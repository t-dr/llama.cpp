"""Compare a Gemma4 Transformers text model with saved llama.cpp token IDs."""

from __future__ import annotations

import argparse
import json
import math
import struct
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import gguf
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

MAX_CONTEXT = 8192
MAX_VOCAB = 1_000_000
MAX_TOKEN_IDS = 1_000_000
MAX_LOGITS_BYTES = 2 * 1024**3


def build_transformers_text_reference(source_dir: Path, output_dir: Path) -> None:
    index = json.loads((source_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    config = json.loads((source_dir / "config.json").read_text())
    text_config = dict(config["text_config"])
    text_config["model_type"] = "gemma4_text"
    text_config["architectures"] = ["Gemma4ForCausalLM"]
    text_config["torch_dtype"] = "float32"

    model_config = Gemma4TextConfig(**text_config)
    with torch.device("meta"):
        meta_model = Gemma4ForCausalLM(model_config)
    expected = {name: tuple(tensor.shape) for name, tensor in meta_model.state_dict().items()}

    tensors = {}
    expert_parts: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for shard in sorted(set(weight_map.values())):
        with safe_open(source_dir / shard, framework="pt", device="cpu") as f:
            for name in f.keys():
                if name.startswith("language_model.model.layers.") and ".experts.switch_glu." in name:
                    parts = name.removeprefix("language_model.model.layers.").split(".")
                    if len(parts) != 5 or parts[1:3] != ["experts", "switch_glu"] or parts[4] != "weight":
                        raise ValueError(f"Unexpected switch_glu tensor: {name}")

                    layer, projection = parts[0], parts[3]
                    value = f.get_tensor(name).clone().contiguous()
                    if projection == "down_proj":
                        key = f"model.layers.{layer}.experts.down_proj"
                        if key in tensors:
                            raise ValueError(f"Duplicate Transformers tensor: {key}")
                        tensors[key] = value
                    elif projection in ("gate_proj", "up_proj"):
                        if projection in expert_parts[layer]:
                            raise ValueError(f"Duplicate switch_glu projection: {name}")
                        expert_parts[layer][projection] = value
                    else:
                        raise ValueError(f"Unknown switch_glu projection: {name}")
                elif name.startswith("language_model.model."):
                    # map the source wrapper to the transformers text-model state names
                    key = "model." + name.removeprefix("language_model.model.")
                    if key in tensors:
                        raise ValueError(f"Duplicate Transformers tensor: {key}")
                    tensors[key] = f.get_tensor(name).clone().contiguous()
                elif name.startswith("language_model.lm_head."):
                    key = name.removeprefix("language_model.")
                    if key in tensors:
                        raise ValueError(f"Duplicate Transformers tensor: {key}")
                    tensors[key] = f.get_tensor(name).clone().contiguous()

    for layer, projections in expert_parts.items():
        if set(projections) != {"gate_proj", "up_proj"}:
            raise ValueError(f"Incomplete switch_glu gate/up pair in layer {layer}")
        gate = projections.pop("gate_proj")
        up = projections.pop("up_proj")
        if gate.shape[0] != up.shape[0] or gate.shape[2:] != up.shape[2:]:
            raise ValueError(f"Incompatible switch_glu gate/up shapes in layer {layer}")
        # transformers stores gate and up together, with gate first
        tensors[f"model.layers.{layer}.experts.gate_up_proj"] = torch.cat((gate, up), dim=1)
        del gate, up

    missing = set(expected) - set(tensors)
    extra = set(tensors) - set(expected)
    shape_mismatches = [
        (name, expected[name], tuple(tensors[name].shape))
        for name in set(expected) & set(tensors)
        if expected[name] != tuple(tensors[name].shape)
    ]
    # this model ties the output head to the token embedding, so the reference can omit it
    if missing - {"lm_head.weight"} or extra or shape_mismatches:
        raise ValueError(
            f"Transformers state layout mismatch: missing={sorted(missing)}, "
            f"extra={sorted(extra)}, shape_mismatches={shape_mismatches[:5]}"
        )
    if "model.embed_tokens.weight" not in tensors:
        raise ValueError("Missing text embedding required to tie lm_head")

    output_dir.mkdir()
    (output_dir / "config.json").write_text(json.dumps(text_config, indent=2) + "\n")
    total_size = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
    save_file(tensors, str(output_dir / "model.safetensors"), metadata={"format": "pt"})
    (output_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": total_size},
                "weight_map": {name: "model.safetensors" for name in tensors},
            },
            indent=2,
        )
        + "\n"
    )
    print("Transformers text tensors:", len(tensors), "bytes:", total_size, flush=True)


def read_token_chunks(
    logits_path: Path,
    max_chunks: int | None,
) -> tuple[int, int, int, np.ndarray]:
    with logits_path.open("rb") as f:
        magic = f.read(8)
        n_ctx = struct.unpack("<I", f.read(4))[0]
        n_vocab = struct.unpack("<i", f.read(4))[0]
        n_chunks = struct.unpack("<i", f.read(4))[0]
        if magic != b"_logits_" or n_ctx < 4 or n_ctx > MAX_CONTEXT:
            raise ValueError("Invalid context or magic in llama-perplexity logits file")
        if n_vocab < 1 or n_vocab > MAX_VOCAB or n_chunks < 1:
            raise ValueError("Invalid vocabulary or chunk count in logits file")
        if n_ctx * n_vocab * 4 > MAX_LOGITS_BYTES:
            raise ValueError("Logits dimensions exceed the validation memory limit")
        chunks_to_read = min(n_chunks, max_chunks) if max_chunks is not None else n_chunks
        token_count = n_ctx * chunks_to_read
        if token_count > MAX_TOKEN_IDS:
            raise ValueError("Token count exceeds the validation limit")
        if logits_path.stat().st_size < 20 + token_count * 4:
            raise ValueError("Truncated token data in llama-perplexity logits file")
        tokens = np.fromfile(f, dtype="<i4", count=token_count)
    if tokens.size != token_count:
        raise ValueError("Invalid llama-perplexity logits file")
    if tokens.size and (tokens.min() < 0 or tokens.max() >= n_vocab):
        raise ValueError("Token ID is outside the logits vocabulary")
    return n_ctx, n_vocab, chunks_to_read, tokens


def get_gguf_value(reader: gguf.GGUFReader, name: str):
    field = reader.fields.get(name)
    if field is None:
        raise ValueError(f"Missing GGUF field: {name}")
    return field.contents()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True,
                        help="Independent dequantized Safetensors checkpoint")
    parser.add_argument("--reference-gguf", type=Path, required=True,
                        help="GGUF used to create the saved token IDs")
    parser.add_argument("--logits-file", type=Path, required=True,
                        help="llama-perplexity file created with --save-all-logits")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--max-chunks", type=int,
                        help="Optional chunk limit for a shorter smoke run")
    args = parser.parse_args()

    if args.threads < 1 or args.threads > 256 or (args.max_chunks is not None and args.max_chunks < 1):
        raise ValueError("threads must be 1..256 and max-chunks must be positive")

    n_ctx, n_vocab, n_chunks, tokens = read_token_chunks(args.logits_file, args.max_chunks)

    reader = gguf.GGUFReader(str(args.reference_gguf))
    add_bos = bool(get_gguf_value(reader, "tokenizer.ggml.add_bos_token"))
    bos_token_id = int(get_gguf_value(reader, "tokenizer.ggml.bos_token_id"))

    torch.set_num_threads(args.threads)
    with tempfile.TemporaryDirectory(prefix="gemma4-transformers-") as tmp:
        text_reference_dir = Path(tmp) / "text-reference"
        build_transformers_text_reference(args.reference_dir, text_reference_dir)
        model, loading = AutoModelForCausalLM.from_pretrained(
            text_reference_dir,
            dtype=torch.float32,
            local_files_only=True,
            low_cpu_mem_usage=False,
            output_loading_info=True,
        )
        model.eval()
        missing = set(loading["missing_keys"])
        unexpected = set(loading["unexpected_keys"])
        if missing or unexpected:
            raise ValueError(
                f"Transformers load mismatch: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        if next(model.parameters()).device.type != "cpu":
            raise ValueError("Transformers reference was not loaded on CPU")
        if model.config.vocab_size != n_vocab:
            raise ValueError(f"Vocab mismatch: logits={n_vocab}, model={model.config.vocab_size}")
        model_bos_id = getattr(model.config, "bos_token_id", None)
        if add_bos and model_bos_id is not None and model_bos_id != bos_token_id:
            raise ValueError(f"BOS mismatch: GGUF={bos_token_id}, Transformers={model_bos_id}")
        max_context = getattr(model.config, "max_position_embeddings", None)
        if max_context is not None and n_ctx > max_context:
            raise ValueError(f"Context {n_ctx} exceeds model limit {max_context}")

        # match llama-perplexity by scoring only the second half of each context
        first = n_ctx // 2
        total_nll = 0.0
        total_tokens = 0
        start = time.time()
        for chunk in range(n_chunks):
            ids = tokens[chunk * n_ctx:(chunk + 1) * n_ctx].copy().astype(np.int64)
            # match the GGUF setting by adding BOS at the start of each chunk
            if add_bos:
                ids[0] = bos_token_id
            input_ids = torch.from_numpy(ids).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)
            with torch.inference_mode():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    logits_to_keep=0,
                )
            logits = outputs.logits[:, first:-1, :].float().contiguous()
            if not torch.isfinite(logits).all():
                raise ValueError(f"Non-finite logits in chunk {chunk + 1}")
            labels = input_ids[:, first + 1:]
            chunk_nll = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="sum",
            )
            if not torch.isfinite(chunk_nll):
                raise ValueError(f"Non-finite loss in chunk {chunk + 1}")
            total_nll += chunk_nll.item()
            total_tokens += labels.numel()
            print(f"chunk {chunk + 1}: nll={chunk_nll.item():.6f}", flush=True)
            del outputs, logits, labels, input_ids, attention_mask

        print("model:", type(model).__name__)
        print("dtype/device:", next(model.parameters()).dtype, next(model.parameters()).device)
        print("chunks/context:", n_chunks, n_ctx)
        print("tokens scored:", total_tokens)
        print("seconds:", round(time.time() - start, 2))
        print("Transformers CPU PPL:", math.exp(total_nll / total_tokens))


if __name__ == "__main__":
    main()
