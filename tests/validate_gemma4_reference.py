"""Build an independent floating-point reference checkpoint for Gemma4 QAT."""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

E2M1 = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.float32)
LOW_NIBBLE = np.array([value & 0x0F for value in range(256)], dtype=np.uint8)
HIGH_NIBBLE = np.array([value >> 4 for value in range(256)], dtype=np.uint8)


def ue4m3_half_scale(value: int) -> float:
    value &= 0x7F
    if value in (0, 0x7F):
        return 0.0
    exponent = value >> 3
    mantissa = value & 0x07
    scale = mantissa * 2.0**-9 if exponent == 0 else (1.0 + mantissa / 8.0) * 2.0**(exponent - 7)
    return scale * 0.5


UE4M3_HALF_SCALE = np.array([ue4m3_half_scale(value) for value in range(256)], dtype=np.float32)


def decode_nvfp4(weight: torch.Tensor, scales: torch.Tensor, group_size: int) -> torch.Tensor:
    if group_size != 16 or scales.dtype != torch.uint8:
        raise ValueError("Expected NVFP4 U8 scales with group size 16")

    row_shape = tuple(weight.shape[:-1])
    n_rows = weight.numel() // weight.shape[-1]
    packed = weight.contiguous().view(torch.uint8).numpy().reshape(n_rows, -1)
    scale_bytes = scales.contiguous().view(torch.uint8).numpy().reshape(n_rows, -1)
    if tuple(scales.shape[:-1]) != row_shape or packed.shape[-1] != scale_bytes.shape[-1] * 8:
        raise ValueError("NVFP4 weight and scale shapes do not match")

    codes = np.stack((LOW_NIBBLE[packed], HIGH_NIBBLE[packed]), axis=-1)
    codes = codes.reshape(n_rows, scale_bytes.shape[-1], 16)
    factor = UE4M3_HALF_SCALE[scale_bytes]
    values = (factor[:, :, None] * E2M1[codes]).reshape(n_rows, -1)
    output_shape = row_shape + (values.shape[-1],)
    return torch.from_numpy(values.reshape(output_shape).astype(np.float16))


def decode_int8(
    weight: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    if group_size != 64 or tuple(scales.shape) != tuple(biases.shape):
        raise ValueError("Expected INT8 groups of 64 with matching scale and bias shapes")

    row_shape = tuple(weight.shape[:-1])
    n_rows = weight.numel() // weight.shape[-1]
    packed = np.array(
        weight.contiguous().view(torch.uint8).numpy().reshape(n_rows, -1),
        dtype=np.uint8,
        copy=True,
    )
    scale_values = scales.float().numpy().reshape(n_rows, -1)
    bias_values = biases.float().numpy().reshape(n_rows, -1)
    if tuple(scales.shape[:-1]) != row_shape or packed.shape[-1] != scale_values.shape[-1] * group_size:
        raise ValueError("INT8 weight and scale shapes do not match")

    values = (
        packed.astype(np.float32).reshape(n_rows, -1, group_size) * scale_values[:, :, None]
        + bias_values[:, :, None]
    ).reshape(n_rows, -1)
    output_shape = row_shape + (values.shape[-1],)
    return torch.from_numpy(values.reshape(output_shape).astype(np.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = args.checkpoint_dir
    output = args.output_dir
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite reference directory: {output}")

    index = json.loads((source / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    config = json.loads((source / "config.json").read_text())
    quant = config.pop("quantization_config")
    default_bits = int(quant.get("bits", 4))
    default_group_size = int(quant.get("group_size", 16))
    layer_config = {
        name: value for name, value in quant.items()
        if isinstance(value, dict) and "bits" in value
    }

    headers = {}
    for shard in sorted(set(weight_map.values())):
        with (source / shard).open("rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            headers[shard] = json.loads(f.read(header_size))

    packed_weights: dict[str, tuple[int, int, str, str | None]] = {}
    sidecars: set[str] = set()
    formats: Counter[int] = Counter()
    for name, shard in weight_map.items():
        if headers[shard][name]["dtype"] != "U32":
            continue
        if not name.endswith(".weight"):
            raise ValueError(f"Unexpected U32 tensor that is not a weight: {name}")

        stem = name.removesuffix(".weight")
        scales = [key for key in (stem + ".scales", stem + ".scales.weight") if key in weight_map]
        if len(scales) != 1:
            raise ValueError(f"Expected one scale tensor for {name}, found {scales}")
        scale_name = scales[0]
        if weight_map[scale_name] != shard:
            raise ValueError(f"Scale tensor is in another shard: {scale_name}")

        layer = layer_config.get(stem, {})
        bits = int(layer.get("bits", default_bits))
        group_size = int(layer.get("group_size", default_group_size))
        bias_name = stem + ".biases"
        bias_name = bias_name if bias_name in weight_map else None
        if bits == 8 and bias_name is None:
            raise ValueError(f"Missing INT8 offset tensor for {name}")
        if bits == 4 and bias_name is not None:
            raise ValueError(f"Unexpected offset tensor for NVFP4 weight {name}")
        if bias_name is not None and weight_map[bias_name] != shard:
            raise ValueError(f"Bias tensor is in another shard: {bias_name}")

        packed_weights[name] = (bits, group_size, scale_name, bias_name)
        sidecars.add(scale_name)
        if bias_name is not None:
            sidecars.add(bias_name)
        formats[bits] += 1

    unmatched = [
        name for name in weight_map
        if name.endswith((".scales", ".scales.weight", ".biases")) and name not in sidecars
    ]
    if unmatched:
        raise ValueError(f"Unmatched quantization sidecars: {unmatched[:5]}")

    output.mkdir(parents=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for filename in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "processor_config.json",
        "generation_config.json",
    ):
        if (source / filename).is_file():
            shutil.copy2(source / filename, output / filename)

    output_weight_map = {}
    total_size = 0
    for shard in sorted(set(weight_map.values())):
        tensors = {}
        with safe_open(source / shard, framework="pt", device="cpu") as f:
            for name in f.keys():
                if name in sidecars:
                    continue
                if name in packed_weights:
                    bits, group_size, scale_name, bias_name = packed_weights[name]
                    weight = f.get_tensor(name).contiguous()
                    scales = f.get_tensor(scale_name).contiguous()
                    if bits == 4:
                        tensor = decode_nvfp4(weight, scales, group_size)
                    elif bits == 8:
                        assert bias_name is not None
                        tensor = decode_int8(weight, scales, f.get_tensor(bias_name).contiguous(), group_size)
                    else:
                        raise ValueError(f"Unsupported bit width {bits} for {name}")
                else:
                    tensor = f.get_tensor(name).contiguous()
                    if tensor.dtype == torch.uint32:
                        raise ValueError(f"Unprocessed U32 tensor: {name}")

                tensors[name] = tensor.contiguous()
                output_weight_map[name] = shard
                total_size += tensor.numel() * tensor.element_size()

        save_file(tensors, str(output / shard), metadata={"format": "pt"})
        print(f"Wrote {shard}: {len(tensors)} tensors")

    output_index = {
        "metadata": {"total_size": total_size},
        "weight_map": output_weight_map,
    }
    (output / "model.safetensors.index.json").write_text(json.dumps(output_index, indent=2) + "\n")
    print(f"Reference ready: {len(output_weight_map)} tensors, {total_size} bytes, formats {dict(formats)}")


if __name__ == "__main__":
    main()
