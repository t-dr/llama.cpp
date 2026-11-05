# Conversion work notes

These notes explain the Gemma4 conversion issue, the fix, and how to repeat
the checks. Paths and model identifiers are placeholders.

## Glossary

### Conversion concepts

Checkpoint: The source model files, usually a config plus one or more
  Safetensors shards. The source framework defines their tensor names and
  storage formats.

Dequantization: Expanding packed values and applying their scale and offset
  to produce F16, BF16, or F32 weights.

GGUF: llama.cpp's model container. A conversion can create a GGUF file
  successfully even when its tensor names or values are not usable by the
  runtime.

Quantization: Storing approximate weights with fewer bits to reduce file and
  memory use. Conversion must decode the source format or preserve it in a
  format the GGUF runtime supports.

Tensor name map: The architecture-specific map from source names such as
  model.layers.0.self_attn.q_proj.weight to canonical GGUF names such as
  blk.0.attn_q.weight. The runtime looks up canonical names.

### Source storage formats

INT8 override (per-layer): Some dense MLP and router weights use packed U32
  words containing 8-bit values, with BF16 block scales and offsets, one pair
  per 64 values. Dequantization is unsigned_value * scale + offset.

NVFP4: A 4-bit floating-point weight format. This checkpoint stores packed
  U32 words with U8 block scales, one scale per 16 values.

U32 packed weight: A 32-bit word used as a container for smaller quantized
  values. It is not necessarily a signed numeric weight; dequantizers read
  its underlying bits.

### GGUF output formats

Q4_0 and Q4_K: GGUF output quantization formats. They differ from the source
  checkpoint's NVFP4 format and are not part of this F16 export.

## Compared with master

At the common base (a91eab67d), ModelBase.map_tensor_name() delegated to the
architecture's TensorNameMap and raised an error for unknown names. It did
not handle Gemma4 switch_glu expert names or this checkpoint's mode=nvfp4
packed weights.

Commit da4d5eec2 added broad regular expressions before the normal map. That
let conversion finish, but the rules intercepted standard names and emitted
noncanonical names, for example attn_q_proj instead of attn_q.

The rules also treated quantization .biases tensors as model biases and
renamed block scales without dequantizing them. The intermediate GGUF had 808
tensors, but the runtime created only 688 and rejected it.

Commit 4e250adb5 removed a redundant local re import. It did not change
the mapping behavior.

## Fixes and why

- map_tensor_name() now tries the architecture's existing map first. This
  preserves canonical names for standard tensors across Llama, Gemma, Qwen,
  and other architectures.

- An exact Gemma4-only alias handles separate switch_glu expert projections
  that the standard map does not cover. It maps gate_proj, up_proj, and
  down_proj to their canonical expert tensor names.

- Gemma4 checkpoints with quantization_config.mode=nvfp4 are dequantized
  before export when the requested output is F16, BF16, or F32. NVFP4 blocks
  use the existing GGUF packing and dequantization implementation.

- Per-layer INT8 overrides unpack U32 words as unsigned bytes and consume
  block scales and offsets. Source .biases tensors are quantizer offsets,
  not runtime linear-layer biases.

- Unsupported output types, missing scale/bias companions, and unmatched
  scale/bias tensors now raise errors instead of producing an invalid GGUF
  that the runtime cannot load.

- Safetensors U32 is represented as a torch I32 container because PyTorch's
  unsigned dtype support is limited. Consumers must reinterpret packed bits;
  arbitrary U32 values are not assumed to be signed weights. The generic
  compressed-tensors path also uses bit shifts and masks on this container.

## Tests and results

### Unit tests

Run the conversion tests from an environment with PyTorch, Safetensors, and
NumPy installed:

```bash
UV_SETUP=/path/to/uv-setup.sh
UV_ROOT=/path/to/uv-root
REPO_ROOT=/path/to/repo
UV_PYTHON=/path/to/python
source "$UV_SETUP"
cd "$UV_ROOT"
PYTHONPATH="$REPO_ROOT" "$UV_PYTHON" -m unittest discover \
    -s "$REPO_ROOT/tests" -p 'test_conversion_*.py' -v
```

The suite has 24 passing tests in two files.

tests/test_conversion_fp8.py covers the existing FP8 conversion cases and a
generic compressed-tensors packed-U32 conversion through Qwen2. That packed
word test checks high-bit values still decode correctly.

tests/test_conversion_gemma4.py checks standard name maps for Llama, Gemma2,
Qwen2, and Gemma4; switch_glu expert aliases; NVFP4 U32 dequantization;
per-layer affine INT8 dequantization; and consumption of quantization scales
and biases.

### Checkpoint conversion

Use a Gemma4 mode=nvfp4 checkpoint with --outtype f16. In a configured
conversion environment, the command is:

```bash
REPO_ROOT=/path/to/repo
CHECKPOINT_DIR=/path/to/checkpoint
OUTPUT_GGUF=/path/to/output.gguf
cd "$REPO_ROOT"
python convert_hf_to_gguf.py "$CHECKPOINT_DIR" \
    --outfile "$OUTPUT_GGUF" --outtype f16 --use-temp-file
```

This machine's uv environment has a PyTorch/OpenBLAS OpenMP import conflict
when the converter script imports torch before NumPy. Import NumPy first and
launch the converter through runpy:

```bash
UV_SETUP=/path/to/uv-setup.sh
REPO_ROOT=/path/to/repo
CHECKPOINT_DIR=/path/to/checkpoint
OUTPUT_GGUF=/path/to/output.gguf
UV_PYTHON=/path/to/python
source "$UV_SETUP"
cd "$REPO_ROOT"
PYTHONPATH=$PWD "$UV_PYTHON" -c \
    'import numpy,runpy,sys; s="convert_hf_to_gguf.py"; sys.argv=[s,*sys.argv[1:]]; runpy.run_path(s,run_name="__main__")' \
    "$CHECKPOINT_DIR" --outfile "$OUTPUT_GGUF" --outtype f16 --use-temp-file
```

The provided checkpoint converted to a new F16 GGUF with 688 tensors. The
runtime loaded it and evaluated the prompt. Do not use the earlier
intermediate output, which still contained quantization offsets as tensors.
The original GGUF was not overwritten.

### CPU-only runtime checks

We used a CPU-only build to isolate conversion correctness from accelerator
selection and execution. The converted model contains about 50.5 GiB of
tensors, while the B70 and RTX 5080 provide about 48.2 GiB of VRAM combined,
before KV cache and runtime buffers. A GPU run would therefore place some
weights or operations on the CPU, mixing backend behavior into this check.
CPU execution can verify that the GGUF loads and computes, but it does not
validate SYCL kernels or GPU performance.

The build was out of source with SYCL and the other GPU backends disabled:

```bash
REPO_ROOT=/path/to/repo
CPU_BUILD_DIR=/tmp/llama-cpu-check
cmake -S "$REPO_ROOT" -B "$CPU_BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
    -DGGML_SYCL=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF \
    -DGGML_METAL=OFF -DGGML_OPENCL=OFF -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_UI=OFF
cmake --build "$CPU_BUILD_DIR" --target llama-cli llama-perplexity --parallel "$(nproc)"
```

Run the CPU-only load/generation and perplexity smoke tests with:

```bash
CPU_BUILD_DIR=/tmp/llama-cpu-check
OUTPUT_GGUF=/path/to/output.gguf
CORPUS=/path/to/corpus.txt
"$CPU_BUILD_DIR/bin/llama-cli" -m "$OUTPUT_GGUF" -ngl 0 -c 128 \
    -p "Say hello in one short sentence." -n 8 -st --simple-io --no-display-prompt
"$CPU_BUILD_DIR/bin/llama-perplexity" -m "$OUTPUT_GGUF" -ngl 0 -c 512 \
    -b 8 -ub 8 --chunks 2 -f "$CORPUS"
```

The CPU CLI smoke test loaded the corrected GGUF, evaluated the prompt, and
exited successfully. It reported about 24 tokens per second for the
8-token run, but its interactive output did not show a useful response. Treat
this as a load/execution smoke test, not a quality check.

The perplexity smoke test reported PPL 28.1078 +/- 5.47278. The sample is too
small for a quality comparison; it only confirms that the model produces
finite scores on CPU.

### Real-checkpoint tensor comparison

Purpose: verify real weight values, not just tensor names, shapes, or successful
model loading. This is a one-off CPU check, not a committed unit test.

Method: read source values from Safetensors, independently decode NVFP4 and
INT8 with NumPy, cast the reference to F16, then compare it to the matching
tensor in the converted GGUF.

The sampled tensors were the first layer's attention query weight, dense MLP
gate weight, and gate expert 0 weight. NVFP4 decoding unpacked U32 words into
nibbles and applied the E2M1 codebook with UE4M3 scales per 16 values. INT8
decoding used unsigned byte values and the BF16 scale and offset per 64 values.

Result: all three matched exactly after F16 rounding, with zero mismatches and
zero maximum absolute difference. This covers representative real tensors and
the expert reshape, but not every tensor or full-model output.

To repeat, set the repository, checkpoint, and output paths, then run this in
the conversion environment:

```bash
source <uv-setup-script>
export REPO_ROOT=/path/to/llama.cpp
export CHECKPOINT_DIR=/path/to/checkpoint
export OUTPUT_GGUF=/path/to/output.gguf
PYTHONPATH="$REPO_ROOT/gguf-py:$REPO_ROOT" <uv-python> - <<'PY'
import json
import os
from pathlib import Path

import numpy as np
import torch
import gguf
from safetensors import safe_open

root = Path(os.environ["CHECKPOINT_DIR"])
weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
reader = gguf.GGUFReader(os.environ["OUTPUT_GGUF"])

prefix = "language_model.model.layers.0."

q_name = prefix + "self_attn.q_proj"
with safe_open(root / weight_map[q_name + ".weight"], framework="pt", device="cpu") as f:
    q_packed = f.get_tensor(q_name + ".weight").clone().contiguous().view(torch.uint8).numpy()
    q_scale = f.get_tensor(q_name + ".scales").clone().contiguous().numpy() & 0x7F
q_codes = np.stack((q_packed & 0x0F, q_packed >> 4), axis=-1)
q_codes = q_codes.reshape(q_packed.shape[0], q_scale.shape[1], 16)
q_exp = (q_scale >> 3).astype(np.int32) & 0x0F
q_man = (q_scale & 0x07).astype(np.float32)
q_factor = np.where(q_exp == 0, q_man * 2.0**-9,
                    (1.0 + q_man / 8.0) * 2.0**(q_exp - 7)) * 0.5
q_factor = np.where((q_scale == 0) | (q_scale == 0x7F), 0.0, q_factor)
e2m1 = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.float32)
q_ref = (q_factor[:, :, None] * e2m1[q_codes]).reshape(q_packed.shape[0], -1).astype(np.float16)
q_out = next(t.data for t in reader.tensors if t.name == "blk.0.attn_q.weight").copy()
print("attention query exact", np.array_equal(q_ref, q_out),
      "mismatches", int(np.count_nonzero(q_ref != q_out)))

m_name = prefix + "mlp.gate_proj"
with safe_open(root / weight_map[m_name + ".weight"], framework="pt", device="cpu") as f:
    m_packed = f.get_tensor(m_name + ".weight").clone().contiguous().view(torch.uint8).numpy()
with safe_open(root / weight_map[m_name + ".scales"], framework="pt", device="cpu") as f:
    m_scale = f.get_tensor(m_name + ".scales").clone().float().numpy()
with safe_open(root / weight_map[m_name + ".biases"], framework="pt", device="cpu") as f:
    m_bias = f.get_tensor(m_name + ".biases").clone().float().numpy()
m_ref = (m_packed.astype(np.float32).reshape(*m_scale.shape, 64) * m_scale[:, :, None] +
         m_bias[:, :, None]).reshape(m_packed.shape[0], -1).astype(np.float16)
m_out = next(t.data for t in reader.tensors if t.name == "blk.0.ffn_gate.weight").copy()
print("INT8 MLP gate exact", np.array_equal(m_ref, m_out),
      "mismatches", int(np.count_nonzero(m_ref != m_out)))

e_name = prefix + "experts.switch_glu.gate_proj"
with safe_open(root / weight_map[e_name + ".weight"], framework="pt", device="cpu") as f:
    e_packed = f.get_tensor(e_name + ".weight")[0].clone().contiguous().view(torch.uint8).numpy()
with safe_open(root / weight_map[e_name + ".scales"], framework="pt", device="cpu") as f:
    e_scale = f.get_tensor(e_name + ".scales")[0].clone().contiguous().numpy() & 0x7F
e_codes = np.stack((e_packed & 0x0F, e_packed >> 4), axis=-1)
e_codes = e_codes.reshape(e_packed.shape[0], e_scale.shape[1], 16)
e_exp = (e_scale >> 3).astype(np.int32) & 0x0F
e_man = (e_scale & 0x07).astype(np.float32)
e_factor = np.where(e_exp == 0, e_man * 2.0**-9,
                    (1.0 + e_man / 8.0) * 2.0**(e_exp - 7)) * 0.5
e_factor = np.where((e_scale == 0) | (e_scale == 0x7F), 0.0, e_factor)
e_ref = (e_factor[:, :, None] * e2m1[e_codes]).reshape(e_packed.shape[0], -1).astype(np.float16)
e_out = next(t.data for t in reader.tensors if t.name == "blk.0.ffn_gate_exps.weight")[0].copy()
print("switch-GLU expert 0 exact", np.array_equal(e_ref, e_out),
      "mismatches", int(np.count_nonzero(e_ref != e_out)))
PY
```

### Full-model reference comparison

Purpose: compare the converted model with a full reference derived from the
same original checkpoint, not just a few tensor samples.

Transformers 5.22.1 recognizes Gemma4 and maps it to
Gemma4ForConditionalGeneration, but its NVFP4 quantizer refuses to load a
pre-quantized NVFP4 checkpoint. The reference therefore used an independent
CPU NumPy pass over the original Safetensors shards. The manual builder is
tests/validate_gemma4_reference.py; it does not call the converter's
dequantization helpers.

Run it with a new output directory; it refuses to overwrite an existing one:

```bash
UV_SETUP=/path/to/uv-setup.sh
REPO_ROOT=/path/to/repo
CHECKPOINT_DIR=/path/to/checkpoint
REFERENCE_DIR=/path/to/new-reference-dir
UV_PYTHON=/path/to/python
source "$UV_SETUP"
PYTHONPATH="$REPO_ROOT" "$UV_PYTHON" "$REPO_ROOT/tests/validate_gemma4_reference.py" \
    --checkpoint-dir "$CHECKPOINT_DIR" --output-dir "$REFERENCE_DIR"
```

This is a manual integration check, not part of the 24-test unit suite. It
reads the full checkpoint and writes about 50 GB of reference weights.

The script decoded all 327 U32 weights: 207 NVFP4 weights and 120 per-layer
INT8 weights. It consumed 327 block-scale tensors and 120 INT8 offset tensors,
preserved the other source dtypes, and kept dequantized INT8 values in F32
until GGUF export. This matters for router weights, which the GGUF exporter
stores in F32. It produced 1,043 tensors, and all matched the corrected
reference checkpoint tensor by tensor.

An earlier reference rounded INT8 weights to F16 too early and differed from
the target in 30 router tensors by one F16 rounding step. That reference was
discarded.

An earlier draft of the reference-builder script also decoded some NVFP4
scales incorrectly. Its output was rejected. The command above uses the
corrected script whose output matched the reference checkpoint.

The corrected reference Safetensors checkpoint contained 1,043 tensors. The
normal F16 converter path wrote a 688-tensor reference GGUF. A streaming
comparison found identical names, shapes, types, and values for all 688
tensors in the reference and target GGUFs. The reference GGUF was made by
running the Checkpoint conversion command above on the new reference directory.

The full tensor scan used:

```bash
REPO_ROOT=/path/to/repo
UV_PYTHON=/path/to/python
REFERENCE_GGUF=/path/to/reference.gguf
CONVERTED_GGUF=/path/to/converted.gguf
export REFERENCE_GGUF CONVERTED_GGUF
PYTHONPATH="$REPO_ROOT/gguf-py" "$UV_PYTHON" - <<'PY'
import numpy as np
import os
import gguf

reference = gguf.GGUFReader(os.environ["REFERENCE_GGUF"])
converted = gguf.GGUFReader(os.environ["CONVERTED_GGUF"])
ref = {tensor.name: tensor for tensor in reference.tensors}
cur = {tensor.name: tensor for tensor in converted.tensors}
assert ref.keys() == cur.keys()

exact = 0
different = []
for name in sorted(ref):
    a, b = ref[name], cur[name]
    if a.tensor_type != b.tensor_type or tuple(a.shape) != tuple(b.shape):
        different.append((name, "shape/type mismatch"))
        continue
    av, bv = a.data.reshape(-1), b.data.reshape(-1)
    mismatches = 0
    for start in range(0, av.size, 1_000_000):
        mismatches += int(np.count_nonzero(av[start:start + 1_000_000] !=
                                           bv[start:start + 1_000_000]))
    if mismatches:
        different.append((name, mismatches))
    else:
        exact += 1
print("exact tensors:", exact, "different tensors:", len(different))
print(different[:20])
PY
```

To compare full-model output, first save baseline log probabilities from the
reference GGUF, then run KL comparison on the target. Both runs use the same
CPU-only build and the baseline file stores the token IDs, so tokenization is
identical:

```bash
CPU_BUILD_DIR=/tmp/llama-cpu-check
REFERENCE_GGUF=/path/to/reference.gguf
CONVERTED_GGUF=/path/to/converted.gguf
CORPUS=/path/to/corpus.txt
REFERENCE_LOGITS=/tmp/reference-logits.bin
"$CPU_BUILD_DIR/bin/llama-perplexity" -m "$REFERENCE_GGUF" -ngl 0 \
    -c 512 -b 8 -ub 8 --chunks 16 -f "$CORPUS" \
    --save-all-logits "$REFERENCE_LOGITS"
"$CPU_BUILD_DIR/bin/llama-perplexity" -m "$CONVERTED_GGUF" -ngl 0 \
    -c 512 -b 8 -ub 8 --kl-divergence \
    --kl-divergence-base "$REFERENCE_LOGITS"
"$CPU_BUILD_DIR/bin/llama-perplexity" -m "$CONVERTED_GGUF" -ngl 0 \
    -c 512 -b 8 -ub 8 --chunks 16 -f "$CORPUS"
```

The reference and target standalone PPL estimates were both 15.4482 +/- 0.97290.
The KL comparison reported mean KL approximately zero, maximum KL 0.000057,
and 100 percent top-token agreement over 16 chunks of 512 tokens.

The KL tool stores baseline log probabilities in a compact 16-bit format, so
its displayed PPL(base) is approximate and differs from the standalone PPL.
Use the standalone PPL runs for the direct PPL comparison; use KL and
top-token agreement for the distribution comparison.

### Still to validate

- A separate Transformers forward pass was not run. The full-model reference
  uses independently dequantized weights, but both GGUF models use the same
  llama.cpp CPU runtime. This isolates conversion from backend differences.

- The real checkpoint validates this Gemma4 encoding. Other Gemma variants
  and other source U32 formats are not claimed to be supported by the
  Gemma4-specific NVFP4 dequantizer.
