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

### Model evaluation

BOS token: The beginning-of-sequence special token. The perplexity comparison
  inserts it at the start of each context chunk.

Cross-entropy: The average negative log-probability assigned to the correct
  next tokens. Its exponential is perplexity.

Forward pass: One run of input tokens through the model layers to produce
  logits.

KL divergence: A measure of how much one next-token probability distribution
  differs from another. A value near zero means the distributions are close.

Logits: Unnormalized scores for the vocabulary tokens. Softmax converts them
  into probabilities.

Log-probability: The logarithm of a token's probability. The KL comparison
  file stores these values in a compact 16-bit format.

Perplexity: The exponent of average negative log-probability over a token
  sequence. Compare perplexity only with the same token IDs and context setup.

Reference checkpoint: A separate set of model weights made through an
  independent dequantization path. It provides expected values for conversion
  checks.

State dict: A mapping from model parameter names to their tensor values.

Softmax: The operation that turns logits into a probability distribution over
  the vocabulary.

Token ID: An integer index into the tokenizer vocabulary. Matching token IDs
  prevents tokenizer differences from affecting a model comparison.

Top-token agreement: The percentage of positions where two models assign the
  highest probability to the same next token.

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

### Secondary BF16 Gemma4 variant check

The local `gemma-4-26B-A4B` source is `Gemma4ForConditionalGeneration` with
30 layers, 128 experts, and top-8 routing. Its two Safetensors shards contain
1,013 BF16 tensors. The config has no NVFP4 quantization mode, and the source
has no packed U32 weights or `.scales`/`.biases` quantization companions. This
checks standard BF16 Gemma4/MoE conversion, not the NVFP4/INT8 dequantizer.

The converter wrote `gemma-4-26B-A4B-f16.gguf` beside the source checkpoint.
The file is 50,505,118,272 bytes and contains 658 tensors (266 F16 and 392
F32), including generated `rope_freqs.weight`. The dry run warned that the
generic RoPE metadata handler does not recognize `proportional`; the
Gemma4-specific converter emits the RoPE frequency tensor used by its
full-attention layers. Sampled Q projection, dense MLP gate, router projection,
and expert-0 fused gate/up values matched the source after the expected F16 or
F32 cast. A CPU `llama-cli` load and four-token generation smoke test
succeeded. This was not a full tensor scan, PPL test, or cross-runtime quality
check.

For the Arc test, the F16 GGUF was quantized on CPU to a separate Q4_K_M file
beside it. The result is 16,795,998,784 bytes; the quantizer reported fallback
quantization for 60 of 658 tensors. The CPU `llama-cli` also loaded this file
and generated four tokens. This derivative is for GPU runtime testing; it
does not replace the F16 conversion check.

To reproduce the quantized derivative:

```bash
CPU_BUILD_DIR=/path/to/cpu-build
INPUT_GGUF=/path/to/gemma-4-26B-A4B-f16.gguf
OUTPUT_GGUF=/path/to/gemma-4-26B-A4B-Q4_K_M.gguf
"$CPU_BUILD_DIR/bin/llama-quantize" "$INPUT_GGUF" "$OUTPUT_GGUF" Q4_K_M 56
```

### CPU-only runtime checks

We use a CPU-only build to isolate conversion correctness from accelerator
selection and execution. The original 31B F16 GGUF is 50,505,137,184 bytes
(47.04 GiB), larger than either GPU individually. CUDA and SYCL memory cannot
be combined, and CPU fallback would mix backend behavior into a GPU test.
CPU execution verifies that a GGUF loads and computes, but does not validate
SYCL kernels or GPU performance.

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

### Transformers forward check

Purpose: check that an independent Transformers implementation can load the
dequantized text model and run a CPU forward pass.

This is a manual Gemma4 integration check, not part of the 24-test unit suite.
The reusable validation helper is tests/validate_gemma4_transformers.py. It
builds a temporary Transformers text checkpoint and removes it when the run
ends; it needs about 50 GB of scratch space.

Using the same token IDs, BOS handling, context, and PPL calculation is a
general cross-runtime method. The checkpoint key remapping and expert fusion
in this helper are specific to this Gemma4 layout.

Transformers could not load the original pre-quantized checkpoint. For this
check, the helper remaps the independently dequantized reference weights to
Gemma4TextModel state names. It moves the language-model prefix to the
Transformers text-model prefix and concatenates each switch_glu gate/up expert
pair into the fused gate_up_proj tensor. The Transformers model ties
lm_head.weight to the token embedding weight.

Run it with the reference checkpoint, reference GGUF, and token/log-probability
file created by the CPU llama.cpp PPL command above:

```bash
UV_SETUP=/path/to/uv-setup.sh
REPO_ROOT=/path/to/repo
REFERENCE_DIR=/path/to/independent-reference
REFERENCE_GGUF=/path/to/reference.gguf
LOGITS_FILE=/path/to/reference-logits.bin
UV_PYTHON=/path/to/python
source "$UV_SETUP"
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 \
PYTHONPATH="$REPO_ROOT/gguf-py:$REPO_ROOT" "$UV_PYTHON" \
    "$REPO_ROOT/tests/validate_gemma4_transformers.py" \
    --reference-dir "$REFERENCE_DIR" --reference-gguf "$REFERENCE_GGUF" \
    --logits-file "$LOGITS_FILE" --threads 32
```

The resulting text checkpoint loaded as Gemma4ForCausalLM with F32 CPU weights
and no missing or unexpected state-dict keys. It used the same 16 chunks of
512 saved token IDs as the llama.cpp PPL test. BOS token ID 2 was inserted at
the start of each chunk, matching llama.cpp. The test scored the last 255
next-token predictions in each chunk, for 4,080 scored tokens total.

Transformers PPL was 15.0768. The llama.cpp reference and converted GGUFs both
reported PPL 15.4482 +/- 0.97290 on those tokens. The difference is about 2.4
percent and falls within the llama.cpp sample uncertainty. Transformers used
F32 parameters while the GGUF stores mixed F16 and F32 weights.

Because PPL does not show per-position agreement, raw logits were compared
next on the same token IDs.

A separate one-off check captured raw llama.cpp logits through the C API and
compared them with Transformers on the same 16 chunks. It scored the same
4,080 positions across a vocabulary of 262,144 tokens. The temporary capture
and comparison tools were kept outside the repository; no production source
was changed.

Transformers PPL from these logits was 15.0771; llama.cpp PPL was 15.4482.
Top-token agreement was 92.6471 percent (3,780 of 4,080 positions). Across all
logit values, raw maximum absolute error was 16.9764 and RMS error was 0.6910.
After subtracting each row's mean difference, centered RMS error was 0.5564.
Mean forward and reverse KL were 0.0552 and 0.0555 nats; their 95th percentiles
were 0.2511 and 0.2461 nats. Mean total variation was 0.1569 (95th percentile
0.5953).

The low top-token agreement prompted a weight-precision control before
investigating model execution.

To isolate weight-storage precision, a second temporary run rounded the 296
F16-mapped Transformers tensors to F16 and back to F32, while keeping the 391
F32-mapped tensors unchanged. This keeps Transformers computation in F32.
Top-token agreement rose slightly to 92.9902 percent (3,794 of 4,080
positions). Mean total variation changed to 0.1555 and centered RMS logit
error to 0.5541. Transformers PPL was 15.0550. Weight rounding had only a
small effect on the cross-runtime differences.

Since weight rounding did not explain the gap, the next check compared layer
activations.

This is not bitwise logits parity. The result records numerical differences
between the F32 Transformers computation and llama.cpp's mixed-precision path;
no logit error threshold was defined. The exact tensor match plus this
precision-matched check does not point to a conversion error.

A temporary layerwise check captured scaled embeddings and all 30 layer outputs
for one 512-token chunk. The scaled embeddings matched exactly. The first
nonzero difference was at layer 0 (relative RMS 0.00138, cosine 0.999999);
relative RMS grew across the stack, reaching 0.115 at layer 28 and 0.073 at
layer 29. Error was not higher at the llama.cpp 8-token batch boundaries. This
shows gradual cross-runtime drift, but does not identify the first differing
operation inside layer 0.

The embedding match and gradual drift led to a closer look inside layer 0.

A temporary layer-0 operation check narrowed the first larger difference:
scaled embeddings, attention normalization, Q/K/V projections and Q/K/V
normalization had relative RMS errors below 3e-7. RoPE outputs were also close
(Q 7.6e-6, K 6.0e-6). The first larger error was after the attention and
output-projection path (relative RMS 0.00126). Shared MLP output error was
0.00051. Router logits differed by 0.00103, with top-8 expert sets matching
for 509 of 512 tokens; normalized expert output error was 0.00505. This points
to attention computation as the first measurable divergence, with a few MoE
routing changes that may amplify it. The exact attention operation or mask
cause is not isolated. The layer uses a 1,024-token sliding window with a
512-token context, and both implementations use attention scale 1.0, so the
window cutoff and attention scale do not explain this probe's difference.

Because Q/K/V projections and RoPE were already close, a flash-attention
on/off comparison tested whether the attention kernel path caused the
divergence.

A full 16-chunk A/B check with llama.cpp flash attention disabled improved
agreement while keeping the same GGUF, tokens, and F16-matched Transformers
weights. The two llama.cpp attention modes had 92.5000 percent top-token
agreement and mean total variation 0.1632. Against Transformers, non-flash
top-token agreement was 94.9510 percent (3,874 of 4,080
positions), mean KL in both directions fell to about 0.0322 nats, and mean
total variation fell to 0.1105. llama.cpp PPL moved from 15.4482 to 15.1378,
closer to the Transformers result of 15.0550. In layer 0, the attention output
relative RMS error fell from 0.00214 to 0.00033. This implicates the default
flash-attention path as a major source of the CPU cross-runtime difference,
but does not establish that either implementation is incorrect.

For the first 512-token chunk, a direct per-layer `kqv_out` comparison between
the two llama.cpp modes showed the relative RMS difference growing from
0.0021 at layer 0 to 0.081 at layer 20 and 0.080 at layer 24. This is
consistent with accumulated attention-path drift rather than one isolated
layer failure.

An independent F64 causal-attention calculation used captured post-RoPE Q/K,
normalized V, GQA head mapping, and scale 1.0. With the Transformers Q/K/V
inputs, it matched the Transformers pre-projection attention output to
1.1e-6 relative RMS. For llama.cpp, the reference rounded cached K/V to F16
and used each attention mode's own Q/K/V inputs. It covered the sliding layer
0 and full-attention layers 5, 11, 17, 23, and 29. Flash relative RMS errors
were 0.0014 to 0.0060; non-flash errors were 0.00018 to 0.00060, about 8 to
14 times closer across these layers. This confirms the flash-path difference
across both layer types at context 512.

A 1,536-token CPU check then crossed the 1,024-token sliding-window cutoff,
using three contiguous saved token chunks with BOS only at the start. It scored
767 positions, 511 at or beyond the cutoff. After the cutoff, non-flash had
96.0861 percent top-token agreement with Transformers versus 92.3679 percent
for flash; mean total variation was 0.0420 versus 0.0704. However, target-token
PPL was 23.7139 for non-flash and 23.2363 for flash, versus 23.1370 for
Transformers. Thus top-token/distribution metrics favored non-flash, while
target-token PPL favored flash on this sample.

Two longer CPU checks had all scored positions at or beyond the sliding-window
cutoff. At context 2,048 (1,023 scored positions), non-flash versus flash had
top-token agreement of 95.6989 versus 92.6686 percent with Transformers, mean
total variation 0.03980 versus 0.06655, and PPL 10.09465 versus 10.10540
(Transformers: 10.06076). At context 4,096
(2,047 scored positions), non-flash versus flash had top-token agreement of
94.4797 versus 92.0371 percent, mean total variation 0.05291 versus 0.08451,
and PPL 4.80441 versus 4.85933 (Transformers: 4.84929). At both lengths,
non-flash was closer on top-token agreement and distribution metrics.
Target-token PPL favored non-flash at 2,048 but flash at 4,096. The Python and
llama.cpp comparisons both used 32 CPU threads.

Mean forward/reverse KL was 0.09782/0.04877 for flash versus
0.04551/0.02182 for non-flash at 2,048; at 4,096 it was 0.16888/0.08423
versus 0.06872/0.03405.

A reliability repeat used four contiguous 2,048-token contexts from the same
8,192-token stream, with BOS at each context start (4,092 scored predictions).
Non-flash versus flash had 94.8925 versus 92.5953 percent top-token agreement,
mean total variation 0.04619 versus 0.07485, and mean forward/reverse KL
0.05415/0.02713 versus 0.12432/0.06303. PPL was 10.46984 for non-flash and
10.57159 for flash, against Transformers at 10.48027. Across the four
contexts, the mean per-context log-PPL ratio (llama.cpp/Transformers) and
standard error were -0.00100 +/- 0.00815 for non-flash and 0.00868 +/- 0.00992
for flash, so this sample does not establish a PPL winner.

The explicit sliding-window F64 reference for layer 0 at context 1,536 gave
relative RMS errors of 0.00593 for flash and 0.000148 for non-flash on
positions at or beyond the cutoff. At context 4,096, layer 0 had relative RMS
errors of 0.00577 for flash and 0.000146 for non-flash after the cutoff; full
attention layer 5 had errors of 0.0260 and 0.000362. The lower non-flash error
therefore persists beyond the window across sliding and full attention in the
sampled CPU layers. End-to-end PPL remains mixed.

No GPU comparison was run. The Arc Pro B70 has 32 GiB physical VRAM and
31.89 GiB usable; the RTX 5080 has 16,303 MiB (15.92 GiB). The 50,505,137,184
byte GGUF is 47.04 GiB, larger than either GPU individually. CUDA and SYCL
devices cannot be combined for one inference allocation, and CPU fallback
would invalidate a GPU-only comparison. A GPU test needs a smaller compatible
checkpoint that fits on one backend, with device placement verified. A SYCL
test must use `feature/sycl-topology-discovery`; its local fixes are not
present on `conversion-update`.

### Still to validate

- The CPU flash-attention A/B is closer to Transformers when flash attention is
  disabled on top-token and distribution metrics, including the four-context
  repeat. PPL differences vary by context and are not decisive in that repeat.
  CPU A/B checks cover contexts 512, 1,536, 2,048, and 4,096; explicit
  attention references cover selected layers at 512, 1,536, and 4,096. A
  full-model GPU comparison needs a smaller compatible checkpoint on one
  backend; do not combine CUDA and SYCL or count CPU fallback as GPU execution.
  Any SYCL check must use `feature/sycl-topology-discovery`. Test relevant GPU
  backends before deciding whether the flash-path difference is an expected
  numerical limit or a correctness issue; do not change defaults based on these
  tests alone.

- The real checkpoint validates this Gemma4 encoding. Other Gemma variants
  and other source U32 formats are not claimed to be supported by the
  Gemma4-specific NVFP4 dequantizer. The BF16 26B A4B check covers standard
  Gemma4/MoE conversion only, not NVFP4/INT8 source formats.
