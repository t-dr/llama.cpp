# SYCL backend working notes (feature/sycl-topology-discovery)

## What the topology probe does

- `ggml-sycl-topology.hpp` collects a PyTorch-DeviceProp-parity field set per
  device via `collect_device()` and `discover_topology()`.
- Enumeration consumes `dpct::dev_mgr`, so `Device::device_id` is the real
  backend index (no remap). Devices outside the pool (other platforms) are
  appended as visible-only and tagged `backend_usable = false`.
- Stored in `info.topology` (member of `ggml_sycl_device_info`, common.hpp).
- Full dump per device at init via `ggml_sycl::describe()`.

Observed on this machine (Arc Pro B70 = bmg_g31, Xeon w7-3465X):
- B70 via level_zero:gpu -> usable; same GPU also visible via opencl:gpu
  (same pci 0xe223 + uuid confirms same silicon).
- Xeon via opencl:cpu -> visible only. 56 CUs, subgroups [4..64],
  partition_max_sub_devices 56, fp16/fp64 via CL emulation-ish config.
- ext_intel_* topology aspects (xe_*, gpu_eu_*) only answer on Level-Zero
  backend; OpenCL reports 0. Deeper queries need L0.

## nsm fix

- `nsm` was `max_compute_units / 16` -> now `gpu_eu_count /
  gpu_eu_count_per_subslice` (aspect-gated, /16 fallback).
- B70: 256 EUs / 8 = 32 Xe cores (was 16, half-modeled).
- `fattn-dpas.cpp` had `nsm * 2` compensation; removed (still 32 on B70,
  fixes over-count on archs where eu_per_subslice != 8).
- `fattn-common` blocks_per_wave + count-equal grid sizing now see real count.

## caps-gated XMX enable (ggml_sycl_xmx_enabled, common.hpp)

- Known arch whitelist unchanged (bmg_g21/g31, lnl_m).
- Unknown arch: enables when device reports is_gpu + has_subgroup_mma +
  has_2d_block_io + 16 in sub_group_sizes. Logs a one-time warning naming the
  arch value. Still overridable via GGML_SYCL_ENABLE_XMX_ESIMD=0.
- Caveat: caps say "DPAS exists", not "Xe2 DPAS shapes match"; silent-wrong-
  result risk remains -> planned init-time DPAS self-test to promote
  "capable" to "verified".

## Temporary debug prints (remove later, keep info here)

Both are unconditional stderr fprintf because ggml-level logs do not reach
llama-cli/server output even at -lv 3 (GGML_LOG_INFO from backend never
visible; even the unconditional topology dump is absent there).

- `[FAPICK]` in ggml/src/ggml-sycl/fattn.cpp: wrapper
  `ggml_sycl_get_best_fattn_kernel` prints picked kernel + nq/nkv/dk/dv +
  K/V types per FA call. Impl renamed to `..._impl`.
  Result on B70: q8_0 K/V decode -> DPAS (dk=dv=256, nq=1..2). No
  ONEDNN/MKL picks observed in the tested workload. F16 K/V decode has no
  XMX kernel at all (falls to VEC/TILE) -- coverage gap, not a gating fix.
- `[MMPICK]` in ggml/src/ggml-sycl/mmvq.cpp: first-30-calls print of the
  XMX gate state and final decision (enabled/type_ok/extra/reorder/ncols,
  later alignment/shape rejects or xmx-dispatched), with separate counters
  for ncols <= 8 and ncols > 8.
  Before commit 4a4de9d24, plain Q4_0 had type_ok=0; the new trait now makes
  type_ok=1. A cold B=16 run can still miss XMX because the weights are not
  reordered before the batch-width gate.
- `[MATMULPICK]` in ggml/src/ggml-sycl/ggml-sycl.cpp: first 24 Q4_0
  dispatches, printing selected route, xmx_batch eligibility, reorder state,
  and W/X dimensions. The first 24 are often prefill ops; do not treat them
  as a decode-only trace.
- `[SCHED]` in ggml/src/ggml-backend.cpp: stderr mirror of the per-op
  backend assignment line inside ggml_backend_sched_print_assignments.
  Gated on sched->debug > 1 -> run with GGML_SCHED_DEBUG=2. Prints
  "node #i OP name -> BACKEND" so CPU-fallback ops are visible.

## Bench numbers (Arc B70, llama-bench, -fa on, -ngl 99)

- qwen35 4B Q4_K_M (2.51 GiB): pp512 2051 t/s, tg128 126.2 t/s
- qwen35 4B Q4_K_M with GGML_SYCL_ENABLE_XMX_ESIMD=0: pp512 2049, tg128
  130.4 t/s -> XMX is ~3% SLOWER at batch-1 decode, no benefit here.
  XMX wins (if any) live in wider-batch shapes.
- gemma4 E4B Q4_0 (measured before Q4_0 XMX support): pp512 1507, tg128
  99.7 t/s. Different model/size so not a controlled XMX test.
- 31B Q4_0 gemma: ~17-23 t/s decode -> roughly linear with model size vs
  the small models: decode is bandwidth-bound, not misconfigured.
  ~130 t/s on 2.5 GiB implies ~325 GB/s effective on B70.

## Glossary

Grouped roughly: the concept, the formats, the hardware, the code paths,
the stack, the knobs.

The compute:

- matmul (matrix multiply): Y = X*W - every layer is one of these, ~95% of
  decode time. X is activations (what the model is computing), W is weights
  (the learned parameters). A "kernel" is the GPU program that does it.
- decode vs prefill: decode = one token at a time (the chat loop - what you
  measure as t/s). prefill = processing the prompt all at once. They have
  different shapes, different fast paths, different bottlenecks.
- batch / ncols: how many activations multiply at once - one row per token
  or per sequence. The "width" that decides which matmul kernel applies.
- t/s: tokens per second - the throughput measure. pp = prefill, tg =
  text-generation/decode, B = batch (parallel sequences).

Quantization (why the weights are weird):

- quantization: shrink weights from 16/32-bit floats down to 2-8 bits per
  value to fit on the card and halve memory traffic. The cost: kernels must
  "dequantize" back to numbers the math units understand before computing.
  Where dequant happens is the whole kernel problem.
- block format: quantized weights live in groups with shared scale/min
  factors, not bare values - a block is a run of values + its own scale.
- Q4_0 (plain block): 32 values, 16 packed 4-bit nibbles + one fp16 scale.
  value = d*(nibble - 8). The simplest format - and the one we just added.
- Q4_K..Q6_K (K-quants): super-block of 256 = 8 groups of 32, bit-packed
  codes + 6-bit scale/min fields. More types of helper data per block -
  more dequant logic, but tighter compression.
- super-block / group of 32: the unit kernels process K in - the DPAS
  instruction wants 32-wide groups, so weights decompose into groups of 32.
- reorder / SOA: the kernel wants all the "quant codes" together, then all
  the scales - a struct-of-arrays layout, not block-by-block. The reorder
  pass converts at load time so the kernel reads contiguous chunks.
- unpack/dequant in-kernel: reading the packed nibbles and expanding them
  to real values inside the matvec - no separate dequant pass needed.

Hardware terms:

- EU (execution unit): the smallest compute unit on an Intel GPU - runs one
  hardware thread of SIMD. 256 EUs on B70.
- Xe core / subslice: a cluster of EUs sharing a scheduler and SLM (shared
  local memory). nsm = Xe core count = EUs / eu_per_subslice = 32 on B70.
  The number that feeds wave/occupancy math.
- sub_group / warp: a 16-wide SIMD lane group - the width kernels launch
  on. B70 supports sizes [16,32]; the XMX kernels need 16.
- XMX / DPAS: Intel's matrix-hardware (tensor cores) and the instruction
  that feeds them - dot-product-accumulate on small int8 tiles. The "fast
  path" this work enables.
- SLM (shared local memory): fast scratch memory shared by a workgroup -
  the tile sizes in FA are bounded by it.
- USM (unified shared memory): device pointers usable from host - the
  alternative to explicit copy calls.
- 2D block load: a hardware-assisted strided load for tile data - the
  load_2d the XMX kernels use. Aspect-gated: has_2d_block_io.

Discovery / devices:

- SYCL device: an abstract compute target the runtime can schedule work on -
  GPUs and CPUs both count, each behind a backend.
- backend: the driver interface a device lives behind - level_zero:gpu,
  opencl:cpu, opencl:gpu. The same Arc shows up twice (L0 and OpenCL) -
  real info for a probe.
- platform: a collection of devices sharing one backend context. dev_mgr
  picks ONE platform - the single-platform model that hid the Xeon.
- aspect: a capability query the device advertises (has_subgroup_mma,
  has_2d_block_io). Aspect-gated queries return 0 when unsupported.
- arch enum: bmg_g21/g31 = Battlemage, lnl_m = Lunar Lake. B70 = bmg_g31 =
  0x500800000. The XMX whitelist keys on these names.
- DeviceProp / topology: the per-device field set our probe collects -
  name, EUs, memory, capabilities, vector widths.

The code paths (what "which kernel" means):

- mmvq (mul_mat_vec_q): quantized weights x narrow activations. Dequantizes
  weight blocks inside the matvec - the default fast path for small ncols.
  Home of the XMX vec path and our Q4_0 trait.
- mmq (mul_mat_q): quantized weights x wider activations - the integer
  dp4a dot-product kernel.
- dmmv / dequant+GEMM: the slow path - dequantizes to fp16/fp32 first then
  normal GEMM. What "sucks" was avoiding.
- reorder_qw_*: one-time weight repack to SOA at load. reorder_qw_q4_0
  existed already - the kernel trait was the missing half.
- xmx_traits_<type>: ESIMD dequant+DPAS specialization per quant type.
  xmx_traits_q4_0 added: min-kind epilogue computes w = d*q - 8*d.
- FATTN kernel variants (best_fattn_kernel): the FA backends - DPAS
  (q4_0/q8_0 KV decode on Xe2, XMX), ONEDNN (native F16 SDPA), MKL (XMX
  GEMM prefill), TILE (generic), VEC (decode fallback). F16-KV decode has
  NO XMX kernel - VEC/TILE only.
- reorder_mmvq / reorder_esimd / reorder_dmmv: per-algo gates listing which
  quant types reorder supports - Q4_0 is in reorder_mmvq but NOT
  reorder_esimd.

State / what the numbers mean:

- N_KV: KV cache length - total tokens kept for attention across sequences.
  We varied it to test whether the B=16 collapse tracks attention context;
  the diagnostic below did not show that relationship.
- opt_feature.reorder / onednn_optimized_gemm: per-device feature bits -
  whether the backend repacks weights for XMX, whether oneDNN GEMM is on.
- nsm: Xe core count feeding blocks_per_wave/workgroup math.
- mm: the dispatch is a chain of can_use_* gates by shape/type - vec vs
  xmx_batch vs mmq vs batched sycl, first to say yes wins.

The stack (bottom to top):

- Level Zero: the low-level GPU driver API - the actual hardware interface.
  The SYCL "ext_oneapi_level_zero" backend sits on it. The real path to Arc.
- OpenCL: the other backend - CPU and GPU devices both live here (Xeon =
  opencl:cpu). Can't answer the deep ext_intel_* topology queries.
- SYCL runtime (DPC++): the device abstraction - platforms, devices,
  queues, USM. Compiles kernels to SPIR-V, driver JITs or AOT-compiles.
- oneAPI: the umbrella - compiler (icx/icpx), SYCL runtime, libraries
  (oneDNN, MKL). "oneAPI" is the whole thing, not a layer.
- ESIMD: Intel's explicit-SIMD extension - register-level vector control.
  The XMX kernels are written in it; that's how they feed DPAS directly.
- DPCT: the CUDA->SYCL porting toolkit - dpct::dev_mgr and the CUDA-shaped
  device_info are its artifacts; why the backend looks like CUDA.
- oneDNN / MKL: library-level GEMM/attention - the ONEDNN FA path and
  onednn_optimized_gemm flag use them.

Env vars:

- ONEAPI_DEVICE_SELECTOR: limits which devices dev_mgr sees.
- ZE_AFFINITY_MASK: Level-Zero visibility mask.
- GGML_SYCL_ENABLE_XMX_ESIMD=0: force the XMX ESIMD path off (bench control).
- GGML_SYCL_KERNEL_DIAG=1: enable FAPICK/MMPICK/MATMULPICK stderr diagnostics.
- GGML_SYCL_XMX_MAX_COLS (default 80): widest ncols the mmvq-xmx path takes.
- GGML_SYCL_FORCE_MMQ / ENABLE_MKL_FA / MKL_FA_DIAG: kernel force/diag
  overrides.

## How to reproduce

All commands assume the SYCL build below and run from the repository root.
`<model.gguf>` means any quantized GGUF; results above used a 4B Q4_K_M and
a 31B Q4_0.

- Batched decode scaling (the B=16 collapse table):
    ./bin/llama-batched-bench -m <model.gguf> --device SYCL0 -ngl 99 \
        -fa on -c 8192 -npp 128 -ntg 128 -npl 1,2,4,8,16
  Rerun with GGML_SYCL_ENABLE_XMX_ESIMD=0 for the non-XMX column. -npl is
  the parallel-sequence count = the batched decode width.
- Cold vs warm Q4_0 path (needs the temporary MMPICK/MATMULPICK prints):
    ./bin/llama-batched-bench -m <model.gguf> --device SYCL0 -ngl 99 \
        -fa on -c 8192 -npp 128 -ntg 32 -npl 16
    ./bin/llama-batched-bench -m <model.gguf> --device SYCL0 -ngl 99 \
        -fa on -c 8192 -npp 128 -ntg 32 -npl 8,16
  The first tests cold B=16. The second runs B=8 before B=16 in one process,
  allowing the smaller batch to reorder weights. Do not use these instrumented
  runs for performance comparisons.
- KV-length diagnostic (fixed B=16 and TG=32; q8_0 KV to fit VRAM):
    ./bin/llama-batched-bench -m <model.gguf> --device SYCL0 -ngl 99 \
        -fa on --cache-type-k q8_0 --cache-type-v q8_0 -c 9216 \
        -npp 32,128,512 -ntg 32 -npl 16
- Standard pp/tg:
    ./bin/llama-bench -m <model.gguf> --device SYCL0 -ngl 99 -fa on \
        -p 512 -n 128 -r 2
  Small-batch pp: -p 16,32,64 -n 32.
- Kernel picks (needs the temporary fprintf prints in the tree):
    ./bin/llama-cli -lv 3 -m <model.gguf> --device SYCL0 -ngl 99 -fa on \
        --cache-type-k q8_0 --cache-type-v q8_0 -f <prompt.txt> -n 16
  stderr shows [FAPICK] per FA call and [MMPICK] for the first 30 matmuls.
  q8_0 KV cache is required for the DPAS FA path.
- Correctness check for a new quant kernel:
    ./bin/llama-perplexity -m <model.gguf> --device SYCL0 -ngl 99 \
        -f <corpus.txt> -c 1024 --chunks 20 -b 8 -ub 8
  For Q4_0 XMX, set GGML_SYCL_KERNEL_DIAG=1 and confirm MMPICK reports
  `xmx-dispatched`; compare PPL with XMX disabled. Interpret differences
  against the reported PPL uncertainty, not as a bitwise equality check.
- Backend op assignment (CPU fallbacks):
    GGML_SCHED_DEBUG=2 ./bin/llama-bench -m <model.gguf> --device SYCL0 \
        -ngl 99 -p 128 -n 32
  With the [SCHED] stderr print: every op shows "node #i OP name -> BACKEND".
- Topology dump only (no model needed):
    ./bin/llama-ls-sycl-device
  Triggers ggml_sycl_init and prints the full per-device topology.

Caveats found while running these:
- ggml-level logs do not reach llama-cli/server output even at -lv 3 - the
  fprintf(stderr) prints are why they were added (env diag vars like
  GGML_SYCL_MKL_FA_DIAG are useless for the same reason).
- llama-cli interactive UI still hides generated text sometimes; use
  llama-perplexity for correctness, not token output.
- -fa needs a value: -fa on. Default is auto.
- 31B model load takes ~60-90s on this card; plan waits accordingly.

## Useful env / tooling

- `GGML_SYCL_MKL_FA_DIAG=1` exists but prints via GGML_LOG_INFO -> invisible
  in tools; the fprintf prints above are the working substitute.
- `llama-ls-sycl-device` triggers ggml_sycl_init -> shows topology dump.
- Build from the repository root (in-source):
    source /opt/intel/oneapi/setvars.sh --force --silent
    cmake . -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_SYCL=ON \
        -DLLAMA_BUILD_TESTS=OFF
    cmake --build . --config Release --parallel $(nproc)
- ONEAPI_DEVICE_SELECTOR / ZE_AFFINITY_MASK shape what dev_mgr can see.

## Q4_0 XMX matmul (added)

- `xmx_traits_q4_0` in esimd.hpp (mirrors `xmx_traits_q8_0`): reorder layout
  [qs: nrows*K/2][d: half per 32]; packed u32 k -> out u32 k (low nibbles) and
  u32 k+4 (high nibbles); -8 offset via xmx_scale_kind::min (m = 8*d).
- Wired: XMX_RUN dispatch in mmvq.cpp + xmx_supports_type in common.hpp.
- reorder_qw_q4_0 already existed and was dispatched.
- Historical check: llama-cli prompt eval on gemma-4-E4B Q4_0 reported
  413.8 vs 170.4 t/s (XMX on/off). The earlier identical-PPL result came from
  `llama-perplexity` built before the Q4_0 XMX trait; it did not verify the
  new kernel.
- Rebuilt correctness check: `llama-perplexity` on gemma-4-E4B Q4_0,
  `-c 1024 --chunks 20 -b 8 -ub 8`: XMX on PPL 3.7856 +/- 0.10649; off
  3.7903 +/- 0.10651. MMPICK logged 30 `xmx-dispatched` statuses on and 0
  off. The estimates agree within uncertainty; this is not bitwise proof.
- Gotcha found: xmx_load_t is a TRANSPOSED 2D load - simd idx = kd*16+n
  (k-major), so unpack selects are contiguous 16-elem blocks per u32, not
  strided columns.

## Historical 31B Q4_0 bench (stale executable)

  XMX on  vs  XMX off (GGML_SYCL_ENABLE_XMX_ESIMD=0):
    pp512  214.9 vs 212.8 t/s
    pp64    57.6 vs  57.6 t/s
    pp32    28.9 vs  28.9 t/s
    pp16    14.5 vs  14.5 t/s
    tg      25.5 vs  25.3 t/s

  These numbers came from `llama-bench` built before the Q4_0 XMX trait was
  added. That executable could not test the new Q4_0 kernel, so the on/off
  similarity is not evidence that the Q4_0 XMX path is neutral. Do not use
  these values to evaluate the new path.

- `[SCHED]` finding: GGML_SCHED_DEBUG=2 on the 4B model showed 26473 ops ->
  SYCL0, 46 -> CPU; the only CPU ops are GET_ROWS on model.input_embed
  (embedding lookup, standard host-side split). No compute ops fall back to
  CPU - no cross-backend stall hiding in the schedule.

## Batched decode scaling (llama-batched-bench, tg t/s, -fa on, SYCL0)

qwen35 4B Q4_K_M:
    B   XMX on   XMX off
    1    117.8    121.8
    2    220.5    191.4
    4    369.0    291.3
    8    554.5    330.8
   16    762.4    186.5    <- non-XMX path regresses below batch-1 rate

gemma4 31B Q4_0 (old, unrebuilt `llama-batched-bench`; invalid for Q4_0 XMX):
    B   XMX on   XMX off
    1     24.3     24.0
    2     47.5     47.0
    4     79.9     78.7
    8     98.4     96.7
   16     31.6     31.6

The executable predated the Q4_0 XMX trait; it did not measure the new path.

gemma4 31B Q4_0 (rebuilt executable, warm B=8 then B=16, F16 KV):
    B   XMX on run 1   XMX on run 2   XMX off run 1   XMX off run 2
    8       177.13         177.19          96.24           95.49
   16       309.64         307.79          31.58           31.58

Each run used `-npp 128 -ntg 128 -npl 8,16 -c 8192 -fa on`. The two runs
per XMX setting had zero FAPICK/MMPICK/MATMULPICK lines; diagnostics were
disabled for timing.

This comparison was repeated twice per setting. `bin/llama-batched-bench`
was rebuilt after the Q4_0 source change; older 31B batched-bench results
used a stale executable and are not comparable.

What these numbers show:
- XMX is the difference between scaling and collapsing at concurrency. On the
  4B, XMX decode keeps scaling to 762 t/s at B=16 while the non-XMX path peaks
  at 330 (B=8) then regresses below batch-1 rate - it hits a resource wall
  (registers/occupancy) once ncols gets wide. Batch-1 XMX is neutral/slightly
  slower, so single-stream users see nothing; servers see 4x.
- On the rebuilt 31B Q4_0 run, XMX reaches 309 t/s at B=16; with XMX disabled,
  throughput is 31.6 t/s. The earlier equal 31.6 results came from a stale
  executable, not a neutral Q4_0 XMX kernel.
- The clean warm comparison now confirms the Q4_0 XMX path is load-bearing:
  about 1.85x at B=8 and 9.8x at B=16. F16-KV decode FA remains TILE at both
  widths, so the measured gain is consistent with the matmul path change.
- A cold B=16 run still misses XMX because weights are not yet reordered.
  Running B=8 first reorders them; the subsequent B=16 call dispatches XMX.
  This is a separate cold-start dispatch issue, not the warm B=16 collapse.

What it leads to:
- Q4_0 XMX matters for concurrent decode on this 31B model; the old benchmark
  missed the kernel because its executable was not rebuilt after the change.
- The pre-Q4_0-XMX KV-length sweep stayed flat across N_KV=2560..8192. It
  tested the generic Q4_0 path, not the newly enabled XMX path.
- Cold B>8 requests may run generic SYCL until a smaller-batch call reorders
  the weights; assess whether the backend should reorder proactively.
- Dispatch map for Q4_0 (verified): reorder_mmvq covers it (weights reorder
  at load); mmvq path handles ncols<=8 or <=80 via can_use_xmx_batch (needs
  xmx_supports_type - now includes Q4_0); reorder_esimd list does NOT cover
  Q4_0 (Q2_K..Q6_K, Q8_0 only); wider/other shapes go mul_mat_q or
  mul_mat_batched_sycl.

## KV-length diagnostic (31B Q4_0, B=16)

Method: hold batch size and generated-token count fixed; vary prompt length
to change the initial KV length. Use q8_0 K/V cache, which matches the server
configuration and needs less VRAM than F16:

    ./bin/llama-batched-bench -m <model.gguf> --device SYCL0 -ngl 99 \
        -fa on --cache-type-k q8_0 --cache-type-v q8_0 -c 9216 \
        -npp 32,128,512 -ntg 32 -npl 16

Results:

    prompt   B   N_KV    S_TG t/s
       32   16   1024       13.67
      128   16   2560       13.66
      512   16   8704       13.98

Interpretation (this executable predated the Q4_0 XMX trait):
- Throughput is effectively flat across these KV lengths on the generic
  non-XMX Q4_0 path. This is not a test of the new Q4_0 XMX path.
- This comparison generates only 32 tokens per sequence, so treat it as a
  diagnostic clue, not a final performance characterization. TG duration is
  about 37 seconds for each row.
- With the default F16 cache, allocation failed before any rows at both
  -c 9216 and -c 16384. The q8_0 sweep completed at -c 9216.

Follow-up using the default cache and the original 128-token TG length:

    ./bin/llama-batched-bench -m <model.gguf> --device SYCL0 -ngl 99 \
        -fa on -c 8192 -npp 32,128,384 -ntg 128 -npl 16

    prompt   B   N_KV    S_TG t/s
       32   16   2560       13.67
      128   16   4096       13.65
      384   16   8192       31.10

The ascending-order result was anomalous: the first two rows took ~150 seconds
and the last ~66 seconds. Repeated in reverse order, including a repeated
prompt=128 point:

    ./bin/llama-batched-bench -m <model.gguf> --device SYCL0 -ngl 99 \
        -fa on -c 8192 -npp 384,128,32,128 -ntg 128 -npl 16

    prompt   B   N_KV    S_TG t/s
      384   16   8192       31.09
      128   16   4096       31.54
       32   16   2560       31.54
      128   16   4096       31.54  <- repeated point

The reverse-order run is stable: the repeated point matches exactly, and
throughput varies by only ~1.4% from N_KV=2560 to 8192. The earlier slow rows
did not reproduce, so their timing was an order/device-state artifact, not
evidence that shorter context is slower. These results characterize the old
generic path only; do not use them to explain or rule out the rebuilt XMX path.

The default F16 cache ran at -c 8192; its allocation failures were only at
-c 9216 and -c 16384. The B=8/B=16 dispatch check follows.

## Matmul/attention path comparison (31B Q4_0, default F16 KV)

Built `llama-batched-bench` after adding `[MATMULPICK]`; ran B=8 and B=16
separately with `-npp 128 -ntg 128`. These diagnostic builds print thousands
of FAPICK lines, so their timing results are not valid for performance
comparison.

Cold B=16 (separate process):
- No MMPICK calls at all. The first 24 Q4_0 dispatches show route=SYCL,
  xmx_batch=0, reordered=0, with X columns=16. The generic SYCL path runs;
  the XMX MMVQ entry point is not reached.
- Decode FA is TILE for F16 K/V.

B=8 (separate process):
- MMPICK reports Q4_0, ncols=8, XMX enabled/type supported, and reordered
  weights. The XMX path is attempted.
- Decode FA is TILE for F16 K/V. Some prefill shapes select MKL instead.

Warm B=8 then B=16 in one process (`-npl 8,16`, `-npp 128 -ntg 32`):
- MMPICK reports ncols=16 for B=16, XMX enabled/type supported, reordered=1,
  and status=xmx-dispatched (60 logged dispatches across the narrow/wide
  counters).
- This confirms state-dependent selection: a prior B<=8 MMVQ call reorders
  weights; the cold B=16 run does not reach that reorder path. At ncols=16,
  `can_use_mul_mat_vec_q` depends on `can_use_xmx_batch`, which requires the
  tensor's reorder flag. Without it, the call falls through to generic SYCL.
- This explains why a cold B=16 request can miss XMX. The clean benchmark
  below confirms that the warm B=8 -> B=16 path gains from XMX.
- F16-KV decode remains TILE at both widths. The F16 decode XMX gap is real,
  but this trace does not show that it limits the warm Q4_0 result.

## Open threads

Done:

- [done, 1f8fec351] Topology probe - full DeviceProp-parity, index-aligned,
  usable vs visible tagging, full dump at init. All fields collected.
- [done, 1f8fec351] nsm fix - EUs/16 -> EUs/eu_per_subslice (32 on B70).
  DPAS *2 compensation removed.
- [done, 1f8fec351] caps-gated XMX enable - unknown arch -> has_subgroup_mma
  + SG16 + 2d_block_io; warns once.
- [done, 4a4de9d24] Q4_0 XMX matmul - xmx_traits_q4_0 + dispatch + type gate.
  Compiles and dispatches in the warm batch test; rebuilt PPL runs agree
  within uncertainty, with XMX dispatch confirmed.
- [done, 4a4de9d24 + clean benchmark] Q4_0 batched decode - on a rebuilt
  binary, warm B=8 -> B=16 reaches 309 t/s with XMX versus 31.6 without.
  About 9.8x at B=16; confirms the Q4_0 trait matters under concurrency.

Open:

1. F16-KV decode FA kernel gap - no XMX path for F16 cache (VEC/TILE only).
   The most common decode config; a kernel-size job.
2. Cold B>8 dispatch - a cold B=16 run misses XMX until a smaller MMVQ call
   reorders the weights. Decide whether to reorder proactively or accept this
   cold-start fallback.
3. Launch params from topology - occupancy/workgroup sizing (hw_threads_per_eu,
   local_mem, cache line) not wired yet.
4. FA eligibility from caps - dpas_supported/onednn gating could consume
   topology fields instead of shape/env only.
5. Init-time DPAS correctness self-test - promote "capable" to "verified" for
   unknown arches (the silent-wrong-result guard).
6. Debug-print cleanup - FAPICK/MMPICK/SCHED still in tree; remove when done.
7. Deep enumeration rework - multi-platform dev_mgr, only if device selection
   ever goes wrong (was deferred - shallow is fine until then).
