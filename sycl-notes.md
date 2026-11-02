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
  XMX gate state (enabled/type_ok/extra/reorder/ncols).
  Result: Q4_K/Q5_K/Q6_K -> all gates pass, XMX taken. Plain Q4_0 ->
  type_ok=0 (not in ggml_sycl_xmx_supports_type list) -> non-XMX mmvq path.
  The 31B gemma file is ftype Q4_0 despite "Q4_K_XL" in the name, which is
  why it never hit XMX matmul. Supporting Q4_0 = kernel/reorder work, not a
  flag.
- `[SCHED]` in ggml/src/ggml-backend.cpp: stderr mirror of the per-op
  backend assignment line inside ggml_backend_sched_print_assignments.
  Gated on sched->debug > 1 -> run with GGML_SCHED_DEBUG=2. Prints
  "node #i OP name -> BACKEND" so CPU-fallback ops are visible.

## Bench numbers (Arc B70, llama-bench, -fa on, -ngl 99)

- qwen35 4B Q4_K_M (2.51 GiB): pp512 2051 t/s, tg128 126.2 t/s
- qwen35 4B Q4_K_M with GGML_SYCL_ENABLE_XMX_ESIMD=0: pp512 2049, tg128
  130.4 t/s -> XMX is ~3% SLOWER at batch-1 decode, no benefit here.
  XMX wins (if any) live in wider-batch shapes.
- gemma4 E4B Q4_0 (4.79 GiB, plain Q4_0 -> no XMX path): pp512 1507,
  tg128 99.7 t/s. Different model/size so not a controlled XMX test.
- 31B Q4_0 gemma: ~17-23 t/s decode -> roughly linear with model size vs
  the small models: decode is bandwidth-bound, not misconfigured.
  ~130 t/s on 2.5 GiB implies ~325 GB/s effective on B70.

## Useful env / tooling

- `GGML_SYCL_MKL_FA_DIAG=1` exists but prints via GGML_LOG_INFO -> invisible
  in tools; the fprintf prints above are the working substitute.
- `llama-ls-sycl-device` triggers ggml_sycl_init -> shows topology dump.
- Build: source /opt/intel/oneapi/setvars.sh, icx/icpx, -DGGML_SYCL=ON
  (steps in ./fuck).
- ONEAPI_DEVICE_SELECTOR / ZE_AFFINITY_MASK shape what dev_mgr can see.

## Q4_0 XMX matmul (added)

- `xmx_traits_q4_0` in esimd.hpp (mirrors `xmx_traits_q8_0`): reorder layout
  [qs: nrows*K/2][d: half per 32]; packed u32 k -> out u32 k (low nibbles) and
  u32 k+4 (high nibbles); -8 offset via xmx_scale_kind::min (m = 8*d).
- Wired: XMX_RUN dispatch in mmvq.cpp + xmx_supports_type in common.hpp.
- reorder_qw_q4_0 already existed and was dispatched.
- VERIFY: llama-perplexity on gemma-4-E4B Q4_0, XMX on vs off -> PPL
  identical (2.7832 +/- 0.21621). pp512 ~2.4x faster with XMX (413 vs 170).
- Gotcha found: xmx_load_t is a TRANSPOSED 2D load - simd idx = kd*16+n
  (k-major), so unpack selects are contiguous 16-elem blocks per u32, not
  strided columns.

## Real-world perf on the 31B Q4_0 (llama-bench, SYCL0, -fa on)

  XMX on  vs  XMX off (GGML_SYCL_ENABLE_XMX_ESIMD=0):
    pp512  214.9 vs 212.8 t/s
    pp64    57.6 vs  57.6 t/s
    pp32    28.9 vs  28.9 t/s
    pp16    14.5 vs  14.5 t/s
    tg      25.5 vs  25.3 t/s

  Conclusion: XMX mmvq is perf-NEUTRAL on B70 at every shape tried. Decode is
  bandwidth-bound (16 GiB -> ~400 GB/s effective, near the card ceiling).
  pp16-64 sizes route through other kernels (mmq/gemm), not mmvq-xmx - the
  identical numbers say the same path ran. XMX engages at ncols <= ~2 (per
  MMPICK) and there it neither wins nor loses. Value of the Q4_0 work is
  coverage/correctness parity, not decode speed.

- `[SCHED]` finding: GGML_SCHED_DEBUG=2 on the 4B model showed 26473 ops ->
  SYCL0, 46 -> CPU; the only CPU ops are GET_ROWS on model.input_embed
  (embedding lookup, standard host-side split). No compute ops fall back to
  CPU - no cross-backend stall hiding in the schedule.

## Open threads

- F16-KV decode has no XMX FA kernel (VEC/TILE only).
- Launch params from topology (occupancy/workgroup) not wired yet.
- FA eligibility still shape/env based; dpas_supported/onnednn eligibility
  could consume caps.
- Init-time DPAS correctness self-test (see caveat above).
