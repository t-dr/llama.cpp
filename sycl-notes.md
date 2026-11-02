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

## Useful env / tooling

- `GGML_SYCL_MKL_FA_DIAG=1` exists but prints via GGML_LOG_INFO -> invisible
  in tools; the fprintf prints above are the working substitute.
- `llama-ls-sycl-device` triggers ggml_sycl_init -> shows topology dump.
- Build: source /opt/intel/oneapi/setvars.sh, icx/icpx, -DGGML_SYCL=ON
  (steps in ./fuck).
- ONEAPI_DEVICE_SELECTOR / ZE_AFFINITY_MASK shape what dev_mgr can see.

## Open threads

- F16-KV decode has no XMX FA kernel (VEC/TILE only).
- Q4_0 has no XMX matmul path.
- Launch params from topology (occupancy/workgroup) not wired yet.
- FA eligibility still shape/env based; dpas_supported/onnednn eligibility
  could consume caps.
- Init-time DPAS correctness self-test (see caveat above).
