//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_TOPOLOGY_HPP
#define GGML_SYCL_TOPOLOGY_HPP

#include <sycl/sycl.hpp>

#include <array>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include "dpct/helper.hpp"

// Device properties in dpct::dev_mgr order, so Device::device_id matches the
// index the backend uses everywhere else.
namespace ggml_sycl {

struct ExecutionUnit {
    int eu_id;
    int max_work_items;
};

struct Tile {
    int tile_id;
    std::vector<ExecutionUnit> eus;
};

struct SubDevice {
    int sub_device_id;
    std::vector<Tile> tiles;
};

struct Device {
    int                    device_id      = -1; // backend index (dpct::dev_mgr order), -1 when not usable
    bool                   backend_usable = false;
    std::string            name;
    std::string            platform_name;
    std::string            backend_name;
    std::string            vendor;
    std::string            driver_version;
    std::string            version;
    sycl::info::device_type device_type = sycl::info::device_type::automatic;
    uint64_t               architecture      = 0;
    bool                   is_available      = false;
    bool                   is_gpu            = false;
    bool                   is_integrated_gpu = false;
    uint32_t               pci_device_id     = 0;
    std::array<unsigned char, 16> uuid = {};

    int                 max_compute_units         = 0;
    int                 max_work_group_size       = 0;
    int                 max_num_sub_groups        = 0;
    int                 max_clock_frequency       = 0;
    uint32_t            max_work_item_dimensions  = 0;
    size_t              max_parameter_size        = 0;
    uint32_t            address_bits              = 0;
    uint32_t            mem_base_addr_align       = 0;
    uint32_t            partition_max_sub_devices = 0;
    size_t              profiling_timer_resolution = 0;
    std::vector<size_t> sub_group_sizes;

    uint32_t gpu_eu_count              = 0;
    uint32_t gpu_eu_count_per_subslice = 0;
    uint32_t gpu_eu_simd_width         = 0;
    uint32_t gpu_hw_threads_per_eu     = 0;

    // xe hierarchy: stack (chiplet/tile) -> region -> cluster -> xe core -> eu
    uint32_t xe_stack_count          = 0;
    uint32_t xe_regions_per_stack    = 0;
    uint32_t xe_clusters_per_region  = 0;
    uint32_t xe_cores_per_cluster    = 0;
    uint32_t eus_per_xe_core         = 0;
    uint32_t max_lanes_per_hw_thread = 0;

    uint64_t                          global_mem_size           = 0;
    sycl::info::global_mem_cache_type global_mem_cache_type     = sycl::info::global_mem_cache_type::none;
    uint32_t                          global_mem_cache_line_size = 0;
    uint64_t                          global_mem_cache_size     = 0;
    sycl::info::local_mem_type        local_mem_type            = sycl::info::local_mem_type::none;
    uint64_t                          local_mem_size            = 0;
    uint64_t                          max_mem_alloc_size        = 0;
    uint32_t                          memory_clock_rate         = 0;
    uint32_t                          memory_bus_width          = 0;

    std::vector<sycl::info::fp_config> half_fp_config;
    std::vector<sycl::info::fp_config> single_fp_config;
    std::vector<sycl::info::fp_config> double_fp_config;

    uint32_t preferred_vector_width_char   = 0;
    uint32_t preferred_vector_width_short  = 0;
    uint32_t preferred_vector_width_int    = 0;
    uint32_t preferred_vector_width_long   = 0;
    uint32_t preferred_vector_width_float  = 0;
    uint32_t preferred_vector_width_double = 0;
    uint32_t preferred_vector_width_half   = 0;

    uint32_t native_vector_width_char   = 0;
    uint32_t native_vector_width_short  = 0;
    uint32_t native_vector_width_int    = 0;
    uint32_t native_vector_width_long   = 0;
    uint32_t native_vector_width_float  = 0;
    uint32_t native_vector_width_double = 0;
    uint32_t native_vector_width_half   = 0;

    bool has_fp16             = false;
    bool has_fp64             = false;
    bool has_atomic64         = false;
    bool has_2d_block_io      = false;
    bool has_subgroup_mma     = false;
    bool has_subgroup_mma_tf32 = false;
    bool has_bf16_conversions = false;

    std::vector<SubDevice> sub_devices; // populated when tile partitioning is used
};

struct Topology {
    std::vector<Device> devices;
};

inline Device collect_device(const sycl::device & dev, int device_id, bool backend_usable) {
    Device d;
    d.device_id      = device_id;
    d.backend_usable = backend_usable;
    try {
        d.name              = dev.get_info<sycl::info::device::name>();
        d.platform_name     = dev.get_platform().get_info<sycl::info::platform::name>();
        d.backend_name      = get_device_backend_and_type(dev);
        d.vendor            = dev.get_info<sycl::info::device::vendor>();
        d.driver_version    = dev.get_info<sycl::info::device::driver_version>();
        d.version           = dev.get_info<sycl::info::device::version>();
        d.device_type       = dev.get_info<sycl::info::device::device_type>();
        d.is_available      = dev.get_info<sycl::info::device::is_available>();
        d.is_gpu            = dev.is_gpu();
        d.is_integrated_gpu = dev.has(sycl::aspect::ext_oneapi_is_integrated_gpu);

        try {
            d.architecture = static_cast<uint64_t>(
                dev.get_info<sycl::ext::oneapi::experimental::info::device::architecture>());
        } catch (const sycl::exception &) {
        }

        d.max_compute_units          = dev.get_info<sycl::info::device::max_compute_units>();
        d.max_work_group_size        = dev.get_info<sycl::info::device::max_work_group_size>();
        d.max_num_sub_groups         = dev.get_info<sycl::info::device::max_num_sub_groups>();
        d.max_clock_frequency        = dev.get_info<sycl::info::device::max_clock_frequency>();
        d.max_work_item_dimensions   = dev.get_info<sycl::info::device::max_work_item_dimensions>();
        d.max_parameter_size         = dev.get_info<sycl::info::device::max_parameter_size>();
        d.address_bits               = dev.get_info<sycl::info::device::address_bits>();
        d.mem_base_addr_align        = dev.get_info<sycl::info::device::mem_base_addr_align>();
        d.partition_max_sub_devices  = dev.get_info<sycl::info::device::partition_max_sub_devices>();
        d.profiling_timer_resolution = dev.get_info<sycl::info::device::profiling_timer_resolution>();
        d.sub_group_sizes            = dev.get_info<sycl::info::device::sub_group_sizes>();

        d.global_mem_size            = dev.get_info<sycl::info::device::global_mem_size>();
        d.global_mem_cache_type      = dev.get_info<sycl::info::device::global_mem_cache_type>();
        d.global_mem_cache_line_size = dev.get_info<sycl::info::device::global_mem_cache_line_size>();
        d.global_mem_cache_size      = dev.get_info<sycl::info::device::global_mem_cache_size>();
        d.local_mem_type             = dev.get_info<sycl::info::device::local_mem_type>();
        d.local_mem_size             = dev.get_info<sycl::info::device::local_mem_size>();
        d.max_mem_alloc_size         = dev.get_info<sycl::info::device::max_mem_alloc_size>();

        d.half_fp_config   = dev.get_info<sycl::info::device::half_fp_config>();
        d.single_fp_config = dev.get_info<sycl::info::device::single_fp_config>();
        d.double_fp_config = dev.get_info<sycl::info::device::double_fp_config>();

#define GGML_SYCL_VEC_WIDTHS(prefix)                                                                   \
    d.prefix##_vector_width_char   = dev.get_info<sycl::info::device::prefix##_vector_width_char>();   \
    d.prefix##_vector_width_short  = dev.get_info<sycl::info::device::prefix##_vector_width_short>();  \
    d.prefix##_vector_width_int    = dev.get_info<sycl::info::device::prefix##_vector_width_int>();    \
    d.prefix##_vector_width_long   = dev.get_info<sycl::info::device::prefix##_vector_width_long>();   \
    d.prefix##_vector_width_float  = dev.get_info<sycl::info::device::prefix##_vector_width_float>();  \
    d.prefix##_vector_width_double = dev.get_info<sycl::info::device::prefix##_vector_width_double>(); \
    d.prefix##_vector_width_half   = dev.get_info<sycl::info::device::prefix##_vector_width_half>()

        GGML_SYCL_VEC_WIDTHS(preferred);
        GGML_SYCL_VEC_WIDTHS(native);
#undef GGML_SYCL_VEC_WIDTHS

        d.has_fp16     = dev.has(sycl::aspect::fp16);
        d.has_fp64     = dev.has(sycl::aspect::fp64);
        d.has_atomic64 = dev.has(sycl::aspect::atomic64);

#define GGML_SYCL_EXT_PROP(field, prop)                                                              \
    d.field = dev.has(sycl::aspect::ext_intel_##prop)                                                \
                  ? dev.get_info<sycl::ext::intel::info::device::prop>()                             \
                  : 0

        GGML_SYCL_EXT_PROP(gpu_eu_count, gpu_eu_count);
        GGML_SYCL_EXT_PROP(gpu_eu_count_per_subslice, gpu_eu_count_per_subslice);
        GGML_SYCL_EXT_PROP(gpu_eu_simd_width, gpu_eu_simd_width);
        GGML_SYCL_EXT_PROP(gpu_hw_threads_per_eu, gpu_hw_threads_per_eu);
        GGML_SYCL_EXT_PROP(pci_device_id, device_id);
        if (dev.has(sycl::aspect::ext_intel_device_info_uuid)) {
            d.uuid = dev.get_info<sycl::ext::intel::info::device::uuid>();
        }
        GGML_SYCL_EXT_PROP(memory_clock_rate, memory_clock_rate);
        GGML_SYCL_EXT_PROP(memory_bus_width, memory_bus_width);
        GGML_SYCL_EXT_PROP(xe_stack_count, xe_stack_count);
        GGML_SYCL_EXT_PROP(xe_regions_per_stack, xe_regions_per_stack);
        GGML_SYCL_EXT_PROP(xe_clusters_per_region, xe_clusters_per_region);
        GGML_SYCL_EXT_PROP(xe_cores_per_cluster, xe_cores_per_cluster);
        GGML_SYCL_EXT_PROP(eus_per_xe_core, eus_per_xe_core);
        GGML_SYCL_EXT_PROP(max_lanes_per_hw_thread, max_lanes_per_hw_thread);
#undef GGML_SYCL_EXT_PROP

        d.has_2d_block_io       = dev.ext_oneapi_supports_cl_extension("cl_intel_subgroup_2d_block_io");
        d.has_subgroup_mma      = dev.ext_oneapi_supports_cl_extension("cl_intel_subgroup_matrix_multiply_accumulate");
        d.has_subgroup_mma_tf32 = dev.ext_oneapi_supports_cl_extension("cl_intel_subgroup_matrix_multiply_accumulate_tensor_float32");
        d.has_bf16_conversions  = dev.ext_oneapi_supports_cl_extension("cl_intel_bfloat16_conversions");
    } catch (const sycl::exception &) {
        // keep what was collected; the rest stays at defaults
    }
    return d;
}

inline Topology discover_topology() {
    Topology topo;
    dpct::dev_mgr * mgr = nullptr;
    try {
        mgr = &dpct::dev_mgr::instance();
        const auto count = mgr->device_count();
        topo.devices.reserve(count);
        for (unsigned int i = 0; i < count; ++i) {
            topo.devices.push_back(collect_device(mgr->get_device(i), static_cast<int>(i), true));
        }
    } catch (const std::exception &) {
        // dev_mgr throws std::runtime_error when no usable platform is found
        mgr = nullptr;
    }
    // every other SYCL device, on any platform/backend: visible but not usable
    for (const auto & platform : sycl::platform::get_platforms()) {
        try {
            for (const auto & dev : platform.get_devices()) {
                if (mgr != nullptr && mgr->get_device_id(dev) != ~0u) {
                    continue;
                }
                topo.devices.push_back(collect_device(dev, -1, false));
            }
        } catch (const sycl::exception &) {
        }
    }
    return topo;
}

inline const char * device_type_str(sycl::info::device_type t) {
    switch (t) {
        case sycl::info::device_type::cpu:         return "cpu";
        case sycl::info::device_type::gpu:         return "gpu";
        case sycl::info::device_type::accelerator: return "accelerator";
        case sycl::info::device_type::custom:      return "custom";
        case sycl::info::device_type::automatic:   return "automatic";
        case sycl::info::device_type::host:        return "host";
        default:                                 return "?";
    }
}

inline const char * cache_type_str(sycl::info::global_mem_cache_type t) {
    switch (t) {
        case sycl::info::global_mem_cache_type::none:       return "none";
        case sycl::info::global_mem_cache_type::read_only:  return "read_only";
        case sycl::info::global_mem_cache_type::read_write: return "read_write";
        default:                                          return "?";
    }
}

inline const char * local_mem_type_str(sycl::info::local_mem_type t) {
    switch (t) {
        case sycl::info::local_mem_type::none:   return "none";
        case sycl::info::local_mem_type::local:  return "local";
        case sycl::info::local_mem_type::global: return "global";
        default:                               return "?";
    }
}

inline const char * fp_config_str(sycl::info::fp_config c) {
    switch (c) {
        case sycl::info::fp_config::denorm:                          return "denorm";
        case sycl::info::fp_config::inf_nan:                         return "inf_nan";
        case sycl::info::fp_config::round_to_nearest:                return "round_to_nearest";
        case sycl::info::fp_config::round_to_zero:                   return "round_to_zero";
        case sycl::info::fp_config::round_to_inf:                    return "round_to_inf";
        case sycl::info::fp_config::fma:                             return "fma";
        case sycl::info::fp_config::correctly_rounded_divide_sqrt:   return "correctly_rounded_divide_sqrt";
        case sycl::info::fp_config::soft_float:                      return "soft_float";
        default:                                                   return "?";
    }
}

inline const char * yn(bool b) { return b ? "y" : "n"; }

inline void appendf(std::string & s, const char * fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    s += buf;
}

inline std::string join_sizes(const std::vector<size_t> & v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) s += ',';
        s += std::to_string(v[i]);
    }
    return s.empty() ? std::string("-") : s;
}

inline std::string join_fp_config(const std::vector<sycl::info::fp_config> & v) {
    std::string s;
    for (const auto c : v) {
        if (!s.empty()) s += ',';
        s += fp_config_str(c);
    }
    return s.empty() ? std::string("-") : s;
}

inline std::string uuid_str(const std::array<unsigned char, 16> & u) {
    std::string s;
    char   buf[4];
    for (const auto b : u) {
        snprintf(buf, sizeof(buf), "%02x", b);
        s += buf;
    }
    return s;
}

inline std::string describe(const Device & d) {
    std::string s;
    appendf(s, "    %s%s\n", d.name.c_str(), d.backend_usable ? "" : " [not usable by backend]");
    appendf(s, "    platform %s, backend %s, vendor %s, driver %s, sycl %s\n",
            d.platform_name.c_str(), d.backend_name.c_str(), d.vendor.c_str(),
            d.driver_version.c_str(), d.version.c_str());
    appendf(s, "    type %s, arch 0x%llx, pci 0x%04x, igpu %s, avail %s, uuid %s\n",
            device_type_str(d.device_type), (unsigned long long) d.architecture,
            d.pci_device_id, yn(d.is_integrated_gpu), yn(d.is_available), uuid_str(d.uuid).c_str());
    appendf(s, "    compute: %u EUs (%u/slice, simd%u, %u thr/eu), %d CUs, wg %d, %d MHz\n",
            d.gpu_eu_count, d.gpu_eu_count_per_subslice, d.gpu_eu_simd_width,
            d.gpu_hw_threads_per_eu, d.max_compute_units, d.max_work_group_size,
            d.max_clock_frequency);
    appendf(s, "    xe: %u stack x %u regions x %u clusters x %u cores x %u eus, %u lanes/thr\n",
            d.xe_stack_count, d.xe_regions_per_stack, d.xe_clusters_per_region,
            d.xe_cores_per_cluster, d.eus_per_xe_core, d.max_lanes_per_hw_thread);
    appendf(s, "    subgroups: sizes [%s], max %d; dims %u, params %zuB, addr %u-bit, align %uB, timer %zuns\n",
            join_sizes(d.sub_group_sizes).c_str(), d.max_num_sub_groups,
            d.max_work_item_dimensions, d.max_parameter_size, d.address_bits,
            d.mem_base_addr_align, d.profiling_timer_resolution);
    appendf(s, "    mem: global %.1f GiB, cache %.1f MiB %s line %uB, local %.1f KiB %s, max alloc %.1f GiB; %u MHz x %u bit\n",
            (double) d.global_mem_size / (1 << 30), (double) d.global_mem_cache_size / (1 << 20),
            cache_type_str(d.global_mem_cache_type), d.global_mem_cache_line_size,
            (double) d.local_mem_size / 1024.0, local_mem_type_str(d.local_mem_type),
            (double) d.max_mem_alloc_size / (double) (1 << 30), d.memory_clock_rate, d.memory_bus_width);
    appendf(s, "    fp: fp16 %s fp64 %s atomic64 %s; half[%s] single[%s] double[%s]\n",
            yn(d.has_fp16), yn(d.has_fp64), yn(d.has_atomic64),
            join_fp_config(d.half_fp_config).c_str(), join_fp_config(d.single_fp_config).c_str(),
            join_fp_config(d.double_fp_config).c_str());
    appendf(s, "    ext: 2d_block_io %s, mma %s, mma_tf32 %s, bf16 %s; max_sub_devices %u\n",
            yn(d.has_2d_block_io), yn(d.has_subgroup_mma), yn(d.has_subgroup_mma_tf32),
            yn(d.has_bf16_conversions), d.partition_max_sub_devices);
    appendf(s, "    vecw pref %u/%u/%u/%u/%u/%u/%u native %u/%u/%u/%u/%u/%u/%u (c/s/i/l/f/d/h)\n",
            d.preferred_vector_width_char, d.preferred_vector_width_short, d.preferred_vector_width_int,
            d.preferred_vector_width_long, d.preferred_vector_width_float, d.preferred_vector_width_double,
            d.preferred_vector_width_half,
            d.native_vector_width_char, d.native_vector_width_short, d.native_vector_width_int,
            d.native_vector_width_long, d.native_vector_width_float, d.native_vector_width_double,
            d.native_vector_width_half);
    return s;
}

} // namespace ggml_sycl

#endif // GGML_SYCL_TOPOLOGY_HPP
