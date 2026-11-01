#ifndef GGML_SYCL_TOPOLOGY_H
#define GGML_SYCL_TOPOLOGY_H

#include <vector>
#include <string>
#include <algorithm>
#include <sycl/sycl.hpp>

namespace ggml::sycl {

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

struct QueueContext {
    int queue_id;
    std::string queue_type;
};

struct Device {
    std::string name;
    int device_id;
    bool is_gpu;
    int max_compute_units;
    int max_work_group_size;
    uint64_t global_mem_size;
    std::vector<SubDevice> sub_devices;
};

struct Topology {
    std::vector<Device> devices;
    std::vector<QueueContext> queue_contexts;

    static Topology discover() {
        Topology topo;
        try {
            for (const auto & platform : ::sycl::platform::get_platforms()) {
                for (const auto & device : platform.get_devices()) {
                    auto dev_type = device.get_info<::sycl::info::device::device_type>();
                    
                    if (dev_type != ::sycl::info::device_type::gpu && 
                        dev_type != ::sycl::info::device_type::cpu) {
                        continue; 
                    }

                    Device dev;
                    dev.name = device.get_info<::sycl::info::device::name>();
                    dev.is_gpu = (dev_type == ::sycl::info::device_type::gpu);
                    dev.device_id = 0; 
                    
                    try {
                        dev.max_compute_units = device.get_info<::sycl::info::device::max_compute_units>();
                    } catch (...) {
                        dev.max_compute_units = 0;
                    }

                    try {
                        dev.max_work_group_size = device.get_info<::sycl::info::device::max_work_group_size>();
                    } catch (...) {
                        dev.max_work_group_size = 0;
                    }

                    try {
                        dev.global_mem_size = device.get_info<::sycl::info::device::global_mem_size>();
                    } catch (...) {
                        dev.global_mem_size = 0;
                    }

                    topo.devices.push_back(dev);
                }
            }

            // Sort devices to prioritize GPUs first, then CPUs
            std::sort(topo.devices.begin(), topo.devices.end(), [](const Device &a, const Device &b) {
                return a.is_gpu && !b.is_gpu; 
            });

            // Re-assign clean sequential device IDs based on preference priority
            for (size_t i = 0; i < topo.devices.size(); ++i) {
                topo.devices[i].device_id = static_cast<int>(i);
            }

        } catch (...) {
            // Fallback or empty topology on exception
        }
        return topo;
    }
};

} // namespace ggml::sycl

#endif // GGML_SYCL_TOPOLOGY_H
