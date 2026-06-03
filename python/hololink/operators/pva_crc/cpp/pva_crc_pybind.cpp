// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// PyBind11 wrapper for PVA CRC

#include <hololink/operators/pva_crc/pva_crc.hpp>
#include <memory>
#include <pybind11/pybind11.h>
#include <stdexcept>

namespace py = pybind11;

// Wrapper class to provide Python-friendly API
class PvaCrcPython {
public:
    PvaCrcPython()
        : m_pva(std::make_unique<PvaCrc>())
    {
    }

    PvaCrcPython(const PvaCrcPython&) = delete;
    PvaCrcPython& operator=(const PvaCrcPython&) = delete;
    PvaCrcPython(PvaCrcPython&&) = default;
    PvaCrcPython& operator=(PvaCrcPython&&) = default;
    ~PvaCrcPython() = default;

    // Initialize with 1D buffer size
    void initialize(uint32_t data_size, uint32_t remaining_size = 0, bool is_first_chunk = true, bool use_dual_vpu = false)
    {
        if (data_size % 4 != 0) {
            throw std::invalid_argument("data_size must be a multiple of 4");
        }
        if (data_size < 32768) {
            throw std::invalid_argument("data_size must be >= 32KB");
        }
        if (data_size > 12 * 1024 * 1024) {
            throw std::invalid_argument("data_size must be <= 12MB");
        }

        int32_t result = m_pva->init(data_size, remaining_size, is_first_chunk, use_dual_vpu);
        if (result != 0) {
            throw std::runtime_error("Failed to initialize PVA CRC with error code: " + std::to_string(result));
        }
    }

    // Async API: Launch computation (non-blocking)
    uintptr_t launch_compute(uintptr_t input_ptr)
    {
        if (!m_pva->isInitialized()) {
            throw std::runtime_error("PVA CRC not initialized. Call initialize() first.");
        }

        void* input_gpu = reinterpret_cast<void*>(input_ptr);
        PvaCrcFence* fence = nullptr;

        int32_t result = m_pva->launchCompute(input_gpu, &fence);
        if (result != 0) {
            throw std::runtime_error("Failed to launch PVA CRC computation with error code: " + std::to_string(result));
        }

        return reinterpret_cast<uintptr_t>(fence);
    }

    // Async API: Check if computation is complete and retrieve results
    // Returns: 0 = complete (results copied), 1 = still computing, <0 = error
    int check_results(uintptr_t fence_ptr, uintptr_t output_ptr, int64_t timeout_us = 0)
    {
        if (!m_pva->isInitialized()) {
            throw std::runtime_error("PVA CRC not initialized.");
        }

        if (fence_ptr == 0) {
            throw std::invalid_argument(
                "Invalid fence handle: fence_ptr is null. "
                "Ensure launch_compute() succeeded before calling check_results().");
        }

        if (output_ptr == 0) {
            throw std::invalid_argument(
                "Invalid output buffer: output_ptr is null. "
                "Ensure output GPU buffer is allocated before calling check_results().");
        }

        PvaCrcFence* fence = reinterpret_cast<PvaCrcFence*>(fence_ptr);
        uint32_t* output_gpu = reinterpret_cast<uint32_t*>(output_ptr);

        int32_t result = m_pva->checkResults(fence, output_gpu, timeout_us);
        return result;
    }

    // Async API: Free fence handle
    void free_fence(uintptr_t fence_ptr)
    {
        if (fence_ptr == 0) {
            return;
        }

        PvaCrcFence* fence = reinterpret_cast<PvaCrcFence*>(fence_ptr);
        m_pva->freeFence(fence);
    }

    bool is_initialized() const
    {
        return m_pva->isInitialized();
    }

private:
    std::unique_ptr<PvaCrc> m_pva;
};

PYBIND11_MODULE(pva_crc, m)
{
    m.doc() = "PVA Hardware-Accelerated CRC32 Computation - Async API\n\n"
              "This module provides Python bindings to NVIDIA PVA (Programmable Vision Accelerator)\n"
              "hardware for computing CRC32 checksums on contiguous buffers.\n\n"
              "Usage (asynchronous for pipelines):\n"
              "    crc = pva_crc.PvaCrc()\n"
              "    crc.initialize(data_size=data_size)\n"
              "    fence = crc.launch_compute(input_data.data.ptr)\n"
              "    # Do other work while PVA computes...\n"
              "    result = crc.check_results(fence, output_crc.data.ptr, timeout_us=2000)\n"
              "    if result == 0: crc.free_fence(fence)  # Complete!\n";

    py::class_<PvaCrcPython>(m, "PvaCrc")
        .def(py::init<>(), "Create a new PVA CRC instance")

        .def("initialize", &PvaCrcPython::initialize,
            py::arg("data_size"),
            py::arg("remaining_size") = 0,
            py::arg("is_first_chunk") = true,
            py::arg("use_dual_vpu") = false,
            "Initialize PVA CRC with 1D buffer size (NEW API with dual VPU support).\n\n"
            "Args:\n"
            "    data_size: Size of buffer in bytes (must be multiple of 4, min 32KB, max 12MB)\n"
            "    remaining_size: Bytes remaining after this chunk (for parallel computation), default 0\n"
            "    is_first_chunk: True to apply preconditioning (for first chunk), default True\n"
            "    use_dual_vpu: Enable dual VPU parallel processing (auto-splits across VPU0+VPU1), default False\n\n"
            "Raises:\n"
            "    ValueError: If parameters don't meet PVA constraints\n"
            "    RuntimeError: If initialization fails")

        .def("launch_compute", &PvaCrcPython::launch_compute,
            py::arg("input_ptr"),
            "Launch async CRC computation (non-blocking).\n\n"
            "Args:\n"
            "    input_ptr: GPU memory address (from CuPy array.data.ptr)\n\n"
            "Returns:\n"
            "    int: Fence handle for checking completion\n\n"
            "Raises:\n"
            "    RuntimeError: If not initialized or launch fails")

        .def("check_results", &PvaCrcPython::check_results,
            py::arg("fence_handle"),
            py::arg("output_ptr"),
            py::arg("timeout_us") = 0,
            "Check if computation is complete and retrieve results.\n\n"
            "Args:\n"
            "    fence_handle: Fence from launch_compute()\n"
            "    output_ptr: GPU memory for output CRC (single uint32_t, from CuPy array.data.ptr)\n"
            "    timeout_us: Timeout in microseconds (0=poll, -1=wait forever)\n\n"
            "Returns:\n"
            "    int: 0 if complete (results ready), 1 if still computing, <0 on error\n\n"
            "The output buffer must be sizeof(uint32_t) bytes (single CRC value)")

        .def("free_fence", &PvaCrcPython::free_fence,
            py::arg("fence_handle"),
            "Free fence handle after retrieving results.\n\n"
            "Args:\n"
            "    fence_handle: Fence from launch_compute()")

        .def("is_initialized", &PvaCrcPython::is_initialized,
            "Check if PVA CRC is initialized and ready to process.\n\n"
            "Returns:\n"
            "    bool: True if initialized, False otherwise");
}
