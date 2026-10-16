/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// PcieDeviceAccess: the module-native receiver's fd on the endpoint's character device node.
// The node is derived by the caller from the channel's peer_ip (the PCI function address,
// "DDDD:BB:DD.F") through hololink/module/pcie/pcie_sysfs.hpp device_node_for_bdf(); minors
// follow probe order, so the BDF, not the node, is what identifies the endpoint. Port of
// the reference bring-up tree's pcie_device.cpp (hsb_pcie/hsdk5_pcie_demo/src/pcie); the only
// place in hololink_module/host/pcie that includes the kernel driver's ABI.

#include "hololink/module/pcie/pcie_device_access.hpp"

#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <fmt/format.h>

#include <hololink/module/pcie/pcie_sysfs.hpp>

#include <hololink/module/logging.hpp>
#include <hololink/module/pcie/hololink_pcie_uapi.h>

namespace hololink::module::pcie {

// The ioctl encoding embeds the argument size, so a struct that drifts from the kernel's
// definition surfaces as ENOTTY rather than as corruption. Pin the layouts here too.
static_assert(sizeof(struct hl_pcie_identity_info) == 64, "hl_pcie_identity_info layout changed");
static_assert(sizeof(struct hl_pcie_dmabuf) == 32, "hl_pcie_dmabuf layout changed");
static_assert(sizeof(struct hl_pcie_msix_wait) == 16, "hl_pcie_msix_wait layout changed");
static_assert(sizeof(struct hl_pcie_msix_ctrl) == 32, "hl_pcie_msix_ctrl layout changed");

namespace {
    constexpr const char* DEVICE_PREFIX = "/dev/" HL_PCIE_DEVICE_NAME;
} // namespace

const char* to_string(PcieStatus status)
{
    switch (status) {
    case PcieStatus::SUCCESS:
        return "SUCCESS";
    case PcieStatus::TIMEOUT:
        return "TIMEOUT";
    case PcieStatus::INVALID_ADDRESS:
        return "INVALID_ADDRESS";
    case PcieStatus::INVALID_DEVICE:
        return "INVALID_DEVICE";
    case PcieStatus::UNSUPPORTED:
        return "UNSUPPORTED";
    case PcieStatus::FAILED:
        return "FAILED";
    }
    return "?";
}

uint16_t PcieDeviceAccess::minor_from_path(const std::string& path)
{
    const std::string prefix(DEVICE_PREFIX);
    if (path.compare(0, prefix.size(), prefix) != 0 || path.size() == prefix.size()) {
        throw std::runtime_error(fmt::format(
            "'{}' is not a hololink_pcie device node (expected {}<minor>)", path, prefix));
    }
    char* end = nullptr;
    const unsigned long minor = std::strtoul(path.c_str() + prefix.size(), &end, 10);
    if (!end || *end != '\0' || minor > 0xFFFF) {
        throw std::runtime_error(fmt::format(
            "'{}' is not a hololink_pcie device node (bad minor)", path));
    }
    return static_cast<uint16_t>(minor);
}

std::string PcieDeviceAccess::discover_bdf(uint16_t minor)
{
    // Port of transport.discover_requester_id: the sysfs class link resolves to the PCI
    // device directory, whose basename is the BDF.
    const std::string link = fmt::format("/sys/class/{}/{}{}/device",
        HL_PCIE_DEVICE_NAME, HL_PCIE_DEVICE_NAME, minor);
    char resolved[PATH_MAX];
    if (!::realpath(link.c_str(), resolved)) {
        HSB_LOG_WARN("Could not resolve {}: {}", link, std::strerror(errno));
        return {};
    }
    std::string path(resolved);
    const size_t slash = path.find_last_of('/');
    return (slash == std::string::npos) ? path : path.substr(slash + 1);
}

uint16_t PcieDeviceAccess::requester_id_from_bdf(const std::string& bdf)
{
    unsigned domain = 0, bus = 0, device = 0, function = 0;
    if (std::sscanf(bdf.c_str(), "%x:%x:%x.%x", &domain, &bus, &device, &function) != 4) {
        HSB_LOG_WARN("Unparsable PCI BDF '{}'", bdf);
        return 0;
    }
    return static_cast<uint16_t>(((bus & 0xFFu) << 8) | ((device & 0x1Fu) << 3) | (function & 0x07u));
}

PcieDeviceAccess::PcieDeviceAccess(const std::string& path, bool allow_host_staging)
    : path_(path)
    , minor_(minor_from_path(path))
    , allow_host_staging_(allow_host_staging)
{
    fd_ = ::open(path_.c_str(), O_RDWR);
    if (fd_ < 0) {
        throw std::runtime_error(device_node_open_error(path_, errno));
    }
}

PcieDeviceAccess::~PcieDeviceAccess()
{
    if (staged_.handle) {
        release(staged_.handle);
    }
    if (dmabuf_fd_ >= 0) {
        ::close(dmabuf_fd_);
    }
    if (fd_ >= 0) {
        ::close(fd_);
    }
}

PcieDeviceIdentity PcieDeviceAccess::identity()
{
    PcieDeviceIdentity id;
    id.minor = minor_;

    struct hl_pcie_identity_info info { };
    if (::ioctl(fd_, HL_PCIE_IOC_IDENTITY, &info) == 0) {
        id.from_ioctl = true;
        id.domain = info.domain;
        id.bus = info.bus;
        id.devfn = info.devfn;
        id.requester_id = info.requester_id;
        id.minor = info.minor;
        id.hsb_version = info.hsb_version;
        id.hsb_date = info.hsb_date;
        id.hsb_status = info.hsb_status;
        id.flags = info.flags;
        std::memcpy(id.uuid, info.uuid, sizeof(id.uuid));
        id.num_msix_vectors = info.num_msix_vectors;
        id.dma_address_bits = info.dma_address_bits;
        id.bar0_len = info.bar0_len;
        id.bdf = fmt::format("{:04x}:{:02x}:{:02x}.{:x}", info.domain, info.bus,
            info.devfn >> 3, info.devfn & 0x7);
        return id;
    }
    if (errno == ENOTTY || errno == EINVAL) {
        HSB_LOG_INFO("{}: driver has no HL_PCIE_IOC_IDENTITY; using sysfs for the BDF", path_);
    } else {
        HSB_LOG_WARN("{}: HL_PCIE_IOC_IDENTITY failed: {}", path_, std::strerror(errno));
    }

    // Fallback, as LocalDmaDeviceAccess::discover_requester_id: the BDF from the sysfs class
    // link, the requester id derived from it, the vector count from MSIX_INFO.
    id.bdf = discover_bdf(minor_);
    if (!id.bdf.empty()) {
        unsigned domain = 0, bus = 0, device = 0, function = 0;
        if (std::sscanf(id.bdf.c_str(), "%x:%x:%x.%x", &domain, &bus, &device, &function) == 4) {
            id.domain = static_cast<uint16_t>(domain);
            id.bus = static_cast<uint8_t>(bus);
            id.devfn = static_cast<uint8_t>(((device & 0x1Fu) << 3) | (function & 0x7u));
        }
        id.requester_id = requester_id_from_bdf(id.bdf);
    }
    MsixVectorInfo msix;
    if (msix_info(0, false, msix) == PcieStatus::SUCCESS) {
        id.num_msix_vectors = msix.num_vectors;
    }
    if (id.bdf.empty() && id.num_msix_vectors == 0) {
        throw std::runtime_error(fmt::format("{}: could not determine device identity", path_));
    }
    return id;
}

PcieStatus PcieDeviceAccess::import_dmabuf(CUdeviceptr buffer, size_t size, DmaMapping& mapping)
{
    // Export the CUDA allocation as a dma-buf. hololink already does exactly this for the
    // RoCE path (roce_receiver.cpp), where ibv_reg_dmabuf_mr performs the import; here the
    // hololink_pcie driver does it, because the endpoint needs a raw device address rather
    // than an RDMA memory region.
    if (dmabuf_fd_ >= 0) {
        HSB_LOG_ERROR("{}: a dma-buf import is already outstanding; release() it first", path_);
        return PcieStatus::FAILED;
    }
    CUresult result = cuMemGetHandleForAddressRange(&dmabuf_fd_, buffer, size,
        CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
    if (result != CUDA_SUCCESS) {
        const char* name = "";
        cuGetErrorName(result, &name);
        HSB_LOG_ERROR("cuMemGetHandleForAddressRange failed ({}); GPUDirect is required for "
                      "the PCIe data path",
            name);
        dmabuf_fd_ = -1;
        return PcieStatus::FAILED;
    }

    struct hl_pcie_dmabuf request { };
    request.fd = dmabuf_fd_;
    if (::ioctl(fd_, HL_PCIE_IOC_DMABUF_IMPORT, &request) < 0) {
        // The driver separates the endpoint's address-generator constraints from transport
        // failures; each needs a different fix, so report them apart.
        PcieStatus status = PcieStatus::INVALID_ADDRESS;
        switch (errno) {
        case EOPNOTSUPP:
            HSB_LOG_ERROR("dma-buf mapped to multiple segments; the endpoint address "
                          "generator requires one contiguous range");
            break;
        case ERANGE:
            HSB_LOG_ERROR("dma-buf mapping crosses a 4 GiB boundary");
            break;
        case EINVAL:
            HSB_LOG_ERROR("dma-buf mapping is not {}-byte aligned", HL_PCIE_DMABUF_ALIGNMENT);
            break;
        default:
            HSB_LOG_ERROR("dma-buf import failed: {}", std::strerror(errno));
            status = PcieStatus::FAILED;
            break;
        }
        ::close(dmabuf_fd_);
        dmabuf_fd_ = -1;
        return status;
    }

    mapping.handle = request.handle;
    mapping.iova = request.iova;
    mapping.size = request.size;
    HSB_LOG_INFO("{}: mapped frame buffer iova={:#x} size={:#x}", path_, mapping.iova, mapping.size);
    return PcieStatus::SUCCESS;
}

PcieStatus PcieDeviceAccess::alloc_host_staged(size_t size, CUdeviceptr& buffer, DmaMapping& mapping)
{
    // No silent fallback to host memory. On bitfiles whose 3DW/4DW TLP selection is wrong for
    // sub-4 GiB targets, a host-staged buffer is emitted with a 64-bit header and a zero upper
    // address -- malformed by spec; observed to bring the link down. Opt in only when the
    // endpoint is known to emit 3DW headers correctly.
    if (!allow_host_staging_) {
        HSB_LOG_ERROR("{}: host staging is disabled; enable it only on an endpoint known to "
                      "handle sub-4 GiB targets",
            path_);
        return PcieStatus::UNSUPPORTED;
    }
    if (staged_.handle) {
        HSB_LOG_ERROR("{}: a staged buffer is already allocated", path_);
        return PcieStatus::FAILED;
    }
    HSB_LOG_WARN("{}: using host-staged memory; this targets a sub-4 GiB address", path_);

    // Coherent host memory from the driver. DMA_ALLOC picks an IOVA inside the range the
    // endpoint can emit, which matters on bitfiles that truncate the address.
    struct hl_pcie_dma_buf request { };
    request.size = size;
    if (::ioctl(fd_, HL_PCIE_IOC_DMA_ALLOC, &request) < 0) {
        HSB_LOG_ERROR("DMA_ALLOC of {:#x} bytes failed: {}", size, std::strerror(errno));
        return PcieStatus::FAILED;
    }

    // vm_pgoff selects the allocation rather than acting as an offset into it.
    const off_t offset
        = static_cast<off_t>(HL_PCIE_DMA_MMAP_BASE_PGOFF + request.handle) * ::getpagesize();
    void* host = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, offset);
    if (host == MAP_FAILED) {
        HSB_LOG_ERROR("mmap of the DMA allocation failed: {}", std::strerror(errno));
        struct hl_pcie_dma_buf free_request = request;
        ::ioctl(fd_, HL_PCIE_IOC_DMA_FREE, &free_request);
        return PcieStatus::FAILED;
    }

    // Page-locked and device-mapped, so kernels reach it over PCIe with no staging copy.
    CUresult result = cuMemHostRegister(host, size, CU_MEMHOSTREGISTER_DEVICEMAP);
    if (result != CUDA_SUCCESS) {
        HSB_LOG_ERROR("cuMemHostRegister failed: {}", static_cast<int>(result));
        ::munmap(host, size);
        struct hl_pcie_dma_buf free_request = request;
        ::ioctl(fd_, HL_PCIE_IOC_DMA_FREE, &free_request);
        return PcieStatus::FAILED;
    }
    result = cuMemHostGetDevicePointer(&buffer, host, 0);
    if (result != CUDA_SUCCESS) {
        HSB_LOG_ERROR("cuMemHostGetDevicePointer failed: {}", static_cast<int>(result));
        cuMemHostUnregister(host);
        ::munmap(host, size);
        struct hl_pcie_dma_buf free_request = request;
        ::ioctl(fd_, HL_PCIE_IOC_DMA_FREE, &free_request);
        return PcieStatus::FAILED;
    }

    staged_.handle = request.handle;
    staged_.host = host;
    staged_.size = size;
    mapping.iova = request.iova;
    mapping.handle = request.handle;
    mapping.size = size;
    HSB_LOG_INFO("{}: staged frame buffer in host memory iova={:#x} size={:#x} device_ptr={:#x}",
        path_, mapping.iova, size, static_cast<unsigned long long>(buffer));
    return PcieStatus::SUCCESS;
}

PcieStatus PcieDeviceAccess::release(uint64_t handle)
{
    if (!handle) {
        return PcieStatus::SUCCESS;
    }

    // Staged host buffers unwind in the reverse of alloc_host_staged().
    if (handle == staged_.handle) {
        PcieStatus status = PcieStatus::SUCCESS;
        cuMemHostUnregister(staged_.host);
        ::munmap(staged_.host, staged_.size);
        struct hl_pcie_dma_buf free_request { };
        free_request.handle = staged_.handle;
        free_request.size = staged_.size;
        if (::ioctl(fd_, HL_PCIE_IOC_DMA_FREE, &free_request) < 0) {
            HSB_LOG_WARN("DMA_FREE failed: {}", std::strerror(errno));
            status = PcieStatus::FAILED;
        }
        staged_ = StagedBuffer {};
        return status;
    }

    PcieStatus status = PcieStatus::SUCCESS;
    struct hl_pcie_dmabuf request { };
    request.handle = handle;
    if (::ioctl(fd_, HL_PCIE_IOC_DMABUF_RELEASE, &request) < 0) {
        HSB_LOG_WARN("dma-buf release failed: {}", std::strerror(errno));
        status = PcieStatus::FAILED;
    }
    if (dmabuf_fd_ >= 0) {
        ::close(dmabuf_fd_);
        dmabuf_fd_ = -1;
    }
    return status;
}

PcieStatus PcieDeviceAccess::msix_info(uint32_t vector, bool reset_count, MsixVectorInfo& info)
{
    struct hl_pcie_msix_info request { };
    request.vector = vector;
    request.reset_count = reset_count ? 1u : 0u;
    if (::ioctl(fd_, HL_PCIE_IOC_MSIX_INFO, &request) < 0) {
        HSB_LOG_ERROR("{}: MSIX_INFO for vector {} failed: {}", path_, vector, std::strerror(errno));
        return errno == EINVAL ? PcieStatus::INVALID_DEVICE : PcieStatus::FAILED;
    }
    info.msg_addr = request.msg_addr;
    info.msg_data = request.msg_data;
    info.num_vectors = request.num_vectors;
    info.enabled = request.enabled != 0;
    info.irq_count = request.irq_count;
    return PcieStatus::SUCCESS;
}

PcieStatus PcieDeviceAccess::msix_ctrl(uint32_t vector, bool enable, uint32_t& refs)
{
    struct hl_pcie_msix_ctrl request { };
    request.vector = vector;
    request.enable = enable ? 1u : 0u;
    if (::ioctl(fd_, HL_PCIE_IOC_MSIX_CTRL, &request) < 0) {
        if (errno == ENOTTY) {
            HSB_LOG_ERROR("{}: the hololink_pcie driver lacks HL_PCIE_IOC_MSIX_CTRL (kernel-owned "
                          "MSI-X); rebuild and reload it from hsb_pcie/driver",
                path_);
            return PcieStatus::UNSUPPORTED;
        }
        HSB_LOG_ERROR("{}: MSIX_CTRL {} for vector {} failed: {}", path_,
            enable ? "enable" : "disable", vector, std::strerror(errno));
        return errno == EINVAL ? PcieStatus::INVALID_DEVICE : PcieStatus::FAILED;
    }
    refs = request.enabled_refs;
    return PcieStatus::SUCCESS;
}

PcieStatus PcieDeviceAccess::wait_frame(uint32_t vector, uint32_t timeout_ms, uint32_t last_count,
    uint32_t& count)
{
    struct hl_pcie_msix_wait request { };
    request.vector = vector;
    request.timeout_ms = timeout_ms;
    request.last_count = last_count;

    const int result = ::ioctl(fd_, HL_PCIE_IOC_MSIX_WAIT, &request);
    // The driver reports the count even on timeout, so publish it either way.
    count = request.irq_count;
    if (result < 0) {
        if (errno == ETIMEDOUT || errno == EINTR) {
            return PcieStatus::TIMEOUT;
        }
        HSB_LOG_ERROR("{}: MSI-X wait failed: {}", path_, std::strerror(errno));
        return PcieStatus::FAILED;
    }
    return PcieStatus::SUCCESS;
}

} // namespace hololink::module::pcie
