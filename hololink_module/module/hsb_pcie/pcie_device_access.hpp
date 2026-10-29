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

#ifndef HOLOLINK_MODULE_PCIE_PCIE_DEVICE_ACCESS_HPP
#define HOLOLINK_MODULE_PCIE_PCIE_DEVICE_ACCESS_HPP

/**
 * @file
 * @brief The receiver's own handle on /dev/hololink_pcieN for the operations that cannot be
 * expressed as BAR0 register writes: making memory addressable by the endpoint (dma-buf
 * import / driver allocation), MSI-X routing information, and blocking for a completion.
 *
 * Port of the reference bring-up tree's pcie_device.cpp
 * (DmaDeviceAccess / LocalDmaDeviceAccess) for the module-native receive path. Differences:
 * the device is opened by its node path, which the caller derives from the channel's
 * `peer_ip` -- the endpoint's PCI function address "DDDD:BB:DD.F" -- through
 * pcie_sysfs.hpp (device_node_for_bdf(); the BDF is authoritative,
 * minors follow probe order); identity comes from HL_PCIE_IOC_IDENTITY with a sysfs
 * fallback; and the statuses are a small local enum
 * rather than a legacy-core status type so this header pulls in nothing from
 * src/hololink/core. Register access is not here: the hsb_pcie module's
 * PcieDataChannelInterfaceV1 programs the endpoint from the values the receiver exposes.
 */

#include <cstddef>
#include <cstdint>
#include <string>

#include <cuda.h>

#include "hololink/module/enumeration_metadata.hpp"

namespace hololink::module::pcie {

/**
 * @brief The /dev/hololink_pcieN node for a channel's enumeration metadata.
 *
 * `peer_ip` is the endpoint's PCI address "DDDD:BB:DD.F", resolved through sysfs to the
 * node the driver has bound to that function right now (the metadata's pcie_device_node
 * is informational and may be stale after a reload), or, as a convenience spelling, a
 * literal node path that names itself.
 *
 * @throws std::runtime_error naming `peer_ip` when it is absent, empty or neither a BDF
 *      nor a node, and naming the BDF when no hololink_pcie device is bound to it.
 */
std::string device_node_from_metadata(const EnumerationMetadata& metadata);

/**
 * @brief The node for one accepted `peer_ip` spelling: a literal /dev/hololink_pcieN path
 * (returned as is) or a BDF, resolved through sysfs to the node bound to it right now.
 * @throws std::runtime_error naming the peer when it is neither, and naming the BDF when
 *      no hololink_pcie device is bound to it.
 */
std::string resolve_device_node(const std::string& peer);

/// Outcome of a device-access operation.
enum class PcieStatus {
    SUCCESS,
    TIMEOUT, ///< wait_frame() elapsed (or was interrupted) with no completion
    INVALID_ADDRESS, ///< the endpoint's address generator cannot use the buffer
    INVALID_DEVICE, ///< the device lacks what was asked of it (MSI-X off, vector out of range)
    UNSUPPORTED, ///< this implementation does not provide the operation
    FAILED, ///< transport / driver failure; see the log
};

const char* to_string(PcieStatus status);

/**
 * @brief What the driver knows about the endpoint, from HL_PCIE_IOC_IDENTITY.
 *
 * `from_ioctl` is false when an older driver lacked the ioctl and only the sysfs-derived
 * fields (bdf, requester_id, minor) are trustworthy.
 */
struct PcieDeviceIdentity {
    uint16_t domain { 0 };
    uint8_t bus { 0 };
    uint8_t devfn { 0 };
    uint16_t requester_id { 0 }; ///< (bus << 8) | devfn, what PCIE_DP_REQUESTER_ID takes
    uint16_t minor { 0 };
    uint32_t hsb_version { 0 };
    uint32_t hsb_date { 0 };
    uint32_t hsb_status { 0 };
    uint32_t flags { 0 }; ///< HL_PCIE_ID_F_*
    uint8_t uuid[16] {};
    uint32_t num_msix_vectors { 0 };
    uint32_t dma_address_bits { 0 };
    uint64_t bar0_len { 0 };
    bool from_ioctl { false };
    std::string bdf; ///< "DDDD:BB:DD.F"
};

/// A buffer the endpoint can address.
struct DmaMapping {
    uint64_t iova { 0 }; ///< what BUF_BASE_HI / BUF_START are programmed with
    uint64_t handle { 0 }; ///< token for release()
    uint64_t size { 0 };
};

/// MSI-X routing for one vector, from HL_PCIE_IOC_MSIX_INFO.
struct MsixVectorInfo {
    uint64_t msg_addr { 0 };
    uint32_t msg_data { 0 };
    uint32_t num_vectors { 0 };
    bool enabled { false };
    uint32_t irq_count { 0 };
};

/**
 * @brief Abstract device access, so PcieReceiver can be tested with no driver.
 */
class PcieDeviceAccessInterface {
public:
    virtual ~PcieDeviceAccessInterface() = default;

    /// Device identity. Throws std::runtime_error only if nothing at all could be learned.
    virtual PcieDeviceIdentity identity() = 0;

    /**
     * @brief Make a CUDA allocation addressable by the endpoint via dma-buf import.
     * @return SUCCESS, INVALID_ADDRESS for the address-generator constraints (scattered
     *         mapping, misaligned, 4 GiB straddle), FAILED otherwise.
     */
    virtual PcieStatus import_dmabuf(CUdeviceptr buffer, size_t size, DmaMapping& mapping) = 0;

    /// Release a mapping from import_dmabuf(). Idempotent for 0.
    virtual PcieStatus release(uint64_t handle) = 0;

    /**
     * @brief MSI-X routing for `vector`. With `reset_count` the driver zeroes the vector's
     * interrupt counter first, so a following wait_frame() starts from a known baseline.
     */
    virtual PcieStatus msix_info(uint32_t vector, bool reset_count, MsixVectorInfo& info) = 0;

    /**
     * @brief Enable or disable delivery of `vector` for this fd (HL_PCIE_IOC_MSIX_CTRL).
     *
     * The kernel owns the MSI-X table; this is the only way userspace turns a vector on or
     * off. Enables are per fd and refcounted across fds in the driver; `refs` receives the
     * number of fds holding the vector enabled after the call. Disabling waits for a handler
     * in flight, so no completion on the vector can wake a waiter afterwards.
     * @return SUCCESS; UNSUPPORTED when the driver predates the ioctl (ENOTTY);
     *         INVALID_DEVICE for a vector the device does not have; FAILED otherwise.
     */
    virtual PcieStatus msix_ctrl(uint32_t vector, bool enable, uint32_t& refs) = 0;

    /**
     * @brief Block until the endpoint signals a completed frame on `vector`.
     * @param timeout_ms 0 polls.
     * @param last_count Count the caller last observed.
     * @param count Receives the current count whether or not the wait timed out.
     * @return SUCCESS when the count advanced, TIMEOUT otherwise (also for EINTR).
     */
    virtual PcieStatus wait_frame(uint32_t vector, uint32_t timeout_ms, uint32_t last_count,
        uint32_t& count)
        = 0;

    /// Short description for diagnostics, e.g. "/dev/hololink_pcie0".
    virtual std::string description() const = 0;
};

/**
 * @brief PcieDeviceAccessInterface backed by this process's own fd on /dev/hololink_pcieN.
 */
class PcieDeviceAccess : public PcieDeviceAccessInterface {
public:
    /**
     * @param path Device node, e.g. "/dev/hololink_pcie0", as resolved from the channel's
     *        `peer_ip` (BDF) by pcie::device_node_for_bdf().
     * @throw std::runtime_error if the device cannot be opened.
     */
    explicit PcieDeviceAccess(const std::string& path);
    ~PcieDeviceAccess() override;

    PcieDeviceAccess(const PcieDeviceAccess&) = delete;
    PcieDeviceAccess& operator=(const PcieDeviceAccess&) = delete;

    PcieDeviceIdentity identity() override;
    PcieStatus import_dmabuf(CUdeviceptr buffer, size_t size, DmaMapping& mapping) override;
    PcieStatus release(uint64_t handle) override;
    PcieStatus msix_info(uint32_t vector, bool reset_count, MsixVectorInfo& info) override;
    PcieStatus msix_ctrl(uint32_t vector, bool enable, uint32_t& refs) override;
    PcieStatus wait_frame(uint32_t vector, uint32_t timeout_ms, uint32_t last_count,
        uint32_t& count) override;
    std::string description() const override { return path_; }

    /// Minor number from a node path ("/dev/hololink_pcie3" -> 3). Throws on a bad path.
    static uint16_t minor_from_path(const std::string& path);

    /// BDF of the endpoint bound at `minor`, from the sysfs class link. Empty if unresolved.
    static std::string discover_bdf(uint16_t minor);

    /// PCI requester id from a "DDDD:BB:DD.F" string. 0 if unparsable.
    static uint16_t requester_id_from_bdf(const std::string& bdf);

private:
    std::string path_;
    uint16_t minor_ { 0 };
    int fd_ { -1 };
    int dmabuf_fd_ { -1 };
};

} // namespace hololink::module::pcie

#endif /* HOLOLINK_MODULE_PCIE_PCIE_DEVICE_ACCESS_HPP */
