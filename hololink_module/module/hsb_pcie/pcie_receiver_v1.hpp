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

#ifndef HOLOLINK_MODULE_HSB_PCIE_PCIE_RECEIVER_V1_HPP
#define HOLOLINK_MODULE_HSB_PCIE_PCIE_RECEIVER_V1_HPP

/*
 * PcieReceiverV1: the hsb_pcie module's PcieReceiverInterfaceV1, the way
 * module_core::RoceReceiverV1 is the RoCE one. The service object lives for the channel;
 * each start() opens the endpoint node (PcieDeviceAccess) and builds a fresh backing
 * pcie::PcieReceiver -- the ring / MSI-X / slot-scan engine in pcie_receiver.{hpp,cpp} --
 * which close() tears down. The two make_* hooks are where a board-specific module
 * substitutes a subclass (a different metadata layout, a different device access), the
 * role make_receiver() plays on the RoCE side.
 */

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/hsb_pcie/pcie_receiver.hpp"
#include "hololink/module/service.hpp"
#include "hololink/module/status.h"

#include "pcie_device_access.hpp"
#include "pcie_receiver.hpp"
#include "pcie_rx_layout.hpp"

namespace hololink::module::hsb_pcie {

class PcieReceiverV1 : public PcieReceiverInterfaceV1,
                       public Service<PcieReceiverV1> {
public:
    static constexpr const char* type_id = "pcie_receiver.hsb_pcie.v1";
    using Service<PcieReceiverV1>::get_service;
    using Service<PcieReceiverV1>::for_each_type_id;
    using ServiceAlias = PcieReceiverInterfaceV1;

    /// Metadata keys configure() reads (stamped by PcieChannelConfigurationV1::use_sensor).
    static constexpr const char* KEY_PEER_IP = "peer_ip";
    static constexpr const char* KEY_SENSOR = "sensor";
    static constexpr const char* KEY_VECTOR = "pcie_vector";

    PcieReceiverV1() = default;
    ~PcieReceiverV1() override;

    /**
     * Takes the endpoint's address (`peer_ip`, the BDF or a literal /dev/hololink_pcieN
     * node) and the sensor index / vector off the metadata, requires the per-channel
     * anchor to be configured (the application fetches the channel first, as on RoCE),
     * and registers with the per-board Hololink so device_lost() invalidates this
     * receiver. Opens nothing: the device node is resolved and opened in start().
     */
    void configure(const EnumerationMetadata& metadata) override;

    hololink_module_status_t start(uint64_t cu_buffer, size_t cu_buffer_size,
        size_t cu_frame_size, size_t cu_page_size, unsigned pages) override;
    void stop() override;
    void close() override;
    void blocking_monitor() override;
    bool get_next_frame(unsigned timeout_ms, PcieReceiverFrameInfoV1& info) override;
    bool frames_ready() override;
    void set_frame_ready(std::function<void()> callback) override;
    uint64_t frame_buffer() override;
    uint32_t sensor() const override { return sensor_; }
    std::string description() const override;

    // PcieReceiverAttachmentV1: what the channel programs the endpoint with. All read off
    // the backing receiver, so they throw (require_backing) before start() and after close():
    // a default here would be programmed into the endpoint.
    uint64_t iova() const override;
    uint64_t window_bytes() const override;
    uint32_t buf_inc_pages() const override;
    uint32_t num_buffers() const override;
    uint32_t requester_id() const override;
    uint32_t vector() const override { return vector_; }
    bool disable_metadata() const override;

    /// The endpoint address configure() took from the metadata (BDF or node path).
    const std::string& peer() const { return peer_; }

protected:
    /* Override hooks for board-specific modules. The defaults open the node the BDF
     * resolves to through sysfs and build the stock pcie::PcieReceiver. */
    virtual std::shared_ptr<pcie::PcieDeviceAccessInterface> make_device_access(
        const std::string& device_node);
    virtual std::shared_ptr<pcie::PcieReceiver> make_backing(
        std::shared_ptr<pcie::PcieDeviceAccessInterface> access,
        const pcie::RxRingConfig& config,
        uint64_t cu_buffer,
        size_t cu_buffer_size);

private:
    /// The current run's backing, or null; a copy taken under backing_mutex_.
    std::shared_ptr<pcie::PcieReceiver> backing() const;
    /// backing(), or std::runtime_error naming `what` when there is no run.
    std::shared_ptr<pcie::PcieReceiver> require_backing(const char* what) const;

    std::string peer_;
    uint32_t sensor_ { 0 };
    uint32_t vector_ { 0 };
    bool configured_ { false };
    // Guards the handoff of access_ / backing_ / frame_ready_callback_ between the thread
    // that drives start() / close() and the monitor thread (blocking_monitor(), and a
    // frame_ready callback that may close() from there): the operator closes the service
    // before it joins the monitor, so the two run concurrently. Held only for the
    // pointer copy or swap, never across a call into the backing.
    mutable std::mutex backing_mutex_;
    std::shared_ptr<pcie::PcieDeviceAccessInterface> access_;
    std::shared_ptr<pcie::PcieReceiver> backing_;
    std::function<void()> frame_ready_callback_;
};

} // namespace hololink::module::hsb_pcie

#endif // HOLOLINK_MODULE_HSB_PCIE_PCIE_RECEIVER_V1_HPP
