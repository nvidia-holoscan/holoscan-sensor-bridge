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

#ifndef HOLOLINK_MODULE_HSB_PCIE_PCIE_DATA_CHANNEL_V1_HPP
#define HOLOLINK_MODULE_HSB_PCIE_PCIE_DATA_CHANNEL_V1_HPP

/**
 * @file
 * @brief PcieDataChannelInterfaceV1 impl: the PCIe counterpart of module_core::RoceDataChannelV1.
 *
 * Has-a relationship with the per-channel anchor (module_core::DataChannelV1, published by
 * HsbIpPublisher::construct_data_channel): holds the shared_ptr<DataChannelInterfaceV1> the
 * publisher hands in, and adds the PCIe transport state -- which sensor / VIP bit / MSI-X
 * vector this channel drives, whether a receiver is attached and armed. Register access is
 * the parent HololinkInterfaceV1's write_uint32 / read_uint32 / or_uint32 / and_uint32
 * surface only, so the module's AddressPolicy applies to the channel like to any client (the
 * PCIE_DP, MSI-X and DP_PKT blocks are all inside the ForwardIdentity BAR0 window).
 *
 * The register sequence in attach_receiver() is the reference receive ring's program() +
 * arm(), carried over verbatim (see docs/pcie_hololink_module_design.md section 3): shared
 * words are touched per sensor only -- DP_PKT_VIP_MASK by read-modify-write of this
 * sensor's bit, the MSI-X table only at this vector's entry -- and every configuration
 * write is read back and verified; only the arm / disarm edges stay posted.
 */

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include "hololink/module/data_channel.hpp"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/hololink.hpp"
#include "hololink/module/hsb_pcie/pcie_data_channel.hpp"
#include "hololink/module/status.h"

namespace hololink::module::hsb_pcie {

class PcieDataChannelV1 : public PcieDataChannelInterfaceV1,
                          public Service<PcieDataChannelV1> {
public:
    static constexpr const char* type_id = "pcie_data_channel.hsb_pcie.v1";
    using Service<PcieDataChannelV1>::get_service;
    using Service<PcieDataChannelV1>::for_each_type_id;
    using ServiceAlias = PcieDataChannelInterfaceV1;

    /// Metadata keys configure() reads (stamped by PcieChannelConfigurationV1::use_sensor).
    static constexpr const char* KEY_SENSOR = "sensor";
    static constexpr const char* KEY_VP_MASK = "vp_mask";
    static constexpr const char* KEY_VECTOR = "pcie_vector";
    static constexpr const char* KEY_DATA_CHANNEL = "data_channel";

    explicit PcieDataChannelV1(std::shared_ptr<DataChannelInterfaceV1> anchor);

    /**
     * Drives the anchor's configure (idempotent via its framework latch), resolves the
     * per-board HololinkV1 (registering with it so device_lost() invalidates this channel
     * too) and reads sensor / vp_mask / pcie_vector off the metadata. Runs once per
     * instance: the channel's addressing is fixed by its (serial, data_channel) key.
     */
    void configure(const EnumerationMetadata& metadata) override;

    hololink_module_status_t attach_receiver(
        std::shared_ptr<PcieReceiverAttachmentV1> receiver,
        uint32_t chunk_pages) override;
    hololink_module_status_t detach_receiver() override;
    hololink_module_status_t arm() override;
    hololink_module_status_t disarm() override;
    hololink_module_status_t diagnostics(PcieChannelDiagnostics& out) override;

    uint32_t sensor() const override { return sensor_; }
    uint32_t vector() const override { return vector_; }
    uint32_t vp_mask() const override { return vp_mask_; }

    /// The per-channel anchor this PCIe wrapper composes with.
    std::shared_ptr<DataChannelInterfaceV1> data_channel() const { return anchor_; }

    bool attached() const;
    bool armed() const;

private:
    /// nullptr when the attachment is usable on this channel, else why it is not.
    const char* validate(const PcieReceiverAttachmentV1& receiver, uint32_t chunk_pages) const;
    /// Write then read back; throws naming `name` on status failure or mismatch.
    void write_verified(uint32_t offset, uint32_t value, const char* name);
    /// Write without readback; throws naming `name` on status failure.
    void write_posted(uint32_t offset, uint32_t value, const char* name);
    uint32_t read_reg(uint32_t offset, const char* name);
    /// Read-modify-write this sensor's bit of DP_PKT_VIP_MASK through the module.
    void set_vip_bit(bool enable);
    /// Throws when configure() has not run (no Hololink to program through).
    void require_configured(const char* what) const;

    std::shared_ptr<DataChannelInterfaceV1> anchor_;
    std::shared_ptr<HololinkInterfaceV1> hololink_;
    std::string instance_;
    uint32_t sensor_ { 0 };
    uint32_t vector_ { 0 };
    uint32_t vp_mask_ { 0 };

    // Serializes attach / detach / arm / disarm on this channel.
    mutable std::mutex mutex_;
    std::shared_ptr<PcieReceiverAttachmentV1> receiver_;
    bool attached_ { false };
    bool armed_ { false };
};

} // namespace hololink::module::hsb_pcie

#endif /* HOLOLINK_MODULE_HSB_PCIE_PCIE_DATA_CHANNEL_V1_HPP */
