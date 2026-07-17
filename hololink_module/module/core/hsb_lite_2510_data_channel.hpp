/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_CORE_HSB_LITE_2510_DATA_CHANNEL_HPP
#define HOLOLINK_MODULE_CORE_HSB_LITE_2510_DATA_CHANNEL_HPP

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>

#include <fmt/format.h>

#include <hololink/core/data_channel.hpp>
#include <hololink/core/hololink.hpp>
#include <hololink/core/metadata.hpp>
#include <hololink/module/page_size.hpp>

#include "hololink/module/logging.hpp"
#include "hololink/module/service.hpp"

#include "linux_data_channel_default.hpp"
#include "roce_data_channel_default.hpp"

namespace hololink::module::module_core {

constexpr uint32_t DP_ADDRESS_0 = 0x08;
constexpr uint32_t DP_ADDRESS_1 = 0x0C;
constexpr uint32_t DP_ADDRESS_2 = 0x10;
constexpr uint32_t DP_ADDRESS_3 = 0x14;
constexpr uint32_t DP_BUFFER_MASK = 0x1C; // each bit enables a buffer

/* Subclass of the legacy hololink::DataChannel — the hook point for
 * channel-side behaviors that differ on the pre-0x2602 FPGA from the
 * modern HSB-Lite. configure_roce is virtual on the base class for
 * exactly this purpose; specific overrides land here as 2510 hardware
 * verification surfaces them. Mirrors the HsbLite2510RoceReceiver
 * pattern (subclass-of-legacy as the override seam, plugged in via the
 * Publisher's construct_roce_data_channel so RoceDataChannelV1's
 * `backing_->configure_roce(...)` call dispatches through this
 * subclass when the loader picked module/hsb_lite_2510/). */
class HsbLite2510DataChannel : public hololink::DataChannel {
public:
    using hololink::DataChannel::DataChannel;

    // This subclass understands the pre-0x2602 memory map, so it accepts
    // FPGAs back to 0x2510 (the oldest hsb_lite_2510 supports).
    int64_t minimum_hsb_ip_version() const override { return 0x2510; }

    void configure_roce(uint64_t frame_memory, size_t frame_size, size_t page_size, unsigned pages, uint32_t local_data_port) override
    {
        verify_hsb_ip_version();

        // Contract enforcement
        if (frame_memory & (hololink::module::PAGE_SIZE - 1)) {
            throw std::runtime_error(fmt::format("frame_memory={:#x} must be {}-byte aligned.", frame_memory, hololink::module::PAGE_SIZE));
        }
        if (page_size & (hololink::module::PAGE_SIZE - 1)) {
            throw std::runtime_error(fmt::format("page_size={:#x} must be {}-byte aligned.", page_size, hololink::module::PAGE_SIZE));
        }
        size_t aligned_frame_size = hololink::module::round_up(frame_size, hololink::module::PAGE_SIZE);
        size_t metadata_size = hololink::module::PAGE_SIZE;
        size_t aligned_frame_with_metadata = aligned_frame_size + metadata_size;
        if (page_size < aligned_frame_with_metadata) {
            throw std::runtime_error(fmt::format("page_size={:#x} must be at least {:#x} bytes.", page_size, aligned_frame_with_metadata));
        }
        if (pages > 4) {
            throw std::runtime_error(fmt::format("pages={} can be at most 4.", pages));
        }
        if (pages < 1) {
            throw std::runtime_error(fmt::format("pages={} must be at least 1.", pages));
        }
        size_t highest_address = frame_memory + page_size * pages;
#define PAGES(x) ((x) >> 7)
        if (PAGES(highest_address) > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error(fmt::format("highest_address={:#x}; pages={:#x} cannot fit in 32-bits.", highest_address, PAGES(highest_address)));
        }

        // Clearing DP_VP_MASK should be unnecessary-- we should only
        // be here following a reset, but be defensive and make
        // sure we're not transmitting anything while we update.
        hololink_->and_uint32(hif_address_ + DP_VP_MASK, ~vp_mask_);

        const uint32_t header_size = 78;
        configure_common(frame_size, header_size, local_data_port);
        hololink_->write_uint32(vp_address_ + DP_QP, qp_number_);
        hololink_->write_uint32(vp_address_ + DP_RKEY, rkey_);
        hololink_->write_uint32(vp_address_ + DP_ADDRESS_0, (pages > 0) ? PAGES(frame_memory) : 0);
        hololink_->write_uint32(vp_address_ + DP_ADDRESS_1, (pages > 1) ? PAGES(frame_memory + page_size) : 0);
        hololink_->write_uint32(vp_address_ + DP_ADDRESS_2, (pages > 2) ? PAGES(frame_memory + (page_size * 2)) : 0);
        hololink_->write_uint32(vp_address_ + DP_ADDRESS_3, (pages > 3) ? PAGES(frame_memory + (page_size * 3)) : 0);
        hololink_->write_uint32(vp_address_ + DP_BUFFER_MASK, (1 << pages) - 1);

        // Restore the DP_VP_MASK to re-enable the sensor.
        hololink_->or_uint32(hif_address_ + DP_VP_MASK, vp_mask_);
    }

    void unconfigure()
    {
        // This stops transmission.
        hololink_->and_uint32(hif_address_ + DP_VP_MASK, ~vp_mask_);
        // Let any in-transit data flush out.
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        // Clear the ROCE configuration.
        hololink_->write_uint32(vp_address_ + DP_BUFFER_MASK, 0);
        hololink_->write_uint32(vp_address_ + DP_BUFFER_LENGTH, 0);
        hololink_->write_uint32(vp_address_ + DP_QP, 0);
        hololink_->write_uint32(vp_address_ + DP_RKEY, 0);
        hololink_->write_uint32(vp_address_ + DP_ADDRESS_0, 0);
        hololink_->write_uint32(vp_address_ + DP_ADDRESS_1, 0);
        hololink_->write_uint32(vp_address_ + DP_ADDRESS_2, 0);
        hololink_->write_uint32(vp_address_ + DP_ADDRESS_3, 0);
        hololink_->write_uint32(vp_address_ + DP_HOST_MAC_LOW, 0);
        hololink_->write_uint32(vp_address_ + DP_HOST_MAC_HIGH, 0);
        hololink_->write_uint32(vp_address_ + DP_HOST_IP, 0);
        hololink_->write_uint32(vp_address_ + DP_HOST_UDP_PORT, 0);

        // Disable the packetizer program.
        packetizer_program_->disable(*hololink_, sif_address_);
    }
};

/* RoceDataChannelV1 specialized for the 2510 FPGA revision —
 * overrides the make_backing hook so configure() builds a
 * HsbLite2510DataChannel for older FPGAs. */
class HsbLite2510RoceDataChannelV1
    : public RoceDataChannelV1,
      public Service<HsbLite2510RoceDataChannelV1> {
public:
    static constexpr const char* type_id = "roce_data_channel.hsb_lite_2510.v1";
    using Service<HsbLite2510RoceDataChannelV1>::get_service;
    using Service<HsbLite2510RoceDataChannelV1>::for_each_type_id;
    using RoceDataChannelV1::RoceDataChannelV1;
    /* Chain walks through module_core::RoceDataChannelV1 (which itself
     * chains to RoceDataChannelInterfaceV1), then emits this subclass's
     * own type_id. Three keys total. */
    using ServiceAlias = RoceDataChannelV1;

protected:
    std::shared_ptr<hololink::DataChannel> make_backing(
        const hololink::Metadata& legacy_metadata,
        std::function<std::shared_ptr<hololink::Hololink>(const hololink::Metadata&)>
            create_hololink) override
    {
        const auto hsb_ip_version_opt = legacy_metadata.get<int64_t>("hsb_ip_version");
        if (!hsb_ip_version_opt) {
            throw std::runtime_error("No hsb_ip_version supplied in enumeration metadata.");
        }
        const auto hsb_ip_version = *hsb_ip_version_opt;
        HSB_LOG_INFO("hsb_ip_version={:#x}", hsb_ip_version);
        if (hsb_ip_version >= 0x2602) {
            return std::make_shared<hololink::DataChannel>(
                legacy_metadata, std::move(create_hololink));
        }
        return std::make_shared<HsbLite2510DataChannel>(
            legacy_metadata, std::move(create_hololink));
    }
};

/* LinuxDataChannelV1 specialized for the 2510 FPGA revision — the
 * software-transport sibling of HsbLite2510RoceDataChannelV1. The Linux
 * path is RoCEv2 over a UDP socket, so it programs the same data-plane
 * memory map; overriding make_backing builds a HsbLite2510DataChannel
 * for older FPGAs, and its configure_roce (which attach_receiver calls)
 * speaks the 2510 layout. No 2510 receiver subclass is needed: the
 * software receiver reads the BTH PSN and RDMA address straight from the
 * packet, which don't vary by FPGA revision. */
class HsbLite2510LinuxDataChannelV1
    : public LinuxDataChannelV1,
      public Service<HsbLite2510LinuxDataChannelV1> {
public:
    static constexpr const char* type_id = "linux_data_channel.hsb_lite_2510.v1";
    using Service<HsbLite2510LinuxDataChannelV1>::get_service;
    using Service<HsbLite2510LinuxDataChannelV1>::for_each_type_id;
    using LinuxDataChannelV1::LinuxDataChannelV1;
    /* Chain walks through module_core::LinuxDataChannelV1 (which itself
     * chains to LinuxDataChannelInterfaceV1), then emits this subclass's
     * own type_id. Three keys total. */
    using ServiceAlias = LinuxDataChannelV1;

protected:
    std::shared_ptr<hololink::DataChannel> make_backing(
        const hololink::Metadata& legacy_metadata,
        std::function<std::shared_ptr<hololink::Hololink>(const hololink::Metadata&)>
            create_hololink) override
    {
        const auto hsb_ip_version_opt = legacy_metadata.get<int64_t>("hsb_ip_version");
        if (!hsb_ip_version_opt) {
            throw std::runtime_error("No hsb_ip_version supplied in enumeration metadata.");
        }
        const auto hsb_ip_version = *hsb_ip_version_opt;
        HSB_LOG_INFO("hsb_ip_version={:#x}", hsb_ip_version);
        if (hsb_ip_version >= 0x2602) {
            return std::make_shared<hololink::DataChannel>(
                legacy_metadata, std::move(create_hololink));
        }
        return std::make_shared<HsbLite2510DataChannel>(
            legacy_metadata, std::move(create_hololink));
    }
};

} // namespace hololink::module::module_core

#endif // HOLOLINK_MODULE_CORE_HSB_LITE_2510_DATA_CHANNEL_HPP
