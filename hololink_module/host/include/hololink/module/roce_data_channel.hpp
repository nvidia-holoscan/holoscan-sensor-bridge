/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_ROCE_DATA_CHANNEL_HPP
#define HOLOLINK_MODULE_ROCE_DATA_CHANNEL_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "enumeration_metadata.hpp"
#include "module.hpp"
#include "service.hpp"
#include "status.h"

namespace hololink::module {

class HololinkInterfaceV1;
class SequencerInterfaceV1;
class RoceReceiverInterfaceV1;

/**
 * @brief Configures an HSB unit for RoCE data transport.
 *
 * This object configures and HSB device with parameters necessary to
 * generate RoCE compatible data packets, including setting
 * QP and RKEY values.  These values come from the
 * RoceReceiverInterfaceV1 passed in to attach_receiver.
 */
class RoceDataChannelInterfaceV1
    : public ConfigurableService<RoceDataChannelInterfaceV1> {
public:
    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    static constexpr const char* type_id = "roce_data_channel.v1";
    static constexpr const char* release_version = "2.7.0";

    /* Per-channel instance_id derivation used by the metadata-form
     * get_service. Throws if the metadata is missing 'serial_number'
     * or 'data_channel'. */
    static std::string locator_id(const EnumerationMetadata& metadata)
    {
        return "serial=" + metadata.get<std::string>("serial_number")
            + ";data_channel=" + std::to_string(metadata.get<int64_t>("data_channel"));
    }

    virtual ~RoceDataChannelInterfaceV1() = default;
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /**
     * @brief Configure the HSB device with settings necessary
     *      for generating RoCE data packets.
     *
     * @param receiver has RoCE configuration parameters.
     * @return HOLOLINK_MODULE_OK on success, or an error status.
     * @throws std::runtime_error if `receiver` is null.
     */
    virtual hololink_module_status_t attach_receiver(
        std::shared_ptr<RoceReceiverInterfaceV1> receiver)
        = 0;

    /**
     * @brief Deconfigure the RoCE configuration.
     *
     * Disconnects the channel from the memory it was directed at, so the
     * device no longer writes to it. Call this before releasing the
     * receiver's memory. It communicates with the device, so it can
     * fail if the device has become unreachable.
     *
     * @return HOLOLINK_MODULE_OK on success, or an error status.
     */
    virtual hololink_module_status_t detach_receiver() = 0;

    /**
     * @brief The sequencer that the device runs each time a frame
     * on this channel has been completely sent.
     *
     * Receivers add operations to it, such as reading back frame
     * metadata, that must happen at the end of each frame.
     *
     * @tparam T Service type to return; defaults to SequencerInterfaceV1.
     * @param allow_null If `false` (the default), throws
     *      std::runtime_error if the sequencer is not available. If
     *      `true`, returns `nullptr` instead.
     * @returns The sequencer for this channel.
     */
    template <typename T = SequencerInterfaceV1>
    std::shared_ptr<T> frame_end_sequencer(bool allow_null = false)
    {
        const std::string instance_id = frame_end_sequencer_instance_id();
        return T::get_service(module(), instance_id.c_str(), allow_null);
    }

    /**
     * @brief The device (HololinkInterfaceV1) this channel belongs to.
     *
     * Same device as returned by DataChannelInterfaceV1::hololink();
     * provided so that code holding only this interface need not look
     * up the other.
     *
     * @tparam T Service type to return; defaults to HololinkInterfaceV1.
     * @param allow_null If `false` (the default), throws
     *      std::runtime_error if the device is not available. If `true`,
     *      returns `nullptr` instead.
     * @returns The device.
     */
    template <typename T = HololinkInterfaceV1>
    std::shared_ptr<T> get_hololink(bool allow_null = false)
    {
        const std::string instance_id = parent_hololink_instance_id();
        return T::get_service(module(), instance_id.c_str(), allow_null);
    }

    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
protected:
    virtual std::string frame_end_sequencer_instance_id() = 0;
    virtual std::string parent_hololink_instance_id() = 0;
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_ROCE_DATA_CHANNEL_HPP
