/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_OPERATORS_NETWORK_RECEIVER_HPP
#define HOLOLINK_MODULE_OPERATORS_NETWORK_RECEIVER_HPP

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include <cuda.h>
#include <holoscan/holoscan.hpp>

#include "hololink/module/data_channel.hpp"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/status.h"

namespace hololink::module::operators {

/* One transport's network receiver (RoCE / Linux / CoE), wrapping its
 * construct / run / destruct lifecycle behind a transport-agnostic
 * surface. HsbController owns one and cycles it construct→run on connect
 * and destruct on loss; it never talks to the SensorFactory. A
 * NetworkReceiverFactory builds an unconfigured one; construct(metadata)
 * resolves the channel + receiver for a specific device. */
class NetworkReceiver {
public:
    struct Config {
        CUcontext frame_context;
        size_t frame_size;
        size_t page_size;
        uint32_t pages;
        uint32_t queue_size;
        size_t metadata_offset;
        // Invoked by the monitor thread when a frame is ready (wakes the
        // operator's compute condition).
        std::function<void()> frame_ready;
        // Per-key renaming for stamp_metadata; identity by default.
        std::function<std::string(const std::string&)> rename_metadata;
    };

    virtual ~NetworkReceiver() = default;

    /* Resolve + build the receiver for `metadata` and attach it to its
     * data channel (no frames flow yet). */
    virtual hololink_module_status_t construct(
        const hololink::module::EnumerationMetadata& metadata,
        const Config& config)
        = 0;

    /* Start the monitor thread — frames begin arriving and Config.frame_ready
     * fires. */
    virtual hololink_module_status_t run() = 0;

    /* Detach, close, and join; the object is reusable via a fresh
     * construct(). */
    virtual void destruct() = 0;

    /* Advance to the next frame within timeout_ms; false on timeout. On
     * success the frame is exposed via frame_memory() / stamp_metadata().
     * `cuda_stream` is the pipeline stream the operator allocated for this
     * compute (from the Holoscan execution context): a software transport
     * issues its host->device frame copy on it so the copy overlaps downstream
     * work ordered on the same stream, and the operator sets that stream on the
     * emitted tensor. A hardware transport that DMAs straight to device memory
     * ignores it. */
    virtual bool get_next_frame(unsigned timeout_ms, CUstream cuda_stream) = 0;
    virtual bool frames_ready() = 0;

    virtual CUdeviceptr frame_memory() = 0;
    virtual std::shared_ptr<void> frame_buffer_owner() = 0;
    virtual void stamp_metadata(holoscan::MetadataDictionary& metadata) = 0;

    /* The transport-agnostic DataChannelInterfaceV1 for this channel — the
     * device_lost() target on loss. Null until construct() has run. */
    virtual std::shared_ptr<hololink::module::DataChannelInterfaceV1>
    data_channel() = 0;
};

/* Builds an unconfigured NetworkReceiver of one transport. The application
 * supplies one (RoCE / Linux); it is the only transport-specific choice. An
 * abstract class held by shared_ptr (not a std::function) so it passes
 * through Holoscan's Arg/Parameter system intact like SensorFactory — a
 * std::function returning non-void is silently convertible to
 * std::function<void()>, which Holoscan's callback-arg overload would grab and
 * strip the return type from. */
class NetworkReceiverFactory {
public:
    virtual ~NetworkReceiverFactory() = default;
    virtual std::shared_ptr<NetworkReceiver> create() = 0;
};

} // namespace hololink::module::operators

#endif // HOLOLINK_MODULE_OPERATORS_NETWORK_RECEIVER_HPP
