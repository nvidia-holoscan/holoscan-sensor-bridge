/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hololink/module/operators/hsb_controller.hpp"

#include <chrono>
#include <memory>
#include <mutex>
#include <utility>

#include "hololink/module/data_channel.hpp"
#include "hololink/module/logging.hpp"
#include "hololink/module/status.h"

namespace hololink::module::operators {

HsbController::HsbController(std::shared_ptr<SensorFactory> sensor_factory,
    std::shared_ptr<NetworkReceiverFactory> network_receiver_factory,
    NetworkReceiver::Config config,
    hololink::module::EnumerationMetadata metadata,
    double watchdog_timeout_s)
    : sensor_factory_(std::move(sensor_factory))
    , network_receiver_factory_(std::move(network_receiver_factory))
    , config_(std::move(config))
    , metadata_(std::move(metadata))
    , watchdog_timeout_s_(watchdog_timeout_s)
{
}

HsbController::~HsbController()
{
    stop();
}

void HsbController::start(std::function<void()> wake)
{
    wake_ = std::move(wake);
    // A frame wakes both a get_next_frame() waiting for one and the operator's compute().
    config_.frame_ready = [this]() {
        wake_frame_waiter();
        if (wake_) {
            wake_();
        }
    };
    {
        std::lock_guard<std::mutex> lock(receiver_mutex_);
        network_receiver_ = network_receiver_factory_
            ? network_receiver_factory_->create()
            : nullptr;
    }
    if (!sensor_factory_) {
        return;
    }
    sensor_factory_->start(
        [this](const hololink::module::EnumerationMetadata& md) { found(md); },
        [this]() { lost(); },
        metadata_,
        watchdog_timeout_s_);
}

void HsbController::stop()
{
    if (sensor_factory_) {
        sensor_factory_->stop();
    }
    {
        std::lock_guard<std::mutex> lock(receiver_mutex_);
        connected_ = false;
        if (network_receiver_) {
            network_receiver_->destruct();
            network_receiver_.reset();
        }
    }
    wake_frame_waiter();
}

void HsbController::found(const hololink::module::EnumerationMetadata& metadata)
{
    // Reactor thread. Build the receiver from the fresh metadata; on
    // failure clean up and stay disconnected (the watchdog reconnects).
    std::lock_guard<std::mutex> lock(receiver_mutex_);
    if (!network_receiver_) {
        return;
    }
    const hololink_module_status_t status
        = network_receiver_->construct(metadata, config_);
    if (status != HOLOLINK_MODULE_OK) {
        HSB_LOG_ERROR("HsbController::found: receiver construct status {}", status);
        network_receiver_->destruct();
        return;
    }
    network_receiver_->run();
    connected_ = true;
}

void HsbController::lost()
{
    // Reactor thread. Tear the receiver down, invalidate the board, then
    // wake compute() once for a fallback tick.
    std::shared_ptr<hololink::module::DataChannelInterfaceV1> data_channel;
    {
        std::lock_guard<std::mutex> lock(receiver_mutex_);
        connected_ = false;
        if (network_receiver_) {
            data_channel = network_receiver_->data_channel();
            network_receiver_->destruct();
        }
    }
    // A get_next_frame() waiting for a frame returns now, so compute() can show the fallback.
    wake_frame_waiter();
    if (data_channel) {
        data_channel->device_lost();
    }
    if (wake_) {
        wake_();
    }
}

void HsbController::wake_frame_waiter()
{
    {
        std::lock_guard<std::mutex> lock(frame_wait_mutex_);
        ++frame_wakes_;
    }
    frame_waiter_.notify_all();
}

bool HsbController::get_next_frame(unsigned timeout_ms,
    hololink::Metadata& metadata, CUstream cuda_stream,
    CUdeviceptr& out_frame, size_t& out_size,
    std::shared_ptr<void>& out_owner)
{
    // Wait for a frame on frame_waiter_, not in the receiver's get_next_frame() with
    // receiver_mutex_ held: lost() takes that mutex on the reactor thread, which would then
    // stall behind this wait on every disconnect. The receiver is used only under the mutex,
    // and only to take a frame it already holds, so lost() and stop() can tear it down while
    // this waits. frame_wakes_ is read before frames_ready(), so a frame that lands in between
    // still ends the wait.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (connected_.load()) {
        uint64_t wakes = 0;
        {
            std::lock_guard<std::mutex> lock(frame_wait_mutex_);
            wakes = frame_wakes_;
        }
        {
            std::lock_guard<std::mutex> lock(receiver_mutex_);
            if (!connected_.load() || !network_receiver_) {
                return false;
            }
            if (network_receiver_->frames_ready()) {
                // A frame is queued, so this doesn't wait.
                if (!network_receiver_->get_next_frame(0, cuda_stream)) {
                    return false;
                }
                if (sensor_factory_) {
                    sensor_factory_->tap();
                }
                network_receiver_->stamp_metadata(metadata);
                out_frame = network_receiver_->frame_memory();
                out_owner = network_receiver_->frame_buffer_owner();
                out_size = config_.frame_size;
                return true;
            }
        }
        std::unique_lock<std::mutex> lock(frame_wait_mutex_);
        if (!frame_waiter_.wait_until(lock, deadline, [&] { return frame_wakes_ != wakes; })) {
            return false;
        }
    }
    return false;
}

bool HsbController::frames_ready()
{
    std::lock_guard<std::mutex> lock(receiver_mutex_);
    // A receive path that failed reports a frame ready but yields none: report none, and leave
    // the connection to the watchdog, which no frame taps any more, to report lost.
    return network_receiver_ && network_receiver_->frames_ready() && !network_receiver_->receive_path_failed();
}

void HsbController::fallback_frame(CUdeviceptr& out_ptr, size_t& out_size)
{
    out_ptr = 0;
    out_size = 0;
    if (sensor_factory_) {
        sensor_factory_->fallback_frame(out_ptr, out_size);
    }
}

} // namespace hololink::module::operators
