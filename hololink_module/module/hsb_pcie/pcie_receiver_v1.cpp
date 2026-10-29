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

#include "pcie_receiver_v1.hpp"

#include <stdexcept>
#include <utility>

#include <fmt/format.h>

#include <hololink/module/data_channel.hpp>
#include <hololink/module/hololink.hpp>
#include <hololink/module/logging.hpp>

#include "hololink_default.hpp" // module_core::HololinkV1::register_associated
#include "pcie_regs.hpp"
#include "pcie_sysfs.hpp"

namespace hololink::module::hsb_pcie {

namespace regs = ::hololink::pcie;

static_assert(PCIE_RX_METADATA_BYTES == pcie::RX_METADATA_BYTES,
    "the public frame-info block size must match the backing receiver's metadata block");

PcieReceiverV1::~PcieReceiverV1()
{
    close();
}

void PcieReceiverV1::configure(const EnumerationMetadata& metadata)
{
    // The endpoint's address. Validated here so a bad record fails at get_service naming
    // the key; resolved through sysfs and opened in start(), when the device has to exist.
    if (!metadata.contains(KEY_PEER_IP)
        || metadata.get<std::string>(KEY_PEER_IP, std::string {}).empty()) {
        throw std::runtime_error(fmt::format(
            "While configuring PcieReceiverInterface: enumeration metadata has no '{}' key; on "
            "the PCIe transport it carries the endpoint's PCI address (DDDD:BB:DD.F)",
            KEY_PEER_IP));
    }
    const std::string peer = metadata.get<std::string>(KEY_PEER_IP);
    if (!pcie::is_bdf(peer) && !pcie::minor_from_device_node(peer)) {
        throw std::runtime_error(fmt::format(
            "While configuring PcieReceiverInterface: '{}' = '{}' is neither a PCI address "
            "DDDD:BB:DD.F nor a /dev/hololink_pcieN node",
            KEY_PEER_IP, peer));
    }

    // Sensor index and vector, as PcieDataChannelV1::configure reads them. The backing
    // receiver waits on vector == sensor (this board's convention, also what
    // PcieChannelConfigurationV1::use_sensor stamps), so anything else is refused here
    // rather than at the channel's attach.
    const int64_t sensor = metadata.get<int64_t>(KEY_SENSOR);
    if (sensor < 0 || sensor >= static_cast<int64_t>(regs::MSIX_TABLE_ENTRIES)) {
        throw std::runtime_error(fmt::format(
            "While configuring PcieReceiverInterface: metadata '{}' = {} is not a valid sensor "
            "index (0..{})",
            KEY_SENSOR, sensor, regs::MSIX_TABLE_ENTRIES - 1));
    }
    const int64_t vector = metadata.get<int64_t>(KEY_VECTOR, sensor);
    if (vector != sensor) {
        throw std::runtime_error(fmt::format(
            "While configuring PcieReceiverInterface: metadata '{}' = {} but this receiver waits "
            "on the vector equal to its sensor index {}",
            KEY_VECTOR, vector, sensor));
    }

    // Anchor-fetch, as RoceReceiverV1: the application is required to construct the channel
    // for this (serial, data_channel) first via PcieDataChannelInterfaceV1::get_service
    // (metadata). If the anchor exists but has no backing Hololink yet, fail fast with a
    // message that names the misordered call site.
    const std::string channel_id = DataChannelInterfaceV1::locator_id(metadata);
    auto channel = DataChannelInterfaceV1::get_service(this->module(), channel_id.c_str());
    if (!channel->hololink()) {
        throw std::runtime_error(
            "While configuring PcieReceiverInterface: parent DataChannelInterface ("
            + channel_id
            + ") has not been configured -- call "
              "PcieDataChannelInterfaceV1::get_service(metadata) for this channel first");
    }
    // Registered with the per-board Hololink so device_lost() invalidates this receiver.
    module_core::HololinkV1::get_service(this->module(),
        HololinkInterfaceV1::locator_id(metadata).c_str())
        ->register_associated(this);

    peer_ = peer;
    sensor_ = static_cast<uint32_t>(sensor);
    vector_ = static_cast<uint32_t>(vector);
    configured_ = true;
}

hololink_module_status_t PcieReceiverV1::start(uint64_t cu_buffer, size_t cu_buffer_size,
    size_t cu_frame_size, size_t cu_page_size, unsigned pages)
{
    if (!configured_) {
        throw std::runtime_error(
            "While starting the PCIe receiver: configure(metadata) has not run -- obtain the "
            "receiver with PcieReceiverInterfaceV1::get_service(metadata)");
    }
    // Release any previous run -- the operator's stop() pairs close() with a monitor-thread
    // join, so by the time we see a fresh start() the prior monitor has exited and dropping
    // it is safe.
    close();

    pcie::RxRingConfig config;
    config.sensor = sensor_;
    config.frame_bytes = cu_frame_size;
    config.page_bytes = cu_page_size;
    config.num_buffers = pages;
    config.disable_metadata = false;

    // The BDF is authoritative: the node the driver has bound to that PCI function right now.
    const std::string device_node = pcie::resolve_device_node(peer_);
    std::shared_ptr<pcie::PcieDeviceAccessInterface> access = make_device_access(device_node);
    const pcie::PcieDeviceIdentity identity = access->identity();
    HSB_LOG_INFO("PcieReceiverV1: {} bdf={} requester_id={:#x} msix_vectors={} sensor={} "
                 "pages={} page_bytes={:#x} frame_bytes={:#x}",
        device_node, identity.bdf, identity.requester_id, identity.num_msix_vectors, sensor_,
        config.num_buffers, config.page_bytes, config.frame_bytes);

    // Built and started before it is published: nothing observes a half-started run.
    std::shared_ptr<pcie::PcieReceiver> backing = make_backing(access, config, cu_buffer, cu_buffer_size);
    {
        std::lock_guard<std::mutex> lock(backing_mutex_);
        if (frame_ready_callback_) {
            backing->set_frame_ready(frame_ready_callback_);
        }
    }
    if (!backing->start()) {
        // The backing logged why (ring or buffer rejected, mapping refused, vector
        // unavailable, ...). Leave nothing half-started behind.
        return HOLOLINK_MODULE_INIT_FAILED;
    }
    std::lock_guard<std::mutex> lock(backing_mutex_);
    access_ = std::move(access);
    backing_ = std::move(backing);
    return HOLOLINK_MODULE_OK;
}

void PcieReceiverV1::stop()
{
    if (const std::shared_ptr<pcie::PcieReceiver> backing = this->backing()) {
        backing->stop();
    }
}

void PcieReceiverV1::close()
{
    // Tear the run down and return to the pre-start() state: description() falls back to
    // the peer address and frame_buffer() / iova() / ... throw again, instead of reporting
    // a buffer the caller may already have freed. The swap happens under backing_mutex_
    // because the monitor thread may be taking its own copy at the same moment (the
    // operator closes before it joins); the backing's close() then waits for the monitor to
    // leave blocking_monitor() (unless this is the monitor thread itself, which holds its
    // own reference), outside the lock, so dropping the last reference here is safe.
    std::shared_ptr<pcie::PcieReceiver> backing;
    std::shared_ptr<pcie::PcieDeviceAccessInterface> access;
    {
        std::lock_guard<std::mutex> lock(backing_mutex_);
        backing = std::move(backing_);
        access = std::move(access_);
    }
    if (backing) {
        backing->close();
    }
}

void PcieReceiverV1::blocking_monitor()
{
    // Hold the backing for the whole run: a frame_ready callback may close() this receiver
    // from the monitor thread, and close() drops backing_ before the call returns.
    require_backing("run the PCIe receiver monitor thread")->blocking_monitor();
}

bool PcieReceiverV1::get_next_frame(unsigned timeout_ms, PcieReceiverFrameInfoV1& info)
{
    pcie::PcieFrameInfo frame;
    if (!require_backing("read the next frame")->get_next_frame(timeout_ms, frame)) {
        return false;
    }
    info.frame_memory = static_cast<uint64_t>(frame.frame_memory);
    info.metadata_memory = static_cast<uint64_t>(frame.metadata_memory);
    info.frame_number = frame.frame_number;
    info.received_s = frame.received_s;
    info.received_ns = frame.received_ns;
    info.bytes_written = frame.bytes_written;
    info.frames = frame.frames;
    info.page_number = frame.page_number;
    info.dropped = frame.dropped;
    info.flags = frame.flags;
    info.raw_metadata = frame.raw_metadata;
    return true;
}

bool PcieReceiverV1::frames_ready()
{
    return require_backing("ask whether a frame is ready")->frames_ready();
}

void PcieReceiverV1::set_frame_ready(std::function<void()> callback)
{
    std::shared_ptr<pcie::PcieReceiver> backing;
    {
        std::lock_guard<std::mutex> lock(backing_mutex_);
        frame_ready_callback_ = std::move(callback);
        backing = backing_;
    }
    if (backing) {
        backing->set_frame_ready(frame_ready_callback_);
    }
}

uint64_t PcieReceiverV1::frame_buffer()
{
    return static_cast<uint64_t>(require_backing("read the frame buffer address")->frame_buffer());
}

std::string PcieReceiverV1::description() const
{
    std::lock_guard<std::mutex> lock(backing_mutex_);
    return access_ ? access_->description() : peer_;
}

// The attachment values the channel programs the endpoint with: never a default, since a
// 0 iova or window would go straight into PCIE_DP; a caller that attaches before start()
// gets the exception instead.
uint64_t PcieReceiverV1::iova() const { return require_backing("read iova")->iova(); }
uint64_t PcieReceiverV1::window_bytes() const { return require_backing("read window_bytes")->window_bytes(); }
uint32_t PcieReceiverV1::buf_inc_pages() const { return require_backing("read buf_inc_pages")->buf_inc_pages(); }
uint32_t PcieReceiverV1::num_buffers() const { return require_backing("read num_buffers")->num_buffers(); }
uint32_t PcieReceiverV1::requester_id() const { return require_backing("read requester_id")->requester_id(); }
bool PcieReceiverV1::disable_metadata() const { return require_backing("read disable_metadata")->disable_metadata(); }

std::shared_ptr<pcie::PcieDeviceAccessInterface> PcieReceiverV1::make_device_access(
    const std::string& device_node)
{
    return std::make_shared<pcie::PcieDeviceAccess>(device_node);
}

std::shared_ptr<pcie::PcieReceiver> PcieReceiverV1::make_backing(
    std::shared_ptr<pcie::PcieDeviceAccessInterface> access,
    const pcie::RxRingConfig& config,
    uint64_t cu_buffer,
    size_t cu_buffer_size)
{
    return std::make_shared<pcie::PcieReceiver>(std::move(access), config,
        static_cast<CUdeviceptr>(cu_buffer), cu_buffer_size);
}

std::shared_ptr<pcie::PcieReceiver> PcieReceiverV1::backing() const
{
    std::lock_guard<std::mutex> lock(backing_mutex_);
    return backing_;
}

std::shared_ptr<pcie::PcieReceiver> PcieReceiverV1::require_backing(const char* what) const
{
    std::shared_ptr<pcie::PcieReceiver> backing = this->backing();
    if (!backing) {
        throw std::runtime_error(fmt::format(
            "While trying to {}: the PCIe receiver has not been started", what));
    }
    return backing;
}

} // namespace hololink::module::hsb_pcie
