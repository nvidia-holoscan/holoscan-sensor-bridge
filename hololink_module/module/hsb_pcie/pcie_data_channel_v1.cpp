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

// PcieDataChannelV1: the PCIe data channel's register programming, ported verbatim from the
// host-side receive ring it replaces (formerly hololink_module/host/pcie, itself a port of
// the reference bring-up tree's PcieRxRing) with the ring's own
// HololinkInterfaceV1 member replaced by the parent Hololink the anchor resolves, and minus
// the ring's MSI-X table writes: the table is generic PCI owned by the kernel, and the
// receiver enables its vector through the driver (HL_PCIE_IOC_MSIX_CTRL). The one shared
// HSB word, DP_PKT_VIP_MASK, is read-modify-written (or_uint32 / and_uint32 on this sensor's
// bit) so a sibling channel is never disturbed.

#include "pcie_data_channel_v1.hpp"

#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include <hololink/module/logging.hpp>
#include <hololink/module/pcie/pcie_regs.hpp>

#include "hololink_default.hpp"

namespace hololink::module::hsb_pcie {

// BAR0 register map.
namespace regs = ::hololink::pcie;

namespace {

    /// Serializes the shared-register sequences (DP_PKT_LEN check + program, DP_PKT_VIP_MASK arm)
    /// of every PcieDataChannelV1 in this process. One board per process is the norm, and the
    /// sequence is a handful of register accesses, so one mutex for all channels is enough.
    std::mutex& shared_register_mutex()
    {
        static std::mutex mutex;
        return mutex;
    }

    /// Every FPGA buffer pointer counts in 128-byte pages, and BUF_START ignores the low 7 bits.
    constexpr uint64_t FPGA_PAGE_BYTES = 128;
    /// Per-frame metadata block the endpoint appends after the payload window.
    constexpr uint64_t RX_METADATA_BYTES = 128;
    /// BUF_PTR_RANGE is 12 bits wide.
    constexpr uint32_t RX_MAX_BUFFERS = 4096;
    /// DP_PKT_LEN is 8 bits wide.
    constexpr uint32_t RX_MAX_CHUNK_PAGES = 255;

    constexpr uint64_t align_up(uint64_t value, uint64_t alignment)
    {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    /// Bytes the ring spans from iova: what the 4 GiB-straddle check has to cover.
    uint64_t ring_span_bytes(const PcieReceiverAttachmentV1& receiver)
    {
        if (receiver.buf_inc_pages() > 0) {
            return static_cast<uint64_t>(receiver.buf_inc_pages()) * FPGA_PAGE_BYTES
                * receiver.num_buffers();
        }
        // A single-slot ring rewrites one window plus its metadata block.
        return align_up(receiver.window_bytes(), FPGA_PAGE_BYTES) + RX_METADATA_BYTES;
    }
} // namespace

PcieDataChannelV1::PcieDataChannelV1(std::shared_ptr<DataChannelInterfaceV1> anchor)
    : anchor_(std::move(anchor))
{
    if (!anchor_) {
        throw std::runtime_error(
            "While constructing PcieDataChannelV1: anchor (DataChannelInterfaceV1) must be non-null");
    }
}

void PcieDataChannelV1::configure(const EnumerationMetadata& metadata)
{
    // Drive the anchor's configure (idempotent via its own framework latch). After this the
    // anchor's enumeration_metadata() / hololink() are usable.
    anchor_->ensure_configured(metadata);

    // The per-board Hololink, reached the way every sibling in the module does, and
    // registered with it so device_lost() invalidates this channel too. Only the
    // HololinkInterfaceV1 surface is kept: the channel never bypasses the address policy.
    auto hololink_impl = module_core::HololinkV1::get_service(
        this->module(), HololinkInterfaceV1::locator_id(metadata).c_str());
    hololink_impl->register_associated(this);
    hololink_ = hololink_impl;

    const int64_t sensor = metadata.get<int64_t>(KEY_SENSOR);
    if (sensor < 0) {
        throw std::runtime_error(fmt::format(
            "While configuring PcieDataChannelV1: metadata '{}' = {} is not a valid sensor index",
            KEY_SENSOR, sensor));
    }
    sensor_ = static_cast<uint32_t>(sensor);
    // Both default to the one-SIF-per-sensor layout this carrier has, which is also what
    // PcieChannelConfigurationV1::use_sensor stamps.
    vp_mask_ = static_cast<uint32_t>(metadata.get<int64_t>(KEY_VP_MASK, int64_t { 1 } << sensor_));
    const int64_t vector = metadata.get<int64_t>(KEY_VECTOR, sensor);
    if (vector < 0 || vector >= static_cast<int64_t>(regs::MSIX_TABLE_ENTRIES)) {
        throw std::runtime_error(fmt::format(
            "While configuring PcieDataChannelV1: metadata '{}' = {} is outside the MSI-X table (0..{})",
            KEY_VECTOR, vector, regs::MSIX_TABLE_ENTRIES - 1));
    }
    vector_ = static_cast<uint32_t>(vector);
    instance_ = PcieDataChannelInterfaceV1::locator_id(metadata);
    HSB_LOG_DEBUG("PcieDataChannelV1 '{}' configured: sensor={} vp_mask={:#x} vector={}",
        instance_, sensor_, vp_mask_, vector_);
}

void PcieDataChannelV1::require_configured(const char* what) const
{
    if (!hololink_) {
        throw std::runtime_error(fmt::format(
            "PcieDataChannelV1: {} before configure(); fetch the channel via "
            "PcieDataChannelInterfaceV1::get_service(metadata) first",
            what));
    }
}

bool PcieDataChannelV1::attached() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return attached_;
}

bool PcieDataChannelV1::armed() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return armed_;
}

// ---------------------------------------------------------------------------
// Register access through the parent HololinkInterfaceV1
// ---------------------------------------------------------------------------

void PcieDataChannelV1::write_posted(uint32_t offset, uint32_t value, const char* name)
{
    const hololink_module_status_t status = hololink_->write_uint32({ offset }, { value });
    if (status != HOLOLINK_MODULE_OK) {
        throw std::runtime_error(fmt::format(
            "PcieDataChannelV1 '{}': write of {} (BAR0 {:#x}) <- {:#x} failed with status {}",
            instance_, name, offset, value, status));
    }
}

uint32_t PcieDataChannelV1::read_reg(uint32_t offset, const char* name)
{
    std::vector<uint32_t> values;
    const hololink_module_status_t status = hololink_->read_uint32({ offset }, values);
    if (status != HOLOLINK_MODULE_OK || values.size() != 1) {
        throw std::runtime_error(fmt::format(
            "PcieDataChannelV1 '{}': read of {} (BAR0 {:#x}) failed with status {}",
            instance_, name, offset, status));
    }
    return values[0];
}

void PcieDataChannelV1::write_verified(uint32_t offset, uint32_t value, const char* name)
{
    write_posted(offset, value, name);
    const uint32_t readback = read_reg(offset, name);
    if (readback != value) {
        throw std::runtime_error(fmt::format(
            "PcieDataChannelV1 '{}': {} (BAR0 {:#x}) wrote {:#x} but reads back {:#x}",
            instance_, name, offset, value, readback));
    }
}

void PcieDataChannelV1::set_vip_bit(bool enable)
{
    // DP_PKT_VIP_MASK is one word shared by every sensor (bit n enables virtual port n), so
    // only this sensor's bit may change: read-modify-write through the module, which
    // serializes the read and the write (Hololink::execute). The read precedes the edge; the
    // edge itself is still a posted write that is never read back -- the arm/disarm edges
    // are where data, metadata and MSI-X TLPs start flowing, and the reference path is only
    // stable when they are not read back.
    const uint32_t address = regs::DP_PKT_BASE + regs::DP_PKT_VIP_MASK;
    const hololink_module_status_t status = enable
        ? hololink_->or_uint32(address, vp_mask_)
        : hololink_->and_uint32(address, ~vp_mask_);
    if (status != HOLOLINK_MODULE_OK) {
        throw std::runtime_error(fmt::format(
            "PcieDataChannelV1 '{}': {} of DP_PKT_VIP_MASK (BAR0 {:#x}) bit {:#x} failed with status {}",
            instance_, enable ? "or_uint32" : "and_uint32", address, vp_mask_, status));
    }
}

// ---------------------------------------------------------------------------
// Attach / detach
// ---------------------------------------------------------------------------

const char* PcieDataChannelV1::validate(const PcieReceiverAttachmentV1& receiver,
    uint32_t chunk_pages) const
{
    if (receiver.vector() != vector_) {
        return "the receiver waits on a different MSI-X vector than this channel routes to";
    }
    if (receiver.num_buffers() < 1) {
        return "num_buffers must be at least 1";
    }
    if (receiver.num_buffers() > RX_MAX_BUFFERS) {
        return "num_buffers must fit in the 12-bit endpoint buffer pointer";
    }
    if (chunk_pages < 1 || chunk_pages > RX_MAX_CHUNK_PAGES) {
        return "chunk_pages must be in [1, 255]";
    }
    if (receiver.requester_id() > 0xFFFFu) {
        return "requester_id must fit in 16 bits";
    }
    if (receiver.window_bytes() == 0) {
        return "window_bytes must be positive";
    }
    if (receiver.window_bytes() > 0xFFFFFFFFull) {
        return "window_bytes must fit in the 32-bit BUF_LEN register";
    }
    if (receiver.num_buffers() > 1 && receiver.buf_inc_pages() == 0) {
        return "a multi-slot ring needs a non-zero slot increment";
    }
    if (receiver.buf_inc_pages() > 0
        && static_cast<uint64_t>(receiver.buf_inc_pages()) * FPGA_PAGE_BYTES
            < align_up(receiver.window_bytes(), FPGA_PAGE_BYTES) + RX_METADATA_BYTES) {
        return "slot increment is smaller than the window plus its metadata block";
    }
    const uint64_t iova = receiver.iova();
    if (iova % FPGA_PAGE_BYTES) {
        return "buffer address must be 128-byte aligned; BUF_START ignores the low 7 bits";
    }
    // BUF_BASE_HI supplies one fixed upper 32 bits for the whole ring, so the buffer may sit
    // anywhere in the 64-bit space but may not straddle a 4 GiB boundary.
    const uint64_t last = iova + ring_span_bytes(receiver) - 1;
    if ((iova >> 32) != (last >> 32)) {
        return "buffer crosses a 4 GiB boundary, which the fixed-MSB address generator "
               "cannot represent";
    }
    return nullptr;
}

hololink_module_status_t PcieDataChannelV1::attach_receiver(
    std::shared_ptr<PcieReceiverAttachmentV1> receiver, uint32_t chunk_pages)
{
    if (!receiver) {
        throw std::runtime_error(
            "While attaching a receiver to the PCIe data channel: receiver is null (start the "
            "receiver before calling attach_receiver)");
    }
    require_configured("attach_receiver");
    std::lock_guard<std::mutex> lock(mutex_);
    // DP_PKT_LEN and DP_PKT_VIP_MASK are one word each for every sensor on the board, and the
    // sibling check below reads the mask that another channel's attach/arm writes. mutex_ is
    // per channel, so the check-program-arm sequence is serialized across channels here
    // (process-wide; across processes the mask itself is the only guard).
    std::lock_guard<std::mutex> board_lock(shared_register_mutex());
    if (attached_) {
        HSB_LOG_ERROR("PcieDataChannelV1 '{}': attach_receiver while a receiver is already "
                      "attached; detach_receiver first",
            instance_);
        return HOLOLINK_MODULE_INVALID_PARAMETER;
    }
    if (const char* error = validate(*receiver, chunk_pages)) {
        HSB_LOG_ERROR("PcieDataChannelV1 '{}': rejecting attachment: {}", instance_, error);
        return HOLOLINK_MODULE_INVALID_PARAMETER;
    }

    const uint64_t iova = receiver->iova();
    try {
        // A previously interrupted process can leave the data path armed against memory the
        // kernel has already reclaimed. Quiesce before touching any destination register:
        // clear this sensor's VIP bit only (the other sensor's ring may be live). The Versa
        // bitfile has no data generator, so the VIP bit is the only producer gate.
        set_vip_bit(false);
        armed_ = false;

        // DP_PKT_LEN is one word for the whole plane. If it already holds a different TLP
        // size and another sensor's VIP bit is set, a sibling channel (this process or
        // another) is streaming with that size; rewriting the word underneath it would
        // corrupt that stream, so refuse. With no sibling armed the word is just whatever
        // the last run or the reset left, and it is ours to program.
        const uint32_t dp_pkt_len = regs::DP_PKT_BASE + regs::DP_PKT_LEN;
        const uint32_t current_len = read_reg(dp_pkt_len, "DP_PKT_LEN");
        if (current_len != chunk_pages) {
            const uint32_t vip_mask
                = read_reg(regs::DP_PKT_BASE + regs::DP_PKT_VIP_MASK, "DP_PKT_VIP_MASK");
            const uint32_t armed_siblings = vip_mask & ~vp_mask_;
            if (armed_siblings != 0) {
                HSB_LOG_ERROR("PcieDataChannelV1 '{}': DP_PKT_LEN (shared by every sensor) is "
                              "{} pages and DP_PKT_VIP_MASK {:#x} shows another sensor armed "
                              "with it; refusing to retarget it to {} pages for sensor {}",
                    instance_, current_len, vip_mask, chunk_pages, sensor_);
                return HOLOLINK_MODULE_INVALID_PARAMETER;
            }
            HSB_LOG_INFO("PcieDataChannelV1 '{}': DP_PKT_LEN (shared by every sensor) read {} "
                         "pages with no sibling armed; programming {}",
                instance_, current_len, chunk_pages);
        }
        write_verified(dp_pkt_len, chunk_pages, "DP_PKT_LEN");

        // No MSI-X table access here: the kernel owns the table and the receiver has
        // already enabled vector_ through the driver. A completion raised before the
        // receiver's first wait is held in the pending-bit array, not lost.

        // Destination: fixed upper 32 bits, requester id, slot increment, start, window,
        // ring depth, metadata control.
        write_verified(regs::pcie_dp_reg(sensor_, regs::PCIE_DP_BUF_BASE_HI),
            static_cast<uint32_t>(iova >> 32), "PCIE_DP_BUF_BASE_HI");
        write_verified(regs::pcie_dp_reg(sensor_, regs::PCIE_DP_REQUESTER_ID),
            receiver->requester_id(), "PCIE_DP_REQUESTER_ID");
        write_verified(regs::pcie_dp_reg(sensor_, regs::PCIE_DP_BUF_INC), receiver->buf_inc_pages(),
            "PCIE_DP_BUF_INC");
        write_verified(regs::pcie_dp_reg(sensor_, regs::PCIE_DP_BUF_START),
            static_cast<uint32_t>(iova & 0xFFFFFFFFull), "PCIE_DP_BUF_START");
        write_verified(regs::pcie_dp_reg(sensor_, regs::PCIE_DP_BUF_LEN),
            static_cast<uint32_t>(receiver->window_bytes()), "PCIE_DP_BUF_LEN");
        write_verified(regs::pcie_dp_reg(sensor_, regs::PCIE_DP_BUF_PTR_RANGE),
            receiver->num_buffers() - 1, "PCIE_DP_BUF_PTR_RANGE");
        write_verified(regs::pcie_dp_reg(sensor_, regs::PCIE_DP_METADATA_CTRL),
            receiver->disable_metadata() ? regs::PCIE_DP_MD_DISABLE_MASK : 0u,
            "PCIE_DP_METADATA_CTRL");

        receiver_ = receiver;
        attached_ = true;

        // Arm: or_uint32 of this sensor's bit; posted, never read back.
        set_vip_bit(true);
        armed_ = true;
    } catch (...) {
        // Leave the endpoint quiet and the channel detached; the caller sees the exception
        // naming the failing register.
        receiver_.reset();
        attached_ = false;
        armed_ = false;
        try {
            set_vip_bit(false);
        } catch (const std::exception& e) {
            HSB_LOG_WARN("PcieDataChannelV1 '{}': disarm after a failed attach also failed: {}",
                instance_, e.what());
        }
        throw;
    }

    HSB_LOG_INFO("PcieDataChannelV1 '{}': sensor {} attached and armed: iova={:#x} "
                 "requester_id={:#x} window={:#x} buf_inc_pages={} buffers={} chunk_pages={} "
                 "msix_vector={}",
        instance_, sensor_, iova, receiver->requester_id(), receiver->window_bytes(),
        receiver->buf_inc_pages(), receiver->num_buffers(), chunk_pages, vector_);
    return HOLOLINK_MODULE_OK;
}

hololink_module_status_t PcieDataChannelV1::detach_receiver()
{
    require_configured("detach_receiver");
    std::lock_guard<std::mutex> lock(mutex_);
    // Disarm first, whether or not anything is attached: this is the "make the endpoint stop
    // writing to this sensor's memory" call, and it must be safe after a failed attach.
    set_vip_bit(false);
    armed_ = false;
    if (!attached_) {
        HSB_LOG_DEBUG("PcieDataChannelV1 '{}': detach_receiver with no receiver attached", instance_);
        return HOLOLINK_MODULE_OK;
    }
    // The vector stays enabled: the receiver owns it and disables it through the driver when
    // it closes, after its monitor has drained.
    receiver_.reset();
    attached_ = false;
    HSB_LOG_INFO("PcieDataChannelV1 '{}': sensor {} detached", instance_, sensor_);
    return HOLOLINK_MODULE_OK;
}

hololink_module_status_t PcieDataChannelV1::arm()
{
    require_configured("arm");
    std::lock_guard<std::mutex> lock(mutex_);
    // Same lock as attach_receiver: an arm racing a sibling's sibling check must not slip in
    // between that check and its DP_PKT_LEN write.
    std::lock_guard<std::mutex> board_lock(shared_register_mutex());
    if (!attached_) {
        HSB_LOG_ERROR("PcieDataChannelV1 '{}': arm() with no receiver attached", instance_);
        return HOLOLINK_MODULE_INVALID_PARAMETER;
    }
    set_vip_bit(true);
    armed_ = true;
    return HOLOLINK_MODULE_OK;
}

hololink_module_status_t PcieDataChannelV1::disarm()
{
    require_configured("disarm");
    std::lock_guard<std::mutex> lock(mutex_);
    set_vip_bit(false);
    armed_ = false;
    return HOLOLINK_MODULE_OK;
}

hololink_module_status_t PcieDataChannelV1::diagnostics(PcieChannelDiagnostics& d)
{
    require_configured("diagnostics");
    std::lock_guard<std::mutex> lock(mutex_);
    const uint32_t sensor_base = regs::PACK_CTRL_BASE + sensor_ * regs::PACK_CTRL_STRIDE;
    const uint32_t entry = regs::msix_table_entry(vector_);
    const auto dp = [this](uint32_t reg) { return regs::pcie_dp_reg(sensor_, reg); };
    d = PcieChannelDiagnostics {};
    d.scratch = read_reg(regs::GLB_SCRATCH, "GLB_SCRATCH");
    d.pack_ctrl = read_reg(sensor_base + 3 * regs::REG_SIZE, "PACK_CTRL");
    d.pack_tvalid = read_reg(sensor_base + regs::PACK_SIF_TVLD_STAT, "PACK_SIF_TVLD_STAT");
    d.pack_psn = read_reg(sensor_base + regs::PACK_PSN_STAT, "PACK_PSN_STAT");
    d.vip_mask = read_reg(regs::DP_PKT_BASE + regs::DP_PKT_VIP_MASK, "DP_PKT_VIP_MASK");
    d.dp_pkt_len = read_reg(regs::DP_PKT_BASE + regs::DP_PKT_LEN, "DP_PKT_LEN");
    d.dp_pkt_stat = read_reg(regs::DP_PKT_BASE + regs::DP_PKT_STAT, "DP_PKT_STAT");
    d.fixed_msb = read_reg(dp(regs::PCIE_DP_BUF_BASE_HI), "PCIE_DP_BUF_BASE_HI");
    d.requester_id = read_reg(dp(regs::PCIE_DP_REQUESTER_ID), "PCIE_DP_REQUESTER_ID");
    d.buf_inc = read_reg(dp(regs::PCIE_DP_BUF_INC), "PCIE_DP_BUF_INC");
    d.buf_start = read_reg(dp(regs::PCIE_DP_BUF_START), "PCIE_DP_BUF_START");
    d.buf_len = read_reg(dp(regs::PCIE_DP_BUF_LEN), "PCIE_DP_BUF_LEN");
    d.msix_addr_lo = read_reg(entry + regs::MSIX_MESSAGE_ADDR_LO, "MSIX_MESSAGE_ADDR_LO");
    d.msix_addr_hi = read_reg(entry + regs::MSIX_MESSAGE_ADDR_HI, "MSIX_MESSAGE_ADDR_HI");
    d.msix_data = read_reg(entry + regs::MSIX_MESSAGE_DATA, "MSIX_MESSAGE_DATA");
    d.msix_vector_control = read_reg(entry + regs::MSIX_VECTOR_CONTROL, "MSIX_VECTOR_CONTROL");
    d.msix_pba_lo = read_reg(regs::MSIX_PBA_BASE + regs::MSIX_PBA_LO, "MSIX_PBA_LO");
    d.msix_pba_hi = read_reg(regs::MSIX_PBA_BASE + regs::MSIX_PBA_HI, "MSIX_PBA_HI");
    d.ptr_range = read_reg(dp(regs::PCIE_DP_BUF_PTR_RANGE), "PCIE_DP_BUF_PTR_RANGE");
    d.metadata_ctrl = read_reg(dp(regs::PCIE_DP_METADATA_CTRL), "PCIE_DP_METADATA_CTRL");
    return HOLOLINK_MODULE_OK;
}

} // namespace hololink::module::hsb_pcie
