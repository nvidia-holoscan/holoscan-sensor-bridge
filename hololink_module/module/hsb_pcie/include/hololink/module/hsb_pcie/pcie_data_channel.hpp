/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_HSB_PCIE_PCIE_DATA_CHANNEL_HPP
#define HOLOLINK_MODULE_HSB_PCIE_PCIE_DATA_CHANNEL_HPP

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "hololink/module/data_channel.hpp"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/module.hpp"
#include "hololink/module/service.hpp"
#include "hololink/module/status.h"

/* Namespace: hololink::module::hsb_pcie, the module's own namespace, so
 * this header follows the hsb_lite convention exactly -- include
 * directory, namespace and module name coincide
 * (hololink/module/hsb_lite/hsb_lite.hpp declares
 * hololink::module::hsb_lite::HsbLiteInterfaceV1). The host-side receive
 * path keeps hololink::module::pcie (hololink/module/pcie/*): that
 * namespace holds what is transport-generic on the host -- the BAR0
 * register map, sysfs resolution, the receiver and its geometry -- while
 * this service is published by the hsb_pcie module and only exists where
 * that module is loaded. Keeping the two apart also avoids two unrelated
 * things both called `pcie` in two include roots; the receiver implements
 * PcieReceiverAttachmentV1 by naming it fully qualified. */
namespace hololink::module::hsb_pcie {

/* What a PCIe receiver hands the channel so the channel can program the
 * endpoint's PCIE_DP block for it: the endpoint-visible address of the
 * ring, the slot layout, the PCI requester id the endpoint must emit,
 * and the completion vector. The receiver owns the memory (dma-buf
 * import / host staging) and the whole interrupt side -- enabling its
 * vector through the driver (HL_PCIE_IOC_MSIX_CTRL) and waiting on it
 * (HL_PCIE_IOC_MSIX_WAIT); the channel owns every HSB register. Nobody
 * in userspace writes the MSI-X table: it is generic PCI, owned by the
 * kernel, and the driver flips a vector's mask with enable_irq() /
 * disable_irq(). Implemented by hololink::module::pcie::PcieReceiver; a
 * test can implement it with constants. The values are read once,
 * during attach_receiver, and must be valid from the receiver's
 * start() until detach_receiver(). */
class PcieReceiverAttachmentV1 {
public:
    static constexpr const char* release_version = UNRELEASED;

    virtual ~PcieReceiverAttachmentV1() = default;

    /* Endpoint-visible address of slot 0 (what BUF_BASE_HI / BUF_START are
     * programmed with). 128-byte aligned; the ring may not straddle a
     * 4 GiB boundary because BUF_BASE_HI is one fixed upper word. */
    virtual uint64_t iova() const = 0;
    /* Payload window per slot, in bytes: PCIE_DP_BUF_LEN. */
    virtual uint64_t window_bytes() const = 0;
    /* Slot stride in 128-byte pages: PCIE_DP_BUF_INC. 0 for a
     * single-slot ring that rewrites the same slot. */
    virtual uint32_t buf_inc_pages() const = 0;
    /* Slots in the ring: PCIE_DP_BUF_PTR_RANGE + 1. In [1, 4096]. */
    virtual uint32_t num_buffers() const = 0;
    /* PCI requester id ((bus << 8) | devfn) the endpoint stamps on its
     * writes: PCIE_DP_REQUESTER_ID. From the receiver's device identity. */
    virtual uint32_t requester_id() const = 0;
    /* MSI-X vector the receiver waits on. Must equal the channel's
     * vector(); the channel rejects an attachment on another vector. The
     * receiver, not the channel, enables it (HL_PCIE_IOC_MSIX_CTRL). */
    virtual uint32_t vector() const = 0;
    /* Set PCIE_DP_MD_DISABLE so the endpoint writes no metadata block. */
    virtual bool disable_metadata() const = 0;
};

/* Register readback of one channel, for logging a configured channel or
 * diagnosing a stalled one. Every field is the raw BAR0 word. */
struct PcieChannelDiagnostics {
    uint32_t scratch { 0 }; ///< GLB_SCRATCH
    uint32_t pack_ctrl { 0 }; ///< PACK_CTRL word 3 of this sensor's packetizer
    uint32_t pack_tvalid { 0 }; ///< PACK_SIF_TVLD_STAT
    uint32_t pack_psn { 0 }; ///< PACK_PSN_STAT
    uint32_t vip_mask { 0 }; ///< DP_PKT_VIP_MASK (shared by every sensor)
    uint32_t dp_pkt_len { 0 }; ///< DP_PKT_LEN (shared by every sensor)
    uint32_t dp_pkt_stat { 0 }; ///< DP_PKT_STAT
    uint32_t fixed_msb { 0 }; ///< PCIE_DP_BUF_BASE_HI
    uint32_t requester_id { 0 }; ///< PCIE_DP_REQUESTER_ID
    uint32_t buf_inc { 0 }; ///< PCIE_DP_BUF_INC
    uint32_t buf_start { 0 }; ///< PCIE_DP_BUF_START
    uint32_t buf_len { 0 }; ///< PCIE_DP_BUF_LEN
    // The MSI-X words are read-only diagnostics: the kernel programs and
    // masks the table, the channel never writes it.
    uint32_t msix_addr_lo { 0 }; ///< this vector's MSIX_MESSAGE_ADDR_LO
    uint32_t msix_addr_hi { 0 }; ///< MSIX_MESSAGE_ADDR_HI
    uint32_t msix_data { 0 }; ///< MSIX_MESSAGE_DATA
    uint32_t msix_vector_control { 0 }; ///< MSIX_VECTOR_CONTROL (1 = masked)
    uint32_t msix_pba_lo { 0 }; ///< MSIX_PBA_LO
    uint32_t msix_pba_hi { 0 }; ///< MSIX_PBA_HI
    uint32_t ptr_range { 0 }; ///< PCIE_DP_BUF_PTR_RANGE
    uint32_t metadata_ctrl { 0 }; ///< PCIE_DP_METADATA_CTRL
};

/* PCIe transport view of a per-channel DataChannelInterfaceV1: the
 * PCIe counterpart of RoceDataChannelInterfaceV1. A separate
 * ConfigurableService -- has-a, not is-a -- whose impl holds a
 * shared_ptr to its DataChannelInterfaceV1 anchor and adds the
 * PCIe-specific transport surface: programming the endpoint's PCIE_DP
 * destination block for a receiver, and arming / disarming this
 * sensor's bit of the shared DP_PKT_VIP_MASK word. The MSI-X table is
 * not part of that surface: it is generic PCI owned by the kernel, and
 * the receiver enables its vector through the driver.
 *
 * Cached under instance_id "serial=<serial_number>;data_channel=<n>"
 * (the same key as the anchor it wraps). The hsb_pcie publisher's
 * construct_board_services fetches the anchor through the Publisher
 * and passes it to the impl's constructor; the impl's
 * configure(metadata) drives the anchor's ensure_configured(metadata)
 * first, so application code calls
 * PcieDataChannelInterfaceV1::get_service(metadata) without constructing
 * the DataChannelInterfaceV1 itself. configure reads `sensor` (the
 * PCIE_DP block), `vp_mask` (this sensor's VIP bit) and
 * `pcie_vector` (its MSI-X vector) off the metadata -- the values
 * ChannelConfigurationInterfaceV1::use_sensor stamps.
 *
 * Every register access goes through the parent HololinkInterfaceV1
 * (write_uint32 / read_uint32 / or_uint32 / and_uint32), so the module's
 * address policy applies to the channel exactly as to any other client.
 *
 * Lifecycle, mirroring RoceDataChannelInterfaceV1 + RoceReceiverInterfaceV1:
 *   - the operator brings its receiver up (memory mapped, completion
 *     counter baselined, its MSI-X vector enabled through the driver)
 *     with no HSB register access;
 *   - attach_receiver(receiver, chunk_pages) programs and arms:
 *       disarm this sensor's VIP bit (and_uint32), DP_PKT_LEN = chunk_pages,
 *       then the PCIE_DP words
 *       BUF_BASE_HI, REQUESTER_ID, BUF_INC, BUF_START, BUF_LEN,
 *       BUF_PTR_RANGE, METADATA_CTRL -- each configuration write read
 *       back and verified -- and finally arm (or_uint32 of the VIP
 *       bit). This is the reference receive ring's register sequence
 *       minus its MSI-X table writes, owned by the channel;
 *   - detach_receiver() disarms (and_uint32). The receiver disables its
 *     vector when it closes.
 * A second attach_receiver without a detach_receiver in between fails
 * with HOLOLINK_MODULE_INVALID_PARAMETER, as does an attachment whose
 * values the endpoint cannot represent (vector mismatch, buffer count,
 * chunk_pages, alignment, 4 GiB straddle). A register write that fails
 * or does not read back throws std::runtime_error naming the register,
 * after the channel has disarmed itself. */
class PcieDataChannelInterfaceV1
    : public ConfigurableService<PcieDataChannelInterfaceV1> {
public:
    static constexpr const char* type_id = "pcie_data_channel.v1";
    static constexpr const char* release_version = UNRELEASED;

    /* Per-channel instance_id derivation used by the metadata-form
     * get_service. Same key as DataChannelInterfaceV1::locator_id --
     * both services key by (serial, data_channel). */
    static std::string locator_id(const EnumerationMetadata& metadata)
    {
        return DataChannelInterfaceV1::locator_id(metadata);
    }

    virtual ~PcieDataChannelInterfaceV1() = default;

    /* Program the endpoint to deliver this sensor's frames into the
     * receiver's ring and arm the data path. The channel reads the
     * destination / routing values it needs off the attachment (see
     * PcieReceiverAttachmentV1); the caller must have already brought the
     * receiver up. `chunk_pages` is the data TLP size in 128-byte pages
     * (DP_PKT_LEN, in [1, 255]); that word is shared by every sensor on
     * the endpoint, so two attached channels must agree on it -- the
     * channel logs a warning when it retargets a value another channel
     * left there. Pair with detach_receiver() before tearing the
     * receiver down. */
    virtual hololink_module_status_t attach_receiver(
        std::shared_ptr<PcieReceiverAttachmentV1> receiver,
        uint32_t chunk_pages)
        = 0;

    /* Disarm this sensor; the PCIE_DP words are left as programmed. Safe
     * to call when nothing is attached. */
    virtual hololink_module_status_t detach_receiver() = 0;

    /* Re-enable / disable the data path for this sensor: a read-modify-
     * write of its bit in DP_PKT_VIP_MASK (or_uint32 / and_uint32). The
     * arm/disarm edges are posted and never read back -- that is where
     * data, metadata and MSI-X TLPs start flowing. arm() requires an
     * attached receiver; disarm() is always allowed. */
    virtual hololink_module_status_t arm() = 0;
    virtual hololink_module_status_t disarm() = 0;

    /* Read back the register set that describes this channel. */
    virtual hololink_module_status_t diagnostics(PcieChannelDiagnostics& out) = 0;

    /* PCIE_DP block index this channel programs. */
    virtual uint32_t sensor() const = 0;
    /* MSI-X vector the endpoint raises for this sensor's completions
     * (this board's convention: vector == sensor). The channel only
     * checks that an attachment agrees; enabling it is the receiver's. */
    virtual uint32_t vector() const = 0;
    /* This sensor's bit in DP_PKT_VIP_MASK. */
    virtual uint32_t vp_mask() const = 0;
};

} // namespace hololink::module::hsb_pcie

#endif // HOLOLINK_MODULE_HSB_PCIE_PCIE_DATA_CHANNEL_HPP
