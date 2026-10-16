/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_OPERATORS_PCIE_RECEIVER_OP_HPP
#define HOLOLINK_MODULE_OPERATORS_PCIE_RECEIVER_OP_HPP

/* PcieReceiverOp: Holoscan operator for the module-native PCIe receive path
 * (hololink::module::pcie::PcieReceiver + the hsb_pcie module's
 * PcieDataChannelInterfaceV1). Port of
 * the reference bring-up tree's receiver operator onto the module
 * operator shape of roce_receiver_op.{hpp,cpp}: no BaseReceiverOp, no legacy core.
 * This header is core-free and forward-declares the receiver types so consumers
 * need neither <cuda.h>-heavy receiver headers nor the driver ABI. */

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <hololink/module/operators/ingress_stall.hpp>
#include <memory>
#include <string>
#include <thread>

#include <cuda.h>
#include <holoscan/holoscan.hpp>

#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/frame_metadata.hpp"
#include "hololink/module/hsb_pcie/pcie_data_channel.hpp"

// Forward declarations so consumers of this header don't need the full
// receiver / ReceiverMemoryDescriptor definitions. The full types are
// reached inside the operator's .cpp.
namespace hololink::module {
class ReceiverMemoryDescriptor;
namespace pcie {
    class PcieReceiver;
    class PcieDeviceAccessInterface;
} // namespace pcie
} // namespace hololink::module

namespace hololink::module::operators {

/* PCIe receiver operator built against the module's V1 surface -- the
 * hsb_pcie module's PcieDataChannelInterfaceV1 for the endpoint's
 * PCIE_DP / MSI-X programming and arming, FrameMetadataInterfaceV1 for
 * the end-of-frame block -- plus the module-native PcieReceiver, which
 * owns the ring memory, the device fd and the completion wait. The same
 * split as RoceReceiverOp's RoceDataChannelInterfaceV1 + RoceReceiverInterfaceV1.
 *
 * The application supplies `enumeration_metadata`; start() derives the
 * device node from its `peer_ip` key -- the endpoint's PCI address
 * "DDDD:BB:DD.F", resolved through sysfs (pcie/pcie_sysfs.hpp); a literal
 * "/dev/hololink_pcieN" is accepted too -- resolves the per-channel
 * PcieDataChannelInterfaceV1 from the metadata (whose `sensor` /
 * `pcie_vector` name the PCIE_DP block and MSI-X vector; the `sensor`
 * parameter, when not -1, must agree with it), and allocates the ring
 * buffer from the frame geometry parameters.
 *
 * Lifecycle:
 *   start()   — resolve services, size + allocate the ring, construct the
 *               PcieReceiver and start it (dma-buf import, MSI-X routing,
 *               baseline), channel->attach_receiver(receiver, chunk_pages)
 *               (program + arm), spawn the monitor thread with
 *               frame_context current, then device_start().
 *   compute() — receiver->get_next_frame, stamp the metadata map (through
 *               FrameMetadataInterfaceV1::decode when the slot block is in
 *               the HSB EOF layout), emit a uint8 device tensor of
 *               frame_dma_bytes at the slot address over "output".
 *   stop()    — device_stop(), channel->detach_receiver (disarm),
 *               receiver->close, join the monitor thread. */
class PcieReceiverOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(PcieReceiverOp);

    ~PcieReceiverOp() override;

    void setup(holoscan::OperatorSpec& spec) override;
    void start() override;
    void stop() override;
    void compute(holoscan::InputContext&,
        holoscan::OutputContext& op_output,
        holoscan::ExecutionContext&) override;

    /* The device node this operator would open for `metadata`: derived from
     * its `peer_ip` key (the BDF) through sysfs. Throws std::runtime_error
     * naming peer_ip when it is absent or malformed, and naming the BDF when
     * no hololink_pcie device is bound to it. Exposed so callers (and tests)
     * can validate metadata without starting the op. */
    static std::string device_node_from_metadata(
        const hololink::module::EnumerationMetadata& metadata);

private:
    // Reports a run of empty polls once, so a stalled ingress names itself.
    IngressStall ingress_stall_;
    // Logged once per stall, next to the ingress warning: the channel's register readback,
    // which separates "the CSI block sees no sensor data" (PACK_SIF_TVLD_STAT not moving)
    // from "data arrives but no completion reaches the host" (DMA / MSI-X path).
    void log_stall_diagnostics();

    holoscan::Parameter<hololink::module::EnumerationMetadata> metadata_;
    holoscan::Parameter<CUcontext> frame_context_;
    // 0 derives the frame size from the geometry; a non-zero value must equal
    // the geometry's frame_dma_bytes() or start() throws.
    holoscan::Parameter<size_t> frame_size_;

    // Sensor-side bring-up hooks (same contract as RoceReceiverOp).
    holoscan::Parameter<std::function<void()>> device_start_;
    holoscan::Parameter<std::function<void()>> device_stop_;

    holoscan::Parameter<std::string> out_tensor_name_;
    holoscan::Parameter<std::function<std::string(const std::string&)>> rename_metadata_;

    // Which sensor interface on the endpoint; doubles as the PCIE_DP block
    // and MSI-X vector. -1 reads the metadata's `sensor` key;
    // an explicit value must match the channel the metadata resolves to.
    holoscan::Parameter<int32_t> sensor_;

    // Ring shape.
    holoscan::Parameter<uint32_t> num_buffers_;
    holoscan::Parameter<uint32_t> chunk_pages_;
    holoscan::Parameter<size_t> window_bytes_;

    // Frame geometry (packed RAW10). These fix the DMA slot layout, so they
    // have to agree with what the sensor is actually emitting.
    holoscan::Parameter<uint32_t> width_;
    holoscan::Parameter<uint32_t> height_;
    holoscan::Parameter<uint32_t> stride_pixels_;
    holoscan::Parameter<uint32_t> crop_x_;
    holoscan::Parameter<uint32_t> line_alignment_;
    holoscan::Parameter<uint32_t> raw_start_byte_;

    // "caller": a ReceiverMemoryDescriptor in frame_context is dma-buf
    // imported (the VRAM path). "device": the device access allocates
    // host-staged memory the endpoint can reach (opt-in, see
    // PcieDeviceAccess::alloc_host_staged).
    holoscan::Parameter<std::string> frame_buffer_source_;

    // Renamed metadata keys, cached once in start().
    std::string flags_key_;
    std::string psn_key_;
    std::string crc_key_;
    std::string frame_number_key_;
    std::string timestamp_s_key_;
    std::string timestamp_ns_key_;
    std::string bytes_written_key_;
    std::string metadata_s_key_;
    std::string metadata_ns_key_;
    std::string received_s_key_;
    std::string received_ns_key_;
    std::string frame_memory_key_;
    std::string page_number_key_;
    std::string dropped_key_;
    std::string frames_key_;

    // Resolved in start(), held until stop().
    std::shared_ptr<hololink::module::hsb_pcie::PcieDataChannelInterfaceV1> channel_;
    std::shared_ptr<hololink::module::FrameMetadataInterfaceV1> frame_metadata_;
    std::shared_ptr<hololink::module::ReceiverMemoryDescriptor> frame_buffer_;
    std::shared_ptr<hololink::module::pcie::PcieDeviceAccessInterface> device_access_;
    std::shared_ptr<hololink::module::pcie::PcieReceiver> receiver_;
    std::shared_ptr<std::thread> monitor_thread_;
    size_t emitted_frame_bytes_ { 0 };
    // True between a successful attach_receiver and stop()'s detach_receiver.
    bool configured_ { false };

    // Frame-ready scheduling, as RoceReceiverOp: compute() runs only when the
    // monitor thread signals a ready frame, so it never blocks the scheduler.
    std::shared_ptr<holoscan::AsynchronousCondition> frame_ready_condition_;
    std::atomic<bool> running_ { false };
    void frame_ready();
};

} // namespace hololink::module::operators

#endif // HOLOLINK_MODULE_OPERATORS_PCIE_RECEIVER_OP_HPP
