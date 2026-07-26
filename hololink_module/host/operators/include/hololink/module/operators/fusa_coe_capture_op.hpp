/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_OPERATORS_FUSA_COE_CAPTURE_OP_HPP
#define HOLOLINK_MODULE_OPERATORS_FUSA_COE_CAPTURE_OP_HPP

#include <atomic>
#include <condition_variable>
#include <ctime>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <hololink/core/csi_controller.hpp>
#include <holoscan/holoscan.hpp>

#include <hololink/operators/fusa_coe_capture/fusa_coe_capture_core.hpp>

#include "hololink/module/coe_data_channel.hpp"
#include "hololink/module/csi_converter.hpp"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/frame_metadata.hpp"

namespace hololink::module::operators {

class FusaCoeCaptureOp : public holoscan::Operator, public hololink::module::csi::CsiConverterV1 {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(FusaCoeCaptureOp)

    ~FusaCoeCaptureOp() override;

    void setup(holoscan::OperatorSpec& spec) override;
    void start() override;
    void stop() override;
    void compute(holoscan::InputContext& op_input,
        holoscan::OutputContext& op_output,
        holoscan::ExecutionContext& context) override;

    uint32_t receiver_start_byte() override;
    uint32_t received_line_bytes(uint32_t line_bytes) override;
    uint32_t transmitted_line_bytes(hololink::module::csi::PixelFormat pixel_format,
        uint32_t pixel_width) override;
    void configure(uint32_t start_byte, uint32_t received_bytes_per_line,
        uint32_t pixel_width, uint32_t pixel_height,
        hololink::module::csi::PixelFormat pixel_format,
        uint32_t trailing_bytes) override;

    // Train a downstream module converter (e.g. the native
    // PackedFormatConverterOp) on the received frame geometry.
    void configure_converter(hololink::module::csi::CsiConverterV1& converter);
    void configure_frame_size(uint32_t frame_size_bytes);

    static nvidia::gxf::Expected<void> buffer_release_callback(void* pointer);

private:
    holoscan::Parameter<std::string> interface_;
    holoscan::Parameter<std::vector<uint8_t>> mac_addr_;
    holoscan::Parameter<uint32_t> timeout_;
    holoscan::Parameter<std::string> out_tensor_name_;
    holoscan::Parameter<EnumerationMetadata> metadata_;
    holoscan::Parameter<std::function<void()>> device_start_;
    holoscan::Parameter<std::function<void()>> device_stop_;
    holoscan::Parameter<std::function<std::string(const std::string&)>> rename_metadata_;

    // Cached (possibly renamed) metadata key names; computed once in start()
    // so compute() does no per-frame string work. A per-leg rename_metadata
    // (e.g. "left_"/"right_" prefixes) lets two capture legs feed one
    // downstream consumer without their metadata keys colliding.
    std::string timestamp_s_key_;
    std::string timestamp_ns_key_;
    std::string metadata_s_key_;
    std::string metadata_ns_key_;
    std::string crc_key_;
    std::string frame_number_key_;
    std::string received_s_key_;
    std::string received_ns_key_;

    hololink::operators::fusa_coe_capture::FusaCoeCaptureCore core_;
    std::shared_ptr<CoeDataChannelInterfaceV1> channel_;
    std::shared_ptr<FrameMetadataInterfaceV1> frame_metadata_;

    // --- async capture plumbing ------------------------------------------
    // FusaCoeCaptureCore::wait_for_acquired_buffer() blocks the caller until a
    // frame lands, so a background monitor thread owns that wait and signals an
    // AsynchronousCondition; compute() then runs on the scheduler thread and
    // does only the fast GXF emit, never blocking the scheduler. This mirrors
    // RoceReceiverOp / LinuxReceiverOp so the default (greedy) scheduler can
    // drive multiple capture legs (e.g. a stereo pair through a FrameAlignerOp)
    // without serializing their blocking waits. The monitor and compute hand
    // off exactly one frame at a time, preserving the core's single
    // acquire -> register_pending_output -> release lifecycle.
    void monitor();

    std::shared_ptr<holoscan::AsynchronousCondition> frame_ready_condition_;
    std::thread monitor_thread_;
    std::atomic<bool> running_ { false };
    std::mutex handoff_mutex_;
    std::condition_variable handoff_cv_;
    bool frame_ready_ = false;
    bool frame_consumed_ = true;
    hololink::operators::fusa_coe_capture::FusaCoeCaptureCore::BufferView pending_buffer_ {};
    struct timespec pending_received_ = {};
};

} // namespace hololink::module::operators

#endif // HOLOLINK_MODULE_OPERATORS_FUSA_COE_CAPTURE_OP_HPP
