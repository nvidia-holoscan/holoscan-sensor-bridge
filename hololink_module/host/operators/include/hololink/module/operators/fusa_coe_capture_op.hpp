/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_OPERATORS_FUSA_COE_CAPTURE_OP_HPP
#define HOLOLINK_MODULE_OPERATORS_FUSA_COE_CAPTURE_OP_HPP

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <capture_metadata_traits.hpp>
#include <hololink/operators/capture_metadata_utils.hpp> // DropCounter

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/lifecycle.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/readiness_source.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include <hololink/operators/fusa_coe_capture/fusa_coe_capture_core.hpp>

#include "hololink/module/coe_data_channel.hpp"
#include "hololink/module/csi_converter.hpp"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/frame_metadata.hpp"

namespace hololink::module::operators {

/**
 * @brief Configuration parameters for FusaCoeCaptureOp.
 */
struct FusaCoeCaptureParams {
    // Enumeration metadata for the channel -- used to resolve the CoeDataChannel and
    // frame-metadata services. Unlike the receiver operators' BaseReceiverOp (whose
    // EnumerationMetadata is a constructor argument), this mirrors native
    // FusaCoeCaptureOp's own choice to carry its resource handle (hololink_channel)
    // inside Params rather than the constructor, since this operator has no CRTP base
    // requiring a non-default constructor.
    hololink::module::EnumerationMetadata metadata;

    std::function<void()> device_start;
    std::function<void()> device_stop;
    std::string interface;
    std::vector<uint8_t> mac_addr;

    // Per-frame capture timeout, in milliseconds. No usable default: FusaCoeCaptureCore
    // rejects a 0 timeout outright.
    uint32_t timeout { 0 };

    // Optional: fail the run after this many consecutive milliseconds without a frame
    // (0 = disabled). compute() then returns ErrorCode::kTimeout, so the run ends
    // RunTermination::kFailed with a diagnostic naming this capture (`name`) and how
    // long it stalled. This replaces HSDK4's capture_stalled()/
    // capture_stall_duration_ms(), which every HSDK4 caller that set this checked after
    // the run to fail it. A shorter, fixed-threshold warning is always logged from
    // Core's own acquire thread regardless of this setting.
    uint32_t stall_timeout_ms { 0 };

    // Capture format: bit depth, sensor CFA, and whether the HSB packetizer is enabled.
    // Always set explicitly -- the defaults are just FusaCoeCaptureCore's own
    // pre-request_format() resting state, not values a caller should rely on.
    hololink::module::csi::PixelFormat pixel_format { hololink::module::csi::PixelFormat::RAW_8 };
    hololink::module::csi::BayerFormat bayer_format { hololink::module::csi::BayerFormat::RGGB };
    bool packetizer_enabled { false };

    // CSI frame geometry, as reported by the sensor driver's
    // configure_converter(*FusaCoeCaptureOp::make_geometry_adapter(geometry,
    // packetizer_enabled)). Ignored when frame_size_bytes is set.
    hololink::module::csi::CsiGeometry geometry;

    // Opaque byte-blob capture: when set, start() calls
    // FusaCoeCaptureCore::configure_frame_size() instead of request_format()+configure(),
    // carrying no CFA/packetizer structure. pixel_format/bayer_format/
    // packetizer_enabled/geometry above are all ignored in this mode. Mirrors the
    // module shell's prior public configure_frame_size() method.
    std::optional<uint32_t> frame_size_bytes;
    // Complete after this many frames (compute() calls), as HSDK4's CountCondition on the
    // capture operator did; 0 or less: no limit. A camera that sends nothing never completes
    // the run; set stall_timeout_ms to bound that case.
    int64_t frame_limit { 0 };
    // Once this operator and every other one that sets frame_limit_ends_run have reached their
    // frame_limit, end the whole run (ExecutionContext::request_completion()), dropping the frames
    // still in the pipeline; see hololink::operators::FrameLimitRunGroup for when a run needs it.
    bool frame_limit_ends_run { false };
    // Names this capture in its stall, resume and failure messages, so a stereo run's log
    // says which camera stalled (HSDK4 printed out_tensor_name there). An operator cannot
    // read its own graph name, so set this to it: std::string(capture.name()).
    std::string name;
};

/**
 * @brief Operator class to capture images using NvFusaCaptureCoe, built against the
 * module's V1 surface (CoeDataChannelInterfaceV1 + FrameMetadataInterfaceV1). Shares the
 * connection-agnostic hololink::operators::fusa_coe_capture::FusaCoeCaptureCore with the
 * native hololink::operators::FusaCoeCaptureOp; only the surrounding shell (service
 * resolution, notification wiring, tensor emission) differs between the two.
 *
 * Device-driven (OnNotified): Core's acquire thread keeps the newest frame and posts a
 * notification (via frame_ready_callback, passed into core_.start()) for each frame
 * compute() can take, which triggers compute(); a frame that a newer one replaces before
 * compute() runs is counted as dropped. An ingress stall (sensor stops sending frames) is
 * detected inside Core itself; compute() fails the run (ErrorCode::kTimeout) once Core
 * reports stalled().
 *
 * Output: two correlated ports, both stamped with the same capture_time/frame_id.
 *  'tensor'   — holoscan::Tensor, the raw CSI pixel data (kCudaDevice, 1-D byte tensor).
 *  'metadata' — hololink::CaptureMetadataT, per-capture hardware metadata.
 */
class FusaCoeCaptureOp : public holoscan::Operator<FusaCoeCaptureParams> {
public:
    static constexpr std::size_t kMaxFrameBytes = 64ULL * 1024 * 1024; // 64 MiB

    /**
     * @brief Build a CsiConverterV1 that captures FUSA CoE CSI geometry into `out`.
     *
     * Sensor drivers call configure_converter(shared_ptr<CsiConverterV1>) to learn how a
     * receiver expects CSI data to be laid out. FusaCoeCaptureOp's HSB-packetizer
     * alignment differs from the generic CsiToBayerOp-flavored rules, so it needs its
     * own adapter using FusaCoeCaptureCore::received_line_bytes()/transmitted_line_bytes().
     *
     * There is no live FusaCoeCaptureOp instance to query before the graph compiles (see
     * holoscan::Graph::op()), so call this -- and camera->configure_converter(adapter) --
     * before declaring the operator, then copy `out` into FusaCoeCaptureParams::geometry.
     */
    static std::shared_ptr<hololink::module::csi::CsiConverterV1> make_geometry_adapter(
        hololink::module::csi::CsiGeometry& out, bool packetizer_enabled);

    void setup(holoscan::OperatorSpec& spec) override;
    [[nodiscard]] holoscan::Contract contract() const override;
    void start() override;
    void stop() override;
    [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
        holoscan::ExecutionContext& context) override;

    // Output ports — public so graph wiring (add_flow) can reference them by address.
    holoscan::Output<holoscan::Tensor> tensor;
    holoscan::Output<hololink::CaptureMetadataT> metadata;

private:
    hololink::operators::fusa_coe_capture::FusaCoeCaptureCore core_;
    std::shared_ptr<CoeDataChannelInterfaceV1> channel_;
    std::shared_ptr<FrameMetadataInterfaceV1> frame_metadata_;

    holoscan::NotificationSource source_;
    holoscan::NotificationSender sender_;

    // Called from Core's acquire thread (via the frame_ready_callback passed to
    // core_.start()) for each frame compute() can take; posts a notification to the HSDK5
    // scheduler to trigger the next compute(). Returns true once durably handled
    // (posted, or a terminal failure no retry would fix); false means "not accepted yet
    // (the sender is not armed, or the source is full), call again" -- see
    // FusaCoeCaptureCore::start()'s frame_ready_callback contract.
    [[nodiscard]] bool frame_ready();

    // Counts frames dropped because the pipeline is behind; reported once a second per counter.
    void count_dropped_frames(
        hololink::operators::DropCounter& counter, uint64_t frames, const char* reason);
    // Frames compute() took but could not publish.
    hololink::operators::DropCounter dropped_frames_;
    // Frames replaced by a newer one before compute() took them.
    hololink::operators::DropCounter replaced_frames_;
    hololink::operators::FrameLimitRunEnd frame_limit_run_end_;

    holoscan::LifecycleStatus on_arm(holoscan::LifecycleContext& context) noexcept;
};

} // namespace hololink::module::operators

#endif // HOLOLINK_MODULE_OPERATORS_FUSA_COE_CAPTURE_OP_HPP
