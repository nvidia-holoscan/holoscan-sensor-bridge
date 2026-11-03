/*
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

// Hardware-free checks of the hololink_module operators: each one the build includes (RoceReceiverOp
// with HOLOLINK_BUILD_ROCE, FusaCoeCaptureOp with HOLOLINK_BUILD_FUSA, the CRC pair with nvcomp)
// composes into a graph that compiles (which also proves it links), and HsbControllerOp shows its
// fallback image at startup with no device connected. The converters and BaseReceiverOp also run
// in sub_frame_pipeline_test, crc_pairing_test and hololink_module_receiver_op_test.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <cuda.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include <capture_metadata_traits.hpp>

#include "hololink/module/csi_converter.hpp"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/operators/csi_to_bayer_op.hpp"
#include "hololink/module/operators/hsb_controller_op.hpp"
#include "hololink/module/operators/image_processor_op.hpp"
#include "hololink/module/operators/linux_receiver_op.hpp"
#include "hololink/module/operators/packed_format_converter_op.hpp"
#include "hololink/module/operators/sensor_factory.hpp"
#if defined(HOLOLINK_BUILD_ROCE)
#include "hololink/module/operators/roce_receiver_op.hpp"
#endif
#if defined(HOLOLINK_MODULE_BUILD_CRC)
#include "hololink/module/operators/compute_crc_op.hpp"
#endif
#if defined(HOLOLINK_BUILD_FUSA)
#include "hololink/module/operators/fusa_coe_capture_op.hpp"
#endif

namespace hololink::tests {

namespace {

    namespace mo = hololink::module::operators;

    // Takes whatever arrives: an image and its metadata.
    class ImageDrainOp : public holoscan::Operator<> {
    public:
        holoscan::Input<holoscan::schema::ImageT> image { "image" };
        holoscan::Input<hololink::CaptureMetadataT> metadata { "metadata" };

        void setup(holoscan::OperatorSpec& spec) override
        {
            (void)spec.input(image, "image").tensor_field(holoscan::schema::ImageTensorFields::data);
            spec.input(metadata, "metadata");
        }

        [[nodiscard]] holoscan::Contract contract() const override
        {
            holoscan::Contract c;
            c.trigger(holoscan::OnEach { image });
            return c;
        }

        [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override
        {
            auto sample = image.receive();
            if (!sample) {
                return holoscan::make_unexpected(std::move(sample).error());
            }
            return {};
        }
    };

    // Copies every tensor it receives to the host.
    struct CapturedTensors {
        std::mutex mutex;
        std::vector<std::vector<uint8_t>> tensors;
    };

    class TensorCaptureOp : public holoscan::Operator<> {
    public:
        explicit TensorCaptureOp(std::shared_ptr<CapturedTensors> captured)
            : captured_(std::move(captured))
        {
        }

        holoscan::Input<holoscan::Tensor> tensor { "tensor" };

        void setup(holoscan::OperatorSpec& spec) override
        {
            spec.input(tensor, "tensor")
                .expects_tensor(holoscan::TensorInputSpec {
                    .representation = holoscan::TensorRepresentation {
                        .memory_kind = holoscan::MemoryKind::kCudaDevice,
                    },
                });
        }

        [[nodiscard]] holoscan::Contract contract() const override
        {
            holoscan::Contract c;
            c.trigger(holoscan::OnEach { tensor });
            return c;
        }

        [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override
        {
            auto sample = tensor.receive();
            if (!sample) {
                return holoscan::make_unexpected(std::move(sample).error());
            }
            std::vector<uint8_t> bytes(static_cast<std::size_t>(sample->data.nbytes()));
            auto copied = sample->data.copy_to_host(bytes.data(), bytes.size(), context.cuda_stream());
            if (!copied) {
                return copied;
            }
            std::lock_guard lock(captured_->mutex);
            captured_->tensors.push_back(std::move(bytes));
            return {};
        }

    private:
        std::shared_ptr<CapturedTensors> captured_;
    };

    // Composes a module converter chain behind \p receiver_output and compiles the graph.
    template <typename Receiver>
    void compose_chain(holoscan::Graph& graph, Receiver& rx)
    {
        auto csi = graph.op<mo::CsiToBayerOp>("csi_to_bayer");
        csi.set_params(mo::CsiToBayerParams {
            .geometry = hololink::module::csi::CsiGeometry {
                .start_byte = 0,
                .bytes_per_line = 96,
                .pixel_width = 72,
                .pixel_height = 48,
                .pixel_format = hololink::module::csi::PixelFormat::RAW_10,
            },
            .bayer_format = hololink::module::csi::BayerFormat::RGGB,
        });
        auto ip = graph.op<mo::ImageProcessorOp>("image_processor");
        ip.set_params(mo::ImageProcessorParams {
            .pixel_format = static_cast<int>(hololink::module::csi::PixelFormat::RAW_10),
            .bayer_format = static_cast<int>(hololink::module::csi::BayerFormat::RGGB),
        });
        auto sink = graph.op<ImageDrainOp>("sink");
        graph.add_flow(rx->tensor, csi->input_tensor);
        graph.add_flow(rx->metadata, csi->input_metadata);
        graph.add_flow(csi->output_tensor, ip->input_tensor);
        graph.add_flow(csi->output_metadata, ip->input_metadata);
        graph.add_flow(ip->output_tensor, sink->image);
        graph.add_flow(ip->output_metadata, sink->metadata);
    }

    void expect_compiles(holoscan::Graph& graph, const std::vector<const char*>& tensor_producers)
    {
        graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
        holoscan::CompileOptions options;
        for (const char* producer : tensor_producers) {
            options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement {
                .operator_path = producer,
                .output_port = "tensor",
                .device = holoscan::DeviceId { 0 },
            });
        }
        auto plan = holoscan::compile(graph, std::move(options));
        if (!plan.ok()) {
            holoscan::log_diagnostics(plan);
        }
        EXPECT_TRUE(plan.ok()) << "graph compilation failed";
    }

} // namespace

TEST(ModuleOperators, ComposeTheLinuxReceiverChain)
{
    holoscan::Graph graph { "module_linux_receiver_chain" };
    auto rx = graph.op<mo::LinuxReceiverOp>(
        "rx", hololink::module::EnumerationMetadata {}, [] {}, [] {});
    rx.set_params(mo::LinuxReceiverParams { .frame_size = 4096, .page_size = 4096 });
    compose_chain(graph, rx);
    expect_compiles(graph, { "rx", "csi_to_bayer", "image_processor" });
}

#if defined(HOLOLINK_BUILD_ROCE)
TEST(ModuleOperators, ComposeTheRoceReceiverChain)
{
    holoscan::Graph graph { "module_roce_receiver_chain" };
    auto rx = graph.op<mo::RoceReceiverOp>(
        "rx", hololink::module::EnumerationMetadata {}, [] {}, [] {});
    rx.set_params(mo::RoceReceiverParams { .frame_size = 4096, .page_size = 4096 });
    compose_chain(graph, rx);
    expect_compiles(graph, { "rx", "csi_to_bayer", "image_processor" });
}
#endif

TEST(ModuleOperators, ComposeThePackedFormatConverter)
{
    holoscan::Graph graph { "module_packed_format_converter" };
    auto rx = graph.op<mo::LinuxReceiverOp>(
        "rx", hololink::module::EnumerationMetadata {}, [] {}, [] {});
    rx.set_params(mo::LinuxReceiverParams { .frame_size = 4096, .page_size = 4096 });
    auto converter = graph.op<mo::PackedFormatConverterOp>("converter");
    converter.set_params(mo::PackedFormatConverterParams {
        .bayer_format = hololink::module::csi::BayerFormat::RGGB,
        .geometry = hololink::module::csi::CsiGeometry {
            .bytes_per_line = 128,
            .pixel_width = 72,
            .pixel_height = 48,
            .pixel_format = hololink::module::csi::PixelFormat::RAW_10,
        },
    });
    auto sink = graph.op<ImageDrainOp>("sink");
    graph.add_flow(rx->tensor, converter->input_tensor);
    graph.add_flow(rx->metadata, converter->input_metadata);
    graph.add_flow(converter->output_tensor, sink->image);
    graph.add_flow(converter->output_metadata, sink->metadata);
    expect_compiles(graph, { "rx", "converter" });
}

#if defined(HOLOLINK_MODULE_BUILD_CRC)
TEST(ModuleOperators, ComposeTheCrcPair)
{
    holoscan::Graph graph { "module_crc_pair" };
    auto rx = graph.op<mo::LinuxReceiverOp>(
        "rx", hololink::module::EnumerationMetadata {}, [] {}, [] {});
    rx.set_params(mo::LinuxReceiverParams { .frame_size = 4096, .page_size = 4096 });
    auto crc_channel = std::make_shared<hololink::operators::CrcChannel>();
    auto compute_crc = graph.op<mo::ComputeCrcOp>("compute_crc", crc_channel);
    compute_crc.set_params(mo::ComputeCrcParams { .frame_size = 4096 });
    auto csi = graph.op<mo::CsiToBayerOp>("csi_to_bayer");
    csi.set_params(mo::CsiToBayerParams {
        .geometry = hololink::module::csi::CsiGeometry {
            .start_byte = 0,
            .bytes_per_line = 96,
            .pixel_width = 72,
            .pixel_height = 48,
            .pixel_format = hololink::module::csi::PixelFormat::RAW_10,
        },
        .bayer_format = hololink::module::csi::BayerFormat::RGGB,
    });
    auto check_crc = graph.op<mo::CheckCrcOp>("check_crc", crc_channel);
    auto sink = graph.op<ImageDrainOp>("sink");
    graph.add_flow(rx->tensor, compute_crc->input_tensor);
    graph.add_flow(rx->metadata, compute_crc->input_metadata);
    graph.add_flow(compute_crc->output_tensor, csi->input_tensor);
    graph.add_flow(compute_crc->output_metadata, csi->input_metadata);
    graph.add_flow(csi->output_tensor, sink->image);
    graph.add_flow(csi->output_metadata, check_crc->input_metadata);
    graph.add_flow(check_crc->output_metadata, sink->metadata);
    expect_compiles(graph, { "rx", "csi_to_bayer" });
}
#endif

#if defined(HOLOLINK_BUILD_FUSA)
namespace {

    // FusaCoeCaptureOp -> PackedFormatConverterOp -> sink, the module FuSa players' default
    // (packetized) path, the geometry trained through make_geometry_adapter() as a sensor driver
    // trains it; returns the compiled plan. Composing and compiling needs no device: the capture
    // resolves its channel in start().
    holoscan::ExecutionPlan compile_a_fusa_coe_capture(int64_t frame_limit)
    {
        constexpr auto kPixelFormat = hololink::module::csi::PixelFormat::RAW_10;
        hololink::module::csi::CsiGeometry geometry;
        auto converter = mo::FusaCoeCaptureOp::make_geometry_adapter(geometry, /*packetizer_enabled=*/true);
        converter->configure(converter->receiver_start_byte(),
            converter->received_line_bytes(converter->transmitted_line_bytes(kPixelFormat, 72)), 72, 48,
            kPixelFormat);

        holoscan::Graph graph { "module_fusa_coe_capture" };
        auto capture = graph.op<mo::FusaCoeCaptureOp>("capture");
        mo::FusaCoeCaptureParams params;
        params.device_start = [] {};
        params.device_stop = [] {};
        params.interface = "lo";
        params.mac_addr = std::vector<uint8_t>(6, 0);
        params.timeout = 1000;
        params.pixel_format = kPixelFormat;
        params.bayer_format = hololink::module::csi::BayerFormat::RGGB;
        params.packetizer_enabled = true;
        params.geometry = geometry;
        params.frame_limit = frame_limit;
        capture.set_params(params);
        auto unpack = graph.op<mo::PackedFormatConverterOp>("converter");
        unpack.set_params(mo::PackedFormatConverterParams {
            .bayer_format = hololink::module::csi::BayerFormat::RGGB,
            .geometry = geometry,
        });
        auto sink = graph.op<ImageDrainOp>("sink");
        graph.add_flow(capture->tensor, unpack->input_tensor);
        graph.add_flow(capture->metadata, unpack->input_metadata);
        graph.add_flow(unpack->output_tensor, sink->image);
        graph.add_flow(unpack->output_metadata, sink->metadata);
        graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
        holoscan::CompileOptions options;
        for (const char* producer : { "capture", "converter" }) {
            options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement {
                .operator_path = producer,
                .output_port = "tensor",
                .device = holoscan::DeviceId { 0 },
            });
        }
        auto plan = holoscan::compile(graph, std::move(options));
        if (!plan.ok()) {
            holoscan::log_diagnostics(plan);
        }
        return plan;
    }

} // namespace

// The capture's frame_limit is the operator's completion rule, as HSDK4's CountCondition on it was
// (the capture is the only operator here with one); without a limit the run has no rule.
TEST(ModuleOperators, ComposeTheFusaCoeCapture)
{
    const holoscan::ExecutionPlan limited = compile_a_fusa_coe_capture(3);
    ASSERT_TRUE(limited.ok()) << "graph compilation failed";
    const auto rules = limited.operator_completion_rules();
    ASSERT_EQ(rules.size(), 1U);
    EXPECT_EQ(rules[0].kind, holoscan::OperatorCompletionRuleKind::kComputeCalls);
    EXPECT_EQ(rules[0].limit, 3U);

    const holoscan::ExecutionPlan unlimited = compile_a_fusa_coe_capture(0);
    ASSERT_TRUE(unlimited.ok()) << "graph compilation failed";
    EXPECT_TRUE(unlimited.operator_completion_rules().empty());
}
#endif

namespace {

    // A SensorFactory with no device behind it: it never connects, so HsbControllerOp only ever
    // has the fallback image to show.
    class FallbackOnlySensorFactory : public mo::SensorFactory {
    public:
        FallbackOnlySensorFactory(CUdeviceptr frame, std::size_t size)
            : frame_(frame)
            , size_(size)
        {
        }

        std::shared_ptr<mo::SensorDevice> new_sensor(const hololink::module::EnumerationMetadata&) override
        {
            return nullptr;
        }
        void start(std::function<void(const hololink::module::EnumerationMetadata&)>, std::function<void()>,
            const hololink::module::EnumerationMetadata&, double) override
        {
        }
        void stop() override { }
        void fallback_frame(CUdeviceptr& out_ptr, std::size_t& out_size) override
        {
            out_ptr = frame_;
            out_size = size_;
        }

    private:
        CUdeviceptr frame_;
        std::size_t size_;
    };

} // namespace

TEST(ModuleOperators, ComposeTheHsbController)
{
    holoscan::Graph graph { "module_hsb_controller" };
    auto controller = graph.op<mo::HsbControllerOp>("hsb_controller", hololink::module::EnumerationMetadata {},
        std::make_shared<FallbackOnlySensorFactory>(0, 0), nullptr);
    controller.set_params(mo::HsbControllerParams { .frame_size = 4096, .page_size = 4096 });
    compose_chain(graph, controller);
    expect_compiles(graph, { "hsb_controller", "csi_to_bayer", "image_processor" });
}

// With no device connected, HsbControllerOp emits the application's fallback image once, at
// startup, and then waits for a device instead of re-emitting it.
TEST(ModuleOperators, HsbControllerShowsItsFallbackImageAtStartup)
{
    constexpr std::size_t kFrameSize = 4096;
    ASSERT_EQ(cuInit(0), CUDA_SUCCESS);
    CUdevice device;
    ASSERT_EQ(cuDeviceGet(&device, 0), CUDA_SUCCESS);
    CUcontext context;
    ASSERT_EQ(cuDevicePrimaryCtxRetain(&context, device), CUDA_SUCCESS);
    ASSERT_EQ(cuCtxSetCurrent(context), CUDA_SUCCESS);
    CUdeviceptr fallback = 0;
    ASSERT_EQ(cuMemAlloc(&fallback, kFrameSize), CUDA_SUCCESS);
    ASSERT_EQ(cuMemsetD8(fallback, 0xC3, kFrameSize), CUDA_SUCCESS);

    auto captured = std::make_shared<CapturedTensors>();
    {
        holoscan::Graph graph { "hsb_controller_fallback" };
        auto controller = graph.op<mo::HsbControllerOp>("hsb_controller",
            hololink::module::EnumerationMetadata {},
            std::make_shared<FallbackOnlySensorFactory>(fallback, kFrameSize), nullptr);
        controller.set_params(mo::HsbControllerParams {
            .frame_context = context,
            .frame_size = kFrameSize,
            .page_size = kFrameSize,
        });
        auto sink = graph.op<TensorCaptureOp>("sink", captured);
        graph.add_flow(controller->tensor, sink->tensor);
        graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
        holoscan::CompileOptions options;
        options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement {
            .operator_path = "hsb_controller",
            .output_port = "tensor",
            .device = holoscan::DeviceId { 0 },
        });
        auto plan = holoscan::compile(graph, std::move(options));
        if (!plan.ok()) {
            holoscan::log_diagnostics(plan);
            FAIL() << "graph compilation failed";
        }
        auto session = holoscan::run_async(plan);
        // The fallback frame comes right away; nothing else should follow it.
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        session.request_stop();
        session.wait();
        EXPECT_NE(session.termination(), holoscan::RunTermination::kFailed);
        for (const auto& diagnostic : session.diagnostics()) {
            if (diagnostic.severity == holoscan::DiagnosticSeverity::kError) {
                ADD_FAILURE() << holoscan::format_diagnostic(diagnostic, plan.graph());
            }
        }
    }
    cuMemFree(fallback);
    cuDevicePrimaryCtxRelease(device);

    std::lock_guard lock(captured->mutex);
    ASSERT_EQ(captured->tensors.size(), 1U);
    EXPECT_EQ(captured->tensors[0], std::vector<uint8_t>(kFrameSize, 0xC3));
}

} // namespace hololink::tests
