// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>

#include <hololink/core/data_channel.hpp>
#include <hololink/operators/operator_catalog.hpp>
#include <holoscan/authoring/register_operator.hpp>
#include <holoscan/authoring/registry.hpp>
#include <holoscan/core/graph.hpp>

#ifdef HOLOLINK_BUILD_ROCE
#include <hololink/operators/roce_receiver/roce_receiver_op.hpp>

#include "fake_receiver.hpp"
#endif

namespace hololink::operators {
namespace {

    TEST(OperatorCatalog, RegistersEveryRowWithoutError)
    {
        auto status = register_hololink_operator_catalog();
        EXPECT_TRUE(status) << status.error().message;
    }

    TEST(OperatorCatalog, ResolvesCsiToBayerOpByRegistryName)
    {
        ASSERT_TRUE(register_hololink_operator_catalog());
        auto row = holoscan::authoring::authoring_registry().operator_type(
            "hololink.operators.CsiToBayerOp");
        ASSERT_TRUE(row);
        holoscan::Graph graph { "operator-catalog-csi-to-bayer" };
        auto added = row->get().add_to_graph_from_assignments(graph, "csi_to_bayer", {});
        EXPECT_TRUE(added) << added.error().message;
    }

    TEST(OperatorCatalog, ResolvesImageProcessorOpByRegistryName)
    {
        ASSERT_TRUE(register_hololink_operator_catalog());
        auto row = holoscan::authoring::authoring_registry().operator_type(
            "hololink.operators.ImageProcessorOp");
        ASSERT_TRUE(row);
        holoscan::Graph graph { "operator-catalog-image-processor" };
        auto added = row->get().add_to_graph_from_assignments(graph, "image_processor", {});
        EXPECT_TRUE(added) << added.error().message;
    }

    TEST(OperatorCatalog, ResolvesUdpTransmitterOpByRegistryName)
    {
        ASSERT_TRUE(register_hololink_operator_catalog());
        auto row = holoscan::authoring::authoring_registry().operator_type(
            "hololink.operators.UdpTransmitterOp");
        ASSERT_TRUE(row);
        holoscan::Graph graph { "operator-catalog-udp-transmitter" };
        auto added = row->get().add_to_graph_from_assignments(graph, "udp_transmitter", {});
        EXPECT_TRUE(added) << added.error().message;
    }

#ifdef HOLOLINK_BUILD_ROCE
    // RoceTransmitterOp is a plain default-constructible Operator<RoceTransmitterParams> (no
    // live-object dependency, unlike RoceReceiverOp elsewhere in this file) -- registered via
    // the simple HOLOSCAN_REGISTER_OPERATOR macro, same shape as UdpTransmitterOp above. Guarded
    // because roce_transmitter, like roce_receiver, is itself only built under HOLOLINK_BUILD_ROCE.
    TEST(OperatorCatalog, ResolvesRoceTransmitterOpByRegistryName)
    {
        ASSERT_TRUE(register_hololink_operator_catalog());
        auto row = holoscan::authoring::authoring_registry().operator_type(
            "hololink.operators.RoceTransmitterOp");
        ASSERT_TRUE(row);
        holoscan::Graph graph { "operator-catalog-roce-transmitter" };
        auto added = row->get().add_to_graph_from_assignments(graph, "roce_transmitter", {});
        EXPECT_TRUE(added) << added.error().message;
    }
#endif // HOLOLINK_BUILD_ROCE

    TEST(OperatorCatalog, ResolvesPackedFormatConverterOpByRegistryName)
    {
        ASSERT_TRUE(register_hololink_operator_catalog());
        auto row = holoscan::authoring::authoring_registry().operator_type(
            "hololink.operators.PackedFormatConverterOp");
        ASSERT_TRUE(row);
        holoscan::Graph graph { "operator-catalog-packed-format-converter" };
        auto added = row->get().add_to_graph_from_assignments(graph, "packed_format_converter", {});
        EXPECT_TRUE(added) << added.error().message;

        const std::array<holoscan::authoring::ParameterAssignment, 2> assignments { {
            { "cuda_device_ordinal", holoscan::authoring::ParameterValue::from(std::int64_t { 0 }) },
            { "bayer_format", holoscan::authoring::ParameterValue::from(std::string_view { "rggb" }) },
        } };
        auto added_with_args = row->get().add_to_graph_from_assignments(
            graph, "packed_format_converter_rggb", assignments);
        EXPECT_TRUE(added_with_args) << added_with_args.error().message;
    }

    TEST(OperatorCatalog, ResolvesSubFrameCombinerOpByRegistryName)
    {
        ASSERT_TRUE(register_hololink_operator_catalog());
        auto row = holoscan::authoring::authoring_registry().operator_type(
            "hololink.operators.SubFrameCombinerOp");
        ASSERT_TRUE(row);
        holoscan::Graph graph { "operator-catalog-sub-frame-combiner" };

        // height is required; empty assignments (discovery mode) must still succeed.
        auto discovery = row->get().add_to_graph_from_assignments(graph, "sub_frame_combiner", {});
        EXPECT_TRUE(discovery) << discovery.error().message;

        const std::array<holoscan::authoring::ParameterAssignment, 1> missing_height { {
            { "cuda_device_ordinal", holoscan::authoring::ParameterValue::from(std::int64_t { 0 }) },
        } };
        auto rejected = row->get().add_to_graph_from_assignments(
            graph, "sub_frame_combiner_missing_height", missing_height);
        EXPECT_FALSE(rejected);

        const std::array<holoscan::authoring::ParameterAssignment, 2> assignments { {
            { "cuda_device_ordinal", holoscan::authoring::ParameterValue::from(std::int64_t { 0 }) },
            { "height", holoscan::authoring::ParameterValue::from(std::int64_t { 1984 }) },
        } };
        auto added = row->get().add_to_graph_from_assignments(
            graph, "sub_frame_combiner_with_height", assignments);
        EXPECT_TRUE(added) << added.error().message;

        // cuda_device_ordinal is optional (SubFrameCombinerParams defaults it to 0).
        const std::array<holoscan::authoring::ParameterAssignment, 1> height_only { {
            { "height", holoscan::authoring::ParameterValue::from(std::int64_t { 1984 }) },
        } };
        auto added_height_only = row->get().add_to_graph_from_assignments(
            graph, "sub_frame_combiner_height_only", height_only);
        EXPECT_TRUE(added_height_only) << added_height_only.error().message;
    }

    TEST(OperatorCatalog, ResolvesDiscardMetadataOpByRegistryName)
    {
        ASSERT_TRUE(register_hololink_operator_catalog());
        auto row = holoscan::authoring::authoring_registry().operator_type(
            "hololink.operators.DiscardMetadataOp");
        ASSERT_TRUE(row);
        holoscan::Graph graph { "operator-catalog-discard-metadata" };
        auto added = row->get().add_to_graph_from_assignments(graph, "discard_metadata", {});
        EXPECT_TRUE(added) << added.error().message;
    }

    // The receiver rows and FusaCoeCaptureOp take their live objects as constructor handles: a
    // required `data_channel` and optional `device_start`/`device_stop` callbacks.

    // A shared-owned channel that never touches a device: DataChannel's factory-taking constructor,
    // with a factory that returns no Hololink (the catalog's own discovery channel does the same).
    std::shared_ptr<hololink::DataChannel> make_test_channel(const std::string& peer_ip = "0.0.0.0")
    {
        hololink::Metadata metadata;
        metadata["peer_ip"] = peer_ip;
        metadata["vp_mask"] = int64_t { 0 };
        metadata["data_plane"] = int64_t { 0 };
        metadata["sensor"] = int64_t { 0 };
        metadata["sif_address"] = int64_t { 0 };
        metadata["vp_address"] = int64_t { 0 };
        metadata["hif_address"] = int64_t { 0 };
        metadata["fpga_uuid"] = std::string { "00000000-0000-0000-0000-000000000000" };
        return std::make_shared<hololink::DataChannel>(
            metadata, [](const hololink::Metadata&) { return std::shared_ptr<hololink::Hololink> {}; });
    }

    holoscan::authoring::ConstructorHandle channel_handle(
        std::shared_ptr<hololink::DataChannel> channel)
    {
        auto handle = holoscan::authoring::ConstructorHandle::from_shared(data_channel_handle_type(),
            std::move(channel));
        EXPECT_TRUE(handle) << handle.error().message;
        return handle ? std::move(*handle) : holoscan::authoring::ConstructorHandle {};
    }

    holoscan::authoring::ConstructorHandle callback_handle(std::function<void()> callback)
    {
        auto handle = holoscan::authoring::ConstructorHandle::from_shared(
            holoscan::authoring::callback_handle_type(),
            std::make_shared<const holoscan::authoring::ConstructorCallback>(std::move(callback)));
        EXPECT_TRUE(handle) << handle.error().message;
        return handle ? std::move(*handle) : holoscan::authoring::ConstructorHandle {};
    }

    // Checks one row that takes the receiver handles: discovery describes them without a device, the
    // row builds from a channel, a callback and `values`, the Graph keeps the channel alive after the
    // caller drops it (and releases it with the Graph), and a missing channel fails before the Graph
    // changes.
    void check_receiver_handle_row(std::string_view row_name,
        std::span<const holoscan::authoring::ParameterAssignment> values, std::size_t handle_count = 3)
    {
        ASSERT_TRUE(register_hololink_operator_catalog());
        auto& registry = holoscan::authoring::authoring_registry();
        auto row = registry.operator_type(row_name);
        ASSERT_TRUE(row);

        const auto described = registry.describe_operator(row_name);
        ASSERT_TRUE(described) << described.error().message;
        ASSERT_EQ(described->constructor_handles.size(), handle_count);
        EXPECT_EQ(described->constructor_handles[0].name, "data_channel");
        EXPECT_EQ(described->constructor_handles[0].type_name, "hololink.DataChannel");
        EXPECT_TRUE(described->constructor_handles[0].required);
        EXPECT_EQ(described->constructor_handles[1].name, "device_start");
        EXPECT_EQ(described->constructor_handles[2].name, "device_stop");
        for (std::size_t callback = 1; callback < 3; ++callback) {
            EXPECT_EQ(described->constructor_handles[callback].type_name, "holoscan.VoidCallback");
            EXPECT_FALSE(described->constructor_handles[callback].required);
        }

        std::weak_ptr<hololink::DataChannel> observed;
        {
            holoscan::Graph graph { "operator-catalog-receiver-handles" };
            {
                auto channel = make_test_channel();
                observed = channel;
                const std::array handles {
                    holoscan::authoring::ConstructorHandleAssignment { "data_channel",
                        channel_handle(std::move(channel)) },
                    holoscan::authoring::ConstructorHandleAssignment { "device_start", callback_handle([] {}) },
                };
                const auto added = holoscan::authoring::add_registered_operator(
                    graph, row->get(), "receiver",
                    holoscan::authoring::OperatorConstructionInputs { .values = values, .handles = handles });
                ASSERT_TRUE(added) << added.error().message;
            }
            EXPECT_FALSE(observed.expired());

            const auto missing_channel = holoscan::authoring::add_registered_operator(
                graph, row->get(), "receiver_without_channel",
                holoscan::authoring::OperatorConstructionInputs { .values = values });
            ASSERT_FALSE(missing_channel);
            EXPECT_NE(missing_channel.error().message.find("AUTHORING_CONSTRUCTOR_HANDLE_REQUIRED"),
                std::string::npos)
                << missing_channel.error().message;
        }
        EXPECT_TRUE(observed.expired());
    }

    holoscan::authoring::ParameterAssignment integer_assignment(std::string_view name,
        std::int64_t value)
    {
        return { name, holoscan::authoring::ParameterValue::from(value) };
    }

    TEST(OperatorCatalog, LinuxReceiverOpTakesItsChannelAndCallbacksAsConstructorHandles)
    {
        // Discovery exercises make_discovery_data_channel(), the fix for a real bug caught during this
        // operator's own development: BaseReceiverOp throws for a null channel, and an earlier version
        // that threw from discovery made holoscan.registry.operators() silently drop the operator from
        // Python's hololink.operators entirely.
        const std::array values { integer_assignment("frame_size", 1024),
            integer_assignment("frame_context", 0) };
        check_receiver_handle_row("hololink.operators.LinuxReceiverOp", values);
    }

    TEST(OperatorCatalog, ReceiverRowsRequireAFrameContext)
    {
        ASSERT_TRUE(register_hololink_operator_catalog());
        auto row = holoscan::authoring::authoring_registry().operator_type(
            "hololink.operators.LinuxReceiverOp");
        ASSERT_TRUE(row);
        holoscan::Graph graph { "operator-catalog-receiver-without-context" };
        const std::array values { integer_assignment("frame_size", 1024) };
        const std::array handles {
            holoscan::authoring::ConstructorHandleAssignment { "data_channel",
                channel_handle(make_test_channel()) },
        };
        const auto rejected = holoscan::authoring::add_registered_operator(
            graph, row->get(), "receiver",
            holoscan::authoring::OperatorConstructionInputs { .values = values, .handles = handles });
        ASSERT_FALSE(rejected);
        EXPECT_NE(rejected.error().message.find("frame_context"), std::string::npos)
            << rejected.error().message;
    }

    // LinuxCoeReceiverOp takes LinuxReceiverOp's handles -- the only new surface is coe_interface/
    // pixel_width/coe_channel. Always built, no HOLOLINK_BUILD_ROCE/HOLOLINK_BUILD_FUSA guard needed
    // (matches linux_coe_receiver's own CMake: no special external dependency).
    TEST(OperatorCatalog, LinuxCoeReceiverOpTakesItsChannelAndCallbacksAsConstructorHandles)
    {
        const std::array values {
            integer_assignment("frame_size", 1024),
            integer_assignment("frame_context", 0),
            holoscan::authoring::ParameterAssignment {
                "coe_interface", holoscan::authoring::ParameterValue::from(std::string_view { "eth0" }) },
        };
        check_receiver_handle_row("hololink.operators.LinuxCoeReceiverOp", values);
    }

#ifdef HOLOLINK_BUILD_ROCE
    // Guarded because roce_receiver, unlike linux_receiver_op, is itself only built under
    // HOLOLINK_BUILD_ROCE, so the row only exists in that configuration.
    TEST(OperatorCatalog, RoceReceiverOpTakesItsChannelAndCallbacksAsConstructorHandles)
    {
        const std::array values { integer_assignment("frame_size", 1024),
            integer_assignment("frame_context", 0) };
        check_receiver_handle_row("hololink.operators.RoceReceiverOp", values);
    }

    // Without ibv_name, the row's RoceReceiverOp uses the IBV device on the network interface that
    // reaches its board, not a fixed one. For a board on the loopback address that is "lo", which
    // no device is bound to, so the run fails at the receiver's start, naming it.
    TEST(OperatorCatalog, RoceReceiverOpPicksTheIbvDeviceThatReachesTheBoard)
    {
        ASSERT_TRUE(register_hololink_operator_catalog());
        auto row = holoscan::authoring::authoring_registry().operator_type(
            "hololink.operators.RoceReceiverOp");
        ASSERT_TRUE(row);
        holoscan::Graph graph { "operator-catalog-roce-receiver-device" };
        const std::array values { integer_assignment("frame_size", 1024),
            integer_assignment("frame_context", 0) };
        const std::array handles {
            holoscan::authoring::ConstructorHandleAssignment { "data_channel",
                channel_handle(make_test_channel("127.0.0.1")) },
        };
        const auto added = holoscan::authoring::add_registered_operator(
            graph, row->get(), "receiver",
            holoscan::authoring::OperatorConstructionInputs { .values = values, .handles = handles });
        ASSERT_TRUE(added) << added.error().message;
        auto receiver = graph.handle<RoceReceiverOp>("receiver");
        auto sink = graph.op<tests::fake_receiver::TensorSinkOp>(
            "sink", std::make_shared<tests::fake_receiver::SinkCount>());
        graph.add_flow(receiver->tensor, sink->tensor);
        graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
        holoscan::CompileOptions compile_options;
        compile_options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement {
            .operator_path = "receiver",
            .output_port = "tensor",
            .device = holoscan::DeviceId { 0 },
        });
        auto plan = holoscan::compile(graph, std::move(compile_options));
        ASSERT_TRUE(plan.ok());

        auto session = holoscan::run_async(plan);
        EXPECT_TRUE(session.wait_for(std::chrono::seconds(10))) << "the run did not end by itself";
        session.request_stop();
        session.wait();
        EXPECT_EQ(session.termination(), holoscan::RunTermination::kFailed);
        std::string diagnostics;
        for (const auto& diagnostic : session.diagnostics()) {
            diagnostics += holoscan::format_diagnostic(diagnostic, plan.graph());
            diagnostics += '\n';
        }
        EXPECT_NE(diagnostics.find("No InfiniBand device is bound to interface \"lo\""), std::string::npos)
            << diagnostics;
    }
#endif // HOLOLINK_BUILD_ROCE

#ifdef HOLOLINK_BUILD_GPUNETIO_ROCE
    // Guarded because gpu_roce_transceiver is itself only built under HOLOLINK_BUILD_GPUNETIO_ROCE, a
    // narrower flag than HOLOLINK_BUILD_ROCE (it links DOCA GPUNetIO/Verbs directly).
    TEST(OperatorCatalog, GpuRoceTransceiverOpTakesItsChannelAndCallbacksAsConstructorHandles)
    {
        const std::array values { integer_assignment("frame_size", 4096),
            integer_assignment("frame_context", 0) };
        // Its other sensors' channels come as a 4th, optional handle.
        check_receiver_handle_row("hololink.operators.GpuRoceTransceiverOp", values, 4);
        const auto described = holoscan::authoring::authoring_registry().describe_operator(
            "hololink.operators.GpuRoceTransceiverOp");
        ASSERT_TRUE(described);
        EXPECT_EQ(described->constructor_handles[3].name, "extra_data_channels");
        EXPECT_EQ(described->constructor_handles[3].type_name, "hololink.DataChannels");
        EXPECT_FALSE(described->constructor_handles[3].required);
    }
#endif // HOLOLINK_BUILD_GPUNETIO_ROCE

#ifdef HOLOLINK_BUILD_FUSA
    // FusaCoeCaptureOp keeps only a raw DataChannel* in its Params, so the Graph keeping the
    // channel's owner alive (checked by check_receiver_handle_row()) is what keeps that pointer
    // valid. frame_size_bytes selects the opaque capture mode, which needs no geometry.
    TEST(OperatorCatalog, FusaCoeCaptureOpTakesItsChannelAndCallbacksAsConstructorHandles)
    {
        const std::array values { integer_assignment("frame_size_bytes", 1024) };
        check_receiver_handle_row("hololink.operators.FusaCoeCaptureOp", values);
    }
#endif // HOLOLINK_BUILD_FUSA

#ifdef HOLOLINK_BUILD_ARGUS_ISP
    // ArgusIspOp is a manual row (see operator_catalog.cpp) -- guarded here because argus_isp,
    // like roce_receiver above, is itself
    // only built under its own flag (HOLOLINK_BUILD_ARGUS_ISP), so the row only exists in that
    // configuration.
    TEST(OperatorCatalog, ResolvesArgusIspOpByRegistryName)
    {
        ASSERT_TRUE(register_hololink_operator_catalog());
        auto row = holoscan::authoring::authoring_registry().operator_type(
            "hololink.operators.ArgusIspOp");
        ASSERT_TRUE(row);
        holoscan::Graph graph { "operator-catalog-argus-isp" };

        // Discovery mode (empty assignments) must still succeed.
        auto discovery = row->get().add_to_graph_from_assignments(graph, "argus_isp", {});
        EXPECT_TRUE(discovery) << discovery.error().message;

        // Exercises real_or<float> (exposure_time_ms/analog_gain), the first float-typed manual-row
        // parameters in this catalog.
        const std::array<holoscan::authoring::ParameterAssignment, 6> assignments { {
            { "bayer_format", holoscan::authoring::ParameterValue::from(std::string_view { "gbrg" }) },
            { "exposure_time_ms", holoscan::authoring::ParameterValue::from(16.67f) },
            { "analog_gain", holoscan::authoring::ParameterValue::from(10.0f) },
            { "pixel_bit_depth", holoscan::authoring::ParameterValue::from(std::int64_t { 10 }) },
            { "camera_index", holoscan::authoring::ParameterValue::from(std::int64_t { 0 }) },
            { "sensor_mode_index", holoscan::authoring::ParameterValue::from(std::int64_t { 0 }) },
        } };
        auto added = row->get().add_to_graph_from_assignments(graph, "argus_isp_configured", assignments);
        EXPECT_TRUE(added) << added.error().message;
    }
#endif // HOLOLINK_BUILD_ARGUS_ISP

#ifdef HOLOLINK_HAS_IMGUI
    TEST(OperatorCatalog, ResolvesSignalGeneratorOpByRegistryName)
    {
        ASSERT_TRUE(register_hololink_operator_catalog());
        auto row = holoscan::authoring::authoring_registry().operator_type(
            "hololink.operators.SignalGeneratorOp");
        ASSERT_TRUE(row);
        holoscan::Graph graph { "operator-catalog-signal-generator" };
        auto discovery = row->get().add_to_graph_from_assignments(graph, "signal_generator", {});
        EXPECT_TRUE(discovery) << discovery.error().message;

        const std::array<holoscan::authoring::ParameterAssignment, 6> assignments { {
            { "samples_count", holoscan::authoring::ParameterValue::from(std::int64_t { 1024 }) },
            { "sampling_interval", holoscan::authoring::ParameterValue::from(std::string_view { "1/64" }) },
            { "in_phase", holoscan::authoring::ParameterValue::from(std::string_view { "cos(2*PI*x)" }) },
            { "quadrature", holoscan::authoring::ParameterValue::from(std::string_view { "sin(2*PI*x)" }) },
            { "period_ns", holoscan::authoring::ParameterValue::from(std::int64_t { 10'000'000 }) },
            { "count", holoscan::authoring::ParameterValue::from(std::int64_t { 10 }) },
        } };
        auto added = row->get().add_to_graph_from_assignments(graph, "signal_generator_args", assignments);
        EXPECT_TRUE(added) << added.error().message;
    }

    TEST(OperatorCatalog, SignalGeneratorOpRejectsInvalidParameters)
    {
        ASSERT_TRUE(register_hololink_operator_catalog());
        auto row = holoscan::authoring::authoring_registry().operator_type(
            "hololink.operators.SignalGeneratorOp");
        ASSERT_TRUE(row);
        holoscan::Graph graph { "operator-catalog-signal-generator-invalid" };

        const std::array<holoscan::authoring::ParameterAssignment, 1> missing_samples_count { {
            { "in_phase", holoscan::authoring::ParameterValue::from(std::string_view { "x" }) },
        } };
        auto missing = row->get().add_to_graph_from_assignments(graph, "missing", missing_samples_count);
        ASSERT_FALSE(missing);
        EXPECT_NE(missing.error().message.find("HOLOLINK_OPERATOR_PARAMETER_REQUIRED"), std::string::npos);

        const std::array<holoscan::authoring::ParameterAssignment, 1> odd_samples_count { {
            { "samples_count", holoscan::authoring::ParameterValue::from(std::int64_t { 6 }) },
        } };
        auto odd = row->get().add_to_graph_from_assignments(graph, "odd", odd_samples_count);
        ASSERT_FALSE(odd);
        EXPECT_NE(odd.error().message.find("AUTHORING_PARAMETER_VALUE_INVALID"), std::string::npos);

        const std::array<holoscan::authoring::ParameterAssignment, 2> bad_interval { {
            { "samples_count", holoscan::authoring::ParameterValue::from(std::int64_t { 8 }) },
            { "sampling_interval", holoscan::authoring::ParameterValue::from(std::string_view { "0.5" }) },
        } };
        auto interval = row->get().add_to_graph_from_assignments(graph, "interval", bad_interval);
        ASSERT_FALSE(interval);
        EXPECT_NE(interval.error().message.find("AUTHORING_PARAMETER_VALUE_INVALID"), std::string::npos);

        const std::array<holoscan::authoring::ParameterAssignment, 2> zero_denominator { {
            { "samples_count", holoscan::authoring::ParameterValue::from(std::int64_t { 8 }) },
            { "sampling_interval", holoscan::authoring::ParameterValue::from(std::string_view { "1/0" }) },
        } };
        auto denominator = row->get().add_to_graph_from_assignments(graph, "denominator", zero_denominator);
        ASSERT_FALSE(denominator);
        EXPECT_NE(denominator.error().message.find("AUTHORING_PARAMETER_VALUE_INVALID"),
            std::string::npos);

        const std::array<holoscan::authoring::ParameterAssignment, 2> zero_period { {
            { "samples_count", holoscan::authoring::ParameterValue::from(std::int64_t { 8 }) },
            { "period_ns", holoscan::authoring::ParameterValue::from(std::int64_t { 0 }) },
        } };
        auto period = row->get().add_to_graph_from_assignments(graph, "period", zero_period);
        ASSERT_FALSE(period);
        EXPECT_NE(period.error().message.find("AUTHORING_PARAMETER_VALUE_INVALID"), std::string::npos);

        const std::array<holoscan::authoring::ParameterAssignment, 2> unknown_parameter { {
            { "samples_count", holoscan::authoring::ParameterValue::from(std::int64_t { 8 }) },
            { "frequency", holoscan::authoring::ParameterValue::from(std::int64_t { 1 }) },
        } };
        auto unknown = row->get().add_to_graph_from_assignments(graph, "unknown", unknown_parameter);
        ASSERT_FALSE(unknown);
        EXPECT_NE(unknown.error().message.find("HOLOLINK_OPERATOR_PARAMETER_UNKNOWN"), std::string::npos);
    }

    TEST(OperatorCatalog, ResolvesIQEncoderOpByRegistryName)
    {
        ASSERT_TRUE(register_hololink_operator_catalog());
        auto row = holoscan::authoring::authoring_registry().operator_type("hololink.operators.IQEncoderOp");
        ASSERT_TRUE(row);
        holoscan::Graph graph { "operator-catalog-iq-encoder" };
        auto discovery = row->get().add_to_graph_from_assignments(graph, "iq_encoder", {});
        EXPECT_TRUE(discovery) << discovery.error().message;

        const std::array<holoscan::authoring::ParameterAssignment, 1> assignments { {
            { "scale", holoscan::authoring::ParameterValue::from(2.0f) },
        } };
        auto added = row->get().add_to_graph_from_assignments(graph, "iq_encoder_scaled", assignments);
        EXPECT_TRUE(added) << added.error().message;
    }

    TEST(OperatorCatalog, ResolvesIQDecoderOpByRegistryName)
    {
        ASSERT_TRUE(register_hololink_operator_catalog());
        auto row = holoscan::authoring::authoring_registry().operator_type("hololink.operators.IQDecoderOp");
        ASSERT_TRUE(row);
        holoscan::Graph graph { "operator-catalog-iq-decoder" };
        auto discovery = row->get().add_to_graph_from_assignments(graph, "iq_decoder", {});
        EXPECT_TRUE(discovery) << discovery.error().message;

        const std::array<holoscan::authoring::ParameterAssignment, 1> missing_max_samples { {
            { "scale", holoscan::authoring::ParameterValue::from(2.0f) },
        } };
        auto missing = row->get().add_to_graph_from_assignments(graph, "missing", missing_max_samples);
        ASSERT_FALSE(missing);
        EXPECT_NE(missing.error().message.find("HOLOLINK_OPERATOR_PARAMETER_REQUIRED"), std::string::npos);

        const std::array<holoscan::authoring::ParameterAssignment, 1> zero_max_samples { {
            { "max_samples", holoscan::authoring::ParameterValue::from(std::int64_t { 0 }) },
        } };
        auto zero = row->get().add_to_graph_from_assignments(graph, "zero", zero_max_samples);
        ASSERT_FALSE(zero);
        EXPECT_NE(zero.error().message.find("AUTHORING_PARAMETER_VALUE_INVALID"), std::string::npos);

        const std::array<holoscan::authoring::ParameterAssignment, 2> assignments { {
            { "max_samples", holoscan::authoring::ParameterValue::from(std::int64_t { 4096 }) },
            { "scale", holoscan::authoring::ParameterValue::from(2.0f) },
        } };
        auto added = row->get().add_to_graph_from_assignments(graph, "iq_decoder_args", assignments);
        EXPECT_TRUE(added) << added.error().message;
    }

    TEST(OperatorCatalog, ResolvesSignalViewerOpByRegistryName)
    {
        ASSERT_TRUE(register_hololink_operator_catalog());
        auto row = holoscan::authoring::authoring_registry().operator_type("hololink.operators.SignalViewerOp");
        ASSERT_TRUE(row);
        holoscan::Graph graph { "operator-catalog-signal-viewer" };
        auto discovery = row->get().add_to_graph_from_assignments(graph, "signal_viewer", {});
        EXPECT_TRUE(discovery) << discovery.error().message;
    }
#endif // HOLOLINK_HAS_IMGUI

} // namespace
} // namespace hololink::operators
