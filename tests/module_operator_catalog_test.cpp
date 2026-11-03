// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>

#include <hololink/module/operators/linux_network_receiver.hpp>
#include <hololink/module/operators/module_operator_catalog.hpp>
#include <hololink/module/operators/sensor_factory.hpp>
#include <holoscan/authoring/register_operator.hpp>
#include <holoscan/authoring/registry.hpp>
#include <holoscan/core/graph.hpp>

// Construct-by-registry-name smoke tests for the hololink::module::operators::* operators
// registered in hololink_module/host/operators/module_operator_catalog.cpp -- mirrors
// tests/operator_catalog_test.cpp's own shape for the native hololink::operators::* family.
// Covers the Phase 1 (auto-registered, no live-object dependency) operators -- CsiToBayerOp,
// ImageProcessorOp, PackedFormatConverterOp, ComputeCrcOp (built only when
// HOLOLINK_MODULE_BUILD_CRC is on) -- Phase 2's first manual row, LinuxReceiverOp, Phase 3's
// RoceReceiverOp/FusaCoeCaptureOp, and Phase 4's HsbControllerOp. CheckCrcOp is deliberately out
// of scope (see compute_crc_op.hpp's note on CheckCrcParams) and not covered here.

namespace hololink::module::operators {
namespace {

    TEST(ModuleOperatorCatalog, RegistersEveryRowWithoutError)
    {
        auto status = register_hololink_module_operator_catalog();
        EXPECT_TRUE(status) << status.error().message;
    }

    TEST(ModuleOperatorCatalog, ResolvesCsiToBayerOpByRegistryName)
    {
        ASSERT_TRUE(register_hololink_module_operator_catalog());
        auto row = holoscan::authoring::authoring_registry().operator_type(
            "hololink_module.operators.CsiToBayerOp");
        ASSERT_TRUE(row);
        holoscan::Graph graph { "module-operator-catalog-csi-to-bayer" };
        auto added = row->get().add_to_graph_from_assignments(graph, "csi_to_bayer", {});
        EXPECT_TRUE(added) << added.error().message;
    }

    TEST(ModuleOperatorCatalog, ResolvesImageProcessorOpByRegistryName)
    {
        ASSERT_TRUE(register_hololink_module_operator_catalog());
        auto row = holoscan::authoring::authoring_registry().operator_type(
            "hololink_module.operators.ImageProcessorOp");
        ASSERT_TRUE(row);
        holoscan::Graph graph { "module-operator-catalog-image-processor" };
        auto added = row->get().add_to_graph_from_assignments(graph, "image_processor", {});
        EXPECT_TRUE(added) << added.error().message;
    }

    TEST(ModuleOperatorCatalog, ResolvesPackedFormatConverterOpByRegistryName)
    {
        ASSERT_TRUE(register_hololink_module_operator_catalog());
        auto row = holoscan::authoring::authoring_registry().operator_type(
            "hololink_module.operators.PackedFormatConverterOp");
        ASSERT_TRUE(row);
        holoscan::Graph graph { "module-operator-catalog-packed-format-converter" };
        auto discovery = row->get().add_to_graph_from_assignments(
            graph, "packed_format_converter", {});
        EXPECT_TRUE(discovery) << discovery.error().message;

        const std::array<holoscan::authoring::ParameterAssignment, 1> assignments { {
            { "bayer_format", holoscan::authoring::ParameterValue::from(std::string_view { "rggb" }) },
        } };
        auto added_with_args = row->get().add_to_graph_from_assignments(
            graph, "packed_format_converter_rggb", assignments);
        EXPECT_TRUE(added_with_args) << added_with_args.error().message;
    }

#ifdef HOLOLINK_MODULE_BUILD_CRC
    TEST(ModuleOperatorCatalog, ResolvesComputeCrcOpByRegistryName)
    {
        ASSERT_TRUE(register_hololink_module_operator_catalog());
        auto row = holoscan::authoring::authoring_registry().operator_type(
            "hololink_module.operators.ComputeCrcOp");
        ASSERT_TRUE(row);
        holoscan::Graph graph { "module-operator-catalog-compute-crc" };
        auto added = row->get().add_to_graph_from_assignments(graph, "compute_crc", {});
        EXPECT_TRUE(added) << added.error().message;
    }
#endif // HOLOLINK_MODULE_BUILD_CRC

    // The receiver rows and FusaCoeCaptureOp take their live objects as constructor handles: the
    // EnumerationMetadata as a required `metadata`, and optional `device_start`/`device_stop`
    // callbacks. HsbControllerOp takes `metadata`, `sensor_factory` and `network_receiver_factory`.

    template <typename T>
    holoscan::authoring::ConstructorHandle make_handle(
        const holoscan::authoring::ConstructorHandleType<T>& type, std::shared_ptr<T> owner)
    {
        auto handle = holoscan::authoring::ConstructorHandle::from_shared(type, std::move(owner));
        EXPECT_TRUE(handle) << handle.error().message;
        return handle ? std::move(*handle) : holoscan::authoring::ConstructorHandle {};
    }

    holoscan::authoring::ConstructorHandle metadata_handle()
    {
        return make_handle(enumeration_metadata_handle_type(),
            std::make_shared<hololink::module::EnumerationMetadata>());
    }

    holoscan::authoring::ConstructorHandle callback_handle(std::function<void()> callback)
    {
        return make_handle(holoscan::authoring::callback_handle_type(),
            std::make_shared<const holoscan::authoring::ConstructorCallback>(std::move(callback)));
    }

    holoscan::authoring::ParameterAssignment integer_assignment(
        std::string_view name, std::int64_t value)
    {
        return { name, holoscan::authoring::ParameterValue::from(value) };
    }

    // Checks one row that takes the receiver handles: discovery describes them without a device (a
    // default-constructed EnumerationMetadata{} never throws, so no placeholder is needed, unlike
    // native's receivers), the row builds from the metadata, a callback and `values`, and a missing
    // `metadata` fails before the Graph changes.
    void check_receiver_handle_row(
        std::string_view row_name, std::span<const holoscan::authoring::ParameterAssignment> values)
    {
        ASSERT_TRUE(register_hololink_module_operator_catalog());
        auto& registry = holoscan::authoring::authoring_registry();
        auto row = registry.operator_type(row_name);
        ASSERT_TRUE(row);

        const auto described = registry.describe_operator(row_name);
        ASSERT_TRUE(described) << described.error().message;
        ASSERT_EQ(described->constructor_handles.size(), 3U);
        EXPECT_EQ(described->constructor_handles[0].name, "metadata");
        EXPECT_EQ(described->constructor_handles[0].type_name, "hololink_module.EnumerationMetadata");
        EXPECT_TRUE(described->constructor_handles[0].required);
        EXPECT_EQ(described->constructor_handles[1].name, "device_start");
        EXPECT_EQ(described->constructor_handles[2].name, "device_stop");
        for (std::size_t callback = 1; callback < 3; ++callback) {
            EXPECT_EQ(described->constructor_handles[callback].type_name, "holoscan.VoidCallback");
            EXPECT_FALSE(described->constructor_handles[callback].required);
        }

        holoscan::Graph graph { "module-operator-catalog-receiver-handles" };
        const std::array handles {
            holoscan::authoring::ConstructorHandleAssignment { "metadata", metadata_handle() },
            holoscan::authoring::ConstructorHandleAssignment { "device_stop", callback_handle([] {}) },
        };
        const auto added = holoscan::authoring::add_registered_operator(graph, row->get(), "receiver",
            holoscan::authoring::OperatorConstructionInputs { .values = values, .handles = handles });
        ASSERT_TRUE(added) << added.error().message;

        const auto missing_metadata = holoscan::authoring::add_registered_operator(graph, row->get(),
            "receiver_without_metadata",
            holoscan::authoring::OperatorConstructionInputs { .values = values });
        ASSERT_FALSE(missing_metadata);
        EXPECT_NE(missing_metadata.error().message.find("AUTHORING_CONSTRUCTOR_HANDLE_REQUIRED"),
            std::string::npos)
            << missing_metadata.error().message;
    }

    TEST(ModuleOperatorCatalog, LinuxReceiverOpTakesItsMetadataAndCallbacksAsConstructorHandles)
    {
        const std::array values {
            integer_assignment("frame_size", 1024),
            integer_assignment("frame_context", 0),
        };
        check_receiver_handle_row("hololink_module.operators.LinuxReceiverOp", values);
    }

#ifdef HOLOLINK_BUILD_ROCE
    // Guarded because roce_receiver_op.cpp, unlike linux_receiver_op.cpp, is itself only built under
    // HOLOLINK_BUILD_ROCE, so the row only exists in that configuration.
    TEST(ModuleOperatorCatalog, RoceReceiverOpTakesItsMetadataAndCallbacksAsConstructorHandles)
    {
        const std::array values {
            integer_assignment("frame_size", 1024),
            integer_assignment("frame_context", 0),
        };
        check_receiver_handle_row("hololink_module.operators.RoceReceiverOp", values);
    }
#endif // HOLOLINK_BUILD_ROCE

#ifdef HOLOLINK_BUILD_FUSA
    TEST(ModuleOperatorCatalog, FusaCoeCaptureOpTakesItsMetadataAndCallbacksAsConstructorHandles)
    {
        // geometry is a nested assignment group (ParameterValue::from(std::span<const
        // ParameterAssignment>), kind kAssignments), not dotted field names -- matches
        // parse_geometry()'s own AssignmentView construction and the shape a real caller (Python's
        // dict kwarg, or examples/*.cpp's own ParameterAssignment construction) produces.
        const std::array<holoscan::authoring::ParameterAssignment, 6> geometry_fields { {
            { "start_byte", holoscan::authoring::ParameterValue::from(std::int64_t { 0 }) },
            { "bytes_per_line", holoscan::authoring::ParameterValue::from(std::int64_t { 2400 }) },
            { "pixel_width", holoscan::authoring::ParameterValue::from(std::int64_t { 1920 }) },
            { "pixel_height", holoscan::authoring::ParameterValue::from(std::int64_t { 1200 }) },
            { "pixel_format", holoscan::authoring::ParameterValue::from(std::string_view { "raw_10" }) },
            { "trailing_bytes", holoscan::authoring::ParameterValue::from(std::int64_t { 0 }) },
        } };
        const std::array<holoscan::authoring::ParameterAssignment, 3> values { {
            { "timeout", holoscan::authoring::ParameterValue::from(std::int64_t { 1500 }) },
            { "pixel_format", holoscan::authoring::ParameterValue::from(std::string_view { "raw_10" }) },
            { "geometry", holoscan::authoring::ParameterValue::from(std::span<const holoscan::authoring::ParameterAssignment> { geometry_fields }) },
        } };
        check_receiver_handle_row("hololink_module.operators.FusaCoeCaptureOp", values);
    }
#endif // HOLOLINK_BUILD_FUSA

    // A SensorFactory that never connects: HsbControllerOp only calls new_sensor() once the board
    // announces itself, which construction-only tests never reach.
    class NoSensorFactory final : public SensorFactory {
    public:
        std::shared_ptr<SensorDevice> new_sensor(const hololink::module::EnumerationMetadata&) override
        {
            return nullptr;
        }
    };

    TEST(ModuleOperatorCatalog, HsbControllerOpTakesItsMetadataAndFactoriesAsConstructorHandles)
    {
        constexpr std::string_view row_name = "hololink_module.operators.HsbControllerOp";
        ASSERT_TRUE(register_hololink_module_operator_catalog());
        auto& registry = holoscan::authoring::authoring_registry();
        auto row = registry.operator_type(row_name);
        ASSERT_TRUE(row);

        // Discovery: null factories are safe, neither is used before start_receiver().
        const auto described = registry.describe_operator(row_name);
        ASSERT_TRUE(described) << described.error().message;
        ASSERT_EQ(described->constructor_handles.size(), 3U);
        EXPECT_EQ(described->constructor_handles[0].name, "metadata");
        EXPECT_EQ(described->constructor_handles[1].name, "sensor_factory");
        EXPECT_EQ(described->constructor_handles[1].type_name, "hololink_module.SensorFactory");
        EXPECT_EQ(described->constructor_handles[2].name, "network_receiver_factory");
        EXPECT_EQ(described->constructor_handles[2].type_name,
            "hololink_module.NetworkReceiverFactory");
        for (const auto& handle : described->constructor_handles) {
            EXPECT_TRUE(handle.required) << handle.name;
        }

        holoscan::Graph graph { "module-operator-catalog-hsb-controller-handles" };
        const std::array values {
            integer_assignment("frame_size", 1024),
            integer_assignment("frame_context", 0),
        };
        const std::array handles {
            holoscan::authoring::ConstructorHandleAssignment { "metadata", metadata_handle() },
            holoscan::authoring::ConstructorHandleAssignment { "sensor_factory",
                make_handle<SensorFactory>(
                    sensor_factory_handle_type(), std::make_shared<NoSensorFactory>()) },
            holoscan::authoring::ConstructorHandleAssignment { "network_receiver_factory",
                make_handle(network_receiver_factory_handle_type(),
                    make_linux_network_receiver_factory()) },
        };
        const auto added = holoscan::authoring::add_registered_operator(graph, row->get(),
            "controller",
            holoscan::authoring::OperatorConstructionInputs { .values = values, .handles = handles });
        ASSERT_TRUE(added) << added.error().message;

        const std::array metadata_only {
            holoscan::authoring::ConstructorHandleAssignment { "metadata", metadata_handle() },
        };
        const auto rejected = holoscan::authoring::add_registered_operator(graph, row->get(),
            "controller_without_factories",
            holoscan::authoring::OperatorConstructionInputs {
                .values = values, .handles = metadata_only });
        ASSERT_FALSE(rejected);
        EXPECT_NE(rejected.error().message.find("AUTHORING_CONSTRUCTOR_HANDLE_REQUIRED"),
            std::string::npos)
            << rejected.error().message;
    }

} // namespace
} // namespace hololink::module::operators
