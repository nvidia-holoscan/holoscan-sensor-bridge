// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "hololink/module/operators/module_operator_catalog.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <cuda.h>
#include <fmt/format.h>

#include <holoscan/authoring/register_operator.hpp>

#include "hololink/module/operators/csi_to_bayer_op.hpp"
#include "hololink/module/operators/hsb_controller_op.hpp"
#include "hololink/module/operators/image_processor_op.hpp"
#include "hololink/module/operators/linux_receiver_op.hpp"
#include "hololink/module/operators/packed_format_converter_op.hpp"

#ifdef HOLOLINK_MODULE_BUILD_CRC
#include "hololink/module/operators/compute_crc_op.hpp"
#endif

#ifdef HOLOLINK_BUILD_ROCE
#include "hololink/module/operators/roce_receiver_op.hpp"
#endif

#ifdef HOLOLINK_BUILD_FUSA
#include "hololink/module/operators/fusa_coe_capture_op.hpp"
#endif

// Phase 1 of the hololink_module Python binding effort: the four hololink::module::operators::*
// operators with no live-object constructor/Params dependency auto-register via a plain
// HOLOSCAN_REGISTER_OPERATOR + the HOLOSCAN_PARAM_SCHEMA each one's own header now declares
// (see csi_to_bayer_op.hpp, image_processor_op.hpp, packed_format_converter_op.hpp,
// compute_crc_op.hpp), mirroring src/hololink/operators/operator_catalog.cpp's own
// CsiToBayerOp/ImageProcessorOp/UdpTransmitterOp/DiscardMetadataOp rows exactly.
//
// Phase 2 added the first manual row: LinuxReceiverOp, whose constructor takes an
// EnumerationMetadata + two std::function<void()> device start/stop callbacks that no
// language-neutral authoring parameter can carry -- they are constructor handles instead
// (see receiver_handle_inputs() below).
//
// Phase 3 adds RoceReceiverOp (the same handles -- RoceReceiverParams(module) has no
// ibv_name/ibv_port fields at all, unlike native's: the ibverbs device is resolved internally
// via the V1 service locator from metadata, not a per-operator parameter; confirmed by reading
// roce_receiver_op.hpp directly, not assumed) and FusaCoeCaptureOp (the same handles, but no
// frame_context: FusaCoeCaptureParams(module) has no such field).
//
// Phase 4 adds HsbControllerOp (constructor takes EnumerationMetadata +
// shared_ptr<SensorFactory> + shared_ptr<NetworkReceiverFactory>, no device_start/device_stop
// callbacks at all; see hsb_controller_handle_inputs() below).
//
// ComputeCrcOp and CheckCrcOp share a hololink::operators::CrcChannel passed to their
// constructors, which has no Python binding yet: the ComputeCrcOp registered here is
// default-constructed, without a channel, so its start() fails, and CheckCrcOp is not
// registered.

namespace hololink::module::operators {
namespace {

    using holoscan::Error;
    using holoscan::ErrorCode;
    using holoscan::expected;
    using holoscan::Graph;
    using holoscan::RuntimeError;
    using holoscan::authoring::ConstructorHandleInput;
    using holoscan::authoring::OpaqueOperatorRef;
    using holoscan::authoring::OperatorConstructionInputs;
    using holoscan::authoring::ParameterAssignment;
    using holoscan::authoring::ParameterValue;
    using holoscan::authoring::ParameterValueKind;
    using holoscan::authoring::RegisteredConstructorInputs;

    // ---------------------------------------------------------------------------
    // Shared assignment-parsing helpers, ported verbatim from
    // src/hololink/operators/operator_catalog.cpp's own anonymous-namespace helpers (that TU's
    // copy isn't exported/shared across translation units, so each manual-row file that needs
    // them keeps its own -- same as every other SDK module with manual rows; there is no shared,
    // non-duplicated version of this in the SDK yet). Trimmed to the subset the rows need: integer,
    // integer-sequence and bool here, string and enum (FusaCoeCaptureOp's row only) below; extend
    // as later rows need more.
    // ---------------------------------------------------------------------------

    [[noreturn]] void parameter_error(std::string_view token, std::string_view field = {},
        std::string_view detail = {})
    {
        std::string message { token };
        if (!field.empty()) {
            message.append(fmt::format(" field_name={}", field));
        }
        if (!detail.empty()) {
            message.append(" ");
            message.append(detail);
        }
        throw RuntimeError { Error { ErrorCode::kInvalidArgument, std::move(message) } };
    }

    // Borrows one structural assignment list while tracking the fields consumed by its parser, so
    // unrecognized fields can be rejected explicitly rather than silently ignored.
    class AssignmentView {
    public:
        AssignmentView(std::span<const ParameterAssignment> assignments, std::string path)
            : assignments_(assignments)
            , path_(std::move(path))
        {
        }

        [[nodiscard]] const ParameterValue* find(std::string_view name) const
        {
            if (std::find(accepted_.begin(), accepted_.end(), name) == accepted_.end()) {
                accepted_.push_back(name);
            }
            const auto found = std::find_if(assignments_.begin(), assignments_.end(),
                [&](const auto& item) { return item.name == name; });
            return found == assignments_.end() ? nullptr : &found->value;
        }

        [[nodiscard]] const ParameterValue& require(std::string_view name) const
        {
            const ParameterValue* value = find(name);
            if (value == nullptr) {
                parameter_error("HOLOLINK_OPERATOR_PARAMETER_REQUIRED", field_path(name));
            }
            return *value;
        }

        void reject_unknown() const
        {
            for (const ParameterAssignment& assignment : assignments_) {
                if (std::find(accepted_.begin(), accepted_.end(), assignment.name)
                    == accepted_.end()) {
                    std::string accepted;
                    for (const std::string_view name : accepted_) {
                        if (!accepted.empty()) {
                            accepted.append(",");
                        }
                        accepted.append(name);
                    }
                    parameter_error("HOLOLINK_OPERATOR_PARAMETER_UNKNOWN", field_path(assignment.name),
                        fmt::format("accepted={}", accepted));
                }
            }
        }

        [[nodiscard]] std::string field_path(std::string_view name) const
        {
            if (path_.empty()) {
                return std::string { name };
            }
            if (name.empty()) {
                return path_;
            }
            return path_ + "." + std::string { name };
        }

    private:
        std::span<const ParameterAssignment> assignments_;
        std::string path_;
        mutable std::vector<std::string_view> accepted_;
    };

    template <typename Integer>
    [[nodiscard]] Integer integer_value(const ParameterValue& value, std::string_view path)
    {
        const auto convert = [&]<typename Source>() -> std::optional<Integer> {
            Source source {};
            if (!value.scalar.try_as(source)) {
                return std::nullopt;
            }
            if (!std::in_range<Integer>(source)) {
                parameter_error("HOLOLINK_OPERATOR_PARAMETER_VALUE_INVALID", path,
                    fmt::format("value={} accepted_range=[{},{}]", source,
                        std::numeric_limits<Integer>::lowest(), std::numeric_limits<Integer>::max()));
            }
            return static_cast<Integer>(source);
        };
        if (value.kind != ParameterValueKind::kScalar) {
            parameter_error("HOLOLINK_OPERATOR_PARAMETER_TYPE_MISMATCH", path);
        }
        if (const auto result = convert.template operator()<std::int64_t>())
            return *result;
        if (const auto result = convert.template operator()<std::uint64_t>())
            return *result;
        if (const auto result = convert.template operator()<std::int32_t>())
            return *result;
        if (const auto result = convert.template operator()<std::uint32_t>())
            return *result;
        parameter_error("HOLOLINK_OPERATOR_PARAMETER_TYPE_MISMATCH", path);
    }

    template <typename Integer>
    [[nodiscard]] Integer integer_or(const AssignmentView& values, std::string_view name,
        Integer default_value)
    {
        const ParameterValue* value = values.find(name);
        return value == nullptr ? default_value : integer_value<Integer>(*value, values.field_path(name));
    }

    // Used by LinuxReceiverOp's `receiver_affinity` -- mirrors native's own helper of the same
    // name/shape (operator_catalog.cpp).
    [[nodiscard]] std::vector<int> integer_sequence_or(const AssignmentView& values, std::string_view name)
    {
        std::vector<int> result;
        const ParameterValue* field = values.find(name);
        if (field == nullptr) {
            return result;
        }
        if (field->kind != ParameterValueKind::kSequence) {
            parameter_error("HOLOLINK_OPERATOR_PARAMETER_TYPE_MISMATCH", values.field_path(name));
        }
        for (const ParameterValue& element : field->sequence_values()) {
            result.push_back(integer_value<int>(element, values.field_path(name)));
        }
        return result;
    }

    [[nodiscard]] bool bool_or(const AssignmentView& values, std::string_view name, bool default_value)
    {
        const ParameterValue* value = values.find(name);
        if (value == nullptr) {
            return default_value;
        }
        bool flag = false;
        if (value->kind != ParameterValueKind::kScalar || !value->scalar.try_as(flag)) {
            parameter_error("HOLOLINK_OPERATOR_PARAMETER_TYPE_MISMATCH", values.field_path(name));
        }
        return flag;
    }

#if defined(HOLOLINK_BUILD_FUSA)
    // string_value/string_or/EnumChoice/enum_value/enum_or are only needed by FusaCoeCaptureOp's
    // row below (interface, pixel_format, bayer_format) -- gated the same way to avoid an unused-
    // function warning in configurations that build this TU without FUSA.

    [[nodiscard]] std::string string_value(const ParameterValue& value, std::string_view path)
    {
        if (value.kind != ParameterValueKind::kString) {
            parameter_error("HOLOLINK_OPERATOR_PARAMETER_TYPE_MISMATCH", path);
        }
        return std::string { value.string };
    }

    [[nodiscard]] std::string string_or(const AssignmentView& values, std::string_view name,
        std::string default_value)
    {
        const ParameterValue* value = values.find(name);
        return value == nullptr ? std::move(default_value) : string_value(*value, values.field_path(name));
    }

    template <typename Enum>
    struct EnumChoice {
        std::string_view name;
        Enum value;
    };

    template <typename Enum, std::size_t Size>
    [[nodiscard]] Enum enum_value(const ParameterValue& value, std::string_view path,
        const std::array<EnumChoice<Enum>, Size>& choices)
    {
        if (value.kind == ParameterValueKind::kString) {
            const auto found = std::find_if(choices.begin(), choices.end(),
                [&](const auto& choice) { return choice.name == value.string; });
            if (found != choices.end()) {
                return found->value;
            }
        } else if (value.kind == ParameterValueKind::kScalar) {
            using Underlying = std::underlying_type_t<Enum>;
            const Underlying numeric = integer_value<Underlying>(value, path);
            const auto found = std::find_if(choices.begin(), choices.end(), [&](const auto& choice) {
                return static_cast<Underlying>(choice.value) == numeric;
            });
            if (found != choices.end()) {
                return found->value;
            }
        }
        parameter_error("HOLOLINK_OPERATOR_PARAMETER_TYPE_MISMATCH", path);
    }

    template <typename Enum, std::size_t Size>
    [[nodiscard]] Enum enum_or(const AssignmentView& values, std::string_view name, Enum default_value,
        const std::array<EnumChoice<Enum>, Size>& choices)
    {
        const ParameterValue* value = values.find(name);
        return value == nullptr ? default_value : enum_value(*value, values.field_path(name), choices);
    }

    // Mirrors native's own kPixelFormats/kBayerFormats (operator_catalog.cpp), using module's own
    // csi::PixelFormat/BayerFormat (deliberately integer-value-identical to native's by design, but
    // a distinct C++ type, so these tables can't just be reused directly).
    constexpr std::array<EnumChoice<hololink::module::csi::PixelFormat>, 3> kModulePixelFormats {
        EnumChoice<hololink::module::csi::PixelFormat> { "raw_8", hololink::module::csi::PixelFormat::RAW_8 },
        EnumChoice<hololink::module::csi::PixelFormat> { "raw_10", hololink::module::csi::PixelFormat::RAW_10 },
        EnumChoice<hololink::module::csi::PixelFormat> { "raw_12", hololink::module::csi::PixelFormat::RAW_12 },
    };

    constexpr std::array<EnumChoice<hololink::module::csi::BayerFormat>, 4> kModuleBayerFormats {
        EnumChoice<hololink::module::csi::BayerFormat> { "bggr", hololink::module::csi::BayerFormat::BGGR },
        EnumChoice<hololink::module::csi::BayerFormat> { "rggb", hololink::module::csi::BayerFormat::RGGB },
        EnumChoice<hololink::module::csi::BayerFormat> { "gbrg", hololink::module::csi::BayerFormat::GBRG },
        EnumChoice<hololink::module::csi::BayerFormat> { "grbg", hololink::module::csi::BayerFormat::GRBG },
    };

    // Parses a nested `geometry` assignment group (a dict from Python, or a directly-nested
    // ParameterAssignment span from C++) into a module::csi::CsiGeometry. Field names match
    // CsiGeometry's own, same shape as native's parse_geometry() (operator_catalog.cpp) and as
    // CsiToBayerOp(module)'s own schema-declared `geometry` field from Phase 1.
    [[nodiscard]] hololink::module::csi::CsiGeometry parse_geometry(const AssignmentView& values)
    {
        const ParameterValue& field = values.require("geometry");
        if (field.kind != ParameterValueKind::kAssignments) {
            parameter_error("HOLOLINK_OPERATOR_PARAMETER_TYPE_MISMATCH", values.field_path("geometry"));
        }
        AssignmentView geometry_values { field.nested_assignments(), values.field_path("geometry") };
        hololink::module::csi::CsiGeometry geometry;
        geometry.start_byte = integer_value<uint32_t>(
            geometry_values.require("start_byte"), geometry_values.field_path("start_byte"));
        geometry.bytes_per_line = integer_value<uint32_t>(
            geometry_values.require("bytes_per_line"), geometry_values.field_path("bytes_per_line"));
        geometry.pixel_width = integer_value<uint32_t>(
            geometry_values.require("pixel_width"), geometry_values.field_path("pixel_width"));
        geometry.pixel_height = integer_value<uint32_t>(
            geometry_values.require("pixel_height"), geometry_values.field_path("pixel_height"));
        geometry.pixel_format = enum_value(geometry_values.require("pixel_format"),
            geometry_values.field_path("pixel_format"), kModulePixelFormats);
        geometry.trailing_bytes = integer_value<uint32_t>(
            geometry_values.require("trailing_bytes"), geometry_values.field_path("trailing_bytes"));
        geometry_values.reject_unknown();
        return geometry;
    }
#endif // HOLOLINK_BUILD_FUSA

    // ---------------------------------------------------------------------------
    // Constructor handles of the module rows (CLARAHOLOS-3438). The receivers and FusaCoeCaptureOp
    // take a required `metadata` (the board's EnumerationMetadata, passed by value to the operator)
    // and two optional `device_start`/`device_stop` callbacks, which Python fills from any callable
    // (typically the camera's start and stop methods). HsbControllerOp takes `metadata`, a
    // `sensor_factory` and a `network_receiver_factory`. The registry checks them before calling the
    // row. The CUDA context the receiver thread makes current is an ordinary `frame_context` value
    // instead: it has no shared owner to pass as a handle.
    // ---------------------------------------------------------------------------

    constexpr holoscan::authoring::ConstructorHandleDescriptor kEnumerationMetadataHandleDescriptor {
        .name = "hololink_module.EnumerationMetadata",
        .description = "Sensor Bridge enumeration metadata (hololink_module.EnumerationMetadata).",
    };

    constexpr holoscan::authoring::ConstructorHandleDescriptor kSensorFactoryHandleDescriptor {
        .name = "hololink_module.SensorFactory",
        .description = "Builds and arms the camera (a hololink_module.SensorFactory subclass).",
    };

    constexpr holoscan::authoring::ConstructorHandleDescriptor kNetworkReceiverFactoryHandleDescriptor {
        .name = "hololink_module.NetworkReceiverFactory",
        .description = "Builds the data-plane receiver (hololink_module.make_*_network_receiver_factory()).",
    };

    constexpr ConstructorHandleInput kMetadataHandleInput {
        .name = "metadata",
        .description = "The enumeration metadata of the Sensor Bridge channel to receive from.",
        .type = &kEnumerationMetadataHandleDescriptor,
    };

    [[nodiscard]] std::span<const ConstructorHandleInput> receiver_handle_inputs()
    {
        // Built at run time: the callback type's descriptor comes from a Holoscan function.
        static const std::array inputs {
            kMetadataHandleInput,
            ConstructorHandleInput {
                .name = "device_start",
                .description = "Called when the operator starts, typically the camera's start method.",
                .type = &holoscan::authoring::callback_handle_type().descriptor(),
                .required = false,
            },
            ConstructorHandleInput {
                .name = "device_stop",
                .description = "Called when the operator stops, typically the camera's stop method.",
                .type = &holoscan::authoring::callback_handle_type().descriptor(),
                .required = false,
            },
        };
        return inputs;
    }

    constexpr std::array kHsbControllerHandleInputs {
        kMetadataHandleInput,
        ConstructorHandleInput {
            .name = "sensor_factory",
            .description = "Builds and arms the camera each time the board (re)connects.",
            .type = &kSensorFactoryHandleDescriptor,
        },
        ConstructorHandleInput {
            .name = "network_receiver_factory",
            .description = "Builds the data-plane receiver each time the board (re)connects.",
            .type = &kNetworkReceiverFactoryHandleDescriptor,
        },
    };

    // The constructor inputs of a row with the given handles and thunks.
    template <typename Thunks>
    [[nodiscard]] const RegisteredConstructorInputs* constructor_inputs(
        std::span<const ConstructorHandleInput> handles)
    {
        static const RegisteredConstructorInputs inputs {
            .handles = handles,
            .add_to_graph_from_inputs = &Thunks::add_from_inputs,
            .add_for_description = &Thunks::add_for_description,
        };
        return &inputs;
    }

    struct ReceiverHandles {
        hololink::module::EnumerationMetadata metadata;
        std::function<void()> device_start;
        std::function<void()> device_stop;
    };

    // Reads receiver_handle_inputs(); omitted callbacks come back empty.
    [[nodiscard]] expected<ReceiverHandles, Error> receiver_handles(
        const OperatorConstructionInputs& inputs)
    {
        auto metadata = inputs.require_shared("metadata", enumeration_metadata_handle_type());
        if (!metadata) {
            return holoscan::make_unexpected(std::move(metadata).error());
        }
        auto device_start = inputs.callback("device_start");
        if (!device_start) {
            return holoscan::make_unexpected(std::move(device_start).error());
        }
        auto device_stop = inputs.callback("device_stop");
        if (!device_stop) {
            return holoscan::make_unexpected(std::move(device_stop).error());
        }
        return ReceiverHandles { **metadata, std::move(*device_start), std::move(*device_stop) };
    }

    // The CUDA context the receiver thread makes current (CUDA context state is thread-local): the
    // numeric value of a CUcontext the application keeps retained, e.g. int(ctx) for the
    // cuda.bindings.driver.CUcontext from cuDevicePrimaryCtxRetain().
    [[nodiscard]] CUcontext frame_context_value(const AssignmentView& values)
    {
        return reinterpret_cast<CUcontext>(integer_value<std::uintptr_t>(
            values.require("frame_context"), values.field_path("frame_context")));
    }

    // ---------------------------------------------------------------------------
    // LinuxReceiverOp(module) -- manual row. BaseReceiverOp(module)'s constructor takes
    // EnumerationMetadata + two std::function<void()> callbacks, none of which a schema-backed
    // parameter can carry -- they arrive as receiver_handle_inputs() above (same shape as native's
    // receiver rows, metadata-by-value instead of shared_ptr<DataChannel>).
    //
    // Unlike native's LinuxReceiverOp, discovery-mode construction here needs no fake-but-valid
    // placeholder channel: EnumerationMetadata is a plain std::map subclass with no custom
    // constructor (confirmed by reading hololink/module/enumeration_metadata.hpp), so a
    // default-constructed EnumerationMetadata{} never throws, and
    // BaseReceiverOp(module)::setup() -- the only method holoscan.registry.operators()'s
    // discovery probe calls -- never dereferences metadata_'s contents (only start_receiver(),
    // never called during discovery, does). Native's own make_discovery_data_channel() exists
    // specifically because DataChannel's constructor *does* throw for a null channel; that
    // constraint doesn't exist here.
    // ---------------------------------------------------------------------------

    constexpr std::array<std::string_view, 2> kLinuxReceiverRequiredParameters {
        "frame_size", "frame_context"
    };

    struct LinuxReceiverRegistrationThunks final
        : holoscan::authoring::ManualOperatorRegistrationThunks<LinuxReceiverOp> {
        static OpaqueOperatorRef add(Graph& graph, std::string_view name,
            hololink::module::EnumerationMetadata metadata, std::function<void()> device_start,
            std::function<void()> device_stop, LinuxReceiverParams params)
        {
            auto op = graph.op<LinuxReceiverOp>(
                name, std::move(metadata), std::move(device_start), std::move(device_stop));
            op.set_params(std::move(params));
            return make_operator_ref(op);
        }

        static expected<OpaqueOperatorRef, Error> add_from_inputs(
            Graph& graph, std::string_view name, const OperatorConstructionInputs& inputs)
        {
            auto handles = receiver_handles(inputs);
            if (!handles) {
                return holoscan::make_unexpected(std::move(handles).error());
            }
            return invoke_add([&] {
                AssignmentView values { inputs.values, {} };
                LinuxReceiverParams params;
                params.frame_context = frame_context_value(values);
                params.frame_size = integer_value<size_t>(
                    values.require("frame_size"), values.field_path("frame_size"));
                params.page_size = integer_or<size_t>(values, "page_size", 0);
                params.pages = integer_or<uint32_t>(values, "pages", 2);
                params.queue_size = integer_or<uint32_t>(values, "queue_size", 1);
                params.receiver_affinity = integer_sequence_or(values, "receiver_affinity");
                params.frame_limit = integer_or<int64_t>(values, "frame_limit", 0);
                params.frame_limit_ends_run = bool_or(values, "frame_limit_ends_run", false);
                values.reject_unknown();
                return add(graph, name, std::move(handles->metadata), std::move(handles->device_start),
                    std::move(handles->device_stop), std::move(params));
            });
        }

        // Discovery: see this section's top-of-section comment above for why a plain
        // default-constructed EnumerationMetadata{} (not a fabricated placeholder) is safe here.
        static expected<OpaqueOperatorRef, Error> add_for_description(
            Graph& graph, std::string_view name)
        {
            return invoke_add([&] {
                return add(graph, name, hololink::module::EnumerationMetadata {}, {}, {},
                    LinuxReceiverParams { .frame_size = 1 });
            });
        }

        static void set_params(const OpaqueOperatorRef&,
            const holoscan::authoring::MaterializedParameters&)
        {
            throw RuntimeError {
                Error { ErrorCode::kNotSupported, "AUTHORING_MANUAL_ROW_PARAMETERS_IMMUTABLE" }
            };
        }
    };

    [[nodiscard]] holoscan::authoring::RegisteredOperatorType linux_receiver_row()
    {
        using Thunks = LinuxReceiverRegistrationThunks;
        return holoscan::authoring::RegisteredOperatorType {
            .name = "hololink_module.operators.LinuxReceiverOp",
            .param_type = &holoscan::param_type_descriptor<holoscan::NoParams>(),
            .param_schema = nullptr,
            .parameter_introspection = holoscan::authoring::OperatorParameterIntrospection::kPassThrough,
            .required_parameters = kLinuxReceiverRequiredParameters,
            .add_to_graph = &Thunks::add_without_constructor_handles,
            .add_to_graph_from_assignments = &Thunks::reject_without_constructor_handles,
            .set_params = &Thunks::set_params,
            .port = &Thunks::port,
            .ports = &Thunks::ports,
            .region_add = &Thunks::region_add,
            .partition_add = &Thunks::partition_add,
            .lifecycle_before = &Thunks::lifecycle_before,
            .lifecycle_allow_self_restart = &Thunks::lifecycle_allow_self_restart,
            .lifecycle_allow_external_restart = &Thunks::lifecycle_allow_external_restart,
            .lifecycle_allow_restart_request = &Thunks::lifecycle_allow_restart_request,
            .constructor_inputs = constructor_inputs<Thunks>(receiver_handle_inputs()),
        };
    }

#ifdef HOLOLINK_BUILD_ROCE
    // ---------------------------------------------------------------------------
    // RoceReceiverOp(module) -- manual row, same shape and reasoning as LinuxReceiverOp(module)
    // above (the same receiver_handle_inputs()). Guarded separately because roce_receiver_op.cpp
    // (unlike linux_receiver_op.cpp) is itself only built under HOLOLINK_BUILD_ROCE -- see
    // hololink_module/host/operators/CMakeLists.txt. Unlike native's RoceReceiverParams, module's
    // has no ibv_name/ibv_port fields at all -- the ibverbs device is resolved internally via the
    // V1 service locator (RoceReceiverInterfaceV1::get_service(metadata_)), not a per-operator
    // parameter; confirmed by reading roce_receiver_op.hpp directly, not assumed.
    // ---------------------------------------------------------------------------

    constexpr std::array<std::string_view, 2> kRoceReceiverRequiredParameters {
        "frame_size", "frame_context"
    };

    struct RoceReceiverRegistrationThunks final
        : holoscan::authoring::ManualOperatorRegistrationThunks<RoceReceiverOp> {
        static OpaqueOperatorRef add(Graph& graph, std::string_view name,
            hololink::module::EnumerationMetadata metadata, std::function<void()> device_start,
            std::function<void()> device_stop, RoceReceiverParams params)
        {
            auto op = graph.op<RoceReceiverOp>(
                name, std::move(metadata), std::move(device_start), std::move(device_stop));
            op.set_params(std::move(params));
            return make_operator_ref(op);
        }

        static expected<OpaqueOperatorRef, Error> add_from_inputs(
            Graph& graph, std::string_view name, const OperatorConstructionInputs& inputs)
        {
            auto handles = receiver_handles(inputs);
            if (!handles) {
                return holoscan::make_unexpected(std::move(handles).error());
            }
            return invoke_add([&] {
                AssignmentView values { inputs.values, {} };
                RoceReceiverParams params;
                params.frame_context = frame_context_value(values);
                params.frame_size = integer_value<size_t>(
                    values.require("frame_size"), values.field_path("frame_size"));
                params.page_size = integer_or<size_t>(values, "page_size", 0);
                params.pages = integer_or<uint32_t>(values, "pages", 2);
                params.queue_size = integer_or<uint32_t>(values, "queue_size", 1);
                params.metadata_offset = integer_or<size_t>(values, "metadata_offset", 0);
                params.frame_limit = integer_or<int64_t>(values, "frame_limit", 0);
                params.frame_limit_ends_run = bool_or(values, "frame_limit_ends_run", false);
                values.reject_unknown();
                return add(graph, name, std::move(handles->metadata), std::move(handles->device_start),
                    std::move(handles->device_stop), std::move(params));
            });
        }

        // Discovery: see LinuxReceiverOp(module)'s row above for why a plain default-constructed
        // EnumerationMetadata{} is safe here.
        static expected<OpaqueOperatorRef, Error> add_for_description(
            Graph& graph, std::string_view name)
        {
            return invoke_add([&] {
                return add(graph, name, hololink::module::EnumerationMetadata {}, {}, {},
                    RoceReceiverParams { .frame_size = 1 });
            });
        }

        static void set_params(const OpaqueOperatorRef&,
            const holoscan::authoring::MaterializedParameters&)
        {
            throw RuntimeError {
                Error { ErrorCode::kNotSupported, "AUTHORING_MANUAL_ROW_PARAMETERS_IMMUTABLE" }
            };
        }
    };

    [[nodiscard]] holoscan::authoring::RegisteredOperatorType roce_receiver_row()
    {
        using Thunks = RoceReceiverRegistrationThunks;
        return holoscan::authoring::RegisteredOperatorType {
            .name = "hololink_module.operators.RoceReceiverOp",
            .param_type = &holoscan::param_type_descriptor<holoscan::NoParams>(),
            .param_schema = nullptr,
            .parameter_introspection = holoscan::authoring::OperatorParameterIntrospection::kPassThrough,
            .required_parameters = kRoceReceiverRequiredParameters,
            .add_to_graph = &Thunks::add_without_constructor_handles,
            .add_to_graph_from_assignments = &Thunks::reject_without_constructor_handles,
            .set_params = &Thunks::set_params,
            .port = &Thunks::port,
            .ports = &Thunks::ports,
            .region_add = &Thunks::region_add,
            .partition_add = &Thunks::partition_add,
            .lifecycle_before = &Thunks::lifecycle_before,
            .lifecycle_allow_self_restart = &Thunks::lifecycle_allow_self_restart,
            .lifecycle_allow_external_restart = &Thunks::lifecycle_allow_external_restart,
            .lifecycle_allow_restart_request = &Thunks::lifecycle_allow_restart_request,
            .constructor_inputs = constructor_inputs<Thunks>(receiver_handle_inputs()),
        };
    }
#endif // HOLOLINK_BUILD_ROCE

#ifdef HOLOLINK_BUILD_FUSA
    // ---------------------------------------------------------------------------
    // FusaCoeCaptureOp(module) -- manual row. Mirrors native's own fusa_coe_capture_row() shape
    // (operator_catalog.cpp) closely: FusaCoeCaptureParams(module) carries its resource handle
    // (metadata, here -- hololink_channel for native) directly in Params rather than the
    // constructor, since FusaCoeCaptureOp has no CRTP base requiring a non-default constructor
    // (unlike LinuxReceiverOp/RoceReceiverOp(module) above) -- so even discovery-mode construction
    // (all-default Params) is otherwise fine; only a real capture needs the metadata, which arrives
    // as receiver_handle_inputs() (FusaCoeCaptureParams(module) has no frame_context field). Also
    // requires calling FusaCoeCaptureOp::make_geometry_adapter()/camera->configure_converter()
    // before construction to resolve geometry -- the Python-facing equivalent of this is
    // resolve_fusa_coe_geometry() in _hololink_module.cpp, mirroring native's own
    // resolve_geometry()/resolve_geometry_for() in _hololink.cpp; geometry arrives here as an
    // already-resolved nested `geometry` assignment group, same shape as CsiToBayerOp(module)'s
    // schema-declared `geometry` field. Unconditionally required (via parse_geometry()'s own
    // values.require("geometry")), matching native's identical behavior -- not reflected in
    // kFusaCoeCaptureRequiredParameters below (neither is native's), but real: a call missing
    // "geometry" is rejected. frame_size_bytes (the opaque byte-blob capture mode
    // FusaCoeCaptureParams also supports) has no Python-facing caller in either tree's registration
    // row currently -- add one if a phase ever needs it.
    // ---------------------------------------------------------------------------

    constexpr std::array<std::string_view, 0> kFusaCoeCaptureRequiredParameters {};

    struct FusaCoeCaptureRegistrationThunks final
        : holoscan::authoring::ManualOperatorRegistrationThunks<FusaCoeCaptureOp> {
        static OpaqueOperatorRef add(Graph& graph, std::string_view name, FusaCoeCaptureParams params)
        {
            auto op = graph.op<FusaCoeCaptureOp>(name);
            op.set_params(std::move(params));
            return make_operator_ref(op);
        }

        static expected<OpaqueOperatorRef, Error> add_from_inputs(
            Graph& graph, std::string_view name, const OperatorConstructionInputs& inputs);

        // Discovery: no metadata, no real capture -- matches FusaCoeCaptureParams's own all-default
        // resting state (metadata == EnumerationMetadata{}, an empty map, never throws -- see
        // LinuxReceiverOp(module)'s row above for the same reasoning).
        static expected<OpaqueOperatorRef, Error> add_for_description(
            Graph& graph, std::string_view name)
        {
            return invoke_add([&] { return add(graph, name, FusaCoeCaptureParams {}); });
        }

        static void set_params(const OpaqueOperatorRef&,
            const holoscan::authoring::MaterializedParameters&)
        {
            throw RuntimeError {
                Error { ErrorCode::kNotSupported, "AUTHORING_MANUAL_ROW_PARAMETERS_IMMUTABLE" }
            };
        }
    };

    expected<OpaqueOperatorRef, Error> FusaCoeCaptureRegistrationThunks::add_from_inputs(
        Graph& graph, std::string_view name, const OperatorConstructionInputs& inputs)
    {
        auto handles = receiver_handles(inputs);
        if (!handles) {
            return holoscan::make_unexpected(std::move(handles).error());
        }
        return invoke_add([&] {
            AssignmentView values { inputs.values, {} };
            FusaCoeCaptureParams params;
            params.metadata = std::move(handles->metadata);
            params.device_start = std::move(handles->device_start);
            params.device_stop = std::move(handles->device_stop);
            params.interface = string_or(values, "interface", "");
            params.timeout = integer_or<uint32_t>(values, "timeout", 0);
            params.stall_timeout_ms = integer_or<uint32_t>(values, "stall_timeout_ms", 0);
            params.frame_limit = integer_or<int64_t>(values, "frame_limit", 0);
            params.frame_limit_ends_run = bool_or(values, "frame_limit_ends_run", false);
            // Names the capture in its stall messages, as the C++ apps do.
            params.name = std::string(name);
            params.pixel_format = enum_or(values, "pixel_format",
                hololink::module::csi::PixelFormat::RAW_8, kModulePixelFormats);
            params.bayer_format = enum_or(values, "bayer_format",
                hololink::module::csi::BayerFormat::RGGB, kModuleBayerFormats);
            params.packetizer_enabled = bool_or(values, "packetizer_enabled", false);
            params.geometry = parse_geometry(values);

            if (const ParameterValue* field = values.find("mac_addr")) {
                if (field->kind != ParameterValueKind::kSequence) {
                    parameter_error(
                        "HOLOLINK_OPERATOR_PARAMETER_TYPE_MISMATCH", values.field_path("mac_addr"));
                }
                for (const ParameterValue& element : field->sequence_values()) {
                    params.mac_addr.push_back(
                        integer_value<uint8_t>(element, values.field_path("mac_addr")));
                }
            }

            values.reject_unknown();
            return add(graph, name, std::move(params));
        });
    }

    [[nodiscard]] holoscan::authoring::RegisteredOperatorType fusa_coe_capture_row()
    {
        using Thunks = FusaCoeCaptureRegistrationThunks;
        return holoscan::authoring::RegisteredOperatorType {
            .name = "hololink_module.operators.FusaCoeCaptureOp",
            .param_type = &holoscan::param_type_descriptor<holoscan::NoParams>(),
            .param_schema = nullptr,
            .parameter_introspection = holoscan::authoring::OperatorParameterIntrospection::kPassThrough,
            .required_parameters = kFusaCoeCaptureRequiredParameters,
            .add_to_graph = &Thunks::add_without_constructor_handles,
            .add_to_graph_from_assignments = &Thunks::reject_without_constructor_handles,
            .set_params = &Thunks::set_params,
            .port = &Thunks::port,
            .ports = &Thunks::ports,
            .region_add = &Thunks::region_add,
            .partition_add = &Thunks::partition_add,
            .lifecycle_before = &Thunks::lifecycle_before,
            .lifecycle_allow_self_restart = &Thunks::lifecycle_allow_self_restart,
            .lifecycle_allow_external_restart = &Thunks::lifecycle_allow_external_restart,
            .lifecycle_allow_restart_request = &Thunks::lifecycle_allow_restart_request,
            .constructor_inputs = constructor_inputs<Thunks>(receiver_handle_inputs()),
        };
    }
#endif // HOLOLINK_BUILD_FUSA

    // ---------------------------------------------------------------------------
    // HsbControllerOp -- manual row. HsbControllerOp's constructor takes EnumerationMetadata +
    // shared_ptr<SensorFactory> + shared_ptr<NetworkReceiverFactory> (confirmed by reading
    // hsb_controller_op.hpp directly), all three as kHsbControllerHandleInputs -- no
    // device_start/device_stop callbacks (the sensor's own start/stop lifecycle runs through
    // SensorFactory::new_sensor()/SensorDevice::stop_sensor() instead, not the generic
    // device_start/device_stop hooks LinuxReceiverOp/RoceReceiverOp(module) use). frame_context is
    // an HsbControllerParams field (like frame_size/page_size/etc.), reached via set_params() after
    // construction, same as LinuxReceiverOp(module)'s row above.
    //
    // Discovery-mode construction needs no fake-but-valid sensor_factory/network_receiver_factory:
    // neither is dereferenced by HsbControllerOp's constructor body (plain member-initializer-list
    // moves, confirmed by reading hsb_controller_op.hpp) or by BaseReceiverOp(module)::setup() (the
    // only method discovery mode calls) -- both are only actually used from start_receiver(), never
    // called during discovery. Null shared_ptrs are therefore safe here, same reasoning as
    // LinuxReceiverOp(module)'s row using a default-constructed EnumerationMetadata{}.
    // ---------------------------------------------------------------------------

    constexpr std::array<std::string_view, 2> kHsbControllerRequiredParameters {
        "frame_size", "frame_context"
    };

    struct HsbControllerRegistrationThunks final
        : holoscan::authoring::ManualOperatorRegistrationThunks<HsbControllerOp> {
        static OpaqueOperatorRef add(Graph& graph, std::string_view name,
            hololink::module::EnumerationMetadata metadata,
            std::shared_ptr<SensorFactory> sensor_factory,
            std::shared_ptr<NetworkReceiverFactory> network_receiver_factory,
            HsbControllerParams params)
        {
            auto op = graph.op<HsbControllerOp>(
                name, std::move(metadata), std::move(sensor_factory), std::move(network_receiver_factory));
            op.set_params(std::move(params));
            return make_operator_ref(op);
        }

        static expected<OpaqueOperatorRef, Error> add_from_inputs(
            Graph& graph, std::string_view name, const OperatorConstructionInputs& inputs)
        {
            auto metadata = inputs.require_shared("metadata", enumeration_metadata_handle_type());
            if (!metadata) {
                return holoscan::make_unexpected(std::move(metadata).error());
            }
            auto sensor_factory = inputs.require_shared("sensor_factory", sensor_factory_handle_type());
            if (!sensor_factory) {
                return holoscan::make_unexpected(std::move(sensor_factory).error());
            }
            auto network_receiver_factory = inputs.require_shared(
                "network_receiver_factory", network_receiver_factory_handle_type());
            if (!network_receiver_factory) {
                return holoscan::make_unexpected(std::move(network_receiver_factory).error());
            }
            return invoke_add([&] {
                AssignmentView values { inputs.values, {} };
                HsbControllerParams params;
                params.frame_context = frame_context_value(values);
                params.frame_size = integer_value<size_t>(
                    values.require("frame_size"), values.field_path("frame_size"));
                params.page_size = integer_or<size_t>(values, "page_size", 0);
                params.pages = integer_or<uint32_t>(values, "pages", 2);
                params.queue_size = integer_or<uint32_t>(values, "queue_size", 1);
                params.metadata_offset = integer_or<size_t>(values, "metadata_offset", 0);
                params.frame_limit = integer_or<int64_t>(values, "frame_limit", 0);
                params.frame_limit_ends_run = bool_or(values, "frame_limit_ends_run", false);
                // watchdog_timeout_s is left at HsbControllerParams's own default (0.5s): no caller
                // in this porting round needs to tune it (the pre-port app never did either), and
                // this file has no real/double-valued assignment-parsing helper yet (unlike native's
                // operator_catalog.cpp's real_value<Real>/real_or<Real>, added there for ArgusIspOp)
                // -- add one here, rather than an awkward integer encoding, if a future caller needs
                // this configurable.
                values.reject_unknown();
                return add(graph, name, **metadata, std::move(*sensor_factory),
                    std::move(*network_receiver_factory), std::move(params));
            });
        }

        // Discovery: see this section's top-of-section comment above for why null
        // sensor_factory/network_receiver_factory shared_ptrs are safe here.
        static expected<OpaqueOperatorRef, Error> add_for_description(
            Graph& graph, std::string_view name)
        {
            return invoke_add([&] {
                return add(graph, name, hololink::module::EnumerationMetadata {}, nullptr, nullptr,
                    HsbControllerParams { .frame_size = 1 });
            });
        }

        static void set_params(const OpaqueOperatorRef&,
            const holoscan::authoring::MaterializedParameters&)
        {
            throw RuntimeError {
                Error { ErrorCode::kNotSupported, "AUTHORING_MANUAL_ROW_PARAMETERS_IMMUTABLE" }
            };
        }
    };

    [[nodiscard]] holoscan::authoring::RegisteredOperatorType hsb_controller_row()
    {
        using Thunks = HsbControllerRegistrationThunks;
        return holoscan::authoring::RegisteredOperatorType {
            .name = "hololink_module.operators.HsbControllerOp",
            .param_type = &holoscan::param_type_descriptor<holoscan::NoParams>(),
            .param_schema = nullptr,
            .parameter_introspection = holoscan::authoring::OperatorParameterIntrospection::kPassThrough,
            .required_parameters = kHsbControllerRequiredParameters,
            .add_to_graph = &Thunks::add_without_constructor_handles,
            .add_to_graph_from_assignments = &Thunks::reject_without_constructor_handles,
            .set_params = &Thunks::set_params,
            .port = &Thunks::port,
            .ports = &Thunks::ports,
            .region_add = &Thunks::region_add,
            .partition_add = &Thunks::partition_add,
            .lifecycle_before = &Thunks::lifecycle_before,
            .lifecycle_allow_self_restart = &Thunks::lifecycle_allow_self_restart,
            .lifecycle_allow_external_restart = &Thunks::lifecycle_allow_external_restart,
            .lifecycle_allow_restart_request = &Thunks::lifecycle_allow_restart_request,
            .constructor_inputs = constructor_inputs<Thunks>(kHsbControllerHandleInputs),
        };
    }

} // namespace

const holoscan::authoring::ConstructorHandleType<hololink::module::EnumerationMetadata>&
enumeration_metadata_handle_type()
{
    static const holoscan::authoring::ConstructorHandleType<hololink::module::EnumerationMetadata>
        type { kEnumerationMetadataHandleDescriptor };
    return type;
}

const holoscan::authoring::ConstructorHandleType<SensorFactory>& sensor_factory_handle_type()
{
    static const holoscan::authoring::ConstructorHandleType<SensorFactory> type {
        kSensorFactoryHandleDescriptor
    };
    return type;
}

const holoscan::authoring::ConstructorHandleType<NetworkReceiverFactory>&
network_receiver_factory_handle_type()
{
    static const holoscan::authoring::ConstructorHandleType<NetworkReceiverFactory> type {
        kNetworkReceiverFactoryHandleDescriptor
    };
    return type;
}

HOLOSCAN_REGISTER_OPERATOR(
    hololink::module::operators::CsiToBayerOp, "hololink_module.operators.CsiToBayerOp");
HOLOSCAN_REGISTER_OPERATOR(
    hololink::module::operators::ImageProcessorOp, "hololink_module.operators.ImageProcessorOp");
HOLOSCAN_REGISTER_OPERATOR(hololink::module::operators::PackedFormatConverterOp,
    "hololink_module.operators.PackedFormatConverterOp");
#ifdef HOLOLINK_MODULE_BUILD_CRC
HOLOSCAN_REGISTER_OPERATOR(
    hololink::module::operators::ComputeCrcOp, "hololink_module.operators.ComputeCrcOp");
#endif

holoscan::expected<void, holoscan::Error> register_hololink_module_operator_catalog()
{
    static const holoscan::expected<void, holoscan::Error> manual_rows_status = [] {
        auto& registry = holoscan::authoring::authoring_registry();
        auto status = registry.register_operator(linux_receiver_row());
#ifdef HOLOLINK_BUILD_ROCE
        if (!status) {
            return status;
        }
        status = registry.register_operator(roce_receiver_row());
#endif
#ifdef HOLOLINK_BUILD_FUSA
        if (!status) {
            return status;
        }
        status = registry.register_operator(fusa_coe_capture_row());
#endif
        if (!status) {
            return status;
        }
        status = registry.register_operator(hsb_controller_row());
        return status;
    }();
    if (!manual_rows_status) {
        return manual_rows_status;
    }
    return holoscan::authoring::authoring_registry().registration_status();
}

} // namespace hololink::module::operators
