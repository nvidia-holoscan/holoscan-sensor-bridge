// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <hololink/operators/operator_catalog.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <functional>
#include <memory>
#include <utility>

#include <cuda.h>
#include <fmt/format.h>

#include <hololink/core/csi_formats.hpp>
#include <hololink/core/data_channel.hpp>
#include <hololink/operators/capture_metadata_utils.hpp>
#include <hololink/operators/compute_crc/compute_crc.hpp>
#include <hololink/operators/csi_to_bayer/csi_to_bayer.hpp>
#include <hololink/operators/discard_metadata/discard_metadata.hpp>
#include <hololink/operators/image_processor/image_processor.hpp>
#include <hololink/operators/linux_coe_receiver/linux_coe_receiver_op.hpp>
#include <hololink/operators/linux_receiver/linux_receiver_op.hpp>
#include <hololink/operators/packed_format_converter/packed_format_converter.hpp>
#include <hololink/operators/sub_frame_combiner/sub_frame_combiner.hpp>
#include <hololink/operators/udp_transmitter/udp_transmitter_op.hpp>
#include <holoscan/authoring/register_operator.hpp>
#include <holoscan/authoring/register_payload.hpp>

// Ports typed Output<hololink::CaptureMetadataT> need their payload identity registered before
// Python's operator-discovery can resolve them (otherwise AuthoringRegistry::payload_type(identity)
// fails with AUTHORING_REGISTRY_UNKNOWN and the operator is silently skipped from discovery,
// since its ports can't be described). Unlike when this comment was first written: do NOT
// register it here anymore. hololink::CaptureMetadataT's identity ("hololink.CaptureMetadataT")
// is now auto-registered by the generated Python codec for capture_metadata.fbs (see
// src/hololink/operators/CMakeLists.txt's capture_metadata_schema_generated target's PYTHON
// flag) the moment a caller does `import capture_metadata` -- both sides derive the same
// identity from the same embedded BFBS reflection, so this is the SAME situation
// schema::ImageT's own identity already was (registered elsewhere, already linked
// transitively): a manual HOLOSCAN_REGISTER_PAYLOAD call here is now a hard
// AUTHORING_REGISTRY_DUPLICATE failure, not a harmless no-op (confirmed by hitting it, not
// assumed) -- don't add one back.

#ifdef HOLOLINK_BUILD_ROCE
#include <hololink/operators/roce_receiver/roce_receiver_op.hpp>
#include <hololink/operators/roce_transmitter/roce_transmitter_op.hpp>
#endif

#ifdef HOLOLINK_BUILD_GPUNETIO_ROCE
#include <hololink/operators/gpu_roce_transceiver/gpu_roce_transceiver_op.hpp>
#endif

#ifdef HOLOLINK_BUILD_FUSA
#include <hololink/operators/fusa_coe_capture/fusa_coe_capture.hpp>
#endif

#ifdef HOLOLINK_BUILD_ARGUS_ISP
#include <hololink/operators/argus_isp/argus_isp.hpp>
#endif

#ifdef HOLOLINK_HAS_IMGUI
#include <chrono>

#include <hololink/common/gui_renderer.hpp>
#include <hololink/operators/iq_dec/iq_dec_op.hpp>
#include <hololink/operators/iq_enc/iq_enc_op.hpp>
#include <hololink/operators/sig_gen/sig_gen_op.hpp>
#include <hololink/operators/sig_viewer/sig_viewer_op.hpp>
#endif

namespace hololink::operators {
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
    // Shared assignment-parsing helpers for the operators below, whose
    // constructors take live positional arguments a schema can't represent
    // (an optional Allocator handle) alongside plain scalars/enums a schema
    // can -- so they register via ManualOperatorRegistrationThunks instead of
    // HOLOSCAN_REGISTER_OPERATOR. Mirrors the local helper pattern every
    // SDK module with manual rows repeats (see e.g.
    // holoscan-sdk/public/modules/inference/operators/src/operator_catalog.cpp);
    // there is no shared, non-duplicated version of this in the SDK yet.
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

    // Mirrors integer_value's multi-source-try shape, but tries the two floating-point carrier
    // widths (ParamScalarValue stores exactly kFloatingPoint/32 or kFloatingPoint/64, no implicit
    // widening between them -- see ScalarValue::try_as in holoscan/core/parameter.hpp) instead of
    // integer_value's four integer widths. Python's float is always a 64-bit double, so values
    // built from Python land here as the `double` branch; the `float` branch exists for parity
    // with C++ callers that already hold a 32-bit value.
    template <typename Real>
    [[nodiscard]] Real real_value(const ParameterValue& value, std::string_view path)
    {
        const auto convert = [&]<typename Source>() -> std::optional<Real> {
            Source source {};
            if (!value.scalar.try_as(source)) {
                return std::nullopt;
            }
            return static_cast<Real>(source);
        };
        if (value.kind != ParameterValueKind::kScalar) {
            parameter_error("HOLOLINK_OPERATOR_PARAMETER_TYPE_MISMATCH", path);
        }
        if (const auto result = convert.template operator()<double>())
            return *result;
        if (const auto result = convert.template operator()<float>())
            return *result;
        parameter_error("HOLOLINK_OPERATOR_PARAMETER_TYPE_MISMATCH", path);
    }

    template <typename Real>
    [[nodiscard]] Real real_or(const AssignmentView& values, std::string_view name, Real default_value)
    {
        const ParameterValue* value = values.find(name);
        return value == nullptr ? default_value : real_value<Real>(*value, values.field_path(name));
    }

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

    // Used by LinuxReceiverOp's `receiver_affinity` -- the only sequence-of-integer field among the
    // manual rows so far (FusaCoeCaptureOp's `mac_addr` parses its own sequence inline instead,
    // since it's the only caller of a sequence-of-uint8_t and pulling that one-line loop into a
    // shared helper isn't worth it; this one has a second caller, so it is).
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

    constexpr std::array<EnumChoice<hololink::csi::BayerFormat>, 4> kBayerFormats {
        EnumChoice<hololink::csi::BayerFormat> { "bggr", hololink::csi::BayerFormat::BGGR },
        EnumChoice<hololink::csi::BayerFormat> { "rggb", hololink::csi::BayerFormat::RGGB },
        EnumChoice<hololink::csi::BayerFormat> { "gbrg", hololink::csi::BayerFormat::GBRG },
        EnumChoice<hololink::csi::BayerFormat> { "grbg", hololink::csi::BayerFormat::GRBG },
    };

#ifdef HOLOLINK_BUILD_FUSA

    constexpr std::array<EnumChoice<hololink::csi::PixelFormat>, 3> kPixelFormats {
        EnumChoice<hololink::csi::PixelFormat> { "raw_8", hololink::csi::PixelFormat::RAW_8 },
        EnumChoice<hololink::csi::PixelFormat> { "raw_10", hololink::csi::PixelFormat::RAW_10 },
        EnumChoice<hololink::csi::PixelFormat> { "raw_12", hololink::csi::PixelFormat::RAW_12 },
    };

    // Parses a nested `geometry` assignment group (a dict from Python, or a directly-nested
    // ParameterAssignment span from C++) into a CsiGeometry. Field names match CsiGeometry's own,
    // which also matches CsiToBayerOp's schema-declared `geometry` field -- see
    // python/hololink/_hololink.cpp's resolve_geometry(), which builds exactly this shape.
    [[nodiscard]] hololink::csi::CsiGeometry parse_geometry(const AssignmentView& values)
    {
        const ParameterValue& field = values.require("geometry");
        if (field.kind != ParameterValueKind::kAssignments) {
            parameter_error("HOLOLINK_OPERATOR_PARAMETER_TYPE_MISMATCH", values.field_path("geometry"));
        }
        AssignmentView geometry_values { field.nested_assignments(), values.field_path("geometry") };
        hololink::csi::CsiGeometry geometry;
        geometry.start_byte = integer_value<uint32_t>(
            geometry_values.require("start_byte"), geometry_values.field_path("start_byte"));
        geometry.bytes_per_line = integer_value<uint32_t>(
            geometry_values.require("bytes_per_line"), geometry_values.field_path("bytes_per_line"));
        geometry.pixel_width = integer_value<uint32_t>(
            geometry_values.require("pixel_width"), geometry_values.field_path("pixel_width"));
        geometry.pixel_height = integer_value<uint32_t>(
            geometry_values.require("pixel_height"), geometry_values.field_path("pixel_height"));
        geometry.pixel_format = enum_value(
            geometry_values.require("pixel_format"), geometry_values.field_path("pixel_format"), kPixelFormats);
        geometry.trailing_bytes = integer_value<uint32_t>(
            geometry_values.require("trailing_bytes"), geometry_values.field_path("trailing_bytes"));
        geometry_values.reject_unknown();
        return geometry;
    }

#endif // HOLOLINK_BUILD_FUSA

    // ---------------------------------------------------------------------------
    // Constructor handles of the receiver and FusaCoeCaptureOp rows. Their live objects can't be
    // Parameters, so each row declares them as constructor-handle inputs (CLARAHOLOS-3438): a
    // required `data_channel` and two optional `device_start`/`device_stop` callbacks, which
    // Python fills from any callable (typically the camera's start and stop methods). The registry
    // checks them before calling the row, and the Graph declaration keeps the channel alive for as
    // long as the operator can run. The CUDA context the receiver thread makes current is an
    // ordinary `frame_context` value instead: it has no shared owner to pass as a handle.
    // ---------------------------------------------------------------------------

    constexpr holoscan::authoring::ConstructorHandleDescriptor kDataChannelHandleDescriptor {
        .name = "hololink.DataChannel",
        .description = "Sensor Bridge data channel from board enumeration (hololink.DataChannel).",
    };

    constexpr holoscan::authoring::ConstructorHandleDescriptor kDataChannelsHandleDescriptor {
        .name = "hololink.DataChannels",
        .description = "Sensor Bridge data channels of several sensors of one board "
                       "(hololink.DataChannels).",
    };

    [[nodiscard]] std::span<const ConstructorHandleInput> receiver_handle_inputs()
    {
        // Built at run time: the callback type's descriptor comes from a Holoscan function.
        static const std::array inputs {
            ConstructorHandleInput {
                .name = "data_channel",
                .description = "The Sensor Bridge data channel to receive from.",
                .type = &kDataChannelHandleDescriptor,
            },
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

    // The constructor inputs of a row whose thunks take receiver_handle_inputs().
    template <typename Thunks>
    [[nodiscard]] const RegisteredConstructorInputs* receiver_constructor_inputs()
    {
        static const RegisteredConstructorInputs inputs {
            .handles = receiver_handle_inputs(),
            .add_to_graph_from_inputs = &Thunks::add_from_inputs,
            .add_for_description = &Thunks::add_for_description,
        };
        return &inputs;
    }

    struct ReceiverHandles {
        std::shared_ptr<hololink::DataChannel> channel;
        std::function<void()> device_start;
        std::function<void()> device_stop;
    };

    // Reads receiver_handle_inputs(); omitted callbacks come back empty.
    [[nodiscard]] expected<ReceiverHandles, Error> receiver_handles(
        const OperatorConstructionInputs& inputs)
    {
        auto channel = inputs.require_shared("data_channel", data_channel_handle_type());
        if (!channel) {
            return holoscan::make_unexpected(std::move(channel).error());
        }
        auto device_start = inputs.callback("device_start");
        if (!device_start) {
            return holoscan::make_unexpected(std::move(device_start).error());
        }
        auto device_stop = inputs.callback("device_stop");
        if (!device_stop) {
            return holoscan::make_unexpected(std::move(device_stop).error());
        }
        return ReceiverHandles { std::move(*channel), std::move(*device_start), std::move(*device_stop) };
    }

    // The CUDA context the receiver thread makes current (CUDA context state is thread-local): the
    // numeric value of a CUcontext the application keeps retained, e.g. int(ctx) for the
    // cuda.bindings.driver.CUcontext from cuDevicePrimaryCtxRetain().
    [[nodiscard]] CUcontext frame_context_value(const AssignmentView& values)
    {
        return reinterpret_cast<CUcontext>(integer_value<std::uintptr_t>(
            values.require("frame_context"), values.field_path("frame_context")));
    }

    // Placeholder channel for registry port-discovery only (see add_for_description() below in the
    // receiver rows). Unlike every other manual row in this file,
    // BaseReceiverOp's constructor throws std::invalid_argument for a null channel, so an
    // all-default/nullptr discovery construction (FusaCoeCaptureOp's approach) or a scalar
    // placeholder (SubFrameCombinerOp's approach) both fall short here --
    // this needs an actual, non-null DataChannel that will not throw either. DataChannel::initialize()
    // unconditionally dereferences several Metadata fields via std::optional::value(), so this fills
    // in just enough fake values to satisfy that, matching holoscan.registry.operators()'s own
    // discovery probe's needs: it only resolves port/name metadata from the constructed instance,
    // never calls start()/stop() or touches the (here, null) Hololink device handle at all.
    [[nodiscard]] std::shared_ptr<hololink::DataChannel> make_discovery_data_channel()
    {
        hololink::Metadata metadata;
        metadata["peer_ip"] = std::string { "0.0.0.0" };
        metadata["vp_mask"] = int64_t { 0 };
        metadata["data_plane"] = int64_t { 0 };
        metadata["sensor"] = int64_t { 0 };
        metadata["sif_address"] = int64_t { 0 };
        metadata["vp_address"] = int64_t { 0 };
        metadata["hif_address"] = int64_t { 0 };
        metadata["fpga_uuid"] = std::string { "00000000-0000-0000-0000-000000000000" };
        // DataChannel's shared_ptr<Hololink>-taking constructor is protected (reserved for
        // programming tools that bypass version checking); this uses the public
        // factory-function-taking constructor instead, with a factory that never actually runs
        // real enumeration/connection logic -- just returns null, which initialize() stores
        // without dereferencing (confirmed by reading it).
        return std::make_shared<hololink::DataChannel>(
            metadata, [](const hololink::Metadata&) { return std::shared_ptr<hololink::Hololink> {}; });
    }

    // ---------------------------------------------------------------------------
    // LinuxReceiverOp -- manual row. Always built (no HOLOLINK_BUILD_ROCE/HOLOLINK_BUILD_FUSA
    // gate), unlike every other manual row in this file. BaseReceiverOp's constructor throws if
    // given a null channel (`throw std::invalid_argument("A Hololink data channel is required")`),
    // unlike FusaCoeCaptureOp/SubFrameCombinerOp above, neither of which
    // validate their own discovery-mode defaults this strictly -- discovered the hard way: an
    // early version of this row threw a dedicated error from its own discovery path instead of
    // faking a placeholder channel, which seemed reasonable (a clearer error than a generic
    // construction failure) until holoscan.registry.operators()'s own discovery probe turned out
    // to construct every registered row to resolve its ports, and silently *drops* any row whose
    // probe throws (logged only as a "[warning] Skipping registered Operator ... during discovery"
    // message) -- so that early version made LinuxReceiverOp invisible from hololink.operators
    // entirely. Fixed by using make_discovery_data_channel() above instead.
    // ---------------------------------------------------------------------------

    constexpr std::array<std::string_view, 2> kLinuxReceiverRequiredParameters {
        "frame_size", "frame_context"
    };

    struct LinuxReceiverRegistrationThunks final
        : holoscan::authoring::ManualOperatorRegistrationThunks<LinuxReceiverOp> {
        static OpaqueOperatorRef add(Graph& graph, std::string_view name,
            std::shared_ptr<hololink::DataChannel> channel, std::function<void()> device_start,
            std::function<void()> device_stop, LinuxReceiverParams params)
        {
            auto op = graph.op<LinuxReceiverOp>(
                name, std::move(channel), std::move(device_start), std::move(device_stop));
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
                params.trim = bool_or(values, "trim", true);
                params.receiver_affinity = integer_sequence_or(values, "receiver_affinity");
                params.pages = integer_or<uint32_t>(values, "pages", 1);
                params.queue_size = integer_or<uint32_t>(values, "queue_size", 1);
                params.frame_limit = integer_or<int64_t>(values, "frame_limit", 0);
                params.frame_limit_ends_run = bool_or(values, "frame_limit_ends_run", false);
                params.watch_ingress = bool_or(values, "watch_ingress", false);
                values.reject_unknown();
                return add(graph, name, std::move(handles->channel), std::move(handles->device_start),
                    std::move(handles->device_stop), std::move(params));
            });
        }

        // Discovery: see make_discovery_data_channel()'s doc comment above for why this (not
        // nullptr, not a thrown error) is what BaseReceiverOp's constructor needs.
        static expected<OpaqueOperatorRef, Error> add_for_description(
            Graph& graph, std::string_view name)
        {
            return invoke_add([&] {
                return add(graph, name, make_discovery_data_channel(), {}, {},
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
            .name = "hololink.operators.LinuxReceiverOp",
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
            .constructor_inputs = receiver_constructor_inputs<Thunks>(),
        };
    }

    // ---------------------------------------------------------------------------
    // LinuxCoeReceiverOp -- manual row, same shape and reasoning as LinuxReceiverOp above (the same
    // receiver_handle_inputs() are exactly what BaseReceiverOp's constructor needs, regardless of
    // transport). Always built (no HOLOLINK_BUILD_ROCE/HOLOLINK_BUILD_FUSA gate), matching
    // linux_coe_receiver's own CMake (no special dependency, unlike roce_receiver's ibverbs
    // requirement). coe_interface/pixel_width/coe_channel/stall_timeout_ms are this row's only
    // genuinely new parameters -- everything else is LinuxReceiverOp's own already-proven field set
    // (frame_context/frame_size/trim/receiver_affinity).
    // ---------------------------------------------------------------------------

    constexpr std::array<std::string_view, 3> kLinuxCoeReceiverRequiredParameters {
        "frame_size", "frame_context", "coe_interface"
    };

    struct LinuxCoeReceiverRegistrationThunks final
        : holoscan::authoring::ManualOperatorRegistrationThunks<LinuxCoeReceiverOp> {
        static OpaqueOperatorRef add(Graph& graph, std::string_view name,
            std::shared_ptr<hololink::DataChannel> channel, std::function<void()> device_start,
            std::function<void()> device_stop, LinuxCoeReceiverParams params)
        {
            auto op = graph.op<LinuxCoeReceiverOp>(
                name, std::move(channel), std::move(device_start), std::move(device_stop));
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
                LinuxCoeReceiverParams params;
                params.frame_context = frame_context_value(values);
                params.frame_size = integer_value<size_t>(
                    values.require("frame_size"), values.field_path("frame_size"));
                params.trim = bool_or(values, "trim", true);
                params.receiver_affinity = integer_sequence_or(values, "receiver_affinity");
                params.coe_interface = string_value(
                    values.require("coe_interface"), values.field_path("coe_interface"));
                params.pixel_width = integer_or<uint32_t>(values, "pixel_width", 0);
                params.coe_channel = integer_or<uint8_t>(values, "coe_channel", 0);
                params.frame_limit = integer_or<int64_t>(values, "frame_limit", 0);
                params.frame_limit_ends_run = bool_or(values, "frame_limit_ends_run", false);
                params.stall_timeout_ms = integer_or<uint32_t>(values, "stall_timeout_ms", 0);
                values.reject_unknown();
                return add(graph, name, std::move(handles->channel), std::move(handles->device_start),
                    std::move(handles->device_stop), std::move(params));
            });
        }

        // Discovery: see make_discovery_data_channel()'s doc comment above (LinuxReceiverOp's row)
        // for why this, not nullptr or a thrown error, is what BaseReceiverOp's constructor needs.
        static expected<OpaqueOperatorRef, Error> add_for_description(
            Graph& graph, std::string_view name)
        {
            return invoke_add([&] {
                return add(graph, name, make_discovery_data_channel(), {}, {},
                    LinuxCoeReceiverParams { .frame_size = 1 });
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

    [[nodiscard]] holoscan::authoring::RegisteredOperatorType linux_coe_receiver_row()
    {
        using Thunks = LinuxCoeReceiverRegistrationThunks;
        return holoscan::authoring::RegisteredOperatorType {
            .name = "hololink.operators.LinuxCoeReceiverOp",
            .param_type = &holoscan::param_type_descriptor<holoscan::NoParams>(),
            .param_schema = nullptr,
            .parameter_introspection = holoscan::authoring::OperatorParameterIntrospection::kPassThrough,
            .required_parameters = kLinuxCoeReceiverRequiredParameters,
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
            .constructor_inputs = receiver_constructor_inputs<Thunks>(),
        };
    }

#ifdef HOLOLINK_BUILD_ROCE
    // ---------------------------------------------------------------------------
    // RoceReceiverOp -- manual row, same shape and reasoning as LinuxReceiverOp above (the same
    // receiver_handle_inputs()). Guarded separately because roce_receiver (unlike linux_receiver) is
    // itself only built under HOLOLINK_BUILD_ROCE -- see src/hololink/operators/CMakeLists.txt.
    // ---------------------------------------------------------------------------

    constexpr std::array<std::string_view, 2> kRoceReceiverRequiredParameters {
        "frame_size", "frame_context"
    };

    struct RoceReceiverRegistrationThunks final
        : holoscan::authoring::ManualOperatorRegistrationThunks<RoceReceiverOp> {
        static OpaqueOperatorRef add(Graph& graph, std::string_view name,
            std::shared_ptr<hololink::DataChannel> channel, std::function<void()> device_start,
            std::function<void()> device_stop, RoceReceiverParams params)
        {
            auto op = graph.op<RoceReceiverOp>(
                name, std::move(channel), std::move(device_start), std::move(device_stop));
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
                params.trim = bool_or(values, "trim", true);
                params.ibv_name = string_or(values, "ibv_name", "");
                params.ibv_port = integer_or<uint32_t>(values, "ibv_port", 1);
                params.pages = integer_or<uint32_t>(values, "pages", 2);
                params.queue_size = integer_or<uint32_t>(values, "queue_size", 1);
                params.metadata_offset = integer_or<size_t>(values, "metadata_offset", 0);
                params.frame_limit = integer_or<int64_t>(values, "frame_limit", 0);
                params.frame_limit_ends_run = bool_or(values, "frame_limit_ends_run", false);
                params.watch_ingress = bool_or(values, "watch_ingress", false);
                values.reject_unknown();
                return add(graph, name, std::move(handles->channel), std::move(handles->device_start),
                    std::move(handles->device_stop), std::move(params));
            });
        }

        // Discovery: see make_discovery_data_channel()'s doc comment (above LinuxReceiverOp's row)
        // for why this, not a thrown error, is required here.
        static expected<OpaqueOperatorRef, Error> add_for_description(
            Graph& graph, std::string_view name)
        {
            return invoke_add([&] {
                return add(graph, name, make_discovery_data_channel(), {}, {},
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
            .name = "hololink.operators.RoceReceiverOp",
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
            .constructor_inputs = receiver_constructor_inputs<Thunks>(),
        };
    }
#endif // HOLOLINK_BUILD_ROCE

#ifdef HOLOLINK_BUILD_GPUNETIO_ROCE
    // ---------------------------------------------------------------------------
    // GpuRoceTransceiverOp -- manual row, same shape as RoceReceiverOp above (the same
    // receiver_handle_inputs() -- BaseReceiverOp gives both operators an identical
    // shared_ptr<DataChannel> + 2-callback constructor). Guarded separately under
    // HOLOLINK_BUILD_GPUNETIO_ROCE, a narrower flag than HOLOLINK_BUILD_ROCE (see
    // src/hololink/operators/CMakeLists.txt): this operator's translation unit links DOCA
    // GPUNetIO/Verbs directly, unlike RoceReceiverOp's plain ibverbs path, so it is only built
    // where that SDK is actually present. `trim` is intentionally not exposed as a settable
    // parameter here: GpuRoceTransceiverParams documents it as unused (the GPU-resident kernel
    // owns the data path end to end), so it stays at the struct's own `false` default -- same
    // "don't add a parsing helper for a knob with no effect" call as HsbControllerOp's
    // watchdog_timeout_s.
    //
    // Several sensors of one board, one QP each: tx_ibv_qp lists the remote QPs ("2,4,6,8"), and
    // the optional extra_data_channels handle (a hololink.DataChannels) the DataChannels of the
    // sensors after data_channel's, in the same order. device_start/device_stop start and stop
    // every sensor's device.
    // ---------------------------------------------------------------------------

    constexpr std::array<std::string_view, 2> kGpuRoceTransceiverRequiredParameters {
        "frame_size", "frame_context"
    };

    // receiver_handle_inputs(), and the DataChannels of the sensors after data_channel's.
    [[nodiscard]] std::span<const ConstructorHandleInput> gpu_roce_transceiver_handle_inputs()
    {
        const auto receiver = receiver_handle_inputs();
        static const std::array inputs {
            receiver[0],
            receiver[1],
            receiver[2],
            ConstructorHandleInput {
                .name = "extra_data_channels",
                .description = "The data channels of the sensors after data_channel's, one per QP of "
                               "tx_ibv_qp after the first, in the same order.",
                .type = &kDataChannelsHandleDescriptor,
                .required = false,
            },
        };
        return inputs;
    }

    template <typename Thunks>
    [[nodiscard]] const RegisteredConstructorInputs* gpu_roce_transceiver_constructor_inputs()
    {
        static const RegisteredConstructorInputs inputs {
            .handles = gpu_roce_transceiver_handle_inputs(),
            .add_to_graph_from_inputs = &Thunks::add_from_inputs,
            .add_for_description = &Thunks::add_for_description,
        };
        return &inputs;
    }

    struct GpuRoceTransceiverRegistrationThunks final
        : holoscan::authoring::ManualOperatorRegistrationThunks<GpuRoceTransceiverOp> {
        static OpaqueOperatorRef add(Graph& graph, std::string_view name,
            std::shared_ptr<hololink::DataChannel> channel, std::function<void()> device_start,
            std::function<void()> device_stop, GpuRoceTransceiverParams params)
        {
            auto op = graph.op<GpuRoceTransceiverOp>(
                name, std::move(channel), std::move(device_start), std::move(device_stop));
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
            auto extra_channels = inputs.optional_shared("extra_data_channels", data_channels_handle_type());
            if (!extra_channels) {
                return holoscan::make_unexpected(std::move(extra_channels).error());
            }
            return invoke_add([&] {
                AssignmentView values { inputs.values, {} };
                GpuRoceTransceiverParams params;
                params.frame_context = frame_context_value(values);
                params.frame_size = integer_value<size_t>(
                    values.require("frame_size"), values.field_path("frame_size"));
                params.ibv_name = string_or(values, "ibv_name", "");
                params.ibv_port = integer_or<uint32_t>(values, "ibv_port", 1);
                params.tx_ibv_qp = string_or(values, "tx_ibv_qp", "2");
                params.gpu_id = integer_or<uint32_t>(values, "gpu_id", 0);
                params.forward = integer_or<uint32_t>(values, "forward", 0);
                params.frame_limit = integer_or<int64_t>(values, "frame_limit", 0);
                values.reject_unknown();
                if (*extra_channels && !(*extra_channels)->channels.empty()) {
                    // The first channel is also the operator's own, which opens the data socket.
                    params.hololink_channels.push_back(handles->channel);
                    for (const auto& channel : (*extra_channels)->channels) {
                        params.hololink_channels.push_back(channel);
                    }
                }
                return add(graph, name, std::move(handles->channel), std::move(handles->device_start),
                    std::move(handles->device_stop), std::move(params));
            });
        }

        // Discovery: see make_discovery_data_channel()'s doc comment (above LinuxReceiverOp's row)
        // for why this, not a thrown error, is required here.
        static expected<OpaqueOperatorRef, Error> add_for_description(
            Graph& graph, std::string_view name)
        {
            return invoke_add([&] {
                return add(graph, name, make_discovery_data_channel(), {}, {},
                    GpuRoceTransceiverParams { .frame_size = 1 });
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

    [[nodiscard]] holoscan::authoring::RegisteredOperatorType gpu_roce_transceiver_row()
    {
        using Thunks = GpuRoceTransceiverRegistrationThunks;
        return holoscan::authoring::RegisteredOperatorType {
            .name = "hololink.operators.GpuRoceTransceiverOp",
            .param_type = &holoscan::param_type_descriptor<holoscan::NoParams>(),
            .param_schema = nullptr,
            .parameter_introspection = holoscan::authoring::OperatorParameterIntrospection::kPassThrough,
            .required_parameters = kGpuRoceTransceiverRequiredParameters,
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
            .constructor_inputs = gpu_roce_transceiver_constructor_inputs<Thunks>(),
        };
    }
#endif // HOLOLINK_BUILD_GPUNETIO_ROCE

    // ---------------------------------------------------------------------------
    // SubFrameCombinerOp -- manual row rather than a schema row, so that height can
    // be a required registry parameter: it has no sensible default (it's the
    // accumulated full-frame height the caller's pipeline expects), and a schema row
    // would accept a missing height and only fail once the operator starts.
    // cuda_device_ordinal and drop_incomplete are optional, defaulting like SubFrameCombinerParams.
    // ---------------------------------------------------------------------------

    constexpr std::array<std::string_view, 1> kSubFrameCombinerRequiredParameters { "height" };

    struct SubFrameCombinerRegistrationThunks final
        : holoscan::authoring::ManualOperatorRegistrationThunks<SubFrameCombinerOp> {
        static OpaqueOperatorRef add(Graph& graph, std::string_view name, int cuda_device_ordinal,
            uint32_t height, bool drop_incomplete)
        {
            auto op = graph.op<SubFrameCombinerOp>(name);
            op.set_params(SubFrameCombinerParams {
                .cuda_device_ordinal = cuda_device_ordinal,
                .height = height,
                .drop_incomplete = drop_incomplete,
            });
            return make_operator_ref(op);
        }

        static OpaqueOperatorRef add_to_graph(Graph& graph, std::string_view name,
            const holoscan::authoring::MaterializedParameters* params)
        {
            if (params != nullptr) {
                parameter_error("AUTHORING_PARAMETER_TYPE_MISMATCH");
            }
            // No sensible discovery-mode default for height; a nonzero placeholder keeps
            // construction succeeding for registry introspection (naming/ports), matching how
            // FusaCoeCaptureParams-shaped rows elsewhere in this catalog handle required fields.
            return add(graph, name, 0, 1U, false);
        }

        static void set_params(const OpaqueOperatorRef&,
            const holoscan::authoring::MaterializedParameters&)
        {
            throw RuntimeError {
                Error { ErrorCode::kNotSupported, "AUTHORING_MANUAL_ROW_PARAMETERS_IMMUTABLE" }
            };
        }
    };

    expected<OpaqueOperatorRef, Error> sub_frame_combiner_add_from_assignments(
        Graph& graph, std::string_view name, std::span<const ParameterAssignment> assignments)
    {
        return SubFrameCombinerRegistrationThunks::invoke_add([&] {
            if (assignments.empty()) {
                return SubFrameCombinerRegistrationThunks::add_to_graph(graph, name, nullptr);
            }
            AssignmentView values { assignments, {} };
            const int cuda_device_ordinal = integer_or<int>(values, "cuda_device_ordinal", 0);
            const uint32_t height = integer_value<uint32_t>(
                values.require("height"), values.field_path("height"));
            const bool drop_incomplete = bool_or(values, "drop_incomplete", false);
            values.reject_unknown();
            return SubFrameCombinerRegistrationThunks::add(
                graph, name, cuda_device_ordinal, height, drop_incomplete);
        });
    }

    [[nodiscard]] holoscan::authoring::RegisteredOperatorType sub_frame_combiner_row()
    {
        using Thunks = SubFrameCombinerRegistrationThunks;
        return holoscan::authoring::RegisteredOperatorType {
            .name = "hololink.operators.SubFrameCombinerOp",
            .param_type = &holoscan::param_type_descriptor<holoscan::NoParams>(),
            .param_schema = nullptr,
            .parameter_introspection = holoscan::authoring::OperatorParameterIntrospection::kPassThrough,
            .required_parameters = kSubFrameCombinerRequiredParameters,
            .add_to_graph = &Thunks::add_to_graph,
            .add_to_graph_from_assignments = &sub_frame_combiner_add_from_assignments,
            .set_params = &Thunks::set_params,
            .port = &Thunks::port,
            .ports = &Thunks::ports,
            .region_add = &Thunks::region_add,
            .partition_add = &Thunks::partition_add,
            .lifecycle_before = &Thunks::lifecycle_before,
            .lifecycle_allow_self_restart = &Thunks::lifecycle_allow_self_restart,
            .lifecycle_allow_external_restart = &Thunks::lifecycle_allow_external_restart,
            .lifecycle_allow_restart_request = &Thunks::lifecycle_allow_restart_request,
        };
    }

#ifdef HOLOLINK_BUILD_FUSA

    // ---------------------------------------------------------------------------
    // FusaCoeCaptureOp -- manual row. FusaCoeCaptureParams embeds a raw DataChannel*
    // and two std::function<void()> callbacks directly in the Params struct (not just the
    // constructor), so even discovery-mode construction (all-default Params) is otherwise fine --
    // only a *real* capture needs the channel, which arrives as the receiver rows'
    // receiver_handle_inputs(). The raw pointer stays valid because the Graph declaration, and every
    // plan compiled from it, keeps the `data_channel` handle's owner. Also requires calling
    // make_geometry_adapter()/configure_converter() before construction to resolve geometry, which
    // happens entirely on the caller's side (Python: resolve_geometry() in _hololink.cpp; C++:
    // examples/fusa_coe_vb1940_player.cpp) -- geometry arrives here as an already-resolved nested
    // `geometry` assignment group, same shape as CsiToBayerOp's schema-declared field.
    // ---------------------------------------------------------------------------

    constexpr std::array<std::string_view, 0> kFusaCoeCaptureRequiredParameters {};

    struct FusaCoeCaptureRegistrationThunks final
        : holoscan::authoring::ManualOperatorRegistrationThunks<FusaCoeCaptureOp> {
        static OpaqueOperatorRef add(Graph& graph, std::string_view name, FusaCoeCaptureParams params)
        {
            auto op = graph.op<FusaCoeCaptureOp>(name, params.cpu_output);
            op.set_params(std::move(params));
            return make_operator_ref(op);
        }

        static expected<OpaqueOperatorRef, Error> add_from_inputs(
            Graph& graph, std::string_view name, const OperatorConstructionInputs& inputs);

        // Discovery: no channel, no real capture -- matches FusaCoeCaptureParams's own all-default
        // resting state (hololink_channel == nullptr).
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
            params.hololink_channel = handles->channel.get();
            params.device_start = std::move(handles->device_start);
            params.device_stop = std::move(handles->device_stop);
            params.interface = string_or(values, "interface", "");
            params.timeout = integer_or<uint32_t>(values, "timeout", 0);
            params.stall_timeout_ms = integer_or<uint32_t>(values, "stall_timeout_ms", 0);
            params.cpu_output = bool_or(values, "cpu_output", false);
            params.frame_limit = integer_or<int64_t>(values, "frame_limit", 0);
            params.frame_limit_ends_run = bool_or(values, "frame_limit_ends_run", false);
            // Names the capture in its stall messages, as the C++ apps do.
            params.name = std::string(name);
            if (const ParameterValue* field = values.find("frame_size_bytes")) {
                params.frame_size_bytes = integer_value<uint32_t>(
                    *field, values.field_path("frame_size_bytes"));
            }
            params.pixel_format = enum_or(
                values, "pixel_format", hololink::csi::PixelFormat::RAW_8, kPixelFormats);
            params.bayer_format = enum_or(
                values, "bayer_format", hololink::csi::BayerFormat::RGGB, kBayerFormats);
            params.include_csi_frame = bool_or(values, "include_csi_frame", false);
            params.packetizer_enabled = false;
            if (const ParameterValue* field = values.find("packetizer_enabled")) {
                bool flag = false;
                if (field->kind != ParameterValueKind::kScalar || !field->scalar.try_as(flag)) {
                    parameter_error(
                        "HOLOLINK_OPERATOR_PARAMETER_TYPE_MISMATCH", values.field_path("packetizer_enabled"));
                }
                params.packetizer_enabled = flag;
            }
            // geometry is meaningless (and start() never reads it) in opaque byte-blob mode -- see
            // FusaCoeCaptureParams::frame_size_bytes's own doc comment -- so only require it (via
            // parse_geometry()'s own unconditional values.require("geometry")) for a real CSI
            // capture.
            if (!params.frame_size_bytes) {
                params.geometry = parse_geometry(values);
            }

            if (const ParameterValue* field = values.find("mac_addr")) {
                if (field->kind != ParameterValueKind::kSequence) {
                    parameter_error("HOLOLINK_OPERATOR_PARAMETER_TYPE_MISMATCH", values.field_path("mac_addr"));
                }
                for (const ParameterValue& element : field->sequence_values()) {
                    params.mac_addr.push_back(
                        integer_value<uint8_t>(element, values.field_path("mac_addr")));
                }
            }

            values.reject_unknown();
            return FusaCoeCaptureRegistrationThunks::add(graph, name, std::move(params));
        });
    }

    [[nodiscard]] holoscan::authoring::RegisteredOperatorType fusa_coe_capture_row()
    {
        using Thunks = FusaCoeCaptureRegistrationThunks;
        return holoscan::authoring::RegisteredOperatorType {
            .name = "hololink.operators.FusaCoeCaptureOp",
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
            .constructor_inputs = receiver_constructor_inputs<Thunks>(),
        };
    }

#endif // HOLOLINK_BUILD_FUSA

#ifdef HOLOLINK_BUILD_ARGUS_ISP

    // ---------------------------------------------------------------------------
    // ArgusIspOp -- manual row: `pool` has no schema-representable handle type in the current
    // registry (ParameterValue carries no opaque object reference), so every
    // caller gets nullptr, matching the operator's own documented default (falls back to an
    // UnboundedAllocator in its constructor). bayer_format is accepted through the same
    // hololink::csi::BayerFormat enum/kBayerFormats table already used by
    // FusaCoeCaptureOp above -- ArgusIspOp's own nested BayerFormat enum
    // (RGGB=1, GBRG=2) agrees with hololink::csi::BayerFormat on both values (both deliberately
    // match NPP's NppiBayerGridPosition), and the constructor itself takes a plain int either way.
    // exposure_time_ms/analog_gain are the first float-typed parameters in this file; see
    // real_value/real_or above.
    // ---------------------------------------------------------------------------

    constexpr std::array<std::string_view, 0> kArgusIspRequiredParameters {};

    struct ArgusIspRegistrationThunks final
        : holoscan::authoring::ManualOperatorRegistrationThunks<ArgusIspOp> {
        static OpaqueOperatorRef add(Graph& graph, std::string_view name, int bayer_format,
            float exposure_time_ms, float analog_gain, int pixel_bit_depth, uint32_t camera_index,
            uint32_t sensor_mode_index)
        {
            return make_operator_ref(graph.op<ArgusIspOp>(name, bayer_format, exposure_time_ms,
                analog_gain, pixel_bit_depth, nullptr, camera_index, sensor_mode_index));
        }

        static OpaqueOperatorRef add_to_graph(Graph& graph, std::string_view name,
            const holoscan::authoring::MaterializedParameters* params)
        {
            if (params != nullptr) {
                parameter_error("AUTHORING_PARAMETER_TYPE_MISMATCH");
            }
            return add(graph, name, static_cast<int>(hololink::csi::BayerFormat::GBRG), 0.0f, 0.0f,
                0, 0, 0);
        }

        static void set_params(const OpaqueOperatorRef&,
            const holoscan::authoring::MaterializedParameters&)
        {
            throw RuntimeError {
                Error { ErrorCode::kNotSupported, "AUTHORING_MANUAL_ROW_PARAMETERS_IMMUTABLE" }
            };
        }
    };

    expected<OpaqueOperatorRef, Error> argus_isp_add_from_assignments(
        Graph& graph, std::string_view name, std::span<const ParameterAssignment> assignments)
    {
        return ArgusIspRegistrationThunks::invoke_add([&] {
            if (assignments.empty()) {
                return ArgusIspRegistrationThunks::add_to_graph(graph, name, nullptr);
            }
            AssignmentView values { assignments, {} };
            const hololink::csi::BayerFormat bayer_format = enum_or(
                values, "bayer_format", hololink::csi::BayerFormat::GBRG, kBayerFormats);
            const float exposure_time_ms = real_or<float>(values, "exposure_time_ms", 0.0f);
            const float analog_gain = real_or<float>(values, "analog_gain", 0.0f);
            const int pixel_bit_depth = integer_or<int>(values, "pixel_bit_depth", 10);
            const uint32_t camera_index = integer_or<uint32_t>(values, "camera_index", 0);
            const uint32_t sensor_mode_index = integer_or<uint32_t>(values, "sensor_mode_index", 0);
            values.reject_unknown();
            return ArgusIspRegistrationThunks::add(graph, name, static_cast<int>(bayer_format),
                exposure_time_ms, analog_gain, pixel_bit_depth, camera_index, sensor_mode_index);
        });
    }

    [[nodiscard]] holoscan::authoring::RegisteredOperatorType argus_isp_row()
    {
        using Thunks = ArgusIspRegistrationThunks;
        return holoscan::authoring::RegisteredOperatorType {
            .name = "hololink.operators.ArgusIspOp",
            .param_type = &holoscan::param_type_descriptor<holoscan::NoParams>(),
            .param_schema = nullptr,
            .parameter_introspection = holoscan::authoring::OperatorParameterIntrospection::kPassThrough,
            .required_parameters = kArgusIspRequiredParameters,
            .add_to_graph = &Thunks::add_to_graph,
            .add_to_graph_from_assignments = &argus_isp_add_from_assignments,
            .set_params = &Thunks::set_params,
            .port = &Thunks::port,
            .ports = &Thunks::ports,
            .region_add = &Thunks::region_add,
            .partition_add = &Thunks::partition_add,
            .lifecycle_before = &Thunks::lifecycle_before,
            .lifecycle_allow_self_restart = &Thunks::lifecycle_allow_self_restart,
            .lifecycle_allow_external_restart = &Thunks::lifecycle_allow_external_restart,
            .lifecycle_allow_restart_request = &Thunks::lifecycle_allow_restart_request,
        };
    }

#endif // HOLOLINK_BUILD_ARGUS_ISP
    // ---------------------------------------------------------------------------
    // ComputeCrcOp / CheckCrcOp -- GPU (nvCOMP) CRC of the received frame. Manual rows: the two
    // share a CrcChannel, which a Graph cannot pass from one operator to another, so both rows
    // take the same hololink.CrcChannel object as the "crc_channel" constructor handle. Python's
    // validation logic is that object's check_crc callback, which the CheckCrcOp row's
    // CallbackCheckCrcOp calls for each frame (Python cannot subclass a registry operator).
    // ComputeCrcOp is instantiated for holoscan::Tensor: it sits on the receiver's raw frame,
    // in front of the converter (MIGRATION.md, CRC Operators). ComputeCrcImageOp is the same
    // operator for a schema::ImageT, e.g. CsiToBayerOp's or BayerDemosaicOp's, paired with the
    // frame's metadata.
    // ---------------------------------------------------------------------------

    constexpr holoscan::authoring::ConstructorHandleDescriptor kCrcChannelHandleDescriptor {
        .name = "hololink.CrcChannel",
        .description = "Links a ComputeCrcOp to its CheckCrcOp (hololink.CrcChannel).",
    };

    [[nodiscard]] std::span<const ConstructorHandleInput> crc_channel_handle_inputs()
    {
        static const std::array inputs {
            ConstructorHandleInput {
                .name = "crc_channel",
                .description = "The hololink.CrcChannel shared by a ComputeCrcOp and its CheckCrcOp.",
                .type = &kCrcChannelHandleDescriptor,
            },
        };
        return inputs;
    }

    template <typename Thunks>
    [[nodiscard]] const RegisteredConstructorInputs* crc_channel_constructor_inputs()
    {
        static const RegisteredConstructorInputs inputs {
            .handles = crc_channel_handle_inputs(),
            .add_to_graph_from_inputs = &Thunks::add_from_inputs,
            .add_for_description = &Thunks::add_for_description,
        };
        return &inputs;
    }

    constexpr std::array<std::string_view, 1> kComputeCrcRequiredParameters { "frame_size" };
    constexpr std::array<std::string_view, 0> kCheckCrcRequiredParameters {};

    template <typename ComputeCrcOpT>
    struct ComputeCrcRegistrationThunks final
        : holoscan::authoring::ManualOperatorRegistrationThunks<ComputeCrcOpT> {
        using Base = holoscan::authoring::ManualOperatorRegistrationThunks<ComputeCrcOpT>;
        using Base::invoke_add;
        using Base::make_operator_ref;

        static OpaqueOperatorRef add(Graph& graph, std::string_view name,
            const std::shared_ptr<CrcCheckChannel>& crc_channel, int cuda_device_ordinal,
            uint64_t frame_size, bool emit_crc = false)
        {
            auto op = graph.op<ComputeCrcOpT>(
                name, crc_channel->channel, cuda_device_ordinal, frame_size, emit_crc);
            return make_operator_ref(op);
        }

        static expected<OpaqueOperatorRef, Error> add_from_inputs(
            Graph& graph, std::string_view name, const OperatorConstructionInputs& inputs)
        {
            auto crc_channel = inputs.require_shared("crc_channel", crc_channel_handle_type());
            if (!crc_channel) {
                return holoscan::make_unexpected(std::move(crc_channel).error());
            }
            return invoke_add([&] {
                AssignmentView values { inputs.values, {} };
                const auto frame_size = integer_value<uint64_t>(
                    values.require("frame_size"), values.field_path("frame_size"));
                const auto cuda_device_ordinal = integer_or<int>(values, "cuda_device_ordinal", 0);
                const auto emit_crc = bool_or(values, "emit_crc", false);
                values.reject_unknown();
                return add(graph, name, *crc_channel, cuda_device_ordinal, frame_size, emit_crc);
            });
        }

        static expected<OpaqueOperatorRef, Error> add_for_description(
            Graph& graph, std::string_view name)
        {
            return invoke_add(
                [&] { return add(graph, name, std::make_shared<CrcCheckChannel>(), 0, 1); });
        }

        static void set_params(const OpaqueOperatorRef&,
            const holoscan::authoring::MaterializedParameters&)
        {
            throw RuntimeError {
                Error { ErrorCode::kNotSupported, "AUTHORING_MANUAL_ROW_PARAMETERS_IMMUTABLE" }
            };
        }
    };

    struct CheckCrcRegistrationThunks final
        : holoscan::authoring::ManualOperatorRegistrationThunks<CallbackCheckCrcOp> {
        static OpaqueOperatorRef add(
            Graph& graph, std::string_view name, std::shared_ptr<CrcCheckChannel> crc_channel)
        {
            auto op = graph.op<CallbackCheckCrcOp>(name, std::move(crc_channel));
            return make_operator_ref(op);
        }

        static expected<OpaqueOperatorRef, Error> add_from_inputs(
            Graph& graph, std::string_view name, const OperatorConstructionInputs& inputs)
        {
            auto crc_channel = inputs.require_shared("crc_channel", crc_channel_handle_type());
            if (!crc_channel) {
                return holoscan::make_unexpected(std::move(crc_channel).error());
            }
            return invoke_add([&] {
                AssignmentView values { inputs.values, {} };
                values.reject_unknown();
                return add(graph, name, std::move(*crc_channel));
            });
        }

        static expected<OpaqueOperatorRef, Error> add_for_description(
            Graph& graph, std::string_view name)
        {
            return invoke_add([&] { return add(graph, name, std::make_shared<CrcCheckChannel>()); });
        }

        static void set_params(const OpaqueOperatorRef&,
            const holoscan::authoring::MaterializedParameters&)
        {
            throw RuntimeError {
                Error { ErrorCode::kNotSupported, "AUTHORING_MANUAL_ROW_PARAMETERS_IMMUTABLE" }
            };
        }
    };

    template <typename Thunks, std::size_t N>
    [[nodiscard]] holoscan::authoring::RegisteredOperatorType crc_row(
        std::string_view name, const std::array<std::string_view, N>& required_parameters)
    {
        return holoscan::authoring::RegisteredOperatorType {
            .name = name,
            .param_type = &holoscan::param_type_descriptor<holoscan::NoParams>(),
            .param_schema = nullptr,
            .parameter_introspection = holoscan::authoring::OperatorParameterIntrospection::kPassThrough,
            .required_parameters = required_parameters,
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
            .constructor_inputs = crc_channel_constructor_inputs<Thunks>(),
        };
    }

    [[nodiscard]] holoscan::authoring::RegisteredOperatorType compute_crc_row()
    {
        return crc_row<ComputeCrcRegistrationThunks<ComputeCrcTensorOp>>(
            "hololink.operators.ComputeCrcOp", kComputeCrcRequiredParameters);
    }

    [[nodiscard]] holoscan::authoring::RegisteredOperatorType compute_crc_image_row()
    {
        return crc_row<ComputeCrcRegistrationThunks<ComputeCrcImageOp>>(
            "hololink.operators.ComputeCrcImageOp", kComputeCrcRequiredParameters);
    }

    [[nodiscard]] holoscan::authoring::RegisteredOperatorType check_crc_row()
    {
        return crc_row<CheckCrcRegistrationThunks>(
            "hololink.operators.CheckCrcOp", kCheckCrcRequiredParameters);
    }

#ifdef HOLOLINK_HAS_IMGUI
    // ---------------------------------------------------------------------------
    // SignalGeneratorOp / IQEncoderOp / IQDecoderOp / SignalViewerOp -- manual rows: these take
    // constructor arguments (Operator<>, no Params). Their GUI renderer is a live object no parameter
    // value can carry, so a `gui` flag selects the process-wide ImGuiRenderer::shared() (a process
    // has one ImGui context, so one renderer, anyway).
    // ---------------------------------------------------------------------------

    [[nodiscard]] std::shared_ptr<hololink::ImGuiRenderer> gui_renderer(bool gui)
    {
        return gui ? hololink::ImGuiRenderer::shared() : nullptr;
    }

    constexpr std::array<std::string_view, 1> kSignalGeneratorRequiredParameters { "samples_count" };

    struct SignalGeneratorRegistrationThunks final
        : holoscan::authoring::ManualOperatorRegistrationThunks<SignalGeneratorOp> {
        static OpaqueOperatorRef add(Graph& graph, std::string_view name, unsigned samples_count,
            const Rational& sampling_interval, std::string in_phase, std::string quadrature,
            std::string cuda_toolkit_include_path, std::shared_ptr<hololink::ImGuiRenderer> renderer,
            std::chrono::nanoseconds period, uint64_t count)
        {
            return make_operator_ref(graph.op<SignalGeneratorOp>(name, samples_count, sampling_interval,
                std::move(in_phase), std::move(quadrature), std::move(cuda_toolkit_include_path),
                std::move(renderer), period, count));
        }

        static OpaqueOperatorRef add_to_graph(Graph& graph, std::string_view name,
            const holoscan::authoring::MaterializedParameters* params)
        {
            if (params != nullptr) {
                parameter_error("AUTHORING_PARAMETER_TYPE_MISMATCH");
            }
            // Discovery placeholders; samples_count must be a multiple of 4.
            return add(graph, name, 4U, Rational(1, 128), "0", "0", "", nullptr, std::chrono::milliseconds(1), 0U);
        }

        static void set_params(const OpaqueOperatorRef&,
            const holoscan::authoring::MaterializedParameters&)
        {
            throw RuntimeError {
                Error { ErrorCode::kNotSupported, "AUTHORING_MANUAL_ROW_PARAMETERS_IMMUTABLE" }
            };
        }
    };

    expected<OpaqueOperatorRef, Error> signal_generator_add_from_assignments(
        Graph& graph, std::string_view name, std::span<const ParameterAssignment> assignments)
    {
        return SignalGeneratorRegistrationThunks::invoke_add([&] {
            if (assignments.empty()) {
                return SignalGeneratorRegistrationThunks::add_to_graph(graph, name, nullptr);
            }
            AssignmentView values { assignments, {} };
            const auto samples_count = integer_value<unsigned>(
                values.require("samples_count"), values.field_path("samples_count"));
            if ((samples_count == 0) || (samples_count % 4)) {
                parameter_error("AUTHORING_PARAMETER_VALUE_INVALID", values.field_path("samples_count"),
                    "detail=must be a positive multiple of 4");
            }
            // "num/den", as on the apps' command lines.
            const std::string sampling_interval_string = string_or(values, "sampling_interval", "1/128");
            std::optional<Rational> sampling_interval;
            try {
                sampling_interval.emplace(sampling_interval_string);
            } catch (const std::exception& e) {
                parameter_error("AUTHORING_PARAMETER_VALUE_INVALID", values.field_path("sampling_interval"),
                    fmt::format("detail={}", e.what()));
            }
            if (sampling_interval->den_ <= 0) {
                parameter_error("AUTHORING_PARAMETER_VALUE_INVALID", values.field_path("sampling_interval"),
                    "detail=the denominator must be positive");
            }
            std::string in_phase = string_or(values, "in_phase", "0");
            std::string quadrature = string_or(values, "quadrature", "0");
            std::string cuda_toolkit_include_path = string_or(values, "cuda_toolkit_include_path", "");
            const auto period_ns = integer_or<int64_t>(values, "period_ns", 1'000'000);
            if (period_ns <= 0) {
                parameter_error("AUTHORING_PARAMETER_VALUE_INVALID", values.field_path("period_ns"),
                    "detail=must be positive");
            }
            const auto count = integer_or<uint64_t>(values, "count", 0);
            const bool gui = bool_or(values, "gui", false);
            values.reject_unknown();
            return SignalGeneratorRegistrationThunks::add(graph, name, samples_count, *sampling_interval,
                std::move(in_phase), std::move(quadrature), std::move(cuda_toolkit_include_path),
                gui_renderer(gui), std::chrono::nanoseconds(period_ns), count);
        });
    }

    [[nodiscard]] holoscan::authoring::RegisteredOperatorType signal_generator_row()
    {
        using Thunks = SignalGeneratorRegistrationThunks;
        return holoscan::authoring::RegisteredOperatorType {
            .name = "hololink.operators.SignalGeneratorOp",
            .param_type = &holoscan::param_type_descriptor<holoscan::NoParams>(),
            .param_schema = nullptr,
            .parameter_introspection = holoscan::authoring::OperatorParameterIntrospection::kPassThrough,
            .required_parameters = kSignalGeneratorRequiredParameters,
            .add_to_graph = &Thunks::add_to_graph,
            .add_to_graph_from_assignments = &signal_generator_add_from_assignments,
            .set_params = &Thunks::set_params,
            .port = &Thunks::port,
            .ports = &Thunks::ports,
            .region_add = &Thunks::region_add,
            .partition_add = &Thunks::partition_add,
            .lifecycle_before = &Thunks::lifecycle_before,
            .lifecycle_allow_self_restart = &Thunks::lifecycle_allow_self_restart,
            .lifecycle_allow_external_restart = &Thunks::lifecycle_allow_external_restart,
            .lifecycle_allow_restart_request = &Thunks::lifecycle_allow_restart_request,
        };
    }

    constexpr std::array<std::string_view, 0> kIQEncoderRequiredParameters {};

    struct IQEncoderRegistrationThunks final
        : holoscan::authoring::ManualOperatorRegistrationThunks<IQEncoderOp> {
        static OpaqueOperatorRef add(Graph& graph, std::string_view name, float scale,
            std::shared_ptr<hololink::ImGuiRenderer> renderer)
        {
            return make_operator_ref(graph.op<IQEncoderOp>(name, scale, std::move(renderer)));
        }

        static OpaqueOperatorRef add_to_graph(Graph& graph, std::string_view name,
            const holoscan::authoring::MaterializedParameters* params)
        {
            if (params != nullptr) {
                parameter_error("AUTHORING_PARAMETER_TYPE_MISMATCH");
            }
            return add(graph, name, 1.0f, nullptr);
        }

        static void set_params(const OpaqueOperatorRef&,
            const holoscan::authoring::MaterializedParameters&)
        {
            throw RuntimeError {
                Error { ErrorCode::kNotSupported, "AUTHORING_MANUAL_ROW_PARAMETERS_IMMUTABLE" }
            };
        }
    };

    expected<OpaqueOperatorRef, Error> iq_encoder_add_from_assignments(
        Graph& graph, std::string_view name, std::span<const ParameterAssignment> assignments)
    {
        return IQEncoderRegistrationThunks::invoke_add([&] {
            if (assignments.empty()) {
                return IQEncoderRegistrationThunks::add_to_graph(graph, name, nullptr);
            }
            AssignmentView values { assignments, {} };
            const auto scale = real_or<float>(values, "scale", 1.0f);
            const bool gui = bool_or(values, "gui", false);
            values.reject_unknown();
            return IQEncoderRegistrationThunks::add(graph, name, scale, gui_renderer(gui));
        });
    }

    [[nodiscard]] holoscan::authoring::RegisteredOperatorType iq_encoder_row()
    {
        using Thunks = IQEncoderRegistrationThunks;
        return holoscan::authoring::RegisteredOperatorType {
            .name = "hololink.operators.IQEncoderOp",
            .param_type = &holoscan::param_type_descriptor<holoscan::NoParams>(),
            .param_schema = nullptr,
            .parameter_introspection = holoscan::authoring::OperatorParameterIntrospection::kPassThrough,
            .required_parameters = kIQEncoderRequiredParameters,
            .add_to_graph = &Thunks::add_to_graph,
            .add_to_graph_from_assignments = &iq_encoder_add_from_assignments,
            .set_params = &Thunks::set_params,
            .port = &Thunks::port,
            .ports = &Thunks::ports,
            .region_add = &Thunks::region_add,
            .partition_add = &Thunks::partition_add,
            .lifecycle_before = &Thunks::lifecycle_before,
            .lifecycle_allow_self_restart = &Thunks::lifecycle_allow_self_restart,
            .lifecycle_allow_external_restart = &Thunks::lifecycle_allow_external_restart,
            .lifecycle_allow_restart_request = &Thunks::lifecycle_allow_restart_request,
        };
    }

    constexpr std::array<std::string_view, 1> kIQDecoderRequiredParameters { "max_samples" };

    struct IQDecoderRegistrationThunks final
        : holoscan::authoring::ManualOperatorRegistrationThunks<IQDecoderOp> {
        static OpaqueOperatorRef add(Graph& graph, std::string_view name, uint64_t max_samples, float scale,
            std::shared_ptr<hololink::ImGuiRenderer> renderer)
        {
            return make_operator_ref(graph.op<IQDecoderOp>(name, max_samples, scale, std::move(renderer)));
        }

        static OpaqueOperatorRef add_to_graph(Graph& graph, std::string_view name,
            const holoscan::authoring::MaterializedParameters* params)
        {
            if (params != nullptr) {
                parameter_error("AUTHORING_PARAMETER_TYPE_MISMATCH");
            }
            // Discovery placeholder; max_samples must be positive.
            return add(graph, name, 4U, 1.0f, nullptr);
        }

        static void set_params(const OpaqueOperatorRef&,
            const holoscan::authoring::MaterializedParameters&)
        {
            throw RuntimeError {
                Error { ErrorCode::kNotSupported, "AUTHORING_MANUAL_ROW_PARAMETERS_IMMUTABLE" }
            };
        }
    };

    expected<OpaqueOperatorRef, Error> iq_decoder_add_from_assignments(
        Graph& graph, std::string_view name, std::span<const ParameterAssignment> assignments)
    {
        return IQDecoderRegistrationThunks::invoke_add([&] {
            if (assignments.empty()) {
                return IQDecoderRegistrationThunks::add_to_graph(graph, name, nullptr);
            }
            AssignmentView values { assignments, {} };
            const auto max_samples = integer_value<uint64_t>(
                values.require("max_samples"), values.field_path("max_samples"));
            if (max_samples == 0) {
                parameter_error("AUTHORING_PARAMETER_VALUE_INVALID", values.field_path("max_samples"),
                    "detail=must be positive");
            }
            const auto scale = real_or<float>(values, "scale", 1.0f);
            const bool gui = bool_or(values, "gui", false);
            values.reject_unknown();
            return IQDecoderRegistrationThunks::add(graph, name, max_samples, scale, gui_renderer(gui));
        });
    }

    [[nodiscard]] holoscan::authoring::RegisteredOperatorType iq_decoder_row()
    {
        using Thunks = IQDecoderRegistrationThunks;
        return holoscan::authoring::RegisteredOperatorType {
            .name = "hololink.operators.IQDecoderOp",
            .param_type = &holoscan::param_type_descriptor<holoscan::NoParams>(),
            .param_schema = nullptr,
            .parameter_introspection = holoscan::authoring::OperatorParameterIntrospection::kPassThrough,
            .required_parameters = kIQDecoderRequiredParameters,
            .add_to_graph = &Thunks::add_to_graph,
            .add_to_graph_from_assignments = &iq_decoder_add_from_assignments,
            .set_params = &Thunks::set_params,
            .port = &Thunks::port,
            .ports = &Thunks::ports,
            .region_add = &Thunks::region_add,
            .partition_add = &Thunks::partition_add,
            .lifecycle_before = &Thunks::lifecycle_before,
            .lifecycle_allow_self_restart = &Thunks::lifecycle_allow_self_restart,
            .lifecycle_allow_external_restart = &Thunks::lifecycle_allow_external_restart,
            .lifecycle_allow_restart_request = &Thunks::lifecycle_allow_restart_request,
        };
    }

    constexpr std::array<std::string_view, 0> kSignalViewerRequiredParameters {};

    struct SignalViewerRegistrationThunks final
        : holoscan::authoring::ManualOperatorRegistrationThunks<SignalViewerOp> {
        static OpaqueOperatorRef add(Graph& graph, std::string_view name,
            std::shared_ptr<hololink::ImGuiRenderer> renderer)
        {
            return make_operator_ref(graph.op<SignalViewerOp>(name, std::move(renderer)));
        }

        static OpaqueOperatorRef add_to_graph(Graph& graph, std::string_view name,
            const holoscan::authoring::MaterializedParameters* params)
        {
            if (params != nullptr) {
                parameter_error("AUTHORING_PARAMETER_TYPE_MISMATCH");
            }
            return add(graph, name, nullptr);
        }

        static void set_params(const OpaqueOperatorRef&,
            const holoscan::authoring::MaterializedParameters&)
        {
            throw RuntimeError {
                Error { ErrorCode::kNotSupported, "AUTHORING_MANUAL_ROW_PARAMETERS_IMMUTABLE" }
            };
        }
    };

    expected<OpaqueOperatorRef, Error> signal_viewer_add_from_assignments(
        Graph& graph, std::string_view name, std::span<const ParameterAssignment> assignments)
    {
        return SignalViewerRegistrationThunks::invoke_add([&] {
            if (assignments.empty()) {
                return SignalViewerRegistrationThunks::add_to_graph(graph, name, nullptr);
            }
            AssignmentView values { assignments, {} };
            const bool gui = bool_or(values, "gui", false);
            values.reject_unknown();
            return SignalViewerRegistrationThunks::add(graph, name, gui_renderer(gui));
        });
    }

    [[nodiscard]] holoscan::authoring::RegisteredOperatorType signal_viewer_row()
    {
        using Thunks = SignalViewerRegistrationThunks;
        return holoscan::authoring::RegisteredOperatorType {
            .name = "hololink.operators.SignalViewerOp",
            .param_type = &holoscan::param_type_descriptor<holoscan::NoParams>(),
            .param_schema = nullptr,
            .parameter_introspection = holoscan::authoring::OperatorParameterIntrospection::kPassThrough,
            .required_parameters = kSignalViewerRequiredParameters,
            .add_to_graph = &Thunks::add_to_graph,
            .add_to_graph_from_assignments = &signal_viewer_add_from_assignments,
            .set_params = &Thunks::set_params,
            .port = &Thunks::port,
            .ports = &Thunks::ports,
            .region_add = &Thunks::region_add,
            .partition_add = &Thunks::partition_add,
            .lifecycle_before = &Thunks::lifecycle_before,
            .lifecycle_allow_self_restart = &Thunks::lifecycle_allow_self_restart,
            .lifecycle_allow_external_restart = &Thunks::lifecycle_allow_external_restart,
            .lifecycle_allow_restart_request = &Thunks::lifecycle_allow_restart_request,
        };
    }
#endif // HOLOLINK_HAS_IMGUI

} // namespace

const holoscan::authoring::ConstructorHandleType<DataChannel>& data_channel_handle_type()
{
    static const holoscan::authoring::ConstructorHandleType<DataChannel> type {
        kDataChannelHandleDescriptor
    };
    return type;
}

const holoscan::authoring::ConstructorHandleType<DataChannels>& data_channels_handle_type()
{
    static const holoscan::authoring::ConstructorHandleType<DataChannels> type {
        kDataChannelsHandleDescriptor
    };
    return type;
}

const holoscan::authoring::ConstructorHandleType<CrcCheckChannel>& crc_channel_handle_type()
{
    static const holoscan::authoring::ConstructorHandleType<CrcCheckChannel> type {
        kCrcChannelHandleDescriptor
    };
    return type;
}

HOLOSCAN_REGISTER_OPERATOR(CsiToBayerOp, "hololink.operators.CsiToBayerOp");
HOLOSCAN_REGISTER_OPERATOR(ImageProcessorOp, "hololink.operators.ImageProcessorOp");
HOLOSCAN_REGISTER_OPERATOR(PackedFormatConverterOp, "hololink.operators.PackedFormatConverterOp");
HOLOSCAN_REGISTER_OPERATOR(UdpTransmitterOp, "hololink.operators.UdpTransmitterOp");
HOLOSCAN_REGISTER_OPERATOR(DiscardMetadataOp, "hololink.operators.DiscardMetadataOp");

#ifdef HOLOLINK_BUILD_ROCE
// RoceTransmitterOp -- already a plain default-constructible Operator<RoceTransmitterParams>
// (no live-object dependency, unlike RoceReceiverOp above), so it needs only this
// self-registration macro, not a manual row. Guarded because roce_transmitter, like
// roce_receiver, is itself only built under HOLOLINK_BUILD_ROCE.
HOLOSCAN_REGISTER_OPERATOR(RoceTransmitterOp, "hololink.operators.RoceTransmitterOp");
#endif

holoscan::expected<void, holoscan::Error> register_hololink_operator_catalog()
{
    static const holoscan::expected<void, holoscan::Error> manual_rows_status = [] {
        auto& registry = holoscan::authoring::authoring_registry();
        auto status = registry.register_operator(sub_frame_combiner_row());
        if (!status) {
            return status;
        }
        status = registry.register_operator(linux_receiver_row());
        if (!status) {
            return status;
        }
        status = registry.register_operator(linux_coe_receiver_row());
        if (!status) {
            return status;
        }
        status = registry.register_operator(compute_crc_row());
        if (!status) {
            return status;
        }
        status = registry.register_operator(compute_crc_image_row());
        if (!status) {
            return status;
        }
        status = registry.register_operator(check_crc_row());
#ifdef HOLOLINK_BUILD_ROCE
        if (!status) {
            return status;
        }
        status = registry.register_operator(roce_receiver_row());
#endif
#ifdef HOLOLINK_BUILD_GPUNETIO_ROCE
        if (!status) {
            return status;
        }
        status = registry.register_operator(gpu_roce_transceiver_row());
#endif
#ifdef HOLOLINK_BUILD_FUSA
        if (!status) {
            return status;
        }
        status = registry.register_operator(fusa_coe_capture_row());
#endif
#ifdef HOLOLINK_BUILD_ARGUS_ISP
        if (!status) {
            return status;
        }
        status = registry.register_operator(argus_isp_row());
#endif
#ifdef HOLOLINK_HAS_IMGUI
        for (auto row : { signal_generator_row(), iq_encoder_row(), iq_decoder_row(), signal_viewer_row() }) {
            if (!status) {
                return status;
            }
            status = registry.register_operator(std::move(row));
        }
#endif
        return status;
    }();
    if (!manual_rows_status) {
        return manual_rows_status;
    }
    return holoscan::authoring::authoring_registry().registration_status();
}

} // namespace hololink::operators
