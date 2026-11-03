// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <holoscan/authoring/constructor_handle.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

#include "hololink/module/enumeration_metadata.hpp"

namespace hololink::module::operators {

class SensorFactory;
class NetworkReceiverFactory;

/// @brief Install the hololink_module Operator rows into the process-local authoring registry.
///
/// Mirrors hololink::operators::register_hololink_operator_catalog() (see
/// src/hololink/operators/operator_catalog.hpp) for the separate, parallel
/// hololink::module::operators::* namespace. Process-idempotent and safe to call from every
/// executable or language module that links this library.
/// @return Success, or the first registration failure retained across repeated calls.
holoscan::expected<void, holoscan::Error> register_hololink_module_operator_catalog();

/// @brief The constructor-handle type of a `hololink::module::EnumerationMetadata`, stable name
/// `hololink_module.EnumerationMetadata`.
///
/// The `metadata` input of the LinuxReceiverOp, RoceReceiverOp, FusaCoeCaptureOp and
/// HsbControllerOp rows has this type; the row copies the metadata into the operator. Python's
/// `hololink_module.EnumerationMetadata` exports a copy of itself through it, so an application
/// passes the metadata itself: `graph.op(hololink_module.operators.LinuxReceiverOp, "rx",
/// metadata=channel_metadata, ...)`. Each descriptor in this header is defined once, in this
/// library.
[[nodiscard]] const holoscan::authoring::ConstructorHandleType<hololink::module::EnumerationMetadata>&
enumeration_metadata_handle_type();

/// @brief The constructor-handle type of HsbControllerOp's `sensor_factory` input, stable name
/// `hololink_module.SensorFactory`; usually a Python `hololink_module.SensorFactory` subclass.
[[nodiscard]] const holoscan::authoring::ConstructorHandleType<SensorFactory>&
sensor_factory_handle_type();

/// @brief The constructor-handle type of HsbControllerOp's `network_receiver_factory` input,
/// stable name `hololink_module.NetworkReceiverFactory`.
[[nodiscard]] const holoscan::authoring::ConstructorHandleType<NetworkReceiverFactory>&
network_receiver_factory_handle_type();

} // namespace hololink::module::operators
