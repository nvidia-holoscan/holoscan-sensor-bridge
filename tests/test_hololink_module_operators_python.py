# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Smoke test for the hololink_module Holoscan-coupled operator
# bindings. Verifies that the per-module sub-package is always
# importable, and that the RoceReceiverOp class symbol resolves when the
# RoCE receiver operator was built into this configuration.

import hololink_module.operators
import pytest

import hololink_module


def test_operator_subpackage_imports():
    # The operators sub-package imports regardless of which operators the
    # build enabled (e.g. RoceReceiverOp requires a RoCE-capable build).
    assert hololink_module.operators is not None


@pytest.mark.accelerated_networking
def test_roce_receiver_op_class_resolves():
    # Per the per-module Python sub-package convention, the
    # operator class is reachable as hololink_module.operators.X.
    # Only present when the RoCE receiver operator was built.
    assert hasattr(hololink_module.operators, "RoceReceiverOp")


def test_core_extension_registered_v1_types():
    # The operator's parameter types resolve through the core
    # extension; the operators sub-package's __init__ imports the
    # core extension first so registrations are visible.
    assert hololink_module.FrameMetadataInterfaceV1 is not None
    assert hololink_module.EnumerationMetadata is not None
