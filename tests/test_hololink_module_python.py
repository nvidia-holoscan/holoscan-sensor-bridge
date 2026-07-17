# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end smoke test for the hololink_module pybind extension.
# Loads the HSB-Lite module .so by UUID through the Python-facing
# Adapter, then reads the enriched metadata back across the binding.

import threading
import time

import hololink_module

HSB_LITE_UUID = "889b7ce3-65a5-4247-8b05-4ff1904c3359"


def test_python_adapter_drives_hsb_lite_module(module_dir):
    adapter = hololink_module.Adapter.get_adapter()
    adapter.set_module_directory(module_dir)

    metadata = hololink_module.EnumerationMetadata()
    metadata["fpga_uuid"] = HSB_LITE_UUID
    metadata["peer_ip"] = "192.168.0.42"
    metadata["serial_number"] = "py-test-001"
    # compat_id is the FPGA's 16-bit IP version field as a numeric
    # value (0x2603 here). The matching .so filename embeds the
    # 4-digit hex rendering "2603".
    metadata["compat_id"] = 0x2603
    # The HSB-Lite EnumerationInterface override reads data_plane
    # off the metadata to compute the per-data-plane address fields.
    # Real bootp populates this from the packet payload; manual
    # enumerations like this smoke test set it explicitly.
    metadata["data_plane"] = 0

    # wait_for_channel discards any cached announcement before it
    # starts waiting, so enumerate must run from another thread that
    # fires after wait_for_channel is already blocked on its cv.
    def _delayed_enumerate():
        time.sleep(0.05)
        adapter.enumerate(metadata)

    enum_thread = threading.Thread(target=_delayed_enumerate)
    enum_thread.start()
    found = adapter.wait_for_channel("192.168.0.42", timeout_s=1.0)
    enum_thread.join()

    # The HSB-Lite EnumerationInterface override stamps these.
    assert found["module_name"] == "hsb_lite"
    assert found["compat_id"] == 0x2603
    assert found["peer_ip"] == "192.168.0.42"
    assert found["serial_number"] == "py-test-001"

    # get_module(metadata) resolves to the same cached Module that
    # enumerate() loaded, without the caller composing a .so path.
    module = adapter.get_module(found)
    assert module is not None


def test_metadata_get_default():
    metadata = hololink_module.EnumerationMetadata()
    metadata["serial_number"] = "abc"
    assert metadata.get("serial_number") == "abc"
    assert metadata.get("missing") is None
    assert metadata.get("missing", "fallback") == "fallback"
