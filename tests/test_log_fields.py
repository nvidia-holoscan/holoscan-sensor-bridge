# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pytest.ini's log_format needs two fields on every log record: the SDK log clock
# (log_timestamp_s) and the thread id (tid). tests/conftest.py adds them; without them, a
# test that logs at the capture level fails on the formatting error.

import logging
import threading

import hololink


def test_log_records_carry_the_log_format_fields(request, caplog):
    with caplog.at_level(logging.WARNING):
        logging.getLogger("hololink.test").warning("log format check")
    record = caplog.records[-1]
    assert record.tid == threading.get_native_id()
    assert 0 <= record.log_timestamp_s <= hololink.log_timestamp_s()
    logging.Formatter(request.config.getini("log_format")).format(record)
