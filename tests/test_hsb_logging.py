# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for HSB logging Python wrappers."""

import pytest

import hololink


class TestHsbLogging:
    """Test HSB logging functionality through Python wrappers."""

    def test_hsb_log_levels_available(self):
        """Test that all HSB log level constants are available."""
        assert hasattr(hololink, "HSB_LOG_LEVEL_TRACE")
        assert hasattr(hololink, "HSB_LOG_LEVEL_DEBUG")
        assert hasattr(hololink, "HSB_LOG_LEVEL_INFO")
        assert hasattr(hololink, "HSB_LOG_LEVEL_WARN")
        assert hasattr(hololink, "HSB_LOG_LEVEL_ERROR")
        assert hasattr(hololink, "HSB_LOG_LEVEL_INVALID")

    def test_hsb_log_functions_available(self):
        """Test that all HSB log functions are available."""
        assert hasattr(hololink, "hsb_log_trace")
        assert hasattr(hololink, "hsb_log_debug")
        assert hasattr(hololink, "hsb_log_info")
        assert hasattr(hololink, "hsb_log_warn")
        assert hasattr(hololink, "hsb_log_error")
        assert callable(hololink.hsb_log_trace)
        assert callable(hololink.hsb_log_debug)
        assert callable(hololink.hsb_log_info)
        assert callable(hololink.hsb_log_warn)
        assert callable(hololink.hsb_log_error)

    def test_hsb_log_level_control_functions(self):
        """Test HSB log level get/set functions."""
        assert hasattr(hololink, "get_hsb_log_level")
        assert hasattr(hololink, "set_hsb_log_level")
        assert callable(hololink.get_hsb_log_level)
        assert callable(hololink.set_hsb_log_level)

    def test_set_and_get_log_level(self):
        """Test setting and getting log levels."""
        # Save original level
        original_level = hololink.get_hsb_log_level()

        try:
            # Test setting different levels
            hololink.set_hsb_log_level(hololink.HSB_LOG_LEVEL_DEBUG)
            assert hololink.get_hsb_log_level() == hololink.HSB_LOG_LEVEL_DEBUG

            hololink.set_hsb_log_level(hololink.HSB_LOG_LEVEL_WARN)
            assert hololink.get_hsb_log_level() == hololink.HSB_LOG_LEVEL_WARN

            hololink.set_hsb_log_level(hololink.HSB_LOG_LEVEL_ERROR)
            assert hololink.get_hsb_log_level() == hololink.HSB_LOG_LEVEL_ERROR
        finally:
            # Restore original level
            hololink.set_hsb_log_level(original_level)

    def test_hsb_log_functions_basic_call(self):
        """Test that HSB log functions can be called without errors."""
        # Save original level and set to TRACE to ensure all messages are logged
        original_level = hololink.get_hsb_log_level()

        try:
            hololink.set_hsb_log_level(hololink.HSB_LOG_LEVEL_TRACE)

            # Test that functions can be called without throwing exceptions
            hololink.hsb_log_trace("Test trace message")
            hololink.hsb_log_debug("Test debug message")
            hololink.hsb_log_info("Test info message")
            hololink.hsb_log_warn("Test warning message")
            hololink.hsb_log_error("Test error message")
        finally:
            # Restore original level
            hololink.set_hsb_log_level(original_level)

    def test_hsb_log_with_different_message_types(self):
        """Test HSB log functions with different message types."""
        original_level = hololink.get_hsb_log_level()

        try:
            hololink.set_hsb_log_level(hololink.HSB_LOG_LEVEL_TRACE)

            # Test with different string types
            hololink.hsb_log_info("Simple message")
            hololink.hsb_log_info("Message with numbers: 123")
            hololink.hsb_log_info("Message with special chars: !@#$%^&*()")
            hololink.hsb_log_info("")  # Empty message

            # Test with longer message
            long_message = "A" * 1000
            hololink.hsb_log_info(long_message)
        finally:
            hololink.set_hsb_log_level(original_level)

    def test_hsb_log_level_filtering(self):
        """Test that log level filtering works correctly."""
        original_level = hololink.get_hsb_log_level()

        try:
            # Set level to WARN - should filter out TRACE, DEBUG, INFO
            hololink.set_hsb_log_level(hololink.HSB_LOG_LEVEL_WARN)

            # These should be filtered (no exception should occur)
            hololink.hsb_log_trace("This should be filtered")
            hololink.hsb_log_debug("This should be filtered")
            hololink.hsb_log_info("This should be filtered")

            # These should pass through
            hololink.hsb_log_warn("This should be logged")
            hololink.hsb_log_error("This should be logged")

        finally:
            hololink.set_hsb_log_level(original_level)

    def test_hsb_log_invalid_arguments(self):
        """Test HSB log functions with invalid arguments."""
        with pytest.raises(TypeError):
            hololink.hsb_log_info()  # Missing required argument

        with pytest.raises(TypeError):
            hololink.hsb_log_info(123)  # Wrong argument type

    def test_hsb_log_level_enum_values(self):
        """Test that HSB log level enum has expected numeric values."""
        # Based on the C++ header, verify the expected values
        assert hololink.HSB_LOG_LEVEL_INVALID == 0
        assert hololink.HSB_LOG_LEVEL_TRACE == 10
        assert hololink.HSB_LOG_LEVEL_DEBUG == 20
        assert hololink.HSB_LOG_LEVEL_INFO == 30
        assert hololink.HSB_LOG_LEVEL_WARN == 40
        assert hololink.HSB_LOG_LEVEL_ERROR == 50
