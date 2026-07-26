# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# See README.md for detailed information.

import sys
import wave
from unittest import mock

import pytest

from examples import audio_recorder_fusa


@pytest.mark.skip_unless_audio
@pytest.mark.skip_unless_fusa
def test_audio_recorder_fusa(tmp_path, frame_limit, hololink_address, capsys):
    # Capture-only (--no-playback): exercises the I2S RX -> FuSa CoE -> decode -> WAV
    # path without needing an ALSA/PulseAudio output device on the test host.
    output = tmp_path / "capture.wav"
    arguments = [
        sys.argv[0],
        "--no-playback",
        "--output",
        str(output),
        "--frame-limit",
        str(frame_limit),
        "--hololink",
        hololink_address,
    ]

    with mock.patch("sys.argv", arguments):
        audio_recorder_fusa.main()

        # check for errors
        captured = capsys.readouterr()
        assert captured.err == ""

    # The run should have produced a valid, non-empty stereo S32 48 kHz WAV.
    with wave.open(str(output), "rb") as wav:
        assert wav.getnchannels() == 2
        assert wav.getsampwidth() == 4
        assert wav.getframerate() == 48000
        assert wav.getnframes() > 0
