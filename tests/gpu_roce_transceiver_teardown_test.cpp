/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "gtest/gtest.h"

#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include <hololink/operators/gpu_roce_transceiver/gpu_roce_transceiver.hpp>
#include <hololink/transport/roce/ibv.hpp>

namespace {

namespace ibv = hololink::transport::roce::ibv;

// Ring geometry is irrelevant here; the QP never carries traffic.
constexpr unsigned kRemoteQp = 0x2;
constexpr std::size_t kFrameSize = 32;
constexpr std::size_t kPageSize = 384;
constexpr unsigned kPages = 128;
constexpr uint32_t kGpuId = 0;
constexpr uint32_t kIbvPort = 1;

struct RoceTestDeviceInfo {
    std::string ibv_name;
    uint32_t ibv_port;
    std::string host_ip;
};

std::optional<RoceTestDeviceInfo> roce_device()
{
    const std::optional<std::string> ibv_name
        = ibv::DeviceList().first_name();
    if (!ibv_name) {
        return std::nullopt;
    }
    ibv::Context context(ibv_name.value());
    const int gid_index = context.query_gid_roce_v2_ip();
    const std::string host_ip
        = ibv::gid_to_ipv4(context.query_gid(kIbvPort, gid_index));
    if (host_ip.empty()) {
        return std::nullopt;
    }
    return RoceTestDeviceInfo { ibv_name.value(), kIbvPort, host_ip };
}

// DOCA reports teardown failures only through its log: the destructors return
// void and swallow every error, so a leaked object is otherwise invisible and
// the process still exits 0.  Redirect both standard streams into a temporary
// file for the duration of the destructor and inspect what DOCA wrote.
class OutputCapture {
public:
    OutputCapture()
        : file_(std::tmpfile())
    {
        if (file_ == nullptr) {
            return;
        }
        fflush(stdout);
        fflush(stderr);
        saved_stdout_ = dup(STDOUT_FILENO);
        saved_stderr_ = dup(STDERR_FILENO);
        dup2(fileno(file_), STDOUT_FILENO);
        dup2(fileno(file_), STDERR_FILENO);
    }

    std::string release()
    {
        if (file_ == nullptr) {
            return {};
        }
        fflush(stdout);
        fflush(stderr);
        if (saved_stdout_ >= 0) {
            dup2(saved_stdout_, STDOUT_FILENO);
            close(saved_stdout_);
            saved_stdout_ = -1;
        }
        if (saved_stderr_ >= 0) {
            dup2(saved_stderr_, STDERR_FILENO);
            close(saved_stderr_);
            saved_stderr_ = -1;
        }

        std::string captured;
        const long size = ftell(file_);
        if (size > 0) {
            captured.resize(static_cast<std::size_t>(size));
            rewind(file_);
            const std::size_t read_bytes
                = fread(captured.data(), 1, captured.size(), file_);
            captured.resize(read_bytes);
        }
        fclose(file_);
        file_ = nullptr;
        return captured;
    }

    ~OutputCapture() { release(); }

    // False if the temporary file could not be created, in which case nothing
    // is redirected and release() returns an empty string.  Callers must check
    // this: an empty log would otherwise satisfy a "no errors were logged"
    // assertion without anything having been observed.
    bool ready() const { return file_ != nullptr; }

private:
    std::FILE* file_ = nullptr;
    int saved_stdout_ = -1;
    int saved_stderr_ = -1;
};

} // namespace

// Destroying the transceiver must release its DOCA objects.  Each destructor
// has to drop the owner before the memory it references -- the CQ before its
// umem, the QP before its umems, and the QP before the CQs it uses -- or DEVX
// refuses with EINVAL and every object leaks for the lifetime of the process.
TEST(GpuRoceTransceiverTeardownTest, DestructorReleasesDocaObjects)
{
    // Deliberately no CUDA runtime probe here: GpuRoceTransceiver creates and
    // pops its own driver context, and retaining the runtime's primary context
    // first makes DOCA's cuStreamCreate fail with 709.  A machine without a
    // usable GPU falls out of start() below instead.
    const auto device = roce_device();
    if (!device) {
        GTEST_SKIP() << "No RoCE-capable InfiniBand device available";
    }

    // The peer is this same port: the QP reaches RTS and is torn down without
    // a packet ever being sent, so no second endpoint is required.
    auto transceiver = std::make_unique<hololink::operators::GpuRoceTransceiver>(
        device->ibv_name.c_str(), device->ibv_port, std::to_string(kRemoteQp), kGpuId,
        kFrameSize, kPageSize, kPages, device->host_ip.c_str(),
        /*forward=*/false, /*rx_only=*/false, /*tx_only=*/false);

    if (!transceiver->start()) {
        GTEST_SKIP() << "GpuRoceTransceiver could not start on "
                     << device->ibv_name;
    }
    transceiver->close();

    std::string teardown_log;
    {
        // Capture only the destructor: start() logs an unrelated GDRCopy error
        // when libgdrapi.so is absent, which is not what this test is about.
        OutputCapture capture;
        ASSERT_TRUE(capture.ready())
            << "std::tmpfile() failed; the teardown log cannot be checked";
        transceiver.reset();
        teardown_log = capture.release();
    }

    EXPECT_EQ(teardown_log.find("[ERR]"), std::string::npos)
        << "DOCA reported errors while destroying the transceiver:\n"
        << teardown_log;
}
