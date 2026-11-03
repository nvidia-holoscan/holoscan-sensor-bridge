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

// Self-contained integration test for UdpTransmitterOp.
//
// Wires TensorSourceOp → UdpTransmitterOp (127.0.0.1:5556) and verifies that
// a UDP receiver gets at least 3 packets containing the expected byte pattern.
//
// Requires GPU access at runtime.  Run inside hololink-prototype with --gpus all.

#include <algorithm>
#include <arpa/inet.h>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include <cuda_runtime_api.h>
#include <dlpack/dlpack.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/domain/dlpack.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include <hololink/operators/udp_transmitter/udp_transmitter_op.hpp>

static constexpr std::size_t kTensorBytes = 256;
static constexpr std::array<int64_t, 1> kShape { static_cast<int64_t>(kTensorBytes) };
static constexpr DLDataType kByteDtype { kDLUInt, 8, 1 };
static constexpr holoscan::TensorRepresentation kDeviceFrame {
    .memory_kind = holoscan::MemoryKind::kCudaDevice,
    .dtype = kByteDtype,
    .rank = 1U,
};

// ---------------------------------------------------------------------------
// UDP receiver running in a background thread.
// ---------------------------------------------------------------------------
struct UdpReceiver {
    std::atomic<int> count { 0 };
    std::atomic<bool> data_ok { true };

    explicit UdpReceiver(std::vector<uint8_t> expected)
        : expected_(std::move(expected))
        , stop_(false)
    {
        sockfd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (sockfd_ < 0) {
            perror("socket");
            std::exit(EXIT_FAILURE);
        }
        // Bind to port 0 so the OS assigns an ephemeral port — avoids conflicts
        // when another process or a parallel test instance holds the port.
        sockaddr_in addr {};
        addr.sin_family = AF_INET;
        addr.sin_port = 0;
        addr.sin_addr.s_addr = INADDR_ANY;
        if (::bind(sockfd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            perror("bind");
            std::exit(EXIT_FAILURE);
        }
        // Discover the assigned port so we can tell the transmitter where to send.
        socklen_t len = sizeof(addr);
        ::getsockname(sockfd_, reinterpret_cast<sockaddr*>(&addr), &len);
        port_ = ntohs(addr.sin_port);

        struct timeval tv {
            .tv_sec = 0, .tv_usec = 100'000
        };
        ::setsockopt(sockfd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

        thread_ = std::thread([this] {
            std::vector<uint8_t> buf(65536);
            while (!stop_) {
                ssize_t n = ::recv(sockfd_, buf.data(), buf.size(), 0);
                if (n <= 0)
                    continue;
                if (static_cast<std::size_t>(n) == expected_.size()
                    && std::equal(buf.begin(), buf.begin() + n, expected_.begin())) {
                    count++;
                } else {
                    data_ok = false;
                    count++;
                }
            }
        });
    }

    ~UdpReceiver()
    {
        stop_ = true;
        thread_.join();
        ::close(sockfd_);
    }

    [[nodiscard]] uint16_t port() const noexcept { return port_; }

private:
    std::vector<uint8_t> expected_;
    std::atomic<bool> stop_;
    int sockfd_ { -1 };
    uint16_t port_ { 0 };
    std::thread thread_;
};

// ---------------------------------------------------------------------------
// Source operator: allocates a CUDA tensor via the HSDK5 loan API, fills
// it with bytes 0..kTensorBytes-1, and emits it at 50 Hz.
// ---------------------------------------------------------------------------
class TensorSourceOp : public holoscan::Operator<> {
public:
    holoscan::Output<holoscan::Tensor> output;

    void setup(holoscan::OperatorSpec& spec) override
    {
        spec.output(output, "output")
            .produces_tensor(holoscan::TensorOutputSpec {
                .representation = kDeviceFrame,
                .bounds = holoscan::TensorBounds {
                    .max_rank = 1U,
                    .max_byte_span = kTensorBytes,
                },
                .storage = holoscan::TensorOutputStorage::kRuntimePool,
            });
    }

    [[nodiscard]] holoscan::Contract contract() const override
    {
        holoscan::Contract c;
        c.trigger(holoscan::OnClock { .period = std::chrono::milliseconds(20) });
        return c;
    }

    void start() override
    {
        pattern_.resize(kTensorBytes);
        std::iota(pattern_.begin(), pattern_.end(), uint8_t { 0 });
    }

    [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
        holoscan::ExecutionContext& context) override
    {
        // Allocate a tensor loan from the runtime-managed pool.
        auto loan = output.allocate_tensor(
            holoscan::TensorLoanRequest {
                .shape = kShape,
                .dtype = kByteDtype,
            });
        if (!loan)
            return holoscan::make_unexpected(std::move(loan).error());

        // Open a CUDA write phase and fill with the pattern.
        auto writer = loan->write(context.cuda_stream());
        if (!writer)
            return holoscan::make_unexpected(std::move(writer).error());

        cudaMemcpyAsync(writer->data(), pattern_.data(), kTensorBytes,
            cudaMemcpyHostToDevice, context.cuda_stream().get());

        auto committed = std::move(*writer).commit();
        if (!committed)
            return holoscan::make_unexpected(std::move(committed).error());

        return output.emit(std::move(*loan),
            holoscan::EmitOptions { .capture_time = context.activation_time() });
    }

private:
    std::vector<uint8_t> pattern_;
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    std::vector<uint8_t> expected(kTensorBytes);
    std::iota(expected.begin(), expected.end(), uint8_t { 0 });

    UdpReceiver receiver(expected);

    holoscan::Graph graph { "udp_transmitter_test" };

    auto source = graph.op<TensorSourceOp>("source");
    auto tx = graph.op<hololink::operators::UdpTransmitterOp>("tx");
    tx.set_params(hololink::operators::UdpTransmitterParams {
        .ip = "127.0.0.1",
        .port = receiver.port(),
        .max_buffer_size = static_cast<uint16_t>(kTensorBytes),
        .queue_size = 1,
        .lossy = true,
    });

    graph.add_flow(source->output, tx->input);
    graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

    // For CUDA device tensors, the runtime needs to know which device hosts the
    // tensor buffer pool.  Declare this via CompileOptions.
    holoscan::CompileOptions options;
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement {
        .operator_path = "source",
        .output_port = "output",
        .device = holoscan::DeviceId { 0 },
    });

    auto plan = holoscan::compile(graph, std::move(options));
    if (!plan.ok()) {
        fprintf(stderr, "FAIL: graph compilation failed\n");
        return 1;
    }

    // Run until 3 packets received or 5 seconds elapse.
    auto session = holoscan::run_async(plan);

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (receiver.count < 3 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

    session.request_stop();
    session.wait();

    if (receiver.count < 3) {
        fprintf(stderr, "FAIL: received %d packet(s), expected >= 3\n", receiver.count.load());
        return 1;
    }
    if (!receiver.data_ok) {
        fprintf(stderr, "FAIL: received data did not match expected pattern\n");
        return 1;
    }

    fprintf(stdout, "PASS: received %d packet(s) with correct data\n", receiver.count.load());
    return 0;
}
