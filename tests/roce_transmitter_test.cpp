/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <dlpack/dlpack.h>
#include <infiniband/verbs.h>
#include <unistd.h>

#include <holoscan/core/compile.hpp>
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

#include <hololink/operators/roce_transmitter/roce_transmitter_buffer.hpp>
#include <hololink/operators/roce_transmitter/roce_transmitter_op.hpp>
#include <hololink/transport/roce/ibv.hpp>

namespace hololink::tests {

namespace ibv = hololink::transport::roce::ibv;

// ---------------------------------------------------------------------------
// Hardware detection — skip tests when no RoCE device is present.
// ---------------------------------------------------------------------------

static std::string get_device_ip(const std::string& device_name, int ib_port)
{
    // Open the device context
    ibv::Context context(device_name);

    // Get the GID index for RoCE v2
    int gid_index = context.query_gid_roce_v2_ip();

    // Get the GID for RoCE v2
    ibv_gid gid = context.query_gid(ib_port, gid_index);
    return ibv::gid_to_ipv4(gid);
}

struct RoceTestDeviceInfo {
    std::string ibv_name;
    uint32_t ibv_port;
    std::string host_ip;
};

static std::optional<RoceTestDeviceInfo> get_roce_test_device_info()
{
    const std::optional<std::string> ibv_name = ibv::DeviceList().first_name();
    if (!ibv_name)
        return std::nullopt;
    uint32_t ibv_port = 1;
    std::string host_ip = get_device_ip(ibv_name.value(), ibv_port);
    if (host_ip.empty())
        return std::nullopt;
    return RoceTestDeviceInfo { ibv_name.value(), ibv_port, host_ip };
}

using Data = std::vector<float>;

// ---------------------------------------------------------------------------
// DataGeneratorOp — HSDK5 source operator that emits one float tensor.
// ---------------------------------------------------------------------------

// Largest payload across all test cases (float_1M = 1 M floats × 4 bytes).
// All DataGeneratorOp instances in the same process must declare the same
// max_byte_span to avoid HSDK5's GRAPH_PARAMETER_PORT_SHAPE_CHANGED error.
static constexpr std::size_t kMaxTestBytes = 1024ULL * 1024 * sizeof(float);

static constexpr DLDataType kFloatDtype { kDLFloat, 32, 1 };
static constexpr holoscan::TensorRepresentation kFloatDeviceFrame {
    .memory_kind = holoscan::MemoryKind::kCudaDevice,
    .dtype = kFloatDtype,
    .rank = 1U,
};

struct DataGeneratorParams {
    Data data {};
};

class DataGeneratorOp : public holoscan::Operator<DataGeneratorParams> {
public:
    // Public: graph wiring and contract() reference this by address.
    holoscan::Output<holoscan::Tensor> output;

    void setup(holoscan::OperatorSpec& spec) override
    {
        // All tests in the same process share one pool — see kMaxTestBytes.
        const std::size_t n_bytes = kMaxTestBytes;
        spec.output(output, "output")
            .produces_tensor(holoscan::TensorOutputSpec {
                .representation = kFloatDeviceFrame,
                .bounds = holoscan::TensorBounds {
                    .max_rank = 1U,
                    .max_byte_span = n_bytes,
                },
                .storage = holoscan::TensorOutputStorage::kRuntimePool,
            });
    }

    [[nodiscard]] holoscan::Contract contract() const override
    {
        holoscan::Contract c;
        c.trigger(holoscan::OnClock { .period = std::chrono::milliseconds(10) });
        return c;
    }

    void start() override { emitted_ = false; }

    [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
        holoscan::ExecutionContext& ctx) override
    {
        // Emit exactly once; subsequent invocations are no-ops until the
        // session is stopped by the test after the receiver confirms arrival.
        if (emitted_)
            return {};
        emitted_ = true;

        const Data& data = params().data;
        const int64_t n = static_cast<int64_t>(data.size());
        const std::array<int64_t, 1> shape { n };

        auto loan = output.allocate_tensor(
            holoscan::TensorLoanRequest { .shape = shape, .dtype = kFloatDtype });
        if (!loan)
            return holoscan::make_unexpected(std::move(loan).error());

        auto writer = loan->write(ctx.cuda_stream());
        if (!writer)
            return holoscan::make_unexpected(std::move(writer).error());

        if (cudaMemcpyAsync(writer->data(), data.data(), data.size() * sizeof(float),
                cudaMemcpyHostToDevice, ctx.cuda_stream().get())
            != cudaSuccess)
            return holoscan::make_unexpected(
                holoscan::Error { holoscan::ErrorCode::kFailure, "cudaMemcpyAsync H2D failed" });

        auto committed = std::move(*writer).commit();
        if (!committed)
            return holoscan::make_unexpected(std::move(committed).error());

        return output.emit(std::move(*loan),
            holoscan::EmitOptions { .capture_time = ctx.activation_time() });
    }

private:
    bool emitted_ { false };
};

// ---------------------------------------------------------------------------
// RoceReceiver — pure ibverbs loopback receiver (no HSDK dependency).
// Allocates a receive buffer, posts a receive WR, then polls for a completion
// and validates the received bytes against a caller-supplied callback.
// ---------------------------------------------------------------------------

class RoceReceiver {
public:
    using DataValidator = std::function<void(const Data&)>;

    // buffer_size is in bytes.
    RoceReceiver(const DataValidator& data_validator, size_t buffer_size,
        const std::string& ibv_name, uint32_t ibv_port, const std::string& remote_ip)
        : data_validator_(data_validator)
        , context_(ibv_name)
        , gid_index_(context_.query_gid_roce_v2_ip())
        // Allocate Protection Domain
        , protection_domain_(context_.allocate_protection_domain())
        // Create Completion Queue
        , completion_queue_(context_.create_completion_queue(10))
        , buffer_(buffer_size / sizeof(float))
        , memory_region_(protection_domain_.register_memory_region(
              &buffer_.front(),
              buffer_.size() * sizeof(float),
              IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE))
        , queue_pair_(
              protection_domain_.create_queue_pair(
                  ibv::QueuePair::InitAttr()
                      .query_pair_type(IBV_QPT_UC)
                      .send_completion_queue(completion_queue_.ptr_)
                      .receive_completion_queue(completion_queue_.ptr_)
                      .capacity(ibv::QueuePair::Capacity()
                                    .max_send_write_read(512)
                                    .max_receive_write_read(512)
                                    .max_send_scatter_gather_entry(1)
                                    .max_receive_scatter_gather_entry(1))))
        , ibv_port_(ibv_port)
        , remote_gid_(ibv::ipv4_to_gid(remote_ip.c_str()))
    {
    }

    uint32_t get_qp_num() const
    {
        return queue_pair_.ptr_->qp_num;
    }

    void connect(uint32_t remote_qp_num)
    {
        auto port_attr = context_.query_port(ibv_port_);
        auto host_if_name = context_.get_host_if_name();
        // Each transition returns 0 on success, an errno value otherwise.
        if (queue_pair_.reset_to_init(ibv_port_)
            || queue_pair_.init_to_rtr(ibv_port_, port_attr, host_if_name,
                remote_gid_, remote_qp_num, gid_index_)
            || queue_pair_.rtr_to_rts())
            throw std::runtime_error("RoceReceiver: QP connect failed");

        // Prepare the receive work request
        ibv::ScatterGatherElement sge(
            reinterpret_cast<uint64_t>(&buffer_.front()),
            buffer_.size() * sizeof(float),
            memory_region_.ptr_->lkey);

        // Send the receive work request
        ibv::ReceiveWorkRequest wr(1, &sge);
        if (queue_pair_.post_receive(wr))
            throw std::runtime_error("RoceReceiver: post_receive failed");
    }

    bool validate(std::chrono::milliseconds timeout)
    {
        std::cout << "Validating data" << std::endl;
        auto deadline = std::chrono::steady_clock::now() + timeout;
        // Wait for completion
        ibv_wc wc;
        while (true) {
            int ret = ibv_poll_cq(completion_queue_.ptr_, 1, &wc);
            if (ret < 0) {
                std::cout << "A failure occurred while trying to read Work Completions from the CQ" << std::endl;
                return false;
            }
            if (ret > 0) {
                if (wc.status != IBV_WC_SUCCESS) {
                    std::cout << "The completion request is not IBV_WC_SUCCESS" << std::endl;
                    return false;
                }
                // completion request is found
                break;
            }
            if (std::chrono::steady_clock::now() >= deadline)
                return false;
            // No completion yet, yield to other threads
            std::this_thread::yield();
        }
        // Validate the data
        data_validator_(buffer_);
        return true;
    }

private:
    DataValidator data_validator_;
    ibv::Context context_;
    int gid_index_;
    ibv::ProtectionDomain protection_domain_;
    ibv::CompletionQueue completion_queue_;
    Data buffer_;
    ibv::MemoryRegion memory_region_;
    ibv::QueuePair queue_pair_;
    uint32_t ibv_port_;
    ::ibv_gid remote_gid_;
};

// ---------------------------------------------------------------------------
// Test helper — builds the graph, runs it, waits for receiver confirmation.
// ---------------------------------------------------------------------------

static void run_roce_transmitter_test(
    const Data& data,
    const RoceReceiver::DataValidator& validator,
    const RoceTestDeviceInfo& device_info)
{
    const uint64_t data_bytes = data.size() * sizeof(float);

    RoceReceiver receiver(validator, data_bytes, device_info.ibv_name,
        device_info.ibv_port, device_info.host_ip);

    holoscan::Graph graph { "roce_tx_test" };

    auto gen = graph.op<DataGeneratorOp>("gen");
    gen.set_params(DataGeneratorParams { .data = data });

    auto tx = graph.op<hololink::operators::RoceTransmitterOp>("tx");
    tx.set_params(hololink::operators::RoceTransmitterParams {
        .ibv_name = device_info.ibv_name,
        .ibv_port = device_info.ibv_port,
        .hololink_ip = device_info.host_ip,
        .ibv_qp = receiver.get_qp_num(),
        .buffer_size = data_bytes,
        .queue_size = 1,
        // on_start: called once the TX QP is connected — connect the RX side
        .on_start = [&receiver](const hololink::operators::RoceTransmitterOp_ConnectionInfo& info) {
            receiver.connect(info.qp_num);
        },
    });

    graph.add_flow(gen->output, tx->input);
    graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

    // Declare the CUDA device hosting the tensor output pool.
    holoscan::CompileOptions options;
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement {
        .operator_path = "gen",
        .output_port = "output",
        .device = holoscan::DeviceId { 0 },
    });

    auto plan = holoscan::compile(graph, std::move(options));
    ASSERT_TRUE(plan.ok()) << "Graph compilation failed";

    auto session = holoscan::run_async(plan);

    // Wait up to 5 seconds for the receiver to confirm data arrival.
    EXPECT_TRUE(receiver.validate(std::chrono::milliseconds(5000)));

    session.request_stop();
    session.wait();
}

// ---------------------------------------------------------------------------
// Tests — each skips gracefully when no RoCE device is present.
// ---------------------------------------------------------------------------

static Data generate_data(size_t size)
{
    Data d(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    std::generate(d.begin(), d.end(), [&]() { return dis(gen); });
    return d;
}

TEST(RoceTransmitterAlignmentTest, HostPageAlignsDmaBufRegistration)
{
    const size_t page_size = static_cast<size_t>(getpagesize());
    const size_t payload_size = 8 * sizeof(float);
    const auto layout = operators::detail::dma_buf_layout(payload_size, page_size);

    ASSERT_LT(payload_size, page_size);
    EXPECT_EQ(layout.registered_size, page_size);
    EXPECT_EQ(layout.registered_size % page_size, 0);
    EXPECT_EQ(layout.allocation_size, layout.registered_size + page_size);

    // Add scratch space so the simulated CUDA allocation can be made deliberately
    // unaligned while retaining layout.allocation_size bytes after its base.
    std::vector<char> storage(layout.allocation_size + page_size);
    char* allocation = storage.data();
    if (reinterpret_cast<std::uintptr_t>(allocation) % page_size == 0) {
        ++allocation;
    }
    ASSERT_NE(reinterpret_cast<std::uintptr_t>(allocation) % page_size, 0);

    char* registered = operators::detail::page_aligned_pointer(allocation, page_size);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(registered) % page_size, 0);
    EXPECT_GE(registered, allocation);
    EXPECT_LT(registered, allocation + page_size);
    EXPECT_LE(registered + layout.registered_size, allocation + layout.allocation_size);
}

TEST(RoceTransmitterTest, float_8)
{
    auto device_info = get_roce_test_device_info();
    if (!device_info)
        GTEST_SKIP() << "No RoCE-capable InfiniBand device available";

    // Keep this payload smaller than a host page: the transmitter must page-align
    // its DMA-BUF registration without changing the number of bytes it sends.
    Data d = generate_data(8);
    run_roce_transmitter_test(
        d, [&d](const Data& received) {
            EXPECT_TRUE(std::equal(d.begin(), d.end(), received.begin()));
        },
        *device_info);
}

TEST(RoceTransmitterTest, float_4K)
{
    auto device_info = get_roce_test_device_info();
    if (!device_info)
        GTEST_SKIP() << "No RoCE-capable InfiniBand device available";

    Data d = generate_data(4 * 1024);
    run_roce_transmitter_test(
        d, [&d](const Data& received) {
            EXPECT_TRUE(std::equal(d.begin(), d.end(), received.begin()));
        },
        *device_info);
}

TEST(RoceTransmitterTest, float_1M)
{
    auto device_info = get_roce_test_device_info();
    if (!device_info)
        GTEST_SKIP() << "No RoCE-capable InfiniBand device available";

    Data d = generate_data(1024 * 1024);
    run_roce_transmitter_test(
        d, [&d](const Data& received) {
            EXPECT_TRUE(std::equal(d.begin(), d.end(), received.begin()));
        },
        *device_info);
}

} // namespace hololink::tests
