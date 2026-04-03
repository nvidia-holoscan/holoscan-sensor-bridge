/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <hololink/common/cuda_error.hpp>
#include <hololink/operators/roce_transmitter/ibv.hpp>
#include <hololink/operators/roce_transmitter/roce_transmitter_op.hpp>

namespace hololink::tests {

static std::string get_device_ip(const std::string& device_name, int ib_port)
{
    // Get list of InfiniBand devices
    operators::ibv::DeviceList device_list;
    auto device = device_list.find(device_name);
    if (!device)
        return "";

    // Open the device context
    auto context = device->get_context();
    if (!context)
        return "";

    // Get the GID index for RoCE v2
    int gid_index = context.query_gid_roce_v2_ip();

    // Get the GID for RoCE v2
    ibv_gid gid = context.query_gid(ib_port, gid_index);
    return operators::ibv::gid_to_ipv4(gid);
}

struct RoceTestDeviceInfo {
    std::string ibv_name;
    uint32_t ibv_port;
    std::string host_ip;
};

static std::optional<RoceTestDeviceInfo> get_roce_test_device_info()
{
    operators::ibv::DeviceList device_list;
    if (device_list.size() == 0) {
        return std::nullopt;
    }
    std::string ibv_name = device_list.get(0).name();
    uint32_t ibv_port = 1;
    std::string host_ip = get_device_ip(ibv_name, ibv_port);
    if (host_ip.empty()) {
        return std::nullopt;
    }
    return RoceTestDeviceInfo { ibv_name, ibv_port, host_ip };
}

using Tensor = std::shared_ptr<holoscan::Tensor>;
using Data = std::vector<float>;
namespace ibv = hololink::operators::ibv;

// Tag parameter so the generator is not nullary: holoscan::Arg otherwise stores any nullary
// callable as std::function<void()> (Arg::set_value_ in holoscan arg.hpp).
struct DataGenTag { };

static Tensor create_tensor(void* context, gxf_uid_t allocator_uid, const Data& data)
{
    nvidia::gxf::Tensor gxf_tensor;
    auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context, allocator_uid);
    if (!gxf_tensor.reshape<float>(
            nvidia::gxf::Shape { static_cast<int>(data.size()) },
            nvidia::gxf::MemoryStorageType::kDevice,
            allocator.value()))
        throw std::runtime_error("Failed to allocate cuda memory");
    auto maybe_dl_ctx = gxf_tensor.toDLManagedTensorContext();
    auto tensor = std::make_shared<holoscan::Tensor>(*maybe_dl_ctx);
    cudaMemcpy(tensor->data(), &data.front(), data.size() * sizeof(float), cudaMemcpyHostToDevice);
    return tensor;
}

class DataGeneratorOp : public holoscan::Operator {
public:
    using DataGenerator = std::function<Data(DataGenTag)>;
    HOLOSCAN_OPERATOR_FORWARD_ARGS(DataGeneratorOp)
    DataGeneratorOp() = default;

    void setup(holoscan::OperatorSpec& spec) override
    {
        spec.output<Tensor>(output_name);

        // Register converters for arguments not defined by Holoscan
        register_converter<DataGenerator>();

        spec.param(data_generator_, "data_generator", "DataGenerator", "Data Generator");
    }

    void initialize() override
    {
        // Create an allocator for the operator
        allocator_ = fragment()->make_resource<holoscan::UnboundedAllocator>("pool");
        // Add the allocator to the operator so that it is initialized
        add_arg(allocator_);

        // Call the base class initialize function
        Operator::initialize();
    }

    void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
        holoscan::ExecutionContext& context) override
    {
        auto data = data_generator_.get()(DataGenTag {});
        auto tensor = create_tensor(context.context(), allocator_->gxf_cid(), data);
        op_output.emit(tensor);
    }

private:
    static inline const char* output_name = "output";
    std::shared_ptr<holoscan::Allocator> allocator_;
    holoscan::Parameter<DataGenerator> data_generator_;
};

class RoceReceiver {
public:
    using DataValidator = std::function<void(const Data&)>;
    RoceReceiver(const DataValidator& data_validator, size_t buffer_size, const std::string& ibv_name, uint32_t ibv_port, const std::string& remote_ip)
        : data_validator_(data_validator)
        , context_(ibv_name)
        , gid_index_(context_.query_gid_roce_v2_ip())
        // Allocate Protection Domain
        , protection_domain_(context_.allocate_protection_domain())
        // Create Completion Queue
        , completion_queue_(context_.create_completion_queue(10))
        , buffer_(buffer_size, 0)
        , memory_region_(protection_domain_.register_memory_region(
              &buffer_.front(),
              buffer_.size(),
              IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE))
        , queue_pair_(
              protection_domain_.create_queue_pair(
                  ibv::QueuePair::InitAttr().query_pair_type(IBV_QPT_UC).send_completion_queue(completion_queue_.ptr_).receive_completion_queue(completion_queue_.ptr_).capacity(ibv::QueuePair::Capacity().max_send_write_read(512).max_receive_write_read(512).max_send_scatter_gather_entry(1).max_receive_scallter_gather_entry(1))))
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
        if (!queue_pair_.reset_to_init(ibv_port_) || !queue_pair_.init_to_rtr(ibv_port_, remote_gid_, remote_qp_num, gid_index_) || !queue_pair_.rtr_to_rts())
            throw std::exception();

        // Prepare the receive work request
        ibv::ScatterGatherElement sge(
            reinterpret_cast<uint64_t>(&buffer_.front()),
            buffer_.size(),
            memory_region_.ptr_->lkey);

        // Send the receive work request
        ibv::ReceiveWorkRequest wr(1, &sge);
        queue_pair_.post_receive(wr);
    }

    bool validate(std::chrono::milliseconds timeout)
    {
        std::cout << "Validating data" << std::endl;
        auto start_time = std::chrono::steady_clock::now();
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

            // Check timeout if specified (timeout > 0)
            if (timeout.count() > 0) {
                auto current_time = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
                if (elapsed >= timeout) {
                    std::cout << "Timeout occurred while waiting for completion" << std::endl;
                    return false;
                }
            }

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
    std::vector<float> buffer_;
    ibv::MemoryRegion memory_region_;
    ibv::QueuePair queue_pair_;
    uint32_t ibv_port_;
    ::ibv_gid remote_gid_;
};

class RoceTransmitterTestApp : public holoscan::Application {
public:
    RoceTransmitterTestApp(size_t roce_transmitter_buffer_size,
        const DataGeneratorOp::DataGenerator& data_generator, const RoceReceiver::DataValidator& data_validator,
        const RoceTestDeviceInfo& device_info)
        : roce_transmitter_buffer_size_(roce_transmitter_buffer_size)
        , data_generator_(data_generator)
        , roce_receiver_(data_validator, roce_transmitter_buffer_size_, device_info.ibv_name, device_info.ibv_port, device_info.host_ip)
        , device_info_(device_info)
    {
    }

    void compose() override
    {
        data_generator_op_ = make_operator<DataGeneratorOp>(
            "DataGenerator",
            make_condition<holoscan::CountCondition>(1),
            holoscan::Arg("data_generator", data_generator_));
        roce_transmitter_op_ = make_operator<hololink::operators::RoceTransmitterOp>(
            "Roce Transmitter",
            holoscan::Arg("ibv_name", device_info_.ibv_name.c_str()),
            holoscan::Arg("ibv_port", device_info_.ibv_port),
            holoscan::Arg("hololink_ip", device_info_.host_ip.c_str()),
            holoscan::Arg("ibv_qp", roce_receiver_.get_qp_num()),
            holoscan::Arg("buffer_size", roce_transmitter_buffer_size_),
            holoscan::Arg("queue_size", static_cast<uint64_t>(1)),
            holoscan::Arg("on_start", hololink::operators::RoceTransmitterOp::OnStartCallback([this](const hololink::operators::RoceTransmitterOp::ConnectionInfo& info) {
                roce_receiver_.connect(info.qp_num);
            })),
            holoscan::Arg("on_stop", hololink::operators::RoceTransmitterOp::OnStopCallback([this](const hololink::operators::RoceTransmitterOp::ConnectionInfo& info) {
                roce_transmitter_op_->flush(std::chrono::milliseconds(1000));
            })));

        add_flow(data_generator_op_, roce_transmitter_op_, { { "output", "input" } });
    }

    void run() override
    {
        holoscan::Application::run();
        EXPECT_TRUE(roce_receiver_.validate(std::chrono::milliseconds(1000)));
    }

private:
    size_t roce_transmitter_buffer_size_;
    const DataGeneratorOp::DataGenerator& data_generator_;
    RoceReceiver roce_receiver_;
    RoceTestDeviceInfo device_info_;

    std::shared_ptr<DataGeneratorOp> data_generator_op_;
    std::shared_ptr<hololink::operators::RoceTransmitterOp> roce_transmitter_op_;
};

static Data generate_data(size_t size)
{
    Data d(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    std::generate(d.begin(), d.end(), [&]() { return dis(gen); });
    return d;
}

TEST(RoceTransmitterTest, float_8)
{
    auto device_info = get_roce_test_device_info();
    if (!device_info) {
        GTEST_SKIP() << "No RoCE-capable InfiniBand device available";
    }
    Data d(generate_data(8));
    RoceTransmitterTestApp app(
        d.size() * sizeof(Data::value_type),
        DataGeneratorOp::DataGenerator([&d](DataGenTag) {
            return d;
        }),
        RoceReceiver::DataValidator([&d](const Data& data) {
            EXPECT_TRUE(std::equal(d.begin(), d.begin() + d.size(), data.begin()));
        }),
        *device_info);
    app.run();
}

TEST(RoceTransmitterTest, float_4K)
{
    auto device_info = get_roce_test_device_info();
    if (!device_info) {
        GTEST_SKIP() << "No RoCE-capable InfiniBand device available";
    }
    Data d(generate_data(4 * 1024));
    RoceTransmitterTestApp app(
        d.size() * sizeof(Data::value_type),
        DataGeneratorOp::DataGenerator([&d](DataGenTag) {
            return d;
        }),
        RoceReceiver::DataValidator([&d](const Data& data) {
            EXPECT_TRUE(std::equal(d.begin(), d.begin() + d.size(), data.begin()));
        }),
        *device_info);
    app.run();
}

TEST(RoceTransmitterTest, float_1M)
{
    auto device_info = get_roce_test_device_info();
    if (!device_info) {
        GTEST_SKIP() << "No RoCE-capable InfiniBand device available";
    }
    Data d(generate_data(4 * 1024));
    RoceTransmitterTestApp app(
        d.size() * sizeof(Data::value_type),
        DataGeneratorOp::DataGenerator([&d](DataGenTag) {
            return d;
        }),
        RoceReceiver::DataValidator([&d](const Data& data) {
            EXPECT_TRUE(std::equal(d.begin(), d.begin() + d.size(), data.begin()));
        }),
        *device_info);
    app.run();
}

} // namespace hololink::tests

#define YAML_CONVERTER(TYPE)                                                                \
    template <>                                                                             \
    struct YAML::convert<TYPE> {                                                            \
        static Node encode(TYPE&) { throw std::runtime_error("Unsupported"); }              \
                                                                                            \
        static bool decode(const Node&, TYPE&) { throw std::runtime_error("Unsupported"); } \
    };

YAML_CONVERTER(hololink::tests::DataGeneratorOp::DataGenerator);
