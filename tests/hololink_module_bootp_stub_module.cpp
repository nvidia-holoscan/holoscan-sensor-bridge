/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Test fixture: a module .so that publishes both an
 * EnumerationInterfaceV1 (records what update_metadata received) and
 * a BootpCaptureV1 service the host gtest queries to observe the
 * recorded values. Used by the bootp end-to-end test to verify the
 * Adapter's listener path delivers the parsed metadata + raw packet
 * bytes through the V1 contract.
 */

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "hololink/module/enumeration.hpp"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/module.hpp"
#include "hololink/module/publisher.hpp"
#include "hololink/module/service_locator.h"

#include "hololink_module_bootp_capture.hpp"

class CaptureState {
public:
    void record(const hololink::module::EnumerationMetadata& metadata,
        const std::uint8_t* raw_packet, std::size_t raw_packet_len)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (raw_packet != nullptr) {
            raw_packet_.assign(raw_packet, raw_packet + raw_packet_len);
        }
        fpga_uuid_ = metadata.get<std::string>("fpga_uuid", std::string {});
        serial_number_ = metadata.get<std::string>("serial_number", std::string {});
        peer_ip_ = metadata.get<std::string>("peer_ip", std::string {});
        compat_id_ = metadata.get<int64_t>("compat_id", int64_t { -1 });
        data_plane_ = metadata.get<int64_t>("data_plane", int64_t { -1 });
        called_ = true;
        cv_.notify_all();
    }

    bool wait(unsigned timeout_ms)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
            [this] { return called_; });
    }

    std::vector<std::uint8_t> raw_packet() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return raw_packet_;
    }
    std::string fpga_uuid() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return fpga_uuid_;
    }
    std::string serial_number() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return serial_number_;
    }
    std::string peer_ip() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return peer_ip_;
    }
    std::int64_t compat_id() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return compat_id_;
    }
    std::int64_t data_plane() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return data_plane_;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool called_ = false;
    std::vector<std::uint8_t> raw_packet_;
    std::string fpga_uuid_;
    std::string serial_number_;
    std::string peer_ip_;
    std::int64_t compat_id_ = -1;
    std::int64_t data_plane_ = -1;
};

class StubEnumeration : public hololink::module::EnumerationInterfaceV1 {
public:
    explicit StubEnumeration(std::shared_ptr<CaptureState> state)
        : state_(std::move(state))
    {
    }

    hololink_module_status_t update_metadata(
        hololink::module::EnumerationMetadata& metadata,
        const std::uint8_t* raw_packet, std::size_t raw_packet_len) override
    {
        state_->record(metadata, raw_packet, raw_packet_len);
        return HOLOLINK_MODULE_OK;
    }

private:
    std::shared_ptr<CaptureState> state_;
};

class StubCapture : public test::BootpCaptureV1 {
public:
    explicit StubCapture(std::shared_ptr<CaptureState> state)
        : state_(std::move(state))
    {
    }

    bool wait_for_metadata(unsigned timeout_ms) override
    {
        return state_->wait(timeout_ms);
    }
    std::vector<std::uint8_t> raw_packet() const override { return state_->raw_packet(); }
    std::string fpga_uuid() const override { return state_->fpga_uuid(); }
    std::string serial_number() const override { return state_->serial_number(); }
    std::int64_t compat_id() const override { return state_->compat_id(); }
    std::int64_t data_plane() const override { return state_->data_plane(); }
    std::string peer_ip() const override { return state_->peer_ip(); }

private:
    std::shared_ptr<CaptureState> state_;
};

// Held alive for the lifetime of the loaded module .so.
static std::shared_ptr<hololink::module::Publisher> g_publisher;
static std::shared_ptr<CaptureState> g_state;
static std::shared_ptr<StubEnumeration> g_enumeration;
static std::shared_ptr<StubCapture> g_capture;

namespace {
class TestPublisher : public hololink::module::Publisher {
public:
    bool construct_service(
        const std::string& /*instance_id*/,
        const std::string& /*type_id*/) override
    {
        return false;
    }
};
} // namespace

extern "C" hololink_module_services_t
hololink_module_init(const hololink_module_init_t* init)
{
    using hololink::module::EnumerationInterfaceV1;
    using hololink::module::ServicePublisher;

    g_publisher = std::make_shared<TestPublisher>();

    auto result = g_publisher->setup(init);
    if (result.status != HOLOLINK_MODULE_OK) {
        return result;
    }

    g_state = std::make_shared<CaptureState>();
    g_enumeration = std::make_shared<StubEnumeration>(g_state);
    g_capture = std::make_shared<StubCapture>(g_state);

    ServicePublisher<EnumerationInterfaceV1>(g_publisher).publish("", g_enumeration);
    ServicePublisher<test::BootpCaptureV1>(g_publisher).publish("", g_capture);

    return result;
}
