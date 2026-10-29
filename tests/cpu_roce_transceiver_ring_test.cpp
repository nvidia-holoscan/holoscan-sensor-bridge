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

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <future>
#include <limits>
#include <memory>
#include <optional>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

#include <arpa/inet.h>
#include <infiniband/verbs.h>

#include <hololink/transport/roce/cpu_roce_transceiver.hpp>
#include <hololink/transport/roce/cpu_roce_transceiver_internal.hpp>
#include <hololink/transport/roce/ibv.hpp>

namespace hololink::transport::roce {

namespace {

    CpuRoceTransceiver::Config host_config()
    {
        CpuRoceTransceiver::Config config;
        config.ibv_name = "unused-without-nic";
        config.rx.frame_layout.payload_size = 128;
        config.rx.stride = 256;
        config.rx.slot_count = 4;
        config.tx = config.rx;
        return config;
    }

    using RxCompletion = detail::RxCompletion;
    using RxCompletionHandler = detail::CpuRoceRxCompletionHandler;

    void expect_invalid_setup(CpuRoceTransceiver::Config config)
    {
        EXPECT_THROW(
            CpuRoceTransceiver(std::move(config)).setup(),
            std::invalid_argument);
    }

    struct PageAlignedBuffer {
        explicit PageAlignedBuffer(std::size_t size)
        {
            const long page = sysconf(_SC_PAGESIZE);
            alignment = page > 0 ? static_cast<std::size_t>(page) : 4096;
            void* allocated = nullptr;
            if (posix_memalign(&allocated, alignment, size) != 0) {
                throw std::runtime_error("posix_memalign failed");
            }
            data = static_cast<std::uint8_t*>(allocated);
            bytes = size;
        }

        ~PageAlignedBuffer()
        {
            std::free(data);
        }

        PageAlignedBuffer(const PageAlignedBuffer&) = delete;
        PageAlignedBuffer& operator=(const PageAlignedBuffer&) = delete;

        std::uint8_t* data = nullptr;
        std::size_t bytes = 0;
        std::size_t alignment = 0;
    };

    struct LocalRoceDevice {
        std::string name;
        std::uint32_t port;
        std::string ip;
    };

    bool environment_flag(const char* name)
    {
        const char* value = std::getenv(name);
        return value != nullptr && value[0] != '\0'
            && std::string(value) != "0";
    }

    std::optional<LocalRoceDevice> find_local_roce_device()
    {
        int device_count = 0;
        ::ibv_device** raw_devices = ibv_get_device_list(&device_count);
        if (raw_devices == nullptr) {
            return std::nullopt;
        }
        struct DeviceListGuard {
            ~DeviceListGuard() { ibv_free_device_list(devices); }
            ::ibv_device** devices;
        } guard { raw_devices };
        for (int i = 0; i < device_count; ++i) {
            const std::string name = ibv_get_device_name(raw_devices[i]);
            try {
                ibv::Context context(name);
                ::ibv_device_attr attributes {};
                if (ibv_query_device(context.ptr_, &attributes) != 0) {
                    continue;
                }
                for (std::uint32_t port = 1; port <= attributes.phys_port_cnt; ++port) {
                    try {
                        const auto gid = detail::query_local_roce_v2_ipv4_gid(
                            context, port);
                        char ip[INET_ADDRSTRLEN] {};
                        if (inet_ntop(
                                AF_INET, gid.gid.raw + 12, ip, sizeof(ip))
                            == nullptr) {
                            continue;
                        }
                        return LocalRoceDevice {
                            name, port, ip
                        };
                    } catch (const ibv::Error&) {
                        // This port has no usable IPv4 RoCE v2 GID. Keep scanning.
                    }
                }
            } catch (const ibv::Error&) {
                // This device cannot be opened or queried. Keep scanning.
            }
        }
        return std::nullopt;
    }

    struct LocalRoceProbe {
        std::optional<LocalRoceDevice> device;
        std::string error;
    };

    LocalRoceProbe probe_local_roce_device()
    {
        try {
            return LocalRoceProbe { find_local_roce_device(), {} };
        } catch (const ibv::Error& error) {
            return LocalRoceProbe { std::nullopt, error.what() };
        }
    }

    std::string missing_roce_message(const LocalRoceProbe& probe)
    {
        return probe.error.empty()
            ? std::string("No IPv4 RoCE v2 device available")
            : probe.error;
    }

    struct ConfiguredRoceEndpoints {
        LocalRoceDevice left;
        LocalRoceDevice right;
    };

    std::optional<ConfiguredRoceEndpoints> configured_roce_endpoints()
    {
        const char* left_device
            = std::getenv("HOLOLINK_CPU_ROCE_LEFT_DEVICE");
        const char* left_port
            = std::getenv("HOLOLINK_CPU_ROCE_LEFT_PORT");
        const char* left_ip = std::getenv("HOLOLINK_CPU_ROCE_LEFT_IP");
        const char* right_device
            = std::getenv("HOLOLINK_CPU_ROCE_RIGHT_DEVICE");
        const char* right_port
            = std::getenv("HOLOLINK_CPU_ROCE_RIGHT_PORT");
        const char* right_ip = std::getenv("HOLOLINK_CPU_ROCE_RIGHT_IP");

        if (left_device == nullptr && left_port == nullptr && left_ip == nullptr
            && right_device == nullptr && right_port == nullptr
            && right_ip == nullptr) {
            return std::nullopt;
        }
        if (left_device == nullptr || left_port == nullptr || left_ip == nullptr
            || right_device == nullptr || right_port == nullptr
            || right_ip == nullptr) {
            throw std::invalid_argument(
                "Two-endpoint RoCE test requires LEFT/RIGHT DEVICE, PORT, and IP");
        }

        const auto parse_port = [](const char* text) {
            std::size_t consumed = 0;
            const auto value = std::stoul(text, &consumed);
            if (text[consumed] != '\0'
                || value > std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument(
                    "Two-endpoint RoCE test port is invalid");
            }
            return static_cast<std::uint32_t>(value);
        };
        ConfiguredRoceEndpoints endpoints {
            { left_device, parse_port(left_port), left_ip },
            { right_device, parse_port(right_port), right_ip },
        };
        if (endpoints.left.name == endpoints.right.name
            && endpoints.left.port == endpoints.right.port
            && endpoints.left.ip == endpoints.right.ip) {
            throw std::invalid_argument(
                "Two-endpoint RoCE test requires distinct endpoint descriptions");
        }
        return endpoints;
    }

    struct RxHandlerHardwarePair {
        std::unique_ptr<CpuRoceTransceiver> subject;
        std::unique_ptr<CpuRoceTransceiver> peer;
    };

    RxHandlerHardwarePair make_rx_handler_datapath_hardware_pair(
        const ConfiguredRoceEndpoints& endpoints,
        CpuRoceTransceiver::TxOperation operation,
        std::uint32_t rx_slot_count,
        std::uint32_t tx_slot_count,
        std::size_t payload_size,
        RxCompletionHandler::Fn handler,
        void* handler_context)
    {
        auto subject_config = host_config();
        subject_config.mode = CpuRoceTransceiver::Mode::Rx;
        subject_config.ibv_name = endpoints.right.name;
        subject_config.ibv_port = endpoints.right.port;
        subject_config.local_ip = endpoints.right.ip;
        subject_config.rx.frame_layout.payload_size = payload_size;
        subject_config.rx.slot_count = rx_slot_count;

        auto peer_config = host_config();
        peer_config.mode = CpuRoceTransceiver::Mode::Tx;
        peer_config.tx_operation = operation;
        peer_config.ibv_name = endpoints.left.name;
        peer_config.ibv_port = endpoints.left.port;
        peer_config.local_ip = endpoints.left.ip;
        peer_config.tx.frame_layout.payload_size = payload_size;
        peer_config.tx.slot_count = tx_slot_count;

        RxHandlerHardwarePair pair {
            std::make_unique<CpuRoceTransceiver>(
                std::move(subject_config)),
            std::make_unique<CpuRoceTransceiver>(std::move(peer_config))
        };
        // Credit count matches the old one-WQE-per-slot depth so a frame past
        // that depth still proves the consumed credit was reposted.
        RxCompletionHandler(*pair.subject)
            .enable_handled_receive(handler, handler_context, rx_slot_count);
        pair.subject->setup();
        pair.peer->setup();

        CpuRoceTransceiver::Peer to_subject;
        to_subject.qp = pair.subject->get_qp_number();
        to_subject.ip = endpoints.right.ip;
        to_subject.rx_base = pair.subject->external_frame_memory();
        to_subject.rx_rkey = pair.subject->get_rkey();
        to_subject.rx_stride = pair.subject->get_rx_ring_stride();
        to_subject.rx_slot_count
            = pair.subject->get_rx_ring_slot_count();
        CpuRoceTransceiver::Peer to_peer;
        to_peer.qp = pair.peer->get_qp_number();
        to_peer.ip = endpoints.left.ip;
        pair.peer->connect(std::move(to_subject));
        pair.subject->connect(std::move(to_peer));
        return pair;
    }

    TEST(CpuRoceTransceiverRingTest, GeometryGettersAreZeroBeforeSetup)
    {
        CpuRoceTransceiver transceiver(host_config());
        EXPECT_EQ(transceiver.get_rx_ring_stride(), 0);
        EXPECT_EQ(transceiver.get_rx_ring_slot_count(), 0);
        EXPECT_EQ(transceiver.get_tx_ring_stride(), 0);
        EXPECT_EQ(transceiver.get_tx_ring_slot_count(), 0);
        EXPECT_EQ(transceiver.get_rx_ring_data_addr(), nullptr);
        EXPECT_EQ(transceiver.get_tx_ring_data_addr(), nullptr);
    }

    TEST(CpuRoceTransceiverRingTest, RoleRingsAllocateSymmetricZeroedStorage)
    {
        detail::RxRing rx(TransceiverAllocation {}, 256, 4);
        detail::TxRing tx(TransceiverAllocation {}, 256, 4);

        ASSERT_NE(rx.data(), nullptr);
        ASSERT_NE(tx.data(), nullptr);
        EXPECT_NE(rx.data(), tx.data());
        EXPECT_EQ(rx.stride(), 256);
        EXPECT_EQ(tx.stride(), 256);
        EXPECT_EQ(rx.slot_count(), 4);
        EXPECT_EQ(tx.slot_count(), 4);

        for (std::uint32_t slot = 0; slot < 4; ++slot) {
            EXPECT_EQ(
                CpuRoceTransceiver::load_flag_acquire(&rx.flags()[slot]), 0);
            EXPECT_EQ(
                CpuRoceTransceiver::load_flag_acquire(&tx.flags()[slot]), 0);
        }
    }

    TEST(CpuRoceTransceiverRingTest, ValidationAcceptsMaximumHsbImmSlotCount)
    {
        auto config = host_config();
        config.rx.slot_count = 4096;
        config.tx.slot_count = 4096;
        EXPECT_NO_THROW(detail::validate_config(config));
    }

    TEST(CpuRoceTransceiverRingTest, ValidationAcceptsIndependentRingGeometry)
    {
        auto config = host_config();
        config.rx.stride = 4096;
        config.rx.slot_count = 8;
        config.rx.frame_layout.payload_size = 3072;
        config.tx.stride = 128;
        config.tx.slot_count = 4;
        config.tx.frame_layout.payload_size = 64;

        EXPECT_NO_THROW(detail::validate_config(config));
    }

    TEST(CpuRoceTransceiverRingTest, ValidationAcceptsNonPowerOfTwoRxSlotCount)
    {
        auto rx_only = host_config();
        rx_only.mode = CpuRoceTransceiver::Mode::Rx;
        rx_only.rx.slot_count = 3;
        EXPECT_NO_THROW(detail::validate_config(rx_only));

        auto forward = host_config();
        forward.mode = CpuRoceTransceiver::Mode::Forward;
        forward.rx.slot_count = 3;
        EXPECT_NO_THROW(detail::validate_config(forward));

        auto duplex = host_config();
        duplex.mode = CpuRoceTransceiver::Mode::Duplex;
        duplex.rx.slot_count = 3;
        duplex.tx.slot_count = 4;
        EXPECT_NO_THROW(detail::validate_config(duplex));
    }

    TEST(CpuRoceTransceiverRingTest, ValidationAcceptsPollCpuForSinglePollerModes)
    {
        for (const auto mode : {
                 CpuRoceTransceiver::Mode::Rx,
                 CpuRoceTransceiver::Mode::Tx,
                 CpuRoceTransceiver::Mode::Forward,
                 CpuRoceTransceiver::Mode::Unified,
             }) {
            auto config = host_config();
            config.mode = mode;
            config.poll_cpu = 0;
            EXPECT_NO_THROW(detail::validate_config(config));
        }
    }

    TEST(CpuRoceTransceiverRingTest, ValidationRejectsPollCpuForDuplex)
    {
        auto config = host_config();
        config.poll_cpu = 0;
        config.poll_cpu_best_effort = true;
        expect_invalid_setup(std::move(config));
    }

    TEST(CpuRoceTransceiverRingTest, ValidationRejectsOutOfRangePollCpu)
    {
        for (const int cpu : { -1, CPU_SETSIZE }) {
            auto config = host_config();
            config.mode = CpuRoceTransceiver::Mode::Forward;
            config.poll_cpu = cpu;
            expect_invalid_setup(std::move(config));
        }
    }

    TEST(CpuRoceTransceiverRingTest, OptionalPollCpuAcceptsAnOutOfRangeCpu)
    {
        for (const auto mode : {
                 CpuRoceTransceiver::Mode::Rx,
                 CpuRoceTransceiver::Mode::Tx,
                 CpuRoceTransceiver::Mode::Forward,
                 CpuRoceTransceiver::Mode::Unified,
             }) {
            auto config = host_config();
            config.mode = mode;
            config.poll_cpu = CPU_SETSIZE;
            config.poll_cpu_best_effort = true;
            EXPECT_NO_THROW(detail::validate_config(config));
        }
    }

    bool same_affinity(const cpu_set_t& left, const cpu_set_t& right)
    {
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (CPU_ISSET(cpu, &left) != CPU_ISSET(cpu, &right)) {
                return false;
            }
        }
        return true;
    }

    TEST(CpuRoceTransceiverRingTest, OptionalPollCpuLeavesTheMaskWhenThePinFails)
    {
        cpu_set_t before;
        CPU_ZERO(&before);
        ASSERT_EQ(::sched_getaffinity(0, sizeof before, &before), 0);

        EXPECT_NO_THROW(detail::ThreadAffinityScope(CPU_SETSIZE, true));
        cpu_set_t after;
        CPU_ZERO(&after);
        ASSERT_EQ(::sched_getaffinity(0, sizeof after, &after), 0);
        EXPECT_TRUE(same_affinity(before, after));

        EXPECT_THROW(
            detail::ThreadAffinityScope(CPU_SETSIZE, false), std::runtime_error);
        CPU_ZERO(&after);
        ASSERT_EQ(::sched_getaffinity(0, sizeof after, &after), 0);
        EXPECT_TRUE(same_affinity(before, after));
    }

    TEST(CpuRoceTransceiverRingTest, ValidationRejectsNonPowerOfTwoTxAndUnified)
    {
        auto tx_only = host_config();
        tx_only.mode = CpuRoceTransceiver::Mode::Tx;
        tx_only.tx.slot_count = 3;
        expect_invalid_setup(std::move(tx_only));

        auto duplex = host_config();
        duplex.mode = CpuRoceTransceiver::Mode::Duplex;
        duplex.tx.slot_count = 3;
        expect_invalid_setup(std::move(duplex));

        auto unified_equal = host_config();
        unified_equal.mode = CpuRoceTransceiver::Mode::Unified;
        unified_equal.rx.slot_count = 3;
        unified_equal.tx.slot_count = 3;
        EXPECT_THROW(
            detail::validate_config(unified_equal), std::invalid_argument);
    }

    TEST(CpuRoceTransceiverRingTest, CallerOwnedMemoryIsUsedWithoutTakingOwnership)
    {
        auto config = host_config();
        std::vector<std::uint8_t> rx(
            config.rx.slot_count * config.rx.stride);
        std::vector<std::uint8_t> tx(
            config.tx.slot_count * config.tx.stride);
        config.rx.payload_memory = HostMemory { rx.data(), rx.size() };
        config.tx.payload_memory = HostMemory { tx.data(), tx.size() };

        {
            detail::RxRing rx_ring(
                config.rx.payload_memory, config.rx.stride, config.rx.slot_count);
            detail::TxRing tx_ring(
                config.tx.payload_memory, config.tx.stride, config.tx.slot_count);
            EXPECT_EQ(rx_ring.data(), rx.data());
            EXPECT_EQ(tx_ring.data(), tx.data());
        }

        rx[0] = 1;
        tx[0] = 2;
        EXPECT_EQ(rx[0], 1);
        EXPECT_EQ(tx[0], 2);
    }

    TEST(CpuRoceTransceiverRingTest, ValidationRejectsOverlappingHostRings)
    {
        const auto modes = {
            CpuRoceTransceiver::Mode::Duplex,
            CpuRoceTransceiver::Mode::Unified,
            CpuRoceTransceiver::Mode::CallerDriven,
        };
        for (const auto mode : modes) {
            auto config = host_config();
            const std::size_t ring_bytes
                = config.rx.slot_count * config.rx.stride;
            PageAlignedBuffer storage(ring_bytes + config.rx.stride);
            config.mode = mode;
            config.rx.payload_memory = HostMemory { storage.data, ring_bytes };
            config.tx.payload_memory
                = HostMemory { storage.data + config.rx.stride, ring_bytes };

            expect_invalid_setup(std::move(config));
        }
    }

    TEST(CpuRoceTransceiverRingTest, ValidationAcceptsAdjacentUnequalHostRings)
    {
        auto config = host_config();
        config.mode = CpuRoceTransceiver::Mode::Duplex;
        config.rx.stride = 4096;
        config.rx.slot_count = 2;
        config.rx.frame_layout.payload_size = 2048;
        config.tx.stride = 4096;
        config.tx.slot_count = 1;
        config.tx.frame_layout.payload_size = 64;
        const std::size_t rx_bytes
            = config.rx.slot_count * config.rx.stride;
        const std::size_t tx_bytes
            = config.tx.slot_count * config.tx.stride;
        PageAlignedBuffer storage(rx_bytes + tx_bytes);
        config.rx.payload_memory = HostMemory { storage.data, rx_bytes };
        config.tx.payload_memory = HostMemory { storage.data + rx_bytes, tx_bytes };

        EXPECT_NO_THROW(detail::validate_config(config));
    }

    TEST(CpuRoceTransceiverRingTest, ValidationRejectsShortTxOverlappingRxTail)
    {
        auto config = host_config();
        config.mode = CpuRoceTransceiver::Mode::Duplex;
        config.rx.stride = 4096;
        config.rx.slot_count = 2;
        config.rx.frame_layout.payload_size = 2048;
        config.tx.stride = 256;
        config.tx.slot_count = 1;
        config.tx.frame_layout.payload_size = 64;
        const std::size_t rx_bytes
            = config.rx.slot_count * config.rx.stride;
        const std::size_t tx_bytes
            = config.tx.slot_count * config.tx.stride;
        PageAlignedBuffer storage(rx_bytes + tx_bytes);
        config.rx.payload_memory = HostMemory { storage.data, rx_bytes };
        // Short TX starts inside the last RX slot so the overlap is only
        // RX's tail. Using the TX length for both ranges would miss it.
        config.tx.payload_memory = HostMemory {
            storage.data + rx_bytes - (tx_bytes / 2), tx_bytes
        };

        expect_invalid_setup(std::move(config));
    }

    TEST(CpuRoceTransceiverRingTest, ValidationRejectsShortRxOverlappingTxTail)
    {
        auto config = host_config();
        const long page = sysconf(_SC_PAGESIZE);
        const std::size_t page_size
            = page > 0 ? static_cast<std::size_t>(page) : 4096;
        config.tx.stride = page_size;
        config.tx.slot_count = 2;
        config.tx.frame_layout.payload_size = 64;
        config.rx.stride = page_size;
        config.rx.slot_count = 1;
        config.rx.frame_layout.payload_size = 64;
        const std::size_t tx_bytes
            = config.tx.slot_count * config.tx.stride;
        const std::size_t rx_bytes
            = config.rx.slot_count * config.rx.stride;
        PageAlignedBuffer storage(tx_bytes);
        config.tx.payload_memory = HostMemory { storage.data, tx_bytes };
        // Short RX starts one OS page after the base, so it is aligned and
        // occupies only TX's tail. Using the RX length for both ranges would
        // treat them as adjacent.
        config.rx.payload_memory
            = HostMemory { storage.data + page_size, rx_bytes };

        config.mode = CpuRoceTransceiver::Mode::Rx;
        EXPECT_NO_THROW(detail::validate_config(config));
        config.mode = CpuRoceTransceiver::Mode::Duplex;
        expect_invalid_setup(std::move(config));
    }

    TEST(CpuRoceTransceiverRingTest, ValidationAllowsOverlappingHostRingsWhenModeUsesOnePayload)
    {
        const auto modes = {
            CpuRoceTransceiver::Mode::Rx,
            CpuRoceTransceiver::Mode::Tx,
            CpuRoceTransceiver::Mode::Forward,
        };
        for (const auto mode : modes) {
            auto config = host_config();
            const std::size_t ring_bytes
                = config.rx.slot_count * config.rx.stride;
            PageAlignedBuffer storage(ring_bytes + config.rx.stride);
            config.mode = mode;
            config.rx.payload_memory = HostMemory { storage.data, ring_bytes };
            config.tx.payload_memory
                = HostMemory { storage.data + config.rx.stride, ring_bytes };

            EXPECT_NO_THROW(detail::validate_config(config));
        }
    }

    TEST(CpuRoceTransceiverRingTest, ValidationIgnoresUnusedRingConfig)
    {
        auto rx_only = host_config();
        rx_only.mode = CpuRoceTransceiver::Mode::Rx;
        rx_only.tx = RingConfig {};
        rx_only.tx.slot_count = 3;
        rx_only.tx.payload_memory = DmaBufMemory {};
        EXPECT_NO_THROW(detail::validate_config(rx_only));

        auto forward = host_config();
        forward.mode = CpuRoceTransceiver::Mode::Forward;
        forward.tx = RingConfig {};
        EXPECT_NO_THROW(detail::validate_config(forward));

        auto tx_only = host_config();
        tx_only.mode = CpuRoceTransceiver::Mode::Tx;
        tx_only.rx = RingConfig {};
        tx_only.rx.payload_memory = HostMemory { nullptr, 1 };
        EXPECT_NO_THROW(detail::validate_config(tx_only));

        auto duplex = host_config();
        duplex.mode = CpuRoceTransceiver::Mode::Duplex;
        duplex.tx = RingConfig {};
        EXPECT_THROW(
            detail::validate_config(duplex), std::invalid_argument);
    }

    TEST(CpuRoceTransceiverRingTest, ReceivePathMtu4096IsRxOnly)
    {
        auto rx = host_config();
        rx.mode = CpuRoceTransceiver::Mode::Rx;
        rx.rx_path_mtu_4096 = true;
        EXPECT_NO_THROW(detail::validate_config(rx));

        auto duplex = host_config();
        duplex.rx_path_mtu_4096 = true;
        EXPECT_THROW(detail::validate_config(duplex), std::invalid_argument);
    }

    TEST(CpuRoceTransceiverRingTest, UnifiedRequiresDispatchBeforeSetup)
    {
        auto config = host_config();
        config.mode = CpuRoceTransceiver::Mode::Unified;
        CpuRoceTransceiver transceiver(std::move(config));

        EXPECT_THROW(transceiver.setup(), std::invalid_argument);
        transceiver.close();
        EXPECT_THROW(transceiver.setup(), std::logic_error);
    }

    TEST(CpuRoceTransceiverRingTest, UnifiedRequiresEqualRingDepths)
    {
        auto config = host_config();
        config.mode = CpuRoceTransceiver::Mode::Unified;
        config.tx.slot_count = config.rx.slot_count * 2;

        EXPECT_THROW(detail::validate_config(config), std::invalid_argument);
    }

    TEST(CpuRoceTransceiverRingTest, CallerDrivenMatchesUnifiedRingRules)
    {
        auto unequal = host_config();
        unequal.mode = CpuRoceTransceiver::Mode::CallerDriven;
        unequal.tx.slot_count = unequal.rx.slot_count * 2;
        EXPECT_THROW(detail::validate_config(unequal), std::invalid_argument);

        auto pinned = host_config();
        pinned.mode = CpuRoceTransceiver::Mode::CallerDriven;
        pinned.poll_cpu = 0;
        EXPECT_THROW(detail::validate_config(pinned), std::invalid_argument);

        auto dmabuf = host_config();
        dmabuf.mode = CpuRoceTransceiver::Mode::CallerDriven;
        dmabuf.rx.payload_memory
            = DmaBufMemory { reinterpret_cast<void*>(0x100000), 10, 0, 1024 };
        EXPECT_THROW(detail::validate_config(dmabuf), std::invalid_argument);

        auto valid = host_config();
        valid.mode = CpuRoceTransceiver::Mode::CallerDriven;
        EXPECT_NO_THROW(detail::validate_config(valid));
    }

    TEST(CpuRoceTransceiverRingTest, CallerDrivenPollIsRejectedOutsideItsMode)
    {
        CpuRoceTransceiver transceiver(host_config());
        std::uint32_t slot = 0;
        EXPECT_THROW(transceiver.rx_poll(&slot), std::logic_error);
        EXPECT_THROW(transceiver.tx_publish(0), std::logic_error);
        EXPECT_THROW(transceiver.rx_poll(nullptr), std::invalid_argument);
    }

    TEST(CpuRoceTransceiverRingTest, CallerDrivenRequiresConnectBeforePoll)
    {
        auto config = host_config();
        config.mode = CpuRoceTransceiver::Mode::CallerDriven;
        CpuRoceTransceiver transceiver(std::move(config));
        std::uint32_t slot = 0;

        EXPECT_THROW(transceiver.blocking_monitor(), std::logic_error);
        // Hooks are the spin path. Before connect() they return false
        // instead of throwing on every iteration.
        EXPECT_FALSE(transceiver.rx_poll(&slot));
        EXPECT_FALSE(transceiver.tx_publish(0));
        // No dispatch callback is required, unlike Unified. Without a device,
        // setup() fails at device open, before a QP exists. A missing
        // callback would be invalid_argument. The second call checks that
        // this failure is not reported as success.
        for (int attempt = 0; attempt < 2; ++attempt) {
            try {
                transceiver.setup();
                FAIL() << "setup without a RoCE device should fail";
            } catch (const std::invalid_argument& error) {
                FAIL() << "CallerDriven setup must not require a dispatch "
                          "callback: "
                       << error.what();
            } catch (const std::runtime_error&) {
            }
        }

        transceiver.close();
        EXPECT_FALSE(transceiver.rx_poll(&slot));
        EXPECT_FALSE(transceiver.tx_publish(0));
        EXPECT_THROW(transceiver.blocking_monitor(), std::logic_error);
        transceiver.close();
    }

    TEST(CpuRoceTransceiverRingTest, CallerDrivenRequestExitKeepsMonitorUsable)
    {
        auto config = host_config();
        config.mode = CpuRoceTransceiver::Mode::CallerDriven;
        CpuRoceTransceiver transceiver(std::move(config));
        transceiver.request_exit();
        // Unconnected, so this still throws, but with the precondition
        // error, not the after-close() one that a set exit flag would cause.
        try {
            transceiver.blocking_monitor();
            FAIL() << "blocking_monitor() before connect() should throw";
        } catch (const std::logic_error& error) {
            EXPECT_NE(std::string(error.what()).find("must be connected"),
                std::string::npos)
                << error.what();
        }
    }

    TEST(CpuRoceTransceiverRingTest, UnifiedAllowsDistinctRingGeometry)
    {
        auto config = host_config();
        config.mode = CpuRoceTransceiver::Mode::Unified;
        config.rx.stride = 4096;
        config.rx.frame_layout.payload_size = 2048;
        config.tx.stride = 128;
        config.tx.frame_layout.payload_size = 64;

        EXPECT_NO_THROW(detail::validate_config(config));
    }

    TEST(CpuRoceTransceiverRingTest, ValidationRejectsInvalidOrUnsupportedConfiguration)
    {
        auto invalid_mode = host_config();
        invalid_mode.mode = static_cast<CpuRoceTransceiver::Mode>(-1);
        expect_invalid_setup(std::move(invalid_mode));

        auto invalid_tx_operation = host_config();
        invalid_tx_operation.tx_operation
            = static_cast<CpuRoceTransceiver::TxOperation>(-1);
        expect_invalid_setup(std::move(invalid_tx_operation));

        auto zero_slots = host_config();
        zero_slots.rx.slot_count = 0;
        expect_invalid_setup(std::move(zero_slots));

        auto not_power_of_two = host_config();
        not_power_of_two.tx.slot_count = 3;
        expect_invalid_setup(std::move(not_power_of_two));

        auto one_duplex_cpu = host_config();
        one_duplex_cpu.rx_cpu = 0;
        expect_invalid_setup(std::move(one_duplex_cpu));

        auto affinity_on_rx = host_config();
        affinity_on_rx.mode = CpuRoceTransceiver::Mode::Rx;
        affinity_on_rx.rx_cpu = 0;
        affinity_on_rx.tx_cpu = 1;
        expect_invalid_setup(std::move(affinity_on_rx));

        auto invalid_duplex_cpu = host_config();
        invalid_duplex_cpu.rx_cpu = -1;
        invalid_duplex_cpu.tx_cpu = 1;
        expect_invalid_setup(std::move(invalid_duplex_cpu));

        auto shared_duplex_cpu = host_config();
        shared_duplex_cpu.rx_cpu = 1;
        shared_duplex_cpu.tx_cpu = 1;
        expect_invalid_setup(std::move(shared_duplex_cpu));

        auto too_many_slots = host_config();
        too_many_slots.rx.slot_count = 8192;
        expect_invalid_setup(std::move(too_many_slots));

        auto null_host = host_config();
        null_host.rx.payload_memory = HostMemory {
            nullptr, null_host.rx.slot_count * null_host.rx.stride
        };
        expect_invalid_setup(std::move(null_host));

        auto undersized_host = host_config();
        std::vector<std::uint8_t> too_small(undersized_host.rx.stride);
        undersized_host.rx.payload_memory
            = HostMemory { too_small.data(), too_small.size() };
        expect_invalid_setup(std::move(undersized_host));

        if constexpr (std::numeric_limits<std::size_t>::max()
            > std::numeric_limits<std::uint32_t>::max()) {
            auto stride_too_large = host_config();
            stride_too_large.tx.stride
                = static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) + 1;
            expect_invalid_setup(std::move(stride_too_large));
        }
    }

    TEST(CpuRoceTransceiverRingTest, ValidationAcceptsRxMetadata)
    {
        auto config = host_config();
        config.mode = CpuRoceTransceiver::Mode::Rx;
        config.rx.frame_layout.metadata_offset = 128;
        config.rx.frame_layout.metadata_size = 128;

        EXPECT_NO_THROW(detail::validate_config(config));

        config.mode = CpuRoceTransceiver::Mode::Forward;
        EXPECT_NO_THROW(detail::validate_config(config));

        config.mode = CpuRoceTransceiver::Mode::Duplex;
        EXPECT_NO_THROW(detail::validate_config(config));
    }

    TEST(CpuRoceTransceiverRingTest, ValidationRejectsInvalidMetadataGeometry)
    {
        auto outside = host_config();
        outside.rx.frame_layout.metadata_offset = 192;
        outside.rx.frame_layout.metadata_size = 128;
        expect_invalid_setup(std::move(outside));

        auto overlaps_payload = host_config();
        overlaps_payload.rx.frame_layout.metadata_offset = 64;
        overlaps_payload.rx.frame_layout.metadata_size = 128;
        expect_invalid_setup(std::move(overlaps_payload));

        auto overflowing = host_config();
        overflowing.rx.frame_layout.metadata_offset
            = std::numeric_limits<std::size_t>::max() - 63;
        overflowing.rx.frame_layout.metadata_size = 128;
        expect_invalid_setup(std::move(overflowing));
    }

    TEST(CpuRoceTransceiverRingTest, ValidationRejectsTxMetadata)
    {
        auto config = host_config();
        config.mode = CpuRoceTransceiver::Mode::Tx;
        config.tx.frame_layout.metadata_offset = 128;
        config.tx.frame_layout.metadata_size = 128;

        expect_invalid_setup(config);

        config.tx.frame_layout.metadata_offset = 64;
        try {
            detail::validate_config(config);
            FAIL() << "expected TX metadata to be rejected";
        } catch (const std::invalid_argument& error) {
            const std::string what { error.what() };
            EXPECT_NE(what.find("TX frame metadata"), std::string::npos)
                << what;
            EXPECT_EQ(what.find("RX metadata"), std::string::npos) << what;
        }
    }

    // Fabricated fds and addresses: layout and zero-IOVA arithmetic only. Live
    // ibv_reg_dmabuf_mr, GPUDirect DMA, CUDA ordering, and zero-SGE credits
    // are CX/GPU qualification, not these tests.
    TEST(CpuRoceTransceiverRingTest, ValidationAcceptsDmaBufMemory)
    {
        auto config = host_config();
        config.mode = CpuRoceTransceiver::Mode::Duplex;
        config.rx.payload_memory
            = DmaBufMemory { reinterpret_cast<void*>(0x100000), 10, 0, 1024 };
        config.tx.payload_memory
            = DmaBufMemory { reinterpret_cast<void*>(0x200000), 11, 0, 1024 };

        EXPECT_NO_THROW(detail::validate_config(config));
    }

    TEST(CpuRoceTransceiverRingTest, ValidationAcceptsDmaBufRxMetadata)
    {
        auto config = host_config();
        config.mode = CpuRoceTransceiver::Mode::Rx;
        config.rx.payload_memory
            = DmaBufMemory { reinterpret_cast<void*>(0x100000), 10, 0, 1024 };
        config.rx.frame_layout.metadata_offset = 128;
        config.rx.frame_layout.metadata_size = 128;

        EXPECT_NO_THROW(detail::validate_config(config));
    }

    TEST(CpuRoceTransceiverRingTest, ValidationRejectsDmaBufInUnified)
    {
        auto rx_dmabuf = host_config();
        rx_dmabuf.mode = CpuRoceTransceiver::Mode::Unified;
        rx_dmabuf.rx.payload_memory
            = DmaBufMemory { reinterpret_cast<void*>(0x100000), 10, 0, 1024 };
        expect_invalid_setup(std::move(rx_dmabuf));

        auto tx_dmabuf = host_config();
        tx_dmabuf.mode = CpuRoceTransceiver::Mode::Unified;
        tx_dmabuf.tx.payload_memory
            = DmaBufMemory { reinterpret_cast<void*>(0x200000), 11, 0, 1024 };
        expect_invalid_setup(std::move(tx_dmabuf));
    }

    TEST(CpuRoceTransceiverRingTest, ValidationRejectsInvalidDmaBufMemory)
    {
        auto null_address = host_config();
        null_address.mode = CpuRoceTransceiver::Mode::Rx;
        null_address.rx.payload_memory
            = DmaBufMemory { nullptr, 10, 0, 1024 };
        expect_invalid_setup(std::move(null_address));

        auto invalid_fd = host_config();
        invalid_fd.mode = CpuRoceTransceiver::Mode::Rx;
        invalid_fd.rx.payload_memory = DmaBufMemory {
            reinterpret_cast<void*>(0x100000), -1, 0, 1024
        };
        expect_invalid_setup(std::move(invalid_fd));

        auto short_span = host_config();
        short_span.mode = CpuRoceTransceiver::Mode::Rx;
        short_span.rx.payload_memory = DmaBufMemory {
            reinterpret_cast<void*>(0x100000), 10, 0, 1023
        };
        expect_invalid_setup(std::move(short_span));

        auto unaligned_offset = host_config();
        unaligned_offset.mode = CpuRoceTransceiver::Mode::Rx;
        unaligned_offset.rx.payload_memory = DmaBufMemory {
            reinterpret_cast<void*>(0x100000), 10, 128, 1024
        };
        expect_invalid_setup(std::move(unaligned_offset));
    }

    TEST(CpuRoceTransceiverRingTest, ValidationRejectsOverlappingDmaBufRings)
    {
        auto config = host_config();
        config.mode = CpuRoceTransceiver::Mode::Duplex;
        config.rx.payload_memory
            = DmaBufMemory { reinterpret_cast<void*>(0x100000), 10, 0, 2048 };
        config.tx.payload_memory
            = DmaBufMemory { reinterpret_cast<void*>(0x200000), 10, 0, 2048 };

        expect_invalid_setup(std::move(config));
    }

    TEST(CpuRoceTransceiverRingTest, ValidationAcceptsGpuPointerMemory)
    {
        auto config = host_config();
        config.mode = CpuRoceTransceiver::Mode::Duplex;
        config.rx.payload_memory = GpuPointerMemory {
            reinterpret_cast<void*>(0x100000), 1024
        };
        config.tx.payload_memory = GpuPointerMemory {
            reinterpret_cast<void*>(0x200000), 1024
        };

        EXPECT_NO_THROW(detail::validate_config(config));
    }

    TEST(CpuRoceTransceiverRingTest, ValidationAcceptsUnalignedDeviceTxPointer)
    {
        auto config = host_config();
        config.mode = CpuRoceTransceiver::Mode::Tx;
        config.tx.payload_memory = GpuPointerMemory {
            reinterpret_cast<void*>(0x100080), 1024
        };

        EXPECT_NO_THROW(detail::validate_config(config));
    }

    TEST(CpuRoceTransceiverRingTest, ValidationRejectsGpuPointerMemoryInUnified)
    {
        auto rx_device = host_config();
        rx_device.mode = CpuRoceTransceiver::Mode::Unified;
        rx_device.rx.payload_memory = GpuPointerMemory {
            reinterpret_cast<void*>(0x100000), 1024
        };
        expect_invalid_setup(std::move(rx_device));

        auto tx_device = host_config();
        tx_device.mode = CpuRoceTransceiver::Mode::Unified;
        tx_device.tx.payload_memory = GpuPointerMemory {
            reinterpret_cast<void*>(0x200000), 1024
        };
        expect_invalid_setup(std::move(tx_device));
    }

    TEST(CpuRoceTransceiverRingTest, ValidationRejectsInvalidGpuPointerMemory)
    {
        auto null_address = host_config();
        null_address.mode = CpuRoceTransceiver::Mode::Rx;
        null_address.rx.payload_memory = GpuPointerMemory { nullptr, 1024 };
        expect_invalid_setup(std::move(null_address));

        auto short_span = host_config();
        short_span.mode = CpuRoceTransceiver::Mode::Rx;
        short_span.rx.payload_memory = GpuPointerMemory {
            reinterpret_cast<void*>(0x100000), 1023
        };
        expect_invalid_setup(std::move(short_span));

        auto unaligned = host_config();
        unaligned.mode = CpuRoceTransceiver::Mode::Rx;
        unaligned.rx.payload_memory = GpuPointerMemory {
            reinterpret_cast<void*>(0x100080), 1024
        };
        expect_invalid_setup(std::move(unaligned));
    }

    TEST(CpuRoceTransceiverRingTest, ValidationRejectsOverlappingDeviceRings)
    {
        auto config = host_config();
        config.mode = CpuRoceTransceiver::Mode::Duplex;
        config.rx.payload_memory = GpuPointerMemory {
            reinterpret_cast<void*>(0x200000), 2048
        };
        config.tx.payload_memory = GpuPointerMemory {
            reinterpret_cast<void*>(0x200200), 2048
        };

        expect_invalid_setup(std::move(config));
    }

    TEST(CpuRoceTransceiverRingTest, ValidationRejectsInvalidVerbsIdentifiers)
    {
        auto empty_device = host_config();
        empty_device.ibv_name.clear();
        expect_invalid_setup(std::move(empty_device));

        for (const std::uint32_t port : { 0u, 256u, 257u }) {
            auto invalid_port = host_config();
            invalid_port.ibv_port = port;
            expect_invalid_setup(std::move(invalid_port));
        }
        auto maximum_port = host_config();
        maximum_port.ibv_port = 255;
        EXPECT_NO_THROW(detail::validate_config(maximum_port));

        auto invalid_local_ip = host_config();
        invalid_local_ip.local_ip = "not-an-ipv4-address";
        expect_invalid_setup(std::move(invalid_local_ip));
    }

    TEST(CpuRoceTransceiverRingTest, SetLocalIpValidatesBeforeSetup)
    {
        CpuRoceTransceiver transceiver(host_config());
        EXPECT_THROW(
            transceiver.set_local_ip("not-an-ipv4-address"),
            std::invalid_argument);
        EXPECT_NO_THROW(transceiver.set_local_ip("192.0.2.1"));
        // Empty restores the default first-IPv4-RoCE-v2-GID selection.
        EXPECT_NO_THROW(transceiver.set_local_ip(""));
    }

    TEST(CpuRoceTransceiverRingTest, SetLocalIpAfterCloseIsLogicError)
    {
        CpuRoceTransceiver transceiver(host_config());
        transceiver.close();
        EXPECT_THROW(
            transceiver.set_local_ip("192.0.2.1"), std::logic_error);
    }

    TEST(CpuRoceTransceiverRingTest, ValidationRejectsUnalignedBorrowedRxHostMemory)
    {
        auto config = host_config();
        const std::size_t ring_bytes
            = config.rx.slot_count * config.rx.stride;
        PageAlignedBuffer storage(ring_bytes + 1);
        config.rx.payload_memory = HostMemory { storage.data + 1, ring_bytes };
        expect_invalid_setup(std::move(config));
    }

    TEST(CpuRoceTransceiverRingTest, ValidationAcceptsUnalignedBorrowedTxHostMemory)
    {
        auto config = host_config();
        const std::size_t ring_bytes
            = config.tx.slot_count * config.tx.stride;
        PageAlignedBuffer storage(ring_bytes + 1);
        config.tx.payload_memory = HostMemory { storage.data + 1, ring_bytes };
        EXPECT_NO_THROW(detail::validate_config(config));
    }

    TEST(CpuRoceTransceiverRingTest, PeerValidationChecksWireFieldWidths)
    {
        auto config = host_config();
        CpuRoceTransceiver::Peer peer;
        peer.ip = "192.0.2.1";
        peer.qp = 1;
        peer.rx_stride = config.tx.frame_layout.payload_size;
        EXPECT_NO_THROW(detail::validate_peer(config, peer));

        peer.qp = 0;
        EXPECT_THROW(detail::validate_peer(config, peer), std::invalid_argument);

        peer.qp = 0x01000000;
        EXPECT_THROW(detail::validate_peer(config, peer), std::invalid_argument);

        peer.qp = 0x00FFFFFF;
        EXPECT_NO_THROW(detail::validate_peer(config, peer));

        peer.ip = "not-an-ipv4-address";
        EXPECT_THROW(detail::validate_peer(config, peer), std::invalid_argument);
    }

    TEST(CpuRoceTransceiverRingTest, PeerValidationChecksSendGeometry)
    {
        auto config = host_config();
        config.mode = CpuRoceTransceiver::Mode::Tx;
        CpuRoceTransceiver::Peer peer;
        peer.ip = "192.0.2.1";
        peer.qp = 1;
        peer.rx_stride = config.tx.stride;
        EXPECT_NO_THROW(detail::validate_peer(config, peer));

        // RX and TX frame layouts are independent. Only the outbound TX
        // payload constrains an ordinary Send to this peer.
        config.rx.frame_layout.payload_size = config.tx.frame_layout.payload_size + 64;

        // The contract is rx_stride >= payload_size rather than equality with
        // the local stride, so a peer may pad or align its slots differently.
        peer.rx_stride = config.tx.frame_layout.payload_size;
        EXPECT_NO_THROW(detail::validate_peer(config, peer));

        peer.rx_stride = config.tx.frame_layout.payload_size - 1;
        EXPECT_THROW(detail::validate_peer(config, peer), std::invalid_argument);

        peer.rx_stride = config.tx.frame_layout.payload_size;
        peer.rx_slot_count = 1;
        EXPECT_LT(peer.rx_slot_count, config.tx.slot_count);
        EXPECT_NO_THROW(detail::validate_peer(config, peer));
    }

    TEST(CpuRoceTransceiverRingTest, PeerValidationChecksWriteWithImmGeometry)
    {
        auto config = host_config();
        config.mode = CpuRoceTransceiver::Mode::Tx;
        config.tx_operation = CpuRoceTransceiver::TxOperation::WriteWithImm;
        CpuRoceTransceiver::Peer peer;
        peer.ip = "192.0.2.1";
        peer.qp = 1;
        peer.rx_rkey = 1;
        peer.rx_stride = config.tx.frame_layout.payload_size;
        peer.rx_slot_count = config.tx.slot_count;
        EXPECT_NO_THROW(detail::validate_peer(config, peer));

        peer.rx_slot_count = 0;
        EXPECT_THROW(detail::validate_peer(config, peer), std::invalid_argument);

        peer.rx_slot_count = config.tx.slot_count;
        config.tx.slot_count = 8;
        peer.rx_slot_count = 4;
        EXPECT_THROW(detail::validate_peer(config, peer), std::invalid_argument);

        config.tx.slot_count = 4;
        peer.rx_slot_count = 8;
        EXPECT_NO_THROW(detail::validate_peer(config, peer));

        peer.rx_stride = config.tx.frame_layout.payload_size - 1;
        EXPECT_THROW(detail::validate_peer(config, peer), std::invalid_argument);

        peer.rx_stride = config.tx.frame_layout.payload_size;
        peer.rx_slot_count = 1;
        config.tx.slot_count = 1;
        peer.rx_base
            = std::numeric_limits<std::uint64_t>::max()
            - (config.tx.frame_layout.payload_size - 1);
        EXPECT_NO_THROW(detail::validate_peer(config, peer));

        peer.rx_base = std::numeric_limits<std::uint64_t>::max();
        EXPECT_THROW(detail::validate_peer(config, peer), std::invalid_argument);
    }

    TEST(CpuRoceTransceiverRingTest, ForwardUsesRxGeometryForPeerValidation)
    {
        auto config = host_config();
        config.mode = CpuRoceTransceiver::Mode::Forward;
        config.tx_operation = CpuRoceTransceiver::TxOperation::Send;
        config.rx.frame_layout.payload_size = 64;
        config.tx.frame_layout.payload_size = 128;

        CpuRoceTransceiver::Peer peer;
        peer.ip = "192.0.2.1";
        peer.qp = 1;
        peer.rx_stride = config.rx.frame_layout.payload_size;
        EXPECT_NO_THROW(detail::validate_peer(config, peer));

        --peer.rx_stride;
        EXPECT_THROW(detail::validate_peer(config, peer), std::invalid_argument);

        config.tx_operation = CpuRoceTransceiver::TxOperation::WriteWithImm;
        peer.rx_stride = config.rx.frame_layout.payload_size;
        peer.rx_rkey = 1;
        peer.rx_slot_count = config.rx.slot_count;
        EXPECT_NO_THROW(detail::validate_peer(config, peer));

        --peer.rx_slot_count;
        EXPECT_THROW(detail::validate_peer(config, peer), std::invalid_argument);
    }

    TEST(CpuRoceTransceiverRingTest, UnifiedUsesTxGeometryForPeerValidation)
    {
        auto config = host_config();
        config.mode = CpuRoceTransceiver::Mode::Unified;
        config.rx.frame_layout.payload_size = 128;
        config.tx.frame_layout.payload_size = 64;

        CpuRoceTransceiver::Peer peer;
        peer.ip = "192.0.2.1";
        peer.qp = 1;
        peer.rx_stride = config.tx.frame_layout.payload_size;
        EXPECT_NO_THROW(detail::validate_peer(config, peer));

        --peer.rx_stride;
        EXPECT_THROW(detail::validate_peer(config, peer), std::invalid_argument);

        config.tx_operation = CpuRoceTransceiver::TxOperation::WriteWithImm;
        peer.rx_stride = config.tx.frame_layout.payload_size;
        peer.rx_rkey = 1;
        peer.rx_slot_count = config.tx.slot_count;
        EXPECT_NO_THROW(detail::validate_peer(config, peer));
    }

    TEST(CpuRoceTransceiverRingTest, PeerValidationIsModeAndOperationAware)
    {
        auto config = host_config();
        CpuRoceTransceiver::Peer peer;
        peer.ip = "192.0.2.1";

        config.mode = CpuRoceTransceiver::Mode::Rx;
        peer.qp = 0;
        EXPECT_NO_THROW(detail::validate_peer(config, peer));

        config.mode = CpuRoceTransceiver::Mode::Duplex;
        config.tx_operation = CpuRoceTransceiver::TxOperation::WriteWithImm;
        peer.qp = 1;
        EXPECT_THROW(detail::validate_peer(config, peer), std::invalid_argument);

        peer.rx_rkey = 1;
        peer.rx_stride = config.tx.frame_layout.payload_size;
        peer.rx_slot_count = config.tx.slot_count;
        EXPECT_NO_THROW(detail::validate_peer(config, peer));

        config.mode = CpuRoceTransceiver::Mode::Tx;
        EXPECT_NO_THROW(detail::validate_peer(config, peer));
    }

    TEST(CpuRoceTransceiverRingTest, SetupAfterCloseIsRejected)
    {
        auto config = host_config();
        CpuRoceTransceiver transceiver(std::move(config));
        transceiver.close();
        transceiver.close();
        EXPECT_THROW(transceiver.setup(), std::logic_error);
        EXPECT_EQ(transceiver.get_rx_ring_data_addr(), nullptr);
        EXPECT_EQ(transceiver.get_tx_ring_flag_addr(), nullptr);
        EXPECT_EQ(transceiver.get_rx_ring_stride(), 0);
        EXPECT_EQ(transceiver.get_rx_ring_slot_count(), 0);
        EXPECT_EQ(transceiver.get_tx_ring_stride(), 0);
        EXPECT_EQ(transceiver.get_tx_ring_slot_count(), 0);
    }

    TEST(CpuRoceTransceiverRxCompletionHandlerTest,
        EnableHandledReceiveIsRxOnlyNonNullAndOneShot)
    {
        const auto handler = +[](void*, RxCompletion) {};
        for (const auto mode : {
                 CpuRoceTransceiver::Mode::Tx,
                 CpuRoceTransceiver::Mode::Duplex,
                 CpuRoceTransceiver::Mode::Forward,
                 CpuRoceTransceiver::Mode::Unified,
             }) {
            auto config = host_config();
            config.mode = mode;
            CpuRoceTransceiver transceiver(std::move(config));
            EXPECT_THROW(
                RxCompletionHandler(transceiver).enable_handled_receive(handler, nullptr, 1),
                std::logic_error);
        }

        auto config = host_config();
        config.mode = CpuRoceTransceiver::Mode::Rx;
        CpuRoceTransceiver transceiver(std::move(config));
        RxCompletionHandler seam(transceiver);
        EXPECT_THROW(seam.enable_handled_receive(nullptr, nullptr, 1), std::invalid_argument);
        EXPECT_THROW(seam.enable_handled_receive(handler, nullptr, 0), std::invalid_argument);
        EXPECT_THROW(seam.enable_handled_receive(handler, nullptr, 4097), std::invalid_argument);
        EXPECT_NO_THROW(seam.enable_handled_receive(handler, &transceiver, 1));
        EXPECT_THROW(seam.enable_handled_receive(handler, nullptr, 1), std::logic_error);

        auto closed_config = host_config();
        closed_config.mode = CpuRoceTransceiver::Mode::Rx;
        CpuRoceTransceiver closed(std::move(closed_config));
        closed.close();
        EXPECT_THROW(
            RxCompletionHandler(closed).enable_handled_receive(handler, nullptr, 1),
            std::logic_error);
    }

    TEST(CpuRoceTransceiverRingTest, ConnectBeforeSetupIsRejected)
    {
        CpuRoceTransceiver transceiver(host_config());

        EXPECT_THROW(
            transceiver.connect(CpuRoceTransceiver::Peer {}), std::logic_error);
        EXPECT_THROW(transceiver.blocking_monitor(), std::logic_error);
    }

    TEST(CpuRoceTransceiverRingTest,
        BlockingMonitorBeforeConnectDoesNotApplyPollCpu)
    {
        auto config = host_config();
        config.mode = CpuRoceTransceiver::Mode::Forward;
        // Out of range: ThreadAffinityScope would throw std::runtime_error.
        // A lifecycle error here means affinity was never reached.
        config.poll_cpu = CPU_SETSIZE;
        CpuRoceTransceiver transceiver(std::move(config));
        try {
            transceiver.blocking_monitor();
            FAIL() << "blocking_monitor should have thrown";
        } catch (const std::logic_error&) {
        } catch (const std::exception& error) {
            FAIL() << "expected lifecycle std::logic_error before affinity; got "
                   << error.what();
        }
    }

    TEST(CpuRoceTransceiverRingTest, SetupCreatesAndClosesLocalVerbsResources)
    {
        const bool required
            = environment_flag("HOLOLINK_CPU_ROCE_HARDWARE_REQUIRED");
        const auto probe = probe_local_roce_device();
        if (!probe.device) {
            if (required) {
                FAIL() << missing_roce_message(probe);
            }
            GTEST_SKIP() << missing_roce_message(probe);
        }
        const auto& device = *probe.device;

        auto config = host_config();
        config.ibv_name = device.name;
        config.ibv_port = device.port;
        // Prove that the post-construction override, rather than the original
        // Config value, is used by setup() for local GID selection.
        config.local_ip = "not-an-ipv4-address";
        CpuRoceTransceiver transceiver(std::move(config));
        transceiver.set_local_ip(device.ip);

        transceiver.setup();
        const auto qp_number = transceiver.get_qp_number();
        const auto rkey = transceiver.get_rkey();
        EXPECT_NE(qp_number, 0);
        EXPECT_NE(rkey, 0);
        EXPECT_EQ(transceiver.external_frame_memory(), 0);
        EXPECT_NE(transceiver.get_rx_ring_data_addr(), nullptr);
        EXPECT_NE(transceiver.get_tx_ring_data_addr(), nullptr);
        const auto dispatch
            = +[](void*, const void*, std::size_t, void*, std::size_t) {
                  return CpuRoceTransceiver::UnifiedDispatchResult {
                      CpuRoceTransceiver::UnifiedDispatchDisposition::Drop, 0
                  };
              };
        EXPECT_THROW(
            transceiver.set_unified_dispatch(dispatch, nullptr),
            std::logic_error);
        EXPECT_THROW(transceiver.set_local_ip(device.ip), std::logic_error);

        transceiver.setup();
        EXPECT_EQ(transceiver.get_qp_number(), qp_number);
        EXPECT_EQ(transceiver.get_rkey(), rkey);

        transceiver.close();
        transceiver.close();
        EXPECT_EQ(transceiver.get_qp_number(), 0);
        EXPECT_EQ(transceiver.get_rkey(), 0);
        EXPECT_EQ(transceiver.get_rx_ring_data_addr(), nullptr);
        EXPECT_EQ(transceiver.get_tx_ring_data_addr(), nullptr);

        auto rx_config = host_config();
        rx_config.mode = CpuRoceTransceiver::Mode::Rx;
        rx_config.ibv_name = device.name;
        rx_config.ibv_port = device.port;
        rx_config.local_ip = device.ip;
        CpuRoceTransceiver rx(std::move(rx_config));
        rx.setup();
        EXPECT_THROW(
            RxCompletionHandler(rx).enable_handled_receive(
                +[](void*, RxCompletion) {}, nullptr, 1),
            std::logic_error);
        rx.close();
    }

    TEST(CpuRoceTransceiverRingTest, SetupAllocatesPayloadRingsByMode)
    {
        const bool required
            = environment_flag("HOLOLINK_CPU_ROCE_HARDWARE_REQUIRED");
        const auto probe = probe_local_roce_device();
        if (!probe.device) {
            if (required) {
                FAIL() << missing_roce_message(probe);
            }
            GTEST_SKIP() << missing_roce_message(probe);
        }
        const auto& device = *probe.device;

        for (const auto mode :
            { CpuRoceTransceiver::Mode::Rx, CpuRoceTransceiver::Mode::Tx,
                CpuRoceTransceiver::Mode::Duplex,
                CpuRoceTransceiver::Mode::Forward,
                CpuRoceTransceiver::Mode::Unified }) {
            auto config = host_config();
            config.mode = mode;
            config.ibv_name = device.name;
            config.ibv_port = device.port;
            config.local_ip = device.ip;
            if (mode == CpuRoceTransceiver::Mode::Rx
                || mode == CpuRoceTransceiver::Mode::Forward) {
                config.tx = RingConfig {};
            } else if (mode == CpuRoceTransceiver::Mode::Tx) {
                config.rx = RingConfig {};
            }
            CpuRoceTransceiver transceiver(std::move(config));
            if (mode == CpuRoceTransceiver::Mode::Unified) {
                transceiver.set_unified_dispatch(
                    +[](void*, const void*, std::size_t, void*, std::size_t) {
                        return CpuRoceTransceiver::UnifiedDispatchResult {
                            CpuRoceTransceiver::UnifiedDispatchDisposition::Drop,
                            0
                        };
                    },
                    nullptr);
            }

            transceiver.setup();

            const bool has_rx = mode != CpuRoceTransceiver::Mode::Tx;
            const bool has_tx = mode == CpuRoceTransceiver::Mode::Tx
                || mode == CpuRoceTransceiver::Mode::Duplex
                || mode == CpuRoceTransceiver::Mode::Unified;
            EXPECT_EQ(
                transceiver.get_rx_ring_data_addr() != nullptr, has_rx);
            EXPECT_EQ(
                transceiver.get_tx_ring_data_addr() != nullptr, has_tx);
            EXPECT_EQ(transceiver.get_rkey() != 0, has_rx);
        }
    }

    TEST(CpuRoceTransceiverRingTest, ConnectsConfiguredTwoEndpoints)
    {
        const bool required
            = environment_flag("HOLOLINK_CPU_ROCE_HARDWARE_REQUIRED");
        const auto endpoints = configured_roce_endpoints();
        if (!endpoints) {
            if (required) {
                FAIL()
                    << "HOLOLINK_CPU_ROCE_HARDWARE_REQUIRED needs configured "
                       "HOLOLINK_CPU_ROCE_{LEFT,RIGHT}_{DEVICE,PORT,IP} and "
                       "a static neighbor entry between them for same-host two-NIC "
                       "(otherwise RTR ETIMEDOUT)";
            }
            GTEST_SKIP()
                << "Set HOLOLINK_CPU_ROCE_{LEFT,RIGHT}_{DEVICE,PORT,IP}; for "
                   "two NICs on one host add "
                   "a static neighbor entry between them first "
                   "(otherwise RTR ETIMEDOUT)";
        }

        auto left_config = host_config();
        left_config.ibv_name = endpoints->left.name;
        left_config.ibv_port = endpoints->left.port;
        left_config.local_ip = endpoints->left.ip;
        auto right_config = host_config();
        right_config.ibv_name = endpoints->right.name;
        right_config.ibv_port = endpoints->right.port;
        right_config.local_ip = endpoints->right.ip;

        CpuRoceTransceiver left(std::move(left_config));
        CpuRoceTransceiver right(std::move(right_config));
        left.setup();
        right.setup();

        CpuRoceTransceiver::Peer to_right;
        to_right.qp = right.get_qp_number();
        to_right.ip = endpoints->right.ip;
        to_right.rx_base = right.external_frame_memory();
        to_right.rx_rkey = right.get_rkey();
        to_right.rx_stride = right.get_rx_ring_stride();
        CpuRoceTransceiver::Peer to_left;
        to_left.qp = left.get_qp_number();
        to_left.ip = endpoints->left.ip;
        to_left.rx_base = left.external_frame_memory();
        to_left.rx_rkey = left.get_rkey();
        to_left.rx_stride = left.get_rx_ring_stride();

        EXPECT_EQ(left.get_path_mtu(), 0u);
        EXPECT_EQ(right.get_path_mtu(), 0u);
        ASSERT_NO_THROW(left.connect(to_right));
        ASSERT_NO_THROW(right.connect(to_left));
        const auto path_mtu_ok = [](std::uint32_t mtu) {
            return mtu == 256 || mtu == 512 || mtu == 1024 || mtu == 2048 || mtu == 4096;
        };
        EXPECT_TRUE(path_mtu_ok(left.get_path_mtu()));
        EXPECT_TRUE(path_mtu_ok(right.get_path_mtu()));

        left.close();
        right.close();
        EXPECT_EQ(left.get_qp_number(), 0);
        EXPECT_EQ(right.get_qp_number(), 0);
        EXPECT_EQ(left.get_rkey(), 0);
        EXPECT_EQ(right.get_rkey(), 0);
        EXPECT_EQ(left.get_path_mtu(), 0u);
        EXPECT_EQ(right.get_path_mtu(), 0u);
    }

    TEST(CpuRoceTransceiverRingTest, BlockingMonitorStopsOnCloseForActiveModes)
    {
        const bool required
            = environment_flag("HOLOLINK_CPU_ROCE_HARDWARE_REQUIRED");
        const auto endpoints = configured_roce_endpoints();
        if (!endpoints) {
            if (required) {
                FAIL()
                    << "HOLOLINK_CPU_ROCE_HARDWARE_REQUIRED needs configured "
                       "HOLOLINK_CPU_ROCE_{LEFT,RIGHT}_{DEVICE,PORT,IP} and "
                       "a static neighbor entry between them for same-host two-NIC "
                       "(otherwise RTR ETIMEDOUT)";
            }
            GTEST_SKIP()
                << "Set HOLOLINK_CPU_ROCE_{LEFT,RIGHT}_{DEVICE,PORT,IP}; for "
                   "two NICs on one host add "
                   "a static neighbor entry between them first "
                   "(otherwise RTR ETIMEDOUT)";
        }

        for (const auto mode :
            { CpuRoceTransceiver::Mode::Rx, CpuRoceTransceiver::Mode::Tx,
                CpuRoceTransceiver::Mode::Duplex,
                CpuRoceTransceiver::Mode::Forward,
                CpuRoceTransceiver::Mode::Unified }) {
            SCOPED_TRACE(static_cast<int>(mode));
            auto left_config = host_config();
            left_config.mode = mode;
            left_config.ibv_name = endpoints->left.name;
            left_config.ibv_port = endpoints->left.port;
            left_config.local_ip = endpoints->left.ip;
            auto right_config = host_config();
            right_config.mode = CpuRoceTransceiver::Mode::Rx;
            right_config.ibv_name = endpoints->right.name;
            right_config.ibv_port = endpoints->right.port;
            right_config.local_ip = endpoints->right.ip;

            CpuRoceTransceiver left(std::move(left_config));
            CpuRoceTransceiver right(std::move(right_config));
            if (mode == CpuRoceTransceiver::Mode::Unified) {
                left.set_unified_dispatch(
                    +[](void*, const void*, std::size_t, void*, std::size_t) {
                        return CpuRoceTransceiver::UnifiedDispatchResult {
                            CpuRoceTransceiver::UnifiedDispatchDisposition::Drop,
                            0
                        };
                    },
                    nullptr);
            }
            left.setup();
            right.setup();

            CpuRoceTransceiver::Peer to_right;
            to_right.qp = right.get_qp_number();
            to_right.ip = endpoints->right.ip;
            to_right.rx_stride = right.get_rx_ring_stride();
            CpuRoceTransceiver::Peer to_left;
            to_left.qp = left.get_qp_number();
            to_left.ip = endpoints->left.ip;
            to_left.rx_stride = left.get_rx_ring_stride();
            ASSERT_NO_THROW(left.connect(to_right));
            ASSERT_NO_THROW(right.connect(to_left));

            auto monitor = std::async(std::launch::async, [&left] {
                left.blocking_monitor();
            });
            if (!detail::CpuRoceMonitorTestAccess::wait_until_active(
                    left, 5000)) {
                left.close();
                right.close();
                FAIL()
                    << "blocking_monitor() never set monitor_active; close() "
                       "would race the prologue";
            }

            auto closer = std::async(std::launch::async, [&left, &right] {
                left.close();
                right.close();
            });
            ASSERT_EQ(
                closer.wait_for(std::chrono::seconds(5)),
                std::future_status::ready)
                << "close() deadlocked waiting for blocking_monitor()";
            closer.get();

            ASSERT_EQ(
                monitor.wait_for(std::chrono::seconds(5)),
                std::future_status::ready)
                << "blocking_monitor hung after close()";
            EXPECT_NO_THROW(monitor.get());
        }
    }

    TEST(CpuRoceTransceiverRxCompletionHandlerHardwareTest,
        WriteWithImmPlacesByImmAndRestoresReceiveCredit)
    {
        const bool required
            = environment_flag("HOLOLINK_CPU_ROCE_HARDWARE_REQUIRED");
        const auto endpoints = configured_roce_endpoints();
        if (!endpoints) {
            if (required) {
                FAIL()
                    << "HOLOLINK_CPU_ROCE_HARDWARE_REQUIRED needs configured "
                       "HOLOLINK_CPU_ROCE_{LEFT,RIGHT}_{DEVICE,PORT,IP} and "
                       "a static neighbor entry between them for "
                       "left-to-right WriteWithImm";
            }
            GTEST_SKIP()
                << "Set HOLOLINK_CPU_ROCE_{LEFT,RIGHT}_{DEVICE,PORT,IP}; for "
                   "left-to-right WriteWithImm on two NICs add "
                   "a static neighbor entry between them first";
        }

        constexpr std::size_t kPayloadSize = 32;
        constexpr std::uint32_t kTxSlotCount = 4;
        constexpr std::uint32_t kRxSlotCount = 8;
        constexpr std::uint32_t kFrameCount = kRxSlotCount + 1;
        constexpr std::uint8_t kUnwritten = 0xA5;
        struct HandlerState {
            std::atomic<std::uint32_t> callbacks { 0 };
            std::atomic<std::uint32_t> credit_id { 0 };
            std::atomic<std::uint32_t> slot { 0 };
            std::optional<std::uint32_t> immediate_data;
        } handler_state;
        const auto handler = +[](void* context, RxCompletion completion) {
            auto& state = *static_cast<HandlerState*>(context);
            state.credit_id.store(
                completion.credit_id, std::memory_order_relaxed);
            const std::uint32_t slot = completion.immediate_data.value_or(0)
                & 0xFFF;
            state.slot.store(slot, std::memory_order_relaxed);
            state.immediate_data = completion.immediate_data;
            state.callbacks.fetch_add(1, std::memory_order_release);
        };
        auto pair = make_rx_handler_datapath_hardware_pair(
            *endpoints, CpuRoceTransceiver::TxOperation::WriteWithImm,
            kRxSlotCount, kTxSlotCount, kPayloadSize, handler, &handler_state);

        auto subject_monitor = std::async(std::launch::async, [&] {
            pair.subject->blocking_monitor();
        });
        auto peer_monitor = std::async(std::launch::async, [&] {
            pair.peer->blocking_monitor();
        });
        const bool monitors_active
            = detail::CpuRoceMonitorTestAccess::wait_until_active(
                  *pair.subject, 5000)
            && detail::CpuRoceMonitorTestAccess::wait_until_active(
                *pair.peer, 5000);
        if (!monitors_active) {
            pair.subject->close();
            pair.peer->close();
            subject_monitor.wait();
            peer_monitor.wait();
            FAIL() << "RX completion-handler monitor did not become active";
        }

        {
            struct ClosePair {
                RxHandlerHardwarePair& pair;
                ~ClosePair()
                {
                    pair.subject->close();
                    pair.peer->close();
                }
            } close_pair { pair };

            auto* const tx_data = pair.peer->get_tx_ring_data_addr();
            auto* const tx_flags = pair.peer->get_tx_ring_flag_addr();
            const auto tx_stride = pair.peer->get_tx_ring_stride();
            auto* const rx_data = pair.subject->get_rx_ring_data_addr();
            auto* const rx_flags = pair.subject->get_rx_ring_flag_addr();
            const auto rx_stride = pair.subject->get_rx_ring_stride();
            ASSERT_NE(tx_data, nullptr);
            ASSERT_NE(tx_flags, nullptr);
            ASSERT_NE(rx_data, nullptr);
            ASSERT_NE(rx_flags, nullptr);
            for (std::uint32_t slot = 0; slot < kRxSlotCount; ++slot) {
                std::memset(
                    rx_data + slot * rx_stride, kUnwritten, kPayloadSize);
            }

            for (std::uint32_t sequence = 0; sequence < kFrameCount;
                 ++sequence) {
                const std::uint32_t payload_slot
                    = sequence % kTxSlotCount;
                auto* const source
                    = tx_data + payload_slot * tx_stride;
                for (std::size_t i = 0; i < kPayloadSize; ++i) {
                    source[i] = static_cast<std::uint8_t>(
                        sequence * 43 + i * 5 + 7);
                }
                CpuRoceTransceiver::store_flag_release(
                    &tx_flags[payload_slot],
                    reinterpret_cast<std::uint64_t>(source));

                const auto callback_deadline
                    = std::chrono::steady_clock::now()
                    + std::chrono::seconds(5);
                while (handler_state.callbacks.load(std::memory_order_acquire)
                        < sequence + 1
                    && std::chrono::steady_clock::now()
                        < callback_deadline) {
                    std::this_thread::sleep_for(
                        std::chrono::microseconds(100));
                }
                ASSERT_EQ(
                    handler_state.callbacks.load(std::memory_order_acquire),
                    sequence + 1);
                EXPECT_EQ(
                    handler_state.credit_id.load(std::memory_order_relaxed),
                    sequence % kRxSlotCount);
                EXPECT_EQ(
                    handler_state.slot.load(std::memory_order_relaxed),
                    payload_slot);
                ASSERT_TRUE(handler_state.immediate_data.has_value());
                EXPECT_EQ(*handler_state.immediate_data, payload_slot);
                EXPECT_EQ(
                    std::memcmp(
                        rx_data + payload_slot * rx_stride, source,
                        kPayloadSize),
                    0);
                for (std::uint32_t slot = kTxSlotCount;
                     slot < kRxSlotCount; ++slot) {
                    const auto* const untouched
                        = rx_data + slot * rx_stride;
                    for (std::size_t i = 0; i < kPayloadSize; ++i) {
                        ASSERT_EQ(untouched[i], kUnwritten)
                            << "sequence=" << sequence << ", slot=" << slot
                            << ", byte=" << i;
                    }
                }
                for (std::uint32_t slot = 0; slot < kRxSlotCount; ++slot) {
                    EXPECT_EQ(
                        CpuRoceTransceiver::load_flag_acquire(&rx_flags[slot]),
                        0);
                }
            }
            // IMM names payload slots 0-3 while eight credits are posted, so
            // frames 4-7 write a different slot than sequential credit order.
            // Frame 8 completes only after credit 0 is reposted. The credit
            // id above is that reposted WQE.
            EXPECT_EQ(
                handler_state.callbacks.load(std::memory_order_acquire),
                kFrameCount);
        }

        ASSERT_EQ(
            subject_monitor.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
        ASSERT_EQ(
            peer_monitor.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
        EXPECT_NO_THROW(subject_monitor.get());
        EXPECT_NO_THROW(peer_monitor.get());
    }

    TEST(CpuRoceTransceiverRxCompletionHandlerHardwareTest,
        HandlerThrowTerminalizesMonitor)
    {
        const bool required
            = environment_flag("HOLOLINK_CPU_ROCE_HARDWARE_REQUIRED");
        const auto endpoints = configured_roce_endpoints();
        if (!endpoints) {
            if (required) {
                FAIL()
                    << "HOLOLINK_CPU_ROCE_HARDWARE_REQUIRED needs configured "
                       "HOLOLINK_CPU_ROCE_{LEFT,RIGHT}_{DEVICE,PORT,IP}";
            }
            GTEST_SKIP()
                << "Set HOLOLINK_CPU_ROCE_{LEFT,RIGHT}_{DEVICE,PORT,IP} for "
                   "the live RX completion-handler failure test";
        }

        auto pair = make_rx_handler_datapath_hardware_pair(
            *endpoints, CpuRoceTransceiver::TxOperation::WriteWithImm, 1, 1, 32,
            +[](void*, RxCompletion) {
                throw std::runtime_error("RX completion handler failed");
            },
            nullptr);
        auto subject_monitor = std::async(std::launch::async, [&] {
            pair.subject->blocking_monitor();
        });
        auto peer_monitor = std::async(std::launch::async, [&] {
            pair.peer->blocking_monitor();
        });
        struct ClosePair {
            RxHandlerHardwarePair& pair;
            ~ClosePair()
            {
                pair.subject->close();
                pair.peer->close();
            }
        } close_pair { pair };

        ASSERT_TRUE(detail::CpuRoceMonitorTestAccess::wait_until_active(
            *pair.subject, 5000));
        ASSERT_TRUE(detail::CpuRoceMonitorTestAccess::wait_until_active(
            *pair.peer, 5000));
        auto* const source = pair.peer->get_tx_ring_data_addr();
        auto* const flag = pair.peer->get_tx_ring_flag_addr();
        ASSERT_NE(source, nullptr);
        ASSERT_NE(flag, nullptr);
        CpuRoceTransceiver::store_flag_release(
            flag, reinterpret_cast<std::uint64_t>(source));

        ASSERT_EQ(
            subject_monitor.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
        try {
            subject_monitor.get();
            FAIL() << "completion-handler failure did not escape blocking_monitor";
        } catch (const std::runtime_error& error) {
            EXPECT_STREQ(error.what(), "RX completion handler failed");
        }
        EXPECT_THROW(
            RxCompletionHandler(*pair.subject)
                .enable_handled_receive(
                    +[](void*, RxCompletion) {}, nullptr, 1),
            std::logic_error);

        pair.subject->close();
        pair.peer->close();
        ASSERT_EQ(
            peer_monitor.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
        EXPECT_NO_THROW(peer_monitor.get());
    }

    // The Duplex coordinator is verbs-free, so its failure paths are covered
    // here directly rather than by planting fault branches in the datapath.
    TEST(CpuRoceTransceiverRingTest, DuplexCoordinatorRethrowsWorkerFailureAndStopsPeer)
    {
        for (const bool fail_rx_body : { true, false }) {
            SCOPED_TRACE(fail_rx_body ? "rx body throws" : "tx body throws");
            std::atomic<bool> exit_flag { false };
            std::atomic<bool> peer_entered { false };
            const detail::DuplexWorkerBody failing
                = [] { throw std::runtime_error("worker body failed"); };
            const detail::DuplexWorkerBody polling = [&] {
                peer_entered.store(true, std::memory_order_release);
                while (!exit_flag.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }
            };

            try {
                detail::run_duplex_workers(
                    exit_flag,
                    fail_rx_body ? failing : polling,
                    fail_rx_body ? polling : failing);
                FAIL() << "expected the worker body to throw";
            } catch (const std::runtime_error& error) {
                // std::system_error from thread construction also derives
                // from runtime_error; the message pins the intended body.
                EXPECT_STREQ(error.what(), "worker body failed");
            }
            // The surviving worker was released instead of polling forever.
            EXPECT_TRUE(exit_flag.load(std::memory_order_acquire));
            EXPECT_TRUE(peer_entered.load(std::memory_order_acquire));
        }
    }

    TEST(CpuRoceTransceiverRingTest, DuplexCoordinatorKeepsOnlyTheFirstFailure)
    {
        std::atomic<bool> exit_flag { false };
        // The TX body cannot throw until the RX failure has been captured,
        // because capture stores the exception before setting exit_flag.
        EXPECT_THROW(
            detail::run_duplex_workers(
                exit_flag,
                [] { throw std::logic_error("first failure"); },
                [&exit_flag] {
                    while (!exit_flag.load(std::memory_order_acquire)) {
                        std::this_thread::yield();
                    }
                    throw std::runtime_error("second failure");
                }),
            std::logic_error);
    }

    TEST(CpuRoceTransceiverRingTest, DuplexCoordinatorJoinsStartedWorkersWhenSpawnFails)
    {
        for (const int failing_spawn : { 0, 1 }) {
            SCOPED_TRACE(failing_spawn == 0 ? "first spawn" : "second spawn");
            std::atomic<bool> exit_flag { false };
            std::atomic<int> bodies_entered { 0 };
            int spawn_calls = 0;
            const detail::DuplexWorkerSpawn spawn
                = [&](detail::DuplexWorkerBody body) {
                      if (spawn_calls++ == failing_spawn) {
                          throw std::runtime_error("worker spawn failed");
                      }
                      return detail::spawn_duplex_worker(std::move(body));
                  };
            const detail::DuplexWorkerBody polling = [&] {
                bodies_entered.fetch_add(1, std::memory_order_acq_rel);
                while (!exit_flag.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }
            };

            // Reaching the assertions at all proves the started worker was
            // joined: a joinable std::thread destructor would terminate.
            EXPECT_THROW(
                detail::run_duplex_workers(exit_flag, polling, polling, spawn),
                std::runtime_error);
            EXPECT_TRUE(exit_flag.load(std::memory_order_acquire));
            EXPECT_EQ(
                bodies_entered.load(std::memory_order_acquire), failing_spawn);
        }
    }

    TEST(CpuRoceTransceiverRingTest, DuplexCoordinatorReturnsWhenBothWorkersExit)
    {
        std::atomic<bool> exit_flag { false };
        std::atomic<int> bodies_entered { 0 };
        const detail::DuplexWorkerBody polling = [&] {
            bodies_entered.fetch_add(1, std::memory_order_acq_rel);
            while (!exit_flag.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
        };

        std::thread closer([&] {
            while (bodies_entered.load(std::memory_order_acquire) < 2
                && !exit_flag.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            if (bodies_entered.load(std::memory_order_acquire) == 2) {
                exit_flag.store(true, std::memory_order_release);
            }
        });
        EXPECT_NO_THROW(
            detail::run_duplex_workers(exit_flag, polling, polling));
        closer.join();
        EXPECT_EQ(bodies_entered.load(std::memory_order_acquire), 2);
    }

    TEST(CpuRoceTransceiverRingTest, TryPinCurrentThreadReportsKernelRejectedCpu)
    {
        std::exception_ptr probe_error;
        bool saw_rejected_cpu = false;
        // Probe on a disposable thread: a CPU missing from this thread's
        // sched_getaffinity() mask may still be permitted by the cpuset, so
        // setaffinity can succeed. Success must not pin the GTest thread.
        std::thread probe([&] {
            try {
                cpu_set_t initial;
                CPU_ZERO(&initial);
                if (sched_getaffinity(0, sizeof(initial), &initial) != 0) {
                    throw std::runtime_error("sched_getaffinity failed");
                }
                for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
                    if (CPU_ISSET(cpu, &initial)) {
                        continue;
                    }
                    if (detail::try_pin_current_thread(cpu) != 0) {
                        saw_rejected_cpu = true;
                        return;
                    }
                }
            } catch (...) {
                probe_error = std::current_exception();
            }
        });
        probe.join();
        if (probe_error != nullptr) {
            std::rethrow_exception(probe_error);
        }
        if (!saw_rejected_cpu) {
            GTEST_SKIP()
                << "no CPU below CPU_SETSIZE was rejected by "
                   "sched_setaffinity; cannot prove the failure path";
        }
    }

    TEST(CpuRoceTransceiverRingTest, TryPinCurrentThreadRejectsOutOfRangeCpu)
    {
        EXPECT_EQ(detail::try_pin_current_thread(-1), EINVAL);
        EXPECT_EQ(detail::try_pin_current_thread(CPU_SETSIZE), EINVAL);
    }

    TEST(CpuRoceTransceiverRingTest, ThreadAffinityScopePinsAndRestores)
    {
        std::exception_ptr probe_error;
        std::thread probe([&] {
            try {
                cpu_set_t original;
                CPU_ZERO(&original);
                if (sched_getaffinity(0, sizeof(original), &original) != 0) {
                    throw std::runtime_error("sched_getaffinity failed");
                }

                int selected_cpu = -1;
                for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
                    if (CPU_ISSET(cpu, &original)) {
                        selected_cpu = cpu;
                        break;
                    }
                }
                if (selected_cpu < 0) {
                    throw std::runtime_error("thread has no available CPU");
                }

                detail::ThreadAffinityScope affinity(selected_cpu);
                {
                    cpu_set_t pinned;
                    CPU_ZERO(&pinned);
                    if (sched_getaffinity(0, sizeof(pinned), &pinned) != 0) {
                        throw std::runtime_error(
                            "sched_getaffinity failed while pinned");
                    }
                    if (CPU_COUNT(&pinned) != 1
                        || !CPU_ISSET(selected_cpu, &pinned)) {
                        throw std::runtime_error(
                            "thread affinity was not restricted to the "
                            "selected CPU");
                    }
                }
                affinity.restore();

                cpu_set_t restored;
                CPU_ZERO(&restored);
                if (sched_getaffinity(0, sizeof(restored), &restored) != 0
                    || !CPU_EQUAL(&original, &restored)) {
                    throw std::runtime_error(
                        "thread affinity was not restored");
                }

                bool body_error_observed = false;
                try {
                    detail::ThreadAffinityScope exceptional(selected_cpu);
                    throw std::runtime_error("body failed");
                } catch (const std::runtime_error&) {
                    body_error_observed = true;
                }
                if (!body_error_observed) {
                    throw std::runtime_error(
                        "body exception was not preserved");
                }
                CPU_ZERO(&restored);
                if (sched_getaffinity(0, sizeof(restored), &restored) != 0
                    || !CPU_EQUAL(&original, &restored)) {
                    throw std::runtime_error(
                        "thread affinity was not restored during unwinding");
                }
            } catch (...) {
                probe_error = std::current_exception();
            }
        });
        probe.join();
        if (probe_error != nullptr) {
            std::rethrow_exception(probe_error);
        }
    }

    TEST(CpuRoceTransceiverRingTest, DuplexCloseAfterMonitorActivationCompletes)
    {
        const bool required
            = environment_flag("HOLOLINK_CPU_ROCE_HARDWARE_REQUIRED");
        const auto endpoints = configured_roce_endpoints();
        if (!endpoints) {
            if (required) {
                FAIL()
                    << "HOLOLINK_CPU_ROCE_HARDWARE_REQUIRED needs configured "
                       "HOLOLINK_CPU_ROCE_{LEFT,RIGHT}_{DEVICE,PORT,IP}";
            }
            GTEST_SKIP()
                << "Set HOLOLINK_CPU_ROCE_{LEFT,RIGHT}_{DEVICE,PORT,IP} for "
                   "Duplex lifecycle tests";
        }

        auto left_config = host_config();
        left_config.mode = CpuRoceTransceiver::Mode::Duplex;
        left_config.ibv_name = endpoints->left.name;
        left_config.ibv_port = endpoints->left.port;
        left_config.local_ip = endpoints->left.ip;
        auto right_config = host_config();
        right_config.mode = CpuRoceTransceiver::Mode::Rx;
        right_config.ibv_name = endpoints->right.name;
        right_config.ibv_port = endpoints->right.port;
        right_config.local_ip = endpoints->right.ip;

        CpuRoceTransceiver left(std::move(left_config));
        CpuRoceTransceiver right(std::move(right_config));
        left.setup();
        right.setup();

        CpuRoceTransceiver::Peer to_right;
        to_right.qp = right.get_qp_number();
        to_right.ip = endpoints->right.ip;
        to_right.rx_stride = right.get_rx_ring_stride();
        CpuRoceTransceiver::Peer to_left;
        to_left.qp = left.get_qp_number();
        to_left.ip = endpoints->left.ip;
        ASSERT_NO_THROW(left.connect(to_right));
        ASSERT_NO_THROW(right.connect(to_left));

        auto monitor = std::async(std::launch::async, [&left] {
            left.blocking_monitor();
        });
        if (!detail::CpuRoceMonitorTestAccess::wait_until_active(
                left, 5000)) {
            left.close();
            right.close();
            FAIL()
                << "blocking_monitor() never set monitor_active; close() "
                   "would race the prologue";
        }

        auto closer = std::async(std::launch::async, [&left, &right] {
            left.close();
            right.close();
        });
        ASSERT_EQ(
            closer.wait_for(std::chrono::seconds(5)),
            std::future_status::ready)
            << "close() deadlocked after monitor_active";
        closer.get();
        ASSERT_EQ(
            monitor.wait_for(std::chrono::seconds(5)),
            std::future_status::ready)
            << "blocking_monitor hung after close()";
        EXPECT_NO_THROW(monitor.get());
    }

    TEST(CpuRoceTransceiverRingTest, TxSendPreservesOrderAcrossAsymmetricReceiveRingAndRearm)
    {
        const bool required
            = environment_flag("HOLOLINK_CPU_ROCE_HARDWARE_REQUIRED");
        const auto endpoints = configured_roce_endpoints();
        if (!endpoints) {
            if (required) {
                FAIL()
                    << "HOLOLINK_CPU_ROCE_HARDWARE_REQUIRED needs configured "
                       "HOLOLINK_CPU_ROCE_{LEFT,RIGHT}_{DEVICE,PORT,IP} and "
                       "a static neighbor entry between them for "
                       "left-to-right Send "
                       "(otherwise RTR ETIMEDOUT)";
            }
            GTEST_SKIP()
                << "Set HOLOLINK_CPU_ROCE_{LEFT,RIGHT}_{DEVICE,PORT,IP}; for "
                   "left-to-right Send on two NICs add "
                   "a static neighbor entry between them first "
                   "(otherwise RTR ETIMEDOUT)";
        }

        auto tx_config = host_config();
        tx_config.mode = CpuRoceTransceiver::Mode::Tx;
        tx_config.ibv_name = endpoints->left.name;
        tx_config.ibv_port = endpoints->left.port;
        tx_config.local_ip = endpoints->left.ip;
        const std::size_t tx_payload_size = 64;
        const std::size_t tx_stride = tx_config.tx.stride;
        const std::uint32_t tx_slot_count = 4;
        const std::uint32_t rx_slot_count = 8;
        tx_config.tx.frame_layout.payload_size = tx_payload_size;
        tx_config.tx.slot_count = tx_slot_count;
        auto rx_config = host_config();
        rx_config.mode = CpuRoceTransceiver::Mode::Rx;
        rx_config.ibv_name = endpoints->right.name;
        rx_config.ibv_port = endpoints->right.port;
        rx_config.local_ip = endpoints->right.ip;
        rx_config.rx.stride = tx_stride * 2;
        rx_config.rx.frame_layout.payload_size = tx_payload_size;
        rx_config.rx.slot_count = rx_slot_count;
        const std::size_t rx_stride = rx_config.rx.stride;
        constexpr std::uint8_t kUnwritten = 0xA5;

        CpuRoceTransceiver tx(std::move(tx_config));
        CpuRoceTransceiver rx(std::move(rx_config));
        tx.setup();
        rx.setup();

        CpuRoceTransceiver::Peer to_rx;
        to_rx.qp = rx.get_qp_number();
        to_rx.ip = endpoints->right.ip;
        to_rx.rx_stride = rx.get_rx_ring_stride();
        CpuRoceTransceiver::Peer to_tx;
        to_tx.qp = tx.get_qp_number();
        to_tx.ip = endpoints->left.ip;
        ASSERT_NO_THROW(tx.connect(to_rx));
        ASSERT_NO_THROW(rx.connect(to_tx));

        auto* const tx_data = tx.get_tx_ring_data_addr();
        auto* const tx_flags = tx.get_tx_ring_flag_addr();
        auto* const rx_data = rx.get_rx_ring_data_addr();
        auto* const rx_flags = rx.get_rx_ring_flag_addr();
        ASSERT_NE(tx_data, nullptr);
        ASSERT_NE(tx_flags, nullptr);
        ASSERT_NE(rx_data, nullptr);
        ASSERT_NE(rx_flags, nullptr);
        ASSERT_EQ(tx.get_tx_ring_slot_count(), tx_slot_count);
        ASSERT_EQ(rx.get_rx_ring_slot_count(), rx_slot_count);
        ASSERT_EQ(tx.get_tx_ring_stride(), tx_stride);
        ASSERT_EQ(rx.get_rx_ring_stride(), rx_stride);

        for (std::uint32_t slot = 0; slot < rx_slot_count; ++slot) {
            std::memset(rx_data + slot * rx_stride, kUnwritten, rx_stride);
        }

        const auto fill_slot = [&](std::uint32_t slot, std::uint8_t generation) {
            auto* const data = tx_data + slot * tx_stride;
            for (std::size_t i = 0; i < tx_payload_size; ++i) {
                data[i] = static_cast<std::uint8_t>(
                    generation * 53 + slot * 17 + i);
            }
        };
        const auto publish_slot = [&](std::uint32_t slot) {
            CpuRoceTransceiver::store_flag_release(
                &tx_flags[slot],
                reinterpret_cast<std::uint64_t>(
                    tx_data + slot * tx_stride));
        };
        const auto rx_matches = [&](std::uint32_t rx_slot, std::uint32_t tx_slot,
                                    std::uint8_t generation) {
            const auto* const data = rx_data + rx_slot * rx_stride;
            for (std::size_t i = 0; i < tx_payload_size; ++i) {
                const auto expected = static_cast<std::uint8_t>(
                    generation * 53 + tx_slot * 17 + i);
                if (data[i] != expected) {
                    return false;
                }
            }
            for (std::size_t i = tx_payload_size; i < rx_stride; ++i) {
                if (data[i] != kUnwritten) {
                    return false;
                }
            }
            return true;
        };

        for (std::uint32_t slot = 0; slot < tx_slot_count; ++slot) {
            fill_slot(slot, 0);
        }
        // Later slots may become ready early, but both TX operations consume
        // the local producer ring in cyclic order. For SEND, remote placement
        // still follows the peer's RQ and is not identified by this slot.
        for (const std::uint32_t slot : { 1u, 3u, 2u }) {
            publish_slot(slot);
        }

        auto tx_monitor = std::async(std::launch::async, [&tx] {
            tx.blocking_monitor();
        });
        auto rx_monitor = std::async(std::launch::async, [&rx] {
            rx.blocking_monitor();
        });
        const bool tx_active
            = detail::CpuRoceMonitorTestAccess::wait_until_active(tx, 5000);
        const bool rx_active
            = detail::CpuRoceMonitorTestAccess::wait_until_active(rx, 5000);
        if (!tx_active || !rx_active) {
            tx.close();
            rx.close();
            FAIL() << "TX/RX monitor did not become active";
        }
        // If the loop scanned ready flags, RX/TX would move before slot 0 is
        // published. The ring-level test separately confirms that a NotReady
        // claim of the expected slot leaves other published slots untouched.
        bool gap_held = true;
        const auto gap_deadline = std::chrono::steady_clock::now()
            + std::chrono::milliseconds(50);
        while (std::chrono::steady_clock::now() < gap_deadline) {
            if (CpuRoceTransceiver::load_flag_acquire(&tx_flags[0]) != 0) {
                gap_held = false;
                break;
            }
            for (const std::uint32_t slot : { 1u, 3u, 2u }) {
                if (CpuRoceTransceiver::load_flag_acquire(&tx_flags[slot])
                    == 0) {
                    gap_held = false;
                    break;
                }
            }
            for (std::uint32_t slot = 0; slot < rx_slot_count; ++slot) {
                if (CpuRoceTransceiver::load_flag_acquire(&rx_flags[slot])
                    != 0) {
                    gap_held = false;
                    break;
                }
            }
            if (!gap_held) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        publish_slot(0);

        const auto wait_until = [&](auto&& predicate) {
            const auto deadline = std::chrono::steady_clock::now()
                + std::chrono::seconds(5);
            while (!predicate()) {
                if (tx_monitor.wait_for(std::chrono::seconds(0))
                        == std::future_status::ready
                    || rx_monitor.wait_for(std::chrono::seconds(0))
                        == std::future_status::ready) {
                    return false;
                }
                if (std::chrono::steady_clock::now() >= deadline) {
                    return false;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            return true;
        };

        const bool first_lap = wait_until([&] {
            for (std::uint32_t slot = 0; slot < tx_slot_count; ++slot) {
                if (CpuRoceTransceiver::load_flag_acquire(&rx_flags[slot])
                    == 0) {
                    return false;
                }
            }
            return true;
        });
        bool first_payloads_match = first_lap;
        if (first_lap) {
            for (std::uint32_t slot = 0; slot < tx_slot_count; ++slot) {
                first_payloads_match
                    = first_payloads_match && rx_matches(slot, slot, 0);
            }
        }

        bool second_lap = false;
        bool second_payloads_match = false;
        bool first_lap_unmoved = false;
        if (first_lap) {
            // Leave RX 0–3 occupied. The next four Sends must land in RX 4–7
            // because placement follows remote WQE order, not the local slot.
            for (std::uint32_t slot = 0; slot < tx_slot_count; ++slot) {
                fill_slot(slot, 1);
                publish_slot(slot);
            }
            second_lap = wait_until([&] {
                for (std::uint32_t slot = 0; slot < tx_slot_count; ++slot) {
                    if (CpuRoceTransceiver::load_flag_acquire(
                            &rx_flags[tx_slot_count + slot])
                        == 0) {
                        return false;
                    }
                }
                return true;
            });
            second_payloads_match = second_lap;
            first_lap_unmoved = second_lap;
            if (second_lap) {
                for (std::uint32_t slot = 0; slot < tx_slot_count; ++slot) {
                    second_payloads_match = second_payloads_match
                        && rx_matches(tx_slot_count + slot, slot, 1);
                    first_lap_unmoved = first_lap_unmoved
                        && rx_matches(slot, slot, 0);
                }
                for (std::uint32_t slot = 0; slot < rx_slot_count; ++slot) {
                    CpuRoceTransceiver::store_flag_release(&rx_flags[slot], 0);
                }
            }
        }

        bool rearmed = false;
        bool rearmed_payload_matches = false;
        if (second_lap) {
            fill_slot(0, 2);
            publish_slot(0);
            rearmed = wait_until([&] {
                return CpuRoceTransceiver::load_flag_acquire(&rx_flags[0])
                    != 0;
            });
            rearmed_payload_matches = rearmed && rx_matches(0, 0, 2);
            if (rearmed) {
                CpuRoceTransceiver::store_flag_release(&rx_flags[0], 0);
            }
        }

        EXPECT_TRUE(first_lap);
        EXPECT_TRUE(first_payloads_match);
        EXPECT_TRUE(gap_held);
        EXPECT_TRUE(second_lap);
        EXPECT_TRUE(second_payloads_match);
        EXPECT_TRUE(first_lap_unmoved);
        EXPECT_TRUE(rearmed);
        EXPECT_TRUE(rearmed_payload_matches);

        tx.close();
        rx.close();
        ASSERT_EQ(
            tx_monitor.wait_for(std::chrono::seconds(5)),
            std::future_status::ready)
            << "TX blocking_monitor hung after close()";
        ASSERT_EQ(
            rx_monitor.wait_for(std::chrono::seconds(5)),
            std::future_status::ready)
            << "RX blocking_monitor hung after close()";
        EXPECT_NO_THROW(tx_monitor.get());
        EXPECT_NO_THROW(rx_monitor.get());
    }

    TEST(CpuRoceTransceiverRingTest, RxStallResumesAfterReleaseAndStopsOnClose)
    {
        const bool required
            = environment_flag("HOLOLINK_CPU_ROCE_HARDWARE_REQUIRED");
        const auto endpoints = configured_roce_endpoints();
        if (!endpoints) {
            if (required) {
                FAIL()
                    << "HOLOLINK_CPU_ROCE_HARDWARE_REQUIRED needs configured "
                       "HOLOLINK_CPU_ROCE_{LEFT,RIGHT}_{DEVICE,PORT,IP}";
            }
            GTEST_SKIP()
                << "Set HOLOLINK_CPU_ROCE_{LEFT,RIGHT}_{DEVICE,PORT,IP} for "
                   "the two-NIC RX Stall test";
        }

        constexpr std::size_t kPayloadSize = 64;
        constexpr std::uint32_t kSlotCount = 2;
        constexpr std::uint32_t kReceiveSlotCount = 8;
        auto tx_config = host_config();
        tx_config.mode = CpuRoceTransceiver::Mode::Tx;
        tx_config.tx_operation
            = CpuRoceTransceiver::TxOperation::WriteWithImm;
        tx_config.ibv_name = endpoints->left.name;
        tx_config.ibv_port = endpoints->left.port;
        tx_config.local_ip = endpoints->left.ip;
        tx_config.tx.frame_layout.payload_size = kPayloadSize;
        tx_config.tx.slot_count = kSlotCount;
        auto rx_config = host_config();
        rx_config.mode = CpuRoceTransceiver::Mode::Rx;
        rx_config.ibv_name = endpoints->right.name;
        rx_config.ibv_port = endpoints->right.port;
        rx_config.local_ip = endpoints->right.ip;
        rx_config.rx.frame_layout.payload_size = kPayloadSize;
        // Four two-slot laps consume the eight initially posted WQEs. This
        // test therefore never uses a sleep as proof that publish-before-rearm
        // has restored remote receive credit.
        rx_config.rx.slot_count = kReceiveSlotCount;

        CpuRoceTransceiver tx(std::move(tx_config));
        CpuRoceTransceiver rx(std::move(rx_config));
        tx.setup();
        rx.setup();

        CpuRoceTransceiver::Peer to_rx;
        to_rx.qp = rx.get_qp_number();
        to_rx.ip = endpoints->right.ip;
        to_rx.rx_base = rx.external_frame_memory();
        to_rx.rx_rkey = rx.get_rkey();
        to_rx.rx_stride = rx.get_rx_ring_stride();
        to_rx.rx_slot_count = rx.get_rx_ring_slot_count();
        CpuRoceTransceiver::Peer to_tx;
        to_tx.qp = tx.get_qp_number();
        to_tx.ip = endpoints->left.ip;
        ASSERT_NO_THROW(tx.connect(to_rx));
        ASSERT_NO_THROW(rx.connect(to_tx));

        auto* const tx_data = tx.get_tx_ring_data_addr();
        auto* const tx_flags = tx.get_tx_ring_flag_addr();
        auto* const rx_data = rx.get_rx_ring_data_addr();
        auto* const rx_flags = rx.get_rx_ring_flag_addr();
        ASSERT_NE(tx_data, nullptr);
        ASSERT_NE(tx_flags, nullptr);
        ASSERT_NE(rx_data, nullptr);
        ASSERT_NE(rx_flags, nullptr);
        const auto tx_stride = tx.get_tx_ring_stride();
        const auto rx_stride = rx.get_rx_ring_stride();

        const auto fill_and_publish
            = [&](std::uint32_t slot, std::uint8_t generation) {
                  auto* const data = tx_data + slot * tx_stride;
                  for (std::size_t i = 0; i < kPayloadSize; ++i) {
                      data[i] = static_cast<std::uint8_t>(
                          generation * 61 + slot * 17 + i);
                  }
                  CpuRoceTransceiver::store_flag_release(
                      &tx_flags[slot],
                      reinterpret_cast<std::uint64_t>(data));
              };
        const auto rx_matches
            = [&](std::uint32_t slot, std::uint8_t generation) {
                  const auto* const data = rx_data + slot * rx_stride;
                  for (std::size_t i = 0; i < kPayloadSize; ++i) {
                      if (data[i]
                          != static_cast<std::uint8_t>(
                              generation * 61 + slot * 17 + i)) {
                          return false;
                      }
                  }
                  return true;
              };

        auto tx_monitor = std::async(std::launch::async, [&tx] {
            tx.blocking_monitor();
        });
        auto rx_monitor = std::async(std::launch::async, [&rx] {
            rx.blocking_monitor();
        });
        const bool monitors_active
            = detail::CpuRoceMonitorTestAccess::wait_until_active(tx, 5000)
            && detail::CpuRoceMonitorTestAccess::wait_until_active(rx, 5000);
        if (!monitors_active) {
            tx.close();
            rx.close();
            FAIL() << "TX/RX monitor did not become active";
        }
        struct CloseMonitors {
            CpuRoceTransceiver& tx;
            CpuRoceTransceiver& rx;
            ~CloseMonitors()
            {
                tx.close();
                rx.close();
            }
        } close_monitors { tx, rx };

        const auto wait_until = [&](auto&& predicate) {
            const auto deadline = std::chrono::steady_clock::now()
                + std::chrono::seconds(5);
            while (!predicate()) {
                if (tx_monitor.wait_for(std::chrono::seconds(0))
                        == std::future_status::ready
                    || rx_monitor.wait_for(std::chrono::seconds(0))
                        == std::future_status::ready
                    || std::chrono::steady_clock::now() >= deadline) {
                    return false;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            return true;
        };
        const auto lap_ready = [&](std::uint8_t generation) {
            return wait_until([&] {
                for (std::uint32_t slot = 0; slot < kSlotCount; ++slot) {
                    if (CpuRoceTransceiver::load_flag_acquire(&rx_flags[slot])
                            == 0
                        || !rx_matches(slot, generation)) {
                        return false;
                    }
                }
                return true;
            });
        };

        for (std::uint32_t slot = 0; slot < kSlotCount; ++slot) {
            fill_and_publish(slot, 0);
        }
        ASSERT_TRUE(lap_ready(0));

        // Keep both publications occupied across a full ring lap. The second
        // slot-0 CQE must wait instead of killing the monitor.
        for (std::uint32_t slot = 0; slot < kSlotCount; ++slot) {
            fill_and_publish(slot, 1);
        }
        ASSERT_TRUE(wait_until([&] {
            return rx.get_rx_collision_count() == 1;
        }));
        EXPECT_EQ(
            rx_monitor.wait_for(std::chrono::seconds(0)),
            std::future_status::timeout);

        // Release both slots so the test does not assume which receive CQE
        // was first in the returned batch.
        for (std::uint32_t slot = 0; slot < kSlotCount; ++slot) {
            CpuRoceTransceiver::store_flag_release(&rx_flags[slot], 0);
        }
        ASSERT_TRUE(lap_ready(1));
        for (std::uint32_t slot = 0; slot < kSlotCount; ++slot) {
            CpuRoceTransceiver::store_flag_release(&rx_flags[slot], 0);
        }

        for (std::uint32_t slot = 0; slot < kSlotCount; ++slot) {
            fill_and_publish(slot, 2);
        }
        ASSERT_TRUE(lap_ready(2));

        // A later occupied lap proves close() interrupts the inner Stall
        // loop; neither monitor may hang waiting for a consumer release.
        for (std::uint32_t slot = 0; slot < kSlotCount; ++slot) {
            fill_and_publish(slot, 3);
        }
        ASSERT_TRUE(wait_until([&] {
            return rx.get_rx_collision_count() == 2;
        }));
        tx.close();
        rx.close();
        ASSERT_EQ(
            tx_monitor.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
        ASSERT_EQ(
            rx_monitor.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
        EXPECT_NO_THROW(tx_monitor.get());
        EXPECT_NO_THROW(rx_monitor.get());
        EXPECT_EQ(rx.get_rx_collision_count(), 2);
    }

    TEST(CpuRoceTransceiverRingTest, ForwardEchoesIntoDuplexReceiveRing)
    {
        const bool required
            = environment_flag("HOLOLINK_CPU_ROCE_HARDWARE_REQUIRED");
        const auto endpoints = configured_roce_endpoints();
        if (!endpoints) {
            if (required) {
                FAIL()
                    << "HOLOLINK_CPU_ROCE_HARDWARE_REQUIRED needs configured "
                       "HOLOLINK_CPU_ROCE_{LEFT,RIGHT}_{DEVICE,PORT,IP}";
            }
            GTEST_SKIP()
                << "Set HOLOLINK_CPU_ROCE_{LEFT,RIGHT}_{DEVICE,PORT,IP} for "
                   "the two-NIC Forward echo test";
        }

        // Above the Send inline threshold so Forward must use registered DMA
        // through the VA-based local RX MR.
        constexpr std::size_t kPayloadSize = 256;
        constexpr std::uint32_t kSlotCount = 4;
        auto duplex_config = host_config();
        duplex_config.mode = CpuRoceTransceiver::Mode::Duplex;
        duplex_config.ibv_name = endpoints->left.name;
        duplex_config.ibv_port = endpoints->left.port;
        duplex_config.local_ip = endpoints->left.ip;
        duplex_config.rx.frame_layout.payload_size = kPayloadSize;
        duplex_config.tx.frame_layout.payload_size = kPayloadSize;
        duplex_config.rx.slot_count = kSlotCount;
        duplex_config.tx.slot_count = kSlotCount;

        auto forward_config = host_config();
        forward_config.mode = CpuRoceTransceiver::Mode::Forward;
        forward_config.ibv_name = endpoints->right.name;
        forward_config.ibv_port = endpoints->right.port;
        forward_config.local_ip = endpoints->right.ip;
        forward_config.rx.frame_layout.payload_size = kPayloadSize;
        forward_config.rx.slot_count = kSlotCount;

        CpuRoceTransceiver duplex(std::move(duplex_config));
        CpuRoceTransceiver forward(std::move(forward_config));
        duplex.setup();
        forward.setup();

        CpuRoceTransceiver::Peer to_forward;
        to_forward.qp = forward.get_qp_number();
        to_forward.ip = endpoints->right.ip;
        to_forward.rx_stride = forward.get_rx_ring_stride();
        CpuRoceTransceiver::Peer to_duplex;
        to_duplex.qp = duplex.get_qp_number();
        to_duplex.ip = endpoints->left.ip;
        to_duplex.rx_stride = duplex.get_rx_ring_stride();
        ASSERT_NO_THROW(duplex.connect(to_forward));
        ASSERT_NO_THROW(forward.connect(to_duplex));

        auto* const tx_data = duplex.get_tx_ring_data_addr();
        auto* const tx_flags = duplex.get_tx_ring_flag_addr();
        auto* const rx_data = duplex.get_rx_ring_data_addr();
        auto* const rx_flags = duplex.get_rx_ring_flag_addr();
        ASSERT_NE(tx_data, nullptr);
        ASSERT_NE(tx_flags, nullptr);
        ASSERT_NE(rx_data, nullptr);
        ASSERT_NE(rx_flags, nullptr);
        ASSERT_EQ(forward.get_tx_ring_data_addr(), nullptr);

        for (std::size_t i = 0; i < kPayloadSize; ++i) {
            tx_data[i] = static_cast<std::uint8_t>(i * 7 + 3);
        }

        auto duplex_monitor = std::async(std::launch::async, [&duplex] {
            duplex.blocking_monitor();
        });
        auto forward_monitor = std::async(std::launch::async, [&forward] {
            forward.blocking_monitor();
        });
        const bool monitors_ready
            = detail::CpuRoceMonitorTestAccess::wait_until_active(duplex, 5000)
            && detail::CpuRoceMonitorTestAccess::wait_until_active(
                forward, 5000);

        bool echoed = false;
        bool payload_matches = false;
        if (monitors_ready) {
            CpuRoceTransceiver::store_flag_release(
                &tx_flags[0], reinterpret_cast<std::uint64_t>(tx_data));
            const auto deadline = std::chrono::steady_clock::now()
                + std::chrono::seconds(5);
            while (std::chrono::steady_clock::now() < deadline) {
                if (CpuRoceTransceiver::load_flag_acquire(&rx_flags[0])
                    != 0) {
                    echoed = true;
                    break;
                }
                if (duplex_monitor.wait_for(std::chrono::seconds(0))
                        == std::future_status::ready
                    || forward_monitor.wait_for(std::chrono::seconds(0))
                        == std::future_status::ready) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            payload_matches = echoed
                && std::memcmp(tx_data, rx_data, kPayloadSize) == 0;
        }

        duplex.close();
        forward.close();
        ASSERT_EQ(
            duplex_monitor.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
        ASSERT_EQ(
            forward_monitor.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
        EXPECT_NO_THROW(duplex_monitor.get());
        EXPECT_NO_THROW(forward_monitor.get());
        EXPECT_TRUE(monitors_ready);
        EXPECT_TRUE(echoed);
        EXPECT_TRUE(payload_matches);
    }

    // Two-NIC CallerDriven exchange. Skipped unless
    // HOLOLINK_CPU_ROCE_{LEFT,RIGHT}_{DEVICE,PORT,IP} are set. Covers both
    // verbs, more than one ring wrap, a held tx_flag, and a send completion
    // queue that is still full when a slot is published again.
    bool exercise_caller_driven_exchange(
        const ConfiguredRoceEndpoints& endpoints,
        CpuRoceTransceiver::TxOperation operation,
        std::size_t payload)
    {
        constexpr std::uint32_t kSlots = 4;
        constexpr std::uint32_t kMessages = kSlots * 2 + 1;
        const bool write_with_imm
            = operation == CpuRoceTransceiver::TxOperation::WriteWithImm;

        const auto make_config = [&](const LocalRoceDevice& device) {
            auto config = host_config();
            config.mode = CpuRoceTransceiver::Mode::CallerDriven;
            config.tx_operation = operation;
            config.ibv_name = device.name;
            config.ibv_port = device.port;
            config.local_ip = device.ip;
            config.rx.frame_layout.payload_size = payload;
            config.tx.frame_layout.payload_size = payload;
            config.rx.slot_count = kSlots;
            config.tx.slot_count = kSlots;
            return config;
        };
        CpuRoceTransceiver left(make_config(endpoints.left));
        CpuRoceTransceiver right(make_config(endpoints.right));
        struct CloseBoth {
            CpuRoceTransceiver& left;
            CpuRoceTransceiver& right;
            ~CloseBoth()
            {
                left.close();
                right.close();
            }
        } close_both { left, right };

        left.setup();
        right.setup();
        const auto peer_of = [](const CpuRoceTransceiver& remote,
                                 const std::string& ip) {
            CpuRoceTransceiver::Peer peer;
            peer.qp = remote.get_qp_number();
            peer.ip = ip;
            peer.rx_base = remote.external_frame_memory();
            peer.rx_rkey = remote.get_rkey();
            peer.rx_stride = remote.get_rx_ring_stride();
            peer.rx_slot_count = remote.get_rx_ring_slot_count();
            return peer;
        };
        left.connect(peer_of(right, endpoints.right.ip));
        right.connect(peer_of(left, endpoints.left.ip));

        auto* const left_tx = left.get_tx_ring_data_addr();
        auto* const left_rx = left.get_rx_ring_data_addr();
        auto* const right_tx = right.get_tx_ring_data_addr();
        auto* const right_rx = right.get_rx_ring_data_addr();
        auto* const right_flags = right.get_tx_ring_flag_addr();
        if (left_tx == nullptr || left_rx == nullptr || right_tx == nullptr
            || right_rx == nullptr || right_flags == nullptr) {
            ADD_FAILURE() << "CallerDriven did not publish both rings";
            return false;
        }
        const auto stride = left.get_tx_ring_stride();

        const auto fill = [payload](std::uint8_t* slot, std::uint8_t sequence) {
            for (std::size_t i = 0; i < payload; ++i) {
                slot[i] = static_cast<std::uint8_t>(sequence * 17 + i);
            }
        };
        const auto matches = [payload](const std::uint8_t* slot,
                                 std::uint8_t sequence) {
            for (std::size_t i = 0; i < payload; ++i) {
                if (slot[i] != static_cast<std::uint8_t>(sequence * 17 + i)) {
                    return false;
                }
            }
            return true;
        };
        const auto poll_for = [](CpuRoceTransceiver& transceiver,
                                  std::uint32_t* slot,
                                  std::chrono::milliseconds budget) {
            const auto deadline = std::chrono::steady_clock::now() + budget;
            while (std::chrono::steady_clock::now() < deadline) {
                if (transceiver.rx_poll(slot)) {
                    return true;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            return false;
        };
        const auto publish_when_complete = [](CpuRoceTransceiver& transceiver,
                                               std::uint32_t slot) {
            const auto deadline = std::chrono::steady_clock::now()
                + std::chrono::seconds(2);
            while (std::chrono::steady_clock::now() < deadline) {
                try {
                    return transceiver.tx_publish(slot);
                } catch (const std::logic_error&) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
            return false;
        };

        for (std::uint32_t sequence = 0; sequence < kMessages; ++sequence) {
            const auto request_slot = sequence % kSlots;
            fill(left_tx + request_slot * stride,
                static_cast<std::uint8_t>(sequence));
            if (!publish_when_complete(left, request_slot)) {
                ADD_FAILURE() << "request publish failed at " << sequence;
                return false;
            }
            std::uint32_t received = 0;
            if (!poll_for(right, &received, std::chrono::seconds(2))) {
                ADD_FAILURE() << "request " << sequence << " did not arrive";
                return false;
            }
            if (write_with_imm && received != request_slot) {
                ADD_FAILURE() << "WriteWithImm request " << sequence
                              << " landed in slot " << received;
                return false;
            }
            if (!matches(
                    right_rx + received * stride,
                    static_cast<std::uint8_t>(sequence))) {
                ADD_FAILURE() << "request " << sequence << " payload mismatch";
                return false;
            }
            auto* const response = right_tx + received * stride;
            for (std::size_t i = 0; i < payload; ++i) {
                response[i] = static_cast<std::uint8_t>(
                    (right_rx + received * stride)[i] ^ 0xA5);
            }
            if (!publish_when_complete(right, received)) {
                ADD_FAILURE() << "response publish failed at " << sequence;
                return false;
            }
            std::uint32_t echoed = 0;
            if (!poll_for(left, &echoed, std::chrono::seconds(2))) {
                ADD_FAILURE() << "response " << sequence << " did not arrive";
                return false;
            }
            if (write_with_imm && echoed != received) {
                ADD_FAILURE() << "WriteWithImm response " << sequence
                              << " landed in slot " << echoed;
                return false;
            }
            const auto* const echoed_bytes = left_rx + echoed * stride;
            for (std::size_t i = 0; i < payload; ++i) {
                if (echoed_bytes[i]
                    != static_cast<std::uint8_t>(
                        (sequence * 17 + i) ^ 0xA5)) {
                    ADD_FAILURE() << "response " << sequence
                                  << " payload mismatch";
                    return false;
                }
            }
        }

        // Hold the slot a long-running reader still owns. The next completion
        // for that slot stays queued until its tx_flag is cleared.
        constexpr std::uint8_t kHeldSequence = 0x41;
        fill(left_tx, kHeldSequence);
        if (!publish_when_complete(left, 0)) {
            ADD_FAILURE() << "held-flag request did not publish";
            return false;
        }
        std::uint32_t held = 0;
        if (!poll_for(right, &held, std::chrono::seconds(2))
            || (write_with_imm && held != 0)
            || !matches(right_rx + held * stride, kHeldSequence)) {
            ADD_FAILURE() << "held-flag request did not arrive";
            return false;
        }
        CpuRoceTransceiver::store_flag_release(&right_flags[held], 1);
        constexpr std::uint8_t kBlockedSequence = 0x42;
        if (write_with_imm) {
            fill(left_tx, kBlockedSequence);
            if (!publish_when_complete(left, 0)) {
                ADD_FAILURE() << "second WriteWithImm to the held slot failed";
                return false;
            }
        } else {
            // Claiming the slot re-arms its receive WQE at the tail. The next
            // three Sends take the other three posted WQEs, and the fourth
            // lands in the held slot. This phase depends on that order.
            for (std::uint32_t i = 1; i < kSlots; ++i) {
                fill(left_tx + i * stride, static_cast<std::uint8_t>(0x50 + i));
                if (!publish_when_complete(left, i)) {
                    ADD_FAILURE() << "Send ahead of the held slot failed";
                    return false;
                }
                std::uint32_t other = 0;
                if (!poll_for(right, &other, std::chrono::seconds(2))
                    || other == held) {
                    ADD_FAILURE()
                        << "tx_flag held slot was returned early";
                    return false;
                }
            }
            fill(left_tx, kBlockedSequence);
            if (!publish_when_complete(left, 0)) {
                ADD_FAILURE() << "Send that reuses the held slot failed";
                return false;
            }
        }
        const auto hold_deadline = std::chrono::steady_clock::now()
            + std::chrono::milliseconds(500);
        while (std::chrono::steady_clock::now() < hold_deadline) {
            std::uint32_t early = 0;
            if (right.rx_poll(&early)) {
                ADD_FAILURE() << "rx_poll returned slot " << early
                              << " while its tx_flag was set";
                return false;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        CpuRoceTransceiver::store_flag_release(&right_flags[held], 0);
        std::uint32_t released = 0;
        if (!poll_for(right, &released, std::chrono::seconds(2))
            || released != held
            || !matches(right_rx + released * stride, kBlockedSequence)) {
            ADD_FAILURE() << "held slot was not returned after tx_flag cleared";
            return false;
        }

        // Publish a full ring without the sender polling, then publish one of
        // those slots again. The second publish has to reap the backed-up
        // send queue before the slot can be reused.
        for (std::uint32_t slot = 0; slot < kSlots; ++slot) {
            fill(
                left_tx + slot * stride,
                static_cast<std::uint8_t>(0x60 + slot));
            if (!publish_when_complete(left, slot)) {
                ADD_FAILURE() << "backlog publish failed for slot " << slot;
                return false;
            }
        }
        bool saw_pattern[kSlots] = {};
        for (std::uint32_t n = 0; n < kSlots; ++n) {
            std::uint32_t slot = 0;
            if (!poll_for(right, &slot, std::chrono::seconds(2))) {
                ADD_FAILURE() << "backlog request " << n << " did not arrive";
                return false;
            }
            bool found = false;
            for (std::uint32_t pattern = 0; pattern < kSlots; ++pattern) {
                if (saw_pattern[pattern]
                    || !matches(
                        right_rx + slot * stride,
                        static_cast<std::uint8_t>(0x60 + pattern))) {
                    continue;
                }
                if (write_with_imm && slot != pattern) {
                    ADD_FAILURE() << "backlog WriteWithImm slot " << slot
                                  << " carried pattern " << pattern;
                    return false;
                }
                saw_pattern[pattern] = true;
                found = true;
                break;
            }
            if (!found) {
                ADD_FAILURE() << "backlog request " << n
                              << " payload mismatch";
                return false;
            }
        }
        fill(left_tx, 0x70);
        if (!publish_when_complete(left, 0)) {
            ADD_FAILURE() << "slot reuse did not reap the backed-up send queue";
            return false;
        }
        std::uint32_t reused = 0;
        if (!poll_for(right, &reused, std::chrono::seconds(2))
            || !matches(right_rx + reused * stride, 0x70)) {
            ADD_FAILURE() << "reused slot's request did not arrive";
            return false;
        }
        return true;
    }

    TEST(CpuRoceTransceiverRingTest, CallerDrivenExchangesOnTwoNics)
    {
        const bool required
            = environment_flag("HOLOLINK_CPU_ROCE_HARDWARE_REQUIRED");
        const auto endpoints = configured_roce_endpoints();
        if (!endpoints) {
            if (required) {
                FAIL()
                    << "HOLOLINK_CPU_ROCE_HARDWARE_REQUIRED needs configured "
                       "HOLOLINK_CPU_ROCE_{LEFT,RIGHT}_{DEVICE,PORT,IP}";
            }
            GTEST_SKIP()
                << "Set HOLOLINK_CPU_ROCE_{LEFT,RIGHT}_{DEVICE,PORT,IP} for "
                   "the two-NIC CallerDriven exchange";
        }

        for (const auto operation :
            { CpuRoceTransceiver::TxOperation::Send,
                CpuRoceTransceiver::TxOperation::WriteWithImm }) {
            const char* const verb
                = operation == CpuRoceTransceiver::TxOperation::Send
                ? "Send"
                : "WriteWithImm";
            // 32 bytes is inline. 256 bytes fills the slot and is read by DMA.
            for (const std::size_t payload :
                { std::size_t { 32 }, std::size_t { 256 } }) {
                SCOPED_TRACE(verb);
                SCOPED_TRACE(payload);
                EXPECT_TRUE(exercise_caller_driven_exchange(
                    *endpoints, operation, payload));
            }
        }
    }

    TEST(CpuRoceTransceiverRingTest, UnifiedDispatchesAndSendsResponse)
    {
        const bool required
            = environment_flag("HOLOLINK_CPU_ROCE_HARDWARE_REQUIRED");
        const auto endpoints = configured_roce_endpoints();
        if (!endpoints) {
            if (required) {
                FAIL()
                    << "HOLOLINK_CPU_ROCE_HARDWARE_REQUIRED needs configured "
                       "HOLOLINK_CPU_ROCE_{LEFT,RIGHT}_{DEVICE,PORT,IP}";
            }
            GTEST_SKIP()
                << "Set HOLOLINK_CPU_ROCE_{LEFT,RIGHT}_{DEVICE,PORT,IP} for "
                   "the two-NIC Unified test";
        }
        // This test requires scatter-to-CQE. Accept only unset (provider
        // default) or "1"; do not infer how libmlx5 parses other values.
        const char* const scatter_to_cqe
            = std::getenv("MLX5_SCATTER_TO_CQE");
        if (scatter_to_cqe != nullptr
            && std::string(scatter_to_cqe) != "1") {
            FAIL()
                << "The 32-byte Unified-response receive regression requires "
                   "scatter-to-CQE; unset MLX5_SCATTER_TO_CQE or set it to 1";
        }

        constexpr std::size_t kRequestSize = 128;
        constexpr std::size_t kResponseCapacity = 64;
        // This 32-byte Unified response empirically took mlx5's inline-scatter
        // path into the Duplex receive SGE on the hardware regression rig.
        constexpr std::size_t kResponseSize = 32;
        constexpr std::uint32_t kSlotCount = 4;

        struct DispatchContext {
            std::size_t expected_rx_size = 0;
            std::size_t expected_tx_capacity = 0;
            std::size_t response_size = 0;
            std::atomic<unsigned> calls { 0 };
            std::atomic<bool> geometry_matches { true };
        } dispatch_context {
            kRequestSize, kResponseCapacity, kResponseSize
        };
        const auto dispatch = +[](void* opaque,
                                   const void* rx_slot,
                                   std::size_t rx_payload_size,
                                   void* tx_slot,
                                   std::size_t tx_capacity) {
            auto& context = *static_cast<DispatchContext*>(opaque);
            if (rx_payload_size != context.expected_rx_size
                || tx_capacity != context.expected_tx_capacity) {
                context.geometry_matches.store(false, std::memory_order_release);
            }
            const auto* const input
                = static_cast<const std::uint8_t*>(rx_slot);
            auto* const output = static_cast<std::uint8_t*>(tx_slot);
            for (std::size_t i = 0; i < context.response_size; ++i) {
                output[i] = static_cast<std::uint8_t>(input[i] ^ 0xA5);
            }
            context.calls.fetch_add(1, std::memory_order_release);
            return CpuRoceTransceiver::UnifiedDispatchResult {
                CpuRoceTransceiver::UnifiedDispatchDisposition::Send,
                context.response_size
            };
        };

        auto duplex_config = host_config();
        duplex_config.mode = CpuRoceTransceiver::Mode::Duplex;
        duplex_config.ibv_name = endpoints->left.name;
        duplex_config.ibv_port = endpoints->left.port;
        duplex_config.local_ip = endpoints->left.ip;
        duplex_config.rx.frame_layout.payload_size = kResponseCapacity;
        duplex_config.tx.frame_layout.payload_size = kRequestSize;
        duplex_config.rx.slot_count = kSlotCount;
        duplex_config.tx.slot_count = kSlotCount;

        auto unified_config = host_config();
        unified_config.mode = CpuRoceTransceiver::Mode::Unified;
        unified_config.ibv_name = endpoints->right.name;
        unified_config.ibv_port = endpoints->right.port;
        unified_config.local_ip = endpoints->right.ip;
        unified_config.rx.frame_layout.payload_size = kRequestSize;
        unified_config.rx.stride = 256;
        unified_config.tx.frame_layout.payload_size = kResponseCapacity;
        unified_config.tx.stride = 128;
        unified_config.rx.slot_count = kSlotCount;
        unified_config.tx.slot_count = kSlotCount;

        CpuRoceTransceiver duplex(std::move(duplex_config));
        CpuRoceTransceiver unified(std::move(unified_config));
        unified.set_unified_dispatch(dispatch, &dispatch_context);
        duplex.setup();
        unified.setup();

        CpuRoceTransceiver::Peer to_unified;
        to_unified.qp = unified.get_qp_number();
        to_unified.ip = endpoints->right.ip;
        to_unified.rx_stride = unified.get_rx_ring_stride();
        CpuRoceTransceiver::Peer to_duplex;
        to_duplex.qp = duplex.get_qp_number();
        to_duplex.ip = endpoints->left.ip;
        to_duplex.rx_stride = duplex.get_rx_ring_stride();
        ASSERT_NO_THROW(duplex.connect(to_unified));
        ASSERT_NO_THROW(unified.connect(to_duplex));

        auto* const request_data = duplex.get_tx_ring_data_addr();
        auto* const request_flags = duplex.get_tx_ring_flag_addr();
        auto* const response_data = duplex.get_rx_ring_data_addr();
        auto* const response_flags = duplex.get_rx_ring_flag_addr();
        ASSERT_NE(request_data, nullptr);
        ASSERT_NE(request_flags, nullptr);
        ASSERT_NE(response_data, nullptr);
        ASSERT_NE(response_flags, nullptr);
        const auto request_stride = duplex.get_tx_ring_stride();
        const auto response_stride = duplex.get_rx_ring_stride();

        auto duplex_monitor = std::async(std::launch::async, [&duplex] {
            duplex.blocking_monitor();
        });
        auto unified_monitor = std::async(std::launch::async, [&unified] {
            unified.blocking_monitor();
        });
        const bool monitors_ready
            = detail::CpuRoceMonitorTestAccess::wait_until_active(duplex, 5000)
            && detail::CpuRoceMonitorTestAccess::wait_until_active(
                unified, 5000);

        bool responses_ready = monitors_ready;
        bool responses_match = monitors_ready;
        if (monitors_ready) {
            // Five requests wrap the four-slot ring and force Unified to
            // retire the prior slot-0 TX completion before invoking the fifth
            // callback on slot 0.
            for (std::uint32_t sequence = 0;
                 responses_ready && sequence < kSlotCount + 1; ++sequence) {
                const auto slot = sequence & (kSlotCount - 1);
                auto* const request_slot
                    = request_data + slot * request_stride;
                auto* const response_slot
                    = response_data + slot * response_stride;
                for (std::size_t i = 0; i < kRequestSize; ++i) {
                    request_slot[i] = static_cast<std::uint8_t>(
                        i * 3 + sequence * 17 + 1);
                }
                CpuRoceTransceiver::store_flag_release(
                    &request_flags[slot],
                    reinterpret_cast<std::uint64_t>(request_slot));

                bool response_ready = false;
                const auto deadline = std::chrono::steady_clock::now()
                    + std::chrono::seconds(5);
                while (std::chrono::steady_clock::now() < deadline) {
                    if (CpuRoceTransceiver::load_flag_acquire(
                            &response_flags[slot])
                        != 0) {
                        response_ready = true;
                        break;
                    }
                    if (duplex_monitor.wait_for(std::chrono::seconds(0))
                            == std::future_status::ready
                        || unified_monitor.wait_for(std::chrono::seconds(0))
                            == std::future_status::ready) {
                        break;
                    }
                    std::this_thread::sleep_for(
                        std::chrono::microseconds(100));
                }
                responses_ready = response_ready;
                for (std::size_t i = 0;
                     responses_match && i < kResponseSize; ++i) {
                    responses_match = response_slot[i]
                        == static_cast<std::uint8_t>(request_slot[i] ^ 0xA5);
                }
                if (response_ready) {
                    CpuRoceTransceiver::store_flag_release(
                        &response_flags[slot], 0);
                }
            }
        }

        duplex.close();
        unified.close();
        ASSERT_EQ(
            duplex_monitor.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
        ASSERT_EQ(
            unified_monitor.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
        EXPECT_NO_THROW(duplex_monitor.get());
        EXPECT_NO_THROW(unified_monitor.get());
        EXPECT_TRUE(monitors_ready);
        EXPECT_TRUE(responses_ready);
        EXPECT_TRUE(responses_match);
        EXPECT_EQ(
            dispatch_context.calls.load(std::memory_order_acquire),
            kSlotCount + 1);
        EXPECT_TRUE(
            dispatch_context.geometry_matches.load(std::memory_order_acquire));
    }

    TEST(CpuRoceTransceiverRingTest, TxWriteWithImmPlacesPayloadByImmNotWqeOrder)
    {
        const bool required
            = environment_flag("HOLOLINK_CPU_ROCE_HARDWARE_REQUIRED");
        const auto endpoints = configured_roce_endpoints();
        if (!endpoints) {
            if (required) {
                FAIL()
                    << "HOLOLINK_CPU_ROCE_HARDWARE_REQUIRED needs configured "
                       "HOLOLINK_CPU_ROCE_{LEFT,RIGHT}_{DEVICE,PORT,IP} and "
                       "a static neighbor entry between them for "
                       "left-to-right WriteWithImm";
            }
            GTEST_SKIP()
                << "Set HOLOLINK_CPU_ROCE_{LEFT,RIGHT}_{DEVICE,PORT,IP}; for "
                   "left-to-right WriteWithImm on two NICs add "
                   "a static neighbor entry between them first";
        }

        // Small Write-with-Imm: payload is placed by remote_addr against the
        // peer-visible rkey; the receive WQE is consumed only for IMM
        // notification. Independent of scatter-to-CQE. Verifies a small
        // remote write still lands on the IMM page, not the receive SGE.
        constexpr std::size_t kPayloadSize = 32;
        auto tx_config = host_config();
        tx_config.mode = CpuRoceTransceiver::Mode::Tx;
        tx_config.tx_operation
            = CpuRoceTransceiver::TxOperation::WriteWithImm;
        tx_config.ibv_name = endpoints->left.name;
        tx_config.ibv_port = endpoints->left.port;
        tx_config.local_ip = endpoints->left.ip;
        const std::uint32_t tx_slot_count = 4;
        const std::uint32_t rx_slot_count = 8;
        tx_config.tx.frame_layout.payload_size = kPayloadSize;
        tx_config.tx.slot_count = tx_slot_count;
        const std::size_t tx_stride = tx_config.tx.stride;
        const std::size_t tx_payload_size
            = tx_config.tx.frame_layout.payload_size;
        auto rx_config = host_config();
        rx_config.mode = CpuRoceTransceiver::Mode::Rx;
        rx_config.ibv_name = endpoints->right.name;
        rx_config.ibv_port = endpoints->right.port;
        rx_config.local_ip = endpoints->right.ip;
        rx_config.rx.frame_layout.payload_size = kPayloadSize;
        rx_config.rx.slot_count = rx_slot_count;
        const std::size_t rx_stride = rx_config.rx.stride;

        CpuRoceTransceiver tx(std::move(tx_config));
        CpuRoceTransceiver rx(std::move(rx_config));
        tx.setup();
        rx.setup();

        CpuRoceTransceiver::Peer to_rx;
        to_rx.qp = rx.get_qp_number();
        to_rx.ip = endpoints->right.ip;
        to_rx.rx_base = rx.external_frame_memory();
        to_rx.rx_rkey = rx.get_rkey();
        to_rx.rx_stride = rx.get_rx_ring_stride();
        to_rx.rx_slot_count = rx.get_rx_ring_slot_count();
        CpuRoceTransceiver::Peer to_tx;
        to_tx.qp = tx.get_qp_number();
        to_tx.ip = endpoints->left.ip;
        ASSERT_NO_THROW(tx.connect(to_rx));
        ASSERT_NO_THROW(rx.connect(to_tx));

        auto* const tx_data = tx.get_tx_ring_data_addr();
        auto* const tx_flags = tx.get_tx_ring_flag_addr();
        auto* const rx_data = rx.get_rx_ring_data_addr();
        auto* const rx_flags = rx.get_rx_ring_flag_addr();
        ASSERT_NE(tx_data, nullptr);
        ASSERT_NE(tx_flags, nullptr);
        ASSERT_NE(rx_data, nullptr);
        ASSERT_NE(rx_flags, nullptr);
        ASSERT_EQ(tx.get_tx_ring_slot_count(), tx_slot_count);
        ASSERT_EQ(rx.get_rx_ring_slot_count(), rx_slot_count);

        const auto fill_slot = [&](std::uint32_t slot, std::uint8_t generation) {
            auto* const data = tx_data + slot * tx_stride;
            for (std::size_t i = 0; i < tx_payload_size; ++i) {
                data[i] = static_cast<std::uint8_t>(
                    generation * 53 + slot * 17 + i);
            }
        };
        const auto publish_slot = [&](std::uint32_t slot) {
            CpuRoceTransceiver::store_flag_release(
                &tx_flags[slot],
                reinterpret_cast<std::uint64_t>(
                    tx_data + slot * tx_stride));
        };
        const auto rx_matches = [&](std::uint32_t rx_slot, std::uint32_t tx_slot,
                                    std::uint8_t generation) {
            const auto* const data = rx_data + rx_slot * rx_stride;
            for (std::size_t i = 0; i < tx_payload_size; ++i) {
                const auto expected = static_cast<std::uint8_t>(
                    generation * 53 + tx_slot * 17 + i);
                if (data[i] != expected) {
                    return false;
                }
            }
            return true;
        };

        for (std::uint32_t slot = 0; slot < tx_slot_count; ++slot) {
            fill_slot(slot, 0);
        }

        auto tx_monitor = std::async(std::launch::async, [&tx] {
            tx.blocking_monitor();
        });
        auto rx_monitor = std::async(std::launch::async, [&rx] {
            rx.blocking_monitor();
        });
        const bool tx_active
            = detail::CpuRoceMonitorTestAccess::wait_until_active(tx, 5000);
        const bool rx_active
            = detail::CpuRoceMonitorTestAccess::wait_until_active(rx, 5000);
        if (!tx_active || !rx_active) {
            tx.close();
            rx.close();
            FAIL() << "TX/RX monitor did not become active";
        }

        {
            struct CloseMonitors {
                CpuRoceTransceiver& tx;
                CpuRoceTransceiver& rx;
                ~CloseMonitors()
                {
                    tx.close();
                    rx.close();
                }
            } close_monitors { tx, rx };

            const auto rx_tail_is_empty = [&] {
                for (std::uint32_t slot = tx_slot_count;
                     slot < rx_slot_count; ++slot) {
                    if (CpuRoceTransceiver::load_flag_acquire(
                            &rx_flags[slot])
                        != 0) {
                        return false;
                    }
                }
                return true;
            };
            const auto wait_until = [&](auto&& predicate) {
                const auto deadline = std::chrono::steady_clock::now()
                    + std::chrono::seconds(5);
                while (!predicate()) {
                    // A wr_id decoder publishes RX 4-7 on the second lap.
                    // Fail immediately instead of waiting out the deadline.
                    if (!rx_tail_is_empty()) {
                        return false;
                    }
                    if (tx_monitor.wait_for(std::chrono::seconds(0))
                            == std::future_status::ready
                        || rx_monitor.wait_for(std::chrono::seconds(0))
                            == std::future_status::ready) {
                        return false;
                    }
                    if (std::chrono::steady_clock::now() >= deadline) {
                        return false;
                    }
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                }
                return true;
            };

            const auto wait_for_lap = [&](std::uint8_t generation) {
                return wait_until([&] {
                    for (std::uint32_t slot = 0; slot < tx_slot_count; ++slot) {
                        if (CpuRoceTransceiver::load_flag_acquire(
                                &rx_flags[slot])
                                == 0
                            || !rx_matches(slot, slot, generation)) {
                            return false;
                        }
                    }
                    return true;
                });
            };

            // Cyclic hole: later slots may be published early, but TX must
            // not skip slot 0. This is independent of the two-lap decoder
            // distinction: WQEs are still posted cyclically, and wr_id != IMM
            // comes from ring geometry rather than WQE posting order.
            for (const std::uint32_t slot : { 1u, 3u, 2u }) {
                publish_slot(slot);
            }
            bool gap_held = true;
            const auto gap_deadline = std::chrono::steady_clock::now()
                + std::chrono::milliseconds(50);
            while (std::chrono::steady_clock::now() < gap_deadline) {
                if (CpuRoceTransceiver::load_flag_acquire(&tx_flags[0]) != 0) {
                    gap_held = false;
                    break;
                }
                for (const std::uint32_t slot : { 1u, 3u, 2u }) {
                    if (CpuRoceTransceiver::load_flag_acquire(&tx_flags[slot])
                        == 0) {
                        gap_held = false;
                        break;
                    }
                }
                for (std::uint32_t slot = 0; slot < rx_slot_count; ++slot) {
                    if (CpuRoceTransceiver::load_flag_acquire(&rx_flags[slot])
                        != 0) {
                        gap_held = false;
                        break;
                    }
                }
                if (!gap_held) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            publish_slot(0);

            // First lap consumes recv WQEs 0,1,2,3. Release those RX flags
            // before the second lap, which consumes recv WQEs 4,5,6,7 while
            // IMM still names pages 0,1,2,3. A decoder that used wr_id would
            // publish RX 4-7 instead.
            const bool first_lap = wait_for_lap(0);
            const bool first_tail_empty = first_lap && rx_tail_is_empty();

            bool second_lap = false;
            bool second_tail_empty = false;
            if (first_lap) {
                for (std::uint32_t slot = 0; slot < tx_slot_count; ++slot) {
                    CpuRoceTransceiver::store_flag_release(&rx_flags[slot], 0);
                }
                for (std::uint32_t slot = 0; slot < tx_slot_count; ++slot) {
                    fill_slot(slot, 1);
                    publish_slot(slot);
                }
                second_lap = wait_for_lap(1);
                second_tail_empty = second_lap && rx_tail_is_empty();
            }

            EXPECT_TRUE(gap_held);
            EXPECT_TRUE(first_lap);
            EXPECT_TRUE(first_tail_empty);
            EXPECT_TRUE(second_lap);
            EXPECT_TRUE(second_tail_empty);
        }

        EXPECT_NO_THROW(tx_monitor.get());
        EXPECT_NO_THROW(rx_monitor.get());
    }

    TEST(CpuRoceTransceiverRingTest, LowMemlockSetupReportsLimits)
    {
        if (!environment_flag(
                "HOLOLINK_CPU_ROCE_EXPECT_MEMLOCK_FAILURE")) {
            GTEST_SKIP()
                << "Set HOLOLINK_CPU_ROCE_EXPECT_MEMLOCK_FAILURE=1 and run "
                   "this test with RLIMIT_MEMLOCK=0";
        }

        const auto probe = probe_local_roce_device();
        ASSERT_TRUE(probe.device) << missing_roce_message(probe);
        const auto& device = *probe.device;

        auto config = host_config();
        config.ibv_name = device.name;
        config.ibv_port = device.port;
        config.local_ip = device.ip;

        // Assumes the provider fails at CQ, MR, or QP creation, in that
        // constructor order (Context and PD are not annotated). On mlx5,
        // CQ is the first pinning allocation and is what the hardware rig
        // hits. A different provider that fails earlier (e.g. Context or
        // PD) would fail this assertion with a message that omits the
        // annotation, not with an unexpected pass.
        try {
            CpuRoceTransceiver(std::move(config)).setup();
            FAIL() << "setup unexpectedly succeeded with the low memlock limit";
        } catch (const ibv::Error& error) {
            const std::string message(error.what());
            // CQ, MR, or QP EPERM/ENOMEM include RLIMIT_MEMLOCK as
            // possible pinned-memory context, not as proof of the cause.
            // PD allocation is not annotated. Finite soft_bytes/hard_bytes
            // are bytes.
            EXPECT_NE(message.find("RLIMIT_MEMLOCK"), std::string::npos);
            EXPECT_NE(message.find("soft_bytes=0"), std::string::npos);
            EXPECT_NE(message.find("hard_bytes="), std::string::npos);
            EXPECT_NE(message.find("possible pinned-memory constraint"),
                std::string::npos);
        }
    }

    TEST(CpuRoceTransceiverRingTest, FailedSetupPublishesNoLocalResources)
    {
        auto config = host_config();
        config.ibv_name = "cpu-roce-device-that-does-not-exist";
        CpuRoceTransceiver transceiver(std::move(config));

        EXPECT_THROW(transceiver.setup(), ibv::Error);
        EXPECT_EQ(transceiver.get_qp_number(), 0);
        EXPECT_EQ(transceiver.get_rkey(), 0);
        EXPECT_EQ(transceiver.get_rx_ring_data_addr(), nullptr);
        EXPECT_EQ(transceiver.get_tx_ring_data_addr(), nullptr);
    }

    TEST(CpuRoceTransceiverRingTest, DispatchCannotBeChangedAfterClose)
    {
        auto config = host_config();
        config.mode = CpuRoceTransceiver::Mode::Unified;
        CpuRoceTransceiver transceiver(std::move(config));
        const auto dispatch
            = +[](void*, const void*, std::size_t, void*, std::size_t) {
                  return CpuRoceTransceiver::UnifiedDispatchResult {
                      CpuRoceTransceiver::UnifiedDispatchDisposition::Drop, 0
                  };
              };

        transceiver.set_unified_dispatch(dispatch, nullptr);
        transceiver.close();
        EXPECT_THROW(
            transceiver.set_unified_dispatch(dispatch, nullptr), std::logic_error);
    }

    TEST(CpuRoceTransceiverRingTest, DispatchCallbackMustNotBeNull)
    {
        auto config = host_config();
        config.mode = CpuRoceTransceiver::Mode::Unified;
        CpuRoceTransceiver transceiver(std::move(config));

        EXPECT_THROW(
            transceiver.set_unified_dispatch(nullptr, nullptr),
            std::invalid_argument);
    }

    TEST(CpuRoceTransceiverRingTest, DispatchRejectedOutsideUnifiedMode)
    {
        auto config = host_config();
        config.mode = CpuRoceTransceiver::Mode::Duplex;
        CpuRoceTransceiver transceiver(std::move(config));
        const auto dispatch
            = +[](void*, const void*, std::size_t, void*, std::size_t) {
                  return CpuRoceTransceiver::UnifiedDispatchResult {
                      CpuRoceTransceiver::UnifiedDispatchDisposition::Drop, 0
                  };
              };

        EXPECT_THROW(
            transceiver.set_unified_dispatch(dispatch, nullptr),
            std::logic_error);
    }

    TEST(CpuRoceTransceiverRingTest, FlagHelpersAcceptVolatilePointers)
    {
        detail::RxRing ring(TransceiverAllocation {}, 256, 4);

        volatile std::uint64_t* rx_flags = ring.flags();
        ASSERT_NE(rx_flags, nullptr);
        const auto address = reinterpret_cast<std::uint64_t>(ring.data());

        EXPECT_EQ(CpuRoceTransceiver::load_flag_acquire(&rx_flags[0]), 0);
        CpuRoceTransceiver::store_flag_release(&rx_flags[0], address);
        EXPECT_EQ(CpuRoceTransceiver::load_flag_acquire(&rx_flags[0]), address);
        CpuRoceTransceiver::store_flag_release(&rx_flags[0], 0);
        EXPECT_EQ(CpuRoceTransceiver::load_flag_acquire(&rx_flags[0]), 0);
    }

    TEST(CpuRoceTransceiverRingTest, SequenceWrapsAcrossSlots)
    {
        detail::TxRing ring(TransceiverAllocation {}, 64, 4);

        EXPECT_EQ(ring.slot_index(0), 0);
        EXPECT_EQ(ring.slot_index(3), 3);
        EXPECT_EQ(ring.slot_index(4), 0);
        EXPECT_EQ(ring.slot_index(9), 1);
    }

    TEST(CpuRoceTransceiverRingTest, RxRingAcceptsNonPowerOfTwoSlotCount)
    {
        detail::RxRing ring(TransceiverAllocation {}, 64, 3);
        EXPECT_EQ(ring.slot_count(), 3);
        EXPECT_EQ(ring.slot_address(2), ring.slot_address(0) + 128);
        EXPECT_EQ(ring.publish_rx(2), detail::RxPublishResult::Published);
        EXPECT_EQ(ring.publish_rx(3), detail::RxPublishResult::InvalidSlot);
    }

    TEST(CpuRoceTransceiverRingTest, TxRingRejectsNonPowerOfTwoSlotCount)
    {
        EXPECT_THROW(
            detail::TxRing(TransceiverAllocation {}, 64, 3),
            std::invalid_argument);
    }

    TEST(CpuRoceTransceiverRingTest, DirectRingStorageRejectsAddressRangeOverflow)
    {
        const auto address = reinterpret_cast<void*>(
            std::numeric_limits<std::uintptr_t>::max() - 31);
        const HostMemory memory { address, 64 };

        EXPECT_THROW(detail::RingStorage(memory, 64, 1), std::invalid_argument);
    }

    TEST(CpuRoceTransceiverRingTest, DmaBufRingSeparatesApplicationAndSgeAddresses)
    {
        const auto application_address = reinterpret_cast<void*>(0x100000);
        const DmaBufMemory memory {
            application_address, 10, 0, 256
        };
        detail::RingStorage storage(memory, 64, 4);

        EXPECT_EQ(
            storage.data(), static_cast<std::uint8_t*>(application_address));
        EXPECT_EQ(storage.slot_address(3), 0x100000u + 3u * 64u);
        EXPECT_EQ(storage.slot_sge_address(3), 3u * 64u);
        ASSERT_TRUE(storage.dmabuf().has_value());
        EXPECT_EQ(storage.dmabuf()->fd, 10);
    }

    TEST(CpuRoceTransceiverRingTest, DmaBufSgesUseZeroRegistrationIova)
    {
        const DmaBufMemory memory {
            reinterpret_cast<void*>(0x100000), 10, 0, 256
        };
        detail::RxRing rx_ring(memory, 64, 4);
        detail::TxRing tx_ring(memory, 64, 4);
        EXPECT_FALSE(rx_ring.accepts_send());
        EXPECT_FALSE(rx_ring.posts_receive_sge());

        const auto rx_sge = detail::make_local_rx_sge(
            rx_ring, 3, 32, 0x55);
        EXPECT_EQ(rx_sge.addr, 3u * 64u);
        EXPECT_EQ(rx_sge.length, 32u);
        EXPECT_EQ(rx_sge.lkey, 0x55u);

        const auto tx_sge = detail::make_local_tx_sge(
            tx_ring, 2, 32, 0x66);
        EXPECT_EQ(tx_sge.addr, 2u * 64u);
        EXPECT_EQ(tx_sge.length, 32u);
        EXPECT_EQ(tx_sge.lkey, 0x66u);
    }

    TEST(CpuRoceTransceiverRingTest, GpuPointerSeparatesRxIovaFromTxSge)
    {
        const auto pointer = reinterpret_cast<void*>(0x100000);
        const GpuPointerMemory memory { pointer, 256 };
        detail::RxRing rx_ring(memory, 64, 4);
        detail::TxRing tx_ring(memory, 64, 4);
        detail::RxRing host_ring(TransceiverAllocation {}, 64, 4);

        EXPECT_TRUE(rx_ring.gpu_pointer());
        EXPECT_FALSE(rx_ring.accepts_send());
        EXPECT_FALSE(rx_ring.posts_receive_sge());
        EXPECT_FALSE(host_ring.gpu_pointer());
        EXPECT_TRUE(host_ring.accepts_send());
        EXPECT_TRUE(host_ring.posts_receive_sge());
        EXPECT_EQ(rx_ring.slot_address(3), 0x100000u + 3u * 64u);
        EXPECT_EQ(rx_ring.slot_sge_address(3), 3u * 64u);
        EXPECT_EQ(tx_ring.slot_sge_address(3), 0x100000u + 3u * 64u);
        EXPECT_TRUE(tx_ring.gpu_pointer());
    }

    TEST(CpuRoceTransceiverRingTest, RxPublishReportsCollision)
    {
        detail::RxRing ring(TransceiverAllocation {}, 64, 2);
        EXPECT_EQ(ring.publish_rx(0), detail::RxPublishResult::Published);
        EXPECT_EQ(ring.publish_rx(0), detail::RxPublishResult::Collision);
        EXPECT_EQ(
            CpuRoceTransceiver::load_flag_acquire(&ring.flags()[0]),
            ring.slot_address(0));
    }

    TEST(CpuRoceTransceiverRingTest, RxPublishesNextCompletionAfterConsumerRelease)
    {
        detail::RxRing ring(TransceiverAllocation {}, 64, 2);
        ASSERT_EQ(ring.publish_rx(0), detail::RxPublishResult::Published);

        CpuRoceTransceiver::store_flag_release(&ring.flags()[0], 0);
        EXPECT_EQ(ring.publish_rx(0), detail::RxPublishResult::Published);
        EXPECT_EQ(
            CpuRoceTransceiver::load_flag_acquire(&ring.flags()[0]),
            ring.slot_address(0));
    }

    TEST(CpuRoceTransceiverRingTest, RxPublishRejectsInvalidExplicitSlot)
    {
        detail::RxRing ring(TransceiverAllocation {}, 64, 2);

        EXPECT_EQ(ring.publish_rx(2), detail::RxPublishResult::InvalidSlot);
        EXPECT_EQ(CpuRoceTransceiver::load_flag_acquire(&ring.flags()[0]), 0);
        EXPECT_EQ(CpuRoceTransceiver::load_flag_acquire(&ring.flags()[1]), 0);
    }

    TEST(CpuRoceTransceiverRingTest, SelectedGidIdentityMatchesUnchangedEntry)
    {
        ::ibv_gid_entry entry {};
        entry.gid = ibv::ipv4_to_gid("192.0.2.10");
        entry.gid_type = IBV_GID_TYPE_ROCE_V2;
        entry.ndev_ifindex = 7;
        const detail::LocalRoceV2Ipv4Gid selected {
            entry.gid, 3, 7, "eth-test"
        };

        EXPECT_TRUE(detail::selected_gid_matches(selected, entry));
    }

    TEST(CpuRoceTransceiverRingTest, SelectedGidIdentityRejectsPathChanges)
    {
        ::ibv_gid_entry entry {};
        entry.gid = ibv::ipv4_to_gid("192.0.2.10");
        entry.gid_type = IBV_GID_TYPE_ROCE_V2;
        entry.ndev_ifindex = 7;
        const detail::LocalRoceV2Ipv4Gid selected {
            entry.gid, 3, 7, "eth-test"
        };

        EXPECT_FALSE(detail::selected_gid_matches(selected, std::nullopt));

        auto changed = entry;
        changed.gid = ibv::ipv4_to_gid("192.0.2.11");
        EXPECT_FALSE(detail::selected_gid_matches(selected, changed));

        changed = entry;
        changed.gid_type = IBV_GID_TYPE_IB;
        EXPECT_FALSE(detail::selected_gid_matches(selected, changed));

        changed = entry;
        changed.ndev_ifindex = 8;
        EXPECT_FALSE(detail::selected_gid_matches(selected, changed));

        changed = entry;
        changed.ndev_ifindex = 0;
        EXPECT_FALSE(detail::selected_gid_matches(selected, changed));
    }

    TEST(CpuRoceTransceiverRingTest, AsyncEventPolicyClassifiesKnownEvents)
    {
        using Disposition = detail::AsyncEventDisposition;
        constexpr std::uint32_t kDevicePort = 2;

        for (const auto event : {
                 IBV_EVENT_CQ_ERR,
                 IBV_EVENT_QP_FATAL,
                 IBV_EVENT_QP_REQ_ERR,
                 IBV_EVENT_QP_ACCESS_ERR,
                 IBV_EVENT_DEVICE_FATAL,
             }) {
            EXPECT_EQ(
                detail::classify_async_event(event, 0, kDevicePort),
                Disposition::Terminal);
        }
        for (const auto event : {
                 IBV_EVENT_COMM_EST,
                 IBV_EVENT_PORT_ACTIVE,
                 IBV_EVENT_LID_CHANGE,
                 IBV_EVENT_PKEY_CHANGE,
                 IBV_EVENT_SM_CHANGE,
                 IBV_EVENT_CLIENT_REREGISTER,
             }) {
            EXPECT_EQ(
                detail::classify_async_event(event, kDevicePort, kDevicePort),
                Disposition::Silent);
        }
        for (const auto event : {
                 IBV_EVENT_SQ_DRAINED,
                 IBV_EVENT_PATH_MIG,
                 IBV_EVENT_PATH_MIG_ERR,
                 IBV_EVENT_SRQ_ERR,
                 IBV_EVENT_SRQ_LIMIT_REACHED,
                 IBV_EVENT_QP_LAST_WQE_REACHED,
             }) {
            EXPECT_EQ(
                detail::classify_async_event(event, 0, kDevicePort),
                Disposition::UnexpectedInvariant);
        }
        EXPECT_EQ(
            detail::classify_async_event(
                IBV_EVENT_PORT_ERR, kDevicePort, kDevicePort),
            Disposition::Terminal);
        EXPECT_EQ(
            detail::classify_async_event(
                IBV_EVENT_PORT_ERR, kDevicePort + 1, kDevicePort),
            Disposition::Silent);
        EXPECT_EQ(
            detail::classify_async_event(
                IBV_EVENT_GID_CHANGE, kDevicePort, kDevicePort),
            Disposition::ValidateSelectedGid);
        EXPECT_EQ(
            detail::classify_async_event(
                IBV_EVENT_GID_CHANGE, kDevicePort + 1, kDevicePort),
            Disposition::Silent);
    }

    TEST(CpuRoceTransceiverRingTest, AsyncEventPolicyHandlesExtendedAbi)
    {
        using Disposition = detail::AsyncEventDisposition;
        const int gid_change = static_cast<int>(IBV_EVENT_GID_CHANGE);

        // WQ_FATAL occupies the intervening ABI value and stays terminal even
        // when compiling against a header too old to name it.
        EXPECT_EQ(
            detail::classify_async_event(
                static_cast<::ibv_event_type>(gid_change + 1), 0, 1),
            Disposition::Terminal);
        // rdma-core v62 appended DEVICE_SPEED_CHANGE at GID_CHANGE + 2.
        EXPECT_EQ(
            detail::classify_async_event(
                static_cast<::ibv_event_type>(gid_change + 2), 0, 1),
            Disposition::Silent);
        // ibv_event_type has no fixed underlying type, so the cast must stay
        // inside the enum's representable range. +3 is unknown on both older
        // headers (max GID_CHANGE) and v62 (max DEVICE_SPEED_CHANGE).
        EXPECT_EQ(
            detail::classify_async_event(
                static_cast<::ibv_event_type>(gid_change + 3), 0, 1),
            Disposition::Terminal);
    }

    TEST(CpuRoceTransceiverRingTest, SendWorkRequestConfiguresInlineAndSignaled)
    {
        ibv::ScatterGatherElement sge(0x12340000, 128, 0x55);
        ibv::SendWorkRequest work_request(42, &sge);
        work_request.inline_data(true);

        EXPECT_EQ(work_request.wr_id, 42);
        EXPECT_EQ(work_request.opcode, IBV_WR_SEND);
        EXPECT_EQ(
            work_request.send_flags,
            IBV_SEND_SIGNALED | IBV_SEND_INLINE);
        EXPECT_EQ(work_request.sg_list, &sge);
        EXPECT_EQ(work_request.num_sge, 1);
        EXPECT_EQ(sge.addr, 0x12340000);
        EXPECT_EQ(sge.length, 128);
        EXPECT_EQ(sge.lkey, 0x55);

        work_request.inline_data(false);
        EXPECT_EQ(work_request.send_flags, IBV_SEND_SIGNALED);
    }

    TEST(CpuRoceTransceiverRingTest, WriteWithImmWorkRequestSetsRemoteAndImm)
    {
        ibv::ScatterGatherElement sge(0x2000, 64, 0x11);
        ibv::WriteWithImmWorkRequest work_request(
            7, &sge, 0x1000, 0x22, htonl(3));
        work_request.inline_data(true);

        EXPECT_EQ(work_request.wr_id, 7);
        EXPECT_EQ(work_request.opcode, IBV_WR_RDMA_WRITE_WITH_IMM);
        EXPECT_EQ(
            work_request.send_flags,
            IBV_SEND_SIGNALED | IBV_SEND_INLINE);
        EXPECT_EQ(work_request.wr.rdma.remote_addr, 0x1000);
        EXPECT_EQ(work_request.wr.rdma.rkey, 0x22);
        EXPECT_EQ(work_request.imm_data, htonl(3));
        EXPECT_EQ(work_request.sg_list, &sge);
        EXPECT_EQ(work_request.num_sge, 1);

        work_request.inline_data(false);
        EXPECT_EQ(work_request.send_flags, IBV_SEND_SIGNALED);
    }

    ::ibv_wc make_recv_cqe(
        ibv_wc_opcode opcode,
        std::uint64_t wr_id,
        std::uint32_t host_imm = 0,
        int wc_flags = 0,
        ibv_wc_status status = IBV_WC_SUCCESS)
    {
        ::ibv_wc wc {};
        wc.status = status;
        wc.opcode = opcode;
        wc.wr_id = wr_id;
        wc.wc_flags = wc_flags;
        wc.imm_data = htonl(host_imm);
        return wc;
    }

    TEST(CpuRoceTransceiverRingTest, TxCqeRequiresExactPostedSequence)
    {
        ::ibv_wc wc {};
        wc.status = IBV_WC_SUCCESS;
        wc.opcode = IBV_WC_SEND;
        wc.wr_id = 7;

        const auto decoded = detail::decode_tx_cqe(wc, 7);
        EXPECT_EQ(decoded.status, detail::TxCqeDecode::Status::Ok);
        EXPECT_EQ(decoded.sequence, 7);

        EXPECT_EQ(
            detail::decode_tx_cqe(wc, 6).status,
            detail::TxCqeDecode::Status::OutOfSequence);
    }

    TEST(CpuRoceTransceiverRingTest, TxCqeValidatesStatusBeforeOpcode)
    {
        ::ibv_wc wc {};
        wc.status = IBV_WC_GENERAL_ERR;
        wc.opcode = IBV_WC_RECV;
        wc.wr_id = 0;

        EXPECT_EQ(
            detail::decode_tx_cqe(wc, 0).status,
            detail::TxCqeDecode::Status::FailedStatus);

        wc.status = IBV_WC_SUCCESS;
        EXPECT_EQ(
            detail::decode_tx_cqe(wc, 0).status,
            detail::TxCqeDecode::Status::UnsupportedOpcode);

        wc.opcode = IBV_WC_RDMA_WRITE;
        EXPECT_EQ(
            detail::decode_tx_cqe(wc, 0, IBV_WC_RDMA_WRITE).status,
            detail::TxCqeDecode::Status::Ok);

        wc.opcode = IBV_WC_SEND;
        EXPECT_EQ(
            detail::decode_tx_cqe(wc, 0, IBV_WC_RDMA_WRITE).status,
            detail::TxCqeDecode::Status::UnsupportedOpcode);
    }

    TEST(CpuRoceTransceiverRingTest, RxCqeSendUsesWrId)
    {
        const auto wc = make_recv_cqe(IBV_WC_RECV, 2);
        const auto decoded = detail::decode_rx_cqe(wc, 4, false);
        EXPECT_EQ(decoded.status, detail::RxCqeDecode::Status::Ok);
        EXPECT_EQ(decoded.payload_slot, 2);
        EXPECT_EQ(decoded.recv_wr_id, 2);

        const auto completion = detail::make_rx_completion(
            wc, static_cast<std::uint32_t>(decoded.recv_wr_id));
        EXPECT_EQ(completion.credit_id, 2u);
        EXPECT_FALSE(completion.immediate_data.has_value());
    }

    TEST(CpuRoceTransceiverRingTest, RxCqeCreditOnlyRejectsSend)
    {
        const auto send = make_recv_cqe(IBV_WC_RECV, 2);
        EXPECT_EQ(
            detail::decode_rx_cqe(send, 4, true).status,
            detail::RxCqeDecode::Status::UnexpectedSendForCreditOnly);

        const auto write_with_imm = make_recv_cqe(
            IBV_WC_RECV_RDMA_WITH_IMM, 2, 3, IBV_WC_WITH_IMM);
        const auto decoded
            = detail::decode_rx_cqe(write_with_imm, 4, true);
        EXPECT_EQ(decoded.status, detail::RxCqeDecode::Status::Ok);
        EXPECT_EQ(decoded.payload_slot, 3);
        EXPECT_EQ(decoded.recv_wr_id, 2);
    }

    TEST(CpuRoceTransceiverRingTest, RxCqeSendWithImmStillUsesWrId)
    {
        const auto wc = make_recv_cqe(
            IBV_WC_RECV, 1, 0xABC003, IBV_WC_WITH_IMM);
        const auto decoded = detail::decode_rx_cqe(wc, 4, false);
        EXPECT_EQ(decoded.status, detail::RxCqeDecode::Status::Ok);
        EXPECT_EQ(decoded.payload_slot, 1);
        EXPECT_EQ(decoded.recv_wr_id, 1);

        const auto completion = detail::make_rx_completion(
            wc, static_cast<std::uint32_t>(decoded.recv_wr_id));
        EXPECT_EQ(completion.credit_id, 1u);
        ASSERT_TRUE(completion.immediate_data.has_value());
        EXPECT_EQ(*completion.immediate_data, 0xABC003);
    }

    TEST(CpuRoceTransceiverRingTest, RxCqeWriteWithImmUsesImmPage)
    {
        const auto wc = make_recv_cqe(
            IBV_WC_RECV_RDMA_WITH_IMM, 2, 0xABC003, IBV_WC_WITH_IMM);
        const auto decoded = detail::decode_rx_cqe(wc, 8, false);
        EXPECT_EQ(decoded.status, detail::RxCqeDecode::Status::Ok);
        EXPECT_EQ(decoded.payload_slot, 3);
        EXPECT_EQ(decoded.recv_wr_id, 2);

        const auto completion = detail::make_rx_completion(
            wc, static_cast<std::uint32_t>(decoded.recv_wr_id));
        EXPECT_EQ(completion.credit_id, 2u);
        ASSERT_TRUE(completion.immediate_data.has_value());
        EXPECT_EQ(*completion.immediate_data, 0xABC003);
    }

    TEST(CpuRoceTransceiverRingTest, RxCqeWriteWithImmWqeOrderDiffersFromPage)
    {
        const auto first = make_recv_cqe(
            IBV_WC_RECV_RDMA_WITH_IMM, 0, 3, IBV_WC_WITH_IMM);
        const auto second = make_recv_cqe(
            IBV_WC_RECV_RDMA_WITH_IMM, 1, 0, IBV_WC_WITH_IMM);
        const auto decoded_first = detail::decode_rx_cqe(first, 4, false);
        const auto decoded_second = detail::decode_rx_cqe(second, 4, false);
        EXPECT_EQ(decoded_first.payload_slot, 3);
        EXPECT_EQ(decoded_first.recv_wr_id, 0);
        EXPECT_EQ(decoded_second.payload_slot, 0);
        EXPECT_EQ(decoded_second.recv_wr_id, 1);
    }

    TEST(CpuRoceTransceiverRingTest, RxCqeRejectsOutOfRangeImmWithoutWrap)
    {
        const auto wc = make_recv_cqe(
            IBV_WC_RECV_RDMA_WITH_IMM, 0, 5, IBV_WC_WITH_IMM);
        const auto decoded = detail::decode_rx_cqe(wc, 4, false);
        EXPECT_EQ(
            decoded.status, detail::RxCqeDecode::Status::InvalidImmSlot);
        EXPECT_EQ(decoded.recv_wr_id, 0);

        // The handled path does not range-check the immediate page. The
        // receiver drops it and the credit is reposted.
        const auto handled = detail::decode_handled_rx_cqe(wc, 4);
        EXPECT_EQ(handled.status, detail::RxCqeDecode::Status::Ok);
        EXPECT_EQ(handled.recv_wr_id, 0);
        const auto completion = detail::make_rx_completion(wc, 0);
        ASSERT_TRUE(completion.immediate_data.has_value());
        EXPECT_EQ(*completion.immediate_data, 5u);
        EXPECT_EQ(completion.credit_id, 0u);
    }

    TEST(CpuRoceTransceiverRingTest, HandledRxCqeRejectsACreditOutsideThePool)
    {
        const auto wc = make_recv_cqe(
            IBV_WC_RECV_RDMA_WITH_IMM, 4, 1, IBV_WC_WITH_IMM);
        EXPECT_EQ(
            detail::decode_handled_rx_cqe(wc, 4).status,
            detail::RxCqeDecode::Status::InvalidWrId);
        const auto missing = make_recv_cqe(
            IBV_WC_RECV_RDMA_WITH_IMM, 0, 1, 0);
        EXPECT_EQ(
            detail::decode_handled_rx_cqe(missing, 4).status,
            detail::RxCqeDecode::Status::MissingImm);
        const auto send = make_recv_cqe(IBV_WC_RECV, 0);
        const auto send_with_imm = make_recv_cqe(
            IBV_WC_RECV, 0, 1, IBV_WC_WITH_IMM);
        EXPECT_EQ(
            detail::decode_handled_rx_cqe(send, 4).status,
            detail::RxCqeDecode::Status::UnexpectedSend);
        EXPECT_EQ(
            detail::decode_handled_rx_cqe(send_with_imm, 4).status,
            detail::RxCqeDecode::Status::UnexpectedSend);
    }

    TEST(CpuRoceTransceiverRingTest, RxCqeLostWriteWithImmDoesNotShiftLaterSlots)
    {
        const auto first = make_recv_cqe(
            IBV_WC_RECV_RDMA_WITH_IMM, 0, 3, IBV_WC_WITH_IMM);
        const auto second = make_recv_cqe(
            IBV_WC_RECV_RDMA_WITH_IMM, 1, 1, IBV_WC_WITH_IMM);
        EXPECT_EQ(detail::decode_rx_cqe(first, 4, false).payload_slot, 3);
        EXPECT_EQ(detail::decode_rx_cqe(second, 4, false).payload_slot, 1);
    }

    TEST(CpuRoceTransceiverRingTest, RxCqeRequiresImmOnWriteWithImm)
    {
        const auto wc = make_recv_cqe(IBV_WC_RECV_RDMA_WITH_IMM, 0, 3, 0);
        EXPECT_EQ(
            detail::decode_rx_cqe(wc, 4, false).status,
            detail::RxCqeDecode::Status::MissingImm);
    }

    TEST(CpuRoceTransceiverRingTest, RxCqeRejectsFailedStatusBeforeOpcode)
    {
        const auto wc = make_recv_cqe(
            IBV_WC_RECV_RDMA_WITH_IMM, 0, 1, IBV_WC_WITH_IMM, IBV_WC_GENERAL_ERR);
        EXPECT_EQ(
            detail::decode_rx_cqe(wc, 4, false).status,
            detail::RxCqeDecode::Status::FailedStatus);
    }

    TEST(CpuRoceTransceiverRingTest, RxCqeRejectsSendWithOutOfRangeWrId)
    {
        const auto wc = make_recv_cqe(IBV_WC_RECV, 4);
        EXPECT_EQ(
            detail::decode_rx_cqe(wc, 4, false).status,
            detail::RxCqeDecode::Status::InvalidWrId);
    }

    TEST(CpuRoceTransceiverRingTest, RxCqeRejectsWriteWithImmWithOutOfRangeWrId)
    {
        const auto wc = make_recv_cqe(
            IBV_WC_RECV_RDMA_WITH_IMM, 4, 1, IBV_WC_WITH_IMM);
        EXPECT_EQ(
            detail::decode_rx_cqe(wc, 4, false).status,
            detail::RxCqeDecode::Status::InvalidWrId);
    }

    TEST(CpuRoceTransceiverRingTest,
        RxCqeRejectsWrIdBeforeNarrowingToSlot)
    {
        constexpr std::uint64_t wide_wr_id = std::uint64_t { 1 } << 32;
        for (const auto opcode :
            { IBV_WC_RECV, IBV_WC_RECV_RDMA_WITH_IMM }) {
            const auto wc = make_recv_cqe(
                opcode, wide_wr_id, 1, IBV_WC_WITH_IMM);
            EXPECT_EQ(
                detail::decode_rx_cqe(wc, 4, false).status,
                detail::RxCqeDecode::Status::InvalidWrId);
        }
    }

    TEST(CpuRoceTransceiverRingTest, RxCqeRejectsUnsupportedOpcode)
    {
        const auto wc = make_recv_cqe(IBV_WC_SEND, 0);
        EXPECT_EQ(
            detail::decode_rx_cqe(wc, 4, false).status,
            detail::RxCqeDecode::Status::UnsupportedOpcode);
    }

    TEST(CpuRoceTransceiverRingTest, LocalRxSgeUsesVirtualSlotAddress)
    {
        detail::RxRing ring(TransceiverAllocation {}, 256, 4);

        const auto sge = detail::make_local_rx_sge(ring, 3, 128, 0x55);

        EXPECT_EQ(sge.addr, ring.slot_address(3));
        EXPECT_EQ(sge.length, 128);
        EXPECT_EQ(sge.lkey, 0x55);
    }

    TEST(CpuRoceTransceiverRingTest, LocalTxSgeUsesVirtualSlotAddress)
    {
        detail::TxRing ring(TransceiverAllocation {}, 256, 4);

        const auto sge = detail::make_local_tx_sge(ring, 3, 128, 0x66);

        EXPECT_EQ(sge.addr, ring.slot_address(3));
        EXPECT_EQ(sge.length, 128);
        EXPECT_EQ(sge.lkey, 0x66);
    }

    TEST(CpuRoceTransceiverRingTest, RxCollisionCountStartsAtZero)
    {
        CpuRoceTransceiver transceiver(host_config());
        EXPECT_EQ(transceiver.get_rx_collision_count(), 0);
        transceiver.close();
        EXPECT_EQ(transceiver.get_rx_collision_count(), 0);
    }

    TEST(CpuRoceTransceiverRingTest, TxClaimReportsNotReady)
    {
        detail::TxRing ring(TransceiverAllocation {}, 64, 2);

        EXPECT_EQ(ring.try_claim_tx(0), detail::TxClaimResult::NotReady);
        EXPECT_EQ(CpuRoceTransceiver::load_flag_acquire(&ring.flags()[0]), 0);
    }

    TEST(CpuRoceTransceiverRingTest, TxClaimSlotDoesNotWrapOutOfRangeIndex)
    {
        detail::TxRing ring(TransceiverAllocation {}, 64, 2);
        CpuRoceTransceiver::store_flag_release(
            &ring.flags()[0], ring.slot_address(0));

        EXPECT_THROW(ring.try_claim_tx(2), std::out_of_range);
        EXPECT_EQ(
            CpuRoceTransceiver::load_flag_acquire(&ring.flags()[0]),
            ring.slot_address(0));
        EXPECT_EQ(ring.try_claim_tx(0), detail::TxClaimResult::Claimed);
    }

    TEST(CpuRoceTransceiverRingTest, TxClaimNotReadyLeavesOtherSlotsPublished)
    {
        detail::TxRing ring(TransceiverAllocation {}, 64, 4);
        for (const std::uint32_t slot : { 1u, 3u, 2u }) {
            CpuRoceTransceiver::store_flag_release(
                &ring.flags()[slot], ring.slot_address(slot));
        }

        EXPECT_EQ(ring.try_claim_tx(0), detail::TxClaimResult::NotReady);
        for (const std::uint32_t slot : { 1u, 3u, 2u }) {
            EXPECT_EQ(
                CpuRoceTransceiver::load_flag_acquire(&ring.flags()[slot]),
                ring.slot_address(slot));
        }
    }

    TEST(CpuRoceTransceiverRingTest, TxClaimConsumesEachPublication)
    {
        detail::TxRing ring(TransceiverAllocation {}, 64, 2);
        const std::uint64_t slot_address = ring.slot_address(1);
        CpuRoceTransceiver::store_flag_release(&ring.flags()[1], slot_address);

        ASSERT_EQ(ring.try_claim_tx(1), detail::TxClaimResult::Claimed);
        EXPECT_EQ(CpuRoceTransceiver::load_flag_acquire(&ring.flags()[1]), 0);

        CpuRoceTransceiver::store_flag_release(
            &ring.flags()[1], slot_address);
        EXPECT_EQ(
            CpuRoceTransceiver::load_flag_acquire(&ring.flags()[1]),
            slot_address);

        // TxRing owns only the public doorbell. The production TX loop's
        // sequence-gap check, tested below, gates WQE lifetime before claim.
        ASSERT_EQ(ring.try_claim_tx(1), detail::TxClaimResult::Claimed);
        EXPECT_EQ(CpuRoceTransceiver::load_flag_acquire(&ring.flags()[1]), 0);
    }

    TEST(CpuRoceTransceiverRingTest, TxRejectsAddressThatIsNotSlotBase)
    {
        detail::TxRing ring(TransceiverAllocation {}, 64, 2);
        CpuRoceTransceiver::store_flag_release(
            &ring.flags()[0], ring.slot_address(0) + 1);

        EXPECT_THROW(ring.try_claim_tx(0), std::runtime_error);
        EXPECT_EQ(CpuRoceTransceiver::load_flag_acquire(&ring.flags()[0]),
            ring.slot_address(0) + 1);
    }

    TEST(CpuRoceTransceiverRingTest, TxCyclicWindowCapacityBoundsOneRingLap)
    {
        EXPECT_TRUE(detail::cyclic_tx_window_has_capacity(0, 0, 1));
        EXPECT_FALSE(detail::cyclic_tx_window_has_capacity(1, 0, 1));

        EXPECT_TRUE(detail::cyclic_tx_window_has_capacity(3, 0, 4));
        EXPECT_FALSE(detail::cyclic_tx_window_has_capacity(4, 0, 4));
        EXPECT_TRUE(detail::cyclic_tx_window_has_capacity(5, 2, 4));
    }

    TEST(CpuRoceTransceiverRingTest, TxCyclicWindowCapacityHandlesSequenceWrap)
    {
        constexpr auto max_sequence
            = std::numeric_limits<std::uint64_t>::max();

        EXPECT_TRUE(
            detail::cyclic_tx_window_has_capacity(1, max_sequence - 1, 4));
        EXPECT_FALSE(
            detail::cyclic_tx_window_has_capacity(1, max_sequence - 2, 4));
    }

    CpuRoceTransceiver::Config caller_tx_config(CpuRoceTransceiver::Mode mode)
    {
        auto config = host_config();
        config.mode = mode;
        config.tx_operation = CpuRoceTransceiver::TxOperation::Send;
        return config;
    }

    TEST(CpuRoceCallerTxContract, ConfigureAndFailGuards)
    {
        using detail::CpuRoceCallerTx;
        {
            CpuRoceTransceiver transceiver(
                caller_tx_config(CpuRoceTransceiver::Mode::Rx));
            CpuRoceCallerTx caller(transceiver);
            EXPECT_THROW(caller.fail(), std::logic_error);
            EXPECT_THROW(caller.configure({}), std::invalid_argument);
            EXPECT_FALSE(caller.configured());
        }
        {
            auto config = caller_tx_config(CpuRoceTransceiver::Mode::Tx);
            config.tx_operation = CpuRoceTransceiver::TxOperation::WriteWithImm;
            CpuRoceTransceiver transceiver(std::move(config));
            CpuRoceCallerTx caller(transceiver);
            EXPECT_THROW(caller.configure({}), std::invalid_argument);
            EXPECT_THROW(caller.fail(), std::logic_error);
        }
        {
            CpuRoceTransceiver transceiver(
                caller_tx_config(CpuRoceTransceiver::Mode::Tx));
            CpuRoceCallerTx caller(transceiver);
            EXPECT_THROW(caller.fail(), std::logic_error);
            CpuRoceCallerTx::Config extra;
            extra.extra_host_bytes = 128;
            EXPECT_THROW(caller.configure(extra), std::invalid_argument);
            caller.configure({});
            EXPECT_TRUE(caller.configured());
            EXPECT_THROW(caller.configure({}), std::logic_error);
            caller.fail();
            EXPECT_THROW(caller.configure({}), std::logic_error);
        }
    }

} // namespace
} // namespace hololink::transport::roce
