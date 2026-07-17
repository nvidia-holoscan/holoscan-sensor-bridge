/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Status-LED application built on the hololink_module API.
 *
 * Runnable companion to the "Device Module Tutorial" user guide. It blinks
 * the Tutorial device's status LED once per second through the device's
 * bespoke service and the hololink_module reactor. There is no data plane:
 * the application only touches the control plane and the LED.
 */

#include <chrono>
#include <functional>
#include <memory>
#include <stdexcept>

#include <unistd.h>

#include "hololink/module/adapter.hpp"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/hololink.hpp"
#include "hololink/module/reactor.hpp"
#include "hololink/module/tutorial/tutorial_device.hpp"

// The board announces itself over bootp; this is the peer IP we wait for.
static const char* const HOLOLINK_IP = "192.168.0.2";
// Seconds to wait for that announcement before giving up.
static constexpr std::chrono::seconds DISCOVERY_TIMEOUT(30);
// Status-LED toggle period.
static constexpr float BLINK_PERIOD_S = 1.0f;

int main()
{
    // Find the board via bootp enumeration and get its metadata.
    auto& adapter = hololink::module::Adapter::get_adapter();
    hololink::module::EnumerationMetadata metadata
        = adapter.wait_for_channel(HOLOLINK_IP, DISCOVERY_TIMEOUT);

    // Bring up the control plane; every register access flows through it.
    auto hololink = hololink::module::HololinkInterfaceV1::get_service(metadata);
    if (hololink->start() != HOLOLINK_MODULE_OK) {
        throw std::runtime_error("HololinkInterface::start failed");
    }
    if (hololink->reset() != HOLOLINK_MODULE_OK) {
        throw std::runtime_error("HololinkInterface::reset failed");
    }

    // Fetch the Tutorial device's bespoke status-LED service.
    auto device
        = hololink::module::tutorial::TutorialDeviceInterfaceV1::get_service(metadata);

    // Reactor alarms are one-shot, so the callback re-arms itself to blink
    // once per second. The reactor runs the callback on its own thread.
    auto reactor = hololink::module::ReactorV1::get_service(
        adapter.host_publisher()->self_module());
    auto state = std::make_shared<bool>(false);
    auto tick = std::make_shared<hololink::module::ReactorV1::Callback>();
    *tick = [device, state, reactor, tick]() {
        *state = !*state;
        device->set_status_led(*state);
        reactor->add_alarm_s(BLINK_PERIOD_S, tick);
    };
    reactor->add_alarm_s(BLINK_PERIOD_S, tick);

    // The reactor thread drives the blinking; this program runs forever.
    for (;;) {
        sleep(60);
    }
}
