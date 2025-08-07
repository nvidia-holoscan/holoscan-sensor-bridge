/**
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include <atomic>
#include <cstdint>
#include <thread>

#include "hololink.hpp"

namespace hololink {

/**
 * Interface for an object that provides JESD configuration.
 */
class JESDConfig {
public:
    virtual ~JESDConfig() {};

    virtual void power_on() = 0;
    virtual void setup_clocks() = 0;
    virtual void configure() = 0;
    virtual void run() = 0;
};

/**
 * Maintains a connection to an HSB SPI Daemon in order to perform SPI
 * communication and respond to JESD204 FSM callbacks.
 *
 * This class is meant to be used by JESDConfig objects which perform
 * device configuration using kernel drivers in a VM (e.g. AD9986Config).
 *
 * When the target VM devices are registered as JESD devices within an
 * ADI JESD204 FSM, the JESDConfig object passed to this thread will be
 * called when the corresponding JESD state transitions are triggered
 * by the drivers in the VM.
 */
class SpiDaemonThread {
public:
    explicit SpiDaemonThread(Hololink& hololink, JESDConfig& jesd_config);

    /**
     * Starts the thread and blocks until the SPI daemon is connected.
     */
    void run();

    /**
     * Stops the thread and blocks until it's complete.
     */
    void stop();

protected:
    // Message types and definition, must match SPI Daemon's definition.
    const int HSB_SPI_MSG_TYPE_SPI = 0;
    const int HSB_SPI_MSG_TYPE_JESD = 1;
    struct hsb_spi_message {
        uint8_t type;
        union {
            struct {
                uint8_t id_cs;
                uint8_t cmd_bytes;
                uint8_t wr_bytes;
                uint8_t rd_bytes;
            } spi;
            struct {
                uint8_t id;
            } jesd;
        } u;
    };

    // List of JESD204 states, per the ADI JESD204 FSM.
    enum jesd204_op {
        JESD204_OP_DEVICE_INIT,
        JESD204_OP_LINK_INIT,
        JESD204_OP_LINK_SUPPORTED,
        JESD204_OP_LINK_PRE_SETUP,
        JESD204_OP_CLK_SYNC_STAGE1,
        JESD204_OP_CLK_SYNC_STAGE2,
        JESD204_OP_CLK_SYNC_STAGE3,
        JESD204_OP_LINK_SETUP,
        JESD204_OP_OPT_SETUP_STAGE1,
        JESD204_OP_OPT_SETUP_STAGE2,
        JESD204_OP_OPT_SETUP_STAGE3,
        JESD204_OP_OPT_SETUP_STAGE4,
        JESD204_OP_OPT_SETUP_STAGE5,
        JESD204_OP_CLOCKS_ENABLE,
        JESD204_OP_LINK_ENABLE,
        JESD204_OP_LINK_RUNNING,
        JESD204_OP_OPT_POST_RUNNING_STAGE,
    };

    Hololink& hololink_;
    JESDConfig& jesd_config_;
    std::thread thread_;
    std::atomic<bool> running_;

    void thread_func();
    int execute_jesd(int jesd_state);
};

/**
 * Configuration for an ADI 9986 + HMC 7044 MxFE.
 * This configuration is supported by a SpiDaemonThread object connected
 * to an HSB SPI Daemon instance running inside an ADI Petalinux VM.
 */
class AD9986Config : public JESDConfig {
public:
    explicit AD9986Config(Hololink& hololink);

    // Set the host pause mapping for the AD9986.
    // 0x01: Interface 0
    // 0x02: Interface 1
    void host_pause_mapping(uint32_t mask);

    // Applies the current configuration.
    void apply();

    // JESDConfig functions.
    virtual void power_on();
    virtual void setup_clocks();
    virtual void configure();
    virtual void run();

private:
    Hololink& hololink_;
    std::unique_ptr<SpiDaemonThread> spi_daemon_thread_;
    std::atomic<bool> jesd_configured_;

    bool task_refclk_sw(uint32_t channel, uint32_t refclk, uint32_t hwseq);
    bool task_set_pma_attribute(uint32_t channel, uint32_t code, uint32_t data);
    void task_pma_analog_reset(uint32_t channel);
    void configure_nvda_jesd_tx(void);
    void configure_nvda_jesd_rx(void);

    uint32_t host_pause_mapping_mask_ = 0;
};

} // namespace hololink
