/**
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
 *
 * See README.md for detailed information.
 */

#include "STM32/data_plane.hpp"
#include "STM32/hsb_emulator.hpp"
#include "STM32/net.hpp"
#include "STM32/stm32_system.h"
#include "STM32/tim.h"
#include "utils.hpp"
#include <string.h>
#include <time.h>

namespace hololink::emulation {

// File-scope static pool of STM32 DataPlane contexts; one slot per data_plane_id. Claimed
// by the DataPlane constructor when `ctxt == nullptr` (the typical case for application code).
// No new/malloc — this is the STM32 way.
struct STM32DataPlaneCtxt DATA_PLANE_CTXT[MAX_DATA_PLANES];

// Per-data-plane and per-sensor register slices, shared by every DataPlane bound to the same
// data_plane_id (resp. sensor_id). The DataPlane constructor wires
//   data_plane_ctxt_->dp_registers        = &DP_REGISTERS[data_plane_id_]
//   data_plane_ctxt_->dp_sensor_registers = &DP_SENSOR_REGISTER[sensor_id_]
// Pre-allocated alongside DATA_PLANE_CTXT so they have the same static lifetime as the rest
// of the STM32 emulator state.
struct DPRegisters DP_REGISTERS[MAX_DATA_PLANES] = {};
struct DPSensorRegisters DP_SENSOR_REGISTER[MAX_SENSORS] = {};

// Helper: downcast the base-class DataPlaneCtxt* to the STM32 extension. Standard-layout
// guarantees zero offset between STM32DataPlaneCtxt and its first `base` member.
static inline STM32DataPlaneCtxt* stm32_ctxt(DataPlaneCtxt* base)
{
    return reinterpret_cast<STM32DataPlaneCtxt*>(base);
}

static void init_bootp_tx_config(STM32DataPlaneCtxt* ctxt)
{
    // initialize the tx buffer attached to the config
    ctxt->tx_buffers = {
        .buffer = &(ctxt->bootp_buffer[0]),
        .len = sizeof(ctxt->bootp_buffer),
        .next = NULL,
    };

    ctxt->tx_config = {
        .Attributes = ETH_TX_PACKETS_FEATURES_CSUM | ETH_TX_PACKETS_FEATURES_CRCPAD,
        .Length = ctxt->tx_buffers.len,
        .TxBuffer = &ctxt->tx_buffers,
        .CRCPadCtrl = ETH_CRC_PAD_INSERT,
        .ChecksumCtrl = ETH_CHECKSUM_IPHDR_PAYLOAD_INSERT_PHDR_CALC,
    };
}

// Build the platform-default context for the public ctor: claim the slot from the file-scope
// DATA_PLANE_CTXT[] static pool corresponding to data_plane_id, and wrap it in a unique_ptr
// with a no-op deleter (static lifetime; never `delete`'d).
static std::unique_ptr<DataPlaneCtxt, std::function<void(DataPlaneCtxt*)>>
make_default_stm32_ctxt(uint8_t data_plane_id)
{
    if (data_plane_id >= MAX_DATA_PLANES) {
        Error_Handler("data_plane_id exceeds maximum number of data planes");
    }
    return {
        &DATA_PLANE_CTXT[data_plane_id].base,
        [](DataPlaneCtxt*) { /* statically allocated */ }
    };
}

// Public ctor: application code path. Delegate to the protected overload with a default
// context claimed from the static pool (STM32 never new/malloc's).
DataPlane::DataPlane(HSBEmulator& hsb_emulator, const IPAddress& ip_address, uint8_t data_plane_id, uint8_t sensor_id)
    : DataPlane(hsb_emulator, ip_address, data_plane_id, sensor_id, make_default_stm32_ctxt(data_plane_id))
{
}

// HSBEmulator must remain alive for as long as the longest-lived DataPlane object it was used to construct.
// IPAddress is both the source IP address and subnet mask to be able to set the appropriate broadcast address (!cannot use INADDR_BROADCAST!).
// `ctxt` is the caller-supplied owning unique_ptr; the caller chose the deleter (typically a
// no-op pointing into DATA_PLANE_CTXT[]). This overload is `protected`: only subclasses reach it.
DataPlane::DataPlane(HSBEmulator& hsb_emulator, const IPAddress& ip_address, uint8_t data_plane_id, uint8_t sensor_id,
    std::unique_ptr<DataPlaneCtxt, std::function<void(DataPlaneCtxt*)>> ctxt)
    : registers_(hsb_emulator.get_memory())
    , ip_address_(ip_address)
    , configuration_(hsb_emulator.get_config())
    , sensor_id_(sensor_id)
    , data_plane_id_(data_plane_id)
    , data_plane_ctxt_(std::move(ctxt))
{
    // validate against platform limits before indexing any pre-allocated arrays
    if (data_plane_id_ >= MAX_DATA_PLANES) {
        Error_Handler("data_plane_id exceeds maximum number of data planes");
    }
    if (sensor_id_ >= MAX_SENSORS) {
        Error_Handler("sensor_id exceeds maximum number of sensors");
    }
    if (!data_plane_ctxt_) {
        Error_Handler("DataPlane: null DataPlaneCtxt passed to protected ctor");
    }

    // Wire the hif (per-data-plane) and vp (per-sensor) slices to the file-scope arrays.
    // Two DataPlanes bound to the same data_plane_id share the same hif slice; two bound to
    // the same sensor_id share the same vp slice.
    data_plane_ctxt_->dp_registers = &DP_REGISTERS[data_plane_id_];
    data_plane_ctxt_->dp_sensor_registers = &DP_SENSOR_REGISTER[sensor_id_];

    // common board initialization
    init(hsb_emulator);

    // board/platform-specific initialization
    STM32DataPlaneCtxt* ext = stm32_ctxt(data_plane_ctxt_.get());
    net_set_ip_address(ip_address_.ip_address);
    ext->eth_handle = &(reinterpret_cast<STM32HSBEmulatorCtxt*>(hsb_emulator.ctxt_.get())->eth_handle);
    init_bootp_packet(ext->bootp_buffer, ip_address_, configuration_);
    init_bootp_tx_config(ext);
}

// Single-threaded main loop: no locking around base.running access.
void DataPlane::start()
{
    if (data_plane_ctxt_->running) {
        return;
    }
    data_plane_ctxt_->running = true;
    if (clock_gettime(CLOCK_REALTIME, &data_plane_ctxt_->start_time)) {
        Error_Handler("Failed to get start time in DataPlane::start()");
    }
}

void DataPlane::stop()
{
    if (!data_plane_ctxt_) {
        return;
    }
    data_plane_ctxt_->running = false;
}

bool DataPlane::is_running()
{
    if (!data_plane_ctxt_) {
        return false;
    }
    return data_plane_ctxt_->running;
}

int64_t DataPlane::send(const DLTensor& tensor, FrameMetadata* frame_metadata)
{
    if (!transmitter_) {
        fprintf(stderr, "DataPlane::send() no transmitter\n");
        return -1;
    }

    if (frame_metadata == nullptr) {
        frame_metadata = DEFAULT_FRAME_METADATA;
    }
    return send((uint8_t*)tensor.data, DLTensor_n_bytes(tensor), frame_metadata);
}

int64_t DataPlane::send(const uint8_t* buffer, size_t buffer_size, FrameMetadata* frame_metadata)
{
    if (!transmitter_) {
        fprintf(stderr, "DataPlane::send_packet() no transmitter\n");
        return -1;
    }

    update_metadata();

    return transmitter_->send(data_plane_ctxt_.get(), buffer, buffer_size, frame_metadata);
}

int DataPlane::broadcast_bootp()
{
    if (!is_running()) {
        return -1;
    }
    STM32DataPlaneCtxt* ext = stm32_ctxt(data_plane_ctxt_.get());
    update_bootp_packet(ext->bootp_buffer, data_plane_ctxt_->start_time);
    return (int)HAL_ETH_Transmit(ext->eth_handle, &ext->tx_config, HSB_DEFAULT_TIMEOUT_MSEC);
}

}
