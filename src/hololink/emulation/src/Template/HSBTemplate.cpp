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
 * TEMPLATE port — empty stub implementations of every platform-side symbol the
 * common emulator code expects. The file is compiled into `emulation_host` (which
 * the transport libraries link against), so all symbols below are visible to the
 * transports without each transport needing its own platform .cpp.
 *
 * To bring this target to life:
 *   1. Define struct extension members in HSBTemplate.hpp (network handles, file
 *      descriptors, threads / mutexes, peripheral pools — whatever your platform
 *      needs) — the `TODO:` markers point at the relevant blocks.
 *   2. Replace the empty function bodies below with calls to your target's HAL /
 *      OS / network stack. The `// TODO:` comment in each function explains what
 *      it's supposed to accomplish on a real port.
 *   3. If your target needs its own static pools (à la STM32) instead of `new`,
 *      replace the `make_*_ctxt()` helpers and the deleters they install.
 *
 * Until those steps are done the build succeeds but no actual data plane traffic
 * is generated and host control-plane reads will return zero.
 */

#include "HSBTemplate.hpp"

#include <functional>
#include <memory>
#include <string>
#include <time.h>

namespace hololink::emulation {

// ============================================================================
// Helpers — downcasts and ctxt-allocation factories.
// ============================================================================

// Downcast helpers used wherever we need a platform-extension view of the
// DataPlaneCtxt-or-HSBEmulatorCtxt-or-I2CControllerCtxt aliased common pointer.
static inline TemplateDataPlaneCtxt* tpl_dp_ctxt(DataPlaneCtxt* base)
{
    return reinterpret_cast<TemplateDataPlaneCtxt*>(base);
}
static inline TemplateHSBEmulatorCtxt* tpl_hsb_ctxt(HSBEmulatorCtxt* base)
{
    return reinterpret_cast<TemplateHSBEmulatorCtxt*>(base);
}
static inline TemplateI2CControllerCtxt* tpl_i2c_ctxt(I2CControllerCtxt* base)
{
    return reinterpret_cast<TemplateI2CControllerCtxt*>(base);
}

// ============================================================================
// net.hpp surface — host stubs
// ============================================================================

// TODO: feed `tp` from your target's wall-clock / monotonic source. The common
// emulator code only ever calls this with CLOCK_REALTIME; tv_sec and tv_nsec
// should be filled in with the current time since the epoch (or since boot if
// your target has no notion of a wall clock — the values are only used for
// FrameMetadata timestamping). Return 0 on success, < 0 on error.
extern "C" int clock_gettime(clockid_t /*clock_id*/, struct timespec* tp)
{
    if (tp) {
        tp->tv_sec = 0;
        tp->tv_nsec = 0;
    }
    return 0;
}

// TODO: build an IPAddress from a textual representation (e.g. "192.168.0.2"). The
// `hsb_control` example and any other application calling IPAddress_from_string()
// expects this to populate the ip_address / subnet / mac fields of IPAddress.
IPAddress IPAddress_from_string(const std::string& /*ip_address*/)
{
    return IPAddress {};
}

// TODO: format an IPAddress as a printable string. Common diagnostic prints
// (src/common/data_plane.cpp) call this.
std::string IPAddress_to_string(const IPAddress& /*ip_address*/)
{
    return std::string {};
}

// ============================================================================
// Wire-emit hook for the host control-plane reply path. The COE / RoCEv2
// equivalents (send_coe_packet / send_rocev2_packet) live in HSBTemplate_coe.cpp
// and HSBTemplate_rocev2.cpp so they sit in the transport-specific archives
// (emulationcoe.a / emulationroce.a).
// ============================================================================

// TODO: emit the reply frame for a host control-plane request. The buffer has
// already been prepared by prepare_control_plane_reply (in common/hsb_emulator.cpp)
// — its `.len` is the outgoing frame length and its headers have been swapped.
void control_plane_reply(HSBEmulatorCtxt* /*ctxt*/, ETH_BufferTypeDef* /*buffer*/) { }

// ============================================================================
// I2C — platform i2c_transaction stub. Called from common/i2c.cpp on every
// host write to the I2C control register.
// ============================================================================

// TODO: decode the host-issued I2C transaction (the low 16 bits of `value` are
// the command, high bits the peripheral address) and either route it to an
// attached I2CPeripheral (Linux pattern) or drive your target's I2C hardware
// (STM32 pattern). Update i2c_ctxt->status / i2c_ctxt->data on completion.
void i2c_transaction(I2CControllerCtxt* /*i2c_ctxt*/, uint32_t /*value*/) { }

// ============================================================================
// I2CController — platform method bodies. Constructor / destructor + start /
// stop / is_running / attach_i2c_peripheral are the platform surface; i2c_*_cb
// implementations live in src/common/i2c.cpp.
// ============================================================================

I2CController::I2CController(HSBEmulator& /*hsb_emulator*/, uint32_t controller_address)
{
    // Heap-allocate the platform extension and hand its `base` pointer to
    // I2CController::ctxt_ with a deleter that knows how to delete the extension.
    // (If your target prefers a static pool — see STM32 for the pattern — claim a
    // pool slot here and use a no-op deleter instead.)
    TemplateI2CControllerCtxt* lctxt = new TemplateI2CControllerCtxt();
    ctxt_.reset(&lctxt->base);
    ctxt_.get_deleter() = [](I2CControllerCtxt* base) {
        delete reinterpret_cast<TemplateI2CControllerCtxt*>(base);
    };
    reset(controller_address);
}

// TODO: start any attached peripherals / I2C hardware. Mark the controller running.
void I2CController::start()
{
    if (ctxt_) {
        ctxt_->running = true;
    }
}

// TODO: stop attached peripherals / I2C hardware. Mark the controller not running.
void I2CController::stop()
{
    if (ctxt_) {
        ctxt_->running = false;
    }
}

bool I2CController::is_running() { return ctxt_ ? ctxt_->running : false; }

// TODO: register `peripheral` on (bus_address, peripheral_address). The Linux
// pattern stores these in a nested unordered_map on the platform extension; STM32
// only supports one controller and doesn't have this. It is only needed if you are also simulating a virtual I2C peripheral.
void I2CController::attach_i2c_peripheral(uint32_t /*bus_address*/, uint16_t /*peripheral_address*/, I2CPeripheral* /*peripheral*/) { }

// ============================================================================
// DataPlane — platform method bodies. Two ctors (public 4-arg + protected
// 5-arg), destructor, start/stop/is_running, two send overloads,
// broadcast_bootp.
// ============================================================================

// Build the platform-default DataPlaneCtxt for the public 4-arg constructor:
// heap-allocate a TemplateDataPlaneCtxt and wrap its `base` in a unique_ptr.
static std::unique_ptr<DataPlaneCtxt, std::function<void(DataPlaneCtxt*)>>
make_default_template_ctxt()
{
    TemplateDataPlaneCtxt* owned = new TemplateDataPlaneCtxt();
    return {
        &owned->base,
        [](DataPlaneCtxt* p) { delete reinterpret_cast<TemplateDataPlaneCtxt*>(p); }
    };
}

DataPlane::DataPlane(HSBEmulator& hsb_emulator, const IPAddress& ip_address, uint8_t data_plane_id, uint8_t sensor_id)
    : DataPlane(hsb_emulator, ip_address, data_plane_id, sensor_id, make_default_template_ctxt())
{
}

DataPlane::DataPlane(HSBEmulator& hsb_emulator, const IPAddress& ip_address, uint8_t data_plane_id, uint8_t sensor_id,
    std::unique_ptr<DataPlaneCtxt, std::function<void(DataPlaneCtxt*)>> ctxt)
    : registers_(hsb_emulator.get_memory())
    , ip_address_(ip_address)
    , configuration_(hsb_emulator.get_config())
    , sensor_id_(sensor_id)
    , data_plane_id_(data_plane_id)
    , data_plane_ctxt_(std::move(ctxt))
{
    if (!data_plane_ctxt_) {
        return; // TODO: call Error_Handler when integrating with the target.
    }
    // Common wiring: register slices + callbacks + add to HSBEmulator. The
    // common init() reads dp_registers/dp_sensor_registers from data_plane_ctxt_,
    // so leave them for the user to wire up to whatever per-data-plane / per-
    // sensor register storage they choose.
    init(hsb_emulator);
    // TODO: per-platform DataPlane initialization (start a BootP thread, prime
    // a HAL transmit config, ...).
}

// DataPlane::~DataPlane() is defined in src/common/data_plane.cpp; don't duplicate here.

// TODO: start the BootP broadcast loop (Linux: spawn a thread; STM32: enable a
// HAL timer / mark the data plane to be polled from the main loop).
void DataPlane::start()
{
    if (data_plane_ctxt_) {
        data_plane_ctxt_->running = true;
        clock_gettime(CLOCK_REALTIME, &data_plane_ctxt_->start_time);
    }
}

// TODO: stop the BootP broadcast loop and join any helper threads.
void DataPlane::stop()
{
    if (data_plane_ctxt_) {
        data_plane_ctxt_->running = false;
    }
}

bool DataPlane::is_running()
{
    return data_plane_ctxt_ ? data_plane_ctxt_->running : false;
}

// TODO: if your target has GPU-resident input tensors, add a bounce buffer to
// TemplateDataPlaneCtxt (see the Linux LinuxDataPlaneCtxt::double_buffer pattern)
// and cudaMemcpy into it here. The default stub forwards the raw pointer assuming
// host-accessible memory.
int64_t DataPlane::send(const DLTensor& tensor, FrameMetadata* frame_metadata)
{
    if (!transmitter_) {
        return -1;
    }
    if (frame_metadata == nullptr) {
        frame_metadata = DEFAULT_FRAME_METADATA;
    }
    return send((uint8_t*)tensor.data, DLTensor_n_bytes(tensor), frame_metadata);
}

// TODO: serialize sends with attach_i2c_peripheral and update_metadata if your
// target uses multi-threading; the Linux implementation takes a mutex here.
int64_t DataPlane::send(const uint8_t* buffer, size_t buffer_size, FrameMetadata* frame_metadata)
{
    if (!transmitter_) {
        return -1;
    }
    update_metadata();
    return transmitter_->send(data_plane_ctxt_.get(), buffer, buffer_size, frame_metadata);
}

// TODO: emit a single BootP broadcast frame using the target's transport. Linux
// sends via a UDP socket; STM32 builds an Ethernet frame and uses HAL_ETH_Transmit.
int DataPlane::broadcast_bootp() { return 0; }

// ============================================================================
// HSBEmulator — platform method bodies. Constructor allocates a
// TemplateHSBEmulatorCtxt and hands its `base` to the public ctxt_ member; the
// HSBEmulator class's read/write/register_*_callback already live in
// src/common/hsb_emulator.cpp.
// ============================================================================

HSBEmulator::HSBEmulator(const HSBConfiguration& config)
    : configuration_(config)
    , i2c_controller_(*this, hololink::I2C_CTRL)
{
    TemplateHSBEmulatorCtxt* lctxt = new TemplateHSBEmulatorCtxt();
    ctxt_.reset(&lctxt->base);
    ctxt_.get_deleter() = [](HSBEmulatorCtxt* base) {
        delete reinterpret_cast<TemplateHSBEmulatorCtxt*>(base);
    };

    // Wire ctxt_->hsb_emulator + ctxt_->register_memory's dispatch ctxt AND register
    // the platform-invariant callbacks (HSB version, PTP, APB RAM, async events,
    // I2C register block). reset() is defined in src/common/hsb_emulator.cpp.
    reset();

    // TODO: platform-specific HSBEmulator initialization (spawn a control-plane
    // listener thread, attach default peripherals, register board callbacks, ...).
}

HSBEmulator::~HSBEmulator() { }

// TODO: start the control-plane listener (Linux: a thread; STM32: nothing — it's
// polled via handle_msgs()) and start every registered DataPlane / I2CController.
void HSBEmulator::start()
{
    if (is_running()) {
        return;
    }
    i2c_controller_.start();

    /* the .build() calls are required here to ensure the DataPlanes, peripheral controllers, and any other contexts you wire in have the ability to register their callbacks before locking them in*/
    ctxt_->cp_write_map.build();
    ctxt_->cp_read_map.build();
    ctxt_->running = true;
}

// TODO: stop every registered DataPlane, the I2CController, and the control
// plane listener. Wait for all helper threads to join before returning.
void HSBEmulator::stop()
{
    if (!is_running()) {
        return;
    }
    i2c_controller_.stop();
    ctxt_->running = false;
}

bool HSBEmulator::is_running()
{
    if (i2c_controller_.is_running()) {
        return true;
    }
    return ctxt_ ? ctxt_->running : false;
}

// TODO: drain any pending control-plane messages off the wire. On STM32 the
// main loop calls this repeatedly; on Linux a worker thread does the equivalent
// and handle_msgs() is a no-op.
int HSBEmulator::handle_msgs() { return 0; }

// TODO: pick a slot for the new DataPlane. The Linux implementation pushes it
// onto a std::vector; the STM32 implementation indexes a fixed-size array.
int HSBEmulator::add_data_plane(DataPlane& /*data_plane*/) { return 0; }

// COEDataPlane / RoCEv2DataPlane subclass bodies and the matching wire-emit
// hooks (send_coe_packet / send_rocev2_packet) live in HSBTemplate_coe.cpp and
// HSBTemplate_rocev2.cpp respectively. Keeping them here would create a circular
// link dependency with the per-transport static libraries (emulationcoe /
// emulationroce) that own the COETransmitter / RoCEv2Transmitter implementations.

// ============================================================================
// Error_Handler — every platform supplies its own. The default here spins.
// Replace with whatever your target's panic / fault path is (route to UART, set
// an LED pattern, trigger a watchdog reset, halt the debugger, ...).
// ============================================================================

extern "C" void Error_Handler(const char* /*str*/)
{
    // TODO: hand off `str` (if not null) to the target's logging path, then halt.
    for (;;) { }
}

// ============================================================================
// msleep — declared in hsb_emulator.hpp. Each port supplies its own. The linux
// build wraps usleep; the STM32 build wraps HAL_Delay in tim.c. Replace this
// stub with your platform's millisecond delay primitive (e.g. a SysTick busy-wait,
// an RTOS sleep, or a HAL delay function).
// ============================================================================

extern "C" int msleep(unsigned /*milliseconds*/)
{
    // TODO: implement using the target's millisecond delay primitive.
    return 0;
}

// ============================================================================
// newlib syscall stubs — required by arm-none-eabi-gcc's libc when linking an
// executable. The bodies here only exist to satisfy the linker; replace them
// with real implementations (UART for _write, a heap arena for _sbrk, ...) when
// porting. STM32 puts the equivalent stubs in src/STM32/stm32_system.c — that
// file is a good reference for a real implementation.
// ============================================================================

extern "C" {

int _getpid(void) { return 1; }

int _kill(int /*pid*/, int /*sig*/) { return -1; }

void _exit(int /*status*/)
{
    // TODO: route to your target's reset / halt path.
    for (;;) { }
}

int _write(int /*file*/, char* /*ptr*/, int len)
{
    // TODO: route to UART / RTT / semihosting. Returning len discards output.
    return len;
}

int _read(int /*file*/, char* /*ptr*/, int /*len*/) { return 0; }

int _close(int /*file*/) { return -1; }

int _isatty(int /*file*/) { return 0; }

int _lseek(int /*file*/, int /*offset*/, int /*whence*/) { return 0; }

// struct stat forward-declared so we don't need to pull in <sys/stat.h>.
struct stat;
int _fstat(int /*file*/, struct stat* /*st*/) { return 0; }

// _sbrk: extend the heap. The simple linker-symbol-based bump allocator pattern
// (use `__heap_start` / `_end` and a `__stack` boundary) is the usual approach;
// see STM32/stm32_system.c::_sbrk for a worked example. Returning -1 (out of
// memory) makes any allocation fail — fine for the empty-stub default.
void* _sbrk(int /*incr*/) { return (void*)-1; }

} // extern "C"

} // namespace hololink::emulation
