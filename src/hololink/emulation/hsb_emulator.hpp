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
 *
 * See README.md for detailed information.
 */

#ifndef EMULATION_HSB_EMULATOR_H
#define EMULATION_HSB_EMULATOR_H

#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "address_map.hpp"
#include "address_memory.hpp"
#include "hsb_config.hpp"
#include "net.hpp"

// Platform policy for the control-plane callback maps. Linux uses the std::vector-backed
// dynamic AddressMap (capacity 0 selects that specialization); STM32 uses fixed-capacity
// arrays sized to the per-build CP_*_HANDLERS_MAX_NUM constants. These defaults match
// the Linux build; STM32 overrides them via CMake compile definitions in src/CMakeLists.txt
// so the same struct definition reaches every translation unit with the right sizes.
#ifndef HSB_CP_WRITE_MAP_SIZE
#define HSB_CP_WRITE_MAP_SIZE 0
#endif
#ifndef HSB_CP_READ_MAP_SIZE
#define HSB_CP_READ_MAP_SIZE 0
#endif

// Target-portable millisecond sleep. Declared with C linkage so the STM32 build's
// tim.c can supply the implementation directly (it forwards to HAL_Delay). The
// linux build supplies it as a free function in hsb_emulator.cpp wrapping usleep,
// and the TEMPLATE port has a stub in HSBTemplate.cpp pending a real implementation.
// Returns 0 on success.
#ifdef __cplusplus
extern "C" {
#endif
int msleep(unsigned milliseconds);
#ifdef __cplusplus
}
#endif

namespace hololink::emulation {

enum ECB_CMD_CODE {
    WR_BYTE = 0x01,
    WR_WORD = 0x02,
    WR_DWORD = 0x04,
    WR_BLOCK = 0x09,
    RMW_BYTE = 0x0A,
    RD_BYTE = 0x11,
    RD_WORD = 0x12,
    RD_DWORD = 0x14,
    RD_BLOCK = 0x19,
    GET_INFO = 0x20,
};

enum ECB_RESPONSE_CODE {
    ECB_SUCCESS = 0x00,
    ECB_ADDRESS_ERROR = 0x03,
    ECB_COMMAND_ERROR = 0x04,
    ECB_FLAG_ERROR = 0x06,
    ECB_SEQUENCE_ERROR = 0x0B
};

template <typename T>
using UniqueDel = std::unique_ptr<T, std::function<void(T*)>>;

// forward declarations
class DataPlane;
class I2CPeripheral;
class I2CController;
struct I2CControllerCtxt;
struct ControlMessage;
class HSBEmulator;

/**
 * @brief callback function type for control plane callback
 * @param ctxt The context to pass to the callback.
 * @param addr_val The address and value to pass to the callback. may be a C style array
 * @param count The number of address and value pairs passed to the callback.
 * @return > 0 for the number of address and successfully processed value pairs. <= 0 for an error. Note 0 is a valid number of consumed pairs but will be treated as an error.
 */
using ControlPlaneCallback_f = std::function<int(void* ctxt, struct AddressValuePair* addr_val, int count)>;

struct ControlPlaneCallback {
    ControlPlaneCallback_f callback;
    void* ctxt;
};

struct AsyncEventCtxt {
    // these are the registers themselves, held in an array for simpler access
    uint32_t data[(CTRL_EVT_SW_EVENT - CTRL_EVENT) / REGISTER_SIZE];
    uint32_t status;
};

// Size of the byte-addressable I2C data window the host can read/write via the control
// plane. Referenced by HSBEmulator (for callback registration ranges), by the common
// I2CControllerCtxt (sizes the `data` array), and by i2c_transaction (validates the
// host-supplied transfer length). Defined once here so platform headers don't have to
// keep their own duplicate `#define`.
#define I2C_DATA_BUFFER_SIZE 0x100u

/**
 * @brief Common, target-invariant per-I2CController context.
 *
 * Each platform defines its own extension struct (LinuxI2CControllerCtxt,
 * STM32I2CControllerCtxt) with `struct I2CControllerCtxt base` as its first member.
 * Standard-layout C++ guarantees `&ext->base == reinterpret_cast<I2CControllerCtxt*>(ext)`,
 * so the I2CController class can hold an `I2CControllerCtxt*` without knowing the platform
 * extension, and platform code downcasts back via `reinterpret_cast<LinuxI2CControllerCtxt*>`
 * (or STM32 equivalent) to reach the platform extras.
 *
 * `running` is a bare bool: I2C transactions execute synchronously inside the host
 * control-plane callback (i2c_configure_cb -> i2c_transaction), there is no worker
 * thread to coordinate. The Linux extension still carries an `i2c_mutex` because
 * attach_i2c_peripheral() can be called from the application thread while the
 * control-plane callback is running on its own thread; STM32 is single-threaded and
 * needs no mutex.
 */
struct I2CControllerCtxt {
    I2CController* i2c_controller { nullptr };
    uint32_t registers[(I2C_REG_CLK_CNT - I2C_REG_CONTROL) / REGISTER_SIZE + 1];
    uint32_t data[I2C_DATA_BUFFER_SIZE / REGISTER_SIZE];
    uint32_t control_address { 0 };
    uint32_t data_address { 0 };
    uint32_t status { 0 };
    bool running { false };
};

/**
 * @brief Common, target-invariant per-HSBEmulator context.
 *
 * Each platform defines its own extension struct (LinuxHSBEmulatorCtxt,
 * STM32HSBEmulatorCtxt) with `struct HSBEmulatorCtxt base` as its first member.
 * Standard-layout C++ guarantees `&ext->base == reinterpret_cast<HSBEmulatorCtxt*>(ext)`,
 * so HSBEmulator can hold a `HSBEmulatorCtxt*` without knowing the platform extension,
 * and platform code can downcast back via `reinterpret_cast<LinuxHSBEmulatorCtxt*>`
 * (or STM32 equivalent) to reach the platform extras.
 *
 * Synchronization: `running` is a plain bool. On platforms where HSBEmulator::start/
 * stop/is_running may be called from concurrent threads (Linux), the platform extension
 * supplies a mutex that the corresponding platform implementations acquire around every
 * read/write of `running`. STM32 is single-threaded (cooperative loop in main()) so no
 * mutex is needed there.
 *
 * AddressMap capacities are macro-driven (HSB_CP_WRITE_MAP_SIZE, HSB_CP_READ_MAP_SIZE)
 * so the same struct definition compiles into a vector-backed map on Linux and a fixed-
 * capacity map on STM32. Defaults at the top of this header match Linux; the STM32 build
 * overrides via CMake compile definitions.
 */
struct HSBEmulatorCtxt {
    HSBEmulator* hsb_emulator { nullptr };
    RegisterMemory register_memory;
    struct PTPConfig ptp_config;
    struct AsyncEventCtxt async_event_ctxt;
    uint32_t apb_ram_data[APB_RAM_DATA_SIZE / REGISTER_SIZE];
    AddressMap<ControlPlaneCallback, HSB_CP_WRITE_MAP_SIZE> cp_write_map;
    AddressMap<ControlPlaneCallback, HSB_CP_READ_MAP_SIZE> cp_read_map;
    // True when HSBEmulator::start() has been called and the control plane is broadcasting.
    // Bare bool; synchronize externally via the platform extension's running_mutex if needed.
    bool running { false };
};

/**
 * @brief class that manages virtual I2C peripherals from the host. This is mostly for the linux emulator where the use case is for virtual sensors
 * @note This should not be instantiated directly by user code. Use HSBEmulator::get_i2c() instead. The default and primary I2CController is at address hololink::I2C_CTRL.
 */
class I2CController {
public:
    friend class HSBEmulator;

    ~I2CController();

    /**
     * python:
     *
     * `def attach_i2c_peripheral(self: hemu.I2CController, bus_address: int, peripheral_address: int, peripheral: hemu.I2CPeripheral)`
     *
     * @brief attach an I2C peripheral as a callback on the specified (bus address, peripheral address) pair.
     *
     * @param bus_address The bus address of the peripheral. For multiplexing or bus expanded addressing, this is the bus address of the peripheral.
     * @param peripheral_address The peripheral address of the peripheral.
     * @param peripheral The peripheral to attach. This is a pointer to the peripheral object. The caller is responsible for managing the lifetime of the peripheral object.
     */
    void attach_i2c_peripheral(uint32_t bus_address, uint16_t peripheral_address, I2CPeripheral* peripheral);

private:
    /**
     * @brief construct an I2C controller on HSB Emulator at the specified address.
     *
     * @param hsb_emulator The HSB emulator to attach to.
     * @param controller_address The address of the I2C controller.
     */
    explicit I2CController(HSBEmulator& hsb_emulator, uint32_t controller_address);

    /**
     * @brief start the I2C controller. Initializes the underlying I2C hardware
     * (STM32) or attached peripherals (Linux) and marks the controller running.
     * No worker thread is launched: host-driven I2C transactions execute synchronously
     * inside the i2c_configure_cb control-plane callback via the free-function
     * i2c_transaction().
     */
    void start();

    /**
     * @brief stop the I2C controller. Stops attached peripherals (Linux) and marks
     * the controller not running.
     */
    void stop();

    /**
     * @brief check if the I2C controller is running.
     * @return true if the I2C controller is running, false otherwise.
     */
    bool is_running();

protected:
    /**
     * @brief Initialize the platform-invariant fields of the already-allocated
     * I2CControllerCtxt the subclass-shared ctxt_ points at. Wires control/data
     * addresses, zeroes the register + data scratch, sets the back-pointer, and
     * marks the controller stopped. Called by the platform-specific I2CController
     * constructor right after it claims a ctxt slot.
     *
     * @param controller_address Base address of the I2C controller in the host
     * register space. Matched against the address the host writes to trigger
     * each I2C transaction.
     */
    void reset(uint32_t controller_address);

    UniqueDel<struct I2CControllerCtxt> ctxt_ = { nullptr };
};

/**
 * @brief The `HSBEmulator` class represents the interface that a host application has to an HSB and acts as the emulation device's controller.
 * It manages the `DataPlane` objects and the `I2CController` and all communication with the internal memory model; see `AddressMemory` for more details.
 */
class HSBEmulator {
public:
    friend class DataPlane;
    friend class I2CController;

    /**
     *
     * python:
     *
     * `hemu.HSBEmulator(config: hemu.HSBConfiguration = hemu.HSB_EMULATOR_CONFIG)`
     *
     * @brief Construct a new HSBEmulator object with the specified configuration.
     * @param config The configuration of the emulator. Two fully populated options
     * are provided in hsb_config.hpp: HSB_EMULATOR_CONFIG or HSB_LEOPARD_EAGLE_CONFIG.
     * @note HSB_EMULATOR_CONFIG is roughly equivalent to a Lattice board.
     * @note HSB_LEOPARD_EAGLE_CONFIG is roughly equivalent to a Leopard Eagle board.
     */
    HSBEmulator(const HSBConfiguration& config);

    /**
     * @brief Construct a new HSBEmulator object. Defaults to HSB_EMULATOR_CONFIG, which is roughly equivalent to a Lattice board.
     */
    HSBEmulator();
    ~HSBEmulator();

    /**
     * @brief Start the emulator. This will start the BootP broadcasting via the DataPlane objects as well as the control thread to listen for control messages from the host.
     *
     * @note It is safe to call this function multiple times and after a subsequent call to stop()
     */
    void start();

    /**
     * @brief Stop the emulator. This will shut down the control message thread, all BootP broadcasts in DataPlane objects. Data transmission is still possible until HSBEmulator object is destroyed
     *
     * @note It is safe to call this function multiple times
     */
    void stop();

    /**
     * @brief Reset the platform-invariant emulator state and (re-)register the control-plane
     * callbacks that are identical on every target (HSB IP version, RESET_REG_CTRL, PTP, APB
     * RAM, async events, I2C register block). Called from each platform's constructor after
     * the ctxt_ and i2c_controller_ are wired up. Platform-specific callback registrations
     * (GPIO/SPI on STM32; Linux peripheral attach) remain in the platform constructor.
     *
     * Future: a default reset callback is applied that will call this, but that callback can be overridden
     */
    void reset();

    /**
     * @brief Check if the emulator is running.
     * @return true if the emulator is running, false otherwise.
     */
    bool is_running();

    /**
     * python:
     *
     * `def write(self: hemu.HSBEmulator, address: int, value: int)`
     *
     * @brief Write a value to a register.
     * @param address The address of the register to write.
     * @param value The value to write.
     * @return 0 on success, 1 on address error.
     */
    int write(uint32_t address, uint32_t value);

    /**
     * python:
     *
     * `def read(self: hemu.HSBEmulator, address: int) -> int`
     *
     * @brief Read a value from a register.
     * @param address The address of the register to read.
     * @param value The value read will be stored here.
     * @return 0 on success, 1 on address error.
     */
    int read(uint32_t address, uint32_t& value);

    /**
     * @brief Set a read callback for a range of addresses. See ControlPlaneCallback_f for more details.
     * @param start_address The start address of the range.
     * @param end_address The end address of the range (exclusive).
     * @param callback The callback to set.
     * @param ctxt The context to pass to the callback.
     * @return 0 on success, else error code.
     * @note This method is not yet implemented for linux targets
     */
    int register_read_callback(uint32_t start_address, uint32_t end_address, ControlPlaneCallback_f callback, void* ctxt = nullptr);

    /**
     * @brief Register a write callback for a range of addresses. See ControlPlaneCallback_f for more details.
     * @param start_address The start address of the range.
     * @param end_address The end address of the range (exclusive).
     * @param callback The callback to set.
     * @param ctxt The context to pass to the callback.
     * @return 0 on success, else error code.
     * @note This method is not yet implemented for linux targets
     */
    int register_write_callback(uint32_t start_address, uint32_t end_address, ControlPlaneCallback_f callback, void* ctxt = nullptr);

    /**
     * @brief method to explicitly handle pending control messages to the HSBEmulator client. This method is required for MCU targets, but optional for linux targets
     */
    int handle_msgs();

    /**
     * python:
     *
     * `def get_i2c(self: hemu.HSBEmulator, controller_address: int = hololink.I2C_CTRL) -> hemu.I2CController`
     *
     * @brief Get a reference to the I2C controller.
     * @return A reference to the I2C controller.
     *
     */
    I2CController& get_i2c(uint32_t controller_address = hololink::I2C_CTRL);

    const HSBConfiguration& get_config() { return configuration_; }

private:
    // utilities for use by friend DataPlane
    /**
     * @brief Add a data plane to the emulator for start()/stop()/is_running() management
     * @param data_plane The data plane to add.
     * @return 0 on success, 1 on failure.
     */
    int add_data_plane(DataPlane& data_plane);

    /**
     * @return The register map of the emulated fpga
     */
    AddressMemory& get_memory();

    UniqueDel<HSBEmulatorCtxt> ctxt_ = { nullptr };

    HSBConfiguration configuration_;
    I2CController i2c_controller_;
};

}

#endif