#ifndef LINUX_I2C_HPP
#define LINUX_I2C_HPP

#include "address_memory.hpp"
#include "hsb_config.hpp"
#include "hsb_emulator.hpp"
#include "i2c_interface.hpp"

#include <memory>
#include <mutex>
#include <unordered_map>

namespace hololink::emulation {

void i2c_transaction(I2CControllerCtxt* i2c_ctxt, uint32_t value);
int i2c_configure_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count);
int i2c_readback_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count);

/**
 * @brief Linux-specific I2CControllerCtxt extension. `base` (the common I2CControllerCtxt)
 * is the first member; the rest is Linux-only state.
 *
 * Note: there is no worker thread, condition variable, or `cmd`/`peripheral_address`
 * scratch state. The host-driven I2C transaction path is fully synchronous now —
 * common/i2c.cpp's i2c_configure_cb calls i2c_transaction(...) directly inside the
 * control-plane callback. The mutex still exists because attach_i2c_peripheral() may
 * be called from the application thread while i2c_transaction is running on the
 * control-plane thread; both read/write i2c_bus_map.
 */
struct LinuxI2CControllerCtxt {
    I2CControllerCtxt base;

    // outer index is bus address, inner index is peripheral address. A map of a map so
    // that buses and peripherals get null/default-initialized as they are accessed.
    std::unordered_map<uint32_t, std::unordered_map<uint16_t, I2CPeripheral*>> i2c_bus_map;

    // Protects i2c_bus_map across the application thread (attach_i2c_peripheral,
    // start/stop iteration) and the control-plane callback thread (i2c_transaction).
    std::mutex i2c_mutex;
};

} // namespace hololink::emulation

#endif
