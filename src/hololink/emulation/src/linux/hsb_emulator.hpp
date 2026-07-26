#ifndef LINUX_HSB_EMULATOR_HPP
#define LINUX_HSB_EMULATOR_HPP

// Linux uses the std::vector-backed AddressMap (HSB_CP_*_MAP_SIZE = 0), which is the
// default in the public hsb_emulator.hpp. No override needed here.

#include "../../hsb_emulator.hpp"
#include "../../net.hpp"
#include "address_map.hpp"
#include "data_plane.hpp"
#include "i2c_interface.hpp"

#include <atomic>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <poll.h>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace hololink::emulation {

void handle_control_packet(HSBEmulatorCtxt* ctxt, ETH_BufferTypeDef* buffer);
int read_hsb_data(void* ctxt, struct AddressValuePair* addr_val, int max_count);
int ptp_readback_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count);
int ptp_configure_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count);
int reset_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count);

struct BSDControlMessage {
    struct sockaddr_in host_addr;
    ETH_BufferTypeDef message_buffer;
    alignas(32) char data[MTU_SIZE];
};

// built-in I2C peripheral for Renesas I2C devices
class RenesasI2CPeripheral : public I2CPeripheral {
public:
    static constexpr uint16_t PERIPHERAL_ADDRESS = 0x09u;
    RenesasI2CPeripheral() = default;

    I2CStatus i2c_transaction(uint16_t peripheral_address, const uint8_t* write_bytes, uint16_t write_size, uint8_t* read_bytes, uint16_t read_size) override
    {
        return I2CStatus::I2C_STATUS_SUCCESS;
    }

    void attach_to_i2c(I2CController& i2c_controller, uint8_t i2c_bus_address) override
    {
        i2c_controller.attach_i2c_peripheral(i2c_bus_address, PERIPHERAL_ADDRESS, this);
    }
};

// Built-in I2C peripheral that emulates the LI 4-channel I2C expander (PCA9544-style) used on
// Leopard CPNX100 boards to share one i2c_bus across multiple cameras at the same peripheral
// address (e.g. two IMX274 sensors at 0x1A). The host driver picks a sensor by writing a one-hot
// OUTPUT_EN byte to peripheral 0x70 before each sensor transaction; we cache that byte here so
// the I2CController can offset bus_en when looking up the actual sensor peripheral.
//
// This peripheral is auto-attached at CAM_I2C_BUS in HSBEmulator's constructor. Applications that
// own peripheral 0x70 themselves (e.g. Vb1940Emulator's VCL_EN_I2C_ADDRESS_1 entry) silently
// override this attach, in which case the redirect logic in i2c_execute simply finds a non-expander
// peripheral at 0x70 and skips the offset (see attach_i2c_peripheral and i2c_execute).
class LII2CExpanderPeripheral : public I2CPeripheral {
public:
    static constexpr uint16_t PERIPHERAL_ADDRESS = 0x70u;
    LII2CExpanderPeripheral() = default;

    I2CStatus i2c_transaction(uint16_t peripheral_address, const uint8_t* write_bytes, uint16_t write_size, uint8_t* read_bytes, uint16_t read_size) override
    {
        // LII2CExpander::configure() issues a single 1-byte write carrying the one-hot OUTPUT_EN
        // selector. Capture that; ignore reads or unexpected sizes.
        if (write_size == 1) {
            output_state_ = write_bytes[0];
        }
        return I2CStatus::I2C_STATUS_SUCCESS;
    }

    void attach_to_i2c(I2CController& i2c_controller, uint8_t i2c_bus_address) override
    {
        i2c_controller.attach_i2c_peripheral(i2c_bus_address, PERIPHERAL_ADDRESS, this);
    }

    // Most-recent OUTPUT_EN byte the host wrote (one-hot: 0b0001/0b0010/0b0100/0b1000 for
    // OUTPUT_1..4; 0 for DEFAULT/disabled).
    uint8_t output_state() const { return output_state_; }

private:
    uint8_t output_state_ { 0 };
};

class ControlThread;

/**
 * @brief Linux-specific HSBEmulatorCtxt extension. `base` (the common HSBEmulatorCtxt) is
 * the first member; the rest is Linux-only state (thread plumbing, lazy register caches,
 * the running_mutex that protects base.running, and the auto-attached I2C peripherals).
 */
struct LinuxHSBEmulatorCtxt {
    HSBEmulatorCtxt base;

    // multiple devices may share this object
    std::vector<DataPlane*> data_plane_list;

    // compositions — owned by HSBEmulator and never shared
    std::unique_ptr<RenesasI2CPeripheral> renesas_i2c = { nullptr };
    std::unique_ptr<LII2CExpanderPeripheral> li_i2c_expander = { nullptr };

    // Per-data-plane and per-sensor register slices, indexed by data_plane_id / sensor_id
    // respectively. DataPlane instances sharing an id share the same slice. Grown lazily by
    // get_or_create_dp_*_registers() when a previously-unseen id is bound. shared_ptr so a
    // slot can be empty (nullptr) for ids that haven't been claimed yet without disturbing
    // the others. Owned by the HSBEmulator and freed at its destruction (which outlives every
    // DataPlane it produced, per the lifetime invariant in data_plane.hpp).
    std::vector<std::shared_ptr<DPRegisters>> dp_registers_cache;
    std::vector<std::shared_ptr<DPSensorRegisters>> dp_sensor_registers_cache;

    // control plane thread
    ControlThread* control_thread_ { nullptr };
    int control_socket_fd_ { -1 };

    // Protects HSBEmulatorCtxt::base.running on Linux (multiple threads call start/stop/
    // is_running concurrently). STM32 has no equivalent — its loop is single-threaded.
    std::mutex running_mutex;
};

std::mutex* get_metadata_mutex(LinuxHSBEmulatorCtxt& ctxt);

/**
 * @brief Look up (or lazily create) the DPRegisters slice for data_plane_id, growing
 * ctxt.dp_registers_cache as needed so that ctxt.dp_registers_cache[data_plane_id] is the
 * live slot. Calling with the same data_plane_id returns the same pointer.
 *
 * Free function (not a class method) so the public HSBEmulator interface stays unchanged.
 * Not thread-safe; DataPlane construction is expected to be single-threaded.
 */
DPRegisters* get_or_create_dp_registers(LinuxHSBEmulatorCtxt& ctxt, uint8_t data_plane_id);

/**
 * @brief Look up (or lazily create) the DPSensorRegisters slice for sensor_id, growing
 * ctxt.dp_sensor_registers_cache as needed so that ctxt.dp_sensor_registers_cache[sensor_id]
 * is the live slot. Calling with the same sensor_id returns the same pointer.
 *
 * Free function (not a class method) so the public HSBEmulator interface stays unchanged.
 * Not thread-safe; DataPlane construction is expected to be single-threaded.
 */
DPSensorRegisters* get_or_create_dp_sensor_registers(LinuxHSBEmulatorCtxt& ctxt, uint8_t sensor_id);

}

#endif
