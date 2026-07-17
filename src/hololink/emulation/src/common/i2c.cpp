#include "i2c.hpp"
#include "../../hsb_config.hpp"
#include "net.hpp"
#include "utils.hpp"

#include <cstring>

namespace hololink::emulation {

// Platform-invariant initialization for ctxt_. Each platform's I2CController
// constructor allocates the right extension struct (LinuxI2CControllerCtxt,
// STM32I2CControllerCtxt, ...), wires ctxt_ to its embedded base, sets the
// deleter, and then calls this method to fill in the shared fields.
void I2CController::reset(uint32_t controller_address)
{
    ctxt_->i2c_controller = this;
    ctxt_->control_address = controller_address;
    ctxt_->data_address = ctxt_->control_address + I2C_REG_DATA_BUFFER;
    ctxt_->status = I2C_IDLE;
    std::memset(ctxt_->registers, 0, sizeof(ctxt_->registers));
    std::memset(ctxt_->data, 0, sizeof(ctxt_->data));
    ctxt_->running = false;
}

int i2c_configure_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count)
{
    I2CControllerCtxt* i2c_ctxt = (I2CControllerCtxt*)ctxt;
    struct AddressValuePair* cur = addr_val;
    int n = 0;
    while (n < max_count) {
        uint32_t address = AVP_GET_ADDRESS(cur);
        if (address >= i2c_ctxt->control_address && address <= i2c_ctxt->control_address + I2C_REG_CLK_CNT) {
            i2c_ctxt->registers[(address - i2c_ctxt->control_address) / REGISTER_SIZE] = AVP_GET_VALUE(cur);
            // if on the control address, execute the transaction command
            if (address == i2c_ctxt->control_address) {
                i2c_transaction(i2c_ctxt, i2c_ctxt->registers[0]);
            }
        } else if (address >= i2c_ctxt->data_address && address < i2c_ctxt->data_address + I2C_DATA_BUFFER_SIZE) {
            i2c_ctxt->data[(address - i2c_ctxt->data_address) / REGISTER_SIZE] = AVP_GET_VALUE(cur);
        } else { // not an I2C address; cannot write to I2C_REG_STATUS
            return n;
        }
        cur++;
        n++;
    }
    return n;
}

int i2c_readback_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count)
{
    I2CControllerCtxt* i2c_ctxt = (I2CControllerCtxt*)ctxt;
    struct AddressValuePair* cur = addr_val;
    int n = 0;
    while (n < max_count) {
        uint32_t address = AVP_GET_ADDRESS(cur);
        if (address >= i2c_ctxt->data_address && address < i2c_ctxt->data_address + I2C_DATA_BUFFER_SIZE) {
            AVP_SET_VALUE(cur, i2c_ctxt->data[(address - i2c_ctxt->data_address) / REGISTER_SIZE]);
        } else if (i2c_ctxt->control_address + I2C_REG_STATUS == address) {
            AVP_SET_VALUE(cur, i2c_ctxt->status);
        } else if (address >= i2c_ctxt->control_address && address <= i2c_ctxt->control_address + I2C_REG_CLK_CNT) {
            AVP_SET_VALUE(cur, i2c_ctxt->registers[(address - i2c_ctxt->control_address) / REGISTER_SIZE]);
        } else { // not an I2C address
            return n;
        }
        cur++;
        n++;
    }
    return n;
}

} // namespace hololink::emulation