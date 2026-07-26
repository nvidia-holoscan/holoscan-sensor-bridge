#include "address_memory.hpp"
#include "hsb_emulator.hpp"
#include "net.hpp"

namespace hololink::emulation {

int RegisterMemory::write(AddressValuePair& address_value)
{
    if (!ctxt_) {
        return 1;
    }
    auto callback = ctxt_->cp_write_map.get(address_value.address);
    int ret = 1;
    if (callback && (1 == callback->callback(callback->ctxt, &address_value, 1))) {
        ret = 0;
    }
    return ret;
}

int RegisterMemory::read(AddressValuePair& address_value)
{
    if (!ctxt_) {
        return 1;
    }
    auto callback = ctxt_->cp_read_map.get(address_value.address);
    int ret = 1;
    if (callback && (1 == callback->callback(callback->ctxt, &address_value, 1))) {
        ret = 0;
    }
    return ret;
}

int RegisterMemory::write_many(AddressValuePair* address_values, int num_addresses)
{
    if (!ctxt_ || !address_values || num_addresses < 0) {
        return 1;
    }
    int ret = 0;
    int i = 0;
    while (i < num_addresses) {
        auto callback = ctxt_->cp_write_map.get(address_values[i].address);
        if (!callback) {
            ret = 1;
            break;
        }
        int ncomsumed = callback->callback(callback->ctxt, address_values + i, num_addresses - i);
        if (0 >= ncomsumed) {
            ret = 1;
            break;
        }
        i += ncomsumed;
    }
    return ret;
}

int RegisterMemory::read_many(AddressValuePair* address_values, int num_addresses)
{
    if (!ctxt_ || !address_values || num_addresses < 0) {
        return 1;
    }
    int ret = 0;
    int i = 0;
    while (i < num_addresses) {
        auto callback = ctxt_->cp_read_map.get(address_values[i].address);
        if (!callback) {
            ret = 1;
            break;
        }
        int ncomsumed = callback->callback(callback->ctxt, address_values + i, num_addresses - i);
        if (0 >= ncomsumed) {
            ret = 1;
            break;
        }
        i += ncomsumed;
    }
    return ret;
}

int RegisterMemory::write_range(uint32_t start_address, uint32_t* values, int num_addresses, int stride)
{
    if (!ctxt_ || !values || num_addresses < 0) {
        return 1;
    }
    int ret = 0;
    while (num_addresses) {
        struct AddressValuePair address_value = { start_address, *values };
        auto callback = ctxt_->cp_write_map.get(start_address);
        if (!callback) {
            ret = 1;
            break;
        }
        int consumed = callback->callback(callback->ctxt, &address_value, 1);
        if (0 >= consumed) {
            ret = 1;
            break;
        }
        num_addresses -= consumed;
        values += consumed * stride;
        start_address += consumed * REGISTER_SIZE;
    }
    return ret;
}

int RegisterMemory::read_range(uint32_t start_address, uint32_t* values, int num_addresses, int stride)
{
    if (!ctxt_ || !values || num_addresses < 0) {
        return 1;
    }
    int ret = 0;
    while (num_addresses) {
        struct AddressValuePair address_value = { start_address, 0 };
        auto callback = ctxt_->cp_read_map.get(start_address);
        if (!callback) {
            ret = 1;
            break;
        }
        int consumed = callback->callback(callback->ctxt, &address_value, 1);
        if (0 >= consumed) {
            ret = 1;
            break;
        }
        *values = address_value.value;
        num_addresses -= consumed;
        values += consumed * stride;
        start_address += consumed * REGISTER_SIZE;
    }
    return ret;
}

} // namespace hololink::emulation