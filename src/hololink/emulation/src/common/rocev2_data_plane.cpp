#include "rocev2_data_plane.hpp"

namespace hololink::emulation {

void RoCEv2DataPlane::update_metadata()
{

    RoCEv2Ctxt* rocev2_metadata = reinterpret_cast<RoCEv2Ctxt*>(data_plane_ctxt_.get());
    if (RoCEv2Ctxt_is_in_use(rocev2_metadata)) {
        return;
    }
    struct DPRegisters* dp_reg = rocev2_metadata->base.base.dp_registers;
    struct DPSensorRegisters* dp_sensor_reg = rocev2_metadata->base.base.dp_sensor_registers;

    // update page
    uint16_t start_page = (uint16_t)((dp_sensor_reg->vp_data[DP_MAX_BUFF / REGISTER_SIZE] >> 16) & 0xFFF);
    uint16_t end_page = (uint16_t)((dp_sensor_reg->vp_data[DP_MAX_BUFF / REGISTER_SIZE] >> 0) & 0xFFF);
    if (page_ < start_page) {
        page_ = start_page;
    } else {
        page_++;
    }
    if (page_ > end_page) {
        page_ = start_page;
    }

    // derived metadata
    rocev2_metadata->payload_size = static_cast<uint16_t>(0xFFFF & (dp_reg->hif_data[DP_PACKET_SIZE / REGISTER_SIZE] * HSB_PAGE_SIZE));
    uint64_t address = dp_sensor_reg->vp_data[DP_PAGE_LSB / REGISTER_SIZE] + ((uint64_t)dp_sensor_reg->vp_data[DP_PAGE_MSB / REGISTER_SIZE] << 32);
    address += ((uint64_t)dp_sensor_reg->vp_data[DP_PAGE_INC / REGISTER_SIZE]) * HSB_PAGE_SIZE * page_;
    rocev2_metadata->virtual_address = address;
    rocev2_metadata->page = (uint16_t)page_;
    rocev2_metadata->metadata_offset = (dp_sensor_reg->vp_data[DP_BUFFER_LENGTH / REGISTER_SIZE] + HSB_PAGE_SIZE - 1) & ~(HSB_PAGE_SIZE - 1);
}
} // namespace hololink::emulation