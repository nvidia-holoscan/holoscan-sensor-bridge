/**
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef SRC_HOLOLINK_DATA_CHANNEL
#define SRC_HOLOLINK_DATA_CHANNEL

#include <memory>
#include <string>

#include "hololink.hpp"
#include "metadata.hpp"

#include <hololink/native/networking.hpp>

namespace hololink {

// Note that these are offsets from VP_START.
constexpr uint32_t DP_PACKET_SIZE = 0x30C;
constexpr uint32_t DP_HOST_MAC_LOW = 0x310;
constexpr uint32_t DP_HOST_MAC_HIGH = 0x314;
constexpr uint32_t DP_HOST_IP = 0x318;
constexpr uint32_t DP_HOST_UDP_PORT = 0x31C;
constexpr uint32_t DP_VIP_MASK = 0x324;

// Fields in DP_ROCE_CFG
// "31:28 = end buf
//  27:24 = start buf
//  23: 0 = qp"
constexpr uint32_t DP_ROCE_CFG = 0x1000;
constexpr uint32_t DP_ROCE_RKEY_0 = 0x1004;
constexpr uint32_t DP_ROCE_VADDR_MSB_0 = 0x1008;
constexpr uint32_t DP_ROCE_VADDR_LSB_0 = 0x100C;
constexpr uint32_t DP_ROCE_BUF_END_MSB_0 = 0x1010;
constexpr uint32_t DP_ROCE_BUF_END_LSB_0 = 0x1014;

class DataChannel {
public:
    /**
     * @brief Construct a new DataChannel object
     *
     * @param metadata
     */
    DataChannel(const Metadata& metadata,
        const std::function<std::shared_ptr<Hololink>(const Metadata& metadata)>& create_hololink
        = Hololink::from_enumeration_metadata);

    /**
     * @brief
     *
     * @param metadata
     * @return true
     * @return false
     */
    static bool enumerated(const Metadata& metadata);

    /**
     * @brief
     *
     * @return std::shared_ptr<Hololink>
     */
    std::shared_ptr<Hololink> hololink() const;

    /**
     * @brief
     *
     * @return std::string
     */
    const std::string& peer_ip() const;

    /**
     * @brief
     *
     * @param qp_number
     * @param rkey
     */
    void authenticate(uint32_t qp_number, uint32_t rkey);

    /**
     * @brief
     *
     * @param frame_address
     * @param frame_size
     * @param local_data_port
     */
    void configure(uint64_t frame_address, uint64_t frame_size, uint32_t local_data_port);

    /**
     * @brief
     *
     * @param reg
     * @param value
     * @return true
     * @return false
     */
    bool write_uint32(uint32_t reg, uint32_t value);

private:
    std::shared_ptr<Hololink> hololink_;
    uint32_t address_;
    std::string peer_ip_;
    uint32_t vip_mask_;
    uint32_t qp_number_;
    uint32_t rkey_;

    void configure_internal(uint64_t frame_size, uint32_t payload_size, uint32_t header_size,
        const native::MacAddress& local_mac, const std::string& local_ip, uint32_t local_data_port,
        uint32_t qp_number, uint32_t rkey, uint64_t address, uint64_t size);
};

} // namespace hololink

#endif /* SRC_HOLOLINK_DATA_CHANNEL */
