// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <hololink/transport/roce/cpu_roce_transceiver.hpp>

#include <utility>

int main()
{
    hololink::transport::roce::CpuRoceTransceiver::Config config;
    config.ibv_name = "not-opened-by-constructor";
    hololink::transport::roce::CpuRoceTransceiver transceiver(
        std::move(config));
    return transceiver.get_qp_number() == 0 ? 0 : 1;
}
