#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Get the directory of the script
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Change to the script's directory
cd "$SCRIPT_DIR" || exit

# Uncomment the following line to customize the signal
# you can then use IQ_EXPRESSIONS in the signal_generator.py script
# to use the custom signal
#EXP_I='--expression-i=cos(2*PI*x)'
#EXP_Q='--expression-q=sin(2*PI*x)'
#IQ_EXPRESSIONS="$EXP_I $EXP_Q"

# IP addresses for the TX and RX
TX_IP="192.168.0.3"
RX_IP="192.168.0.2"

# The app picks the RoCE device on each board's network interface. Options given to this
# script go to the app too, e.g. --tx-ibv-name=mlx5_3 --rx-ibv-name=mlx5_0 to pick the
# devices yourself (see /sys/class/infiniband).
python3 signal_generator.py --tx-hololink=$TX_IP --rx-hololink=$RX_IP --udp-ip="127.0.0.1" "$@"
