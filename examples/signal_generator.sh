#!/bin/bash

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

shopt -s nullglob
LC_COLLATE=C INFINIBANDS=(/sys/class/infiniband/*)
shopt -u nullglob
if [ ${#INFINIBANDS[@]} -lt 2 ]; then
    echo "Error: at least 2 InfiniBand devices required, found ${#INFINIBANDS[@]}" >&2
    exit 1
fi
# The RoCE interfaces used for the TX and RX
TX_INTERFACE=`basename ${INFINIBANDS[1]}`
RX_INTERFACE=`basename ${INFINIBANDS[0]}`

# IP addresses for the TX and RX
TX_IP="192.168.0.3"
RX_IP="192.168.0.2"

TX_HOLOLINK="--tx-hololink=$TX_IP --tx-ibv-name=$TX_INTERFACE"
RX_HOLOLINK="--rx-hololink=$RX_IP --rx-ibv-name=$RX_INTERFACE"

python3 signal_generator.py $TX_HOLOLINK $RX_HOLOLINK --udp-ip="127.0.0.1"
