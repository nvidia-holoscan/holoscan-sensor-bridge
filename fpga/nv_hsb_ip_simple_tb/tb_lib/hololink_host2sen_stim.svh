// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`ifndef HOLOLINK_HOST2SEN_STIM_SVH
`define HOLOLINK_HOST2SEN_STIM_SVH

//==============================================================================
// Host-to-Sensor (TX) stimulus library  (UC RoCE SEND path)
//
// DESCRIPTION:
//   Drives host-initiated "push" traffic to the sensors.  The host builds a
//   RoCE UC SEND message (BTH + payload + dummy ICRC), wraps it in an
//   Ethernet/IP/UDP frame addressed to the FPGA data UDP port, and transmits it
//   on the host ingress AXIS via send_eth_packet().  The DUT reassembles the
//   SEND message per QP and streams the payload out of the matching sensor TX
//   interface (o_sif_axis[sensor]).
//
//   The target sensor is selected purely by the BTH dest_qp field
//   (sensor s -> QP (s+1)<<1), mirroring the IP-level basic_host2sen UC test.
//   No ROCE_QP_RAM programming is required for the UC path; only the per-sensor
//   tx_strm_buff block is configured (small ALMOST_FULL threshold so the DUT
//   starts draining to the sensor as soon as a little data is buffered).
//
//   Every payload byte the host emits is also pushed into the per-sensor
//   expected_stx_bytes[] queue.  hololink_host2sen_scbd.svh streams the sensor
//   egress and byte-compares it against that queue (tlast is ignored).
//==============================================================================

// Traffic shape knobs (overridable via +define if ever needed).
`ifndef HOST2SEN_MSGS_PER_SENSOR
`define HOST2SEN_MSGS_PER_SENSOR 3
`endif
`ifndef HOST2SEN_MSG_BYTES
`define HOST2SEN_MSG_BYTES 1024
`endif
`ifndef HOST2SEN_PKT_PAYLOAD_MAX
`define HOST2SEN_PKT_PAYLOAD_MAX 256
`endif
// The TX-buffer thresholds are not knobs here: program_tx_strm_buff() derives them
// per sensor from the def's HOST_WIDTH, SIF_TX_WIDTH and SIF_TX_BUF_SIZE, which are
// the same values HOLOLINK_top hands the buffer as HOST_DWIDTH, SENSOR_DWIDTH and
// FIFO_DEPTH.  A hardcoded number cannot be right for every platform: tx_stream_buffer
// gates its sensor egress on afull_seen (o_axis_tvalid = afull_seen_sync &
// axis_buffer_tvalid), so nothing leaves the buffer until buffer_count first reaches
// ALMOST_FULL, and buffer_count counts entries of max(HOST_WIDTH, SIF_TX_WIDTH) bits
// (axis_buffer_tx's CENTER_WIDTH).  64 entries is 512B on a 64-bit host but 4KB on a
// 512-bit one, more than this test ever sends, which leaves the gate shut and the
// sensor silent; on a shallow buffer it also exceeds the $clog2(FIFO_DEPTH)+1-bit
// threshold field and truncates.

`ifdef SENSOR_TX_IF_INST
//------------------------------------------------------------------------------
// program_tx_strm_buff: configure the per-sensor host-to-sensor stream buffer.
//   Order mirrors the IP-level init sequence: pause map -> thresholds -> quanta
//   -> latch thresholds (ctrl=111) -> run (ctrl=011).
//------------------------------------------------------------------------------
task automatic program_tx_strm_buff();
    int entry_bytes;
    int buf_entries;
    int msg_entries;
    int afull_entries;
    int aempty_entries;
    `TB_LOG_LOW(LOG_INFO, "=== Programming host-to-sensor TX stream buffers ===");
    for (int s = 0; s < `SENSOR_TX_IF_INST; s++) begin
        // Thresholds and SIF_TX_BUF_SIZE both count buffer entries, which are
        // axis_buffer_tx's CENTER_WIDTH = max(in, out) bits wide.
        entry_bytes    = ((`HOST_WIDTH > SIF_TX_WIDTH[s]) ? `HOST_WIDTH : SIF_TX_WIDTH[s]) / 8;
        buf_entries    = SIF_TX_BUF_SIZE[s];
        msg_entries    = `HOST2SEN_MSG_BYTES / entry_bytes;
        // Half a message keeps AFULL reachable on the first message; half the buffer
        // keeps it inside a buffer too shallow to ever hold that much.  The AFULL/
        // AEMPTY hysteresis keeps the DUT off Ethernet PAUSE on nearly every cycle.
        afull_entries  = ((msg_entries < buf_entries) ? msg_entries : buf_entries) / 2;
        aempty_entries = afull_entries / 4;
        if (afull_entries  < 1) afull_entries  = 1;
        if (aempty_entries < 1) aempty_entries = 1;
        `TB_LOG_LOW(LOG_INFO, $sformatf(
                 "TX_STRM_BUFF[%0d]: entry=%0dB depth=%0d afull=%0d aempty=%0d entries pause_map=0x1",
                 s, entry_bytes, buf_entries, afull_entries, aempty_entries));
        // Backpressure host 0 (single-host simple TB) when almost full.
        ecb_write(reg_tx_strm_buff(s, TX_STRM_OFF_PAUSE_MAP), 32'h0000_0001);
        ecb_write(reg_tx_strm_buff(s, TX_STRM_OFF_AFULL),     afull_entries);
        ecb_write(reg_tx_strm_buff(s, TX_STRM_OFF_AEMPTY),    aempty_entries);
        ecb_write(reg_tx_strm_buff(s, TX_STRM_OFF_QUANTA),    32'h0000_FFFF);
        // Latch programmable thresholds (prog_thresh_ena rising edge), then run.
        ecb_write(reg_tx_strm_buff(s, TX_STRM_OFF_CTRL), TX_STRM_CTRL_LATCH);
        ecb_write(reg_tx_strm_buff(s, TX_STRM_OFF_CTRL), TX_STRM_CTRL_RUN);
    end
endtask

//------------------------------------------------------------------------------
// build_host2sen_send_frame: assemble a single UC SEND RoCE packet and wrap it
//   in a complete Ethernet/IP/UDP frame addressed to the FPGA data UDP port.
//   BTH layout (12B): opcode, SMPad_ver, P_Key[2], fecn/becn/rsvd,
//                     dest_qp[3], ack_req, psn[3]   (big-endian fields).
//   Followed by payload then 4 dummy ICRC bytes (not validated on the UC path).
//------------------------------------------------------------------------------
task automatic build_host2sen_send_frame(input  int          sensor_id,
                                         input  logic [7:0]  payload[],
                                         input  int          payload_len,
                                         input  logic [7:0]  opcode,
                                         input  logic [23:0] psn,
                                         output logic [7:0]  eth_frame[],
                                         output int          frame_size);
    automatic logic [7:0]  udp_payload[];
    automatic logic [23:0] dqp;
    automatic int          idx;

    dqp = tx_dest_qp(sensor_id);
    udp_payload = new[ROCE_BTH_BYTES + payload_len + ROCE_ICRC_BYTES];
    idx = 0;

    // --- BTH (Base Transport Header) ---
    udp_payload[idx++] = opcode;              // opcode
    udp_payload[idx++] = 8'h00;               // SMPad_ver (no pad, ver 0)
    udp_payload[idx++] = ROCE_BTH_PKEY[15:8]; // P_Key MSB
    udp_payload[idx++] = ROCE_BTH_PKEY[7:0];  // P_Key LSB
    udp_payload[idx++] = 8'h00;               // fecn/becn/rsvd
    udp_payload[idx++] = dqp[23:16];          // dest_qp MSB
    udp_payload[idx++] = dqp[15:8];
    udp_payload[idx++] = dqp[7:0];            // dest_qp LSB
    udp_payload[idx++] = 8'h00;               // ack_req/rsvd
    udp_payload[idx++] = psn[23:16];          // PSN MSB
    udp_payload[idx++] = psn[15:8];
    udp_payload[idx++] = psn[7:0];            // PSN LSB

    // --- Payload ---
    for (int i = 0; i < payload_len; i++)
        udp_payload[idx++] = payload[i];

    // --- ICRC (dummy trailer, stripped by the DUT) ---
    for (int i = 0; i < ROCE_ICRC_BYTES; i++)
        udp_payload[idx++] = 8'h00;

    build_ethernet_frame(udp_payload, udp_payload.size(),
                         DEFAULT_FPGA_MAC, TB_HOST_MAC,        // dst=FPGA, src=host
                         TB_FPGA_IP,       TB_HOST_IP,         // dst=FPGA, src=host
                         ETH_UDP_ROCE_PORT, ETH_UDP_ROCE_PORT, // FPGA data UDP port
                         eth_frame, frame_size);
endtask

//------------------------------------------------------------------------------
// send_host2sen_message: push msg_bytes of incrementing data to one sensor as a
//   single UC SEND message, segmented into <=HOST2SEN_PKT_PAYLOAD_MAX packets.
//   Every payload byte is recorded in expected_stx_bytes[sensor_id].
//------------------------------------------------------------------------------
task automatic send_host2sen_message(input int sensor_id, input int msg_bytes);
    automatic logic [7:0] payload[];
    automatic logic [7:0] eth_frame[];
    automatic int         frame_size;
    automatic int         remaining;
    automatic int         this_len;
    automatic int         pkt_idx;
    automatic int         num_pkts;
    automatic logic [7:0] opcode;
    automatic int         pmax;

    pmax     = `HOST2SEN_PKT_PAYLOAD_MAX;
    num_pkts = (msg_bytes + pmax - 1) / pmax;
    remaining = msg_bytes;
    pkt_idx   = 0;

    while (remaining > 0) begin
        // Honor an in-flight Ethernet PAUSE from the DUT before sending the next
        // packet (host_tx_paused is driven by the pause countdown in the TB top).
        if (host_tx_paused) begin
            `TB_LOG_MED(LOG_INFO, $sformatf("Host->Sensor[%0d] TX paused by DUT - waiting", sensor_id));
            wait (!host_tx_paused);
            `TB_LOG_MED(LOG_INFO, $sformatf("Host->Sensor[%0d] TX resumed", sensor_id));
        end

        this_len = (remaining < pmax) ? remaining : pmax;
        payload  = new[this_len];
        for (int i = 0; i < this_len; i++) begin
            payload[i] = stx_gen_byte[sensor_id][7:0];
            expected_stx_bytes[sensor_id].push_back(payload[i]);
            stx_gen_byte[sensor_id]++;
        end

        if (num_pkts == 1)                 opcode = ROCE_OP_SEND_ONLY;
        else if (pkt_idx == 0)             opcode = ROCE_OP_SEND_FIRST;
        else if (pkt_idx == num_pkts - 1)  opcode = ROCE_OP_SEND_LAST;
        else                               opcode = ROCE_OP_SEND_MIDDLE;

        build_host2sen_send_frame(sensor_id, payload, this_len, opcode,
                                  host2sen_psn[sensor_id][23:0], eth_frame, frame_size);
        send_eth_packet(0, eth_frame, frame_size);

        host2sen_bytes_sent[sensor_id] += this_len;
        host2sen_psn[sensor_id]++;
        pkt_idx++;
        remaining -= this_len;
    end
endtask

//------------------------------------------------------------------------------
// run_host2sen_traffic: top-level TX stimulus.  Sends HOST2SEN_MSGS_PER_SENSOR
//   messages to every sensor TX interface.
//------------------------------------------------------------------------------
task automatic run_host2sen_traffic();
    `TB_LOG_LOW(LOG_INFO, "=== HOST-TO-SENSOR (TX) TRAFFIC START ===");
    for (int m = 0; m < `HOST2SEN_MSGS_PER_SENSOR; m++) begin
        for (int s = 0; s < `SENSOR_TX_IF_INST; s++) begin
            `TB_LOG_LOW(LOG_INFO, $sformatf(
                "Host->Sensor[%0d] message %0d: %0d bytes, dest_qp=0x%06x",
                s, m, `HOST2SEN_MSG_BYTES, tx_dest_qp(s)));
            send_host2sen_message(s, `HOST2SEN_MSG_BYTES);
        end
    end
    `TB_LOG_LOW(LOG_INFO, "=== HOST-TO-SENSOR (TX) TRAFFIC DONE ===");
endtask
`endif // SENSOR_TX_IF_INST

`endif // HOLOLINK_HOST2SEN_STIM_SVH
