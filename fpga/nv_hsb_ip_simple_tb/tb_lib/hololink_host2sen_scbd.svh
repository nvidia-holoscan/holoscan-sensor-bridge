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

`ifndef HOLOLINK_HOST2SEN_SCBD_SVH
`define HOLOLINK_HOST2SEN_SCBD_SVH

//==============================================================================
// Host-to-Sensor (TX) streaming scoreboard
//
// DESCRIPTION:
//   Pure streaming byte comparator for one sensor TX egress interface.  It does
//   NOT model packet structure or tlast: whatever byte stream the host pushed
//   (recorded in expected_stx_bytes[sensor]) must reappear, in order, on the
//   sensor egress.  Each valid byte (tkeep=1) on o_sif_axis_tdata is popped from
//   the expected queue and compared; any mismatch or extra byte is an error.
//==============================================================================

`ifdef SENSOR_TX_IF_INST
//------------------------------------------------------------------------------
// monitor_sensor_egress: run forever, collecting and checking sensor 's' egress.
//------------------------------------------------------------------------------
task automatic monitor_sensor_egress(input int s);
    automatic logic [7:0] exp_byte;
    automatic logic [7:0] got_byte;

    i_sif_axis_tready[s] = 1'b1;  // always accept from the DUT

    forever begin
        @(posedge i_sif_tx_clk[s]);
        if (o_sif_axis_tvalid[s] && i_sif_axis_tready[s]) begin
            for (int lane = 0; lane < `DATAKEEP_WIDTH; lane++) begin
                if (o_sif_axis_tkeep[s][lane]) begin
                    got_byte = o_sif_axis_tdata[s][lane*8 +: 8];
                    if (expected_stx_bytes[s].size() == 0) begin
                        errors_detected++;
                        host2sen_mismatches[s]++;
                        `TB_LOG_LOW(LOG_ERR, $sformatf(
                            "Host->Sensor[%0d] unexpected egress byte 0x%02x (expected queue empty)",
                            s, got_byte));
                    end
                    else begin
                        exp_byte = expected_stx_bytes[s].pop_front();
                        host2sen_bytes_checked[s]++;
                        if (got_byte !== exp_byte) begin
                            errors_detected++;
                            host2sen_mismatches[s]++;
                            `TB_LOG_LOW(LOG_ERR, $sformatf(
                                "Host->Sensor[%0d] MISMATCH at byte %0d: got=0x%02x exp=0x%02x",
                                s, host2sen_bytes_checked[s]-1, got_byte, exp_byte));
                        end
                    end
                end
            end
        end
    end
endtask

//------------------------------------------------------------------------------
// report_host2sen_results: summarize the TX path and set the final test_status.
//------------------------------------------------------------------------------
task automatic report_host2sen_results();
    automatic bit pass;
    pass = 1;

    `TB_LOG_LOW(LOG_INFO, "=== HOST-TO-SENSOR (TX) RESULTS SUMMARY ===");
    for (int s = 0; s < `SENSOR_TX_IF_INST; s++) begin
        `TB_LOG_LOW(LOG_INFO, $sformatf(
            "  Sensor[%0d]: dest_qp=0x%06x sent=%0d B checked=%0d B mismatches=%0d leftover_expected=%0d",
            s, tx_dest_qp(s), host2sen_bytes_sent[s], host2sen_bytes_checked[s],
            host2sen_mismatches[s], expected_stx_bytes[s].size()));

        if (host2sen_bytes_sent[s] == 0) begin
            pass = 0;
            `TB_LOG_LOW(LOG_ERR, $sformatf("  Sensor[%0d]: no host-to-sensor data was sent", s));
        end
        if (host2sen_mismatches[s] != 0)
            pass = 0;
        if (expected_stx_bytes[s].size() != 0) begin
            pass = 0;
            `TB_LOG_LOW(LOG_ERR, $sformatf(
                "  Sensor[%0d]: %0d expected bytes never received at egress",
                s, expected_stx_bytes[s].size()));
        end
        if (host2sen_bytes_checked[s] != host2sen_bytes_sent[s]) begin
            pass = 0;
            `TB_LOG_LOW(LOG_ERR, $sformatf(
                "  Sensor[%0d]: checked(%0d) != sent(%0d) bytes",
                s, host2sen_bytes_checked[s], host2sen_bytes_sent[s]));
        end
    end

    if (pass && errors_detected == 0) begin
        test_status = 1;
        `TB_LOG_LOW(LOG_PASS, "=== HOST-TO-SENSOR TEST PASSED ===");
    end
    else begin
        test_status = 2;
        `TB_LOG_LOW(LOG_FAIL, "=== HOST-TO-SENSOR TEST FAILED ===");
        `TB_LOG_LOW(LOG_FAIL, $sformatf("  - %0d total errors detected", errors_detected));
    end
endtask
`endif // SENSOR_TX_IF_INST

`endif // HOLOLINK_HOST2SEN_SCBD_SVH
