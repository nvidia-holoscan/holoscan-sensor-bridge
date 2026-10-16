// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Static packetizer datapath.  This preserves the buffered-cycle and virtual
// port structure of packetizer.sv, but removes all pattern RAM and the
// programmable odd/even sorting network.
//
// The catalog fixes the sorting network, so the geometry knobs the
// programmable packetizer takes from the platform are all determined here:
// the schedule depth comes from the enabled patterns, every virtual port
// commits a full DIN_WIDTH word, and the dynamic-commit resolution is unused.

module packetizer_static
  import packetizer_pkg::*;
#(
  parameter integer DIN_WIDTH     = 64,
  parameter integer VP_COUNT      = 1,
  parameter integer DOUT_WIDTH    = 64,
  parameter integer MTU           = 1500,
  // Optional pruning mask.  The default keeps every pattern the generated
  // catalog provides; a platform may narrow it to save area.
  parameter logic [63:0] PATTERN_MASK = {64{1'b1}},
  localparam integer DOUT_KWIDTH  = DOUT_WIDTH/8,
  // vp_ctrl only ever carries the commit bit, so the port takes one whole
  // DIN_WIDTH word at a time and needs no select field.
  localparam integer CTRL_WIDTH_VP = 2
) (
  input                                           i_sclk,
  input                                           i_srst,
  input                                           i_pclk,
  input                                           i_prst,
  input  [3:0]                                    pattern_cycle,
  input  [5:0]                                    pattern_select,
  output logic                                    pattern_supported,
  output logic [15:0]                             pattern_ld_arr,
  output logic [3:0]                              pattern_max_addr,
  output logic [3:0]                              pattern_latency,
  output [VP_COUNT-1:0]                           vp_is_empty,
  output [VP_COUNT-1:0]                           vp_almost_full,
  input  [DIN_WIDTH-1:0]                          din,
  input                                           din_valid,
  input                                           din_bypass,
  input                                           din_wait,
  input                                           din_stall,
  input                                           din_tlast,
  input                                           duplicate,
  output [VP_COUNT-1:0]                           o_axis_tvalid,
  output [VP_COUNT-1:0]                           o_axis_tlast,
  output [DOUT_WIDTH-1:0]                         o_axis_tdata [VP_COUNT-1:0],
  output [DOUT_KWIDTH-1:0]                        o_axis_tkeep [VP_COUNT-1:0],
  output [VP_COUNT-1:0]                           o_axis_tuser,
  input  [VP_COUNT-1:0]                           i_axis_tready
);

  // Patterns this instance can actually route: what the catalog provides,
  // narrowed by the optional platform pruning mask.
  localparam logic [63:0] ENABLED_PATTERNS = PATTERN_MASK & FIXED_PATTERN_ALL_MASK;
  localparam integer SELECT_WIDTH = fixed_pattern_select_width(DIN_WIDTH);
  localparam integer MAX_ROWS     = 16;

  // Deepest history any enabled pattern asks for.  Sizing the pipeline from
  // the catalog is what lets every routable pattern reach its selects: the
  // generator sets each map's selects within DIN_WIDTH*num_cycles, and
  // SELECT_WIDTH is derived from the same product.  An unsupported DIN_WIDTH
  // maps to zero and leaves the floor of one beat.
  function automatic integer max_pattern_num_cycles(input integer din_width,
                                                    input logic [63:0] enabled);
    logic [FIXED_PATTERN_MAP_WIDTH-1:0] map;
    integer id;
    integer cycles;
    begin
      max_pattern_num_cycles = 1;
      for (id = 1; id < 64; id = id + 1) begin
        if (enabled[id]) begin
          map    = fixed_pattern_map(din_width, id[5:0]);
          cycles = map[FIXED_PATTERN_NUM_CYCLES_LSB+:4];
          if (cycles > max_pattern_num_cycles) begin
            max_pattern_num_cycles = cycles;
          end
        end
      end
    end
  endfunction

  localparam integer NUM_CYCLES     = max_pattern_num_cycles(DIN_WIDTH, ENABLED_PATTERNS);
  localparam integer FLAT_DIN_WIDTH = DIN_WIDTH * NUM_CYCLES;

  logic [DIN_WIDTH-1:0] din_pipe [NUM_CYCLES-1:0];
  logic [FLAT_DIN_WIDTH-1:0] sorter_din;
  logic [DIN_WIDTH-1:0] sort_dout_nxt;
  logic [DIN_WIDTH-1:0] sort_dout;
  logic [1:0]           din_valid_pipe;
  logic [1:0]           din_tlast_pipe;
  logic                 bypass_pipe;
  // packetizer.sv reaches the sorter with the row that was current one cycle
  // before the beat arrived on din, so match that phase here.
  logic [3:0]           pattern_cycle_pipe [1:0];
  logic                 adv_pipe;

  logic [FIXED_PATTERN_MAP_WIDTH-1:0] input_pattern_map;

  // Schedule lookup for the incoming pattern ID.  Only the header fields are
  // used, so this reduces to a mux on the low bits.  An unsupported DIN_WIDTH
  // resolves to zero, which reports the pattern as unsupported and falls back
  // to bypass.
  assign input_pattern_map = fixed_pattern_map(DIN_WIDTH, pattern_select);

  always_comb begin
    pattern_supported = (pattern_select != 0) &&
                        ENABLED_PATTERNS[pattern_select] &&
                        (input_pattern_map[FIXED_PATTERN_ROWS_LSB+:5] != 0);
    pattern_ld_arr = input_pattern_map[FIXED_PATTERN_LD_ARR_LSB+:16];
    pattern_max_addr = input_pattern_map[FIXED_PATTERN_MAX_ADDR_LSB+:4];
    pattern_latency = input_pattern_map[FIXED_PATTERN_LATENCY_LSB+:4];
  end

  integer p;
  always_ff @(posedge i_sclk) begin
    if (i_srst) begin
      din_valid_pipe    <= '0;
      din_tlast_pipe    <= '0;
      bypass_pipe       <= '0;
      pattern_cycle_pipe[0] <= '0;
      pattern_cycle_pipe[1] <= '0;
      for (p = 0; p < NUM_CYCLES; p = p + 1) begin
        din_pipe[p] <= '0;
      end
    end
    else begin
      din_valid_pipe[0] <= din_valid && !din_wait;
      din_valid_pipe[1] <= din_valid_pipe[0];
      din_tlast_pipe[0] <= din_tlast;
      din_tlast_pipe[1] <= din_tlast_pipe[0];
      bypass_pipe       <= din_bypass;
      pattern_cycle_pipe[0] <= pattern_cycle;
      pattern_cycle_pipe[1] <= pattern_cycle_pipe[0];

      for (p = 1; p < NUM_CYCLES; p = p + 1) begin
        din_pipe[p] <= adv_pipe ? din_pipe[p-1] : din_pipe[p];
      end
      din_pipe[0] <= adv_pipe ? din : din_pipe[0];
    end
  end

  assign adv_pipe = din_valid && (!din_stall || din_wait);

  genvar cycle_idx;
  generate
    for (cycle_idx = 0; cycle_idx < NUM_CYCLES; cycle_idx = cycle_idx + 1) begin : gen_flatten_din
      assign sorter_din[cycle_idx*DIN_WIDTH+:DIN_WIDTH] = din_pipe[cycle_idx];
    end
  endgenerate

  // One comb arm per routable pattern.  Each map is a localparam so the
  // per-bit selects fold to wiring and a row mux.  Bypass (ID 0) and any
  // pattern the platform pruned are still present so the runtime mux has a
  // legal default.  Only the DIN_WIDTH results are muxed, then registered
  // once to keep the original one-cycle sorter latency.
  logic [DIN_WIDTH-1:0] pattern_dout [64];
  logic [5:0]           sort_pattern_sel;

  genvar pat_id, row, obit;
  generate
    for (pat_id = 0; pat_id < 64; pat_id = pat_id + 1) begin : gen_pattern
      if ((pat_id == 0) || ENABLED_PATTERNS[pat_id]) begin : gen_enabled
        localparam logic [FIXED_PATTERN_MAP_WIDTH-1:0] PATTERN_MAP =
            fixed_pattern_map(DIN_WIDTH, pat_id);
        localparam integer PATTERN_ROWS = PATTERN_MAP[FIXED_PATTERN_ROWS_LSB+:5];

        if (PATTERN_ROWS == 0) begin : gen_bypass
          assign pattern_dout[pat_id] = sorter_din[0+:DIN_WIDTH];
        end
        else begin : gen_routed
          logic [DIN_WIDTH-1:0] row_dout [MAX_ROWS];

          for (row = 0; row < MAX_ROWS; row = row + 1) begin : gen_row
            for (obit = 0; obit < DIN_WIDTH; obit = obit + 1) begin : gen_obit
              localparam integer SELECT = PATTERN_MAP[
                  FIXED_PATTERN_HEADER_WIDTH +
                  ((row*DIN_WIDTH)+obit)*SELECT_WIDTH
                  +:SELECT_WIDTH
              ];
              if ((row < PATTERN_ROWS) && (SELECT < FLAT_DIN_WIDTH)) begin : gen_route
                assign row_dout[row][obit] = sorter_din[SELECT];
              end
              else begin : gen_clear
                assign row_dout[row][obit] = 1'b0;
              end
            end
          end

          assign pattern_dout[pat_id] = row_dout[pattern_cycle_pipe[1]];
        end
      end
      else begin : gen_disabled
        assign pattern_dout[pat_id] = '0;
      end
    end
  endgenerate

  // packetizer_top holds pattern_select constant for the complete transaction.
  // Keep honoring the original tuser[0] runtime bypass for embedded traffic.
  assign sort_pattern_sel = !bypass_pipe && ENABLED_PATTERNS[pattern_select] ?
                            pattern_select : 6'd0;
  assign sort_dout_nxt    = pattern_dout[sort_pattern_sel];

  always_ff @(posedge i_sclk) begin
    sort_dout <= sort_dout_nxt;
  end

  genvar vp_idx;
  generate
    for (vp_idx = 0; vp_idx < VP_COUNT; vp_idx = vp_idx + 1) begin : gen_vp
      logic [CTRL_WIDTH_VP-1:0] vp_ctrl;

      // The fixed packetizer commits to VP0.  The existing duplicate control
      // may mirror the same stream to additional VPs.
      always_comb begin
        vp_ctrl    = '0;
        vp_ctrl[0] = (vp_idx == 0) || duplicate;
      end

      // RESOLUTION only scales the dynamic-size commits, which this datapath
      // does not use, so it keeps the virtual_port default.
      virtual_port #(
        .DIN_WIDTH  ( DIN_WIDTH                    ),
        .VP_SIZE    ( DIN_WIDTH                    ),
        .DOUT_WIDTH ( DOUT_WIDTH                   ),
        .CTRL_WIDTH ( CTRL_WIDTH_VP                ),
        .DYNAMIC_VP ( 0                            ),
        .MTU        ( MTU                          )
      ) u_vp (
        .i_sclk        ( i_sclk                    ),
        .i_srst        ( i_srst                    ),
        .i_pclk        ( i_pclk                    ),
        .i_prst        ( i_prst                    ),
        .din           ( sort_dout                 ),
        .din_valid     ( din_valid_pipe[1]         ),
        .din_tlast     ( din_tlast_pipe[1]         ),
        .ctrl          ( vp_ctrl                   ),
        .is_empty      ( vp_is_empty[vp_idx]       ),
        .almost_full   ( vp_almost_full[vp_idx]    ),
        .o_axis_tvalid ( o_axis_tvalid[vp_idx]     ),
        .o_axis_tlast  ( o_axis_tlast[vp_idx]      ),
        .o_axis_tdata  ( o_axis_tdata[vp_idx]      ),
        .o_axis_tkeep  ( o_axis_tkeep[vp_idx]      ),
        .o_axis_tuser  ( o_axis_tuser[vp_idx]      ),
        .i_axis_tready ( i_axis_tready[vp_idx]     )
      );
    end
  endgenerate


endmodule
