// An assertions file for basic checks on AXI-Stream interface
// Based on the FV interface file fv/hololink/interface/FV_AXI_S_check.sv

module axis_checker #(
   parameter  STBL_CHECK     = 1,
   parameter  NLST_BT_B2B    = 1,
   parameter  MIN_PKTL_CHK   = 1,
   parameter  MAX_PKTL_CHK   = 0,
   parameter  AXI_TDATA      = 64,
   localparam AXI_TKEEP      = AXI_TDATA/8,
   parameter  AXI_TUSER      = 1,
   parameter  PKT_MIN_LENGTH = 64,
   parameter  PKT_MAX_LENGTH = 4096,
   parameter  SIMULATION     = 0,
   parameter  BWIDTH         = 8,
   parameter  WWIDTH         = 2*BWIDTH,
   parameter  DWWIDTH        = 2*WWIDTH,
   parameter  QWIDTH         = 8*BWIDTH
)(
   input                        clk,
   input                        rst,
   input                        axis_tvalid,
   input                        axis_tlast,
   input      [AXI_TDATA-1 : 0] axis_tdata,
   input      [AXI_TKEEP-1 : 0] axis_tkeep,
   input      [AXI_TUSER-1 : 0] axis_tuser,
   input                        axis_tready,
   output reg [DWWIDTH-1   : 0] byte_count,
   output     [DWWIDTH-1   : 0] byte_count_nxt
);

assign byte_count_nxt = (byte_count + $countones(axis_tkeep));

always@ (posedge clk) begin
   if(rst) begin
      byte_count <= 'd0;
   end
   else begin
      if(axis_tvalid && axis_tready) begin
         if(axis_tlast) begin
            byte_count <= 'd0;
         end
         else begin
            byte_count <= byte_count_nxt;
         end
      end
   end
end

generate
if (STBL_CHECK) begin
asrt_if_tvld_hi_no_trdy_thn_tvld_stbl_nxt_cyc: assert property (
   @(posedge clk) disable iff(rst)
   (axis_tvalid && (!axis_tready)) |-> ##1 axis_tvalid
);
asrt_if_tvld_hi_no_trdy_thn_tlast_stbl_nxt_cyc: assert property (
   @(posedge clk) disable iff(rst)
   (axis_tvalid && (!axis_tready)) |-> ##1 (axis_tlast == $past(axis_tlast))
);
asrt_if_tvld_hi_no_trdy_thn_tdata_stbl_nxt_cyc: assert property (
   @(posedge clk) disable iff(rst)
   (axis_tvalid && (!axis_tready)) |-> ##1 (axis_tdata == $past(axis_tdata))
);
asrt_if_tvld_hi_no_trdy_thn_tkeep_stbl_nxt_cyc: assert property (
   @(posedge clk) disable iff(rst)
   (axis_tvalid && (!axis_tready)) |-> ##1 (axis_tkeep == $past(axis_tkeep))
);
asrt_if_tvld_hi_no_trdy_thn_tuser_stbl_nxt_cyc: assert property (
   @(posedge clk) disable iff(rst)
   (axis_tvalid && (!axis_tready)) |-> ##1 (axis_tuser == $past(axis_tuser))
);
end
if (NLST_BT_B2B) begin
asrt_if_tvld_hi_no_tlst_thn_tvld_hi_nxt_cyc: assert property (
   @(posedge clk) disable iff(rst)
   (axis_tvalid && (!axis_tlast)) |-> ##1 axis_tvalid
);
end
asrt_if_tvld_hi_no_tlast_thn_tkeep_1s: assert property (
   @(posedge clk) disable iff(rst)
   (axis_tvalid && (!axis_tlast)) |-> (axis_tkeep == '1)
);
asrt_if_tvld_hi_tlast_hi_thn_tkeep_gt_0: assert property (
   @(posedge clk) disable iff(rst)
   (axis_tvalid && axis_tlast) |-> (axis_tkeep > 'd0)
);
asrt_if_tvld_lst_bt_thn_tkp_condition: assert property (
   @(posedge clk) disable iff(rst)
   (axis_tvalid && axis_tlast) |->
   ((axis_tkeep>>$countones(axis_tkeep)) == 'd0)
);
// Note: we had to add an extra MSB to axis_tkeep to account for the legal case where axis_tkeep is all 1's
asrt_if_tvld_lst_bt_thn_tkp_pls_1_is_2pwr: assert property (
   @(posedge clk) disable iff(rst)
   (axis_tvalid && axis_tlast) |-> ($countones({1'b0, axis_tkeep} + 'd1) == 'd1)
);
`ifdef FV_ASSERT_ON
// Disabling the minimum packet size check in DV as per Bugs 5888880 and 5851797
if (MIN_PKTL_CHK && (AXI_TKEEP < PKT_MIN_LENGTH)) begin
// Usually we want AXIS packets of at least 64 bytes since it's Ethernet minimum
asrt_pkt_min_size: assert property (
   @(posedge clk) disable iff(rst)
   (axis_tvalid && (byte_count < (PKT_MIN_LENGTH - AXI_TKEEP))) |-> (!axis_tlast)
);
end
`endif
if (MAX_PKTL_CHK && (AXI_TKEEP < PKT_MAX_LENGTH)) begin
// Usually we want AXIS packets of at most 4096 bytes
asrt_pkt_max_size: assert property (
   @(posedge clk) disable iff(rst)
   (axis_tvalid && (byte_count >= (PKT_MAX_LENGTH - AXI_TKEEP))) |-> axis_tlast
);
end
if (SIMULATION) begin
// tdata cannot be x when tvalid is high
asrt_dv_only_tdata_not_x: assert property (
    @(posedge clk) disable iff (rst)
    (!axis_tvalid) || (!$isunknown(axis_tdata))
);
end
endgenerate

endmodule
