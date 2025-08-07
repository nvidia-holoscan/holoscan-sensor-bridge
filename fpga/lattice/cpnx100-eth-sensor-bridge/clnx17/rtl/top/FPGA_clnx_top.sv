
// Bajoran CrossLink-NX FPGA Top Level
module FPGA_clnx_top #(
  parameter [15:0] FPGA_VERSION = 16'h0101
)(
    input           PCLK_IN                 , // 125MHz
    input           RESET_N                 , // board reset push button
    input           SWRST_N                 ,
    // LVDS Data Output

    input           LVDS_CAM_RDY            ,
    output  [1:0]   LVDS_CAM_DCLK           ,
    output  [10:0]  LVDS_CAM_DATA [2]       ,
    // MIPI CAM Data Input
    inout   [1:0]   MIPI_CAM_CLK_P          ,
    inout   [1:0]   MIPI_CAM_CLK_N          ,
    inout   [3:0]   MIPI_CAM_DATA_P [2]     ,
    inout   [3:0]   MIPI_CAM_DATA_N [2]     ,
    // On-Board Device Control and Status
    input           CTRL_SPI_SCSN           ,
    input           CTRL_SPI_SSCK           ,
    input           CTRL_SPI_MOSI           ,
    output          CTRL_SPI_MISO           ,
    // SPI Flash Ctrl
    output          FLASH_SPI_MCSN          ,
    output          FLASH_SPI_MSCK          ,
    inout   [3:0]   FLASH_SPI_MOSI
);

//------------------------------------------------------------------------------
// Clock and Reset
//------------------------------------------------------------------------------

  logic eclk_i;
  logic eclk90_i;
  logic aclk_25;
  logic aclk_60;
  logic locked_lvds_pll;
  logic rst_n;

  assign rst_n = RESET_N & SWRST_N;

  lvds_pll u_lvds_pll(
    .clki_i   ( PCLK_IN         ),
    .rstn_i   ( rst_n           ),
    .clkop_o  ( eclk_i          ), // 500 MHz
    .clkos_o  ( eclk90_i        ), // 500 MHz 90 Degree
    .clkos2_o ( aclk_25         ), // 25 MHz
    .clkos3_o ( aclk_60         ), // 60 MHz
    .lock_o   ( locked_lvds_pll )
  );

  logic s_rst_n;
  logic a_rst_n;

  reset_sync u_rst (
    .i_clk     ( aclk_60         ),
    .i_arst_n  ( rst_n           ),
    .i_srst    ( 1'b0            ),
    .i_locked  ( locked_lvds_pll ),
    .o_arst    (                 ),
    .o_arst_n  ( a_rst_n         ),
    .o_srst    (                 ),
    .o_srst_n  ( s_rst_n         )
  );

//------------------------------------------------------------------------------
// MIPI Bridge: MIPI -> FIFO -> LVDS
//------------------------------------------------------------------------------
  localparam CAM_INST = 2;
  genvar i;

  logic [7:0]  rt_mipi_debug_en;
  logic [1:0]  mipi_en;
  logic [1:0]  soft_rstn;

  logic [1:0]  lmmi_ready;
  logic [7:0]  lmmi_rdata [1:0];
  logic [1:0]  lmmi_rdata_valid;
  logic [7:0]  lmmi_wdata;
  logic        lmmi_wr_rdn;
  logic [7:0]  lmmi_offset;
  logic [1:0]  lmmi_request;

  logic [7:0]  mipi_debug_length;

// Disable Normal MIPI CSI2 RX IP in MFG Test image
  generate
    for (i=0; i<CAM_INST; i=i+1) begin: mipi_rx
      mipi_bridge mipi_bridge_inst (
        // clock and reset
        .rst_n              ( rst_n                     ),   // async reset - synchronized within module
        .eclk_i             ( eclk_i                    ),   // 500MHz LVDS clk
        .eclk90_i           ( eclk90_i                  ),   // 500MHz LVDS clk 90 Degree
        .clk_25             ( aclk_25                   ),   // 25MHz Clock
        .mipi_sync_clk      ( aclk_60                   ),   // 60MHz
        .lmmi_clk           ( aclk_25                   ),   // lmmi_clk
        .pll_locked         ( locked_lvds_pll           ),
        .soft_rstn          ( soft_rstn            [i]  ),
        .mipi_en            ( mipi_en              [i]  ),   // Enable MIPI Core
        // LMMI Interface
        .lmmi_ready         ( lmmi_ready           [i]  ),
        .lmmi_rdata         ( lmmi_rdata           [i]  ),
        .lmmi_rdata_valid   ( lmmi_rdata_valid     [i]  ),
        .lmmi_wdata         ( lmmi_wdata                ),
        .lmmi_wr_rdn        ( lmmi_wr_rdn               ),
        .lmmi_offset        ( lmmi_offset               ),
        .lmmi_request       ( lmmi_request         [i]  ),
        // RX - MIPI Interface
        .mipi_clk_p_io      ( MIPI_CAM_CLK_P       [i]  ), // MIPI CLK
        .mipi_clk_n_io      ( MIPI_CAM_CLK_N       [i]  ),
        .mipi_data_p_io     ( MIPI_CAM_DATA_P      [i]  ), // MIPI DATA
        .mipi_data_n_io     ( MIPI_CAM_DATA_N      [i]  ),
        // TX - LVDS Interface
        .lvds_rdy_i         ( LVDS_CAM_RDY              ),
        .lvds_data_o        ( LVDS_CAM_DATA        [i]  ),
        .lvds_clk_o         ( LVDS_CAM_DCLK        [i]  )
      );
    end
  endgenerate

//------------------------------------------------------------------------------
// SPI CL Control + Regtbl
//------------------------------------------------------------------------------

  logic CTRL_SPI_SCSN_sync;
  logic CTRL_SPI_SSCK_sync;
  logic CTRL_SPI_MOSI_sync;
  logic FLASH_SPI_MOSI_1_sync;
  logic spi_flash_fwd;

  data_sync #(
    .DATA_WIDTH  ( 4                    ),
    .RESET_VALUE ( 1'b1                 )
  ) spi_sync (
    .clk         ( aclk_60               ),
    .rst_n       ( s_rst_n               ),
    .sync_in     ( {CTRL_SPI_SCSN     ,CTRL_SPI_SSCK     ,CTRL_SPI_MOSI     ,FLASH_SPI_MOSI[1]}     ),
    .sync_out    ( {CTRL_SPI_SCSN_sync,CTRL_SPI_SSCK_sync,CTRL_SPI_MOSI_sync,FLASH_SPI_MOSI_1_sync} )
  );

// Convert 4 bit SDIO to 2 bit MOSI/MISO
  wire [3:0] SDIO;
  assign SDIO[3:2] = 2'bzz;
  assign SDIO[0] = (1) ? CTRL_SPI_MOSI_sync : 1'bz;

  FPGA_spi_peri_ctrl_fsm # (
    .FPGA_VERSION(FPGA_VERSION)
  ) spi_peri_ctrl  (
    .clk                ( aclk_60               ),
    .rst_n              ( s_rst_n               ),
    .mipi_debug_en      ( rt_mipi_debug_en      ),
    .mipi_en            ( mipi_en               ),
    .soft_rstn          ( soft_rstn             ),
    .spi_flash_fwd      ( spi_flash_fwd         ),
    .debug_length       ( mipi_debug_length     ),
    .lmmi_ready         ( lmmi_ready            ),
    .lmmi_rdata         ( lmmi_rdata            ),
    .lmmi_rdata_valid   ( lmmi_rdata_valid      ),
    .lmmi_wdata         ( lmmi_wdata            ),
    .lmmi_wr_rdn        ( lmmi_wr_rdn           ),
    .lmmi_offset        ( lmmi_offset           ),
    .lmmi_request       ( lmmi_request          ),
    .CS_N               ( CTRL_SPI_SCSN_sync    ),
    .SCK                ( CTRL_SPI_SSCK_sync    ),
    .SDIO               ( SDIO[3:0]             )
  );

//------------------------------------------------------------------------------
// SPI Flash Control
//------------------------------------------------------------------------------

  reg r_flash_spi_mscn;
  reg r_flash_spi_msck;
  reg r_flash_spi_mosi;
  reg r_ctrl_spi_miso;

  always_ff @ (posedge aclk_60) begin
    if (spi_flash_fwd) begin
      r_flash_spi_mscn <= CTRL_SPI_SCSN_sync;
      r_flash_spi_msck <= CTRL_SPI_SSCK_sync;
      r_flash_spi_mosi <= CTRL_SPI_MOSI_sync;
      r_ctrl_spi_miso  <= FLASH_SPI_MOSI_1_sync;
    end
    else begin
      r_flash_spi_mscn <= 1'b1;
      r_flash_spi_msck <= 1'b1;
      r_flash_spi_mosi <= 1'bz;
      r_ctrl_spi_miso  <= SDIO[1];
    end
  end

  assign FLASH_SPI_MCSN = r_flash_spi_mscn;
  assign FLASH_SPI_MSCK = r_flash_spi_msck;
  assign FLASH_SPI_MOSI[0] = (spi_flash_fwd) ? r_flash_spi_mosi : 1'bz;
  assign FLASH_SPI_MOSI[1] = 1'bz;
  assign FLASH_SPI_MOSI[3:2] = 2'b11; //{WP_N,HOLD_N}
  assign CTRL_SPI_MISO  = r_ctrl_spi_miso;

endmodule
