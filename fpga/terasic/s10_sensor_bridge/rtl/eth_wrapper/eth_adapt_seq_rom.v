module eth_adapt_seq_rom (
    input                   clk     ,
    input          [ 7:0]   addr    ,
    output logic   [71:0]   data
);

localparam CMD_WR   =  4'h1;
localparam CMD_RD   =  4'h2;
localparam CMD_VAL  =  4'h3;
localparam CMD_POLL =  4'h4;
localparam CMD_RMW  =  4'h5;

`ifdef ETH_25Gx4
localparam XCVR_BASE_0 = 24'h1_00000;
localparam XCVR_BASE_1 = 24'h3_00000;
localparam XCVR_BASE_2 = 24'h5_00000;
localparam XCVR_BASE_3 = 24'h7_00000;
`else
localparam XCVR_BASE_0 = 24'h20_0000>>2;
localparam XCVR_BASE_1 = 24'h40_0000>>2;
localparam XCVR_BASE_2 = 24'h60_0000>>2;
localparam XCVR_BASE_3 = 24'h80_0000>>2;
`endif
localparam RSFEC_BASE  = 24'h1_0000;

    always @ (posedge clk) begin
        case (addr)         
            //                  PAD       , ADDR                            , DATA  , MASK  , DATAVAL , CMD     
`ifdef ETH_25G
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //START: Assert Rx and Tx Digital Resets on Eth IP
            8'd000:     data = {12'h00   , (32'h310                   << 2) , 8'h06 , 8'h00 , 8'h00   , CMD_WR    };    
            //END: Assert Rx and Tx Digital Resets on Eth IP
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////
            
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //START: PMA Analog reset on all 4 XCVR channels in Eth IP 
            8'd001:     data = {12'h00   , ((XCVR_BASE_0 + 32'h200)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd002:     data = {12'h00   , ((XCVR_BASE_0 + 32'h201)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd003:     data = {12'h00   , ((XCVR_BASE_0 + 32'h202)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd004:     data = {12'h00   , ((XCVR_BASE_0 + 32'h203)   << 2) , 8'h81 , 8'h00 , 8'h00   , CMD_WR    };
            8'd005:     data = {12'h00   , ((XCVR_BASE_0 + 32'h207)   << 2) , 8'h00 , 8'hFF , 8'h80   , CMD_POLL  };            
            8'd006:     data = {12'h00   , ((XCVR_BASE_0 + 32'h095)   << 2) , 8'h00 , 8'h20 , 8'h00   , CMD_RMW   };            
            8'd007:     data = {12'h00   , ((XCVR_BASE_0 + 32'h091)   << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };            
            8'd008:     data = {12'h00   , (32'h310                   << 2) , 8'h05 , 8'h00 , 8'h00   , CMD_WR    };
            8'd009:     data = {12'h00   , (32'h310                   << 2) , 8'h04 , 8'h00 , 8'h00   , CMD_WR    };
            
            
            8'd010:     data = {12'h00   , ((XCVR_BASE_0 + 32'h40143) << 2) , 8'h80 , 8'h00 , 8'h00   , CMD_WR    };
            8'd011:     data = {12'h00   , ((XCVR_BASE_0 + 32'h40144) << 2) , 8'h00 , 8'h01 , 8'h01   , CMD_POLL  };
            //  The following sequence is for all 4 channels
            8'd012:     data = {12'h00   , ((XCVR_BASE_0 + 32'h200)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd013:     data = {12'h00   , ((XCVR_BASE_0 + 32'h201)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd014:     data = {12'h00   , ((XCVR_BASE_0 + 32'h202)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd015:     data = {12'h00   , ((XCVR_BASE_0 + 32'h203)   << 2) , 8'h94 , 8'h00 , 8'h00   , CMD_WR    };
            8'd016:     data = {12'h00   , ((XCVR_BASE_0 + 32'h207)   << 2) , 8'h00 , 8'hFF , 8'h80   , CMD_POLL  };
            8'd017:     data = {12'h00   , ((XCVR_BASE_0 + 32'h8A)    << 2) , 8'h80 , 8'h00 , 8'h00   , CMD_WR    };
            8'd018:     data = {12'h00   , ((XCVR_BASE_0 + 32'h84)    << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };
            8'd019:     data = {12'h00   , ((XCVR_BASE_0 + 32'h85)    << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };
            8'd020:     data = {12'h00   , ((XCVR_BASE_0 + 32'h86)    << 2) , 8'h08 , 8'h00 , 8'h00   , CMD_WR    };
            8'd021:     data = {12'h00   , ((XCVR_BASE_0 + 32'h87)    << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd022:     data = {12'h00   , ((XCVR_BASE_0 + 32'h90)    << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };
            8'd023:     data = {12'h00   , ((XCVR_BASE_0 + 32'h8A)    << 2) , 8'h00 , 8'h80 , 8'h80   , CMD_POLL  };
            8'd024:     data = {12'h00   , ((XCVR_BASE_0 + 32'h8B)    << 2) , 8'h00 , 8'h01 , 8'h00   , CMD_POLL  };
            8'd025:     data = {12'h00   , ((XCVR_BASE_0 + 32'h200)   << 2) , 8'hE1 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd026:     data = {12'h00   , ((XCVR_BASE_0 + 32'h201)   << 2) , 8'h03 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd027:     data = {12'h00   , ((XCVR_BASE_0 + 32'h202)   << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };      
            8'd028:     data = {12'h00   , ((XCVR_BASE_0 + 32'h203)   << 2) , 8'h96 , 8'h00 , 8'h00   , CMD_WR    };
            //8'd001:     data = {12'h00   , ((XCVR_BASE_0 + 32'h207) << 2) , 8'h00 , 8'hFF , 8'h80   , CMD_POLL    };    //Verify no POLL is needed here
            //  Get Cal Status
            8'd029:     data = {12'h00   , ((XCVR_BASE_0 + 32'h200)   << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd030:     data = {12'h00   , ((XCVR_BASE_0 + 32'h201)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd031:     data = {12'h00   , ((XCVR_BASE_0 + 32'h202)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };      
            8'd032:     data = {12'h00   , ((XCVR_BASE_0 + 32'h203)   << 2) , 8'h97 , 8'h00 , 8'h00   , CMD_WR    };
            
            8'd033:     data = {12'h00   , ((XCVR_BASE_0 + 32'h200)   << 2) , 8'hE2 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd034:     data = {12'h00   , ((XCVR_BASE_0 + 32'h201)   << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd035:     data = {12'h00   , ((XCVR_BASE_0 + 32'h202)   << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };      
            8'd036:     data = {12'h00   , ((XCVR_BASE_0 + 32'h203)   << 2) , 8'h96 , 8'h00 , 8'h00   , CMD_WR    };
            8'd037:     data = {12'h00   , ((XCVR_BASE_0 + 32'h200)   << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd038:     data = {12'h00   , ((XCVR_BASE_0 + 32'h201)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd039:     data = {12'h00   , ((XCVR_BASE_0 + 32'h202)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };      
            8'd040:     data = {12'h00   , ((XCVR_BASE_0 + 32'h203)   << 2) , 8'h97 , 8'h00 , 8'h00   , CMD_WR    };
            8'd041:     data = {12'h00   , (32'h310                   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };    
            8'd042:     data = {12'h00   , ((XCVR_BASE_0 + 32'h207)   << 2) , 8'h00 , 8'hFF , 8'h80   , CMD_POLL  };
            
            
            8'd043:     data = {12'h00   , ((XCVR_BASE_0 + 32'h8A)    << 2) , 8'h80 , 8'h00 , 8'h00   , CMD_WR    };
            8'd044:     data = {12'h00   , ((XCVR_BASE_0 + 32'h84)    << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd045:     data = {12'h00   , ((XCVR_BASE_0 + 32'h85)    << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };
            8'd046:     data = {12'h00   , ((XCVR_BASE_0 + 32'h86)    << 2) , 8'h08 , 8'h00 , 8'h00   , CMD_WR    };
            8'd047:     data = {12'h00   , ((XCVR_BASE_0 + 32'h87)    << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd048:     data = {12'h00   , ((XCVR_BASE_0 + 32'h90)    << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };
            8'd049:     data = {12'h00   , ((XCVR_BASE_0 + 32'h8A)    << 2) , 8'h00 , 8'h80 , 8'h80   , CMD_POLL  };
            8'd050:     data = {12'h00   , ((XCVR_BASE_0 + 32'h8B)    << 2) , 8'h00 , 8'h01 , 8'h00   , CMD_POLL  };
            8'd051:     data = {12'h00   , ((RSFEC_BASE + 32'h8B)     << 2) , 8'h00 , 8'h01 , 8'h00   , CMD_POLL  };
`else
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //START: Assert Rx and Tx Digital Resets on Eth IP
            8'd000:     data = {12'h00   , (32'h310                   << 2) , 8'h06 , 8'h00 , 8'h00   , CMD_WR    };    
            //END: Assert Rx and Tx Digital Resets on Eth IP
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////
            
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //START: PMA Analog reset on all 4 XCVR channels in Eth IP 
            8'd001:     data = {12'h00   , ((XCVR_BASE_0 + 32'h200)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd002:     data = {12'h00   , ((XCVR_BASE_0 + 32'h201)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd003:     data = {12'h00   , ((XCVR_BASE_0 + 32'h202)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd004:     data = {12'h00   , ((XCVR_BASE_0 + 32'h203)   << 2) , 8'h81 , 8'h00 , 8'h00   , CMD_WR    };
            8'd005:     data = {12'h00   , ((XCVR_BASE_0 + 32'h207)   << 2) , 8'h00 , 8'hFF , 8'h80   , CMD_POLL  };
            8'd006:     data = {12'h00   , ((XCVR_BASE_1 + 32'h200)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd007:     data = {12'h00   , ((XCVR_BASE_1 + 32'h201)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd008:     data = {12'h00   , ((XCVR_BASE_1 + 32'h202)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd009:     data = {12'h00   , ((XCVR_BASE_1 + 32'h203)   << 2) , 8'h81 , 8'h00 , 8'h00   , CMD_WR    };
            8'd010:     data = {12'h00   , ((XCVR_BASE_1 + 32'h207)   << 2) , 8'h00 , 8'hFF , 8'h80   , CMD_POLL  };
            8'd011:     data = {12'h00   , ((XCVR_BASE_2 + 32'h200)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd012:     data = {12'h00   , ((XCVR_BASE_2 + 32'h201)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd013:     data = {12'h00   , ((XCVR_BASE_2 + 32'h202)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd014:     data = {12'h00   , ((XCVR_BASE_2 + 32'h203)   << 2) , 8'h81 , 8'h00 , 8'h00   , CMD_WR    };
            8'd015:     data = {12'h00   , ((XCVR_BASE_2 + 32'h207)   << 2) , 8'h00 , 8'hFF , 8'h80   , CMD_POLL  };
            8'd016:     data = {12'h00   , ((XCVR_BASE_3 + 32'h200)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd017:     data = {12'h00   , ((XCVR_BASE_3 + 32'h201)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd018:     data = {12'h00   , ((XCVR_BASE_3 + 32'h202)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd019:     data = {12'h00   , ((XCVR_BASE_3 + 32'h203)   << 2) , 8'h81 , 8'h00 , 8'h00   , CMD_WR    };
            8'd020:     data = {12'h00   , ((XCVR_BASE_3 + 32'h207)   << 2) , 8'h00 , 8'hFF , 8'h80   , CMD_POLL  };
            //END: PMA Analog reset on all 4 XCVR channels in Eth IP 
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////
            
            ////////////////////////// THIS IS WHERE THE 100MSEC WAIT WOULD BE ///////////////////////////////////////////
            
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //START: Enable PMA calibration when loading new settings for all 4 XCVR channels in Eth IP
            8'd021:     data = {12'h00   , ((XCVR_BASE_0 + 32'h095)   << 2) , 8'h00 , 8'h20 , 8'h00   , CMD_RMW   };
            8'd022:     data = {12'h00   , ((XCVR_BASE_1 + 32'h095)   << 2) , 8'h00 , 8'h20 , 8'h00   , CMD_RMW   };
            8'd023:     data = {12'h00   , ((XCVR_BASE_2 + 32'h095)   << 2) , 8'h00 , 8'h20 , 8'h00   , CMD_RMW   };
            8'd024:     data = {12'h00   , ((XCVR_BASE_3 + 32'h095)   << 2) , 8'h00 , 8'h20 , 8'h00   , CMD_RMW   };
            //END: Enable PMA calibration when loading new settings for all 4 XCVR channels in Eth IP
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////
            
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////                ///ADD POLL FOR 0x91[0] to 0x0?
            //START: Reload SERDES Settings for all 4 XCVR channels in Eth IP
            8'd025:     data = {12'h00   , ((XCVR_BASE_0 + 32'h091)   << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };
            8'd026:     data = {12'h00   , ((XCVR_BASE_1 + 32'h091)   << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };
            8'd027:     data = {12'h00   , ((XCVR_BASE_2 + 32'h091)   << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };
            8'd028:     data = {12'h00   , ((XCVR_BASE_3 + 32'h091)   << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };
            //END: Reload SERDES Settings for all 4 XCVR channels in Eth IP
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////
            
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////                ///Change to assert eios (keep tx and rx high. Deassert eios. Deassert tx.
            //START: De-assert Tx digital reset, keep RX asserted, and assert EIOS reset
            8'd029:     data = {12'h00   , (32'h310                   << 2) , 8'h05 , 8'h00 , 8'h00   , CMD_WR    };
            //END: De-assert Tx digital reset, keep RX asserted, and assert EIOS reset
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////
            
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //START: De-assert EIOS reset
            8'd030:     data = {12'h00   , (32'h310                   << 2) , 8'h04 , 8'h00 , 8'h00   , CMD_WR    };
            //END: De-assert EIOS reset
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////
            
            //Need a wait state here. Not sure why yet. Perhaps monitoring IP stats will tell us...
            
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //START: Load PMA Configuration for Initial Adaptation Mode
            //  This write/poll is for channel 0 only because this register only exists in channel 0 (per E-tile doc).
            8'd031:     data = {12'h00   , ((XCVR_BASE_0 + 32'h40143) << 2) , 8'h80 , 8'h00 , 8'h00   , CMD_WR    };
            8'd032:     data = {12'h00   , ((XCVR_BASE_0 + 32'h40144) << 2) , 8'h00 , 8'h01 , 8'h01   , CMD_POLL  };
            //  The following sequence is for all 4 channels
            8'd033:     data = {12'h00   , ((XCVR_BASE_0 + 32'h200)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd034:     data = {12'h00   , ((XCVR_BASE_0 + 32'h201)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd035:     data = {12'h00   , ((XCVR_BASE_0 + 32'h202)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd036:     data = {12'h00   , ((XCVR_BASE_0 + 32'h203)   << 2) , 8'h94 , 8'h00 , 8'h00   , CMD_WR    };
            8'd037:     data = {12'h00   , ((XCVR_BASE_0 + 32'h207)   << 2) , 8'h00 , 8'hFF , 8'h80   , CMD_POLL  };
            8'd038:     data = {12'h00   , ((XCVR_BASE_1 + 32'h200)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd039:     data = {12'h00   , ((XCVR_BASE_1 + 32'h201)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd040:     data = {12'h00   , ((XCVR_BASE_1 + 32'h202)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd041:     data = {12'h00   , ((XCVR_BASE_1 + 32'h203)   << 2) , 8'h94 , 8'h00 , 8'h00   , CMD_WR    };
            8'd042:     data = {12'h00   , ((XCVR_BASE_1 + 32'h207)   << 2) , 8'h00 , 8'hFF , 8'h80   , CMD_POLL  };
            8'd043:     data = {12'h00   , ((XCVR_BASE_2 + 32'h200)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd044:     data = {12'h00   , ((XCVR_BASE_2 + 32'h201)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd045:     data = {12'h00   , ((XCVR_BASE_2 + 32'h202)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd046:     data = {12'h00   , ((XCVR_BASE_2 + 32'h203)   << 2) , 8'h94 , 8'h00 , 8'h00   , CMD_WR    };
            8'd047:     data = {12'h00   , ((XCVR_BASE_2 + 32'h207)   << 2) , 8'h00 , 8'hFF , 8'h80   , CMD_POLL  };
            8'd048:     data = {12'h00   , ((XCVR_BASE_3 + 32'h200)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd049:     data = {12'h00   , ((XCVR_BASE_3 + 32'h201)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd050:     data = {12'h00   , ((XCVR_BASE_3 + 32'h202)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd051:     data = {12'h00   , ((XCVR_BASE_3 + 32'h203)   << 2) , 8'h94 , 8'h00 , 8'h00   , CMD_WR    };
            8'd052:     data = {12'h00   , ((XCVR_BASE_3 + 32'h207)   << 2) , 8'h00 , 8'hFF , 8'h80   , CMD_POLL  };
            //END: Load PMA Configuration for Initial Adaptation Mode
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////
            
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //START: Enable serial loopback mode for all channels
            8'd053:     data = {12'h00   , ((XCVR_BASE_0 + 32'h8A)    << 2) , 8'h80 , 8'h00 , 8'h00   , CMD_WR    };
            8'd054:     data = {12'h00   , ((XCVR_BASE_0 + 32'h84)    << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };
            8'd055:     data = {12'h00   , ((XCVR_BASE_0 + 32'h85)    << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };
            8'd056:     data = {12'h00   , ((XCVR_BASE_0 + 32'h86)    << 2) , 8'h08 , 8'h00 , 8'h00   , CMD_WR    };
            8'd057:     data = {12'h00   , ((XCVR_BASE_0 + 32'h87)    << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd058:     data = {12'h00   , ((XCVR_BASE_0 + 32'h90)    << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };
            8'd059:     data = {12'h00   , ((XCVR_BASE_0 + 32'h8A)    << 2) , 8'h00 , 8'h80 , 8'h80   , CMD_POLL  };
            8'd060:     data = {12'h00   , ((XCVR_BASE_0 + 32'h8B)    << 2) , 8'h00 , 8'h01 , 8'h00   , CMD_POLL  };
            //data0=eth_reconfig_read(intf,ch,0x88)
            //data1=eth_reconfig_read(intf,ch,0x89)
            8'd061:     data = {12'h00   , ((XCVR_BASE_1 + 32'h8A)    << 2) , 8'h80 , 8'h00 , 8'h00   , CMD_WR    };
            8'd062:     data = {12'h00   , ((XCVR_BASE_1 + 32'h84)    << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };
            8'd063:     data = {12'h00   , ((XCVR_BASE_1 + 32'h85)    << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };
            8'd064:     data = {12'h00   , ((XCVR_BASE_1 + 32'h86)    << 2) , 8'h08 , 8'h00 , 8'h00   , CMD_WR    };
            8'd065:     data = {12'h00   , ((XCVR_BASE_1 + 32'h87)    << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd066:     data = {12'h00   , ((XCVR_BASE_1 + 32'h90)    << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };
            8'd067:     data = {12'h00   , ((XCVR_BASE_1 + 32'h8A)    << 2) , 8'h00 , 8'h80 , 8'h80   , CMD_POLL  };
            8'd068:     data = {12'h00   , ((XCVR_BASE_1 + 32'h8B)    << 2) , 8'h00 , 8'h01 , 8'h00   , CMD_POLL  };
            //data0=eth_reconfig_read(intf,ch,0x88)
            //data1=eth_reconfig_read(intf,ch,0x89)
            8'd069:     data = {12'h00   , ((XCVR_BASE_2 + 32'h8A)    << 2) , 8'h80 , 8'h00 , 8'h00   , CMD_WR    };
            8'd070:     data = {12'h00   , ((XCVR_BASE_2 + 32'h84)    << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };
            8'd071:     data = {12'h00   , ((XCVR_BASE_2 + 32'h85)    << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };
            8'd072:     data = {12'h00   , ((XCVR_BASE_2 + 32'h86)    << 2) , 8'h08 , 8'h00 , 8'h00   , CMD_WR    };
            8'd073:     data = {12'h00   , ((XCVR_BASE_2 + 32'h87)    << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd074:     data = {12'h00   , ((XCVR_BASE_2 + 32'h90)    << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };
            8'd075:     data = {12'h00   , ((XCVR_BASE_2 + 32'h8A)    << 2) , 8'h00 , 8'h80 , 8'h80   , CMD_POLL  };
            8'd076:     data = {12'h00   , ((XCVR_BASE_2 + 32'h8B)    << 2) , 8'h00 , 8'h01 , 8'h00   , CMD_POLL  };
            //data0=eth_reconfig_read(intf,ch,0x88)
            //data1=eth_reconfig_read(intf,ch,0x89)
            8'd077:     data = {12'h00   , ((XCVR_BASE_3 + 32'h8A)    << 2) , 8'h80 , 8'h00 , 8'h00   , CMD_WR    };
            8'd078:     data = {12'h00   , ((XCVR_BASE_3 + 32'h84)    << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };
            8'd079:     data = {12'h00   , ((XCVR_BASE_3 + 32'h85)    << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };
            8'd080:     data = {12'h00   , ((XCVR_BASE_3 + 32'h86)    << 2) , 8'h08 , 8'h00 , 8'h00   , CMD_WR    };
            8'd081:     data = {12'h00   , ((XCVR_BASE_3 + 32'h87)    << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd082:     data = {12'h00   , ((XCVR_BASE_3 + 32'h90)    << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };
            8'd083:     data = {12'h00   , ((XCVR_BASE_3 + 32'h8A)    << 2) , 8'h00 , 8'h80 , 8'h80   , CMD_POLL  };
            8'd084:     data = {12'h00   , ((XCVR_BASE_3 + 32'h8B)    << 2) , 8'h00 , 8'h01 , 8'h00   , CMD_POLL  };
            //data0=eth_reconfig_read(intf,ch,0x88)
            //data1=eth_reconfig_read(intf,ch,0x89)
            //END: Enable serial loopback mode for all channels
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////
            
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //START: Perform adaptation for each channel - internal mode
            8'd085:     data = {12'h00   , ((XCVR_BASE_0 + 32'h200)   << 2) , 8'hE1 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd086:     data = {12'h00   , ((XCVR_BASE_0 + 32'h201)   << 2) , 8'h03 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd087:     data = {12'h00   , ((XCVR_BASE_0 + 32'h202)   << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };      
            8'd088:     data = {12'h00   , ((XCVR_BASE_0 + 32'h203)   << 2) , 8'h96 , 8'h00 , 8'h00   , CMD_WR    };
            //8'd001:     data = {12'h00   , ((XCVR_BASE_0 + 32'h207) << 2) , 8'h00 , 8'hFF , 8'h80   , CMD_POLL    };    //Verify no POLL is needed here
            //  Get Cal Status
            8'd089:     data = {12'h00   , ((XCVR_BASE_0 + 32'h200)   << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd090:     data = {12'h00   , ((XCVR_BASE_0 + 32'h201)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd091:     data = {12'h00   , ((XCVR_BASE_0 + 32'h202)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };      
            8'd092:     data = {12'h00   , ((XCVR_BASE_0 + 32'h203)   << 2) , 8'h97 , 8'h00 , 8'h00   , CMD_WR    };
            //  
            8'd093:     data = {12'h00   , ((XCVR_BASE_1 + 32'h200)   << 2) , 8'hE1 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd094:     data = {12'h00   , ((XCVR_BASE_1 + 32'h201)   << 2) , 8'h03 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd095:     data = {12'h00   , ((XCVR_BASE_1 + 32'h202)   << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };      
            8'd096:     data = {12'h00   , ((XCVR_BASE_1 + 32'h203)   << 2) , 8'h96 , 8'h00 , 8'h00   , CMD_WR    };
            // Get Cal Status 
            8'd097:     data = {12'h00   , ((XCVR_BASE_1 + 32'h200)   << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd098:     data = {12'h00   , ((XCVR_BASE_1 + 32'h201)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd099:     data = {12'h00   , ((XCVR_BASE_1 + 32'h202)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };      
            8'd100:     data = {12'h00   , ((XCVR_BASE_1 + 32'h203)   << 2) , 8'h97 , 8'h00 , 8'h00   , CMD_WR    };
            //  
            8'd101:     data = {12'h00   , ((XCVR_BASE_2 + 32'h200)   << 2) , 8'hE1 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd102:     data = {12'h00   , ((XCVR_BASE_2 + 32'h201)   << 2) , 8'h03 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd103:     data = {12'h00   , ((XCVR_BASE_2 + 32'h202)   << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };      
            8'd104:     data = {12'h00   , ((XCVR_BASE_2 + 32'h203)   << 2) , 8'h96 , 8'h00 , 8'h00   , CMD_WR    };
            //Get Cal Status  
            8'd105:     data = {12'h00   , ((XCVR_BASE_2 + 32'h200)   << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd106:     data = {12'h00   , ((XCVR_BASE_2 + 32'h201)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd107:     data = {12'h00   , ((XCVR_BASE_2 + 32'h202)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };      
            8'd108:     data = {12'h00   , ((XCVR_BASE_2 + 32'h203)   << 2) , 8'h97 , 8'h00 , 8'h00   , CMD_WR    };
            //  
            8'd109:     data = {12'h00   , ((XCVR_BASE_3 + 32'h200)   << 2) , 8'hE1 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd110:     data = {12'h00   , ((XCVR_BASE_3 + 32'h201)   << 2) , 8'h03 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd111:     data = {12'h00   , ((XCVR_BASE_3 + 32'h202)   << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };      
            8'd112:     data = {12'h00   , ((XCVR_BASE_3 + 32'h203)   << 2) , 8'h96 , 8'h00 , 8'h00   , CMD_WR    };
            //Get Cal Status  
            8'd113:     data = {12'h00   , ((XCVR_BASE_3 + 32'h200)   << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd114:     data = {12'h00   , ((XCVR_BASE_3 + 32'h201)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd115:     data = {12'h00   , ((XCVR_BASE_3 + 32'h202)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };      
            8'd116:     data = {12'h00   , ((XCVR_BASE_3 + 32'h203)   << 2) , 8'h97 , 8'h00 , 8'h00   , CMD_WR    };
            //END: Perform adaptation for each channel - internal mode
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////
            
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //START: Perform adaptation for each channel - external mode
            8'd117:     data = {12'h00   , ((XCVR_BASE_0 + 32'h200)   << 2) , 8'hE2 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd118:     data = {12'h00   , ((XCVR_BASE_0 + 32'h201)   << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd119:     data = {12'h00   , ((XCVR_BASE_0 + 32'h202)   << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };      
            8'd120:     data = {12'h00   , ((XCVR_BASE_0 + 32'h203)   << 2) , 8'h96 , 8'h00 , 8'h00   , CMD_WR    };
            //8'd001:     data = {12'h00   , ((XCVR_BASE_0 + 32'h207) << 2) , 8'h00 , 8'hFF , 8'h80   , CMD_POLL    };    //Verify no POLL is needed here
            //  Get Cal Status
            8'd121:     data = {12'h00   , ((XCVR_BASE_0 + 32'h200)   << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd122:     data = {12'h00   , ((XCVR_BASE_0 + 32'h201)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd123:     data = {12'h00   , ((XCVR_BASE_0 + 32'h202)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };      
            8'd124:     data = {12'h00   , ((XCVR_BASE_0 + 32'h203)   << 2) , 8'h97 , 8'h00 , 8'h00   , CMD_WR    };
            //  
            8'd125:     data = {12'h00   , ((XCVR_BASE_1 + 32'h200)   << 2) , 8'hE2 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd126:     data = {12'h00   , ((XCVR_BASE_1 + 32'h201)   << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd127:     data = {12'h00   , ((XCVR_BASE_1 + 32'h202)   << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };      
            8'd128:     data = {12'h00   , ((XCVR_BASE_1 + 32'h203)   << 2) , 8'h96 , 8'h00 , 8'h00   , CMD_WR    };
            // Get Cal Status 
            8'd129:     data = {12'h00   , ((XCVR_BASE_1 + 32'h200)   << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd130:     data = {12'h00   , ((XCVR_BASE_1 + 32'h201)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd131:     data = {12'h00   , ((XCVR_BASE_1 + 32'h202)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };      
            8'd132:     data = {12'h00   , ((XCVR_BASE_1 + 32'h203)   << 2) , 8'h97 , 8'h00 , 8'h00   , CMD_WR    };
            //  
            8'd133:     data = {12'h00   , ((XCVR_BASE_2 + 32'h200)   << 2) , 8'hE2 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd134:     data = {12'h00   , ((XCVR_BASE_2 + 32'h201)   << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd135:     data = {12'h00   , ((XCVR_BASE_2 + 32'h202)   << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };      
            8'd136:     data = {12'h00   , ((XCVR_BASE_2 + 32'h203)   << 2) , 8'h96 , 8'h00 , 8'h00   , CMD_WR    };
            //Get Cal Status  
            8'd137:     data = {12'h00   , ((XCVR_BASE_2 + 32'h200)   << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd138:     data = {12'h00   , ((XCVR_BASE_2 + 32'h201)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd139:     data = {12'h00   , ((XCVR_BASE_2 + 32'h202)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };      
            8'd140:     data = {12'h00   , ((XCVR_BASE_2 + 32'h203)   << 2) , 8'h97 , 8'h00 , 8'h00   , CMD_WR    };
            //  
            8'd141:     data = {12'h00   , ((XCVR_BASE_3 + 32'h200)   << 2) , 8'hE2 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd142:     data = {12'h00   , ((XCVR_BASE_3 + 32'h201)   << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd143:     data = {12'h00   , ((XCVR_BASE_3 + 32'h202)   << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };      
            8'd144:     data = {12'h00   , ((XCVR_BASE_3 + 32'h203)   << 2) , 8'h96 , 8'h00 , 8'h00   , CMD_WR    };
            //Get Cal Status  
            8'd145:     data = {12'h00   , ((XCVR_BASE_3 + 32'h200)   << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd146:     data = {12'h00   , ((XCVR_BASE_3 + 32'h201)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };        
            8'd147:     data = {12'h00   , ((XCVR_BASE_3 + 32'h202)   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };      
            8'd148:     data = {12'h00   , ((XCVR_BASE_3 + 32'h203)   << 2) , 8'h97 , 8'h00 , 8'h00   , CMD_WR    };
            //END: Perform adaptation for each channel - external mode
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////
            
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //START: De-assert RX reset  
            8'd149:     data = {12'h00   , (32'h310                   << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };    
            //END: De-assert RX reset  
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////
            
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //START: Check adaptation status for each channel  
            8'd150:     data = {12'h00   , ((XCVR_BASE_0 + 32'h207)   << 2) , 8'h00 , 8'hFF , 8'h80   , CMD_POLL  };
            8'd151:     data = {12'h00   , ((XCVR_BASE_1 + 32'h207)   << 2) , 8'h00 , 8'hFF , 8'h80   , CMD_POLL  };
            8'd152:     data = {12'h00   , ((XCVR_BASE_2 + 32'h207)   << 2) , 8'h00 , 8'hFF , 8'h80   , CMD_POLL  };
            8'd153:     data = {12'h00   , ((XCVR_BASE_3 + 32'h207)   << 2) , 8'h00 , 8'hFF , 8'h80   , CMD_POLL  };
            //END: Check adaptation status for each channel  
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////
            
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //START: Disable serial loopback mode for all channels
            8'd154:     data = {12'h00   , ((XCVR_BASE_0 + 32'h8A)    << 2) , 8'h80 , 8'h00 , 8'h00   , CMD_WR    };
            8'd155:     data = {12'h00   , ((XCVR_BASE_0 + 32'h84)    << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd156:     data = {12'h00   , ((XCVR_BASE_0 + 32'h85)    << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };
            8'd157:     data = {12'h00   , ((XCVR_BASE_0 + 32'h86)    << 2) , 8'h08 , 8'h00 , 8'h00   , CMD_WR    };
            8'd158:     data = {12'h00   , ((XCVR_BASE_0 + 32'h87)    << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd159:     data = {12'h00   , ((XCVR_BASE_0 + 32'h90)    << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };
            8'd160:     data = {12'h00   , ((XCVR_BASE_0 + 32'h8A)    << 2) , 8'h00 , 8'h80 , 8'h80   , CMD_POLL  };
            8'd161:     data = {12'h00   , ((XCVR_BASE_0 + 32'h8B)    << 2) , 8'h00 , 8'h01 , 8'h00   , CMD_POLL  };
            //data0=eth_reconfig_read(intf,ch,0x88)   
            //data1=eth_reconfig_read(intf,ch,0x89)   
            8'd162:     data = {12'h00   , ((XCVR_BASE_1 + 32'h8A)    << 2) , 8'h80 , 8'h00 , 8'h00   , CMD_WR    };
            8'd163:     data = {12'h00   , ((XCVR_BASE_1 + 32'h84)    << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd164:     data = {12'h00   , ((XCVR_BASE_1 + 32'h85)    << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };
            8'd165:     data = {12'h00   , ((XCVR_BASE_1 + 32'h86)    << 2) , 8'h08 , 8'h00 , 8'h00   , CMD_WR    };
            8'd166:     data = {12'h00   , ((XCVR_BASE_1 + 32'h87)    << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd167:     data = {12'h00   , ((XCVR_BASE_1 + 32'h90)    << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };
            8'd168:     data = {12'h00   , ((XCVR_BASE_1 + 32'h8A)    << 2) , 8'h00 , 8'h80 , 8'h80   , CMD_POLL  };
            8'd169:     data = {12'h00   , ((XCVR_BASE_1 + 32'h8B)    << 2) , 8'h00 , 8'h01 , 8'h00   , CMD_POLL  };
            //data0=eth_reconfig_read(intf,ch,0x88)   
            //data1=eth_reconfig_read(intf,ch,0x89)   
            8'd170:     data = {12'h00   , ((XCVR_BASE_2 + 32'h8A)    << 2) , 8'h80 , 8'h00 , 8'h00   , CMD_WR    };
            8'd171:     data = {12'h00   , ((XCVR_BASE_2 + 32'h84)    << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd172:     data = {12'h00   , ((XCVR_BASE_2 + 32'h85)    << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };
            8'd173:     data = {12'h00   , ((XCVR_BASE_2 + 32'h86)    << 2) , 8'h08 , 8'h00 , 8'h00   , CMD_WR    };
            8'd174:     data = {12'h00   , ((XCVR_BASE_2 + 32'h87)    << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd175:     data = {12'h00   , ((XCVR_BASE_2 + 32'h90)    << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };
            8'd176:     data = {12'h00   , ((XCVR_BASE_2 + 32'h8A)    << 2) , 8'h00 , 8'h80 , 8'h80   , CMD_POLL  };
            8'd177:     data = {12'h00   , ((XCVR_BASE_2 + 32'h8B)    << 2) , 8'h00 , 8'h01 , 8'h00   , CMD_POLL  };
            //data0=eth_reconfig_read(intf,ch,0x88)   
            //data1=eth_reconfig_read(intf,ch,0x89)   
            8'd178:     data = {12'h00   , ((XCVR_BASE_3 + 32'h8A)    << 2) , 8'h80 , 8'h00 , 8'h00   , CMD_WR    };
            8'd179:     data = {12'h00   , ((XCVR_BASE_3 + 32'h84)    << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd180:     data = {12'h00   , ((XCVR_BASE_3 + 32'h85)    << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };
            8'd181:     data = {12'h00   , ((XCVR_BASE_3 + 32'h86)    << 2) , 8'h08 , 8'h00 , 8'h00   , CMD_WR    };
            8'd182:     data = {12'h00   , ((XCVR_BASE_3 + 32'h87)    << 2) , 8'h00 , 8'h00 , 8'h00   , CMD_WR    };
            8'd183:     data = {12'h00   , ((XCVR_BASE_3 + 32'h90)    << 2) , 8'h01 , 8'h00 , 8'h00   , CMD_WR    };
            8'd184:     data = {12'h00   , ((XCVR_BASE_3 + 32'h8A)    << 2) , 8'h00 , 8'h80 , 8'h80   , CMD_POLL  };
            8'd185:     data = {12'h00   , ((XCVR_BASE_3 + 32'h8B)    << 2) , 8'h00 , 8'h01 , 8'h00   , CMD_POLL  };
            //data0=eth_reconfig_read(intf,ch,0x88)
            //data1=eth_reconfig_read(intf,ch,0x89)
            //END: Disable serial loopback mode for all channels
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////
            
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //START: Check RSFEC Status
            8'd186:     data = {12'h00   , ((RSFEC_BASE + 32'h8B)     << 2) , 8'h00 , 8'h01 , 8'h00   , CMD_POLL  };
            //END: Check RSFEC Status
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////
                                     //VCS coverage off
`endif
            default:    data = '0;
                                     //VCS coverage on
        endcase
    end
    
endmodule
