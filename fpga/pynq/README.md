# PYNQ RFSoC build

Build the RFSoC PYNQ project with Vivado.

## Steps

1. **Set the Vivado installation path**\
   Open `rfsoc-pynq/build/build.sh` and set `VIVADO_PATH` to the Xilinx installation
   root for your environment. The script expects Vivado at
   `${VIVADO_PATH}/Vivado/2024.1/bin/vivado`.

   For example, if Vivado is installed at `/tools/Xilinx/Vivado/2024.1/bin/vivado`, set
   `VIVADO_PATH="/tools/Xilinx"`.

1. **Run the build script**\
   From this `pynq` directory, run `cd rfsoc-pynq/build` and then `sh build.sh`.

   The script creates `project_folder`, runs Vivado in batch mode with `build.tcl`, then
   returns to the build directory.
