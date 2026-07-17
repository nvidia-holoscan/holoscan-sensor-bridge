# PYNQ RFSoC build

Build the RFSoC PYNQ project with Vivado.

## Steps

1. **Download `nv_hsb_ip`**\
   Place the `nv_hsb_ip` directory at the same level as `pynq` (sibling of `pynq`).\
   Layout should look like:

   ```
   <parent>/
   ├── nv_hsb_ip/
   └── pynq/
       ├── rfsoc-pynq/
       │   └── build/
       └── README.md
   ```

1. **Build with Vivado**\
   Run the build from within the `build` folder. You can use the provided script
   directly after updating the Vivado path in `build.sh` for your environment:

   ```bash
   cd pynq/rfsoc-pynq/build
   ./build.sh
   ```

   The script creates `project_folder`, runs Vivado in batch mode with `build.tcl`, then
   returns to the build directory.
