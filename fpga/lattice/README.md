# Lattice build

## Steps

1. Update Installation and License path Update
   "cpnx100-eth-sensor-bridge/lattice_env.sh": **RADIANT_PATH** to the Lattice Radiant
   Installed Path **LM_PATH** to the license server path, if applicable.

1. Known Radiant Issues For Radiant versions 2025.2 and older, there is a known issue
   with the DDR IP generation. In "${RADIANT_PATH}/ip/lifcl/ddr/metadata.xml" remove
   this line:

   ```none
   drc                = "ext_check_gearing_value_valid(GEARING, CLK_FREQ, INTERFACE_TYPE)"
   ```

1. Build Flow

   ```bash
   cd cpnx100-eth-sensor-bridge/cpnx100/build
   ./build.sh
   ```

   The build flow will generate a project folder with today's date under the "build"
   directory. If the build is successful, it will copy the generated bitfile to
   "cpnx100-eth-sensor-bridge/cpnx100/bitfile"

   The build flow for "clnx17" is the same. For Lattice CPNX100-ETH-SENSOR-BRIDGE, both
   the cpnx100 and clnx17 bitfiles are needed.
