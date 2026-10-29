# Third-party notices

This inventory documents copied source and preserved upstream license terms;
it does not grant rights to relicense or redistribute those components. These
components retain their own license scopes.

| Material | Included path | Upstream source | Applicable terms |
|---|---|---|---|
| Analog Devices meta-adi recipe | `configurator/meta-hsb/recipes-support/libiio/libiio_%.bbappend` | [meta-adi, revision b266043ccba46738ef32aab617d37d3fb605b0a7](https://github.com/analogdevicesinc/meta-adi/tree/b266043ccba46738ef32aab617d37d3fb605b0a7); recipe content matches this revision excluding the local attribution preamble | MIT; [copyright and license](third_party_licenses/meta-adi-MIT.txt), retained from the matching revision. |
| Microchip Ethernet sensor bridge RTL | `fpga/microchip/mfp200-eth-sensor-bridge/rtl/top/PF_ESB_top.v`; `fpga/microchip/mfp200-eth-sensor-bridge/rtl/MAC_Hololink/HOLOLINK_def.svh`; `fpga/microchip/mfp200-eth-sensor-bridge/rtl/MAC_Hololink/HOLOLINK_top_wrapper.v` | [Microchip AN5522](https://www.microchip.com/en-us/application-notes/an5522) | Microchip device-restricted RTL terms; [copyright and preserved notice](third_party_licenses/Microchip-RTL.txt). Use is subject to the accompanying agreement and restricted to Microchip products. Applicability to the definitions and wrapper files remains subject to review; other design files require provenance and terms verification. |
| STM32CubeMX linker script | `src/hololink/emulation/src/STM32/STM32F767ZI/STM32F767XX_FLASH.ld` | STM32CubeMX / STMicroelectronics | BSD-3-Clause; [copyright and preserved notice](third_party_licenses/STM32CubeMX-BSD.txt). |

Build-time dependencies include DLPack, fmt, GoogleTest, pybind11, STM32 firmware
packages, and zlib. They are fetched or supplied externally; distributions
containing their code must retain their applicable upstream licenses and notices.
