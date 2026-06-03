# `eeprom_utility.py`

Programs the HSB I2C EEPROM over **ECB/UDP** (port **8192**). Run from this directory:

```bash
python3 eeprom_utility.py
```

Optional arguments (defaults shown):

```bash
python3 eeprom_utility.py \
  --host 192.168.0.101 \
  --dest 192.168.0.2 \
  --eeprom-reg-addr-bits 8 \
  --apb_clk_freq 19531250
```

- `--host` — local bind address for the control socket.
- `--dest` — board IP.
- `--eeprom-reg-addr-bits` — `8` or `16` (match your FPGA `EEPROM_REG_ADDR_BITS`).
- `--apb_clk_freq` — APB frequency in Hz (used for I2C clock divider; default =
  156.25e6/8).

Then use the interactive commands; see `--help` and the on-screen menu.

### eeprom utility commands

| Command | Arguments       | What it does                                                                                                                                                                                |
| ------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `q`     | —               | Quit.                                                                                                                                                                                       |
| `d`     | —               | Dump 256 bytes (page reads), show CRC result, optional CRC fix; **loads** the in-memory copy for later `wbc`.                                                                               |
| `e`     | —               | Erase to `0xFF` and set CRC (with confirmation).                                                                                                                                            |
| `r`     | `<file>`        | Read full EEPROM to `<file>` (`@addr  byte` lines).                                                                                                                                         |
| `w`     | `<file>`        | Program EEPROM from `<file>`, 8 bytes per page, fix CRC.                                                                                                                                    |
| `wrc`   | `<file>`        | Like `w`, then re-read and compare.                                                                                                                                                         |
| `rb`    | `<addr>`        | Read one byte. `<addr>` = hex EEPROM address (two digits in 8-bit mode, e.g. `44` = byte 68).                                                                                               |
| `wb`    | `<addr> <byte>` | Write one byte **without** recalculating CRC (with confirmation).                                                                                                                           |
| `wbc`   | `<addr> <byte>` | Write one byte, recompute data CRC (bytes 0x00…0xFE) and update byte **0xFF**. **Requires** a prior `d` or `r` / `w` so the in-memory image matches the part. If not loaded, run `d` first. |

### Programming the Ethernet MAC and Board Serial Number in EEPROM (8-bit addressing)

| EEPROM byte (decimal) | MAC bits        |
| --------------------: | --------------- |
|                    68 | MAC[47:40]      |
|                    69 | MAC[39:32]      |
|                    70 | MAC[31:24]      |
|                    71 | MAC[23:16]      |
|                    72 | MAC[15:8]       |
|                    73 | MAC[7:0]        |
|                    74 | Board SN[55:48] |
|                    75 | Board SN[47:40] |
|                    76 | Board SN[39:32] |
|                    77 | Board SN[31:24] |
|                    78 | Board SN[23:16] |
|                    79 | Board SN[15:8]  |
|                    80 | Board SN[7:0]   |

After loading the current image (e.g. `d`):

```text
wbc 44 12
wbc 45 34
wbc 46 56
wbc 47 78
wbc 48 9A
wbc 49 BC
```

(Example MAC `12:34:56:78:9A:BC`; use your own **two** hex digit values per line.)

For `--eeprom-reg-addr-bits 8`, use **two**-digit hex addresses: `44`…`49` (EEPROM bytes
68…73). For **16**-bit address mode, use **four** digits, e.g. `0044`…`0049`. above
example using `--eprom-reg-addr-bits 16` is:

```text
wbc 0044 12
wbc 0045 34
wbc 0046 56
wbc 0047 78
wbc 0048 9A
wbc 0049 BC
```
