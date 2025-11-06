# Peripheral Interface

The Holoscan Sensor Bridge IP supports several software defined peripheral interfaces;
SPI, I2C, and GPIO. Each of the protocols instantiate a general-purpose core to handle
all protocol specific requirements and allows for software to set up transactions which
are then executed by the Holoscan Sensor Bridge IP.

Multiple endpoints can be supported using one single core instance. This reduces the
overall resource utilization by not requiring multiple instances of the given core.
However, this creates the requirement that only one transaction can be set up and
executed at a time per protocol, since all endpoints share the same physical registers.

The peripheral cores and its transactions are set up through ECB packets. This system
allows software to load a series of data bytes into a buffer, which are then written out
to the peripheral. Any data coming from the peripheral will be stored into the same
buffer which can then be read through ECB reads.

The peripheral interfaces are clocked using the "i_apb_clk" and reset with "o_apb_rst".
A prescaler register can be configured to the desired peripheral interface frequency.

## SPI

The SPI core has a 4-bit bi-directional signal data signal (SDIO), a chip select (CS_N)
and clock signal (SCK) for each endpoint connected. The bi-directional data signal is
broken up into a 4-bit input and 4-bit output signal, with an output enable (OEN) signal
for tri-stating. This SPI core supports single SPI mode using SDIO[0] as an output
(MOSI) and SDIO[1] as an input (MISO). Dual and Quad SPI are supported using 2 and 4
bits of the SDIO signal respectively.

The top-level input should be synchronized to "i_apb_clk" domain. The following code
snippets demonstrate how to connect the Holoscan Sensor Bridge SPI interface to top
level ports, where “i” is the endpoint index. SPI_SCK, SPI_CSN, SPI_MOSI, SPI_MISO, and
SPI_SDIO are top level ports.

![simple_spi](simple_spi.png)

Figure 1. Simple SPI Connectivity

```none
assign SPI_SCK = o_spi_sck [i];
assign SPI_CSN = o_spi_csn [i];
assign SPI_MOSI = o_spi_sdio [i];
assign i_spi_sdio[i] = {2'b0, SPI_MISO, 1'b0};
```

![dual_quad_spi](dual_quad_spi.png)

Figure 2. Dual/Quad SPI Connectivity

Dual SPI Assignment

```none
assign SPI_SCK = o_spi_sck [i];
assign SPI_CSN = o_spi_csn [i];
assign SPI_SDIO [1:0] = o_spi_oen [i] ? o_spi_sdio[i][1:0] : 2'bz;
assign i_spi_sdio [i] = {2’b0,SPI_SDIO};
```

Quad SPI Assignment

```none
assign SPI_SCK = o_spi_sck [i];
assign SPI_CSN = o_spi_csn [i];
assign SPI_SDIO = o_spi_oen [i] ? o_spi_sdio[i] : 4'bz;
assign i_spi_sdio [i] = SPI_SDIO;
```

The base address for SPI CTRL FSM is 0x0300_0000.

## I2C

The I2C core has two bidirectional signals, sda and scl. The core implements this
bidirectional signal using an input signal, and an output enable signal. When the output
enable signal is high, the corresponding signal should be left floating, only when
output enable is low should the output be pulled low. This behavior is consistent with
the I2C protocol.

I2C clock stretching is supported by the I2C core.

The top-level input should be synchronized to "i_apb_clk" clock domain, and glitch
filtering should also be added according to the I2C protocol.

The following code snippet demonstrates how to connect the Holoscan Sensor Bridge I2C
interface to top level ports, where “i” is the endpoint index. I2C_SCL and I2C_SDA are
top level ports.

![i2c](i2c.png)

Figure 3. I2C Connectivity

I2C Assignment

```none
assign i_i2c_scl[i] = o_i2c_scl_en[i] ? I2C_SCL : 1'b0;
assign i_i2c_sda[i] = o_i2c_sda_en[i] ? I2C_SDA : 1'b0;
assign I2C_SCL = o_i2c_scl_en[i] ? 1'bz : 1'b0;
assign I2C_SDA = o_i2c_sda_en[i] ? 1'bz : 1'b0;
```

The base address for I2C CTRL FSM is 0x0300_0200.

**\*Note: I2C is only verified and tested at 400kHz speed mode.**

## GPIO

The Holoscan Sensor Bridge IP supports General Purpose I/O (GPIO) signals for status and
control functionality. The GPIO signals can be set to be an input or an output signals.
GPIO signals are set as inputs by default.

The GPIO signals can connect to internal user logic or to the top level onto the board.
Examples for GPIO control are toggling on-board pins, internal straps to IP
configuration, LED control, and more.

The reset value of the GPIO control signals can be defined using the "GPIO_RESET_VALUE"
parameter.

Examples for GPIO status are sensor status signals, on-board and internal calibration
done signal, software reset count, and more.

GPIO input signals will cross domain clock (CDC) into "i_apb_clk" cloGck domain.

GPIO output signals will cross domain clock (CDC) into "i_hif_clk" clock domain.

![1709593052977](images/peripheral_interface/1709593052977.png)

Figure 4. GPIO Connectivity
