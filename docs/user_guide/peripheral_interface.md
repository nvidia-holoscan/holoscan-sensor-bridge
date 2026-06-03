# Peripheral Interface

The Holoscan Sensor Bridge IP supports several software defined peripheral interfaces;
SPI, I2C, UART, and GPIO. Each of the protocols instantiate a general-purpose core to
handle all protocol specific requirements and allows for software to set up transactions
which are then executed by the Holoscan Sensor Bridge IP.

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

## UART

The UART core provides a standard asynchronous serial interface with configurable frame
format and programmable baud rate through the UART divisor register. The core supports
common UART baud rates 9600, 38400, 57600, and 115200 (and other rates that can be
generated from the APB clock using the baud divisor setting).

The UART interface should be connected at the top level with one TX output and one RX
input per endpoint. Connect the HSB UART TX output to the external UART RX pin, and
connect the external UART TX pin into the HSB UART RX input. If hardware flow control is
required, also connect RTS/CTS between the HSB IP and the external interface. The UART
core already includes internal RX glitch filtering support, so additional external
glitch filter logic is not required.

UART TX and RX data is accessed through the APB interface.

The base address for UART CTRL FSM is 0x0300_0400.

## GPIO

The Holoscan Sensor Bridge IP supports General Purpose I/O (GPIO) signals for status and
control functionality.

The GPIO signals can connect to internal user logic or to the top level of the board.
Examples for GPIO control are toggling on-board pins, internal straps to IP
configuration, LED control, and more.

Examples for GPIO status are sensor status signals, on-board and internal calibration
done signal, software reset count, and more.

The reset value of the GPIO control signals can be defined using the `GPIO_RESET_VALUE`
parameter.

The following code snippet demonstrates how to connect the Holoscan Sensor Bridge GPIO
interface to top level as INOUT ports, where “i” is the endpoint index.

GPIO Assignment

```none
assign i_gpio[i] = GPIO[i];
assign GPIO[i] = o_gpio_dir[i] ? 1'bz : o_gpio[i];
```

GPIO input signals will cross domain clock (CDC) into "i_apb_clk" clock domain. GPIO
output signals will cross domain clock (CDC) into "i_hif_clk" clock domain.

![1709593052977](images/peripheral_interface/1709593052977.png)

Figure 4. GPIO Connectivity

In the Lattice CPNX100-ETH-SENSOR-BRIDGE devkit, the GPIOs from HSB IP are connected to
the Test Points and the Jetson Connector. The HSB IP GPIO mapping in Lattice devkit is
listed below.

Lattice CPNX100-ETH-SENSOR-BRIDGE GPIO Pins

| **HSB IP** | **Lattice HSB** |
| ---------- | --------------- |
| GPIO[0]    | J20, pin 3      |
| GPIO[1]    | J20, pin 5      |
| GPIO[2]    | J20, pin 7      |
| GPIO[3]    | J20, pin 9      |
| GPIO[4]    | J20, pin 11     |
| GPIO[5]    | J20, pin 13     |
| GPIO[6]    | J20, pin 15     |
| GPIO[7]    | J20, pin 17     |
| GPIO[8]    | J20, pin 4      |
| GPIO[9]    | J20, pin 6      |
| GPIO[10]   | J20, pin 8      |
| GPIO[11]   | J20, pin 10     |
| GPIO[12]   | J20, pin 12     |
| GPIO[13]   | J20, pin 14     |
| GPIO[14]   | J20, pin 16     |
| GPIO[15]   | J20, pin 18     |
| GPIO[16]   | J9, pin 76      |
| GPIO[17]   | J9, pin 78      |
| GPIO[18]   | J9, pin 84      |
| GPIO[19]   | J9, pin 85      |
| GPIO[20]   | J9, pin 90      |
| GPIO[21]   | J9, pin 92      |
| GPIO[22]   | J9, pin 96      |
| GPIO[23]   | J9, pin 97      |
| GPIO[24]   | J9, pin 98      |
| GPIO[25]   | J9, pin 119     |
| GPIO[26]   | J9, pin 86      |
| GPIO[27]   | J9, pin 103     |
| GPIO[28]   | J9, pin 104     |
| GPIO[29]   | J9, pin 106     |
| GPIO[30]   | J9, pin 117     |
