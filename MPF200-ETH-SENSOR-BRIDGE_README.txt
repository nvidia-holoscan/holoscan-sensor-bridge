
Microchip PolarFire MPF200-ETH-SENSOR-BRIDGE kit purchasing information and and technical information is available at https://www.microchip.com/en-us/development-tool/MPF200-ETH-SENSOR-BRIDGE

Prgramming MPF200-ETH-SENSOR-BRIDGE FPGA design using Holoscan sensor bridge software

1. Connect the ethernet cable to J6 connector on  MPF200-ETH-SENSOR-BRIDGE to AGX/IGX ethernet port. Check if Nvidia AGX/IGX is connected to Holoscan board by doing a ping test by issuing following command.
	a. ping 192.168.0.2
2. In new terminal. Issue the following command to change directory to hololink-sensor-bridge folder
	a.	cd <PATH/TO/hololink-sensor-bridge>
3. Run the below commands to tranfer .spi(job/RTL design) file to on board SPI flash of MPF200-ETH-SENSOR-BRIDGE board.
	a. 	xhost + <enter>
	b.	sh docker/demo.sh <enter>
			b.1:The above command runs holoscan-sensor-bridge docker container
	c.	polarfire_esb --flash
4. Step 3.c downloads FPGA design file from internet and takes around 50 minutes to transfer the design file to on-board flash.
5. To program the FPGA with the .spi that is in the on-board SPI flash run the below command within the holoscan-sensor-bridge docker contianer
	a.	polarfire_esb --program
6. Step 5.a programs FPGA with design file in SPI flash. It takes around 1 minute to program the FPGA



Running imx477 applications

1. Run the docker container by changing directory to holoscan-sensor-bridge folder
	a. cd <PATH/TO/hololink-sensor-bridge>
2. Connect Ethernet cable from AGX/IGX to  J6 connector on  MPF200-ETH-SENSOR-BRIDGE and connect the camera to J14 MIPI connector.  Run the below commands to run linux_imx477 example for camera 1
	a. xhost +
	b. sh docker/demo.sh <enter>
		b.1: It runs holoscan-sensor-bridge docker container
	c. python examples/linux_imx477_player.py
		c.1 The above command display video from camera 1


3.  Connect Ethernet cable from AGX/IGX to  J3 connector on  MPF200-ETH-SENSOR-BRIDGE and connect the camera to J17 MIPI connector. Run the below commands to run linux_imx477 example for camera 2.
	a. xhost +
	b. sh docker/demo.sh <enter>
		b.1: It runs holoscan-sensor-bridge docker container
	c. python examples/linux_imx477_player.py --cam 2
		c.1 The above command display video from camera 2


note1 : Camera that is connected to J14 is refered as "cam 1" and camera that is connected to J17 is refered as "cam 2" on  MPF200-ETH-SENSOR-BRIDGE board
note2 : To work with "cam 1" connect ethernet cable to J6 connector. To work with "cam 2" connect ethernet cable to J3 connector on MPF200-ETH-SENSOR-BRIDGE board.


Imx477 sensor driver

1. The driver supports streaming video from two cameras connected one at a time. Either "cam 1" or "cam 2" works at a time
2. Imx477 sensor driver currently supports RGB8 format, 4K resolution, 60 FPS. The constructor of Imx477 camera sensor is invoked as shown below
	camera = hololink_module.sensors.imx477.imx477(hololink_channel, args.cam )
    a. cam arugument specifies which camera to select. By default cam 1 is selected
