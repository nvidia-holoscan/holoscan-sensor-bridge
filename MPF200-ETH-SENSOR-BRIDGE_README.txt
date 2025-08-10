
Microchip PolarFire MPF200-ETH-SENSOR-BRIDGE kit purchasing information and and technical information is available at https://www.microchip.com/en-us/development-tool/MPF200-ETH-SENSOR-BRIDGE



Programming MPF200-ETH-SENSOR-BRIDGE FPGA design using Holoscan sensor bridge software
--------------------------------------------------------------------------------------

1. Connect the ethernet cable to J6 connector on  MPF200-ETH-SENSOR-BRIDGE to AGX/IGX ethernet port. Check if Nvidia AGX/IGX is connected to Holoscan board by doing a ping test by issuing following command.
	a. ping 192.168.0.2
2. In new terminal. Issue the following command to change directory to hololink-sensor-bridge folder
	a.	cd <PATH/TO/hololink-sensor-bridge>
3. Run the below commands to transfer .spi(job/RTL design) file to on board SPI flash of MPF200-ETH-SENSOR-BRIDGE board.
	a. 	xhost + <enter>
	b.	sh docker/demo.sh <enter>
			b.1:The above command runs holoscan-sensor-bridge docker container
	c.	polarfire_esb flash --fpga-bit-version 2506
			c.1 The above command flashes(Transfers to on board SPI flash) 2506 version bit file.
	d.	polarfire_esb --force flash --fpga-bit-version 2506
			d.1 The above command flashes(Transfers to on board SPI flash) 2506 version bit file.
			d.2 Note: Use command switch "--force" when FPGA is running older version of bit file like 2407 or 2412
4. Step 3.c(or 3.d) downloads FPGA design file from internet and takes around 50 minutes to transfer the design file to on-board flash.
5. To program the FPGA with the .spi that is in the on-board SPI flash run the below command within the holoscan-sensor-bridge docker container
	a.	polarfire_esb --program
 			a.1 programs FPGA with design file in SPI flash. It takes around 1 minute to program the FPGA
	b.	polarfire_esb --force --program
 			b.1 programs FPGA with design file in SPI flash. It takes around 1 minute to program the FPGA.
 			b.2 Note: Use command switch "--force" when FPGA is running older version of bit file like 2407
6. Use the new "--force" switch if (new/latest) holoscan ethernet sensor bridge software is not unable to detect ethernet packets. This situation arise if FPGA is running older bit file and NVIDIA
   AGX/IGX running newer holoscan sensor bridge software.

Running imx477 applications - single camera
--------------------------------------------

1. Run the docker container by changing directory to holoscan-sensor-bridge folder
	a. cd <PATH/TO/hololink-sensor-bridge>
2. Connect Ethernet cable from AGX/IGX to  J6 connector on  MPF200-ETH-SENSOR-BRIDGE and connect the camera to J14 MIPI connector.  Run the below commands to run linux_imx477 example for camera 0
	a. xhost +
	b. sh docker/demo.sh <enter>
		b.1: It runs holoscan-sensor-bridge docker container
	c. python examples/linux_imx477_player.py
		c.1 The above command display video from camera 0


3.  Connect Ethernet cable from AGX/IGX to  J3 connector on  MPF200-ETH-SENSOR-BRIDGE and connect the camera to J17 MIPI connector. Run the below commands to run linux_imx477 example for camera 1.
	a. xhost +
	b. sh docker/demo.sh <enter>
		b.1: It runs holoscan-sensor-bridge docker container
	c. python examples/linux_imx477_player.py --cam 1
		c.1 The above command display video from camera 1


note1 : Camera that is connected to J14 is referred as "cam 0" and camera that is connected to J17 is referred as "cam 1" on  MPF200-ETH-SENSOR-BRIDGE board
note2 : To work with "cam 0" connect ethernet cable to J6 connector. To work with "cam 1" connect ethernet cable to J3 connector on MPF200-ETH-SENSOR-BRIDGE board.



Running imx477 application - Tao People net
-------------------------------------------

1. Run the docker container by changing directory to holoscan-sensor-bridge folder
	a. cd <PATH/TO/hololink-sensor-bridge>
2. Follow the steps from below and the link mentioned in 2.c to setup environment for running tao people net
	a. xhost +
	b. sh docker/demo.sh <enter>
		b.1: It runs holoscan-sensor-bridge docker container
	c. https://docs.nvidia.com/holoscan/sensor-bridge/latest/examples.html#running-the-tao-peoplenet-example
3. Connect Ethernet cable from AGX to J6 connector on MPF200-ETH-SENSOR-BRIDGE. Connect the camera to J14 MIPI connector.  Run the below commands to run linux_tao_peoplenet_imx477 for camera 0 
	a. python examples/linux_tao_peoplenet_imx477.py
		1.1 The above command runs tao peoplenet model on video from camera 0



Running imx477 application - body pose estimation
--------------------------------------------------

1. Run the docker container by changing directory to holoscan-sensor-bridge folder
	a. cd <PATH/TO/hololink-sensor-bridge>
2. Follow the steps from below and the link mentioned in 2.c to setup environment for running tao people net
	a. xhost +
	b. sh docker/demo.sh <enter>
		b.1: It runs holoscan-sensor-bridge docker container
	c. https://docs.nvidia.com/holoscan/sensor-bridge/latest/examples.html#running-the-body-pose-example
3. Connect Ethernet cable from AGX to J6 connector on MPF200-ETH-SENSOR-BRIDGE. Connect the camera to J14 MIPI connector.  Run the below commands to run linux_body_pose_estimation_imx477 for camera 0 
	a. python examples/linux_body_pose_estimation_imx477.py
		1.1 The above command runs body pose estimation model on video from camera 0



Running imx477 application - stereo camera using AGX + connectX-6 dx
--------------------------------------------------------------------

1. Run the docker container by changing directory to holoscan-sensor-bridge folder
	a. cd <PATH/TO/hololink-sensor-bridge>
2. Connect Ethernet cables from AGX + connectX-6 dx to  J6 and J3 connectors on  MPF200-ETH-SENSOR-BRIDGE. Connect the cameras to J14 and J17 MIPI connectors.  Run the below commands to run stereo_imx477_player for camera 0 and camera 1 
	a. xhost +
	b. sh docker/demo.sh <enter>
		b.1: It runs holoscan-sensor-bridge docker container
	c. python examples/stereo_imx477_player.py
		c.1 The above command display video from both camera 0 and camera 1.
		
Running imx477 application - stereo camera using only AGX

1. Run the docker container by changing directory to holoscan-sensor-bridge folder
        a. cd <PATH/TO/hololink-sensor-bridge>
2. Connect Ethernet cables from AGX + connectX-6 dx to  J6 and J3 connectors on  MPF200-ETH-SENSOR-BRIDGE. Connect the cameras to J14 and J17 MIPI connectors.  Run the below commands to run linux_single_network_stereo_imx477_player for camera 0 and camera 1
        a. xhost +
        b. sh docker/demo.sh <enter>
                b.1: It runs holoscan-sensor-bridge docker container
        c. python examples/linux_single_network_stereo_imx477_player.py
                c.1 The above command display video from both camera 0 and camera 1 via single ethernet port.

Running imx477 application - latency measurement using AGX + connectX-6 dx

1. Run the docker container by changing directory to holoscan-sensor-bridge folder
        a. cd <PATH/TO/hololink-sensor-bridge>
2. Connect Ethernet cables from AGX + connectX-6 dx to  J6 and J3 connectors on  MPF200-ETH-SENSOR-BRIDGE. Connect the cameras to J14 and J17 MIPI connectors.  Run the below commands to run imx477_latency for camera 0 and camera 1
        a. xhost +
        b. sh docker/demo.sh <enter>
                b.1: It runs holoscan-sensor-bridge docker container
        c. python examples/imx477_latency.py
                c.1 The above command gives a complete latency report.

Imx477 sensor driver
--------------------
1. The driver supports streaming video from two cameras connected one at a time. Driver also supports stereo camera i.e two cameras simultaneously.
2. Imx477 sensor driver currently supports RGB8 format, 4K resolution, 60 FPS. The constructor of Imx477 camera sensor is invoked as shown below
	camera = hololink_module.sensors.imx477.imx477(hololink_channel, args.cam, args.resolution)
    a. cam arugument specifies which camera to select. By default cam 0 is selected
    b. resolution argument specifies either "4k" or "1080p" default is 4k, resolution can be selected by passing --resolution=1080p flag while running demos
         resolution for stereo camera using only AGX is fixed to 1080p
