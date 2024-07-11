# Release Notes

## 1.1.0-EA, July 10, 2024

*Note that this is an early access release, so performance is not guaranteed and support
may be limited.*

### Dependencies

- IGX: [IGX-SW 1.0 Production Release](https://developer.nvidia.com/igx-downloads)
- AGX: Use [SDK Manager](https://developer.nvidia.com/sdk-manager) to set up JetPack 6.0
  release 2.
- Holoscan Sensor Bridge, 10G; FPGA v2405
- ISP libraries: Contact NVIDIA for ISP libraries in order to run
  `linux_hwisp_player.py`.

Be sure and follow the installation instructions included with the release. To generate
documentation, in the host system, run `sh docs/make_docs.sh`, then use your browser to
look at `docs/user_guide/_build/html/index.html`.

### Known Anomalies

- AGX ingress data packets dropped.

  When running e.g. `python3 examples/linux_imx274_player.py`, the network stack in AGX
  can go into a state where ingress data packets are dropped. In this case, the
  application console will display the message "Ingress frame timeout" and `ifconfig`
  will show a quickly increasing number of packets in the "RX dropped" status.
  `ping 192.168.0.2` usually works in this case. In some cases,
  `sudo ifconfig <interface> down; sudo ifconfig <interface> up` corrects the problem.
  Occurrences of this problem are much worse in 4K video mode; running in 1080p
  (`python3 examples/linux_imx274_player.py --camera-mode=1`) is much more reliable.
