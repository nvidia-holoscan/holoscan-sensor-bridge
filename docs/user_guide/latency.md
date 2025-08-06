# Holoscan sensor bridge data latency

### Holoscan SDK metadata and HSB

HSB devices send a block of metadata following each received data frame. That metadata
includes:

- `frame_number` counts of the number of data frames from this sensor sent by the FPGA
- `timestamp_s` and `timestamp_ns` are the PTP timestamp when the first data for the
  current frame arrived at the FPGA.
- `metadata_s` and `metadata_ns` are the PTP timestamp recorded when the metadata packet
  is sent out-- which exactly follows the last byte in the received data frame.

When host PTP support is properly configured, this time is synchronized with the host
time to within one microsecond. HSDK operators can access this metadata using
[APIs provided by the Holoscan SDK](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_create_app.html#dynamic-application-metadata).
Notes:

- Timestamps are comparable to the clock values read from the
  `clock_gettime(CLOCK_REALTIME, &timespec)` API.
- Receiver operators produce different sets of metadata based on the unique
  characteristics of that implementation. For example, data specific to RoceReceiverOp
  may not appear in the metadata presented by LinuxReceiverOperator. Applications may
  choose to access metadata using `metadata.get("parameter_name", 0)` to provide a
  useful default value when the specifically named metadata isn't set by the framework.
- Be sure and call the application `is_metadata_enabled(true)` method at initialization
  time; otherwise each operator will only see an empty metadata structure.

### Measuring sensor data latency

In `examples/imx274_latency.py`, you can see a pipeline that records additional
timestamps, and uses these timestamps to issue a latency report:

- `operator_s` and `operator_ns` are recorded by the operator following the network
  receiver operator. This is the time at which pipeline operators can actually access
  received sensor data.
- `completed_s` and `completed_ns` are recorded by the last operator in the example
  pipeline, after visualization is complete.

The receiver operator also records `received_s` and `received_ns` which are recorded at
the time the CPU wakes up with the end-of-frame interrupt. This occurs in a background
thread, independent of the pipeline. The times listed below are computed by combining
`(name)_s` and `(name)_ns` into a single floating point seconds value. Time values
displayed are all typical but will vary.

<img src="latency.svg" alt="HSB Latency" width="100%"/>

- `frame_end - frame_start` is the time that the sensor requires to transfer an entire
  frame of data into the FPGA. For the IMX274 in 4k RAW10 mode, this is typically
  15.8ms.
- `received - frame_end` shows how long it takes for the CPU to wake up in the
  background thread due to end-of-frame indication. On an IGX with accelerated
  networking, this is typically 120us.
- `operator - received` is the time that it takes for the next pipeline operator to
  execute with the currently received data. On IGX, if the pipeline is idle, this time
  is typically around 1ms.
- `completed - operator` is the time required for the rest of the pipeline to complete.
  For IGX, executing the naive example ISP and visualizer, this time is typically about
  2.4ms.

The sample application therefore shows, for each video frame, almost 16ms data
acquisition time followed by almost 4ms of processing time, for a total of under 20ms
latency. In this application, frames are delivered at 60FPS, which means that each new
frame starts at a 16ms interval; reception of the next frame goes on in the background
while the current frame processing is underway.
