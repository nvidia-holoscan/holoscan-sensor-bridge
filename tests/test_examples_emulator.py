import os
import signal
import subprocess
import time

import pytest
from camera_example_conf import (
    camera_properties,
    get_camera_count_from_json_config,
    get_ib_device_from_ifname,
    handle_failed_subprocess,
    has_ib_interface,
    has_sipl_interface,
    vb1940_emulator_examples,
    vb1940_loopback_cases,
    vb1940_loopback_cases_extended,
    vb1940_loopback_cases_roce,
    vb1940_loopback_cases_sipl,
)

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_namespace_pid(namespace, command):
    try:
        pid_cmd = [
            os.path.join(script_dir, "scripts", "nspid.sh"),
            namespace,
            command[0],  # only take the running binary name as flags throw off the grep
        ]
        if len(command) > 1 and not command[1].startswith("-"):
            pid_cmd[-1] = pid_cmd[-1] + " " + command[1]
        nspid_process = subprocess.run(
            pid_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return int(nspid_process.stdout.decode("utf-8").strip())
    except Exception as e:
        print(f"Error getting namespace PID: {e}")
        return None


def emulator_loopback(host_command, emulator_command, timeout_s, isolated_interface):

    time_start = time.time()
    time_end = time_start + timeout_s
    # start the emulator command and define the appropriate kill function
    emulator_process = None
    if isolated_interface != "lo":
        # if the emulator is running in an isolated namespace, the process that needs to be killed is within the namespace
        # killing the process from the scripts/nsexec.sh basically makes the emulator process an orphan that root/sudo needs to clean up
        emulator_process = subprocess.Popen(
            [os.path.join(script_dir, "scripts", "nsexec.sh"), isolated_interface]
            + emulator_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        nspid = get_namespace_pid(isolated_interface, emulator_command)
        while nspid is None and time.time() < time_end:
            time.sleep(1)
            nspid = get_namespace_pid(isolated_interface, emulator_command)
        if nspid is None:
            raise RuntimeError(
                f"Timeout reached after {timeout_s} seconds while waiting for emulator process to start. cmd: {emulator_command}. May require manual cleanup of namespace: {isolated_interface}"
            )

        # why is this here? because ci/lint.sh says we cannot assign a lambda to a variable
        def nsexec_kill():
            os.kill(nspid, signal.SIGKILL)

        emulator_kill = nsexec_kill
    else:
        emulator_process = subprocess.Popen(
            emulator_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        emulator_kill = emulator_process.kill

    # start the host command
    host_process = subprocess.Popen(
        host_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    host_kill = host_process.kill

    # restart timeout for the main loop
    time_end = time.time() + timeout_s
    while time.time() < time_end:
        # success is host process exiting normally. If it exits with error code, kill the emulator and raise an error for a failed test
        if (
            host_process.poll() is not None
        ):  # only success is if host process exits normally
            emulator_kill()  # regardless of host return state, kill the emulator
            if host_process.returncode != 0:
                handle_failed_subprocess(host_process, "Host", host_kill)
                handle_failed_subprocess(emulator_process, "Emulator")
                print(f"test failed in {time.time() - time_start} seconds", end=" ")
                raise RuntimeError(
                    f"Host subprocess exited early with code: {host_process.returncode}"
                )
            print(
                f"test completed successfully in {time.time() - time_start} seconds",
                end=" ",
            )
            return

        # if emulator process exits early, kill the host process and raise an error
        if emulator_process.poll() is not None:
            handle_failed_subprocess(host_process, "Host", host_kill)
            handle_failed_subprocess(emulator_process, "Emulator")
            print(f"test failed in {time.time() - time_start} seconds", end=" ")
            raise RuntimeError(
                f"Emulator unexpectedly exited early with code: {emulator_process.returncode}"
            )
        time.sleep(1)
    # timeout reached, kill both processes and raise an error
    handle_failed_subprocess(host_process, "Host", host_kill)
    handle_failed_subprocess(emulator_process, "Emulator", emulator_kill)
    print(f"test failed in {time.time() - time_start} seconds", end=" ")
    raise RuntimeError(f"Host subprocess timed out after {timeout_s} seconds")


# tests vb1940 emulator configurations in loopback mode by
# - running a subprocess for both the host and emulator applications from their respective examples/ folders
# - sends tests frames for "frame_limit" number of frames at frame rate determined by the camera mode
# - successful test is the host application exits normally without error code and no early exit from emulator process
@pytest.mark.skip_unless_cuda_subprocesses
def test_emulator_vb1940_loopback(
    transport,
    camera_count,
    camera_mode,
    gpu,
    frame_limit,
    headless,
    hololink_address,
    hw_loopback,
    json_config,
):
    camera = camera_properties["vb1940"]
    if camera_count < 1 or camera_count > 2:
        raise ValueError(f"Invalid camera count: {camera_count}")
    if camera_mode < 0 or camera_mode > 2:
        raise ValueError(f"Invalid camera mode: {camera_mode}. must be 0, 1, or 2")
    frame_rate = 30  # for all vb1940 modes
    if frame_limit < 1:
        raise ValueError(f"Invalid frame limit: {frame_limit}. must be greater than 0")
    if frame_limit > 60:
        frame_limit = 60
    if not isinstance(gpu, bool):
        raise ValueError(f"Invalid GPU flag: {gpu}. must be a boolean")

    if hw_loopback is None:
        assert transport in ("linux", "coe")
        emu_interface = "lo"
        host_interface = "lo"
        # loopback can only have one valid IP address robust to transport type and cannot be accelerated. use 127.0.0.1 instead of whatever user applied
        hololink_address = "127.0.0.1"
        host_transport = transport
        emu_transport = transport
    else:
        emu_interface, host_interface = hw_loopback
        assert host_interface != emu_interface
        if transport == "sipl":
            emu_transport = "coe"
            assert host_interface.startswith("mgbe")
        elif transport == "roce":
            emu_transport = "linux"
        else:
            emu_transport = transport
        host_transport = transport

    host_command = [
        "python3",
        camera["examples"][host_transport][camera_count],
        "--frame-limit",
        str(frame_limit),
        "--camera-mode",
        str(camera_mode),
        "--hololink",
        hololink_address,
    ]

    if host_transport == "coe":
        host_command.extend(["--coe-interface", host_interface])
    elif host_transport == "roce":
        host_command.extend(["--ibv-name", get_ib_device_from_ifname(host_interface)])
    elif host_transport == "sipl":
        # remove cmaera-mode and hololink options that are unsupported
        host_command = host_command[:-4]
        if camera_count == 1:
            host_command.extend(["--json-config", json_config])
        elif camera_count == 2:
            host_command.extend(["--json-config", json_config])
        else:
            raise ValueError(
                f"Invalid camera count: {camera_count}. must be 1 or 2 for sipl transport"
            )

    if headless:
        host_command.append("--headless")
    emu_command = [
        "python3",
        vb1940_emulator_examples[emu_transport][camera_count],
        "--frame-rate",
        str(frame_rate),
    ]
    if gpu:
        emu_command.append("--gpu")
    emu_command.append(hololink_address)

    timeout_s = (
        2 * frame_limit / frame_rate + camera["configure_timeout"] * camera_count
    )

    print(f"host_command: {host_command}")
    print(f"emu_command: {emu_command}")
    emulator_loopback(host_command, emu_command, timeout_s, emu_interface)


# dynamically generate test cases for the 2 test functions in this file based on the command line options
def pytest_generate_tests(metafunc):
    if metafunc.function == test_emulator_vb1940_loopback:
        parameters = vb1940_loopback_cases
        if metafunc.config.getoption("--emulator"):
            parameters += vb1940_loopback_cases_extended
        if metafunc.config.getoption(
            "--hw-loopback"
        ) is not None and not metafunc.config.getoption("--unaccelerated-only"):
            if (
                has_sipl_interface()
                and metafunc.config.getoption("--json-config") is not None
            ):
                camera_count = get_camera_count_from_json_config(
                    metafunc.config.getoption("--json-config")
                )
                for test_case in vb1940_loopback_cases_sipl:
                    parameters.append(
                        (test_case[0], camera_count, test_case[2], test_case[3])
                    )
            if has_ib_interface():
                parameters += vb1940_loopback_cases_roce
        metafunc.parametrize("transport,camera_count,camera_mode,gpu", parameters)


if __name__ == "__main__":
    pytest.main()
