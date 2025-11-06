import os
import sysconfig
from distutils.core import setup

STANDALONE_VERSION = "2.5"


def get_version():
    version = STANDALONE_VERSION
    # try to get the value from HSB TOT VERSION file
    try:
        with open("../../../VERSION", "r") as f:
            version = f.read().strip()
    except Exception:
        print(
            f"failed to get HSB TOT version. using standalone version {STANDALONE_VERSION}"
        )
    return version


def install_data_files(module_path, exts=None):
    if exts is None:
        exts = {".so"}
    files = []
    search_path = os.path.join(".", module_path)
    for file in os.listdir(search_path):
        if os.path.splitext(file)[1] in exts:
            files.append(os.path.join(search_path, file))
    target_path = os.path.join(sysconfig.get_paths()["purelib"], module_path)
    print(f"installing {files} to {target_path}")
    return (target_path, files)


setup(
    name="hololink",
    version=get_version(),
    description="Holoscan Sensor Bridge Emulation",
    url="https://github.com/nvidia-holoscan/holoscan-sensor-bridge",
    packages=["hololink", "hololink.emulation", "hololink.emulation.sensors"],
    package_dir={"hololink": "hololink"},
    data_files=[
        install_data_files("hololink/emulation"),
        install_data_files("hololink/emulation/sensors"),
    ],
)
