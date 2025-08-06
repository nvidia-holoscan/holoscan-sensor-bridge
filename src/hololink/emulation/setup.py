import os
import sysconfig
from distutils.core import setup


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
    version="2.2.0",
    description="Holoscan Sensor Bridge Emulation",
    url="https://github.com/nvidia-holoscan/holoscan-sensor-bridge",
    packages=["hololink", "hololink.emulation"],
    package_dir={"hololink": "hololink"},
    data_files=[install_data_files("hololink/emulation")],
)
