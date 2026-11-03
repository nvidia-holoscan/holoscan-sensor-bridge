# hololink.emulation.sensors (build-staged package)

This directory holds no sources of its own. `python/setup.py` lists
`hololink/emulation/sensors` in `packages=`, and setuptools insists that every listed
package directory exists in the source tree before the CMake build runs, so the directory
must be present in git. The package's actual content comes from the emulator tree:

- `src/hololink/emulation/python/hololink/emulation/sensors/__init__.py`, staged here by
  `python/hololink/emulation/CMakeLists.txt`, re-exports the `sensors` submodule of the
  `_emulation` extension (built from
  `src/hololink/emulation/python/hololink/emulation/_emulation.cpp`).

Keep the emulator's Python sources in one place under `src/hololink/emulation/python`, so
the Emulator-only build (`cmake -S src/hololink/emulation`) ships the identical package.
