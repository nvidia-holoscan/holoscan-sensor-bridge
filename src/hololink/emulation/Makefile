SHELL = /bin/bash
CXX = g++
CXX_FLAGS_ = -Wall -Wno-unused-variable -g3 $(CUDA_CFLAGS) $(ZLIB_CFLAGS) $(CXX_FLAGS)
IFLAGS_ = -I../.. -I.
LFLAGS_ =  $(CUDA_LFLAGS) $(ZLIB_LFLAGS)

# python environment parameters
PY_VENV = env
ENV_PYTHON = $(PY_VENV)/bin/python3
ENV_PIP = $(ENV_PYTHON) -m pip

#specify the folder for dlpack
DLPACK = dlpack

# specify CUDA package config
CUDA_PKG_CONFIG = "cudart-$$(nvcc --version | grep release | awk '{print $$5}' | sed -e 's/.$$//')"

# specify CUDA flags
CUDA_CFLAGS = `pkg-config $(CUDA_PKG_CONFIG) --cflags`
CUDA_LFLAGS = `pkg-config $(CUDA_PKG_CONFIG) --libs`

# specify ZLIB flags
ZLIB_CFLAGS = `pkg-config zlib --cflags`
ZLIB_LFLAGS = `pkg-config zlib --libs`

HSB_EMULATOR_SRCS = utils.cpp linux_transmitter.cpp data_plane.cpp net.cpp hsb_emulator.cpp mem_register.cpp linux_data_plane.cpp hsb_config.cpp
HSB_EMULATOR_OBJS = utils.o linux_transmitter.o data_plane.o net.o hsb_emulator.o mem_register.o linux_data_plane.o hsb_config.o
HSB_EMULATOR_DEPS = $(DLPACK) hsb_emulator.hpp linux_transmitter.hpp data_plane.hpp net.hpp mem_register.hpp utils.hpp linux_data_plane.hpp base_transmitter.hpp hsb_config.hpp
EXAMPLES_OBJS = examples/serve_frames.o

all: examples libhololink_emulation.so libhololink_emulation.a $(PY_VENV)/bin/activate

FORCE:

$(DLPACK):
	mkdir -p $(DLPACK)
	curl -o $(DLPACK)/dlpack.h https://raw.githubusercontent.com/dmlc/dlpack/main/include/dlpack/dlpack.h


%.o: %.cpp $(DLPACK)
	$(CXX) $(CXX_FLAGS_) -fPIC $(IFLAGS_) -c -o $@ $<

libhololink_emulation.so: $(HSB_EMULATOR_DEPS) $(HSB_EMULATOR_OBJS)
	$(CXX) -shared -fPIC -o libhololink_emulation.so $(HSB_EMULATOR_OBJS) $(LFLAGS_)

libhololink_emulation.a: $(HSB_EMULATOR_DEPS) $(HSB_EMULATOR_OBJS)
	ar -rcs libhololink_emulation.a $(HSB_EMULATOR_OBJS)

examples/serve_frames: $(HSB_EMULATOR_DEPS) $(HSB_EMULATOR_OBJS) $(EXAMPLES_OBJS)
	$(CXX) $(CXX_FLAGS_) $(IFLAGS_) -o examples/serve_frames $(HSB_EMULATOR_OBJS) examples/serve_frames.o $(LFLAGS_)

examples: examples/serve_frames

# builds the python environment
$(PY_VENV)/bin/activate: _emulation.so requirements.txt setup.py
	if [ -z "$(NO_PYTHON)" ] ; then \
		python3 -m venv $(PY_VENV) ; \
		source $(PY_VENV)/bin/activate ; \
		$(ENV_PIP) install -r requirements.txt; \
		if [ -n "$(PY_GPU)" ]; then \
			$(ENV_PIP) install cupy-cuda12x ; \
		fi ; \
		$(ENV_PIP) install . ; \
		deactivate ; \
	fi

# builds the python extension module
_emulation.so: $(HSB_EMULATOR_DEPS) $(HSB_EMULATOR_OBJS) hololink/emulation/_emulation.cpp
	if [ -z "$(NO_PYTHON)" ] ; then \
		$(CXX) $(CXX_FLAGS_) $(IFLAGS_) -shared -fPIC $$(python3 -m pybind11 --includes) $(HSB_EMULATOR_OBJS) hololink/emulation/_emulation.cpp -o _emulation.so $(CUDA_LFLAGS) $(LFLAGS_) ; \
		cp _emulation.so hololink/emulation/_emulation$$(python3-config --extension-suffix) ; \
	fi

# clean only the python environment and python build portions
clean_py_venv:
	if [[ "`realpath .`" == "`realpath $(PY_VENV)`"* ]] ; then \
		rm -rf $(PY_VENV)/bin $(PY_VENV)/lib $(PY_VENV)/include $(PY_VENV)/lib64 $(PY_VENV)/pyvenv.cfg build *.egg-info ; \
	else \
		rm -rf $(PY_VENV) build *.egg-info ; \
	fi
	rm -rf hololink/__pycache__ hololink/emulation/__pycache__

# clean everything except for dlpack
reset: clean_py_venv
	rm -f *.o examples/*.o *.so hololink/emulation/*$$(python3-config --extension-suffix) *.a examples/serve_frames

# clean everything
clean: reset
	rm -rf dlpack
