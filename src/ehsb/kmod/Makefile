obj-m += hololink_pcie_drv.o

# Driver version reported by MODULE_VERSION(). CMake passes the repository VERSION as
# HL_PCIE_DRV_VERSION=X.Y.Z on the make command line (it reaches this sub-make); a bare
# `make -C src/ehsb/kmod` leaves it unset and the driver's built-in fallback applies.
ifneq ($(HL_PCIE_DRV_VERSION),)
ccflags-y += -DHL_PCIE_DRV_VERSION=\"$(HL_PCIE_DRV_VERSION)\"
endif

KDIR ?= /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)

# host1x syncpoint support (HL_PCIE_IOC_GET_SYNCPOINT / HL_PCIE_IOC_ARM) links against
# host1x_syncpt_* symbols exported by the host1x module of the target kernel. With
# CONFIG_MODVERSIONS the .ko records a CRC per imported symbol, and insmod refuses the module
# ("Invalid parameters"; dmesg "disagrees about version of symbol host1x_syncpt_*") when
# that CRC is not the one the loaded host1x exports. Which file holds the right CRCs depends
# on who provides host1x for the target kernel:
#   - in-tree host1x (IGX Orin, 7.0.0-nvidia: /lib/modules/<release>/kernel/drivers/gpu/
#     host1x/): the base kernel's $(KDIR)/Module.symvers, which kbuild uses by itself;
#   - out-of-tree host1x (IGX Thor R38 / Jetson JetPack: host1x(O) under updates/ or
#     extra/): that tree's own Module.symvers, passed through KBUILD_EXTRA_SYMBOLS. The
#     base kernel's file may still list in-tree host1x CRCs there -- that is the Thor
#     mismatch -- so the out-of-tree file must win.
# The search below picks the out-of-tree file when one is installed for the target release:
#   /usr/src/nvidia/nvidia-public/Module.symvers                     JetPack 6 IGX
#   /usr/src/nvidia/nvidia-oot/Module.symvers                        JetPack 6 Jetson
#   /usr/src/linux-headers-tegra-oot-*-<release>/nvidia-oot/Module.symvers
#                                                                    IGX Thor (R38, Ubuntu kernel)
# <release> is taken from KDIR (/lib/modules/<release>/build), so a -DKERNEL_DIR build for
# another kernel looks next to that kernel's headers; NVIDIA_SYMVERS_ROOT relocates the
# search for a sysroot; NVIDIA_PUBLIC_SYMVERS=<file> overrides it. It then warns -- rather
# than building a module that cannot load -- when the target's host1x is out-of-tree
# (modinfo -k <release> -n host1x not under kernel/) and no out-of-tree file was found, or
# when no CRC source has the symbols at all.
KERNEL_RELEASE := $(notdir $(patsubst %/build,%,$(KDIR)))
NVIDIA_SYMVERS_ROOT ?= /usr/src
NVIDIA_SYMVERS_CANDIDATES := \
	$(NVIDIA_SYMVERS_ROOT)/nvidia/nvidia-public/Module.symvers \
	$(NVIDIA_SYMVERS_ROOT)/nvidia/nvidia-oot/Module.symvers \
	$(NVIDIA_SYMVERS_ROOT)/linux-headers-tegra-oot-*-$(KERNEL_RELEASE)/nvidia-oot/Module.symvers
NVIDIA_PUBLIC_SYMVERS ?= $(firstword $(wildcard $(NVIDIA_SYMVERS_CANDIDATES)))
ifneq ($(wildcard $(NVIDIA_PUBLIC_SYMVERS)),)
KBUILD_EXTRA_SYMBOLS += $(NVIDIA_PUBLIC_SYMVERS)
else ifeq ($(KERNELRELEASE),)
# Top-level invocation only (kbuild re-reads this file with KERNELRELEASE set).
HOST1X_KO := $(shell modinfo -k $(KERNEL_RELEASE) -n host1x 2>/dev/null)
HOST1X_IN_TREE := $(if $(findstring /kernel/,$(HOST1X_KO)),yes,)
BASE_HAS_HOST1X := $(shell grep -qsw host1x_syncpt_alloc $(KDIR)/Module.symvers && echo yes)
ifneq ($(NVIDIA_PUBLIC_SYMVERS),)
$(warning hololink_pcie_drv: NVIDIA_PUBLIC_SYMVERS=$(NVIDIA_PUBLIC_SYMVERS) does not exist; \
	the host1x_syncpt_* CRCs will come from the base kernel's Module.symvers and insmod \
	may refuse the module ("disagrees about version of symbol").)
else ifeq ($(HOST1X_KO)$(BASE_HAS_HOST1X),)
$(warning hololink_pcie_drv: $(KERNEL_RELEASE) has no host1x module installed and \
	$(KDIR)/Module.symvers has no host1x_syncpt_* CRCs; the module will build but needs \
	host1x loaded, and may be refused, on the target.)
else ifeq ($(HOST1X_IN_TREE)$(if $(HOST1X_KO),,skip),)
$(warning hololink_pcie_drv: $(KERNEL_RELEASE)'s host1x is out-of-tree ($(HOST1X_KO)) but \
	no out-of-tree Module.symvers was found under $(NVIDIA_SYMVERS_ROOT) (tried \
	nvidia/nvidia-public, nvidia/nvidia-oot, \
	linux-headers-tegra-oot-*-$(KERNEL_RELEASE)/nvidia-oot); the host1x_syncpt_* CRCs \
	would come from the base kernel's in-tree host1x and insmod would refuse the module \
	("disagrees about version of symbol"). Install that kernel's out-of-tree headers or \
	set NVIDIA_PUBLIC_SYMVERS=<file>.)
endif
endif

all:
	$(MAKE) -C $(KDIR) M=$(PWD) KBUILD_EXTRA_SYMBOLS="$(KBUILD_EXTRA_SYMBOLS)" modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
