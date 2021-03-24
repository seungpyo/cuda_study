#
# Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
#
# See COPYRIGHT for license information
#

# External dependencies
include common.mk
CXX ?= /usr/bin/g++
CC ?= /usr/bin/gcc

# Build/install location
# location where the build will be installed
NVSHMEM_PREFIX ?= /usr/local/nvshmem/
# build location
NVSHMEM_BUILDDIR ?= $(abspath build)
# MPI/SHMEM Support
# Whether to build with MPI support. If yes, MPI_HOME should be set
NVSHMEM_MPI_SUPPORT ?= 1
# Is the MPI being built against is OpenMPI?
NVSHMEM_MPI_IS_OMPI ?= 1
# MPI install location
MPI_HOME ?= /usr/local/ompi/
# Whether to build with SHMEM support
NVSHMEM_SHMEM_SUPPORT ?= 1
# SHMEM install location
SHMEM_HOME ?= $(MPI_HOME)
# GDRCopy install/headers location
GDRCOPY_HOME ?= /usr/local/gdrdrv
# NCCL install/headers location
NCCL_HOME ?= /usr/local/nccl
NVSHMEM_USE_NCCL ?= 0
# Whether to build with MPI support. If yes, MPI_HOME should be set
NVSHMEM_USE_GDRCOPY ?= 0
# Include support for PMIx as the process manager interface
NVSHMEM_PMIX_SUPPORT ?= 0
PMIX_HOME ?= /usr
# One of the below can be set to 1 to override the default PMI
NVSHMEM_DEFAULT_PMIX ?= 0
NVSHMEM_DEFAULT_PMI2 ?= 0

# NVSHMEM internal features
# Enable debug mode, builds library with -g
NVSHMEM_DEBUG ?= 0
NVSHMEM_TRACE ?= 0
NVSHMEM_COMPLEX_SUPPORT ?= 0
NVSHMEM_VERBOSE ?= 0
NVSHMEM_DEVEL ?= 0
NVSHMEM_DISABLE_COLL_POLL ?= 1
NVSHMEM_GPU_COLL_USE_LDST ?= 0
# Timeout if stuck for long in wait loops in device
NVSHMEM_TIMEOUT_DEVICE_POLLING ?= 0

#NVSHMEM_LMPICXX ?= -lmpi_cxx #: Not required for openmpi-4.0.0 unless configured with --enable-mpi-cxx
ARCH := $(shell uname -m)


mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))

ifeq ($(ARCH), x86_64)
CXXFLAGS   := -fPIC -fpermissive -I$(CUDA_INC) -msse 
CXXFLAGS += -DNVSHMEM_X86_64
NVCUFLAGS := -DNVSHMEM_X86_64
else
ifeq ($(ARCH), ppc64le)
CXXFLAGS   := -fPIC -fpermissive -I$(CUDA_INC) -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS
CXXFLAGS += -DNVSHMEM_PPC64LE
NVCUFLAGS += -DNVSHMEM_PPC64LE
endif
endif
NVCUFLAGS  += -Xcompiler -fPIC -Xcompiler -fpermissive -ccbin $(CXX) $(NVCC_GENCODE) -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS

ifeq ($(NVSHMEM_PMIX_SUPPORT), 1)
CXXFLAGS  += -DNVSHMEM_PMIX_SUPPORT -I$(PMIX_HOME)/include
else
# Don't allow PMIX to be set as default unless it's in the build
NVSHMEM_DEFAULT_PMIX := 0
endif

ifeq ($(NVSHMEM_DEFAULT_PMIX), 1)
CXXFLAGS  += -DNVSHMEM_DEFAULT_PMIX
else
ifeq ($(NVSHMEM_DEFAULT_PMI2), 1)
CXXFLAGS  += -DNVSHMEM_DEFAULT_PMI2
endif
endif

ifeq ($(NVSHMEM_MPI_SUPPORT), 1)
CXXFLAGS  += -I$(MPI_HOME)/include -DNVSHMEM_MPI_SUPPORT
NVCUFLAGS += -I$(MPI_HOME)/include -DNVSHMEM_MPI_SUPPORT
endif

ifeq ($(NVSHMEM_USE_GDRCOPY), 1)
CXXFLAGS  += -I$(GDRCOPY_HOME)/include -DNVSHMEM_USE_GDRCOPY
NVCUFLAGS += -I$(GDRCOPY_HOME)/include -DNVSHMEM_USE_GDRCOPY
endif

ifeq ($(NVSHMEM_SHMEM_SUPPORT), 1)
CXXFLAGS  += -I$(SHMEM_HOME)/include -DNVSHMEM_SHMEM_SUPPORT
NVCUFLAGS += -I$(SHMEM_HOME)/include -DNVSHMEM_SHMEM_SUPPORT
endif

ifeq ($(NVSHMEM_USE_NCCL), 1)
CXXFLAGS  += -I$(NCCL_HOME)/include -DNVSHMEM_USE_NCCL
NVCUFLAGS += -I$(NCCL_HOME)/include -DNVSHMEM_USE_NCCL
endif

ifeq ($(NVSHMEM_COMPLEX_SUPPORT), 1)
CXXFLAGS  += -DNVSHMEM_COMPLEX_SUPPORT
NVCUFLAGS += -DNVSHMEM_COMPLEX_SUPPORT
endif

ifeq ($(NVSHMEM_MPI_IS_OMPI), 1)
CXXFLAGS  += -DNVSHMEM_MPI_IS_OMPI
NVCUFLAGS += -DNVSHMEM_MPI_IS_OMPI
endif

ifeq ($(NVSHMEM_DISABLE_COLL_POLL), 1)
CXXFLAGS  += -DNVSHMEM_DISABLE_COLL_POLL
NVCUFLAGS += -DNVSHMEM_DISABLE_COLL_POLL
endif
ifeq ($(NVSHMEM_GPU_COLL_USE_LDST), 1)
CXXFLAGS  += -DNVSHMEM_GPU_COLL_USE_LDST
NVCUFLAGS += -DNVSHMEM_GPU_COLL_USE_LDST
endif

ifeq ($(NVSHMEM_TIMEOUT_DEVICE_POLLING), 1)
CXXFLAGS  += -DNVSHMEM_TIMEOUT_DEVICE_POLLING
NVCUFLAGS += -DNVSHMEM_TIMEOUT_DEVICE_POLLING
endif

ifeq ($(NVSHMEM_TRACE), 1)
NVCUFLAGS += -DENABLE_TRACE
CXXFLAGS  += -DENABLE_TRACE
endif

ifeq ($(NVSHMEM_DEBUG), 0)
NVCUFLAGS += -O3
CXXFLAGS  += -O3
else
NVCUFLAGS += -O0 -g -G -DENABLE_TRACE
CXXFLAGS  += -O0 -g -DENABLE_TRACE
endif

ifneq ($(NVSHMEM_VERBOSE), 0)
NVCUFLAGS += -lineinfo -Xptxas -v -Xcompiler -Wall,-Wextra
CXXFLAGS  += -Wall -Wextra
endif

ifneq ($(NVSHMEM_DEVEL), 0)
NVCUFLAGS += -Werror all-warnings
CXXFLAGS  += -Werror
endif

ifneq (, $(filter 1, $(NVSHMEM_TIMEOUT_DEVICE_POLLING) $(NVSHMEM_DEBUG)))
NVCUFLAGS += --maxrregcount 128
else
NVCUFLAGS += --maxrregcount 64
endif

.PHONY : default 
default : lib

LICENSE_FILES := NVSHMEM-SLA.txt COPYRIGHT.txt
LICENSE_TARGETS := $(LICENSE_FILES:%=$(NVSHMEM_BUILDDIR)/%)
lic: $(LICENSE_TARGETS)

${NVSHMEM_BUILDDIR}/%.txt: %.txt
	@printf "Copying    %-35s > %s\n" $< $@
	mkdir -p ${NVSHMEM_BUILDDIR}
	cp $< $@

INCEXPORTS_NVSHMEM  := nvshmem.h nvshmemx.h
INCEXPORTS := nvshmem_api.h nvshmemx_api.h nvshmemx_error.h nvshmem_coll_api.h nvshmemx_coll_api.h nvshmem_constants.h nvshmemi_constants.h nvshmem_common.cuh nvshmemi_util.h nvshmem_defines.h nvshmemx_defines.h nvshmem.h nvshmemx.h nvshmem_types.h

LIBSRCFILES := bootstrap/bootstrap.cpp
LIBSRCFILES += pmi/pmi-2/pmi2_api.c pmi/pmi-2/pmi2_util.c
LIBSRCFILES += pmi/simple-pmi/simple_pmi.cpp pmi/simple-pmi/simple_pmiutil.cpp
LIBSRCFILES += bootstrap/bootstrap_pmi.cpp
ifeq ($(NVSHMEM_PMIX_SUPPORT), 1)
LIBSRCFILES += bootstrap/bootstrap_pmix.cpp bootstrap/bootstrap_loader.cpp
endif
ifeq ($(NVSHMEM_MPI_SUPPORT), 1) 
LIBSRCFILES += bootstrap/bootstrap_mpi.cpp  
endif
ifeq ($(NVSHMEM_SHMEM_SUPPORT), 1) 
LIBSRCFILES += bootstrap/bootstrap_shmem.cpp  
endif
LIBSRCFILES += init/init.cpp init/init_device.cu init/query.cu
LIBSRCFILES += launch/collective_launch.cpp
LIBSRCFILES += util/util.cpp util/cs.cpp util/env_vars.cpp
LIBSRCFILES += topo/topo.cpp
LIBSRCFILES += comm/proxy/proxy.cu comm/device/proxy_device.cu
LIBSRCFILES += comm/transport.cpp comm/transports/p2p/p2p.cpp comm/transports/ibrc/ibrc.cpp
LIBSRCFILES += coll/host/cpu_coll.cpp
LIBSRCFILES += coll/host/barrier.cpp
LIBSRCFILES += coll/host/broadcast.cpp
LIBSRCFILES += coll/host/alltoall.cpp
LIBSRCFILES += coll/host/collect.cpp
LIBSRCFILES += coll/host/rdxn.cpp
LIBSRCFILES += coll/host/barrier_on_stream.cpp
LIBSRCFILES += coll/host/broadcast_on_stream.cpp
LIBSRCFILES += coll/host/alltoall_on_stream.cpp
LIBSRCFILES += coll/host/collect_on_stream.cpp
LIBSRCFILES += coll/host/rdxn_on_stream.cpp
LIBSRCFILES += coll/device/gpu_coll.cu
LIBSRCFILES += coll/device/gpu_coll_dev.cu coll/device/recexchalgo.cu
LIBSRCFILES += coll/device/barrier.cu
LIBSRCFILES += coll/device/bcast.cu
LIBSRCFILES += coll/device/collect.cu
LIBSRCFILES += coll/device/alltoall.cu
LIBSRCFILES += coll/device/rdxn.cu coll/device/rdxn_threadgroup.cu
LIBSRCFILES += mem/mem.cpp mem/dlmalloc.c mem/device/mem.cu
LIBSRCFILES += comm/host/putget.cpp comm/host/amo.cu comm/host/fence.cpp comm/host/sync.cpp comm/host/quiet.cpp comm/host/proxy/rma.cu
LIBSRCFILES += comm/host/quiet_on_stream.cu
LIBSRCFILES += comm/host/cuda_interface_sync.cu
LIBSRCFILES += comm/device/put.cu comm/device/get.cu comm/device/amo.cu
LIBSRCFILES += comm/device/put_threadgroup.cu comm/device/get_threadgroup.cu
LIBSRCFILES += team/team.cu team/team_internal.cu

LIBNAME     := libnvshmem.a

INCDIR := $(NVSHMEM_BUILDDIR)/include
LIBDIR := $(NVSHMEM_BUILDDIR)/lib
OBJDIR_NVSHMEM := $(NVSHMEM_BUILDDIR)/obj_nvshmem

INCTARGETS := $(patsubst %, $(INCDIR)/%, $(INCEXPORTS))

LIBTARGET  := $(LIBNAME)
LIBOBJ     := $(patsubst %.cpp, $(OBJDIR_NVSHMEM)/%.o, $(filter %.cpp, $(LIBSRCFILES)))
LIBOBJ     += $(patsubst %.c, $(OBJDIR_NVSHMEM)/%.o, $(filter %.c, $(LIBSRCFILES)))
LIBOBJ     += $(patsubst %.cu, $(OBJDIR_NVSHMEM)/%.o, $(filter %.cu, $(LIBSRCFILES)))
LIBOBJ     += $(OBJDIR_NVSHMEM)/bootstrap/bootstrap_pmi2.o
LIBINC     := -Isrc/include -Isrc/util -Isrc/bootstrap -Isrc/comm -Isrc/coll/host -Isrc/coll/device -Isrc/coll -Isrc/topo
LIBINC     += -Isrc/pmi/pmi-2 -Isrc/pmi/simple-pmi

ifeq ($(NVSHMEM_PMIX_SUPPORT), 1)
PLUGINS    := $(LIBDIR)/nvshmem_pmix.so
endif

lib : $(INCTARGETS) $(LIBDIR)/$(LIBTARGET) $(PLUGINS)
$(LIBDIR)/$(LIBTARGET) : $(LIBOBJ)
	@mkdir -p $(LIBDIR)
	$(NVCC) -lib -o $@ $(LIBOBJ)

$(LIBDIR)/nvshmem_pmix.so: src/bootstrap/bootstrap_pmix.cpp
	@mkdir -p $(LIBDIR)
	$(CXX) -shared -fpic $(LIBINC) $(CXXFLAGS) $< -L$(PMIX_HOME)/lib -lpmix -o $@

$(INCDIR)/%.h : src/include/%.h
	@mkdir -p $(INCDIR)
	cp -f $< $@

$(INCDIR)/%.cuh : src/include/%.cuh
	@mkdir -p $(INCDIR)
	cp -f $< $@

$(OBJDIR_NVSHMEM)/%.o : src/%.cpp
	@mkdir -p `dirname $@`
	$(CXX) -c $(LIBINC) $(CXXFLAGS) $< -o $@

$(OBJDIR_NVSHMEM)/bootstrap/bootstrap_pmi2.o : src/bootstrap/bootstrap_pmi.cpp
	@mkdir -p `dirname $@`
	$(CXX) -c $(LIBINC) -DNVSHMEM_BUILD_PMI2 $(CXXFLAGS) $< -o $@

$(OBJDIR_NVSHMEM)/%.o : src/%.c
	@mkdir -p `dirname $@`
	$(CC) -c $(LIBINC) $(CXXFLAGS) $< -o $@

$(OBJDIR_NVSHMEM)/%.o : src/%.cu
	@mkdir -p `dirname $@`
	$(NVCC) -c $(LIBINC) $(NVCUFLAGS) -rdc=true $< -o $@

clean :
	rm -rf $(NVSHMEM_BUILDDIR)

install : lib
	mkdir -p $(NVSHMEM_PREFIX)/lib
	mkdir -p $(NVSHMEM_PREFIX)/include
	cp -v $(NVSHMEM_BUILDDIR)/lib/* $(NVSHMEM_PREFIX)/lib/
	cp -P -v $(NVSHMEM_BUILDDIR)/include/nvshmem* $(NVSHMEM_PREFIX)/include
	cp -P -v $(NVSHMEM_BUILDDIR)/include/* $(NVSHMEM_PREFIX)/include/
