# Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See COPYRIGHT.txt for license information

# Dependencies
include ../common.mk
NVSHMEM_HOME ?= /usr/local/nvshmem/

# MPI/SHMEM support
# Whether to build with MPI support
NVSHMEM_MPI_SUPPORT ?= 1
MPI_HOME ?= /usr/local/ompi
# Name of the mpi library. For example, for spectrum-mpi it would be -lmpi_ibm
NVSHMEM_LMPI ?= -lmpi
# Whether to build with SHMEM support
NVSHMEM_SHMEM_SUPPORT ?= 0
SHMEM_HOME ?= $(MPI_HOME)

# NVSHMEM features
NVSHMEM_DEBUG ?= 0
NVSHMEM_VERBOSE ?= 0
#NVSHMEM_LMPICXX ?= -lmpi_cxx

# Define this on the command line to change the format of perftest readouts to
# machine readable format.
NVSHMEM_MACHINE_READABLE_OUTPUT ?= 0

MPI_LIBS := $(NVSHMEM_LMPI) $(NVSHMEM_LMPICXX)
SHMEM_LIBS := -loshmem

mkfile_path := $(abspath $(firstword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))

NVSHMEM_PERFTEST_INSTALL ?= $(mkfile_dir)/perftest_install

TESTCUFLAGS  := -dc -ccbin $(CXX) $(NVCC_GENCODE)

# Use addprefix so that we can specify more than one path
TESTLDFLAGS :=

TESTINC := -I$(CUDA_INC) -I$(mkfile_dir)/common

ifeq ($(NVSHMEM_SHMEM_SUPPORT), 1)
TESTINC += -I$(SHMEM_HOME)/include -DNVSHMEM_SHMEM_SUPPORT
endif
TESTINC += -I$(NVSHMEM_HOME)/include

ifeq ($(NVSHMEM_MPI_SUPPORT), 1)
TESTINC += -I$(MPI_HOME)/include -DNVSHMEM_MPI_SUPPORT
endif

ifeq ($(NVSHMEM_DEBUG), 0)
TESTCUFLAGS  += -O3 
else
TESTCUFLAGS  += -O0 -g -G -lineinfo -D_NVSHMEM_DEBUG
endif

ifeq ($(NVSHMEM_MACHINE_READABLE_OUTPUT), 1)
TESTCUFLAGS  += -DNVSHMEM_MACHINE_READABLE_OUTPUT
endif

ifneq ($(NVSHMEM_VERBOSE), 0)
TESTCUFLAGS  += -lineinfo -Xptxas -v -Xcompiler -Wall,-Wextra
endif


TESTLDFLAGS += -lcuda -L$(CUDA_HOME)/lib64 -lcudart
TESTLDFLAGS += $(NVSHMEM_HOME)/lib/libnvshmem.a

ifeq ($(NVSHMEM_MPI_SUPPORT), 1)
TESTLDFLAGS += -L$(MPI_HOME)/lib $(MPI_LIBS)
endif
ifeq ($(NVSHMEM_SHMEM_SUPPORT), 1)
TESTLDFLAGS += -L$(SHMEM_HOME)/lib $(SHMEM_LIBS)
endif
 
.PHONY : default 
default : perftest

PERFCUHELPERFILES := common/utils.cu
PERFCUSRCFILES := host/pt-to-pt/stream_latency.cu
PERFCUSRCFILES += device/pt-to-pt/shmem_p_latency.cu device/pt-to-pt/shmem_p_bw.cu 
PERFCUSRCFILES += device/pt-to-pt/shmem_g_latency.cu device/pt-to-pt/shmem_g_bw.cu 
PERFCUSRCFILES += device/pt-to-pt/shmem_st_bw.cu device/pt-to-pt/shmem_p_ping_pong_latency.cu
PERFCUSRCFILES += device/pt-to-pt/shmem_put_atomic_ping_pong_latency.cu
PERFCUSRCFILES += device/pt-to-pt/shmem_put_signal_ping_pong_latency.cu
PERFCUSRCFILES += device/pt-to-pt/shmem_signal_ping_pong_latency.cu
PERFCUSRCFILES += device/pt-to-pt/shmem_put_latency.cu device/pt-to-pt/shmem_put_ping_pong_latency.cu 
PERFCUSRCFILES += device/pt-to-pt/shmem_get_latency.cu
PERFCUSRCFILES += device/pt-to-pt/shmem_put_bw.cu 
PERFCUSRCFILES += device/pt-to-pt/shmem_get_bw.cu 
PERFCUSRCFILES += device/coll/barrier_latency.cu
PERFCUSRCFILES += device/coll/sync_latency.cu
PERFCUSRCFILES += device/coll/bcast_latency.cu
PERFCUSRCFILES += device/coll/collect_latency.cu
PERFCUSRCFILES += device/coll/alltoall_latency.cu
PERFCUSRCFILES += device/coll/reduction_latency.cu
PERFCXXSRCFILES := host/pt-to-pt/bw.cpp host/pt-to-pt/latency.cpp
PERFCXXSRCFILES += host/coll/barrier_all.cpp host/coll/barrier.cpp
PERFCXXSRCFILES += host/coll/sync_all.cpp host/coll/sync.cpp
PERFCXXSRCFILES += host/coll/alltoall.cpp
PERFCXXSRCFILES += host/coll/broadcast.cpp
PERFCXXSRCFILES += host/coll/collect.cpp
PERFCXXSRCFILES += host/coll/reduction.cpp
PERFCXXSRCFILES += host/coll/barrier_on_stream.cpp host/coll/barrier_all_on_stream.cpp
PERFCXXSRCFILES += host/coll/sync_on_stream.cpp host/coll/sync_all_on_stream.cpp
PERFCXXSRCFILES += host/coll/alltoall_on_stream.cpp
PERFCXXSRCFILES += host/coll/broadcast_on_stream.cpp
PERFCXXSRCFILES += host/coll/collect_on_stream.cpp
PERFCXXSRCFILES += host/coll/reduction_on_stream.cpp

LIBDIR := $(NVSHMEM_HOME)/lib
OBJDIR := $(mkfile_dir)/obj

CUPERFHELPEROBJ := $(patsubst %.cu, $(OBJDIR)/%.o, $(filter %.cu, $(PERFCUHELPERFILES)))
CUPERFBIN := $(patsubst %.cu, $(OBJDIR)/%, $(filter %.cu, $(PERFCUSRCFILES)))
CXXPERFBIN := $(patsubst %.cpp, $(OBJDIR)/%, $(filter %.cpp, $(PERFCXXSRCFILES)))

perftest : $(CUPERFHELPEROBJ) $(CUPERFBIN) $(CXXPERFBIN)
 
$(OBJDIR)/common/%.o : common/%.cu
	@mkdir -p `dirname $@`
	$(NVCC) $(TESTCUFLAGS) $(TESTINC) $< -c -o $@ 

$(OBJDIR)/device/pt-to-pt/% : device/pt-to-pt/%.cu $(CUPERFHELPEROBJ) 
	@mkdir -p `dirname $@`
	$(NVCC) $(TESTCUFLAGS) $(TESTINC) $< -c -o $@.o
	$(NVCC) $(NVCC_GENCODE) $@.o $(CUPERFHELPEROBJ) -o $@ $(TESTLDFLAGS)
	@rm $@.o

$(OBJDIR)/device/coll/% : device/coll/%.cu $(CUPERFHELPEROBJ)
	@mkdir -p `dirname $@`
	$(NVCC) $(TESTCUFLAGS) $(TESTINC) $< -c -o $@.o
	$(NVCC) $(NVCC_GENCODE) $@.o $(CUPERFHELPEROBJ) -o $@ $(TESTLDFLAGS)
	@rm $@.o

$(OBJDIR)/host/coll/% : host/coll/%.cpp $(CUPERFHELPEROBJ)
	@mkdir -p `dirname $@`
	$(NVCC) $(TESTCUFLAGS) $(TESTINC) $< -c -o $@.o
	$(NVCC) $(NVCC_GENCODE) $@.o $(CUPERFHELPEROBJ) -o $@ $(TESTLDFLAGS)
	@rm $@.o

$(OBJDIR)/host/pt-to-pt/% : host/pt-to-pt/%.cpp $(CUPERFHELPEROBJ)
	@mkdir -p `dirname $@`
	$(NVCC) $(TESTCUFLAGS) $(TESTINC) $< -c -o $@.o
	$(NVCC) $(NVCC_GENCODE) $@.o $(CUPERFHELPEROBJ) -o $@ $(TESTLDFLAGS)
	@rm $@.o

$(OBJDIR)/host/pt-to-pt/% : host/pt-to-pt/%.cu $(CUPERFHELPEROBJ)
	@mkdir -p `dirname $@`
	$(NVCC) $(TESTCUFLAGS) $(TESTINC) $< -c -o $@.o
	$(NVCC) $(NVCC_GENCODE) $@.o $(CUPERFHELPEROBJ) -o $@ $(TESTLDFLAGS)
	@rm $@.o

clean :
	rm -rf $(OBJDIR)
	rm -rf $(NVSHMEM_PERFTEST_INSTALL)

install : perftest
	mkdir -p $(NVSHMEM_PERFTEST_INSTALL)
	cp -r -v $(OBJDIR)/device/ $(NVSHMEM_PERFTEST_INSTALL)/
	cp -r -v $(OBJDIR)/host/ $(NVSHMEM_PERFTEST_INSTALL)/
