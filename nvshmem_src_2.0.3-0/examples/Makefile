# Copyright (c) 2018-2020, NVIDIA CORPORATION.  All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See COPYRIGHT.txt for license information

include ../common.mk

NVSHMEM_HOME ?= /usr/local/nvshmem-0.1

NVSHMEM_MPI_SUPPORT ?= 0
MPI_HOME ?= /usr/local/openmpi-3.0.1

NVSHMEM_DEBUG ?= 0
NVSHMEM_VERBOSE ?= 0

mkfile_path := $(abspath $(firstword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))

EXAMPLES_INSTALL ?= $(mkfile_dir)/examples_install

TESTCUFLAGS  := -ccbin $(CXX) 

# Use addprefix so that we can specify more than one path
TESTLDFLAGS :=

TESTINC := -I$(CUDA_INC) -I$(NVSHMEM_HOME)/include  -I$(mkfile_dir)/common

ifeq ($(NVSHMEM_MPI_SUPPORT), 1)
MPI_LIB = -lmpi  
TESTINC += -I$(MPI_HOME)/include -DENABLE_MPI_SUPPORT
endif

ifeq ($(NVSHMEM_DEBUG), 0)
TESTCUFLAGS  += -O3 
else
TESTCUFLAGS  += -O0 -g -G -lineinfo
endif

ifneq ($(NVSHMEM_VERBOSE), 0)
TESTCUFLAGS  += -lineinfo -Xptxas -v -Xcompiler -Wall,-Wextra
endif

TESTLDFLAGS += -L$(NVSHMEM_HOME)/lib -lnvshmem -lcuda -L$(CUDA_HOME)/lib64 -lcudart  
ifeq ($(NVSHMEM_MPI_SUPPORT), 1)
TESTLDFLAGS += -L$(MPI_HOME)/lib $(MPI_LIB)
endif
 
.PHONY : default 
default : examples

EXAMPLECUSRCFILES := \
    collective-launch.cu        \
    on-stream.cu                \
    thread-group.cu             \
    put_block.cu                \
    dev-guide-ring.cu

ifeq ($(NVSHMEM_MPI_SUPPORT), 1)
EXAMPLECUSRCFILES += mpi-based-init.cu dev-guide-ring-mpi.cu
endif

OBJDIR := $(mkfile_dir)/obj

CUPERFBIN   := $(patsubst %.cu, $(OBJDIR)/%, $(filter %.cu, $(EXAMPLECUSRCFILES)))
CXXPERFBIN  := $(patsubst %.cpp, $(OBJDIR)/%, $(filter %.cpp, $(PERFCXXSRCFILES)))

examples : $(CUPERFBIN) $(CXXPERFBIN)
 
$(OBJDIR)/% : %.cu
	@printf "Compiling %-25s > %-25s\n" $< $@
	mkdir -p `dirname $@`
	$(NVCC) $(NVCC_GENCODE) $(TESTCUFLAGS) $(TESTINC) -rdc=true $< -o $@ $(TESTLDFLAGS)

clean :
	rm -rf $(OBJDIR)

install : examples
	mkdir -p $(EXAMPLES_INSTALL)
	cp -r -v $(OBJDIR)/* $(EXAMPLES_INSTALL)/
