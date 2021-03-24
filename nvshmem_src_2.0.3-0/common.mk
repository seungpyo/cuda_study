#
# Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
#
# See LICENCE.txt for license information
#

CUDA_HOME ?= /usr/local/cuda-11.2
CUDA_DRV ?= $(CUDA_HOME)/lib64/stubs
CUDA_INC ?= $(CUDA_HOME)/include
CUDA_LIB ?= $(CUDA_HOME)/lib64
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_VERSION = $(strip $(shell $(NVCC) --version | grep release | sed 's/^.*release \([0-9]\+\.[0-9]\+\).*/\1/'))
CUDA_MAJOR = $(shell echo $(CUDA_VERSION) | cut -d "." -f 1)
CUDA_MINOR = $(shell echo $(CUDA_VERSION) | cut -d "." -f 2)

# Better define NVCC_GENCODE in your environment to the minimal set
# of archs to reduce compile time.
NVCC_GENCODE ?= -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70
ifeq ($(shell test "0$(CUDA_MAJOR)" -ge 11; echo $$?),0)
NVCC_GENCODE += -gencode=arch=compute_80,code=sm_80
endif
