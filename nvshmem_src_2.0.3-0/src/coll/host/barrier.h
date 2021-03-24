/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef NVSHMEMI_BARRIER_CPU_H
#define NVSHMEMI_BARRIER_CPU_H 1
#include "barrier_common.h"
void nvshmemi_sync(int PE_start, int PE_stride, int PE_size, long *pSync, long *counter);
#endif
