/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef NVSHMEMI_BARRIER_COMMON_CPU_H
#define NVSHMEMI_BARRIER_COMMON_CPU_H
#include <cuda.h>
#include <cuda_runtime.h>

#if __cplusplus
extern "C" {
#endif
int call_barrier_on_stream_kern(int PE_start, int PE_stride, int PE_size, long *pSync,
                                long *counter, cudaStream_t stream);

int call_sync_on_stream_kern(int PE_start, int PE_stride, int PE_size, long *pSync,
                             long *counter, cudaStream_t stream);
#if __cplusplus
}
#endif

#endif /* NVSHMEMI_BARRIER_COMMON_CPU_H */
