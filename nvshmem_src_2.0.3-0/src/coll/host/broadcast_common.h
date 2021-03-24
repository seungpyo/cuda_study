/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef NVSHMEMI_BCAST_COMMON_CPU_H
#define NVSHMEMI_BCAST_COMMON_CPU_H
#include <cuda.h>
#include <cuda_runtime.h>

#if __cplusplus
extern "C" {
#endif
#define CALL_TYPENAME_BCAST_ON_STREAM_KERN(TYPENAME, TYPE)                                              \
    void call_##TYPENAME##_broadcast_on_stream_kern(void *dest, const void *source, size_t nelems,      \
                                               int PE_root, int PE_start, int logPE_stride,             \
                                               int PE_size, long *pSync, cudaStream_t stream);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(CALL_TYPENAME_BCAST_ON_STREAM_KERN)
#undef CALL_TYPENAME_BCAST_ON_STREAM_KERN

#if __cplusplus
}
#endif

#endif
