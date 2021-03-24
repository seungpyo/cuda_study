/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef NVSHMEMI_COLLECT_ON_STREAM_CPU_H
#define NVSHMEMI_COLLECT_ON_STREAM_CPU_H
#include "collect_common.h"

#define DECL_NVSHMEMXI_TYPENAME_COLLECT_ON_STREAM(TYPENAME, TYPE)                                  \
    void nvshmemxi_##TYPENAME##_collect_on_stream(TYPE *dest, const TYPE *source, size_t nelems,    \
                                     int PE_start, int PE_stride, int PE_size, long *pSync, cudaStream_t stream);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DECL_NVSHMEMXI_TYPENAME_COLLECT_ON_STREAM)
#undef DECL_NVSHMEMXI_TYPENAME_COLLECT_ON_STREAM

#endif /* NVSHMEMI_COLLECT_ON_STREAM_CPU_H */
