/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _CUDA_INTERFACE_SYNC_H_
#define _CUDA_INTERFACE_SYNC_H_

#define DECL_CUDA_INTERFACE_TYPE_WAIT(type, TYPE) \
    cudaError_t cuda_interface_##type##_wait(volatile TYPE *ivar, TYPE cmp_value);

NVSHMEMI_REPT_FOR_WAIT_TYPES(DECL_CUDA_INTERFACE_TYPE_WAIT)
#undef DECL_CUDA_INTERFACE_TYPE_WAIT

#define DECL_CUDA_INTERFACE_TYPE_WAIT_ON_STREAM(type, TYPE)                                 \
    cudaError_t cuda_interface_##type##_wait_on_stream(volatile TYPE *ivar, TYPE cmp_value, \
                                                       cudaStream_t cstream);

NVSHMEMI_REPT_FOR_WAIT_TYPES(DECL_CUDA_INTERFACE_TYPE_WAIT_ON_STREAM)
#undef DECL_CUDA_INTERFACE_TYPE_WAIT_ON_STREAM

#define DECL_CUDA_INTERFACE_TYPE_WAIT_UNTIL(type, TYPE) \
    cudaError_t cuda_interface_##type##_wait_until(volatile TYPE *ivar, int cmp, TYPE cmp_value);

NVSHMEMI_REPT_FOR_WAIT_TYPES(DECL_CUDA_INTERFACE_TYPE_WAIT_UNTIL)
#undef DECL_CUDA_INTERFACE_TYPE_WAIT_UNTIL

#define DECL_CUDA_INTERFACE_TYPE_WAIT_UNTIL_ON_STREAM(type, TYPE) \
    cudaError_t cuda_interface_##type##_wait_until_on_stream(     \
        volatile TYPE *ivar, int cmp, TYPE cmp_value, cudaStream_t cstream);

NVSHMEMI_REPT_FOR_WAIT_TYPES(DECL_CUDA_INTERFACE_TYPE_WAIT_UNTIL_ON_STREAM)
#undef DECL_CUDA_INTERFACE_TYPE_WAIT_UNTIL_ON_STREAM

#endif
