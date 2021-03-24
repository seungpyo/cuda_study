/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmem_internal.h"

#include "sync_kernel_entrypoints.cuh"

#define CUDA_INTERFACE_TYPE_WAIT(type, TYPE)                                        \
    cudaError_t cuda_interface_##type##_wait(volatile TYPE *ivar, TYPE cmp_value) { \
        void *kernelParams[] = {&ivar, &cmp_value};                                 \
        dim3 gdim(1), bdim(1);                                                      \
        void (*funcPtr)(TYPE *, TYPE) = WaitKernel<TYPE>;                           \
        return cudaLaunchKernel((const void *)funcPtr, gdim, bdim, kernelParams, 0, \
                                nvshmemi_state->my_stream);                          \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(CUDA_INTERFACE_TYPE_WAIT)
#undef CUDA_INTERFACE_TYPE_WAIT

#define CUDA_INTERFACE_TYPE_WAIT_ON_STREAM(type, TYPE)                                        \
    cudaError_t cuda_interface_##type##_wait_on_stream(volatile TYPE *ivar, TYPE cmp_value,   \
                                                       cudaStream_t cstream) {                \
        void *kernelParams[] = {&ivar, &cmp_value};                                           \
        dim3 gdim(1), bdim(1);                                                                \
        void (*funcPtr)(TYPE *, TYPE) = WaitKernel<TYPE>;                                     \
        return cudaLaunchKernel((const void *)funcPtr, gdim, bdim, kernelParams, 0, cstream); \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(CUDA_INTERFACE_TYPE_WAIT_ON_STREAM)
#undef CUDA_INTERFACE_TYPE_WAIT_ON_STREAM

#define CUDA_INTERFACE_TYPE_WAIT_UNTIL(type, TYPE)                                                 \
    cudaError_t cuda_interface_##type##_wait_until(volatile TYPE *ivar, int cmp, TYPE cmp_value) { \
        void *kernelParams[] = {&ivar, &cmp_value};                                                \
        dim3 gdim(1), bdim(1);                                                                     \
        void (*funcPtr)(TYPE *, TYPE);                                                             \
        switch (cmp) {                                                                             \
            case NVSHMEM_CMP_EQ:                                                                   \
                funcPtr = WaitUntilKernel<TYPE, equal_to<TYPE> >;                                  \
                break;                                                                             \
            case NVSHMEM_CMP_NE:                                                                   \
                funcPtr = WaitUntilKernel<TYPE, not_equal_to<TYPE> >;                              \
                break;                                                                             \
            case NVSHMEM_CMP_GT:                                                                   \
                funcPtr = WaitUntilKernel<TYPE, greater_than<TYPE> >;                              \
                break;                                                                             \
            case NVSHMEM_CMP_LE:                                                                   \
                funcPtr = WaitUntilKernel<TYPE, less_equal_to<TYPE> >;                             \
                break;                                                                             \
            case NVSHMEM_CMP_LT:                                                                   \
                funcPtr = WaitUntilKernel<TYPE, less_than<TYPE> >;                                 \
                break;                                                                             \
            case NVSHMEM_CMP_GE:                                                                   \
                funcPtr = WaitUntilKernel<TYPE, greater_equal_to<TYPE> >;                          \
                break;                                                                             \
        }                                                                                          \
        return cudaLaunchKernel((const void *)funcPtr, gdim, bdim, kernelParams, 0,                \
                                nvshmemi_state->my_stream);                                         \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(CUDA_INTERFACE_TYPE_WAIT_UNTIL)
#undef CUDA_INTERFACE_TYPE_WAIT_UNTIL

#define CUDA_INTERFACE_TYPE_WAIT_UNTIL_ON_STREAM(type, TYPE)                                  \
    cudaError_t cuda_interface_##type##_wait_until_on_stream(                                 \
        volatile TYPE *ivar, int cmp, TYPE cmp_value, cudaStream_t cstream) {                 \
        void *kernelParams[] = {&ivar, &cmp_value};                                           \
        dim3 gdim(1), bdim(1);                                                                \
        void (*funcPtr)(TYPE *, TYPE);                                                        \
        switch (cmp) {                                                                        \
            case NVSHMEM_CMP_EQ:                                                              \
                funcPtr = WaitUntilKernel<TYPE, equal_to<TYPE> >;                             \
                break;                                                                        \
            case NVSHMEM_CMP_NE:                                                              \
                funcPtr = WaitUntilKernel<TYPE, not_equal_to<TYPE> >;                         \
                break;                                                                        \
            case NVSHMEM_CMP_GT:                                                              \
                funcPtr = WaitUntilKernel<TYPE, greater_than<TYPE> >;                         \
                break;                                                                        \
            case NVSHMEM_CMP_LE:                                                              \
                funcPtr = WaitUntilKernel<TYPE, less_equal_to<TYPE> >;                        \
                break;                                                                        \
            case NVSHMEM_CMP_LT:                                                              \
                funcPtr = WaitUntilKernel<TYPE, less_than<TYPE> >;                            \
                break;                                                                        \
            case NVSHMEM_CMP_GE:                                                              \
                funcPtr = WaitUntilKernel<TYPE, greater_equal_to<TYPE> >;                     \
                break;                                                                        \
        }                                                                                     \
        return cudaLaunchKernel((const void *)funcPtr, gdim, bdim, kernelParams, 0, cstream); \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(CUDA_INTERFACE_TYPE_WAIT_UNTIL_ON_STREAM)
#undef CUDA_INTERFACE_TYPE_WAIT_UNTIL_ON_STREAM
