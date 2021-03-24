/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "../sync_kernels.cuh"

template <typename T>
__global__ void WaitKernel(T *ivar, T cmp_value) {
    Wait(ivar, cmp_value);
}

template __global__ void WaitKernel<int>(int *ivar, int cmp_value);
template __global__ void WaitKernel<long>(long *ivar, long cmp_value);
template __global__ void WaitKernel<long long>(long long *ivar, long long cmp_value);
template __global__ void WaitKernel<short>(short *ivar, short cmp_value);

#define DECL_CMPRTS(TYPE)                \
    template struct equal_to<TYPE>;      \
    template struct not_equal_to<TYPE>;  \
    template struct greater_than<TYPE>;  \
    template struct less_equal_to<TYPE>; \
    template struct less_than<TYPE>;     \
    template struct greater_equal_to<TYPE>;

DECL_CMPRTS(int)
DECL_CMPRTS(long long)
DECL_CMPRTS(short)
DECL_CMPRTS(long)

template <typename T, typename Op>
__global__ void WaitUntilKernel(T *ivar, T cmp_value) {
    Op o;
    o(ivar, cmp_value);
}
