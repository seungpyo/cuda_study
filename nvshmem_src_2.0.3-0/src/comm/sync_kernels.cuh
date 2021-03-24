/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

template <typename T>
__device__ __forceinline__ void Wait(T *ivar, T cmp_value) {
    // printf("Wait kernel ivar %p *ivar %ld cmp_value %ld\n", ivar, *ivar, cmp_value);
    while (*ivar == cmp_value) {
    };
}

template <typename T>
struct equal_to {
    __device__ __forceinline__ void operator()(T *ivar, T cmp_value) {
        do {
            if (*ivar == cmp_value) break;
        } while (true);
    }
};

template <typename T>
struct not_equal_to {
    __device__ __forceinline__ void operator()(T *ivar, T cmp_value) {
        do {
            if (*ivar != cmp_value) break;
        } while (true);
    }
};

template <typename T>
struct greater_than {
    __device__ __forceinline__ void operator()(T *ivar, T cmp_value) {
        do {
            if (*ivar > cmp_value) break;
        } while (true);
    }
};

template <typename T>
struct less_equal_to {
    __device__ __forceinline__ void operator()(T *ivar, T cmp_value) {
        do {
            if (*ivar <= cmp_value) break;
        } while (true);
    }
};

template <typename T>
struct less_than {
    __device__ __forceinline__ void operator()(T *ivar, T cmp_value) {
        do {
            if (*ivar < cmp_value) break;
        } while (true);
    }
};

template <typename T>
struct greater_equal_to {
    __device__ __forceinline__ void operator()(T *ivar, T cmp_value) {
        do {
            if (*ivar >= cmp_value) break;
        } while (true);
    }
};
