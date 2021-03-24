/****
 * Copyright (c) 2016-2019, NVIDIA Corporation.  All rights reserved.
 *
 * See COPYRIGHT for license information
 ****/

#ifndef __DEVICE_UTIL_H
#define __DEVICE_UTIL_H

#include <assert.h>
#include <stdio.h>

#ifdef __CUDA_ARCH__
__device__ inline int nvshmemi_thread_id_in_warp() {
    int myIdx;
    asm volatile("mov.u32  %0, %laneid;" : "=r"(myIdx));
    return myIdx;
}

__device__ inline int nvshmemi_warp_size() {
    return ((blockDim.x * blockDim.y * blockDim.z) < warpSize)
               ? (blockDim.x * blockDim.y * blockDim.z)
               : warpSize;
}

__device__ inline void nvshmemi_warp_sync() { __syncwarp(); }


__device__ inline int nvshmemi_thread_id_in_block() {
    return (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y);
}

__device__ inline int nvshmemi_block_size() { return (blockDim.x * blockDim.y * blockDim.z); }

__device__ inline void nvshmemi_block_sync() { __syncthreads(); }


__device__ inline int nvshmemi_thread_id_in_thread() {
    return 0;
}

__device__ inline int nvshmemi_thread_size() { return 1; }

__device__ inline void nvshmemi_thread_sync() {}



#endif

static inline void nvshmemi_bit_set(unsigned char *ptr, size_t size, size_t index)
{
    assert(size > 0 && (index < size * CHAR_BIT));

    size_t which_byte = index / size;
    ptr[which_byte] |= (1 << (index % CHAR_BIT));

    return;
}

static inline void nvshmemi_bit_clear(unsigned char *ptr, size_t size, size_t index)
{
    assert(size > 0 && (index < size * CHAR_BIT));

    size_t which_byte = index / size;
    ptr[which_byte] &= ~(1 << (index % CHAR_BIT));

    return;
}

static inline unsigned char nvshmemi_bit_fetch(unsigned char *ptr, size_t index)
{
    return (ptr[index / CHAR_BIT] >> (index % CHAR_BIT)) & 1;
}

static inline size_t nvshmemi_bit_1st_nonzero(const unsigned char *ptr, const size_t size)
{
    /* The following ignores endianess: */
    for(size_t i = 0; i < size; i++) {
        unsigned char bit_val = ptr[i];
        for (size_t j = 0; bit_val && j < CHAR_BIT; j++) {
            if (bit_val & 1) return i * CHAR_BIT + j;
            bit_val >>= 1;
        }
    }

    return (size_t)-1;
}

/* Create a bit string of the format AAAAAAAA.BBBBBBBB into str for the byte
 * array passed via ptr. */
static inline void nvshmemi_bit_to_string(char *str, size_t str_size,
                                  unsigned char *ptr, size_t ptr_size)
{
    size_t off = 0;

    for (size_t i = 0; i < ptr_size; i++) {
        for (size_t j = 0; j < CHAR_BIT; j++) {
            off += snprintf(str+off, str_size-off, "%s",
                            (ptr[i] & (1 << (CHAR_BIT-1-j))) ? "1" : "0");
            if (off >= str_size) return;
        }
        if (i < ptr_size - 1) {
            off += snprintf(str+off, str_size-off, ".");
            if (off >= str_size) return;
        }
    }
}

/* Return -1 if `global_pe` is not in the given active set.
 * If `global_pe` is in the active set, return the PE index within this set. */
__host__ __device__ static inline int nvshmemi_pe_in_active_set(int global_pe, int PE_start, int PE_stride, int PE_size)
{
    int n = (global_pe - PE_start) / PE_stride;
    if (global_pe < PE_start || (global_pe - PE_start) % PE_stride || n >= PE_size)
        return -1;
    else {
        return n;
    }
}

template<typename T>
__global__ void nvshmemi_init_array_kernel(T *array, int len, T val) {
    for (int i = 0; i < len; i++)
        array[i] = val;
}
#endif
