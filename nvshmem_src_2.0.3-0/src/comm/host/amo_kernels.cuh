/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <stdint.h>

/*inc*/
template <typename T>
__device__ __forceinline__ void AtomicInc(T *dest) {
    atomicAdd(dest, (T)1); /*XXX: CUDA atomicInc only supports unsigned int*/
}

template <typename T1, typename T2>
__device__ __forceinline__ void AtomicInc(T1 *dest) {
    atomicAdd((T2 *)dest, (T2)1);
}

/*finc, fetch*/
#define FINC_FETCH_DEVICE_FUNC(Opname1, Opname2, value)       \
    template <typename T>                                     \
    __device__ __forceinline__ T Atomic##Opname1(T *dest) {   \
        return atomic##Opname2(dest, (T)value);               \
    }                                                         \
                                                              \
    template <typename T1, typename T2>                       \
    __device__ __forceinline__ T1 Atomic##Opname1(T1 *dest) { \
        T2 temp = atomic##Opname2((T2 *)dest, (T2)value);     \
        return *((T1 *)&temp);                                \
    }

FINC_FETCH_DEVICE_FUNC(Finc, Add, 1)
FINC_FETCH_DEVICE_FUNC(Fetch, Or, 0)

/*and, or, xor, add, set*/
#define AND_OR_XOR_ADD_SET_DEVICE_FUNC(Opname1, Opname2)                  \
    template <typename T>                                                 \
    __device__ __forceinline__ void Atomic##Opname1(T *dest, T value) {   \
        atomic##Opname2(dest, value);                                     \
    }                                                                     \
                                                                          \
    template <typename T1, typename T2>                                   \
    __device__ __forceinline__ void Atomic##Opname1(T1 *dest, T1 value) { \
        atomic##Opname2((T2 *)dest, *((T2 *)&value));                     \
    }

AND_OR_XOR_ADD_SET_DEVICE_FUNC(And, And)
AND_OR_XOR_ADD_SET_DEVICE_FUNC(Or, Or)
AND_OR_XOR_ADD_SET_DEVICE_FUNC(Xor, Xor)
AND_OR_XOR_ADD_SET_DEVICE_FUNC(Add, Add)
AND_OR_XOR_ADD_SET_DEVICE_FUNC(Set, Exch)

/*fand, for, fxor, fadd, swap*/
#define FAND_FOR_FXOR_FADD_SWAP_DEVICE_FUNC(Opname1, Opname2)           \
    template <typename T>                                               \
    __device__ __forceinline__ T Atomic##Opname1(T *dest, T value) {    \
        return atomic##Opname2(dest, value);                            \
    }                                                                   \
                                                                        \
    template <typename T1, typename T2>                                 \
    __device__ __forceinline__ T1 Atomic##Opname1(T1 *dest, T1 value) { \
        T2 temp = atomic##Opname2((T2 *)dest, *((T2 *)&value));         \
        return *((T1 *)&temp);                                          \
    }

FAND_FOR_FXOR_FADD_SWAP_DEVICE_FUNC(Fand, And)
FAND_FOR_FXOR_FADD_SWAP_DEVICE_FUNC(For, Or)
FAND_FOR_FXOR_FADD_SWAP_DEVICE_FUNC(Fxor, Xor)
FAND_FOR_FXOR_FADD_SWAP_DEVICE_FUNC(Fadd, Add)
FAND_FOR_FXOR_FADD_SWAP_DEVICE_FUNC(Swap, Exch)

template <typename T>
__device__ __forceinline__ T AtomicCswap(T *dest, T cond, T value) {
    return atomicCAS(dest, cond, value);
}

template <typename T1, typename T2>
__device__ __forceinline__ T1 AtomicCswap(T1 *dest, T1 cond, T1 value) {
    T2 temp = (T1)atomicCAS((T2 *)dest, (T2)cond, *((T2 *)&value));
    return *((T1 *)&temp);
}
