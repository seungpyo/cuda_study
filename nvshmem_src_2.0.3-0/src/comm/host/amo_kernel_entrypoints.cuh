/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "amo_kernels.cuh"

/*inc*/
template <typename T>
__global__ void AtomicIncKernel(T *dest, T *ret, T val, T cmp) {
    AtomicInc<T>(dest);
}

template <typename T1, typename T2>
__global__ void AtomicIncKernel(T1 *dest, T1 *ret, T1 val, T1 cmp) {
    AtomicInc<T1, T2>(dest);
}

/*finc, fetch*/
#define FINC_FETCH_GLOBAL_FUNC(Opname)                                            \
    template <typename T>                                                         \
    __global__ void Atomic##Opname##Kernel(T *dest, T *ret, T val, T cmp) {       \
        *ret = Atomic##Opname<T>(dest);                                           \
        /*printf("[After fetch/finc] %lx\n", *ret);*/                             \
    }                                                                             \
                                                                                  \
    /*template __global__ void AtomicFincKernel <int> (int *dest, int *retval);*/ \
                                                                                  \
    template <typename T1, typename T2>                                           \
    __global__ void Atomic##Opname##Kernel(T1 *dest, T1 *ret, T1 val, T1 cmp) {   \
        *ret = Atomic##Opname<T1, T2>(dest);                                      \
        /*printf("[After fetch/finc using other type] %lx\n", *ret);*/            \
    }

FINC_FETCH_GLOBAL_FUNC(Finc)
FINC_FETCH_GLOBAL_FUNC(Fetch)

/*and, or, xor, add, set*/
#define AND_OR_XOR_ADD_SET_GLOBAL_FUNC(Opname)                                  \
    template <typename T>                                                       \
    __global__ void Atomic##Opname##Kernel(T *dest, T *ret, T val, T cmp) {     \
        Atomic##Opname<T>(dest, val);                                           \
    }                                                                           \
                                                                                \
    template <typename T1, typename T2>                                         \
    __global__ void Atomic##Opname##Kernel(T1 *dest, T1 *ret, T1 val, T1 cmp) { \
        Atomic##Opname<T1, T2>(dest, val);                                      \
    }

AND_OR_XOR_ADD_SET_GLOBAL_FUNC(And)
AND_OR_XOR_ADD_SET_GLOBAL_FUNC(Or)
AND_OR_XOR_ADD_SET_GLOBAL_FUNC(Xor)
AND_OR_XOR_ADD_SET_GLOBAL_FUNC(Add)
AND_OR_XOR_ADD_SET_GLOBAL_FUNC(Set)

/*fand, for, fxor, fadd, swap*/
#define FAND_FOR_FXOR_FADD_SWAP_GLOBAL_FUNC(Opname)                             \
    template <typename T>                                                       \
    __global__ void Atomic##Opname##Kernel(T *dest, T *ret, T val, T cmp) {     \
        *ret = Atomic##Opname<T>(dest, val);                                    \
    }                                                                           \
                                                                                \
    template <typename T1, typename T2>                                         \
    __global__ void Atomic##Opname##Kernel(T1 *dest, T1 *ret, T1 val, T1 cmp) { \
        *ret = Atomic##Opname<T1, T2>(dest, val);                               \
        /*printf("[After swap/fadd/for/... using other type] %f\n", *retval);*/ \
    }

FAND_FOR_FXOR_FADD_SWAP_GLOBAL_FUNC(Fand)
FAND_FOR_FXOR_FADD_SWAP_GLOBAL_FUNC(For)
FAND_FOR_FXOR_FADD_SWAP_GLOBAL_FUNC(Fxor)
FAND_FOR_FXOR_FADD_SWAP_GLOBAL_FUNC(Fadd)
FAND_FOR_FXOR_FADD_SWAP_GLOBAL_FUNC(Swap)

/*cswap*/
template <typename T>
__global__ void AtomicCswapKernel(T *dest, T *ret, T val, T cond) {
    *ret = AtomicCswap<T>(dest, cond, val);
}

template <typename T1, typename T2>
__global__ void AtomicCswapKernel(T1 *dest, T1 *ret, T1 val, T1 cond) {
    *ret = AtomicCswap<T1, T2>(dest, cond, val);
}
