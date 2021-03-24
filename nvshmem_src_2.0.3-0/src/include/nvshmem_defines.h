/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */

#ifndef _NVSHMEM_DEFINES_H_
#define _NVSHMEM_DEFINES_H_

#include "nvshmem_common.cuh"
#include "nvshmemi_util.h"
#include "nvshmemi_constants.h"
#include "nvshmemx_api.h"

template <typename T>
__device__ void nvshmemi_proxy_rma_p(void *rptr, T value, int pe);
template <nvshmemi_op_t op>
__device__ void nvshmemi_proxy_rma_nbi(void *rptr, void *lptr, size_t nelems, int pe);
template <typename T>
__device__ void nvshmemi_proxy_amo_nonfetch(void *rptr, const T value, int pe,
               nvshmemi_amo_t op);
template <typename T>
__device__ void nvshmemi_proxy_amo_fetch(void *rptr, void *lptr, T value, T compare, int pe,
               nvshmemi_amo_t op);
template<typename T>
__device__ T nvshmemi_proxy_rma_g(void *source, int pe);
__device__ void nvshmemi_proxy_fence();
__device__ void nvshmemi_proxy_quiet();
__device__ void nvshmemi_proxy_quiet_no_membar();
__device__ void nvshmemi_proxy_enforce_consistency_at_target();
__device__ void nvshmemi_proxy_enforce_consistency_at_target_no_membar();

#ifdef __CUDA_ARCH__
template <typename T>
__device__ inline void p(T *dest, const T value, int pe) {
    const void *peer_base_addr =
        (void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + pe);
    if (peer_base_addr) {
        T *dest_actual = (T *)((char *)(peer_base_addr) +
                               ((char *)dest - (char *)(nvshmemi_heap_base_d)));
        *dest_actual = value;
    } else {
        nvshmemi_proxy_rma_p<T>((void *)dest, value, pe);
    }
}

template <typename T>
__device__ inline T g(const T *source, int pe) {
    const void *peer_base_addr =
        (void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + pe);
    if (peer_base_addr) {
        T *source_actual =
            (T *)((char *)(peer_base_addr) +
                  ((char *)source - (char *)(nvshmemi_heap_base_d)));
        return *source_actual;
    } else {
        return nvshmemi_proxy_rma_g<T>((void*)source, pe);
    }
}

template <typename T>
__device__ inline void put(T *dest, const T *source, size_t nelems, int pe) {
    void *peer_base_addr = (void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + pe);
    if (peer_base_addr) {
        T *dest_actual = (T *)((char *)(peer_base_addr) +
                               ((char *)dest - (char *)(nvshmemi_heap_base_d)));
        for (size_t i = 0; i < nelems; i++) {
            *((T *)dest_actual + i) = *((T *)source + i);
        }
    } else {
        nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_PUT>((void *)dest, (void *)source, nelems * sizeof(T), pe);
        nvshmemi_proxy_quiet();
    }
}

template <typename T>
__device__ inline void put_signal(T *dest, const T *source, size_t nelems,
                                  uint64_t *sig_addr, uint64_t signal, int sig_op, int pe, bool is_nbi) {
    void *peer_base_addr = (void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + pe);
    if (peer_base_addr) {
        put<T>(dest, source, nelems, pe);
        __threadfence_system();
        nvshmemx_signal_op(sig_addr, signal, sig_op, pe);
        
    } else {
        nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_PUT>((void *)dest, (void *)source, nelems * sizeof(T), pe);
        nvshmemi_proxy_amo_nonfetch<uint64_t>((void *)sig_addr, signal, pe, (nvshmemi_amo_t)sig_op);
        if (is_nbi == 0)
            nvshmemi_proxy_quiet();
    }
}

template <typename T>
__device__ inline void get(T *dest, const T *source, size_t nelems, int pe) {
    void *peer_base_addr = (void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + pe);
    if (peer_base_addr) {
        T *source_actual =
            (T *)((char *)(peer_base_addr) +
                  ((char *)source - (char *)(nvshmemi_heap_base_d)));
        for (size_t i = 0; i < nelems; i++) {
            *((T *)dest + i) = *((T *)source_actual + i);
        }
    } else {
        nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_GET>((void *)source, (void *)dest, nelems * sizeof(T), pe);
        nvshmemi_proxy_quiet();
    }
}

template <typename T>
__device__ inline void put_nbi(T *dest, const T *source, size_t nelems, int pe) {
    void *peer_base_addr = (void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + pe);
    if (peer_base_addr) {
        T *dest_actual = (T *)((char *)(peer_base_addr) +
                               ((char *)dest - (char *)(nvshmemi_heap_base_d)));
        for (size_t i = 0; i < nelems; i++) {
            *((T *)dest_actual + i) = *((T *)source + i);
        }
    } else {
        nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_PUT>((void *)dest, (void *)source, nelems * sizeof(T), pe);
    }
}

template <typename T>
__device__ inline void get_nbi(T *dest, const T *source, size_t nelems, int pe) {
    void *peer_base_addr = (void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + pe);
    if (peer_base_addr) {
        T *source_actual =
            (T *)((char *)(peer_base_addr) +
                  ((char *)source - (char *)(nvshmemi_heap_base_d)));
        for (size_t i = 0; i < nelems; i++) {
            *((T *)dest + i) = *((T *)source_actual + i);
        }
    } else {
        nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_GET>((void *)source, (void *)dest, nelems * sizeof(T), pe);
    }
}

__device__ inline uint64_t nvshmem_signal_fetch(uint64_t *sig_addr) {
    return *((volatile uint64_t*) sig_addr);
}

#endif /* __CUDA__ARCH__ */

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __CUDA_ARCH__

/*__device__ nvshmem_p*/
#define NVSHMEMI_TYPENAME_P_IMPL(TYPENAME, TYPE)                                            \
    __device__ inline void nvshmem_##TYPENAME##_p(TYPE* dest, const TYPE value, int pe) {   \
        p<TYPE>(dest, value, pe);                                                           \
    }
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_TYPENAME_P_IMPL)
#undef NVSHMEMI_TYPENAME_P_IMPL

/*__device__ nvshmem_g*/
#define NVSHMEMI_TYPENAME_G_IMPL(TYPENAME, TYPE)                                    \
    __device__ inline TYPE nvshmem_##TYPENAME##_g(const TYPE *source, int pe) {    \
        return g<TYPE>(source, pe);                                                 \
    }
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_TYPENAME_G_IMPL)
#undef NVSHMEMI_TYPENAME_G_IMPL

/*__device__ nvshmem_<typename>_put*/
#define NVSHMEMI_TYPENAME_PUT_IMPL(TYPENAME, TYPE)                                                          \
    __device__ inline void nvshmem_##TYPENAME##_put(TYPE *dest, const TYPE *source, size_t nelems, int pe) {\
        put<TYPE>(dest, source, nelems, pe);                                                                \
    }
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_TYPENAME_PUT_IMPL)
#undef NVSHMEMI_TYPENAME_PUT_IMPL

/*__device__ nvshmem_<typename>_put_signal*/
#define NVSHMEMI_TYPENAME_PUT_SIGNAL_IMPL(TYPENAME, TYPE)                                                            \
    __device__ inline void nvshmem_##TYPENAME##_put_signal(TYPE *dest, const TYPE *source, size_t nelems,            \
                                                           uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {\
        put_signal<TYPE>(dest, source, nelems, sig_addr, signal, sig_op, pe, 0);                                     \
    }
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_TYPENAME_PUT_SIGNAL_IMPL)
#undef NVSHMEMI_TYPENAME_PUT_SIGNAL_IMPL

/*__device__ nvshmem_<typename>_get*/
#define NVSHMEMI_TYPENAME_GET_IMPL(TYPENAME, TYPE)                                                          \
    __device__ inline void nvshmem_##TYPENAME##_get(TYPE *dest, const TYPE *source, size_t nelems, int pe) {\
        get<TYPE>(dest, source, nelems, pe);                                                                \
    }
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_TYPENAME_GET_IMPL)
#undef NVSHMEMI_TYPENAME_GET_IMPL

/*__device__ nvshmem_put<bits>*/
__device__ inline void nvshmem_put8(void *dest, const void *source, size_t nelems, int pe) {
    put<int8_t>((int8_t *)dest, (const int8_t *)source, nelems, pe);
}
__device__ inline void nvshmem_put16(void *dest, const void *source, size_t nelems, int pe) {
    put<int16_t>((int16_t *)dest, (const int16_t *)source, nelems, pe);
}
__device__ inline void nvshmem_put32(void *dest, const void *source, size_t nelems, int pe) {
    put<int32_t>((int32_t *)dest, (const int32_t *)source, nelems, pe);
}
__device__ inline void nvshmem_put64(void *dest, const void *source, size_t nelems, int pe) {
    put<int64_t>((int64_t *)dest, (const int64_t *)source, nelems, pe);
}
__device__ inline void nvshmem_put128(void *dest, const void *source, size_t nelems, int pe) {
    put<int4>((int4 *)dest, (const int4 *)source, nelems, pe);
}

/*__device__ nvshmem_put<bits>_signal*/
#define NVSHMEMI_SIZE_PUT_SIGNAL_IMPL(BITS)                                                            \
    __device__ inline void nvshmem_put##BITS##_signal(void *dest, const void *source, size_t nelems,   \
                                                      uint64_t *sig_addr, uint64_t signal, int sig_op, \
                                                      int pe) {                                        \
        nvshmem_putmem_signal(dest, source, nelems*(BITS/8), sig_addr, signal, sig_op, pe);            \
    }
NVSHMEMI_REPT_FOR_SIZES(NVSHMEMI_SIZE_PUT_SIGNAL_IMPL)

/*__device__ nvshmem_get<bits>*/
__device__ inline void nvshmem_get8(void *dest, const void *source, size_t nelems, int pe) {
    get<int8_t>((int8_t *)dest, (const int8_t *)source, nelems, pe);
}
__device__ inline void nvshmem_get16(void *dest, const void *source, size_t nelems, int pe) {
    get<int16_t>((int16_t *)dest, (const int16_t *)source, nelems, pe);
}
__device__ inline void nvshmem_get32(void *dest, const void *source, size_t nelems, int pe) {
    get<int32_t>((int32_t *)dest, (const int32_t *)source, nelems, pe);
}
__device__ inline void nvshmem_get64(void *dest, const void *source, size_t nelems, int pe) {
    get<int64_t>((int64_t *)dest, (const int64_t *)source, nelems, pe);
}
__device__ inline void nvshmem_get128(void *dest, const void *source, size_t nelems, int pe) {
    get<int4>((int4 *)dest, (const int4 *)source, nelems, pe);
}

/*__device__ nvshmem_putmem*/
__device__ inline void nvshmem_putmem(void *dest, const void *source, size_t bytes, int pe) {
    put<char>((char *)dest, (const char *)source, bytes, pe);
}

/*__device__ nvshmem_putmem_signal*/
__device__ inline void nvshmem_putmem_signal(void *dest, const void *source, size_t bytes,
                                             uint64_t *sig_addr, uint64_t signal, int sig_op,
                                             int pe) {
    put_signal<char>((char *)dest, (const char *)source, bytes, sig_addr, signal, sig_op, pe, 0);
}

/*__device__ nvshmem_getmem*/
__device__ inline void nvshmem_getmem(void *dest, const void *source, size_t bytes, int pe) {
    get<char>((char *)dest, (const char *)source, bytes, pe);
}

/*__device__ nvshmem_<typename>_put_nbi*/
#define NVSHMEMI_TYPENAME_PUT_NBI_IMPL(TYPENAME, TYPE)                                                          \
    __device__ inline void nvshmem_##TYPENAME##_put_nbi(TYPE *dest, const TYPE *source, size_t nelems, int pe) {\
        put_nbi<TYPE>(dest, source, nelems, pe);                                                                \
    }
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_TYPENAME_PUT_NBI_IMPL)
#undef NVSHMEMI_TYPENAME_PUT_NBI_IMPL

/*__device__ nvshmem_<typename>_put_signal_nbi*/
#define NVSHMEMI_TYPENAME_PUT_SIGNAL_NBI_IMPL(TYPENAME, TYPE)                                                        \
    __device__ inline void nvshmem_##TYPENAME##_put_signal_nbi(TYPE *dest, const TYPE *source, size_t nelems,        \
                                                           uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {\
        put_signal<TYPE>(dest, source, nelems, sig_addr, signal, sig_op, pe, 1);                                     \
    }
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_TYPENAME_PUT_SIGNAL_NBI_IMPL)
#undef NVSHMEMI_TYPENAME_PUT_SIGNAL_NBI_IMPL

/*__device__ nvshmem_<typename>_get_nbi*/
#define NVSHMEMI_TYPENAME_GET_NBI_IMPL(TYPENAME, TYPE)                                                          \
    __device__ inline void nvshmem_##TYPENAME##_get_nbi(TYPE *dest, const TYPE *source, size_t nelems, int pe) {\
        get_nbi<TYPE>(dest, source, nelems, pe);                                                                \
    }
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_TYPENAME_GET_NBI_IMPL)
#undef NVSHMEMI_TYPENAME_GET_NBI_IMPL

/*__device__ nvshmem_put<bits>_nbi*/
__device__ inline void nvshmem_put8_nbi(void *dest, const void *source, size_t nelems, int pe) {
    put_nbi<int8_t>((int8_t *)dest, (const int8_t *)source, nelems, pe);
}
__device__ inline void nvshmem_put16_nbi(void *dest, const void *source, size_t nelems, int pe) {
    put_nbi<int16_t>((int16_t *)dest, (const int16_t *)source, nelems, pe);
}
__device__ inline void nvshmem_put32_nbi(void *dest, const void *source, size_t nelems, int pe) {
    put_nbi<int32_t>((int32_t *)dest, (const int32_t *)source, nelems, pe);
}
__device__ inline void nvshmem_put64_nbi(void *dest, const void *source, size_t nelems, int pe) {
    put_nbi<int64_t>((int64_t *)dest, (const int64_t *)source, nelems, pe);
}
__device__ inline void nvshmem_put128_nbi(void *dest, const void *source, size_t nelems, int pe) {
    put_nbi<int4>((int4 *)dest, (const int4 *)source, nelems, pe);
}
/*__device__ nvshmem_get<bits>_nbi*/
__device__ inline void nvshmem_get8_nbi(void *dest, const void *source, size_t nelems, int pe) {
    get_nbi<int8_t>((int8_t *)dest, (const int8_t *)source, nelems, pe);
}
__device__ inline void nvshmem_get16_nbi(void *dest, const void *source, size_t nelems, int pe) {
    get_nbi<int16_t>((int16_t *)dest, (const int16_t *)source, nelems, pe);
}
__device__ inline void nvshmem_get32_nbi(void *dest, const void *source, size_t nelems, int pe) {
    get_nbi<int32_t>((int32_t *)dest, (const int32_t *)source, nelems, pe);
}
__device__ inline void nvshmem_get64_nbi(void *dest, const void *source, size_t nelems, int pe) {
    get_nbi<int64_t>((int64_t *)dest, (const int64_t *)source, nelems, pe);
}
__device__ inline void nvshmem_get128_nbi(void *dest, const void *source, size_t nelems, int pe) {
    get_nbi<int4>((int4 *)dest, (const int4 *)source, nelems, pe);
}
/*__device__ nvshmem_putmem_nbi*/
__device__ inline void nvshmem_putmem_nbi(void *dest, const void *source, size_t bytes, int pe) {
    put_nbi<char>((char *)dest, (const char *)source, bytes, pe);
}
/*__device__ nvshmem_getmem_nbi*/
__device__ inline void nvshmem_getmem_nbi(void *dest, const void *source, size_t bytes, int pe) {
    get_nbi<char>((char *)dest, (const char *)source, bytes, pe);
}

/**** TEST API ****/
#define NVSHMEM_TEST(Name, Type)                                                \
    __device__ inline int nvshmem_##Name##_test(Type *ivar, int cmp, Type cmp_value) { \
        int return_value = nvshmemi_test<Type>(ivar, cmp, cmp_value);                    \
        if (return_value == 1)                                                  \
            nvshmemi_syncapi_update_mem();                                      \
        return return_value;                                                    \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEM_TEST)
#undef NVSHMEM_TEST

#define NVSHMEM_TEST_ALL(Name, Type)                                                        \
    __device__ inline int nvshmem_##Name##_test_all(Type *ivars, size_t nelems, const int *status, \
                                             int cmp, Type cmp_value) {                     \
        bool test_set_is_empty = true;                                                      \
        for (size_t i = 0; i < nelems; i++) {                                               \
            if (!status || status[i] == 0) {                                                \
                if (nvshmemi_test<Type>(&ivars[i], cmp, cmp_value) == 0) return 0;          \
                test_set_is_empty = false;                                                  \
            }                                                                               \
        }                                                                                   \
        if (test_set_is_empty == false)                                                     \
            nvshmemi_syncapi_update_mem();                                                  \
                                                                                            \
        return 1;                                                                           \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEM_TEST_ALL)
#undef NVSHMEM_TEST_ALL

#define NVSHMEM_TEST_ANY(Name, Type)                                                           \
    __device__ inline size_t nvshmem_##Name##_test_any(Type *ivars, size_t nelems, const int *status, \
                                                int cmp, Type cmp_value) {                     \
        unsigned long long start_idx = atomicAdd(&test_wait_any_start_idx_d, 1);               \
        for (size_t i = 0; i < nelems; i++) {                                                  \
            size_t idx = (i + (size_t)start_idx) % nelems;                                     \
            if (!status || status[idx] == 0) {                                                 \
                if (nvshmemi_test<Type>(&ivars[idx], cmp, cmp_value) == 1) {                   \
                    nvshmemi_syncapi_update_mem();                                             \
                    return idx;                                                                \
                }                                                                              \
            }                                                                                  \
        }                                                                                      \
                                                                                               \
        return SIZE_MAX;                                                                       \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEM_TEST_ANY)
#undef NVSHMEM_TEST_ANY

#define NVSHMEM_TEST_SOME(Name, Type)                                                                 \
    __device__ inline size_t nvshmem_##Name##_test_some(Type *ivars, size_t nelems, size_t *indices,  \
                                                 const int *status, int cmp, Type cmp_value) {        \
        size_t num_satisfied = 0;                                                                     \
        for (size_t i = 0; i < nelems; i++) {                                                         \
            if (!status || status[i] == 0) {                                                          \
                if (nvshmemi_test<Type>(&ivars[i], cmp, cmp_value) == 1) {                            \
                    indices[num_satisfied++] = i;                                                     \
                }                                                                                     \
            }                                                                                         \
        }                                                                                             \
                                                                                                      \
        if (num_satisfied > 0)                                                                        \
            nvshmemi_syncapi_update_mem();                                                            \
                                                                                                      \
        return num_satisfied;                                                                         \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEM_TEST_SOME)
#undef NVSHMEM_TEST_SOME


#define NVSHMEM_TEST_ALL_VECTOR(Name, Type)                                                        \
    __device__ inline int nvshmem_##Name##_test_all_vector(Type *ivars, size_t nelems, const int *status, \
                                             int cmp, Type *cmp_value) {                     \
        bool test_set_is_empty = true;                                                      \
        for (size_t i = 0; i < nelems; i++) {                                               \
            if (!status || status[i] == 0) {                                                \
                if (nvshmemi_test<Type>(&ivars[i], cmp, cmp_value[i]) == 0) return 0;          \
                test_set_is_empty = false;                                                  \
            }                                                                               \
        }                                                                                   \
        if (test_set_is_empty == false)                                                     \
            nvshmemi_syncapi_update_mem();                                                  \
                                                                                            \
        return 1;                                                                           \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEM_TEST_ALL_VECTOR)
#undef NVSHMEM_TEST_ALL_VECTOR

#define NVSHMEM_TEST_ANY_VECTOR(Name, Type)                                                           \
    __device__ inline size_t nvshmem_##Name##_test_any_vector(Type *ivars, size_t nelems, const int *status, \
                                                int cmp, Type *cmp_value) {                     \
        unsigned long long start_idx = atomicAdd(&test_wait_any_start_idx_d, 1);               \
        for (size_t i = 0; i < nelems; i++) {                                                  \
            size_t idx = (i + (size_t)start_idx) % nelems;                                     \
            if (!status || status[idx] == 0) {                                                 \
                if (nvshmemi_test<Type>(&ivars[idx], cmp, cmp_value[idx]) == 1) {                   \
                    nvshmemi_syncapi_update_mem();                                             \
                    return idx;                                                                \
                }                                                                              \
            }                                                                                  \
        }                                                                                      \
                                                                                               \
        return SIZE_MAX;                                                                       \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEM_TEST_ANY_VECTOR)
#undef NVSHMEM_TEST_ANY_VECTOR

#define NVSHMEM_TEST_SOME_VECTOR(Name, Type)                                                                 \
    __device__ inline size_t nvshmem_##Name##_test_some_vector(Type *ivars, size_t nelems, size_t *indices,  \
                                                 const int *status, int cmp, Type *cmp_value) {        \
        size_t num_satisfied = 0;                                                                     \
        for (size_t i = 0; i < nelems; i++) {                                                         \
            if (!status || status[i] == 0) {                                                          \
                if (nvshmemi_test<Type>(&ivars[i], cmp, cmp_value[i]) == 1) {                            \
                    indices[num_satisfied++] = i;                                                     \
                }                                                                                     \
            }                                                                                         \
        }                                                                                             \
                                                                                                      \
        if (num_satisfied > 0)                                                                        \
            nvshmemi_syncapi_update_mem();                                                            \
                                                                                                      \
        return num_satisfied;                                                                         \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEM_TEST_SOME_VECTOR)
#undef NVSHMEM_TEST_SOME_VECTOR

/**** WAIT API ****/
#define NVSHMEM_WAIT_UNTIL(Name, Type)                                                        \
    __device__ inline void nvshmem_##Name##_wait_until(Type *ivar, int cmp, Type cmp_value) { \
        nvshmemi_wait_until<Type>(ivar, cmp, cmp_value);                                      \
                                                                                              \
        nvshmemi_syncapi_update_mem();                                                        \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEM_WAIT_UNTIL)
#undef NVSHMEM_WAIT_UNTIL

__device__ inline void nvshmem_wait_until(long *ivar, int cmp, long cmp_value) {
    nvshmem_long_wait_until(ivar, cmp, cmp_value);
}

__device__ inline uint64_t nvshmem_signal_wait_until(uint64_t *sig_addr, int cmp, uint64_t cmp_val) {
    return nvshmemi_signal_wait_until(sig_addr, cmp, cmp_val);
}

#define NVSHMEM_WAIT(Name, Type)                                                \
    __device__ inline void nvshmem_##Name##_wait(Type *ivar, Type cmp_value) {  \
        nvshmemi_wait_until_not_equals<Type>(ivar, cmp_value, NVSHMEMI_CALL_SITE_WAIT_NE);        \
                                                                                \
        nvshmemi_syncapi_update_mem();                                          \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEM_WAIT)
#undef NVSHMEM_WAIT

__device__ inline void nvshmem_wait(long *ivar, long cmp_value) { nvshmem_long_wait(ivar, cmp_value); }

#define NVSHMEM_WAIT_UNTIL_ALL(Name, Type)                                                                \
    __device__ inline void nvshmem_##Name##_wait_until_all(Type *ivars, size_t nelems, const int *status, \
                                                    int cmp, Type cmp_value) {                            \
        bool waited = false;                                                                              \
        for (size_t i = 0; i < nelems; i++) {                                                             \
            if (!status || status[i] == 0) {                                                              \
                waited = true;                                                                            \
                nvshmemi_wait_until<Type>(&ivars[i], cmp, cmp_value);                                     \
            }                                                                                             \
        }                                                                                                 \
                                                                                                          \
        if (waited)                                                                                       \
            nvshmemi_syncapi_update_mem();                                                                \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEM_WAIT_UNTIL_ALL)
#undef NVSHMEM_WAIT_UNTIL_ALL

#define NVSHMEM_WAIT_UNTIL_ANY(Name, Type)                                        \
    __device__ inline size_t nvshmem_##Name##_wait_until_any(                     \
        Type *ivars, size_t nelems, const int *status, int cmp, Type cmp_value) { \
        bool wait_set_is_empty = true;                                            \
        size_t idx;                                                               \
        if (nelems == 0) return SIZE_MAX;                                         \
        unsigned long long start_idx = atomicAdd(&test_wait_any_start_idx_d, 1);  \
                                                                                  \
        for (size_t i = 0;; i++) {                                                \
            idx = (i + (size_t)start_idx) % nelems;                               \
            if (!status || status[idx] == 0) {                                    \
                wait_set_is_empty = false;                                        \
                if (nvshmemi_test<Type>(&ivars[idx], cmp, cmp_value)) break;      \
            } else if (i >= nelems && wait_set_is_empty)                          \
                break;                                                            \
        }                                                                         \
                                                                                  \
        if (wait_set_is_empty == false)                                           \
            nvshmemi_syncapi_update_mem();                                        \
                                                                                  \
        return wait_set_is_empty ? SIZE_MAX : idx;                                \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEM_WAIT_UNTIL_ANY)
#undef NVSHMEM_WAIT_UNTIL_ANY

#define NVSHMEM_WAIT_UNTIL_SOME(Name, Type)                                                        \
    __device__ inline size_t nvshmem_##Name##_wait_until_some(                                     \
        Type *ivars, size_t nelems, size_t *indices, const int *status, int cmp, Type cmp_value) { \
        size_t i;                                                                                  \
        int num_satisfied = 0;                                                                     \
        bool wait_set_is_empty = true;                                                             \
        for (i = 0; i < nelems; i++) {                                                             \
            if (!status || status[i] == 0) {                                                       \
                wait_set_is_empty = false;                                                         \
                if (nvshmem_##Name##_test(&ivars[i], cmp, cmp_value) == 1)                         \
                    indices[num_satisfied++] = i;                                                  \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        if (wait_set_is_empty == false && num_satisfied == 0) { /* do wait_any*/                   \
            indices[num_satisfied++] =                                                             \
                nvshmem_##Name##_wait_until_any(ivars, nelems, status, cmp, cmp_value);            \
        }                                                                                          \
                                                                                                   \
        if (num_satisfied > 0)                                                                     \
            nvshmemi_syncapi_update_mem();                                                         \
                                                                                                   \
        return num_satisfied;                                                                      \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEM_WAIT_UNTIL_SOME)
#undef NVSHMEM_WAIT_UNTIL_SOME


#define NVSHMEM_WAIT_UNTIL_ALL_VECTOR(Name, Type)                                                                \
    __device__ inline void nvshmem_##Name##_wait_until_all_vector(Type *ivars, size_t nelems, const int *status, \
                                                    int cmp, Type *cmp_value) {                            \
        bool waited = false;                                                                              \
        for (size_t i = 0; i < nelems; i++) {                                                             \
            if (!status || status[i] == 0) {                                                              \
                waited = true;                                                                            \
                nvshmemi_wait_until<Type>(&ivars[i], cmp, cmp_value[i]);                                  \
            }                                                                                             \
        }                                                                                                 \
                                                                                                          \
        if (waited)                                                                                       \
            nvshmemi_syncapi_update_mem();                                                                \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEM_WAIT_UNTIL_ALL_VECTOR)
#undef NVSHMEM_WAIT_UNTIL_ALL_VECTOR

#define NVSHMEM_WAIT_UNTIL_ANY_VECTOR(Name, Type)                                 \
    __device__ inline size_t nvshmem_##Name##_wait_until_any_vector(                     \
        Type *ivars, size_t nelems, const int *status, int cmp, Type *cmp_value) { \
        bool wait_set_is_empty = true;                                            \
        size_t idx;                                                               \
        if (nelems == 0) return SIZE_MAX;                                         \
        unsigned long long start_idx = atomicAdd(&test_wait_any_start_idx_d, 1);  \
                                                                                  \
        for (size_t i = 0;; i++) {                                                \
            idx = (i + (size_t)start_idx) % nelems;                               \
            if (!status || status[idx] == 0) {                                    \
                wait_set_is_empty = false;                                        \
                if (nvshmemi_test<Type>(&ivars[idx], cmp, cmp_value[idx])) break; \
            } else if (i >= nelems && wait_set_is_empty)                          \
                break;                                                            \
        }                                                                         \
                                                                                  \
        if (wait_set_is_empty == false)                                           \
            nvshmemi_syncapi_update_mem();                                        \
                                                                                  \
                                                                                  \
        return wait_set_is_empty ? SIZE_MAX : idx;                                \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEM_WAIT_UNTIL_ANY_VECTOR)
#undef NVSHMEM_WAIT_UNTIL_ANY_VECTOR

#define NVSHMEM_WAIT_UNTIL_SOME_VECTOR(Name, Type)                                                 \
    __device__ inline size_t nvshmem_##Name##_wait_until_some_vector(                                     \
        Type *ivars, size_t nelems, size_t *indices, const int *status, int cmp, Type *cmp_value) {\
        size_t i;                                                                                  \
        int num_satisfied = 0;                                                                     \
        bool wait_set_is_empty = true;                                                             \
        for (i = 0; i < nelems; i++) {                                                             \
            if (!status || status[i] == 0) {                                                       \
                wait_set_is_empty = false;                                                         \
                if (nvshmem_##Name##_test(&ivars[i], cmp, cmp_value[i]) == 1)                      \
                    indices[num_satisfied++] = i;                                                  \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        if (wait_set_is_empty == false && num_satisfied == 0) { /* do wait_any*/                   \
            indices[num_satisfied++] =                                                             \
                nvshmem_##Name##_wait_until_any_vector(ivars, nelems, status, cmp, cmp_value);     \
        }                                                                                          \
                                                                                                   \
        if (num_satisfied > 0)                                                                     \
            nvshmemi_syncapi_update_mem();                                                         \
                                                                                                   \
        return num_satisfied;                                                                      \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEM_WAIT_UNTIL_SOME_VECTOR)
#undef NVSHMEM_WAIT_UNTIL_SOME_VECTOR

/* nvshmem_quiet and nvshmem_fence API */
__device__ inline void nvshmem_quiet() {
    if (nvshmemi_proxy_d && (nvshmemi_job_connectivity_d > NVSHMEMI_JOB_GPU_LDST_ATOMICS)) { 
        nvshmemi_proxy_quiet();
    } else {
        __threadfence_system(); /* Use __threadfence_system instead of __threadfence
                                 for data visibility in case of intra-node GPU transfers */
    }
}

__device__ inline void nvshmem_fence() {
    if (nvshmemi_proxy_d && (nvshmemi_job_connectivity_d > NVSHMEMI_JOB_GPU_LDST_ATOMICS)
        && !nvshmemi_proxy_ops_are_ordered_d) { 
        nvshmemi_proxy_fence();
    }
    __threadfence_system(); /* Use __threadfence_system instead of __threadfence
                               for data visibility in case of intra-node GPU transfers */
}

#define NVSHMEM_TYPE_ATOMIC_FETCH_ADD(Name, Type)                                                              \
    __device__ inline Type nvshmem_##Name##_atomic_fetch_add(Type *target, Type value, int pe) {               \
        void *peer_base_addr =                                                                     \
            (void *)__ldg((const unsigned long long *)nvshmemi_peer_heap_base_d + pe);              \
       if (nvshmemi_job_connectivity_d <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {             \
            Type *target_actual =                                                                  \
                (Type *)((char *)peer_base_addr + ((char *)target - (char *)nvshmemi_heap_base_d)); \
                                                                                                   \
            return ((Type)atomicAdd(target_actual, value));                                        \
        } else {                                                                                   \
	    Type retval;									   \
            nvshmemi_proxy_amo_fetch<Type>((void *)target, (void *)&retval, (Type)value, 0, pe, NVSHMEMI_AMO_FETCH_ADD);      \
            return retval;                                                                         \
        }                                                                                          \
    }

#define NVSHMEM_TYPE_ATOMIC_FETCH_ADD_CAST(Name, Type, subType)                                                \
    __device__ inline Type nvshmem_##Name##_atomic_fetch_add(Type *target, Type value, int pe) {               \
        void *peer_base_addr =                                                                     \
            (void *)__ldg((const unsigned long long *)nvshmemi_peer_heap_base_d + pe);              \
       if (nvshmemi_job_connectivity_d <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {             \
            Type *target_actual =                                                                  \
                (Type *)((char *)peer_base_addr + ((char *)target - (char *)nvshmemi_heap_base_d)); \
                                                                                                   \
            return (Type)atomicAdd((subType *)target_actual, *((subType *)&value));                \
        } else {                                                                                   \
            Type retval;									   \
	    nvshmemi_proxy_amo_fetch<Type>((void *)target, (void *)&retval, (Type)value, 0, pe, NVSHMEMI_AMO_FETCH_ADD);      \
	    return retval;									   \
        }                                                                                          \
    }

NVSHMEM_TYPE_ATOMIC_FETCH_ADD(int, int)
NVSHMEM_TYPE_ATOMIC_FETCH_ADD_CAST(long, long, unsigned long long int)
NVSHMEM_TYPE_ATOMIC_FETCH_ADD(uint, unsigned int)
NVSHMEM_TYPE_ATOMIC_FETCH_ADD_CAST(ulong, unsigned long, unsigned int)
NVSHMEM_TYPE_ATOMIC_FETCH_ADD(ulonglong, unsigned long long int)
NVSHMEM_TYPE_ATOMIC_FETCH_ADD_CAST(int32, int32_t, int)
NVSHMEM_TYPE_ATOMIC_FETCH_ADD_CAST(uint32, uint32_t, unsigned int)
NVSHMEM_TYPE_ATOMIC_FETCH_ADD_CAST(uint64, uint64_t, unsigned long long int)
NVSHMEM_TYPE_ATOMIC_FETCH_ADD_CAST(size, size_t, unsigned long long int)
/*the following types are not implemented for FADD becuase of lack of CUDA support
 * ptrdiff_t
 * longlong
 * int64_t
 */
#undef NVSHMEM_TYPE_ATOMIC_FETCH_ADD
#undef NVSHMEM_TYPE_ATOMIC_FETCH_ADD_CAST	

#define NVSHMEM_TYPE_ATOMIC_ADD_EMULATE(Name, Type)                                 \
    __device__ inline void nvshmem_##Name##_atomic_add(Type *target, Type value, int pe) { \
       /*need a better check for case when to use only proxy-based atomics*/          \
       if (nvshmemi_job_connectivity_d <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {             \
            nvshmem_##Name##_atomic_fetch_add(target, value, pe);                      \
       } else {                                                                       \
            nvshmemi_proxy_amo_nonfetch<Type>((void *)target, value, pe, NVSHMEMI_AMO_ADD);   \
       }									      \
   }

NVSHMEM_TYPE_ATOMIC_ADD_EMULATE(int, int)
NVSHMEM_TYPE_ATOMIC_ADD_EMULATE(long, long)
NVSHMEM_TYPE_ATOMIC_ADD_EMULATE(uint, unsigned int)
NVSHMEM_TYPE_ATOMIC_ADD_EMULATE(ulong, unsigned long)
NVSHMEM_TYPE_ATOMIC_ADD_EMULATE(ulonglong, unsigned long long)
NVSHMEM_TYPE_ATOMIC_ADD_EMULATE(int32, int32_t)
NVSHMEM_TYPE_ATOMIC_ADD_EMULATE(uint32, uint32_t)
NVSHMEM_TYPE_ATOMIC_ADD_EMULATE(uint64, uint64_t)
NVSHMEM_TYPE_ATOMIC_ADD_EMULATE(size, size_t)
/*the following types are not implemented for ADD becuase of lack of CUDA support
 * ptrdiff_t
 * longlong
 * int64_t
 */

#undef NVSHMEM_TYPE_ATOMIC_ADD_EMULATE

#define NVSHMEM_TYPE_ATOMIC_FETCH_INC(Name, Type)                                                         \
    __device__ inline Type nvshmem_##Name##_atomic_fetch_inc(Type *target, int pe) {                           \
        void *peer_base_addr =                                                                     \
            (void *)__ldg((const unsigned long long *)nvshmemi_peer_heap_base_d + pe);              \
        if (nvshmemi_job_connectivity_d <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {             \
            Type *target_actual =                                                                  \
                (Type *)((char *)peer_base_addr + ((char *)target - (char *)nvshmemi_heap_base_d)); \
                                                                                                   \
            return atomicInc(target_actual, UINT_MAX);                                             \
        } else {                                                                                   \
            Type retval;									   \
	    nvshmemi_proxy_amo_fetch<Type>((void *)target, (void *)&retval, (Type)1, 0, pe, NVSHMEMI_AMO_FETCH_INC);      \
	    return retval;									   \
        }                                                                                          \
    }

#define NVSHMEM_TYPE_ATOMIC_FETCH_INC_CAST(Name, Type, subType)                                           \
    __device__ inline Type nvshmem_##Name##_atomic_fetch_inc(Type *target, int pe) {                           \
        void *peer_base_addr =                                                                     \
            (void *)__ldg((const unsigned long long *)nvshmemi_peer_heap_base_d + pe);              \
        if (nvshmemi_job_connectivity_d <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {             \
            Type *target_actual =                                                                  \
                (Type *)((char *)peer_base_addr + ((char *)target - (char *)nvshmemi_heap_base_d)); \
                                                                                                   \
            return (Type)atomicInc((subType *)target_actual, UINT_MAX);                            \
        } else {                                                                                   \
            Type retval;									   \
	    nvshmemi_proxy_amo_fetch<Type>((void *)target, (void *)&retval, (Type)1, 0, pe, NVSHMEMI_AMO_FETCH_INC);      \
	    return retval;									   \
        }                                                                                          \
    }

#define NVSHMEM_TYPE_ATOMIC_FETCH_INC_EMULATE(Name, Type)                \
    __device__ inline Type nvshmem_##Name##_atomic_fetch_inc(Type *target, int pe) { \
        return nvshmem_##Name##_atomic_fetch_add(target, (Type)1, pe);        \
    }

NVSHMEM_TYPE_ATOMIC_FETCH_INC_EMULATE(int, int)
NVSHMEM_TYPE_ATOMIC_FETCH_INC_EMULATE(long, long)
NVSHMEM_TYPE_ATOMIC_FETCH_INC(uint, unsigned int)
NVSHMEM_TYPE_ATOMIC_FETCH_INC_EMULATE(ulong, unsigned long)
NVSHMEM_TYPE_ATOMIC_FETCH_INC_EMULATE(ulonglong, unsigned long long)
NVSHMEM_TYPE_ATOMIC_FETCH_INC_EMULATE(int32, int32_t)
NVSHMEM_TYPE_ATOMIC_FETCH_INC_CAST(uint32, uint32_t, unsigned int)
NVSHMEM_TYPE_ATOMIC_FETCH_INC_EMULATE(uint64, uint64_t)
NVSHMEM_TYPE_ATOMIC_FETCH_INC_EMULATE(size, size_t)
/*the following types are not implemented for INC becuase of lack of CUDA support
 * ptrdiff_t
 * longlong
 * int64_t
 */
	
#undef NVSHMEM_TYPE_ATOMIC_FETCH_INC
#undef NVSHMEM_TYPE_ATOMIC_FETCH_INC_CAST
#undef NVSHMEM_TYPE_ATOMIC_FETCH_INC_EMULATE

#define NVSHMEM_TYPE_ATOMIC_INC_EMULATE(Name, Type)                                                      \
    __device__ inline void nvshmem_##Name##_atomic_inc(Type *target, int pe) {                           \
       /*need a better check for case when to use only proxy-based atomcis*/                      \
       if (nvshmemi_job_connectivity_d <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {             \
            nvshmem_##Name##_atomic_fetch_inc(target, pe);                                                    \
       } else {                                                                       \
            nvshmemi_proxy_amo_nonfetch<Type>((void *)target, (Type)1, pe, NVSHMEMI_AMO_ADD);             \
       }                                                                                          \
     }

NVSHMEM_TYPE_ATOMIC_INC_EMULATE(int, int)
NVSHMEM_TYPE_ATOMIC_INC_EMULATE(long, long)
NVSHMEM_TYPE_ATOMIC_INC_EMULATE(uint, unsigned int)
NVSHMEM_TYPE_ATOMIC_INC_EMULATE(ulong, unsigned long)
NVSHMEM_TYPE_ATOMIC_INC_EMULATE(ulonglong, unsigned long long)
NVSHMEM_TYPE_ATOMIC_INC_EMULATE(int32, int32_t)
NVSHMEM_TYPE_ATOMIC_INC_EMULATE(uint32, uint32_t)
NVSHMEM_TYPE_ATOMIC_INC_EMULATE(uint64, uint64_t)
NVSHMEM_TYPE_ATOMIC_INC_EMULATE(size, size_t)
/*the following types are not implemented for INC becuase of lack of CUDA support
 * ptrdiff_t
 * longlong
 * int64_t
 */
#undef NVSHMEM_TYPE_ATOMIC_INC_EMULATE
#endif /* __CUDA_ARCH__ */

#ifdef __cplusplus
}
#endif
#endif /* _NVSHMEM_DEFINES_H_ */
