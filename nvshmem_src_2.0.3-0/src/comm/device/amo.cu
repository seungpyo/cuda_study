/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmem_internal.h"

#ifdef __CUDA_ARCH__

/* For reading peer_base_addr, we are casting argument of __ldg to
   unsigned long long * because __ldg does not take void ** as argument
*/

#define NVSHMEM_TYPE_COMPARE_SWAP(Name, Type)                                                             \
    __device__ Type nvshmem_##Name##_atomic_compare_swap(Type *target, Type compare, Type value, int pe) {       \
        void *peer_base_addr =                                                                     \
            (void *)__ldg((const unsigned long long *)nvshmemi_peer_heap_base_d + pe);             \
        if (nvshmemi_job_connectivity_d <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {             	   \
            Type *target_actual =                                                                  \
                (Type *)((char *)peer_base_addr + ((char *)target - (char *)nvshmemi_heap_base_d)); \
                                                                                                   \
            return (Type)atomicCAS(target_actual, compare, value);                                 \
        } else {                                                                                   \
            Type retval;                                                                           \
            nvshmemi_proxy_amo_fetch<Type>((void *)target, (void *)&retval, (Type)value, compare, pe, NVSHMEMI_AMO_COMPARE_SWAP);      \
            return retval;                                                                         \
        }                                                                                          \
    }

#define NVSHMEM_TYPE_COMPARE_SWAP_CAST(Name, Type, subType)                                               \
    __device__ Type nvshmem_##Name##_atomic_compare_swap(Type *target, Type compare, Type value, int pe) {       \
        void *peer_base_addr =                                                                     \
            (void *)__ldg((const unsigned long long *)nvshmemi_peer_heap_base_d + pe);              \
        if (nvshmemi_job_connectivity_d <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {                        \
            Type *target_actual =                                                                  \
                (Type *)((char *)peer_base_addr + ((char *)target - (char *)nvshmemi_heap_base_d)); \
                                                                                                   \
            return (Type)atomicCAS((subType *)target_actual, *((subType *)&compare),               \
                                   *((subType *)&value));                                          \
        } else {                                                                                   \
            Type retval;                                                                           \
            nvshmemi_proxy_amo_fetch<Type>((void *)target, (void *)&retval, (Type)value, compare, pe, NVSHMEMI_AMO_COMPARE_SWAP);      \
            return retval;                                                                         \
        }                                                                                          \
    }

NVSHMEM_TYPE_COMPARE_SWAP(int, int)
NVSHMEM_TYPE_COMPARE_SWAP_CAST(long, long, unsigned long long int)
NVSHMEM_TYPE_COMPARE_SWAP_CAST(longlong, long long, unsigned long long int)
NVSHMEM_TYPE_COMPARE_SWAP(uint, unsigned int)
NVSHMEM_TYPE_COMPARE_SWAP_CAST(ulong, unsigned long, unsigned long long int)
NVSHMEM_TYPE_COMPARE_SWAP(ulonglong, unsigned long long int)
NVSHMEM_TYPE_COMPARE_SWAP_CAST(int32, int32_t, int)
NVSHMEM_TYPE_COMPARE_SWAP_CAST(int64, int64_t, unsigned long long int)
NVSHMEM_TYPE_COMPARE_SWAP_CAST(uint32, uint32_t, unsigned int)
NVSHMEM_TYPE_COMPARE_SWAP_CAST(uint64, uint64_t, unsigned long long int)
NVSHMEM_TYPE_COMPARE_SWAP_CAST(size, size_t, unsigned long long int)
NVSHMEM_TYPE_COMPARE_SWAP_CAST(ptrdiff, ptrdiff_t, unsigned long long int)

#define NVSHMEM_TYPE_FETCH_AND(Name, Type)                                                         \
    __device__ Type nvshmem_##Name##_atomic_fetch_and(Type *target, Type value, int pe) {                 \
        void *peer_base_addr =                                                                     \
            (void *)__ldg((const unsigned long long *)nvshmemi_peer_heap_base_d + pe);              \
        if (nvshmemi_job_connectivity_d <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {                        \
            Type *target_actual =                                                                  \
                (Type *)((char *)peer_base_addr + ((char *)target - (char *)nvshmemi_heap_base_d)); \
                                                                                                   \
            return atomicAnd(target_actual, value);                                                \
        } else {                                                                                   \
            Type retval;                                                                           \
            nvshmemi_proxy_amo_fetch<Type>((void *)target, (void *)&retval, (Type)value, 0, pe, NVSHMEMI_AMO_FETCH_AND);      \
            return retval;                                                                         \
        }                                                                                          \
    }

#define NVSHMEM_TYPE_FETCH_AND_CAST(Name, Type, subType)                                           \
    __device__ Type nvshmem_##Name##_atomic_fetch_and(Type *target, Type value, int pe) {                 \
        void *peer_base_addr =                                                                     \
            (void *)__ldg((const unsigned long long *)nvshmemi_peer_heap_base_d + pe);              \
        if (nvshmemi_job_connectivity_d <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {                        \
            Type *target_actual =                                                                  \
                (Type *)((char *)peer_base_addr + ((char *)target - (char *)nvshmemi_heap_base_d)); \
                                                                                                   \
            return atomicAnd((subType *)target_actual, *((subType *)&value));                      \
        } else {                                                                                   \
            Type retval;                                                                           \
            nvshmemi_proxy_amo_fetch<Type>((void *)target, (void *)&retval, (Type)value, 0, pe, NVSHMEMI_AMO_FETCH_AND);      \
            return retval;                                                                         \
        }                                                                                          \
    }

NVSHMEM_TYPE_FETCH_AND(uint, unsigned int)
NVSHMEM_TYPE_FETCH_AND_CAST(ulong, unsigned long, unsigned long long int)
NVSHMEM_TYPE_FETCH_AND(ulonglong, unsigned long long)
NVSHMEM_TYPE_FETCH_AND_CAST(int32, int32_t, unsigned int)
NVSHMEM_TYPE_FETCH_AND_CAST(int64, int64_t, unsigned long long int)
NVSHMEM_TYPE_FETCH_AND_CAST(uint32, uint32_t, unsigned int)
NVSHMEM_TYPE_FETCH_AND_CAST(uint64, uint64_t, unsigned long long int)

#define NVSHMEM_TYPE_AND_EMULATE(Name, Type)                                 \
    __device__ void nvshmem_##Name##_atomic_and(Type *target, Type value, int pe) { \
       if (nvshmemi_job_connectivity_d <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {             \
    	    nvshmem_##Name##_atomic_fetch_and(target, (Type)value, pe);                     \
       } else {                                                                       \
            nvshmemi_proxy_amo_nonfetch<Type>((void *)target, value, pe, NVSHMEMI_AMO_AND);   \
       }										      \
    }

NVSHMEM_TYPE_AND_EMULATE(uint, unsigned int)
NVSHMEM_TYPE_AND_EMULATE(ulong, unsigned long)
NVSHMEM_TYPE_AND_EMULATE(ulonglong, unsigned long long)
NVSHMEM_TYPE_AND_EMULATE(int32, int32_t)
NVSHMEM_TYPE_AND_EMULATE(int64, int64_t)
NVSHMEM_TYPE_AND_EMULATE(uint32, uint32_t)
NVSHMEM_TYPE_AND_EMULATE(uint64, uint64_t)

#define NVSHMEM_TYPE_FETCH_OR(Name, Type)                                                          \
    __device__ Type nvshmem_##Name##_atomic_fetch_or(Type *target, Type value, int pe) {                  \
        void *peer_base_addr =                                                                     \
            (void *)__ldg((const unsigned long long *)nvshmemi_peer_heap_base_d + pe);              \
        if (nvshmemi_job_connectivity_d <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {                        \
            Type *target_actual =                                                                  \
                (Type *)((char *)peer_base_addr + ((char *)target - (char *)nvshmemi_heap_base_d)); \
                                                                                                   \
            return atomicOr(target_actual, value);                                                 \
        } else {                                                                                   \
            Type retval;                                                                           \
            nvshmemi_proxy_amo_fetch<Type>((void *)target, (void *)&retval, (Type)value, 0, pe, NVSHMEMI_AMO_FETCH_OR);      \
            return retval;                                                                         \
        }                                                                                          \
    }

#define NVSHMEM_TYPE_FETCH_OR_CAST(Name, Type, subType)                                            \
    __device__ Type nvshmem_##Name##_atomic_fetch_or(Type *target, Type value, int pe) {                  \
        void *peer_base_addr =                                                                     \
            (void *)__ldg((const unsigned long long *)nvshmemi_peer_heap_base_d + pe);              \
        if (nvshmemi_job_connectivity_d <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {                        \
            Type *target_actual =                                                                  \
                (Type *)((char *)peer_base_addr + ((char *)target - (char *)nvshmemi_heap_base_d)); \
                                                                                                   \
            return atomicOr((subType *)target_actual, *((subType *)&value));                       \
        } else {                                                                                   \
            Type retval;                                                                           \
            nvshmemi_proxy_amo_fetch<Type>((void *)target, (void *)&retval, (Type)value, 0, pe, NVSHMEMI_AMO_FETCH_OR);      \
            return retval;                                                                         \
        }                                                                                          \
    }

NVSHMEM_TYPE_FETCH_OR(uint, unsigned int)
NVSHMEM_TYPE_FETCH_OR_CAST(ulong, unsigned long, unsigned long long int)
NVSHMEM_TYPE_FETCH_OR(ulonglong, unsigned long long)
NVSHMEM_TYPE_FETCH_OR_CAST(int32, int32_t, unsigned int)
NVSHMEM_TYPE_FETCH_OR_CAST(int64, int64_t, unsigned long long int)
NVSHMEM_TYPE_FETCH_OR_CAST(uint32, uint32_t, unsigned int)
NVSHMEM_TYPE_FETCH_OR_CAST(uint64, uint64_t, unsigned long long int)

#define NVSHMEM_TYPE_OR_EMULATE(Name, Type)                                 \
    __device__ void nvshmem_##Name##_atomic_or(Type *target, Type value, int pe) { \
       if (nvshmemi_job_connectivity_d <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {             \
    	    nvshmem_##Name##_atomic_fetch_or(target, (Type)value, pe);                     \
       } else {                                                                       \
            nvshmemi_proxy_amo_nonfetch<Type>((void *)target, value, pe, NVSHMEMI_AMO_OR);   \
       }										      \
    }

NVSHMEM_TYPE_OR_EMULATE(uint, unsigned int)
NVSHMEM_TYPE_OR_EMULATE(ulong, unsigned long)
NVSHMEM_TYPE_OR_EMULATE(ulonglong, unsigned long long)
NVSHMEM_TYPE_OR_EMULATE(int32, int32_t)
NVSHMEM_TYPE_OR_EMULATE(int64, int64_t)
NVSHMEM_TYPE_OR_EMULATE(uint32, uint32_t)
NVSHMEM_TYPE_OR_EMULATE(uint64, uint64_t)

#define NVSHMEM_TYPE_FETCH_XOR(Name, Type)                                                         \
    __device__ Type nvshmem_##Name##_atomic_fetch_xor(Type *target, Type value, int pe) {                 \
        void *peer_base_addr =                                                                     \
            (void *)__ldg((const unsigned long long *)nvshmemi_peer_heap_base_d + pe);              \
        if (nvshmemi_job_connectivity_d <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {                        \
            Type *target_actual =                                                                  \
                (Type *)((char *)peer_base_addr + ((char *)target - (char *)nvshmemi_heap_base_d)); \
                                                                                                   \
            return atomicXor(target_actual, value);                                                \
        } else {                                                                                   \
            Type retval;                                                                           \
            nvshmemi_proxy_amo_fetch<Type>((void *)target, (void *)&retval, (Type)value, 0, pe, NVSHMEMI_AMO_FETCH_XOR);      \
            return retval;                                                                         \
        }                                                                                          \
    }

#define NVSHMEM_TYPE_FETCH_XOR_CAST(Name, Type, subType)                                           \
    __device__ Type nvshmem_##Name##_atomic_fetch_xor(Type *target, Type value, int pe) {                 \
        void *peer_base_addr =                                                                     \
            (void *)__ldg((const unsigned long long *)nvshmemi_peer_heap_base_d + pe);              \
        if (nvshmemi_job_connectivity_d <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {                        \
            Type *target_actual =                                                                  \
                (Type *)((char *)peer_base_addr + ((char *)target - (char *)nvshmemi_heap_base_d)); \
                                                                                                   \
            return atomicXor((subType *)target_actual, *((subType *)&value));                      \
        } else {                                                                                   \
            Type retval;                                                                           \
            nvshmemi_proxy_amo_fetch<Type>((void *)target, (void *)&retval, (Type)value, 0, pe, NVSHMEMI_AMO_FETCH_XOR);      \
            return retval;                                                                         \
        }                                                                                          \
    }

NVSHMEM_TYPE_FETCH_XOR(uint, unsigned int)
NVSHMEM_TYPE_FETCH_XOR_CAST(ulong, unsigned long, unsigned long long int)
NVSHMEM_TYPE_FETCH_XOR(ulonglong, unsigned long long)
NVSHMEM_TYPE_FETCH_XOR_CAST(int32, int32_t, unsigned int)
NVSHMEM_TYPE_FETCH_XOR_CAST(int64, int64_t, unsigned long long int)
NVSHMEM_TYPE_FETCH_XOR_CAST(uint32, uint32_t, unsigned int)
NVSHMEM_TYPE_FETCH_XOR_CAST(uint64, uint64_t, unsigned long long int)

#define NVSHMEM_TYPE_XOR_EMULATE(Name, Type)                                 \
    __device__ void nvshmem_##Name##_atomic_xor(Type *target, Type value, int pe) { \
       if (nvshmemi_job_connectivity_d <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {             \
    	    nvshmem_##Name##_atomic_fetch_xor(target, (Type)value, pe);                     \
       } else {                                                                       \
            nvshmemi_proxy_amo_nonfetch<Type>((void *)target, value, pe, NVSHMEMI_AMO_XOR);   \
       }										      \
    }

NVSHMEM_TYPE_XOR_EMULATE(uint, unsigned int)
NVSHMEM_TYPE_XOR_EMULATE(ulong, unsigned long)
NVSHMEM_TYPE_XOR_EMULATE(ulonglong, unsigned long long)
NVSHMEM_TYPE_XOR_EMULATE(int32, int32_t)
NVSHMEM_TYPE_XOR_EMULATE(int64, int64_t)
NVSHMEM_TYPE_XOR_EMULATE(uint32, uint32_t)
NVSHMEM_TYPE_XOR_EMULATE(uint64, uint64_t)

#define NVSHMEM_TYPE_SWAP(Name, Type)                                                              \
    __device__ Type nvshmem_##Name##_atomic_swap(Type *target, Type value, int pe) {                      \
        void *peer_base_addr =                                                                     \
            (void *)__ldg((const unsigned long long *)nvshmemi_peer_heap_base_d + pe);              \
        if (nvshmemi_job_connectivity_d <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {                        \
            Type *target_actual =                                                                  \
                (Type *)((char *)peer_base_addr + ((char *)target - (char *)nvshmemi_heap_base_d)); \
                                                                                                   \
            return (Type)atomicExch(target_actual, value);                                         \
        } else {                                                                                   \
            Type retval;                                                                           \
            nvshmemi_proxy_amo_fetch<Type>((void *)target, (void *)&retval, (Type)value, 0, pe, NVSHMEMI_AMO_SWAP);  \
            return retval;                                                                         \
        }                                                                                          \
    }

#define NVSHMEM_TYPE_SWAP_CAST(Name, Type, subType)                                                \
    __device__ Type nvshmem_##Name##_atomic_swap(Type *target, Type value, int pe) {                      \
        void *peer_base_addr =                                                                     \
            (void *)__ldg((const unsigned long long *)nvshmemi_peer_heap_base_d + pe);              \
        if (nvshmemi_job_connectivity_d <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {                        \
            Type *target_actual =                                                                  \
                (Type *)((char *)peer_base_addr + ((char *)target - (char *)nvshmemi_heap_base_d)); \
                                                                                                   \
            return (Type)atomicExch((subType *)target_actual, *((subType *)&value));               \
        } else {                                                                                   \
            Type retval;                                                                           \
            nvshmemi_proxy_amo_fetch<Type>((void *)target, (void *)&retval, (Type)value, 0, pe, NVSHMEMI_AMO_SWAP);  \
            return retval;                                                                         \
        }                                                                                          \
    }

NVSHMEM_TYPE_SWAP(int, int)
NVSHMEM_TYPE_SWAP_CAST(long, long, int)
NVSHMEM_TYPE_SWAP_CAST(longlong, long long, unsigned long long int)
NVSHMEM_TYPE_SWAP(uint, unsigned int)
NVSHMEM_TYPE_SWAP_CAST(ulong, unsigned long, unsigned long long int)
NVSHMEM_TYPE_SWAP(ulonglong, unsigned long long)
NVSHMEM_TYPE_SWAP_CAST(int32, int32_t, unsigned int)
NVSHMEM_TYPE_SWAP_CAST(int64, int64_t, unsigned long long int)
NVSHMEM_TYPE_SWAP_CAST(uint32, uint32_t, unsigned int)
NVSHMEM_TYPE_SWAP_CAST(uint64, uint64_t, unsigned long long int)
NVSHMEM_TYPE_SWAP(float, float)
NVSHMEM_TYPE_SWAP_CAST(double, double, unsigned long long)
NVSHMEM_TYPE_SWAP_CAST(size, size_t, unsigned long long int)
NVSHMEM_TYPE_SWAP_CAST(ptrdiff, ptrdiff_t, unsigned long long int)

#define NVSHMEM_TYPE_FETCH_EMULATE(Name, Type)                     \
    __device__ Type nvshmem_##Name##_atomic_fetch(Type *target, int pe) { \
        return nvshmem_##Name##_atomic_fetch_or(target, (Type)0, pe);     \
    }

#define NVSHMEM_TYPE_FETCH_EMULATE_CAST(Name, Type, subName, subType)                   \
    __device__ Type nvshmem_##Name##_atomic_fetch(Type *target, int pe) {                      \
        subType temp = nvshmem_##subName##_atomic_fetch_or((subType *)target, (subType)0, pe); \
        return *((Type *)&temp);                                                        \
    }

NVSHMEM_TYPE_FETCH_EMULATE_CAST(int, int, uint, unsigned int)
NVSHMEM_TYPE_FETCH_EMULATE_CAST(long, long, ulonglong, unsigned long long int)
NVSHMEM_TYPE_FETCH_EMULATE_CAST(longlong, long long, ulonglong, unsigned long long int)
NVSHMEM_TYPE_FETCH_EMULATE(uint, unsigned int)
NVSHMEM_TYPE_FETCH_EMULATE(ulong, unsigned long)
NVSHMEM_TYPE_FETCH_EMULATE(ulonglong, unsigned long long)
NVSHMEM_TYPE_FETCH_EMULATE(int32, int32_t)
NVSHMEM_TYPE_FETCH_EMULATE(int64, int64_t)
NVSHMEM_TYPE_FETCH_EMULATE(uint32, uint32_t)
NVSHMEM_TYPE_FETCH_EMULATE(uint64, uint64_t)
NVSHMEM_TYPE_FETCH_EMULATE_CAST(float, float, uint, unsigned int)
NVSHMEM_TYPE_FETCH_EMULATE_CAST(double, double, ulonglong, unsigned long long int)
NVSHMEM_TYPE_FETCH_EMULATE_CAST(size, size_t, ulonglong, unsigned long long int)
NVSHMEM_TYPE_FETCH_EMULATE_CAST(ptrdiff, ptrdiff_t, ulonglong, unsigned long long int)

#define NVSHMEM_TYPE_SET_EMULATE(Name, Type)                                 \
    __device__ void nvshmem_##Name##_atomic_set(Type *target, Type value, int pe) { \
       if (nvshmemi_job_connectivity_d <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {             \
    	    nvshmem_##Name##_atomic_swap(target, value, pe);                            \
       } else {                                                                       \
            nvshmemi_proxy_amo_nonfetch<Type>((void *)target, value, pe, NVSHMEMI_AMO_SET);   \
       }										      \
    }

NVSHMEM_TYPE_SET_EMULATE(int, int)
NVSHMEM_TYPE_SET_EMULATE(long, long)
NVSHMEM_TYPE_SET_EMULATE(longlong, long long)
NVSHMEM_TYPE_SET_EMULATE(uint, unsigned int)
NVSHMEM_TYPE_SET_EMULATE(ulong, unsigned long)
NVSHMEM_TYPE_SET_EMULATE(ulonglong, unsigned long long)
NVSHMEM_TYPE_SET_EMULATE(int32, int32_t)
NVSHMEM_TYPE_SET_EMULATE(int64, int64_t)
NVSHMEM_TYPE_SET_EMULATE(uint32, uint32_t)
NVSHMEM_TYPE_SET_EMULATE(uint64, uint64_t)
NVSHMEM_TYPE_SET_EMULATE(float, float)
NVSHMEM_TYPE_SET_EMULATE(double, double)
NVSHMEM_TYPE_SET_EMULATE(size, size_t)
NVSHMEM_TYPE_SET_EMULATE(ptrdiff, ptrdiff_t)

#endif
