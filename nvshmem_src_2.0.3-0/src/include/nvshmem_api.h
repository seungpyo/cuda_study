/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */

#ifndef _NVSHMEM_API_H_
#define _NVSHMEM_API_H_

#include <stdint.h>
#include <stddef.h>
#include "nvshmem_types.h"
#include "nvshmem_common.cuh"
#include "nvshmem_constants.h"
#include "nvshmem_coll_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Library initialization
void nvshmem_init();
int nvshmem_init_thread(int requested, int *provided);
void nvshmem_query_thread(int *provided);
void nvshmem_finalize();


// PE info query
NVSHMEMI_HOSTDEVICE_PREFIX int nvshmem_my_pe();
NVSHMEMI_HOSTDEVICE_PREFIX int nvshmem_n_pes();
NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_info_get_version(int *major, int *minor);
NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_info_get_name(char *name);

// Heap management
void *nvshmem_malloc(size_t size);
void *nvshmem_calloc(size_t count, size_t size);
void nvshmem_free(void *ptr);
void *nvshmem_realloc(void *ptr, size_t size);
void *nvshmem_align(size_t alignment, size_t size);
NVSHMEMI_HOSTDEVICE_PREFIX void *nvshmem_ptr(const void *ptr, int pe);

//////////////////// OpenSHMEM 1.3 Atomics ////////////////////

/* inc */
#define NVSHMEMI_DECL_TYPE_INC(type, TYPE, opname) \
    NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_##type##_atomic_##opname(TYPE *dest, int pe);

/* finc, fetch */
#define NVSHMEMI_DECL_TYPE_FINC_FETCH(type, TYPE, opname) \
    NVSHMEMI_HOSTDEVICE_PREFIX TYPE nvshmem_##type##_atomic_##opname(TYPE *dest, int pe);

/* add, set */
#define NVSHMEMI_DECL_TYPE_ADD_SET(type, TYPE, opname) \
    NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_##type##_atomic_##opname(TYPE *dest, TYPE value, int pe);

/* fadd, swap */
#define NVSHMEMI_DECL_TYPE_FADD_SWAP(type, TYPE, opname) \
    NVSHMEMI_HOSTDEVICE_PREFIX TYPE nvshmem_##type##_atomic_##opname(TYPE *dest, TYPE value, int pe);

/* cswap */
#define NVSHMEMI_DECL_TYPE_CSWAP(type, TYPE, opname)                                             \
    NVSHMEMI_HOSTDEVICE_PREFIX TYPE nvshmem_##type##_atomic_##opname(TYPE *dest, TYPE cond, TYPE value, \
                                                              int pe);

NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(INC, inc)
NVSHMEMI_REPT_OPGROUP_FOR_STANDARD_AMO(INC, inc)

NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(FINC_FETCH, fetch_inc)
NVSHMEMI_REPT_OPGROUP_FOR_STANDARD_AMO(FINC_FETCH, fetch_inc)

NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(FINC_FETCH, fetch)
NVSHMEMI_REPT_OPGROUP_FOR_STANDARD_AMO(FINC_FETCH, fetch)
NVSHMEMI_REPT_OPGROUP_FOR_EXTENDED_AMO(FINC_FETCH, fetch)

NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(ADD_SET, add)
NVSHMEMI_REPT_OPGROUP_FOR_STANDARD_AMO(ADD_SET, add)

NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(ADD_SET, set)
NVSHMEMI_REPT_OPGROUP_FOR_STANDARD_AMO(ADD_SET, set)
NVSHMEMI_REPT_OPGROUP_FOR_EXTENDED_AMO(ADD_SET, set)

NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(FADD_SWAP, fetch_add)
NVSHMEMI_REPT_OPGROUP_FOR_STANDARD_AMO(FADD_SWAP, fetch_add)

NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(FADD_SWAP, swap)
NVSHMEMI_REPT_OPGROUP_FOR_STANDARD_AMO(FADD_SWAP, swap)
NVSHMEMI_REPT_OPGROUP_FOR_EXTENDED_AMO(FADD_SWAP, swap)

NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(CSWAP, compare_swap)
NVSHMEMI_REPT_OPGROUP_FOR_STANDARD_AMO(CSWAP, compare_swap)

#undef NVSHMEMI_DECL_TYPE_INC
#undef NVSHMEMI_DECL_TYPE_FINC_FETCH
#undef NVSHMEMI_DECL_TYPE_ADD_SET
#undef NVSHMEMI_DECL_TYPE_FADD_SWAP
#undef NVSHMEMI_DECL_TYPE_CSWAP

//////////////////// OpenSHMEM 1.4 Atomics ////////////////////

/* and, or, xor */
#define NVSHMEMI_DECL_TYPE_AND_OR_XOR(type, TYPE, opname)               \
    NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_##type##_atomic_##opname(   \
            TYPE *dest, TYPE value, int pe);

/* fand, for, fxor */
#define NVSHMEMI_DECL_TYPE_FAND_FOR_FXOR(type, TYPE, opname)                    \
    NVSHMEMI_HOSTDEVICE_PREFIX TYPE nvshmem_##type##_atomic_fetch_##opname(     \
            TYPE *dest, TYPE value, int pe);

NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(AND_OR_XOR, and)
NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(AND_OR_XOR, or)
NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(AND_OR_XOR, xor)

NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(FAND_FOR_FXOR, and)
NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(FAND_FOR_FXOR, or)
NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(FAND_FOR_FXOR, xor)

#undef NVSHMEMI_DECL_TYPE_AND_OR_XOR
#undef NVSHMEMI_DECL_TYPE_FAND_FOR_FXOR

//////////////////// Put ////////////////////

#define NVSHMEMI_DECL_TYPE_P(NAME, TYPE) \
    NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_##NAME##_p(TYPE *dest, const TYPE value, int pe);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_DECL_TYPE_P)
#undef NVSHMEMI_DECL_TYPE_P

#define NVSHMEMI_DECL_TYPE_PUT(NAME, TYPE)                                               \
    NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_##NAME##_put(TYPE *dest, const TYPE *source, \
                                                         size_t nelems, int pe);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_DECL_TYPE_PUT)
#undef NVSHMEMI_DECL_TYPE_PUT

#define NVSHMEMI_DECL_SIZE_PUT(NAME)                                                  \
    NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_put##NAME(void *dest, const void *source, \
                                                      size_t nelems, int pe);

NVSHMEMI_REPT_FOR_SIZES(NVSHMEMI_DECL_SIZE_PUT)
#undef NVSHMEMI_DECL_SIZE_PUT

NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_putmem(void *dest, const void *source, size_t bytes,
                                               int pe);

#define NVSHMEMI_DECL_TYPE_IPUT(NAME, TYPE)                \
    NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_##NAME##_iput( \
        TYPE *dest, const TYPE *source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_DECL_TYPE_IPUT)
#undef NVSHMEMI_DECL_TYPE_PUT

#define NVSHMEMI_DECL_SIZE_IPUT(NAME)                   \
    NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_iput##NAME( \
        void *dest, const void *source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe);

NVSHMEMI_REPT_FOR_SIZES(NVSHMEMI_DECL_SIZE_IPUT)
#undef NVSHMEMI_DECL_SIZE_IPUT

#define NVSHMEMI_DECL_TYPE_PUT_NBI(NAME, TYPE)                                               \
    NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_##NAME##_put_nbi(TYPE *dest, const TYPE *source, \
                                                             size_t nelems, int pe);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_DECL_TYPE_PUT_NBI)
#undef NVSHMEM_DECL_TYPE_PUT_NBI

#define NVSHMEMI_DECL_SIZE_PUT_NBI(NAME)                                                    \
    NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_put##NAME##_nbi(void *dest, const void *source, \
                                                            size_t nelems, int pe);

NVSHMEMI_REPT_FOR_SIZES(NVSHMEMI_DECL_SIZE_PUT_NBI)
#undef NVSHMEMI_DECL_SIZE_PUT_NBI

NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_putmem_nbi(void *dest, const void *source, size_t bytes,
                                                   int pe);

//////////////////// Get ////////////////////

#define NVSHMEMI_DECL_TYPE_G(NAME, TYPE) \
    NVSHMEMI_HOSTDEVICE_PREFIX TYPE nvshmem_##NAME##_g(const TYPE *src, int pe);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_DECL_TYPE_G)
#undef NVSHMEMI_DECL_TYPE_G

#define NVSHMEMI_DECL_TYPE_GET(NAME, TYPE)                                               \
    NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_##NAME##_get(TYPE *dest, const TYPE *source, \
                                                         size_t nelems, int pe);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_DECL_TYPE_GET)
#undef NVSHMEMI_DECL_TYPE_GET

#define NVSHMEMI_DECL_SIZE_GET(NAME)                                                  \
    NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_get##NAME(void *dest, const void *source, \
                                                      size_t nelems, int pe);

NVSHMEMI_REPT_FOR_SIZES(NVSHMEMI_DECL_SIZE_GET)
#undef NVSHMEMI_DECL_SIZE_GET

NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_getmem(void *dest, const void *source, size_t bytes,
                                               int pe);

#define NVSHMEMI_DECL_TYPE_IGET(NAME, TYPE)                \
    NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_##NAME##_iget( \
        TYPE *dest, const TYPE *source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_DECL_TYPE_IGET)
#undef NVSHMEMI_DECL_TYPE_IGET

#define NVSHMEMI_DECL_SIZE_IGET(NAME)                   \
    NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_iget##NAME( \
        void *dest, const void *source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe);

NVSHMEMI_REPT_FOR_SIZES(NVSHMEMI_DECL_SIZE_IGET)
#undef NVSHMEMI_DECL_SIZE_IGET

#define NVSHMEMI_DECL_TYPE_GET_NBI(NAME, TYPE)                                               \
    NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_##NAME##_get_nbi(TYPE *dest, const TYPE *source, \
                                                             size_t nelems, int pe);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_DECL_TYPE_GET_NBI)
#undef NVSHMEMI_DECL_TYPE_GET_NBI

#define NVSHMEMI_DECL_SIZE_GET_NBI(NAME)                                                    \
    NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_get##NAME##_nbi(void *dest, const void *source, \
                                                            size_t nelems, int pe);

NVSHMEMI_REPT_FOR_SIZES(NVSHMEMI_DECL_SIZE_GET_NBI)
#undef NVSHMEMI_DECL_SIZE_GET_NBI

NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_getmem_nbi(void *dest, const void *source, size_t bytes,
                                                   int pe);
/* Signal API */
#define NVSHMEMI_DECL_PUT_SIGNAL(TYPENAME, TYPE)                                                        \
    NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_##TYPENAME##_put_signal(TYPE *dest, const TYPE *source,       \
                                                                  size_t nelems, uint64_t *sig_addr,    \
                                                                  uint64_t signal, int sig_op, int pe);
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_DECL_PUT_SIGNAL)
#undef NVSHMEMI_DECL_PUT_SIGNAL

#define NVSHMEMI_DECL_PUT_SIGNAL_NBI(TYPENAME, TYPE)                                                        \
    NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_##TYPENAME##_put_signal_nbi(TYPE *dest, const TYPE *source,       \
                                                                      size_t nelems, uint64_t *sig_addr,    \
                                                                      uint64_t signal, int sig_op, int pe);
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_DECL_PUT_SIGNAL_NBI)
#undef NVSHMEMI_DECL_PUT_SIGNAL_NBI

#ifdef __CUDACC__
#define NVSHMEMI_DECL_SIZE_PUT_SIGNAL(BITS)                                                     \
    __device__ void nvshmem_put##BITS##_signal(void *dest, const void *source, size_t nelems,   \
                                               uint64_t *sig_addr, uint64_t signal, int sig_op, \
                                               int pe);
NVSHMEMI_REPT_FOR_SIZES(NVSHMEMI_DECL_SIZE_PUT_SIGNAL)
#undef NVSHMEMI_DECL_SIZE_PUT_SIGNAL

__device__ void nvshmem_putmem_signal(void *dest, const void *source, size_t bytes,
                                      uint64_t *sig_addr, uint64_t signal, int sig_op,
                                      int pe);
#endif

NVSHMEMI_HOSTDEVICE_PREFIX uint64_t nvshmem_signal_fetch(uint64_t *sig_addr);


//////////////////// Point-to-Point Synchronization ////////////////////

NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_quiet();
NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_fence();
NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_wait(long *ivar, long cmp_value);
NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_wait_until(long *ivar, int cmp, long cmp_value);
#ifdef __CUDACC__
__device__ uint64_t nvshmem_signal_wait_until(uint64_t *sig_addr, int cmp, uint64_t cmp_val);
#endif

#define NVSHMEMI_DECL_WAIT(NAME, TYPE) \
    NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_##NAME##_wait(TYPE *ivar, TYPE cmp_value);

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMI_DECL_WAIT)
#undef NVSHMEMI_DECL_WAIT

#define NVSHMEMI_DECL_WAIT_UNTIL(NAME, TYPE)                                         \
    NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_##NAME##_wait_until(TYPE *ivar, int cmp, \
                                                                TYPE cmp_value);

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMI_DECL_WAIT_UNTIL)
#undef NVSHMEMI_DECL_WAIT_UNTIL

#define NVSHMEMI_DECL_WAIT_UNTIL_ALL(NAME, TYPE)                     \
    NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_##NAME##_wait_until_all( \
        TYPE *ivar, size_t nelems, const int *status, int cmp, TYPE cmp_value);

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMI_DECL_WAIT_UNTIL_ALL)
#undef NVSHMEMI_DECL_WAIT_UNTIL_ALL

#define NVSHMEMI_DECL_WAIT_UNTIL_ANY(NAME, TYPE)                       \
    NVSHMEMI_HOSTDEVICE_PREFIX size_t nvshmem_##NAME##_wait_until_any( \
        TYPE *ivar, size_t nelems, const int *status, int cmp, TYPE cmp_value);

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMI_DECL_WAIT_UNTIL_ANY)
#undef NVSHMEMI_DECL_WAIT_UNTIL_ANY

#define NVSHMEMI_DECL_WAIT_UNTIL_SOME(NAME, TYPE)                       \
    NVSHMEMI_HOSTDEVICE_PREFIX size_t nvshmem_##NAME##_wait_until_some( \
        TYPE *ivar, size_t nelems, size_t *indices, const int *status, int cmp, TYPE cmp_value);

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMI_DECL_WAIT_UNTIL_SOME)
#undef NVSHMEMI_DECL_WAIT_UNTIL_SOME

#define NVSHMEMI_DECL_WAIT_UNTIL_ALL_VECTOR(NAME, TYPE)                     \
    NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_##NAME##_wait_until_all_vector(   \
        TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE *cmp_values);

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMI_DECL_WAIT_UNTIL_ALL_VECTOR)
#undef NVSHMEMI_DECL_WAIT_UNTIL_ALL_VECTOR

#define NVSHMEMI_DECL_WAIT_UNTIL_ANY_VECTOR(NAME, TYPE)                         \
    NVSHMEMI_HOSTDEVICE_PREFIX size_t nvshmem_##NAME##_wait_until_any_vector(     \
        TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE *cmp_values);

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMI_DECL_WAIT_UNTIL_ANY_VECTOR)
#undef NVSHMEMI_DECL_WAIT_UNTIL_ANY_VECTOR

#define NVSHMEMI_DECL_WAIT_UNTIL_SOME_VECTOR(NAME, TYPE)                         \
    NVSHMEMI_HOSTDEVICE_PREFIX size_t nvshmem_##NAME##_wait_until_some_vector(     \
        TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, TYPE *cmp_values);

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMI_DECL_WAIT_UNTIL_SOME_VECTOR)
#undef NVSHMEMI_DECL_WAIT_UNTIL_SOME_VECTOR

#define NVSHMEMI_DECL_TEST(NAME, TYPE) \
    NVSHMEMI_HOSTDEVICE_PREFIX int nvshmem_##NAME##_test(TYPE *ivar, int cmp, TYPE cmp_value);

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMI_DECL_TEST)
#undef NVSHMEMI_DECL_TEST

#define NVSHMEMI_DECL_TEST_ALL(Name, Type)                    \
    NVSHMEMI_HOSTDEVICE_PREFIX int nvshmem_##Name##_test_all( \
        Type *ivars, size_t nelems, const int *status, int cmp, Type cmp_value);

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMI_DECL_TEST_ALL)
#undef NVSHMEMI_DECL_TEST_ALL

#define NVSHMEMI_DECL_TEST_ANY(Name, Type)                       \
    NVSHMEMI_HOSTDEVICE_PREFIX size_t nvshmem_##Name##_test_any( \
        Type *ivars, size_t nelems, const int *status, int cmp, Type cmp_value);

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMI_DECL_TEST_ANY)
#undef NVSHMEMI_DECL_TEST_ANY

#define NVSHMEMI_DECL_TEST_SOME(Name, Type)                       \
    NVSHMEMI_HOSTDEVICE_PREFIX size_t nvshmem_##Name##_test_some( \
        Type *ivars, size_t nelems, size_t *indices, const int *status, int cmp, Type cmp_value);

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMI_DECL_TEST_SOME)
#undef NVSHMEMI_DECL_TEST_SOME

#define NVSHMEMI_DECL_TEST_ALL_VECTOR(NAME, TYPE)                     \
    NVSHMEMI_HOSTDEVICE_PREFIX int nvshmem_##NAME##_test_all_vector(   \
        TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE *cmp_values);

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMI_DECL_TEST_ALL_VECTOR)
#undef NVSHMEMI_DECL_TEST_ALL_VECTOR

#define NVSHMEMI_DECL_TEST_ANY_VECTOR(NAME, TYPE)                         \
    NVSHMEMI_HOSTDEVICE_PREFIX size_t nvshmem_##NAME##_test_any_vector(     \
        TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE *cmp_values);

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMI_DECL_TEST_ANY_VECTOR)
#undef NVSHMEMI_DECL_TEST_ANY_VECTOR

#define NVSHMEMI_DECL_TEST_SOME_VECTOR(NAME, TYPE)                         \
    NVSHMEMI_HOSTDEVICE_PREFIX size_t nvshmem_##NAME##_test_some_vector(     \
        TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, TYPE *cmp_values);

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMI_DECL_TEST_SOME_VECTOR)
#undef NVSHMEMI_DECL_TEST_SOME_VECTOR

//////////////////// Teams API ////////////////////

#define NVSHMEM_TEAM_WORLD  0
#define NVSHMEM_TEAM_SHARED 1
#define NVSHMEMX_TEAM_NODE  2
#define NVSHMEM_TEAM_INVALID -1


NVSHMEMI_HOSTDEVICE_PREFIX int nvshmem_team_my_pe(nvshmem_team_t team);
NVSHMEMI_HOSTDEVICE_PREFIX int nvshmem_team_n_pes(nvshmem_team_t team);


void nvshmem_team_get_config(nvshmem_team_t team, nvshmem_team_config_t *config);
NVSHMEMI_HOSTDEVICE_PREFIX int nvshmem_team_translate_pe(nvshmem_team_t src_team, int src_pe, nvshmem_team_t dest_team);
int nvshmem_team_split_strided(nvshmem_team_t parent_team, int PE_start, int PE_stride,
                                int PE_size, const nvshmem_team_config_t *config,
                                long config_mask, nvshmem_team_t *new_team);
int nvshmem_team_split_2d(nvshmem_team_t parent_team, int xrange,
                          const nvshmem_team_config_t *xaxis_config,
                          long xaxis_mask, nvshmem_team_t *xaxis_team,
                          const nvshmem_team_config_t *yaxis_config,
                          long yaxis_mask, nvshmem_team_t *yaxis_team);
void nvshmem_team_destroy(nvshmem_team_t team);

//////////////////// Deprecated API ////////////////////

static inline void nvstart_pes(int npes) __attribute__((deprecated));
static inline int NVSHMEMI_HOSTDEVICE_PREFIX nv_num_pes(void) __attribute__((deprecated));
static inline int NVSHMEMI_HOSTDEVICE_PREFIX nv_my_pe(void) __attribute__((deprecated));
static inline void *nvshmalloc(size_t size) __attribute__((deprecated));
static inline void nvshfree(void *ptr) __attribute__((deprecated));
static inline void *nvshrealloc(void *ptr, size_t size) __attribute__((deprecated));
static inline void *nvshmemalign(size_t alignment, size_t size) __attribute__((deprecated));

static inline void nvstart_pes(int npes) {
    NVSHMEMI_UNUSED_ARG(npes);
    nvshmem_init();
}
static inline int NVSHMEMI_HOSTDEVICE_PREFIX nv_num_pes(void) { return nvshmem_n_pes(); }
static inline int NVSHMEMI_HOSTDEVICE_PREFIX nv_my_pe(void) { return nvshmem_my_pe(); }
static inline void *nvshmalloc(size_t size) { return nvshmem_malloc(size); }
static inline void nvshfree(void *ptr) { nvshmem_free(ptr); }
static inline void *nvshrealloc(void *ptr, size_t size) { return nvshmem_realloc(ptr, size); }
static inline void *nvshmemalign(size_t alignment, size_t size) {
    return nvshmem_align(alignment, size);
}

static inline void nvshmem_clear_cache_inv(void) __attribute__((deprecated));
static inline void nvshmem_set_cache_inv(void) __attribute__((deprecated));
static inline void nvshmem_clear_cache_line_inv(void *dest) __attribute__((deprecated));
static inline void nvshmem_set_cache_line_inv(void *dest) __attribute__((deprecated));
static inline void nvshmem_udcflush(void) __attribute__((deprecated));
static inline void nvshmem_udcflush_line(void *dest) __attribute__((deprecated));

static inline void nvshmem_clear_cache_inv(void) {}
static inline void nvshmem_set_cache_inv(void) {}
static inline void nvshmem_clear_cache_line_inv(void *dest) { NVSHMEMI_UNUSED_ARG(dest); }
static inline void nvshmem_set_cache_line_inv(void *dest) { NVSHMEMI_UNUSED_ARG(dest); }
static inline void nvshmem_udcflush(void) {}
static inline void nvshmem_udcflush_line(void *dest) { NVSHMEMI_UNUSED_ARG(dest); }

#ifdef __cplusplus
}
#endif

#endif
