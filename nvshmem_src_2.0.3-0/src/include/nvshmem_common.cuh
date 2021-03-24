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

#ifndef _NVSHMEM_COMMON_H_
#define _NVSHMEM_COMMON_H_

#include <cuda_runtime.h>
#ifdef NVSHMEM_COMPLEX_SUPPORT
#include <complex.h>
#endif
#include "nvshmem_constants.h"
#include "nvshmemi_constants.h"

#ifdef __CUDA_ARCH__
#  ifdef NVSHMEMI_HOST_ONLY
#    define NVSHMEMI_HOSTDEVICE_PREFIX __host__
#  else
#    define NVSHMEMI_HOSTDEVICE_PREFIX __host__ __device__
#  endif
#else
#define NVSHMEMI_HOSTDEVICE_PREFIX
#endif

#define NVSHMEMI_UNUSED_ARG(ARG) (void)(ARG)

#define NVSHPRI_float       "%0.2f"
#define NVSHPRI_double      "%0.2f"
#define NVSHPRI_char        "%hhd"
#define NVSHPRI_schar       "%hhd"
#define NVSHPRI_short       "%hd"
#define NVSHPRI_int         "%d"
#define NVSHPRI_long        "%ld"
#define NVSHPRI_longlong    "%lld"
#define NVSHPRI_uchar       "%hhu"
#define NVSHPRI_ushort      "%hu"
#define NVSHPRI_uint        "%u"
#define NVSHPRI_ulong       "%lu"
#define NVSHPRI_ulonglong   "%llu"
#define NVSHPRI_int8        "%" PRIi8
#define NVSHPRI_int16       "%" PRIi16
#define NVSHPRI_int32       "%" PRIi32
#define NVSHPRI_int64       "%" PRIi64
#define NVSHPRI_uint8       "%" PRIu8
#define NVSHPRI_uint16      "%" PRIu16
#define NVSHPRI_uint32      "%" PRIu32
#define NVSHPRI_uint64      "%" PRIu64
#define NVSHPRI_size        "%zu"
#define NVSHPRI_ptrdiff     "%zu"
#define NVSHPRI_bool        "%s"
#define NVSHPRI_string      "\"%s\""

#define NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(OPGRPNAME, opname)                \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(uint, unsigned int, opname)                  \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(ulong, unsigned long, opname)                \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(ulonglong, unsigned long long, opname)       \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(int32, int32_t, opname)                      \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(uint32, uint32_t, opname)                    \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(int64, int64_t, opname)                      \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(uint64, uint64_t, opname)

#define NVSHMEMI_REPT_OPGROUP_FOR_STANDARD_AMO(OPGRPNAME, opname)               \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(int, int, opname)                            \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(long, long, opname)                          \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(longlong, long long, opname)                 \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(size, size_t, opname)                        \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(ptrdiff, ptrdiff_t, opname)

#define NVSHMEMI_REPT_OPGROUP_FOR_EXTENDED_AMO(OPGRPNAME, opname)               \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(float, float, opname)                        \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(double, double, opname)

/* Note: The "long double" type is not supported */
#define NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_FN_TEMPLATE)      \
    NVSHMEMI_FN_TEMPLATE(float, float)                                  \
    NVSHMEMI_FN_TEMPLATE(double, double)                                \
    NVSHMEMI_FN_TEMPLATE(char, char)                                    \
    NVSHMEMI_FN_TEMPLATE(short, short)                                  \
    NVSHMEMI_FN_TEMPLATE(schar, signed char)                            \
    NVSHMEMI_FN_TEMPLATE(int, int)                                      \
    NVSHMEMI_FN_TEMPLATE(long, long)                                    \
    NVSHMEMI_FN_TEMPLATE(longlong, long long)                           \
    NVSHMEMI_FN_TEMPLATE(uchar, unsigned char)                          \
    NVSHMEMI_FN_TEMPLATE(ushort, unsigned short)                        \
    NVSHMEMI_FN_TEMPLATE(uint, unsigned int)                            \
    NVSHMEMI_FN_TEMPLATE(ulong, unsigned long)                          \
    NVSHMEMI_FN_TEMPLATE(ulonglong, unsigned long long)                 \
    NVSHMEMI_FN_TEMPLATE(int8, int8_t)                                  \
    NVSHMEMI_FN_TEMPLATE(int16, int16_t)                                \
    NVSHMEMI_FN_TEMPLATE(int32, int32_t)                                \
    NVSHMEMI_FN_TEMPLATE(int64, int64_t)                                \
    NVSHMEMI_FN_TEMPLATE(uint8, uint8_t)                                \
    NVSHMEMI_FN_TEMPLATE(uint16, uint16_t)                              \
    NVSHMEMI_FN_TEMPLATE(uint32, uint32_t)                              \
    NVSHMEMI_FN_TEMPLATE(uint64, uint64_t)                              \
    NVSHMEMI_FN_TEMPLATE(size, size_t)                                  \
    NVSHMEMI_FN_TEMPLATE(ptrdiff, ptrdiff_t)

#define NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE(NVSHMEMI_FN_TEMPLATE, SC)      \
    NVSHMEMI_FN_TEMPLATE(SC, float, float)                                  \
    NVSHMEMI_FN_TEMPLATE(SC, double, double)                                \
    NVSHMEMI_FN_TEMPLATE(SC, char, char)                                    \
    NVSHMEMI_FN_TEMPLATE(SC, short, short)                                  \
    NVSHMEMI_FN_TEMPLATE(SC, schar, signed char)                            \
    NVSHMEMI_FN_TEMPLATE(SC, int, int)                                      \
    NVSHMEMI_FN_TEMPLATE(SC, long, long)                                    \
    NVSHMEMI_FN_TEMPLATE(SC, longlong, long long)                           \
    NVSHMEMI_FN_TEMPLATE(SC, uchar, unsigned char)                          \
    NVSHMEMI_FN_TEMPLATE(SC, ushort, unsigned short)                        \
    NVSHMEMI_FN_TEMPLATE(SC, uint, unsigned int)                            \
    NVSHMEMI_FN_TEMPLATE(SC, ulong, unsigned long)                          \
    NVSHMEMI_FN_TEMPLATE(SC, ulonglong, unsigned long long)                 \
    NVSHMEMI_FN_TEMPLATE(SC, int8, int8_t)                                  \
    NVSHMEMI_FN_TEMPLATE(SC, int16, int16_t)                                \
    NVSHMEMI_FN_TEMPLATE(SC, int32, int32_t)                                \
    NVSHMEMI_FN_TEMPLATE(SC, int64, int64_t)                                \
    NVSHMEMI_FN_TEMPLATE(SC, uint8, uint8_t)                                \
    NVSHMEMI_FN_TEMPLATE(SC, uint16, uint16_t)                              \
    NVSHMEMI_FN_TEMPLATE(SC, uint32, uint32_t)                              \
    NVSHMEMI_FN_TEMPLATE(SC, uint64, uint64_t)                              \
    NVSHMEMI_FN_TEMPLATE(SC, size, size_t)                                  \
    NVSHMEMI_FN_TEMPLATE(SC, ptrdiff, ptrdiff_t)

#define NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX, SC_PREFIX)      \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, float, float)                                  \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, double, double)                                \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, char, char)                                    \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, short, short)                                  \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, schar, signed char)                            \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, int, int)                                      \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, long, long)                                    \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, longlong, long long)                           \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, uchar, unsigned char)                          \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, ushort, unsigned short)                        \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, uint, unsigned int)                            \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, ulong, unsigned long)                          \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, ulonglong, unsigned long long)                 \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, int8, int8_t)                                  \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, int16, int16_t)                                \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, int32, int32_t)                                \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, int64, int64_t)                                \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, uint8, uint8_t)                                \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, uint16, uint16_t)                              \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, uint32, uint32_t)                              \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, uint64, uint64_t)                              \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, size, size_t)                                  \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, ptrdiff, ptrdiff_t)

#define NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_AND_SCOPES2(NVSHMEMI_FN_TEMPLATE)                      \
    NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, thread, , )              \
    NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, warp, _warp, x)          \
    NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, block, _block, x)

#define NVSHMEMI_REPT_FOR_SIZES(NVSHMEMI_FN_TEMPLATE)                   \
    NVSHMEMI_FN_TEMPLATE(8)                                             \
    NVSHMEMI_FN_TEMPLATE(16)                                            \
    NVSHMEMI_FN_TEMPLATE(32)                                            \
    NVSHMEMI_FN_TEMPLATE(64)                                            \
    NVSHMEMI_FN_TEMPLATE(128)

#define NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEMI_FN_TEMPLATE)         \
    NVSHMEMI_FN_TEMPLATE(8, int8_t)                                     \
    NVSHMEMI_FN_TEMPLATE(16, int16_t)                                   \
    NVSHMEMI_FN_TEMPLATE(32, int32_t)                                   \
    NVSHMEMI_FN_TEMPLATE(64, int64_t)                                   \
    NVSHMEMI_FN_TEMPLATE(128, int4)


#define NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMI_FN_TEMPLATE)              \
    NVSHMEMI_FN_TEMPLATE(short, short)                                  \
    NVSHMEMI_FN_TEMPLATE(int, int)                                      \
    NVSHMEMI_FN_TEMPLATE(long, long)                                    \
    NVSHMEMI_FN_TEMPLATE(longlong, long long)                           \
    NVSHMEMI_FN_TEMPLATE(ushort, unsigned short)                        \
    NVSHMEMI_FN_TEMPLATE(uint, unsigned int)                            \
    NVSHMEMI_FN_TEMPLATE(ulong, unsigned long)                          \
    NVSHMEMI_FN_TEMPLATE(ulonglong, unsigned long long)                 \
    NVSHMEMI_FN_TEMPLATE(int32, int32_t)                                \
    NVSHMEMI_FN_TEMPLATE(int64, int64_t)                                \
    NVSHMEMI_FN_TEMPLATE(uint32, uint32_t)                              \
    NVSHMEMI_FN_TEMPLATE(uint64, uint64_t)                              \
    NVSHMEMI_FN_TEMPLATE(size, size_t)                                  \
    NVSHMEMI_FN_TEMPLATE(ptrdiff, ptrdiff_t)

#define NVSHMEMX_REPT_FOR_SIGNAL_TYPES(NVSHMEMI_FN_TEMPLATE)            \
    NVSHMEMI_FN_TEMPLATE(short, short)                                  \
    NVSHMEMI_FN_TEMPLATE(ushort, unsigned short)                        \
    NVSHMEMI_FN_TEMPLATE(int, int)                                      \
    NVSHMEMI_FN_TEMPLATE(uint, unsigned int)                            \
    NVSHMEMI_FN_TEMPLATE(long, long)                                    \
    NVSHMEMI_FN_TEMPLATE(ulong, unsigned long)                          \
    NVSHMEMI_FN_TEMPLATE(longlong, long long)                           \
    NVSHMEMI_FN_TEMPLATE(ulonglong, unsigned long long)                 \
    NVSHMEMI_FN_TEMPLATE(int32, int32_t)                                \
    NVSHMEMI_FN_TEMPLATE(int64, int64_t)                                \
    NVSHMEMI_FN_TEMPLATE(uint32, uint32_t)                              \
    NVSHMEMI_FN_TEMPLATE(uint64, uint64_t)                              \
    NVSHMEMI_FN_TEMPLATE(size, size_t)                                  \
    NVSHMEMI_FN_TEMPLATE(ptrdiff, ptrdiff_t)

#define NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, opname) \
    NVSHMEMI_FN_TEMPLATE(uchar, unsigned char, opname)                       \
    NVSHMEMI_FN_TEMPLATE(ushort, unsigned short, opname)                     \
    NVSHMEMI_FN_TEMPLATE(uint, unsigned int, opname)                         \
    NVSHMEMI_FN_TEMPLATE(ulong, unsigned long, opname)                       \
    NVSHMEMI_FN_TEMPLATE(ulonglong, unsigned long long, opname)              \
    NVSHMEMI_FN_TEMPLATE(int8, int8_t, opname)                               \
    NVSHMEMI_FN_TEMPLATE(int16, int16_t, opname)                             \
    NVSHMEMI_FN_TEMPLATE(int32, int32_t, opname)                             \
    NVSHMEMI_FN_TEMPLATE(int64, int64_t, opname)                             \
    NVSHMEMI_FN_TEMPLATE(uint8, uint8_t, opname)                             \
    NVSHMEMI_FN_TEMPLATE(uint16, uint16_t, opname)                           \
    NVSHMEMI_FN_TEMPLATE(uint32, uint32_t, opname)                           \
    NVSHMEMI_FN_TEMPLATE(uint64, uint64_t, opname)                           \
    NVSHMEMI_FN_TEMPLATE(size, size_t, opname)

/* Note: The "long double" type is not supported */
#define NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, opname)   \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, opname)        \
    NVSHMEMI_FN_TEMPLATE(char, char, opname)                                    \
    NVSHMEMI_FN_TEMPLATE(schar, signed char, opname)                            \
    NVSHMEMI_FN_TEMPLATE(short, short, opname)                                  \
    NVSHMEMI_FN_TEMPLATE(int, int, opname)                                      \
    NVSHMEMI_FN_TEMPLATE(long, long, opname)                                    \
    NVSHMEMI_FN_TEMPLATE(longlong, long long, opname)                           \
    NVSHMEMI_FN_TEMPLATE(float, float, opname)                                  \
    NVSHMEMI_FN_TEMPLATE(double, double, opname)

#ifdef NVSHMEM_COMPLEX_SUPPORT
#define NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, opname)  \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, opname)   \
    NVSHMEMI_FN_TEMPLATE(complexf, double complex, opname)                  \
    NVSHMEMI_FN_TEMPLATE(complexd, float complex, opname)
#else
#define NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, opname)  \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, opname)
#endif


#define NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE(NVSHMEMI_FN_TEMPLATE, SC, opname) \
    NVSHMEMI_FN_TEMPLATE(SC, uchar, unsigned char, opname)                          \
    NVSHMEMI_FN_TEMPLATE(SC, ushort, unsigned short, opname)                        \
    NVSHMEMI_FN_TEMPLATE(SC, uint, unsigned int, opname)                            \
    NVSHMEMI_FN_TEMPLATE(SC, ulong, unsigned long, opname)                          \
    NVSHMEMI_FN_TEMPLATE(SC, ulonglong, unsigned long long, opname)                 \
    NVSHMEMI_FN_TEMPLATE(SC, int8, int8_t, opname)                                  \
    NVSHMEMI_FN_TEMPLATE(SC, int16, int16_t, opname)                                \
    NVSHMEMI_FN_TEMPLATE(SC, int32, int32_t, opname)                                \
    NVSHMEMI_FN_TEMPLATE(SC, int64, int64_t, opname)                                \
    NVSHMEMI_FN_TEMPLATE(SC, uint8, uint8_t, opname)                                \
    NVSHMEMI_FN_TEMPLATE(SC, uint16, uint16_t, opname)                              \
    NVSHMEMI_FN_TEMPLATE(SC, uint32, uint32_t, opname)                              \
    NVSHMEMI_FN_TEMPLATE(SC, uint64, uint64_t, opname)                              \
    NVSHMEMI_FN_TEMPLATE(SC, size, size_t, opname)

/* Note: The "long double" type is not supported */
#define NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE(NVSHMEMI_FN_TEMPLATE, SC, opname)    \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE(NVSHMEMI_FN_TEMPLATE, SC, opname)         \
    NVSHMEMI_FN_TEMPLATE(SC, char, char, opname)                                                \
    NVSHMEMI_FN_TEMPLATE(SC, schar, signed char, opname)                                        \
    NVSHMEMI_FN_TEMPLATE(SC, short, short, opname)                                              \
    NVSHMEMI_FN_TEMPLATE(SC, int, int, opname)                                                  \
    NVSHMEMI_FN_TEMPLATE(SC, long, long, opname)                                                \
    NVSHMEMI_FN_TEMPLATE(SC, longlong, long long, opname)                                       \
    NVSHMEMI_FN_TEMPLATE(SC, float, float, opname)                                              \
    NVSHMEMI_FN_TEMPLATE(SC, double, double, opname)

#ifdef NVSHMEM_COMPLEX_SUPPORT
#define NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES_WITH_SCOPE(NVSHMEMI_FN_TEMPLATE, SC, opname)   \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE(NVSHMEMI_FN_TEMPLATE, SC, opname)    \
    NVSHMEMI_FN_TEMPLATE(SC, complexf, double complex, opname)                              \
    NVSHMEMI_FN_TEMPLATE(SC, complexd, float complex, opname)
#else
#define NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES_WITH_SCOPE(NVSHMEMI_FN_TEMPLATE, SC, opname)   \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE(NVSHMEMI_FN_TEMPLATE, SC, opname)
#endif

#define NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX, SC_PREFIX, opname) \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, uchar, unsigned char, opname)                          \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, ushort, unsigned short, opname)                        \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, uint, unsigned int, opname)                            \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, ulong, unsigned long, opname)                          \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, ulonglong, unsigned long long, opname)                 \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, int8, int8_t, opname)                                  \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, int16, int16_t, opname)                                \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, int32, int32_t, opname)                                \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, int64, int64_t, opname)                                \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, uint8, uint8_t, opname)                                \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, uint16, uint16_t, opname)                              \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, uint32, uint32_t, opname)                              \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, uint64, uint64_t, opname)                              \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, size, size_t, opname)

/* Note: The "long double" type is not supported */
#define NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX, SC_PREFIX, opname)    \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX, SC_PREFIX, opname)         \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, char, char, opname)                                                \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, schar, signed char, opname)                                        \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, short, short, opname)                                              \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, int, int, opname)                                                  \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, long, long, opname)                                                \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, longlong, long long, opname)                                       \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, float, float, opname)                                              \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, double, double, opname)

#ifdef NVSHMEM_COMPLEX_SUPPORT
#define NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX, SC_PREFIX, opname)   \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX, SC_PREFIX, opname)    \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, complexf, double complex, opname)                              \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, complexd, float complex, opname)
#else
#define NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX, SC_PREFIX, opname)   \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX, SC_PREFIX, opname)
#endif

#define NVSHMEMI_REPT_TYPES_AND_OPS_FOR_REDUCE(NVSHMEMI_FN_TEMPLATE)    \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, and)   \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, or)    \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, xor)   \
                                                                        \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, min)  \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, max)  \
                                                                        \
    NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, prod)    \
    NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, sum)

#define NVSHMEMI_REPT_TYPES_AND_OPS_FOR_REDUCE_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX, SC_PREFIX)    \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX, SC_PREFIX, and)  \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX, SC_PREFIX, or)   \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX, SC_PREFIX, xor)  \
                                                                        \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX, SC_PREFIX, min) \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX, SC_PREFIX, max) \
                                                                        \
    NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX, SC_PREFIX, prod)   \
    NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX, SC_PREFIX, sum)

#define NVSHMEMI_REPT_TYPES_AND_OPS_AND_SCOPES2_FOR_REDUCE(NVSHMEMI_FN_TEMPLATE)                \
    NVSHMEMI_REPT_TYPES_AND_OPS_FOR_REDUCE_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, thread, , )        \
    NVSHMEMI_REPT_TYPES_AND_OPS_FOR_REDUCE_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, warp, _warp, x)    \
    NVSHMEMI_REPT_TYPES_AND_OPS_FOR_REDUCE_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, block, _block, x)

#define NVSHMEMI_REPT_FOR_SCOPES2(NVSHMEMI_FN_TEMPLATE) \
    NVSHMEMI_FN_TEMPLATE(thread, , )                    \
    NVSHMEMI_FN_TEMPLATE(warp, _warp, x)                \
    NVSHMEMI_FN_TEMPLATE(block, _block, x)
    
#define NVSHMEMI_REPT_FOR_SCOPE(NVSHMEMI_FN_TEMPLATE)   \
    NVSHMEMI_FN_TEMPLATE(thread)                        \
    NVSHMEMI_FN_TEMPLATE(warp)                          \
    NVSHMEMI_FN_TEMPLATE(block)

#define NVSHMEMI_DECL_THREAD_IDX_warp()                         \
    ;                                                           \
    int myIdx;                                                  \
    asm volatile("mov.u32  %0, %laneid;" : "=r"(myIdx));

#define NVSHMEMI_DECL_THREADGROUP_SIZE_warp()                           \
    ;                                                                   \
    int groupSize = ((blockDim.x * blockDim.y * blockDim.z) < warpSize) \
                        ? (blockDim.x * blockDim.y * blockDim.z)        \
                        : warpSize;

#define NVSHMEMI_DECL_THREAD_IDX_block()                        \
    ;                                                           \
    int myIdx = (threadIdx.x + threadIdx.y * blockDim.x +       \
                 threadIdx.z * blockDim.x * blockDim.y);

#define NVSHMEMI_DECL_THREADGROUP_SIZE_block()                  \
    ;                                                           \
    int groupSize = (blockDim.x * blockDim.y * blockDim.z);


#define NVSHMEMI_DECL_THREAD_IDX_thread()                       \
    ;                                                           \
    int myIdx = 0;

#define NVSHMEMI_DECL_THREADGROUP_SIZE_thread()                 \
    ;                                                           \
    int groupSize = 1;

#define NVSHMEMI_SYNC_warp() \
    ;                        \
    __syncwarp();

#define NVSHMEMI_SYNC_block() \
    ;                         \
    __syncthreads();

#define NVSHMEMI_SYNC_thread() ;

enum nvshmemi_call_site_id {
    NVSHMEMI_CALL_SITE_BARRIER = 0,
    NVSHMEMI_CALL_SITE_BARRIER_WARP,
    NVSHMEMI_CALL_SITE_BARRIER_THREADBLOCK,
    NVSHMEMI_CALL_SITE_WAIT_UNTIL_GE,
    NVSHMEMI_CALL_SITE_WAIT_UNTIL_EQ,
    NVSHMEMI_CALL_SITE_WAIT_UNTIL_NE,
    NVSHMEMI_CALL_SITE_WAIT_UNTIL_GT,
    NVSHMEMI_CALL_SITE_WAIT_UNTIL_LT,
    NVSHMEMI_CALL_SITE_WAIT_UNTIL_LE,
    NVSHMEMI_CALL_SITE_WAIT_NE,
    NVSHMEMI_CALL_SITE_PROXY_CHECK_CHANNEL_AVAILABILITY,
    NVSHMEMI_CALL_SITE_PROXY_QUIET,
    NVSHMEMI_CALL_SITE_PROXY_ENFORCE_CONSISTENCY_AT_TARGET,
    NVSHMEMI_CALL_SITE_SIGNAL_WAIT_UNTIL_GE,
    NVSHMEMI_CALL_SITE_SIGNAL_WAIT_UNTIL_EQ,
    NVSHMEMI_CALL_SITE_SIGNAL_WAIT_UNTIL_NE,
    NVSHMEMI_CALL_SITE_SIGNAL_WAIT_UNTIL_GT,
    NVSHMEMI_CALL_SITE_SIGNAL_WAIT_UNTIL_LT,
    NVSHMEMI_CALL_SITE_SIGNAL_WAIT_UNTIL_LE,
    NVSHMEMI_CALL_SITE_AMO_FETCH_WAIT_FLAG,
    NVSHMEMI_CALL_SITE_AMO_FETCH_WAIT_DATA,
};

#define TIMEOUT_NCYCLES 1e10

typedef struct {
    uint64_t signal;
    uint64_t caller;
    uint64_t signal_addr;
    uint64_t signal_val_found;
    uint64_t signal_val_expected;
} nvshmemi_timeout_t;

extern __constant__ int nvshmemi_mype_d;
extern __constant__ int nvshmemi_npes_d;
extern __constant__ int nvshmemi_node_mype_d;
extern __constant__ int nvshmemi_node_npes_d;
extern __constant__ int *nvshmemi_p2p_attrib_native_atomic_support_d;
extern __constant__ int nvshmemi_proxy_d;
extern __constant__ int nvshmemi_atomics_sync_d;
extern __constant__ int nvshmemi_job_connectivity_d;
extern __constant__ bool nvshmemi_proxy_ops_are_ordered_d;
extern __constant__ void *nvshmemi_heap_base_d;
extern __constant__ size_t nvshmemi_heap_size_d;
extern __constant__ void **nvshmemi_peer_heap_base_d;

extern __device__ nvshmemi_timeout_t *nvshmemi_timeout_d;
extern __device__ unsigned long long test_wait_any_start_idx_d;
__device__ void nvshmemi_proxy_enforce_consistency_at_target();

template <typename T>
__device__ inline void nvshmemi_check_timeout_and_log(long long int start, int caller, uintptr_t signal_addr,
                                             T signal_val_found, T signal_val_expected) {
    long long int now;
    asm volatile("mov.u64  %0, %globaltimer;" : "=l"(now));
    if ((now - start) > TIMEOUT_NCYCLES) {
        nvshmemi_timeout_t *timeout_d = nvshmemi_timeout_d;
        timeout_d->caller = caller;
        timeout_d->signal_addr = signal_addr;
        *(T *)(&timeout_d->signal_val_found) = signal_val_found;
        *(T *)(&timeout_d->signal_val_expected) = signal_val_expected;
        *((volatile uint64_t *)(&timeout_d->signal)) = 1;
    }
}

template <typename T>
__device__ inline void nvshmemi_wait_until_greater_than(volatile T *addr, T val, int caller) {
#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
    long long int start;
    asm volatile("mov.u64  %0, %globaltimer;" : "=l"(start));
#endif
    while (*addr <= val) {
#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
        nvshmemi_check_timeout_and_log<T>(start, caller, (uintptr_t)addr, *addr, val);
#endif
    }
}

template <typename T>
__device__ inline void nvshmemi_wait_until_greater_than_equals(volatile T *addr, T val, int caller) {
#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
    long long int start;
    asm volatile("mov.u64  %0, %globaltimer;" : "=l"(start));
#endif
    while (*addr < val) {
#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
        nvshmemi_check_timeout_and_log<T>(start, caller, (uintptr_t)addr, *addr, val);
#endif
    }
}

template <typename T>
__device__ inline void nvshmemi_wait_until_lesser_than(volatile T *addr, T val, int caller) {
#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
    long long int start;
    asm volatile("mov.u64  %0, %globaltimer;" : "=l"(start));
#endif
    while (*addr >= val) {
#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
        nvshmemi_check_timeout_and_log<T>(start, caller, (uintptr_t)addr, *addr, val);
#endif
    }
}

template <typename T>
__device__ inline void nvshmemi_wait_until_lesser_than_equals(volatile T *addr, T val, int caller) {
#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
    long long int start;
    asm volatile("mov.u64  %0, %globaltimer;" : "=l"(start));
#endif
    while (*addr > val) {
#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
        nvshmemi_check_timeout_and_log<T>(start, caller, (uintptr_t)addr, *addr, val);
#endif
    }
}

template <typename T>
__device__ inline void nvshmemi_wait_until_equals(volatile T *addr, T val, int caller) {
#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
    long long int start;
    asm volatile("mov.u64  %0, %globaltimer;" : "=l"(start));
#endif
    while (*addr != val) {
#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
        nvshmemi_check_timeout_and_log<T>(start, caller, (uintptr_t)addr, *addr, val);
#endif
    }
}

template <typename T>
__device__ inline void nvshmemi_wait_until_not_equals(volatile T *addr, T val, int caller) {
#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
    long long int start;
    asm volatile("mov.u64  %0, %globaltimer;" : "=l"(start));
#endif
    while (*addr == val) {
#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
        nvshmemi_check_timeout_and_log<T>(start, caller, (uintptr_t)addr, *addr, val);
#endif
    }
}

template <typename T>
__device__ inline void nvshmemi_wait_until_greater_than_equals_add(volatile T *addr, uint64_t toadd, T val,
                                                          int caller) {
#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
    long long int start;
    asm volatile("mov.u64  %0, %globaltimer;" : "=l"(start));
#endif
    T valataddr;
    do {
        valataddr = *addr;
#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
        nvshmemi_check_timeout_and_log<T>(start, caller, (uintptr_t)addr, valataddr + toadd, val);
#endif
    } while (valataddr + toadd < val);
}

template <typename T>
__device__ inline int nvshmemi_test(volatile T *ivar, int cmp, T cmp_value) {
    int return_value = 0;
    if (NVSHMEM_CMP_GE == cmp) {
        if (*ivar >= cmp_value) return_value = 1;
    } else if (NVSHMEM_CMP_EQ == cmp) {
        if (*ivar == cmp_value) return_value = 1;
    } else if (NVSHMEM_CMP_NE == cmp) {
        if (*ivar != cmp_value) return_value = 1;
    } else if (NVSHMEM_CMP_GT == cmp) {
        if (*ivar > cmp_value) return_value = 1;
    } else if (NVSHMEM_CMP_LT == cmp) {
        if (*ivar < cmp_value) return_value = 1;
    } else if (NVSHMEM_CMP_LE == cmp) {
        if (*ivar <= cmp_value) return_value = 1;
    }
    return return_value;
}

template <typename T>
__device__ inline void nvshmemi_wait_until(volatile T *ivar, int cmp, T cmp_value) {
    if (NVSHMEM_CMP_GE == cmp) {
        nvshmemi_wait_until_greater_than_equals<T>(ivar, cmp_value, NVSHMEMI_CALL_SITE_WAIT_UNTIL_GE);
    } else if (NVSHMEM_CMP_EQ == cmp) {
        nvshmemi_wait_until_equals<T>(ivar, cmp_value, NVSHMEMI_CALL_SITE_WAIT_UNTIL_EQ);
    } else if (NVSHMEM_CMP_NE == cmp) {
        nvshmemi_wait_until_not_equals<T>(ivar, cmp_value, NVSHMEMI_CALL_SITE_WAIT_UNTIL_NE);
    } else if (NVSHMEM_CMP_GT == cmp) {
        nvshmemi_wait_until_greater_than<T>(ivar, cmp_value, NVSHMEMI_CALL_SITE_WAIT_UNTIL_GT);
    } else if (NVSHMEM_CMP_LT == cmp) {
        nvshmemi_wait_until_lesser_than<T>(ivar, cmp_value, NVSHMEMI_CALL_SITE_WAIT_UNTIL_LT);
    } else if (NVSHMEM_CMP_LE == cmp) {
        nvshmemi_wait_until_lesser_than_equals<T>(ivar, cmp_value, NVSHMEMI_CALL_SITE_WAIT_UNTIL_LE);
    }
}

#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
#define NVSHMEM_WAIT_UNTIL_RETURN_LOOP_BODY(site)                                                                \
    local_signal = *signal;                                                                                      \
    nvshmemi_check_timeout_and_log<uint64_t>(start, site, (uintptr_t)signal, local_signal, cmp_value);
#else
#define NVSHMEM_WAIT_UNTIL_RETURN_LOOP_BODY(site) local_signal = *signal;
#endif

__device__ inline uint64_t nvshmemi_signal_wait_until(volatile uint64_t *signal, int cmp, uint64_t cmp_value) {
    uint64_t local_signal;
#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
    long long int start;
    asm volatile("mov.u64  %0, %globaltimer;" : "=l"(start));
#endif

    switch (cmp) {
        case NVSHMEM_CMP_GE:
            do {
                NVSHMEM_WAIT_UNTIL_RETURN_LOOP_BODY(NVSHMEMI_CALL_SITE_SIGNAL_WAIT_UNTIL_GE);
            } while (local_signal < cmp_value);
            break;
        case NVSHMEM_CMP_EQ:
            do {
                NVSHMEM_WAIT_UNTIL_RETURN_LOOP_BODY(NVSHMEMI_CALL_SITE_SIGNAL_WAIT_UNTIL_EQ);
            } while (local_signal != cmp_value);
            break;
        case NVSHMEM_CMP_NE:
            do {
                NVSHMEM_WAIT_UNTIL_RETURN_LOOP_BODY(NVSHMEMI_CALL_SITE_SIGNAL_WAIT_UNTIL_NE);
            } while (local_signal == cmp_value);
            break;
        case NVSHMEM_CMP_GT:
            do {
                NVSHMEM_WAIT_UNTIL_RETURN_LOOP_BODY(NVSHMEMI_CALL_SITE_SIGNAL_WAIT_UNTIL_GT);
            } while (local_signal <= cmp_value);
            break;
        case NVSHMEM_CMP_LT:
            do {
                NVSHMEM_WAIT_UNTIL_RETURN_LOOP_BODY(NVSHMEMI_CALL_SITE_SIGNAL_WAIT_UNTIL_LT);
            } while (local_signal >= cmp_value);
            break;
        case NVSHMEM_CMP_LE:
            do {
                NVSHMEM_WAIT_UNTIL_RETURN_LOOP_BODY(NVSHMEMI_CALL_SITE_SIGNAL_WAIT_UNTIL_LE);
            } while (local_signal > cmp_value);
            break;
    }
    return local_signal;
}

#ifdef __CUDA_ARCH__
__device__ inline void nvshmemi_syncapi_update_mem() {
    __threadfence(); /* 1. Ensures consitency op is not called before the prior test/wait condition has been met
                        2. Needed to prevent reorder of instructions after sync api (when the following if condition is false) */
    if (nvshmemi_job_connectivity_d > NVSHMEMI_JOB_GPU_PROXY) { 
       nvshmemi_proxy_enforce_consistency_at_target();
    }	
}
#endif
#endif
