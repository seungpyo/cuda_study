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

#ifndef _NVSHMEM_CONSTANTS_H_
#define _NVSHMEM_CONSTANTS_H_

#include "nvshmemi_constants.h"

/* This is not the NVSHMEM release version, it is the supported OpenSHMEM spec version. */
#define NVSHMEM_MAJOR_VERSION 1
#define NVSHMEM_MINOR_VERSION 3
#define _NVSHMEM_MAJOR_VERSION NVSHMEM_MAJOR_VERSION
#define _NVSHMEM_MINOR_VERSION NVSHMEM_MINOR_VERSION

#define NVSHMEM_VENDOR_STRING "NVSHMEM v2.0.2"
#define _NVSHMEM_VENDOR_STRING NVSHMEM_VENDOR_STRING

#define NVSHMEM_MAX_NAME_LEN 256
#define _NVSHMEM_MAX_NAME_LEN NVSHMEM_MAX_NAME_LEN

enum nvshmemi_cmp_type {
    NVSHMEM_CMP_EQ = 0,
    NVSHMEM_CMP_NE,
    NVSHMEM_CMP_GT,
    NVSHMEM_CMP_LE,
    NVSHMEM_CMP_LT,
    NVSHMEM_CMP_GE
};

enum nvshmemi_thread_support {
    NVSHMEM_THREAD_SINGLE = 0,
    NVSHMEM_THREAD_FUNNELED,
    NVSHMEM_THREAD_SERIALIZED,
    NVSHMEM_THREAD_MULTIPLE
};

enum nvshmem_signal_ops {
    NVSHMEM_SIGNAL_SET = NVSHMEMI_AMO_SIGNAL_SET,
    NVSHMEM_SIGNAL_ADD = NVSHMEMI_AMO_SIGNAL_ADD
};

#endif
