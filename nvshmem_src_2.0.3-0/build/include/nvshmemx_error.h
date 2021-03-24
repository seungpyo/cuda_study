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

#ifndef _NVSHMEMX_ERROR_H_
#define _NVSHMEMX_ERROR_H_

#ifdef __cplusplus
extern "C" {
#endif

enum nvshmemx_status {
    NVSHMEMX_SUCCESS = 0,
    NVSHMEMX_ERROR_INVALID_VALUE,
    NVSHMEMX_ERROR_OUT_OF_MEMORY,
    NVSHMEMX_ERROR_NOT_SUPPORTED,
    NVSHMEMX_ERROR_SYMMETRY,
    NVSHMEMX_ERROR_GPU_NOT_SELECTED,
    NVSHMEMX_ERROR_COLLECTIVE_LAUNCH_FAILED,
    NVSHMEMX_ERROR_INTERNAL
};

#ifdef __cplusplus
}
#endif

#endif
