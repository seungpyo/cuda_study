/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "nvshmem_internal.h"

__host__ __device__ int nvshmem_my_pe(void) {
#ifdef __CUDA_ARCH__
    return nvshmemi_mype_d;
#else
    return nvshmemi_state->mype;
#endif
}

__host__ __device__ int nvshmem_n_pes(void) {
#ifdef __CUDA_ARCH__
    return nvshmemi_npes_d;
#else
    return nvshmemi_state->npes;
#endif
}

__host__ __device__ void nvshmem_info_get_name(char *name) {
    size_t i;
    const char *str = NVSHMEM_VENDOR_STRING;

    /* Copy up to NVSHMEM_MAX_NAME_LEN-1 chars, then add NULL terminator */
    for (i = 0; i < NVSHMEM_MAX_NAME_LEN-1 && str[i] != '\0'; i++)
        name[i] = str[i];

    name[i] = '\0';
}

__host__ __device__ void nvshmem_info_get_version(int *major, int *minor) {
    *major = NVSHMEM_MAJOR_VERSION;
    *minor = NVSHMEM_MINOR_VERSION;
}

__host__ __device__ int nvshmemx_my_pe(nvshmemx_team_t team) {
    return nvshmem_team_my_pe((nvshmem_team_t) team);
}

__host__ __device__ int nvshmemx_n_pes(nvshmemx_team_t team) {
    return nvshmem_team_n_pes((nvshmem_team_t) team);
}
