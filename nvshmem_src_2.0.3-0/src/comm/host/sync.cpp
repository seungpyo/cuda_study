/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "nvshmem_internal.h"

#include "cuda_interface_sync.h"

#define NVSHMEM_TYPE_WAIT(type, TYPE)                                                           \
    void nvshmem_##type##_wait(TYPE *ivar, TYPE cmp_value) {                                    \
        /*SHMEM_CHECK_STATE_AND_INIT();*/                                                       \
        int status = 0;                                                                         \
        status = cuda_interface_##type##_wait(ivar, cmp_value);                                 \
        /*status = cuLaunchKernel (nvshmemi_state->cufunction_wait_"#type", 1, 1, 1, 1, 1, 1, 0, \
         * nvshmemi_state->my_stream, kernelParams, NULL);*/                                     \
        if (status) {                                                                           \
            ERROR_PRINT("[%d] cudaLaunchKernel()/shmem_" #type "_wait() failed",                \
                        nvshmemi_state->mype);                                                   \
            goto out;                                                                           \
        }                                                                                       \
        status = cuStreamSynchronize(nvshmemi_state->my_stream);                                 \
        if (status) {                                                                           \
            ERROR_PRINT("[%d] cuStreamSynchronize()/shmem_" #type "_wait() failed",             \
                        nvshmemi_state->mype);                                                   \
            goto out;                                                                           \
        }                                                                                       \
    out:                                                                                        \
        return;                                                                                 \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEM_TYPE_WAIT)
#undef NVSHMEM_TYPE_WAIT

#define NVSHMEMX_TYPE_WAIT_ON_STREAM(type, TYPE)                                            \
    void nvshmemx_##type##_wait_on_stream(TYPE *ivar, TYPE cmp_value,                       \
                                          cudaStream_t cstream) {                           \
        int status = 0;                                                                     \
        status = cuda_interface_##type##_wait_on_stream(ivar, cmp_value, cstream);          \
        if (status) {                                                                       \
            ERROR_PRINT("[%d] cudaLaunchKernel()/shmemx_" #type "_wait_on_stream() failed", \
                        nvshmemi_state->mype);                                               \
            goto out;                                                                       \
        }                                                                                   \
    out:                                                                                    \
        return;                                                                             \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMX_TYPE_WAIT_ON_STREAM)
#undef NVSHMEMX_TYPE_WAIT_ON_STREAM

void nvshmem_wait(long *ivar, long cmp_value) { /*wait until *ivar != cmp_value*/
    int status = 0;
    status = cuda_interface_long_wait(ivar, cmp_value);
    if (status) {
        ERROR_PRINT("[%d] cudaLaunchKernel()/shmem_wait() failed %d", nvshmemi_state->mype, status);
        goto out;
    }
    status = cuStreamSynchronize(nvshmemi_state->my_stream);
    if (status) {
        ERROR_PRINT("[%d] cuStreamSynchronize()/shmem_wait() failed", nvshmemi_state->mype);
        goto out;
    }
out:
    return;
}

void nvshmemx_wait_on_stream(long *ivar, long cmp_value,
                             cudaStream_t cstream) { /*wait until *ivar != cmp_value*/
    int status = 0;
    status = cuda_interface_long_wait_on_stream(ivar, cmp_value, cstream);
    if (status) {
        ERROR_PRINT("[%d] cudaLaunchKernel()/shmemx_wait_on_stream() failed", nvshmemi_state->mype);
        goto out;
    }
out:
    return;
}

#define NVSHMEM_TYPE_WAIT_UNTIL(type, TYPE)                                                       \
    void nvshmem_##type##_wait_until(TYPE *ivar, int cmp,                                         \
                                     TYPE cmp_value) { /*wait until *ivar cmp cmp_value == true*/ \
        int status = 0;                                                                           \
        switch (cmp) {                                                                            \
            case NVSHMEM_CMP_EQ:                                                                  \
                break;                                                                            \
            case NVSHMEM_CMP_NE:                                                                  \
                break;                                                                            \
            case NVSHMEM_CMP_GT:                                                                  \
                break;                                                                            \
            case NVSHMEM_CMP_LE:                                                                  \
                break;                                                                            \
            case NVSHMEM_CMP_LT:                                                                  \
                break;                                                                            \
            case NVSHMEM_CMP_GE:                                                                  \
                break;                                                                            \
            default:                                                                              \
                ERROR_PRINT("[%d] Invalid comparator/shmem_" #type "_wait_until()",               \
                            nvshmemi_state->mype);                                                 \
                goto out;                                                                         \
        }                                                                                         \
        status = cuda_interface_##type##_wait_until(ivar, cmp, cmp_value);                        \
        if (status) {                                                                             \
            ERROR_PRINT("[%d] cudaLaunchKernel()/shmem_" #type "_wait_until() failed",            \
                        nvshmemi_state->mype);                                                     \
            goto out;                                                                             \
        }                                                                                         \
        status = cuStreamSynchronize(nvshmemi_state->my_stream);                                   \
        if (status) {                                                                             \
            ERROR_PRINT("[%d] cuStreamSynchronize()/shmem_" #type "_wait_until() failed",         \
                        nvshmemi_state->mype);                                                     \
            goto out;                                                                             \
        }                                                                                         \
    out:                                                                                          \
        return;                                                                                   \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEM_TYPE_WAIT_UNTIL)
#undef NVSHMEM_TYPE_WAIT_UNTIL


#define NVSHMEMX_TYPE_WAIT_UNTIL_ON_STREAM(type, TYPE)                                            \
    void nvshmemx_##type##_wait_until_on_stream(                                                  \
        TYPE *ivar, int cmp, TYPE cmp_value,                                                      \
        cudaStream_t cstream) { /*wait until *ivar cmp cmp_value == true*/                        \
        int status = 0;                                                                           \
        switch (cmp) {                                                                            \
            case NVSHMEM_CMP_EQ:                                                                  \
                break;                                                                            \
            case NVSHMEM_CMP_NE:                                                                  \
                break;                                                                            \
            case NVSHMEM_CMP_GT:                                                                  \
                break;                                                                            \
            case NVSHMEM_CMP_LE:                                                                  \
                break;                                                                            \
            case NVSHMEM_CMP_LT:                                                                  \
                break;                                                                            \
            case NVSHMEM_CMP_GE:                                                                  \
                break;                                                                            \
            default:                                                                              \
                ERROR_PRINT("[%d] Invalid comparator/shmem_" #type "_wait_until()",               \
                            nvshmemi_state->mype);                                                 \
                goto out;                                                                         \
        }                                                                                         \
        status = cuda_interface_##type##_wait_until_on_stream(ivar, cmp, cmp_value, cstream);     \
        if (status) {                                                                             \
            ERROR_PRINT("[%d] cudaLaunchKernel()/shmemx_" #type "_wait_until_on_stream() failed", \
                        nvshmemi_state->mype);                                                     \
            goto out;                                                                             \
        }                                                                                         \
    out:                                                                                          \
        return;                                                                                   \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMX_TYPE_WAIT_UNTIL_ON_STREAM)
#undef NVSHMEMX_TYPE_WAIT_UNTIL_ON_STREAM

void nvshmem_wait_until(long *ivar, int cmp,
                        long cmp_value) { /*wait until *ivar cmp cmp_value == true*/
    int status = 0;
    switch (cmp) {
        case NVSHMEM_CMP_EQ:
            break;
        case NVSHMEM_CMP_NE:
            break;
        case NVSHMEM_CMP_GT:
            break;
        case NVSHMEM_CMP_LE:
            break;
        case NVSHMEM_CMP_LT:
            break;
        case NVSHMEM_CMP_GE:
            break;
        default:
            ERROR_PRINT("[%d] Invalid comparator/shmem_wait_until()", nvshmemi_state->mype);
            goto out;
    }
    status = cuda_interface_long_wait_until(ivar, cmp, cmp_value);
    if (status) {
        ERROR_PRINT("[%d] cudaLaunchKernel()/shmem_wait_until() failed", nvshmemi_state->mype);
        goto out;
    }
    status = cuStreamSynchronize(nvshmemi_state->my_stream);
    if (status) {
        ERROR_PRINT("[%d] cuStreamSynchronize()/shmem_wait_until() failed", nvshmemi_state->mype);
        goto out;
    }
out:
    return;
}

void nvshmemx_wait_until_on_stream(
    long *ivar, int cmp, long cmp_value,
    cudaStream_t cstream) { /*wait until *ivar cmp cmp_value == true*/
    int status = 0;
    switch (cmp) {
        case NVSHMEM_CMP_EQ:
            break;
        case NVSHMEM_CMP_NE:
            break;
        case NVSHMEM_CMP_GT:
            break;
        case NVSHMEM_CMP_LE:
            break;
        case NVSHMEM_CMP_LT:
            break;
        case NVSHMEM_CMP_GE:
            break;
        default:
            ERROR_PRINT("[%d] Invalid comparator/shmem_wait_until()", nvshmemi_state->mype);
            goto out;
    }
    status = cuda_interface_long_wait_until_on_stream(ivar, cmp, cmp_value, cstream);
    if (status) {
        ERROR_PRINT("[%d] cudaLaunchKernel()/shmemx_wait_until_on_stream() failed",
                    nvshmemi_state->mype);
        goto out;
    }
out:
    return;
}
