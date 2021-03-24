/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmemx_error.h"
#include "util.h"
#include "gpu_coll.h"
#include "nvshmem_internal.h"

__device__ gpu_coll_env_params_t gpu_coll_env_params_var_d;


__device__ int reduce_recexch_step1_sendto_d;
__device__ int *reduce_recexch_step1_recvfrom_d;
__device__ int reduce_recexch_step1_nrecvs_d;
__device__ int **reduce_recexch_step2_nbrs_d;
__device__ int reduce_recexch_step2_nphases_d;
__device__ int reduce_recexch_p_of_k_d;
__device__ int reduce_recexch_reduce_recexch_digit_d;
__device__ int *digit_d;

extern "C" int init_shm_kernel_shm_ptr() {
    int status = 0;

    int *step1_recvfrom = NULL, **step2_nbrs = NULL;
    int *digit = NULL;
    int k, max_phases;

    status = cudaMemcpyToSymbol(gpu_coll_env_params_var_d, &gpu_coll_env_params_var,
                                sizeof(gpu_coll_env_params_t));
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "memcopy to symbol failed \n");

    /* Allocate memory for performing reduce recursive exchange algorithm */
    k = gpu_coll_env_params_var.reduce_recexch_kval;
    assert(k > 1);

    status = cuMemAlloc((CUdeviceptr *) &step1_recvfrom, sizeof(int) * (k - 1));
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cudaMalloc failed\n");

    status = cudaMemcpyToSymbol(reduce_recexch_step1_recvfrom_d, &step1_recvfrom, sizeof(void *));
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "memcopy to symbol failed \n");

    max_phases = log(nvshmemi_state->npes) / log(k) + 1; /* The '+ 1' makes it a conservative calculation, max_pahses >= 1 */

    status = cuMemAlloc((CUdeviceptr *) &step2_nbrs, sizeof(int *) * max_phases);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cudaMalloc failed\n");
    status = cuMemAlloc((CUdeviceptr *) &digit, sizeof(int) * max_phases);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cudaMalloc failed\n");

    for (int i = 0; i < max_phases; i++) {
        void *dev_ptr;
        status = cudaMalloc(&dev_ptr, sizeof(int) * (k - 1));
        NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cudaMalloc failed\n");
        status = cuMemcpyHtoD((CUdeviceptr)((int **)step2_nbrs + i), &dev_ptr, sizeof(int *));
        NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cuMemcpyHtoD failed\n");
    }

    status = cudaMemcpyToSymbol(reduce_recexch_step2_nbrs_d, &step2_nbrs, sizeof(void**));
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "memcopy to symbol failed \n");
    status = cudaMemcpyToSymbol(digit_d, &digit, sizeof(int *));
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "memcopy to symbol failed \n");

    nvshmemi_recexchalgo_get_neighbors(nvshmemi_state->mype, nvshmemi_state->npes);

    CUDA_CHECK(cuStreamSynchronize(0));

    goto fn_out;
out:
fn_out:
    return status;
}
