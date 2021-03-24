/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmem_internal.h"

#include <stdio.h>
#include "nvshmemx_error.h"
#include "util.h"
#include <algorithm>

extern int (*nvshmemi_transport_init_op[NVSHMEM_TRANSPORT_COUNT])(nvshmem_transport_t *transport);

void *heap_base_array_dptr = NULL;
void *heap_base_actual_array_dptr = NULL;
int *p2p_attrib_native_atomic_support_array_dptr = NULL;
int nvshmemi_job_connectivity;
__constant__ int nvshmemi_mype_d;
__constant__ int nvshmemi_npes_d;
__constant__ int nvshmemi_node_mype_d;
__constant__ int nvshmemi_node_npes_d;
__constant__ void **nvshmemi_peer_heap_base_d;
__constant__ void **nvshmem_peer_heap_base_actual_d;
__constant__ void *nvshmemi_heap_base_d;
__constant__ size_t nvshmemi_heap_size_d;
__constant__ int *nvshmemi_p2p_attrib_native_atomic_support_d;
__constant__ int nvshmemi_proxy_d;
__constant__ int nvshmemi_atomics_sync_d;
__constant__ int nvshmemi_job_connectivity_d;
__constant__ bool nvshmemi_proxy_ops_are_ordered_d;
__constant__ int barrier_dissem_kval_d;
__constant__ int barrier_tg_dissem_kval_d;
__device__ unsigned long long test_wait_any_start_idx_d;

int set_job_connectivity (nvshmemi_state_t *state) {
    int status;
    int *job_connectivity_all; 
    bool proxy_ops_are_ordered = true;

    status = cuMemAlloc((CUdeviceptr *)&p2p_attrib_native_atomic_support_array_dptr,
                        (state->npes) * sizeof(int));
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                 "device p2p native atomic support flag array allocation failed \n");

    state->p2p_attrib_native_atomic_support = (int *)calloc(state->npes, sizeof(int));
    NULL_ERROR_JMP(state->p2p_attrib_native_atomic_support, status, NVSHMEMX_ERROR_OUT_OF_MEMORY,
                   out, "memory allocation for atomic support array \n");

    // detrmine job level connectivity among GPUs
    nvshmemi_job_connectivity = NVSHMEMI_JOB_GPU_LDST_ATOMICS;
    for (int i = 0; i < state->npes; i++) {
        int peer_connectivity = NVSHMEMI_JOB_GPU_PROXY;
        void *enforce_cst = NULL;
        // for each PE, pick the best connectivity of any transport
        for (int j = 0; j < NVSHMEM_TRANSPORT_COUNT; j++) {
            if (state->transports[j]) {
                if (state->transports[j]->cap[i] & NVSHMEM_TRANSPORT_CAP_MAP_GPU_ATOMICS) {
                    state->p2p_attrib_native_atomic_support[i] = 1;
                    peer_connectivity = (int)NVSHMEMI_JOB_GPU_LDST_ATOMICS;
                } else if (state->transports[j]->cap[i] &
                           (NVSHMEM_TRANSPORT_CAP_MAP_GPU_ST | NVSHMEM_TRANSPORT_CAP_MAP_GPU_LD)) {
                    peer_connectivity = std::min(peer_connectivity, (int)NVSHMEMI_JOB_GPU_LDST);
                } else {
                    peer_connectivity = std::min(peer_connectivity, (int)NVSHMEMI_JOB_GPU_PROXY);
                    enforce_cst = (void *)state->transports[j]->host_ops.enforce_cst_at_target;
                }
            }
        }
        if ((peer_connectivity == NVSHMEMI_JOB_GPU_PROXY) && (enforce_cst)) {
            peer_connectivity = NVSHMEMI_JOB_GPU_PROXY_CST;
        }
        // for the job, pick the weakest connecitivity to any remote PEs
        nvshmemi_job_connectivity = std::max(nvshmemi_job_connectivity, peer_connectivity);
    }

    //agree on maximumg distance for job_connectivity among all PEs
    job_connectivity_all = (int *)malloc(sizeof(int) * state->npes);
    NULL_ERROR_JMP(job_connectivity_all, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                   "memory allocation for job_connecitivty_all failed \n");

    status = state->boot_handle.allgather((void *)&nvshmemi_job_connectivity, (void *)job_connectivity_all, 
                   sizeof(int), &state->boot_handle);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "allgather of job_connectivity failed \n");
    for (int i = 0; i < state->npes; i++) {
        nvshmemi_job_connectivity = std::max(nvshmemi_job_connectivity, job_connectivity_all[i]);
    }
    free(job_connectivity_all);

    status = cuMemcpyHtoDAsync((CUdeviceptr)p2p_attrib_native_atomic_support_array_dptr,
                               (const int *)state->p2p_attrib_native_atomic_support,
                               sizeof(int) * state->npes, state->my_stream);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                 "p2p native atomic support flag array initialization failed \n");

    status = cudaMemcpyToSymbolAsync(nvshmemi_job_connectivity_d, &nvshmemi_job_connectivity,
                                     sizeof(int), 0, cudaMemcpyHostToDevice,
                                     (cudaStream_t)state->my_stream);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "memcopy to symbol failed \n");

    // check if all proxy ops are ordered
    for (int i = 0; i < NVSHMEM_TRANSPORT_COUNT; i++) {
        if (state->transports[i] && (state->transports[i]->host_ops.fence != NULL))
            proxy_ops_are_ordered = false;
    }
    status = cudaMemcpyToSymbol(nvshmemi_proxy_ops_are_ordered_d, &proxy_ops_are_ordered,
                                     sizeof(bool), 0, cudaMemcpyHostToDevice);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "memcopy to symbol failed \n");

out:
    return status;
}

int nvshmemi_init_device_state(nvshmemi_state_t *state) {
    int status = CUDA_SUCCESS;
    int dev_count;
    pcie_id_t *pcie_ids = NULL;
    CUdevice *cudev = NULL;
    int use_proxy = 0;
    int atomics_sync = 0;
    int zero = 0;

    status = cuMemAlloc((CUdeviceptr *)&heap_base_array_dptr, (state->npes) * sizeof(void *));
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                 "device peer heap base allocation failed \n");

    status =
        cuMemAlloc((CUdeviceptr *)&heap_base_actual_array_dptr, (state->npes) * sizeof(void *));
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                 "device peer heap base actual allocation failed \n");

    /*P2P specific*/
    status = cuDeviceGetCount(&dev_count);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cuDeviceGetCount failed \n");

    cudev = (CUdevice *)malloc(sizeof(CUdevice) * dev_count);
    NULL_ERROR_JMP(cudev, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                   "memory allocation for cudev failed \n");

    pcie_ids = (pcie_id_t *)malloc(sizeof(pcie_id_t) * dev_count);
    NULL_ERROR_JMP(pcie_ids, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                   "memory allocation for pcie_id failed \n");

    for (int i = 0; i < dev_count; i++) {
        status = cuDeviceGet(&cudev[i], i);
        NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cuDeviceGet failed \n");

        status = nvshmemi_get_pcie_attrs(&pcie_ids[i], cudev[i]);
        NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                     "nvshmemi_get_pcie_attrs failed \n");
    }

    status = set_job_connectivity(state);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "set_job_connectivity failed \n");

    status =
        cuMemcpyHtoDAsync((CUdeviceptr)heap_base_array_dptr, (const void *)state->peer_heap_base,
                          sizeof(void *) * state->npes, state->my_stream);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                 "peer heap base initialization failed \n");

    status = cuMemcpyHtoDAsync((CUdeviceptr)heap_base_actual_array_dptr,
                               (const void *)state->peer_heap_base_actual,
                               sizeof(void *) * state->npes, state->my_stream);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                 "peer heap base actual initialization failed \n");

    status = cuStreamSynchronize(state->my_stream);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                 " stream synchronize failed\n");

    INFO(NVSHMEM_INIT,
         "[%d] heap_base_array_dptr %p p2p_attrib_native_atomic_support_array_dptr %p", state->mype,
         heap_base_array_dptr, p2p_attrib_native_atomic_support_array_dptr);

    if (state->transports[NVSHMEM_TRANSPORT_ID_IBRC] &&
        nvshmemi_transport_init_op[NVSHMEM_TRANSPORT_ID_IBRC] &&
        state->transports[NVSHMEM_TRANSPORT_ID_IBRC]->is_successfully_initialized &&
        nvshmemi_job_connectivity >= NVSHMEMI_JOB_GPU_LDST)
        use_proxy = 1;

    status = cudaMemcpyToSymbolAsync(nvshmemi_proxy_d, &use_proxy, sizeof(int), 0,
                                     cudaMemcpyHostToDevice, (cudaStream_t)state->my_stream);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "memcopy to symbol failed \n");

    if (nvshmemi_options.ASSERT_ATOMICS_SYNC) atomics_sync = 1;

    status = cudaMemcpyToSymbolAsync(nvshmemi_atomics_sync_d, &atomics_sync, sizeof(int), 0,
                                     cudaMemcpyHostToDevice, (cudaStream_t)state->my_stream);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "memcopy to symbol failed \n");

    status =
        cudaMemcpyToSymbolAsync(nvshmemi_peer_heap_base_d, &heap_base_array_dptr, sizeof(void *), 0,
                                cudaMemcpyHostToDevice, (cudaStream_t)state->my_stream);

    INFO(NVSHMEM_INIT,
         "[%d] status %d cudaErrorInvalidValue %d cudaErrorInvalidSymbol %d "
         "cudaErrorInvalidMemcpyDirection %d cudaErrorNoKernelImageForDevice %d",
         state->mype, status, cudaErrorInvalidValue, cudaErrorInvalidSymbol,
         cudaErrorInvalidMemcpyDirection, cudaErrorNoKernelImageForDevice);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "memcopy to symbol failed \n");

    status = cudaMemcpyToSymbolAsync(nvshmem_peer_heap_base_actual_d, &heap_base_actual_array_dptr,
                                     sizeof(void *), 0, cudaMemcpyHostToDevice,
                                     (cudaStream_t)state->my_stream);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "memcopy to symbol failed \n");

    status = cudaMemcpyToSymbolAsync(nvshmemi_p2p_attrib_native_atomic_support_d,
                                     &p2p_attrib_native_atomic_support_array_dptr, sizeof(int *), 0,
                                     cudaMemcpyHostToDevice, (cudaStream_t)state->my_stream);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "memcopy to symbol failed \n");

    status = cudaMemcpyToSymbolAsync(nvshmemi_heap_base_d, &state->heap_base, sizeof(void *), 0,
                                     cudaMemcpyHostToDevice, (cudaStream_t)state->my_stream);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "memcopy to symbol failed \n");

    status = cudaMemcpyToSymbolAsync(nvshmemi_heap_size_d, &state->heap_size, sizeof(size_t), 0,
                                     cudaMemcpyHostToDevice, (cudaStream_t)state->my_stream);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "memcopy to symbol failed \n");

    status = cudaMemcpyToSymbolAsync(nvshmemi_mype_d, &state->mype, sizeof(int), 0,
                                     cudaMemcpyHostToDevice, (cudaStream_t)state->my_stream);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "memcopy to symbol failed \n");

    status = cudaMemcpyToSymbolAsync(nvshmemi_npes_d, &state->npes, sizeof(int), 0,
                                     cudaMemcpyHostToDevice, (cudaStream_t)state->my_stream);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "memcopy to symbol failed \n");

    status = cudaMemcpyToSymbolAsync(nvshmemi_node_mype_d, &state->mype_node, sizeof(int), 0,
                                     cudaMemcpyHostToDevice, (cudaStream_t)state->my_stream);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "memcopy to symbol failed \n");

    status = cudaMemcpyToSymbolAsync(nvshmemi_node_npes_d, &state->npes_node, sizeof(int), 0,
                                     cudaMemcpyHostToDevice, (cudaStream_t)state->my_stream);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "memcopy to symbol failed \n");

    status = cudaMemcpyToSymbolAsync(barrier_dissem_kval_d, &nvshmemi_options.BARRIER_DISSEM_KVAL,
                                     sizeof(int), 0, cudaMemcpyHostToDevice,
                                     (cudaStream_t)state->my_stream);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "memcopy to symbol failed \n");

    status = cudaMemcpyToSymbolAsync(barrier_tg_dissem_kval_d,
                                     &nvshmemi_options.BARRIER_TG_DISSEM_KVAL, sizeof(int), 0,
                                     cudaMemcpyHostToDevice, (cudaStream_t)state->my_stream);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "memcopy to symbol failed \n");

    status = cuStreamSynchronize(state->my_stream);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                 " stream synchronize failed\n");

    cudaMemcpyToSymbol(test_wait_any_start_idx_d, &zero, sizeof(int), 0);

out:
    if (status) {
        if (heap_base_array_dptr) free(heap_base_array_dptr);
        if (heap_base_actual_array_dptr) free(heap_base_actual_array_dptr);
        if (cudev) free(cudev);
        if (pcie_ids) free(pcie_ids);
    }
    return status;
}
