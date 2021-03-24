/****
 * Copyright (c) 2016-2018, NVIDIA Corporation.  All rights reserved.
 *
 * See COPYRIGHT for license information
 ****/

#include "nvshmem.h"
#include "nvshmemx.h"
#include "nvshmem_internal.h"

#include <iostream>
#include <cassert>
#include "util.h"

int nvshmemi_setup_collective_launch(nvshmemi_state_t *state) {
    int leastPriority, greatestPriority, status = 0;
    status = cuDeviceGetAttribute(&(state->cu_dev_attrib.multi_processor_count),
                                  CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, state->cudevice);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                 "cuDeviceGetAttribute of CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT failed \n");
    state->cu_dev_attrib.cooperative_launch = 0;

    status = cuDeviceGetAttribute(&(state->cu_dev_attrib.cooperative_launch),
                                  CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, state->cudevice);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                 "cuDeviceGetAttribute CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH failed \n");

    if (!state->cu_dev_attrib.cooperative_launch)
        WARN_PRINT(
            "Cooperative launch not supported on PE %d; GPU-side synchronize may cause hang\n",
            state->mype);

    status = cuCtxGetStreamPriorityRange(&leastPriority, &greatestPriority);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                 "cuCtxGetStreamPriorityRange failed \n");

    status = cuStreamCreateWithPriority(&state->claunch_params.stream, CU_STREAM_NON_BLOCKING,
                                        greatestPriority);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                 "cuStreamCreateWithPriority failed \n");

    status = cuEventCreate(&state->claunch_params.begin_event, CU_EVENT_DISABLE_TIMING);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                 "cuEventCreate for begin event failed \n");

    status = cuEventCreate(&state->claunch_params.end_event, CU_EVENT_DISABLE_TIMING);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                 "cuEventCreate for end event failed \n");

out:
    return status;
}

int nvshmemi_teardown_collective_launch(nvshmemi_state_t *state) {
    int status = 0;

    if (!state->initialized) goto out;

    status = cuStreamDestroy(state->claunch_params.stream);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cuStreamDestroy failed \n");

    status = cuEventDestroy(state->claunch_params.begin_event);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                 "cuEventDestroy for begin event failed \n");

    status = cuEventDestroy(state->claunch_params.end_event);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                 "cuEventDestroy for end event failed \n");

out:
    return status;
}

inline int nvshmemi_minv(int *vec, int count) {
    int minval = INT_MAX;
    for (int i = 0; i < count; i++) {
        if (vec[i] < minval) {
            minval = vec[i];
        }
    }
    return minval;
}

inline int nvshmemi_maxv(int *vec, int count) {
    int maxval = INT_MIN;
    for (int i = 0; i < count; i++) {
        if (vec[i] > maxval) {
            maxval = vec[i];
        }
    }
    return maxval;
}

int nvshmemx_collective_launch(const void *func, dim3 gridDims, dim3 blockDims, void **args,
                               size_t sharedMem, cudaStream_t stream) {
    NVSHMEM_CHECK_STATE_AND_INIT();

    int multiProcessorCount;
    int blockSize = blockDims.x * blockDims.y * blockDims.z;
    int maxBlocksSM;
    int gridSize = -1;
    int launchFailed = 1;
    CUresult cures;
    cudaError_t cudares;
    int status = 0;

    // XXX: Supports the user passing a non-zero grid but of differing size across ranks
    if (gridDims.x == 0 && gridDims.y == 0 && gridDims.z == 0) {
        gridSize = 0;
    } else if (gridDims.x != 0 && gridDims.y != 0 && gridDims.z != 0) {
        gridSize = gridDims.x * gridDims.y * gridDims.z;
    }  // else
       // some but not all grid dim being 0 is illegal
       // XXX: if some ranks pass an illegal grid, others error out

    // get min blocks per SM, error out if 0 for any GPU
    status =
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksSM, func, blockSize, sharedMem);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                 "cudaOccupancyMaxActiveBlocksPerMultiprocessor failed \n");

    multiProcessorCount = nvshmemi_state->cu_dev_attrib.multi_processor_count;
    INFO(NVSHMEM_COLL, "[%d] SM count %d  CTA/SM count %d", nvshmemi_state->mype,
         multiProcessorCount, maxBlocksSM);
    if (gridSize == 0) { /*XXX : auto sizing */
        // two alternatives - run the minimum supported grid (>0) on all GPUs (global communication
        // needed) or run the maximum supported grid on each GPU (local decision)
        // XXX: Launches maximum supported grid (>0) on associated GPU
        if (maxBlocksSM > 0) { /*Launch will work only if all GPUs can run at least one CTA*/
            launchFailed = 0;
        }
        gridDims.x = maxBlocksSM * multiProcessorCount;
        gridDims.y = 1;
        gridDims.z = 1;
    } else if (gridSize > 0) { /* XXX : legal grid is provided by user*/
        if ((maxBlocksSM > 0) && (gridSize <= maxBlocksSM * multiProcessorCount)) { /*Works*/
            launchFailed = 0;
        }
    }

    INFO(NVSHMEM_COLL, "nvshmemi_maxv allgather target %p source %p nbytes %ld", &launchFailed,
         nvshmemi_state->scratch, sizeof(int));
    status =
        nvshmemi_state->boot_handle.allgather((void *)&launchFailed, (void *)nvshmemi_state->scratch,
                                             sizeof(int), &nvshmemi_state->boot_handle);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_COLLECTIVE_LAUNCH_FAILED, out,
                 "allgather of launch capability failed \n");

    launchFailed = nvshmemi_maxv(nvshmemi_state->scratch, nvshmemi_state->npes);
    NZ_ERROR_JMP(launchFailed, NVSHMEMX_ERROR_COLLECTIVE_LAUNCH_FAILED, out,
                 "One or more PEs cannot launch \n");

    status =
        cuEventRecord(nvshmemi_state->claunch_params.begin_event, static_cast<CUstream>(stream));
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_COLLECTIVE_LAUNCH_FAILED, out,
                 "Recording begin event failed \n");

    status = cuStreamWaitEvent(nvshmemi_state->claunch_params.stream,
                               nvshmemi_state->claunch_params.begin_event, 0);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_COLLECTIVE_LAUNCH_FAILED, out,
                 "Waiting on stream for begin event failed \n");

    if (nvshmemi_state->cu_dev_attrib.cooperative_launch) {
        status = cudaLaunchCooperativeKernel(func, gridDims, blockDims, args, sharedMem,
                                             nvshmemi_state->claunch_params.stream);
        NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_COLLECTIVE_LAUNCH_FAILED, out,
                     "Cooperative kernel launch failed \n");
    } else {
        status = cudaLaunchKernel(func, gridDims, blockDims, args, sharedMem,
                                  nvshmemi_state->claunch_params.stream);
        NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_COLLECTIVE_LAUNCH_FAILED, out,
                     "Kernel launch failed \n");
    }

    status = cuEventRecord(nvshmemi_state->claunch_params.end_event,
                           nvshmemi_state->claunch_params.stream);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_COLLECTIVE_LAUNCH_FAILED, out,
                 "Recording end event failed \n");

    status = cuStreamWaitEvent(static_cast<CUstream>(stream),
                               nvshmemi_state->claunch_params.end_event, 0);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_COLLECTIVE_LAUNCH_FAILED, out,
                 "Waiting on stream for end failed \n");

out:
    return status;
}

int nvshmemx_collective_launch_query_gridsize(const void *func, dim3 blockDims, void **args,
                                              size_t sharedMem, int *gridsize) {
    int multiProcessorCount;
    int blockSize = blockDims.x * blockDims.y * blockDims.z;
    int maxBlocksSM;
    int status = 0;

    multiProcessorCount = nvshmemi_state->cu_dev_attrib.multi_processor_count;
    // get min blocks per SM, error out if 0 for any GPU
    status =
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksSM, func, blockSize, sharedMem);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                 "cudaOccupancyMaxActiveBlocksPerMultiprocessor failed \n");

    // XXX: Returns maximum supported grid (including 0) on associated GPU
    *gridsize = maxBlocksSM * multiProcessorCount;  // XXX:caller chooses dimension of grid

out:
    return status;
}
