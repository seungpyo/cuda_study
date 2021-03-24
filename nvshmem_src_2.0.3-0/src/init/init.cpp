/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "nvshmem_internal.h"

#include <stdlib.h>
#include <string.h>
#include "topo.h"
#include "util.h"
#include "cpu_coll.h"
#include "gpu_coll.h"
#include "unistd.h"
#include "debug.h"

nvshmemi_state_t *nvshmemi_state;
nvshmem_options_t nvshmem_options;
const char *p_err_str;
int nvshmem_debug_level;
uint64_t nvshmem_debug_mask = NVSHMEM_INIT;  // Default debug sub-system mask is INIT
pthread_mutex_t nvshmem_debug_output_lock;
FILE *nvshmem_debug_file = stdout;

#ifdef ENABLE_TRACE
std::chrono::high_resolution_clock::time_point nvshmem_epoch;
#endif

int nvshmemi_bootstrap(int flags, nvshmemx_init_attr_t *nvshmem_attr, nvshmemi_state_t *state) {
    int status = 0;
    uint64_t myHostHash = 0;
    uint64_t *hostHash = 0;
    int mype_node = 0, npes_node = 0;

    if (flags & NVSHMEMX_INIT_WITH_MPI_COMM) {
        bootstrap_attr_t boot_attr;
        boot_attr.mpi_comm = nvshmem_attr->mpi_comm;
        status = bootstrap_init(BOOTSTRAP_MPI, &boot_attr, &state->boot_handle);
    } else if (flags & NVSHMEMX_INIT_WITH_SHMEM) {
        status = bootstrap_init(BOOTSTRAP_SHMEM, NULL, &state->boot_handle);
    } else {
        status = bootstrap_init(BOOTSTRAP_PMI, NULL, &state->boot_handle);
    }
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "bootstrap_init failed \n");

    state->mype = state->boot_handle.pg_rank;
    state->npes = state->boot_handle.pg_size;

    myHostHash = getHostHash();
    hostHash = (uint64_t *)malloc(sizeof(uint64_t) * state->npes);
    status = state->boot_handle.allgather((void *)&myHostHash, (void *)hostHash, sizeof(uint64_t),
                                          &state->boot_handle);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "allgather of host hashes failed \n");
    for (int i = 0; i < state->npes; i++) {
        if (hostHash[i] == myHostHash) {
            if (i == state->mype) mype_node = npes_node;
            npes_node++;
        }
    }
    state->mype_node = mype_node;
    state->npes_node = npes_node;

out:
    return status;
}

int nvshmemi_get_cucontext(nvshmemi_state_t *state) {
    int status = 0;
    int leastPriority, greatestPriority;

    status = cuInit(0);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "CUDA init failed \n");

    status = cuCtxGetDevice(&state->cudevice);
    if (status) {
        TRACE(NVSHMEM_INIT, "GPU not selected, cuCtxGetDevice failed, err: %d", status);
        status = NVSHMEMX_ERROR_GPU_NOT_SELECTED;
        goto out;
    } else {
        CUresult cres = cuCtxSynchronize();
        if (cres) {
            INFO(NVSHMEM_INIT,
                 "[%d] nvshmemi_get_cucontext->cuCtxSynchronize->%d(CUDA_ERROR_NOT_INITIALIZED %d) "
                 "my_stream %llu",
                 state->mype, cres, CUDA_ERROR_NOT_INITIALIZED, state->my_stream);
            status = cuDevicePrimaryCtxRetain(&state->cucontext, state->cudevice);
            NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                         "get primary context failed \n");

            status = cuCtxSetCurrent(state->cucontext);
            NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                         "set context failed \n");

            INFO(NVSHMEM_INIT, "retained primary context for device: %d", state->cudevice);

        } else {
            INFO(NVSHMEM_INIT,
                 "[%d] nvshmemi_get_cucontext->cuCtxSynchronize->CUDA_SUCCESS) my_stream %llu",
                 state->mype, cres, state->my_stream);
            status = cuCtxGetCurrent(&state->cucontext);

            INFO(NVSHMEM_INIT,
                 "int get_cucontext, queried and saved context for device: %d context: %llu",
                 state->cudevice, state->cucontext);
            NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                         "get context failed \n");
        }
        status = cuCtxGetStreamPriorityRange(&leastPriority, &greatestPriority);
        NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                     "cudaDeviceGetStreamPriorityRange failed \n");

        status =
            cuStreamCreateWithPriority(&state->my_stream, CU_STREAM_NON_BLOCKING, greatestPriority);
        NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                     "cudaStreamCreateWithPriority failed \n");
        INFO(NVSHMEM_INIT,
             "[%d] nvshmemi_get_cucontext->cuCtxGetDevice->%d(CUDA_ERROR_INVALID_CONTEXT %d) "
             "cuStreamCreateWithPriority my_stream %llu",
             state->mype, cres, CUDA_ERROR_INVALID_CONTEXT, state->my_stream);
    }
out:
    return status;
}

int nvshmemi_setup_stream_priorities(nvshmemi_state_t *state) {
    int status = 0;
    int leastPriority, greatestPriority;

    status = cuCtxGetStreamPriorityRange(&leastPriority, &greatestPriority);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                 "cudaDeviceGetStreamPriorityRange failed \n");

    status =
        cuStreamCreateWithPriority(&state->my_stream, CU_STREAM_NON_BLOCKING, greatestPriority);
    NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                 "cudaStreamCreateWithPriority failed \n");
out:
    return status;
}

int nvshmemi_setup_memory_ordering(nvshmemi_state_t *state, int selected_transport) {
    int status = 0;

    state->fence[selected_transport] = state->transports[selected_transport]->host_ops.fence;
    state->quiet[selected_transport] = state->transports[selected_transport]->host_ops.quiet;

    return status;
}

int nvshmemi_teardown_handles(nvshmemi_state_t *state) {
    int status = 0;
    free(state->rma);
    free(state->selected_transport_for_rma);
    free(state->amo);
    free(state->selected_transport_for_amo);
    free(state->fence);
    free(state->quiet);
    for (int i = 0; i < MAX_PEER_STREAMS; i++) {
        status = cuStreamDestroy(state->custreams[i]);
        NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                     "cuStreamDestroy failed \n");
        status = cuEventDestroy(state->cuevents[i]);
        NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out, "cuEventDestroy failed \n");
    }
out:
    return status;
}

static int nvshmemi_setup_nvshmem_handles(nvshmemi_state_t *state) {
    int status = 0;
    state->rma = (rma_handle *)calloc(state->npes, sizeof(rma_handle));
    state->amo = (amo_handle *)calloc(state->npes, sizeof(amo_handle));
    state->fence = (fence_handle *)calloc(NVSHMEM_TRANSPORT_COUNT, sizeof(fence_handle));
    state->quiet = (quiet_handle *)calloc(NVSHMEM_TRANSPORT_COUNT, sizeof(quiet_handle));
    state->selected_transport_for_rma = (int *)calloc(state->npes, sizeof(int));
    state->selected_transport_for_amo = (int *)calloc(state->npes, sizeof(int));
    for (int pe = 0; pe < state->npes; pe++) {
        state->rma[pe] = 0;
        state->amo[pe] = 0;
        state->selected_transport_for_rma[pe] = -1;
        state->selected_transport_for_amo[pe] = -1;
    }
    for (int t = 0; t < NVSHMEM_TRANSPORT_COUNT; t++) {
        state->fence[t] = 0;
        state->quiet[t] = 0;
    }
    int memory_ordering_initialized = 0;
    int tbitmap;
    for (int i = 0; i < state->npes; i++) {
        bool amo_initialized = false, rma_initialized = false;
        tbitmap = state->transport_bitmap;
        for (int j = 0; j < NVSHMEM_TRANSPORT_COUNT; j++) {
            if (!(state->transports[j])) continue;

	    if ((tbitmap & 1)) {
                if (!rma_initialized &&
                    (state->transports[j]->cap[i] &
                     (NVSHMEM_TRANSPORT_CAP_CPU_READ | NVSHMEM_TRANSPORT_CAP_CPU_WRITE))) {
                    state->rma[i] = state->transports[j]->host_ops.rma;
                    rma_initialized = true;
                    state->selected_transport_for_rma[i] = j;
                }

                if (!amo_initialized &&
                    (state->transports[j]->cap[i] & NVSHMEM_TRANSPORT_CAP_CPU_ATOMICS)) {
                    state->amo[i] = state->transports[j]->host_ops.amo;
                    amo_initialized = true;
                    state->selected_transport_for_amo[i] = j;
                }

		if (((state->selected_transport_for_amo[i] == j) ||
                     (state->selected_transport_for_rma[i] == j)) &&
                    !(memory_ordering_initialized && (1 << j))) {
                    nvshmemi_setup_memory_ordering(state, j);
                    memory_ordering_initialized |= 1 << j;
                }
            }
            tbitmap >>= 1;
        }
    }
out:
    return status;
}

static int nvshmemi_setup_cuda_handles(nvshmemi_state_t *state) {
    int status = 0;
    state->custreams = (CUstream *)malloc(MAX_PEER_STREAMS * sizeof(CUstream));
    state->cuevents = (CUevent *)malloc(MAX_PEER_STREAMS * sizeof(CUevent));
    int leastPriority, greatestPriority;
    status = cuCtxGetStreamPriorityRange(&leastPriority, &greatestPriority);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                 "cudaDeviceGetStreamPriorityRange failed \n");
    for (int i = 0; i < MAX_PEER_STREAMS; i++) {
        status = cuStreamCreateWithPriority(&state->custreams[i], CU_STREAM_NON_BLOCKING,
                                            greatestPriority);
        NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                     "cuStreamCreateWithPriority failed \n");
        status = cuEventCreate(&state->cuevents[i], CU_EVENT_DISABLE_TIMING);
        NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out, "cuEventCreate failed \n");
    }
    state->curets = (CUdeviceptr *)malloc(state->npes * sizeof(CUdeviceptr));
    for (int i = 0; i < state->npes; i++) {
        status = cuMemAlloc(&state->curets[i], sizeof(long long));
        NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                     "cuMemAlloc failed \n");  // XXX:renamed curetval to curets
    }
out:
    return status;
}

int nvshmemi_common_init(nvshmemi_state_t *state) {
    int status = 0;
    char *value;

    if (state->initialized) return 0;

    status = nvshmemi_get_cucontext(state);
    NZ_DEBUG_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem get cucontext failed \n");

    status = nvshmemi_detect_same_device(state);
    NZ_DEBUG_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem detect same device failed \n");

    status = nvshmemi_setup_stream_priorities(state);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem setup stream priorities failed \n");

    status = nvshmemi_setup_local_heap(state);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem setup local heap failed \n");

    status = nvshmemi_transport_init(state);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem detect topo failed \n");

    status = nvshmemi_build_transport_map(state);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "building transport map failed \n");

    status = nvshmemi_setup_cuda_handles(state);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cuda handles setup failed \n");

    status = nvshmemi_setup_nvshmem_handles(state);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem handles setup failed \n");

    status = nvshmemi_setup_connections(state);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem setup connections failed \n");

    status = nvshmemi_setup_symmetric_heap(state);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem setup heap failed \n");

    status = nvshmemi_setup_collective_launch(state);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem setup collective launch failed \n");

    status = nvshmemi_init_device_state(state);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem device state setup failed \n");

    state->initialized = 1;

    // coll init uses nvshmem_malloc directly
    // better to have state->initialized = 1
    status = nvshmemi_coll_common_cpu_init();
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cpu collective setup failed \n");

    status = nvshmemi_coll_common_gpu_init();
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gpu collective setup failed \n");

    // enable proxy only if IB transport is initialized
    if (nvshmemi_job_connectivity >= NVSHMEMI_JOB_GPU_LDST &&
        state->transports[NVSHMEM_TRANSPORT_ID_IBRC] &&
        state->transports[NVSHMEM_TRANSPORT_ID_IBRC]->is_successfully_initialized) {
        status = nvshmemi_proxy_init(state);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "proxy initialization failed \n");
    }

    nvshmemi_state->boot_handle.barrier(&nvshmemi_state->boot_handle);
    status = nvshmemi_team_init();
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "team setup failed \n");
    
    nvshmemi_barrier_all();
out:
    return status;
}

int nvshmemi_try_common_init(nvshmemi_state_t *state) {
    int status = 0;

    status = nvshmemi_common_init(state);

    if (status) {
        INFO(NVSHMEM_INIT, "nvshmemi_common_init failed, continuing");
        status = 0;
    }

    return status;
}

int nvshmemi_init_thread(int requested, int *provided) {
    int status = 0;

    NVSHMEMU_THREAD_CS_INIT();

    nvshmemi_options_init();
    init_debug();

    nvshmemi_state = (nvshmemi_state_t *)calloc(1, sizeof(nvshmemi_state_t));
    NULL_ERROR_JMP(nvshmemi_state, status, NVSHMEMX_ERROR_INTERNAL, out,
                   "nvshmemi_init_thread/calloc failed \n");

    status = nvshmemi_bootstrap(0, NULL, nvshmemi_state);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem_bootstrap failed \n");

    if (0 == nvshmemi_state->mype) {
        if (nvshmemi_options.VERSION)
            printf("%s\n", NVSHMEM_VENDOR_STRING);

        if (nvshmemi_options.DEBUG_provided) {
            int runtimeVersion, driverVersion;
            cudaError_t err;

            printf("NVSHMEM configuration:\n");

            printf("  %-28s %d\n", "CUDA API", CUDA_VERSION);

            err = cudaRuntimeGetVersion(&runtimeVersion);
            if (err != cudaSuccess) runtimeVersion = -1;
            printf("  %-28s %d\n", "CUDA Runtime", runtimeVersion);

            err = cudaDriverGetVersion(&driverVersion);
            if (err != cudaSuccess) driverVersion = -1;
            printf("  %-28s %d\n", "CUDA Driver", driverVersion);

            printf("  %-28s %s %s\n", "Build Timestamp", __DATE__, __TIME__);
            printf("  %-28s%s", "Build Options",
#ifdef NVSHMEM_X86_64
                    " NVSHMEM_X86_64"
#endif
#ifdef NVSHMEM_PPC64LE
                    " NVSHMEM_PPC64LE"
#endif
#ifdef NVSHMEM_MPI_SUPPORT
                    " NVSHMEM_MPI_SUPPORT"
#endif
#ifdef NVSHMEM_SHMEM_SUPPORT
                    " NVSHMEM_SHMEM_SUPPORT"
#endif
#ifdef NVSHMEM_PMIX_SUPPORT
                    " NVSHMEM_PMIX_SUPPORT"
#endif
#ifdef NVSHMEM_DEFAULT_PMIX
                    " NVSHMEM_DEFAULT_PMIX"
#endif
#ifdef NVSHMEM_DEFAULT_PMI2
                    " NVSHMEM_DEFAULT_PMI2"
#endif
#ifdef NVSHMEM_USE_GDRCOPY
                    " NVSHMEM_USE_GDRCOPY"
#endif
#ifdef NVSHMEM_COMPLEX_SUPPORT
                    " NVSHMEM_COMPLEX_SUPPORT"
#endif
#ifdef NVSHMEM_MPI_IS_OMPI
                    " NVSHMEM_MPI_IS_OMPI"
#endif
#ifdef NVSHMEM_DISABLE_COLL_POLL
                    " NVSHMEM_DISABLE_COLL_POLL"
#endif
#ifdef NVSHMEM_GPU_COLL_USE_LDST
                    " NVSHMEM_GPU_COLL_USE_LDST"
#endif
#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
                    " NVSHMEM_TIMEOUT_DEVICE_POLLING"
#endif
#ifdef NVSHMEM_ENABLE_TRACE
                    " NVSHMEM_ENABLE_TRACE"
#endif
                    "\n");

            printf("\n");
        }
        if (nvshmemi_options.INFO)
            nvshmemi_options_print();
    }

    nvshmemi_state->scratch = (int *)calloc(
        nvshmemi_state->npes, sizeof(int)); /*XXX:scratch used by nvshmemi_try_common_init*/

    status = nvshmemi_try_common_init(nvshmemi_state);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem common init failed \n");

    *provided = NVSHMEM_THREAD_SERIALIZED;

out:
    if (status) NVSHMEMU_THREAD_CS_FINALIZE();

    return status;
}

void nvshmem_init() {
    int status = 0, requested = NVSHMEM_THREAD_SERIALIZED, provided;

    status = nvshmemi_init_thread(requested, &provided);
    NZ_EXIT(status, "aborting due to error in nvshmemi_init_thread \n");
}

int nvshmem_init_thread(int requested, int *provided) {
    int status = 0;

    status = nvshmemi_init_thread(requested, provided);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem_init_thread failed \n");
out:
    return status;
}

int nvshmemx_init_thread(int requested, int *provided) {
    return nvshmem_init_thread(requested, provided);
}

void nvshmem_query_thread(int *provided) { *provided = NVSHMEM_THREAD_SERIALIZED; }

void nvshmemx_query_thread(int *provided) { nvshmem_query_thread(provided); }

int nvshmemx_init_attr(unsigned int flags, nvshmemx_init_attr_t *attr) {
    int status;

    NVSHMEMU_THREAD_CS_INIT();

    nvshmemi_options_init();
    init_debug();

    nvshmemi_state = (nvshmemi_state_t *)calloc(1, sizeof(nvshmemi_state_t));
    p_err_str = (char *)malloc(MAX_LENGTH_ERROR_STRING);

    status = nvshmemi_bootstrap(flags, attr, nvshmemi_state);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem_bootstrap failed \n");

    if (0 == nvshmemi_state->mype) {
        if (nvshmemi_options.VERSION)
            printf("%s\n", NVSHMEM_VENDOR_STRING);

        if (nvshmemi_options.INFO)
            nvshmemi_options_print();
    }

    status = nvshmemi_try_common_init(nvshmemi_state);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem topo init failed \n");

    nvshmemi_state->scratch = (int *)calloc(nvshmemi_state->npes, sizeof(int));

out:
    return status;
}

void nvshmem_finalize() {
    int status = 0;
    int pid = getpid();

    INFO(NVSHMEM_INIT, "[%d] in nvshmem_finalize:", pid);

    if (nvshmemi_state->initialized) {
        nvshmemi_barrier_all();
        nvshmemx_quiet_on_stream(nvshmemi_state->my_stream); /* wait for signal ops from barrier to complete */
        cudaDeviceSynchronize();

        /* teams cleanup */
        status = nvshmemi_team_finalize();
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "Teams cleanup failed \n");

        /*cleaning up proxy*/
        if (nvshmemi_job_connectivity >= NVSHMEMI_JOB_GPU_LDST &&
            nvshmemi_state->transports[NVSHMEM_TRANSPORT_ID_IBRC] &&
            nvshmemi_state->transports[NVSHMEM_TRANSPORT_ID_IBRC]->is_successfully_initialized) {
            status = nvshmemi_proxy_finalize(nvshmemi_state);
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "proxy cleanup failed \n");
        }
        
        /* collective cleanup */
        status = nvshmemi_coll_common_cpu_finalize();
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "CPU collectives cleanup failed \n");

        status = nvshmemi_coll_common_gpu_finalize();
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "GPU collectives cleanup failed \n");

        status = nvshmemi_teardown_handles(nvshmemi_state);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "handles cleanup failed \n");

        status = nvshmemi_cleanup_symmetric_heap(nvshmemi_state);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "symmetric heap cleanup failed \n");

        status = nvshmemi_transport_finalize(nvshmemi_state);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem transport finalize failed \n");

        status = nvshmemi_teardown_collective_launch(nvshmemi_state);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "collective launch cleanup failed \n");
    } else
        nvshmemi_state->boot_handle.barrier(&nvshmemi_state->boot_handle);

    status = bootstrap_finalize(&nvshmemi_state->boot_handle);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "bootstrap_finalize failed \n");

    /* cleanup state */
    if (nvshmemi_state->scratch_size) free(nvshmemi_state->scratch_space);

    if (nvshmemi_state->scratch) free(nvshmemi_state->scratch);
    free(nvshmemi_state);

    NVSHMEMU_THREAD_CS_FINALIZE();

out:
    if (status) {
        ERROR_PRINT("aborting due to error in nvshmem_finalize \n");
        exit(-1);
    }
}

void init_debug() {
    const char *nvshmem_debug = nvshmemi_options.DEBUG;
    if (!nvshmemi_options.DEBUG_provided && !nvshmemi_options.DEBUG_SUBSYS_provided) {
        nvshmem_debug_level = NVSHMEM_LOG_NONE;
    } else if (strcmp_case_insensitive(nvshmem_debug, "VERSION") == 0) {
        nvshmem_debug_level = NVSHMEM_LOG_VERSION;
    } else if (strcmp_case_insensitive(nvshmem_debug, "WARN") == 0) {
        nvshmem_debug_level = NVSHMEM_LOG_WARN;
    } else if (strcmp_case_insensitive(nvshmem_debug, "INFO") == 0) {
        nvshmem_debug_level = NVSHMEM_LOG_INFO;
    } else if (strcmp_case_insensitive(nvshmem_debug, "ABORT") == 0) {
        nvshmem_debug_level = NVSHMEM_LOG_ABORT;
    } else if (strcmp_case_insensitive(nvshmem_debug, "TRACE") == 0) {
        nvshmem_debug_level = NVSHMEM_LOG_TRACE;
    }
    else {
        /* OpenSHMEM spec treats SHMEM_DEBUG as a boolean, enable INFO logging
         * when user-supplied value does match one of the above. */
        nvshmem_debug_level = NVSHMEM_LOG_INFO;
    }

    /* Parse the NVSHMEM_DEBUG_SUBSYS env var
     * This can be a comma separated list such as INIT,COLL
     * or ^INIT,COLL etc
     */
    /* Note: strtok will modify the string, operate on a copy */
    char *nvshmem_debug_subsys = strdup(nvshmemi_options.DEBUG_SUBSYS);
    if (nvshmem_debug_subsys != NULL) {
        char *subsys = strtok(nvshmem_debug_subsys, ",");
        while (subsys != NULL) {
            int invert = 0;
            uint64_t mask = 0;
            if (subsys[0] == '^') {
                invert = 1;
                subsys++;
            }
            if (strcmp_case_insensitive(subsys, "INIT") == 0) {
                mask = NVSHMEM_INIT;
            } else if (strcmp_case_insensitive(subsys, "COLL") == 0) {
                mask = NVSHMEM_COLL;
            } else if (strcmp_case_insensitive(subsys, "P2P") == 0) {
                mask = NVSHMEM_P2P;
            } else if (strcmp_case_insensitive(subsys, "PROXY") == 0) {
                mask = NVSHMEM_PROXY;
            } else if (strcmp_case_insensitive(subsys, "TRANSPORT") == 0) {
                mask = NVSHMEM_TRANSPORT;
            } else if (strcmp_case_insensitive(subsys, "MEM") == 0) {
                mask = NVSHMEM_MEM;
            } else if (strcmp_case_insensitive(subsys, "BOOTSTRAP") == 0) {
                mask = NVSHMEM_BOOTSTRAP;
            } else if (strcmp_case_insensitive(subsys, "TOPO") == 0) {
                mask = NVSHMEM_TOPO;
            } else if (strcmp_case_insensitive(subsys, "UTIL") == 0) {
                mask = NVSHMEM_UTIL;
            } else if (strcmp_case_insensitive(subsys, "ALL") == 0) {
                mask = NVSHMEM_ALL;
            } else {
                mask = 0;
                WARN("Unrecognized value in DEBUG_SUBSYS: %s%s", invert ? "^" : "", subsys);
            }
            if (mask) {
                if (invert)
                    nvshmem_debug_mask &= ~mask;
                else
                    nvshmem_debug_mask |= mask;
            }
            subsys = strtok(NULL, ",");
        }

        free(nvshmem_debug_subsys);
    }

    /* Parse and expand the NVSHMEM_DEBUG_FILE path and
     * then create the debug file. But don't bother unless the
     * NVSHMEM_DEBUG level is > VERSION
     */
    const char *nvshmem_debug_filename = nvshmemi_options.DEBUG_FILE;
    if (nvshmem_debug_level > NVSHMEM_LOG_VERSION && nvshmemi_options.DEBUG_FILE_provided) {
        int c = 0;
        char debugFn[PATH_MAX + 1] = "";
        char *dfn = debugFn;
        while (nvshmem_debug_filename[c] != '\0' && c < PATH_MAX) {
            if (nvshmem_debug_filename[c++] != '%') {
                *dfn++ = nvshmem_debug_filename[c - 1];
                continue;
            }
            switch (nvshmem_debug_filename[c++]) {
                case '%':  // Double %
                    *dfn++ = '%';
                    break;
                case 'h':  // %h = hostname
                    char hostname[1024];
                    getHostName(hostname, 1024);
                    dfn += snprintf(dfn, PATH_MAX, "%s", hostname);
                    break;
                case 'p':  // %p = pid
                    dfn += snprintf(dfn, PATH_MAX, "%d", getpid());
                    break;
                default:  // Echo everything we don't understand
                    *dfn++ = '%';
                    *dfn++ = nvshmem_debug_filename[c - 1];
                    break;
            }
        }
        *dfn = '\0';
        if (debugFn[0] != '\0') {
            FILE *file = fopen(debugFn, "w");
            if (file != NULL) {
                INFO(NVSHMEM_ALL, "DEBUG file is '%s'", debugFn);
                nvshmem_debug_file = file;
            }
        }
    }
    pthread_mutex_init(&nvshmem_debug_output_lock, NULL);

#ifdef ENABLE_TRACE
    nvshmem_epoch = std::chrono::high_resolution_clock::now();
#endif
}
