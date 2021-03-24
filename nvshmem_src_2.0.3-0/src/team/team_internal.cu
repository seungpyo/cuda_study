/*
* Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
*
* See COPYRIGHT for license information
*/

#include "nvshmem.h"
#include "nvshmemx.h"
#include "nvshmemi_coll.h"
#include "nvshmem_internal.h"
#include "nvshmemi_team.h"
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include "util.h"

enum {
    NVSHMEM_TEAM_WORLD_INDEX = 0,
    NVSHMEM_TEAM_SHARED_INDEX,
    NVSHMEM_TEAM_NODE_INDEX,
    NVSHMEM_TEAMS_MIN
};

static long NVSHMEMI_TEAMS_MAX=20;
#define PSYNC_SIZE_PER_TEAM (NVSHMEMI_SYNC_SIZE +                \
                             NVSHMEMI_ALLTOALL_SYNC_SIZE +       \
                             NVSHMEMI_BCAST_SYNC_SIZE +          \
                             NVSHMEMI_COLLECT_SYNC_SIZE +        \
                             NVSHMEMI_REDUCE_MIN_WRKDATA_SIZE +  \
                             NVSHMEMI_REDUCE_SYNC_SIZE)

nvshmemi_team_t nvshmemi_team_world;
nvshmemi_team_t nvshmemi_team_shared;
nvshmemi_team_t nvshmemi_team_node;

__device__ nvshmemi_team_t nvshmemi_team_world_d;
__device__ nvshmemi_team_t nvshmemi_team_shared_d;
__device__ nvshmemi_team_t nvshmemi_team_node_d;

nvshmemi_team_t **nvshmemi_team_pool;
long *nvshmemi_psync_pool;
long *nvshmemi_sync_counter;

nvshmemi_team_t **nvshmemi_device_team_pool;

__device__ nvshmemi_team_t **nvshmemi_team_pool_d;
__device__ long *nvshmemi_psync_pool_d;
__device__ long *nvshmemi_sync_counter_d;

static unsigned char *psync_pool_avail;
static unsigned char *psync_pool_avail_reduced;
static unsigned char *device_psync_pool_avail;
static unsigned char *device_psync_pool_avail_reduced;

static int *team_ret_val;
static int *team_ret_val_reduced;
static int *device_team_ret_val;
static int *device_team_ret_val_reduced;

/* Checks whether a PE has a consistent stride given (start, stride, size).
 * This function is useful within a loop across PE IDs, and sets 'start',
 * 'stride' and 'size' accordingly upon exiting the loop. It also assumes
 * 'start' and 'stride' are initialized to a negative number and 'size' to 0.
 * If an inconsistent stride is found, returns -1. */
static inline int check_for_linear_stride(int pe, int *start, int *stride, int *size) {
    if (*start < 0) {
        *start = pe;
        (*size)++;
    } else if (*stride < 0) {
        *stride = pe - *start;
        (*size)++;
    } else if ((pe - *start) % *stride != 0) {
        WARN_PRINT("Detected non-uniform stride inserting PE %d into <%d, %d, %d>\n", pe, *start,
                   *stride, *size);
        return -1;
    } else {
        (*size)++;
    }
    return 0;
}

size_t nvshmemi_get_teams_mem_requirement() {
    return sizeof(long) * NVSHMEMI_TEAMS_MAX * PSYNC_SIZE_PER_TEAM + /* psync's */
           2 * N_PSYNC_BYTES +  /* psync_pool_avail */
           2 * sizeof(int) +    /* team_ret_val */
           sizeof(long) * NVSHMEMI_TEAMS_MAX /* storing counters */
#ifdef NVSHMEM_USE_NCCL
           + sizeof(ncclUniqueId)
#endif
           ;
            
}

#ifdef NVSHMEM_USE_NCCL
void nvshmemi_team_init_nccl_comm(nvshmemi_team_t *teami) {
    ncclUniqueId Id;
    int start = teami->start;
    int stride = teami->stride;
    int size = teami->size;
    long *pWrk = nvshmemi_team_get_psync(teami, REDUCE);
    if (teami->my_pe == 0) {
        NCCL_CHECK(nccl_ftable.GetUniqueId(&Id));
        CUDA_RUNTIME_CHECK(cudaMemcpy(pWrk, &Id, sizeof(ncclUniqueId), cudaMemcpyHostToDevice));
        for (int i = 0; i < size; i++) {
            nvshmem_char_put_nbi((char *)pWrk, (const char *)pWrk,
                                 sizeof(ncclUniqueId), start + i * stride);
        }
        nvshmemi_barrier(start, stride, size, 
                         nvshmemi_team_get_psync(teami, SYNC),
                         nvshmemi_team_get_sync_counter(teami)); /* assumes barrier does not use NCCL */
    } else {
        nvshmemi_barrier(start, stride, size, 
                         nvshmemi_team_get_psync(teami, SYNC),
                         nvshmemi_team_get_sync_counter(teami)); /* assumes barrier does not use NCCL */
        CUDA_RUNTIME_CHECK(cudaMemcpy(&Id, pWrk, sizeof(ncclUniqueId), cudaMemcpyDeviceToHost));
    }
    INFO(NVSHMEM_TEAM, "Calling ncclCommInitRank, teami->size = %d, teami->my_pe = %d\n", teami->size, teami->my_pe);
    NCCL_CHECK(nccl_ftable.CommInitRank(&teami->nccl_comm, teami->size, Id, teami->my_pe));
}
#endif  /* NVSHMEM_USE_NCCL */

/* Team Management Routines */

int nvshmemi_team_init(void) {
    long psync_len;
    int start, stride, size;
    nvshmemi_team_t *team_addr;
    int status = 0;

    /* Initialize NVSHMEM_TEAM_WORLD */
    nvshmemi_team_world.psync_idx = NVSHMEM_TEAM_WORLD_INDEX;
    nvshmemi_team_world.start = 0;
    nvshmemi_team_world.stride = 1;
    nvshmemi_team_world.size = nvshmemi_state->npes;
    nvshmemi_team_world.my_pe = nvshmemi_state->mype;
    nvshmemi_team_world.config_mask = 0;
    memset(&nvshmemi_team_world.config, 0, sizeof(nvshmem_team_config_t));
    cudaMemcpyToSymbol(nvshmemi_team_world_d, &nvshmemi_team_world, sizeof(nvshmemi_team_t), 0);

    /* Initialize NVSHMEM_TEAM_SHARED */
    nvshmemi_team_shared.psync_idx = NVSHMEM_TEAM_SHARED_INDEX;
    nvshmemi_team_shared.my_pe = 0;
    nvshmemi_team_shared.config_mask = 0;
    memset(&nvshmemi_team_shared.config, 0, sizeof(nvshmem_team_config_t));

    nvshmemi_team_shared.start = nvshmemi_state->mype;
    nvshmemi_team_shared.stride = 1;
    nvshmemi_team_shared.size = 1;

    cudaMemcpyToSymbol(nvshmemi_team_shared_d, &nvshmemi_team_shared, sizeof(nvshmemi_team_t), 0);
    INFO(NVSHMEM_INIT, "NVSHMEM_TEAM_SHARED: start=%d, stride=%d, size=%d\n",
         nvshmemi_team_shared.start, nvshmemi_team_shared.stride, nvshmemi_team_shared.size);

    /* Initialize NVSHMEM_TEAM_NODE */
    nvshmemi_team_node.psync_idx = NVSHMEM_TEAM_NODE_INDEX;
    nvshmemi_team_node.my_pe = nvshmemi_state->mype_node;
    nvshmemi_team_node.config_mask = 0;
    memset(&nvshmemi_team_node.config, 0, sizeof(nvshmem_team_config_t));

    uint64_t myHostHash = getHostHash();
    uint64_t *hostHash = (uint64_t *)malloc(sizeof(uint64_t) * nvshmemi_state->npes);
    NULL_ERROR_JMP(hostHash, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, cleanup, "hostHash allocation failed \n");
    status = nvshmemi_state->boot_handle.allgather((void *)&myHostHash, (void *)hostHash, sizeof(uint64_t),
                                          &nvshmemi_state->boot_handle);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, cleanup, "allgather of host hashes failed\n");

    /* Search for on-node peer PEs while checking for a consistent stride */
    start = -1; stride = -1; size = 0;

    for (int pe = 0; pe < nvshmemi_state->npes; pe++) {
        if (hostHash[pe] != myHostHash)
            continue;
        
        int ret = check_for_linear_stride(pe, &start, &stride, &size);
        if (ret < 0) {
            start = nvshmemi_state->mype;
            stride = 1;
            size = 1;
            break;
        }
    }
    free(hostHash);
    assert(start >= 0 && size > 0);
    nvshmemi_team_node.start = start;
    nvshmemi_team_node.stride = (stride == -1) ? 1 : stride;
    nvshmemi_team_node.size = size;
    cudaMemcpyToSymbol(nvshmemi_team_node_d, &nvshmemi_team_node, sizeof(nvshmemi_team_t), 0);

    INFO(NVSHMEM_INIT, "NVSHMEMX_TEAM_NODE: start=%d, stride=%d, size=%d\n",
         nvshmemi_team_node.start, nvshmemi_team_node.stride, nvshmemi_team_node.size);

    if (NVSHMEMI_TEAMS_MAX < NVSHMEM_TEAMS_MIN) NVSHMEMI_TEAMS_MAX = NVSHMEM_TEAMS_MIN;

    if (NVSHMEMI_TEAMS_MAX > N_PSYNC_BYTES * CHAR_BIT) {
        ERROR_EXIT("Requested %ld teams, but only %d are supported\n", NVSHMEMI_TEAMS_MAX,
                   N_PSYNC_BYTES * CHAR_BIT);
        goto cleanup;
    }


    nvshmemi_team_pool = (nvshmemi_team_t **) malloc(NVSHMEMI_TEAMS_MAX * sizeof(nvshmemi_team_t *));
    NULL_ERROR_JMP(nvshmemi_team_pool, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, cleanup, "nvshmemi_team_pool allocation failed \n");
    CUDA_RUNTIME_CHECK(cudaMalloc((void **)&nvshmemi_device_team_pool, NVSHMEMI_TEAMS_MAX * sizeof(nvshmemi_team_t *)));
    CUDA_RUNTIME_CHECK(cudaMemcpyToSymbol(nvshmemi_team_pool_d, &nvshmemi_device_team_pool, sizeof(nvshmemi_team_t **), 0));    

    for (long i = 0; i < NVSHMEMI_TEAMS_MAX; i++) {
        nvshmemi_team_pool[i] = NULL;
    }

    nvshmemi_init_array_kernel<nvshmemi_team_t *><<<1, 1>>>(nvshmemi_device_team_pool, NVSHMEMI_TEAMS_MAX, NULL);
    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

    nvshmemi_team_pool[NVSHMEM_TEAM_WORLD_INDEX] = &nvshmemi_team_world;
    nvshmemi_team_pool[NVSHMEM_TEAM_SHARED_INDEX] = &nvshmemi_team_shared;
    nvshmemi_team_pool[NVSHMEM_TEAM_NODE_INDEX] = &nvshmemi_team_node;
    
    /* Allocate pSync pool, each with the maximum possible size requirement */
    /* Create two pSyncs per team for back-to-back collectives and one for barriers.
     * Array organization:
     *
     * [ (world) (shared) (team 1) (team 2) ...  (world) (shared) (team 1) (team 2) ... ]
     *  <----------- groups 1 & 2-------------->|<------------- group 3 ---------------->
     *  <--- (bcast, collect, reduce, etc.) --->|<------ (barriers and syncs) ---------->
     * */
    psync_len = NVSHMEMI_TEAMS_MAX * PSYNC_SIZE_PER_TEAM;
    nvshmemi_psync_pool = (long *)nvshmemi_malloc(sizeof(long) * psync_len);
    NULL_ERROR_JMP(nvshmemi_psync_pool, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, cleanup, "nvshmemi_psync_pool allocation failed \n");

    CUDA_RUNTIME_CHECK(cudaMemcpyToSymbol(nvshmemi_psync_pool_d, &nvshmemi_psync_pool, sizeof(long *)));

    nvshmemi_init_array_kernel<long><<<1, 1>>>(nvshmemi_psync_pool, psync_len, NVSHMEMI_SYNC_VALUE);
    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

    nvshmemi_sync_counter = (long *)nvshmemi_malloc(NVSHMEMI_TEAMS_MAX * sizeof(long));
    NULL_ERROR_JMP(nvshmemi_sync_counter, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, cleanup, "nvshmemi_sync_counter allocation failed \n");

    CUDA_RUNTIME_CHECK(cudaMemcpyToSymbol(nvshmemi_sync_counter_d, &nvshmemi_sync_counter, sizeof(long *)));

    nvshmemi_init_array_kernel<long><<<1, 1>>>(nvshmemi_sync_counter, NVSHMEMI_TEAMS_MAX, 1);
    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

    /* Convenience pointer to the group-3 pSync array (for barriers and syncs): */
    psync_pool_avail = (unsigned char *)malloc(2 * N_PSYNC_BYTES);
    NULL_ERROR_JMP(psync_pool_avail, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, cleanup, "psync_pool_avail allocation failed \n");
    psync_pool_avail_reduced = &psync_pool_avail[N_PSYNC_BYTES];

    device_psync_pool_avail = (unsigned char *)nvshmemi_malloc(2 * N_PSYNC_BYTES);
    NULL_ERROR_JMP(device_psync_pool_avail, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, cleanup, "device_psync_pool_avail allocation failed \n");
    device_psync_pool_avail_reduced = &device_psync_pool_avail[N_PSYNC_BYTES];
    /* Initialize the psync bits to 1, making all slots available: */
    memset(psync_pool_avail, 0, 2 * N_PSYNC_BYTES);
    for (size_t i = 0; i < (size_t)NVSHMEMI_TEAMS_MAX; i++) {
        nvshmemi_bit_set(psync_pool_avail, N_PSYNC_BYTES, i);
    }

    /* Set the bits for NVSHMEM_TEAM_WORLD, NVSHMEM_TEAM_SHARED, NVSHMEMX_TEAM_NODE to 0: */
    nvshmemi_bit_clear(psync_pool_avail, N_PSYNC_BYTES, NVSHMEM_TEAM_WORLD_INDEX);
    nvshmemi_bit_clear(psync_pool_avail, N_PSYNC_BYTES, NVSHMEM_TEAM_SHARED_INDEX);
    nvshmemi_bit_clear(psync_pool_avail, N_PSYNC_BYTES, NVSHMEM_TEAM_NODE_INDEX);

    /* Initialize an integer used to agree on an equal return value across PEs in team creation: */
    team_ret_val = (int *)malloc(sizeof(int) * 2);
    NULL_ERROR_JMP(team_ret_val, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, cleanup, "team_ret_val allocation failed \n");
    team_ret_val_reduced = &team_ret_val[1];

    device_team_ret_val = (int *)nvshmemi_malloc(sizeof(int) * 2);
    NULL_ERROR_JMP(team_ret_val, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, cleanup, "device_team_ret_val allocation failed \n");
    device_team_ret_val_reduced = &device_team_ret_val[1];

    nvshmemi_state->boot_handle.barrier(&nvshmemi_state->boot_handle); /* To ensure neccessary setup has been done all PEs */
#ifdef NVSHMEM_USE_NCCL
    if (nvshmemi_use_nccl) {
        /* Setup NCCL usage */
        nvshmemi_team_init_nccl_comm(&nvshmemi_team_world);
        nvshmemi_team_init_nccl_comm(&nvshmemi_team_shared);
        nvshmemi_team_init_nccl_comm(&nvshmemi_team_node);
    }
#endif /* NVSHMEM_USE_NCCL */

#if defined(NVSHMEM_PPC64LE)
    if (nvshmemi_use_nccl) {
        /* Set GPU thread stack size to be max stack size of any kernel invoked by NCCL.
           The value 1256 has been obtained by profiling all NCCL kernels in NCCL 2.8.3-1.
           This value is being set to prevent any memory config during application run
           as that can lead to potential deadlock */
        if (nvshmemi_options.CUDA_LIMIT_STACK_SIZE_provided) {
            CUDA_RUNTIME_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, nvshmemi_options.CUDA_LIMIT_STACK_SIZE));
            if (nvshmemi_options.CUDA_LIMIT_STACK_SIZE < 1256)
                WARN_PRINT("CUDA stack size limit has been set to less than 1256.\n"
                           "This can lead to hangs because a NCCL kernel can need up\n"
                           "to 1256 bytes");
        }
        else
            CUDA_RUNTIME_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 1256));
    } else if (nvshmemi_options.CUDA_LIMIT_STACK_SIZE_provided) {
        CUDA_RUNTIME_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, nvshmemi_options.CUDA_LIMIT_STACK_SIZE));
    }
#endif

    CUDA_RUNTIME_CHECK(cudaGetSymbolAddress((void **)&team_addr, nvshmemi_team_world_d));
    CUDA_RUNTIME_CHECK(cudaMemcpy(&nvshmemi_device_team_pool[NVSHMEM_TEAM_WORLD_INDEX], &team_addr, sizeof(nvshmemi_team_t *), cudaMemcpyHostToDevice));
    CUDA_RUNTIME_CHECK(cudaGetSymbolAddress((void **)&team_addr, nvshmemi_team_shared_d));
    CUDA_RUNTIME_CHECK(cudaMemcpy(&nvshmemi_device_team_pool[NVSHMEM_TEAM_SHARED_INDEX], &team_addr, sizeof(nvshmemi_team_t *), cudaMemcpyHostToDevice));
    CUDA_RUNTIME_CHECK(cudaGetSymbolAddress((void **)&team_addr, nvshmemi_team_node_d));
    CUDA_RUNTIME_CHECK(cudaMemcpy(&nvshmemi_device_team_pool[NVSHMEM_TEAM_NODE_INDEX], &team_addr, sizeof(nvshmemi_team_t *), cudaMemcpyHostToDevice));
    
    return status;

cleanup:
    if (nvshmemi_team_pool) {
        free(nvshmemi_team_pool);
        nvshmemi_team_pool = NULL;
        cudaFree(nvshmemi_device_team_pool);
        nvshmemi_device_team_pool = NULL;
    }
    if (nvshmemi_psync_pool) {
        nvshmemi_free(nvshmemi_psync_pool);
        nvshmemi_psync_pool = NULL;
    }
    if (psync_pool_avail) {
        free(psync_pool_avail);
        psync_pool_avail = NULL;
    }
    if (device_psync_pool_avail) {
        nvshmemi_free(device_psync_pool_avail);
        device_psync_pool_avail = NULL;
    }
    if (team_ret_val) {
        free(team_ret_val);
        team_ret_val = NULL;
    }
    if (device_team_ret_val) {
        nvshmemi_free(device_team_ret_val);
        device_team_ret_val = NULL;
    }

    return status;
}

int nvshmemi_team_finalize(void) {
    /* Destroy all undestroyed teams */
    for (int32_t i = 0; i < NVSHMEMI_TEAMS_MAX; i++) {
        if (nvshmemi_team_pool[i] != NULL) nvshmemi_team_destroy(nvshmemi_team_pool[i]);
    }

    free(nvshmemi_team_pool);
    CUDA_RUNTIME_CHECK(cudaFree(nvshmemi_device_team_pool));
    
    nvshmemi_free(nvshmemi_psync_pool);
    nvshmemi_free(nvshmemi_sync_counter);

    free(psync_pool_avail);
    nvshmemi_free(device_psync_pool_avail);
    free(team_ret_val);
    nvshmemi_free(device_team_ret_val);

    return 0;
}

int nvshmemi_team_translate_pe(nvshmemi_team_t *src_team, int src_pe, nvshmemi_team_t *dest_team) {
    int src_pe_world, dest_pe = -1;

    if (src_pe > src_team->size) return -1;

    src_pe_world = src_team->start + src_pe * src_team->stride;
#ifdef __CUDA_ARCH__
    assert(src_pe_world >= src_team->start && src_pe_world < nvshmemi_npes_d);
#else
    assert(src_pe_world >= src_team->start && src_pe_world < nvshmemi_state->npes);
#endif

    dest_pe = nvshmemi_pe_in_active_set(src_pe_world, dest_team->start, dest_team->stride,
                                        dest_team->size);

    return dest_pe;
}

int nvshmemi_team_split_strided(nvshmemi_team_t *parent_team, int PE_start, int PE_stride,
                                int PE_size, const nvshmem_team_config_t *config, long config_mask,
                                nvshmem_team_t *new_team) {
    *new_team = NVSHMEM_TEAM_INVALID;
    nvshmem_barrier(parent_team->psync_idx);

    int global_PE_start = nvshmemi_team_pe(parent_team, PE_start);
    int global_PE_end = global_PE_start + PE_stride * (PE_size - 1);

    if (PE_start < 0 || PE_start >= parent_team->size || PE_size <= 0 ||
        PE_size > parent_team->size || PE_stride < 1) {
        WARN_PRINT("Invalid <start, stride, size>: child <%d, %d, %d>, parent <%d, %d, %d>\n",
                       PE_start, PE_stride, PE_size, parent_team->start, parent_team->stride,
                       parent_team->size);
        return -1;
    }

    if (global_PE_start >= nvshmemi_state->npes || global_PE_end >= nvshmemi_state->npes) {
        WARN_PRINT("Starting PE (%d) or ending PE (%d) is invalid\n", global_PE_start,
                       global_PE_end);
        return -1;
    }

    int my_pe = nvshmemi_pe_in_active_set(nvshmemi_state->mype, global_PE_start, PE_stride, PE_size);
    
    long *psync_reduce = nvshmemi_team_get_psync(parent_team, REDUCE);
    nvshmemi_team_t *myteam = NULL;
    *team_ret_val = 0;
    *team_ret_val_reduced = 0;

    if (my_pe != -1) {
        char bit_str[NVSHMEMI_DIAG_STRLEN];

        myteam = (nvshmemi_team_t *)calloc(1, sizeof(nvshmemi_team_t));

        myteam->my_pe = my_pe;
        myteam->start = global_PE_start;
        myteam->stride = PE_stride;
        myteam->size = PE_size;
        if (config) {
            myteam->config = *config;
            myteam->config_mask = config_mask;
        }
        myteam->psync_idx = -1;
        nvshmemi_bit_to_string(bit_str, NVSHMEMI_DIAG_STRLEN, psync_pool_avail,
                               N_PSYNC_BYTES);
        
        CUDA_RUNTIME_CHECK(cudaMemcpy(device_psync_pool_avail, psync_pool_avail, N_PSYNC_BYTES, cudaMemcpyHostToDevice));
        CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
        nvshmemi_uchar_and_reduce((unsigned char *)device_psync_pool_avail_reduced, 
                                  (const unsigned char *)device_psync_pool_avail, N_PSYNC_BYTES,
                                  myteam->start, PE_stride, PE_size, 
                                  (unsigned char *)psync_reduce,
                                  (long *)(psync_reduce + NVSHMEMI_REDUCE_MIN_WRKDATA_SIZE));
        CUDA_RUNTIME_CHECK(cudaMemcpy(psync_pool_avail_reduced, device_psync_pool_avail_reduced,
                           N_PSYNC_BYTES, cudaMemcpyDeviceToHost));

        /* We cannot release the psync here, because this reduction may not
         * have been performed on the entire parent team. */
        nvshmemi_bit_to_string(bit_str, NVSHMEMI_DIAG_STRLEN, psync_pool_avail_reduced,
                               N_PSYNC_BYTES);

        /* Select the least signficant nonzero bit, which corresponds to an available pSync. */
        myteam->psync_idx = nvshmemi_bit_1st_nonzero(psync_pool_avail_reduced, N_PSYNC_BYTES);

        nvshmemi_bit_to_string(bit_str, NVSHMEMI_DIAG_STRLEN, psync_pool_avail_reduced,
                               N_PSYNC_BYTES);
        if (myteam->psync_idx == -1 || myteam->psync_idx >= NVSHMEMI_TEAMS_MAX) {
            WARN_PRINT(
                "No more teams available (max = %ld), try increasing NVSHMEM_TEAMS_MAX\n",
                NVSHMEMI_TEAMS_MAX);
            /* No psync was available, but must call barrier across parent team before returning. */
            myteam->psync_idx = -1;
            *team_ret_val = 1;
        } else {
            /* Set the selected psync bit to 0, reserving that slot */
            nvshmemi_bit_clear(psync_pool_avail, N_PSYNC_BYTES, myteam->psync_idx);

            *new_team = myteam->psync_idx;

            nvshmemi_team_pool[myteam->psync_idx] = myteam;
#ifdef NVSHMEM_USE_NCCL
            nvshmemi_team_init_nccl_comm(myteam);
#endif
            nvshmemi_team_t *device_team_addr;
            CUDA_RUNTIME_CHECK(cudaMalloc((void **)&device_team_addr, sizeof(nvshmemi_team_t)));
            CUDA_RUNTIME_CHECK(cudaMemcpy(device_team_addr, myteam, sizeof(nvshmemi_team_t), cudaMemcpyHostToDevice));
            CUDA_RUNTIME_CHECK(cudaMemcpy(&nvshmemi_device_team_pool[myteam->psync_idx], &device_team_addr, sizeof(nvshmemi_team_t *), cudaMemcpyHostToDevice));
        }
    }

    /* This barrier on the parent team eliminates problematic race conditions
     * during psync allocation between back-to-back team creations. */
    
    nvshmem_quiet();
    //nvshmem_barrier(parent_team->start, parent_team->stride, parent_team->size, psync);
    nvshmem_team_sync(parent_team->psync_idx);

    /* This OR reduction assures all PEs return the same value.  */
    CUDA_RUNTIME_CHECK(cudaMemcpy(device_team_ret_val, team_ret_val, sizeof(int), cudaMemcpyHostToDevice));
    nvshmemi_int_max_reduce(device_team_ret_val_reduced, device_team_ret_val, 1, parent_team->start,
                            parent_team->stride, parent_team->size, (int *)psync_reduce, 
                            (long *)(psync_reduce + NVSHMEMI_REDUCE_MIN_WRKDATA_SIZE));
    CUDA_RUNTIME_CHECK(cudaMemcpy(team_ret_val_reduced, device_team_ret_val_reduced, sizeof(int), cudaMemcpyDeviceToHost));


    /* If no team was available, print some team triplet info and return nonzero. */
    if (my_pe >= 0 && myteam != NULL && myteam->psync_idx == -1) {
        WARN_PRINT("Team split strided failed: child <%d, %d, %d>, parent <%d, %d, %d>\n",
                       global_PE_start, PE_stride, PE_size, parent_team->start, parent_team->stride,
                       parent_team->size);
    }
    return *team_ret_val_reduced;
}

int nvshmemi_team_split_2d(nvshmemi_team_t *parent_team, int xrange,
                           const nvshmem_team_config_t *xaxis_config, long xaxis_mask,
                           nvshmem_team_t *xaxis_team, const nvshmem_team_config_t *yaxis_config,
                           long yaxis_mask, nvshmem_team_t *yaxis_team) {
    *xaxis_team = NVSHMEM_TEAM_INVALID;
    *yaxis_team = NVSHMEM_TEAM_INVALID;

    if (xrange > parent_team->size) {
        xrange = parent_team->size;
    }

    const int parent_stride = parent_team->stride;
    const int parent_size = parent_team->size;
    const int num_xteams = ceil(parent_size / (float)xrange);
    const int num_yteams = xrange;

    int start = 0;
    int ret = 0;

    for (int i = 0; i < num_xteams; i++) {
        nvshmem_team_t my_xteam;
        int xsize = (i == num_xteams - 1 && parent_size % xrange) ? parent_size % xrange : xrange;

        ret = nvshmemi_team_split_strided(parent_team, start, parent_stride, xsize, xaxis_config,
                                          xaxis_mask, &my_xteam);
        if (ret) {
            ERROR_PRINT("Creation of x-axis team %d of %d failed\n", i + 1, num_xteams);
        }
        start += xrange;

        if (my_xteam != NVSHMEM_TEAM_INVALID) {
            assert(*xaxis_team == NVSHMEM_TEAM_INVALID);
            *xaxis_team = my_xteam;
        }
    }

    start = 0;

    for (int i = 0; i < num_yteams; i++) {
        nvshmem_team_t my_yteam;
        int remainder = parent_size % xrange;
        int yrange = parent_size / xrange;
        int ysize = (remainder && i < remainder) ? yrange + 1 : yrange;

        ret = nvshmemi_team_split_strided(parent_team, start, xrange * parent_stride, ysize,
                                          yaxis_config, yaxis_mask, &my_yteam);
        if (ret) {
            ERROR_PRINT("Creation of y-axis team %d of %d failed\n", i + 1, num_yteams);
        }
        start += 1;

        if (my_yteam != NVSHMEM_TEAM_INVALID) {
            assert(*yaxis_team == NVSHMEM_TEAM_INVALID);
            *yaxis_team = my_yteam;
        }
    }

    nvshmem_quiet();
    nvshmem_team_sync(parent_team->psync_idx);

    return 0;
}

void nvshmemi_team_destroy(nvshmemi_team_t *team) {
    int idx = team->psync_idx;
    if (nvshmemi_bit_fetch(psync_pool_avail, idx)) {
        ERROR_PRINT("Destroying a team without an active pSync");
    }
    
    /* Since it is a collective routine, perform a barrier */
    nvshmem_barrier(idx);

    nvshmemi_bit_set(psync_pool_avail, N_PSYNC_BYTES, idx);

    nvshmemi_team_pool[idx] = NULL;
    CUDA_RUNTIME_CHECK(cudaMemset(&nvshmemi_device_team_pool[idx], 0, sizeof(nvshmemi_team_t *)));

    nvshmemi_init_array_kernel<long><<<1, 1>>>(&nvshmemi_sync_counter[idx], 1, 1);
    nvshmemi_init_array_kernel<long><<<1, 1>>>(&nvshmemi_psync_pool[idx * PSYNC_SIZE_PER_TEAM], 
                                      PSYNC_SIZE_PER_TEAM, NVSHMEMI_SYNC_VALUE);
    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

    if (team != &nvshmemi_team_world && team != &nvshmemi_team_shared && team != &nvshmemi_team_node) {
        free(team);
        nvshmemi_team_t *device_team_addr;
        CUDA_RUNTIME_CHECK(cudaMemcpy((void **)&device_team_addr, &nvshmemi_device_team_pool[idx], 
                           sizeof(nvshmemi_team_t *), cudaMemcpyDeviceToHost));
        CUDA_RUNTIME_CHECK(cudaFree(device_team_addr));
   }
}

long *nvshmemi_team_get_psync(nvshmemi_team_t *team, nvshmemi_team_op_t op) {
    long *team_psync;
#ifdef __CUDA_ARCH__
    team_psync = &nvshmemi_psync_pool_d[team->psync_idx * PSYNC_SIZE_PER_TEAM];
#else
    team_psync = &nvshmemi_psync_pool[team->psync_idx * PSYNC_SIZE_PER_TEAM];
#endif /* __CUDA_ARCH__ */
    switch(op) {
        case SYNC:
            return team_psync;
        case ALLTOALL:
            return &team_psync[NVSHMEMI_SYNC_SIZE];
        case BCAST:
            return &team_psync[NVSHMEMI_SYNC_SIZE +
                               NVSHMEMI_ALLTOALL_SYNC_SIZE];
        case COLLECT:
            return &team_psync[NVSHMEMI_SYNC_SIZE +
                               NVSHMEMI_ALLTOALL_SYNC_SIZE +
                               NVSHMEMI_BCAST_SYNC_SIZE];
        case REDUCE:
            return &team_psync[NVSHMEMI_SYNC_SIZE +
                               NVSHMEMI_ALLTOALL_SYNC_SIZE +
                               NVSHMEMI_BCAST_SYNC_SIZE +
                               NVSHMEMI_COLLECT_SYNC_SIZE];
        default:
            printf("Incorrect argument to nvshmemi_team_get_psync\n");
            return NULL;
    }
}

long *nvshmemi_team_get_sync_counter(nvshmemi_team_t *team) {
#ifdef __CUDA_ARCH__
    return &nvshmemi_sync_counter_d[team->psync_idx];
#else
    return &nvshmemi_sync_counter[team->psync_idx];
#endif /* __CUDA_ARCH__ */
}
