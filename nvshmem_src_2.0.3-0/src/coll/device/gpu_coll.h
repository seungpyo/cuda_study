/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _NVSHMEMI_COLL_GPU_H_
#define _NVSHMEMI_COLL_GPU_H_

#include "cuda.h"
#include "nvshmem_internal.h"
#include "common.cuh"

/* structs */

typedef struct gpu_coll_env_params {
    int gpu_intm_rdxn_size;
    int reduce_recexch_kval;
} gpu_coll_env_params_t;

extern int nvshm_gpu_coll_initialized;
extern gpu_coll_env_params_t gpu_coll_env_params_var;

extern __device__ gpu_coll_env_params_t gpu_coll_env_params_var_d;

extern __device__ int reduce_recexch_step1_sendto_d;
extern __device__ int *reduce_recexch_step1_recvfrom_d;
extern __device__ int reduce_recexch_step1_nrecvs_d;
extern __device__ int **reduce_recexch_step2_nbrs_d;
extern __device__ int reduce_recexch_step2_nphases_d;
extern __device__ int reduce_recexch_p_of_k_d;
extern __device__ int reduce_recexch_reduce_recexch_digit_d;
extern __device__ int *digit_d;


/* function declarations */
extern int nvshmemi_coll_common_gpu_init(void);
extern int nvshmemi_coll_common_gpu_return_modes(void);
extern int nvshmemi_coll_common_gpu_finalize(void);

void nvshmemi_recexchalgo_get_neighbors(int my_pe, int PE_size);

/* macro definitions */
#define MAX_THREADS_PER_CTA 512

#define NVSHMEMI_COLL_GPU_ERR_POP()                                                \
    do {                                                                           \
        fprintf(stderr, "Error at %s:%d in %s", __FILE__, __LINE__, __FUNCTION__); \
        goto fn_fail;                                                              \
    } while (0)

#if __cplusplus
extern "C" {
#endif
int init_shm_kernel_shm_ptr();
#if __cplusplus
}
#endif


/* This is signaling function used in barrier algorithm.
nvshmem_<type>_signal function cannot be used in barrier because it uses a
combination of P2P path and IB path depending on how the peer GPU is
connected. In contrast to that, this fuction uses either P2P path (when all GPUs
are NVLink connected) or IB path (when any of the GPU is not NVLink connected).

Using this function in barrier is necessary to ensure any previous RMA
operations are visible. When combination of P2P and IB path are used
as in nvshmem_<type>_signal function, it can lead to race conditions.
For example NVLink writes (of data and signal) can overtake IB writes.
And hence the data may not be visible after the barrier operation.
*/

template <typename T>
__device__ inline void nvshmemi_signal_for_barrier(T *dest, const T value, int pe) {
   const void *peer_base_addr =
       (void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + pe);
   if (nvshmemi_job_connectivity_d <= NVSHMEMI_JOB_GPU_LDST_ATOMICS ||
       (nvshmemi_job_connectivity_d == NVSHMEMI_JOB_GPU_LDST && nvshmemi_proxy_d == 0)) {
       volatile T *dest_actual = (volatile T *)((char *)(peer_base_addr) +
                              ((char *)dest - (char *)(nvshmemi_heap_base_d)));
       *dest_actual = value;
   } else {
       nvshmemi_proxy_amo_nonfetch<T>((void *)dest, value, pe, NVSHMEMI_AMO_SIGNAL);
   }
}


#endif /* NVSHMEMI_COLL_GPU_H */
