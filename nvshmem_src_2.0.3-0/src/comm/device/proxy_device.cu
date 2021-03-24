/****
 * Copyright (c) 2014, NVIDIA Corporation.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright notice,
 *      this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *    * Neither the name of the NVIDIA Corporation nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * The U.S. Department of Energy funded the development of this software
 * under subcontract 7078610 with Lawrence Berkeley National Laboratory.
 *
 ****/

#include <math.h>

#include "../proxy/proxy.h"
#include "nvshmem_internal.h"
#include "nvshmemx_error.h"

__device__ nvshmemi_timeout_t *nvshmemi_timeout_d;
// this will have to be arrays of pointers if more than one channels are used
__device__ void *proxy_channels_buf_d;       /* requests are written in this buffer */
__device__ char *proxy_channel_g_buf_d;
__device__ uint64_t proxy_channel_g_buf_head_d;     /* next location to be assigned to a thread */
__constant__ uint64_t proxy_channel_g_buf_size_d;   /* Total size of g_buf in bytes */
__constant__ uint64_t proxy_channel_g_buf_log_size_d;   /* Total size of g_buf in bytes */
char *proxy_channel_g_buf;
uint64_t proxy_channel_g_buf_head;     /* next location to be assigned to a thread */
uint64_t proxy_channel_g_buf_size;   /* Total size of g_buf in bytes */
uint64_t proxy_channel_g_buf_log_size;   /* Total size of g_buf in bytes */
__device__ uint64_t *proxy_channels_issue_d; /* last byte of the last request */
__device__ uint64_t
    *proxy_channels_complete_d; /* shared betwen CPU and GPU threads - only write by CPU thread and
                                   read by GPU threads. This is allocated on the system memory */
__device__ uint64_t proxy_channels_complete_local_d; /* shared only between GPU threads */
__device__ uint64_t *proxy_channels_quiet_issue_d;
__device__ uint64_t *proxy_channels_quiet_ack_d;
__device__ uint64_t *proxy_channels_cst_issue_d;
__device__ uint64_t *proxy_channels_cst_ack_d;
__constant__ uint64_t proxy_channel_buf_size_d; /* Maximum number of inflight requests in bytes OR
                                                   maximum channel length */
__constant__ uint32_t proxy_channel_buf_logsize_d;

__forceinline__ __device__ void check_channel_availability(uint64_t tail_idx) {
    uint64_t complete;
    complete = *((volatile uint64_t *)&proxy_channels_complete_local_d);
    if ((complete + proxy_channel_buf_size_d - 1) < tail_idx) {
        nvshmemi_wait_until_greater_than_equals_add<uint64_t>(proxy_channels_complete_d, proxy_channel_buf_size_d - 1, tail_idx, NVSHMEMI_CALL_SITE_PROXY_CHECK_CHANNEL_AVAILABILITY);
        atomicMax((unsigned long long int *)&proxy_channels_complete_local_d, complete);
        __threadfence_system();  // XXX: prevents store to buf_d reordered to before load from
                                 // complete_d (breaks rma)
    }
}

__device__ void nvshmemi_proxy_quiet() {
    uint64_t quiet_issue;
    quiet_issue = (*(volatile uint64_t *)proxy_channels_issue_d);
    atomicMax((unsigned long long int *)proxy_channels_quiet_issue_d, quiet_issue);

    nvshmemi_wait_until_greater_than_equals<uint64_t>(proxy_channels_quiet_ack_d, quiet_issue, NVSHMEMI_CALL_SITE_PROXY_QUIET);
    __threadfence_system();  // XXX: prevents store to issue_d reordered to before load from
                             // quiet_ack_d (breaks quiet -> rma)
}

__device__ void nvshmemi_proxy_quiet_no_membar() { /* This function is same as nvshmemi_proxy_quiet
                                                      except that it does not have __threadfence_sysmte() call
                                                      at the end */
    uint64_t quiet_issue;
    quiet_issue = (*(volatile uint64_t *)proxy_channels_issue_d);
    atomicMax((unsigned long long int *)proxy_channels_quiet_issue_d, quiet_issue);

    nvshmemi_wait_until_greater_than_equals<uint64_t>(proxy_channels_quiet_ack_d, quiet_issue, NVSHMEMI_CALL_SITE_PROXY_QUIET);
}

__device__ void nvshmemi_proxy_enforce_consistency_at_target() {
    uint64_t cst_issue;
    cst_issue = (*(volatile uint64_t *)proxy_channels_issue_d);
    atomicMax((unsigned long long int *)proxy_channels_cst_issue_d, cst_issue);

    nvshmemi_wait_until_greater_than_equals<uint64_t>(proxy_channels_cst_ack_d, cst_issue, NVSHMEMI_CALL_SITE_PROXY_ENFORCE_CONSISTENCY_AT_TARGET);
    __threadfence_system();  // XXX: prevents store to issue_d reordered to before load from
                             // cst_ack_d (breaks cst -> rma)
}

__device__ void nvshmemi_proxy_enforce_consistency_at_target_no_membar() {/* This function is same as nvshmemi_proxy_enforce_consistency_at_target()
                                                                             except that it does not have __threadfence_sysmte() call
                                                                             at the end */
    uint64_t cst_issue;
    cst_issue = (*(volatile uint64_t *)proxy_channels_issue_d);
    atomicMax((unsigned long long int *)proxy_channels_cst_issue_d, cst_issue);

    nvshmemi_wait_until_greater_than_equals<uint64_t>(proxy_channels_cst_ack_d, cst_issue, NVSHMEMI_CALL_SITE_PROXY_ENFORCE_CONSISTENCY_AT_TARGET);
}

__forceinline__ __device__ void transfer_dma(void *rptr, void *lptr, size_t bytes, int pe, int channel_op) {
    uint64_t idx, tail_idx, *req;
    int size = sizeof(channel_request_t);
    int group_size = 1;
    void *buf_ptr = proxy_channels_buf_d;
    void *base_ptr = nvshmemi_heap_base_d;

    __threadfence();

    /* idx is an every increasing counter. Since it is 64 bit integer, practically
    it will not overflow */
    idx = atomicAdd((unsigned long long int *)proxy_channels_issue_d, size);
    tail_idx = idx + (size - 1);

    // flow-control
    check_channel_availability(tail_idx);

    req = (uint64_t *)((uint8_t *)buf_ptr + (idx & (proxy_channel_buf_size_d - 1)));
    uint64_t curr_flag = !((idx >> proxy_channel_buf_logsize_d) &
                           1); /* curr_flag is either 0 or 1. Starting at idx = 0 to idx =
                                  proxy_channel_buf_size_d - 1, it will be 1, then for next
                                  proxy_channel_buf_size_d idx values it will be 0, and so on. */
    uint64_t roffset = (uint64_t)((char *)rptr - (char *)base_ptr);
    uint64_t loffset = (uint64_t)((char *)lptr - (char *)base_ptr);
    uint64_t op = channel_op;
    uint16_t pe_u16 = pe;
    uint64_t size_u64 = bytes;

    // assumption that wrap around does not occur in the middle of the request,
    // issue is always incremenet in multiples of sizeof(channel_request_t)

    /* base_request_t
     * 32 | 8 | 8 | 8 | 8
     * roffset_high | roffset_low | op | group_size | flag */
    *((volatile uint64_t *)req) =
        (uint64_t)((roffset << 24) | (op << 16) | (group_size << 8) | curr_flag);

    /* put_dma_request_0
     * 32 | 16 | 8 | 8
     * loffset_high | loffset_low | resv | flag */
    req++;
    *((volatile uint64_t *)req) = (uint64_t)(loffset << 16 | curr_flag);

    /* put_dma_request_1
     * 32 | 16 | 8 | 8
     * size_high | size_low | resv | flag */
    req++;
    *((volatile uint64_t *)req) = (uint64_t)(size_u64 << 16 | curr_flag);

    /* put_dma_request_2
     * 32 | 16 | 8 | 8
     * resv2 | pe | resv1 | flag */
    req++;
    *((volatile uint64_t *)req) = (uint64_t)((pe_u16 << 16) | curr_flag);
}

/*XXX : Only no const version is used*/
template <nvshmemi_op_t channel_op>
__device__ void nvshmemi_proxy_rma(void *rptr, void *lptr, size_t bytes, int pe) {
    assert(0);
    /*XXX:to be used for 1) inline 2) DMA with ack 3) DMA by staging in another buffer*/
}
template __device__ void nvshmemi_proxy_rma<NVSHMEMI_OP_PUT>(void *rptr, void *lptr, size_t bytes,
                                                        int pe);
template __device__ void nvshmemi_proxy_rma<NVSHMEMI_OP_GET>(void *rptr, void *lptr, size_t bytes,
                                                        int pe);


template<typename T>
__device__ T nvshmemi_proxy_rma_g(void * source, int pe) {
    uint64_t counter = atomicAdd((unsigned long long int *) &proxy_channel_g_buf_head_d, 1);
    uint64_t idx = counter*sizeof(g_elem_t);
    uint64_t idx_in_buf = idx & (proxy_channel_g_buf_size_d - 1);
    g_elem_t *elem = (g_elem_t *)(proxy_channel_g_buf_d + idx_in_buf);
    uint64_t flag = (idx >> proxy_channel_g_buf_log_size_d)*2;

    /* wait until elemet can be used */
    while(elem->flag < flag);

    nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_G>(source, (void *)elem, sizeof(T), pe);
    nvshmemi_proxy_quiet_no_membar();

    T return_val = *(T *)(&(elem->data));
    __threadfence();
    /* release the element for the next thread */
    elem->flag += 2;

    return return_val;
}
template __device__ char nvshmemi_proxy_rma_g<char>(void* source, int pe);
template __device__ unsigned char nvshmemi_proxy_rma_g<unsigned char>(void *source, int pe);
template __device__ short nvshmemi_proxy_rma_g<short>(void *source, int pe);
template __device__ unsigned short nvshmemi_proxy_rma_g<unsigned short>(void *source, int pe);
template __device__ int nvshmemi_proxy_rma_g<int>(void *source, int pe);
template __device__ unsigned int nvshmemi_proxy_rma_g<unsigned int>(void *source, int pe);
template __device__ long nvshmemi_proxy_rma_g<long>(void *source, int pe);
template __device__ unsigned long nvshmemi_proxy_rma_g<unsigned long>(void *source, int pe);
template __device__ float nvshmemi_proxy_rma_g<float>(void *source, int pe);
template __device__ long long nvshmemi_proxy_rma_g<long long>(void *source, int pe);
template __device__ unsigned long long nvshmemi_proxy_rma_g<unsigned long long>(void *source, int pe);
template __device__ double nvshmemi_proxy_rma_g<double>(void *source, int pe);

template <nvshmemi_op_t channel_op>
__device__ void nvshmemi_proxy_rma_nbi(void *rptr, void *lptr, size_t bytes, int pe) {
    if (!bytes) return;
    transfer_dma(rptr, lptr, bytes, pe, channel_op);
}
template __device__ void nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_PUT>(void *rptr, void *lptr, size_t bytes,
                                                            int pe);
template __device__ void nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_GET>(void *rptr, void *lptr, size_t bytes,
                                                            int pe);
template __device__ void nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_G>(void *rptr, void *lptr, size_t bytes,
                                                            int pe);

template <typename T>
__forceinline__ __device__ void transfer_inline(void *rptr, T value, int pe,
                                                nvshmemi_op_t optype) {
    uint64_t idx, tail_idx, *req;
    int size = sizeof(channel_request_t);
    int group_size = 1;
    void *buf_ptr = proxy_channels_buf_d;
    void *base_ptr = nvshmemi_heap_base_d;

    idx = atomicAdd((unsigned long long int *)proxy_channels_issue_d, size);
    tail_idx = idx + (size - 1);

    // flow-control
    check_channel_availability(tail_idx);

    req = (uint64_t *)((uint8_t *)buf_ptr + (idx & (proxy_channel_buf_size_d - 1)));
    uint64_t curr_flag = !((idx >> proxy_channel_buf_logsize_d) & 1);
    uint64_t roffset = (uint64_t)((char *)rptr - (char *)base_ptr);
    uint64_t op = optype;
    uint16_t pe_u16 = pe;
    uint64_t size_u64 = sizeof(T);

    // assumption that wrap around does not occur in the middle of the request,
    // issue is always incremenet in multiples of sizeof(channel_request_t)

    /* base_request_t
     * 32 | 8 | 8 | 8 | 8
     * roffset_high | roffset_low | op | group_size | flag */
    *((volatile uint64_t *)req) =
        (uint64_t)((roffset << 24) | (op << 16) | (group_size << 8) | curr_flag);

    /* put_inline_request_0
     * 32 | 16 | 8 | 8
     * lvalue (low) | pe | resv | flag */
    req++;
    uint64_t lvalue_low = *((uint32_t *)&value);
    *((volatile uint64_t *)req) = (lvalue_low << 32 | ((uint64_t)pe_u16 << 16) | curr_flag);

    /* put_inline_request_1
     * 32 | 16 | 8 | 8
     * lvalue(high) | size | resv | flag */
    req++;
    uint64_t lvalue_high = (size_u64 > 4) ? ((*((uint64_t *)&value)) >> 32) : 0;
    *((volatile uint64_t *)req) = (lvalue_high << 32 | size_u64 << 16 | curr_flag);

    /* update flags on all 64bit chunks of the channel request size */
    req++;
    *((volatile uint64_t *)req) = (uint64_t)(curr_flag);
}

template <typename T>
__device__ void nvshmemi_proxy_rma_p(void *rptr, const T value, int pe) {
    transfer_inline<T>(rptr, value, pe, NVSHMEMI_OP_P);
}

template __device__ void nvshmemi_proxy_rma_p<char>(void *rptr, const char value,
                                        int pe);
template __device__ void nvshmemi_proxy_rma_p<unsigned char>(
    					void *rptr, const unsigned char value, int pe);
template __device__ void nvshmemi_proxy_rma_p<short>(void *rptr,
                                        const short value, int pe);
template __device__ void nvshmemi_proxy_rma_p<unsigned short>(
    					void *rptr, const unsigned short value, int pe);
template __device__ void nvshmemi_proxy_rma_p<int>(void *rptr, const int value,
                                        int pe);
template __device__ void nvshmemi_proxy_rma_p<unsigned int>(
    					void *rptr, const unsigned int value, int pe);
template __device__ void nvshmemi_proxy_rma_p<long>(void *rptr, const long value,
                                        int pe);
template __device__ void nvshmemi_proxy_rma_p<unsigned long>(
    					void *rptr, const unsigned long value, int pe);
template __device__ void nvshmemi_proxy_rma_p<long long>(void *rptr, const long long value,
                                        int pe);
template __device__ void nvshmemi_proxy_rma_p<unsigned long long>(
    					void *rptr, const unsigned long long value, int pe);
template __device__ void nvshmemi_proxy_rma_p<float>(void *rptr,
                                        const float value, int pe);
template __device__ void nvshmemi_proxy_rma_p<double>(void *rptr,
                                        const double value, int pe);

template <typename T>
__device__ void amo (void *rptr, T swap_add, T compare, int pe, nvshmemi_amo_t amo_op) {
    uint64_t idx, tail_idx, *req;
    int size = sizeof(channel_request_t);
    int group_size = 1;
    void *buf_ptr = proxy_channels_buf_d;
    void *base_ptr = nvshmemi_heap_base_d;

    idx = atomicAdd((unsigned long long int *)proxy_channels_issue_d, size);
    tail_idx = idx + (size - 1);

    // flow-control
    check_channel_availability(tail_idx);

    req = (uint64_t *)((uint8_t *)buf_ptr + (idx & (proxy_channel_buf_size_d - 1)));
    uint64_t curr_flag = !((idx >> proxy_channel_buf_logsize_d) & 1);
    uint64_t roffset = (uint64_t)((char *)rptr - (char *)base_ptr);
    uint64_t op = NVSHMEMI_OP_AMO;
    uint64_t amo = amo_op;
    uint16_t pe_u16 = pe;
    uint64_t size_u64 = sizeof(T);

    // assumption that wrap around does not occur in the middle of the request,
    // issue is always incremenet in multiples of sizeof(channel_request_t)

    /* base_request_t
     * 32 | 8 | 8 | 8 | 8
     * roffset_high | roffset_low | op | group_size | flag */
    *((volatile uint64_t *)req) =
        (uint64_t)((roffset << 24) | (op << 16) | (group_size << 8) | curr_flag);

    /* amo_request_0
     * 32 | 16 | 8 | 8
     * swap_add_low | pe | amo | flag */
    req++;
    uint64_t swap_add_low = *((uint32_t *)&swap_add);
    *((volatile uint64_t *)req) = (swap_add_low << 32 | ((uint64_t)pe_u16 << 16) | (amo << 8)| curr_flag);

    /* amo_request_1
     * 32 | 16 | 8 | 8
     * swap_add_high | size | compare_low | flag */
    req++;
    uint64_t swap_add_high = (size_u64 > 4) ? ((*((uint64_t *)&swap_add)) >> 32) : 0;
    uint64_t compare_low = *((uint8_t *)&compare);
    *((volatile uint64_t *)req) = (swap_add_high << 32 | size_u64 << 16 | compare_low << 8 | curr_flag);

    /* amo_request_2
     * 32 | 16 | 8 | 8
     * comapare_high | flag */
    req++;
    uint64_t compare_high = (size_u64 > 4) ? ((*((uint64_t *)&compare)) >> 32) : 0;
    *((volatile uint64_t *)req) = (compare_high << 8 | curr_flag);
}

#define NVSHMEMI_REPT_FOR_STANDARD_AMO_TYPES(NVSHMEMI_FN_TEMPLATE) \
   NVSHMEMI_FN_TEMPLATE(short) \
   NVSHMEMI_FN_TEMPLATE(unsigned short) \
   NVSHMEMI_FN_TEMPLATE(int) \
   NVSHMEMI_FN_TEMPLATE(unsigned int) \
   NVSHMEMI_FN_TEMPLATE(long) \
   NVSHMEMI_FN_TEMPLATE(unsigned long) \
   NVSHMEMI_FN_TEMPLATE(long long) \
   NVSHMEMI_FN_TEMPLATE(unsigned long long)

#define NVSHMEMI_REPT_FOR_EXTENDED_AMO_TYPES(NVSHMEMI_FN_TEMPLATE) \
   NVSHMEMI_FN_TEMPLATE(float) \
   NVSHMEMI_FN_TEMPLATE(double)

template <typename T>
__device__ void nvshmemi_proxy_amo_nonfetch(void *rptr, T swap_add, int pe, nvshmemi_amo_t op) {
    return amo<T>(rptr, swap_add, 0, pe, op);
}

#define NVSHMEMI_DECL_PROXY_AMO_NONFETCH(Type) \
template __device__ void nvshmemi_proxy_amo_nonfetch<Type>(void *rptr, Type swap_add, int  pe, nvshmemi_amo_t op);

NVSHMEMI_REPT_FOR_STANDARD_AMO_TYPES(NVSHMEMI_DECL_PROXY_AMO_NONFETCH)
NVSHMEMI_REPT_FOR_EXTENDED_AMO_TYPES(NVSHMEMI_DECL_PROXY_AMO_NONFETCH)

template <typename T>
__device__ void nvshmemi_proxy_amo_fetch(void *rptr, void *lptr, T swap_add, 
		T compare, int pe, nvshmemi_amo_t op) {
    uint64_t counter = atomicAdd((unsigned long long int *) &proxy_channel_g_buf_head_d, 1);
    uint64_t idx = counter*sizeof(g_elem_t);
    uint64_t idx_in_buf = idx & (proxy_channel_g_buf_size_d - 1);
    g_elem_t *elem = (g_elem_t *)(proxy_channel_g_buf_d + idx_in_buf);
    uint64_t flag = (idx >> proxy_channel_g_buf_log_size_d)*2;

    /* wait until elemet can be used */
    nvshmemi_wait_until_greater_than_equals<uint64_t>((volatile uint64_t *)&elem->flag, flag, NVSHMEMI_CALL_SITE_AMO_FETCH_WAIT_FLAG);

    amo<T>(rptr, swap_add, compare, pe, op);

    nvshmemi_wait_until_greater_than_equals<uint64_t>((volatile uint64_t *)&elem->flag, flag + 1, NVSHMEMI_CALL_SITE_AMO_FETCH_WAIT_DATA);
    __threadfence();

    T return_val = *(T *)(&(elem->data));
    __threadfence();

    /* release the element for the next thread */
    elem->flag += 1;
    
    *((T *)lptr) = return_val;
}

#define NVSHMEMI_DECL_PROXY_AMO_FETCH(Type) \
template __device__ void nvshmemi_proxy_amo_fetch<Type>(void *rptr, void *lptr, Type swap_add, \
	Type compare, int  pe, nvshmemi_amo_t op);

NVSHMEMI_REPT_FOR_STANDARD_AMO_TYPES(NVSHMEMI_DECL_PROXY_AMO_FETCH)
NVSHMEMI_REPT_FOR_EXTENDED_AMO_TYPES(NVSHMEMI_DECL_PROXY_AMO_FETCH)

#undef NVSHMEMI_DECL_PROXY_AMO_FETCH
#undef NVSHMEMI_DECL_PROXY_AMO_NONFETCH
#undef NVSHMEMI_REPT_FOR_STANDARD_AMO_TYPES
#undef NVSHMEMI_REPT_FOR_EXTENDED_AMO_TYPES

__device__ void nvshmemi_proxy_fence() {
    // making it a no-op as it is a no-op for IB RC, the only transport
    uint64_t idx, tail_idx, *req;
    int size = sizeof(channel_request_t);

    // assumption that wrap around does not occur in the middle of the request,
    // issue is always incremenet in multiples of sizeof(channel_request_t)

    idx = atomicAdd((unsigned long long int *)proxy_channels_issue_d, size);
    tail_idx = idx + (size - 1);

    // flow-control
    check_channel_availability(tail_idx);

    req = (uint64_t *)((uint64_t)proxy_channels_buf_d + (idx & (proxy_channel_buf_size_d - 1)));
    uint64_t curr_flag = !((idx >> proxy_channel_buf_logsize_d) & 1);
    uint64_t op = NVSHMEMI_OP_FENCE;

    /* base_request_t
     * 32 | 8 | 8 | 8 | 8
     * resv | resv | op | resv | flag */
    *((volatile uint64_t *)req) = (uint64_t)((op << 16) | curr_flag);

    /* update flags on all 64bit chunks of the channel request size */
    req++;
    *((volatile uint64_t *)req) = (uint64_t)(curr_flag);
    req++;
    *((volatile uint64_t *)req) = (uint64_t)(curr_flag);
    req++;
    *((volatile uint64_t *)req) = (uint64_t)(curr_flag);

    return;
}

int nvshmemi_proxy_setup_device_channels(proxy_state_t *state) {
    int status = 0;

    CUDA_RUNTIME_CHECK(cudaMemcpyToSymbol(proxy_channel_buf_size_d,
                                          (const void *)&state->channel_bufsize, sizeof(uint64_t),
                                          0, cudaMemcpyHostToDevice));

    CUDA_RUNTIME_CHECK(cudaMemcpyToSymbol(proxy_channel_buf_logsize_d,
                                          (const void *)&state->channel_bufsize_log,
                                          sizeof(uint32_t), 0, cudaMemcpyHostToDevice));

    CUDA_CHECK(cuMemAlloc((CUdeviceptr *)&state->channels_device,
                          sizeof(proxy_channel_t) * state->channel_count));

    INFO(NVSHMEM_PROXY, "channel buf: %p complete: %p quiet_issue: %p quiet_ack: %p",
         state->channels[0].buf, state->channels[0].complete, state->channels[0].quiet_issue,
         state->channels[0].quiet_ack);

    uint64_t *temp_buf_dptr;
    uint64_t *temp_complete_dptr;
    uint64_t *temp_quiet_issue_dptr;
    uint64_t *temp_quiet_ack_dptr;
    uint64_t *temp_cst_issue_dptr;
    uint64_t *temp_cst_ack_dptr;
    nvshmemi_timeout_t *nvshmemi_timeout_dptr;

    CUDA_CHECK(cuMemHostGetDevicePointer((CUdeviceptr *)&temp_buf_dptr, state->channels[0].buf, 0));
    CUDA_CHECK(cuMemHostGetDevicePointer((CUdeviceptr *)&temp_complete_dptr,
                                         state->channels[0].complete, 0));
    CUDA_CHECK(cuMemHostGetDevicePointer((CUdeviceptr *)&temp_quiet_issue_dptr,
                                         state->channels[0].quiet_issue, 0));
    CUDA_CHECK(cuMemHostGetDevicePointer((CUdeviceptr *)&temp_quiet_ack_dptr,
                                         state->channels[0].quiet_ack, 0));
    CUDA_CHECK(cuMemHostGetDevicePointer((CUdeviceptr *)&temp_cst_issue_dptr,
                                         state->channels[0].cst_issue, 0));
    CUDA_CHECK(cuMemHostGetDevicePointer((CUdeviceptr *)&temp_cst_ack_dptr,
                                         state->channels[0].cst_ack, 0));
    CUDA_CHECK(cuMemHostGetDevicePointer((CUdeviceptr *)&nvshmemi_timeout_dptr,
                                         state->nvshmemi_timeout, 0));

    INFO(NVSHMEM_PROXY,
         "channel device_ptr buf: %p issue: %p complete: %p quiet_issue: %p quiet_ack: %p \n",
         temp_buf_dptr, state->channels[0].issue, temp_complete_dptr, temp_quiet_issue_dptr,
         temp_quiet_ack_dptr);

    CUDA_RUNTIME_CHECK(cudaMemcpyToSymbol(proxy_channels_buf_d, &temp_buf_dptr, sizeof(uint64_t *),
                                          0, cudaMemcpyHostToDevice));
    CUDA_RUNTIME_CHECK(cudaMemcpyToSymbol(proxy_channels_issue_d, &state->channels[0].issue,
                                          sizeof(uint64_t *), 0, cudaMemcpyHostToDevice));
    CUDA_RUNTIME_CHECK(cudaMemcpyToSymbol(proxy_channels_complete_d, &temp_complete_dptr,
                                          sizeof(uint64_t *), 0, cudaMemcpyHostToDevice));
    CUDA_RUNTIME_CHECK(cudaMemcpyToSymbol(proxy_channels_quiet_issue_d, &temp_quiet_issue_dptr,
                                          sizeof(uint64_t *), 0, cudaMemcpyHostToDevice));
    CUDA_RUNTIME_CHECK(cudaMemcpyToSymbol(proxy_channels_quiet_ack_d, &temp_quiet_ack_dptr,
                                          sizeof(uint64_t *), 0, cudaMemcpyHostToDevice));
    CUDA_RUNTIME_CHECK(cudaMemcpyToSymbol(proxy_channels_cst_issue_d, &temp_cst_issue_dptr,
                                          sizeof(uint64_t *), 0, cudaMemcpyHostToDevice));
    CUDA_RUNTIME_CHECK(cudaMemcpyToSymbol(proxy_channels_cst_ack_d, &temp_cst_ack_dptr,
                                          sizeof(uint64_t *), 0, cudaMemcpyHostToDevice));
    CUDA_RUNTIME_CHECK(cudaMemcpyToSymbol(nvshmemi_timeout_d, &nvshmemi_timeout_dptr,
                                          sizeof(nvshmemi_timeout_t *), 0, cudaMemcpyHostToDevice));

    proxy_channel_g_buf_size =  NUM_G_BUF_ELEMENTS * sizeof(g_elem_t);
    proxy_channel_g_buf_log_size = (uint64_t)log2((double)proxy_channel_g_buf_size);
    proxy_channel_g_buf_head = 0;
    proxy_channel_g_buf = (char *)nvshmemi_malloc(proxy_channel_g_buf_size);
    NULL_ERROR_JMP(proxy_channel_g_buf, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, 
		    out, "failed allocating proxy_channel_g_buf");

    CUDA_RUNTIME_CHECK(cudaMemcpyToSymbol(proxy_channel_g_buf_size_d, &proxy_channel_g_buf_size, sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
    CUDA_RUNTIME_CHECK(cudaMemcpyToSymbol(proxy_channel_g_buf_log_size_d, &proxy_channel_g_buf_log_size, sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
    CUDA_RUNTIME_CHECK(cudaMemcpyToSymbol(proxy_channel_g_buf_head_d, &proxy_channel_g_buf_head, sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
    CUDA_RUNTIME_CHECK(cudaMemcpyToSymbol(proxy_channel_g_buf_d, &proxy_channel_g_buf, sizeof(char *), 0, cudaMemcpyHostToDevice));

    assert(proxy_channel_g_buf_size % sizeof(g_elem_t) == 0);

out:
    return status;
}
