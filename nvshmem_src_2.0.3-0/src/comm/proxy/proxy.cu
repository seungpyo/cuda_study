/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */
#include "nvshmem.h"
#include <unistd.h>
#include "proxy.h"
#include "nvshmem_internal.h"
#include "nvshmemx_error.h"
#include "util.h"

// progress channels
static base_request_t **channel_req;

void *nvshmemi_proxy_progress(void *in);

inline void proxy_update_processed(proxy_channel_t *ch, int bytes) {
    ch->processed += bytes;

    if ((ch->processed - ch->last_sync) >= 1024) {
        *ch->complete = ch->processed;
        ch->last_sync = ch->processed;
        TRACE(NVSHMEM_PROXY, "updated processed to device %llu", ch->processed);
    }
}

int nvshmemi_proxy_create_channels(proxy_state_t *proxy_state) {
    int status = 0;

    proxy_channel_t *channels =
        (proxy_channel_t *)malloc(sizeof(proxy_channel_t) * proxy_state->channel_count);
    NULL_ERROR_JMP(channels, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                   "failed allocating channels");
    memset(channels, 0, sizeof(proxy_channel_t) * proxy_state->channel_count);

    CUDA_CHECK(cuMemAllocHost((void **)&proxy_state->nvshmemi_timeout,
                              sizeof(nvshmemi_timeout_t))); /* GPU writes, CPU reads */
    memset(proxy_state->nvshmemi_timeout, 0, sizeof(nvshmemi_timeout_t));

    for (int i = 0; i < proxy_state->channel_count; i++) {
        // for put/get
        CUDA_CHECK(cuMemAllocHost((void **)&channels[i].buf,
                                  proxy_state->channel_bufsize)); /* CPU reads, GPU writes */
        memset(channels[i].buf, 0, proxy_state->channel_bufsize);
        assert(proxy_state->channel_bufsize % sizeof(channel_request_t) == 0);

        CUDA_CHECK(cuMemAlloc((CUdeviceptr *)&channels[i].issue,
                              sizeof(uint64_t))); /* issue is not accessed through LD/ST by CPU
                                                     thread, therefore on device memory */
        CUDA_CHECK(cuMemsetD8((CUdeviceptr)channels[i].issue, 0, sizeof(uint64_t)));

        CUDA_CHECK(cuMemAllocHost((void **)&channels[i].complete,
                                  sizeof(uint64_t))); /* CPU writes, GPU reads */
        CUDA_CHECK(cuMemAllocHost((void **)&channels[i].quiet_issue,
                                  sizeof(uint64_t))); /* CPU reads, GPU writes */
        CUDA_CHECK(cuMemAllocHost((void **)&channels[i].quiet_ack,
                                  sizeof(uint64_t))); /* CPU writes, GPU reads */
        CUDA_CHECK(cuMemAllocHost((void **)&channels[i].cst_issue,
                                  sizeof(uint64_t))); /* CPU reads, GPU writes */
        CUDA_CHECK(cuMemAllocHost((void **)&channels[i].cst_ack,
                                  sizeof(uint64_t))); /* CPU writes, GPU reads */

        *channels[i].complete = 0;
        *channels[i].quiet_issue = 0;
        *channels[i].quiet_ack = 0;
        channels[i].last_quiet_issue = 0;
        *channels[i].cst_issue = 0;
        *channels[i].cst_ack = 0;
        channels[i].last_cst_issue = 0;
    }

    proxy_state->channels = channels;

out:
    return status;
}

int nvshmemi_proxy_setup_connections(proxy_state_t *proxy_state) {
    int status;
    nvshmemi_state_t *state = proxy_state->nvshmemi_state;
    nvshmemt_ep_handle_t *local_ep_handles = NULL, *ep_handles = NULL;
    nvshmemt_ep_t *ep = NULL;
    struct nvshmem_transport **transport = NULL;
    int *transport_id;

    ep = proxy_state->ep = (nvshmemt_ep_t *)calloc(state->npes, sizeof(nvshmemt_ep_t));
    NULL_ERROR_JMP(ep, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                   "failed allocating space for endpoints \n");

    transport = proxy_state->transport =
        (struct nvshmem_transport **)calloc(state->npes, sizeof(nvshmemt_ep_t));
    NULL_ERROR_JMP(transport, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                   "failed allocating space for transports \n");

    transport_id = proxy_state->transport_id = (int *)calloc(state->npes, sizeof(int));
    NULL_ERROR_JMP(transport_id, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                   "failed allocating space for transport id \n");

    // this can just be npes long if alltoall is used instead of allgather
    local_ep_handles = (nvshmemt_ep_handle_t *)calloc(state->npes, sizeof(nvshmemt_ep_handle_t));
    NULL_ERROR_JMP(local_ep_handles, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                   "failed allocating space for ep handles \n");

    ep_handles = (nvshmemt_ep_handle_t *)calloc(state->npes, sizeof(nvshmemt_ep_handle_t));
    NULL_ERROR_JMP(ep_handles, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                   "failed allocating space for ep handles \n");

    for (int j = 0; j < state->npes; j++) {
        for (int i = 0; i < NVSHMEM_TRANSPORT_COUNT; i++) {
            int transport_bit = (1 << i);
            // assumes symmetry of transport list at all PEs
            if (!((state->transport_bitmap) & transport_bit)) continue;
            struct nvshmem_transport *tcurr = state->transports[i];

            // finding the first transport with CPU WRITE capability
            if (!(tcurr->cap[j] &
                  (NVSHMEM_TRANSPORT_CAP_CPU_WRITE | NVSHMEM_TRANSPORT_CAP_CPU_READ)))
                continue;

            // assuming the transport is connected - IB RC
            assert(tcurr->attr & NVSHMEM_TRANSPORT_ATTR_CONNECTED);

            int dev_id = 0;
            dev_id = tcurr->dev_id;
            transport[j] = tcurr;
            transport_id[j] = i;
            proxy_state->transport_bitmap |= transport_bit;
            status = tcurr->host_ops.ep_create((ep + j), dev_id, tcurr);
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "transport create ep failed \n");

            status = tcurr->host_ops.ep_get_handle(local_ep_handles + j, ep[j]);
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "transport get ep handle failed \n");

            break;
        }
    }

    // this could be more efficient with an alltoall
    status = state->boot_handle.alltoall((void *)local_ep_handles, (void *)ep_handles,
                                         sizeof(nvshmemt_ep_handle_t), &state->boot_handle);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "allgather of ep handles failed \n");

    for (int j = 0; j < state->npes; j++) {
        struct nvshmem_transport *tcurr = transport[j];

        status = tcurr->host_ops.ep_connect(ep[j], ep_handles[j]);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "transport create connect failed \n");
    }

    status = state->boot_handle.barrier(&state->boot_handle);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "barrier failed \n");

out:
    if (status) {
        if (ep) free(ep);
        if (transport) free(transport);
    }
    if (local_ep_handles) free(local_ep_handles);
    if (ep_handles) free(ep_handles);
    return status;
}

int nvshmemi_proxy_init(nvshmemi_state_t *state) {
    int status = 0;

    INFO(NVSHMEM_PROXY, "[%d] in proxy_init", state->mype);

    proxy_state_t *proxy_state = (proxy_state_t *)malloc(sizeof(proxy_state_t));

    proxy_state->channel_bufsize_log = CHANNEL_BUF_SIZE_LOG;
    proxy_state->channel_bufsize = (1 << CHANNEL_BUF_SIZE_LOG);
    proxy_state->channel_count = CHANNEL_COUNT;
    proxy_state->nvshmemi_state = state;

    status = nvshmemi_proxy_create_channels(proxy_state);
    if (status) {
        fprintf(stderr, "channel creation failed \n");
        exit(-1);
    }

    status = nvshmemi_proxy_setup_device_channels(proxy_state);
    if (status) {
        fprintf(stderr, "channel creation failed \n");
        exit(-1);
    }

    status = nvshmemi_proxy_setup_connections(proxy_state);
    if (status) {
        fprintf(stderr, "connection setup failed \n");
        exit(-1);
    }

    INFO(NVSHMEM_PROXY, "[%d] after setting up proxy channels on device", state->mype);

    CUDA_CHECK(cuStreamCreate(&proxy_state->stream, CU_STREAM_NON_BLOCKING));
    CUDA_CHECK(cuStreamCreate(&proxy_state->queue_stream_out, CU_STREAM_NON_BLOCKING));
    CUDA_CHECK(cuStreamCreate(&proxy_state->queue_stream_in, CU_STREAM_NON_BLOCKING));
    CUDA_CHECK(cuEventCreate(&proxy_state->cuev, CU_EVENT_DEFAULT));

    proxy_state->progress_params.state = proxy_state;
    proxy_state->progress_params.stop = 0;
    proxy_state->finalize_count = 0;
    proxy_state->quiet_in_progress = PROXY_QUIET_STATUS_CHANNELS_INACTIVE;
    proxy_state->cst_in_progress = PROXY_CST_STATUS_CHANNELS_INACTIVE;
    proxy_state->issued_get = 0;

    INFO(NVSHMEM_PROXY, "[%d] creating proxy thread", state->mype);

    status = pthread_create(&proxy_state->progress_thread, NULL, nvshmemi_proxy_progress,
                            (void *)&proxy_state->progress_params);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "pthread creation failed \n");

    state->proxy = (void *)proxy_state;

out:
    return status;
}

inline int process_channel_dma(proxy_state_t *state, proxy_channel_t *ch, int *is_processed) {
    int status = 0;
    uint64_t *base_ptr;
    base_request_t *base_req;
    put_dma_request_0_t *dma_req_0;
    put_dma_request_1_t *dma_req_1;
    put_dma_request_2_t *dma_req_2;
    int pe;
    size_t size;
    uint8_t flag;
    uint64_t roffset, loffset;
    nvshmemi_state_t *nvshmemi_state = state->nvshmemi_state;

    base_ptr = (uint64_t *)WRAPPED_CHANNEL_BUF(state, ch, ch->processed);
    flag = COUNTER_TO_FLAG(state, ch->processed);

    base_req = (base_request_t *)base_ptr;
    roffset = (uint64_t)((base_req->roffset_high << 8) | (base_req->roffset_low));

    base_ptr++;
    dma_req_0 = (put_dma_request_0_t *)base_ptr;
    while (*((volatile uint8_t *)&dma_req_0->flag) != flag)
        ;

    base_ptr++;
    dma_req_1 = (put_dma_request_1_t *)base_ptr;
    while (*((volatile uint8_t *)&dma_req_1->flag) != flag)
        ;

    base_ptr++;
    dma_req_2 = (put_dma_request_2_t *)base_ptr;
    while (*((volatile uint8_t *)&dma_req_2->flag) != flag)
        ;

#if defined(NVSHMEM_PPC64LE)
    __sync_synchronize();  // XXX : prevents load from buf_d reordered to before load from issue_d
                           // (breaks rma)
#elif defined(NVSHMEM_X86_64)
    asm volatile("" : : : "memory");
#endif
    loffset = (uint64_t)((dma_req_0->loffset_high << 16) | (dma_req_0->loffset_low));
    size = (size_t)((dma_req_1->size_high << 16) | (dma_req_1->size_low));
    pe = dma_req_2->pe;
    TRACE(NVSHMEM_PROXY, "process_channel_dma loffset %p pe %d", loffset, pe);

    // issue transport DMA
    {
        rma_memdesc_t localdesc, remotedesc;
        rma_bytesdesc_t bytes;
        rma_verb_t verb;
        void *remote_actual =
            (void *)((char *)(nvshmemi_state->peer_heap_base_actual[pe]) + roffset);
        void *local = (void *)((char *)(nvshmemi_state->heap_base) + loffset);
        struct nvshmem_transport *tcurr = state->transport[pe];
        int t = state->transport_id[pe];
        nvshmemt_ep_t ep = state->ep[pe];
        nvshmem_mem_handle_t *handles = nvshmemi_state->handles;
        int tcount = nvshmemi_state->transport_count;

        // incrementing G buf head corresponding to the device
        if ((nvshmemi_op_t)base_req->op == NVSHMEMI_OP_G) proxy_channel_g_buf_head++;

        verb.desc = (nvshmemi_op_t)base_req->op;
        verb.is_nbi = 1;

        localdesc.ptr = local;
        localdesc.handle = handles[nvshmemi_state->mype * tcount + t];
        remotedesc.ptr = remote_actual;
        remotedesc.handle = handles[pe * tcount + t];

        bytes.nelems = size;
        bytes.elembytes = 1;

        status = tcurr->host_ops.rma(ep, verb, remotedesc, localdesc, bytes);
        if (unlikely(status)) {
            ERROR_PRINT("aborting due to error in process_channel_dma\n");
            exit(-1);
        }
    }
#if defined(NVSHMEM_PPC64LE)
    __sync_synchronize();  // XXX: prevents complete_d store reordered to before return from
                           // ibv_post_send (breaks rma -> quiet)
#endif

    *is_processed = 1;

    proxy_update_processed(ch, sizeof(channel_request_t));
    TRACE(NVSHMEM_PROXY,
          "[%d] process_channel_put_dma/proxy_update_processed processed %ld complete %ld",
          state->nvshmemi_state->mype, ch->processed, *ch->complete);

    return status;
}

inline int process_channel_inline(proxy_state_t *state, proxy_channel_t *ch, int *is_processed) {
    int status = 0;
    uint64_t *base_ptr;
    base_request_t *base_req;
    put_inline_request_0_t *inline_req_0;
    put_inline_request_1_t *inline_req_1;
    uint8_t flag;
    uint64_t roffset;
    nvshmemi_state_t *nvshmemi_state = state->nvshmemi_state;

    base_ptr = (uint64_t *)WRAPPED_CHANNEL_BUF(state, ch, ch->processed);
    flag = COUNTER_TO_FLAG(state, ch->processed);

    base_req = (base_request_t *)base_ptr;
    roffset = (uint64_t)((base_req->roffset_high << 8) | (base_req->roffset_low));

    base_ptr++;
    inline_req_0 = (put_inline_request_0_t *)base_ptr;
    while (*((volatile uint8_t *)&inline_req_0->flag) != flag)
        ;

    base_ptr++;
    inline_req_1 = (put_inline_request_1_t *)base_ptr;
    while (*((volatile uint8_t *)&inline_req_1->flag) != flag)
        ;

#if defined(NVSHMEM_PPC64LE)
    __sync_synchronize();  // XXX : prevents load from buf_d reordered to before load from issue_d
                           // (was present in dma function, was missing in inline function, breaks
                           // rma)
#elif defined(NVSHMEM_X86_64)
    asm volatile("" : : : "memory");
#endif

    uint32_t pe = inline_req_0->pe;
    uint32_t size = inline_req_1->size;
    uint64_t lvalue;

    lvalue = inline_req_0->lvalue_low;
    if (size == 8) {
        lvalue = lvalue | ((uint64_t)inline_req_1->lvalue_high << 32);
    }

    // issue transport DMA
    {
        rma_memdesc_t localdesc, remotedesc;
        rma_bytesdesc_t bytes;
        rma_verb_t verb;
        void *remote_actual =
            (void *)((char *)(nvshmemi_state->peer_heap_base_actual[pe]) + roffset);
        void *local = (void *)&lvalue;
        struct nvshmem_transport *tcurr = state->transport[pe];
        int t = state->transport_id[pe];
        nvshmemt_ep_t ep = state->ep[pe];
        nvshmem_mem_handle_t *handles = nvshmemi_state->handles;
        int tcount = nvshmemi_state->transport_count;

        verb.desc = NVSHMEMI_OP_P;
        verb.is_nbi = 0;

        localdesc.ptr = local;
        remotedesc.ptr = remote_actual;
        remotedesc.handle = handles[pe * tcount + t];

        bytes.nelems = 1;
        bytes.elembytes = size;

        status = tcurr->host_ops.rma(ep, verb, remotedesc, localdesc, bytes);
        if (unlikely(status)) {
            ERROR_PRINT("aborting due to error in process_channel_dma\n");
            exit(-1);
        }
    }
#if defined(NVSHMEM_PPC64LE)
    __sync_synchronize();  // XXX: prevents complete_d store reordered to before return from
                           // ibv_post_cq (breaks rma -> quiet)
#endif

    *is_processed = 1;

    proxy_update_processed(ch, sizeof(channel_request_t));
    TRACE(NVSHMEM_PROXY,
          "[%d] process_channel_put_dma/proxy_update_processed processed %ld complete %ld",
          state->nvshmemi_state->mype, ch->processed, *ch->complete);

    return status;
}

int process_channel_amo(proxy_state_t *state, proxy_channel_t *ch, int *is_processed) {
    int status = 0;
    uint64_t *base_ptr;
    base_request_t *base_req;
    amo_request_0_t *req_0;
    amo_request_1_t *req_1;
    amo_request_2_t *req_2;
    uint8_t flag;
    uint64_t roffset;
    nvshmemi_state_t *nvshmemi_state = state->nvshmemi_state;

    base_ptr = (uint64_t *)WRAPPED_CHANNEL_BUF(state, ch, ch->processed);
    flag = COUNTER_TO_FLAG(state, ch->processed);

    base_req = (base_request_t *)base_ptr;
    roffset = (uint64_t)((base_req->roffset_high << 8) | (base_req->roffset_low));

    base_ptr++;
    req_0 = (amo_request_0_t *)base_ptr;
    while (*((volatile uint8_t *)&req_0->flag) != flag)
        ;

    base_ptr++;
    req_1 = (amo_request_1_t *)base_ptr;
    while (*((volatile uint8_t *)&req_1->flag) != flag)
        ;

    base_ptr++;
    req_2 = (amo_request_2_t *)base_ptr;
    while (*((volatile uint8_t *)&req_2->flag) != flag)
        ;

#if defined(NVSHMEM_PPC64LE)
    __sync_synchronize();  // XXX : prevents load from buf_d reordered to before load from issue_d
                           // (was present in dma function, was missing in inline function, breaks
                           // rma)
#elif defined(_NVSHMEM_X86_64)
    asm volatile("" : : : "memory");
#endif

    uint32_t pe = req_0->pe;
    uint32_t size = req_1->size;
    nvshmemi_amo_t amo_op = (nvshmemi_amo_t)req_0->amo;
    uint64_t lvalue, cvalue;

    lvalue = req_0->swap_add_low;
    lvalue = lvalue | ((uint64_t)req_1->swap_add_high << 32);

    if (amo_op == NVSHMEMI_AMO_COMPARE_SWAP) {
        cvalue = (*((uint64_t *)req_2) & 0xFFFFFFFFFFFFFF00);
        cvalue |= req_1->compare_low;
    }

    // issue transport amo
    {
        amo_verb_t verb;
        amo_bytesdesc_t bytes;
        amo_memdesc_t memdesc;
        void *remote_actual =
            (void *)((char *)(nvshmemi_state->peer_heap_base_actual[pe]) + roffset);
        int t = state->transport_id[pe];
        struct nvshmem_transport *tcurr = state->transport[pe];
        nvshmemt_ep_t ep = state->ep[pe];
        nvshmem_mem_handle_t *handles = nvshmemi_state->handles;
        int tcount = nvshmemi_state->transport_count;

        verb.desc = amo_op;

        memset(&memdesc, 0, sizeof(amo_memdesc_t));
        memdesc.ptr = remote_actual;
        memdesc.val = lvalue;
        memdesc.cmp = cvalue;
        memdesc.handle = handles[pe * tcount + t];
        // pick spot in g buffer for fetch value
        if ((amo_op > NVSHMEMI_AMO_END_OF_NONFETCH)) {
            uint64_t offset =
                ((proxy_channel_g_buf_head * sizeof(g_elem_t)) & (proxy_channel_g_buf_size - 1));
            memdesc.retptr = (void *)(proxy_channel_g_buf + offset);
            memdesc.retflag = ((proxy_channel_g_buf_head * sizeof(g_elem_t)) >> proxy_channel_g_buf_log_size) * 2 + 1;
            proxy_channel_g_buf_head++;
        }
        bytes.elembytes = size;

        status = tcurr->host_ops.amo(ep, NULL, verb, memdesc, bytes);
        if (unlikely(status)) {
            ERROR_PRINT("aborting due to error in process_channel_dma\n");
            exit(-1);
        }
    }

#if defined(NVSHMEM_PPC64LE)
    __sync_synchronize();  // XXX: prevents complete_d store reordered to before return from
                           // ibv_post_cq (breaks rma -> quiet)
#endif

    *is_processed = 1;

    proxy_update_processed(ch, sizeof(channel_request_t));
    INFO(NVSHMEM_PROXY,
         "[%d] process_channel_put_dma/proxy_update_processed processed %ld complete %ld \n",
         state->nvshmemi_state->mype, ch->processed, *ch->complete);

    return status;
}

void enforce_cst(proxy_state_t *proxy_state) {
    nvshmemi_state_t *state = proxy_state->nvshmemi_state;
    int status = 0;

    if (nvshmemi_options.BYPASS_FLUSH) return;

#if defined(NVSHMEM_PPC64LE)
    status = cuEventRecord(proxy_state->cuev, proxy_state->stream);
    if (unlikely(status != CUDA_SUCCESS)) {
        ERROR_EXIT("cuEventRecord() failed in the proxy thread \n");
    }
#elif defined(NVSHMEM_X86_64)
    for (int i = 0; i < NVSHMEM_TRANSPORT_COUNT; i++) {
        if (!((state->transport_bitmap) & (1 << i))) continue;
        struct nvshmem_transport *tcurr = state->transports[i];
        if (!tcurr->host_ops.enforce_cst) continue;

        // assuming the transport is connected - IB RC
        if (tcurr->attr & NVSHMEM_TRANSPORT_ATTR_CONNECTED) {
            status = tcurr->host_ops.enforce_cst();
            if (status) {
                ERROR_PRINT("aborting due to error in progress_cst \n");
                exit(-1);
            }
        }
    }
#endif
}

inline void quiet_ack_channels(proxy_state_t *proxy_state) {
    for (int i = 0; i < proxy_state->channel_count; i++) {
        proxy_channel_t *ch = (proxy_state->channels + i);
        *((volatile uint64_t *)ch->quiet_ack) = ch->last_quiet_issue;
        TRACE(NVSHMEM_PROXY, "[%d] quiet_ack_channels quiet_ack %ld",
              proxy_state->nvshmemi_state->mype, *ch->quiet_ack);
    }
}

inline int quiet_channels_check(proxy_state_t *proxy_state) {
    int start_quiet = 0;

    for (int i = 0; i < proxy_state->channel_count; i++) {
        proxy_channel_t *ch = (proxy_state->channels + i);
        if (*((volatile uint64_t *)ch->quiet_issue) > ch->last_quiet_issue) {
            ch->last_quiet_issue = *((volatile uint64_t *)ch->quiet_issue);
            start_quiet = 1;
            TRACE(NVSHMEM_PROXY, "[%d] host proxy: received quiet on channel %d from GPU", getpid(),
                  i);
        }
    }

    return start_quiet;
}

inline int quiet_channels_test(proxy_state_t *proxy_state) {
    int processed = 1;

    for (int i = 0; i < proxy_state->channel_count; i++) {
        proxy_channel_t *ch = (proxy_state->channels + i);
        if (ch->processed < ch->last_quiet_issue) {
            TRACE(NVSHMEM_PROXY, "[%d] quiet_channels_test last_quiet_issue %ld processed %ld",
                  proxy_state->nvshmemi_state->mype, ch->last_quiet_issue, ch->processed);
            processed = 0;
        } else {
            TRACE(NVSHMEM_PROXY,
                  "processing quiet for channel %d from GPU "
                  "ch->processed: %llu ch->last_quiet_issue: %llu",
                  i, ch->processed, ch->last_quiet_issue);
        }
    }

    return processed;
}

inline void progress_quiet(proxy_state_t *proxy_state) {
    // quiet processing at source
    if (proxy_state->quiet_in_progress == PROXY_QUIET_STATUS_CHANNELS_INACTIVE) {
        if (quiet_channels_check(proxy_state)) {
            proxy_state->quiet_in_progress = PROXY_QUIET_STATUS_CHANNELS_IN_PROGRESS;
            TRACE(NVSHMEM_PROXY, "[%d] quiet_progress PROXY_QUIET_STATUS_CHANNELS_IN_PROGRESS",
                  proxy_state->nvshmemi_state->mype);
        }
    }

    if (proxy_state->quiet_in_progress == PROXY_QUIET_STATUS_CHANNELS_IN_PROGRESS) {
        if (quiet_channels_test(proxy_state)) {
            proxy_state->quiet_in_progress = PROXY_QUIET_STATUS_CHANNELS_DONE;
            TRACE(NVSHMEM_PROXY, "[%d] quiet_progress PROXY_QUIET_STATUS_CHANNELS_DONE",
                  proxy_state->nvshmemi_state->mype);
        }
    }

    if (proxy_state->quiet_in_progress == PROXY_QUIET_STATUS_CHANNELS_DONE) {
        nvshmemi_state_t *state = proxy_state->nvshmemi_state;

        // issue quiet on connections to all peers, we might want to make transport level quiet a
        // non-blocking call
        for (int i = 0; i < state->npes; i++) {
            struct nvshmem_transport *tcurr;
            nvshmemt_ep_t epcurr;
            int status = 0;

            tcurr = proxy_state->transport[i];
            if (tcurr == NULL) continue;
            epcurr = proxy_state->ep[i];
            status = tcurr->host_ops.quiet(epcurr);
            if (unlikely(status)) {
                ERROR_PRINT("aborting due to error in progress_quiet \n");
                exit(-1);
            }
        }
#if defined(NVSHMEM_PPC64LE)
        __sync_synchronize();  // XXX: prevents quiet_ack_d store reordered to before return from
                               // ibv_poll_cq
#endif

        if (proxy_state->issued_get) {
            enforce_cst(proxy_state);
            proxy_state->issued_get = 0;
        }

        quiet_ack_channels(proxy_state);
        proxy_state->quiet_in_progress = PROXY_QUIET_STATUS_CHANNELS_INACTIVE;
    }
}

inline void cst_ack_channels(proxy_state_t *proxy_state) {
    for (int i = 0; i < proxy_state->channel_count; i++) {
        proxy_channel_t *ch = (proxy_state->channels + i);
        *((volatile uint64_t *)ch->cst_ack) = ch->last_cst_issue;
        TRACE(NVSHMEM_PROXY, "[%d] cst_ack_channels cst_ack %ld", proxy_state->nvshmemi_state->mype,
              *ch->cst_ack);
    }
}

inline int cst_channels_check(proxy_state_t *proxy_state) {
    int start_cst = 0;

    for (int i = 0; i < proxy_state->channel_count; i++) {
        proxy_channel_t *ch = (proxy_state->channels + i);
        if (*((volatile uint64_t *)ch->cst_issue) > ch->last_cst_issue) {
            ch->last_cst_issue = *((volatile uint64_t *)ch->cst_issue);
            start_cst = 1;
            TRACE(NVSHMEM_PROXY, "[%d] host proxy: received cst on channel %d from GPU %ld",
                  proxy_state->nvshmemi_state->mype, i, *ch->cst_issue);
        }
    }

    return start_cst;
}

inline void progress_cst(proxy_state_t *proxy_state) {
    if (proxy_state->cst_in_progress == PROXY_CST_STATUS_CHANNELS_INACTIVE) {
        if (cst_channels_check(proxy_state)) {
            proxy_state->cst_in_progress = PROXY_CST_STATUS_CHANNELS_ACTIVE;
            TRACE(NVSHMEM_PROXY, "[%d] cst_progress PROXY_CST_STATUS_CHANNELS_IN_PROGRESS",
                  proxy_state->nvshmemi_state->mype);
        }
    }

    if (proxy_state->cst_in_progress == PROXY_CST_STATUS_CHANNELS_ACTIVE) {
        enforce_cst(proxy_state);
#if defined(NVSHMEM_PPC64LE)
        __sync_synchronize();  // XXX: prevents cst_ack_d store reordered to before return from
                               // cuEventRecord
#endif
        cst_ack_channels(proxy_state);
        proxy_state->cst_in_progress = PROXY_CST_STATUS_CHANNELS_INACTIVE;
    }
}

inline int process_channel_fence(proxy_state_t *proxy_state, proxy_channel_t *ch) {
    int status = 0;
    nvshmemi_state_t *state = proxy_state->nvshmemi_state;

    for (int i = 0; i < state->npes; i++) {
        struct nvshmem_transport *tcurr;
        nvshmemt_ep_t epcurr;

        if (i == state->mype) continue;

        tcurr = proxy_state->transport[i];
        epcurr = proxy_state->ep[i];

        if(tcurr->host_ops.fence)
            status = tcurr->host_ops.fence(epcurr);
        if (unlikely(status)) {
            ERROR_PRINT("aborting due to error in process_fence \n");
            exit(-1);
        }
    }

    proxy_update_processed(ch, sizeof(channel_request_t));
    return 0;
}

inline void progress_channels(proxy_state_t *proxy_state) {
    int status = 0;

    for (int i = 0; i < proxy_state->channel_count; i++) {
        proxy_channel_t *ch = (proxy_state->channels + i);

        uint64_t counter;
        int flag;

        counter = ch->processed;

        if (likely(channel_req[i] == NULL)) {
            flag = COUNTER_TO_FLAG(proxy_state, counter);
            channel_req[i] = (base_request_t *)WRAPPED_CHANNEL_BUF(proxy_state, ch, counter);
            if (*((volatile uint8_t *)&channel_req[i]->flag) != flag) {
                channel_req[i] =
                    NULL;  // XXX:this store should prevent the next load to be reordered, so fence
                           // should not be needed; but fence below unhangs barrier test
            } else {
                TRACE(NVSHMEM_PROXY, "[%d] progress_channels found new channeL_req %p counter %ld",
                      proxy_state->nvshmemi_state->mype, channel_req[i], counter);
            }
        }

#if defined(NVSHMEM_PPC64LE)
        __sync_synchronize();  // XXX: this makes a difference for barrier but not for get_nbi; this
                               // is Load/Load ordering point, fence could be needed for x86_64 (if
                               // data dependency is not enough)
#endif
        // NOTE: all process function except process_channel_dma either processes
        // the complete request of does not process it at all
        if (channel_req[i]) {
            TRACE(NVSHMEM_PROXY, "[%d] progress_channels new request channel_req %p counter %ld",
                  proxy_state->nvshmemi_state->mype, channel_req[i], counter);
            int is_processed = 1;
            switch (channel_req[i]->op) {
                case NVSHMEMI_OP_PUT:
                    TRACE(NVSHMEM_PROXY, "host proxy: received PUT \n");
                    is_processed = 0;
                    status = process_channel_dma(proxy_state, ch, &is_processed);
                    NZ_EXIT(status, "error in process_channel_dma<PUT>\n");
                    break;
                case NVSHMEMI_OP_G:
                case NVSHMEMI_OP_GET:
                    TRACE(NVSHMEM_PROXY, "host proxy: received GET \n");
                    is_processed = 0;
                    status = process_channel_dma(proxy_state, ch, &is_processed);
                    if (likely(is_processed)) proxy_state->issued_get = 1;
                    NZ_EXIT(status, "error in process_channel_dma<GET>\n");
                    break;
                case NVSHMEMI_OP_P:
                    TRACE(NVSHMEM_PROXY, "host proxy: received P_CHAR \n");
                    is_processed = 0;
                    status = process_channel_inline(proxy_state, ch, &is_processed);
                    NZ_EXIT(status, "error in process_channel_inline<char>\n");
                    break;
                case NVSHMEMI_OP_AMO:
                    is_processed = 0;
                    status = process_channel_amo(proxy_state, ch, &is_processed);
                    NZ_EXIT(status, "error in process_channel_inline<char>\n");
                    break;
                case NVSHMEMI_OP_FENCE:
                    TRACE(NVSHMEM_PROXY, "host proxy: received FENCE \n");
                    status = process_channel_fence(proxy_state, ch);
                    NZ_EXIT(status, "error in process_channel_fence\n");
                    break;
                default:
                    fprintf(stderr, "invalid op type encountered in proxy \n");
                    exit(-1);
            }

            if (likely(is_processed)) {
                channel_req[i] = NULL;
            } else {
                // request is only partially processed, use the same request in the next
                // iteration
            }
        } /*if(channel_req[i])*/
    }
}

void progress_transports(proxy_state_t *proxy_state) {
    int status = 0;
    nvshmemi_state_t *state = proxy_state->nvshmemi_state;

    for (int i = 0; i < NVSHMEM_TRANSPORT_COUNT; i++) {
        if (!((proxy_state->transport_bitmap) & (1 << i))) continue;

        struct nvshmem_transport *tcurr = state->transports[i];

        if (tcurr->host_ops.progress == NULL) continue;

        status = tcurr->host_ops.progress(tcurr);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "transport %d progress failed \n", i);
    }
out:
    NZ_EXIT(status, "error in progress_transport \n");
}

// this has to be call before channels are torn down
void force_flush(proxy_state_t *proxy_state) {}

inline void progress(proxy_state_t *proxy_state) {
    // progress quiet ops
    progress_quiet(proxy_state);

    // progress cst ops
    progress_cst(proxy_state);

#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
    nvshmemi_timeout_t *timeout = proxy_state->nvshmemi_timeout;
    if (timeout->signal) {
        const char *str = "";
        switch (timeout->caller) {
            case NVSHMEMI_CALL_SITE_BARRIER:
                str = "nvshmem_barrier";
                break;
            case NVSHMEMI_CALL_SITE_BARRIER_WARP:
                str = "nvshmem_barrier_warp";
                break;
            case NVSHMEMI_CALL_SITE_BARRIER_THREADBLOCK:
                str = "nvshmem_barrier_block";
                break;
            case NVSHMEMI_CALL_SITE_WAIT_UNTIL_GE:
                str = "nvshmem_wait_until_ge";
                break;
            case NVSHMEMI_CALL_SITE_WAIT_UNTIL_EQ:
                str = "nvshmem_wait_until_eq";
                break;
            case NVSHMEMI_CALL_SITE_WAIT_UNTIL_NE:
                str = "nvshmem_wait_until_ne";
                break;
            case NVSHMEMI_CALL_SITE_WAIT_UNTIL_GT:
                str = "nvshmem_wait_until_gt";
                break;
            case NVSHMEMI_CALL_SITE_WAIT_UNTIL_LT:
                str = "nvshmem_wait_until_lt";
                break;
            case NVSHMEMI_CALL_SITE_WAIT_UNTIL_LE:
                str = "nvshmem_wait_until_le";
                break;
            case NVSHMEMI_CALL_SITE_WAIT_NE:
                str = "nvshmem_wait_ne";
                break;
            case NVSHMEMI_CALL_SITE_PROXY_CHECK_CHANNEL_AVAILABILITY:
                str = "check_channel_availability";
                break;
            case NVSHMEMI_CALL_SITE_PROXY_QUIET:
                str = "nvshmemi_proxy_quiet";
                break;
            case NVSHMEMI_CALL_SITE_PROXY_ENFORCE_CONSISTENCY_AT_TARGET:
                str = "nvshmemi_proxy_enforce_consistency_at_target";
                break;
            case NVSHMEMI_CALL_SITE_AMO_FETCH_WAIT_FLAG:
                str = "nvshmemi_call_site_amo_fetch_wait_flag";
                break;
            case NVSHMEMI_CALL_SITE_AMO_FETCH_WAIT_DATA:
                str = "nvshmemi_call_site_amo_fetch_wait_data";
                break;
            default: { str = "unknown call site, exiting"; }
        }
        ERROR_PRINT("received timeout signal from GPU thread(s) in %s\n", str);
        ERROR_PRINT("signal addr %" PRIu64 " signal val found %" PRIu64 " signal val expected %" PRIu64 "\n",
                    timeout->signal_addr, timeout->signal_val_found, timeout->signal_val_expected);
        exit(-1);
    }
#endif

    // progress channels
    progress_channels(proxy_state);

    // progress transports
    progress_transports(proxy_state);
}

void *nvshmemi_proxy_progress(void *in) {
    proxy_progress_params_t *params = (proxy_progress_params_t *)in;
    proxy_state_t *proxy_state = params->state;

    // set context on the current thread
    INFO(NVSHMEM_PROXY, "setting current CUDA context to saved context: %llu",
         proxy_state->nvshmemi_state->cucontext);
    CUresult curesult = CUDA_SUCCESS;
    curesult = cuCtxSetCurrent(proxy_state->nvshmemi_state->cucontext);
    if (curesult != CUDA_SUCCESS) {
        ERROR_EXIT("failed setting context on the proxy thread \n");
    }

    // setup progress channels
    channel_req = (base_request_t **)calloc(proxy_state->channel_count, sizeof(base_request_t *));

    // call progress until stop is signalled
    do {
        progress(proxy_state);
    } while (!*((volatile int *)&params->stop));

    free(channel_req);

    return NULL;
}

int nvshmemi_proxy_finalize(nvshmemi_state_t *state) {
    proxy_state_t *proxy_state = (proxy_state_t *)state->proxy;

    proxy_state->progress_params.stop = 1;

    pthread_join(proxy_state->progress_thread, NULL);

    CUDA_CHECK(cuStreamDestroy(proxy_state->stream));

    // free up all proxy state
    return 0;
}
