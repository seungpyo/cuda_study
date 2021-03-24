/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmem_internal.h"
#include <pthread.h>
#include "common.h"
#include <assert.h>
#include <stdint.h>
#include "transport.h"

#define CHANNEL_COUNT 1

#define CHANNEL_BUF_SIZE_LOG 22
#define CHANNEL_BUF_SIZE (1 << CHANNEL_BUF_SIZE_LOG)
#define COUNTER_TO_FLAG(state, counter) ((uint8_t)(!((counter >> state->channel_bufsize_log) & 1)))
#define WRAPPED_CHANNEL_BUF(state, ch, counter) (ch->buf + (counter & (state->channel_bufsize - 1)))
#define WRAPPED_CHANNEL_BUF_(state, buf, counter) (buf + (counter & (state->channel_bufsize - 1)))

enum {
    PROXY_QUIET_STATUS_CHANNELS_INACTIVE = 0,
    PROXY_QUIET_STATUS_CHANNELS_IN_PROGRESS,
    PROXY_QUIET_STATUS_CHANNELS_DONE
};

enum { PROXY_CST_STATUS_CHANNELS_INACTIVE = 0, PROXY_CST_STATUS_CHANNELS_ACTIVE };

typedef struct {
    uint64_t data[4];
} channel_request_t;

/* base_request_t
 * 32 | 8 | 8 | 8 | 8
 * roffset_high | roffset_low | op | group_size | flag */
typedef struct base_request {
    volatile uint8_t flag;
    uint8_t groupsize;
    uint8_t op;
    uint8_t roffset_low;   // target is remote
    uint32_t roffset_high; /*used as pe for base-only requests*/
} base_request_t;

/* put_dma_request_0
 * 32 | 16 | 8 | 8
 * loffset_high | loffset_low | pe | flag */
typedef struct put_dma_request_0 {
    volatile uint8_t flag;
    uint8_t resv;
    uint16_t loffset_low;  // source is local
    uint32_t loffset_high;
} put_dma_request_0_t;

/* put_dma_request_1
 * 32 | 16 | 8 | 8
 * size_high | size_low | resv | flag */
typedef struct put_dma_request_1 {
    volatile uint8_t flag;
    uint8_t resv;
    uint16_t size_low;
    uint32_t size_high;
} put_dma_request_1_t;

/* put_dma_request_2
 * 32 | 16 | 8 | 8
 * resv2 | pe | resv | flag */
typedef struct put_dma_request_2 {
    volatile uint8_t flag;
    uint8_t resv;
    uint16_t pe;
    uint32_t resv1;
} put_dma_request_2_t;

/* put_inline_request_0
 * 32 | 16 | 8 | 8
 * loffset_high | loffset_low | pe | flag */
typedef struct put_inline_request_0 {
    volatile uint8_t flag;
    uint8_t resv;
    uint16_t pe;  
    uint32_t lvalue_low;
} put_inline_request_0_t;

/* put_inline_request_1
 * 32 | 16 | 8 | 8
 * size_high | size_low | resv | flag */
typedef struct put_inline_request_1 {
    volatile uint8_t flag;
    uint8_t resv;
    uint16_t size;
    uint32_t lvalue_high;
} put_inline_request_1_t;

/* amo_request_0
 * 32 | 16 | 8 | 8
 * lvalue_low | pe | amo | flag */
typedef struct amo_request_0 {
    volatile uint8_t flag;
    uint8_t amo;
    uint16_t pe;  
    uint32_t swap_add_low;
} amo_request_0_t;

/* amo_request_1
 * 32 | 16 | 8 | 8
 * lvalue_high | resv | size | flag */
typedef struct amo_request_1 {
    volatile uint8_t flag;
    uint8_t compare_low;
    uint16_t size;
    uint32_t swap_add_high;
} amo_request_1_t;

/* amo_request_2
 * 56 | 8
 * compare_high | flag */
typedef struct amo_request_2 {
    volatile uint8_t flag;
    uint8_t compare_high[7];
} amo_request_2_t;

typedef struct {
    struct proxy_state *state;
    int stop;
} proxy_progress_params_t;

typedef struct proxy_channel {
    char *buf;
    uint64_t *issue;
    uint64_t *complete;
    uint64_t *quiet_issue;
    uint64_t *quiet_ack;
    uint64_t last_quiet_issue;
    uint64_t *cst_issue;
    uint64_t *cst_ack;
    uint64_t last_cst_issue;
    uint64_t processed;
    uint64_t last_sync;
} proxy_channel_t;

typedef struct proxy_state {
    nvshmemt_ep_t *ep;
    int *transport_id;
    int transport_bitmap;
    struct nvshmem_transport **transport;
    int quiet_in_progress;
    int cst_in_progress;
    int quiet_ack_count;
    uint64_t channel_bufsize_log;
    uint64_t channel_bufsize;
    int channel_count;
    proxy_channel_t *channels;
    proxy_channel_t *channels_device;
    uint64_t channel_g_bufsize;
    int channel_in_progress;
    pthread_t progress_thread;
    proxy_progress_params_t progress_params;
    nvshmemi_state_t *nvshmemi_state;
    int *quiet_incoming_in_progress_pe;
    CUstream stream;
    CUstream queue_stream_out;
    CUstream queue_stream_in;
    CUevent cuev;
    int finalize_count;
    int issued_get;
    nvshmemi_timeout_t* nvshmemi_timeout;
} proxy_state_t;

int nvshmemi_proxy_setup_device_channels(proxy_state_t *state);

extern __device__ char *proxy_channel_g_buf_d;           /* buffer space for shmem_g requests */
extern __device__ uint64_t proxy_channel_g_buf_head_d;     /* next location to be assigned to a thread */
extern __constant__ uint64_t proxy_channel_g_buf_size_d;   /* Total size of g_buf in bytes */
extern __constant__ uint64_t proxy_channel_g_buf_log_size_d;   /* Total size of g_buf in bytes */

extern char *proxy_channel_g_buf;
extern uint64_t proxy_channel_g_buf_head;     /* next location to be assigned to a thread */
extern uint64_t proxy_channel_g_buf_size;   /* Total size of g_buf in bytes */
extern uint64_t proxy_channel_g_buf_log_size;   /* Total size of g_buf in bytes */

