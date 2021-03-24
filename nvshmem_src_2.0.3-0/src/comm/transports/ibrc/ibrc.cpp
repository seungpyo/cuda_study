/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmem_internal.h"

#include <string.h>
#include <assert.h>
#include <map>
#include <vector>
#include <deque>
#include <dlfcn.h>
#include "ibrc.h"
#include "nvshmemx_error.h"
#include <asm/types.h>
#ifdef NVSHMEM_X86_64
#include <immintrin.h>
#endif

#ifdef NVSHMEM_USE_GDRCOPY
#include "gdrapi.h"
#endif

#define IBRC_MAX_INLINE_SIZE 128

int ibrc_srq_depth;
#define IBRC_SRQ_MASK (ibrc_srq_depth - 1)

int ibrc_qp_depth;
#define IBRC_REQUEST_QUEUE_MASK (ibrc_qp_depth - 1)
#define IBRC_BUF_SIZE 64

#if defined(NVSHMEM_X86_64)
#define IBRC_CACHELINE 64
#elif defined(NVSHMEM_PPC64LE)
#define IBRC_CACHELINE 128
#else
#error Unknown cache line size
#endif

#define MAX_NUM_HCAS 16
#define MAX_NUM_PORTS 4
#define MAX_NUM_PES_PER_NODE 32
#ifdef NVSHMEM_USE_GDRCOPY
#define BAR_READ_BUFSIZE (2 * 1024 * 1024)
#else
#define BAR_READ_BUFSIZE (sizeof(uint64_t))
#endif

enum { WAIT_ANY = 0, WAIT_ALL = 1 };

int MAX_RD_ATOMIC; /* Maximum number of RDMA Read & Atomic operations that can be outstanding per QP
                    */

typedef struct {
    void *devices;
    int *dev_ids;
    int *port_ids;
    int n_dev_ids;
} transport_ibrc_state_t;

struct ibrc_request {
    struct ibv_send_wr sr;
    struct ibv_send_wr *bad_sr;
    struct ibv_sge sge;
};

struct ibrc_atomic_op {
    nvshmemi_amo_t op;
    void *addr;
    void *retaddr;
    uint32_t retrkey;
    uint64_t retflag;
    uint32_t elembytes;
    uint64_t compare;
    uint64_t swap_add;
};

typedef struct ibrc_buf {
    struct ibv_recv_wr rwr;
    struct ibv_recv_wr *bad_rwr;
    struct ibv_sge sge;
    int qp_num;
    char buf[IBRC_BUF_SIZE];
} ibrc_buf_t;
ibrc_buf_t *bpool;
int bpool_size;
static std::vector<void *> bpool_free;
static std::deque<void *> bqueue_toprocess;

struct ibrc_device {
    struct ibv_device *dev;
    struct ibv_context *context;
    struct ibv_pd *pd;
    struct ibv_device_attr device_attr;
    struct ibv_port_attr port_attr[MAX_NUM_PORTS];
    union ibv_gid gid[MAX_NUM_PORTS];
    // bpool information
    struct ibv_srq *srq;
    int srq_posted;
    struct ibv_mr *bpool_mr;
    struct ibv_cq *recv_cq;
    struct ibv_cq *send_cq;
};

struct ibrc_ep {
    int devid;
    int portid;
    struct ibv_qp *qp;
    struct ibv_cq *send_cq;
    struct ibv_cq *recv_cq;
    struct ibrc_request *req;
    volatile uint64_t head_op_id;
    volatile uint64_t tail_op_id;
    void *ibrc_state;
};

struct ibrc_ep_handle {
    uint32_t qpn;
    uint16_t lid;
    //ROCE
    uint64_t spn;
    uint64_t iid;
};

struct ibrc_mem_handle {
    uint32_t lkey;
    uint32_t rkey;
    void *mr;
};

typedef struct ibrc_mem_handle_info {
    struct ibv_mr *mr;
    void *ptr;
    size_t size;
#ifdef NVSHMEM_USE_GDRCOPY
    void *cpu_ptr;
    void *cpu_ptr_base;
    gdr_mh_t mh;
#endif
} ibrc_mem_handle_info_t;
ibrc_mem_handle_info_t *dummy_local_mem;
pthread_mutex_t ibrc_mutex_recv_progress;
pthread_mutex_t ibrc_mutex_send_progress;

static std::vector<ibrc_mem_handle_info_t> mem_handle_cache;
static std::map<unsigned int, long unsigned int> qp_map;
static uint64_t connected_qp_count;

struct ibrc_ep *ibrc_cst_ep;
static int use_ib_native_atomics = 1;
static int use_gdrcopy = 0;
#ifdef NVSHMEM_USE_GDRCOPY
static gdr_t gdr_desc;
struct gdrcopy_function_table {
    gdr_t (*open)();
    int (*close)(gdr_t g);
    int (*pin_buffer)(gdr_t g, unsigned long addr, size_t size, uint64_t p2p_token,
                      uint32_t va_space, gdr_mh_t *handle);
    int (*unpin_buffer)(gdr_t g, gdr_mh_t handle);
    int (*get_info)(gdr_t g, gdr_mh_t handle, gdr_info_t *info);
    int (*map)(gdr_t g, gdr_mh_t handle, void **va, size_t size);
    int (*unmap)(gdr_t g, gdr_mh_t handle, void *va, size_t size);
    int (*copy_from_mapping)(gdr_mh_t handle, void *h_ptr, const void *map_d_ptr, size_t size);
    int (*copy_to_mapping)(gdr_mh_t handle, const void *map_d_ptr, void *h_ptr, size_t size);
    void (*runtime_get_version)(int *major, int *minor);
    int (*driver_get_version)(gdr_t g, int *major, int *minor);
};
static struct gdrcopy_function_table gdrcopy_ftable;
static void *gdrcopy_handle = NULL;
static volatile uint64_t atomics_received = 0;
static volatile uint64_t atomics_processed = 0;
static volatile uint64_t atomics_issued = 0;
static volatile uint64_t atomics_completed = 0;
static volatile uint64_t atomics_acked = 0;

struct gdrmap_heap_info {
    void *gpu_ptr;
    gdr_mh_t mh;
    void *mapped_ptr;
    int size;
};
static struct gdrmap_heap_info gdrmap_heap;
#endif

static struct ibrc_function_table ftable;
static void *ibv_handle;

struct ibrc_hca_info {
    char name[64];
    int port;
    int count;
    int found;
};

int nvshmemt_ibrc_init(nvshmem_transport_t *transport);
int check_poll_avail(struct ibrc_ep *ep, int wait_predicate);
int progress_send(transport_ibrc_state_t *ibrc_state);
int poll_recv(transport_ibrc_state_t *ibrc_state);

ibrc_mem_handle_info_t get_mem_handle_info(void *gpu_ptr) {
    assert(!mem_handle_cache.empty());

    // assuming there is only one region (shmem heap) that is registered with IB
    ibrc_mem_handle_info_t mem_handle_info = mem_handle_cache.back();

    return mem_handle_info;
}

inline int refill_srq(struct ibrc_device *device) {
    int status = 0;

    while ((device->srq_posted < ibrc_srq_depth) && !bpool_free.empty()) {
        ibrc_buf_t *buf = (ibrc_buf_t *)bpool_free.back();

        buf->rwr.next = NULL;
        buf->rwr.wr_id = (uint64_t)buf;
        buf->rwr.sg_list = &(buf->sge);
        buf->rwr.num_sge = 1;

        buf->sge.addr = (uint64_t)buf->buf;
        buf->sge.length = IBRC_BUF_SIZE;
        buf->sge.lkey = device->bpool_mr->lkey;

        status = ibv_post_srq_recv(device->srq, &buf->rwr, &buf->bad_rwr);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_post_srq_recv failed \n");

        bpool_free.pop_back();
        device->srq_posted++;
    }

out:
    return status;
}

int parse_hca_list(const char *string, struct ibrc_hca_info *hca_list, int max_count) {
    if (!string) return 0;

    const char *ptr = string;
    // Ignore "^" name, will be detected outside of this function
    if (ptr[0] == '^') ptr++;

    int if_num = 0;
    int if_counter = 0;
    int segment_counter = 0;
    char c;
    do {
        c = *ptr;
        if (c == ':') {
            if (segment_counter == 0) {
                if (if_counter > 0) {
                    hca_list[if_num].name[if_counter] = '\0';
                    hca_list[if_num].port = atoi(ptr + 1);
                    hca_list[if_num].found = 0;
                    if_num++;
                    if_counter = 0;
                    segment_counter++;
                }
            } else {
                hca_list[if_num - 1].count = atoi(ptr + 1);
                segment_counter = 0;
            }
            c = *(ptr + 1);
            while (c != ',' && c != ':' && c != '\0') {
                ptr++;
                c = *(ptr + 1);
            }
        } else if (c == ',' || c == '\0') {
            if (if_counter > 0) {
                hca_list[if_num].name[if_counter] = '\0';
                hca_list[if_num].found = 0;
                if_num++;
                if_counter = 0;
            }
            segment_counter = 0;
        } else {
            if (if_counter == 0) {
                hca_list[if_num].port = -1;
                hca_list[if_num].count = 1;
            }
            hca_list[if_num].name[if_counter] = c;
            if_counter++;
        }
        ptr++;
    } while (if_num < max_count && c);

    INFO(NVSHMEM_INIT, "Begin - Parsed HCA list provided by user - ");
    for (int i = 0; i < if_num; i++) {
        INFO(NVSHMEM_INIT,
             "Parsed HCA list provided by user - i=%d (of %d), name=%s, port=%d, count=%d", i,
             if_num, hca_list[i].name, hca_list[i].port, hca_list[i].count);
    }
    INFO(NVSHMEM_INIT, "End - Parsed HCA list provided by user");

    return if_num;
}

int nvshmemt_ibrc_show_info(nvshmem_mem_handle_t *mem_handles, int transport_id,
                            int transport_count, nvshmemt_ep_t *eps, int ep_count, int npes,
                            int mype) {
    for (int i = 0; i < npes; ++i) {
        INFO(NVSHMEM_TRANSPORT, "[%d] mem_handle %d : %p", mype, transport_id,
             &mem_handles[i * transport_count + transport_id]);
        struct ibrc_mem_handle *mem_handle =
            (struct ibrc_mem_handle *)&mem_handles[i * transport_count + transport_id];
        (struct ibrc_mem_handle *)&mem_handles[i * transport_count + transport_id];
        INFO(NVSHMEM_TRANSPORT, "[%d] lkey %x rkey %x mr %p", mype, mem_handle->lkey,
             mem_handle->rkey, mem_handle->mr);
        if (i != mype) {
            for (int j = 0; j < ep_count; ++j) {
                /*XXX : not implemented*/
            }
        }
    }
    return 0;
}

int nvshmemt_ibrc_get_device_count(int *ndev, nvshmem_transport_t t) {
    int status = 0;

    struct nvshmem_transport *transport = (struct nvshmem_transport *)t;
    transport_ibrc_state_t *ibrc_state = (transport_ibrc_state_t *)transport->state;

    *ndev = ibrc_state->n_dev_ids;

out:
    return status;
}

static int ib_iface_get_mlx_path(const char *ib_name, char **path) {
    int status = NVSHMEMX_SUCCESS;

    char device_path[MAXPATHSIZE];
    snprintf(device_path, MAXPATHSIZE, "/sys/class/infiniband/%s/device", ib_name);
    *path = realpath(device_path, NULL);
    NULL_ERROR_JMP(*path, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "realpath failed \n");

out:
    return status;
}

int nvshmemt_ibrc_get_pci_path(int dev, char **pci_path, nvshmem_transport_t t) {
    int status = NVSHMEMX_SUCCESS;

    struct nvshmem_transport *transport = (struct nvshmem_transport *)t;
    transport_ibrc_state_t *ibrc_state = (transport_ibrc_state_t *)transport->state;
    int dev_id = ibrc_state->dev_ids[dev];
    const char *ib_name =
        (const char *)((struct ibrc_device *)ibrc_state->devices)[dev_id].dev->name;

    status = ib_iface_get_mlx_path(ib_name, pci_path);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ib_iface_get_mlx_path failed \n");

out:
    return status;
}

int nvshmemt_ibrc_can_reach_peer(int *access, struct nvshmem_transport_pe_info *peer_info,
                                 nvshmem_transport_t t) {
    int status = 0;

    *access = NVSHMEM_TRANSPORT_CAP_CPU_WRITE | NVSHMEM_TRANSPORT_CAP_CPU_READ |
              NVSHMEM_TRANSPORT_CAP_CPU_ATOMICS;

out:
    return status;
}

static int ep_create(struct ibrc_ep **ep_ptr, int devid, transport_ibrc_state_t *ibrc_state) {
    int status = 0;
    struct ibrc_ep *ep;
    struct ibv_qp_init_attr init_attr;
    struct ibv_qp_attr attr;
    int flags;
    struct ibrc_device *device =
        ((struct ibrc_device *)ibrc_state->devices + ibrc_state->dev_ids[devid]);
    int portid = ibrc_state->port_ids[devid];
    struct ibv_port_attr port_attr = device->port_attr[ibrc_state->port_ids[devid] - 1];
    struct ibv_context *context = device->context;
    struct ibv_pd *pd = device->pd;

    // algining ep structure to prevent split tranactions when accessing head_op_id and
    // tail_op_id which can be used in inter-thread synchronization
    // TODO: use atomic variables instead to rely on language memory model guarantees
    status = posix_memalign((void **)&ep, IBRC_CACHELINE, sizeof(struct ibrc_ep));
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "ep allocation failed \n");
    memset((void *)ep, 0, sizeof(struct ibrc_ep));

    if (!device->send_cq) {
        device->send_cq = ftable.create_cq(context, device->device_attr.max_cqe, NULL, NULL, 0);
        NULL_ERROR_JMP(device->send_cq, status, NVSHMEMX_ERROR_INTERNAL, out,
                       "cq creation failed \n");
    }
    assert(device->send_cq != NULL);
    ep->send_cq = device->send_cq;

    if (!device->srq) {
        struct ibv_srq_init_attr srq_init_attr;
        memset(&srq_init_attr, 0, sizeof(srq_init_attr));

        srq_init_attr.attr.max_wr = ibrc_srq_depth;
        srq_init_attr.attr.max_sge = 1;

        device->srq = ftable.create_srq(pd, &srq_init_attr);
        NULL_ERROR_JMP(device->srq, status, NVSHMEMX_ERROR_INTERNAL, out, "srq creation failed \n");

        device->recv_cq = ftable.create_cq(context, ibrc_srq_depth, NULL, NULL, 0);
        NULL_ERROR_JMP(device->recv_cq, status, NVSHMEMX_ERROR_INTERNAL, out,
                       "cq creation failed \n");
    }
    assert(device->recv_cq != NULL);
    ep->recv_cq = device->recv_cq;

    memset(&init_attr, 0, sizeof(struct ibv_qp_init_attr));
    init_attr.srq = device->srq;
    init_attr.send_cq = ep->send_cq;
    init_attr.recv_cq = ep->recv_cq;
    init_attr.qp_type = IBV_QPT_RC;
    init_attr.cap.max_send_wr = ibrc_qp_depth;
    init_attr.cap.max_recv_wr = 0;
    init_attr.cap.max_send_sge = 1;
    init_attr.cap.max_recv_sge = 0;
    init_attr.cap.max_inline_data = IBRC_MAX_INLINE_SIZE;

    ep->qp = ftable.create_qp(pd, &init_attr);
    NULL_ERROR_JMP(ep->qp, status, NVSHMEMX_ERROR_INTERNAL, out, "qp creation failed \n");

    memset(&attr, 0, sizeof(struct ibv_qp_attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = portid;
    attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_ATOMIC;
    flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;

    status = ftable.modify_qp(ep->qp, &attr, flags);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_modify_qp failed \n");

    ep->req = (struct ibrc_request *)malloc(sizeof(struct ibrc_request) * ibrc_qp_depth);
    NULL_ERROR_JMP(ep->req, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "req allocation failed \n");
    ep->head_op_id = 0;
    ep->tail_op_id = 0;
    ep->ibrc_state = (void *)ibrc_state;
    ep->devid = ibrc_state->dev_ids[devid];
    ep->portid = portid;

    // insert qp into map
    qp_map.insert(std::make_pair((unsigned int)ep->qp->qp_num, (long unsigned int)ep));

    *ep_ptr = ep;

out:
    return status;
}

static int ep_connect(struct ibrc_ep *ep, struct ibrc_ep_handle *ep_handle) {
    int status = 0;
    struct ibv_qp_attr attr;
    int flags;
    int devid = ep->devid;
    int portid = ep->portid;
    transport_ibrc_state_t *ibrc_state = (transport_ibrc_state_t *)ep->ibrc_state;
    struct ibrc_device *device = ((struct ibrc_device *)ibrc_state->devices + devid);
    struct ibv_port_attr *port_attr = device->port_attr + (portid - 1);

    memset(&attr, 0, sizeof(struct ibv_qp_attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = port_attr->active_mtu;
    attr.dest_qp_num = ep_handle->qpn;
    attr.rq_psn = 0;
    if (port_attr->lid == 0) {
        attr.ah_attr.is_global = 1;
        attr.ah_attr.grh.dgid.global.subnet_prefix = ep_handle->spn;
        attr.ah_attr.grh.dgid.global.interface_id = ep_handle->iid;
        attr.ah_attr.grh.flow_label = 0;
        attr.ah_attr.grh.sgid_index = nvshmemi_options.IB_GID_INDEX;
        attr.ah_attr.grh.hop_limit = 255;
        attr.ah_attr.grh.traffic_class = nvshmemi_options.IB_TRAFFIC_CLASS;
    } else { 
       attr.ah_attr.dlid = ep_handle->lid;
       attr.ah_attr.is_global = 0;
    }
    attr.max_dest_rd_atomic = MAX_RD_ATOMIC;
    attr.min_rnr_timer = 12;
    attr.ah_attr.sl = nvshmemi_options.IB_SL;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = portid;
    flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
            IBV_QP_MIN_RNR_TIMER | IBV_QP_MAX_DEST_RD_ATOMIC;

    status = ftable.modify_qp(ep->qp, &attr, flags);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_modify_qp failed \n");

    memset(&attr, 0, sizeof(struct ibv_qp_attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = 0;
    attr.timeout = 20;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.max_rd_atomic = MAX_RD_ATOMIC;
    flags = IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
            IBV_QP_MAX_QP_RD_ATOMIC;

    status = ftable.modify_qp(ep->qp, &attr, flags);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_modify_qp failed \n");

    // register and post receive buffer pool
    if (!device->bpool_mr) {
        device->bpool_mr = ftable.reg_mr(
            device->pd, bpool, bpool_size * sizeof(ibrc_buf_t),
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        NULL_ERROR_JMP(device->bpool_mr, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                       "mem registration failed \n");

        assert(device->srq != NULL);

        status = refill_srq(device);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "refill_srq failed \n");
    }

    connected_qp_count++;
out:
    return status;
}

int ep_get_handle(struct ibrc_ep_handle *ep_handle, struct ibrc_ep *ep) {
    int status = 0;
    transport_ibrc_state_t *ibrc_state = (transport_ibrc_state_t *)ep->ibrc_state;
    struct ibrc_device *device = ((struct ibrc_device *)ibrc_state->devices + ep->devid);

    assert(sizeof(struct ibrc_ep_handle) <= NVSHMEM_EP_HANDLE_SIZE);

    ep_handle->lid = device->port_attr[ep->portid - 1].lid;
    ep_handle->qpn = ep->qp->qp_num;
    if (ep_handle->lid == 0) { 
        ep_handle->spn = device->gid[ep->portid - 1].global.subnet_prefix;
        ep_handle->iid = device->gid[ep->portid - 1].global.interface_id;
    }

out:
    return status;
}

int setup_cst_loopback(transport_ibrc_state_t *ibrc_state, int dev_id) {
    int status = 0;
    struct ibrc_device *device =
        ((struct ibrc_device *)ibrc_state->devices + ibrc_state->dev_ids[dev_id]);
    struct ibrc_ep_handle cst_ep_handle;

    status = ep_create(&ibrc_cst_ep, dev_id, ibrc_state);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ep_create cst failed \n");

    status = ep_get_handle(&cst_ep_handle, ibrc_cst_ep);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ep_get_handle failed \n");

    status = ep_connect(ibrc_cst_ep, &cst_ep_handle);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ep_connect failed \n");
out:
    return status;
}

int nvshmemt_ibrc_get_mem_handle(nvshmem_mem_handle_t *mem_handle, void *buf, size_t length,
                                 int dev_id, nvshmem_transport_t t) {
    int status = 0;
    struct nvshmem_transport *transport = (struct nvshmem_transport *)t;
    transport_ibrc_state_t *ibrc_state = (transport_ibrc_state_t *)transport->state;
    struct ibrc_device *device =
        ((struct ibrc_device *)ibrc_state->devices + ibrc_state->dev_ids[dev_id]);
    struct ibrc_mem_handle_info handle_info;
    struct ibrc_mem_handle *handle = (struct ibrc_mem_handle *)mem_handle;

    struct ibv_mr *mr = NULL;

    assert(sizeof(struct ibrc_mem_handle) <= NVSHMEM_MEM_HANDLE_SIZE);

    mr = ftable.reg_mr(device->pd, buf, length,
                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                           IBV_ACCESS_REMOTE_ATOMIC);
    NULL_ERROR_JMP(mr, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "mem registration failed \n");

    handle->lkey = mr->lkey;
    handle->rkey = mr->rkey;
    handle->mr = (void *)mr;
    handle_info.mr = mr;
    handle_info.ptr = buf;
    handle_info.size = length;
    INFO(NVSHMEM_TRANSPORT, "ibv_reg_mr handle %p handle->mr %x", handle, handle->mr);

#ifdef NVSHMEM_USE_GDRCOPY
    if (use_gdrcopy) {
        status =
            gdrcopy_ftable.pin_buffer(gdr_desc, (unsigned long)buf, length, 0, 0, &handle_info.mh);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gdrcopy pin_buffer failed \n");

        status = gdrcopy_ftable.map(gdr_desc, handle_info.mh, &handle_info.cpu_ptr_base, length);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gdrcopy map failed \n");

        gdr_info_t info;
        status = gdrcopy_ftable.get_info(gdr_desc, handle_info.mh, &info);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gdrcopy get_info failed \n");

        // remember that mappings start on a 64KB boundary, so let's
        // calculate the offset from the head of the mapping to the
        // beginning of the buffer
        uintptr_t off;
        off = (uintptr_t)buf - info.va;
        handle_info.cpu_ptr = (void *)((uintptr_t)handle_info.cpu_ptr_base + off);
    }
#endif

    mem_handle_cache.push_back(handle_info);

    if (!dummy_local_mem) {
        dummy_local_mem = (ibrc_mem_handle_info_t *)malloc(sizeof(ibrc_mem_handle_info_t));
        NULL_ERROR_JMP(dummy_local_mem, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                       "dummy_local_mem allocation failed\n");

        dummy_local_mem->ptr = malloc(sizeof(uint64_t));
        NULL_ERROR_JMP(dummy_local_mem->ptr, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                       "dummy_mem allocation failed\n");

        dummy_local_mem->mr = ftable.reg_mr(device->pd, dummy_local_mem->ptr, sizeof(uint64_t),
                                            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                                IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
        NULL_ERROR_JMP(dummy_local_mem->mr, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                       "mem registration failed \n");
    }
out:
    return status;
}

int nvshmemt_ibrc_release_mem_handle(nvshmem_mem_handle_t mem_handle) {
    int status = 0;
    struct ibrc_mem_handle *handle = (struct ibrc_mem_handle *)&mem_handle;

    INFO(NVSHMEM_TRANSPORT, "ibv_dereg_mr handle %p handle->mr %x", handle, handle->mr);
    status = ftable.dereg_mr((struct ibv_mr *)handle->mr);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_dereg_mr failed \n");

out:
    return status;
}

int nvshmemt_ibrc_finalize(nvshmem_transport_t transport) {
    int status = 0;
    struct nvshmem_transport *t = (struct nvshmem_transport *)transport;
    transport_ibrc_state_t *ibrc_state = (transport_ibrc_state_t *)t->state;

    while (!mem_handle_cache.empty()) {
        ibrc_mem_handle_info_t handle_info = mem_handle_cache.back();

#ifdef NVSHMEM_USE_GDRCOPY
        if (use_gdrcopy) {
            status = gdrcopy_ftable.unmap(gdr_desc, handle_info.mh, handle_info.cpu_ptr_base,
                                          handle_info.size);
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gdr_unmap failed\n");

            status = gdrcopy_ftable.unpin_buffer(gdr_desc, handle_info.mh);
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gdr_unpin failed\n");

            status = gdrcopy_ftable.close(gdr_desc);
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gdr_close failed\n");
        }
#endif
        mem_handle_cache.pop_back();
    }

    // clear qp map
    qp_map.clear();

    if (dummy_local_mem) {
        status = ftable.dereg_mr(dummy_local_mem->mr);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_dereg_mr failed \n");
        free(dummy_local_mem);
    }

#ifdef NVSHMEM_USE_GDRCOPY
    if (use_gdrcopy && gdrcopy_handle) {
        status = dlclose(gdrcopy_handle);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "dlclose() failed\n");
    }
#endif

    if (bpool != NULL) {
        while (!bpool_free.empty()) bpool_free.pop_back();

        free(bpool);
    }

    status = dlclose(ibv_handle);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "dlclose() failed\n");

    status = pthread_mutex_destroy(&ibrc_mutex_send_progress);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "pthread_mutex_destroy failed\n");

    status = pthread_mutex_destroy(&ibrc_mutex_recv_progress);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "pthread_mutex_destroy failed\n");

out:
    return status;
}

#ifdef NVSHMEM_USE_GDRCOPY
template <typename T>
int perform_gdrcopy_amo(struct ibrc_ep *ep, gdr_mh_t mh, struct ibrc_atomic_op *op, void *ptr) {
    int status = 0;

    T old_value, new_value;
    // FIXME: gdrcopy causing duplicate copies for small transfers, using direct LD/ST until this resolved
    //status = gdrcopy_ftable.copy_from_mapping(mh, &old_value, ptr, sizeof(T));
    //NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gdr copy from mapping failed\n");
#if __cplusplus >= 201103L
    // assert size is 64-bit or smaller, issued as single tansaction
    static_assert(sizeof(T) <= 8, "static_assert(sizeof(T) >= 8) failed");
#endif
    old_value = *((volatile T *)ptr);

    switch (op->op) {
        case NVSHMEMI_AMO_SIGNAL:
        case NVSHMEMI_AMO_SIGNAL_SET:
        case NVSHMEMI_AMO_SET:
        case NVSHMEMI_AMO_SWAP: {
            new_value = *((T *)&op->swap_add);
            break;
        }
        case NVSHMEMI_AMO_ADD:
        case NVSHMEMI_AMO_SIGNAL_ADD:
        case NVSHMEMI_AMO_FETCH_ADD: {
            new_value = old_value + (*((T *)&op->swap_add));
            break;
        }
        case NVSHMEMI_AMO_OR:
        case NVSHMEMI_AMO_FETCH_OR: {
            new_value = old_value | (*((T *)&op->swap_add));
            break;
        }
        case NVSHMEMI_AMO_AND:
        case NVSHMEMI_AMO_FETCH_AND: {
            new_value = old_value & (*((T *)&op->swap_add));
            break;
        }
        case NVSHMEMI_AMO_XOR:
        case NVSHMEMI_AMO_FETCH_XOR: {
            new_value = old_value ^ (*((T *)&op->swap_add));
            break;
        }
        case NVSHMEMI_AMO_COMPARE_SWAP: {
            new_value = (old_value == *((T *)&op->compare)) ? *((T *)&op->swap_add) : old_value;
            break;
        }
        default: {
            ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "RMA/AMO verb %d not implemented\n",
                      op->op);
        }
    }

    // FIXME: gdrcopy causing duplicate copies for small transfers, using direct LD/ST until this resolved
    // status = gdrcopy_ftable.copy_to_mapping(mh, ptr, (void *)&new_value, sizeof(T));
    // NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gdr copy to mapping failed\n");
    *((volatile T *)ptr) = new_value;
#if defined(NVSHMEM_X86_64)
    _mm_sfence();
#elif defined(NVSHMEM_PPC64LE)
    asm volatile("sync");
#endif

    {
        transport_ibrc_state_t *ibrc_state = (transport_ibrc_state_t *)ep->ibrc_state;
        struct ibrc_device *device = ((struct ibrc_device *)ibrc_state->devices + ep->devid);
        struct ibv_send_wr *sr, **bad_sr;
        struct ibv_sge *sge;
        struct ibrc_request *req;
        int op_id;
        nvshmemi_amo_t ack;
        g_elem_t ret;

        // wait for one send request to become avaialble on the ep
        int outstanding_count = (ibrc_qp_depth - 1);
        while ((ep->head_op_id - ep->tail_op_id) > outstanding_count) {
            status = progress_send(ibrc_state);
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                         "progress_send failed, outstanding_count: %d\n", outstanding_count);

            // already in processing a recv request
            // only poll recv cq
            status = poll_recv(ibrc_state);
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                         "poll_recv failed, outstanding_count: %d\n", outstanding_count);
        }

        op_id = ep->head_op_id & IBRC_REQUEST_QUEUE_MASK;  // ep->head_op_id % ibrc_qp_depth
        ep->head_op_id++;

        sr = &(ep->req + op_id)->sr;
        bad_sr = &(ep->req + op_id)->bad_sr;
        sge = &(ep->req + op_id)->sge;

        memset(sr, 0, sizeof(ibv_send_wr));
        if (op->op > NVSHMEMI_AMO_END_OF_NONFETCH) {
            ret.data = ret.flag = 0;
            *((T *)&ret.data) = old_value;
            ret.flag = op->retflag;

            sr->next = NULL;
            sr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
            sr->send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
            sr->wr_id = NVSHMEMI_AMO_END_OF_NONFETCH;
            sr->num_sge = 1;
            sr->sg_list = sge;

            sr->imm_data = (uint32_t)NVSHMEMI_AMO_ACK;
            sr->wr.rdma.remote_addr = (uint64_t)op->retaddr;
            sr->wr.rdma.rkey = op->retrkey;
            sge->length = sizeof(g_elem_t);
            sge->addr = (uintptr_t)&ret;
            sge->lkey = 0;
        } else {
            ack = NVSHMEMI_AMO_ACK;

            sr->next = NULL;
            sr->opcode = IBV_WR_SEND;
            sr->send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
            sr->wr_id = NVSHMEMI_AMO_ACK;
            sr->num_sge = 1;
            sr->sg_list = sge;

            // dummy send
            sge->length = sizeof(nvshmemi_amo_t);
            sge->addr = (uintptr_t)&ack;
            sge->lkey = 0;
        }

        status = ibv_post_send(ep->qp, sr, bad_sr);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_poll_cq failed \n");
    }

out:
    return status;
}

int poll_recv(transport_ibrc_state_t *ibrc_state) {
    int status = 0;
    int n_devs = ibrc_state->n_dev_ids;
    static int atomic_in_progress = 0;

    // poll all CQs available
    for (int i = 0; i < n_devs; i++) {
        struct ibv_wc wc;
        int devid = ibrc_state->dev_ids[i];
        struct ibrc_device *device = ((struct ibrc_device *)ibrc_state->devices + devid);

        if (!device->recv_cq) continue;

        int ne = ibv_poll_cq(device->recv_cq, 1, &wc);
        if (ne < 0) {
            status = ne;
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_poll_cq failed \n");
        } else if (ne) {
            uint64_t idx;
            struct ibrc_atomic_op *op;

            assert(ne == 1);
            ibrc_buf_t *buf = (ibrc_buf_t *)wc.wr_id;
            if (wc.wc_flags & IBV_WC_WITH_IMM) {
                atomics_acked++;
                TRACE(NVSHMEM_TRANSPORT, "[%d] atomic acked : %llu \n", getpid(), atomics_acked);
                bpool_free.push_back((void *)buf);
            } else {
                struct ibrc_atomic_op *op = (struct ibrc_atomic_op *)buf->buf;
                if (op->op == NVSHMEMI_AMO_ACK) {
                    atomics_acked++;
                    TRACE(NVSHMEM_TRANSPORT, "[%d] atomic acked : %llu \n", getpid(),
                          atomics_acked);
                    bpool_free.push_back((void *)buf);
                } else {
                    buf->qp_num = wc.qp_num;
                    atomics_received++;
                    TRACE(NVSHMEM_TRANSPORT, "[%d] atomic received, enqueued : %llu \n", getpid(),
                          atomics_received);
                    bqueue_toprocess.push_back((void *)buf);
                }
            }
            device->srq_posted--;
        }

        status = refill_srq(device);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "refill_sqr failed \n");
    }

out:
    return status;
}

int process_recv(transport_ibrc_state_t *ibrc_state) {
    int status = 0;

    if (!bqueue_toprocess.empty()) {
        ibrc_buf_t *buf = (ibrc_buf_t *)bqueue_toprocess.front();
        struct ibrc_ep *ep = (struct ibrc_ep *)qp_map.find((unsigned int)buf->qp_num)->second;
        struct ibrc_atomic_op *op = (struct ibrc_atomic_op *)buf->buf;
        ibrc_mem_handle_info_t mem_handle_info = get_mem_handle_info((void *)op->addr);
        void *ptr = (void *)((uintptr_t)mem_handle_info.cpu_ptr +
                             ((uintptr_t)op->addr - (uintptr_t)mem_handle_info.ptr));

        switch (op->elembytes) {
            case 2:
                perform_gdrcopy_amo<uint16_t>(ep, mem_handle_info.mh, op, ptr);
                break;
            case 4:
                perform_gdrcopy_amo<uint32_t>(ep, mem_handle_info.mh, op, ptr);
                break;
            case 8:
                perform_gdrcopy_amo<uint64_t>(ep, mem_handle_info.mh, op, ptr);
                break;
            default:
                ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "invalid element size encountered \n");
        }
        atomics_processed++;
        TRACE(NVSHMEM_TRANSPORT, "[%d] atomic dequeued and processed : %llu \n", getpid(),
              atomics_processed);

        bqueue_toprocess.pop_front();
        bpool_free.push_back((void *)buf);
    }

out:
    return status;
}

int progress_recv(transport_ibrc_state_t *ibrc_state) {
    int status = 0;

    pthread_mutex_lock(&ibrc_mutex_recv_progress);

    status = poll_recv(ibrc_state);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "poll recv failed \n");

    status = process_recv(ibrc_state);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "process recv failed \n");

out:
    pthread_mutex_unlock(&ibrc_mutex_recv_progress);
    return status;
}
#endif

int progress_send(transport_ibrc_state_t *ibrc_state) {
    int status = 0;
    struct ibrc_ep *ep;
    int n_devs = ibrc_state->n_dev_ids;

    pthread_mutex_lock(&ibrc_mutex_send_progress);

    for (int i = 0; i < n_devs; i++) {
        struct ibv_wc wc;
        int devid = ibrc_state->dev_ids[i];
        struct ibrc_device *device = ((struct ibrc_device *)ibrc_state->devices + devid);

        if (!device->send_cq) continue;

        int ne = ibv_poll_cq(device->send_cq, 1, &wc);
        if (ne < 0) {
            status = ne;
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_poll_cq failed \n");
        } else if (ne) {
            if (wc.status) {
                status = wc.status;
                NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                             "ibv_poll_cq failed, status: %d\n", wc.status);
            }

            assert(ne == 1);
            if (wc.wr_id == NVSHMEMI_OP_AMO) {
#ifdef NVSHMEM_USE_GDRCOPY
                atomics_completed++;
                TRACE(NVSHMEM_TRANSPORT, "[%d] atomic completed : %llu \n", getpid(),
                      atomics_completed);
#endif
            }

            struct ibrc_ep *ep = (struct ibrc_ep *)qp_map.find((unsigned int)wc.qp_num)->second;
            ep->tail_op_id += ne;
        }
    }

out:
    pthread_mutex_unlock(&ibrc_mutex_send_progress);
    return status;
}

int nvshmemt_ibrc_progress(nvshmem_transport_t t) {
    int status = 0;
    struct nvshmem_transport *transport = (struct nvshmem_transport *)t;
    transport_ibrc_state_t *ibrc_state = (transport_ibrc_state_t *)transport->state;

    status = progress_send(ibrc_state);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "progress_send failed, \n");

#ifdef NVSHMEM_USE_GDRCOPY
    status = progress_recv(ibrc_state);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "progress failed \n");
#endif

out:
    return status;
}

int check_poll_avail(struct ibrc_ep *ep, int wait_predicate) {
    int status = 0;
    int outstanding_count = (ibrc_qp_depth - 1);
    if (wait_predicate == WAIT_ALL) outstanding_count = 0;
    transport_ibrc_state_t *ibrc_state = (transport_ibrc_state_t *)ep->ibrc_state;

    /* poll until space becomes in local send qp and space in receive qp at target for atomics
     * assuming connected qp cout is symmetric across all processes,
     * connected_qp_count+1 to avoid completely emptying the recv qp at target, leading to perf
     * issues*/
    while (((ep->head_op_id - ep->tail_op_id) > outstanding_count)
#ifdef NVSHMEM_USE_GDRCOPY
           || ((atomics_issued - atomics_acked) > (ibrc_srq_depth / (connected_qp_count + 1)))
#endif
    ) {
        status = progress_send(ibrc_state);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                     "progress_send failed, outstanding_count: %d\n", outstanding_count);

#ifdef NVSHMEM_USE_GDRCOPY
        status = progress_recv(ibrc_state);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "progress_recv failed \n");
#endif
    }

out:
    return status;
}

int nvshmemt_ibrc_rma(nvshmemt_ep_t tep, rma_verb_t verb, rma_memdesc_t remote, rma_memdesc_t local,
                      rma_bytesdesc_t bytesdesc) {
    int status = 0;
    struct ibrc_ep *ep = (struct ibrc_ep *)tep;
    transport_ibrc_state_t *ibrc_state = (transport_ibrc_state_t *)ep->ibrc_state;
    struct ibrc_device *device = ((struct ibrc_device *)ibrc_state->devices + ep->devid);
    struct ibv_send_wr *sr, **bad_sr;
    struct ibv_sge *sge;
    struct ibrc_request *req;
    int op_id;

    status = check_poll_avail(ep, WAIT_ANY);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "check_poll failed \n");

    op_id = ep->head_op_id & IBRC_REQUEST_QUEUE_MASK;  // ep->head_op_id % ibrc_qp_depth

    sr = &(ep->req + op_id)->sr;
    bad_sr = &(ep->req + op_id)->bad_sr;
    sge = &(ep->req + op_id)->sge;

    memset(sr, 0, sizeof(ibv_send_wr));

    sr->next = NULL;
    sr->send_flags = IBV_SEND_SIGNALED;
    sr->wr_id = NVSHMEMI_OP_PUT;
    sr->num_sge = 1;
    sr->sg_list = sge;

    sr->wr.rdma.remote_addr = (uint64_t)remote.ptr;
    sr->wr.rdma.rkey = ((struct ibrc_mem_handle *)&remote.handle)->rkey;
    sge->length = bytesdesc.nelems * bytesdesc.elembytes;
    sge->addr = (uintptr_t)local.ptr;
    sge->lkey = ((struct ibrc_mem_handle *)&local.handle)->lkey;
    if (verb.desc == NVSHMEMI_OP_P) {
        sr->opcode = IBV_WR_RDMA_WRITE;
        sr->send_flags |= IBV_SEND_INLINE;
        TRACE(NVSHMEM_TRANSPORT, "[PUT] remote_addr %p addr %p rkey %d lkey %d length %lx",
              sr->wr.rdma.remote_addr, sge->addr, sr->wr.rdma.rkey, sge->lkey, sge->length);
    } else if (verb.desc == NVSHMEMI_OP_GET || verb.desc == NVSHMEMI_OP_G) {
        sr->opcode = IBV_WR_RDMA_READ;
        TRACE(NVSHMEM_TRANSPORT, "[GET] remote_addr %p addr %p rkey %d lkey %d length %lx",
              sr->wr.rdma.remote_addr, sge->addr, sr->wr.rdma.rkey, sge->lkey, sge->length);
    } else if (verb.desc == NVSHMEMI_OP_PUT) {
        sr->opcode = IBV_WR_RDMA_WRITE;
        TRACE(NVSHMEM_TRANSPORT, "[PUT] remote_addr %p addr %p rkey %d lkey %d length %lx",
              sr->wr.rdma.remote_addr, sge->addr, sr->wr.rdma.rkey, sge->lkey, sge->length);
    } else {
        ERROR_PRINT("RMA/AMO verb not implemented\n");
        exit(-1);
    }

    TRACE(NVSHMEM_TRANSPORT, "[%d] ibrc post_send dest handle %p rkey %x src handle %p lkey %x",
          getpid(), remote.handle, sr->wr.rdma.rkey, local.handle, sge->lkey);
    status = ibv_post_send(ep->qp, sr, bad_sr);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_poll_cq failed \n");

    ep->head_op_id++;

    if (unlikely(!verb.is_nbi && verb.desc != NVSHMEMI_OP_P)) {
        check_poll_avail(ep, WAIT_ALL /*1*/);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "check_poll failed \n");
    }
out:
    return status;
}

int nvshmemt_ibrc_amo(nvshmemt_ep_t tep, void *curetptr, amo_verb_t verb, amo_memdesc_t remote,
                      amo_bytesdesc_t bytesdesc) {
    int status = 0;
    struct ibrc_ep *ep = (struct ibrc_ep *)tep;
    transport_ibrc_state_t *ibrc_state = (transport_ibrc_state_t *)ep->ibrc_state;
    struct ibrc_device *device = ((struct ibrc_device *)ibrc_state->devices + ep->devid);
    struct ibv_send_wr *sr, **bad_sr;
    struct ibv_sge *sge;
    struct ibrc_request *req;
    int op_id;
    int op_prepared = 0;
    struct ibrc_atomic_op op;

    status = check_poll_avail(ep, WAIT_ANY);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "check_poll failed \n");

    op_id = ep->head_op_id & IBRC_REQUEST_QUEUE_MASK;  // ep->head_op_id % ibrc_qp_depth
    sr = &(ep->req + op_id)->sr;
    bad_sr = &(ep->req + op_id)->bad_sr;
    sge = &(ep->req + op_id)->sge;

    memset(sr, 0, sizeof(ibv_send_wr));
    memset(sge, 0, sizeof(ibv_sge));

    sr->num_sge = 1;
    sr->sg_list = sge;
    sr->wr_id = NVSHMEMI_OP_AMO;
    sr->next = NULL;

    if (use_ib_native_atomics) {
        if (verb.desc == NVSHMEMI_AMO_SIGNAL_ADD) {
            if (bytesdesc.elembytes = 8) {
                sr->opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
                sr->send_flags = IBV_SEND_SIGNALED;

                sr->wr.atomic.remote_addr = (uint64_t)remote.ptr;
                sr->wr.atomic.rkey = ((struct ibrc_mem_handle *)&remote.handle)->rkey;
                sr->wr.atomic.compare_add = remote.val;

                sge->length = bytesdesc.elembytes;
                sge->addr = (uintptr_t)dummy_local_mem->ptr;
                sge->lkey = dummy_local_mem->mr->lkey;
                goto post_op;
            }
        }
    }

#ifdef NVSHMEM_USE_GDRCOPY
    // if gdrcopy is available, use it for all atomics to guarantee
    // atomicity across different ops
    if (use_gdrcopy) {
        ibrc_mem_handle_info_t mem_handle_info;

        // assuming GDRCopy availability is uniform on all nodes
        op.op = verb.desc;
        op.addr = remote.ptr;
        op.retaddr = remote.retptr;
        op.retflag = remote.retflag;
        op.compare = remote.cmp;
        op.swap_add = remote.val;
        op.elembytes = bytesdesc.elembytes;

        // send rkey info
        assert(!mem_handle_cache.empty());
        mem_handle_info = mem_handle_cache.back();
        op.retrkey = mem_handle_info.mr->rkey;

        sr->opcode = IBV_WR_SEND;
        sr->send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
        sge->length = sizeof(struct ibrc_atomic_op);
        assert(sge->length <= IBRC_BUF_SIZE);
        sge->addr = (uintptr_t)&op;
        sge->lkey = 0;

        atomics_issued++;
        TRACE(NVSHMEM_TRANSPORT, "[%d] atomic issued : %llu \n", getpid(), atomics_issued);
        goto post_op;
    }
#endif

    if (use_ib_native_atomics) {
        if (verb.desc == NVSHMEMI_AMO_ADD) {
            if (bytesdesc.elembytes = 8) {
                sr->opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
                sr->send_flags = IBV_SEND_SIGNALED;

                sr->wr.atomic.remote_addr = (uint64_t)remote.ptr;
                sr->wr.atomic.rkey = ((struct ibrc_mem_handle *)&remote.handle)->rkey;
                sr->wr.atomic.compare_add = remote.val;

                sge->length = bytesdesc.elembytes;
                sge->addr = (uintptr_t)dummy_local_mem->ptr;
                sge->lkey = dummy_local_mem->mr->lkey;
                goto post_op;
            }
        } else if (verb.desc == NVSHMEMI_AMO_SIGNAL) {
            sr->opcode = IBV_WR_RDMA_WRITE;
            sr->send_flags = IBV_SEND_SIGNALED;
            sr->send_flags |= IBV_SEND_INLINE;

            sr->wr.rdma.remote_addr = (uint64_t)remote.ptr;
            sr->wr.rdma.rkey = ((struct ibrc_mem_handle *)&remote.handle)->rkey;

            sge->length = bytesdesc.elembytes;
            sge->addr = (uintptr_t)&remote.val;
            sge->lkey = 0;
            goto post_op;
        }
    }

    ERROR_EXIT("RMA/AMO verb %d not implemented\n", verb.desc);

post_op:
    status = ibv_post_send(ep->qp, sr, bad_sr);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_exp_post_send failed \n");

    ep->head_op_id++;

out:
    return status;
}

int nvshmemt_ibrc_enforce_cst_at_target() {
    int status = 0;
    ibrc_mem_handle_info_t mem_handle_info;

    if (mem_handle_cache.empty()) return status;

    // pick the last region that was inserted
    mem_handle_info = mem_handle_cache.back();

#ifdef NVSHMEM_USE_GDRCOPY
    if (use_gdrcopy) {
        int temp;
        gdrcopy_ftable.copy_from_mapping(mem_handle_info.mh, &temp, mem_handle_info.cpu_ptr,
                                         sizeof(int));
        return status;
    }
#endif

    struct ibrc_ep *ep = ibrc_cst_ep;
    transport_ibrc_state_t *ibrc_state = (transport_ibrc_state_t *)ep->ibrc_state;
    struct ibv_send_wr *sr, **bad_sr;
    struct ibv_sge *sge;
    struct ibrc_request *req;
    int op_id;

    op_id = ep->head_op_id & IBRC_REQUEST_QUEUE_MASK;  // ep->head_op_id % ibrc_qp_depth
    sr = &(ep->req + op_id)->sr;
    bad_sr = &(ep->req + op_id)->bad_sr;
    sge = &(ep->req + op_id)->sge;

    sr->next = NULL;
    sr->send_flags = IBV_SEND_SIGNALED;
    sr->num_sge = 1;
    sr->sg_list = sge;

    sr->opcode = IBV_WR_RDMA_READ;
    sr->wr.rdma.remote_addr = (uint64_t)mem_handle_info.ptr;
    sr->wr.rdma.rkey = mem_handle_info.mr->rkey;

    sge->length = sizeof(int);
    sge->addr = (uintptr_t)mem_handle_info.ptr;
    sge->lkey = mem_handle_info.mr->lkey;

    status = ibv_post_send(ep->qp, sr, bad_sr);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_post_send failed \n");

    ep->head_op_id++;

    status = check_poll_avail(ep, WAIT_ALL);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "check_poll failed \n");

out:
    return status;
}

int nvshmemt_ibrc_quiet(nvshmemt_ep_t tep) {
    int status = 0;
    struct ibrc_ep *ep = (struct ibrc_ep *)tep;
    static uint64_t quiet_count = 0;

    status = check_poll_avail(ep, WAIT_ALL /*1*/);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "check_poll failed \n");

    quiet_count++;
#ifdef NVSHMEM_USE_GDRCOPY
    while (atomics_acked < atomics_issued) {
        status = progress_recv((transport_ibrc_state_t *)ep->ibrc_state);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "progress failed \n");
    }
#endif
out:
    return status;
}

int nvshmemt_ibrc_ep_create(nvshmemt_ep_t *tep, int devid, nvshmem_transport_t t) {
    int status = 0;
    struct ibrc_ep *ep;
    struct nvshmem_transport *transport = (struct nvshmem_transport *)t;
    transport_ibrc_state_t *ibrc_state = (transport_ibrc_state_t *)transport->state;

    status = ep_create(&ep, devid, ibrc_state);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ep_create failed\n");
    *tep = ep;

    // setup loopback connection on the first device used.
    if (!ibrc_cst_ep) {
        status = setup_cst_loopback(ibrc_state, devid);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cst setup failed \n");
    }

out:
    return status;
}

int nvshmemt_ibrc_ep_get_handle(nvshmemt_ep_handle_t *ep_handle, nvshmemt_ep_t tep) {
    int status = 0;
    struct ibrc_ep *ep = (struct ibrc_ep *)tep;
    transport_ibrc_state_t *ibrc_state = (transport_ibrc_state_t *)ep->ibrc_state;
    struct ibrc_device *device = ((struct ibrc_device *)ibrc_state->devices + ep->devid);
    struct ibrc_ep_handle *ep_handle_ptr = (struct ibrc_ep_handle *)ep_handle;

    assert(sizeof(struct ibrc_ep_handle) <= NVSHMEM_EP_HANDLE_SIZE);

    status = ep_get_handle(ep_handle_ptr, ep);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ep_get_handle failed \n");

out:
    return status;
}

int nvshmemt_ibrc_ep_destroy(nvshmemt_ep_t tep) {
    int status = 0;
    struct ibrc_ep *ep = (struct ibrc_ep *)tep;

    status = check_poll_avail(ep, WAIT_ALL /*1*/);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "check_poll failed \n");

    // TODO: clean up qp, cq, etc.

out:
    return status;
}

int nvshmemt_ibrc_ep_connect(nvshmemt_ep_t tep, nvshmemt_ep_handle_t remote_ep_handle) {
    int status = 0;
    struct ibrc_ep *ep = (struct ibrc_ep *)tep;
    struct ibrc_ep_handle *ep_handle = (struct ibrc_ep_handle *)&remote_ep_handle;

    status = ep_connect(ep, ep_handle);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ep_connect failed \n");

out:
    return status;
}

#define LOAD_SYM(handle, symbol, funcptr)  \
    do {                                   \
        void **cast = (void **)&funcptr;   \
        void *tmp = dlsym(handle, symbol); \
        *cast = tmp;                       \
    } while (0)

int nvshmemt_ibrc_init(nvshmem_transport_t *t) {
    int status = 0;
    struct nvshmem_transport *transport;
    transport_ibrc_state_t *ibrc_state;
    struct ibv_device **dev_list = NULL;
    int num_devices;
    struct ibrc_device *device;
    char *value = NULL;
    std::vector<std::string> nic_names_n_pes;
    std::vector<std::string> nic_names;
    int exclude_list = 0;
    int pes_counted = 0;
    struct ibrc_hca_info hca_list[MAX_NUM_HCAS];
    struct ibrc_hca_info pe_hca_mapping[MAX_NUM_PES_PER_NODE];
    int hca_list_count = 0, pe_hca_map_count = 0, user_selection = 0;
    int offset = 0;

    ibv_handle = dlopen("libibverbs.so", RTLD_LAZY);
    if (ibv_handle == NULL) {
        INFO(NVSHMEM_INIT, "libibverbs not found on the system; skipping IB RC transport \n");
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }

    LOAD_SYM(ibv_handle, "ibv_get_device_list", ftable.get_device_list);
    LOAD_SYM(ibv_handle, "ibv_get_device_name", ftable.get_device_name);
    LOAD_SYM(ibv_handle, "ibv_open_device", ftable.open_device);
    LOAD_SYM(ibv_handle, "ibv_close_device", ftable.close_device);
    LOAD_SYM(ibv_handle, "ibv_query_port", ftable.query_port);
    LOAD_SYM(ibv_handle, "ibv_query_device", ftable.query_device);
    LOAD_SYM(ibv_handle, "ibv_alloc_pd", ftable.alloc_pd);
    LOAD_SYM(ibv_handle, "ibv_reg_mr", ftable.reg_mr);
    LOAD_SYM(ibv_handle, "ibv_dereg_mr", ftable.dereg_mr);
    LOAD_SYM(ibv_handle, "ibv_create_cq", ftable.create_cq);
    LOAD_SYM(ibv_handle, "ibv_create_qp", ftable.create_qp);
    LOAD_SYM(ibv_handle, "ibv_create_srq", ftable.create_srq);
    LOAD_SYM(ibv_handle, "ibv_modify_qp", ftable.modify_qp);
    LOAD_SYM(ibv_handle, "ibv_query_gid", ftable.query_gid);

    if (nvshmemi_options.DISABLE_IB_NATIVE_ATOMICS) {
        use_ib_native_atomics = 0;
    }
    ibrc_srq_depth = nvshmemi_options.SRQ_DEPTH;
    ibrc_qp_depth = nvshmemi_options.QP_DEPTH;

#ifdef NVSHMEM_USE_GDRCOPY
    use_gdrcopy = 1;
    if (nvshmemi_options.DISABLE_GDRCOPY) {
        use_gdrcopy = 0;
    }

    gdrcopy_handle = dlopen("libgdrapi.so", RTLD_LAZY);
    if (!gdrcopy_handle) use_gdrcopy = 0;

    if (use_gdrcopy) {
        LOAD_SYM(gdrcopy_handle, "gdr_runtime_get_version", gdrcopy_ftable.runtime_get_version);
        if (!gdrcopy_ftable.runtime_get_version) {
            WARN_PRINT("GDRCopy library found by version older than 2.0, skipping use \n");
            use_gdrcopy = 0;
            goto skip_gdrcopy_dlsym;
        }
        LOAD_SYM(gdrcopy_handle, "gdr_runtime_get_version", gdrcopy_ftable.driver_get_version);
        LOAD_SYM(gdrcopy_handle, "gdr_open", gdrcopy_ftable.open);
        LOAD_SYM(gdrcopy_handle, "gdr_close", gdrcopy_ftable.close);
        LOAD_SYM(gdrcopy_handle, "gdr_pin_buffer", gdrcopy_ftable.pin_buffer);
        LOAD_SYM(gdrcopy_handle, "gdr_unpin_buffer", gdrcopy_ftable.unpin_buffer);
        LOAD_SYM(gdrcopy_handle, "gdr_map", gdrcopy_ftable.map);
        LOAD_SYM(gdrcopy_handle, "gdr_unmap", gdrcopy_ftable.unmap);
        LOAD_SYM(gdrcopy_handle, "gdr_get_info", gdrcopy_ftable.get_info);
        LOAD_SYM(gdrcopy_handle, "gdr_copy_from_mapping", gdrcopy_ftable.copy_from_mapping);
        LOAD_SYM(gdrcopy_handle, "gdr_copy_to_mapping", gdrcopy_ftable.copy_to_mapping);

        gdr_desc = gdrcopy_ftable.open();
        if (!gdr_desc) {
            use_gdrcopy = 0;
            WARN_PRINT("GDRCopy open call failed, falling back to not using GDRCopy \n");
        }
    }
skip_gdrcopy_dlsym:
#endif

    transport = (struct nvshmem_transport *)malloc(sizeof(struct nvshmem_transport));
    memset(transport, 0, sizeof(struct nvshmem_transport));
    transport->is_successfully_initialized =
        false; /* set it to true after everything has been successfully initialized */

    ibrc_state = (transport_ibrc_state_t *)malloc(sizeof(transport_ibrc_state_t));
    NULL_ERROR_JMP(ibrc_state, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                   "p2p state allocation failed \n");

    dev_list = ftable.get_device_list(&num_devices);
    NULL_ERROR_JMP(dev_list, status, NVSHMEMX_ERROR_INTERNAL, out, "get_device_list failed \n");

    ibrc_state->devices = calloc(MAX_NUM_HCAS, sizeof(struct ibrc_device));
    NULL_ERROR_JMP(ibrc_state->devices, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                   "get_device_list failed \n");

    ibrc_state->dev_ids = (int *)malloc(MAX_NUM_PES_PER_NODE * sizeof(int));
    NULL_ERROR_JMP(ibrc_state->dev_ids, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                   "malloc failed \n");

    ibrc_state->port_ids = (int *)malloc(MAX_NUM_PES_PER_NODE * sizeof(int));
    NULL_ERROR_JMP(ibrc_state->port_ids, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                   "malloc failed \n");

    status = pthread_mutex_init(&ibrc_mutex_send_progress, NULL);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "pthread_mutex_init failed \n");

    status = pthread_mutex_init(&ibrc_mutex_recv_progress, NULL);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "pthread_mutex_init failed \n");

    if (nvshmemi_options.HCA_LIST_provided) {
        user_selection = 1;
        exclude_list = (nvshmemi_options.HCA_LIST[0] == '^');
        hca_list_count = parse_hca_list(nvshmemi_options.HCA_LIST, hca_list, MAX_NUM_HCAS);
    }

    if (nvshmemi_options.HCA_PE_MAPPING_provided) {
        if (hca_list_count) {
            WARN_PRINT(
                "Found conflicting parameters NVSHMEM_HCA_LIST and NVSHMEM_HCA_PE_MAPPING, "
                "ignoring "
                "NVSHMEM_HCA_PE_MAPPING \n");
        } else {
            user_selection = 1;
            pe_hca_map_count = parse_hca_list(nvshmemi_options.HCA_PE_MAPPING, pe_hca_mapping,
                                              MAX_NUM_PES_PER_NODE);
        }
    }

    INFO(NVSHMEM_INIT,
         "Begin - Enumerating IB devices in the system ([<dev_id, device_name, num_ports>]) - ");
    for (int i = 0; i < num_devices; i++) {
        device = (struct ibrc_device *)ibrc_state->devices + i;
        device->dev = dev_list[i];

        device->context = ftable.open_device(device->dev);
        if (!device->context) {
            INFO(NVSHMEM_INIT, "open_device failed for IB device at index %d \n", i);
            continue;
        }

        const char *name = ftable.get_device_name(device->dev);
        NULL_ERROR_JMP(name, status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_get_device_name failed \n");

        status = ftable.query_device(device->context, &device->device_attr);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_query_device failed \n");

        MAX_RD_ATOMIC = (device->device_attr).max_qp_rd_atom;
        INFO(NVSHMEM_INIT,
             "Enumerated IB devices in the system - device id=%d (of %d), name=%s, num_ports=%d", i,
             num_devices, name, device->device_attr.phys_port_cnt);
        int device_used = 0;
        for (int p = 1; p <= device->device_attr.phys_port_cnt; p++) {
            int allowed_device = 1;
            int replicate_count = 1;
            if (hca_list_count) {
                // filter out based on user hca list
                allowed_device = exclude_list;
                for (int j = 0; j < hca_list_count; j++) {
                    if (!strcmp(hca_list[j].name, name)) {
                        if (hca_list[j].port == -1 || hca_list[j].port == p) {
                            hca_list[j].found = 1;
                            allowed_device = !exclude_list;
                        }
                    }
                }
            } else if (pe_hca_map_count) {
                // filter devices based on user hca-pe mapping
                allowed_device = 0;
                for (int j = 0; j < pe_hca_map_count; j++) {
                    if (!strcmp(pe_hca_mapping[j].name, name)) {
                        if (pe_hca_mapping[j].port == -1 || pe_hca_mapping[j].port == p) {
                            allowed_device = 1;
                            pe_hca_mapping[j].found = 1;
                            replicate_count = pe_hca_mapping[j].count;
                        }
                    }
                }
            }

            if (!allowed_device) {
                continue;
            } else {
                status = ftable.query_port(device->context, p, &device->port_attr[p - 1]);
                NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_port_query failed \n");

                if ((device->port_attr[p - 1].state != IBV_PORT_ACTIVE) 
                    || (device->port_attr[p - 1].link_layer != IBV_LINK_LAYER_INFINIBAND
                    && device->port_attr[p - 1].link_layer != IBV_LINK_LAYER_ETHERNET)) { 

                    if (user_selection) {
                        WARN_PRINT(
                            "found inactive port or port with non-IB link layer protocol, "
                            "skipping...\n");
                    }
                    continue;
                }

                status = ftable.query_gid(device->context, p, nvshmemi_options.IB_GID_INDEX, &device->gid[p - 1]);
                NULL_ERROR_JMP(dev_list, status, NVSHMEMX_ERROR_INTERNAL, out, "query_gid failed \n");

                device->pd = ftable.alloc_pd(device->context);
                NULL_ERROR_JMP(device->pd, status, NVSHMEMX_ERROR_INTERNAL, out,
                               "ibv_alloc_pd failed \n");

                for (int k = 0; k < replicate_count; k++) {
                    ibrc_state->dev_ids[offset] = i;
                    ibrc_state->port_ids[offset] = p;
                    offset++;
                }

                device_used = 1;
            }
        }

        if (!device_used) {
            status = ftable.close_device(device->context);
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_port_query failed \n");
        }
    }
    INFO(NVSHMEM_INIT, "End - Enumerating IB devices in the system");

    ibrc_state->n_dev_ids = offset;
    INFO(NVSHMEM_INIT,
         "Begin - Ordered list of devices for assignment (after processing user provdied env vars "
         "(if any))  - ");
    for (int i = 0; i < ibrc_state->n_dev_ids; i++) {
        INFO(NVSHMEM_INIT,
             "Ordered list of devices for assignment - idx=%d (of %d), device id=%d, port_num=%d",
             i, ibrc_state->n_dev_ids, ibrc_state->dev_ids[i], ibrc_state->port_ids[i]);
    }
    INFO(NVSHMEM_INIT,
         "End - Ordered list of devices for assignment (after processing user provdied env vars "
         "(if any))");

    if (!ibrc_state->n_dev_ids) {
        INFO(NVSHMEM_INIT, "no active IB device found, exiting");
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }

    // print devices that were not found
    if (hca_list_count) {
        for (int j = 0; j < hca_list_count; j++) {
            if (hca_list[j].found != 1) {
                WARN_PRINT("cound not find user specified HCA name: %s port: %d, skipping\n",
                           hca_list[j].name, hca_list[j].port);
            }
        }
    } else if (pe_hca_map_count) {
        // filter devices based on user hca-pe mapping
        for (int j = 0; j < pe_hca_map_count; j++) {
            if (pe_hca_mapping[j].found != 1) {
                WARN_PRINT("cound not find user specified HCA name: %s port: %d, skipping\n",
                           pe_hca_mapping[j].name, pe_hca_mapping[j].port);
            }
        }
    }

    // allocate buffer pool
    bpool_size = ibrc_srq_depth;
    bpool = (ibrc_buf_t *)calloc(bpool_size, sizeof(ibrc_buf_t));
    NULL_ERROR_JMP(bpool, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                   "buf poll allocation failed \n");
    for (int i = 0; i < bpool_size; i++) {
        bpool_free.push_back((void *)(bpool + i));
    }

    transport->host_ops.get_device_count = nvshmemt_ibrc_get_device_count;
    transport->host_ops.get_pci_path = nvshmemt_ibrc_get_pci_path;
    transport->host_ops.can_reach_peer = nvshmemt_ibrc_can_reach_peer;
    transport->host_ops.ep_create = nvshmemt_ibrc_ep_create;
    transport->host_ops.ep_get_handle = nvshmemt_ibrc_ep_get_handle;
    transport->host_ops.ep_connect = nvshmemt_ibrc_ep_connect;
    transport->host_ops.ep_destroy = nvshmemt_ibrc_ep_destroy;
    transport->host_ops.get_mem_handle = nvshmemt_ibrc_get_mem_handle;
    transport->host_ops.release_mem_handle = nvshmemt_ibrc_release_mem_handle;
    transport->host_ops.rma = nvshmemt_ibrc_rma;
    transport->host_ops.amo = nvshmemt_ibrc_amo;
    transport->host_ops.fence = NULL;
    transport->host_ops.quiet = nvshmemt_ibrc_quiet;
    transport->host_ops.finalize = nvshmemt_ibrc_finalize;
    transport->host_ops.show_info = nvshmemt_ibrc_show_info;
    transport->host_ops.progress = nvshmemt_ibrc_progress;

    transport->host_ops.enforce_cst = nvshmemt_ibrc_enforce_cst_at_target;
#ifndef NVSHMEM_PPC64LE
    if (!use_gdrcopy)
#endif
        transport->host_ops.enforce_cst_at_target = nvshmemt_ibrc_enforce_cst_at_target;

    transport->attr = NVSHMEM_TRANSPORT_ATTR_CONNECTED;
    transport->state = (void *)ibrc_state;
    transport->is_successfully_initialized = true;

    *t = transport;

out:
    return status;
}
