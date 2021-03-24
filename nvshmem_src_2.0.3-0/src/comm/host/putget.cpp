/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "nvshmem_internal.h"
#include "transport.h"
#include "util.h"

template <nvshmemi_op_t desc, int is_nbi>
int nvshmemi_proxy_rma_launcher(void *args[], cudaStream_t cstrm);

static int nvshmemi_p2p_rma(CUstream custrm, CUevent cuev, rma_verb_t verb, rma_memdesc_t dest,
                            rma_memdesc_t src, rma_bytesdesc_t bytesdesc) {
    int status = 0;
    bool is_contig = ((bytesdesc.srcstride == 1) && (bytesdesc.deststride == 1)) ? true : false;
    bool is_single_word = ((verb.desc == NVSHMEMI_OP_P) || (verb.desc == NVSHMEMI_OP_G)) ? true : false;
    if (verb.is_stream) {
        if (verb.is_nbi) {
            status = cuEventRecord(cuev, (CUstream)verb.cstrm);
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cuEventRecord() failed\n");
            status = cuStreamWaitEvent(custrm, cuev, 0);
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cuStreamWaitEvent() failed\n");
            if (is_contig) { /*can include iput,iget in future*/
                status = cuMemcpyDtoDAsync((CUdeviceptr)dest.ptr, (CUdeviceptr)src.ptr,
                                           bytesdesc.nelems * bytesdesc.elembytes, custrm);
                NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cuMemcpyDtoDAsync() failed\n");
            }
        } else { /*!is_nbi*/
            if (is_contig) {
                if (is_single_word) {
                    if (verb.desc == NVSHMEMI_OP_P) {
                        status = cuMemcpyHtoDAsync((CUdeviceptr)dest.ptr, src.ptr,
                                                   bytesdesc.nelems * bytesdesc.elembytes,
                                                   (CUstream)verb.cstrm);
                        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                     "cuMemcpyHtoDAsync() failed\n");
                    } else { /*!is P*/
                        status = cuMemcpyDtoHAsync(dest.ptr, (CUdeviceptr)src.ptr,
                                                   bytesdesc.nelems * bytesdesc.elembytes,
                                                   (CUstream)verb.cstrm);
                        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                     "cuMemcpyDtoHAsync() failed\n");
                    }
                } else { /*!is_single_word*/
                    status = cuMemcpyDtoDAsync((CUdeviceptr)dest.ptr, (CUdeviceptr)src.ptr,
                                               bytesdesc.nelems * bytesdesc.elembytes,
                                               (CUstream)verb.cstrm);
                    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                 "cuMemcpyDtoDAsync() failed \n");
                }
            } else { /*!is_contig*/
                CUDA_MEMCPY2D pCopy;
                memset(&pCopy, 0, sizeof(pCopy));
                pCopy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
                pCopy.dstDevice = (CUdeviceptr)dest.ptr;
                pCopy.dstPitch = bytesdesc.deststride * bytesdesc.elembytes;
                pCopy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
                pCopy.srcDevice = (CUdeviceptr)src.ptr;
                pCopy.srcPitch = bytesdesc.srcstride * bytesdesc.elembytes;
                pCopy.WidthInBytes = bytesdesc.elembytes;
                pCopy.Height = bytesdesc.nelems;
                status = cuMemcpy2DAsync(&pCopy, (CUstream)verb.cstrm);
                NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cuMemcpy2DAsync() failed\n");
            } /*is_contig*/
        }     /*is_nbi*/
    } else {  /*!is_stream*/
        if (verb.is_nbi) {
            if (is_contig) { /*can include iput,iget in future*/
                status = cuMemcpyDtoDAsync((CUdeviceptr)dest.ptr, (CUdeviceptr)src.ptr,
                                           bytesdesc.nelems * bytesdesc.elembytes, custrm);
                NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cuMemcpyDtoDAsync() failed\n");
            }
        } else { /*!is_nbi*/
            if (is_contig) {
                if (is_single_word) {
                    if (verb.desc == NVSHMEMI_OP_P) {
                        status = cuMemcpyHtoDAsync((CUdeviceptr)dest.ptr, src.ptr,
                                                   bytesdesc.nelems * bytesdesc.elembytes, custrm);
                        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                     "cuMemcpyHtoDAsync() failed\n");
                    } else { /*!is P*/
                        status = cuMemcpyDtoHAsync(dest.ptr, (CUdeviceptr)src.ptr,
                                                   bytesdesc.nelems * bytesdesc.elembytes, custrm);
                        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                     "cuMemcpyDtoHAsync() failed\n");
                    }
                } else { /*!is_single_word*/
                    status = cuMemcpyDtoDAsync((CUdeviceptr)dest.ptr, (CUdeviceptr)src.ptr,
                                               bytesdesc.nelems * bytesdesc.elembytes, custrm);
                    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                 "cuMemcpyDtoDAsync() failed\n");
                }
            } else { /*!is_contig*/
                CUDA_MEMCPY2D pCopy;
                memset(&pCopy, 0, sizeof(pCopy));
                pCopy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
                pCopy.dstDevice = (CUdeviceptr)dest.ptr;
                pCopy.dstPitch = bytesdesc.deststride * bytesdesc.elembytes;
                pCopy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
                pCopy.srcDevice = (CUdeviceptr)src.ptr;
                pCopy.srcPitch = bytesdesc.srcstride * bytesdesc.elembytes;
                pCopy.WidthInBytes = bytesdesc.elembytes;
                pCopy.Height = bytesdesc.nelems;
                status = cuMemcpy2DAsync(&pCopy, custrm);
                NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cuMemcpy2DAsync() failed\n");
            } /*is_contig*/
            status = cuStreamSynchronize(custrm);
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cuStreamSynchronize() failed \n");
        } /*is_nbi*/
    }     /*is_stream*/
out:
    return status;
}

static void nvshmemi_prepare_and_post_rma(const char *apiname, nvshmemi_op_t desc, int is_nbi,
                                          int is_stream, void *destptr, void *srcptr,
                                          ptrdiff_t deststride, ptrdiff_t srcstride, int nelems,
                                          size_t elembytes, int pe, cudaStream_t cstrm) {
    int status = 0;
    rma_verb_t verb = {desc, is_nbi, is_stream, cstrm};
    rma_bytesdesc_t bytesdesc = {(size_t)nelems, (int)elembytes, srcstride, deststride};
    void *destptr_actual = destptr, *srcptr_actual = srcptr;
    rma_memdesc_t dest, src;
    memset(&src, 0, sizeof(rma_memdesc_t));
    memset(&dest, 0, sizeof(rma_memdesc_t));
    dest.ptr = (void *)destptr_actual;
    src.ptr = (void *)srcptr_actual;
    if (nvshmemi_state->peer_heap_base[pe]) {
        CUstream custrm = nvshmemi_state->custreams[pe % MAX_PEER_STREAMS];
        CUevent cuev = nvshmemi_state->cuevents[pe % MAX_PEER_STREAMS];
        if ((verb.desc == NVSHMEMI_OP_P) || (verb.desc == NVSHMEMI_OP_PUT)) {
            NVSHMEMU_MAPPED_PTR_TRANSLATE(destptr_actual, destptr, pe)
            dest.ptr = (void *)destptr_actual;
        } else {
            NVSHMEMU_MAPPED_PTR_TRANSLATE(srcptr_actual, srcptr, pe)
            src.ptr = (void *)srcptr_actual;
        }
        status = nvshmemi_p2p_rma(custrm, cuev, verb, dest, src, bytesdesc);
    } else {
        int t = nvshmemi_state->selected_transport_for_rma[pe];
        if (t < 0) {
            ERROR_EXIT("[%d] rma not supported on transport to pe: %d \n", nvshmemi_state->mype, pe);
        }

        nvshmemt_ep_t ep;
        int tcount = nvshmemi_state->transport_count;
        int mype = nvshmemi_state->mype;
        struct nvshmem_transport *tcurr = nvshmemi_state->transports[t];
        int ep_offset = pe * tcurr->ep_count;
        ep = tcurr->ep[ep_offset];
        nvshmem_mem_handle_t *handles = nvshmemi_state->handles;
        if (verb.desc == NVSHMEMI_OP_PUT) {
            dest.handle = handles[pe * tcount + t];
            src.handle = handles[mype * tcount + t];
        } else if (verb.desc == NVSHMEMI_OP_P) {
            dest.handle = handles[pe * tcount + t];
        } else if (verb.desc == NVSHMEMI_OP_GET) {
            dest.handle = handles[mype * tcount + t];
            src.handle = handles[pe * tcount + t];
        } else {
            status = NVSHMEMX_ERROR_INTERNAL;
            ERROR_PRINT("NOT IMPLEMENTED %s \n", apiname);
            goto out;
        }
        if ((bytesdesc.srcstride > 1) || (bytesdesc.deststride > 1)) {
            status = NVSHMEMX_ERROR_INTERNAL;
            ERROR_PRINT("NOT IMPLEMENTED %s \n", apiname);
            goto out;
        }
        if (!verb.is_stream) {
            rma_memdesc_t remote, local;
            if ((verb.desc == NVSHMEMI_OP_PUT) || (verb.desc == NVSHMEMI_OP_P)) {
                NVSHMEMU_UNMAPPED_PTR_TRANSLATE(destptr_actual, destptr, pe)
                dest.ptr = (void *)destptr_actual;
                remote = dest;
                local = src;
            } else if (verb.desc == NVSHMEMI_OP_GET) {
                NVSHMEMU_UNMAPPED_PTR_TRANSLATE(srcptr_actual, srcptr, pe)
                src.ptr = (void *)srcptr_actual;
                remote = src;
                local = dest;
            }
            status = nvshmemi_state->rma[pe](ep, verb, remote, local, bytesdesc);
        } else {
            void *rptr, *lptr;
            if ((verb.desc == NVSHMEMI_OP_PUT) || (verb.desc == NVSHMEMI_OP_P)) {
                dest.ptr = (void *)destptr_actual;
                rptr = dest.ptr;
                lptr = src.ptr;
            } else if (verb.desc == NVSHMEMI_OP_GET) {
                src.ptr = (void *)srcptr_actual;
                rptr = src.ptr;
                lptr = dest.ptr;
            }
            void *args[] = {&rptr, &lptr, &bytesdesc, &pe};
            if (is_nbi) {
                if (verb.desc == NVSHMEMI_OP_PUT) {
                    status = nvshmemi_proxy_rma_launcher<NVSHMEMI_OP_PUT, 1>(args, cstrm);
                } else if (verb.desc == NVSHMEMI_OP_GET) {
                    status = nvshmemi_proxy_rma_launcher<NVSHMEMI_OP_GET, 1>(args, cstrm);
                }
            } else {
                /*!is_nbi*/
                if (verb.desc == NVSHMEMI_OP_PUT) {
                    status = nvshmemi_proxy_rma_launcher<NVSHMEMI_OP_PUT, 0>(args, cstrm);
                } else if (verb.desc == NVSHMEMI_OP_GET) {
                    status = nvshmemi_proxy_rma_launcher<NVSHMEMI_OP_GET, 0>(args, cstrm);
                }
            }
            if (status) {
                ERROR_PRINT("cudaLaunchKernel() failed in %s \n", apiname);
            }
        }
    }
out:
    if (status) {
        ERROR_PRINT("aborting due to error in %s \n", apiname);
        exit(-1);
    }
}

/***** Put APIs ******/

#define NVSHMEM_TYPE_PUT(Name, TYPE)                                                             \
    void nvshmem_##Name##_put(TYPE *dest, const TYPE *source, size_t nelems, int pe) {           \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                          \
        /*INFO(NVSHMEM_P2P, "[%d] bulk put : (remote)dest %p, (local)source %p, %d elements,     \
         * remote PE %d", nvshmemi_state->mype, dest, source, nelems, pe);*/                      \
        nvshmemi_prepare_and_post_rma(                                                           \
            "nvshmem_" #Name "_put", NVSHMEMI_OP_PUT, NO_NBI, NO_ASYNC, (void *)dest, (void *)source,        \
            DEST_STRIDE_CONTIG, SRC_STRIDE_CONTIG, nelems, sizeof(TYPE), pe, NOT_A_CUDA_STREAM); \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEM_TYPE_PUT)
#undef NVSHMEM_TYPE_PUT


#define NVSHMEMX_TYPE_PUT_ON_STREAM(Name, TYPE)                                                 \
    void nvshmemx_##Name##_put_on_stream(TYPE *dest, const TYPE *source, size_t nelems, int pe, \
                                         cudaStream_t cstrm) {                                  \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                         \
        nvshmemi_prepare_and_post_rma("nvshmem_" #Name "_put_on_stream", NVSHMEMI_OP_PUT, NO_NBI, ASYNC,    \
                                      (void *)dest, (void *)source, DEST_STRIDE_CONTIG,         \
                                      SRC_STRIDE_CONTIG, nelems, sizeof(TYPE), pe, cstrm);      \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_TYPE_PUT_ON_STREAM)
#undef NVSHMEMX_TYPE_PUT_ON_STREAM

#define NVSHMEM_PUTSIZE(Name, Type)                                                                \
    void nvshmem_put##Name(void *dest, const void *source, size_t nelems, int pe) {                \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                            \
        nvshmemi_prepare_and_post_rma("nvshmem_put" #Name "", NVSHMEMI_OP_PUT, NO_NBI, NO_ASYNC, (void *)dest, \
                                      (void *)source, DEST_STRIDE_CONTIG, SRC_STRIDE_CONTIG,       \
                                      nelems, sizeof(Type), pe, NOT_A_CUDA_STREAM);                        \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEM_PUTSIZE)
#undef NVSHMEM_PUTSIZE

#define NVSHMEMX_PUTSIZE_ON_STREAM(Name, Type)                                                 \
    void nvshmemx_put##Name##_on_stream(void *dest, const void *source, size_t nelems, int pe, \
                                        cudaStream_t cstrm) {                                  \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                        \
        nvshmemi_prepare_and_post_rma("nvshmemx_put" #Name "_on_stream", NVSHMEMI_OP_PUT, NO_NBI, ASYNC,   \
                                      (void *)dest, (void *)source, DEST_STRIDE_CONTIG,        \
                                      SRC_STRIDE_CONTIG, nelems, sizeof(Type), pe, cstrm);     \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEMX_PUTSIZE_ON_STREAM)
#undef NVSHMEMX_PUTSIZE_ON_STREAM

/*XXX:Should other comm/put APIs call into nvshmem_putmem (suggested by SP)*/
void nvshmem_putmem(void *dest, const void *source, size_t bytes, int pe) {
    NVSHMEM_CHECK_STATE_AND_INIT();
    INFO(NVSHMEM_P2P,
         "[%d] untyped put : (remote)dest %p, (local)source %p, %d bytes, remote PE %d",
         nvshmemi_state->mype, dest, source, bytes, pe);
    nvshmemi_prepare_and_post_rma("nvshmem_putmem", NVSHMEMI_OP_PUT, NO_NBI, NO_ASYNC, (void *)dest,
                                  (void *)source, DEST_STRIDE_CONTIG, SRC_STRIDE_CONTIG, bytes, 1,
                                  pe, NOT_A_CUDA_STREAM);
}

void nvshmemx_putmem_on_stream(void *dest, const void *source, size_t bytes, int pe,
                               cudaStream_t cstrm) {
    NVSHMEM_CHECK_STATE_AND_INIT();
    INFO(NVSHMEM_P2P,
         "[%d] untyped put : (remote)dest %p, (local)source %p, %d bytes, remote PE %d",
         nvshmemi_state->mype, dest, source, bytes, pe);
    nvshmemi_prepare_and_post_rma("nvshmemx_putmem_on_stream", NVSHMEMI_OP_PUT, NO_NBI, ASYNC, (void *)dest,
                                  (void *)source, DEST_STRIDE_CONTIG, SRC_STRIDE_CONTIG, bytes, 1,
                                  pe, cstrm);
}

#define NVSHMEM_TYPE_P(Name, TYPE)                                                              \
    void nvshmem_##Name##_p(TYPE *dest, const TYPE value, int pe) {                             \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                         \
        nvshmemi_prepare_and_post_rma("nvshmem_" #Name "_p", NVSHMEMI_OP_P, NO_NBI, NO_ASYNC, (void *)dest, \
                                      (void *)&value, DEST_STRIDE_CONTIG, SRC_STRIDE_CONTIG, 1, \
                                      sizeof(TYPE), pe, NOT_A_CUDA_STREAM);                     \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEM_TYPE_P)
#undef NVSHMEM_TYPE_P

#define NVSHMEMX_TYPE_P_ON_STREAM(Name, TYPE)                                                      \
    void nvshmemx_##Name##_p_on_stream(TYPE *dest, const TYPE value, int pe, cudaStream_t cstrm) { \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                            \
        nvshmemi_prepare_and_post_rma("nvshmem_" #Name "_p_on_stream", NVSHMEMI_OP_P, NO_NBI, ASYNC,           \
                                      (void *)dest, (void *)&value, DEST_STRIDE_CONTIG,            \
                                      SRC_STRIDE_CONTIG, 1, sizeof(TYPE), pe, cstrm);              \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_TYPE_P_ON_STREAM)
#undef NVSHMEMX_TYPE_P_ON_STREAM

#define NVSHMEM_TYPE_IPUT(Name, TYPE)                                                         \
    void nvshmem_##Name##_iput(TYPE *dest, const TYPE *source, ptrdiff_t dst, ptrdiff_t sst,  \
                               size_t nelems, int pe) {                                       \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                       \
        nvshmemi_prepare_and_post_rma("nvshmem_" #Name "_iput", NVSHMEMI_OP_PUT, NO_NBI, NO_ASYNC,        \
                                      (void *)dest, (void *)source, dst, sst, nelems,         \
                                      sizeof(TYPE), pe, NOT_A_CUDA_STREAM);                   \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEM_TYPE_IPUT)
#undef NVSHMEM_TYPE_IPUT

#define NVSHMEMX_TYPE_IPUT_ON_STREAM(Name, TYPE)                                              \
    void nvshmemx_##Name##_iput_on_stream(TYPE *dest, const TYPE *source, ptrdiff_t dst,      \
                                          ptrdiff_t sst, size_t nelems, int pe,               \
                                          cudaStream_t cstrm) {                               \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                       \
        nvshmemi_prepare_and_post_rma("nvshmem_" #Name "_iput_on_stream", NVSHMEMI_OP_PUT, NO_NBI, ASYNC, \
                                      (void *)dest, (void *)source, dst, sst, nelems,         \
                                      sizeof(TYPE), pe, cstrm);                               \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_TYPE_IPUT_ON_STREAM)
#undef NVSHMEMX_TYPE_IPUT_ON_STREAM

#define NVSHMEM_IPUTSIZE(Name, Type)                                                          \
    void nvshmem_iput##Name(void *dest, const void *source, ptrdiff_t dst, ptrdiff_t sst,     \
                            size_t nelems, int pe) {                                          \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                       \
        nvshmemi_prepare_and_post_rma("nvshmem_iput" #Name "", NVSHMEMI_OP_PUT, NO_NBI, NO_ASYNC,         \
                                      (void *)dest, (void *)source, dst, sst, nelems,         \
                                      sizeof(Type), pe, NOT_A_CUDA_STREAM);                   \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEM_IPUTSIZE)
#undef NVSHMEM_IPUTSIZE

#define NVSHMEMX_IPUTSIZE_ON_STREAM(Name, Type)                                               \
    void nvshmemx_iput##Name##_on_stream(void *dest, const void *source, ptrdiff_t dst,       \
                                         ptrdiff_t sst, size_t nelems, int pe,                \
                                         cudaStream_t cstrm) {                                \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                       \
        nvshmemi_prepare_and_post_rma("nvshmem_iput" #Name "_on_stream", NVSHMEMI_OP_PUT, NO_NBI, ASYNC,  \
                                      (void *)dest, (void *)source, dst, sst, nelems,         \
                                      sizeof(Type), pe, cstrm);                                       \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEMX_IPUTSIZE_ON_STREAM)
#undef NVSHMEMX_IPUTSIZE_ON_STREAM

#define NVSHMEM_TYPE_PUT_NBI(type, TYPE)                                                         \
    void nvshmem_##type##_put_nbi(TYPE *dest, const TYPE *source, size_t nelems, int pe) {       \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                          \
        nvshmemi_prepare_and_post_rma(                                                           \
            "nvshmem_" #type "_put_nbi", NVSHMEMI_OP_PUT, NBI, NO_ASYNC, (void *)dest, (void *)source,       \
            DEST_STRIDE_CONTIG, SRC_STRIDE_CONTIG, nelems, sizeof(TYPE), pe, NOT_A_CUDA_STREAM); \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEM_TYPE_PUT_NBI)
#undef NVSHMEM_TYPE_PUT_NBI

#define NVSHMEMX_TYPE_PUT_NBI_ON_STREAM(type, TYPE)                                           \
    void nvshmemx_##type##_put_nbi_on_stream(TYPE *dest, const TYPE *source, size_t nelems,   \
                                             int pe, cudaStream_t cstrm) {                    \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                       \
        nvshmemi_prepare_and_post_rma("nvshmem_" #type "_put_nbi_on_stream", NVSHMEMI_OP_PUT, NBI, ASYNC, \
                                      (void *)dest, (void *)source, DEST_STRIDE_CONTIG,       \
                                      SRC_STRIDE_CONTIG, nelems, sizeof(TYPE), pe, cstrm);    \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_TYPE_PUT_NBI_ON_STREAM)
#undef NVSHMEMX_TYPE_PUT_NBI_ON_STREAM

#define NVSHMEM_PUTSIZE_NBI(Name, Type)                                                        \
    void nvshmem_put##Name##_nbi(void *dest, const void *source, size_t nelems, int pe) {      \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                        \
        nvshmemi_prepare_and_post_rma("nvshmem_put" #Name "_nbi", NVSHMEMI_OP_PUT, NBI, NO_ASYNC,          \
                                      (void *)dest, (void *)source, DEST_STRIDE_CONTIG,        \
                                      SRC_STRIDE_CONTIG, nelems, sizeof(Type), pe, NOT_A_CUDA_STREAM); \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEM_PUTSIZE_NBI)
#undef NVSHMEM_PUTSIZE_NBI

#define NVSHMEMX_PUTSIZE_NBI_ON_STREAM(Name, Type)                                                 \
    void nvshmemx_put##Name##_nbi_on_stream(void *dest, const void *source, size_t nelems, int pe, \
                                            cudaStream_t cstrm) {                                  \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                            \
        nvshmemi_prepare_and_post_rma("nvshmem_put" #Name "_nbi_on_stream", NVSHMEMI_OP_PUT, NBI, ASYNC,       \
                                      (void *)dest, (void *)source, DEST_STRIDE_CONTIG,            \
                                      SRC_STRIDE_CONTIG, nelems, sizeof(Type), pe, cstrm);         \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEMX_PUTSIZE_NBI_ON_STREAM)
#undef NVSHMEMX_PUTSIZE_NBI_ON_STREAM

void nvshmem_putmem_nbi(void *dest, const void *source, size_t bytes, int pe) {
    NVSHMEM_CHECK_STATE_AND_INIT();
    INFO(NVSHMEM_P2P,
         "[%d] untyped put : (remote)dest %p, (local)source %p, %d bytes, remote PE %d",
         nvshmemi_state->mype, dest, source, bytes, pe);
    nvshmemi_prepare_and_post_rma("nvshmem_putmem_nbi", NVSHMEMI_OP_PUT, NBI, NO_ASYNC, (void *)dest,
                                  (void *)source, DEST_STRIDE_CONTIG, SRC_STRIDE_CONTIG, bytes, 1,
                                  pe, NOT_A_CUDA_STREAM);
}

void nvshmemx_putmem_nbi_on_stream(void *dest, const void *source, size_t bytes, int pe,
                                   cudaStream_t cstrm) {
    NVSHMEM_CHECK_STATE_AND_INIT();
    INFO(NVSHMEM_P2P,
         "[%d] untyped put : (remote)dest %p, (local)source %p, %d bytes, remote PE %d",
         nvshmemi_state->mype, dest, source, bytes, pe);
    nvshmemi_prepare_and_post_rma("nvshmem_putmem_nbi_on_stream", NVSHMEMI_OP_PUT, NBI, ASYNC, (void *)dest,
                                  (void *)source, DEST_STRIDE_CONTIG, SRC_STRIDE_CONTIG, bytes, 1,
                                  pe, cstrm);
}

/***** Get APIs ******/

#define NVSHMEM_TYPE_GET(Name, TYPE)                                                             \
    void nvshmem_##Name##_get(TYPE *dest, const TYPE *source, size_t nelems, int pe) {           \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                          \
        nvshmemi_prepare_and_post_rma(                                                           \
            "nvshmem_" #Name "_get", NVSHMEMI_OP_GET, NO_NBI, NO_ASYNC, (void *)dest, (void *)source,        \
            DEST_STRIDE_CONTIG, SRC_STRIDE_CONTIG, nelems, sizeof(TYPE), pe, NOT_A_CUDA_STREAM); \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEM_TYPE_GET)
#undef NVSHMEM_TYPE_GET

#define NVSHMEMX_TYPE_GET_ON_STREAM(Name, TYPE)                                                 \
    void nvshmemx_##Name##_get_on_stream(TYPE *dest, const TYPE *source, size_t nelems, int pe, \
                                         cudaStream_t cstrm) {                                  \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                         \
        nvshmemi_prepare_and_post_rma("nvshmem_" #Name "_get_on_stream", NVSHMEMI_OP_GET, NO_NBI, ASYNC,    \
                                      (void *)dest, (void *)source, DEST_STRIDE_CONTIG,         \
                                      SRC_STRIDE_CONTIG, nelems, sizeof(TYPE), pe, cstrm);      \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_TYPE_GET_ON_STREAM)
#undef NVSHMEMX_TYPE_GET_ON_STREAM

#define NVSHMEM_GETSIZE(Name, Type)                                                                \
    void nvshmem_get##Name(void *dest, const void *source, size_t nelems, int pe) {                \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                            \
        nvshmemi_prepare_and_post_rma("nvshmem_get" #Name "", NVSHMEMI_OP_GET, NO_NBI, NO_ASYNC, (void *)dest, \
                                      (void *)source, DEST_STRIDE_CONTIG, SRC_STRIDE_CONTIG,       \
                                      nelems, sizeof(Type), pe, NOT_A_CUDA_STREAM);                \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEM_GETSIZE)
#undef NVSHMEM_GETSIZE

#define NVSHMEMX_GETSIZE_ON_STREAM(Name, Type)                                                 \
    void nvshmemx_get##Name##_on_stream(void *dest, const void *source, size_t nelems, int pe, \
                                        cudaStream_t cstrm) {                                  \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                        \
        nvshmemi_prepare_and_post_rma("nvshmemx_get" #Name "_on_stream", NVSHMEMI_OP_GET, NO_NBI, ASYNC,   \
                                      (void *)dest, (void *)source, DEST_STRIDE_CONTIG,        \
                                      SRC_STRIDE_CONTIG, nelems, sizeof(Type), pe, cstrm);     \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEMX_GETSIZE_ON_STREAM)
#undef NVSHMEMX_GETSIZE_ON_STREAM

void nvshmem_getmem(void *dest, const void *source, size_t bytes, int pe) {
    NVSHMEM_CHECK_STATE_AND_INIT();
    INFO(NVSHMEM_P2P,
         "[%d] untyped get : (local)dest %p, (remote)source %p, %d bytes, remote PE %d",
         nvshmemi_state->mype, dest, source, bytes, pe);
    nvshmemi_prepare_and_post_rma("nvshmem_getmem", NVSHMEMI_OP_GET, NO_NBI, NO_ASYNC, (void *)dest,
                                  (void *)source, DEST_STRIDE_CONTIG, SRC_STRIDE_CONTIG, bytes, 1,
                                  pe, NOT_A_CUDA_STREAM);
}

void nvshmemx_getmem_on_stream(void *dest, const void *source, size_t bytes, int pe,
                               cudaStream_t cstrm) {
    NVSHMEM_CHECK_STATE_AND_INIT();
    INFO(NVSHMEM_P2P,
         "[%d] untyped get : (local)dest %p, (remote)source %p, %d bytes, remote PE %d",
         nvshmemi_state->mype, dest, source, bytes, pe);
    nvshmemi_prepare_and_post_rma("nvshmemx_getmem_on_stream", NVSHMEMI_OP_GET, NO_NBI, ASYNC, (void *)dest,
                                  (void *)source, DEST_STRIDE_CONTIG, SRC_STRIDE_CONTIG, bytes, 1,
                                  pe, cstrm);
}

#define NVSHMEM_TYPE_G(Name, TYPE)                                                                \
    TYPE nvshmem_##Name##_g(const TYPE *source, int pe) {                                         \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                           \
        TYPE value;                                                                               \
        INFO(NVSHMEM_P2P, "[%d] single get : (remote)source %p, remote PE %d",                    \
             nvshmemi_state->mype, source, pe);                                                    \
        nvshmemi_prepare_and_post_rma("nvshmem_" #Name "_g", NVSHMEMI_OP_G, NO_NBI, NO_ASYNC, (void *)&value, \
                                      (void *)source, DEST_STRIDE_CONTIG, SRC_STRIDE_CONTIG, 1,   \
                                      sizeof(TYPE), pe, NOT_A_CUDA_STREAM);                       \
        return value;                                                                             \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEM_TYPE_G)
#undef NVSHMEM_TYPE_G

#define NVSHMEMX_TYPE_G_ON_STREAM(Name, TYPE)                                             \
    TYPE nvshmemx_##Name##_g_on_stream(const TYPE *source, int pe, cudaStream_t cstrm) {  \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                   \
        TYPE value;                                                                       \
        INFO(NVSHMEM_P2P, "[%d] single get : (remote)source %p, remote PE %d",            \
             nvshmemi_state->mype, source, pe);                                            \
        nvshmemi_prepare_and_post_rma("nvshmem_" #Name "_g_on_stream", NVSHMEMI_OP_G, NO_NBI, ASYNC,  \
                                      (void *)&value, (void *)source, DEST_STRIDE_CONTIG, \
                                      SRC_STRIDE_CONTIG, 1, sizeof(TYPE), pe, cstrm);     \
        return value;                                                                     \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_TYPE_G_ON_STREAM)
#undef NVSHMEMX_TYPE_G_ON_STREAM

#define NVSHMEM_TYPE_IGET(Name, TYPE)                                                         \
    void nvshmem_##Name##_iget(TYPE *dest, const TYPE *source, ptrdiff_t dst, ptrdiff_t sst,  \
                               size_t nelems, int pe) {                                       \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                       \
        nvshmemi_prepare_and_post_rma("nvshmem_" #Name "_iget", NVSHMEMI_OP_GET, NO_NBI, NO_ASYNC,        \
                                      (void *)dest, (void *)source, dst, sst, nelems,         \
                                      sizeof(TYPE), pe, NOT_A_CUDA_STREAM);                   \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEM_TYPE_IGET)
#undef NVSHMEM_TYPE_IGET

#define NVSHMEMX_TYPE_IGET_ON_STREAM(Name, TYPE)                                              \
    void nvshmemx_##Name##_iget_on_stream(TYPE *dest, const TYPE *source, ptrdiff_t dst,      \
                                          ptrdiff_t sst, size_t nelems, int pe,               \
                                          cudaStream_t cstrm) {                               \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                       \
        nvshmemi_prepare_and_post_rma("nvshmem_" #Name "_iget_on_stream", NVSHMEMI_OP_GET, NO_NBI, ASYNC, \
                                      (void *)dest, (void *)source, dst, sst, nelems,         \
                                      sizeof(TYPE), pe, cstrm);                               \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_TYPE_IGET_ON_STREAM)
#undef NVSHMEMX_TYPE_IGET_ON_STREAM

#define NVSHMEM_IGETSIZE(Name, Type)                                                          \
    void nvshmem_iget##Name(void *dest, const void *source, ptrdiff_t dst, ptrdiff_t sst,     \
                            size_t nelems, int pe) {                                          \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                       \
        nvshmemi_prepare_and_post_rma("nvshmem_iget" #Name "", NVSHMEMI_OP_GET, NO_NBI, NO_ASYNC,         \
                                      (void *)dest, (void *)source, dst, sst, nelems,         \
                                      sizeof(Type), pe, NOT_A_CUDA_STREAM);                   \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEM_IGETSIZE)
#undef NVSHMEM_IGETSIZE

#define NVSHMEMX_IGETSIZE_ON_STREAM(Name, Type)                                               \
    void nvshmemx_iget##Name##_on_stream(void *dest, const void *source, ptrdiff_t dst,       \
                                         ptrdiff_t sst, size_t nelems, int pe,                \
                                         cudaStream_t cstrm) {                                \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                       \
        nvshmemi_prepare_and_post_rma("nvshmem_iget" #Name "_on_stream", NVSHMEMI_OP_GET, NO_NBI, ASYNC,  \
                                      (void *)dest, (void *)source, dst, sst, nelems,         \
                                      sizeof(Type), pe, cstrm);                               \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEMX_IGETSIZE_ON_STREAM)
#undef NVSHMEMX_IGETSIZE_ON_STREAM

#define NVSHMEM_TYPE_GET_NBI(type, TYPE)                                                         \
    void nvshmem_##type##_get_nbi(TYPE *dest, const TYPE *source, size_t nelems, int pe) {       \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                          \
        nvshmemi_prepare_and_post_rma(                                                           \
            "nvshmem_" #type "_get_nbi", NVSHMEMI_OP_GET, NBI, NO_ASYNC, (void *)dest, (void *)source,       \
            DEST_STRIDE_CONTIG, SRC_STRIDE_CONTIG, nelems, sizeof(TYPE), pe, NOT_A_CUDA_STREAM); \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEM_TYPE_GET_NBI)
#undef NVSHMEM_TYPE_GET_NBI

#define NVSHMEMX_TYPE_GET_NBI_ON_STREAM(type, TYPE)                                           \
    void nvshmemx_##type##_get_nbi_on_stream(TYPE *dest, const TYPE *source, size_t nelems,   \
                                             int pe, cudaStream_t cstrm) {                    \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                       \
        nvshmemi_prepare_and_post_rma("nvshmem_" #type "_get_nbi_on_stream", NVSHMEMI_OP_GET, NBI, ASYNC, \
                                      (void *)dest, (void *)source, DEST_STRIDE_CONTIG,       \
                                      SRC_STRIDE_CONTIG, nelems, sizeof(TYPE), pe, cstrm);    \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_TYPE_GET_NBI_ON_STREAM)
#undef NVSHMEMX_TYPE_GET_NBI_ON_STREAM

#define NVSHMEM_GETSIZE_NBI(Name, Type)                                                        \
    void nvshmem_get##Name##_nbi(void *dest, const void *source, size_t nelems, int pe) {      \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                        \
        nvshmemi_prepare_and_post_rma("nvshmem_get" #Name "_nbi", NVSHMEMI_OP_GET, NBI, NO_ASYNC,          \
                                      (void *)dest, (void *)source, DEST_STRIDE_CONTIG,        \
                                      SRC_STRIDE_CONTIG, nelems, sizeof(Type), pe, NOT_A_CUDA_STREAM); \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEM_GETSIZE_NBI)
#undef NVSHMEM_GETSIZE_NBI

#define NVSHMEMX_GETSIZE_NBI_ON_STREAM(Name, Type)                                                 \
    void nvshmemx_get##Name##_nbi_on_stream(void *dest, const void *source, size_t nelems, int pe, \
                                            cudaStream_t cstrm) {                                  \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                            \
        nvshmemi_prepare_and_post_rma("nvshmem_get" #Name "_nbi_on_stream", NVSHMEMI_OP_GET, NBI, ASYNC,       \
                                      (void *)dest, (void *)source, DEST_STRIDE_CONTIG,            \
                                      SRC_STRIDE_CONTIG, nelems, sizeof(Type), pe, cstrm);         \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEMX_GETSIZE_NBI_ON_STREAM)
#undef NVSHMEMX_GETSIZE_NBI_ON_STREAM

void nvshmem_getmem_nbi(void *dest, const void *source, size_t bytes, int pe) {
    NVSHMEM_CHECK_STATE_AND_INIT();
    INFO(NVSHMEM_P2P,
         "[%d] untyped get : (local)dest %p, (remote)source %p, %d bytes, remote PE %d",
         nvshmemi_state->mype, dest, source, bytes, pe);
    nvshmemi_prepare_and_post_rma("nvshmem_getmem_nbi", NVSHMEMI_OP_GET, NBI, NO_ASYNC, (void *)dest,
                                  (void *)source, DEST_STRIDE_CONTIG, SRC_STRIDE_CONTIG, bytes, 1,
                                  pe, NOT_A_CUDA_STREAM);
}

void nvshmemx_getmem_nbi_on_stream(void *dest, const void *source, size_t bytes, int pe,
                                   cudaStream_t cstrm) {
    NVSHMEM_CHECK_STATE_AND_INIT();
    INFO(NVSHMEM_P2P,
         "[%d] untyped get : (local)dest %p, (remote)source %p, %d bytes, remote PE %d",
         nvshmemi_state->mype, dest, source, bytes, pe);
    nvshmemi_prepare_and_post_rma("nvshmem_getmem_nbi_on_stream", NVSHMEMI_OP_GET, NBI, ASYNC, (void *)dest,
                                  (void *)source, DEST_STRIDE_CONTIG, SRC_STRIDE_CONTIG, bytes, 1,
                                  pe, cstrm);
}
