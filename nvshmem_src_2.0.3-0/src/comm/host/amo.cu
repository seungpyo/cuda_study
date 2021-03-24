/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#define NVSHMEMI_HOST_ONLY
#include "nvshmem_api.h"
#include "nvshmem_internal.h"
#include "nvshmemx_error.h"
#include "amo_kernel_entrypoints.cuh"

template <typename T>
int nvshmemi_p2p_amo_base(amo_verb_t verb, CUstream custrm, T *targetptr, T *retptr, T *curetptr,
                          T *valptr, T *cmpptr, amo_bytesdesc_t bytesdesc, const void *handle) {
    int status = 0;
    T val = 0, cmp = 0, ret = 0;
    if (verb.is_val) {
        val = *valptr;
        if (verb.is_cmp) {
            cmp = *cmpptr;
        }
    }
    void *args[] = {&targetptr, &curetptr, &val, &cmp};
    status = cudaLaunchKernel(handle, 1, 1, args, 0, custrm);
    if (status) {
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cudaLaunchKernel() failed\n");
    }
    if (verb.is_fetch) {
        status = cuMemcpyDtoHAsync(&ret, (CUdeviceptr)curetptr, bytesdesc.elembytes,
                                   custrm); /*XXX:replace by GDRcopy*/
        if (status) {
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cuMemcpyDtoHAsync() failed\n");
        }
        status = cuStreamSynchronize(custrm);
        if (status) {
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cuStreamSynchronize() failed\n");
        }
        *retptr = ret;
    }
out:
    return status;
}

static int nvshmemi_p2p_amo_bitwise(amo_verb_t verb, CUstream custrm, void *targetptr, void *retptr,
                                    void *curetptr, void *valptr, void *cmpptr,
                                    amo_bytesdesc_t bytesdesc, const void **handles) {
    int status = 0;
    switch (bytesdesc.name_type) {
        case UINT:
            status = nvshmemi_p2p_amo_base<unsigned int>(
                verb, custrm, (unsigned int *)targetptr, (unsigned int *)retptr,
                (unsigned int *)curetptr, (unsigned int *)valptr, (unsigned int *)cmpptr, bytesdesc,
                handles[UINT]);
            break;
        case ULONG:
            status = nvshmemi_p2p_amo_base<unsigned long>(
                verb, custrm, (unsigned long *)targetptr, (unsigned long *)retptr,
                (unsigned long *)curetptr, (unsigned long *)valptr, (unsigned long *)cmpptr,
                bytesdesc, handles[ULONG]);
            break;
        case ULONGLONG:
            status = nvshmemi_p2p_amo_base<unsigned long long>(
                verb, custrm, (unsigned long long *)targetptr, (unsigned long long *)retptr,
                (unsigned long long *)curetptr, (unsigned long long *)valptr,
                (unsigned long long *)cmpptr, bytesdesc, handles[ULONGLONG]);
            break;
        case INT32:
            status = nvshmemi_p2p_amo_base<int32_t>(
                verb, custrm, (int32_t *)targetptr, (int32_t *)retptr, (int32_t *)curetptr,
                (int32_t *)valptr, (int32_t *)cmpptr, bytesdesc, handles[INT32]);
            break;
        case INT64:
            status = nvshmemi_p2p_amo_base<int64_t>(
                verb, custrm, (int64_t *)targetptr, (int64_t *)retptr, (int64_t *)curetptr,
                (int64_t *)valptr, (int64_t *)cmpptr, bytesdesc, handles[INT64]);
            break;
        case UINT32:
            status = nvshmemi_p2p_amo_base<uint32_t>(
                verb, custrm, (uint32_t *)targetptr, (uint32_t *)retptr, (uint32_t *)curetptr,
                (uint32_t *)valptr, (uint32_t *)cmpptr, bytesdesc, handles[UINT32]);
            break;
        case UINT64:
            status = nvshmemi_p2p_amo_base<uint64_t>(
                verb, custrm, (uint64_t *)targetptr, (uint64_t *)retptr, (uint64_t *)curetptr,
                (uint64_t *)valptr, (uint64_t *)cmpptr, bytesdesc, handles[UINT64]);
            break;
        default:
            status = NVSHMEMX_ERROR_INTERNAL;
            fprintf(stderr, "[%d] Invalid AMO type %d\n", nvshmemi_state->mype, bytesdesc.name_type);
    }
    return status;
}

static int nvshmemi_p2p_amo_standard(amo_verb_t verb, CUstream custrm, void *targetptr,
                                     void *retptr, void *curetptr, void *valptr, void *cmpptr,
                                     amo_bytesdesc_t bytesdesc, const void **handles) {
    int status = 0;
    switch (bytesdesc.name_type) {
        case INT:
            status = nvshmemi_p2p_amo_base<int>(verb, custrm, (int *)targetptr, (int *)retptr,
                                                (int *)curetptr, (int *)valptr, (int *)cmpptr,
                                                bytesdesc, handles[INT]);
            break;
        case LONG:
            status = nvshmemi_p2p_amo_base<long>(verb, custrm, (long *)targetptr, (long *)retptr,
                                                 (long *)curetptr, (long *)valptr, (long *)cmpptr,
                                                 bytesdesc, handles[LONG]);
            break;
        case LONGLONG:
            status = nvshmemi_p2p_amo_base<long long>(
                verb, custrm, (long long *)targetptr, (long long *)retptr, (long long *)curetptr,
                (long long *)valptr, (long long *)cmpptr, bytesdesc, handles[LONGLONG]);
            break;
        case SIZE:
            status = nvshmemi_p2p_amo_base<size_t>(
                verb, custrm, (size_t *)targetptr, (size_t *)retptr, (size_t *)curetptr,
                (size_t *)valptr, (size_t *)cmpptr, bytesdesc, handles[SIZE]);
            break;
        case PTRDIFF:
            status = nvshmemi_p2p_amo_base<ptrdiff_t>(
                verb, custrm, (ptrdiff_t *)targetptr, (ptrdiff_t *)retptr, (ptrdiff_t *)curetptr,
                (ptrdiff_t *)valptr, (ptrdiff_t *)cmpptr, bytesdesc, handles[PTRDIFF]);
            break;
        default:
            status = nvshmemi_p2p_amo_bitwise(verb, custrm, targetptr, retptr, curetptr, valptr,
                                              cmpptr, bytesdesc, handles);
    }
    return status;
}

static int nvshmemi_p2p_amo_extended(amo_verb_t verb, CUstream custrm, void *targetptr,
                                     void *retptr, void *curetptr, void *valptr, void *cmpptr,
                                     amo_bytesdesc_t bytesdesc, const void **handles) {
    int status = 0;
    if (bytesdesc.name_type == FLOAT) {
        status = nvshmemi_p2p_amo_base<float>(verb, custrm, (float *)targetptr, (float *)retptr,
                                              (float *)curetptr, (float *)valptr, (float *)cmpptr,
                                              bytesdesc, handles[FLOAT]);
    } else if (bytesdesc.name_type == DOUBLE) {
        status = nvshmemi_p2p_amo_base<double>(verb, custrm, (double *)targetptr, (double *)retptr,
                                               (double *)curetptr, (double *)valptr,
                                               (double *)cmpptr, bytesdesc, handles[DOUBLE]);
    } else {
        nvshmemi_p2p_amo_standard(verb, custrm, targetptr, retptr, curetptr, valptr, cmpptr,
                                  bytesdesc, handles);
    }
    return status;
}

static int nvshmemi_p2p_amo_set(amo_verb_t verb, CUstream custrm, void *targetptr, void *retptr,
                                void *curetptr, void *valptr, void *cmpptr,
                                amo_bytesdesc_t bytesdesc) {
    const void *handles[] = {(void *)AtomicSetKernel<unsigned int>,
                             (void *)AtomicSetKernel<unsigned long, unsigned long long>,
                             (void *)AtomicSetKernel<unsigned long long>,
                             (void *)AtomicSetKernel<int32_t, int>,
                             (void *)AtomicSetKernel<int64_t, unsigned long long int>,
                             (void *)AtomicSetKernel<uint32_t, unsigned int>,
                             (void *)AtomicSetKernel<uint64_t, unsigned long long int>,
                             (void *)AtomicSetKernel<int>,
                             (void *)AtomicSetKernel<long, int>,
                             (void *)AtomicSetKernel<long long, unsigned long long int>,
                             (void *)AtomicSetKernel<size_t, unsigned long long int>,
                             (void *)AtomicSetKernel<ptrdiff_t, unsigned long long int>,
                             (void *)AtomicSetKernel<float, unsigned int>,
                             (void *)AtomicSetKernel<double, unsigned long long int>};
    return nvshmemi_p2p_amo_extended(verb, custrm, targetptr, retptr, curetptr, valptr, cmpptr,
                                     bytesdesc, handles);
}

static int nvshmemi_p2p_amo_inc(amo_verb_t verb, CUstream custrm, void *targetptr, void *retptr,
                                void *curetptr, void *valptr, void *cmpptr,
                                amo_bytesdesc_t bytesdesc) {
    /*XXX not implemented types : long long, ptrdiff_t, int64_t*/
    const void *handles[] = {
        (void *)AtomicIncKernel<unsigned int>,
        (void *)AtomicIncKernel<unsigned long, unsigned long long>,
        (void *)AtomicIncKernel<unsigned long long>,
        (void *)AtomicIncKernel<int32_t, int>,
        0 /*AtomicIncKernel<int64_t>*/,
        (void *)AtomicIncKernel<uint32_t, unsigned int>,
        (void *)AtomicIncKernel<uint64_t, unsigned long long int>,
        (void *)AtomicIncKernel<int>,
        (void *)AtomicIncKernel<long, int>,
        0 /*AtomicIncKernel<long long>*/,
        (void *)AtomicIncKernel<size_t, unsigned long long int>,
        0 /*AtomicIncKernel<ptrdiff_t>*/
    };
    return nvshmemi_p2p_amo_standard(verb, custrm, targetptr, retptr, curetptr, valptr, cmpptr,
                                     bytesdesc, handles);
}

static int nvshmemi_p2p_amo_add(amo_verb_t verb, CUstream custrm, void *targetptr, void *retptr,
                                void *curetptr, void *valptr, void *cmpptr,
                                amo_bytesdesc_t bytesdesc) {
    /*XXX not implemented types : long long, ptrdiff_t, int64_t*/
    const void *handles[] = {
        (void *)AtomicAddKernel<unsigned int>,
        (void *)AtomicAddKernel<unsigned long, unsigned long long>,
        (void *)AtomicAddKernel<unsigned long long>,
        (void *)AtomicAddKernel<int32_t, int>,
        0 /*AtomicAddKernel<int64_t>*/,
        (void *)AtomicAddKernel<uint32_t, unsigned int>,
        (void *)AtomicAddKernel<uint64_t, unsigned long long int>,
        (void *)AtomicAddKernel<int>,
        (void *)AtomicAddKernel<long, int>,
        0 /*AtomicAddKernel<long long>*/,
        (void *)AtomicAddKernel<size_t, unsigned long long int>,
        0 /*AtomicAddKernel<ptrdiff_t>*/
    };
    return nvshmemi_p2p_amo_standard(verb, custrm, targetptr, retptr, curetptr, valptr, cmpptr,
                                     bytesdesc, handles);
}

static int nvshmemi_p2p_amo_and(amo_verb_t verb, CUstream custrm, void *targetptr, void *retptr,
                                void *curetptr, void *valptr, void *cmpptr,
                                amo_bytesdesc_t bytesdesc) {
    const void *handles[] = {(void *)AtomicAndKernel<unsigned int>,
                             (void *)AtomicAndKernel<unsigned long, unsigned long long>,
                             (void *)AtomicAndKernel<unsigned long long>,
                             (void *)AtomicAndKernel<int32_t, int>,
                             (void *)AtomicAndKernel<int64_t, unsigned long long int>,
                             (void *)AtomicAndKernel<uint32_t, unsigned int>,
                             (void *)AtomicAndKernel<uint64_t, unsigned long long int>};
    return nvshmemi_p2p_amo_bitwise(verb, custrm, targetptr, retptr, curetptr, valptr, cmpptr,
                                    bytesdesc, handles);
}

static int nvshmemi_p2p_amo_or(amo_verb_t verb, CUstream custrm, void *targetptr, void *retptr,
                               void *curetptr, void *valptr, void *cmpptr,
                               amo_bytesdesc_t bytesdesc) {
    const void *handles[] = {(void *)AtomicOrKernel<unsigned int>,
                             (void *)AtomicOrKernel<unsigned long, unsigned long long>,
                             (void *)AtomicOrKernel<unsigned long long>,
                             (void *)AtomicOrKernel<int32_t, int>,
                             (void *)AtomicOrKernel<int64_t, unsigned long long int>,
                             (void *)AtomicOrKernel<uint32_t, unsigned int>,
                             (void *)AtomicOrKernel<uint64_t, unsigned long long int>};
    return nvshmemi_p2p_amo_bitwise(verb, custrm, targetptr, retptr, curetptr, valptr, cmpptr,
                                    bytesdesc, handles);
}

static int nvshmemi_p2p_amo_xor(amo_verb_t verb, CUstream custrm, void *targetptr, void *retptr,
                                void *curetptr, void *valptr, void *cmpptr,
                                amo_bytesdesc_t bytesdesc) {
    const void *handles[] = {(void *)AtomicXorKernel<unsigned int>,
                             (void *)AtomicXorKernel<unsigned long, unsigned long long>,
                             (void *)AtomicXorKernel<unsigned long long>,
                             (void *)AtomicXorKernel<int32_t, int>,
                             (void *)AtomicXorKernel<int64_t, unsigned long long int>,
                             (void *)AtomicXorKernel<uint32_t, unsigned int>,
                             (void *)AtomicXorKernel<uint64_t, unsigned long long int>};
    return nvshmemi_p2p_amo_bitwise(verb, custrm, targetptr, retptr, curetptr, valptr, cmpptr,
                                    bytesdesc, handles);
}

static int nvshmemi_p2p_amo_fetch(amo_verb_t verb, CUstream custrm, void *targetptr, void *retptr,
                                  void *curetptr, void *valptr, void *cmpptr,
                                  amo_bytesdesc_t bytesdesc) {
    const void *handles[] = {(void *)AtomicFetchKernel<unsigned int>,
                             (void *)AtomicFetchKernel<unsigned long, unsigned long long>,
                             (void *)AtomicFetchKernel<unsigned long long>,
                             (void *)AtomicFetchKernel<int32_t, int>,
                             (void *)AtomicFetchKernel<int64_t, unsigned long long int>,
                             (void *)AtomicFetchKernel<uint32_t, unsigned int>,
                             (void *)AtomicFetchKernel<uint64_t, unsigned long long int>,
                             (void *)AtomicFetchKernel<int>,
                             (void *)AtomicFetchKernel<long, int>,
                             (void *)AtomicFetchKernel<long long, unsigned long long int>,
                             (void *)AtomicFetchKernel<size_t, unsigned long long int>,
                             (void *)AtomicFetchKernel<ptrdiff_t, unsigned long long int>,
                             (void *)AtomicFetchKernel<float, unsigned int>,
                             (void *)AtomicFetchKernel<double, unsigned long long int>};
    return nvshmemi_p2p_amo_extended(verb, custrm, targetptr, retptr, curetptr, valptr, cmpptr,
                                     bytesdesc, handles);
}

static int nvshmemi_p2p_amo_fetch_inc(amo_verb_t verb, CUstream custrm, void *targetptr, void *retptr,
                                 void *curetptr, void *valptr, void *cmpptr,
                                 amo_bytesdesc_t bytesdesc) {
    /*XXX not implemented types : long long, ptrdiff_t, int64_t*/
    const void *handles[] = {
        (void *)AtomicFincKernel<unsigned int>,
        (void *)AtomicFincKernel<unsigned long, unsigned long long>,
        (void *)AtomicFincKernel<unsigned long long>,
        (void *)AtomicFincKernel<int32_t, int>,
        0 /*AtomicFincKernel<int64_t>*/,
        (void *)AtomicFincKernel<uint32_t, unsigned int>,
        (void *)AtomicFincKernel<uint64_t, unsigned long long int>,
        (void *)AtomicFincKernel<int>,
        (void *)AtomicFincKernel<long, int>,
        0 /*AtomicFincKernel<long long>*/,
        (void *)AtomicFincKernel<size_t, unsigned long long int>,
        0 /*AtomicFincKernel<ptrdiff_t>*/
    };
    return nvshmemi_p2p_amo_standard(verb, custrm, targetptr, retptr, curetptr, valptr, cmpptr,
                                     bytesdesc, handles);
}

static int nvshmemi_p2p_amo_fetch_add(amo_verb_t verb, CUstream custrm, void *targetptr, void *retptr,
                                 void *curetptr, void *valptr, void *cmpptr,
                                 amo_bytesdesc_t bytesdesc) {
    /*XXX not implemented types : long long, ptrdiff_t, int64_t*/
    const void *handles[] = {
        (void *)AtomicFaddKernel<unsigned int>,
        (void *)AtomicFaddKernel<unsigned long, unsigned long long>,
        (void *)AtomicFaddKernel<unsigned long long>,
        (void *)AtomicFaddKernel<int32_t, int>,
        0 /*AtomicFaddKernel<int64_t>*/,
        (void *)AtomicFaddKernel<uint32_t, unsigned int>,
        (void *)AtomicFaddKernel<uint64_t, unsigned long long int>,
        (void *)AtomicFaddKernel<int>,
        (void *)AtomicFaddKernel<long, int>,
        0 /*AtomicFaddKernel<long long>*/,
        (void *)AtomicFaddKernel<size_t, unsigned long long int>,
        0 /*AtomicFaddKernel<ptrdiff_t>*/
    };
    return nvshmemi_p2p_amo_standard(verb, custrm, targetptr, retptr, curetptr, valptr, cmpptr,
                                     bytesdesc, handles);
}

static int nvshmemi_p2p_amo_swap(amo_verb_t verb, CUstream custrm, void *targetptr, void *retptr,
                                 void *curetptr, void *valptr, void *cmpptr,
                                 amo_bytesdesc_t bytesdesc) {
    const void *handles[] = {(void *)AtomicSwapKernel<unsigned int>,
                             (void *)AtomicSwapKernel<unsigned long, unsigned long long>,
                             (void *)AtomicSwapKernel<unsigned long long>,
                             (void *)AtomicSwapKernel<int32_t, int>,
                             (void *)AtomicSwapKernel<int64_t, unsigned long long int>,
                             (void *)AtomicSwapKernel<uint32_t, unsigned int>,
                             (void *)AtomicSwapKernel<uint64_t, unsigned long long int>,
                             (void *)AtomicSwapKernel<int>,
                             (void *)AtomicSwapKernel<long, int>,
                             (void *)AtomicSwapKernel<long long, unsigned long long int>,
                             (void *)AtomicSwapKernel<size_t, unsigned long long int>,
                             (void *)AtomicSwapKernel<ptrdiff_t, unsigned long long int>,
                             (void *)AtomicSwapKernel<float, unsigned int>,
                             (void *)AtomicSwapKernel<double, unsigned long long int>};
    return nvshmemi_p2p_amo_extended(verb, custrm, targetptr, retptr, curetptr, valptr, cmpptr,
                                     bytesdesc, handles);
}

static int nvshmemi_p2p_amo_compare_swap(amo_verb_t verb, CUstream custrm, void *targetptr, void *retptr,
                                  void *curetptr, void *valptr, void *cmpptr,
                                  amo_bytesdesc_t bytesdesc) {
    const void *handles[] = {(void *)AtomicCswapKernel<unsigned int>,
                             (void *)AtomicCswapKernel<unsigned long, unsigned long long>,
                             (void *)AtomicCswapKernel<unsigned long long>,
                             (void *)AtomicCswapKernel<int32_t, int>,
                             (void *)AtomicCswapKernel<int64_t, unsigned long long int>,
                             (void *)AtomicCswapKernel<uint32_t, unsigned int>,
                             (void *)AtomicCswapKernel<uint64_t, unsigned long long int>,
                             (void *)AtomicCswapKernel<int>,
                             (void *)AtomicCswapKernel<long, int>,
                             (void *)AtomicCswapKernel<long long, unsigned long long int>,
                             (void *)AtomicCswapKernel<size_t, unsigned long long int>,
                             (void *)AtomicCswapKernel<ptrdiff_t, unsigned long long int>};
    return nvshmemi_p2p_amo_standard(verb, custrm, targetptr, retptr, curetptr, valptr, cmpptr,
                                     bytesdesc, handles);
}

static int nvshmemi_p2p_amo_fetch_and(amo_verb_t verb, CUstream custrm, void *targetptr, void *retptr,
                                 void *curetptr, void *valptr, void *cmpptr,
                                 amo_bytesdesc_t bytesdesc) {
    const void *handles[] = {(void *)AtomicFandKernel<unsigned int>,
                             (void *)AtomicFandKernel<unsigned long, unsigned long long>,
                             (void *)AtomicFandKernel<unsigned long long>,
                             (void *)AtomicFandKernel<int32_t, int>,
                             (void *)AtomicFandKernel<int64_t, unsigned long long int>,
                             (void *)AtomicFandKernel<uint32_t, unsigned int>,
                             (void *)AtomicFandKernel<uint64_t, unsigned long long int>};
    return nvshmemi_p2p_amo_bitwise(verb, custrm, targetptr, retptr, curetptr, valptr, cmpptr,
                                    bytesdesc, handles);
}

static int nvshmemi_p2p_amo_fetch_or(amo_verb_t verb, CUstream custrm, void *targetptr, void *retptr,
                                void *curetptr, void *valptr, void *cmpptr,
                                amo_bytesdesc_t bytesdesc) {
    const void *handles[] = {(void *)AtomicForKernel<unsigned int>,
                             (void *)AtomicForKernel<unsigned long, unsigned long long>,
                             (void *)AtomicForKernel<unsigned long long>,
                             (void *)AtomicForKernel<int32_t, int>,
                             (void *)AtomicForKernel<int64_t, unsigned long long int>,
                             (void *)AtomicForKernel<uint32_t, unsigned int>,
                             (void *)AtomicForKernel<uint64_t, unsigned long long int>};
    return nvshmemi_p2p_amo_bitwise(verb, custrm, targetptr, retptr, curetptr, valptr, cmpptr,
                                    bytesdesc, handles);
}

static int nvshmemi_p2p_amo_fetch_xor(amo_verb_t verb, CUstream custrm, void *targetptr, void *retptr,
                                 void *curetptr, void *valptr, void *cmpptr,
                                 amo_bytesdesc_t bytesdesc) {
    const void *handles[] = {(void *)AtomicFxorKernel<unsigned int>,
                             (void *)AtomicFxorKernel<unsigned long, unsigned long long>,
                             (void *)AtomicFxorKernel<unsigned long long>,
                             (void *)AtomicFxorKernel<int32_t, int>,
                             (void *)AtomicFxorKernel<int64_t, unsigned long long int>,
                             (void *)AtomicFxorKernel<uint32_t, unsigned int>,
                             (void *)AtomicFxorKernel<uint64_t, unsigned long long int>};
    return nvshmemi_p2p_amo_bitwise(verb, custrm, targetptr, retptr, curetptr, valptr, cmpptr,
                                    bytesdesc, handles);
}

static int nvshmemi_p2p_amo(CUstream custrm, CUevent cuev, void *curetptr, amo_verb_t verb,
                            amo_memdesc_t target, amo_bytesdesc_t bytesdesc) {
    int status = 0;
    switch (verb.desc) {
        /*ret NULL*/
        case NVSHMEMI_AMO_SET:
            status = nvshmemi_p2p_amo_set(verb, custrm, target.ptr, target.retptr, curetptr,
                                          target.valptr, target.cmpptr, bytesdesc); /*cmp NULL*/
            break;
        case NVSHMEMI_AMO_INC:
            status =
                nvshmemi_p2p_amo_inc(verb, custrm, target.ptr, target.retptr, curetptr,
                                     target.valptr, target.cmpptr, bytesdesc); /*val, cmp NULL*/
            break;
        case NVSHMEMI_AMO_ADD:
            status = nvshmemi_p2p_amo_add(verb, custrm, target.ptr, target.retptr, curetptr,
                                          target.valptr, target.cmpptr, bytesdesc); /*cmp NULL*/
            break;
        case NVSHMEMI_AMO_AND:
            status = nvshmemi_p2p_amo_and(verb, custrm, target.ptr, target.retptr, curetptr,
                                          target.valptr, target.cmpptr, bytesdesc); /*cmp NULL*/
            break;
        case NVSHMEMI_AMO_OR:
            status = nvshmemi_p2p_amo_or(verb, custrm, target.ptr, target.retptr, curetptr,
                                         target.valptr, target.cmpptr, bytesdesc); /*cmp NULL*/
            break;
        case NVSHMEMI_AMO_XOR:
            status = nvshmemi_p2p_amo_xor(verb, custrm, target.ptr, target.retptr, curetptr,
                                          target.valptr, target.cmpptr, bytesdesc); /*cmp NULL*/
            break;
        /*ret !NULL*/
        case NVSHMEMI_AMO_FETCH:
            status =
                nvshmemi_p2p_amo_fetch(verb, custrm, target.ptr, target.retptr, curetptr,
                                       target.valptr, target.cmpptr, bytesdesc); /*val, cmp NULL*/
            break;
        case NVSHMEMI_AMO_FETCH_INC:
            status =
                nvshmemi_p2p_amo_fetch_inc(verb, custrm, target.ptr, target.retptr, curetptr,
                                      target.valptr, target.cmpptr, bytesdesc); /*val, cmp NULL*/
            break;
        case NVSHMEMI_AMO_FETCH_ADD:
            status = nvshmemi_p2p_amo_fetch_add(verb, custrm, target.ptr, target.retptr, curetptr,
                                           target.valptr, target.cmpptr, bytesdesc); /*cmp NULL*/
            break;
        case NVSHMEMI_AMO_SWAP:
            status = nvshmemi_p2p_amo_swap(verb, custrm, target.ptr, target.retptr, curetptr,
                                           target.valptr, target.cmpptr, bytesdesc); /*cmp NULL*/
            break;
        case NVSHMEMI_AMO_COMPARE_SWAP:
            status = nvshmemi_p2p_amo_compare_swap(verb, custrm, target.ptr, target.retptr, curetptr,
                                            target.valptr, target.cmpptr, bytesdesc);
            break;
        case NVSHMEMI_AMO_FETCH_AND:
            status = nvshmemi_p2p_amo_fetch_and(verb, custrm, target.ptr, target.retptr, curetptr,
                                           target.valptr, target.cmpptr, bytesdesc); /*cmp NULL*/
            break;
        case NVSHMEMI_AMO_FETCH_OR:
            status = nvshmemi_p2p_amo_fetch_or(verb, custrm, target.ptr, target.retptr, curetptr,
                                          target.valptr, target.cmpptr, bytesdesc); /*cmp NULL*/
            break;
        case NVSHMEMI_AMO_FETCH_XOR:
            status = nvshmemi_p2p_amo_fetch_xor(verb, custrm, target.ptr, target.retptr, curetptr,
                                           target.valptr, target.cmpptr, bytesdesc); /*cmp NULL*/
            break;
    }
    return status;
}

static void nvshmemi_prepare_and_post_amo(nvshmemi_amo_t desc, void *targetptr, void *retptr,
                                          void *valptr, void *cmpptr, size_t elembytes, int pe,
                                          int nameoftype, const char *apiname) {
    int status = 0;
    amo_verb_t verb;
    amo_memdesc_t target;
    amo_bytesdesc_t bytesdesc;
    verb.desc = desc;
    switch (desc) {
        case NVSHMEMI_AMO_INC:
            verb.is_val = 0;
            verb.is_cmp = 0;
            verb.is_fetch = 0;
            break;
        case NVSHMEMI_AMO_SET:
        case NVSHMEMI_AMO_ADD:
        case NVSHMEMI_AMO_AND:
        case NVSHMEMI_AMO_OR:
        case NVSHMEMI_AMO_XOR:
            verb.is_val = 1;
            verb.is_cmp = 0;
            verb.is_fetch = 0;
            break;
        case NVSHMEMI_AMO_FETCH:
        case NVSHMEMI_AMO_FETCH_INC:
            verb.is_val = 0;
            verb.is_cmp = 0;
            verb.is_fetch = 1;
            break;
        case NVSHMEMI_AMO_SWAP:
        case NVSHMEMI_AMO_FETCH_ADD:
        case NVSHMEMI_AMO_FETCH_AND:
        case NVSHMEMI_AMO_FETCH_OR:
        case NVSHMEMI_AMO_FETCH_XOR:
            verb.is_val = 1;
            verb.is_cmp = 0;
            verb.is_fetch = 1;
            break;
        case NVSHMEMI_AMO_COMPARE_SWAP:
            verb.is_val = 1;
            verb.is_cmp = 1;
            verb.is_fetch = 1;
            break;
    }
    bytesdesc.elembytes = elembytes;
    bytesdesc.name_type = nameoftype;
    volatile void *targetptr_actual =
        (volatile void *)((char *)(nvshmemi_state->peer_heap_base[pe]) +
                          ((char *)targetptr - (char *)(nvshmemi_state->heap_base)));
    target.ptr = (void *)targetptr_actual;
    target.retptr = retptr;
    target.valptr = valptr;
    target.cmpptr = cmpptr;
    void *curetptr = (void *)nvshmemi_state->curets[pe];
    if (targetptr_actual) {
        CUstream custrm = nvshmemi_state->custreams[pe % MAX_PEER_STREAMS];
        CUevent cuev = nvshmemi_state->cuevents[pe % MAX_PEER_STREAMS];
        if (nvshmemi_state
                ->p2p_attrib_native_atomic_support[pe]) { /*AMO not supported for P2P over PCIE*/
            status = nvshmemi_p2p_amo(custrm, cuev, curetptr, verb, target,
                                      bytesdesc); /*bypass transport for P2P*/
        } else {
            ERROR_PRINT("[%d] %s to PE %d does not have P2P path\n", nvshmemi_state->mype, apiname,
                        pe);
        }
    } else {
        int t = nvshmemi_state->selected_transport_for_amo[pe];
        if (t < 0) {
            ERROR_EXIT("[%d] amo not supported on transport to pe: %d \n", nvshmemi_state->mype, pe);
        }

        nvshmemt_ep_t ep;
        int tcount = nvshmemi_state->transport_count;
        struct nvshmem_transport *tcurr = nvshmemi_state->transports[t];
        int ep_offset = pe * tcurr->ep_count;
        ep = tcurr->ep[ep_offset];
        nvshmem_mem_handle_t *handles = nvshmemi_state->handles;
        target.handle = handles[pe * tcount + t];
        status = nvshmemi_state->amo[pe](ep, curetptr, verb, target, bytesdesc);
    }
    if (status) {
        ERROR_EXIT("[%d] aborting due to error in %s \n", nvshmemi_state->mype, apiname);
    }
}

#define NVSHMEM_TYPE_INC(Name, NameIdx, TYPE)                                                  \
    void nvshmem_##Name##_atomic_inc(TYPE *target, int pe) {                                          \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                        \
        nvshmemi_prepare_and_post_amo(NVSHMEMI_AMO_INC, (void *)target, 0, 0, 0, sizeof(TYPE), pe, NameIdx, \
                                      "nvshmem_" #Name "_atomic_inc");                                \
    }

#define NVSHMEM_TYPE_INC_NOT_IMPLEMENTED(Name, NameIdx, TYPE)                             \
    void nvshmem_##Name##_atomic_inc(TYPE *target, int pe) {                                     \
        ERROR_PRINT("[%d] nvshmem_" #Name "_atomic_inc() not implemented", nvshmemi_state->mype); \
    }

NVSHMEM_TYPE_INC(uint, UINT, unsigned int)
NVSHMEM_TYPE_INC(ulong, ULONG, unsigned long)
NVSHMEM_TYPE_INC(ulonglong, ULONGLONG, unsigned long long)
NVSHMEM_TYPE_INC(int32, INT32, int32_t)
NVSHMEM_TYPE_INC(uint32, UINT32, uint32_t)
NVSHMEM_TYPE_INC_NOT_IMPLEMENTED(int64, INT64, int64_t) /*XXX:not implemented*/
NVSHMEM_TYPE_INC(uint64, UINT64, uint64_t)
NVSHMEM_TYPE_INC(int, INT, int)
NVSHMEM_TYPE_INC(long, LONG, long)
NVSHMEM_TYPE_INC_NOT_IMPLEMENTED(longlong, LONGLONG, long long) /*XXX:not implemented*/
NVSHMEM_TYPE_INC(size, SIZE, size_t)
NVSHMEM_TYPE_INC_NOT_IMPLEMENTED(ptrdiff, PTRDIFF, ptrdiff_t) /*XXX:not implemented*/

#define NVSHMEM_TYPE_ADD(Name, NameIdx, TYPE)                                              \
    void nvshmem_##Name##_atomic_add(TYPE *target, TYPE value, int pe) {                          \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                    \
        nvshmemi_prepare_and_post_amo(NVSHMEMI_AMO_ADD, (void *)target, 0, &value, 0, sizeof(TYPE), pe, \
                                      NameIdx, "nvshmem_" #Name "_atomic_add");                   \
    }

#define NVSHMEM_TYPE_ADD_NOT_IMPLEMENTED(Name, NameIdx, TYPE)                             \
    void nvshmem_##Name##_atomic_add(TYPE *target, TYPE value, int pe) {                         \
        ERROR_PRINT("[%d] nvshmem_" #Name "_atomic_add() not implemented", nvshmemi_state->mype); \
    }

NVSHMEM_TYPE_ADD(uint, UINT, unsigned int)
NVSHMEM_TYPE_ADD(ulong, ULONG, unsigned long)
NVSHMEM_TYPE_ADD(ulonglong, ULONGLONG, unsigned long long)
NVSHMEM_TYPE_ADD(int32, INT32, int32_t)
NVSHMEM_TYPE_ADD(uint32, UINT32, uint32_t)
NVSHMEM_TYPE_ADD_NOT_IMPLEMENTED(int64, INT64, int64_t) /*XXX:not implemented*/
NVSHMEM_TYPE_ADD(uint64, UINT64, uint64_t)
NVSHMEM_TYPE_ADD(int, INT, int)
NVSHMEM_TYPE_ADD(long, LONG, long)
NVSHMEM_TYPE_ADD_NOT_IMPLEMENTED(longlong, LONGLONG, long long) /*XXX:not implemented*/
NVSHMEM_TYPE_ADD(size, SIZE, size_t)
NVSHMEM_TYPE_ADD_NOT_IMPLEMENTED(ptrdiff, PTRDIFF, ptrdiff_t) /*XXX:not implemented*/

#define NVSHMEM_TYPE_SET(Name, NameIdx, TYPE)                                              \
    void nvshmem_##Name##_atomic_set(TYPE *target, TYPE value, int pe) {                          \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                    \
        nvshmemi_prepare_and_post_amo(NVSHMEMI_AMO_SET, (void *)target, 0, &value, 0, sizeof(TYPE), pe, \
                                      NameIdx, "nvshmem_" #Name "_atomic_set");                   \
    }
NVSHMEM_TYPE_SET(uint, UINT, unsigned int)
NVSHMEM_TYPE_SET(ulong, ULONG, unsigned long)
NVSHMEM_TYPE_SET(ulonglong, ULONGLONG, unsigned long long)
NVSHMEM_TYPE_SET(int32, INT32, int32_t)
NVSHMEM_TYPE_SET(uint32, UINT32, uint32_t)
NVSHMEM_TYPE_SET(int64, INT64, int64_t)
NVSHMEM_TYPE_SET(uint64, UINT64, uint64_t)
NVSHMEM_TYPE_SET(int, INT, int)
NVSHMEM_TYPE_SET(long, LONG, long)
NVSHMEM_TYPE_SET(longlong, LONGLONG, long long)
NVSHMEM_TYPE_SET(size, SIZE, size_t)
NVSHMEM_TYPE_SET(ptrdiff, PTRDIFF, ptrdiff_t)
NVSHMEM_TYPE_SET(float, FLOAT, float)
NVSHMEM_TYPE_SET(double, DOUBLE, double)

#define NVSHMEM_TYPE_AND(Name, NameIdx, TYPE)                                              \
    void nvshmem_##Name##_atomic_and(TYPE *target, TYPE value, int pe) {                   \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                    \
        nvshmemi_prepare_and_post_amo(NVSHMEMI_AMO_AND, (void *)target, 0, &value, 0, sizeof(TYPE), pe, \
                                      NameIdx, "nvshmem_" #Name "_atomic_and");                   \
    }
NVSHMEM_TYPE_AND(uint, UINT, unsigned int)
NVSHMEM_TYPE_AND(ulong, ULONG, unsigned long)
NVSHMEM_TYPE_AND(ulonglong, ULONGLONG, unsigned long long)
NVSHMEM_TYPE_AND(int32, INT32, int32_t)
NVSHMEM_TYPE_AND(uint32, UINT32, uint32_t)
NVSHMEM_TYPE_AND(int64, INT64, int64_t)
NVSHMEM_TYPE_AND(uint64, UINT64, uint64_t)

#define NVSHMEM_TYPE_OR(Name, NameIdx, TYPE)                                                       \
    void nvshmem_##Name##_atomic_or(TYPE *target, TYPE value, int pe) {                            \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                            \
        nvshmemi_prepare_and_post_amo(NVSHMEMI_AMO_OR, (void *)target, 0, &value, 0, sizeof(TYPE), pe, NameIdx, \
                                      "nvshmem_" #Name "_atomic_or");                                     \
    }
NVSHMEM_TYPE_OR(uint, UINT, unsigned int)
NVSHMEM_TYPE_OR(ulong, ULONG, unsigned long)
NVSHMEM_TYPE_OR(ulonglong, ULONGLONG, unsigned long long)
NVSHMEM_TYPE_OR(int32, INT32, int32_t)
NVSHMEM_TYPE_OR(uint32, UINT32, uint32_t)
NVSHMEM_TYPE_OR(int64, INT64, int64_t)
NVSHMEM_TYPE_OR(uint64, UINT64, uint64_t)

#define NVSHMEM_TYPE_XOR(Name, NameIdx, TYPE)                                              \
    void nvshmem_##Name##_atomic_xor(TYPE *target, TYPE value, int pe) {                   \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                    \
        nvshmemi_prepare_and_post_amo(NVSHMEMI_AMO_XOR, (void *)target, 0, &value, 0, sizeof(TYPE), pe, \
                                      NameIdx, "nvshmem_" #Name "_atomic_xor");                   \
    }
NVSHMEM_TYPE_XOR(uint, UINT, unsigned int)
NVSHMEM_TYPE_XOR(ulong, ULONG, unsigned long)
NVSHMEM_TYPE_XOR(ulonglong, ULONGLONG, unsigned long long)
NVSHMEM_TYPE_XOR(int32, INT32, int32_t)
NVSHMEM_TYPE_XOR(uint32, UINT32, uint32_t)
NVSHMEM_TYPE_XOR(int64, INT64, int64_t)
NVSHMEM_TYPE_XOR(uint64, UINT64, uint64_t)

#define NVSHMEM_TYPE_FETCH(Name, NameIdx, TYPE)                                                    \
    TYPE nvshmem_##Name##_atomic_fetch(TYPE *target, int pe) {                                            \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                            \
        TYPE ret;                                                                                  \
        nvshmemi_prepare_and_post_amo(NVSHMEMI_AMO_FETCH, (void *)target, (void *)&ret, 0, 0, sizeof(TYPE), pe, \
                                      NameIdx, "nvshmem_" #Name "_atomic_fetch");                         \
        return ret;                                                                                \
    }
NVSHMEM_TYPE_FETCH(uint, UINT, unsigned int)
NVSHMEM_TYPE_FETCH(ulong, ULONG, unsigned long)
NVSHMEM_TYPE_FETCH(ulonglong, ULONGLONG, unsigned long long)
NVSHMEM_TYPE_FETCH(int32, INT32, int32_t)
NVSHMEM_TYPE_FETCH(uint32, UINT32, uint32_t)
NVSHMEM_TYPE_FETCH(int64, INT64, int64_t)
NVSHMEM_TYPE_FETCH(uint64, UINT64, uint64_t)
NVSHMEM_TYPE_FETCH(int, INT, int)
NVSHMEM_TYPE_FETCH(long, LONG, long)
NVSHMEM_TYPE_FETCH(longlong, LONGLONG, long long)
NVSHMEM_TYPE_FETCH(size, SIZE, size_t)
NVSHMEM_TYPE_FETCH(ptrdiff, PTRDIFF, ptrdiff_t)
NVSHMEM_TYPE_FETCH(float, FLOAT, float)
NVSHMEM_TYPE_FETCH(double, DOUBLE, double)

#define NVSHMEM_TYPE_FETCH_INC(Name, NameIdx, TYPE)                                                     \
    TYPE nvshmem_##Name##_atomic_fetch_inc(TYPE *target, int pe) {                                             \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                            \
        TYPE ret;                                                                                  \
        nvshmemi_prepare_and_post_amo(NVSHMEMI_AMO_FETCH_INC, (void *)target, (void *)&ret, 0, 0, sizeof(TYPE), \
                                      pe, NameIdx, "nvshmem_" #Name "_atomic_fetch_inc");                      \
        return ret;                                                                                \
    }

#define NVSHMEM_TYPE_FETCH_INC_NOT_IMPLEMENTED(Name, NameIdx, TYPE)                             \
    TYPE nvshmem_##Name##_atomic_fetch_inc(TYPE *target, int pe) {                                     \
        ERROR_PRINT("[%d] nvshmem_" #Name "_atomic_fetch_inc() not implemented", nvshmemi_state->mype); \
        return 0;                                                                          \
    }

NVSHMEM_TYPE_FETCH_INC(uint, UINT, unsigned int)
NVSHMEM_TYPE_FETCH_INC(ulong, ULONG, unsigned long)
NVSHMEM_TYPE_FETCH_INC(ulonglong, ULONGLONG, unsigned long long)
NVSHMEM_TYPE_FETCH_INC(int32, INT32, int32_t)
NVSHMEM_TYPE_FETCH_INC(uint32, UINT32, uint32_t)
NVSHMEM_TYPE_FETCH_INC_NOT_IMPLEMENTED(int64, INT64, int64_t) /*XXX:not implemented*/
NVSHMEM_TYPE_FETCH_INC(uint64, UINT64, uint64_t)
NVSHMEM_TYPE_FETCH_INC(int, INT, int)
NVSHMEM_TYPE_FETCH_INC(long, LONG, long)
NVSHMEM_TYPE_FETCH_INC_NOT_IMPLEMENTED(longlong, LONGLONG, long long) /*XXX:not implemented*/
NVSHMEM_TYPE_FETCH_INC(size, SIZE, size_t)
NVSHMEM_TYPE_FETCH_INC_NOT_IMPLEMENTED(ptrdiff, PTRDIFF, ptrdiff_t) /*XXX:not implemented*/

#define NVSHMEM_TYPE_FETCH_ADD(Name, NameIdx, TYPE)                                              \
    TYPE nvshmem_##Name##_atomic_fetch_add(TYPE *target, TYPE value, int pe) {                          \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                     \
        TYPE ret;                                                                           \
        nvshmemi_prepare_and_post_amo(NVSHMEMI_AMO_FETCH_ADD, (void *)target, (void *)&ret, &value, 0,   \
                                      sizeof(TYPE), pe, NameIdx, "nvshmem_" #Name "_atomic_fetch_add"); \
        return ret;                                                                         \
    }

#define NVSHMEM_TYPE_FETCH_ADD_NOT_IMPLEMENTED(Name, NameIdx, TYPE)                             \
    TYPE nvshmem_##Name##_atomic_fetch_add(TYPE *target, TYPE value, int pe) {                         \
        ERROR_PRINT("[%d] nvshmem_" #Name "_atomic_fadd() not implemented", nvshmemi_state->mype); \
        return 0;                                                                          \
    }

NVSHMEM_TYPE_FETCH_ADD(uint, UINT, unsigned int)
NVSHMEM_TYPE_FETCH_ADD(ulong, ULONG, unsigned long)
NVSHMEM_TYPE_FETCH_ADD(ulonglong, ULONGLONG, unsigned long long)
NVSHMEM_TYPE_FETCH_ADD(int32, INT32, int32_t)
NVSHMEM_TYPE_FETCH_ADD(uint32, UINT32, uint32_t)
NVSHMEM_TYPE_FETCH_ADD_NOT_IMPLEMENTED(int64, INT64, int64_t) /*XXX:not implemented*/
NVSHMEM_TYPE_FETCH_ADD(uint64, UINT64, uint64_t)
NVSHMEM_TYPE_FETCH_ADD(int, INT, int)
NVSHMEM_TYPE_FETCH_ADD(long, LONG, long)
NVSHMEM_TYPE_FETCH_ADD_NOT_IMPLEMENTED(longlong, LONGLONG, long long) /*XXX:not implemented*/
NVSHMEM_TYPE_FETCH_ADD(size, SIZE, size_t)
NVSHMEM_TYPE_FETCH_ADD_NOT_IMPLEMENTED(ptrdiff, PTRDIFF, ptrdiff_t) /*XXX:not implemented*/

#define NVSHMEM_TYPE_SWAP(Name, NameIdx, TYPE)                                                     \
    TYPE nvshmem_##Name##_atomic_swap(TYPE *target, TYPE value, int pe) {                                 \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                            \
        TYPE ret;                                                                                  \
        nvshmemi_prepare_and_post_amo(NVSHMEMI_AMO_SWAP, (void *)target, (void *)&ret, &value, 0, sizeof(TYPE), \
                                      pe, NameIdx, "nvshmem_" #Name "_atomic_swap");                      \
        return ret;                                                                                \
    }
NVSHMEM_TYPE_SWAP(uint, UINT, unsigned int)
NVSHMEM_TYPE_SWAP(ulong, ULONG, unsigned long)
NVSHMEM_TYPE_SWAP(ulonglong, ULONGLONG, unsigned long long)
NVSHMEM_TYPE_SWAP(int32, INT32, int32_t)
NVSHMEM_TYPE_SWAP(uint32, UINT32, uint32_t)
NVSHMEM_TYPE_SWAP(int64, INT64, int64_t)
NVSHMEM_TYPE_SWAP(uint64, UINT64, uint64_t)
NVSHMEM_TYPE_SWAP(int, INT, int)
NVSHMEM_TYPE_SWAP(long, LONG, long)
NVSHMEM_TYPE_SWAP(longlong, LONGLONG, long long)
NVSHMEM_TYPE_SWAP(size, SIZE, size_t)
NVSHMEM_TYPE_SWAP(ptrdiff, PTRDIFF, ptrdiff_t)
NVSHMEM_TYPE_SWAP(float, FLOAT, float)
NVSHMEM_TYPE_SWAP(double, DOUBLE, double)

#define NVSHMEM_TYPE_COMPARE_SWAP(Name, NameIdx, TYPE)                                                  \
    TYPE nvshmem_##Name##_atomic_compare_swap(TYPE *target, TYPE cond, TYPE value, int pe) {                   \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                          \
        TYPE ret;                                                                                \
        nvshmemi_prepare_and_post_amo(NVSHMEMI_AMO_COMPARE_SWAP, (void *)target, (void *)&ret, &value, &cond, \
                                      sizeof(TYPE), pe, NameIdx, "nvshmem_" #Name "atomic_compare_swap");     \
        return ret;                                                                              \
    }
NVSHMEM_TYPE_COMPARE_SWAP(uint, UINT, unsigned int)
NVSHMEM_TYPE_COMPARE_SWAP(ulong, ULONG, unsigned long)
NVSHMEM_TYPE_COMPARE_SWAP(ulonglong, ULONGLONG, unsigned long long)
NVSHMEM_TYPE_COMPARE_SWAP(int32, INT32, int32_t)
NVSHMEM_TYPE_COMPARE_SWAP(uint32, UINT32, uint32_t)
NVSHMEM_TYPE_COMPARE_SWAP(int64, INT64, int64_t)
NVSHMEM_TYPE_COMPARE_SWAP(uint64, UINT64, uint64_t)
NVSHMEM_TYPE_COMPARE_SWAP(int, INT, int)
NVSHMEM_TYPE_COMPARE_SWAP(long, LONG, long)
NVSHMEM_TYPE_COMPARE_SWAP(longlong, LONGLONG, long long)
NVSHMEM_TYPE_COMPARE_SWAP(size, SIZE, size_t)
NVSHMEM_TYPE_COMPARE_SWAP(ptrdiff, PTRDIFF, ptrdiff_t)

#define NVSHMEM_TYPE_FETCH_AND(Name, NameIdx, TYPE)                                              \
    TYPE nvshmem_##Name##_atomic_fetch_and(TYPE *target, TYPE value, int pe) {              \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                     \
        TYPE ret;                                                                           \
        nvshmemi_prepare_and_post_amo(NVSHMEMI_AMO_FETCH_AND, (void *)target, (void *)&ret, &value, 0,   \
                                      sizeof(TYPE), pe, NameIdx, "nvshmem_" #Name "_atomic_fetch_and"); \
        return ret;                                                                         \
    }
NVSHMEM_TYPE_FETCH_AND(uint, UINT, unsigned int)
NVSHMEM_TYPE_FETCH_AND(ulong, ULONG, unsigned long)
NVSHMEM_TYPE_FETCH_AND(ulonglong, ULONGLONG, unsigned long long)
NVSHMEM_TYPE_FETCH_AND(int32, INT32, int32_t)
NVSHMEM_TYPE_FETCH_AND(uint32, UINT32, uint32_t)
NVSHMEM_TYPE_FETCH_AND(int64, INT64, int64_t)
NVSHMEM_TYPE_FETCH_AND(uint64, UINT64, uint64_t)

#define NVSHMEM_TYPE_FETCH_OR(Name, NameIdx, TYPE)                                              \
    TYPE nvshmem_##Name##_atomic_fetch_or(TYPE *target, TYPE value, int pe) {              \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                    \
        TYPE ret;                                                                          \
        nvshmemi_prepare_and_post_amo(NVSHMEMI_AMO_FETCH_OR, (void *)target, (void *)&ret, &value, 0,   \
                                      sizeof(TYPE), pe, NameIdx, "nvshmem_" #Name "_atomic_fetch_or"); \
        return ret;                                                                        \
    }
NVSHMEM_TYPE_FETCH_OR(uint, UINT, unsigned int)
NVSHMEM_TYPE_FETCH_OR(ulong, ULONG, unsigned long)
NVSHMEM_TYPE_FETCH_OR(ulonglong, ULONGLONG, unsigned long long)
NVSHMEM_TYPE_FETCH_OR(int32, INT32, int32_t)
NVSHMEM_TYPE_FETCH_OR(uint32, UINT32, uint32_t)
NVSHMEM_TYPE_FETCH_OR(int64, INT64, int64_t)
NVSHMEM_TYPE_FETCH_OR(uint64, UINT64, uint64_t)

#define NVSHMEM_TYPE_FETCH_XOR(Name, NameIdx, TYPE)                                              \
    TYPE nvshmem_##Name##_atomic_fetch_xor(TYPE *target, TYPE value, int pe) {              \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                     \
        TYPE ret;                                                                           \
        nvshmemi_prepare_and_post_amo(NVSHMEMI_AMO_FETCH_XOR, (void *)target, (void *)&ret, &value, 0,   \
                                      sizeof(TYPE), pe, NameIdx, "nvshmem_" #Name "_atomic_fetch_xor"); \
        return ret;                                                                         \
    }
NVSHMEM_TYPE_FETCH_XOR(uint, UINT, unsigned int)
NVSHMEM_TYPE_FETCH_XOR(ulong, ULONG, unsigned long)
NVSHMEM_TYPE_FETCH_XOR(ulonglong, ULONGLONG, unsigned long long)
NVSHMEM_TYPE_FETCH_XOR(int32, INT32, int32_t)
NVSHMEM_TYPE_FETCH_XOR(uint32, UINT32, uint32_t)
NVSHMEM_TYPE_FETCH_XOR(int64, INT64, int64_t)
NVSHMEM_TYPE_FETCH_XOR(uint64, UINT64, uint64_t)
