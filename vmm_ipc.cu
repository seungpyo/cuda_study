#include <iostream>
#include <cstdlib>
#include <vector>
#include <cuda.h>
#include "cuutils.h"

#define CUUTIL_DEBUG
#define SIZE_TO_ALLOC 1024
#define ROUND_UP(x, m) ((m) > 0 ? (((x) + (m) - 1) / (m)) * (m) : (x))

void * ec_malloc(size_t sz) {
    void * ptr = nullptr;
    if((ptr = (void *)malloc(sz)) == nullptr) {
        std::cout << "malloc failed" << std::endl;
    }
    return ptr;
}


int main() {
    int device_id = 0;
    int supportsVMM = 0;
    CUUTIL_ERRCHK(cudaFree(0));  // Force and check the initialization of the runtime

    CUUTIL_ERRCHK(cuCtxGetDevice(&device_id));
    std::cout << "device id = " << device_id << std::endl;
    CUUTIL_ERRCHK(cuDeviceGetAttribute(&supportsVMM, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, device_id));
    
    size_t granularity = 0;
    CUmemGenericAllocationHandle allocHandle;
    CUmemAccessDesc accessDesc;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_id;
    accessDesc.location = prop.location;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUdeviceptr ptr;

    CUUTIL_ERRCHK(cuMemGetAllocationGranularity(&granularity, &prop,
                                            CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    size_t padded_size = ROUND_UP(SIZE_TO_ALLOC, granularity);
    CUUTIL_ERRCHK(cuMemCreate(&allocHandle, padded_size, &prop, 0));

    std::cout << "granularity = " << granularity << ", padded_size = " << padded_size << std::endl;
    CUUTIL_ERRCHK(cuMemAddressReserve(&ptr, padded_size, 0, 0, 0));    
    std::cout << "Reserved VA = 0x" << std::hex << ptr << std::endl;

    CUUTIL_ERRCHK(cuMemMap(ptr, padded_size, 0, allocHandle, 0));
    CUUTIL_ERRCHK(cuMemSetAccess(ptr, padded_size, &accessDesc, 1));

    CUUTIL_ERRCHK(cudaMemset((void *)ptr, (int)'A', padded_size));

    char * host_src = (char *)ec_malloc(padded_size);
    char * host_dst = (char *)ec_malloc(padded_size);

    memset(host_src, 'S', padded_size);
    memset(host_dst, 'D', padded_size);
    CUUTIL_ERRCHK(cudaMemcpy((void *)ptr, (void *)host_src, padded_size, cudaMemcpyHostToDevice));
    CUUTIL_ERRCHK(cudaMemcpy((void *)host_dst, (void *)ptr, padded_size, cudaMemcpyDeviceToHost));
    std::cout << "peeking host dst" << std::endl;
    std::cout << host_dst[0] << host_dst[1] << host_dst[2] << host_dst[3] << std::endl;
    
    CUUTIL_ERRCHK(cuMemUnmap(ptr, padded_size));
    CUUTIL_ERRCHK(cuMemRelease(allocHandle));
    CUUTIL_ERRCHK(cuMemAddressFree(ptr, padded_size));
    free(host_src);
    free(host_dst);
    return 0;
}