#include "MemMapManager.h"

MemMapManager * MemMapManager::instance_ = nullptr;
std::once_flag MemMapManager::singletonFlag_;
const char MemMapManager::name[128] = "MemMapManager";
const char MemMapManager::endpointName[128] = "MemMapManager_Server_EndPoint";
const char MemMapManager::barrierName[128] = "MemMapManager_Server_Barrier";

MemMapManager::MemMapManager() {

    char barrierToRemove[128] = "/dev/shm/sem.";
    strcat(barrierToRemove, MemMapManager::barrierName);
    unlink(barrierToRemove);
    unlink(MemMapManager::endpointName);

    CUUTIL_ERRCHK(cuInit(0));
    
    CUUTIL_ERRCHK(cuDeviceGetCount(&device_count_));
    std::cout << "MemMapManager Server detected " << device_count_ << " GPU(s)" << std::endl;
    devices_.resize(device_count_);

    char gpuName[128];
    for(int i = 0; i < device_count_; ++i) {
        CUUTIL_ERRCHK(cuDeviceGet(&devices_[i], i));
        CUUTIL_ERRCHK(cuDeviceGetName(gpuName, 128, devices_[i]));
        std::cout << "* Device " << devices_[i] << ": " << gpuName << std::endl;
    }
    CUUTIL_ERRCHK(cuCtxCreate(&ctx_, 0, devices_[0]));
    
    if((ipc_sock_fd_ = socket(AF_UNIX, SOCK_DGRAM, 0)) == -1) {
        panic("MemMapManager: Failed to open server socket");
    }

    bzero(&server_addr, sizeof(server_addr));
    server_addr.sun_family = AF_UNIX;
    size_t name_len = strlen(MemMapManager::endpointName);
    if (name_len >= sizeof(server_addr.sun_path)) {
        panic("MemMapManager: Name is too long");
    }
    strncpy(server_addr.sun_path, MemMapManager::endpointName, name_len);

    if (bind(ipc_sock_fd_, (struct sockaddr *)&server_addr, SUN_LEN(&server_addr)) < 0) {
        std::cout << getpid() << " server_addr: " << server_addr.sun_path << std::endl;
        panic("MemMapManager::MemMapManager: Binding IPC server socket failed");
    }

    ProcessInfo serverProcess;
    serverProcess.SetContext(ctx_);
    if (Register(serverProcess) != M3INTERNAL_OK) {
        panic("Server process failed to register itself\n");
    }

    sem_t * sem = sem_open(MemMapManager::barrierName, O_CREAT, S_IRUSR|S_IWUSR, 0);
    if (sem == SEM_FAILED) {
        std::cout << MemMapManager::barrierName << std::endl;
        panic("MemMapManager: Failed to open semaphore");
    }
    sem_post(sem);
    sem_close(sem);                                                                                                                 

    Server();
    
}

void MemMapManager::Server() {

    bool halt = false;
    M3InternalErrorType m3Err;
    std::vector<CUmemGenericAllocationHandle> allocHandles;
    std::vector<shareable_handle_t> shHandles;
    uint32_t numShareableHandles;
    bool shHandleAlreadyExists;


    while(!halt) {
        MemMapRequest req;
        MemMapResponse res;

        struct sockaddr_un client_addr;
        socklen_t client_addr_len = sizeof(client_addr);
        bzero(&client_addr, client_addr_len);
        if (recvfrom(ipc_sock_fd_, (void *)&req, sizeof(req), 0, (struct sockaddr *)&client_addr, &client_addr_len) < 0) {
            panic("MemMapManager::MemMapManager: failed to receive IPC message");
        }
        res.dst = req.src;
        switch (req.cmd) {
            case CMD_ECHO:
                res.status = STATUSCODE_ACK;
                break;
            case CMD_HALT:
                res.status = STATUSCODE_ACK;
                halt = true;
                break;
            case CMD_REGISTER:
                res.status = STATUSCODE_ACK;
                Register(res.dst);
                break;
            case CMD_ALLOCATE:
                // We assume that client request CMD_GETROUNDEDALLOCATIONSIZE before CMD_ALLOCATE.
                // Thus, no size rounding is provided in this command.
                res.status = STATUSCODE_ACK;
                
                shHandles.clear();
                shHandleAlreadyExists = false;
                for (auto& it : memIdToShHandle_) {
                    if(it.first == req.memId) {
                        if (!shHandleAlreadyExists) {
                            shHandleAlreadyExists = true;
                            res.numShareableHandles = 1;
                        } else {
                            res.numShareableHandles++;
                        }
                        shHandles.push_back(it.second);
                    }
                }
                /*
                std::cout << "dump of memIdToShHandle_:";
                for(auto & it : memIdToShHandle_) {
                    std::cout << it.first << " -> " << it.second << ", ";
                }
                std::cout << std::endl;
                std::cout << "shareable handle cache hit: ";
                for(auto & sh : shHandles) {
                    std::cout << sh << ", ";
                }
                std::cout << std::endl;
                */

                if (!shHandleAlreadyExists) {
                    // MEM_POOL_NUM_ENTRY should be defined in memory pool class definition.
                    // uint32_t numShareableHandles = (req.size + MEM_POOL_NUM_ENTRY - 1) / MEM_POOL_NUM_ENTRY;
                    // For now, we just cut the region into half.
                    numShareableHandles = 1;
                    res.numShareableHandles = numShareableHandles;
                    shHandles.resize(numShareableHandles);
                    m3Err = Allocate(req.src, req.alignment, req.size, shHandles, allocHandles);
                    if (m3Err != M3INTERNAL_OK) {
                        printf("M3 Internal Error Code %d\n", m3Err);
                        panic("Server failed to Allocate()");
                    }
                    for(auto& sh : shHandles) {
                        memIdToShHandle_[req.memId] = sh;
                        shHandletoMemId_[sh] = req.memId;
                    }
                    for(auto& ah : allocHandles) {
                        CUUTIL_ERRCHK(cuMemRelease(ah));
                    }
                }
                break;
            case CMD_GETROUNDEDALLOCATIONSIZE:
                res.status = STATUSCODE_ACK;
                res.roundedSize = GetRoundedAllocationSize(req.size);
                break;
            default:
                res.status = STATUSCODE_NYI;
                break;
        }

        if (sendto(ipc_sock_fd_, (const void *)&res, sizeof(res), 0, (struct sockaddr *)&client_addr, sizeof(client_addr)) < 0) {
            panic("MemMapManager::MemMapManager: failed to send IPC message");
        }

        if (req.cmd == CMD_ALLOCATE) {
            strncpy(res.memId, req.memId, MAX_MEMID_LEN);
            for(auto sh : shHandles) {
                res.shareableHandle = sh;
                if (sendShareableHandle(ipc_sock_fd_, &client_addr, res.shareableHandle) < 0) {
                    panic("MemMapManager::MemMapManager: failed to send res.shsareableHandle");
                }
            }
        }
        
    }

}


MemMapManager::~MemMapManager() {
    char barrierToRemove[128] = "/dev/shm/sem.";
    strcat(barrierToRemove, MemMapManager::barrierName);
    unlink(barrierToRemove);

    close(ipc_sock_fd_);
    unlink(MemMapManager::endpointName);

}


MemMapResponse MemMapManager::RequestRegister(ProcessInfo &pInfo, int sock_fd) {

    MemMapRequest req;
    req.src = pInfo;
    req.cmd = CMD_REGISTER;
    MemMapResponse res = MemMapManager::Request(sock_fd, req, &server_addr);
    return res;

}

M3InternalErrorType MemMapManager::Register(ProcessInfo &pInfo) {

    auto processIterator = std::find(subscribers_.begin(), subscribers_.end(), pInfo);
    if (processIterator != subscribers_.end()) {
        // Process is already subscribing memory server. Ignored.
        return M3INTERNAL_DUPLICATE_REGISTER;
    }

    for(auto subscriber : subscribers_) {
        int a2b, b2a;
        CUUTIL_ERRCHK(cuDeviceCanAccessPeer(&a2b, pInfo.device, subscriber.device));
        CUUTIL_ERRCHK(cuDeviceCanAccessPeer(&b2a, subscriber.device, pInfo.device));
            
        if (a2b && b2a) {
            cuCtxSetCurrent(pInfo.ctx);
            cuCtxEnablePeerAccess(subscriber.ctx, 0);
            cuCtxSetCurrent(subscriber.ctx);
            cuCtxEnablePeerAccess(pInfo.ctx, 0);
        }
    }

    subscribers_.push_back(pInfo);
    return M3INTERNAL_OK;
}

MemMapResponse MemMapManager::RequestAllocate(ProcessInfo &pInfo, int sock_fd, size_t alignment, size_t num_bytes) {
    
    // Request from M3 server.
    // RequestAllocate should be separately implemented from other Request APIs.
    MemMapRequest req;
    req.src = pInfo;
    req.cmd = CMD_ALLOCATE;
    req.alignment = 1024;
    req.size = num_bytes;
    strncpy(req.memId, "myRedundantMemoryId", MAX_MEMID_LEN);


    MemMapResponse res;   
    res.status = STATUSCODE_ACK;

    int shHandleCount = 0;
    std::vector<shareable_handle_t> shHandles;

    sem_t * sem;
    sem = sem_open(MemMapManager::barrierName, O_CREAT, S_IRUSR|S_IWUSR, 0);
    if (sem == SEM_FAILED) {
        panic("Request: Failed to open semaphore");
    }
    sem_wait(sem);


    socklen_t server_addr_len = SUN_LEN(&server_addr);

    if (sendto(sock_fd, (const void *)&req, sizeof(req), 0, (struct sockaddr *)&server_addr, server_addr_len) < 0) {
        perror("Request sendto() call failure");
        res.status = STATUSCODE_SOCKERR;
    }

    if (recvfrom(sock_fd, (void *)&res, sizeof(res), 0, (struct sockaddr *)&server_addr, &server_addr_len) < 0) {
        perror("MemMapManager::RequestRegister failed to receive RequestRegister result");
        res.status = STATUSCODE_SOCKERR;
    }

    do {
        if (recvShareableHandle(sock_fd, &res.shareableHandle) < 0) {
            perror("MemMapManager::RequestRegister failed to receive RequestAllocate result");
            res.status = STATUSCODE_SOCKERR;    
        }
        shHandles.push_back(res.shareableHandle);
    } while(++shHandleCount < res.numShareableHandles);

    sem_post(sem);
    sem_close(sem);
    
    CUmemAccessDesc accessDescriptor;
    accessDescriptor.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDescriptor.location.id = pInfo.device;
    accessDescriptor.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    CUmemGenericAllocationHandle allocHandle;
    res.d_ptr = (CUdeviceptr)nullptr;
    CUUTIL_ERRCHK(cuMemAddressReserve(&res.d_ptr, req.size, alignment, 0, 0));

    assert(res.numShareableHandles > 0);
    size_t chunkSize = req.size / res.numShareableHandles;

    for(int i = 0; i < res.numShareableHandles; ++i) {
        CUUTIL_ERRCHK(cuMemImportFromShareableHandle(
            &allocHandle, (void *)(uintptr_t)shHandles[i], CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
        CUUTIL_ERRCHK(cuMemMap(res.d_ptr + i * chunkSize, chunkSize, 0, allocHandle, 0));
        CUUTIL_ERRCHK(cuMemRelease(allocHandle));
        close(shHandles[i]);
    }

    CUUTIL_ERRCHK(cuMemSetAccess(res.d_ptr, req.size, &accessDescriptor, 1));
    return res;

}

MemMapResponse MemMapManager::RequestRoundedAllocationSize(ProcessInfo &pInfo, int sock_fd, size_t num_bytes) {

    MemMapRequest req;
    req.src = pInfo;
    req.cmd = CMD_GETROUNDEDALLOCATIONSIZE;
    req.size = num_bytes;
    MemMapResponse res = MemMapManager::Request(sock_fd, req, &server_addr);
    return res;

}

size_t MemMapManager::GetRoundedAllocationSize(size_t num_bytes) {

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    // Use GPU 0 as Memory server device, for now.
    prop.location.id = devices_[0];
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    size_t granularity = 0;
    CUUTIL_ERRCHK(cuMemGetAllocationGranularity(
        &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    if (num_bytes % granularity) {
       num_bytes += (granularity - (num_bytes % granularity));
    }

    return num_bytes;

}


M3InternalErrorType MemMapManager::Allocate(ProcessInfo &pInfo, size_t alignment, size_t num_bytes, std::vector<shareable_handle_t>& shHandle, std::vector<CUmemGenericAllocationHandle>& allocHandle) {

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    // Use GPU 0 as Memory server device, for now.
    // Since I don't have NVLINK-supported environment now,
    // I'll just set pInfo.device = 0 in child process.
    prop.location.id = pInfo.device;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

    uint32_t num_handles = shHandle.size();
    allocHandle.resize(num_handles);
    size_t chunk_size = GetRoundedAllocationSize(num_bytes / num_handles);
    assert(num_bytes % chunk_size == 0);

    for(int i = 0; i < num_handles; ++i) {
        CUUTIL_ERRCHK( cuMemCreate(&allocHandle[i], chunk_size, &prop, 0) );
        CUUTIL_ERRCHK( cuMemExportToShareableHandle((void *)&shHandle[i], allocHandle[i], CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0) );
    }
    return M3INTERNAL_OK;

}

MemMapResponse MemMapManager::RequestDeAllocate(ProcessInfo &pInfo, int sock_fd, shareable_handle_t shHandle) {
    MemMapRequest req;
    req.src = pInfo;
    req.cmd = CMD_DEALLOCATE;
    req.shareableHandle = shHandle;
    MemMapResponse res;
    res = Request(sock_fd, req, &server_addr);
    return res;
}

M3InternalErrorType MemMapManager::DeAllocate(ProcessInfo &pInfo, shareable_handle_t shHandle) {
    return M3INTERNAL_NYI;
}



std::string MemMapManager::DebugString() const {

    std::string dbg;
    dbg.append("name: ");
    dbg.append(MemMapManager::name);
    dbg.append(", ipc_name: ");
    dbg.append(MemMapManager::endpointName);
    dbg.append(", ipc_sock_fd: ");
    dbg.append(std::to_string(ipc_sock_fd_));

    return dbg;

}


MemMapResponse MemMapManager::Request(int sock_fd, MemMapRequest req, struct sockaddr_un * remote_addr) {
    
    sem_t * sem;
    sem = sem_open(MemMapManager::barrierName, O_CREAT, S_IRUSR|S_IWUSR, 0);
    if (sem == SEM_FAILED) {
        panic("Request: Failed to open semaphore");
    }
    sem_wait(sem);
    

    MemMapResponse res;
    res.status = STATUSCODE_ACK;

    uint32_t shHandleCnt = 0;
    std::vector<shareable_handle_t> shHandles;
    
    socklen_t remote_addr_len = SUN_LEN(remote_addr);
    if (sendto(sock_fd, (const void *)&req, sizeof(req), 0, (struct sockaddr *)remote_addr, remote_addr_len) < 0) {
        perror("Request sendto() call failure");
        printf("tried to open %s\n", remote_addr->sun_path);
        res.status = STATUSCODE_SOCKERR;
    }

    if (recvfrom(sock_fd, (void *)&res, sizeof(res), 0, (struct sockaddr *)remote_addr, &remote_addr_len) < 0) {
        perror("MemMapManager::RequestRegister failed to receive RequestRegister result");
        res.status = STATUSCODE_SOCKERR;
    }

    sem_post(sem);
    sem_close(sem);

    return res;

}



void panic(const char * msg) {

    perror(msg);
    exit(EXIT_FAILURE);

}

static int sendShareableHandle(int sock_fd, struct sockaddr_un * client_addr, shareable_handle_t shHandle) {

    struct msghdr msg;
    struct iovec iov[1];

    union {
        struct cmsghdr cm;
        char control[CMSG_SPACE(sizeof(int))];
    } control_un;

    struct cmsghdr *cmptr;
    ssize_t readResult;
    socklen_t len = sizeof(*client_addr);

    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof(control_un.control);

    cmptr = CMSG_FIRSTHDR(&msg);
    cmptr->cmsg_len = CMSG_LEN(sizeof(int));
    cmptr->cmsg_level = SOL_SOCKET;
    cmptr->cmsg_type = SCM_RIGHTS;

    memmove(CMSG_DATA(cmptr), &shHandle, sizeof(shHandle));

    msg.msg_name = (void *)client_addr;
    msg.msg_namelen = sizeof(struct sockaddr_un);

    iov[0].iov_base = (void *)"";
    iov[0].iov_len = 1;
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;

    ssize_t sendResult = sendmsg(sock_fd, &msg, 0);
    if (sendResult <= 0) {
        perror("IPC failure: Sending data over socket failed");
        return -1;
    }
    return 0;

}

static int recvShareableHandle(int sock_fd, shareable_handle_t *shHandle) {
    struct msghdr msg = {0};
    struct iovec iov[1];
    struct cmsghdr cm;

    // Union to guarantee alignment requirements for control array
    union {
        struct cmsghdr cm;
        char control[CMSG_SPACE(sizeof(int))];
    } control_un;

    struct cmsghdr *cmptr;
    ssize_t n;
    int receivedfd;
    char dummy_buffer[1];
    ssize_t sendResult;

    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof(control_un.control);

    iov[0].iov_base = (void *)dummy_buffer;
    iov[0].iov_len = sizeof(dummy_buffer);

    msg.msg_iov = iov;
    msg.msg_iovlen = 1;

    if ((n = recvmsg(sock_fd, &msg, 0)) <= 0) {
        perror("IPC failure: Receiving data over socket failed");
        return -1;
    }

    if (((cmptr = CMSG_FIRSTHDR(&msg)) != NULL) &&
        (cmptr->cmsg_len == CMSG_LEN(sizeof(int)))) {
    if ((cmptr->cmsg_level != SOL_SOCKET) || (cmptr->cmsg_type != SCM_RIGHTS)) {
        return -1;
    }

    memmove(&receivedfd, CMSG_DATA(cmptr), sizeof(receivedfd));
    *(int *)shHandle = receivedfd;
    } else {
    return -1;
    }

    return 0;
}
