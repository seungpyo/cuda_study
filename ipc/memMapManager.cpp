#include "MemMapManager.h"
#include "shm_barrier.h"

MemMapManager * MemMapManager::instance_ = nullptr;
std::once_flag MemMapManager::singletonFlag_;
const char MemMapManager::name[128] = "MemMapManager";
const char MemMapManager::endpointName[128] = "MemMapManager_Server_EndPoint";

MemMapManager::MemMapManager() {
    
    unlink(MemMapManager::endpointName);
    if (sharedMemoryCreate(barrier_name, sizeof(shmStruct),  &shmInfo) < 0) {
        panic("main, sharedMemoryOpen");
    }
    volatile shmStruct * shm = (volatile shmStruct *)shmInfo.addr;
    
    CUUTIL_ERRCHK(cuInit(0));
    CUUTIL_ERRCHK(cuDeviceGetCount(&device_count_));
    devices_.resize(device_count_);
    for(int i = 0; i < device_count_; ++i) {
        CUUTIL_ERRCHK(cuDeviceGet(&devices_[i], i));
    }

    std::cout << "MemMapManager using " << device_count_ << " devices :" << std::endl;
    for(int i = 0; i < device_count_; ++i) {
        std::cout << devices_[i] << std::endl;
    }
    
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

    // barrierWait(&shm->barrier, &shm->sense,2);
    waitServerInit(&shm->sense, true);
    Server();
}

void MemMapManager::Server() {
    bool halt = false;
    while(!halt) {
        MemMapRequest req;
        MemMapResponse res;
        struct sockaddr_un client_addr;
        socklen_t client_addr_len = sizeof(client_addr);
        if (recvfrom(ipc_sock_fd_, (void *)&req, sizeof(req), 0, (struct sockaddr *)&client_addr, &client_addr_len) < 0) {
            panic("MemMapManager::MemMapManager: failed to receive IPC message");
        }
        std::cout << "M3 Server recv cmd : " << req.cmd << std::endl;
        switch (req.cmd) {
            case CMD_HALT:
                res.status = STATUSCODE_ACK;
                halt = true;
            case CMD_REGISTER:
                res.status = STATUSCODE_ACK;
                Register(req.src);
                break;
            case CMD_ALLOCATE:
                res.status = STATUSCODE_ACK;
                res.shareableHandle = Allocate(req.src, req.alignment, req.size);
                break;
            default:
                res.status = STATUSCODE_NYI;
                break;
        }
        
        if (sendto(ipc_sock_fd_, (const void *)&res, sizeof(res), 0, (struct sockaddr *)&client_addr, sizeof(client_addr)) < 0) {
            panic("MemMapManager::MemMapManager: failed to send IPC message");
        }
    }
}

MemMapManager::~MemMapManager() {
    unlink(MemMapManager::endpointName);
    close(ipc_sock_fd_);
}


MemMapStatusCode MemMapManager::RequestRegister(ProcessInfo &pInfo, int sock_fd) {

    MemMapRequest req;
    req.src = pInfo;
    req.cmd = CMD_REGISTER;
    MemMapResponse res = MemMapManager::Request(sock_fd, req, &server_addr);
    return res.status;
}

void MemMapManager::Register(ProcessInfo &pInfo) {

    auto processIterator = std::find(subscribers.begin(), subscribers.end(), pInfo);
    if (processIterator != subscribers.end()) {
        // Process is already subscribing memory server. Ignored.
        return;
    }
    subscribers.push_back(pInfo);
   
}

MemMapStatusCode MemMapManager::RequestAllocate(ProcessInfo &pInfo, int sock_fd, size_t alignment, size_t num_bytes, shareable_handle_t * shHandle) {
    
    MemMapRequest req;
    req.src = pInfo;
    req.cmd = CMD_ALLOCATE;
    MemMapResponse res = MemMapManager::Request(sock_fd, req, &server_addr);
    return res.status;

}



shareable_handle_t MemMapManager::Allocate(ProcessInfo &pInfo, size_t alignment, size_t num_bytes) {
    
    return (shareable_handle_t)nullptr;
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
    MemMapResponse res;
    res.status = STATUSCODE_ACK;
    socklen_t remote_addr_len = SUN_LEN(remote_addr);
    if (sendto(sock_fd, (const void *)&req, sizeof(req), 0, (struct sockaddr *)remote_addr, remote_addr_len) < 0) {
        perror(remote_addr->sun_path);
        // perror("MemMapManager::RequestRegister failed to send RequestRegister request; maybe server_addr is invalid?");
        res.status = STATUSCODE_SOCKERR;
    }
    if (recvfrom(sock_fd, (void *)&res, sizeof(res), 0, (struct sockaddr *)remote_addr, &remote_addr_len) < 0) {
        perror("MemMapManager::RequestRegister failed to receive RequestRegister result");
        res.status = STATUSCODE_SOCKERR;
    }
    return res;
}

void panic(const char * msg) {
    perror(msg);
    exit(EXIT_FAILURE);
}