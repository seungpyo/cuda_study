#include "MemMapManager.h"

void test_singleton(void) {
    MemMapManager *m3 = MemMapManager::Instance();
    MemMapManager *m3_dup = MemMapManager::Instance();
    std::cout << m3->DebugString() << std::endl;
    std::cout << m3_dup->DebugString() << std::endl;
    std::cout << m3 << ", " << m3_dup << std::endl;
}



int main() {
    // Ensure that MemMapManager instance exists before forking.
    pid_t pid;
    if ((pid = fork()) == 0) {
        CUUTIL_ERRCHK(cuInit(0));
        CUcontext ctx;
        CUdevice device;
        
        CUstream stream_h2d, stream_d2h;
        int multiProcessorCount;

        // And then, launch client process.
        // sharedMemoryInfo shmInfo;
        ProcessInfo pInfo;
        pInfo.pid = getpid();

        // Open and bind socket to client's unique IPC file.
        int sock_fd = 0;
        struct sockaddr_un client_addr;

        if ((sock_fd = socket(AF_UNIX, SOCK_DGRAM, 0)) == -1) {
            panic("MemMapManager::RequestRegister failed to open socket");
        }

        bzero(&client_addr, sizeof(client_addr));
        client_addr.sun_family = AF_UNIX;
        pInfo.AddressString(client_addr.sun_path, 100);

        
        if (bind(sock_fd, (struct sockaddr *)&client_addr, SUN_LEN(&client_addr)) < 0) {
            panic("MemMapManager::RequestRegister failed to bind client socket");
        }


        if (sharedMemoryOpen(barrier_name, sizeof(shmStruct),  &shmInfo) != 0) {
            panic("main, sharedMemoryOpen");
        }
        volatile shmStruct * shm = (volatile shmStruct *)shmInfo.addr;
        waitServerInit(&shm->sense, &shm->counter, false);

        pInfo.device_ordinal = 0;
        CUUTIL_ERRCHK(cuDeviceGet(&pInfo.device, pInfo.device_ordinal));
        CUUTIL_ERRCHK(cuCtxCreate(&pInfo.ctx, 0, pInfo.device));
        MemMapResponse res;
        res = MemMapManager::RequestRegister(pInfo, sock_fd);
        if(res.status != STATUSCODE_ACK) {
            panic("Failed to RequestRegister M3");
        }
        pInfo = res.dst;

        
        CUUTIL_ERRCHK(cuStreamCreate(&stream_h2d, CU_STREAM_NON_BLOCKING));
        CUUTIL_ERRCHK(cuStreamCreate(&stream_d2h, CU_STREAM_NON_BLOCKING));

        std::vector<CUdeviceptr> d_ptr;
        for(int t = 0; t < 10; ++t) {
            MemMapResponse res;
            res = MemMapManager::RequestRoundedAllocationSize(pInfo, sock_fd, 32*1024);
            if (res.status != STATUSCODE_ACK) {
                panic("Failed to get rounded allocation size");
            }
            res =  MemMapManager::RequestAllocate(pInfo, sock_fd, 1024, res.roundedSize);
            if(res.status != STATUSCODE_ACK) {
                std::cout << res.status << std::endl;
                panic("Failed to RequestAllocate M3");
            }
            d_ptr.push_back(res.d_ptr);
            auto addr = d_ptr[t];
            char test_string[128] = "can you see me?";
            char recv_string[128];
            CUUTIL_ERRCHK(cuMemcpyHtoD(addr, test_string, 128));
            CUUTIL_ERRCHK(cuMemcpyDtoH(recv_string, addr, 128));
            std::cout << "test_string: " << test_string << std::endl;
            std::cout << "recv_string: " << recv_string << std::endl;
        }
        
        

        MemMapRequest req;
        req.src = pInfo;
        req.cmd = CMD_HALT;
        res = MemMapManager::Request(sock_fd, req, &server_addr);
        if (res.status != STATUSCODE_ACK) {
            panic("Failed to halt M3 instance");
        }

        sharedMemoryClose(&shmInfo);
    } else {
        // parent process as a demo server.
        
        volatile shmStruct * shm = (volatile shmStruct *)shmInfo.addr;
        MemMapManager *m3 = MemMapManager::Instance();
        
        sharedMemoryClose(&shmInfo);
        int wait_status;
        wait(&wait_status);
    }
    return 0;
}