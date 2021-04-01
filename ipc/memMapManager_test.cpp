#include "MemMapManager.h"

void test_MultiGPUAllocate(char * unit, size_t factor) {
    size_t num_bytes = factor;
    if(!strcmp(unit, "g")) {
        num_bytes <<= 30;
    } else if(!strcmp(unit, "m")) {
        num_bytes <<= 20;
    } else if(!strcmp(unit, "k")) {
        num_bytes <<= 10;
    } else { // default is gigabyte.
        num_bytes <<= 30;
    }

    pid_t pid;
    if ((pid = fork()) == 0) {

        std::cout << "client sleeping for 5 ms" << std::endl;
        usleep(5 * 1000);

        CUUTIL_ERRCHK(cuInit(0));
    
        CUcontext ctx;
        CUdevice dev = 1;
        CUUTIL_ERRCHK(cuCtxCreate(&ctx, 0, dev));
        ProcessInfo pInfo;
        pInfo.SetContext(ctx);

        std::cout << "Client process Info:" << std::endl;
        std::cout << pInfo.DebugString() << std::endl;

        int sock_fd = 0;
        struct sockaddr_un client_addr;
        if ((sock_fd = socket(AF_UNIX, SOCK_DGRAM, 0)) == -1) {
            panic("MemMapManager::RequestRegister failed to open socket");
        }
        bzero(&client_addr, sizeof(client_addr));
        client_addr.sun_family = AF_UNIX;
        strcpy(client_addr.sun_path, pInfo.AddressString().c_str());
        if (bind(sock_fd, (struct sockaddr *)&client_addr, SUN_LEN(&client_addr)) < 0) {
            panic("MemMapManager::RequestRegister failed to bind client socket");
        }
        
        MemMapRequest req;
        MemMapResponse res;

        res = MemMapManager::RequestRegister(pInfo, sock_fd);
        if (res.status != STATUSCODE_ACK) {
            panic("MemMapManager::RequestRegister failed");
        }
        std::cout << "succesfully registered!" << std::endl;

        res = MemMapManager::RequestRoundedAllocationSize(pInfo, sock_fd, num_bytes);
        if (res.status != STATUSCODE_ACK) {
            panic("MemMapManager::RequestRegister failed");
        }
        std::cout << "succesfully rounded to 0x" << std::hex << res.roundedSize << " bytes" << std::endl;

        res = MemMapManager::RequestAllocate(pInfo, sock_fd, 1024, res.roundedSize);
        if (res.status != STATUSCODE_ACK) {
            panic("MemMapManager::RequestRegister failed");
        }
        std::cout << "succesfully allocated at " << res.d_ptr << std::endl;

        req.src = pInfo;
        req.cmd = CMD_HALT;
        res = MemMapManager::Request(sock_fd, req, &server_addr);
        if (res.status != STATUSCODE_ACK) {
            panic("Failed to halt M3 instance");
        }
        unlink(pInfo.AddressString().c_str());

    } else {
        CUUTIL_ERRCHK(cuInit(0));
        MemMapManager *m3 = MemMapManager::Instance();
        int wait_status;
        wait(&wait_status);
    }
}

void test_singleton(void) {
    MemMapManager *m3 = MemMapManager::Instance();
    MemMapManager *m3_dup = MemMapManager::Instance();
    std::cout << m3->DebugString() << std::endl;
    std::cout << m3_dup->DebugString() << std::endl;
    std::cout << m3 << ", " << m3_dup << std::endl;
}

void test_Allocate(void) {
    pid_t pid;
    if ((pid = fork()) == 0) {
        CUUTIL_ERRCHK(cuInit(0));
        CUcontext ctx;
        CUdevice device;
        
        CUstream stream_h2d, stream_d2h;
        int multiProcessorCount;

        // And then, launch client process.
        ProcessInfo pInfo;

        // Open and bind socket to client's unique IPC file.
        int sock_fd = 0;
        struct sockaddr_un client_addr;

        if ((sock_fd = socket(AF_UNIX, SOCK_DGRAM, 0)) == -1) {
            panic("MemMapManager::RequestRegister failed to open socket");
        }

        bzero(&client_addr, sizeof(client_addr));
        client_addr.sun_family = AF_UNIX;
        strcpy(client_addr.sun_path, pInfo.AddressString().c_str());

        
        if (bind(sock_fd, (struct sockaddr *)&client_addr, SUN_LEN(&client_addr)) < 0) {
            panic("MemMapManager::RequestRegister failed to bind client socket");
        }

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
        bool pass = true;
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
            pass = pass && !strcmp(test_string, recv_string);
            if (!pass) {
                break;
            }
        }
        if (pass) {
            std::cout << "ALLOCATE TEST PASSED" << std::endl;
        } else {
            std::cout << "ALLOCATE TEST FAILED" << std::endl;
        }
        

        MemMapRequest req;
        req.src = pInfo;
        req.cmd = CMD_HALT;
        res = MemMapManager::Request(sock_fd, req, &server_addr);
        if (res.status != STATUSCODE_ACK) {
            panic("Failed to halt M3 instance");
        }
        unlink(pInfo.AddressString().c_str());
    } else {
        // parent process as a demo server.
        MemMapManager *m3 = MemMapManager::Instance();
        int wait_status;
        wait(&wait_status);
    }

}


int main(int argc, char *argv[]) {
        if(argc != 3) {
            printf("Usage: [%s] <unit> <Number of unit> \n", argv[0]);
            exit(EXIT_SUCCESS);
        }

        
        test_MultiGPUAllocate(argv[1], (size_t) atoi(argv[2]));
        return 0;
}
