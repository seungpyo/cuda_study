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
        
        // barrierWait(&shm->barrier, &shm->sense, 2);
        std::cout << "Waiting for server init..." << std::endl;
        waitServerInit(&shm->sense, &shm->counter, false);
        std::cout << "registering client..." << std::endl;
        if(MemMapManager::RequestRegister(pInfo, sock_fd) != STATUSCODE_ACK) {
            panic("Failed to RequestRegister M3");
        }

        
        for(int t = 0; t < 10; ++t) {
            shareable_handle_t shHandle;
            MemMapStatusCode status;
            if((status = MemMapManager::RequestAllocate(pInfo, sock_fd, 0, 1024, &shHandle)) != STATUSCODE_ACK) {
                std::cout << status << std::endl;
                panic("Failed to RequestAllocate M3");
            }
            printf("M3 allocated shareable handle %p\n", shHandle);
        }
        
        

        MemMapRequest req;
        req.src = pInfo;
        req.cmd = CMD_HALT;
        MemMapResponse res = MemMapManager::Request(sock_fd, req, &server_addr);
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