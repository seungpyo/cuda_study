#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string>
#include <vector>
#include <algorithm>
#include <thread>
#include <mutex>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <atomic>
#include "cuda.h"

void panic(const char * msg) {
    perror(msg);
    exit(EXIT_FAILURE);
}
static const char * barrier_name = "MemMapManagerBarrier";
struct sockaddr_un server_addr;

typedef struct sharedMemoryInfo_st {
    void *addr;
    size_t size;
    int shmFd;
} sharedMemoryInfo;
sharedMemoryInfo shmInfo;

int sharedMemoryCreate(const char *name, size_t sz, sharedMemoryInfo *info) {
  int status = 0;

  info->size = sz;

  info->shmFd = shm_open(name, O_RDWR | O_CREAT, 0777);
  if (info->shmFd < 0) {
    return errno;
  }

  status = ftruncate(info->shmFd, sz);
  if (status != 0) {
    return status;
  }

  info->addr = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, info->shmFd, 0);
  if (info->addr == NULL) {
    return errno;
  }

  return 0;
}

int sharedMemoryOpen(const char *name, size_t sz, sharedMemoryInfo *info) {
  info->size = sz;

  info->shmFd = shm_open(name, O_RDWR, 0777);
  if (info->shmFd < 0) {
    return errno;
  }

  info->addr = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, info->shmFd, 0);
  if (info->addr == NULL) {
    return errno;
  }

  return 0;
}

void sharedMemoryClose(sharedMemoryInfo *info) {
  if (info->addr) {
    munmap(info->addr, info->size);
  }
  if (info->shmFd) {
    close(info->shmFd);
  }
}

typedef struct shmStruct_st {
  size_t nprocesses;
  int barrier;
  int sense;
} shmStruct;
#define cpu_atomic_add32(a, x) __sync_add_and_fetch(a, x)
static void barrierWait(volatile int *barrier, volatile int *sense,
                        unsigned int n) {
  int count;
  // Check-in
  
  count = cpu_atomic_add32(barrier, 1);
  if (count == n) {  // Last one in
    *sense = 1;
  }
  while (!*sense);
  // Check-out
  count = cpu_atomic_add32(barrier, -1);
  if (count == 0) {  // Last one out
    *sense = 0;
  }
  while (*sense);
}

class ProcessInfo {
    public:
        pid_t pid;
        bool operator ==(const ProcessInfo &other) const {
            bool ret = true;
            ret = ret && (pid == other.pid);
            return ret;
        }
        void AddressString(char *addrStr, size_t max_len) {
            snprintf(addrStr, max_len, "%d_ipc", pid);
        }
};
enum MemMapCmd {
    REGISTER,
    DEREGISTER,
    ALLOCATE,
    DEALLOCATE,
    IMPORT
};

enum MemMapStatusCode {
    ACK,
    NYI
};

typedef struct MemMapRequestSt {
    MemMapCmd cmd;
    ProcessInfo src;
    size_t size, alignment;
    ProcessInfo importSrc;
} MemMapRequest;

typedef struct MemMapResponseSt {
    MemMapStatusCode status;
    uintptr_t shareableHandle;
} MemMapResponse;



class MemMapManager {
    public:
        ~MemMapManager();
        static MemMapManager* Instance() {
            std::call_once(singletonFlag_, [](){
                instance_ = new MemMapManager();
            });
            return instance_;
        }
        
        std::string Name() { return name; }
        static std::string EndPoint() { return endpointName; }

        static void Subscribe(ProcessInfo &pInfo);
        void RegisterProcess(ProcessInfo &pInfo);

        void* Allocate(ProcessInfo &pInfo, size_t alignment, size_t num_bytes);
        void DeAllocate(ProcessInfo &pInfo, void* d_ptr);

        std::string DebugString() const;
        static const char name[128];
        static const char endpointName[128];

    private:
        MemMapManager();
        static MemMapManager * instance_;
        static std::once_flag singletonFlag_;
        int ipc_sock_fd_;
        std::vector<ProcessInfo> subscribers;
};

MemMapManager * MemMapManager::instance_ = nullptr;
std::once_flag MemMapManager::singletonFlag_;
const char MemMapManager::name[128] = "MemMapManager";
const char MemMapManager::endpointName[128] = "MemMapManager_Endpoints";

MemMapManager::MemMapManager() {
    
    unlink(MemMapManager::endpointName);
    
    
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
        panic("MemMapManager::MemMapManager: Binding IPC server socket failed");
    }

    if (sharedMemoryOpen(barrier_name, sizeof(shmStruct),  &shmInfo) < 0) {
        panic("main, sharedMemoryOpen");
    }
    volatile shmStruct * shm = (volatile shmStruct *)shmInfo.addr;
    shm->barrier = 0;
    shm->nprocesses = 2;
    barrierWait(&shm->barrier, &shm->sense, (unsigned int)(shm->nprocesses));

    while(true) {
        MemMapRequest req;
        MemMapResponse res;
        struct sockaddr_un client_addr;
        socklen_t client_addr_len = sizeof(client_addr);
        if (recvfrom(ipc_sock_fd_, (void *)&req, sizeof(req), 0, (struct sockaddr *)&client_addr, &client_addr_len) < 0) {
            panic("MemMapManager::MemMapManager: failed to receive IPC message");
        }
        std::cout << "Received : {  cmd: " << req.cmd << "}" << std::endl;
        switch (req.cmd) {
            case REGISTER:
                res.status = ACK;
                RegisterProcess(req.src);
                break;
            default:
                res.status = NYI;
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

/*
    Subscribe() is called by user processes.
*/
void MemMapManager::Subscribe(ProcessInfo &pInfo) {

    int sock_fd = 0;
    struct sockaddr_un client_addr;

    if ((sock_fd = socket(AF_UNIX, SOCK_DGRAM, 0)) == -1) {
        panic("MemMapManager::Subscribe failed to open socket");
    }

    bzero(&client_addr, sizeof(client_addr));
    client_addr.sun_family = AF_UNIX;
    pInfo.AddressString(client_addr.sun_path, 100);

    if (bind(sock_fd, (struct sockaddr *)&client_addr, SUN_LEN(&client_addr)) < 0) {
        panic("MemMapManager::Subscribe failed to bind client socket");
    }

    MemMapRequest req;
    req.src = pInfo;
    req.cmd = REGISTER;
    MemMapResponse res;
    if (sendto(sock_fd, (const void *)&req, sizeof(req), 0, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        panic("MemMapManager::Subscribe failed to send subscribe request");
    }
    socklen_t server_addr_len = sizeof(server_addr);
    if (recvfrom(sock_fd, (void *)&res, sizeof(res), 0, (struct sockaddr *)&server_addr, &server_addr_len) < 0) {
        panic("MemMapManager::Subscribe failed to receive subscribe result");
    }
    if (res.status != ACK) {
        panic("MemMapManager::Subscribe Server Error");
        return;
    }
    printf("Successfully registered process!\n");
}

void MemMapManager::RegisterProcess(ProcessInfo &pInfo) {

    auto processIterator = std::find(subscribers.begin(), subscribers.end(), pInfo);
    if (processIterator != subscribers.end()) {
        // Process is already subscribing memory server. Ignored.
        return;
    }
    subscribers.push_back(pInfo);
   
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
        // And then, launch client process.
        // sharedMemoryInfo shmInfo;
        ProcessInfo pInfo;
        pInfo.pid = getpid();

        if (sharedMemoryOpen(barrier_name, sizeof(shmStruct),  &shmInfo) < 0) {
            panic("main, sharedMemoryOpen");
        }
        volatile shmStruct * shm = (volatile shmStruct *)shmInfo.addr;
        shm->nprocesses = 2;
        barrierWait(&shm->barrier, &shm->sense, (unsigned int)(shm->nprocesses));

        MemMapManager::Subscribe(pInfo);
        sharedMemoryClose(&shmInfo);
    } else {
        // parent process as a demo server.
        
        volatile shmStruct * shm = (volatile shmStruct *)shmInfo.addr;
        MemMapManager *m3 = MemMapManager::Instance();
        
        sharedMemoryClose(&shmInfo);
        
    }
    int wait_status;
    wait(&wait_status);
    return 0;
}