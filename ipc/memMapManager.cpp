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
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include "cuda.h"

void panic(const char * msg) {
    perror(msg);
    exit(EXIT_FAILURE);
}

class ProcessInfo {
    public:
        pid_t pid;
        bool operator ==(const ProcessInfo &other) const {
            bool ret = true;
            ret = ret && (pid == other.pid);
            return ret;
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
    
    struct sockaddr_un server_sun;
    if((ipc_sock_fd_ = socket(AF_UNIX, SOCK_DGRAM, 0)) == -1) {
        panic("MemMapManager: Failed to open server socket");
    }
    bzero(&server_sun, sizeof(server_sun));
    server_sun.sun_family = AF_UNIX;
    size_t name_len = strlen(MemMapManager::endpointName);
    if (name_len >= sizeof(server_sun.sun_path)) {
        panic("MemMapManager: Name is too long");
    }
    strncpy(server_sun.sun_path, MemMapManager::endpointName, name_len);

    if (bind(ipc_sock_fd_, (struct sockaddr *)&server_sun, SUN_LEN(&server_sun)) < 0) {
        panic("MemMapManager::MemMapManager: Binding IPC server socket failed");
    }

    while(true) {
        MemMapRequest req;
        MemMapResponse res;
        if (recv(ipc_sock_fd_, &req, sizeof(req), 0) < 0) {
            panic("MemMapManager::MemMapManager: failed to receive IPC message");
        }
        switch (req.cmd) {
            case REGISTER:
                res.status = ACK;
                RegisterProcess(req.src);
                break;
            default:
                res.status = NYI;
                break;
        }
        if (send(ipc_sock_fd_, &res, sizeof(res), 0) < 0) {
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
    struct sockaddr_un client_sun;

    if ((sock_fd = socket(AF_UNIX, SOCK_DGRAM, 0)) == -1) {
        panic("MemMapManager::Subscribe failed to open socket");
    }

    bzero(&client_sun, sizeof(client_sun));
    client_sun.sun_family = AF_UNIX;
    strcpy(client_sun.sun_path, MemMapManager::endpointName);
    std::cout << "pid " << getpid() << " binding..." << std::endl;
    if (bind(sock_fd, (struct sockaddr *)&client_sun, SUN_LEN(&client_sun)) < 0) {
        panic(client_sun.sun_path);
        panic("MemMapManager::Subscribe failed to bind client socket");
    }

    MemMapRequest req;
    req.src = pInfo;
    req.cmd = REGISTER;
    MemMapResponse res;
    if (send(sock_fd, &req, sizeof(req), 0) < 0) {
        panic("MemMapManager::Subscribe failed to send subscribe request");
    }
    if (recv(sock_fd, &res, sizeof(res), 0) < 0) {
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
        // child process as a demo client.
        ProcessInfo pInfo;
        pInfo.pid = getpid();
        try {
            // sleep(3);
            MemMapManager::Subscribe(pInfo);
        }
        catch(char const * msg) {
            std::cout << msg << std::endl;
        }
    } else {
        // parent process as a demo server.
    }
    try {
        MemMapManager* m3 = MemMapManager::Instance();
    } catch(char const * msg) {
        std::cout << msg << std::endl;
    }
    return 0;
}