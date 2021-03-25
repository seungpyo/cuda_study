#pragma once

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
#include "cuutils.h"
#include "shm_barrier.h"


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
    CMD_INVALID,
    CMD_HALT,
    CMD_REGISTER,
    CMD_DEREGISTER,
    CMD_ALLOCATE,
    CMD_DEALLOCATE,
    CMD_IMPORT
};

enum MemMapStatusCode {
    STATUSCODE_INVALID,
    STATUSCODE_ACK,
    STATUSCODE_NYI,
    STATUSCODE_SOCKERR
};

typedef struct MemMapRequestSt {
    MemMapCmd cmd;
    ProcessInfo src;
    size_t size, alignment;
    ProcessInfo importSrc;
} MemMapRequest;

typedef uintptr_t shareable_handle_t;
typedef struct MemMapResponseSt {
    MemMapStatusCode status;
    shareable_handle_t shareableHandle;
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

        static MemMapResponse Request(int sock_fd, MemMapRequest req, struct sockaddr_un * remote_addr);

        static MemMapStatusCode RequestRegister(ProcessInfo &pInfo, int sock_fd);
        void Register(ProcessInfo &pInfo);

        static MemMapStatusCode RequestAllocate(ProcessInfo &pInfo, int sock_fd, size_t alignment, size_t num_bytes, shareable_handle_t * shHandle);
        shareable_handle_t Allocate(ProcessInfo &pInfo, size_t alignment, size_t num_bytes);

        static MemMapStatusCode RequestDeAllocate(ProcessInfo &pInfo, int sock_fd, shareable_handle_t shHandle);
        void DeAllocate(ProcessInfo &pInfo, shareable_handle_t shHandle);

        std::string DebugString() const;
        static const char name[128];
        static const char endpointName[128];

    private:
        MemMapManager();
        void Server();
        static MemMapManager * instance_;
        static std::once_flag singletonFlag_;
        int ipc_sock_fd_;
        std::vector<CUdevice> devices_;
        int device_count_;
        std::vector<ProcessInfo> subscribers;
};



static struct sockaddr_un server_addr = { 
    AF_UNIX,
    "MemMapManager_Server_EndPoint"
};

void panic(const char * msg);

