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
        int device_ordinal;
        CUdevice device;
        CUcontext ctx;

        ProcessInfo(void) : ProcessInfo(0) {}

        ProcessInfo(int _device_ordinal) {
            pid = getpid();
            device_ordinal = _device_ordinal;
            CUUTIL_ERRCHK(cuDeviceGet(&device, device_ordinal));
        }

        ProcessInfo(const ProcessInfo& pInfo) {
            if (this != &pInfo)
            {
                pid = pInfo.pid;
                device_ordinal = pInfo.device_ordinal;
                device = pInfo.device;
                ctx = pInfo.ctx;
            }
        }
        /*
        ProcessInfo& operator =(const ProcessInfo& pInfo)
        {
            std::cout << "boom!" << std::endl;
            *this = ProcessInfo(pInfo);
            return *this;
        }
        */

        bool operator ==(const ProcessInfo &other) const {
            bool ret = true;
            ret = ret && (pid == other.pid);
            return ret;
        }
        void AddressString(char *addrStr, size_t max_len) {
            snprintf(addrStr, max_len, "%d_ipc", pid);
        }
        std::string DebugString() {
            char buf[1024];
            sprintf(buf, "pid = %d\n", pid);
            sprintf(buf+strlen(buf), "device = %d\n", device);
            sprintf(buf+strlen(buf), "device_ordinal = %d\n", device_ordinal);
            std::string s(buf);
            return s;
        }
};

enum MemMapCmd {
    CMD_INVALID,
    CMD_HALT,
    CMD_REGISTER,
    CMD_DEREGISTER,
    CMD_ALLOCATE,
    CMD_DEALLOCATE,
    CMD_IMPORT,
    CMD_GETROUNDEDALLOCATIONSIZE
};

enum MemMapStatusCode {
    STATUSCODE_INVALID,
    STATUSCODE_ACK,
    STATUSCODE_NYI,
    STATUSCODE_SOCKERR
};

class MemMapRequest {
    public:
        MemMapRequest() : MemMapRequest(CMD_INVALID) {}
        MemMapRequest(MemMapCmd _cmd) {
            cmd = _cmd;
            size = 0;
            alignment = 0;
        }
        MemMapCmd cmd;
        ProcessInfo src;
        size_t size, alignment;
        ProcessInfo importSrc;
};

typedef uintptr_t shareable_handle_t;
class MemMapResponse {
    public:
        MemMapResponse(void) : MemMapResponse(STATUSCODE_INVALID) {}
        MemMapResponse(MemMapStatusCode _status) {
            status = _status;
            shareableHandle = (shareable_handle_t)nullptr;
            roundedSize = 0;
            d_ptr = (CUdeviceptr)nullptr;
        }
        MemMapStatusCode status;
        ProcessInfo dst;
        shareable_handle_t shareableHandle;
        size_t roundedSize;
        CUdeviceptr d_ptr;

        std::string DebugString() {
            char buf[1024];
            sprintf(buf, "* status code = %d\n", status);
            sprintf(buf+strlen(buf), "* shareableHandle = %p\n", shareableHandle);
            sprintf(buf+strlen(buf), "* roundedSize = %p\n", roundedSize);
            sprintf(buf+strlen(buf), "* d_ptr = %p\n", d_ptr);
            sprintf(buf+strlen(buf), "* Destination process info\n");
            sprintf(buf+strlen(buf), "* %s", dst.DebugString().c_str());
            std::string s(buf);
            return s;
        }
};



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

        static MemMapResponse RequestRegister(ProcessInfo &pInfo, int sock_fd);
        void Register(ProcessInfo &pInfo);

        static MemMapResponse RequestAllocate(ProcessInfo &pInfo, int sock_fd, size_t alignment, size_t num_bytes);
        shareable_handle_t Allocate(ProcessInfo &pInfo, size_t alignment, size_t num_bytes);

        static MemMapResponse RequestDeAllocate(ProcessInfo &pInfo, int sock_fd, shareable_handle_t shHandle);
        void DeAllocate(ProcessInfo &pInfo, shareable_handle_t shHandle);

        static MemMapResponse RequestRoundedAllocationSize(ProcessInfo &pInfo, int sock_fd, size_t num_bytes);

        std::string DebugString() const;
        static const char name[128];
        static const char endpointName[128];

        CUcontext ctx(void) { return ctx_; }

    private:
        MemMapManager();
        void Server();
        size_t GetRoundedAllocationSize(size_t num_bytes);

        static MemMapManager * instance_;
        static std::once_flag singletonFlag_;
        int ipc_sock_fd_;
        std::vector<CUdevice> devices_;
        int device_count_;
        std::vector<ProcessInfo> subscribers_;
        CUcontext ctx_;
        std::vector<std::pair<CUcontext, CUcontext>> edges_; 
};



static struct sockaddr_un server_addr = { 
    AF_UNIX,
    "MemMapManager_Server_EndPoint"
};

void panic(const char * msg);


static int sendShareableHandle(int sock_fd, struct sockaddr_un * client_addr, shareable_handle_t shHandle);
static int recvShareableHandle(int sock_fd, shareable_handle_t *shHandle);