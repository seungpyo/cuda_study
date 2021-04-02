#pragma once

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <assert.h>
#include <string>
#include <vector>
#include <unordered_map>
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
#include <semaphore.h>
#include "cuda.h"
#include "cuutils.h"

typedef uintptr_t shareable_handle_t;

enum M3InternalErrorType {
    M3INTERNAL_INVALIDCODE,
    M3INTERNAL_NYI,
    M3INTERNAL_OK,
    M3INTERNAL_DUPLICATE_REGISTER,
    M3INTERNAL_ENTRY_NOT_FOUND,
};

class ProcessInfo {
    public:
        pid_t pid;
        int device_ordinal;
        CUdevice device;
        CUcontext ctx;

        ProcessInfo(void) : ProcessInfo(0) {}

        ProcessInfo(int _device_ordinal) {
            pid = getpid();
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

        void SetContext(CUcontext &_ctx) {
            ctx = _ctx;
            CUUTIL_ERRCHK( cuCtxGetDevice(&device) );
            // For now, we just assume that device ordinal is same as device id.
            device_ordinal = device;
        }

        bool operator ==(const ProcessInfo &other) const {
            bool ret = true;
            ret = ret && (pid == other.pid);
            return ret;
        }

        std::string AddressString() {
            char buf[1024];
            sprintf(buf, "pid_%d", pid);
            std::string s(buf);
            return s;
        }

        std::string DebugString() {
            char buf[1024];
            sprintf(buf+strlen(buf), "ProcessInfo:\n");
            sprintf(buf+strlen(buf), "* pid = %d\n", pid);
            sprintf(buf+strlen(buf), "* device = %d\n", device);
            sprintf(buf+strlen(buf), "* device_ordinal = %d\n", device_ordinal);
            std::string s(buf);
            return s;
        }
};

enum MemMapCmd {
    CMD_INVALID,
    CMD_ECHO,
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

#define MAX_MEMID_LEN 256

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
        shareable_handle_t shareableHandle;
        char memId[MAX_MEMID_LEN];
        size_t size, alignment;
        ProcessInfo importSrc;
};

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
        char memId[MAX_MEMID_LEN];
        size_t roundedSize;
        CUdeviceptr d_ptr;
        uint32_t numShareableHandles;

        std::string DebugString() {
            char buf[1024];
            sprintf(buf+strlen(buf), "* status code = %d\n", status);
            sprintf(buf+strlen(buf), "* roundedSize = %p\n", roundedSize);
            sprintf(buf+strlen(buf), "* d_ptr = %p\n", d_ptr);
            sprintf(buf+strlen(buf), "* numShareableHandles = %u\n", numShareableHandles);
            sprintf(buf+strlen(buf), "* shareableHandle = %p\n", shareableHandle);
            sprintf(buf+strlen(buf), "* memId = %s\n", memId);
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

        static MemMapResponse Request(int sock_fd, MemMapRequest req, struct sockaddr_un * remote_addr);
        static MemMapResponse RequestRegister(ProcessInfo &pInfo, int sock_fd);
        static MemMapResponse RequestAllocate(ProcessInfo &pInfo, int sock_fd, size_t alignment, size_t num_bytes);
        static MemMapResponse RequestDeAllocate(ProcessInfo &pInfo, int sock_fd, shareable_handle_t shHandle);
        static MemMapResponse RequestRoundedAllocationSize(ProcessInfo &pInfo, int sock_fd, size_t num_bytes);

        std::string DebugString() const;
        std::string Name() { return name; }
        static std::string EndPoint() { return endpointName; }
        CUcontext ctx(void) { return ctx_; }

        static const char name[128];
        static const char endpointName[128];
        static const char barrierName[128];

        

    private:
        MemMapManager();
        void Server();

        M3InternalErrorType Register(ProcessInfo &pInfo);
        M3InternalErrorType Allocate(ProcessInfo &pInfo, size_t alignment, size_t num_bytes, std::vector<shareable_handle_t> &shHandle, std::vector<CUmemGenericAllocationHandle> &allocHandle);
        M3InternalErrorType DeAllocate(ProcessInfo &pInfo, shareable_handle_t shHandle);
        size_t GetRoundedAllocationSize(size_t num_bytes);

        static MemMapManager * instance_;
        static std::once_flag singletonFlag_;

        std::vector<CUdevice> devices_;
        int device_count_;
        CUcontext ctx_;

        int ipc_sock_fd_;
        std::vector<ProcessInfo> subscribers_;

        std::unordered_map<std::string, shareable_handle_t> memIdToShHandle_;
        std::unordered_map<shareable_handle_t, std::string> shHandletoMemId_;        


};

static struct sockaddr_un server_addr = { 
    AF_UNIX,
    "MemMapManager_Server_EndPoint"
};

void panic(const char * msg);
static int sendShareableHandle(int sock_fd, struct sockaddr_un * client_addr, shareable_handle_t shHandle);
static int recvShareableHandle(int sock_fd, shareable_handle_t *shHandle);
