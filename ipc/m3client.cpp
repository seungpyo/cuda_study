#include "MemMapManager.h"

int main() {
    CUUTIL_ERRCHK(cuInit(0));
    CUcontext ctx;
    CUdevice device = 0;
    ProcessInfo pInfo;
    CUUTIL_ERRCHK(cuCtxCreate(&ctx, 0, device));
    pInfo.SetContext(ctx);

    MemMapRequest req;
    MemMapResponse res;
    char cmd[128];
    while (true) {
        printf("> ");
        scanf("%s", cmd);
        printf("GOT: %s\n", cmd);
        if(!strcmp(cmd, "q")) {
            printf("Bye\n");
            break;
        }

        if(!strcmp(cmd, "alloc")) {
            char memId[MAX_MEMID_LEN];
            scanf("%s", memId);
            uint32_t factor;
            scanf("%u", &factor);
            char unit[8];
            scanf("%s", &unit);

            size_t num_bytes = factor;
            switch(unit[0]) {
                case 'g':
                    num_bytes <<= 10;
                case 'm':
                    num_bytes <<= 10;
                case 'k':
                    num_bytes <<= 10;
                    break;
                default:
                    printf("Invalid unit %c!\n", unit);
                    continue;
            }
            struct sockaddr_un client_addr;
            client_addr.sun_family = AF_UNIX;
            strncpy(client_addr.sun_path, "m3shell_ipc", 108);
            int sock_fd = ipcOpenAndBindSocket(&client_addr);
            res = MemMapManager::RequestRoundedAllocationSize(pInfo, sock_fd, num_bytes);
            if(res.status != STATUSCODE_ACK) {
                printf("Failed to round %u bytes.\n", num_bytes);
                continue;
            }
            num_bytes = res.roundedSize;
            res = MemMapManager::RequestAllocate(pInfo, sock_fd, memId, 1024, num_bytes);
            if(res.status != STATUSCODE_ACK) {
                printf("Failed to allocate %u bytes.\n", num_bytes);
                continue;
            }
            printf("Successfully allocated %lu bytes @ [%p : %p].\n", num_bytes, res.d_ptr, res.d_ptr + num_bytes -1);
            close(sock_fd);
        }

    }

    return 0;
}