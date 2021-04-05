#include "MemMapManager.h"

int main() {
    CUUTIL_ERRCHK(cuInit(0));
    CUcontext ctx;
    CUdevice device = 0;
    ProcessInfo pInfo;
    CUUTIL_ERRCHK(cuCtxCreate(&ctx, 0, device));
    pInfo.SetContext(ctx);

    struct sockaddr_un client_addr;
    client_addr.sun_family = AF_UNIX;
    strncpy(client_addr.sun_path, "m3shell_ipc", sizeof(client_addr.sun_path));

    MemMapRequest req;
    MemMapResponse res;
    char cmd[128];
    while (true) {
        unlink("m3shell_ipc");
        printf("> ");
        scanf("%s", cmd);

        if(!strcmp(cmd, "q") || !strcmp(cmd, "quit") || !strcmp(cmd, "exit")) {
            printf("Bye\n");
            break;
        }

        if (!strcmp(cmd, "h") || !strcmp(cmd, "help")) {
            printf("Memory Map Manager testing shell\n");
            printf("PLEASE MAKE SURE THAT THE SERVER PROGRAM IS RUNNING.\n");
            printf("Usage:\n");
            printf("halt : Halts the M3 server\n");
            printf("echo <N> : Sends ECHO command and receives ACK for N times\n");
            printf("alloc <memory id> <factor> <g | m | k> : Allocates <factor> <g | m | k> bytes of GPU memory with ID <memory id>\n");
            printf("exit: exits the shell\n");
            printf("help: prints out this help message\n");
        }

        if(!strcmp(cmd, "halt")) {
            int sock_fd = ipcOpenAndBindSocket(&client_addr);
            ipcHaltM3Server(sock_fd, pInfo);
            close(sock_fd);
        }

        if(!strcmp(cmd, "echo")) {
            int rep;
            scanf("%d", &rep);
            
            int sock_fd = ipcOpenAndBindSocket(&client_addr);
            req.src = pInfo;
            req.cmd = CMD_ECHO;
            for(int i = 0; i < rep; ++i) {
                res = MemMapManager::Request(sock_fd, req, &server_addr);
                if(res.status != STATUSCODE_ACK) {
                    printf("Failed to Echo\n");
                    continue;
                } else {
                    printf("Echo back # %d\n", i);
                }
            }
            close(sock_fd);
            continue;
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