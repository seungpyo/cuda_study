/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef __TOPO_H
#define __TOPO_H

enum pci_distance { PATH_PIX = 0, PATH_PXB = 1, PATH_PHB = 2, PATH_NODE = 3, PATH_SYS = 4 };

static int getNumaId(char *path) {
    char npath[PATH_MAX];
    snprintf(npath, PATH_MAX, "%s/numa_node", path);
    npath[PATH_MAX - 1] = '\0';

    int numaId = -1;
    FILE *file = fopen(npath, "r");
    if (file == NULL) return -1;
    if (fscanf(file, "%d", &numaId) == EOF) {
        fclose(file);
        return -1;
    }
    fclose(file);

    return numaId;
}

int nvshmemi_detect_same_device(nvshmemi_state_t *state);
int nvshmemi_build_transport_map(nvshmemi_state_t *state);
int get_device_by_distance(int *device, nvshmemi_state_t *state, struct nvshmem_transport *tcurr);

#endif
