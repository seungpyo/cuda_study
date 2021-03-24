/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include "util.h"
#include "nvshmemx_error.h"
#include "bootstrap.h"
#include "bootstrap_internal.h"

#define GET_SYMBOL(lib_handle, name, var, status)                                              \
    do {                                                                                       \
        void **var_ptr = (void **)&(var);                                                      \
        void *tmp = (void *)dlsym(lib_handle, name);                                           \
        NULL_ERROR_JMP(tmp, status, NVSHMEMX_ERROR_INTERNAL, out,                              \
                       "Bootstrap failed to get symbol '%s'\n\t%s\n", name, dlerror());        \
        *var_ptr = tmp;                                                                        \
    } while (0)

static void *plugin_hdl;
static char *plugin_name;

static int (*bootstrap_plugin_init)(bootstrap_handle_t *handle);
static int (*bootstrap_plugin_finalize)(bootstrap_handle_t *handle);

static int bootstrap_loader_finalize(bootstrap_handle_t *handle) {
    int status = bootstrap_plugin_finalize(handle);

    if (status != 0)
        ERROR_PRINT("Bootstrap plugin finalize failed for '%s'\n", plugin_name);

    dlclose(plugin_hdl);
    free(plugin_name);

    return 0;
}

int bootstrap_loader_init(const char *plugin, bootstrap_handle_t *handle) {
    int status = 0;

    dlerror(); /* Clear any existing error */
    plugin_name = strdup(plugin);
    plugin_hdl  = dlopen(plugin, RTLD_NOW);
    NULL_ERROR_JMP(plugin_hdl, status, -1, error, "Bootstrap unable to load '%s'\n\t%s\n", plugin, dlerror());

    dlerror(); /* Clear any existing error */
    GET_SYMBOL(plugin_hdl, "nvshmemi_bootstrap_plugin_init", bootstrap_plugin_init, status);
    GET_SYMBOL(plugin_hdl, "nvshmemi_bootstrap_plugin_finalize", bootstrap_plugin_finalize, status);

    handle->finalize = bootstrap_loader_finalize;

    status = bootstrap_plugin_init(handle);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, error, "Bootstrap plugin init failed for '%s'\n", plugin);

    goto out;

error:
    if (plugin_hdl != NULL)
        dlclose(plugin_hdl);
    if (plugin_name)
        free(plugin_name);

out:
    return status;
}
