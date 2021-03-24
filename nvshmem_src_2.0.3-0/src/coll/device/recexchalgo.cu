/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "gpu_coll.h"

/* This is a utility function for the recursive exchange algorithm.
 * It calculates the PEs to which data is sent or received from during the different steps
 * of the algorithm. The recursive exchange algorithm is divided into 3 steps.
 * In step 1, when the number of PEs are not a power of k, some PEs send their data
 * to other PEs that participate in the power-of-k recursive exchange algorithm in Step 2.
 * Step 2 has log_k (PE_size) phases. In each phase k PEs exchange data with each other.
 * In Step 3, PEs from step 2 send the final reduced data to PEs that did not participate in Step 2.
*/
__host__ void nvshmemi_recexchalgo_get_neighbors(int my_pe, int num_pes) {
    int i, j, k;
    int p_of_k = 1, log_p_of_k = 0, rem, T, newpe;
    int step1_sendto, step1_nrecvs, step2_nphases;
    INFO(NVSHMEM_COLL, "step 1 nbr calculation started, num_pes = %d", num_pes);

    k = gpu_coll_env_params_var.reduce_recexch_kval;
    if (num_pes < k) /* If size of the active set is less than k, reduce the value of k */
        k = (num_pes > 2) ? num_pes : 2;

    /* Calculate p_of_k, p_of_k is the largest power of k that is less than num_pes */
    while (p_of_k <= num_pes) {
        p_of_k *= k;
        log_p_of_k++;
    }
    p_of_k /= k;
    log_p_of_k--;

    step2_nphases = log_p_of_k;
    int *step1_recvfrom = (int *)malloc(sizeof(int) * (k - 1));
    int **step2_nbrs = (int **)malloc(sizeof(int *) * step2_nphases);
    for (int i = 0; i < step2_nphases; i++) {
        step2_nbrs[i] = (int *)malloc(sizeof(int) * (k - 1));
    }

    rem = num_pes - p_of_k;
    /* rem is the number of PEs that do not particpate in Step 2
     * We need to identify these non-participating PEs. This is done in the following way.
     * The first T PEs are divided into sets of k consecutive PEs each.
     * In each of these sets, the first k-1 PEs are the non-participating
     * PEs while the last PE is the participating PE.
     * The non-participating PEs send their data to the participating PE
     * in their corresponding set.
     */
    T = (rem * k) / (k - 1);

    INFO(NVSHMEM_COLL, "step 1 nbr calculation started. T is %d", T);
    step1_nrecvs = 0;
    step1_sendto = -1;

    /* Step 1 */
    if (my_pe < T) {
        if (my_pe % k != (k - 1)) {                    /* I am a non-participating PE */
            step1_sendto = my_pe + (k - 1 - my_pe % k); /* partipating PE to send the data to */
            /* if the corresponding participating PE is not in T,
             * then send to the Tth PE to preserve non-commutativity */
            if (step1_sendto > T - 1) step1_sendto = T;
            newpe = -1; /* tag this PE as non-participating */
        } else {          /* participating PE */
            for (i = 0; i < k - 1; i++) {
                step1_recvfrom[i] = my_pe - i - 1;
            }
            step1_nrecvs = k - 1;
            newpe = my_pe / k; /* this is the new PE amongst the set of participating PEs */
        }
    } else { /* PE >= T */
        newpe = my_pe - rem;

        if (my_pe == T && (T - 1) % k != k - 1 && T >= 1) {
            int nsenders = (T - 1) % k + 1; /* number of PEs sending their data to me in Step 1 */

            for (j = nsenders - 1; j >= 0; j--) {
                step1_recvfrom[nsenders - 1 - j] = T - nsenders + j;
            }
            step1_nrecvs = nsenders;
        }
    }

    INFO(NVSHMEM_COLL, "step 1 nbr computation completed");

    /* Step 2 */
    if (step1_sendto == -1) { /* calulate step2_nbrs only for participating PEs */
        int *digit = (int *)malloc(sizeof(int) * step2_nphases);
        assert(digit != NULL);
        int temppe = newpe;
        int mask = 0x1;
        int phase = 0, cbit, cnt, nbr, power;

        /* calculate the digits in base k representation of newpe */
        for (i = 0; i < log_p_of_k; i++) digit[i] = 0;

        int remainder, i_digit = 0;
        while (temppe != 0) {
            remainder = temppe % k;
            temppe = temppe / k;
            digit[i_digit] = remainder;
            i_digit++;
        }

        while (mask < p_of_k) {
            cbit =
                digit[phase]; /* phase_th digit changes in this phase, obtain its original value */
            cnt = 0;
            for (i = 0; i < k; i++) { /* there are k-1 neighbors */
                if (i != cbit) {      /* do not generate yourself as your nieighbor */
                    digit[phase] = i; /* this gets us the base k representation of the neighbor */

                    /* calculate the base 10 value of the neighbor PE */
                    nbr = 0;
                    power = 1;
                    for (j = 0; j < log_p_of_k; j++) {
                        nbr += digit[j] * power;
                        power *= k;
                    }

                    /* calculate its real PE and store it */
                    step2_nbrs[phase][cnt] =
                        (nbr < rem / (k - 1)) ? (nbr * k) + (k - 1) : nbr + rem;
                    cnt++;
                }
            }
            INFO(NVSHMEM_COLL, "step 2, phase %d nbr calculation completed", phase);
            digit[phase] = cbit; /* reset the digit to original value */
            phase++;
            mask *= k;
        }
    }

    // Copy the data to device memory
    cudaMemcpyToSymbol(reduce_recexch_step1_sendto_d, &step1_sendto, sizeof(int));
    void *dev_ptr;
    cudaMemcpyFromSymbol(&dev_ptr, reduce_recexch_step1_recvfrom_d, sizeof(int *));
    cuMemcpyHtoD((CUdeviceptr)dev_ptr, step1_recvfrom, sizeof(int) * step1_nrecvs);
    cudaMemcpyToSymbol(reduce_recexch_step1_nrecvs_d, &step1_nrecvs, sizeof(int));

    cudaMemcpyFromSymbol(&dev_ptr, reduce_recexch_step2_nbrs_d, sizeof(int **));
    void *dev_ptr_2;
    for (int i = 0; i < step2_nphases; i++) {
        cuMemcpyDtoH(&dev_ptr_2, (CUdeviceptr)((int **)dev_ptr + i), sizeof(int *));
        cudaDeviceSynchronize();
        cuMemcpyHtoD((CUdeviceptr)dev_ptr_2, step2_nbrs[i], sizeof(int) * (k - 1));
        cudaDeviceSynchronize();
    }
    cudaMemcpyToSymbol(reduce_recexch_step2_nphases_d, &step2_nphases, sizeof(int));
}
