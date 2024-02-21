/*
 * MIT License
 *
 * Copyright (c) 2019-2024 Tskit Developers
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "testlib.h"
#include <tskit/stats.h>

#include <unistd.h>
#include <stdlib.h>
#include <float.h>

static bool
multi_mutations_exist(tsk_treeseq_t *ts, tsk_id_t start, tsk_id_t end)
{
    int ret;
    tsk_id_t j;
    tsk_site_t site;

    for (j = start; j < TSK_MIN((tsk_id_t) tsk_treeseq_get_num_sites(ts), end); j++) {
        ret = tsk_treeseq_get_site(ts, j, &site);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        if (site.mutations_length > 1) {
            return true;
        }
    }
    return false;
}

static void
verify_ld(tsk_treeseq_t *ts)
{
    int ret;
    tsk_size_t num_sites = tsk_treeseq_get_num_sites(ts);
    tsk_site_t *sites = tsk_malloc(num_sites * sizeof(tsk_site_t));
    int *num_site_mutations = tsk_malloc(num_sites * sizeof(int));
    tsk_ld_calc_t ld_calc;
    double *r2, *r2_prime, x;
    tsk_id_t j;
    tsk_size_t num_r2_values;
    double eps = 1e-6;

    r2 = tsk_calloc(num_sites, sizeof(double));
    r2_prime = tsk_calloc(num_sites, sizeof(double));
    CU_ASSERT_FATAL(r2 != NULL);
    CU_ASSERT_FATAL(r2_prime != NULL);
    CU_ASSERT_FATAL(sites != NULL);
    CU_ASSERT_FATAL(num_site_mutations != NULL);

    ret = tsk_ld_calc_init(&ld_calc, ts);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tsk_ld_calc_print_state(&ld_calc, _devnull);

    for (j = 0; j < (tsk_id_t) num_sites; j++) {
        ret = tsk_treeseq_get_site(ts, j, sites + j);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        num_site_mutations[j] = (int) sites[j].mutations_length;
        ret = tsk_ld_calc_get_r2(&ld_calc, j, j, &x);
        if (num_site_mutations[j] <= 1) {
            CU_ASSERT_EQUAL_FATAL(ret, 0);
            CU_ASSERT_DOUBLE_EQUAL_FATAL(x, 1.0, eps);
        } else {
            CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_ONLY_INFINITE_SITES);
        }
    }

    if (num_sites > 0) {
        /* Some checks in the forward direction */
        ret = tsk_ld_calc_get_r2_array(
            &ld_calc, 0, TSK_DIR_FORWARD, num_sites, DBL_MAX, r2, &num_r2_values);
        if (multi_mutations_exist(ts, 0, (tsk_id_t) num_sites)) {
            CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_ONLY_INFINITE_SITES);
        } else {
            CU_ASSERT_EQUAL_FATAL(ret, 0);
            CU_ASSERT_EQUAL_FATAL(num_r2_values, num_sites - 1);
        }
        tsk_ld_calc_print_state(&ld_calc, _devnull);

        ret = tsk_ld_calc_get_r2_array(&ld_calc, (tsk_id_t) num_sites - 2,
            TSK_DIR_FORWARD, num_sites, DBL_MAX, r2_prime, &num_r2_values);
        if (multi_mutations_exist(ts, (tsk_id_t) num_sites - 2, (tsk_id_t) num_sites)) {
            CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_ONLY_INFINITE_SITES);
        } else {
            CU_ASSERT_EQUAL_FATAL(ret, 0);
            CU_ASSERT_EQUAL_FATAL(num_r2_values, 1);
        }
        tsk_ld_calc_print_state(&ld_calc, _devnull);

        ret = tsk_ld_calc_get_r2_array(
            &ld_calc, 0, TSK_DIR_FORWARD, num_sites, DBL_MAX, r2_prime, &num_r2_values);
        if (multi_mutations_exist(ts, 0, (tsk_id_t) num_sites)) {
            CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_ONLY_INFINITE_SITES);
        } else {
            CU_ASSERT_EQUAL_FATAL(ret, 0);
            CU_ASSERT_EQUAL_FATAL(num_r2_values, num_sites - 1);
            for (j = 0; j < (tsk_id_t) num_r2_values; j++) {
                CU_ASSERT_EQUAL_FATAL(r2[j], r2_prime[j]);
                ret = tsk_ld_calc_get_r2(&ld_calc, 0, j + 1, &x);
                CU_ASSERT_EQUAL_FATAL(ret, 0);
                CU_ASSERT_DOUBLE_EQUAL_FATAL(r2[j], x, eps);
            }
        }

        /* Some checks in the reverse direction */
        ret = tsk_ld_calc_get_r2_array(&ld_calc, (tsk_id_t) num_sites - 1,
            TSK_DIR_REVERSE, num_sites, DBL_MAX, r2, &num_r2_values);
        if (multi_mutations_exist(ts, 0, (tsk_id_t) num_sites)) {
            CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_ONLY_INFINITE_SITES);
        } else {
            CU_ASSERT_EQUAL_FATAL(ret, 0);
            CU_ASSERT_EQUAL_FATAL(num_r2_values, num_sites - 1);
        }
        tsk_ld_calc_print_state(&ld_calc, _devnull);

        ret = tsk_ld_calc_get_r2_array(
            &ld_calc, 1, TSK_DIR_REVERSE, num_sites, DBL_MAX, r2_prime, &num_r2_values);
        if (multi_mutations_exist(ts, 0, 2)) {
            CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_ONLY_INFINITE_SITES);
        } else {
            CU_ASSERT_EQUAL_FATAL(ret, 0);
            CU_ASSERT_EQUAL_FATAL(num_r2_values, 1);
        }

        ret = tsk_ld_calc_get_r2_array(&ld_calc, (tsk_id_t) num_sites - 1,
            TSK_DIR_REVERSE, num_sites, DBL_MAX, r2_prime, &num_r2_values);
        if (multi_mutations_exist(ts, 0, (tsk_id_t) num_sites)) {
            CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_ONLY_INFINITE_SITES);
        } else {
            CU_ASSERT_EQUAL_FATAL(ret, 0);
            CU_ASSERT_EQUAL_FATAL(num_r2_values, num_sites - 1);
            tsk_ld_calc_print_state(&ld_calc, _devnull);

            for (j = 0; j < (tsk_id_t) num_r2_values; j++) {
                CU_ASSERT_EQUAL_FATAL(r2[j], r2_prime[j]);
                ret = tsk_ld_calc_get_r2(&ld_calc, (tsk_id_t) num_sites - 1,
                    (tsk_id_t) num_sites - j - 2, &x);
                CU_ASSERT_EQUAL_FATAL(ret, 0);
                CU_ASSERT_DOUBLE_EQUAL_FATAL(r2[j], x, eps);
            }
        }

        /* Check some error conditions */
        ret = tsk_ld_calc_get_r2_array(
            &ld_calc, 0, 0, num_sites, DBL_MAX, r2, &num_r2_values);
        CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    }

    /* Check some error conditions */
    for (j = (tsk_id_t) num_sites; j < (tsk_id_t) num_sites + 2; j++) {
        ret = tsk_ld_calc_get_r2_array(
            &ld_calc, j, TSK_DIR_FORWARD, num_sites, DBL_MAX, r2, &num_r2_values);
        CU_ASSERT_EQUAL(ret, TSK_ERR_SITE_OUT_OF_BOUNDS);
        ret = tsk_ld_calc_get_r2(&ld_calc, j, 0, r2);
        CU_ASSERT_EQUAL(ret, TSK_ERR_SITE_OUT_OF_BOUNDS);
        ret = tsk_ld_calc_get_r2(&ld_calc, 0, j, r2);
        CU_ASSERT_EQUAL(ret, TSK_ERR_SITE_OUT_OF_BOUNDS);
    }

    tsk_ld_calc_free(&ld_calc);
    free(r2);
    free(r2_prime);
    free(sites);
    free(num_site_mutations);
}

/* FIXME: this test is weak and should check the return value somehow.
 * We should also have simplest and single tree tests along with separate
 * tests for the error conditions. This should be done as part of the general
 * stats framework.
 */
static void
verify_genealogical_nearest_neighbours(tsk_treeseq_t *ts)
{
    int ret;
    const tsk_id_t *samples;
    const tsk_id_t *sample_sets[2];
    tsk_size_t sample_set_size[2];
    tsk_size_t num_samples = tsk_treeseq_get_num_samples(ts);
    double *A = tsk_malloc(2 * num_samples * sizeof(double));
    CU_ASSERT_FATAL(A != NULL);

    samples = tsk_treeseq_get_samples(ts);

    sample_sets[0] = samples;
    sample_set_size[0] = num_samples / 2;
    sample_sets[1] = samples + sample_set_size[0];
    sample_set_size[1] = num_samples - sample_set_size[0];

    ret = tsk_treeseq_genealogical_nearest_neighbours(
        ts, samples, num_samples, sample_sets, sample_set_size, 2, 0, A);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    sample_sets[0] = samples;
    sample_set_size[0] = 1;
    sample_sets[1] = samples + 1;
    sample_set_size[1] = 1;

    ret = tsk_treeseq_genealogical_nearest_neighbours(
        ts, samples, num_samples, sample_sets, sample_set_size, 2, 0, A);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    free(A);
}

/* FIXME: this test is weak and should check the return value somehow.
 * We should also have simplest and single tree tests along with separate
 * tests for the error conditions. This should be done as part of the general
 * stats framework.
 */
static void
verify_mean_descendants(tsk_treeseq_t *ts)
{
    int ret;
    tsk_id_t *samples;
    const tsk_id_t *sample_sets[2];
    tsk_size_t sample_set_size[2];
    tsk_size_t num_samples = tsk_treeseq_get_num_samples(ts);
    double *C = tsk_malloc(2 * tsk_treeseq_get_num_nodes(ts) * sizeof(double));
    CU_ASSERT_FATAL(C != NULL);

    samples = tsk_malloc(num_samples * sizeof(*samples));
    tsk_memcpy(samples, tsk_treeseq_get_samples(ts), num_samples * sizeof(*samples));

    sample_sets[0] = samples;
    sample_set_size[0] = num_samples / 2;
    sample_sets[1] = samples + sample_set_size[0];
    sample_set_size[1] = num_samples - sample_set_size[0];

    ret = tsk_treeseq_mean_descendants(ts, sample_sets, sample_set_size, 2, 0, C);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* Check some error conditions */
    ret = tsk_treeseq_mean_descendants(ts, sample_sets, sample_set_size, 0, 0, C);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    samples[0] = -1;
    ret = tsk_treeseq_mean_descendants(ts, sample_sets, sample_set_size, 2, 0, C);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_NODE_OUT_OF_BOUNDS);
    samples[0] = (tsk_id_t) tsk_treeseq_get_num_nodes(ts) + 1;
    ret = tsk_treeseq_mean_descendants(ts, sample_sets, sample_set_size, 2, 0, C);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_NODE_OUT_OF_BOUNDS);

    free(samples);
    free(C);
}

/* Check the divergence matrix by running against the stats API equivalent
 * code.
 */
static void
verify_divergence_matrix(tsk_treeseq_t *ts, tsk_flags_t options)
{
    int ret;
    const tsk_size_t n = tsk_treeseq_get_num_samples(ts);
    const tsk_id_t *samples = tsk_treeseq_get_samples(ts);
    tsk_size_t sample_set_sizes[n];
    tsk_id_t index_tuples[2 * n * n];
    double D1[n * n], D2[n * n];
    tsk_size_t i, j, k;

    for (j = 0; j < n; j++) {
        sample_set_sizes[j] = 1;
        for (k = 0; k < n; k++) {
            index_tuples[2 * (j * n + k)] = (tsk_id_t) j;
            index_tuples[2 * (j * n + k) + 1] = (tsk_id_t) k;
        }
    }
    ret = tsk_treeseq_divergence(
        ts, n, sample_set_sizes, samples, n * n, index_tuples, 0, NULL, options, D1);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_treeseq_divergence_matrix(
        ts, n, sample_set_sizes, samples, 0, NULL, options, D2);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    for (j = 0; j < n; j++) {
        for (k = 0; k < n; k++) {
            i = j * n + k;
            /* printf("%d\t%d\t%f\t%f\n", (int) j, (int) k, D1[i], D2[i]); */
            if (j == k) {
                CU_ASSERT_EQUAL(D2[i], 0);
            } else {
                CU_ASSERT_DOUBLE_EQUAL(D1[i], D2[i], 1E-6);
            }
        }
    }
}

typedef struct {
    int call_count;
    int error_on;
    int error_code;
} general_stat_error_params_t;

static int
general_stat_error(tsk_size_t TSK_UNUSED(K), const double *TSK_UNUSED(X), tsk_size_t M,
    double *Y, void *params)
{
    int ret = 0;
    CU_ASSERT_FATAL(M == 1);
    Y[0] = 0;
    general_stat_error_params_t *the_params = (general_stat_error_params_t *) params;
    if (the_params->call_count == the_params->error_on) {
        ret = the_params->error_code;
    }
    the_params->call_count++;
    return ret;
}

static void
verify_window_errors(tsk_treeseq_t *ts, tsk_flags_t mode)
{
    int ret;
    tsk_size_t num_samples = tsk_treeseq_get_num_samples(ts);
    double *W = tsk_calloc(num_samples, sizeof(double));
    /* node mode requires this much space at least */
    double *sigma = tsk_calloc(tsk_treeseq_get_num_nodes(ts), sizeof(double));
    double windows[] = { 0, 0, 0 };
    tsk_flags_t options = mode;

    /* Window errors */
    ret = tsk_treeseq_general_stat(
        ts, 1, W, 1, general_stat_error, NULL, 0, windows, options, sigma);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_NUM_WINDOWS);

    ret = tsk_treeseq_general_stat(
        ts, 1, W, 1, general_stat_error, NULL, 2, windows, options, sigma);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_WINDOWS);

    windows[0] = -1;
    ret = tsk_treeseq_general_stat(
        ts, 1, W, 1, general_stat_error, NULL, 2, windows, options, sigma);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_WINDOWS);

    windows[1] = -1;
    ret = tsk_treeseq_general_stat(
        ts, 1, W, 1, general_stat_error, NULL, 1, windows, options, sigma);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_WINDOWS);

    windows[0] = 10;
    ret = tsk_treeseq_general_stat(
        ts, 1, W, 1, general_stat_error, NULL, 2, windows, options, sigma);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_WINDOWS);

    windows[0] = 0;
    windows[2] = tsk_treeseq_get_sequence_length(ts) + 1;
    ret = tsk_treeseq_general_stat(
        ts, 1, W, 1, general_stat_error, NULL, 2, windows, options, sigma);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_WINDOWS);

    windows[0] = 0;
    windows[1] = -1;
    windows[2] = tsk_treeseq_get_sequence_length(ts);
    ret = tsk_treeseq_general_stat(
        ts, 1, W, 1, general_stat_error, NULL, 2, windows, options, sigma);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_WINDOWS);

    free(W);
    free(sigma);
}

static void
verify_summary_func_errors(tsk_treeseq_t *ts, tsk_flags_t mode)
{
    int ret;
    tsk_size_t num_samples = tsk_treeseq_get_num_samples(ts);
    double *W = tsk_calloc(num_samples, sizeof(double));
    /* We need this much space for NODE mode */
    double *sigma = tsk_calloc(tsk_treeseq_get_num_nodes(ts), sizeof(double));
    int j;
    general_stat_error_params_t params;
    CU_ASSERT_FATAL(W != NULL);

    /* Errors in the summary function */
    j = 1;
    while (true) {
        params.call_count = 0;
        params.error_on = j;
        params.error_code = -j;
        ret = tsk_treeseq_general_stat(ts, 1, W, 1, general_stat_error, &params, 0, NULL,
            TSK_STAT_POLARISED | mode, sigma);
        if (ret == 0) {
            break;
        }
        CU_ASSERT_EQUAL_FATAL(ret, params.error_code);
        j++;
    }
    CU_ASSERT_FATAL(j > 1);

    j = 1;
    while (true) {
        params.call_count = 0;
        params.error_on = j;
        params.error_code = -j;
        ret = tsk_treeseq_general_stat(
            ts, 1, W, 1, general_stat_error, &params, 0, NULL, mode, sigma);
        if (ret == 0) {
            break;
        }
        CU_ASSERT_EQUAL_FATAL(ret, params.error_code);
        j++;
    }
    CU_ASSERT_FATAL(j > 1);

    free(W);
    free(sigma);
}

static void
verify_branch_general_stat_errors(tsk_treeseq_t *ts)
{
    verify_summary_func_errors(ts, TSK_STAT_BRANCH);
    verify_window_errors(ts, TSK_STAT_BRANCH);
}

static void
verify_site_general_stat_errors(tsk_treeseq_t *ts)
{
    verify_window_errors(ts, TSK_STAT_SITE);
    verify_summary_func_errors(ts, TSK_STAT_SITE);
}

static void
verify_node_general_stat_errors(tsk_treeseq_t *ts)
{
    verify_window_errors(ts, TSK_STAT_NODE);
    verify_summary_func_errors(ts, TSK_STAT_NODE);
}

static void
verify_one_way_weighted_func_errors(tsk_treeseq_t *ts, one_way_weighted_method *method)
{
    int ret;
    tsk_size_t num_samples = tsk_treeseq_get_num_samples(ts);
    double *weights = tsk_malloc(num_samples * sizeof(double));
    double bad_windows[] = { 0, -1 };
    double result;
    tsk_size_t j;

    for (j = 0; j < num_samples; j++) {
        weights[j] = 1.0;
    }

    ret = method(ts, 0, weights, 0, NULL, 0, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_INSUFFICIENT_WEIGHTS);

    ret = method(ts, 1, weights, 1, bad_windows, 0, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_WINDOWS);

    free(weights);
}

static void
verify_one_way_weighted_covariate_func_errors(
    tsk_treeseq_t *ts, one_way_covariates_method *method)
{
    int ret;
    tsk_size_t num_samples = tsk_treeseq_get_num_samples(ts);
    double *weights = tsk_malloc(num_samples * sizeof(double));
    double *covariates = NULL;
    double bad_windows[] = { 0, -1 };
    double result;
    tsk_size_t j;

    for (j = 0; j < num_samples; j++) {
        weights[j] = 1.0;
    }

    ret = method(ts, 0, weights, 0, covariates, 0, NULL, 0, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_INSUFFICIENT_WEIGHTS);

    ret = method(ts, 1, weights, 0, covariates, 1, bad_windows, 0, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_WINDOWS);

    free(weights);
}

static void
verify_one_way_stat_func_errors(tsk_treeseq_t *ts, one_way_sample_stat_method *method)
{
    int ret;
    tsk_id_t num_nodes = (tsk_id_t) tsk_treeseq_get_num_nodes(ts);
    tsk_id_t samples[] = { 0, 1, 2, 3 };
    tsk_size_t sample_set_sizes = 4;
    double windows[] = { 0, 0, 0 };
    double result;

    ret = method(ts, 0, &sample_set_sizes, samples, 0, NULL, 0, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_INSUFFICIENT_SAMPLE_SETS);

    samples[0] = TSK_NULL;
    ret = method(ts, 1, &sample_set_sizes, samples, 0, NULL, 0, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_NODE_OUT_OF_BOUNDS);
    samples[0] = -10;
    ret = method(ts, 1, &sample_set_sizes, samples, 0, NULL, 0, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_NODE_OUT_OF_BOUNDS);
    samples[0] = num_nodes;
    ret = method(ts, 1, &sample_set_sizes, samples, 0, NULL, 0, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_NODE_OUT_OF_BOUNDS);
    samples[0] = num_nodes + 1;
    ret = method(ts, 1, &sample_set_sizes, samples, 0, NULL, 0, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_NODE_OUT_OF_BOUNDS);

    samples[0] = num_nodes - 1;
    ret = method(ts, 1, &sample_set_sizes, samples, 0, NULL, 0, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_SAMPLES);

    samples[0] = 1;
    ret = method(ts, 1, &sample_set_sizes, samples, 0, NULL, 0, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_DUPLICATE_SAMPLE);

    samples[0] = 0;
    sample_set_sizes = 0;
    ret = method(ts, 1, &sample_set_sizes, samples, 0, NULL, 0, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_EMPTY_SAMPLE_SET);

    sample_set_sizes = 4;
    /* Window errors */
    ret = method(ts, 1, &sample_set_sizes, samples, 0, windows, 0, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_NUM_WINDOWS);

    ret = method(ts, 1, &sample_set_sizes, samples, 2, windows, 0, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_WINDOWS);
}

static void
verify_two_way_stat_func_errors(tsk_treeseq_t *ts, general_sample_stat_method *method)
{
    int ret;
    tsk_id_t samples[] = { 0, 1, 2, 3 };
    tsk_size_t sample_set_sizes[] = { 2, 2 };
    tsk_id_t set_indexes[] = { 0, 1 };
    double result;

    ret = method(ts, 0, sample_set_sizes, samples, 1, set_indexes, 0, NULL,
        TSK_STAT_SITE, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_INSUFFICIENT_SAMPLE_SETS);
    ret = method(ts, 1, sample_set_sizes, samples, 1, set_indexes, 0, NULL,
        TSK_STAT_SITE, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_INSUFFICIENT_SAMPLE_SETS);

    ret = method(ts, 2, sample_set_sizes, samples, 0, set_indexes, 0, NULL,
        TSK_STAT_SITE, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_INSUFFICIENT_INDEX_TUPLES);

    set_indexes[0] = -1;
    ret = method(ts, 2, sample_set_sizes, samples, 1, set_indexes, 0, NULL,
        TSK_STAT_SITE, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_SAMPLE_SET_INDEX);
    set_indexes[0] = 0;
    set_indexes[1] = 2;
    ret = method(ts, 2, sample_set_sizes, samples, 1, set_indexes, 0, NULL,
        TSK_STAT_SITE, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_SAMPLE_SET_INDEX);
}

static void
verify_two_way_weighted_stat_func_errors(
    tsk_treeseq_t *ts, two_way_weighted_method *method)
{
    int ret;
    tsk_id_t indexes[] = { 0, 0, 0, 1 };
    double bad_windows[] = { -1, -1 };
    double weights[10];
    double result[10];

    memset(weights, 0, sizeof(weights));

    ret = method(ts, 2, weights, 2, indexes, 0, NULL, result, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = method(ts, 0, weights, 2, indexes, 0, NULL, result, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_INSUFFICIENT_WEIGHTS);

    ret = method(ts, 2, weights, 2, indexes, 1, bad_windows, result, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_WINDOWS);
}

static void
verify_three_way_stat_func_errors(tsk_treeseq_t *ts, general_sample_stat_method *method)
{
    int ret;
    tsk_id_t samples[] = { 0, 1, 2, 3 };
    tsk_size_t sample_set_sizes[] = { 1, 1, 2 };
    tsk_id_t set_indexes[] = { 0, 1, 2 };
    double result;

    ret = method(ts, 0, sample_set_sizes, samples, 1, set_indexes, 0, NULL,
        TSK_STAT_SITE, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_INSUFFICIENT_SAMPLE_SETS);
    ret = method(ts, 1, sample_set_sizes, samples, 1, set_indexes, 0, NULL,
        TSK_STAT_SITE, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_INSUFFICIENT_SAMPLE_SETS);
    ret = method(ts, 2, sample_set_sizes, samples, 1, set_indexes, 0, NULL,
        TSK_STAT_SITE, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_INSUFFICIENT_SAMPLE_SETS);

    ret = method(ts, 3, sample_set_sizes, samples, 0, set_indexes, 0, NULL,
        TSK_STAT_SITE, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_INSUFFICIENT_INDEX_TUPLES);

    set_indexes[0] = -1;
    ret = method(ts, 3, sample_set_sizes, samples, 1, set_indexes, 0, NULL,
        TSK_STAT_SITE, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_SAMPLE_SET_INDEX);
    set_indexes[0] = 0;
    set_indexes[1] = 3;
    ret = method(ts, 3, sample_set_sizes, samples, 1, set_indexes, 0, NULL,
        TSK_STAT_SITE, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_SAMPLE_SET_INDEX);
}

static void
verify_four_way_stat_func_errors(tsk_treeseq_t *ts, general_sample_stat_method *method)
{
    int ret;
    tsk_id_t samples[] = { 0, 1, 2, 3 };
    tsk_size_t sample_set_sizes[] = { 1, 1, 1, 1 };
    tsk_id_t set_indexes[] = { 0, 1, 2, 3 };
    double result;

    ret = method(ts, 0, sample_set_sizes, samples, 1, set_indexes, 0, NULL,
        TSK_STAT_SITE, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_INSUFFICIENT_SAMPLE_SETS);
    ret = method(ts, 1, sample_set_sizes, samples, 1, set_indexes, 0, NULL,
        TSK_STAT_SITE, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_INSUFFICIENT_SAMPLE_SETS);
    ret = method(ts, 2, sample_set_sizes, samples, 1, set_indexes, 0, NULL,
        TSK_STAT_SITE, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_INSUFFICIENT_SAMPLE_SETS);
    ret = method(ts, 3, sample_set_sizes, samples, 1, set_indexes, 0, NULL,
        TSK_STAT_SITE, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_INSUFFICIENT_SAMPLE_SETS);

    ret = method(ts, 4, sample_set_sizes, samples, 0, set_indexes, 0, NULL,
        TSK_STAT_SITE, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_INSUFFICIENT_INDEX_TUPLES);

    set_indexes[0] = -1;
    ret = method(ts, 4, sample_set_sizes, samples, 1, set_indexes, 0, NULL,
        TSK_STAT_SITE, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_SAMPLE_SET_INDEX);
    set_indexes[0] = 0;
    set_indexes[1] = 4;
    ret = method(ts, 4, sample_set_sizes, samples, 1, set_indexes, 0, NULL,
        TSK_STAT_SITE, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_SAMPLE_SET_INDEX);
}

static int
general_stat_identity(
    tsk_size_t K, const double *restrict X, tsk_size_t M, double *Y, void *params)
{
    tsk_size_t k;
    CU_ASSERT_FATAL(M == K);
    CU_ASSERT_FATAL(params == NULL);

    for (k = 0; k < K; k++) {
        Y[k] = X[k];
    }
    return 0;
}

static void
verify_branch_general_stat_identity(tsk_treeseq_t *ts)
{
    CU_ASSERT_FATAL(ts != NULL);

    int ret;
    tsk_size_t num_samples = tsk_treeseq_get_num_samples(ts);
    double *W = tsk_malloc(num_samples * sizeof(double));
    tsk_id_t *nodes = tsk_malloc(tsk_treeseq_get_num_nodes(ts) * sizeof(*nodes));
    tsk_id_t u;
    tsk_size_t num_nodes;
    double s, branch_length;
    double *sigma = tsk_malloc(tsk_treeseq_get_num_trees(ts) * sizeof(*sigma));
    tsk_tree_t tree;
    tsk_size_t j;
    CU_ASSERT_FATAL(W != NULL);
    CU_ASSERT_FATAL(nodes != NULL);

    for (j = 0; j < num_samples; j++) {
        W[j] = 1;
    }

    ret = tsk_treeseq_general_stat(ts, 1, W, 1, general_stat_identity, NULL,
        tsk_treeseq_get_num_trees(ts), tsk_treeseq_get_breakpoints(ts),
        TSK_STAT_BRANCH | TSK_STAT_POLARISED | TSK_STAT_SPAN_NORMALISE, sigma);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_tree_init(&tree, ts, 0);
    CU_ASSERT_EQUAL(ret, 0);

    for (ret = tsk_tree_first(&tree); ret == TSK_TREE_OK; ret = tsk_tree_next(&tree)) {
        ret = tsk_tree_preorder(&tree, nodes, &num_nodes);
        CU_ASSERT_EQUAL_FATAL(ret, 0);

        s = 0;
        for (j = 0; j < num_nodes; j++) {
            u = nodes[j];
            ret = tsk_tree_get_branch_length(&tree, u, &branch_length);
            CU_ASSERT_EQUAL_FATAL(ret, 0);
            s += branch_length * (double) tree.num_samples[u];
        }
        CU_ASSERT_DOUBLE_EQUAL_FATAL(sigma[tree.index], s, 1e-6);
    }
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    free(nodes);
    tsk_tree_free(&tree);
    free(W);
    free(sigma);
}

static int
general_stat_sum(
    tsk_size_t K, const double *restrict X, tsk_size_t M, double *Y, void *params)
{
    tsk_size_t k, m;
    double s = 0;
    CU_ASSERT_FATAL(params == NULL);

    s = 0;
    for (k = 0; k < K; k++) {
        s += X[k];
    }
    for (m = 0; m < M; m++) {
        Y[m] = s;
    }
    return 0;
}

static void
verify_general_stat_dims(
    tsk_treeseq_t *ts, tsk_size_t K, tsk_size_t M, tsk_flags_t options)
{
    int ret;
    tsk_size_t num_samples = tsk_treeseq_get_num_samples(ts);
    double *W = tsk_malloc(K * num_samples * sizeof(double));
    /* We need this much space for NODE mode; no harm for other modes. */
    double *sigma = tsk_calloc(tsk_treeseq_get_num_nodes(ts) * M, sizeof(double));
    tsk_size_t j, k;
    CU_ASSERT_FATAL(W != NULL);

    for (j = 0; j < num_samples; j++) {
        for (k = 0; k < K; k++) {
            W[j * K + k] = 1;
        }
    }
    ret = tsk_treeseq_general_stat(
        ts, K, W, M, general_stat_sum, NULL, 0, NULL, options, sigma);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    free(W);
    free(sigma);
}

static void
verify_general_stat_windows(
    tsk_treeseq_t *ts, tsk_size_t num_windows, tsk_flags_t options)
{
    int ret;
    tsk_size_t num_samples = tsk_treeseq_get_num_samples(ts);
    double *W = tsk_malloc(num_samples * sizeof(double));
    tsk_size_t M = 5;
    /* We need this much space for NODE mode; no harm for other modes. */
    double *sigma
        = tsk_calloc(M * tsk_treeseq_get_num_nodes(ts) * num_windows, sizeof(double));
    double *windows = tsk_malloc((num_windows + 1) * sizeof(*windows));
    double L = tsk_treeseq_get_sequence_length(ts);
    tsk_size_t j;
    CU_ASSERT_FATAL(W != NULL);
    CU_ASSERT_FATAL(sigma != NULL);
    CU_ASSERT_FATAL(windows != NULL);

    for (j = 0; j < num_samples; j++) {
        W[j] = 1;
    }
    windows[0] = 0;
    windows[num_windows] = L;
    for (j = 1; j < num_windows; j++) {
        windows[j] = ((double) j) * L / (double) num_windows;
    }
    ret = tsk_treeseq_general_stat(
        ts, 1, W, M, general_stat_sum, NULL, num_windows, windows, options, sigma);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    free(W);
    free(sigma);
    free(windows);
}

static void
verify_default_general_stat(tsk_treeseq_t *ts)
{
    int ret;
    tsk_size_t K = 2;
    tsk_size_t M = 1;
    tsk_size_t num_samples = tsk_treeseq_get_num_samples(ts);
    double *W = tsk_malloc(K * num_samples * sizeof(double));
    double sigma1, sigma2;
    tsk_size_t j, k;
    CU_ASSERT_FATAL(W != NULL);

    for (j = 0; j < num_samples; j++) {
        for (k = 0; k < K; k++) {
            W[j * K + k] = 1;
        }
    }
    ret = tsk_treeseq_general_stat(
        ts, K, W, M, general_stat_sum, NULL, 0, NULL, TSK_STAT_SITE, &sigma1);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_treeseq_general_stat(
        ts, K, W, M, general_stat_sum, NULL, 0, NULL, 0, &sigma2);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(sigma1, sigma2);
    free(W);
}

static void
verify_general_stat(tsk_treeseq_t *ts, tsk_flags_t mode)
{
    CU_ASSERT_FATAL(ts != NULL);
    verify_general_stat_dims(ts, 4, 2, mode);
    verify_general_stat_dims(ts, 4, 2, mode | TSK_STAT_POLARISED);
    verify_general_stat_dims(ts, 1, 20, mode);
    verify_general_stat_dims(ts, 1, 20, mode | TSK_STAT_POLARISED);
    verify_general_stat_dims(ts, 100, 1, mode);
    verify_general_stat_dims(ts, 100, 1, mode | TSK_STAT_POLARISED);
    verify_general_stat_dims(ts, 10, 12, mode);
    verify_general_stat_dims(ts, 10, 12, mode | TSK_STAT_POLARISED);
    verify_general_stat_windows(ts, 1, mode);
    verify_general_stat_windows(ts, 1, mode | TSK_STAT_SPAN_NORMALISE);
    verify_general_stat_windows(ts, 2, mode);
    verify_general_stat_windows(ts, 2, mode | TSK_STAT_SPAN_NORMALISE);
    verify_general_stat_windows(ts, 3, mode);
    verify_general_stat_windows(ts, 3, mode | TSK_STAT_SPAN_NORMALISE);
    verify_general_stat_windows(ts, 10, mode);
    verify_general_stat_windows(ts, 10, mode | TSK_STAT_SPAN_NORMALISE);
    verify_general_stat_windows(ts, 100, mode);
    verify_general_stat_windows(ts, 100, mode | TSK_STAT_SPAN_NORMALISE);
}

static void
verify_afs(tsk_treeseq_t *ts)
{
    int ret;
    tsk_size_t n = tsk_treeseq_get_num_samples(ts);
    tsk_size_t sample_set_sizes[2];
    const tsk_id_t *samples = tsk_treeseq_get_samples(ts);
    double *result = tsk_malloc(n * n * sizeof(*result));

    CU_ASSERT_FATAL(sample_set_sizes != NULL);

    sample_set_sizes[0] = n - 2;
    sample_set_sizes[1] = 2;
    ret = tsk_treeseq_allele_frequency_spectrum(
        ts, 2, sample_set_sizes, samples, 0, NULL, 0, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_treeseq_allele_frequency_spectrum(
        ts, 2, sample_set_sizes, samples, 0, NULL, TSK_STAT_POLARISED, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_treeseq_allele_frequency_spectrum(ts, 2, sample_set_sizes, samples, 0,
        NULL, TSK_STAT_POLARISED | TSK_STAT_SPAN_NORMALISE, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_treeseq_allele_frequency_spectrum(ts, 2, sample_set_sizes, samples, 0,
        NULL, TSK_STAT_BRANCH | TSK_STAT_POLARISED | TSK_STAT_SPAN_NORMALISE, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_treeseq_allele_frequency_spectrum(ts, 2, sample_set_sizes, samples, 0,
        NULL, TSK_STAT_BRANCH | TSK_STAT_SPAN_NORMALISE, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    free(result);
}

static void
test_general_stat_input_errors(void)
{
    tsk_treeseq_t ts;
    double result;
    double W;
    int ret;

    tsk_treeseq_from_text(&ts, 1, single_tree_ex_nodes, single_tree_ex_edges, NULL,
        single_tree_ex_sites, single_tree_ex_mutations, NULL, NULL, 0);

    /* Bad input dimensions */
    ret = tsk_treeseq_general_stat(
        &ts, 0, &W, 1, general_stat_sum, NULL, 0, NULL, 0, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_STATE_DIMS);

    ret = tsk_treeseq_general_stat(
        &ts, 1, &W, 0, general_stat_sum, NULL, 0, NULL, 0, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_RESULT_DIMS);

    /* Multiple stats*/
    ret = tsk_treeseq_general_stat(&ts, 1, &W, 1, general_stat_sum, NULL, 0, NULL,
        TSK_STAT_SITE | TSK_STAT_BRANCH, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_MULTIPLE_STAT_MODES);
    ret = tsk_treeseq_general_stat(&ts, 1, &W, 1, general_stat_sum, NULL, 0, NULL,
        TSK_STAT_SITE | TSK_STAT_NODE, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_MULTIPLE_STAT_MODES);
    ret = tsk_treeseq_general_stat(&ts, 1, &W, 1, general_stat_sum, NULL, 0, NULL,
        TSK_STAT_BRANCH | TSK_STAT_NODE, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_MULTIPLE_STAT_MODES);

    tsk_treeseq_free(&ts);
}

static void
test_empty_ts_ld(void)
{
    tsk_treeseq_t ts;

    tsk_treeseq_from_text(
        &ts, 1, single_tree_ex_nodes, "", NULL, NULL, NULL, NULL, NULL, 0);

    verify_ld(&ts);
    tsk_treeseq_free(&ts);
}

static void
test_empty_ts_mean_descendants(void)
{
    tsk_treeseq_t ts;

    tsk_treeseq_from_text(
        &ts, 1, single_tree_ex_nodes, "", NULL, NULL, NULL, NULL, NULL, 0);
    verify_mean_descendants(&ts);
    tsk_treeseq_free(&ts);
}

static void
test_empty_ts_genealogical_nearest_neighbours(void)
{
    tsk_treeseq_t ts;

    tsk_treeseq_from_text(
        &ts, 1, single_tree_ex_nodes, "", NULL, NULL, NULL, NULL, NULL, 0);
    verify_genealogical_nearest_neighbours(&ts);
    tsk_treeseq_free(&ts);
}

static void
test_empty_ts_general_stat(void)
{
    tsk_treeseq_t ts;

    tsk_treeseq_from_text(
        &ts, 1, single_tree_ex_nodes, "", NULL, NULL, NULL, NULL, NULL, 0);
    verify_branch_general_stat_identity(&ts);
    verify_default_general_stat(&ts);
    verify_general_stat(&ts, TSK_STAT_BRANCH);
    verify_general_stat(&ts, TSK_STAT_SITE);
    verify_general_stat(&ts, TSK_STAT_NODE);
    tsk_treeseq_free(&ts);
}

static void
test_empty_ts_afs(void)
{
    tsk_treeseq_t ts;

    tsk_treeseq_from_text(
        &ts, 1, single_tree_ex_nodes, "", NULL, NULL, NULL, NULL, NULL, 0);
    verify_afs(&ts);
    tsk_treeseq_free(&ts);
}

static void
test_single_tree_ld(void)
{
    tsk_treeseq_t ts;

    tsk_treeseq_from_text(&ts, 1, single_tree_ex_nodes, single_tree_ex_edges, NULL,
        single_tree_ex_sites, single_tree_ex_mutations, NULL, NULL, 0);
    verify_ld(&ts);
    tsk_treeseq_free(&ts);
}

static void
test_single_tree_mean_descendants(void)
{
    tsk_treeseq_t ts;

    tsk_treeseq_from_text(&ts, 1, single_tree_ex_nodes, single_tree_ex_edges, NULL,
        single_tree_ex_sites, single_tree_ex_mutations, NULL, NULL, 0);
    verify_mean_descendants(&ts);
    tsk_treeseq_free(&ts);
}

static void
test_single_tree_genealogical_nearest_neighbours(void)
{
    tsk_treeseq_t ts;

    tsk_treeseq_from_text(&ts, 1, single_tree_ex_nodes, single_tree_ex_edges, NULL,
        single_tree_ex_sites, single_tree_ex_mutations, NULL, NULL, 0);
    verify_genealogical_nearest_neighbours(&ts);
    tsk_treeseq_free(&ts);
}

static void
test_single_tree_general_stat(void)
{
    tsk_treeseq_t ts;

    tsk_treeseq_from_text(&ts, 1, single_tree_ex_nodes, single_tree_ex_edges, NULL,
        single_tree_ex_sites, single_tree_ex_mutations, NULL, NULL, 0);
    verify_branch_general_stat_identity(&ts);
    verify_default_general_stat(&ts);
    verify_general_stat(&ts, TSK_STAT_BRANCH);
    verify_general_stat(&ts, TSK_STAT_SITE);
    verify_general_stat(&ts, TSK_STAT_NODE);
    tsk_treeseq_free(&ts);
}

static void
test_single_tree_general_stat_errors(void)
{
    tsk_treeseq_t ts;

    tsk_treeseq_from_text(&ts, 1, single_tree_ex_nodes, single_tree_ex_edges, NULL,
        single_tree_ex_sites, single_tree_ex_mutations, NULL, NULL, 0);
    verify_branch_general_stat_errors(&ts);
    verify_site_general_stat_errors(&ts);
    verify_node_general_stat_errors(&ts);
    tsk_treeseq_free(&ts);
}

static void
test_single_tree_divergence_matrix(void)
{
    tsk_treeseq_t ts;
    int ret;
    double result[16];
    double D_branch[16] = { 0, 2, 6, 6, 2, 0, 6, 6, 6, 6, 0, 4, 6, 6, 4, 0 };
    double D_site[16] = { 0, 1, 1, 0, 1, 0, 2, 1, 1, 2, 0, 1, 0, 1, 1, 0 };

    tsk_size_t sample_set_sizes[] = { 2, 2 };

    tsk_treeseq_from_text(&ts, 1, single_tree_ex_nodes, single_tree_ex_edges, NULL,
        single_tree_ex_sites, single_tree_ex_mutations, NULL, NULL, 0);

    ret = tsk_treeseq_divergence_matrix(
        &ts, 0, NULL, NULL, 0, NULL, TSK_STAT_BRANCH, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(16, result, D_branch);

    ret = tsk_treeseq_divergence_matrix(
        &ts, 0, NULL, NULL, 0, NULL, TSK_STAT_SITE, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(16, result, D_site);

    ret = tsk_treeseq_divergence_matrix(
        &ts, 2, sample_set_sizes, NULL, 0, NULL, TSK_STAT_SITE, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_treeseq_divergence_matrix(
        &ts, 2, sample_set_sizes, NULL, 0, NULL, TSK_STAT_BRANCH, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    sample_set_sizes[0] = 3;
    sample_set_sizes[1] = 1;
    ret = tsk_treeseq_divergence_matrix(
        &ts, 2, sample_set_sizes, NULL, 0, NULL, TSK_STAT_BRANCH, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_treeseq_divergence_matrix(
        &ts, 2, sample_set_sizes, NULL, 0, NULL, TSK_STAT_SITE, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* assert_arrays_almost_equal(4, result, D_site); */

    verify_divergence_matrix(&ts, TSK_STAT_BRANCH);
    verify_divergence_matrix(&ts, TSK_STAT_BRANCH | TSK_STAT_SPAN_NORMALISE);
    verify_divergence_matrix(&ts, TSK_STAT_SITE);
    verify_divergence_matrix(&ts, TSK_STAT_SITE | TSK_STAT_SPAN_NORMALISE);

    tsk_treeseq_free(&ts);
}

static void
test_single_tree_divergence_matrix_internal_samples(void)
{
    tsk_treeseq_t ts;
    int ret;
    double *result = malloc(16 * sizeof(double));
    double D[16] = { 0, 2, 4, 3, 2, 0, 4, 3, 4, 4, 0, 1, 3, 3, 1, 0 };

    const char *nodes = "1  0   -1   -1\n" /* 2.00┊    6    ┊ */
                        "1  0   -1   -1\n" /*     ┊  ┏━┻━┓  ┊ */
                        "1  0   -1   -1\n" /* 1.00┊  4   5* ┊ */
                        "0  0   -1   -1\n" /*     ┊ ┏┻┓ ┏┻┓ ┊ */
                        "0  1   -1   -1\n" /* 0.00┊ 0 1 2 3 ┊ */
                        "1  1   -1   -1\n" /*     0 * * *   1 */
                        "0  2   -1   -1\n";
    const char *edges = "0  1   4   0,1\n"
                        "0  1   5   2,3\n"
                        "0  1   6   4,5\n";
    /* One mutations per branch so we get the same as the branch length value */
    const char *sites = "0.1  A\n"
                        "0.2  A\n"
                        "0.3  A\n"
                        "0.4  A\n"
                        "0.5  A\n"
                        "0.6  A\n";
    const char *mutations = "0  0  T  -1\n"
                            "1  1  T  -1\n"
                            "2  2  T  -1\n"
                            "3  3  T  -1\n"
                            "4  4  T  -1\n"
                            "5  5  T  -1\n";
    tsk_id_t samples[] = { 0, 1, 2, 5 };
    tsk_size_t sizes[] = { 1, 1, 1, 1 };

    tsk_treeseq_from_text(&ts, 1, nodes, edges, NULL, sites, mutations, NULL, NULL, 0);

    ret = tsk_treeseq_divergence_matrix(
        &ts, 0, NULL, NULL, 0, NULL, TSK_STAT_BRANCH, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(16, result, D);
    ret = tsk_treeseq_divergence_matrix(
        &ts, 0, NULL, NULL, 0, NULL, TSK_STAT_SITE, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(16, result, D);

    ret = tsk_treeseq_divergence_matrix(
        &ts, 4, sizes, samples, 0, NULL, TSK_STAT_BRANCH, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(16, result, D);
    ret = tsk_treeseq_divergence_matrix(
        &ts, 4, sizes, samples, 0, NULL, TSK_STAT_SITE, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(16, result, D);

    ret = tsk_treeseq_divergence_matrix(
        &ts, 4, NULL, samples, 0, NULL, TSK_STAT_BRANCH, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(16, result, D);
    ret = tsk_treeseq_divergence_matrix(
        &ts, 4, NULL, samples, 0, NULL, TSK_STAT_SITE, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(16, result, D);

    verify_divergence_matrix(&ts, TSK_STAT_BRANCH);
    verify_divergence_matrix(&ts, TSK_STAT_BRANCH | TSK_STAT_SPAN_NORMALISE);
    verify_divergence_matrix(&ts, TSK_STAT_SITE);
    verify_divergence_matrix(&ts, TSK_STAT_SITE | TSK_STAT_SPAN_NORMALISE);

    tsk_treeseq_free(&ts);
    free(result);
}

static void
test_single_tree_divergence_matrix_multi_root(void)
{
    tsk_treeseq_t ts;
    int ret;
    double result[16];
    double D_branch[16] = { 0, 2, 3, 3, 2, 0, 3, 3, 3, 3, 0, 4, 3, 3, 4, 0 };

    const char *nodes = "1  0   -1   -1\n"
                        "1  0   -1   -1\n"  /* 2.00┊      5  ┊ */
                        "1  0   -1   -1\n"  /* 1.00┊  4      ┊ */
                        "1  0   -1   -1\n"  /*     ┊ ┏┻┓ ┏┻┓ ┊ */
                        "0  1   -1   -1\n"  /* 0.00┊ 0 1 2 3 ┊ */
                        "0  2   -1   -1\n"; /*     0 * * * * 1 */
    const char *edges = "0  1   4   0,1\n"
                        "0  1   5   2,3\n";
    /* Two mutations per branch */
    const char *sites = "0.1  A\n"
                        "0.2  A\n"
                        "0.3  A\n"
                        "0.4  A\n";
    const char *mutations = "0  0  B  -1\n"
                            "0  0  C  0\n"
                            "1  1  B  -1\n"
                            "1  1  C  2\n"
                            "2  2  B  -1\n"
                            "2  2  C  4\n"
                            "2  2  D  5\n"
                            "2  2  E  6\n"
                            "3  3  B  -1\n"
                            "3  3  C  8\n"
                            "3  3  D  9\n"
                            "3  3  E  10\n";

    tsk_treeseq_from_text(&ts, 1, nodes, edges, NULL, sites, mutations, NULL, NULL, 0);

    ret = tsk_treeseq_divergence_matrix(
        &ts, 0, NULL, NULL, 0, NULL, TSK_STAT_BRANCH, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(16, result, D_branch);

    verify_divergence_matrix(&ts, TSK_STAT_BRANCH);
    verify_divergence_matrix(&ts, TSK_STAT_BRANCH | TSK_STAT_SPAN_NORMALISE);
    verify_divergence_matrix(&ts, TSK_STAT_SITE);
    verify_divergence_matrix(&ts, TSK_STAT_SITE | TSK_STAT_SPAN_NORMALISE);

    tsk_treeseq_free(&ts);
}

static void
test_paper_ex_ld(void)
{
    tsk_treeseq_t ts;
    tsk_ld_calc_t ld_calc;
    double r2[3];
    tsk_size_t num_r2_values;
    int ret;

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);
    verify_ld(&ts);

    /* Check early exit corner cases */
    ret = tsk_ld_calc_init(&ld_calc, &ts);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_ld_calc_get_r2_array(
        &ld_calc, 0, TSK_DIR_FORWARD, 1, DBL_MAX, r2, &num_r2_values);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(num_r2_values, 1);

    ret = tsk_ld_calc_get_r2_array(
        &ld_calc, 2, TSK_DIR_REVERSE, 1, DBL_MAX, r2, &num_r2_values);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(num_r2_values, 1);

    tsk_ld_calc_free(&ld_calc);
    tsk_treeseq_free(&ts);
}

static void
test_paper_ex_mean_descendants(void)
{
    tsk_treeseq_t ts;

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);
    verify_mean_descendants(&ts);
    tsk_treeseq_free(&ts);
}

static void
test_paper_ex_genealogical_nearest_neighbours(void)
{
    tsk_treeseq_t ts;

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);
    verify_genealogical_nearest_neighbours(&ts);
    tsk_treeseq_free(&ts);
}

static void
test_paper_ex_general_stat(void)
{
    tsk_treeseq_t ts;

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);
    verify_branch_general_stat_identity(&ts);
    verify_default_general_stat(&ts);
    verify_general_stat(&ts, TSK_STAT_BRANCH);
    verify_general_stat(&ts, TSK_STAT_SITE);
    verify_general_stat(&ts, TSK_STAT_NODE);
    tsk_treeseq_free(&ts);
}

static void
test_paper_ex_general_stat_errors(void)
{
    tsk_treeseq_t ts;

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);
    verify_branch_general_stat_errors(&ts);
    verify_site_general_stat_errors(&ts);
    verify_node_general_stat_errors(&ts);
    tsk_treeseq_free(&ts);
}

static void
test_paper_ex_diversity_errors(void)
{
    tsk_treeseq_t ts;

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);
    verify_one_way_stat_func_errors(&ts, tsk_treeseq_diversity);
    tsk_treeseq_free(&ts);
}

static void
test_paper_ex_diversity(void)
{
    tsk_treeseq_t ts;
    tsk_id_t samples[] = { 0, 1, 2, 3 };
    tsk_size_t sample_set_sizes = 4;
    double pi;
    int ret;

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);

    ret = tsk_treeseq_diversity(
        &ts, 1, &sample_set_sizes, samples, 0, NULL, TSK_STAT_SITE, &pi);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_DOUBLE_EQUAL_FATAL(pi, 1.5, 1e-6);

    /* A sample set size of 1 leads to NaN */
    sample_set_sizes = 1;
    ret = tsk_treeseq_diversity(
        &ts, 1, &sample_set_sizes, samples, 0, NULL, TSK_STAT_SITE, &pi);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT(tsk_isnan(pi));

    tsk_treeseq_free(&ts);
}

static void
test_paper_ex_trait_covariance_errors(void)
{
    tsk_treeseq_t ts;

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);
    verify_one_way_weighted_func_errors(&ts, tsk_treeseq_trait_covariance);
    tsk_treeseq_free(&ts);
}

static void
test_paper_ex_trait_covariance(void)
{
    tsk_treeseq_t ts;
    double result;
    double *weights;
    tsk_size_t j;
    int ret;

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);
    weights = tsk_malloc(4 * sizeof(double));
    weights[0] = weights[1] = 0.0;
    weights[2] = weights[3] = 1.0;

    ret = tsk_treeseq_trait_covariance(&ts, 1, weights, 0, NULL, TSK_STAT_SITE, &result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_DOUBLE_EQUAL_FATAL(result, 1.0 / 12.0, 1e-6);

    /* weights of 0 leads to 0 */
    for (j = 0; j < 4; j++) {
        weights[j] = 0.0;
    }
    ret = tsk_treeseq_trait_covariance(&ts, 1, weights, 0, NULL, TSK_STAT_SITE, &result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_DOUBLE_EQUAL_FATAL(result, 0.0, 1e-6);

    tsk_treeseq_free(&ts);
    free(weights);
}

static void
test_paper_ex_trait_correlation_errors(void)
{
    tsk_treeseq_t ts;

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);
    verify_one_way_weighted_func_errors(&ts, tsk_treeseq_trait_correlation);
    tsk_treeseq_free(&ts);
}

static void
test_paper_ex_trait_correlation(void)
{
    tsk_treeseq_t ts;
    double result;
    double *weights;
    int ret;

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);
    weights = tsk_malloc(4 * sizeof(double));
    weights[0] = weights[1] = 0.0;
    weights[2] = weights[3] = 1.0;

    ret = tsk_treeseq_trait_correlation(
        &ts, 1, weights, 0, NULL, TSK_STAT_SITE, &result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_DOUBLE_EQUAL_FATAL(result, 1.0, 1e-6);

    tsk_treeseq_free(&ts);
    free(weights);
}

static void
test_paper_ex_trait_linear_model_errors(void)
{
    tsk_treeseq_t ts;

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);
    verify_one_way_weighted_covariate_func_errors(&ts, tsk_treeseq_trait_linear_model);
    tsk_treeseq_free(&ts);
}

static void
test_paper_ex_trait_linear_model(void)
{
    tsk_treeseq_t ts;
    double result;
    double *weights;
    double *covariates;
    int ret;

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);
    weights = tsk_malloc(4 * sizeof(double));
    covariates = tsk_malloc(8 * sizeof(double));
    weights[0] = weights[1] = 0.0;
    weights[2] = weights[3] = 1.0;
    covariates[0] = covariates[1] = 0.0;
    covariates[2] = covariates[3] = 1.0;
    covariates[4] = covariates[6] = 0.0;
    covariates[5] = covariates[7] = 1.0;

    ret = tsk_treeseq_trait_linear_model(
        &ts, 1, weights, 2, covariates, 0, NULL, TSK_STAT_SITE, &result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_DOUBLE_EQUAL_FATAL(result, 0.0, 1e-6);

    tsk_treeseq_free(&ts);
    free(weights);
    free(covariates);
}

static void
test_paper_ex_segregating_sites_errors(void)
{
    tsk_treeseq_t ts;

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);
    verify_one_way_stat_func_errors(&ts, tsk_treeseq_segregating_sites);
    tsk_treeseq_free(&ts);
}

static void
test_paper_ex_segregating_sites(void)
{
    tsk_treeseq_t ts;
    tsk_id_t samples[] = { 0, 1, 2, 3 };
    tsk_size_t sample_set_sizes = 4;
    double segsites;
    int ret;

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);

    ret = tsk_treeseq_segregating_sites(
        &ts, 1, &sample_set_sizes, samples, 0, NULL, TSK_STAT_SITE, &segsites);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_DOUBLE_EQUAL_FATAL(segsites, 3.0, 1e-6);

    /* A sample set size of 1 leads to 0 */
    sample_set_sizes = 1;
    ret = tsk_treeseq_segregating_sites(
        &ts, 1, &sample_set_sizes, samples, 0, NULL, TSK_STAT_SITE, &segsites);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_DOUBLE_EQUAL_FATAL(segsites, 0.0, 1e-6);

    tsk_treeseq_free(&ts);
}

static void
test_paper_ex_Y1_errors(void)
{
    tsk_treeseq_t ts;

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);
    verify_one_way_stat_func_errors(&ts, tsk_treeseq_Y1);
    tsk_treeseq_free(&ts);
}

static void
test_paper_ex_Y1(void)
{
    tsk_treeseq_t ts;
    tsk_id_t samples[] = { 0, 1, 2, 3 };
    tsk_size_t sample_set_sizes = 4;
    double result;
    int ret;

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);

    ret = tsk_treeseq_Y1(&ts, 1, &sample_set_sizes, samples, 0, NULL, 0, &result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* A sample set size of < 2 leads to NaN */
    sample_set_sizes = 1;
    ret = tsk_treeseq_Y1(&ts, 1, &sample_set_sizes, samples, 0, NULL, 0, &result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT(tsk_isnan(result));

    tsk_treeseq_free(&ts);
}

static void
test_paper_ex_divergence_errors(void)
{
    tsk_treeseq_t ts;

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);
    verify_two_way_stat_func_errors(&ts, tsk_treeseq_divergence);
    tsk_treeseq_free(&ts);
}

static void
test_paper_ex_divergence(void)
{
    tsk_treeseq_t ts;
    tsk_id_t samples[] = { 0, 1, 2, 3 };
    tsk_size_t sample_set_sizes[] = { 2, 2 };
    tsk_id_t set_indexes[] = { 0, 1 };
    double result;
    int ret;

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);

    ret = tsk_treeseq_divergence(&ts, 2, sample_set_sizes, samples, 1, set_indexes, 0,
        NULL, TSK_STAT_SITE, &result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* sample_set[0] size = 1 with indexes = (0, 0) leads to NaN */
    sample_set_sizes[0] = 1;
    set_indexes[1] = 0;
    ret = tsk_treeseq_divergence(&ts, 2, sample_set_sizes, samples, 1, set_indexes, 0,
        NULL, TSK_STAT_SITE, &result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT(tsk_isnan(result));

    tsk_treeseq_free(&ts);
}

static void
test_paper_ex_genetic_relatedness(void)
{
    tsk_treeseq_t ts;
    tsk_id_t samples[] = { 0, 1, 2, 3 };
    tsk_size_t sample_set_sizes[] = { 2, 2 };
    tsk_id_t set_indexes[] = { 0, 0 };
    double result;
    int ret;

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);

    ret = tsk_treeseq_genetic_relatedness(&ts, 2, sample_set_sizes, samples, 1,
        set_indexes, 0, NULL, TSK_STAT_SITE, &result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tsk_treeseq_free(&ts);
}

static void
test_paper_ex_genetic_relatedness_errors(void)
{
    tsk_treeseq_t ts;

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);
    verify_two_way_stat_func_errors(&ts, tsk_treeseq_genetic_relatedness);
    tsk_treeseq_free(&ts);
}

static void
test_paper_ex_genetic_relatedness_weighted(void)
{
    tsk_treeseq_t ts;
    double weights[] = { 1.2, 0.1, 0.0, 0.0, 3.4, 5.0, 1.0, -1.0 };
    tsk_id_t indexes[] = { 0, 0, 0, 1 };
    double result[100];
    tsk_size_t num_weights;
    int ret;

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);

    for (num_weights = 1; num_weights < 3; num_weights++) {
        ret = tsk_treeseq_genetic_relatedness_weighted(
            &ts, num_weights, weights, 2, indexes, 0, NULL, result, TSK_STAT_SITE);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        ret = tsk_treeseq_genetic_relatedness_weighted(
            &ts, num_weights, weights, 2, indexes, 0, NULL, result, TSK_STAT_BRANCH);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        ret = tsk_treeseq_genetic_relatedness_weighted(
            &ts, num_weights, weights, 2, indexes, 0, NULL, result, TSK_STAT_NODE);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
    }

    tsk_treeseq_free(&ts);
}

static void
test_paper_ex_genetic_relatedness_weighted_errors(void)
{
    tsk_treeseq_t ts;

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);
    verify_two_way_weighted_stat_func_errors(
        &ts, tsk_treeseq_genetic_relatedness_weighted);
    tsk_treeseq_free(&ts);
}

static void
test_paper_ex_Y2_errors(void)
{
    tsk_treeseq_t ts;

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);
    verify_two_way_stat_func_errors(&ts, tsk_treeseq_Y2);
    tsk_treeseq_free(&ts);
}

static void
test_paper_ex_Y2(void)
{
    tsk_treeseq_t ts;
    tsk_id_t samples[] = { 0, 1, 2, 3 };
    tsk_size_t sample_set_sizes[] = { 2, 2 };
    tsk_id_t set_indexes[] = { 0, 1 };
    double result;
    int ret;

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);

    ret = tsk_treeseq_Y2(&ts, 2, sample_set_sizes, samples, 1, set_indexes, 0, NULL,
        TSK_STAT_SITE, &result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* sample_set_size of 1 leads to NaN */
    sample_set_sizes[1] = 1;
    ret = tsk_treeseq_Y2(&ts, 2, sample_set_sizes, samples, 1, set_indexes, 0, NULL,
        TSK_STAT_SITE, &result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT(tsk_isnan(result));

    tsk_treeseq_free(&ts);
}

static void
test_paper_ex_f2_errors(void)
{
    tsk_treeseq_t ts;

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);
    verify_two_way_stat_func_errors(&ts, tsk_treeseq_f2);
    tsk_treeseq_free(&ts);
}

static void
test_paper_ex_f2(void)
{
    tsk_treeseq_t ts;
    tsk_id_t samples[] = { 0, 1, 2, 3 };
    tsk_size_t sample_set_sizes[] = { 2, 2 };
    tsk_id_t set_indexes[] = { 0, 1 };
    double result;
    int ret;

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);

    ret = tsk_treeseq_f2(&ts, 2, sample_set_sizes, samples, 1, set_indexes, 0, NULL,
        TSK_STAT_SITE, &result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* sample_set_size of 1 leads to NaN */
    sample_set_sizes[0] = 1;
    ret = tsk_treeseq_f2(&ts, 2, sample_set_sizes, samples, 1, set_indexes, 0, NULL,
        TSK_STAT_SITE, &result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT(tsk_isnan(result));

    /* sample_set_size of 1 leads to NaN */
    sample_set_sizes[0] = 2;
    sample_set_sizes[1] = 1;
    ret = tsk_treeseq_f2(&ts, 2, sample_set_sizes, samples, 1, set_indexes, 0, NULL,
        TSK_STAT_SITE, &result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT(tsk_isnan(result));

    tsk_treeseq_free(&ts);
}

static void
test_paper_ex_Y3_errors(void)
{
    tsk_treeseq_t ts;

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);
    verify_three_way_stat_func_errors(&ts, tsk_treeseq_Y3);
    tsk_treeseq_free(&ts);
}

static void
test_paper_ex_Y3(void)
{
    tsk_treeseq_t ts;
    tsk_id_t samples[] = { 0, 1, 2, 3 };
    tsk_size_t sample_set_sizes[] = { 2, 1, 1 };
    tsk_id_t set_indexes[] = { 0, 1, 2 };
    double result;
    int ret;

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);

    ret = tsk_treeseq_Y3(&ts, 3, sample_set_sizes, samples, 1, set_indexes, 0, NULL,
        TSK_STAT_SITE, &result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    tsk_treeseq_free(&ts);
}

static void
test_paper_ex_f3_errors(void)
{
    tsk_treeseq_t ts;

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);
    verify_three_way_stat_func_errors(&ts, tsk_treeseq_f3);
    tsk_treeseq_free(&ts);
}

static void
test_paper_ex_f3(void)
{
    tsk_treeseq_t ts;
    tsk_id_t samples[] = { 0, 1, 2, 3 };
    tsk_size_t sample_set_sizes[] = { 2, 1, 1 };
    tsk_id_t set_indexes[] = { 0, 1, 2 };
    double result;
    int ret;

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);

    ret = tsk_treeseq_f3(&ts, 3, sample_set_sizes, samples, 1, set_indexes, 0, NULL,
        TSK_STAT_SITE, &result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* sample_set_size of 1 leads to NaN */
    sample_set_sizes[0] = 1;
    ret = tsk_treeseq_f3(&ts, 3, sample_set_sizes, samples, 1, set_indexes, 0, NULL,
        TSK_STAT_SITE, &result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT(tsk_isnan(result));

    tsk_treeseq_free(&ts);
}

static void
test_paper_ex_f4_errors(void)
{
    tsk_treeseq_t ts;

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);
    verify_four_way_stat_func_errors(&ts, tsk_treeseq_f4);
    tsk_treeseq_free(&ts);
}

static void
test_paper_ex_f4(void)
{
    tsk_treeseq_t ts;
    tsk_id_t samples[] = { 0, 1, 2, 3 };
    tsk_size_t sample_set_sizes[] = { 1, 1, 1, 1 };
    tsk_id_t set_indexes[] = { 0, 1, 2, 3 };
    double result;
    int ret;

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);

    ret = tsk_treeseq_f4(&ts, 4, sample_set_sizes, samples, 1, set_indexes, 0, NULL,
        TSK_STAT_SITE, &result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tsk_treeseq_free(&ts);
}

static void
test_paper_ex_afs_errors(void)
{
    tsk_treeseq_t ts;
    tsk_size_t sample_set_sizes[] = { 2, 2 };
    tsk_id_t samples[] = { 0, 1, 2, 3 };
    double result[10]; /* not thinking too hard about the actual value needed */
    int ret;

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);

    verify_one_way_stat_func_errors(&ts, tsk_treeseq_allele_frequency_spectrum);

    ret = tsk_treeseq_allele_frequency_spectrum(
        &ts, 2, sample_set_sizes, samples, 0, NULL, TSK_STAT_NODE, result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_UNSUPPORTED_STAT_MODE);

    ret = tsk_treeseq_allele_frequency_spectrum(&ts, 2, sample_set_sizes, samples, 0,
        NULL, TSK_STAT_BRANCH | TSK_STAT_SITE, result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_MULTIPLE_STAT_MODES);

    tsk_treeseq_free(&ts);
}

static void
test_paper_ex_afs(void)
{
    tsk_treeseq_t ts;
    tsk_id_t samples[] = { 0, 1, 2, 3 };
    tsk_size_t sample_set_sizes[] = { 4, 0 };
    double result[25];
    int ret;

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);
    /* we have two singletons and one tripleton */

    ret = tsk_treeseq_allele_frequency_spectrum(
        &ts, 1, sample_set_sizes, samples, 0, NULL, 0, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(result[0], 0);
    CU_ASSERT_EQUAL_FATAL(result[1], 3.0);
    CU_ASSERT_EQUAL_FATAL(result[2], 0);

    ret = tsk_treeseq_allele_frequency_spectrum(
        &ts, 1, sample_set_sizes, samples, 0, NULL, TSK_STAT_POLARISED, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(result[0], 0);
    CU_ASSERT_EQUAL_FATAL(result[1], 2.0);
    CU_ASSERT_EQUAL_FATAL(result[2], 0);
    CU_ASSERT_EQUAL_FATAL(result[3], 1.0);
    CU_ASSERT_EQUAL_FATAL(result[4], 0);

    verify_afs(&ts);
    tsk_treeseq_free(&ts);
}

static void
test_paper_ex_divergence_matrix(void)
{
    tsk_treeseq_t ts;

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);

    verify_divergence_matrix(&ts, TSK_STAT_BRANCH);
    verify_divergence_matrix(&ts, TSK_STAT_BRANCH | TSK_STAT_SPAN_NORMALISE);
    verify_divergence_matrix(&ts, TSK_STAT_SITE);
    verify_divergence_matrix(&ts, TSK_STAT_SITE | TSK_STAT_SPAN_NORMALISE);

    tsk_treeseq_free(&ts);
}

static void
test_nonbinary_ex_ld(void)
{
    tsk_treeseq_t ts;

    tsk_treeseq_from_text(&ts, 100, nonbinary_ex_nodes, nonbinary_ex_edges, NULL,
        nonbinary_ex_sites, nonbinary_ex_mutations, NULL, NULL, 0);
    verify_ld(&ts);
    tsk_treeseq_free(&ts);
}

static void
test_nonbinary_ex_mean_descendants(void)
{
    tsk_treeseq_t ts;

    tsk_treeseq_from_text(&ts, 100, nonbinary_ex_nodes, nonbinary_ex_edges, NULL,
        nonbinary_ex_sites, nonbinary_ex_mutations, NULL, NULL, 0);
    verify_mean_descendants(&ts);
    tsk_treeseq_free(&ts);
}

static void
test_nonbinary_ex_genealogical_nearest_neighbours(void)
{
    tsk_treeseq_t ts;

    tsk_treeseq_from_text(&ts, 100, nonbinary_ex_nodes, nonbinary_ex_edges, NULL,
        nonbinary_ex_sites, nonbinary_ex_mutations, NULL, NULL, 0);
    verify_genealogical_nearest_neighbours(&ts);
    tsk_treeseq_free(&ts);
}

static void
test_nonbinary_ex_general_stat(void)
{
    tsk_treeseq_t ts;

    tsk_treeseq_from_text(&ts, 100, nonbinary_ex_nodes, nonbinary_ex_edges, NULL,
        nonbinary_ex_sites, nonbinary_ex_mutations, NULL, NULL, 0);
    verify_branch_general_stat_identity(&ts);
    verify_default_general_stat(&ts);
    verify_general_stat(&ts, TSK_STAT_BRANCH);
    verify_general_stat(&ts, TSK_STAT_SITE);
    verify_general_stat(&ts, TSK_STAT_NODE);
    tsk_treeseq_free(&ts);
}

static void
test_nonbinary_ex_general_stat_errors(void)
{
    tsk_treeseq_t ts;

    tsk_treeseq_from_text(&ts, 100, nonbinary_ex_nodes, nonbinary_ex_edges, NULL,
        nonbinary_ex_sites, nonbinary_ex_mutations, NULL, NULL, 0);
    verify_branch_general_stat_errors(&ts);
    verify_site_general_stat_errors(&ts);
    verify_node_general_stat_errors(&ts);
    tsk_treeseq_free(&ts);
}

static void
test_caterpillar_tree_ld(void)
{
    tsk_treeseq_t *ts = caterpillar_tree(50, 20, 1);
    tsk_ld_calc_t ld_calc;
    double r2[20];
    tsk_size_t num_r2_values;
    int ret = tsk_ld_calc_init(&ld_calc, ts);

    CU_ASSERT_EQUAL_FATAL(ret, 0);

    verify_ld(ts);

    ret = tsk_ld_calc_get_r2_array(
        &ld_calc, 0, TSK_DIR_FORWARD, 5, DBL_MAX, r2, &num_r2_values);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(num_r2_values, 5);

    ret = tsk_ld_calc_get_r2_array(
        &ld_calc, 10, TSK_DIR_REVERSE, 5, DBL_MAX, r2, &num_r2_values);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(num_r2_values, 5);

    tsk_ld_calc_free(&ld_calc);
    tsk_treeseq_free(ts);
    free(ts);
}

static void
test_ld_multi_mutations(void)
{
    tsk_treeseq_t *ts = caterpillar_tree(4, 2, 2);
    tsk_ld_calc_t ld_calc;
    double r2;
    int ret = tsk_ld_calc_init(&ld_calc, ts);

    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_ld_calc_get_r2(&ld_calc, 0, 1, &r2);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_ONLY_INFINITE_SITES);

    tsk_ld_calc_free(&ld_calc);
    tsk_treeseq_free(ts);
    free(ts);
}

static void
test_ld_silent_mutations(void)
{
    tsk_treeseq_t *base_ts = caterpillar_tree(4, 2, 1);
    tsk_table_collection_t tables;
    tsk_treeseq_t ts;
    tsk_ld_calc_t ld_calc;
    double r2;
    int ret = tsk_table_collection_copy(base_ts->tables, &tables, 0);

    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tables.mutations.derived_state[1] = '0';

    ret = tsk_treeseq_init(&ts, &tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_ld_calc_init(&ld_calc, &ts);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_ld_calc_get_r2(&ld_calc, 0, 1, &r2);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_SILENT_MUTATIONS_NOT_SUPPORTED);
    tsk_ld_calc_free(&ld_calc);
    tsk_treeseq_free(&ts);

    tsk_table_collection_free(&tables);
    tsk_treeseq_free(base_ts);
    free(base_ts);
}

static void
test_paper_ex_two_site(void)
{
    tsk_treeseq_t ts;
    double result[27];
    tsk_size_t s, result_size, num_sample_sets;
    int ret;

    double truth_one_set[9] = { 1, 0.1111111111111111, 0.1111111111111111,
        0.1111111111111111, 1, 1, 0.1111111111111111, 1, 1 };
    double truth_two_sets[18] = { 1, 1, 0.1111111111111111, 0.1111111111111111,
        0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111,
        1, 1, 1, 1, 0.1111111111111111, 0.1111111111111111, 1, 1, 1, 1 };
    double truth_three_sets[27] = { 1, 1, NAN, 0.1111111111111111, 0.1111111111111111,
        NAN, 0.1111111111111111, 0.1111111111111111, NAN, 0.1111111111111111,
        0.1111111111111111, NAN, 1, 1, 1, 1, 1, 1, 0.1111111111111111,
        0.1111111111111111, NAN, 1, 1, 1, 1, 1, 1 };

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);

    tsk_size_t sample_set_sizes[3];
    tsk_id_t sample_sets[ts.num_samples * 3];
    tsk_size_t num_sites = ts.tables->sites.num_rows;
    tsk_id_t *row_sites = tsk_malloc(num_sites * sizeof(*row_sites));
    tsk_id_t *col_sites = tsk_malloc(num_sites * sizeof(*col_sites));

    // First sample set contains all of the samples
    sample_set_sizes[0] = ts.num_samples;
    num_sample_sets = 1;
    for (s = 0; s < ts.num_samples; s++) {
        sample_sets[s] = (tsk_id_t) s;
    }
    for (s = 0; s < num_sites; s++) {
        row_sites[s] = (tsk_id_t) s;
        col_sites[s] = (tsk_id_t) s;
    }

    result_size = num_sites * num_sites;
    tsk_memset(result, 0, sizeof(*result) * result_size * num_sample_sets);
    ret = tsk_treeseq_r2(&ts, num_sample_sets, sample_set_sizes, sample_sets, num_sites,
        row_sites, num_sites, col_sites, 0, result);

    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(result_size * num_sample_sets, result, truth_one_set);

    // Second sample set contains all of the samples
    sample_set_sizes[1] = ts.num_samples;
    num_sample_sets = 2;
    for (s = ts.num_samples; s < ts.num_samples * 2; s++) {
        sample_sets[s] = (tsk_id_t) s - (tsk_id_t) ts.num_samples;
    }

    tsk_memset(result, 0, sizeof(*result) * result_size * num_sample_sets);
    ret = tsk_treeseq_r2(&ts, num_sample_sets, sample_set_sizes, sample_sets, num_sites,
        row_sites, num_sites, col_sites, 0, result);

    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(result_size * num_sample_sets, result, truth_two_sets);

    // Third sample set contains the first two samples
    sample_set_sizes[2] = 2;
    num_sample_sets = 3;
    for (s = ts.num_samples * 2; s < (ts.num_samples * 3) - 2; s++) {
        sample_sets[s] = (tsk_id_t) s - (tsk_id_t) ts.num_samples * 2;
    }

    tsk_memset(result, 0, sizeof(*result) * result_size * num_sample_sets);
    ret = tsk_treeseq_r2(&ts, num_sample_sets, sample_set_sizes, sample_sets, num_sites,
        row_sites, num_sites, col_sites, 0, result);

    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal_nan(
        result_size * num_sample_sets, result, truth_three_sets);

    tsk_treeseq_free(&ts);
    tsk_safe_free(row_sites);
    tsk_safe_free(col_sites);
}

static void
test_two_site_correlated_multiallelic(void)
{
    const char *nodes = "1   0   -1\n"
                        "1   0   -1\n"
                        "1   0   -1\n"
                        "1   0   -1\n"
                        "1   0   -1\n"
                        "1   0   -1\n"
                        "1   0   -1\n"
                        "1   0   -1\n"
                        "1   0   -1\n"
                        "0   2   -1\n"
                        "0   4   -1\n"
                        "0   6   -1\n"
                        "0   8   -1\n"
                        "0   10  -1\n"
                        "0   12  -1\n"
                        "0   14  -1\n"
                        "0   16  -1\n";
    const char *edges = "0   20   9    0,1\n"
                        "0   20   10   2,9\n"
                        "0   20   11   4,5\n"
                        "0   20   12   6,11\n"
                        "0   20   13   7,8\n"
                        "0   20   14   3,10\n"
                        "0   10   15   12\n"
                        "10  20   15   13\n"
                        "0   10   15   14\n"
                        "10  20   15   14\n"
                        "10  20   16   12\n"
                        "0   10   16   13\n"
                        "0   10   16   15\n"
                        "10  20   16   15\n";
    const char *sites = "7   A\n"
                        "13  G\n";
    const char *mutations = "0   15  T  -1\n"
                            "0   14  G   0\n"
                            "1   15  T  -1\n"
                            "1   13  C   2\n";

    int ret;

    tsk_treeseq_t ts;
    tsk_size_t s, result_size;

    double truth_D[4] = { 0.043209876543209874, -0.018518518518518517,
        -0.018518518518518517, 0.05555555555555555 };
    double truth_D2[4] = { 0.023844603634269844, 0.02384460363426984,
        0.02384460363426984, 0.02384460363426984 };
    double truth_r2[4] = { 1, 1, 1, 1 };
    double truth_D_prime[4] = { 0.7777777777777777, 0.4444444444444444,
        0.4444444444444444, 0.6666666666666666 };
    double truth_r[4] = { 0.18377223398316206, -0.12212786219416509,
        -0.12212786219416509, 0.2609542781331212 };
    double truth_Dz[4] = { 0.0033870175616860566, 0.003387017561686057,
        0.003387017561686057, 0.003387017561686057 };
    double truth_pi2[4] = { 0.04579247743399549, 0.04579247743399549,
        0.04579247743399549, 0.0457924774339955 };

    tsk_treeseq_from_text(&ts, 20, nodes, edges, NULL, sites, mutations, NULL, NULL, 0);

    tsk_size_t num_sample_sets = 1;
    tsk_size_t sample_set_sizes[1] = { ts.num_samples };
    tsk_id_t sample_sets[ts.num_samples];
    tsk_size_t num_sites = ts.tables->sites.num_rows;
    tsk_id_t *row_sites = tsk_malloc(num_sites * sizeof(*row_sites));
    tsk_id_t *col_sites = tsk_malloc(num_sites * sizeof(*col_sites));
    result_size = num_sites * num_sites;
    double result[result_size];

    for (s = 0; s < ts.num_samples; s++) {
        sample_sets[s] = (tsk_id_t) s;
    }
    for (s = 0; s < num_sites; s++) {
        row_sites[s] = (tsk_id_t) s;
        col_sites[s] = (tsk_id_t) s;
    }

    tsk_memset(result, 0, sizeof(*result) * result_size * num_sample_sets);
    ret = tsk_treeseq_D(&ts, num_sample_sets, sample_set_sizes, sample_sets, num_sites,
        row_sites, num_sites, col_sites, 0, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(result_size, result, truth_D);

    tsk_memset(result, 0, sizeof(*result) * result_size * num_sample_sets);
    ret = tsk_treeseq_D2(&ts, num_sample_sets, sample_set_sizes, sample_sets, num_sites,
        row_sites, num_sites, col_sites, 0, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(result_size, result, truth_D2);

    tsk_memset(result, 0, sizeof(*result) * result_size * num_sample_sets);
    ret = tsk_treeseq_r2(&ts, num_sample_sets, sample_set_sizes, sample_sets, num_sites,
        row_sites, num_sites, col_sites, 0, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(result_size, result, truth_r2);

    tsk_memset(result, 0, sizeof(*result) * result_size * num_sample_sets);
    ret = tsk_treeseq_D_prime(&ts, num_sample_sets, sample_set_sizes, sample_sets,
        num_sites, row_sites, num_sites, col_sites, 0, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(result_size, result, truth_D_prime);

    tsk_memset(result, 0, sizeof(*result) * result_size * num_sample_sets);
    ret = tsk_treeseq_r(&ts, num_sample_sets, sample_set_sizes, sample_sets, num_sites,
        row_sites, num_sites, col_sites, 0, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(result_size, result, truth_r);

    tsk_memset(result, 0, sizeof(*result) * result_size * num_sample_sets);
    ret = tsk_treeseq_Dz(&ts, num_sample_sets, sample_set_sizes, sample_sets, num_sites,
        row_sites, num_sites, col_sites, 0, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(result_size, result, truth_Dz);

    tsk_memset(result, 0, sizeof(*result) * result_size * num_sample_sets);
    ret = tsk_treeseq_pi2(&ts, num_sample_sets, sample_set_sizes, sample_sets, num_sites,
        row_sites, num_sites, col_sites, 0, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(result_size, result, truth_pi2);

    tsk_treeseq_free(&ts);
    tsk_safe_free(row_sites);
    tsk_safe_free(col_sites);
}

static void
test_two_site_uncorrelated_multiallelic(void)
{
    const char *nodes = "1   0  -1\n"
                        "1   0  -1\n"
                        "1   0  -1\n"
                        "1   0  -1\n"
                        "1   0  -1\n"
                        "1   0  -1\n"
                        "1   0  -1\n"
                        "1   0  -1\n"
                        "1   0  -1\n"
                        "0   2  -1\n"
                        "0   4  -1\n"
                        "0   6  -1\n"
                        "0   8  -1\n"
                        "0   10 -1\n"
                        "0   12 -1\n"
                        "0   14 -1\n"
                        "0   16 -1\n"
                        "0   2  -1\n"
                        "0   4  -1\n"
                        "0   6  -1\n"
                        "0   8  -1\n"
                        "0   10 -1\n"
                        "0   12 -1\n"
                        "0   14 -1\n"
                        "0   16 -1\n";
    const char *edges = "0     10    9      0,1\n"
                        "10    20    17     0,3\n"
                        "0     10    10     2,9\n"
                        "10    20    18     6,17\n"
                        "0     10    11     3,4\n"
                        "10    20    19     1,4\n"
                        "0     10    12     5,11\n"
                        "10    20    20     7,19\n"
                        "0     10    13     6,7\n"
                        "10    20    21     2,5\n"
                        "0     10    14     8,13\n"
                        "10    20    22     8,21\n"
                        "0     10    15     10,12\n"
                        "10    20    23     18,20\n"
                        "0     10    16     14,15\n"
                        "10    20    24     22,23\n";
    const char *sites = "7   A\n"
                        "13  G\n";
    const char *mutations = "0   15  T  -1\n"
                            "0   12  G   0\n"
                            "1   23  T  -1\n"
                            "1   20  A   2\n";

    tsk_treeseq_t ts;

    int ret;

    double truth_D[4] = { 0.05555555555555555, 0.0, 0.0, 0.05555555555555555 };
    double truth_D2[4] = { 0.024691358024691357, 0.0, 0.0, 0.024691358024691357 };
    double truth_r2[4] = { 1, 0, 0, 1 };
    double truth_D_prime[4] = { 0.6666666666666665, 0.0, 0.0, 0.6666666666666665 };
    double truth_r[4] = { 0.24999999999999997, 0.0, 0.0, 0.24999999999999997 };
    double truth_Dz[4] = { 0.0, 0.0, 0.0, 0.0 };
    double truth_pi2[4] = { 0.04938271604938272, 0.04938271604938272,
        0.04938271604938272, 0.04938271604938272 };

    tsk_treeseq_from_text(&ts, 20, nodes, edges, NULL, sites, mutations, NULL, NULL, 0);

    tsk_size_t s;
    tsk_size_t num_sample_sets = 1;
    tsk_size_t num_sites = ts.tables->sites.num_rows;
    tsk_id_t *row_sites = tsk_malloc(num_sites * sizeof(*row_sites));
    tsk_id_t *col_sites = tsk_malloc(num_sites * sizeof(*col_sites));
    tsk_size_t sample_set_sizes[1] = { ts.num_samples };
    tsk_id_t sample_sets[ts.num_samples];
    tsk_size_t result_size = num_sites * num_sites;
    double result[result_size];

    for (s = 0; s < ts.num_samples; s++) {
        sample_sets[s] = (tsk_id_t) s;
    }
    for (s = 0; s < num_sites; s++) {
        row_sites[s] = (tsk_id_t) s;
        col_sites[s] = (tsk_id_t) s;
    }

    tsk_memset(result, 0, sizeof(*result) * result_size * num_sample_sets);
    ret = tsk_treeseq_D(&ts, num_sample_sets, sample_set_sizes, sample_sets, num_sites,
        row_sites, num_sites, col_sites, 0, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(result_size, result, truth_D);

    tsk_memset(result, 0, sizeof(*result) * result_size * num_sample_sets);
    ret = tsk_treeseq_D2(&ts, num_sample_sets, sample_set_sizes, sample_sets, num_sites,
        row_sites, num_sites, col_sites, 0, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(result_size, result, truth_D2);

    tsk_memset(result, 0, sizeof(*result) * result_size * num_sample_sets);
    ret = tsk_treeseq_r2(&ts, num_sample_sets, sample_set_sizes, sample_sets, num_sites,
        row_sites, num_sites, col_sites, 0, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(result_size, result, truth_r2);

    tsk_memset(result, 0, sizeof(*result) * result_size * num_sample_sets);
    ret = tsk_treeseq_D_prime(&ts, num_sample_sets, sample_set_sizes, sample_sets,
        num_sites, row_sites, num_sites, col_sites, 0, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(result_size, result, truth_D_prime);

    tsk_memset(result, 0, sizeof(*result) * result_size * num_sample_sets);
    ret = tsk_treeseq_r(&ts, num_sample_sets, sample_set_sizes, sample_sets, num_sites,
        row_sites, num_sites, col_sites, 0, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(result_size, result, truth_r);

    tsk_memset(result, 0, sizeof(*result) * result_size * num_sample_sets);
    ret = tsk_treeseq_Dz(&ts, num_sample_sets, sample_set_sizes, sample_sets, num_sites,
        row_sites, num_sites, col_sites, 0, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(result_size, result, truth_Dz);

    tsk_memset(result, 0, sizeof(*result) * result_size * num_sample_sets);
    ret = tsk_treeseq_pi2(&ts, num_sample_sets, sample_set_sizes, sample_sets, num_sites,
        row_sites, num_sites, col_sites, 0, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(result_size, result, truth_pi2);

    tsk_treeseq_free(&ts);
    tsk_safe_free(row_sites);
    tsk_safe_free(col_sites);
}

static void
test_two_site_backmutation(void)
{
    const char *nodes
        = "1 0  -1\n1 0  -1\n1 0  -1\n1 0  -1\n1 0  -1\n1 0  -1\n1 0  -1\n1 0  -1\n"
          "1 0  -1\n1 0  -1\n1 0  -1\n1 0  -1\n1 0  -1\n1 0  -1\n1 0  -1\n1 0  -1\n"
          "1 0  -1\n1 0  -1\n1 0  -1\n1 0  -1\n1 0  -1\n1 0  -1\n1 0  -1\n1 0  -1\n"
          "1 0  -1\n1 0  -1\n1 0  -1\n1 0  -1\n1 0  -1\n1 0  -1\n1 0  -1\n1 0  -1\n"
          "1 0  -1\n1 0  -1\n1 0  -1\n0 2  -1\n0 4  -1\n0 6  -1\n0 8  -1\n0 10 -1\n"
          "0 12 -1\n0 14 -1\n0 16 -1\n0 18 -1\n0 20 -1\n0 22 -1\n0 24 -1\n0 26 -1\n"
          "0 28 -1\n0 30 -1\n0 32 -1\n0 34 -1\n0 36 -1\n0 38 -1\n0 40 -1\n0 42 -1\n"
          "0 44 -1\n0 46 -1\n0 48 -1\n0 50 -1\n0 52 -1\n0 54 -1\n0 56 -1\n0 58 -1\n"
          "0 60 -1\n0 62 -1\n0 64 -1\n0 66 -1\n0 68 -1\n";

    const char *edges
        = "0 10 35 0,1\n0 10 36 2,35\n0 10 37 3,36\n0 10 38 4,37\n0 10 39 5,38\n"
          "0 10 40 6,39\n0 10 41 7,40\n0 10 42 8,41\n0 10 43 9,42\n0 10 44 10,43\n"
          "0 10 45 11,44\n0 10 46 12,45\n0 10 47 13,46\n0 10 48 14,47\n0 10 49 15,48\n"
          "0 10 50 16,49\n0 10 51 17,50\n0 10 52 18,51\n0 10 53 19,52\n0 10 54 20,53\n"
          "0 10 55 21,54\n0 10 56 22,55\n0 10 57 23,56\n0 10 58 24,57\n0 10 59 25,58\n"
          "0 10 60 26,59\n0 10 61 27,60\n0 10 62 28,61\n0 10 63 29,62\n0 10 64 30,63\n"
          "0 10 65 31,64\n0 10 66 32,65\n0 10 67 33,66\n0 10 68 34,67\n";

    const char *sites = "1    A\n"
                        "4.5  T\n";

    const char *mutations = "0  50  T  -1\n"
                            "0  48  G   0\n"
                            "0  46  A   1\n"
                            "1  62  G  -1\n"
                            "1  60  T   3\n"
                            "1  58  A   4\n";

    int ret;

    tsk_treeseq_t ts;
    tsk_treeseq_from_text(&ts, 10, nodes, edges, NULL, sites, mutations, NULL, NULL, 0);

    tsk_size_t num_sample_sets = 1;
    tsk_size_t num_sites = ts.tables->sites.num_rows;
    tsk_id_t *row_sites = tsk_malloc(num_sites * sizeof(*row_sites));
    tsk_id_t *col_sites = tsk_malloc(num_sites * sizeof(*col_sites));
    tsk_size_t sample_set_sizes[1] = { ts.num_samples };
    tsk_id_t sample_sets[ts.num_samples];
    tsk_size_t result_size = num_sites * num_sites;
    double result[result_size];
    tsk_size_t s;

    double truth_r2[4] = { 0.999999999999999, 0.042923862278701, 0.042923862278701, 1. };

    for (s = 0; s < ts.num_samples; s++) {
        sample_sets[s] = (tsk_id_t) s;
    }
    for (s = 0; s < num_sites; s++) {
        row_sites[s] = (tsk_id_t) s;
        col_sites[s] = (tsk_id_t) s;
    }

    tsk_memset(result, 0, sizeof(*result) * result_size * num_sample_sets);
    ret = tsk_treeseq_r2(&ts, num_sample_sets, sample_set_sizes, sample_sets, num_sites,
        row_sites, num_sites, col_sites, 0, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(result_size, result, truth_r2);

    tsk_treeseq_free(&ts);
    tsk_safe_free(row_sites);
    tsk_safe_free(col_sites);
}

static void
test_paper_ex_two_site_subset(void)
{
    tsk_treeseq_t ts;
    double result[4];
    int ret;
    tsk_size_t s, result_size;
    tsk_size_t sample_set_sizes[1];
    tsk_size_t num_sample_sets;
    tsk_id_t row_sites[2] = { 0, 1 };
    tsk_id_t col_sites[2] = { 1, 2 };
    double result_truth_1[4] = { 0.1111111111111111, 0.1111111111111111, 1, 1 };
    double result_truth_2[1] = { 0.1111111111111111 };
    double result_truth_3[4] = { 0.1111111111111111, 1, 0.1111111111111111, 1 };

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);

    tsk_id_t sample_sets[ts.num_samples];

    sample_set_sizes[0] = ts.num_samples;
    num_sample_sets = 1;
    for (s = 0; s < ts.num_samples; s++) {
        sample_sets[s] = (tsk_id_t) s;
    }

    result_size = 2 * 2;
    tsk_memset(result, 0, sizeof(*result) * result_size * num_sample_sets);
    ret = tsk_treeseq_r2(&ts, num_sample_sets, sample_set_sizes, sample_sets, 2,
        row_sites, 2, col_sites, 0, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(result_size * num_sample_sets, result, result_truth_1);

    result_size = 1 * 1;
    tsk_memset(result, 0, sizeof(*result) * result_size * num_sample_sets);
    col_sites[0] = 2;
    ret = tsk_treeseq_r2(&ts, num_sample_sets, sample_set_sizes, sample_sets, 1,
        row_sites, 1, col_sites, 0, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(result_size * num_sample_sets, result, result_truth_2);

    result_size = 2 * 2;
    tsk_memset(result, 0, sizeof(*result) * result_size * num_sample_sets);
    row_sites[0] = 1;
    row_sites[1] = 2;
    col_sites[0] = 0;
    col_sites[1] = 1;
    ret = tsk_treeseq_r2(&ts, num_sample_sets, sample_set_sizes, sample_sets, 2,
        row_sites, 2, col_sites, 0, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(result_size * num_sample_sets, result, result_truth_3);

    tsk_treeseq_free(&ts);
}

static void
test_two_locus_stat_input_errors(void)
{
    tsk_treeseq_t ts;
    int ret;

    tsk_treeseq_from_text(&ts, 1, single_tree_ex_nodes, single_tree_ex_edges, NULL,
        single_tree_ex_sites, single_tree_ex_mutations, NULL, NULL, 0);

    tsk_size_t num_sites = ts.tables->sites.num_rows;
    tsk_id_t *row_sites = tsk_malloc(num_sites * sizeof(*row_sites));
    tsk_id_t *col_sites = tsk_malloc(num_sites * sizeof(*col_sites));
    tsk_size_t sample_set_sizes[1] = { ts.num_samples };
    tsk_size_t num_sample_sets = 1;
    tsk_id_t sample_sets[ts.num_samples];
    tsk_size_t result_size = num_sites * num_sites;
    double result[result_size];
    tsk_size_t s;

    for (s = 0; s < ts.num_samples; s++) {
        sample_sets[s] = (tsk_id_t) s;
    }
    for (s = 0; s < num_sites; s++) {
        row_sites[s] = (tsk_id_t) s;
        col_sites[s] = (tsk_id_t) s;
    }

    sample_set_sizes[0] = ts.num_samples;
    num_sample_sets = 1;
    for (s = 0; s < ts.num_samples; s++) {
        sample_sets[s] = (tsk_id_t) s;
    }

    sample_sets[1] = 0;
    ret = tsk_treeseq_r2(&ts, num_sample_sets, sample_set_sizes, sample_sets, num_sites,
        row_sites, num_sites, col_sites, 0, result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_DUPLICATE_SAMPLE);
    sample_sets[1] = 1;

    ret = tsk_treeseq_r2(&ts, num_sample_sets, sample_set_sizes, sample_sets, num_sites,
        row_sites, num_sites, col_sites, TSK_STAT_SITE | TSK_STAT_BRANCH, result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_MULTIPLE_STAT_MODES);

    ret = tsk_treeseq_r2(&ts, num_sample_sets, sample_set_sizes, sample_sets, num_sites,
        row_sites, num_sites, col_sites, TSK_STAT_BRANCH, result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_UNSUPPORTED_STAT_MODE);

    ret = tsk_treeseq_r2(&ts, 0, sample_set_sizes, sample_sets, num_sites, row_sites,
        num_sites, col_sites, 0, result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_INSUFFICIENT_SAMPLE_SETS);

    sample_set_sizes[0] = 0;
    ret = tsk_treeseq_r2(&ts, num_sample_sets, sample_set_sizes, sample_sets, num_sites,
        row_sites, num_sites, col_sites, 0, result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_EMPTY_SAMPLE_SET);
    sample_set_sizes[0] = ts.num_samples;

    sample_sets[1] = 10;
    ret = tsk_treeseq_r2(&ts, num_sample_sets, sample_set_sizes, sample_sets, num_sites,
        row_sites, num_sites, col_sites, 0, result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_NODE_OUT_OF_BOUNDS);
    sample_sets[1] = 1;

    row_sites[0] = 1000;
    ret = tsk_treeseq_r2(&ts, num_sample_sets, sample_set_sizes, sample_sets, num_sites,
        row_sites, num_sites, col_sites, 0, result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_SITE_OUT_OF_BOUNDS);
    row_sites[0] = 0;

    col_sites[num_sites - 1] = (tsk_id_t) num_sites;
    ret = tsk_treeseq_r2(&ts, num_sample_sets, sample_set_sizes, sample_sets, num_sites,
        row_sites, num_sites, col_sites, 0, result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_SITE_OUT_OF_BOUNDS);
    col_sites[num_sites - 1] = (tsk_id_t) num_sites - 1;

    row_sites[0] = 1;
    row_sites[1] = 0;
    ret = tsk_treeseq_r2(&ts, num_sample_sets, sample_set_sizes, sample_sets, num_sites,
        row_sites, num_sites, col_sites, 0, result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_UNSORTED_SITES);
    row_sites[0] = 0;
    row_sites[1] = 1;

    row_sites[0] = 1;
    row_sites[1] = 1;
    ret = tsk_treeseq_r2(&ts, num_sample_sets, sample_set_sizes, sample_sets, num_sites,
        row_sites, num_sites, col_sites, 0, result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_UNSORTED_SITES);
    row_sites[0] = 0;
    row_sites[1] = 1;

    // Not an error condition, but we want to record this behavior
    ret = tsk_treeseq_r2(&ts, num_sample_sets, sample_set_sizes, sample_sets, 0, NULL, 0,
        NULL, 0, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    tsk_treeseq_free(&ts);
    tsk_safe_free(row_sites);
    tsk_safe_free(col_sites);
}

static void
test_simplest_divergence_matrix(void)
{
    const char *nodes = "1  0   0\n"
                        "1  0   0\n"
                        "0  1   0\n";
    const char *edges = "0  1   2   0,1\n";
    const char *sites = "0.1  A\n"
                        "0.6  A\n";
    const char *mutations = "0  0  B  -1\n"
                            "1  0  B  -1\n";
    tsk_treeseq_t ts;
    tsk_id_t sample_ids[] = { 0, 1 };
    double D_branch[4] = { 0, 2, 2, 0 };
    double D_site[4] = { 0, 2, 2, 0 };
    double result[4];
    int ret;

    tsk_treeseq_from_text(&ts, 1, nodes, edges, NULL, sites, mutations, NULL, NULL, 0);

    ret = tsk_treeseq_divergence_matrix(
        &ts, 2, NULL, sample_ids, 0, NULL, TSK_STAT_BRANCH, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(4, D_branch, result);

    ret = tsk_treeseq_divergence_matrix(&ts, 2, NULL, sample_ids, 0, NULL,
        TSK_STAT_BRANCH | TSK_STAT_SPAN_NORMALISE, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(4, D_branch, result);

    ret = tsk_treeseq_divergence_matrix(&ts, 2, NULL, sample_ids, 0, NULL, 0, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(4, D_site, result);

    ret = tsk_treeseq_divergence_matrix(
        &ts, 2, NULL, sample_ids, 0, NULL, TSK_STAT_SPAN_NORMALISE, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(4, D_site, result);

    ret = tsk_treeseq_divergence_matrix(
        &ts, 2, NULL, sample_ids, 0, NULL, TSK_STAT_SITE, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(4, D_site, result);

    ret = tsk_treeseq_divergence_matrix(
        &ts, 0, NULL, NULL, 0, NULL, TSK_STAT_BRANCH, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(4, D_branch, result);

    ret = tsk_treeseq_divergence_matrix(
        &ts, 0, NULL, NULL, 0, NULL, TSK_STAT_SITE, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(4, D_site, result);

    ret = tsk_treeseq_divergence_matrix(
        &ts, 0, NULL, NULL, 0, NULL, TSK_STAT_NODE, result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_UNSUPPORTED_STAT_MODE);

    ret = tsk_treeseq_divergence_matrix(
        &ts, 0, NULL, NULL, 0, NULL, TSK_STAT_POLARISED, result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_STAT_POLARISED_UNSUPPORTED);

    ret = tsk_treeseq_divergence_matrix(
        &ts, 0, NULL, NULL, 0, NULL, TSK_STAT_SITE | TSK_STAT_BRANCH, result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_MULTIPLE_STAT_MODES);

    sample_ids[0] = -1;
    ret = tsk_treeseq_divergence_matrix(&ts, 2, NULL, sample_ids, 0, NULL, 0, result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_NODE_OUT_OF_BOUNDS);

    sample_ids[0] = 3;
    ret = tsk_treeseq_divergence_matrix(&ts, 2, NULL, sample_ids, 0, NULL, 0, result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_NODE_OUT_OF_BOUNDS);

    sample_ids[0] = 1;
    ret = tsk_treeseq_divergence_matrix(&ts, 2, NULL, sample_ids, 0, NULL, 0, result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_DUPLICATE_SAMPLE);
    ret = tsk_treeseq_divergence_matrix(
        &ts, 2, NULL, sample_ids, 0, NULL, TSK_STAT_BRANCH, result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_DUPLICATE_SAMPLE);

    sample_ids[0] = 2;
    ret = tsk_treeseq_divergence_matrix(&ts, 2, NULL, sample_ids, 0, NULL, 0, result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_SAMPLES);

    tsk_treeseq_free(&ts);
}

static void
test_simplest_divergence_matrix_windows(void)
{
    const char *nodes = "1  0   0\n"
                        "1  0   0\n"
                        "0  1   0\n";
    const char *edges = "0  1   2   0,1\n";
    const char *sites = "0.1  A\n"
                        "0.6  A\n";
    const char *mutations = "0  0  B  -1\n"
                            "1  0  B  -1\n";
    tsk_treeseq_t ts;
    tsk_id_t sample_ids[] = { 0, 1 };
    double D_branch[8] = { 0, 1, 1, 0, 0, 1, 1, 0 };
    double D_site[8] = { 0, 1, 1, 0, 0, 1, 1, 0 };
    double result[8];
    double windows[] = { 0, 0.5, 1 };
    int ret;

    tsk_treeseq_from_text(&ts, 1, nodes, edges, NULL, sites, mutations, NULL, NULL, 0);

    ret = tsk_treeseq_divergence_matrix(&ts, 2, NULL, sample_ids, 2, windows, 0, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(8, D_site, result);
    ret = tsk_treeseq_divergence_matrix(
        &ts, 2, NULL, sample_ids, 2, windows, TSK_STAT_BRANCH, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(8, D_branch, result);

    /* Windows for the second half */
    ret = tsk_treeseq_divergence_matrix(
        &ts, 2, NULL, sample_ids, 1, windows + 1, TSK_STAT_SITE, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(4, D_site, result);
    ret = tsk_treeseq_divergence_matrix(
        &ts, 2, NULL, sample_ids, 1, windows + 1, TSK_STAT_BRANCH, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(4, D_branch, result);

    ret = tsk_treeseq_divergence_matrix(&ts, 2, NULL, sample_ids, 0, windows, 0, result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_NUM_WINDOWS);

    windows[0] = -1;
    ret = tsk_treeseq_divergence_matrix(&ts, 2, NULL, sample_ids, 2, windows, 0, result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_WINDOWS);

    windows[0] = 0.45;
    windows[2] = 1.5;
    ret = tsk_treeseq_divergence_matrix(&ts, 2, NULL, sample_ids, 2, windows, 0, result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_WINDOWS);

    windows[0] = 0.55;
    windows[2] = 1.0;
    ret = tsk_treeseq_divergence_matrix(&ts, 2, NULL, sample_ids, 2, windows, 0, result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_WINDOWS);

    tsk_treeseq_free(&ts);
}

static void
test_simplest_divergence_matrix_internal_sample(void)
{
    const char *nodes = "1  0   0\n"
                        "1  0   0\n"
                        "1  1   0\n";
    const char *edges = "0  1   2   0,1\n";
    tsk_treeseq_t ts;
    tsk_id_t sample_ids[] = { 0, 1, 2 };
    double result[9];
    double D_branch[9] = { 0, 2, 1, 2, 0, 1, 1, 1, 0 };
    double D_site[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    int ret;

    tsk_treeseq_from_text(&ts, 1, nodes, edges, NULL, NULL, NULL, NULL, NULL, 0);

    ret = tsk_treeseq_divergence_matrix(
        &ts, 3, NULL, sample_ids, 0, NULL, TSK_STAT_BRANCH, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(9, D_branch, result);

    ret = tsk_treeseq_divergence_matrix(
        &ts, 3, NULL, sample_ids, 0, NULL, TSK_STAT_SITE, result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    assert_arrays_almost_equal(9, D_site, result);

    tsk_treeseq_free(&ts);
}

static void
test_multiroot_divergence_matrix(void)
{
    tsk_treeseq_t ts;

    tsk_treeseq_from_text(&ts, 10, multiroot_ex_nodes, multiroot_ex_edges, NULL,
        multiroot_ex_sites, multiroot_ex_mutations, NULL, NULL, 0);

    verify_divergence_matrix(&ts, TSK_STAT_BRANCH);
    verify_divergence_matrix(&ts, TSK_STAT_BRANCH | TSK_STAT_SPAN_NORMALISE);
    verify_divergence_matrix(&ts, TSK_STAT_SITE);
    verify_divergence_matrix(&ts, TSK_STAT_SITE | TSK_STAT_SPAN_NORMALISE);

    tsk_treeseq_free(&ts);
}

int
main(int argc, char **argv)
{
    CU_TestInfo tests[] = {
        { "test_general_stat_input_errors", test_general_stat_input_errors },

        { "test_empty_ts_ld", test_empty_ts_ld },
        { "test_empty_ts_mean_descendants", test_empty_ts_mean_descendants },
        { "test_empty_ts_genealogical_nearest_neighbours",
            test_empty_ts_genealogical_nearest_neighbours },
        { "test_empty_ts_general_stat", test_empty_ts_general_stat },
        { "test_empty_ts_afs", test_empty_ts_afs },

        { "test_single_tree_ld", test_single_tree_ld },
        { "test_single_tree_mean_descendants", test_single_tree_mean_descendants },
        { "test_single_tree_genealogical_nearest_neighbours",
            test_single_tree_genealogical_nearest_neighbours },
        { "test_single_tree_general_stat", test_single_tree_general_stat },
        { "test_single_tree_general_stat_errors", test_single_tree_general_stat_errors },
        { "test_single_tree_divergence_matrix", test_single_tree_divergence_matrix },
        { "test_single_tree_divergence_matrix_internal_samples",
            test_single_tree_divergence_matrix_internal_samples },
        { "test_single_tree_divergence_matrix_multi_root",
            test_single_tree_divergence_matrix_multi_root },

        { "test_paper_ex_ld", test_paper_ex_ld },
        { "test_paper_ex_mean_descendants", test_paper_ex_mean_descendants },
        { "test_paper_ex_genealogical_nearest_neighbours",
            test_paper_ex_genealogical_nearest_neighbours },
        { "test_paper_ex_general_stat_errors", test_paper_ex_general_stat_errors },
        { "test_paper_ex_general_stat", test_paper_ex_general_stat },
        { "test_paper_ex_trait_covariance_errors",
            test_paper_ex_trait_covariance_errors },
        { "test_paper_ex_trait_covariance", test_paper_ex_trait_covariance },
        { "test_paper_ex_trait_correlation_errors",
            test_paper_ex_trait_correlation_errors },
        { "test_paper_ex_trait_correlation", test_paper_ex_trait_correlation },
        { "test_paper_ex_trait_linear_model_errors",
            test_paper_ex_trait_linear_model_errors },
        { "test_paper_ex_trait_linear_model", test_paper_ex_trait_linear_model },
        { "test_paper_ex_diversity_errors", test_paper_ex_diversity_errors },
        { "test_paper_ex_diversity", test_paper_ex_diversity },
        { "test_paper_ex_segregating_sites_errors",
            test_paper_ex_segregating_sites_errors },
        { "test_paper_ex_segregating_sites", test_paper_ex_segregating_sites },
        { "test_paper_ex_Y1_errors", test_paper_ex_Y1_errors },
        { "test_paper_ex_Y1", test_paper_ex_Y1 },
        { "test_paper_ex_divergence_errors", test_paper_ex_divergence_errors },
        { "test_paper_ex_divergence", test_paper_ex_divergence },
        { "test_paper_ex_genetic_relatedness_errors",
            test_paper_ex_genetic_relatedness_errors },
        { "test_paper_ex_genetic_relatedness", test_paper_ex_genetic_relatedness },
        { "test_paper_ex_genetic_relatedness_weighted",
            test_paper_ex_genetic_relatedness_weighted },
        { "test_paper_ex_genetic_relatedness_weighted_errors",
            test_paper_ex_genetic_relatedness_weighted_errors },
        { "test_paper_ex_Y2_errors", test_paper_ex_Y2_errors },
        { "test_paper_ex_Y2", test_paper_ex_Y2 },
        { "test_paper_ex_f2_errors", test_paper_ex_f2_errors },
        { "test_paper_ex_f2", test_paper_ex_f2 },
        { "test_paper_ex_Y3_errors", test_paper_ex_Y3_errors },
        { "test_paper_ex_Y3", test_paper_ex_Y3 },
        { "test_paper_ex_f3_errors", test_paper_ex_f3_errors },
        { "test_paper_ex_f3", test_paper_ex_f3 },
        { "test_paper_ex_f4_errors", test_paper_ex_f4_errors },
        { "test_paper_ex_f4", test_paper_ex_f4 },
        { "test_paper_ex_afs_errors", test_paper_ex_afs_errors },
        { "test_paper_ex_afs", test_paper_ex_afs },
        { "test_paper_ex_divergence_matrix", test_paper_ex_divergence_matrix },

        { "test_nonbinary_ex_ld", test_nonbinary_ex_ld },
        { "test_nonbinary_ex_mean_descendants", test_nonbinary_ex_mean_descendants },
        { "test_nonbinary_ex_genealogical_nearest_neighbours",
            test_nonbinary_ex_genealogical_nearest_neighbours },
        { "test_nonbinary_ex_general_stat", test_nonbinary_ex_general_stat },
        { "test_nonbinary_ex_general_stat_errors",
            test_nonbinary_ex_general_stat_errors },

        { "test_caterpillar_tree_ld", test_caterpillar_tree_ld },
        { "test_ld_multi_mutations", test_ld_multi_mutations },
        { "test_ld_silent_mutations", test_ld_silent_mutations },

        { "test_paper_ex_two_site", test_paper_ex_two_site },
        { "test_two_site_correlated_multiallelic",
            test_two_site_correlated_multiallelic },
        { "test_two_site_uncorrelated_multiallelic",
            test_two_site_uncorrelated_multiallelic },
        { "test_two_site_backmutation", test_two_site_backmutation },
        { "test_paper_ex_two_site_subset", test_paper_ex_two_site_subset },
        { "test_two_locus_stat_input_errors", test_two_locus_stat_input_errors },

        { "test_simplest_divergence_matrix", test_simplest_divergence_matrix },
        { "test_simplest_divergence_matrix_windows",
            test_simplest_divergence_matrix_windows },
        { "test_simplest_divergence_matrix_internal_sample",
            test_simplest_divergence_matrix_internal_sample },
        { "test_multiroot_divergence_matrix", test_multiroot_divergence_matrix },

        { NULL, NULL },
    };
    return test_main(tests, argc, argv);
}
