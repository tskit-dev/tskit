/*
 * MIT License
 *
 * Copyright (c) 2019 Tskit Developers
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
    tsk_site_t *sites = malloc(num_sites * sizeof(tsk_site_t));
    int *num_site_mutations = malloc(num_sites * sizeof(int));
    tsk_ld_calc_t ld_calc;
    double *r2, *r2_prime, x;
    tsk_id_t j;
    tsk_size_t num_r2_values;
    double eps = 1e-6;

    r2 = calloc(num_sites, sizeof(double));
    r2_prime = calloc(num_sites, sizeof(double));
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
            tsk_ld_calc_print_state(&ld_calc, _devnull);
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
        tsk_ld_calc_print_state(&ld_calc, _devnull);

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

    if (num_sites > 3) {
        /* Check for some basic distance calculations */
        j = (tsk_id_t) num_sites / 2;
        x = sites[j + 1].position - sites[j].position;
        ret = tsk_ld_calc_get_r2_array(
            &ld_calc, j, TSK_DIR_FORWARD, num_sites, x, r2, &num_r2_values);
        if (multi_mutations_exist(ts, j, (tsk_id_t) num_sites)) {
            CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_ONLY_INFINITE_SITES);
        } else {
            CU_ASSERT_EQUAL_FATAL(ret, 0);
            CU_ASSERT_EQUAL_FATAL(num_r2_values, 1);
        }

        x = sites[j].position - sites[j - 1].position;
        ret = tsk_ld_calc_get_r2_array(
            &ld_calc, j, TSK_DIR_REVERSE, num_sites, x, r2, &num_r2_values);
        if (multi_mutations_exist(ts, 0, j + 1)) {
            CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_ONLY_INFINITE_SITES);
        } else {
            CU_ASSERT_EQUAL_FATAL(ret, 0);
            CU_ASSERT_EQUAL_FATAL(num_r2_values, 1);
        }
    }

    /* Check some error conditions */
    for (j = (tsk_id_t) num_sites; j < (tsk_id_t) num_sites + 2; j++) {
        ret = tsk_ld_calc_get_r2_array(
            &ld_calc, j, TSK_DIR_FORWARD, num_sites, DBL_MAX, r2, &num_r2_values);
        CU_ASSERT_EQUAL(ret, TSK_ERR_OUT_OF_BOUNDS);
        ret = tsk_ld_calc_get_r2(&ld_calc, j, 0, r2);
        CU_ASSERT_EQUAL(ret, TSK_ERR_OUT_OF_BOUNDS);
        ret = tsk_ld_calc_get_r2(&ld_calc, 0, j, r2);
        CU_ASSERT_EQUAL(ret, TSK_ERR_OUT_OF_BOUNDS);
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
    size_t sample_set_size[2];
    size_t num_samples = tsk_treeseq_get_num_samples(ts);
    double *A = malloc(2 * num_samples * sizeof(double));
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
    size_t sample_set_size[2];
    size_t num_samples = tsk_treeseq_get_num_samples(ts);
    double *C = malloc(2 * tsk_treeseq_get_num_nodes(ts) * sizeof(double));
    CU_ASSERT_FATAL(C != NULL);

    samples = malloc(num_samples * sizeof(*samples));
    memcpy(samples, tsk_treeseq_get_samples(ts), num_samples * sizeof(*samples));

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

typedef struct {
    int call_count;
    int error_on;
    int error_code;
} general_stat_error_params_t;

static int
general_stat_error(
    size_t TSK_UNUSED(K), const double *TSK_UNUSED(X), size_t M, double *Y, void *params)
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
    size_t num_samples = tsk_treeseq_get_num_samples(ts);
    double *W = calloc(num_samples, sizeof(double));
    /* node mode requires this much space at least */
    double *sigma = calloc(tsk_treeseq_get_num_nodes(ts), sizeof(double));
    double windows[] = { 0, 0, 0 };
    tsk_flags_t options = mode;

    /* Window errors */
    ret = tsk_treeseq_general_stat(
        ts, 1, W, 1, general_stat_error, NULL, 0, windows, sigma, options);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_NUM_WINDOWS);

    ret = tsk_treeseq_general_stat(
        ts, 1, W, 1, general_stat_error, NULL, 2, windows, sigma, options);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_WINDOWS);

    windows[0] = 10;
    ret = tsk_treeseq_general_stat(
        ts, 1, W, 1, general_stat_error, NULL, 2, windows, sigma, options);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_WINDOWS);

    windows[0] = 0;
    windows[2] = tsk_treeseq_get_sequence_length(ts) + 1;
    ret = tsk_treeseq_general_stat(
        ts, 1, W, 1, general_stat_error, NULL, 2, windows, sigma, options);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_WINDOWS);

    windows[0] = 0;
    windows[1] = -1;
    windows[2] = tsk_treeseq_get_sequence_length(ts);
    ret = tsk_treeseq_general_stat(
        ts, 1, W, 1, general_stat_error, NULL, 2, windows, sigma, options);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_WINDOWS);

    free(W);
    free(sigma);
}

static void
verify_summary_func_errors(tsk_treeseq_t *ts, tsk_flags_t mode)
{
    int ret;
    size_t num_samples = tsk_treeseq_get_num_samples(ts);
    double *W = calloc(num_samples, sizeof(double));
    /* We need this much space for NODE mode */
    double *sigma = calloc(tsk_treeseq_get_num_nodes(ts), sizeof(double));
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
            sigma, TSK_STAT_POLARISED | mode);
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
            ts, 1, W, 1, general_stat_error, &params, 0, NULL, sigma, mode);
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
    // we don't have any specific errors for this function
    // but we might add some in the future
    int ret;
    tsk_size_t num_samples = tsk_treeseq_get_num_samples(ts);
    double *weights = malloc(num_samples * sizeof(double));
    double result;
    size_t j;

    for (j = 0; j < num_samples; j++) {
        weights[j] = 1.0;
    }

    ret = method(ts, 0, weights, 0, NULL, &result, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_STATE_DIMS);

    free(weights);
}

static void
verify_one_way_weighted_covariate_func_errors(
    tsk_treeseq_t *ts, one_way_covariates_method *method)
{
    // we don't have any specific errors for this function
    // but we might add some in the future
    int ret;
    tsk_size_t num_samples = tsk_treeseq_get_num_samples(ts);
    double *weights = malloc(num_samples * sizeof(double));
    double *covariates = NULL;
    double result;
    size_t j;

    for (j = 0; j < num_samples; j++) {
        weights[j] = 1.0;
    }

    ret = method(ts, 0, weights, 0, covariates, 0, NULL, &result, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_STATE_DIMS);

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

    ret = method(ts, 0, &sample_set_sizes, samples, 0, NULL, &result, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_INSUFFICIENT_SAMPLE_SETS);

    samples[0] = TSK_NULL;
    ret = method(ts, 1, &sample_set_sizes, samples, 0, NULL, &result, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_NODE_OUT_OF_BOUNDS);
    samples[0] = -10;
    ret = method(ts, 1, &sample_set_sizes, samples, 0, NULL, &result, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_NODE_OUT_OF_BOUNDS);
    samples[0] = num_nodes;
    ret = method(ts, 1, &sample_set_sizes, samples, 0, NULL, &result, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_NODE_OUT_OF_BOUNDS);
    samples[0] = num_nodes + 1;
    ret = method(ts, 1, &sample_set_sizes, samples, 0, NULL, &result, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_NODE_OUT_OF_BOUNDS);

    samples[0] = num_nodes - 1;
    ret = method(ts, 1, &sample_set_sizes, samples, 0, NULL, &result, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_SAMPLES);

    samples[0] = 1;
    ret = method(ts, 1, &sample_set_sizes, samples, 0, NULL, &result, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_DUPLICATE_SAMPLE);

    samples[0] = 0;
    sample_set_sizes = 0;
    ret = method(ts, 1, &sample_set_sizes, samples, 0, NULL, &result, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_EMPTY_SAMPLE_SET);

    sample_set_sizes = 4;
    /* Window errors */
    ret = method(ts, 1, &sample_set_sizes, samples, 0, windows, &result, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_NUM_WINDOWS);

    ret = method(ts, 1, &sample_set_sizes, samples, 2, windows, &result, 0);
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

    ret = method(ts, 0, sample_set_sizes, samples, 1, set_indexes, 0, NULL, &result,
        TSK_STAT_SITE);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_INSUFFICIENT_SAMPLE_SETS);
    ret = method(ts, 1, sample_set_sizes, samples, 1, set_indexes, 0, NULL, &result,
        TSK_STAT_SITE);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_INSUFFICIENT_SAMPLE_SETS);

    ret = method(ts, 2, sample_set_sizes, samples, 0, set_indexes, 0, NULL, &result,
        TSK_STAT_SITE);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_INSUFFICIENT_INDEX_TUPLES);

    set_indexes[0] = -1;
    ret = method(ts, 2, sample_set_sizes, samples, 1, set_indexes, 0, NULL, &result,
        TSK_STAT_SITE);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_SAMPLE_SET_INDEX);
    set_indexes[0] = 0;
    set_indexes[1] = 2;
    ret = method(ts, 2, sample_set_sizes, samples, 1, set_indexes, 0, NULL, &result,
        TSK_STAT_SITE);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_SAMPLE_SET_INDEX);
}

static void
verify_three_way_stat_func_errors(tsk_treeseq_t *ts, general_sample_stat_method *method)
{
    int ret;
    tsk_id_t samples[] = { 0, 1, 2, 3 };
    tsk_size_t sample_set_sizes[] = { 1, 1, 2 };
    tsk_id_t set_indexes[] = { 0, 1, 2 };
    double result;

    ret = method(ts, 0, sample_set_sizes, samples, 1, set_indexes, 0, NULL, &result,
        TSK_STAT_SITE);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_INSUFFICIENT_SAMPLE_SETS);
    ret = method(ts, 1, sample_set_sizes, samples, 1, set_indexes, 0, NULL, &result,
        TSK_STAT_SITE);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_INSUFFICIENT_SAMPLE_SETS);
    ret = method(ts, 2, sample_set_sizes, samples, 1, set_indexes, 0, NULL, &result,
        TSK_STAT_SITE);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_INSUFFICIENT_SAMPLE_SETS);

    ret = method(ts, 3, sample_set_sizes, samples, 0, set_indexes, 0, NULL, &result,
        TSK_STAT_SITE);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_INSUFFICIENT_INDEX_TUPLES);

    set_indexes[0] = -1;
    ret = method(ts, 3, sample_set_sizes, samples, 1, set_indexes, 0, NULL, &result,
        TSK_STAT_SITE);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_SAMPLE_SET_INDEX);
    set_indexes[0] = 0;
    set_indexes[1] = 3;
    ret = method(ts, 3, sample_set_sizes, samples, 1, set_indexes, 0, NULL, &result,
        TSK_STAT_SITE);
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

    ret = method(ts, 0, sample_set_sizes, samples, 1, set_indexes, 0, NULL, &result,
        TSK_STAT_SITE);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_INSUFFICIENT_SAMPLE_SETS);
    ret = method(ts, 1, sample_set_sizes, samples, 1, set_indexes, 0, NULL, &result,
        TSK_STAT_SITE);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_INSUFFICIENT_SAMPLE_SETS);
    ret = method(ts, 2, sample_set_sizes, samples, 1, set_indexes, 0, NULL, &result,
        TSK_STAT_SITE);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_INSUFFICIENT_SAMPLE_SETS);
    ret = method(ts, 3, sample_set_sizes, samples, 1, set_indexes, 0, NULL, &result,
        TSK_STAT_SITE);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_INSUFFICIENT_SAMPLE_SETS);

    ret = method(ts, 4, sample_set_sizes, samples, 0, set_indexes, 0, NULL, &result,
        TSK_STAT_SITE);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_INSUFFICIENT_INDEX_TUPLES);

    set_indexes[0] = -1;
    ret = method(ts, 4, sample_set_sizes, samples, 1, set_indexes, 0, NULL, &result,
        TSK_STAT_SITE);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_SAMPLE_SET_INDEX);
    set_indexes[0] = 0;
    set_indexes[1] = 4;
    ret = method(ts, 4, sample_set_sizes, samples, 1, set_indexes, 0, NULL, &result,
        TSK_STAT_SITE);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_SAMPLE_SET_INDEX);
}

static int
general_stat_identity(
    size_t K, const double *restrict X, size_t M, double *Y, void *params)
{
    size_t k;
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
    size_t num_samples = tsk_treeseq_get_num_samples(ts);
    double *W = malloc(num_samples * sizeof(double));
    tsk_id_t *stack = malloc(tsk_treeseq_get_num_nodes(ts) * sizeof(*stack));
    tsk_id_t root, u, v;
    int stack_top;
    double s, branch_length;
    double *sigma = malloc(tsk_treeseq_get_num_trees(ts) * sizeof(*sigma));
    tsk_tree_t tree;
    size_t j;
    CU_ASSERT_FATAL(W != NULL);
    CU_ASSERT_FATAL(stack != NULL);

    for (j = 0; j < num_samples; j++) {
        W[j] = 1;
    }

    ret = tsk_treeseq_general_stat(ts, 1, W, 1, general_stat_identity, NULL,
        tsk_treeseq_get_num_trees(ts), tsk_treeseq_get_breakpoints(ts), sigma,
        TSK_STAT_BRANCH | TSK_STAT_POLARISED | TSK_STAT_SPAN_NORMALISE);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_tree_init(&tree, ts, 0);
    CU_ASSERT_EQUAL(ret, 0);

    for (ret = tsk_tree_first(&tree); ret == 1; ret = tsk_tree_next(&tree)) {
        s = 0;
        for (root = tree.left_root; root != TSK_NULL; root = tree.right_sib[root]) {
            stack_top = 0;
            stack[stack_top] = root;
            while (stack_top >= 0) {
                u = stack[stack_top];
                stack_top--;

                for (v = tree.right_child[u]; v != TSK_NULL; v = tree.left_sib[v]) {
                    branch_length
                        = ts->tables->nodes.time[u] - ts->tables->nodes.time[v];
                    /* printf("u = %d v= %d s = %d bl = %f \n", u, v,
                     * tree.num_samples[v], */
                    /*         branch_length); */
                    s += branch_length * tree.num_samples[v];
                    stack_top++;
                    stack[stack_top] = v;
                }
            }
        }
        CU_ASSERT_DOUBLE_EQUAL_FATAL(sigma[tree.index], s, 1e-6);
    }
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    free(stack);
    tsk_tree_free(&tree);
    free(W);
    free(sigma);
}

static int
general_stat_sum(size_t K, const double *restrict X, size_t M, double *Y, void *params)
{
    size_t k, m;
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
verify_general_stat_dims(tsk_treeseq_t *ts, size_t K, size_t M, tsk_flags_t options)
{
    int ret;
    size_t num_samples = tsk_treeseq_get_num_samples(ts);
    double *W = malloc(K * num_samples * sizeof(double));
    /* We need this much space for NODE mode; no harm for other modes. */
    double *sigma = calloc(tsk_treeseq_get_num_nodes(ts) * M, sizeof(double));
    size_t j, k;
    CU_ASSERT_FATAL(W != NULL);

    for (j = 0; j < num_samples; j++) {
        for (k = 0; k < K; k++) {
            W[j * K + k] = 1;
        }
    }
    ret = tsk_treeseq_general_stat(
        ts, K, W, M, general_stat_sum, NULL, 0, NULL, sigma, options);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    free(W);
    free(sigma);
}

static void
verify_general_stat_windows(tsk_treeseq_t *ts, size_t num_windows, tsk_flags_t options)
{
    int ret;
    size_t num_samples = tsk_treeseq_get_num_samples(ts);
    double *W = malloc(num_samples * sizeof(double));
    size_t M = 5;
    /* We need this much space for NODE mode; no harm for other modes. */
    double *sigma
        = calloc(M * tsk_treeseq_get_num_nodes(ts) * num_windows, sizeof(double));
    double *windows = malloc((num_windows + 1) * sizeof(*windows));
    double L = tsk_treeseq_get_sequence_length(ts);
    size_t j;
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
        ts, 1, W, M, general_stat_sum, NULL, num_windows, windows, sigma, options);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    free(W);
    free(sigma);
    free(windows);
}

static void
verify_default_general_stat(tsk_treeseq_t *ts)
{
    int ret;
    size_t K = 2;
    size_t M = 1;
    size_t num_samples = tsk_treeseq_get_num_samples(ts);
    double *W = malloc(K * num_samples * sizeof(double));
    double sigma1, sigma2;
    size_t j, k;
    CU_ASSERT_FATAL(W != NULL);

    for (j = 0; j < num_samples; j++) {
        for (k = 0; k < K; k++) {
            W[j * K + k] = 1;
        }
    }
    ret = tsk_treeseq_general_stat(
        ts, K, W, M, general_stat_sum, NULL, 0, NULL, &sigma1, TSK_STAT_SITE);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_treeseq_general_stat(
        ts, K, W, M, general_stat_sum, NULL, 0, NULL, &sigma2, 0);
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
    double *result = malloc(n * n * sizeof(*result));

    CU_ASSERT_FATAL(sample_set_sizes != NULL);

    sample_set_sizes[0] = n - 2;
    sample_set_sizes[1] = 2;
    ret = tsk_treeseq_allele_frequency_spectrum(
        ts, 2, sample_set_sizes, samples, 0, NULL, result, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_treeseq_allele_frequency_spectrum(
        ts, 2, sample_set_sizes, samples, 0, NULL, result, TSK_STAT_POLARISED);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_treeseq_allele_frequency_spectrum(ts, 2, sample_set_sizes, samples, 0,
        NULL, result, TSK_STAT_POLARISED | TSK_STAT_SPAN_NORMALISE);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_treeseq_allele_frequency_spectrum(ts, 2, sample_set_sizes, samples, 0,
        NULL, result, TSK_STAT_BRANCH | TSK_STAT_POLARISED | TSK_STAT_SPAN_NORMALISE);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_treeseq_allele_frequency_spectrum(ts, 2, sample_set_sizes, samples, 0,
        NULL, result, TSK_STAT_BRANCH | TSK_STAT_SPAN_NORMALISE);
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
        &ts, 0, &W, 1, general_stat_sum, NULL, 0, NULL, &result, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_STATE_DIMS);

    ret = tsk_treeseq_general_stat(
        &ts, 1, &W, 0, general_stat_sum, NULL, 0, NULL, &result, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_RESULT_DIMS);

    /* Multiple stats*/
    ret = tsk_treeseq_general_stat(&ts, 1, &W, 1, general_stat_sum, NULL, 0, NULL,
        &result, TSK_STAT_SITE | TSK_STAT_BRANCH);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_MULTIPLE_STAT_MODES);
    ret = tsk_treeseq_general_stat(&ts, 1, &W, 1, general_stat_sum, NULL, 0, NULL,
        &result, TSK_STAT_SITE | TSK_STAT_NODE);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_MULTIPLE_STAT_MODES);
    ret = tsk_treeseq_general_stat(&ts, 1, &W, 1, general_stat_sum, NULL, 0, NULL,
        &result, TSK_STAT_BRANCH | TSK_STAT_NODE);
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
test_paper_ex_ld(void)
{
    tsk_treeseq_t ts;

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);
    verify_ld(&ts);
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
        &ts, 1, &sample_set_sizes, samples, 0, NULL, &pi, TSK_STAT_SITE);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_DOUBLE_EQUAL_FATAL(pi, 1.5, 1e-6);

    /* A sample set size of 1 leads to NaN */
    sample_set_sizes = 1;
    ret = tsk_treeseq_diversity(
        &ts, 1, &sample_set_sizes, samples, 0, NULL, &pi, TSK_STAT_SITE);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT(isnan((float) pi));

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
    size_t j;
    int ret;

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);
    weights = malloc(4 * sizeof(double));
    weights[0] = weights[1] = 0.0;
    weights[2] = weights[3] = 1.0;

    ret = tsk_treeseq_trait_covariance(&ts, 1, weights, 0, NULL, &result, TSK_STAT_SITE);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_DOUBLE_EQUAL_FATAL(result, 1.0 / 12.0, 1e-6);

    /* weights of 0 leads to 0 */
    for (j = 0; j < 4; j++) {
        weights[j] = 0.0;
    }
    ret = tsk_treeseq_trait_covariance(&ts, 1, weights, 0, NULL, &result, TSK_STAT_SITE);
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
    weights = malloc(4 * sizeof(double));
    weights[0] = weights[1] = 0.0;
    weights[2] = weights[3] = 1.0;

    ret = tsk_treeseq_trait_correlation(
        &ts, 1, weights, 0, NULL, &result, TSK_STAT_SITE);
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
    weights = malloc(4 * sizeof(double));
    covariates = malloc(8 * sizeof(double));
    weights[0] = weights[1] = 0.0;
    weights[2] = weights[3] = 1.0;
    covariates[0] = covariates[1] = 0.0;
    covariates[2] = covariates[3] = 1.0;
    covariates[4] = covariates[6] = 0.0;
    covariates[5] = covariates[7] = 1.0;

    ret = tsk_treeseq_trait_linear_model(
        &ts, 1, weights, 2, covariates, 0, NULL, &result, TSK_STAT_SITE);
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
        &ts, 1, &sample_set_sizes, samples, 0, NULL, &segsites, TSK_STAT_SITE);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_DOUBLE_EQUAL_FATAL(segsites, 3.0, 1e-6);

    /* A sample set size of 1 leads to 0 */
    sample_set_sizes = 1;
    ret = tsk_treeseq_segregating_sites(
        &ts, 1, &sample_set_sizes, samples, 0, NULL, &segsites, TSK_STAT_SITE);
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

    ret = tsk_treeseq_Y1(&ts, 1, &sample_set_sizes, samples, 0, NULL, &result, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* A sample set size of < 2 leads to NaN */
    sample_set_sizes = 1;
    ret = tsk_treeseq_Y1(&ts, 1, &sample_set_sizes, samples, 0, NULL, &result, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT(isnan((float) result));

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
        NULL, &result, TSK_STAT_SITE);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* sample_set[0] size = 1 with indexes = (0, 0) leads to NaN */
    sample_set_sizes[0] = 1;
    set_indexes[1] = 0;
    ret = tsk_treeseq_divergence(&ts, 2, sample_set_sizes, samples, 1, set_indexes, 0,
        NULL, &result, TSK_STAT_SITE);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT(isnan((float) result));

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
        set_indexes, 0, NULL, &result, TSK_STAT_SITE);
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
        &result, TSK_STAT_SITE);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* sample_set_size of 1 leads to NaN */
    sample_set_sizes[1] = 1;
    ret = tsk_treeseq_Y2(&ts, 2, sample_set_sizes, samples, 1, set_indexes, 0, NULL,
        &result, TSK_STAT_SITE);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT(isnan((float) result));

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
        &result, TSK_STAT_SITE);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* sample_set_size of 1 leads to NaN */
    sample_set_sizes[0] = 1;
    ret = tsk_treeseq_f2(&ts, 2, sample_set_sizes, samples, 1, set_indexes, 0, NULL,
        &result, TSK_STAT_SITE);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT(isnan((float) result));

    /* sample_set_size of 1 leads to NaN */
    sample_set_sizes[0] = 2;
    sample_set_sizes[1] = 1;
    ret = tsk_treeseq_f2(&ts, 2, sample_set_sizes, samples, 1, set_indexes, 0, NULL,
        &result, TSK_STAT_SITE);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT(isnan((float) result));

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
        &result, TSK_STAT_SITE);
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
        &result, TSK_STAT_SITE);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* sample_set_size of 1 leads to NaN */
    sample_set_sizes[0] = 1;
    ret = tsk_treeseq_f3(&ts, 3, sample_set_sizes, samples, 1, set_indexes, 0, NULL,
        &result, TSK_STAT_SITE);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT(isnan((float) result));

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
        &result, TSK_STAT_SITE);
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
        &ts, 2, sample_set_sizes, samples, 0, NULL, result, TSK_STAT_NODE);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_UNSUPPORTED_STAT_MODE);

    ret = tsk_treeseq_allele_frequency_spectrum(&ts, 2, sample_set_sizes, samples, 0,
        NULL, result, TSK_STAT_BRANCH | TSK_STAT_SITE);
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
        &ts, 1, sample_set_sizes, samples, 0, NULL, result, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(result[0], 0);
    CU_ASSERT_EQUAL_FATAL(result[1], 3.0);
    CU_ASSERT_EQUAL_FATAL(result[2], 0);

    ret = tsk_treeseq_allele_frequency_spectrum(
        &ts, 1, sample_set_sizes, samples, 0, NULL, result, TSK_STAT_POLARISED);
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

        { "test_nonbinary_ex_ld", test_nonbinary_ex_ld },
        { "test_nonbinary_ex_mean_descendants", test_nonbinary_ex_mean_descendants },
        { "test_nonbinary_ex_genealogical_nearest_neighbours",
            test_nonbinary_ex_genealogical_nearest_neighbours },
        { "test_nonbinary_ex_general_stat", test_nonbinary_ex_general_stat },
        { "test_nonbinary_ex_general_stat_errors",
            test_nonbinary_ex_general_stat_errors },

        { NULL, NULL },
    };
    return test_main(tests, argc, argv);
}
