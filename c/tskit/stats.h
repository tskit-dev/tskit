/*
 * MIT License
 *
 * Copyright (c) 2019 Tskit Developers
 * Copyright (c) 2016-2017 University of Oxford
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

#ifndef TSK_STATS_H
#define TSK_STATS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <tskit/trees.h>

int tsk_treeseq_genealogical_nearest_neighbours(tsk_treeseq_t *self, tsk_id_t *focal,
    size_t num_focal, tsk_id_t **reference_sets, size_t *reference_set_size,
    size_t num_reference_sets, tsk_flags_t options, double *ret_array);
int tsk_treeseq_mean_descendants(tsk_treeseq_t *self, tsk_id_t **reference_sets,
    size_t *reference_set_size, size_t num_reference_sets, tsk_flags_t options,
    double *ret_array);

/* TODO change all these size_t's to tsk_size_t */

typedef int general_stat_func_t(size_t K, double *X, size_t M, double *Y, void *params);

int tsk_treeseq_general_stat(tsk_treeseq_t *self, size_t K, double *W, size_t M,
    general_stat_func_t *f, void *f_params, size_t num_windows, double *windows,
    double *sigma, tsk_flags_t options);

/* One way weighted stats */

typedef int one_way_weighted_method(tsk_treeseq_t *self, tsk_size_t num_weights,
    double *weights, tsk_size_t num_windows, double *windows, double *result,
    tsk_flags_t options);

int tsk_treeseq_trait_covariance(tsk_treeseq_t *self, tsk_size_t num_weights,
    double *weights, tsk_size_t num_windows, double *windows, double *result,
    tsk_flags_t options);
int tsk_treeseq_trait_correlation(tsk_treeseq_t *self, tsk_size_t num_weights,
    double *weights, tsk_size_t num_windows, double *windows, double *result,
    tsk_flags_t options);

/* One way weighted stats with covariates */

typedef int one_way_covariates_method(tsk_treeseq_t *self, tsk_size_t num_weights,
    double *weights, tsk_size_t num_covariates, double *covariates,
    tsk_size_t num_windows, double *windows, double *result, tsk_flags_t options);

int tsk_treeseq_trait_regression(tsk_treeseq_t *self, tsk_size_t num_weights,
    double *weights, tsk_size_t num_covariates, double *covariates,
    tsk_size_t num_windows, double *windows, double *result, tsk_flags_t options);

/* One way sample set stats */

typedef int one_way_sample_stat_method(tsk_treeseq_t *self, tsk_size_t num_sample_sets,
    tsk_size_t *sample_set_sizes, tsk_id_t *sample_sets, tsk_size_t num_windows,
    double *windows, double *result, tsk_flags_t options);

int tsk_treeseq_diversity(tsk_treeseq_t *self, tsk_size_t num_sample_sets,
    tsk_size_t *sample_set_sizes, tsk_id_t *sample_sets, tsk_size_t num_windows,
    double *windows, double *result, tsk_flags_t options);
int tsk_treeseq_segregating_sites(tsk_treeseq_t *self, tsk_size_t num_sample_sets,
    tsk_size_t *sample_set_sizes, tsk_id_t *sample_sets, tsk_size_t num_windows,
    double *windows, double *result, tsk_flags_t options);
int tsk_treeseq_Y1(tsk_treeseq_t *self, tsk_size_t num_sample_sets,
    tsk_size_t *sample_set_sizes, tsk_id_t *sample_sets, tsk_size_t num_windows,
    double *windows, double *result, tsk_flags_t options);
int tsk_treeseq_allele_frequency_spectrum(tsk_treeseq_t *self,
    tsk_size_t num_sample_sets, tsk_size_t *sample_set_sizes, tsk_id_t *sample_sets,
    tsk_size_t num_windows, double *windows, double *result, tsk_flags_t options);

/* Two way sample set stats */

typedef int general_sample_stat_method(tsk_treeseq_t *self, tsk_size_t num_sample_sets,
    tsk_size_t *sample_set_sizes, tsk_id_t *sample_sets, tsk_size_t num_indexes,
    tsk_id_t *indexes, tsk_size_t num_windows, double *windows, double *result,
    tsk_flags_t options);

int tsk_treeseq_divergence(tsk_treeseq_t *self, tsk_size_t num_sample_sets,
    tsk_size_t *sample_set_sizes, tsk_id_t *sample_sets, tsk_size_t num_index_tuples,
    tsk_id_t *index_tuples, tsk_size_t num_windows, double *windows, double *result,
    tsk_flags_t options);
int tsk_treeseq_Y2(tsk_treeseq_t *self, tsk_size_t num_sample_sets,
    tsk_size_t *sample_set_sizes, tsk_id_t *sample_sets, tsk_size_t num_index_tuples,
    tsk_id_t *index_tuples, tsk_size_t num_windows, double *windows, double *result,
    tsk_flags_t options);
int tsk_treeseq_f2(tsk_treeseq_t *self, tsk_size_t num_sample_sets,
    tsk_size_t *sample_set_sizes, tsk_id_t *sample_sets, tsk_size_t num_index_tuples,
    tsk_id_t *index_tuples, tsk_size_t num_windows, double *windows, double *result,
    tsk_flags_t options);

/* Three way sample set stats */
int tsk_treeseq_Y3(tsk_treeseq_t *self, tsk_size_t num_sample_sets,
    tsk_size_t *sample_set_sizes, tsk_id_t *sample_sets, tsk_size_t num_index_tuples,
    tsk_id_t *index_tuples, tsk_size_t num_windows, double *windows, double *result,
    tsk_flags_t options);
int tsk_treeseq_f3(tsk_treeseq_t *self, tsk_size_t num_sample_sets,
    tsk_size_t *sample_set_sizes, tsk_id_t *sample_sets, tsk_size_t num_index_tuples,
    tsk_id_t *index_tuples, tsk_size_t num_windows, double *windows, double *result,
    tsk_flags_t options);

/* Four way sample set stats */
int tsk_treeseq_f4(tsk_treeseq_t *self, tsk_size_t num_sample_sets,
    tsk_size_t *sample_set_sizes, tsk_id_t *sample_sets, tsk_size_t num_index_tuples,
    tsk_id_t *index_tuples, tsk_size_t num_windows, double *windows, double *result,
    tsk_flags_t options);

/****************************************************************************/
/* LD calculator */
/****************************************************************************/

typedef struct {
    tsk_tree_t *outer_tree;
    tsk_tree_t *inner_tree;
    tsk_size_t num_sites;
    int tree_changed;
    tsk_treeseq_t *tree_sequence;
} tsk_ld_calc_t;

int tsk_ld_calc_init(tsk_ld_calc_t *self, tsk_treeseq_t *tree_sequence);
int tsk_ld_calc_free(tsk_ld_calc_t *self);
void tsk_ld_calc_print_state(tsk_ld_calc_t *self, FILE *out);
int tsk_ld_calc_get_r2(tsk_ld_calc_t *self, tsk_id_t a, tsk_id_t b, double *r2);
int tsk_ld_calc_get_r2_array(tsk_ld_calc_t *self, tsk_id_t a, int direction,
    tsk_size_t max_sites, double max_distance, double *r2, tsk_size_t *num_r2_values);

#ifdef __cplusplus
}
#endif
#endif
