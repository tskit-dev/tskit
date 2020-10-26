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

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include <tskit/stats.h>

#define GET_2D_ROW(array, row_len, row) (array + (((size_t)(row_len)) * (size_t) row))

static inline double *
GET_3D_ROW(
    double *base, size_t num_nodes, size_t output_dim, size_t window_index, tsk_id_t u)
{
    size_t offset = window_index * num_nodes * output_dim + ((size_t) u) * output_dim;
    return base + offset;
}

/* Increments the n-dimensional array with the specified shape by the specified value at
 * the specified coordinate. */
static inline void
increment_nd_array_value(double *array, tsk_size_t n, const tsk_size_t *shape,
    const tsk_size_t *coordinate, double value)
{
    size_t offset = 0;
    size_t product = 1;
    int k;

    for (k = (int) n - 1; k >= 0; k--) {
        tsk_bug_assert(coordinate[k] < shape[k]);
        offset += coordinate[k] * product;
        product *= shape[k];
    }
    array[offset] += value;
}

/* TODO flatten the reference sets input here and follow the same pattern used
 * in diversity, divergence, etc. */
int TSK_WARN_UNUSED
tsk_treeseq_genealogical_nearest_neighbours(tsk_treeseq_t *self, tsk_id_t *focal,
    size_t num_focal, tsk_id_t **reference_sets, size_t *reference_set_size,
    size_t num_reference_sets, tsk_flags_t TSK_UNUSED(options), double *ret_array)
{
    int ret = 0;
    tsk_id_t u, v, p;
    size_t j;
    /* TODO It's probably not worth bothering with the int16_t here. */
    int16_t k, focal_reference_set;
    /* We use the K'th element of the array for the total. */
    const int16_t K = (int16_t)(num_reference_sets + 1);
    size_t num_nodes = self->tables->nodes.num_rows;
    const tsk_id_t num_edges = (tsk_id_t) self->tables->edges.num_rows;
    const tsk_id_t *restrict I = self->tables->indexes.edge_insertion_order;
    const tsk_id_t *restrict O = self->tables->indexes.edge_removal_order;
    const double *restrict edge_left = self->tables->edges.left;
    const double *restrict edge_right = self->tables->edges.right;
    const tsk_id_t *restrict edge_parent = self->tables->edges.parent;
    const tsk_id_t *restrict edge_child = self->tables->edges.child;
    const double sequence_length = self->tables->sequence_length;
    tsk_id_t tj, tk, h;
    double left, right, *A_row, scale, tree_length;
    tsk_id_t *restrict parent = malloc(num_nodes * sizeof(*parent));
    double *restrict length = calloc(num_focal, sizeof(*length));
    uint32_t *restrict ref_count = calloc(((size_t) K) * num_nodes, sizeof(*ref_count));
    int16_t *restrict reference_set_map = malloc(num_nodes * sizeof(*reference_set_map));
    uint32_t *restrict row = NULL;
    uint32_t *restrict child_row = NULL;
    uint32_t total, delta;

    /* We support a max of 8K focal sets */
    if (num_reference_sets == 0 || num_reference_sets > (INT16_MAX - 1)) {
        /* TODO: more specific error */
        ret = TSK_ERR_BAD_PARAM_VALUE;
        goto out;
    }
    if (parent == NULL || ref_count == NULL || reference_set_map == NULL
        || length == NULL) {
        ret = TSK_ERR_NO_MEMORY;
        goto out;
    }

    memset(parent, 0xff, num_nodes * sizeof(*parent));
    memset(reference_set_map, 0xff, num_nodes * sizeof(*reference_set_map));
    memset(ret_array, 0, num_focal * num_reference_sets * sizeof(*ret_array));

    total = 0; /* keep the compiler happy */

    /* Set the initial conditions and check the input. */
    for (k = 0; k < (int16_t) num_reference_sets; k++) {
        for (j = 0; j < reference_set_size[k]; j++) {
            u = reference_sets[k][j];
            if (u < 0 || u >= (tsk_id_t) num_nodes) {
                ret = TSK_ERR_NODE_OUT_OF_BOUNDS;
                goto out;
            }
            if (reference_set_map[u] != TSK_NULL) {
                /* FIXME Technically inaccurate here: duplicate focal not sample */
                ret = TSK_ERR_DUPLICATE_SAMPLE;
                goto out;
            }
            reference_set_map[u] = k;
            row = GET_2D_ROW(ref_count, K, u);
            row[k] = 1;
            /* Also set the count for the total among all sets */
            row[K - 1] = 1;
        }
    }
    for (j = 0; j < num_focal; j++) {
        u = focal[j];
        if (u < 0 || u >= (tsk_id_t) num_nodes) {
            ret = TSK_ERR_NODE_OUT_OF_BOUNDS;
            goto out;
        }
    }

    /* Iterate over the trees */
    tj = 0;
    tk = 0;
    left = 0;
    while (tj < num_edges || left < sequence_length) {
        while (tk < num_edges && edge_right[O[tk]] == left) {
            h = O[tk];
            tk++;
            u = edge_child[h];
            v = edge_parent[h];
            parent[u] = TSK_NULL;
            child_row = GET_2D_ROW(ref_count, K, u);
            while (v != TSK_NULL) {
                row = GET_2D_ROW(ref_count, K, v);
                for (k = 0; k < K; k++) {
                    row[k] -= child_row[k];
                }
                v = parent[v];
            }
        }
        while (tj < num_edges && edge_left[I[tj]] == left) {
            h = I[tj];
            tj++;
            u = edge_child[h];
            v = edge_parent[h];
            parent[u] = v;
            child_row = GET_2D_ROW(ref_count, K, u);
            while (v != TSK_NULL) {
                row = GET_2D_ROW(ref_count, K, v);
                for (k = 0; k < K; k++) {
                    row[k] += child_row[k];
                }
                v = parent[v];
            }
        }
        right = sequence_length;
        if (tj < num_edges) {
            right = TSK_MIN(right, edge_left[I[tj]]);
        }
        if (tk < num_edges) {
            right = TSK_MIN(right, edge_right[O[tk]]);
        }

        tree_length = right - left;
        /* Process this tree */
        for (j = 0; j < num_focal; j++) {
            u = focal[j];
            focal_reference_set = reference_set_map[u];
            delta = focal_reference_set != -1;
            p = u;
            while (p != TSK_NULL) {
                row = GET_2D_ROW(ref_count, K, p);
                total = row[K - 1];
                if (total > delta) {
                    break;
                }
                p = parent[p];
            }
            if (p != TSK_NULL) {
                length[j] += tree_length;
                scale = tree_length / (total - delta);
                A_row = GET_2D_ROW(ret_array, num_reference_sets, j);
                for (k = 0; k < K - 1; k++) {
                    A_row[k] += row[k] * scale;
                }
                if (focal_reference_set != -1) {
                    /* Remove the contribution for the reference set u belongs to and
                     * insert the correct value. The long-hand version is
                     * A_row[k] = A_row[k] - row[k] * scale + (row[k] - 1) * scale;
                     * which cancels to give: */
                    A_row[focal_reference_set] -= scale;
                }
            }
        }

        /* Move on to the next tree */
        left = right;
    }

    /* Divide by the accumulated length for each node to normalise */
    for (j = 0; j < num_focal; j++) {
        A_row = GET_2D_ROW(ret_array, num_reference_sets, j);
        if (length[j] > 0) {
            for (k = 0; k < K - 1; k++) {
                A_row[k] /= length[j];
            }
        }
    }
out:
    /* Can't use msp_safe_free here because of restrict */
    if (parent != NULL) {
        free(parent);
    }
    if (ref_count != NULL) {
        free(ref_count);
    }
    if (reference_set_map != NULL) {
        free(reference_set_map);
    }
    if (length != NULL) {
        free(length);
    }
    return ret;
}

int TSK_WARN_UNUSED
tsk_treeseq_mean_descendants(tsk_treeseq_t *self, tsk_id_t **reference_sets,
    size_t *reference_set_size, size_t num_reference_sets,
    tsk_flags_t TSK_UNUSED(options), double *ret_array)
{
    int ret = 0;
    tsk_id_t u, v;
    size_t j;
    int32_t k;
    /* We use the K'th element of the array for the total. */
    const int32_t K = (int32_t)(num_reference_sets + 1);
    size_t num_nodes = self->tables->nodes.num_rows;
    const tsk_id_t num_edges = (tsk_id_t) self->tables->edges.num_rows;
    const tsk_id_t *restrict I = self->tables->indexes.edge_insertion_order;
    const tsk_id_t *restrict O = self->tables->indexes.edge_removal_order;
    const double *restrict edge_left = self->tables->edges.left;
    const double *restrict edge_right = self->tables->edges.right;
    const tsk_id_t *restrict edge_parent = self->tables->edges.parent;
    const tsk_id_t *restrict edge_child = self->tables->edges.child;
    const double sequence_length = self->tables->sequence_length;
    tsk_id_t tj, tk, h;
    double left, right, length, *restrict C_row;
    tsk_id_t *restrict parent = malloc(num_nodes * sizeof(*parent));
    uint32_t *restrict ref_count = calloc(num_nodes * ((size_t) K), sizeof(*ref_count));
    double *restrict last_update = calloc(num_nodes, sizeof(*last_update));
    double *restrict total_length = calloc(num_nodes, sizeof(*total_length));
    uint32_t *restrict row, *restrict child_row;

    if (num_reference_sets == 0 || num_reference_sets > (INT32_MAX - 1)) {
        /* TODO: more specific error */
        ret = TSK_ERR_BAD_PARAM_VALUE;
        goto out;
    }
    if (parent == NULL || ref_count == NULL || last_update == NULL
        || total_length == NULL) {
        ret = TSK_ERR_NO_MEMORY;
        goto out;
    }
    /* TODO add check for duplicate values in the reference sets */

    memset(parent, 0xff, num_nodes * sizeof(*parent));
    memset(ret_array, 0, num_nodes * num_reference_sets * sizeof(*ret_array));

    /* Set the initial conditions and check the input. */
    for (k = 0; k < (int32_t) num_reference_sets; k++) {
        for (j = 0; j < reference_set_size[k]; j++) {
            u = reference_sets[k][j];
            if (u < 0 || u >= (tsk_id_t) num_nodes) {
                ret = TSK_ERR_NODE_OUT_OF_BOUNDS;
                goto out;
            }
            row = GET_2D_ROW(ref_count, K, u);
            row[k] = 1;
            /* Also set the count for the total among all sets */
            row[K - 1] = 1;
        }
    }

    /* Iterate over the trees */
    tj = 0;
    tk = 0;
    left = 0;
    while (tj < num_edges || left < sequence_length) {
        while (tk < num_edges && edge_right[O[tk]] == left) {
            h = O[tk];
            tk++;
            u = edge_child[h];
            v = edge_parent[h];
            parent[u] = TSK_NULL;
            child_row = GET_2D_ROW(ref_count, K, u);
            while (v != TSK_NULL) {
                row = GET_2D_ROW(ref_count, K, v);
                if (last_update[v] != left) {
                    if (row[K - 1] > 0) {
                        length = left - last_update[v];
                        C_row = GET_2D_ROW(ret_array, num_reference_sets, v);
                        for (k = 0; k < (int32_t) num_reference_sets; k++) {
                            C_row[k] += length * row[k];
                        }
                        total_length[v] += length;
                    }
                    last_update[v] = left;
                }
                for (k = 0; k < K; k++) {
                    row[k] -= child_row[k];
                }
                v = parent[v];
            }
        }
        while (tj < num_edges && edge_left[I[tj]] == left) {
            h = I[tj];
            tj++;
            u = edge_child[h];
            v = edge_parent[h];
            parent[u] = v;
            child_row = GET_2D_ROW(ref_count, K, u);
            while (v != TSK_NULL) {
                row = GET_2D_ROW(ref_count, K, v);
                if (last_update[v] != left) {
                    if (row[K - 1] > 0) {
                        length = left - last_update[v];
                        C_row = GET_2D_ROW(ret_array, num_reference_sets, v);
                        for (k = 0; k < (int32_t) num_reference_sets; k++) {
                            C_row[k] += length * row[k];
                        }
                        total_length[v] += length;
                    }
                    last_update[v] = left;
                }
                for (k = 0; k < K; k++) {
                    row[k] += child_row[k];
                }
                v = parent[v];
            }
        }
        right = sequence_length;
        if (tj < num_edges) {
            right = TSK_MIN(right, edge_left[I[tj]]);
        }
        if (tk < num_edges) {
            right = TSK_MIN(right, edge_right[O[tk]]);
        }
        left = right;
    }

    /* Add the stats for the last tree and divide by the total length that
     * each node was an ancestor to > 0 of the reference nodes. */
    for (v = 0; v < (tsk_id_t) num_nodes; v++) {
        row = GET_2D_ROW(ref_count, K, v);
        C_row = GET_2D_ROW(ret_array, num_reference_sets, v);
        if (row[K - 1] > 0) {
            length = sequence_length - last_update[v];
            total_length[v] += length;
            for (k = 0; k < (int32_t) num_reference_sets; k++) {
                C_row[k] += length * row[k];
            }
        }
        if (total_length[v] > 0) {
            length = total_length[v];
            for (k = 0; k < (int32_t) num_reference_sets; k++) {
                C_row[k] /= length;
            }
        }
    }

out:
    /* Can't use msp_safe_free here because of restrict */
    if (parent != NULL) {
        free(parent);
    }
    if (ref_count != NULL) {
        free(ref_count);
    }
    if (last_update != NULL) {
        free(last_update);
    }
    if (total_length != NULL) {
        free(total_length);
    }
    return ret;
}

/***********************************
 * General stats framework
 ***********************************/

static int
tsk_treeseq_check_windows(tsk_treeseq_t *self, size_t num_windows, double *windows)
{
    int ret = TSK_ERR_BAD_WINDOWS;
    size_t j;

    if (num_windows < 1) {
        ret = TSK_ERR_BAD_NUM_WINDOWS;
        goto out;
    }
    /* TODO these restrictions can be lifted later if we want a specific interval. */
    if (windows[0] != 0) {
        goto out;
    }
    if (windows[num_windows] != self->tables->sequence_length) {
        goto out;
    }
    for (j = 0; j < num_windows; j++) {
        if (windows[j] >= windows[j + 1]) {
            goto out;
        }
    }
    ret = 0;
out:
    return ret;
}

/* TODO make these functions more consistent in how the arguments are ordered */

static inline void
update_state(double *X, size_t state_dim, tsk_id_t dest, tsk_id_t source, int sign)
{
    size_t k;
    double *X_dest = GET_2D_ROW(X, state_dim, dest);
    double *X_source = GET_2D_ROW(X, state_dim, source);

    for (k = 0; k < state_dim; k++) {
        X_dest[k] += sign * X_source[k];
    }
}

static inline int
update_node_summary(tsk_id_t u, size_t result_dim, double *node_summary, double *X,
    size_t state_dim, general_stat_func_t *f, void *f_params)
{
    double *X_u = GET_2D_ROW(X, state_dim, u);
    double *summary_u = GET_2D_ROW(node_summary, result_dim, u);

    return f(state_dim, X_u, result_dim, summary_u, f_params);
}

static inline void
update_running_sum(tsk_id_t u, double sign, const double *restrict branch_length,
    const double *summary, size_t result_dim, double *running_sum)
{
    const double *summary_u = GET_2D_ROW(summary, result_dim, u);
    const double x = sign * branch_length[u];
    size_t m;

    for (m = 0; m < result_dim; m++) {
        running_sum[m] += x * summary_u[m];
    }
}

static int
tsk_treeseq_branch_general_stat(tsk_treeseq_t *self, size_t state_dim,
    double *sample_weights, size_t result_dim, general_stat_func_t *f, void *f_params,
    size_t num_windows, double *windows, double *result, tsk_flags_t TSK_UNUSED(options))
{
    int ret = 0;
    tsk_id_t u, v;
    size_t j, k, tree_index, window_index;
    size_t num_nodes = self->tables->nodes.num_rows;
    const tsk_id_t num_edges = (tsk_id_t) self->tables->edges.num_rows;
    const tsk_id_t *restrict I = self->tables->indexes.edge_insertion_order;
    const tsk_id_t *restrict O = self->tables->indexes.edge_removal_order;
    const double *restrict edge_left = self->tables->edges.left;
    const double *restrict edge_right = self->tables->edges.right;
    const tsk_id_t *restrict edge_parent = self->tables->edges.parent;
    const tsk_id_t *restrict edge_child = self->tables->edges.child;
    const double *restrict time = self->tables->nodes.time;
    const double sequence_length = self->tables->sequence_length;
    tsk_id_t *restrict parent = malloc(num_nodes * sizeof(*parent));
    double *restrict branch_length = calloc(num_nodes, sizeof(*branch_length));
    tsk_id_t tj, tk, h;
    double t_left, t_right, w_left, w_right, left, right, scale;
    double *state_u, *weight_u, *result_row, *summary_u;
    double *state = calloc(num_nodes * state_dim, sizeof(*state));
    double *summary = calloc(num_nodes * result_dim, sizeof(*summary));
    double *running_sum = calloc(result_dim, sizeof(*running_sum));

    if (parent == NULL || branch_length == NULL || state == NULL || running_sum == NULL
        || summary == NULL) {
        ret = TSK_ERR_NO_MEMORY;
        goto out;
    }
    memset(parent, 0xff, num_nodes * sizeof(*parent));

    /* Set the initial conditions */
    for (j = 0; j < self->num_samples; j++) {
        u = self->samples[j];
        state_u = GET_2D_ROW(state, state_dim, u);
        weight_u = GET_2D_ROW(sample_weights, state_dim, j);
        memcpy(state_u, weight_u, state_dim * sizeof(*state_u));
        summary_u = GET_2D_ROW(summary, result_dim, u);
        ret = f(state_dim, state_u, result_dim, summary_u, f_params);
        if (ret != 0) {
            goto out;
        }
    }
    memset(result, 0, num_windows * result_dim * sizeof(*result));

    /* Iterate over the trees */
    tj = 0;
    tk = 0;
    t_left = 0;
    tree_index = 0;
    window_index = 0;
    while (tj < num_edges || t_left < sequence_length) {
        while (tk < num_edges && edge_right[O[tk]] == t_left) {
            h = O[tk];
            tk++;

            u = edge_child[h];
            update_running_sum(u, -1, branch_length, summary, result_dim, running_sum);
            parent[u] = TSK_NULL;
            branch_length[u] = 0;

            u = edge_parent[h];
            while (u != TSK_NULL) {
                update_running_sum(
                    u, -1, branch_length, summary, result_dim, running_sum);
                update_state(state, state_dim, u, edge_child[h], -1);
                ret = update_node_summary(
                    u, result_dim, summary, state, state_dim, f, f_params);
                if (ret != 0) {
                    goto out;
                }
                update_running_sum(
                    u, +1, branch_length, summary, result_dim, running_sum);
                u = parent[u];
            }
        }

        while (tj < num_edges && edge_left[I[tj]] == t_left) {
            h = I[tj];
            tj++;

            u = edge_child[h];
            v = edge_parent[h];
            parent[u] = v;
            branch_length[u] = time[v] - time[u];
            update_running_sum(u, +1, branch_length, summary, result_dim, running_sum);

            u = v;
            while (u != TSK_NULL) {
                update_running_sum(
                    u, -1, branch_length, summary, result_dim, running_sum);
                update_state(state, state_dim, u, edge_child[h], +1);
                ret = update_node_summary(
                    u, result_dim, summary, state, state_dim, f, f_params);
                if (ret != 0) {
                    goto out;
                }
                update_running_sum(
                    u, +1, branch_length, summary, result_dim, running_sum);
                u = parent[u];
            }
        }

        t_right = sequence_length;
        if (tj < num_edges) {
            t_right = TSK_MIN(t_right, edge_left[I[tj]]);
        }
        if (tk < num_edges) {
            t_right = TSK_MIN(t_right, edge_right[O[tk]]);
        }

        while (windows[window_index] < t_right) {
            tsk_bug_assert(window_index < num_windows);
            w_left = windows[window_index];
            w_right = windows[window_index + 1];
            left = TSK_MAX(t_left, w_left);
            right = TSK_MIN(t_right, w_right);
            scale = (right - left);
            tsk_bug_assert(scale > 0);
            result_row = GET_2D_ROW(result, result_dim, window_index);
            for (k = 0; k < result_dim; k++) {
                result_row[k] += running_sum[k] * scale;
            }

            if (w_right <= t_right) {
                window_index++;
            } else {
                /* This interval crosses a tree boundary, so we update it again in the */
                /* for the next tree */
                break;
            }
        }
        /* Move to the next tree */
        t_left = t_right;
        tree_index++;
    }
    tsk_bug_assert(window_index == num_windows);
out:
    /* Can't use msp_safe_free here because of restrict */
    if (parent != NULL) {
        free(parent);
    }
    if (branch_length != NULL) {
        free(branch_length);
    }
    tsk_safe_free(state);
    tsk_safe_free(summary);
    tsk_safe_free(running_sum);
    return ret;
}

static int
get_allele_weights(tsk_site_t *site, double *state, size_t state_dim,
    double *total_weight, tsk_size_t *ret_num_alleles, double **ret_allele_states)
{
    int ret = 0;
    size_t k;
    tsk_mutation_t mutation, parent_mut;
    tsk_size_t mutation_index, allele, num_alleles, alt_allele_length;
    /* The allele table */
    size_t max_alleles = site->mutations_length + 1;
    const char **alleles = malloc(max_alleles * sizeof(*alleles));
    tsk_size_t *allele_lengths = calloc(max_alleles, sizeof(*allele_lengths));
    double *allele_states = calloc(max_alleles * state_dim, sizeof(*allele_states));
    double *allele_row, *state_row;
    const char *alt_allele;

    if (alleles == NULL || allele_lengths == NULL || allele_states == NULL) {
        ret = TSK_ERR_NO_MEMORY;
        goto out;
    }

    tsk_bug_assert(state != NULL);
    alleles[0] = site->ancestral_state;
    allele_lengths[0] = site->ancestral_state_length;
    memcpy(allele_states, total_weight, state_dim * sizeof(*allele_states));
    num_alleles = 1;

    for (mutation_index = 0; mutation_index < site->mutations_length; mutation_index++) {
        mutation = site->mutations[mutation_index];
        /* Compute the allele index for this derived state value. */
        allele = 0;
        while (allele < num_alleles) {
            if (mutation.derived_state_length == allele_lengths[allele]
                && memcmp(
                       mutation.derived_state, alleles[allele], allele_lengths[allele])
                       == 0) {
                break;
            }
            allele++;
        }
        if (allele == num_alleles) {
            tsk_bug_assert(allele < max_alleles);
            alleles[allele] = mutation.derived_state;
            allele_lengths[allele] = mutation.derived_state_length;
            num_alleles++;
        }

        /* Add the state for the the mutation's node to this allele */
        state_row = GET_2D_ROW(state, state_dim, mutation.node);
        allele_row = GET_2D_ROW(allele_states, state_dim, allele);
        for (k = 0; k < state_dim; k++) {
            allele_row[k] += state_row[k];
        }

        /* Get the index for the alternate allele that we must substract from */
        alt_allele = site->ancestral_state;
        alt_allele_length = site->ancestral_state_length;
        if (mutation.parent != TSK_NULL) {
            parent_mut = site->mutations[mutation.parent - site->mutations[0].id];
            alt_allele = parent_mut.derived_state;
            alt_allele_length = parent_mut.derived_state_length;
        }
        allele = 0;
        while (allele < num_alleles) {
            if (alt_allele_length == allele_lengths[allele]
                && memcmp(alt_allele, alleles[allele], allele_lengths[allele]) == 0) {
                break;
            }
            allele++;
        }
        tsk_bug_assert(allele < num_alleles);

        allele_row = GET_2D_ROW(allele_states, state_dim, allele);
        for (k = 0; k < state_dim; k++) {
            allele_row[k] -= state_row[k];
        }
    }
    *ret_num_alleles = num_alleles;
    *ret_allele_states = allele_states;
    allele_states = NULL;
out:
    tsk_safe_free(alleles);
    tsk_safe_free(allele_lengths);
    tsk_safe_free(allele_states);
    return ret;
}

static int
compute_general_stat_site_result(tsk_site_t *site, double *state, size_t state_dim,
    size_t result_dim, general_stat_func_t *f, void *f_params, double *total_weight,
    bool polarised, double *result)
{
    int ret = 0;
    size_t k;
    tsk_size_t allele, num_alleles;
    double *allele_states;
    double *result_tmp = calloc(result_dim, sizeof(*result_tmp));

    if (result_tmp == NULL) {
        ret = TSK_ERR_NO_MEMORY;
        goto out;
    }
    memset(result, 0, result_dim * sizeof(*result));

    ret = get_allele_weights(
        site, state, state_dim, total_weight, &num_alleles, &allele_states);
    if (ret != 0) {
        goto out;
    }
    /* Sum over the allele weights. Skip the ancestral state if this is a polarised stat
     */
    for (allele = polarised ? 1 : 0; allele < num_alleles; allele++) {
        ret = f(state_dim, GET_2D_ROW(allele_states, state_dim, allele), result_dim,
            result_tmp, f_params);
        if (ret != 0) {
            goto out;
        }
        for (k = 0; k < result_dim; k++) {
            result[k] += result_tmp[k];
        }
    }
out:
    tsk_safe_free(result_tmp);
    tsk_safe_free(allele_states);
    return ret;
}

static int
tsk_treeseq_site_general_stat(tsk_treeseq_t *self, size_t state_dim,
    double *sample_weights, size_t result_dim, general_stat_func_t *f, void *f_params,
    size_t num_windows, double *windows, double *result, tsk_flags_t options)
{
    int ret = 0;
    tsk_id_t u, v;
    size_t j, k, tree_site, tree_index, window_index;
    size_t num_nodes = self->tables->nodes.num_rows;
    const tsk_id_t num_edges = (tsk_id_t) self->tables->edges.num_rows;
    const tsk_id_t *restrict I = self->tables->indexes.edge_insertion_order;
    const tsk_id_t *restrict O = self->tables->indexes.edge_removal_order;
    const double *restrict edge_left = self->tables->edges.left;
    const double *restrict edge_right = self->tables->edges.right;
    const tsk_id_t *restrict edge_parent = self->tables->edges.parent;
    const tsk_id_t *restrict edge_child = self->tables->edges.child;
    const double sequence_length = self->tables->sequence_length;
    tsk_id_t *restrict parent = malloc(num_nodes * sizeof(*parent));
    tsk_site_t *site;
    tsk_id_t tj, tk, h;
    double t_left, t_right;
    double *state_u, *weight_u, *result_row;
    double *state = calloc(num_nodes * state_dim, sizeof(*state));
    double *total_weight = calloc(state_dim, sizeof(*total_weight));
    double *site_result = calloc(result_dim, sizeof(*site_result));
    bool polarised = false;

    if (parent == NULL || state == NULL || total_weight == NULL || site_result == NULL) {
        ret = TSK_ERR_NO_MEMORY;
        goto out;
    }
    memset(parent, 0xff, num_nodes * sizeof(*parent));

    if (options & TSK_STAT_POLARISED) {
        polarised = true;
    }

    /* Set the initial conditions */
    for (j = 0; j < self->num_samples; j++) {
        u = self->samples[j];
        state_u = GET_2D_ROW(state, state_dim, u);
        weight_u = GET_2D_ROW(sample_weights, state_dim, j);
        memcpy(state_u, weight_u, state_dim * sizeof(*state_u));
        for (k = 0; k < state_dim; k++) {
            total_weight[k] += weight_u[k];
        }
    }
    memset(result, 0, num_windows * result_dim * sizeof(*result));

    /* Iterate over the trees */
    tj = 0;
    tk = 0;
    t_left = 0;
    tree_index = 0;
    window_index = 0;
    while (tj < num_edges || t_left < sequence_length) {
        while (tk < num_edges && edge_right[O[tk]] == t_left) {
            h = O[tk];
            tk++;
            u = edge_child[h];
            v = edge_parent[h];
            while (v != TSK_NULL) {
                update_state(state, state_dim, v, u, -1);
                v = parent[v];
            }
            parent[u] = TSK_NULL;
        }

        while (tj < num_edges && edge_left[I[tj]] == t_left) {
            h = I[tj];
            tj++;
            u = edge_child[h];
            v = edge_parent[h];
            parent[u] = v;
            while (v != TSK_NULL) {
                update_state(state, state_dim, v, u, +1);
                v = parent[v];
            }
        }
        t_right = sequence_length;
        if (tj < num_edges) {
            t_right = TSK_MIN(t_right, edge_left[I[tj]]);
        }
        if (tk < num_edges) {
            t_right = TSK_MIN(t_right, edge_right[O[tk]]);
        }

        /* Update the sites */
        for (tree_site = 0; tree_site < self->tree_sites_length[tree_index];
             tree_site++) {
            site = self->tree_sites[tree_index] + tree_site;
            ret = compute_general_stat_site_result(site, state, state_dim, result_dim, f,
                f_params, total_weight, polarised, site_result);
            if (ret != 0) {
                goto out;
            }

            while (windows[window_index + 1] <= site->position) {
                window_index++;
                tsk_bug_assert(window_index < num_windows);
            }
            tsk_bug_assert(windows[window_index] <= site->position);
            tsk_bug_assert(site->position < windows[window_index + 1]);
            result_row = GET_2D_ROW(result, result_dim, window_index);
            for (k = 0; k < result_dim; k++) {
                result_row[k] += site_result[k];
            }
        }
        tree_index++;
        t_left = t_right;
    }
out:
    /* Can't use msp_safe_free here because of restrict */
    if (parent != NULL) {
        free(parent);
    }
    tsk_safe_free(state);
    tsk_safe_free(total_weight);
    tsk_safe_free(site_result);
    return ret;
}

static inline void
increment_row(size_t length, double multiplier, double *source, double *dest)
{
    size_t j;

    for (j = 0; j < length; j++) {
        dest[j] += multiplier * source[j];
    }
}

static int
tsk_treeseq_node_general_stat(tsk_treeseq_t *self, size_t state_dim,
    double *sample_weights, size_t result_dim, general_stat_func_t *f, void *f_params,
    size_t num_windows, double *windows, double *result, tsk_flags_t TSK_UNUSED(options))
{
    int ret = 0;
    tsk_id_t u, v;
    size_t j, window_index;
    size_t num_nodes = self->tables->nodes.num_rows;
    const tsk_id_t num_edges = (tsk_id_t) self->tables->edges.num_rows;
    const tsk_id_t *restrict I = self->tables->indexes.edge_insertion_order;
    const tsk_id_t *restrict O = self->tables->indexes.edge_removal_order;
    const double *restrict edge_left = self->tables->edges.left;
    const double *restrict edge_right = self->tables->edges.right;
    const tsk_id_t *restrict edge_parent = self->tables->edges.parent;
    const tsk_id_t *restrict edge_child = self->tables->edges.child;
    const double sequence_length = self->tables->sequence_length;
    tsk_id_t *restrict parent = malloc(num_nodes * sizeof(*parent));
    tsk_id_t tj, tk, h;
    double *state_u, *weight_u;
    double *state = calloc(num_nodes * state_dim, sizeof(*state));
    double *node_summary = calloc(num_nodes * result_dim, sizeof(*node_summary));
    double *last_update = calloc(num_nodes, sizeof(*last_update));
    double t_left, t_right, w_right;

    if (parent == NULL || state == NULL || node_summary == NULL || last_update == NULL) {
        ret = TSK_ERR_NO_MEMORY;
        goto out;
    }
    memset(parent, 0xff, num_nodes * sizeof(*parent));
    memset(result, 0, num_windows * num_nodes * result_dim * sizeof(*result));

    /* Set the initial conditions */
    for (j = 0; j < self->num_samples; j++) {
        u = self->samples[j];
        state_u = GET_2D_ROW(state, state_dim, u);
        weight_u = GET_2D_ROW(sample_weights, state_dim, j);
        memcpy(state_u, weight_u, state_dim * sizeof(*state_u));
    }
    for (u = 0; u < (tsk_id_t) num_nodes; u++) {
        ret = update_node_summary(
            u, result_dim, node_summary, state, state_dim, f, f_params);
        if (ret != 0) {
            goto out;
        }
    }

    /* Iterate over the trees */
    tj = 0;
    tk = 0;
    t_left = 0;
    window_index = 0;
    while (tj < num_edges || t_left < sequence_length) {
        tsk_bug_assert(window_index < num_windows);
        while (tk < num_edges && edge_right[O[tk]] == t_left) {
            h = O[tk];
            tk++;
            u = edge_child[h];
            v = edge_parent[h];
            while (v != TSK_NULL) {
                increment_row(result_dim, t_left - last_update[v],
                    GET_2D_ROW(node_summary, result_dim, v),
                    GET_3D_ROW(result, num_nodes, result_dim, window_index, v));
                last_update[v] = t_left;
                update_state(state, state_dim, v, u, -1);
                ret = update_node_summary(
                    v, result_dim, node_summary, state, state_dim, f, f_params);
                if (ret != 0) {
                    goto out;
                }
                v = parent[v];
            }
            parent[u] = TSK_NULL;
        }

        while (tj < num_edges && edge_left[I[tj]] == t_left) {
            h = I[tj];
            tj++;
            u = edge_child[h];
            v = edge_parent[h];
            parent[u] = v;
            while (v != TSK_NULL) {
                increment_row(result_dim, t_left - last_update[v],
                    GET_2D_ROW(node_summary, result_dim, v),
                    GET_3D_ROW(result, num_nodes, result_dim, window_index, v));
                last_update[v] = t_left;
                update_state(state, state_dim, v, u, +1);
                ret = update_node_summary(
                    v, result_dim, node_summary, state, state_dim, f, f_params);
                if (ret != 0) {
                    goto out;
                }
                v = parent[v];
            }
        }

        t_right = sequence_length;
        if (tj < num_edges) {
            t_right = TSK_MIN(t_right, edge_left[I[tj]]);
        }
        if (tk < num_edges) {
            t_right = TSK_MIN(t_right, edge_right[O[tk]]);
        }

        while (window_index < num_windows && windows[window_index + 1] <= t_right) {
            w_right = windows[window_index + 1];
            /* Flush the contributions of all nodes to the current window */
            for (u = 0; u < (tsk_id_t) num_nodes; u++) {
                tsk_bug_assert(last_update[u] < w_right);
                increment_row(result_dim, w_right - last_update[u],
                    GET_2D_ROW(node_summary, result_dim, u),
                    GET_3D_ROW(result, num_nodes, result_dim, window_index, u));
                last_update[u] = w_right;
            }
            window_index++;
        }

        t_left = t_right;
    }
out:
    /* Can't use msp_safe_free here because of restrict */
    if (parent != NULL) {
        free(parent);
    }
    tsk_safe_free(state);
    tsk_safe_free(node_summary);
    tsk_safe_free(last_update);
    return ret;
}

static void
span_normalise(size_t num_windows, double *windows, size_t row_size, double *array)
{
    size_t window_index, k;
    double span, *row;

    for (window_index = 0; window_index < num_windows; window_index++) {
        span = windows[window_index + 1] - windows[window_index];
        row = GET_2D_ROW(array, row_size, window_index);
        for (k = 0; k < row_size; k++) {
            row[k] /= span;
        }
    }
}

typedef struct {
    general_stat_func_t *f;
    void *f_params;
    double *total_weight;
    double *total_minus_state;
    double *result_tmp;
} unpolarised_summary_func_args;

static int
unpolarised_summary_func(
    size_t state_dim, double *state, size_t result_dim, double *result, void *params)
{
    int ret = 0;
    unpolarised_summary_func_args *upargs = (unpolarised_summary_func_args *) params;
    const double *total_weight = upargs->total_weight;
    double *total_minus_state = upargs->total_minus_state;
    double *result_tmp = upargs->result_tmp;
    size_t k, m;

    ret = upargs->f(state_dim, state, result_dim, result, upargs->f_params);
    if (ret != 0) {
        goto out;
    }
    for (k = 0; k < state_dim; k++) {
        total_minus_state[k] = total_weight[k] - state[k];
    }
    ret = upargs->f(
        state_dim, total_minus_state, result_dim, result_tmp, upargs->f_params);
    if (ret != 0) {
        goto out;
    }
    for (m = 0; m < result_dim; m++) {
        result[m] += result_tmp[m];
    }
out:
    return ret;
}

/* Abstracts the running of node and branch stats where the summary function
 * is run twice when non-polarised. We replace the call to the input summary
 * function with a call of the required form when non-polarised, simplifying
 * the implementation and memory management for the node and branch stats.
 */
static int
tsk_polarisable_func_general_stat(tsk_treeseq_t *self, size_t state_dim,
    double *sample_weights, size_t result_dim, general_stat_func_t *f, void *f_params,
    size_t num_windows, double *windows, double *result, tsk_flags_t options)
{
    int ret = 0;
    bool stat_branch = !!(options & TSK_STAT_BRANCH);
    bool polarised = options & TSK_STAT_POLARISED;
    general_stat_func_t *wrapped_f = f;
    void *wrapped_f_params = f_params;
    double *weight_u;
    unpolarised_summary_func_args upargs;
    tsk_size_t j, k;

    memset(&upargs, 0, sizeof(upargs));
    if (!polarised) {
        upargs.f = f;
        upargs.f_params = f_params;
        upargs.total_weight = calloc(state_dim, sizeof(double));
        upargs.total_minus_state = calloc(state_dim, sizeof(double));
        upargs.result_tmp = calloc(result_dim, sizeof(double));

        if (upargs.total_weight == NULL || upargs.total_minus_state == NULL
            || upargs.result_tmp == NULL) {
            ret = TSK_ERR_NO_MEMORY;
            goto out;
        }

        /* Compute the total weight */
        for (j = 0; j < self->num_samples; j++) {
            weight_u = GET_2D_ROW(sample_weights, state_dim, j);
            for (k = 0; k < state_dim; k++) {
                upargs.total_weight[k] += weight_u[k];
            }
        }

        wrapped_f = unpolarised_summary_func;
        wrapped_f_params = &upargs;
    }

    if (stat_branch) {
        ret = tsk_treeseq_branch_general_stat(self, state_dim, sample_weights,
            result_dim, wrapped_f, wrapped_f_params, num_windows, windows, result,
            options);
    } else {
        ret = tsk_treeseq_node_general_stat(self, state_dim, sample_weights, result_dim,
            wrapped_f, wrapped_f_params, num_windows, windows, result, options);
    }
out:
    tsk_safe_free(upargs.total_weight);
    tsk_safe_free(upargs.total_minus_state);
    tsk_safe_free(upargs.result_tmp);
    return ret;
}

int
tsk_treeseq_general_stat(tsk_treeseq_t *self, size_t state_dim, double *sample_weights,
    size_t result_dim, general_stat_func_t *f, void *f_params, size_t num_windows,
    double *windows, double *result, tsk_flags_t options)
{
    int ret = 0;
    bool stat_site = !!(options & TSK_STAT_SITE);
    bool stat_branch = !!(options & TSK_STAT_BRANCH);
    bool stat_node = !!(options & TSK_STAT_NODE);
    double default_windows[] = { 0, self->tables->sequence_length };
    size_t row_size;

    /* If no mode is specified, we default to site mode */
    if (!(stat_site || stat_branch || stat_node)) {
        stat_site = true;
    }
    /* It's an error to specify more than one mode */
    if (stat_site + stat_branch + stat_node > 1) {
        ret = TSK_ERR_MULTIPLE_STAT_MODES;
        goto out;
    }

    if (state_dim < 1) {
        ret = TSK_ERR_BAD_STATE_DIMS;
        goto out;
    }
    if (result_dim < 1) {
        ret = TSK_ERR_BAD_RESULT_DIMS;
        goto out;
    }
    if (windows == NULL) {
        num_windows = 1;
        windows = default_windows;
    } else {
        ret = tsk_treeseq_check_windows(self, num_windows, windows);
        if (ret != 0) {
            goto out;
        }
    }

    if (stat_site) {
        ret = tsk_treeseq_site_general_stat(self, state_dim, sample_weights, result_dim,
            f, f_params, num_windows, windows, result, options);
    } else {
        ret = tsk_polarisable_func_general_stat(self, state_dim, sample_weights,
            result_dim, f, f_params, num_windows, windows, result, options);
    }

    if (options & TSK_STAT_SPAN_NORMALISE) {
        row_size = result_dim;
        if (stat_node) {
            row_size = result_dim * tsk_treeseq_get_num_nodes(self);
        }
        span_normalise(num_windows, windows, row_size, result);
    }

out:
    return ret;
}

static int
check_set_indexes(tsk_size_t num_sets, tsk_size_t num_set_indexes, tsk_id_t *set_indexes)
{
    int ret = 0;
    tsk_size_t j;

    for (j = 0; j < num_set_indexes; j++) {
        if (set_indexes[j] < 0 || set_indexes[j] >= (tsk_id_t) num_sets) {
            ret = TSK_ERR_BAD_SAMPLE_SET_INDEX;
            goto out;
        }
    }
out:
    return ret;
}

static int
tsk_treeseq_check_sample_sets(tsk_treeseq_t *self, tsk_size_t num_sample_sets,
    tsk_size_t *sample_set_sizes, tsk_id_t *sample_sets)
{
    int ret = 0;
    size_t j, k, l;
    const tsk_id_t num_nodes = (tsk_id_t) self->tables->nodes.num_rows;
    tsk_id_t u, sample_index;

    if (num_sample_sets == 0) {
        ret = TSK_ERR_INSUFFICIENT_SAMPLE_SETS;
        goto out;
    }
    j = 0;
    for (k = 0; k < num_sample_sets; k++) {
        if (sample_set_sizes[k] == 0) {
            ret = TSK_ERR_EMPTY_SAMPLE_SET;
            goto out;
        }
        for (l = 0; l < sample_set_sizes[k]; l++) {
            u = sample_sets[j];
            if (u < 0 || u >= num_nodes) {
                ret = TSK_ERR_NODE_OUT_OF_BOUNDS;
                goto out;
            }
            sample_index = self->sample_index_map[u];
            if (sample_index == TSK_NULL) {
                ret = TSK_ERR_BAD_SAMPLES;
                goto out;
            }
            j++;
        }
    }
out:
    return ret;
}

typedef struct {
    tsk_size_t num_samples;
} weight_stat_params_t;

typedef struct {
    tsk_size_t num_samples;
    tsk_size_t num_covariates;
    double *V;
} covariates_stat_params_t;

typedef struct {
    tsk_id_t *sample_sets;
    tsk_size_t num_sample_sets;
    tsk_size_t *sample_set_sizes;
    tsk_id_t *set_indexes;
} sample_count_stat_params_t;

static int
tsk_treeseq_sample_count_stat(tsk_treeseq_t *self, tsk_size_t num_sample_sets,
    tsk_size_t *sample_set_sizes, tsk_id_t *sample_sets, tsk_size_t result_dim,
    tsk_id_t *set_indexes, general_stat_func_t *f, tsk_size_t num_windows,
    double *windows, double *result, tsk_flags_t options)
{
    int ret = 0;
    const size_t num_samples = self->num_samples;
    size_t j, k, l;
    tsk_id_t u, sample_index;
    double *weights = NULL;
    double *weight_row;
    sample_count_stat_params_t args = { .sample_sets = sample_sets,
        .num_sample_sets = num_sample_sets,
        .sample_set_sizes = sample_set_sizes,
        .set_indexes = set_indexes };

    ret = tsk_treeseq_check_sample_sets(
        self, num_sample_sets, sample_set_sizes, sample_sets);
    if (ret != 0) {
        goto out;
    }
    weights = calloc(num_samples * num_sample_sets, sizeof(*weights));
    if (weights == NULL) {
        ret = TSK_ERR_NO_MEMORY;
        goto out;
    }
    j = 0;
    for (k = 0; k < num_sample_sets; k++) {
        for (l = 0; l < sample_set_sizes[k]; l++) {
            u = sample_sets[j];
            sample_index = self->sample_index_map[u];
            weight_row = GET_2D_ROW(weights, num_sample_sets, sample_index);
            if (weight_row[k] != 0) {
                ret = TSK_ERR_DUPLICATE_SAMPLE;
                goto out;
            }
            weight_row[k] = 1;
            j++;
        }
    }
    ret = tsk_treeseq_general_stat(self, num_sample_sets, weights, result_dim, f, &args,
        num_windows, windows, result, options);
out:
    tsk_safe_free(weights);
    return ret;
}

/***********************************
 * Allele frequency spectrum
 ***********************************/

static inline void
fold(tsk_size_t *restrict coordinate, const tsk_size_t *restrict dims,
    tsk_size_t num_dims)
{
    tsk_size_t k;
    double n = 0;
    int s = 0;

    for (k = 0; k < num_dims; k++) {
        tsk_bug_assert(coordinate[k] < dims[k]);
        n += dims[k] - 1;
        s += (int) coordinate[k];
    }
    n /= 2;
    k = num_dims;
    while (s == n && k > 0) {
        k--;
        n -= ((double) (dims[k] - 1)) / 2;
        s -= (int) coordinate[k];
    }
    if (s > n) {
        for (k = 0; k < num_dims; k++) {
            s = (int) (dims[k] - 1 - coordinate[k]);
            tsk_bug_assert(s >= 0);
            coordinate[k] = (tsk_size_t) s;
        }
    }
}

static int
tsk_treeseq_update_site_afs(tsk_treeseq_t *self, tsk_site_t *site, double *total_counts,
    double *counts, tsk_size_t num_sample_sets, tsk_size_t window_index,
    tsk_size_t *result_dims, double *result, tsk_flags_t options)
{
    int ret = 0;
    tsk_size_t afs_size;
    tsk_size_t k, allele, num_alleles, all_samples;
    double increment, *afs, *allele_counts, *allele_count;
    tsk_size_t *coordinate = malloc(num_sample_sets * sizeof(*coordinate));
    bool polarised = !!(options & TSK_STAT_POLARISED);
    const tsk_size_t K = num_sample_sets + 1;

    if (coordinate == NULL) {
        ret = TSK_ERR_NO_MEMORY;
        goto out;
    }
    ret = get_allele_weights(
        site, counts, K, total_counts, &num_alleles, &allele_counts);
    if (ret != 0) {
        goto out;
    }

    afs_size = result_dims[num_sample_sets];
    afs = result + afs_size * window_index;

    increment = polarised ? 1 : 0.5;
    /* Sum over the allele weights. Skip the ancestral state if polarised. */
    for (allele = polarised ? 1 : 0; allele < num_alleles; allele++) {
        allele_count = GET_2D_ROW(allele_counts, K, allele);
        all_samples = (tsk_size_t) allele_count[num_sample_sets];
        if (all_samples > 0 && all_samples < self->num_samples) {
            for (k = 0; k < num_sample_sets; k++) {
                coordinate[k] = (tsk_size_t) allele_count[k];
            }
            if (!polarised) {
                fold(coordinate, result_dims, num_sample_sets);
            }
            increment_nd_array_value(
                afs, num_sample_sets, result_dims, coordinate, increment);
        }
    }
out:
    tsk_safe_free(coordinate);
    tsk_safe_free(allele_counts);
    return ret;
}

static int
tsk_treeseq_site_allele_frequency_spectrum(tsk_treeseq_t *self,
    tsk_size_t num_sample_sets, const tsk_size_t *sample_set_sizes, double *counts,
    tsk_size_t num_windows, double *windows, tsk_size_t *result_dims, double *result,
    tsk_flags_t options)
{
    int ret = 0;
    tsk_id_t u, v;
    tsk_size_t tree_site, tree_index, window_index;
    size_t num_nodes = self->tables->nodes.num_rows;
    const tsk_id_t num_edges = (tsk_id_t) self->tables->edges.num_rows;
    const tsk_id_t *restrict I = self->tables->indexes.edge_insertion_order;
    const tsk_id_t *restrict O = self->tables->indexes.edge_removal_order;
    const double *restrict edge_left = self->tables->edges.left;
    const double *restrict edge_right = self->tables->edges.right;
    const tsk_id_t *restrict edge_parent = self->tables->edges.parent;
    const tsk_id_t *restrict edge_child = self->tables->edges.child;
    const double sequence_length = self->tables->sequence_length;
    tsk_id_t *restrict parent = malloc(num_nodes * sizeof(*parent));
    tsk_site_t *site;
    tsk_id_t tj, tk, h;
    tsk_size_t j;
    const tsk_size_t K = num_sample_sets + 1;
    double t_left, t_right;
    double *total_counts = malloc((1 + num_sample_sets) * sizeof(*total_counts));

    if (parent == NULL || total_counts == NULL) {
        ret = TSK_ERR_NO_MEMORY;
        goto out;
    }
    memset(parent, 0xff, num_nodes * sizeof(*parent));

    for (j = 0; j < num_sample_sets; j++) {
        total_counts[j] = sample_set_sizes[j];
    }
    total_counts[num_sample_sets] = self->num_samples;

    /* Iterate over the trees */
    tj = 0;
    tk = 0;
    t_left = 0;
    tree_index = 0;
    window_index = 0;
    while (tj < num_edges || t_left < sequence_length) {
        while (tk < num_edges && edge_right[O[tk]] == t_left) {
            h = O[tk];
            tk++;
            u = edge_child[h];
            v = edge_parent[h];
            while (v != TSK_NULL) {
                update_state(counts, K, v, u, -1);
                v = parent[v];
            }
            parent[u] = TSK_NULL;
        }

        while (tj < num_edges && edge_left[I[tj]] == t_left) {
            h = I[tj];
            tj++;
            u = edge_child[h];
            v = edge_parent[h];
            parent[u] = v;
            while (v != TSK_NULL) {
                update_state(counts, K, v, u, +1);
                v = parent[v];
            }
        }
        t_right = sequence_length;
        if (tj < num_edges) {
            t_right = TSK_MIN(t_right, edge_left[I[tj]]);
        }
        if (tk < num_edges) {
            t_right = TSK_MIN(t_right, edge_right[O[tk]]);
        }

        /* Update the sites */
        for (tree_site = 0; tree_site < self->tree_sites_length[tree_index];
             tree_site++) {
            site = self->tree_sites[tree_index] + tree_site;
            while (windows[window_index + 1] <= site->position) {
                window_index++;
                tsk_bug_assert(window_index < num_windows);
            }
            ret = tsk_treeseq_update_site_afs(self, site, total_counts, counts,
                num_sample_sets, window_index, result_dims, result, options);
            if (ret != 0) {
                goto out;
            }
            tsk_bug_assert(windows[window_index] <= site->position);
            tsk_bug_assert(site->position < windows[window_index + 1]);
        }
        tree_index++;
        t_left = t_right;
    }
out:
    /* Can't use msp_safe_free here because of restrict */
    if (parent != NULL) {
        free(parent);
    }
    tsk_safe_free(total_counts);
    return ret;
}

static int TSK_WARN_UNUSED
tsk_treeseq_update_branch_afs(tsk_treeseq_t *self, tsk_id_t u, double right,
    const double *restrict branch_length, double *restrict last_update,
    const double *counts, tsk_size_t num_sample_sets, size_t window_index,
    const tsk_size_t *result_dims, double *result, tsk_flags_t options)
{
    int ret = 0;
    tsk_size_t afs_size;
    tsk_size_t k;
    double *afs;
    tsk_size_t *coordinate = malloc(num_sample_sets * sizeof(*coordinate));
    bool polarised = !!(options & TSK_STAT_POLARISED);
    const double *count_row = GET_2D_ROW(counts, num_sample_sets + 1, u);
    double x = (right - last_update[u]) * branch_length[u];
    const tsk_size_t all_samples = (tsk_size_t) count_row[num_sample_sets];

    if (coordinate == NULL) {
        ret = TSK_ERR_NO_MEMORY;
        goto out;
    }

    if (0 < all_samples && all_samples < self->num_samples) {
        if (!polarised) {
            x *= 0.5;
        }
        afs_size = result_dims[num_sample_sets];
        afs = result + afs_size * window_index;
        for (k = 0; k < num_sample_sets; k++) {
            coordinate[k] = (tsk_size_t) count_row[k];
        }
        if (!polarised) {
            fold(coordinate, result_dims, num_sample_sets);
        }
        increment_nd_array_value(afs, num_sample_sets, result_dims, coordinate, x);
    }
    last_update[u] = right;
out:
    tsk_safe_free(coordinate);
    return ret;
}

static int
tsk_treeseq_branch_allele_frequency_spectrum(tsk_treeseq_t *self,
    tsk_size_t num_sample_sets, double *counts, tsk_size_t num_windows, double *windows,
    tsk_size_t *result_dims, double *result, tsk_flags_t options)
{
    int ret = 0;
    tsk_id_t u, v;
    size_t window_index;
    size_t num_nodes = self->tables->nodes.num_rows;
    const tsk_id_t num_edges = (tsk_id_t) self->tables->edges.num_rows;
    const tsk_id_t *restrict I = self->tables->indexes.edge_insertion_order;
    const tsk_id_t *restrict O = self->tables->indexes.edge_removal_order;
    const double *restrict edge_left = self->tables->edges.left;
    const double *restrict edge_right = self->tables->edges.right;
    const tsk_id_t *restrict edge_parent = self->tables->edges.parent;
    const tsk_id_t *restrict edge_child = self->tables->edges.child;
    const double *restrict node_time = self->tables->nodes.time;
    const double sequence_length = self->tables->sequence_length;
    tsk_id_t *restrict parent = malloc(num_nodes * sizeof(*parent));
    double *restrict last_update = calloc(num_nodes, sizeof(*last_update));
    double *restrict branch_length = calloc(num_nodes, sizeof(*branch_length));
    tsk_id_t tj, tk, h;
    double t_left, t_right, w_right;
    const tsk_size_t K = num_sample_sets + 1;

    if (parent == NULL || last_update == NULL) {
        ret = TSK_ERR_NO_MEMORY;
        goto out;
    }
    memset(parent, 0xff, num_nodes * sizeof(*parent));

    /* Iterate over the trees */
    tj = 0;
    tk = 0;
    t_left = 0;
    window_index = 0;
    while (tj < num_edges || t_left < sequence_length) {
        tsk_bug_assert(window_index < num_windows);
        while (tk < num_edges && edge_right[O[tk]] == t_left) {
            h = O[tk];
            tk++;
            u = edge_child[h];
            v = edge_parent[h];
            ret = tsk_treeseq_update_branch_afs(self, u, t_left, branch_length,
                last_update, counts, num_sample_sets, window_index, result_dims, result,
                options);
            if (ret != 0) {
                goto out;
            }
            while (v != TSK_NULL) {
                ret = tsk_treeseq_update_branch_afs(self, v, t_left, branch_length,
                    last_update, counts, num_sample_sets, window_index, result_dims,
                    result, options);
                if (ret != 0) {
                    goto out;
                }
                update_state(counts, K, v, u, -1);
                v = parent[v];
            }
            parent[u] = TSK_NULL;
            branch_length[u] = 0;
        }

        while (tj < num_edges && edge_left[I[tj]] == t_left) {
            h = I[tj];
            tj++;
            u = edge_child[h];
            v = edge_parent[h];
            parent[u] = v;
            branch_length[u] = node_time[v] - node_time[u];
            while (v != TSK_NULL) {
                ret = tsk_treeseq_update_branch_afs(self, v, t_left, branch_length,
                    last_update, counts, num_sample_sets, window_index, result_dims,
                    result, options);
                if (ret != 0) {
                    goto out;
                }
                update_state(counts, K, v, u, +1);
                v = parent[v];
            }
        }

        t_right = sequence_length;
        if (tj < num_edges) {
            t_right = TSK_MIN(t_right, edge_left[I[tj]]);
        }
        if (tk < num_edges) {
            t_right = TSK_MIN(t_right, edge_right[O[tk]]);
        }

        while (window_index < num_windows && windows[window_index + 1] <= t_right) {
            w_right = windows[window_index + 1];
            /* Flush the contributions of all nodes to the current window */
            for (u = 0; u < (tsk_id_t) num_nodes; u++) {
                tsk_bug_assert(last_update[u] < w_right);
                ret = tsk_treeseq_update_branch_afs(self, u, w_right, branch_length,
                    last_update, counts, num_sample_sets, window_index, result_dims,
                    result, options);
                if (ret != 0) {
                    goto out;
                }
            }
            window_index++;
        }

        t_left = t_right;
    }
out:
    /* Can't use msp_safe_free here because of restrict */
    if (parent != NULL) {
        free(parent);
    }
    if (last_update != NULL) {
        free(last_update);
    }
    if (branch_length != NULL) {
        free(branch_length);
    }
    return ret;
}

int
tsk_treeseq_allele_frequency_spectrum(tsk_treeseq_t *self, tsk_size_t num_sample_sets,
    tsk_size_t *sample_set_sizes, tsk_id_t *sample_sets, tsk_size_t num_windows,
    double *windows, double *result, tsk_flags_t options)
{
    int ret = 0;
    bool stat_site = !!(options & TSK_STAT_SITE);
    bool stat_branch = !!(options & TSK_STAT_BRANCH);
    bool stat_node = !!(options & TSK_STAT_NODE);
    double default_windows[] = { 0, self->tables->sequence_length };
    const size_t num_nodes = self->tables->nodes.num_rows;
    const size_t K = num_sample_sets + 1;
    size_t j, k, l, afs_size;
    tsk_id_t u;
    tsk_size_t *result_dims = NULL;
    /* These counts should really be ints, but we use doubles so that we can
     * reuse code from the general_stats code paths. */
    double *counts = NULL;
    double *count_row;

    if (stat_node) {
        ret = TSK_ERR_UNSUPPORTED_STAT_MODE;
        goto out;
    }
    /* If no mode is specified, we default to site mode */
    if (!(stat_site || stat_branch)) {
        stat_site = true;
    }
    /* It's an error to specify more than one mode */
    if (stat_site + stat_branch > 1) {
        ret = TSK_ERR_MULTIPLE_STAT_MODES;
        goto out;
    }
    if (windows == NULL) {
        num_windows = 1;
        windows = default_windows;
    } else {
        ret = tsk_treeseq_check_windows(self, num_windows, windows);
        if (ret != 0) {
            goto out;
        }
    }
    ret = tsk_treeseq_check_sample_sets(
        self, num_sample_sets, sample_set_sizes, sample_sets);
    if (ret != 0) {
        goto out;
    }

    /* the last element of result_dims stores the total size of the dimenensions */
    result_dims = malloc((num_sample_sets + 1) * sizeof(*result_dims));
    counts = calloc(num_nodes * K, sizeof(*counts));
    if (counts == NULL || result_dims == NULL) {
        ret = TSK_ERR_NO_MEMORY;
        goto out;
    }
    afs_size = 1;
    j = 0;
    for (k = 0; k < num_sample_sets; k++) {
        result_dims[k] = 1 + sample_set_sizes[k];
        afs_size *= result_dims[k];
        for (l = 0; l < sample_set_sizes[k]; l++) {
            u = sample_sets[j];
            count_row = GET_2D_ROW(counts, K, u);
            if (count_row[k] != 0) {
                ret = TSK_ERR_DUPLICATE_SAMPLE;
                goto out;
            }
            count_row[k] = 1;
            j++;
        }
    }
    for (j = 0; j < self->num_samples; j++) {
        u = self->samples[j];
        count_row = GET_2D_ROW(counts, K, u);
        count_row[num_sample_sets] = 1;
    }
    result_dims[num_sample_sets] = (tsk_size_t) afs_size;

    memset(result, 0, num_windows * afs_size * sizeof(*result));
    if (stat_site) {
        ret = tsk_treeseq_site_allele_frequency_spectrum(self, num_sample_sets,
            sample_set_sizes, counts, num_windows, windows, result_dims, result,
            options);
    } else {
        ret = tsk_treeseq_branch_allele_frequency_spectrum(self, num_sample_sets, counts,
            num_windows, windows, result_dims, result, options);
    }

    if (options & TSK_STAT_SPAN_NORMALISE) {
        span_normalise(num_windows, windows, afs_size, result);
    }
out:
    tsk_safe_free(counts);
    tsk_safe_free(result_dims);
    return ret;
}

/***********************************
 * One way stats
 ***********************************/

static int
diversity_summary_func(size_t state_dim, double *state, size_t TSK_UNUSED(result_dim),
    double *result, void *params)
{
    sample_count_stat_params_t args = *(sample_count_stat_params_t *) params;
    const double *x = state;
    double n;
    size_t j;

    for (j = 0; j < state_dim; j++) {
        n = (double) args.sample_set_sizes[j];
        result[j] = x[j] * (n - x[j]) / (n * (n - 1));
    }
    return 0;
}

int
tsk_treeseq_diversity(tsk_treeseq_t *self, tsk_size_t num_sample_sets,
    tsk_size_t *sample_set_sizes, tsk_id_t *sample_sets, tsk_size_t num_windows,
    double *windows, double *result, tsk_flags_t options)
{
    return tsk_treeseq_sample_count_stat(self, num_sample_sets, sample_set_sizes,
        sample_sets, num_sample_sets, NULL, diversity_summary_func, num_windows, windows,
        result, options);
}

static int
trait_covariance_summary_func(size_t state_dim, double *state,
    size_t TSK_UNUSED(result_dim), double *result, void *params)
{
    weight_stat_params_t args = *(weight_stat_params_t *) params;
    const double n = (double) args.num_samples;
    const double *x = state;
    size_t j;

    for (j = 0; j < state_dim; j++) {
        result[j] = (x[j] * x[j]) / (2 * (n - 1) * (n - 1));
    }
    return 0;
}

int
tsk_treeseq_trait_covariance(tsk_treeseq_t *self, tsk_size_t num_weights,
    double *weights, tsk_size_t num_windows, double *windows, double *result,
    tsk_flags_t options)
{
    tsk_size_t num_samples = self->num_samples;
    size_t j, k;
    int ret;
    double *row, *new_row;
    double *means = calloc(num_weights, sizeof(double));
    double *new_weights = malloc((num_weights + 1) * num_samples * sizeof(double));
    weight_stat_params_t args = { num_samples = self->num_samples };

    if (new_weights == NULL || means == NULL) {
        ret = TSK_ERR_NO_MEMORY;
        goto out;
    }

    // center weights
    for (j = 0; j < num_samples; j++) {
        row = GET_2D_ROW(weights, num_weights, j);
        for (k = 0; k < num_weights; k++) {
            means[k] += row[k];
        }
    }
    for (k = 0; k < num_weights; k++) {
        means[k] /= (double) num_samples;
    }
    for (j = 0; j < num_samples; j++) {
        row = GET_2D_ROW(weights, num_weights, j);
        new_row = GET_2D_ROW(new_weights, num_weights, j);
        for (k = 0; k < num_weights; k++) {
            new_row[k] = row[k] - means[k];
        }
    }

    ret = tsk_treeseq_general_stat(self, num_weights, new_weights, num_weights,
        trait_covariance_summary_func, &args, num_windows, windows, result, options);

out:
    tsk_safe_free(means);
    tsk_safe_free(new_weights);
    return ret;
}

static int
trait_correlation_summary_func(size_t state_dim, double *state,
    size_t TSK_UNUSED(result_dim), double *result, void *params)
{
    weight_stat_params_t args = *(weight_stat_params_t *) params;
    const double n = (double) args.num_samples;
    const double *x = state;
    double p;
    size_t j;

    p = x[state_dim - 1];
    for (j = 0; j < state_dim - 1; j++) {
        if ((p > 0.0) && (p < 1.0)) {
            result[j] = (x[j] * x[j]) / (2 * (p * (1 - p)) * n * (n - 1));
        } else {
            result[j] = 0.0;
        }
    }
    return 0;
}

int
tsk_treeseq_trait_correlation(tsk_treeseq_t *self, tsk_size_t num_weights,
    double *weights, tsk_size_t num_windows, double *windows, double *result,
    tsk_flags_t options)
{
    tsk_size_t num_samples = self->num_samples;
    size_t j, k;
    int ret;
    double *means = calloc(num_weights, sizeof(double));
    double *meansqs = calloc(num_weights, sizeof(double));
    double *sds = calloc(num_weights, sizeof(double));
    double *row, *new_row;
    double *new_weights = malloc((num_weights + 1) * num_samples * sizeof(double));
    weight_stat_params_t args = { num_samples = self->num_samples };

    if (new_weights == NULL || means == NULL || meansqs == NULL || sds == NULL) {
        ret = TSK_ERR_NO_MEMORY;
        goto out;
    }

    if (num_weights < 1) {
        ret = TSK_ERR_BAD_STATE_DIMS;
        goto out;
    }

    // center and scale weights
    for (j = 0; j < num_samples; j++) {
        row = GET_2D_ROW(weights, num_weights, j);
        for (k = 0; k < num_weights; k++) {
            means[k] += row[k];
            meansqs[k] += row[k] * row[k];
        }
    }
    for (k = 0; k < num_weights; k++) {
        means[k] /= (double) num_samples;
        meansqs[k] -= means[k] * means[k] * (double) num_samples;
        meansqs[k] /= (double) (num_samples - 1);
        sds[k] = sqrt(meansqs[k]);
    }
    for (j = 0; j < num_samples; j++) {
        row = GET_2D_ROW(weights, num_weights, j);
        new_row = GET_2D_ROW(new_weights, num_weights + 1, j);
        for (k = 0; k < num_weights; k++) {
            new_row[k] = (row[k] - means[k]) / sds[k];
        }
        // set final row to 1/n to compute frequency
        new_row[num_weights] = 1.0 / (double) num_samples;
    }

    ret = tsk_treeseq_general_stat(self, num_weights + 1, new_weights, num_weights,
        trait_correlation_summary_func, &args, num_windows, windows, result, options);

out:
    tsk_safe_free(means);
    tsk_safe_free(meansqs);
    tsk_safe_free(sds);
    tsk_safe_free(new_weights);
    return ret;
}

static int
trait_regression_summary_func(
    size_t state_dim, double *state, size_t result_dim, double *result, void *params)
{
    covariates_stat_params_t args = *(covariates_stat_params_t *) params;
    const double num_samples = (double) args.num_samples;
    const tsk_size_t k = args.num_covariates;
    const double *V = args.V;
    ;
    const double *x = state;
    const double *v;
    double m, a, denom, z;
    size_t i, j;
    // x[0], ..., x[result_dim - 1] contains the traits, W
    // x[result_dim], ..., x[state_dim - 2] contains the covariates, Z
    // x[state_dim - 1] has the number of samples below the node

    m = x[state_dim - 1];
    for (i = 0; i < result_dim; i++) {
        if ((m > 0.0) && (m < num_samples)) {
            v = GET_2D_ROW(V, k, i);
            a = x[i];
            denom = m;
            for (j = 0; j < k; j++) {
                z = x[result_dim + j];
                a -= z * v[j];
                denom -= z * z;
            }
            // denom is the length of projection of the trait onto the subspace
            // spanned by the covariates, so if it is zero then the system is
            // singular and the solution is nonunique. This numerical tolerance
            // could be smaller without hitting floating-point error, but being
            // a tiny bit conservative about when the trait is almost in the
            // span of the covariates is probably good.
            if (denom < 1e-8) {
                result[i] = 0.0;
            } else {
                result[i] = (a * a) / (2 * denom * denom);
            }
        } else {
            result[i] = 0.0;
        }
    }
    return 0;
}

int
tsk_treeseq_trait_regression(tsk_treeseq_t *self, tsk_size_t num_weights,
    double *weights, tsk_size_t num_covariates, double *covariates,
    tsk_size_t num_windows, double *windows, double *result, tsk_flags_t options)
{
    tsk_size_t num_samples = self->num_samples;
    size_t i, j, k;
    int ret;
    double *v, *w, *z, *new_row;
    double *V = calloc(num_covariates * num_weights, sizeof(double));
    double *new_weights
        = malloc((num_weights + num_covariates + 1) * num_samples * sizeof(double));

    covariates_stat_params_t args
        = { .num_samples = self->num_samples, .num_covariates = num_covariates, .V = V };

    // We assume that the covariates have been *already standardised*,
    // so that (a) 1 is in the span of the columns, and
    // (b) their crossproduct is the identity.
    // We could do this instead here with gsl linalg.

    if (new_weights == NULL || V == NULL) {
        ret = TSK_ERR_NO_MEMORY;
        goto out;
    }

    if (num_weights < 1) {
        ret = TSK_ERR_BAD_STATE_DIMS;
        goto out;
    }

    // V = weights^T (matrix mult) covariates
    for (k = 0; k < num_samples; k++) {
        w = GET_2D_ROW(weights, num_weights, k);
        z = GET_2D_ROW(covariates, num_covariates, k);
        for (i = 0; i < num_weights; i++) {
            v = GET_2D_ROW(V, num_covariates, i);
            for (j = 0; j < num_covariates; j++) {
                v[j] += w[i] * z[j];
            }
        }
    }

    for (k = 0; k < num_samples; k++) {
        w = GET_2D_ROW(weights, num_weights, k);
        z = GET_2D_ROW(covariates, num_covariates, k);
        new_row = GET_2D_ROW(new_weights, num_covariates + num_weights + 1, k);
        for (i = 0; i < num_weights; i++) {
            new_row[i] = w[i];
        }
        for (i = 0; i < num_covariates; i++) {
            new_row[i + num_weights] = z[i];
        }
        // set final row to 1 to count alleles
        new_row[num_weights + num_covariates] = 1.0;
    }

    ret = tsk_treeseq_general_stat(self, num_weights + num_covariates + 1, new_weights,
        num_weights, trait_regression_summary_func, &args, num_windows, windows, result,
        options);

out:
    tsk_safe_free(V);
    tsk_safe_free(new_weights);
    return ret;
}

static int
segregating_sites_summary_func(size_t state_dim, double *state,
    size_t TSK_UNUSED(result_dim), double *result, void *params)
{
    sample_count_stat_params_t args = *(sample_count_stat_params_t *) params;
    const double *x = state;
    double n;
    size_t j;

    // this works because sum_{i=1}^k (1-p_i) = k-1
    for (j = 0; j < state_dim; j++) {
        n = (double) args.sample_set_sizes[j];
        result[j] = (x[j] > 0) * (1 - x[j] / n);
    }
    return 0;
}

int
tsk_treeseq_segregating_sites(tsk_treeseq_t *self, tsk_size_t num_sample_sets,
    tsk_size_t *sample_set_sizes, tsk_id_t *sample_sets, tsk_size_t num_windows,
    double *windows, double *result, tsk_flags_t options)
{
    return tsk_treeseq_sample_count_stat(self, num_sample_sets, sample_set_sizes,
        sample_sets, num_sample_sets, NULL, segregating_sites_summary_func, num_windows,
        windows, result, options);
}

static int
Y1_summary_func(size_t TSK_UNUSED(state_dim), double *state, size_t result_dim,
    double *result, void *params)
{
    sample_count_stat_params_t args = *(sample_count_stat_params_t *) params;
    const double *x = state;
    double ni, denom, numer;
    size_t i;

    for (i = 0; i < result_dim; i++) {
        ni = args.sample_set_sizes[i];
        denom = ni * (ni - 1) * (ni - 2);
        numer = x[i] * (ni - x[i]) * (ni - x[i] - 1);
        result[i] = numer / denom;
    }
    return 0;
}

int
tsk_treeseq_Y1(tsk_treeseq_t *self, tsk_size_t num_sample_sets,
    tsk_size_t *sample_set_sizes, tsk_id_t *sample_sets, tsk_size_t num_windows,
    double *windows, double *result, tsk_flags_t options)
{
    return tsk_treeseq_sample_count_stat(self, num_sample_sets, sample_set_sizes,
        sample_sets, num_sample_sets, NULL, Y1_summary_func, num_windows, windows,
        result, options);
}

/***********************************
 * Two way stats
 ***********************************/

static int
check_sample_stat_inputs(tsk_size_t num_sample_sets, tsk_size_t tuple_size,
    tsk_size_t num_index_tuples, tsk_id_t *index_tuples)
{
    int ret = 0;

    if (num_sample_sets < tuple_size) {
        ret = TSK_ERR_INSUFFICIENT_SAMPLE_SETS;
        goto out;
    }
    if (num_index_tuples < 1) {
        ret = TSK_ERR_INSUFFICIENT_INDEX_TUPLES;
        goto out;
    }
    ret = check_set_indexes(
        num_sample_sets, tuple_size * num_index_tuples, index_tuples);
    if (ret != 0) {
        goto out;
    }
out:
    return ret;
}

static int
divergence_summary_func(size_t TSK_UNUSED(state_dim), double *state, size_t result_dim,
    double *result, void *params)
{
    sample_count_stat_params_t args = *(sample_count_stat_params_t *) params;
    const double *x = state;
    double ni, nj, denom;
    tsk_id_t i, j;
    size_t k;

    for (k = 0; k < result_dim; k++) {
        i = args.set_indexes[2 * k];
        j = args.set_indexes[2 * k + 1];
        ni = args.sample_set_sizes[i];
        nj = args.sample_set_sizes[j];
        denom = ni * (nj - (i == j));
        result[k] = x[i] * (nj - x[j]) / denom;
    }
    return 0;
}

int
tsk_treeseq_divergence(tsk_treeseq_t *self, tsk_size_t num_sample_sets,
    tsk_size_t *sample_set_sizes, tsk_id_t *sample_sets, tsk_size_t num_index_tuples,
    tsk_id_t *index_tuples, tsk_size_t num_windows, double *windows, double *result,
    tsk_flags_t options)
{
    int ret = 0;
    ret = check_sample_stat_inputs(num_sample_sets, 2, num_index_tuples, index_tuples);
    if (ret != 0) {
        goto out;
    }
    ret = tsk_treeseq_sample_count_stat(self, num_sample_sets, sample_set_sizes,
        sample_sets, num_index_tuples, index_tuples, divergence_summary_func,
        num_windows, windows, result, options);
out:
    return ret;
}

static int
Y2_summary_func(size_t TSK_UNUSED(state_dim), double *state, size_t result_dim,
    double *result, void *params)
{
    sample_count_stat_params_t args = *(sample_count_stat_params_t *) params;
    const double *x = state;
    double ni, nj, denom;
    tsk_id_t i, j;
    size_t k;

    for (k = 0; k < result_dim; k++) {
        i = args.set_indexes[2 * k];
        j = args.set_indexes[2 * k + 1];
        ni = args.sample_set_sizes[i];
        nj = args.sample_set_sizes[j];
        denom = ni * nj * (nj - 1);
        result[k] = x[i] * (nj - x[j]) * (nj - x[j] - 1) / denom;
    }
    return 0;
}

int
tsk_treeseq_Y2(tsk_treeseq_t *self, tsk_size_t num_sample_sets,
    tsk_size_t *sample_set_sizes, tsk_id_t *sample_sets, tsk_size_t num_index_tuples,
    tsk_id_t *index_tuples, tsk_size_t num_windows, double *windows, double *result,
    tsk_flags_t options)
{
    int ret = 0;
    ret = check_sample_stat_inputs(num_sample_sets, 2, num_index_tuples, index_tuples);
    if (ret != 0) {
        goto out;
    }
    ret = tsk_treeseq_sample_count_stat(self, num_sample_sets, sample_set_sizes,
        sample_sets, num_index_tuples, index_tuples, Y2_summary_func, num_windows,
        windows, result, options);
out:
    return ret;
}

static int
f2_summary_func(size_t TSK_UNUSED(state_dim), double *state, size_t result_dim,
    double *result, void *params)
{
    sample_count_stat_params_t args = *(sample_count_stat_params_t *) params;
    const double *x = state;
    double ni, nj, denom, numer;
    tsk_id_t i, j;
    size_t k;

    for (k = 0; k < result_dim; k++) {
        i = args.set_indexes[2 * k];
        j = args.set_indexes[2 * k + 1];
        ni = args.sample_set_sizes[i];
        nj = args.sample_set_sizes[j];
        denom = ni * (ni - 1) * nj * (nj - 1);
        numer = x[i] * (x[i] - 1) * (nj - x[j]) * (nj - x[j] - 1)
                - x[i] * (ni - x[i]) * (nj - x[j]) * x[j];
        result[k] = numer / denom;
    }
    return 0;
}

int
tsk_treeseq_f2(tsk_treeseq_t *self, tsk_size_t num_sample_sets,
    tsk_size_t *sample_set_sizes, tsk_id_t *sample_sets, tsk_size_t num_index_tuples,
    tsk_id_t *index_tuples, tsk_size_t num_windows, double *windows, double *result,
    tsk_flags_t options)
{
    int ret = 0;
    ret = check_sample_stat_inputs(num_sample_sets, 2, num_index_tuples, index_tuples);
    if (ret != 0) {
        goto out;
    }
    ret = tsk_treeseq_sample_count_stat(self, num_sample_sets, sample_set_sizes,
        sample_sets, num_index_tuples, index_tuples, f2_summary_func, num_windows,
        windows, result, options);
out:
    return ret;
}

/***********************************
 * Three way stats
 ***********************************/

static int
Y3_summary_func(size_t TSK_UNUSED(state_dim), double *state, size_t result_dim,
    double *result, void *params)
{
    sample_count_stat_params_t args = *(sample_count_stat_params_t *) params;
    const double *x = state;
    double ni, nj, nk, denom, numer;
    tsk_id_t i, j, k;
    size_t tuple_index;

    for (tuple_index = 0; tuple_index < result_dim; tuple_index++) {
        i = args.set_indexes[3 * tuple_index];
        j = args.set_indexes[3 * tuple_index + 1];
        k = args.set_indexes[3 * tuple_index + 2];
        ni = args.sample_set_sizes[i];
        nj = args.sample_set_sizes[j];
        nk = args.sample_set_sizes[k];
        denom = ni * nj * nk;
        numer = x[i] * (nj - x[j]) * (nk - x[k]);
        result[tuple_index] = numer / denom;
    }
    return 0;
}

int
tsk_treeseq_Y3(tsk_treeseq_t *self, tsk_size_t num_sample_sets,
    tsk_size_t *sample_set_sizes, tsk_id_t *sample_sets, tsk_size_t num_index_tuples,
    tsk_id_t *index_tuples, tsk_size_t num_windows, double *windows, double *result,
    tsk_flags_t options)
{
    int ret = 0;
    ret = check_sample_stat_inputs(num_sample_sets, 3, num_index_tuples, index_tuples);
    if (ret != 0) {
        goto out;
    }
    ret = tsk_treeseq_sample_count_stat(self, num_sample_sets, sample_set_sizes,
        sample_sets, num_index_tuples, index_tuples, Y3_summary_func, num_windows,
        windows, result, options);
out:
    return ret;
}

static int
f3_summary_func(size_t TSK_UNUSED(state_dim), double *state, size_t result_dim,
    double *result, void *params)
{
    sample_count_stat_params_t args = *(sample_count_stat_params_t *) params;
    const double *x = state;
    double ni, nj, nk, denom, numer;
    tsk_id_t i, j, k;
    size_t tuple_index;

    for (tuple_index = 0; tuple_index < result_dim; tuple_index++) {
        i = args.set_indexes[3 * tuple_index];
        j = args.set_indexes[3 * tuple_index + 1];
        k = args.set_indexes[3 * tuple_index + 2];
        ni = args.sample_set_sizes[i];
        nj = args.sample_set_sizes[j];
        nk = args.sample_set_sizes[k];
        denom = ni * (ni - 1) * nj * nk;
        numer = x[i] * (x[i] - 1) * (nj - x[j]) * (nk - x[k])
                - x[i] * (ni - x[i]) * (nj - x[j]) * x[k];
        result[tuple_index] = numer / denom;
    }
    return 0;
}

int
tsk_treeseq_f3(tsk_treeseq_t *self, tsk_size_t num_sample_sets,
    tsk_size_t *sample_set_sizes, tsk_id_t *sample_sets, tsk_size_t num_index_tuples,
    tsk_id_t *index_tuples, tsk_size_t num_windows, double *windows, double *result,
    tsk_flags_t options)
{
    int ret = 0;
    ret = check_sample_stat_inputs(num_sample_sets, 3, num_index_tuples, index_tuples);
    if (ret != 0) {
        goto out;
    }
    ret = tsk_treeseq_sample_count_stat(self, num_sample_sets, sample_set_sizes,
        sample_sets, num_index_tuples, index_tuples, f3_summary_func, num_windows,
        windows, result, options);
out:
    return ret;
}

/***********************************
 * Four way stats
 ***********************************/

static int
f4_summary_func(size_t TSK_UNUSED(state_dim), double *state, size_t result_dim,
    double *result, void *params)
{
    sample_count_stat_params_t args = *(sample_count_stat_params_t *) params;
    const double *x = state;
    double ni, nj, nk, nl, denom, numer;
    tsk_id_t i, j, k, l;
    size_t tuple_index;

    for (tuple_index = 0; tuple_index < result_dim; tuple_index++) {
        i = args.set_indexes[4 * tuple_index];
        j = args.set_indexes[4 * tuple_index + 1];
        k = args.set_indexes[4 * tuple_index + 2];
        l = args.set_indexes[4 * tuple_index + 3];
        ni = args.sample_set_sizes[i];
        nj = args.sample_set_sizes[j];
        nk = args.sample_set_sizes[k];
        nl = args.sample_set_sizes[l];
        denom = ni * nj * nk * nl;
        numer = x[i] * x[k] * (nj - x[j]) * (nl - x[l])
                - x[i] * x[l] * (nj - x[j]) * (nk - x[k]);
        result[tuple_index] = numer / denom;
    }
    return 0;
}

int
tsk_treeseq_f4(tsk_treeseq_t *self, tsk_size_t num_sample_sets,
    tsk_size_t *sample_set_sizes, tsk_id_t *sample_sets, tsk_size_t num_index_tuples,
    tsk_id_t *index_tuples, tsk_size_t num_windows, double *windows, double *result,
    tsk_flags_t options)
{
    int ret = 0;
    ret = check_sample_stat_inputs(num_sample_sets, 4, num_index_tuples, index_tuples);
    if (ret != 0) {
        goto out;
    }
    ret = tsk_treeseq_sample_count_stat(self, num_sample_sets, sample_set_sizes,
        sample_sets, num_index_tuples, index_tuples, f4_summary_func, num_windows,
        windows, result, options);
out:
    return ret;
}

/****************************************************************************/
/* LD calculator */
/****************************************************************************/

static void
tsk_ld_calc_check_state(tsk_ld_calc_t *self)
{
    uint32_t u;
    uint32_t num_nodes = (uint32_t) tsk_treeseq_get_num_nodes(self->tree_sequence);
    tsk_tree_t *tA = self->outer_tree;
    tsk_tree_t *tB = self->inner_tree;

    tsk_bug_assert(tA->index == tB->index);

    /* The inner tree's mark values should all be zero. */
    for (u = 0; u < num_nodes; u++) {
        tsk_bug_assert(tA->marked[u] == 0);
        tsk_bug_assert(tB->marked[u] == 0);
    }
}

void
tsk_ld_calc_print_state(tsk_ld_calc_t *self, FILE *out)
{
    fprintf(out, "tree_sequence = %p\n", (void *) self->tree_sequence);
    fprintf(out, "outer tree index = %d\n", (int) self->outer_tree->index);
    fprintf(out, "outer tree interval = (%f, %f)\n", self->outer_tree->left,
        self->outer_tree->right);
    fprintf(out, "inner tree index = %d\n", (int) self->inner_tree->index);
    fprintf(out, "inner tree interval = (%f, %f)\n", self->inner_tree->left,
        self->inner_tree->right);
    tsk_ld_calc_check_state(self);
}

int TSK_WARN_UNUSED
tsk_ld_calc_init(tsk_ld_calc_t *self, tsk_treeseq_t *tree_sequence)
{
    int ret = TSK_ERR_GENERIC;

    memset(self, 0, sizeof(tsk_ld_calc_t));
    self->tree_sequence = tree_sequence;
    self->num_sites = tsk_treeseq_get_num_sites(tree_sequence);
    self->outer_tree = malloc(sizeof(tsk_tree_t));
    self->inner_tree = malloc(sizeof(tsk_tree_t));
    if (self->outer_tree == NULL || self->inner_tree == NULL) {
        ret = TSK_ERR_NO_MEMORY;
        goto out;
    }
    ret = tsk_tree_init(self->outer_tree, self->tree_sequence, TSK_SAMPLE_LISTS);
    if (ret != 0) {
        goto out;
    }
    ret = tsk_tree_init(self->inner_tree, self->tree_sequence, 0);
    if (ret != 0) {
        goto out;
    }
    ret = tsk_tree_first(self->outer_tree);
    if (ret < 0) {
        goto out;
    }
    ret = tsk_tree_first(self->inner_tree);
    if (ret < 0) {
        goto out;
    }
    ret = 0;
out:
    return ret;
}

int
tsk_ld_calc_free(tsk_ld_calc_t *self)
{
    if (self->inner_tree != NULL) {
        tsk_tree_free(self->inner_tree);
        free(self->inner_tree);
    }
    if (self->outer_tree != NULL) {
        tsk_tree_free(self->outer_tree);
        free(self->outer_tree);
    }
    return 0;
}

/* Position the two trees so that the specified site is within their
 * interval.
 */
static int TSK_WARN_UNUSED
tsk_ld_calc_position_trees(tsk_ld_calc_t *self, tsk_id_t site_index)
{
    int ret = TSK_ERR_GENERIC;
    tsk_site_t mut;
    double x;
    tsk_tree_t *tA = self->outer_tree;
    tsk_tree_t *tB = self->inner_tree;

    ret = tsk_treeseq_get_site(self->tree_sequence, site_index, &mut);
    if (ret != 0) {
        goto out;
    }
    x = mut.position;
    tsk_bug_assert(tA->index == tB->index);
    while (x >= tA->right) {
        ret = tsk_tree_next(tA);
        if (ret < 0) {
            goto out;
        }
        tsk_bug_assert(ret == 1);
        ret = tsk_tree_next(tB);
        if (ret < 0) {
            goto out;
        }
        tsk_bug_assert(ret == 1);
    }
    while (x < tA->left) {
        ret = tsk_tree_prev(tA);
        if (ret < 0) {
            goto out;
        }
        tsk_bug_assert(ret == 1);
        ret = tsk_tree_prev(tB);
        if (ret < 0) {
            goto out;
        }
        tsk_bug_assert(ret == 1);
    }
    ret = 0;
    tsk_bug_assert(x >= tA->left && x < tB->right);
    tsk_bug_assert(tA->index == tB->index);
out:
    return ret;
}

static double
tsk_ld_calc_overlap_within_tree(tsk_ld_calc_t *self, tsk_site_t sA, tsk_site_t sB)
{
    const tsk_tree_t *t = self->inner_tree;
    const tsk_node_table_t *nodes = &self->tree_sequence->tables->nodes;
    tsk_id_t u, v, nAB;

    tsk_bug_assert(sA.mutations_length == 1);
    tsk_bug_assert(sB.mutations_length == 1);
    u = sA.mutations[0].node;
    v = sB.mutations[0].node;
    if (nodes->time[u] > nodes->time[v]) {
        v = sA.mutations[0].node;
        u = sB.mutations[0].node;
    }
    while (u != v && u != TSK_NULL) {
        u = t->parent[u];
    }
    nAB = 0;
    if (u == v) {
        nAB = TSK_MIN(
            t->num_samples[sA.mutations[0].node], t->num_samples[sB.mutations[0].node]);
    }
    return (double) nAB;
}

static inline int TSK_WARN_UNUSED
tsk_ld_calc_set_tracked_samples(tsk_ld_calc_t *self, tsk_site_t sA)
{
    int ret = 0;

    tsk_bug_assert(sA.mutations_length == 1);
    ret = tsk_tree_set_tracked_samples_from_sample_list(
        self->inner_tree, self->outer_tree, sA.mutations[0].node);
    return ret;
}

static int TSK_WARN_UNUSED
tsk_ld_calc_get_r2_array_forward(tsk_ld_calc_t *self, tsk_id_t source_index,
    tsk_size_t max_sites, double max_distance, double *r2, tsk_size_t *num_r2_values)
{
    int ret = TSK_ERR_GENERIC;
    tsk_site_t sA, sB;
    double fA, fB, fAB, D;
    int tracked_samples_set = 0;
    tsk_tree_t *tA, *tB;
    double n = (double) tsk_treeseq_get_num_samples(self->tree_sequence);
    tsk_id_t j;
    double nAB;

    tA = self->outer_tree;
    tB = self->inner_tree;
    ret = tsk_treeseq_get_site(self->tree_sequence, source_index, &sA);
    if (ret != 0) {
        goto out;
    }
    if (sA.mutations_length > 1) {
        ret = TSK_ERR_ONLY_INFINITE_SITES;
        goto out;
    }
    fA = ((double) tA->num_samples[sA.mutations[0].node]) / n;
    tsk_bug_assert(fA > 0);
    tB->mark = 1;
    for (j = 0; j < (tsk_id_t) max_sites; j++) {
        if (source_index + j + 1 >= (tsk_id_t) self->num_sites) {
            break;
        }
        ret = tsk_treeseq_get_site(self->tree_sequence, (source_index + j + 1), &sB);
        if (ret != 0) {
            goto out;
        }
        if (sB.mutations_length > 1) {
            ret = TSK_ERR_ONLY_INFINITE_SITES;
            goto out;
        }
        if (sB.position - sA.position > max_distance) {
            break;
        }
        while (sB.position >= tB->right) {
            ret = tsk_tree_next(tB);
            if (ret < 0) {
                goto out;
            }
            tsk_bug_assert(ret == 1);
        }
        fB = ((double) tB->num_samples[sB.mutations[0].node]) / n;
        tsk_bug_assert(fB > 0);
        if (sB.position < tA->right) {
            nAB = tsk_ld_calc_overlap_within_tree(self, sA, sB);
        } else {
            if (!tracked_samples_set && tB->marked[sA.mutations[0].node] == 1) {
                tracked_samples_set = 1;
                ret = tsk_ld_calc_set_tracked_samples(self, sA);
                if (ret != 0) {
                    goto out;
                }
            }
            if (tracked_samples_set) {
                nAB = (double) tB->num_tracked_samples[sB.mutations[0].node];
            } else {
                nAB = tsk_ld_calc_overlap_within_tree(self, sA, sB);
            }
        }
        fAB = nAB / n;
        D = fAB - fA * fB;
        r2[j] = D * D / (fA * fB * (1 - fA) * (1 - fB));
    }

    /* Now rewind back the inner iterator and unmark all nodes that
     * were set to 1 as we moved forward. */
    tB->mark = 0;
    while (tB->index > tA->index) {
        ret = tsk_tree_prev(tB);
        if (ret < 0) {
            goto out;
        }
        tsk_bug_assert(ret == 1);
    }
    *num_r2_values = (tsk_size_t) j;
    ret = 0;
out:
    return ret;
}

static int TSK_WARN_UNUSED
tsk_ld_calc_get_r2_array_reverse(tsk_ld_calc_t *self, tsk_id_t source_index,
    tsk_size_t max_sites, double max_distance, double *r2, tsk_size_t *num_r2_values)
{
    int ret = TSK_ERR_GENERIC;
    tsk_site_t sA, sB;
    double fA, fB, fAB, D;
    int tracked_samples_set = 0;
    tsk_tree_t *tA, *tB;
    double n = (double) tsk_treeseq_get_num_samples(self->tree_sequence);
    double nAB;
    tsk_id_t j, site_index;

    tA = self->outer_tree;
    tB = self->inner_tree;
    ret = tsk_treeseq_get_site(self->tree_sequence, source_index, &sA);
    if (ret != 0) {
        goto out;
    }
    if (sA.mutations_length > 1) {
        ret = TSK_ERR_ONLY_INFINITE_SITES;
        goto out;
    }
    fA = ((double) tA->num_samples[sA.mutations[0].node]) / n;
    tsk_bug_assert(fA > 0);
    tB->mark = 1;
    for (j = 0; j < (tsk_id_t) max_sites; j++) {
        site_index = source_index - j - 1;
        if (site_index < 0) {
            break;
        }
        ret = tsk_treeseq_get_site(self->tree_sequence, site_index, &sB);
        if (ret != 0) {
            goto out;
        }
        if (sB.mutations_length > 1) {
            ret = TSK_ERR_ONLY_INFINITE_SITES;
            goto out;
        }
        if (sA.position - sB.position > max_distance) {
            break;
        }
        while (sB.position < tB->left) {
            ret = tsk_tree_prev(tB);
            if (ret < 0) {
                goto out;
            }
            tsk_bug_assert(ret == 1);
        }
        fB = ((double) tB->num_samples[sB.mutations[0].node]) / n;
        tsk_bug_assert(fB > 0);
        if (sB.position >= tA->left) {
            nAB = tsk_ld_calc_overlap_within_tree(self, sA, sB);
        } else {
            if (!tracked_samples_set && tB->marked[sA.mutations[0].node] == 1) {
                tracked_samples_set = 1;
                ret = tsk_ld_calc_set_tracked_samples(self, sA);
                if (ret != 0) {
                    goto out;
                }
            }
            if (tracked_samples_set) {
                nAB = (double) tB->num_tracked_samples[sB.mutations[0].node];
            } else {
                nAB = tsk_ld_calc_overlap_within_tree(self, sA, sB);
            }
        }
        fAB = nAB / n;
        D = fAB - fA * fB;
        r2[j] = D * D / (fA * fB * (1 - fA) * (1 - fB));
    }

    /* Now fast forward the inner iterator and unmark all nodes that
     * were set to 1 as we moved back. */
    tB->mark = 0;
    while (tB->index < tA->index) {
        ret = tsk_tree_next(tB);
        if (ret < 0) {
            goto out;
        }
        tsk_bug_assert(ret == 1);
    }
    *num_r2_values = (tsk_size_t) j;
    ret = 0;
out:
    return ret;
}

int TSK_WARN_UNUSED
tsk_ld_calc_get_r2_array(tsk_ld_calc_t *self, tsk_id_t a, int direction,
    tsk_size_t max_sites, double max_distance, double *r2, tsk_size_t *num_r2_values)
{
    int ret = TSK_ERR_GENERIC;

    if (a < 0 || a >= (tsk_id_t) self->num_sites) {
        ret = TSK_ERR_OUT_OF_BOUNDS;
        goto out;
    }
    ret = tsk_ld_calc_position_trees(self, a);
    if (ret != 0) {
        goto out;
    }
    if (direction == TSK_DIR_FORWARD) {
        ret = tsk_ld_calc_get_r2_array_forward(
            self, a, max_sites, max_distance, r2, num_r2_values);
    } else if (direction == TSK_DIR_REVERSE) {
        ret = tsk_ld_calc_get_r2_array_reverse(
            self, a, max_sites, max_distance, r2, num_r2_values);
    } else {
        ret = TSK_ERR_BAD_PARAM_VALUE;
    }
out:
    return ret;
}

int TSK_WARN_UNUSED
tsk_ld_calc_get_r2(tsk_ld_calc_t *self, tsk_id_t a, tsk_id_t b, double *r2)
{
    int ret = TSK_ERR_GENERIC;
    tsk_site_t sA, sB;
    double fA, fB, fAB, D;
    tsk_tree_t *tA, *tB;
    double n = (double) tsk_treeseq_get_num_samples(self->tree_sequence);
    double nAB;
    tsk_id_t tmp;

    if (a < 0 || b < 0 || a >= (tsk_id_t) self->num_sites
        || b >= (tsk_id_t) self->num_sites) {
        ret = TSK_ERR_OUT_OF_BOUNDS;
        goto out;
    }
    if (a > b) {
        tmp = a;
        a = b;
        b = tmp;
    }
    ret = tsk_ld_calc_position_trees(self, a);
    if (ret != 0) {
        goto out;
    }
    /* We can probably do a lot better than this implementation... */
    tA = self->outer_tree;
    tB = self->inner_tree;
    ret = tsk_treeseq_get_site(self->tree_sequence, a, &sA);
    if (ret != 0) {
        goto out;
    }
    ret = tsk_treeseq_get_site(self->tree_sequence, b, &sB);
    if (ret != 0) {
        goto out;
    }
    if (sA.mutations_length > 1 || sB.mutations_length > 1) {
        ret = TSK_ERR_ONLY_INFINITE_SITES;
        goto out;
    }
    tsk_bug_assert(sA.mutations_length == 1);
    /* tsk_bug_assert(tA->parent[sA.mutations[0].node] != TSK_NULL); */
    fA = ((double) tA->num_samples[sA.mutations[0].node]) / n;
    tsk_bug_assert(fA > 0);
    ret = tsk_ld_calc_set_tracked_samples(self, sA);
    if (ret != 0) {
        goto out;
    }

    while (sB.position >= tB->right) {
        ret = tsk_tree_next(tB);
        if (ret < 0) {
            goto out;
        }
        tsk_bug_assert(ret == 1);
    }
    /* tsk_bug_assert(tB->parent[sB.mutations[0].node] != TSK_NULL); */
    fB = ((double) tB->num_samples[sB.mutations[0].node]) / n;
    tsk_bug_assert(fB > 0);
    nAB = (double) tB->num_tracked_samples[sB.mutations[0].node];
    fAB = nAB / n;
    D = fAB - fA * fB;
    *r2 = D * D / (fA * fB * (1 - fA) * (1 - fB));

    /* Now rewind the inner iterator back. */
    while (tB->index > tA->index) {
        ret = tsk_tree_prev(tB);
        if (ret < 0) {
            goto out;
        }
        tsk_bug_assert(ret == 1);
    }
    ret = 0;
out:
    return ret;
}
