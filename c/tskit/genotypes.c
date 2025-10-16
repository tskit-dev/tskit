/*
 * MIT License
 *
 * Copyright (c) 2019-2025 Tskit Developers
 * Copyright (c) 2016-2018 University of Oxford
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

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#include <tskit/genotypes.h>

static inline uint32_t
tsk_haplotype_ctz64(uint64_t x)
{
#if defined(_MSC_VER)
    unsigned long index;
    _BitScanForward64(&index, x);
    return (uint32_t) index;
#else
    return (uint32_t) __builtin_ctzll(x);
#endif
}

static inline void
tsk_haplotype_bitset_clear(uint64_t *bits, tsk_size_t idx)
{
    tsk_size_t word = idx >> 6;
    uint64_t mask = UINT64_C(1) << (idx & 63);
    bits[word] &= ~mask;
}

static inline tsk_size_t
tsk_haplotype_bitset_next(
    const uint64_t *bits, tsk_size_t num_words, tsk_size_t start, tsk_size_t limit)
{
    tsk_size_t word = start >> 6;
    uint64_t mask, value;

    if (start >= limit || word >= num_words) {
        return limit;
    }
    mask = UINT64_MAX << (start & 63);
    value = bits[word] & mask;
    while (value == 0) {
        word++;
        if (word >= num_words) {
            return limit;
        }
        value = bits[word];
    }
    start = (word << 6) + tsk_haplotype_ctz64(value);
    return start < limit ? start : limit;
}

static void
tsk_haplotype_reset_bitset(const tsk_haplotype_t *self)
{
    if (self->num_bit_words > 0) {
        tsk_memcpy(self->unresolved_bits, self->initial_bits,
            self->num_bit_words * sizeof(*self->unresolved_bits));
    }
}

static int
tsk_haplotype_build_parent_index(tsk_haplotype_t *self)
{
    int ret = 0;
    const tsk_table_collection_t *tables = self->tree_sequence->tables;
    const tsk_edge_table_t *edges = &tables->edges;
    const tsk_id_t *edges_child = edges->child;
    tsk_size_t num_edges = edges->num_rows;
    int32_t *child_counts = NULL;

    if (num_edges == 0) {
        self->parent_edge_index = NULL;
        if (self->num_nodes > 0) {
            self->parent_index_range
                = tsk_calloc(self->num_nodes * 2, sizeof(*self->parent_index_range));
            if (self->parent_index_range == NULL) {
                ret = tsk_trace_error(TSK_ERR_NO_MEMORY);
                goto out;
            }
        } else {
            self->parent_index_range = NULL;
        }
        goto out;
    }

    self->parent_edge_index = tsk_malloc(num_edges * sizeof(*self->parent_edge_index));
    self->parent_index_range
        = tsk_malloc(self->num_nodes * 2 * sizeof(*self->parent_index_range));
    child_counts = tsk_calloc(self->num_nodes, sizeof(*child_counts));
    if (self->parent_edge_index == NULL
        || (self->num_nodes > 0 && self->parent_index_range == NULL)
        || child_counts == NULL) {
        ret = tsk_trace_error(TSK_ERR_NO_MEMORY);
        goto out;
    }

    for (tsk_size_t j = 0; j < num_edges; j++) {
        tsk_id_t child = edges_child[j];
        if (child >= 0 && child < (tsk_id_t) self->num_nodes) {
            if (child_counts[child] == INT32_MAX) {
                ret = tsk_trace_error(TSK_ERR_UNSUPPORTED_OPERATION);
                goto out;
            }
            child_counts[child]++;
        }
    }

    int32_t current_start = 0;
    for (tsk_size_t u = 0; u < (tsk_size_t) self->num_nodes; u++) {
        int32_t offset = (int32_t)(u * 2);
        self->parent_index_range[offset] = current_start;
        self->parent_index_range[offset + 1] = current_start;
        current_start += child_counts[u];
    }

    for (tsk_size_t j = 0; j < num_edges; j++) {
        tsk_id_t child = edges_child[j];
        if (child >= 0 && child < (tsk_id_t) self->num_nodes) {
            int32_t end_offset = (int32_t)(child * 2 + 1);
            int32_t pos = self->parent_index_range[end_offset];
            self->parent_edge_index[pos] = (tsk_id_t) j;
            self->parent_index_range[end_offset] = pos + 1;
        }
    }

    for (tsk_size_t u = 0; u < (tsk_size_t) self->num_nodes; u++) {
        int32_t offset = (int32_t)(u * 2);
        int32_t end = self->parent_index_range[offset + 1];
        self->parent_index_range[offset] = end - child_counts[u];
    }

out:
    if (ret != 0) {
        tsk_safe_free(self->parent_edge_index);
        self->parent_edge_index = NULL;
        tsk_safe_free(self->parent_index_range);
        self->parent_index_range = NULL;
    }
    tsk_safe_free(child_counts);
    return ret;
}

static int
tsk_haplotype_build_mutation_index(tsk_haplotype_t *self)
{
    int ret = 0;
    tsk_size_t j;
    const tsk_table_collection_t *tables = self->tree_sequence->tables;
    const tsk_mutation_table_t *mutations = &tables->mutations;
    int32_t *counts = NULL;
    tsk_size_t total_mutations = 0;
    tsk_id_t site_start = self->site_start;
    tsk_id_t site_stop = self->site_stop;

    counts = tsk_calloc(self->num_nodes, sizeof(*counts));
    if (self->num_nodes > 0 && counts == NULL) {
        ret = tsk_trace_error(TSK_ERR_NO_MEMORY);
        goto out;
    }

    for (j = 0; j < mutations->num_rows; j++) {
        tsk_id_t node = mutations->node[j];
        tsk_id_t site = mutations->site[j];
        if (site < site_start || site >= site_stop) {
            continue;
        }
        if (node >= 0 && node < (tsk_id_t) self->num_nodes) {
            if (counts[node] == INT32_MAX) {
                ret = tsk_trace_error(TSK_ERR_UNSUPPORTED_OPERATION);
                goto out;
            }
            counts[node]++;
        }
    }

    self->node_mutation_offsets
        = tsk_malloc((self->num_nodes + 1) * sizeof(*self->node_mutation_offsets));
    if (self->node_mutation_offsets == NULL) {
        ret = tsk_trace_error(TSK_ERR_NO_MEMORY);
        goto out;
    }
    self->node_mutation_offsets[0] = 0;
    for (j = 0; j < self->num_nodes; j++) {
        total_mutations += (tsk_size_t) counts[j];
        if (total_mutations > INT32_MAX) {
            ret = tsk_trace_error(TSK_ERR_UNSUPPORTED_OPERATION);
            goto out;
        }
        self->node_mutation_offsets[j + 1] = (int32_t) total_mutations;
    }

    self->node_mutation_sites
        = tsk_malloc(total_mutations * sizeof(*self->node_mutation_sites));
    self->node_mutation_states
        = tsk_malloc(total_mutations * sizeof(*self->node_mutation_states));
    if ((total_mutations > 0)
        && (self->node_mutation_sites == NULL || self->node_mutation_states == NULL)) {
        ret = tsk_trace_error(TSK_ERR_NO_MEMORY);
        goto out;
    }

    for (j = 0; j < self->num_nodes; j++) {
        counts[j] = self->node_mutation_offsets[j];
    }
    for (j = mutations->num_rows; j > 0; j--) {
        tsk_size_t mut_index = j - 1;
        tsk_id_t node = mutations->node[mut_index];
        tsk_id_t site = mutations->site[mut_index];
        if (site < site_start || site >= site_stop) {
            continue;
        }
        if (node >= 0 && node < (tsk_id_t) self->num_nodes) {
            tsk_size_t start = mutations->derived_state_offset[mut_index];
            tsk_size_t stop = mutations->derived_state_offset[mut_index + 1];
            tsk_size_t length = stop - start;
            uint8_t allele;

            if (length != 1) {
                ret = tsk_trace_error(TSK_ERR_UNSUPPORTED_OPERATION);
                goto out;
            }
            allele = (uint8_t) mutations->derived_state[start];
            if (allele > 0x7F) {
                ret = tsk_trace_error(TSK_ERR_UNSUPPORTED_OPERATION);
                goto out;
            }
            self->node_mutation_sites[counts[node]] = (int32_t)(site - site_start);
            self->node_mutation_states[counts[node]] = allele;
            counts[node]++;
        }
    }

out:
    tsk_safe_free(counts);
    return ret;
}

static int
tsk_haplotype_build_ancestral_states(tsk_haplotype_t *self)
{
    int ret = 0;
    const tsk_table_collection_t *tables = self->tree_sequence->tables;
    const tsk_site_table_t *sites = &tables->sites;
    tsk_id_t site_start = self->site_start;
    tsk_size_t j;

    if (self->num_sites == 0) {
        self->ancestral_states = NULL;
        return 0;
    }

    self->ancestral_states
        = tsk_malloc((tsk_size_t) self->num_sites * sizeof(*self->ancestral_states));
    if (self->ancestral_states == NULL) {
        return tsk_trace_error(TSK_ERR_NO_MEMORY);
    }

    for (j = 0; j < (tsk_size_t) self->num_sites; j++) {
        tsk_id_t site = site_start + (tsk_id_t) j;
        tsk_size_t start = sites->ancestral_state_offset[site];
        tsk_size_t stop = sites->ancestral_state_offset[site + 1];
        tsk_size_t length = stop - start;
        uint8_t allele;
        if (length != 1) {
            ret = tsk_trace_error(TSK_ERR_UNSUPPORTED_OPERATION);
            goto out;
        }
        allele = (uint8_t) sites->ancestral_state[start];
        if (allele > 0x7F) {
            ret = tsk_trace_error(TSK_ERR_UNSUPPORTED_OPERATION);
            goto out;
        }
        self->ancestral_states[j] = allele;
    }

out:
    if (ret != 0) {
        tsk_safe_free(self->ancestral_states);
        self->ancestral_states = NULL;
    }
    return ret;
}

static int
tsk_haplotype_build_edge_intervals(tsk_haplotype_t *self)
{
    int ret = 0;
    const tsk_table_collection_t *tables = self->tree_sequence->tables;
    const tsk_edge_table_t *edges = &tables->edges;
    const double *positions = tables->sites.position + self->site_start;
    tsk_size_t num_edges = edges->num_rows;
    tsk_size_t j;

    if (num_edges == 0) {
        self->edge_start_index = NULL;
        self->edge_end_index = NULL;
        return 0;
    }

    self->edge_start_index = tsk_malloc(num_edges * sizeof(*self->edge_start_index));
    self->edge_end_index = tsk_malloc(num_edges * sizeof(*self->edge_end_index));
    if (self->edge_start_index == NULL || self->edge_end_index == NULL) {
        ret = tsk_trace_error(TSK_ERR_NO_MEMORY);
        goto out;
    }

    if (self->num_sites == 0) {
        for (j = 0; j < num_edges; j++) {
            self->edge_start_index[j] = 0;
            self->edge_end_index[j] = 0;
        }
        goto out;
    }

    for (j = 0; j < num_edges; j++) {
        double left = edges->left[j];
        double right = edges->right[j];
        tsk_size_t start
            = tsk_search_sorted(positions, (tsk_size_t) self->num_sites, left);
        tsk_size_t end
            = tsk_search_sorted(positions, (tsk_size_t) self->num_sites, right);
        if (start > (tsk_size_t) self->num_sites) {
            start = (tsk_size_t) self->num_sites;
        }
        if (end > (tsk_size_t) self->num_sites) {
            end = (tsk_size_t) self->num_sites;
        }
        self->edge_start_index[j] = (int32_t) start;
        self->edge_end_index[j] = (int32_t) end;
    }

out:
    if (ret != 0) {
        tsk_safe_free(self->edge_start_index);
        tsk_safe_free(self->edge_end_index);
        self->edge_start_index = NULL;
        self->edge_end_index = NULL;
    }
    return ret;
}

static int
tsk_haplotype_alloc_bitset(tsk_haplotype_t *self)
{
    tsk_size_t j;

    self->num_bit_words = ((tsk_size_t) self->num_sites + 63) >> 6;
    if (self->num_bit_words == 0) {
        self->unresolved_bits = NULL;
        self->initial_bits = NULL;
        return 0;
    }
    self->unresolved_bits
        = tsk_malloc(self->num_bit_words * sizeof(*self->unresolved_bits));
    self->initial_bits = tsk_malloc(self->num_bit_words * sizeof(*self->initial_bits));
    if (self->unresolved_bits == NULL || self->initial_bits == NULL) {
        return tsk_trace_error(TSK_ERR_NO_MEMORY);
    }
    for (j = 0; j < self->num_bit_words; j++) {
        self->initial_bits[j] = UINT64_MAX;
    }
    if ((tsk_size_t) self->num_sites % 64 != 0) {
        uint32_t bits = (uint32_t)((tsk_size_t) self->num_sites & 63);
        self->initial_bits[self->num_bit_words - 1] = (UINT64_C(1) << bits) - 1;
    }
    return 0;
}

int
tsk_haplotype_init(tsk_haplotype_t *self, const tsk_treeseq_t *tree_sequence,
    tsk_id_t site_start, tsk_id_t site_stop)
{
    int ret = 0;
    const tsk_table_collection_t *tables;
    const tsk_site_table_t *sites;
    tsk_size_t total_sites;

    if (tree_sequence == NULL) {
        return tsk_trace_error(TSK_ERR_BAD_PARAM_VALUE);
    }

    tsk_memset(self, 0, sizeof(*self));
    self->tree_sequence = tree_sequence;

    tables = tree_sequence->tables;
    sites = &tables->sites;
    total_sites = sites->num_rows;

    if (site_start < 0 || site_stop < site_start || site_stop > (tsk_id_t) total_sites) {
        ret = tsk_trace_error(TSK_ERR_BAD_PARAM_VALUE);
        goto out;
    }

    self->site_start = (int32_t) site_start;
    self->site_stop = (int32_t) site_stop;
    self->num_sites = (int32_t)(site_stop - site_start);
    self->num_nodes = tables->nodes.num_rows;
    self->num_edges = tables->edges.num_rows;
    self->site_positions = sites->position + site_start;

    ret = tsk_haplotype_build_parent_index(self);
    if (ret != 0) {
        goto out;
    }
    ret = tsk_haplotype_build_mutation_index(self);
    if (ret != 0) {
        goto out;
    }
    ret = tsk_haplotype_build_ancestral_states(self);
    if (ret != 0) {
        goto out;
    }
    ret = tsk_haplotype_build_edge_intervals(self);
    if (ret != 0) {
        goto out;
    }
    ret = tsk_haplotype_alloc_bitset(self);
    if (ret != 0) {
        goto out;
    }
    if (self->num_edges > 0) {
        self->edge_stack = tsk_malloc(self->num_edges * sizeof(*self->edge_stack));
        self->stack_interval_start
            = tsk_malloc(self->num_edges * sizeof(*self->stack_interval_start));
        self->stack_interval_end
            = tsk_malloc(self->num_edges * sizeof(*self->stack_interval_end));
        self->parent_interval_start
            = tsk_malloc(self->num_edges * sizeof(*self->parent_interval_start));
        self->parent_interval_end
            = tsk_malloc(self->num_edges * sizeof(*self->parent_interval_end));
        if (self->edge_stack == NULL || self->stack_interval_start == NULL
            || self->stack_interval_end == NULL || self->parent_interval_start == NULL
            || self->parent_interval_end == NULL) {
            ret = tsk_trace_error(TSK_ERR_NO_MEMORY);
            goto out;
        }
    }

    self->initialised = true;

out:
    if (ret != 0) {
        tsk_haplotype_free(self);
    }
    return ret;
}

int
tsk_haplotype_decode(tsk_haplotype_t *self, tsk_id_t node, int8_t *haplotype)
{
    tsk_size_t stack_top = 0;
    const tsk_table_collection_t *tables;
    const tsk_edge_table_t *edges;
    const tsk_id_t *edge_parent;
    int32_t interval_start, interval_end;
    int32_t mut_start, mut_end;
    tsk_size_t idx;
    tsk_size_t parent_count;
    uint64_t *bits;

    if (self == NULL || haplotype == NULL) {
        return tsk_trace_error(TSK_ERR_BAD_PARAM_VALUE);
    }
    if (!self->initialised) {
        return tsk_trace_error(TSK_ERR_BAD_PARAM_VALUE);
    }
    if (node < 0 || node >= (tsk_id_t) self->num_nodes) {
        return tsk_trace_error(TSK_ERR_NODE_OUT_OF_BOUNDS);
    }
    if (self->num_sites == 0) {
        return 0;
    }

    tables = self->tree_sequence->tables;
    edges = &tables->edges;
    edge_parent = edges->parent;
    bits = self->unresolved_bits;

    for (idx = 0; idx < (tsk_size_t) self->num_sites; idx++) {
        haplotype[idx] = (int8_t) self->ancestral_states[idx];
    }
    tsk_haplotype_reset_bitset(self);

    mut_start = self->node_mutation_offsets[node];
    mut_end = self->node_mutation_offsets[node + 1];
    for (int32_t m = mut_start; m < mut_end; m++) {
        int32_t site = self->node_mutation_sites[m];
        if (site >= 0 && site < self->num_sites
            && tsk_haplotype_bitset_next(
                   bits, self->num_bit_words, (tsk_size_t) site, (tsk_size_t) site + 1)
                   == (tsk_size_t) site) {
            haplotype[site] = (int8_t) self->node_mutation_states[m];
            tsk_haplotype_bitset_clear(bits, (tsk_size_t) site);
        }
    }

    int32_t child_start = 0;
    int32_t child_stop = 0;
    if (self->parent_index_range != NULL) {
        int32_t range_offset = node * 2;
        child_start = self->parent_index_range[range_offset];
        child_stop = self->parent_index_range[range_offset + 1];
    }
    for (int32_t i = child_start; i < child_stop; i++) {
        tsk_id_t edge = self->parent_edge_index[i];
        int32_t start = self->edge_start_index[edge];
        int32_t end = self->edge_end_index[edge];
        if (start >= end) {
            continue;
        }
        if (tsk_haplotype_bitset_next(
                bits, self->num_bit_words, (tsk_size_t) start, (tsk_size_t) end)
            < (tsk_size_t) end) {
            self->edge_stack[stack_top] = edge;
            self->stack_interval_start[stack_top] = start;
            self->stack_interval_end[stack_top] = end;
            stack_top++;
        }
    }

    while (stack_top > 0) {
        stack_top--;
        tsk_id_t edge = self->edge_stack[stack_top];
        tsk_id_t ancestor = edge_parent[edge];
        interval_start = self->stack_interval_start[stack_top];
        interval_end = self->stack_interval_end[stack_top];

        if (ancestor >= 0) {
            mut_start = self->node_mutation_offsets[ancestor];
            mut_end = self->node_mutation_offsets[ancestor + 1];
            for (int32_t m = mut_start; m < mut_end; m++) {
                int32_t site = self->node_mutation_sites[m];
                if (site >= interval_start && site < interval_end
                    && tsk_haplotype_bitset_next(bits, self->num_bit_words,
                           (tsk_size_t) site, (tsk_size_t) site + 1)
                           == (tsk_size_t) site) {
                    haplotype[site] = (int8_t) self->node_mutation_states[m];
                    tsk_haplotype_bitset_clear(bits, (tsk_size_t) site);
                }
            }
        }

        parent_count = 0;
        if (ancestor >= 0 && self->parent_index_range != NULL) {
            int32_t range_offset = ancestor * 2;
            child_start = self->parent_index_range[range_offset];
            child_stop = self->parent_index_range[range_offset + 1];
            for (int32_t i = child_start; i < child_stop; i++) {
                tsk_id_t parent_edge = self->parent_edge_index[i];
                int32_t parent_start = self->edge_start_index[parent_edge];
                int32_t parent_end = self->edge_end_index[parent_edge];
                if (parent_start < interval_start) {
                    parent_start = interval_start;
                }
                if (parent_end > interval_end) {
                    parent_end = interval_end;
                }
                if (parent_start >= parent_end) {
                    continue;
                }
                if (tsk_haplotype_bitset_next(bits, self->num_bit_words,
                        (tsk_size_t) parent_start, (tsk_size_t) parent_end)
                    < (tsk_size_t) parent_end) {
                    self->edge_stack[stack_top] = parent_edge;
                    self->stack_interval_start[stack_top] = parent_start;
                    self->stack_interval_end[stack_top] = parent_end;
                    stack_top++;
                    self->parent_interval_start[parent_count] = parent_start;
                    self->parent_interval_end[parent_count] = parent_end;
                    parent_count++;
                }
            }
        } else {
            child_start = 0;
            child_stop = 0;
        }

        idx = tsk_haplotype_bitset_next(bits, self->num_bit_words,
            (tsk_size_t) interval_start, (tsk_size_t) interval_end);
        while ((int32_t) idx < interval_end) {
            bool covered = false;
            for (tsk_size_t p = 0; p < parent_count; p++) {
                if (self->parent_interval_start[p] <= (int32_t) idx
                    && (int32_t) idx < self->parent_interval_end[p]) {
                    covered = true;
                    break;
                }
            }
            if (covered) {
                idx = tsk_haplotype_bitset_next(
                    bits, self->num_bit_words, idx + 1, (tsk_size_t) interval_end);
            } else {
                tsk_haplotype_bitset_clear(bits, idx);
                idx = tsk_haplotype_bitset_next(
                    bits, self->num_bit_words, idx, (tsk_size_t) interval_end);
            }
        }
    }

    idx = tsk_haplotype_bitset_next(
        bits, self->num_bit_words, 0, (tsk_size_t) self->num_sites);
    while (idx < (tsk_size_t) self->num_sites) {
        tsk_haplotype_bitset_clear(bits, idx);
        idx = tsk_haplotype_bitset_next(
            bits, self->num_bit_words, idx, (tsk_size_t) self->num_sites);
    }

    return 0;
}

int
tsk_haplotype_free(tsk_haplotype_t *self)
{
    if (self == NULL) {
        return 0;
    }
    tsk_safe_free(self->ancestral_states);
    tsk_safe_free(self->node_mutation_offsets);
    tsk_safe_free(self->node_mutation_sites);
    tsk_safe_free(self->node_mutation_states);
    tsk_safe_free(self->parent_edge_index);
    tsk_safe_free(self->parent_index_range);
    tsk_safe_free(self->edge_start_index);
    tsk_safe_free(self->edge_end_index);
    tsk_safe_free(self->edge_stack);
    tsk_safe_free(self->stack_interval_start);
    tsk_safe_free(self->stack_interval_end);
    tsk_safe_free(self->parent_interval_start);
    tsk_safe_free(self->parent_interval_end);
    tsk_safe_free(self->unresolved_bits);
    tsk_safe_free(self->initial_bits);
    self->tree_sequence = NULL;
    self->site_positions = NULL;
    self->initialised = false;
    return 0;
}

/* ======================================================== *
 * Variant generator
 * ======================================================== */

void
tsk_variant_print_state(const tsk_variant_t *self, FILE *out)
{
    tsk_size_t j;

    fprintf(out, "tsk_variant state\n");
    fprintf(out, "user_alleles = %lld\n", (long long) self->user_alleles);
    fprintf(out, "num_alleles = %lld\n", (long long) self->num_alleles);
    for (j = 0; j < self->num_alleles; j++) {
        fprintf(out, "\tlen = %lld, '%.*s'\n", (long long) self->allele_lengths[j],
            (int) self->allele_lengths[j], self->alleles[j]);
    }
    fprintf(out, "num_samples = %lld\n", (long long) self->num_samples);
}

void
tsk_vargen_print_state(const tsk_vargen_t *self, FILE *out)
{
    tsk_variant_print_state(&self->variant, out);
}

/* Copy the fixed allele mapping specified by the user into local
 * memory. */
static int
tsk_variant_copy_alleles(tsk_variant_t *self, const char **alleles)
{
    int ret = 0;
    tsk_size_t j;
    size_t total_len, allele_len, offset;

    self->num_alleles = self->max_alleles;

    total_len = 0;
    for (j = 0; j < self->num_alleles; j++) {
        allele_len = strlen(alleles[j]);
        self->allele_lengths[j] = (tsk_size_t) allele_len;
        total_len += allele_len;
    }
    self->user_alleles_mem = tsk_malloc(total_len * sizeof(char *));
    if (self->user_alleles_mem == NULL) {
        ret = tsk_trace_error(TSK_ERR_NO_MEMORY);
        goto out;
    }
    offset = 0;
    for (j = 0; j < self->num_alleles; j++) {
        strcpy(self->user_alleles_mem + offset, alleles[j]);
        self->alleles[j] = self->user_alleles_mem + offset;
        offset += (size_t) self->allele_lengths[j];
    }
out:
    return ret;
}

static int
variant_init_samples_and_index_map(tsk_variant_t *self,
    const tsk_treeseq_t *tree_sequence, const tsk_id_t *samples, tsk_size_t num_samples,
    size_t num_samples_alloc, tsk_flags_t options)
{
    int ret = 0;
    const tsk_flags_t *flags = tree_sequence->tables->nodes.flags;
    tsk_size_t j, num_nodes;
    bool impute_missing = !!(options & TSK_ISOLATED_NOT_MISSING);
    tsk_id_t u;

    num_nodes = tsk_treeseq_get_num_nodes(tree_sequence);
    self->alt_samples = tsk_malloc(num_samples_alloc * sizeof(*samples));
    self->alt_sample_index_map
        = tsk_malloc(num_nodes * sizeof(*self->alt_sample_index_map));
    if (self->alt_samples == NULL || self->alt_sample_index_map == NULL) {
        ret = tsk_trace_error(TSK_ERR_NO_MEMORY);
        goto out;
    }
    tsk_memcpy(self->alt_samples, samples, num_samples * sizeof(*samples));
    tsk_memset(self->alt_sample_index_map, 0xff,
        num_nodes * sizeof(*self->alt_sample_index_map));
    /* Create the reverse mapping */
    for (j = 0; j < num_samples; j++) {
        u = samples[j];
        if (u < 0 || u >= (tsk_id_t) num_nodes) {
            ret = tsk_trace_error(TSK_ERR_NODE_OUT_OF_BOUNDS);
            goto out;
        }
        if (self->alt_sample_index_map[u] != TSK_NULL) {
            ret = tsk_trace_error(TSK_ERR_DUPLICATE_SAMPLE);
            goto out;
        }
        /* We can only detect missing data for samples */
        if (!impute_missing && !(flags[u] & TSK_NODE_IS_SAMPLE)) {
            ret = tsk_trace_error(TSK_ERR_MUST_IMPUTE_NON_SAMPLES);
            goto out;
        }
        self->alt_sample_index_map[samples[j]] = (tsk_id_t) j;
    }
out:
    return ret;
}

int
tsk_variant_init(tsk_variant_t *self, const tsk_treeseq_t *tree_sequence,
    const tsk_id_t *samples, tsk_size_t num_samples, const char **alleles,
    tsk_flags_t options)
{
    int ret = 0;
    tsk_size_t max_alleles_limit, max_alleles;
    tsk_size_t num_samples_alloc;

    tsk_memset(self, 0, sizeof(tsk_variant_t));

    /* Set site id to NULL to indicate the variant is not decoded */
    self->site.id = TSK_NULL;

    self->tree_sequence = tree_sequence;
    ret = tsk_tree_init(
        &self->tree, tree_sequence, samples == NULL ? TSK_SAMPLE_LISTS : 0);
    if (ret != 0) {
        goto out;
    }

    if (samples != NULL) {
        /* Take a copy of the samples so we don't have to manage the lifecycle*/
        self->samples = tsk_malloc(num_samples * sizeof(*samples));
        if (self->samples == NULL) {
            ret = tsk_trace_error(TSK_ERR_NO_MEMORY);
            goto out;
        }
        tsk_memcpy(self->samples, samples, num_samples * sizeof(*samples));
        self->num_samples = num_samples;
    }

    self->options = options;

    max_alleles_limit = INT32_MAX;

    if (alleles == NULL) {
        self->user_alleles = false;
        max_alleles = 4; /* Arbitrary --- we'll rarely have more than this */
    } else {
        self->user_alleles = true;
        /* Count the input alleles. The end is designated by the NULL sentinel. */
        for (max_alleles = 0; alleles[max_alleles] != NULL; max_alleles++)
            ;
        if (max_alleles > max_alleles_limit) {
            ret = tsk_trace_error(TSK_ERR_TOO_MANY_ALLELES);
            goto out;
        }
        if (max_alleles == 0) {
            ret = tsk_trace_error(TSK_ERR_ZERO_ALLELES);
            goto out;
        }
    }
    self->max_alleles = max_alleles;
    self->alleles = tsk_calloc(max_alleles, sizeof(*self->alleles));
    self->allele_lengths = tsk_malloc(max_alleles * sizeof(*self->allele_lengths));
    if (self->alleles == NULL || self->allele_lengths == NULL) {
        ret = tsk_trace_error(TSK_ERR_NO_MEMORY);
        goto out;
    }
    if (self->user_alleles) {
        ret = tsk_variant_copy_alleles(self, alleles);
        if (ret != 0) {
            goto out;
        }
    }
    if (self->samples == NULL) {
        self->num_samples = tsk_treeseq_get_num_samples(tree_sequence);
        self->samples = tsk_malloc(self->num_samples * sizeof(*self->samples));
        if (self->samples == NULL) {
            ret = tsk_trace_error(TSK_ERR_NO_MEMORY);
            goto out;
        }
        tsk_memcpy(self->samples, tsk_treeseq_get_samples(tree_sequence),
            self->num_samples * sizeof(*self->samples));

        self->sample_index_map = tsk_treeseq_get_sample_index_map(tree_sequence);
        num_samples_alloc = self->num_samples;
    } else {
        num_samples_alloc = self->num_samples;
        ret = variant_init_samples_and_index_map(self, tree_sequence, self->samples,
            self->num_samples, (size_t) num_samples_alloc, self->options);
        if (ret != 0) {
            goto out;
        }
        self->sample_index_map = self->alt_sample_index_map;
    }
    /* When a list of samples is given, we use the traversal based algorithm
     * which doesn't use sample list tracking in the tree */
    if (self->alt_samples != NULL) {
        self->traversal_stack = tsk_malloc(
            tsk_treeseq_get_num_nodes(tree_sequence) * sizeof(*self->traversal_stack));
        if (self->traversal_stack == NULL) {
            ret = tsk_trace_error(TSK_ERR_NO_MEMORY);
            goto out;
        }
    }

    self->genotypes = tsk_malloc(num_samples_alloc * sizeof(*self->genotypes));
    if (self->genotypes == NULL || self->alleles == NULL
        || self->allele_lengths == NULL) {
        ret = tsk_trace_error(TSK_ERR_NO_MEMORY);
        goto out;
    }

out:
    return ret;
}

int
tsk_vargen_init(tsk_vargen_t *self, const tsk_treeseq_t *tree_sequence,
    const tsk_id_t *samples, tsk_size_t num_samples, const char **alleles,
    tsk_flags_t options)
{
    int ret = 0;

    tsk_bug_assert(tree_sequence != NULL);
    tsk_memset(self, 0, sizeof(tsk_vargen_t));

    self->tree_sequence = tree_sequence;
    ret = tsk_variant_init(
        &self->variant, tree_sequence, samples, num_samples, alleles, options);
    if (ret != 0) {
        goto out;
    }
    ret = 0;
out:
    return ret;
}

int
tsk_variant_free(tsk_variant_t *self)
{
    if (self->tree_sequence != NULL) {
        tsk_tree_free(&self->tree);
    }
    tsk_safe_free(self->genotypes);
    tsk_safe_free(self->alleles);
    tsk_safe_free(self->allele_lengths);
    tsk_safe_free(self->user_alleles_mem);
    tsk_safe_free(self->samples);
    tsk_safe_free(self->alt_samples);
    tsk_safe_free(self->alt_sample_index_map);
    tsk_safe_free(self->traversal_stack);
    return 0;
}

int
tsk_vargen_free(tsk_vargen_t *self)
{
    tsk_variant_free(&self->variant);
    return 0;
}

static int
tsk_variant_expand_alleles(tsk_variant_t *self)
{
    int ret = 0;
    void *p;
    tsk_size_t hard_limit = INT32_MAX;

    if (self->max_alleles == hard_limit) {
        ret = tsk_trace_error(TSK_ERR_TOO_MANY_ALLELES);
        goto out;
    }
    self->max_alleles = TSK_MIN(hard_limit, self->max_alleles * 2);
    p = tsk_realloc(self->alleles, self->max_alleles * sizeof(*self->alleles));
    if (p == NULL) {
        ret = tsk_trace_error(TSK_ERR_NO_MEMORY);
        goto out;
    }
    self->alleles = p;
    p = tsk_realloc(
        self->allele_lengths, self->max_alleles * sizeof(*self->allele_lengths));
    if (p == NULL) {
        ret = tsk_trace_error(TSK_ERR_NO_MEMORY);
        goto out;
    }
    self->allele_lengths = p;
out:
    return ret;
}

/* The following pair of functions are identical except one handles 8 bit
 * genotypes and the other handles 16 bit genotypes. This is done for performance
 * reasons as this is a key function and for common alleles can entail
 * iterating over millions of samples. The compiler hints are included for the
 * same reason.
 */
static int TSK_WARN_UNUSED
tsk_variant_update_genotypes_sample_list(
    tsk_variant_t *self, tsk_id_t node, tsk_id_t derived)
{
    int32_t *restrict genotypes = self->genotypes;
    const tsk_id_t *restrict list_left = self->tree.left_sample;
    const tsk_id_t *restrict list_right = self->tree.right_sample;
    const tsk_id_t *restrict list_next = self->tree.next_sample;
    tsk_id_t index, stop;
    int ret = 0;

    tsk_bug_assert(derived < INT32_MAX);

    index = list_left[node];
    if (index != TSK_NULL) {
        stop = list_right[node];
        while (true) {

            ret += genotypes[index] == TSK_MISSING_DATA;
            genotypes[index] = (int32_t) derived;
            if (index == stop) {
                break;
            }
            index = list_next[index];
        }
    }

    return ret;
}

/* The following functions implement the genotype setting by traversing
 * down the tree to the samples. We're not so worried about performance here
 * because this should only be used when we have a very small number of samples,
 * and so we use a visit function to avoid duplicating code.
 */

typedef int (*visit_func_t)(tsk_variant_t *, tsk_id_t, tsk_id_t);

static int TSK_WARN_UNUSED
tsk_variant_traverse(
    tsk_variant_t *self, tsk_id_t node, tsk_id_t derived, visit_func_t visit)
{
    int ret = 0;
    tsk_id_t *restrict stack = self->traversal_stack;
    const tsk_id_t *restrict left_child = self->tree.left_child;
    const tsk_id_t *restrict right_sib = self->tree.right_sib;
    const tsk_id_t *restrict sample_index_map = self->sample_index_map;
    tsk_id_t u, v, sample_index;
    int stack_top;
    int no_longer_missing = 0;

    stack_top = 0;
    stack[0] = node;
    while (stack_top >= 0) {
        u = stack[stack_top];
        sample_index = sample_index_map[u];
        if (sample_index != TSK_NULL) {
            ret = visit(self, sample_index, derived);
            if (ret < 0) {
                goto out;
            }
            no_longer_missing += ret;
        }
        stack_top--;
        for (v = left_child[u]; v != TSK_NULL; v = right_sib[v]) {
            stack_top++;
            stack[stack_top] = v;
        }
    }
    ret = no_longer_missing;
out:
    return ret;
}

static int
tsk_variant_visit(tsk_variant_t *self, tsk_id_t sample_index, tsk_id_t derived)
{
    int ret = 0;
    int32_t *restrict genotypes = self->genotypes;

    tsk_bug_assert(derived < INT32_MAX);
    tsk_bug_assert(sample_index != -1);

    ret = genotypes[sample_index] == TSK_MISSING_DATA;
    genotypes[sample_index] = (int32_t) derived;

    return ret;
}

static int TSK_WARN_UNUSED
tsk_variant_update_genotypes_traversal(
    tsk_variant_t *self, tsk_id_t node, tsk_id_t derived)
{
    return tsk_variant_traverse(self, node, derived, tsk_variant_visit);
}

static tsk_size_t
tsk_variant_mark_missing(tsk_variant_t *self)
{
    tsk_size_t num_missing = 0;
    const tsk_id_t *restrict left_child = self->tree.left_child;
    const tsk_id_t *restrict right_sib = self->tree.right_sib;
    const tsk_id_t *restrict sample_index_map = self->sample_index_map;
    const tsk_id_t N = self->tree.virtual_root;
    int32_t *restrict genotypes = self->genotypes;
    tsk_id_t root, sample_index;

    for (root = left_child[N]; root != TSK_NULL; root = right_sib[root]) {
        if (left_child[root] == TSK_NULL) {
            sample_index = sample_index_map[root];
            if (sample_index != TSK_NULL) {
                genotypes[sample_index] = TSK_MISSING_DATA;
                num_missing++;
            }
        }
    }
    return num_missing;
}

static tsk_id_t
tsk_variant_get_allele_index(tsk_variant_t *self, const char *allele, tsk_size_t length)
{
    tsk_id_t ret = -1;
    tsk_size_t j;

    for (j = 0; j < self->num_alleles; j++) {
        if (length == self->allele_lengths[j]
            && tsk_memcmp(allele, self->alleles[j], length) == 0) {
            ret = (tsk_id_t) j;
            break;
        }
    }
    return ret;
}

int
tsk_variant_decode(
    tsk_variant_t *self, tsk_id_t site_id, tsk_flags_t TSK_UNUSED(options))
{
    int ret = 0;
    tsk_id_t allele_index;
    tsk_size_t j, num_missing;
    int no_longer_missing;
    tsk_mutation_t mutation;
    bool impute_missing = !!(self->options & TSK_ISOLATED_NOT_MISSING);
    bool by_traversal = self->alt_samples != NULL;
    int (*update_genotypes)(tsk_variant_t *, tsk_id_t, tsk_id_t);
    tsk_size_t (*mark_missing)(tsk_variant_t *);

    if (self->tree_sequence == NULL) {
        ret = tsk_trace_error(TSK_ERR_VARIANT_CANT_DECODE_COPY);
        goto out;
    }

    ret = tsk_treeseq_get_site(self->tree_sequence, site_id, &self->site);
    if (ret != 0) {
        goto out;
    }

    ret = tsk_tree_seek(&self->tree, self->site.position, 0);
    if (ret != 0) {
        goto out;
    }

    /* When we have no specified samples we need sample lists to be active
     * on the tree, as indicated by the presence of left_sample */
    if (!by_traversal && self->tree.left_sample == NULL) {
        ret = tsk_trace_error(TSK_ERR_NO_SAMPLE_LISTS);
        goto out;
    }

    /* For now we use a traversal method to find genotypes when we have a
     * specified set of samples, but we should provide the option to do it
     * via tracked_samples in the tree also. There will be a tradeoff: if
     * we only have a small number of samples, it's probably better to
     * do it by traversal. For large sets of samples though, it may be
     * better to use the sample list infrastructure. */

    mark_missing = tsk_variant_mark_missing;
    update_genotypes = tsk_variant_update_genotypes_sample_list;
    if (by_traversal) {
        update_genotypes = tsk_variant_update_genotypes_traversal;
    }

    if (self->user_alleles) {
        allele_index = tsk_variant_get_allele_index(
            self, self->site.ancestral_state, self->site.ancestral_state_length);
        if (allele_index == -1) {
            ret = tsk_trace_error(TSK_ERR_ALLELE_NOT_FOUND);
            goto out;
        }
    } else {
        /* Ancestral state is always allele 0 */
        self->alleles[0] = self->site.ancestral_state;
        self->allele_lengths[0] = self->site.ancestral_state_length;
        self->num_alleles = 1;
        allele_index = 0;
    }

    /* The algorithm for generating the allelic state of every sample works by
     * examining each mutation in order, and setting the state for all the
     * samples under the mutation's node. For complex sites where there is
     * more than one mutation, we depend on the ordering of mutations being
     * correct. Specifically, any mutation that is above another mutation in
     * the tree must be visited first. This is enforced using the mutation.parent
     * field, where we require that a mutation's parent must appear before it
     * in the list of mutations. This guarantees the correctness of this algorithm.
     */
    for (j = 0; j < self->num_samples; j++) {
        self->genotypes[j] = (int32_t) allele_index;
    }

    /* We mark missing data *before* updating the genotypes because
     * mutations directly over samples should not be missing */
    num_missing = 0;
    if (!impute_missing) {
        num_missing = mark_missing(self);
    }
    for (j = 0; j < self->site.mutations_length; j++) {
        mutation = self->site.mutations[j];
        /* Compute the allele index for this derived state value. */
        allele_index = tsk_variant_get_allele_index(
            self, mutation.derived_state, mutation.derived_state_length);
        if (allele_index == -1) {
            if (self->user_alleles) {
                ret = tsk_trace_error(TSK_ERR_ALLELE_NOT_FOUND);
                goto out;
            }
            if (self->num_alleles == self->max_alleles) {
                ret = tsk_variant_expand_alleles(self);
                if (ret != 0) {
                    goto out;
                }
            }
            allele_index = (tsk_id_t) self->num_alleles;
            self->alleles[allele_index] = mutation.derived_state;
            self->allele_lengths[allele_index] = mutation.derived_state_length;
            self->num_alleles++;
        }

        no_longer_missing = update_genotypes(self, mutation.node, allele_index);
        if (no_longer_missing < 0) {
            ret = no_longer_missing;
            goto out;
        }
        /* Update genotypes returns the number of missing values marked
         * not-missing */
        num_missing -= (tsk_size_t) no_longer_missing;
    }
    self->has_missing_data = num_missing > 0;
out:
    return ret;
}

int
tsk_variant_restricted_copy(const tsk_variant_t *self, tsk_variant_t *other)
{
    int ret = 0;
    tsk_size_t total_len, offset, j;

    /* Copy everything */
    tsk_memcpy(other, self, sizeof(*other));
    /* Tree sequence left as NULL and zero'd tree is a way of indicating this variant is
     * fixed and cannot be further decoded. */
    other->tree_sequence = NULL;
    tsk_memset(&other->tree, sizeof(other->tree), 0);
    other->traversal_stack = NULL;
    other->samples = NULL;
    other->sample_index_map = NULL;
    other->alt_samples = NULL;
    other->alt_sample_index_map = NULL;
    other->user_alleles_mem = NULL;

    total_len = 0;
    for (j = 0; j < self->num_alleles; j++) {
        total_len += self->allele_lengths[j];
    }
    other->samples = tsk_malloc(other->num_samples * sizeof(*other->samples));
    other->genotypes = tsk_malloc(other->num_samples * sizeof(*other->genotypes));
    other->user_alleles_mem = tsk_malloc(total_len * sizeof(*other->user_alleles_mem));
    other->allele_lengths
        = tsk_malloc(other->num_alleles * sizeof(*other->allele_lengths));
    other->alleles = tsk_malloc(other->num_alleles * sizeof(*other->alleles));
    if (other->samples == NULL || other->genotypes == NULL
        || other->user_alleles_mem == NULL || other->allele_lengths == NULL
        || other->alleles == NULL) {
        ret = tsk_trace_error(TSK_ERR_NO_MEMORY);
        goto out;
    }
    tsk_memcpy(
        other->samples, self->samples, other->num_samples * sizeof(*other->samples));
    tsk_memcpy(other->genotypes, self->genotypes,
        other->num_samples * sizeof(*other->genotypes));
    tsk_memcpy(other->allele_lengths, self->allele_lengths,
        other->num_alleles * sizeof(*other->allele_lengths));
    offset = 0;
    for (j = 0; j < other->num_alleles; j++) {
        tsk_memcpy(other->user_alleles_mem + offset, self->alleles[j],
            other->allele_lengths[j] * sizeof(*other->user_alleles_mem));
        other->alleles[j] = other->user_alleles_mem + offset;
        offset += other->allele_lengths[j];
    }

out:
    return ret;
}

int
tsk_vargen_next(tsk_vargen_t *self, tsk_variant_t **variant)
{
    int ret = 0;

    if ((tsk_size_t) self->site_index < tsk_treeseq_get_num_sites(self->tree_sequence)) {
        ret = tsk_variant_decode(&self->variant, self->site_index, 0);
        if (ret != 0) {
            goto out;
        }
        self->site_index++;
        *variant = &self->variant;
        ret = 1;
    }
out:
    return ret;
}
