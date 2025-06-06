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

#ifndef __TESTLIB_H__
#define __TESTLIB_H__

#define _GNU_SOURCE
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#include <CUnit/Basic.h>
#include <tskit/trees.h>

/* Global variables used in the test suite */

extern char *_tmp_file_name;
extern FILE *_devnull;

int test_main(CU_TestInfo *tests, int argc, char **argv);

void tsk_treeseq_from_text(tsk_treeseq_t *ts, double sequence_length, const char *nodes,
    const char *edges, const char *migrations, const char *sites, const char *mutations,
    const char *individuals, const char *provenance, tsk_flags_t tc_options);
tsk_treeseq_t *caterpillar_tree(
    tsk_size_t num_samples, tsk_size_t num_sites, tsk_size_t num_mutations);

void parse_nodes(const char *text, tsk_node_table_t *node_table);
void parse_edges(const char *text, tsk_edge_table_t *edge_table);
void parse_sites(const char *text, tsk_site_table_t *site_table);
void parse_mutations(const char *text, tsk_mutation_table_t *mutation_table);
void parse_individuals(const char *text, tsk_individual_table_t *individual_table);

void unsort_edges(tsk_edge_table_t *edges, size_t start);

/* Use a macro so we can get line numbers at roughly the right place */
#define assert_arrays_almost_equal(len, a, b)                                           \
    {                                                                                   \
        do {                                                                            \
            tsk_size_t _j;                                                              \
            for (_j = 0; _j < len; _j++) {                                              \
                CU_ASSERT_DOUBLE_EQUAL(a[_j], b[_j], 1e-9);                             \
            }                                                                           \
        } while (0);                                                                    \
    }

#define assert_arrays_equal(len, a, b)                                                  \
    {                                                                                   \
        do {                                                                            \
            tsk_size_t _j;                                                              \
            for (_j = 0; _j < len; _j++) {                                              \
                CU_ASSERT_EQUAL(a[_j], b[_j]);                                          \
            }                                                                           \
        } while (0);                                                                    \
    }

/* Array equality if the arrays contain NaN values
   NB: the float cast for NaNs is for mingw, which complains without */
#define assert_arrays_almost_equal_nan(len, a, b)                                       \
    {                                                                                   \
        do {                                                                            \
            tsk_size_t _j;                                                              \
            for (_j = 0; _j < len; _j++) {                                              \
                if (isnan((float) a[_j]) || isnan((float) b[_j])) {                     \
                    CU_ASSERT_EQUAL_FATAL(isnan((float) a[_j]), isnan((float) b[_j]));  \
                } else {                                                                \
                    CU_ASSERT_DOUBLE_EQUAL(a[_j], b[_j], 1e-9);                         \
                }                                                                       \
            }                                                                           \
        } while (0);                                                                    \
    }

extern const char *single_tree_ex_nodes;
extern const char *single_tree_ex_edges;
extern const char *single_tree_ex_sites;
extern const char *single_tree_ex_mutations;

extern const char *multiple_tree_ex_nodes;
extern const char *multiple_tree_ex_edges;

extern const char *odd_tree1_ex_nodes;
extern const char *odd_tree1_ex_edges;

extern const char *multi_root_tree_ex_nodes;
extern const char *multi_root_tree_ex_edges;

extern const char *multi_path_tree_ex_nodes;
extern const char *multi_path_tree_ex_edges;

extern const char *nonbinary_ex_nodes;
extern const char *nonbinary_ex_edges;
extern const char *nonbinary_ex_sites;
extern const char *nonbinary_ex_mutations;

extern const char *unary_ex_nodes;
extern const char *unary_ex_edges;
extern const char *unary_ex_sites;
extern const char *unary_ex_mutations;

extern const char *internal_sample_ex_nodes;
extern const char *internal_sample_ex_edges;
extern const char *internal_sample_ex_sites;
extern const char *internal_sample_ex_mutations;

extern const char *multiroot_ex_nodes;
extern const char *multiroot_ex_edges;
extern const char *multiroot_ex_sites;
extern const char *multiroot_ex_mutations;

extern const char *empty_ex_nodes;
extern const char *empty_ex_edges;

extern const char *paper_ex_nodes;
extern const char *paper_ex_edges;
extern const char *paper_ex_sites;
extern const char *paper_ex_mutations;
extern const char *paper_ex_individuals;

extern const char *missing_ex_nodes;
extern const char *missing_ex_edges;

#endif
