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
#include <tskit/genotypes.h>

#include <unistd.h>
#include <stdlib.h>

static void
test_simplest_missing_data(void)
{
    const char *nodes =
        "1  0   0\n"
        "1  0   0\n";
    const char *sites =
        "0.0    A\n";
    tsk_treeseq_t ts;
    tsk_vargen_t vargen;
    tsk_variant_t *var;
    int ret;

    tsk_treeseq_from_text(&ts, 1, nodes, "", NULL, sites, NULL, NULL, NULL);
    CU_ASSERT_EQUAL(tsk_treeseq_get_num_samples(&ts), 2);
    CU_ASSERT_EQUAL(tsk_treeseq_get_num_sites(&ts), 1);

    ret = tsk_vargen_init(&vargen, &ts, NULL, 0, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_vargen_next(&vargen, &var);
    CU_ASSERT_EQUAL_FATAL(ret, 1);
    CU_ASSERT_EQUAL(var->site->position, 0.0);
    CU_ASSERT_TRUE(var->has_missing_data);
    CU_ASSERT_EQUAL(var->genotypes.i8[0], TSK_MISSING_DATA);
    CU_ASSERT_EQUAL(var->genotypes.i8[1], TSK_MISSING_DATA);
    ret = tsk_vargen_next(&vargen, &var);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tsk_vargen_free(&vargen);

    ret = tsk_vargen_init(&vargen, &ts, NULL, 0, NULL, TSK_16_BIT_GENOTYPES);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_vargen_next(&vargen, &var);
    CU_ASSERT_EQUAL_FATAL(ret, 1);
    CU_ASSERT_EQUAL(var->site->position, 0.0);
    CU_ASSERT_TRUE(var->has_missing_data);
    CU_ASSERT_EQUAL(var->genotypes.i16[0], TSK_MISSING_DATA);
    CU_ASSERT_EQUAL(var->genotypes.i16[1], TSK_MISSING_DATA);
    ret = tsk_vargen_next(&vargen, &var);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tsk_vargen_free(&vargen);

    ret = tsk_vargen_init(&vargen, &ts, NULL, 0, NULL, TSK_IMPUTE_MISSING_DATA);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_vargen_next(&vargen, &var);
    CU_ASSERT_EQUAL_FATAL(ret, 1);
    CU_ASSERT_EQUAL(var->site->position, 0.0);
    CU_ASSERT_FALSE(var->has_missing_data);
    CU_ASSERT_EQUAL(var->genotypes.i8[0], 0);
    CU_ASSERT_EQUAL(var->genotypes.i8[1], 0);
    ret = tsk_vargen_next(&vargen, &var);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tsk_vargen_free(&vargen);

    tsk_treeseq_free(&ts);
}

static void
test_simplest_missing_data_user_alleles(void)
{
    const char *nodes =
        "1  0   0\n"
        "1  0   0\n";
    const char *sites =
        "0.0    A\n";
    tsk_treeseq_t ts;
    tsk_vargen_t vargen;
    tsk_variant_t *var;
    const char *alleles[] = {"A", NULL};
    int ret;

    tsk_treeseq_from_text(&ts, 1, nodes, "", NULL, sites, NULL, NULL, NULL);
    CU_ASSERT_EQUAL(tsk_treeseq_get_num_samples(&ts), 2);
    CU_ASSERT_EQUAL(tsk_treeseq_get_num_sites(&ts), 1);

    ret = tsk_vargen_init(&vargen, &ts, NULL, 0, alleles, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_vargen_next(&vargen, &var);
    CU_ASSERT_EQUAL_FATAL(ret, 1);
    CU_ASSERT_EQUAL(var->site->position, 0.0);
    CU_ASSERT_TRUE(var->has_missing_data);
    CU_ASSERT_EQUAL(var->genotypes.i8[0], TSK_MISSING_DATA);
    CU_ASSERT_EQUAL(var->genotypes.i8[1], TSK_MISSING_DATA);
    ret = tsk_vargen_next(&vargen, &var);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tsk_vargen_free(&vargen);

    ret = tsk_vargen_init(&vargen, &ts, NULL, 0, NULL, TSK_16_BIT_GENOTYPES);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_vargen_next(&vargen, &var);
    CU_ASSERT_EQUAL_FATAL(ret, 1);
    CU_ASSERT_EQUAL(var->site->position, 0.0);
    CU_ASSERT_TRUE(var->has_missing_data);
    CU_ASSERT_EQUAL(var->genotypes.i16[0], TSK_MISSING_DATA);
    CU_ASSERT_EQUAL(var->genotypes.i16[1], TSK_MISSING_DATA);
    ret = tsk_vargen_next(&vargen, &var);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tsk_vargen_free(&vargen);

    ret = tsk_vargen_init(&vargen, &ts, NULL, 0, NULL, TSK_IMPUTE_MISSING_DATA);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_vargen_next(&vargen, &var);
    CU_ASSERT_EQUAL_FATAL(ret, 1);
    CU_ASSERT_EQUAL(var->site->position, 0.0);
    CU_ASSERT_FALSE(var->has_missing_data);
    CU_ASSERT_EQUAL(var->genotypes.i8[0], 0);
    CU_ASSERT_EQUAL(var->genotypes.i8[1], 0);
    ret = tsk_vargen_next(&vargen, &var);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tsk_vargen_free(&vargen);

    tsk_treeseq_free(&ts);
}

static void
test_single_tree_user_alleles(void)
{
    int ret = 0;
    const char *sites =
        "0.0    G\n"
        "0.125  A\n"
        "0.25   C\n"
        "0.5    A\n";
    const char *mutations =
        "0    0     T   -1\n"
        "1    1     C   -1\n"
        "2    0     G   -1\n"
        "2    1     A   -1\n"
        "2    2     T   -1\n"  // A bunch of different sample mutations
        "3    4     T   -1\n"
        "3    0     A   5\n"; // A back mutation from T -> A
    tsk_treeseq_t ts;
    tsk_vargen_t vargen;
    tsk_variant_t *var;
    const char *alleles[] = {"A", "C", "G", "T", NULL};

    tsk_treeseq_from_text(&ts, 1, single_tree_ex_nodes, single_tree_ex_edges, NULL,
            sites, mutations, NULL, NULL);
    ret = tsk_vargen_init(&vargen, &ts, NULL, 0, alleles, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    tsk_vargen_print_state(&vargen, _devnull);

    ret = tsk_vargen_next(&vargen, &var);
    CU_ASSERT_EQUAL_FATAL(ret, 1);
    CU_ASSERT_EQUAL(var->site->position, 0.0);
    CU_ASSERT_EQUAL_FATAL(var->num_alleles, 4);
    CU_ASSERT_EQUAL(var->allele_lengths[0], 1);
    CU_ASSERT_EQUAL(var->allele_lengths[1], 1);
    CU_ASSERT_EQUAL(var->allele_lengths[2], 1);
    CU_ASSERT_EQUAL(var->allele_lengths[3], 1);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[0], "A", 1);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[1], "C", 1);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[2], "G", 1);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[3], "T", 1);
    CU_ASSERT_FALSE(var->has_missing_data);
    CU_ASSERT_EQUAL(var->genotypes.i8[0], 3);
    CU_ASSERT_EQUAL(var->genotypes.i8[1], 2);
    CU_ASSERT_EQUAL(var->genotypes.i8[2], 2);
    CU_ASSERT_EQUAL(var->genotypes.i8[3], 2);

    ret = tsk_vargen_next(&vargen, &var);
    CU_ASSERT_EQUAL_FATAL(ret, 1);
    CU_ASSERT_EQUAL(var->site->position, 0.125);
    CU_ASSERT_EQUAL(var->num_alleles, 4);
    CU_ASSERT_EQUAL(var->allele_lengths[0], 1);
    CU_ASSERT_EQUAL(var->allele_lengths[1], 1);
    CU_ASSERT_EQUAL(var->allele_lengths[2], 1);
    CU_ASSERT_EQUAL(var->allele_lengths[3], 1);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[0], "A", 1);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[1], "C", 1);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[2], "G", 1);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[3], "T", 1);
    CU_ASSERT_FALSE(var->has_missing_data);
    CU_ASSERT_EQUAL(var->genotypes.i8[0], 0);
    CU_ASSERT_EQUAL(var->genotypes.i8[1], 1);
    CU_ASSERT_EQUAL(var->genotypes.i8[2], 0);
    CU_ASSERT_EQUAL(var->genotypes.i8[3], 0);

    ret = tsk_vargen_next(&vargen, &var);
    CU_ASSERT_EQUAL_FATAL(ret, 1);
    CU_ASSERT_EQUAL(var->site->position, 0.25);
    CU_ASSERT_EQUAL(var->num_alleles, 4);
    CU_ASSERT_EQUAL(var->allele_lengths[0], 1);
    CU_ASSERT_EQUAL(var->allele_lengths[1], 1);
    CU_ASSERT_EQUAL(var->allele_lengths[2], 1);
    CU_ASSERT_EQUAL(var->allele_lengths[3], 1);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[0], "A", 1);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[1], "C", 1);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[2], "G", 1);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[3], "T", 1);
    CU_ASSERT_FALSE(var->has_missing_data);
    CU_ASSERT_EQUAL(var->genotypes.i8[0], 2);
    CU_ASSERT_EQUAL(var->genotypes.i8[1], 0);
    CU_ASSERT_EQUAL(var->genotypes.i8[2], 3);
    CU_ASSERT_EQUAL(var->genotypes.i8[3], 1);

    ret = tsk_vargen_next(&vargen, &var);
    CU_ASSERT_EQUAL_FATAL(ret, 1);
    CU_ASSERT_EQUAL(var->site->position, 0.5);
    CU_ASSERT_EQUAL(var->num_alleles, 4);
    CU_ASSERT_EQUAL(var->allele_lengths[0], 1);
    CU_ASSERT_EQUAL(var->allele_lengths[1], 1);
    CU_ASSERT_EQUAL(var->allele_lengths[2], 1);
    CU_ASSERT_EQUAL(var->allele_lengths[3], 1);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[0], "A", 1);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[1], "C", 1);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[2], "G", 1);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[3], "T", 1);
    CU_ASSERT_FALSE(var->has_missing_data);
    CU_ASSERT_EQUAL(var->genotypes.i8[0], 0);
    CU_ASSERT_EQUAL(var->genotypes.i8[1], 3);
    CU_ASSERT_EQUAL(var->genotypes.i8[2], 0);
    CU_ASSERT_EQUAL(var->genotypes.i8[3], 0);

    ret = tsk_vargen_next(&vargen, &var);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    tsk_vargen_free(&vargen);
    tsk_treeseq_free(&ts);
}

static void
test_single_tree_char_alphabet(void)
{
    int ret = 0;
    const char *sites =
        "0.0    A\n"
        "0.125  A\n"
        "0.25   C\n"
        "0.5    A\n";
    const char *mutations =
        "0    0     T   -1\n"
        "1    1     TTTAAGGG   -1\n"
        "2    0     G   -1\n"
        "2    1     AT  -1\n"
        "2    2     T   -1\n"  // A bunch of different sample mutations
        "3    4     T   -1\n"
        "3    0     A   5\n"; // A back mutation from T -> A
    tsk_treeseq_t ts;
    tsk_vargen_t vargen;
    tsk_variant_t *var;

    tsk_treeseq_from_text(&ts, 1, single_tree_ex_nodes, single_tree_ex_edges, NULL,
            sites, mutations, NULL, NULL);
    ret = tsk_vargen_init(&vargen, &ts, NULL, 0, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_vargen_next(&vargen, &var);
    CU_ASSERT_EQUAL_FATAL(ret, 1);
    CU_ASSERT_EQUAL(var->site->position, 0.0);
    CU_ASSERT_EQUAL(var->num_alleles, 2);
    CU_ASSERT_EQUAL(var->allele_lengths[0], 1);
    CU_ASSERT_EQUAL(var->allele_lengths[1], 1);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[0], "A", 1);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[1], "T", 1);
    CU_ASSERT_FALSE(var->has_missing_data);
    CU_ASSERT_EQUAL(var->genotypes.i8[0], 1);
    CU_ASSERT_EQUAL(var->genotypes.i8[1], 0);
    CU_ASSERT_EQUAL(var->genotypes.i8[2], 0);
    CU_ASSERT_EQUAL(var->genotypes.i8[3], 0);

    ret = tsk_vargen_next(&vargen, &var);
    CU_ASSERT_EQUAL_FATAL(ret, 1);
    CU_ASSERT_EQUAL(var->site->position, 0.125);
    CU_ASSERT_EQUAL(var->num_alleles, 2);
    CU_ASSERT_EQUAL(var->allele_lengths[0], 1);
    CU_ASSERT_EQUAL(var->allele_lengths[1], 8);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[0], "A", 1);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[1], "TTTAAGGG", 8);
    CU_ASSERT_FALSE(var->has_missing_data);
    CU_ASSERT_EQUAL(var->genotypes.i8[0], 0);
    CU_ASSERT_EQUAL(var->genotypes.i8[1], 1);
    CU_ASSERT_EQUAL(var->genotypes.i8[2], 0);
    CU_ASSERT_EQUAL(var->genotypes.i8[3], 0);

    ret = tsk_vargen_next(&vargen, &var);
    CU_ASSERT_EQUAL_FATAL(ret, 1);
    CU_ASSERT_EQUAL(var->site->position, 0.25);
    CU_ASSERT_EQUAL(var->num_alleles, 4);
    CU_ASSERT_EQUAL(var->allele_lengths[0], 1);
    CU_ASSERT_EQUAL(var->allele_lengths[1], 1);
    CU_ASSERT_EQUAL(var->allele_lengths[2], 2);
    CU_ASSERT_EQUAL(var->allele_lengths[3], 1);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[0], "C", 1);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[1], "G", 1);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[2], "AT", 1);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[3], "T", 1);
    CU_ASSERT_FALSE(var->has_missing_data);
    CU_ASSERT_EQUAL(var->genotypes.i8[0], 1);
    CU_ASSERT_EQUAL(var->genotypes.i8[1], 2);
    CU_ASSERT_EQUAL(var->genotypes.i8[2], 3);
    CU_ASSERT_EQUAL(var->genotypes.i8[3], 0);

    ret = tsk_vargen_next(&vargen, &var);
    CU_ASSERT_EQUAL_FATAL(ret, 1);
    CU_ASSERT_EQUAL(var->site->position, 0.5);
    CU_ASSERT_EQUAL(var->num_alleles, 2);
    CU_ASSERT_EQUAL(var->allele_lengths[0], 1);
    CU_ASSERT_EQUAL(var->allele_lengths[1], 1);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[0], "A", 1);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[1], "T", 1);
    CU_ASSERT_FALSE(var->has_missing_data);
    CU_ASSERT_EQUAL(var->genotypes.i8[0], 0);
    CU_ASSERT_EQUAL(var->genotypes.i8[1], 1);
    CU_ASSERT_EQUAL(var->genotypes.i8[2], 0);
    CU_ASSERT_EQUAL(var->genotypes.i8[3], 0);

    ret = tsk_vargen_next(&vargen, &var);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    tsk_vargen_free(&vargen);
    tsk_treeseq_free(&ts);
}

static void
test_single_tree_binary_alphabet(void)
{
    int ret = 0;
    tsk_treeseq_t ts;
    tsk_vargen_t vargen;
    tsk_variant_t *var;

    tsk_treeseq_from_text(&ts, 1, single_tree_ex_nodes, single_tree_ex_edges, NULL,
            single_tree_ex_sites, single_tree_ex_mutations, NULL, NULL);
    ret = tsk_vargen_init(&vargen, &ts, NULL, 0, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tsk_vargen_print_state(&vargen, _devnull);

    ret = tsk_vargen_next(&vargen, &var);
    CU_ASSERT_EQUAL_FATAL(ret, 1);
    CU_ASSERT_EQUAL(var->genotypes.i8[0], 0);
    CU_ASSERT_EQUAL(var->genotypes.i8[1], 0);
    CU_ASSERT_EQUAL(var->genotypes.i8[2], 1);
    CU_ASSERT_EQUAL(var->genotypes.i8[3], 0);
    CU_ASSERT_EQUAL(var->num_alleles, 2);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[0], "0", 1);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[1], "1", 1);
    CU_ASSERT_EQUAL(var->site->id, 0);
    CU_ASSERT_EQUAL(var->site->mutations_length, 1);

    ret = tsk_vargen_next(&vargen, &var);
    CU_ASSERT_EQUAL_FATAL(ret, 1);
    CU_ASSERT_EQUAL(var->genotypes.i8[0], 0);
    CU_ASSERT_EQUAL(var->genotypes.i8[1], 1);
    CU_ASSERT_EQUAL(var->genotypes.i8[2], 0);
    CU_ASSERT_EQUAL(var->genotypes.i8[3], 0);
    CU_ASSERT_EQUAL(var->num_alleles, 2);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[0], "0", 1);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[1], "1", 1);
    CU_ASSERT_EQUAL(var->site->id, 1);
    CU_ASSERT_EQUAL(var->site->mutations_length, 2);

    ret = tsk_vargen_next(&vargen, &var);
    CU_ASSERT_EQUAL_FATAL(ret, 1);
    CU_ASSERT_EQUAL(var->genotypes.i8[0], 1);
    CU_ASSERT_EQUAL(var->genotypes.i8[1], 1);
    CU_ASSERT_EQUAL(var->genotypes.i8[2], 1);
    CU_ASSERT_EQUAL(var->genotypes.i8[3], 1);
    CU_ASSERT_EQUAL(var->num_alleles, 2);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[0], "0", 1);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[1], "1", 1);
    CU_ASSERT_EQUAL(var->site->id, 2);
    CU_ASSERT_EQUAL(var->site->mutations_length, 4);

    ret = tsk_vargen_next(&vargen, &var);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_vargen_free(&vargen);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tsk_treeseq_free(&ts);
}

static void
test_single_tree_non_samples(void)
{
    int ret = 0;
    tsk_treeseq_t ts;
    tsk_vargen_t vargen;
    tsk_variant_t *var;
    /* Non sample internal nodes we want to generate genotypes for */
    tsk_id_t samples[] = {4, 5};

    tsk_treeseq_from_text(&ts, 1, single_tree_ex_nodes, single_tree_ex_edges, NULL,
            single_tree_ex_sites, single_tree_ex_mutations, NULL, NULL);
    /* It's an error to hand in non-samples without imputation turned on */
    ret = tsk_vargen_init(&vargen, &ts, samples, 2, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_MUST_IMPUTE_NON_SAMPLES);
    tsk_vargen_free(&vargen);

    ret = tsk_vargen_init(&vargen, &ts, samples, 2, NULL, TSK_IMPUTE_MISSING_DATA);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tsk_vargen_print_state(&vargen, _devnull);

    ret = tsk_vargen_next(&vargen, &var);
    CU_ASSERT_EQUAL_FATAL(ret, 1);
    CU_ASSERT_EQUAL(var->genotypes.i8[0], 0);
    CU_ASSERT_EQUAL(var->genotypes.i8[1], 0);
    CU_ASSERT_EQUAL(var->num_alleles, 2);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[0], "0", 1);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[1], "1", 1);
    CU_ASSERT_EQUAL(var->site->id, 0);
    CU_ASSERT_EQUAL(var->site->mutations_length, 1);

    ret = tsk_vargen_next(&vargen, &var);
    CU_ASSERT_EQUAL_FATAL(ret, 1);
    CU_ASSERT_EQUAL(var->genotypes.i8[1], 0);
    CU_ASSERT_EQUAL(var->genotypes.i8[0], 1);
    CU_ASSERT_EQUAL(var->num_alleles, 2);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[0], "0", 1);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[1], "1", 1);
    CU_ASSERT_EQUAL(var->site->id, 1);
    CU_ASSERT_EQUAL(var->site->mutations_length, 2);

    ret = tsk_vargen_next(&vargen, &var);
    CU_ASSERT_EQUAL_FATAL(ret, 1);
    CU_ASSERT_EQUAL(var->genotypes.i8[0], 0);
    CU_ASSERT_EQUAL(var->genotypes.i8[1], 0);
    CU_ASSERT_EQUAL(var->num_alleles, 2);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[0], "0", 1);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[1], "1", 1);
    CU_ASSERT_EQUAL(var->site->id, 2);
    CU_ASSERT_EQUAL(var->site->mutations_length, 4);

    ret = tsk_vargen_next(&vargen, &var);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_vargen_free(&vargen);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tsk_treeseq_free(&ts);
}

static void
test_single_tree_errors(void)
{
    int ret;
    tsk_treeseq_t ts;
    tsk_vargen_t vargen;
    tsk_id_t samples[] = {0, 3};

    tsk_treeseq_from_text(&ts, 1, single_tree_ex_nodes, single_tree_ex_edges, NULL,
            single_tree_ex_sites, single_tree_ex_mutations, NULL, NULL);
    ret = tsk_vargen_init(&vargen, &ts, samples, 2, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tsk_vargen_free(&vargen);

    samples[0] = -1;
    ret = tsk_vargen_init(&vargen, &ts, samples, 2, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_OUT_OF_BOUNDS);
    tsk_vargen_free(&vargen);

    samples[0] = 7;
    ret = tsk_vargen_init(&vargen, &ts, samples, 2, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_OUT_OF_BOUNDS);
    tsk_vargen_free(&vargen);

    samples[0] = 3;
    ret = tsk_vargen_init(&vargen, &ts, samples, 2, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_DUPLICATE_SAMPLE);
    tsk_vargen_free(&vargen);

    tsk_treeseq_free(&ts);
}

static void
test_single_tree_user_alleles_errors(void)
{
    int ret;
    tsk_treeseq_t ts;
    tsk_vargen_t vargen;
    tsk_variant_t *var;
    int j;
    /* The maximium number of alleles is 127. We need space for one more plus the
     * sentinel */
    const int max_alleles = 129;
    const char * acct_alleles[] = {"A", "C", "G", "T", NULL};
    const char * zero_allele[] = {"0", NULL};
    const char * no_alleles[] = {NULL};
    const char * many_alleles[max_alleles];
    tsk_id_t samples[] = {0, 3};

    tsk_treeseq_from_text(&ts, 1, single_tree_ex_nodes, single_tree_ex_edges, NULL,
            single_tree_ex_sites, single_tree_ex_mutations, NULL, NULL);

    /* these are 0/1 alleles */
    ret = tsk_vargen_init(&vargen, &ts, samples, 2, acct_alleles, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_vargen_next(&vargen, &var);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_ALLELE_NOT_FOUND);
    tsk_vargen_free(&vargen);

    /* pass just the 0 allele alleles at all */
    ret = tsk_vargen_init(&vargen, &ts, samples, 2, zero_allele, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_vargen_next(&vargen, &var);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_ALLELE_NOT_FOUND);
    tsk_vargen_free(&vargen);

    /* Empty allele list is an error */
    ret = tsk_vargen_init(&vargen, &ts, samples, 2, no_alleles, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_ZERO_ALLELES);
    tsk_vargen_free(&vargen);

    for (j = 0; j < max_alleles; j++) {
        many_alleles[j] = "0";
    }
    many_alleles[128] = NULL;
    ret = tsk_vargen_init(&vargen, &ts, samples, 2, many_alleles, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_TOO_MANY_ALLELES);
    tsk_vargen_free(&vargen);

    tsk_treeseq_free(&ts);
}

static void
test_single_tree_subsample(void)
{
    int ret = 0;
    tsk_treeseq_t ts;
    tsk_vargen_t vargen;
    tsk_variant_t *var;
    tsk_id_t samples[] = {0, 3};

    tsk_treeseq_from_text(&ts, 1, single_tree_ex_nodes, single_tree_ex_edges, NULL,
            single_tree_ex_sites, single_tree_ex_mutations, NULL, NULL);
    ret = tsk_vargen_init(&vargen, &ts, samples, 2, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tsk_vargen_print_state(&vargen, _devnull);

    ret = tsk_vargen_next(&vargen, &var);
    CU_ASSERT_EQUAL_FATAL(ret, 1);
    CU_ASSERT_EQUAL(var->genotypes.i8[0], 0);
    CU_ASSERT_EQUAL(var->genotypes.i8[1], 0);
    CU_ASSERT_EQUAL(var->num_alleles, 2);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[0], "0", 1);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[1], "1", 1);
    CU_ASSERT_EQUAL(var->site->id, 0);
    CU_ASSERT_EQUAL(var->site->mutations_length, 1);

    ret = tsk_vargen_next(&vargen, &var);
    CU_ASSERT_EQUAL_FATAL(ret, 1);
    CU_ASSERT_EQUAL(var->genotypes.i8[0], 0);
    CU_ASSERT_EQUAL(var->genotypes.i8[1], 0);
    CU_ASSERT_EQUAL(var->num_alleles, 2);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[0], "0", 1);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[1], "1", 1);
    CU_ASSERT_EQUAL(var->site->id, 1);
    CU_ASSERT_EQUAL(var->site->mutations_length, 2);

    ret = tsk_vargen_next(&vargen, &var);
    CU_ASSERT_EQUAL_FATAL(ret, 1);
    CU_ASSERT_EQUAL(var->genotypes.i8[0], 1);
    CU_ASSERT_EQUAL(var->genotypes.i8[1], 1);
    CU_ASSERT_EQUAL(var->num_alleles, 2);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[0], "0", 1);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[1], "1", 1);
    CU_ASSERT_EQUAL(var->site->id, 2);
    CU_ASSERT_EQUAL(var->site->mutations_length, 4);

    ret = tsk_vargen_next(&vargen, &var);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_vargen_free(&vargen);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* Zero samples */
    ret = tsk_vargen_init(&vargen, &ts, samples, 0, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tsk_vargen_print_state(&vargen, _devnull);

    ret = tsk_vargen_next(&vargen, &var);
    CU_ASSERT_EQUAL_FATAL(ret, 1);
    CU_ASSERT_EQUAL(var->num_alleles, 2);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[0], "0", 1);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[1], "1", 1);
    CU_ASSERT_EQUAL(var->site->id, 0);
    CU_ASSERT_EQUAL(var->site->mutations_length, 1);

    ret = tsk_vargen_next(&vargen, &var);
    CU_ASSERT_EQUAL_FATAL(ret, 1);
    CU_ASSERT_EQUAL(var->num_alleles, 2);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[0], "0", 1);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[1], "1", 1);
    CU_ASSERT_EQUAL(var->site->id, 1);
    CU_ASSERT_EQUAL(var->site->mutations_length, 2);

    ret = tsk_vargen_next(&vargen, &var);
    CU_ASSERT_EQUAL_FATAL(ret, 1);
    CU_ASSERT_EQUAL(var->num_alleles, 2);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[0], "0", 1);
    CU_ASSERT_NSTRING_EQUAL(var->alleles[1], "1", 1);
    CU_ASSERT_EQUAL(var->site->id, 2);
    CU_ASSERT_EQUAL(var->site->mutations_length, 4);

    ret = tsk_vargen_next(&vargen, &var);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_vargen_free(&vargen);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    tsk_treeseq_free(&ts);
}

static void
test_single_tree_many_alleles(void)
{
    int ret = 0;
    tsk_treeseq_t ts;
    tsk_vargen_t vargen;
    tsk_variant_t *var;
    tsk_size_t num_alleles = 257;
    tsk_id_t j, k, l;
    tsk_flags_t options;
    char alleles[num_alleles];
    tsk_table_collection_t tables;

    tsk_treeseq_from_text(&ts, 1, single_tree_ex_nodes, single_tree_ex_edges, NULL,
            NULL, NULL, NULL, NULL);
    ret = tsk_treeseq_copy_tables(&ts, &tables, 0);
    CU_ASSERT_FATAL(ret == 0);
    tsk_treeseq_free(&ts);
    memset(alleles, 'X', (size_t) num_alleles);
    ret = tsk_site_table_add_row(&tables.sites, 0, "Y", 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);

    /* Add j mutations over a single node. */
    for (j = 0; j < (tsk_id_t) num_alleles; j++) {
        /* When j = 0 we get a parent of -1, which is the NULL_NODE */
        ret = tsk_mutation_table_add_row(&tables.mutations, 0, 0, j - 1, alleles,
                (tsk_size_t) j, NULL, 0);
        CU_ASSERT_FATAL(ret >= 0);
        ret = tsk_treeseq_init(&ts, &tables, TSK_BUILD_INDEXES);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        for (l = 0; l < 2; l++) {
            options = 0;
            if (l == 1) {
                options = TSK_16_BIT_GENOTYPES;
            }
            ret = tsk_vargen_init(&vargen, &ts, NULL, 0, NULL, options);
            CU_ASSERT_EQUAL_FATAL(ret, 0);
            tsk_vargen_print_state(&vargen, _devnull);
            ret = tsk_vargen_next(&vargen, &var);
            /* We have j + 2 alleles. So, if j >= 126, we should fail with 8bit
             * genotypes */
            if (l == 0 && j >= 126) {
                CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_TOO_MANY_ALLELES);
            } else {
                CU_ASSERT_EQUAL_FATAL(ret, 1);
                CU_ASSERT_NSTRING_EQUAL(var->alleles[0], "Y", 1);
                for (k = 1; k < (tsk_id_t) var->num_alleles; k++) {
                    CU_ASSERT_EQUAL(k - 1, (tsk_id_t) var->allele_lengths[k]);
                    CU_ASSERT_NSTRING_EQUAL(var->alleles[k], alleles, var->allele_lengths[k]);
                }
                CU_ASSERT_EQUAL(var->num_alleles, (tsk_size_t) j + 2);
            }
            ret = tsk_vargen_free(&vargen);
            CU_ASSERT_EQUAL_FATAL(ret, 0);
        }
        tsk_treeseq_free(&ts);
    }
    tsk_table_collection_free(&tables);
}

static void
test_single_tree_inconsistent_mutations(void)
{
    const char *sites =
        "0.0     0\n"
        "0.1     0\n"
        "0.2     0\n";
    const char *mutations =
        "0    0     1\n"
        "1    1     1\n"
        "2    4     1\n"
        "2    0     1\n";
    tsk_treeseq_t ts;
    tsk_variant_t *var;
    tsk_vargen_t vargen;
    tsk_flags_t options[] = {0, TSK_16_BIT_GENOTYPES};
    tsk_id_t all_samples[] = {0, 1, 2, 3};
    tsk_id_t *samples[] = {NULL, all_samples};
    size_t num_samples = 4;
    size_t s, f;
    int ret;

    tsk_treeseq_from_text(&ts, 1, single_tree_ex_nodes, single_tree_ex_edges, NULL,
            sites, mutations, NULL, NULL);

    for (s = 0; s < 2; s++) {
        for (f = 0; f < sizeof(options) / sizeof(*options); f++) {
            ret = tsk_vargen_init(&vargen, &ts, samples[s], num_samples, NULL, options[f]);
            CU_ASSERT_EQUAL_FATAL(ret, 0);
            ret = tsk_vargen_next(&vargen, &var);
            CU_ASSERT_EQUAL_FATAL(ret, 1);
            ret = tsk_vargen_next(&vargen, &var);
            CU_ASSERT_EQUAL_FATAL(ret, 1);
            ret = tsk_vargen_next(&vargen, &var);
            CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_INCONSISTENT_MUTATIONS);
            tsk_vargen_free(&vargen);
        }
    }

    tsk_treeseq_free(&ts);
}

int
main(int argc, char **argv)
{
    CU_TestInfo tests[] = {
        {"test_simplest_missing_data", test_simplest_missing_data},
        {"test_simplest_missing_data_user_alleles", test_simplest_missing_data_user_alleles},
        {"test_single_tree_user_alleles", test_single_tree_user_alleles},
        {"test_single_tree_char_alphabet", test_single_tree_char_alphabet},
        {"test_single_tree_binary_alphabet", test_single_tree_binary_alphabet},
        {"test_single_tree_non_samples", test_single_tree_non_samples},
        {"test_single_tree_errors", test_single_tree_errors},
        {"test_single_tree_user_alleles_errors", test_single_tree_user_alleles_errors},
        {"test_single_tree_subsample", test_single_tree_subsample},
        {"test_single_tree_many_alleles", test_single_tree_many_alleles},
        {"test_single_tree_inconsistent_mutations", test_single_tree_inconsistent_mutations},
        {NULL, NULL},
    };

    return test_main(tests, argc, argv);
}
