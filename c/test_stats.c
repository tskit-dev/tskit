#include "testlib.h"
#include "tsk_stats.h"

#include <unistd.h>
#include <stdlib.h>
#include <float.h>

static bool
multi_mutations_exist(tsk_treeseq_t *ts, size_t start, size_t end)
{
    int ret;
    size_t j;
    tsk_site_t site;

    for (j = start; j < TSK_MIN(tsk_treeseq_get_num_sites(ts), end); j++) {
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
    size_t num_sites = tsk_treeseq_get_num_sites(ts);
    tsk_site_t *sites = malloc(num_sites * sizeof(tsk_site_t));
    int *num_site_mutations = malloc(num_sites * sizeof(int));
    tsk_ld_calc_t ld_calc;
    double *r2, *r2_prime, x;
    size_t j, num_r2_values;
    double eps = 1e-6;

    r2 = calloc(num_sites, sizeof(double));
    r2_prime = calloc(num_sites, sizeof(double));
    CU_ASSERT_FATAL(r2 != NULL);
    CU_ASSERT_FATAL(r2_prime != NULL);
    CU_ASSERT_FATAL(sites != NULL);
    CU_ASSERT_FATAL(num_site_mutations != NULL);

    ret = tsk_ld_calc_alloc(&ld_calc, ts);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tsk_ld_calc_print_state(&ld_calc, _devnull);

    for (j = 0; j < num_sites; j++) {
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
        ret = tsk_ld_calc_get_r2_array(&ld_calc, 0, TSK_DIR_FORWARD,
                num_sites, DBL_MAX, r2, &num_r2_values);
        if (multi_mutations_exist(ts, 0, num_sites)) {
            CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_ONLY_INFINITE_SITES);
        } else {
            CU_ASSERT_EQUAL_FATAL(ret, 0);
            CU_ASSERT_EQUAL_FATAL(num_r2_values, num_sites - 1);
        }
        tsk_ld_calc_print_state(&ld_calc, _devnull);

        ret = tsk_ld_calc_get_r2_array(&ld_calc, num_sites - 2, TSK_DIR_FORWARD,
                num_sites, DBL_MAX, r2_prime, &num_r2_values);
        if (multi_mutations_exist(ts, num_sites - 2, num_sites)) {
            CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_ONLY_INFINITE_SITES);
        } else {
            CU_ASSERT_EQUAL_FATAL(ret, 0);
            CU_ASSERT_EQUAL_FATAL(num_r2_values, 1);
        }
        tsk_ld_calc_print_state(&ld_calc, _devnull);

        ret = tsk_ld_calc_get_r2_array(&ld_calc, 0, TSK_DIR_FORWARD,
                num_sites, DBL_MAX, r2_prime, &num_r2_values);
        if (multi_mutations_exist(ts, 0, num_sites)) {
            CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_ONLY_INFINITE_SITES);
        } else {
            CU_ASSERT_EQUAL_FATAL(ret, 0);
            CU_ASSERT_EQUAL_FATAL(num_r2_values, num_sites - 1);
            tsk_ld_calc_print_state(&ld_calc, _devnull);
            for (j = 0; j < num_r2_values; j++) {
                CU_ASSERT_EQUAL_FATAL(r2[j], r2_prime[j]);
                ret = tsk_ld_calc_get_r2(&ld_calc, 0, j + 1, &x);
                CU_ASSERT_EQUAL_FATAL(ret, 0);
                CU_ASSERT_DOUBLE_EQUAL_FATAL(r2[j], x, eps);
            }

        }

        /* Some checks in the reverse direction */
        ret = tsk_ld_calc_get_r2_array(&ld_calc, num_sites - 1,
                TSK_DIR_REVERSE, num_sites, DBL_MAX,
                r2, &num_r2_values);
        if (multi_mutations_exist(ts, 0, num_sites)) {
            CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_ONLY_INFINITE_SITES);
        } else {
            CU_ASSERT_EQUAL_FATAL(ret, 0);
            CU_ASSERT_EQUAL_FATAL(num_r2_values, num_sites - 1);
        }
        tsk_ld_calc_print_state(&ld_calc, _devnull);

        ret = tsk_ld_calc_get_r2_array(&ld_calc, 1, TSK_DIR_REVERSE,
                num_sites, DBL_MAX, r2_prime, &num_r2_values);
        if (multi_mutations_exist(ts, 0, 2)) {
            CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_ONLY_INFINITE_SITES);
        } else {
            CU_ASSERT_EQUAL_FATAL(ret, 0);
            CU_ASSERT_EQUAL_FATAL(num_r2_values, 1);
        }
        tsk_ld_calc_print_state(&ld_calc, _devnull);

        ret = tsk_ld_calc_get_r2_array(&ld_calc, num_sites - 1,
                TSK_DIR_REVERSE, num_sites, DBL_MAX,
                r2_prime, &num_r2_values);
        if (multi_mutations_exist(ts, 0, num_sites)) {
            CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_ONLY_INFINITE_SITES);
        } else {
            CU_ASSERT_EQUAL_FATAL(ret, 0);
            CU_ASSERT_EQUAL_FATAL(num_r2_values, num_sites - 1);
            tsk_ld_calc_print_state(&ld_calc, _devnull);

            for (j = 0; j < num_r2_values; j++) {
                CU_ASSERT_EQUAL_FATAL(r2[j], r2_prime[j]);
                ret = tsk_ld_calc_get_r2(&ld_calc, num_sites - 1,
                        num_sites - j - 2, &x);
                CU_ASSERT_EQUAL_FATAL(ret, 0);
                CU_ASSERT_DOUBLE_EQUAL_FATAL(r2[j], x, eps);
            }
        }

        /* Check some error conditions */
        ret = tsk_ld_calc_get_r2_array(&ld_calc, 0, 0, num_sites, DBL_MAX,
            r2, &num_r2_values);
        CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    }

    if (num_sites > 3) {
        /* Check for some basic distance calculations */
        j = num_sites / 2;
        x = sites[j + 1].position - sites[j].position;
        ret = tsk_ld_calc_get_r2_array(&ld_calc, j, TSK_DIR_FORWARD, num_sites,
                x, r2, &num_r2_values);
        if (multi_mutations_exist(ts, j, num_sites)) {
            CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_ONLY_INFINITE_SITES);
        } else {
            CU_ASSERT_EQUAL_FATAL(ret, 0);
            CU_ASSERT_EQUAL_FATAL(num_r2_values, 1);
        }

        x = sites[j].position - sites[j - 1].position;
        ret = tsk_ld_calc_get_r2_array(&ld_calc, j, TSK_DIR_REVERSE, num_sites,
                x, r2, &num_r2_values);
        if (multi_mutations_exist(ts, 0, j + 1)) {
            CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_ONLY_INFINITE_SITES);
        } else {
            CU_ASSERT_EQUAL_FATAL(ret, 0);
            CU_ASSERT_EQUAL_FATAL(num_r2_values, 1);
        }
    }

    /* Check some error conditions */
    for (j = num_sites; j < num_sites + 2; j++) {
        ret = tsk_ld_calc_get_r2_array(&ld_calc, j, TSK_DIR_FORWARD,
                num_sites, DBL_MAX, r2, &num_r2_values);
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

static void
test_single_tree(void)
{
    tsk_treeseq_t ts;

    tsk_treeseq_from_text(&ts, 1, single_tree_ex_nodes, single_tree_ex_edges,
            NULL, single_tree_ex_sites, single_tree_ex_mutations, NULL, NULL);
    verify_ld(&ts);
    tsk_treeseq_free(&ts);
}

static void
test_paper_example(void)
{
    tsk_treeseq_t ts;

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges,
            NULL, paper_ex_sites, paper_ex_mutations, paper_ex_individuals, NULL);
    verify_ld(&ts);
    tsk_treeseq_free(&ts);
}

static void
test_nonbinary_example(void)
{
    tsk_treeseq_t ts;

    tsk_treeseq_from_text(&ts, 100, nonbinary_ex_nodes, nonbinary_ex_edges,
            NULL, nonbinary_ex_sites, nonbinary_ex_mutations, NULL, NULL);
    verify_ld(&ts);
    tsk_treeseq_free(&ts);
}

int
main(int argc, char **argv)
{
    CU_TestInfo tests[] = {
        {"test_single_tree", test_single_tree},
        {"test_paper_example", test_paper_example},
        {"test_nonbinary_example", test_nonbinary_example},
        {NULL, NULL},
    };
    return test_main(tests, argc, argv);
}
