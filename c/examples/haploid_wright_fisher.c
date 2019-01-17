#include <stdio.h>
#include <stdlib.h>
#include <err.h>

#include <gsl/gsl_rng.h>

#include <tskit/tables.h>


#define check_error(val) if (val < 0) {\
    errx(EXIT_FAILURE, "line %d: %s", __LINE__, tsk_strerror(val));\
}

void
simulate(tsk_table_collection_t *tables, int N, int T, int simplify_interval, gsl_rng *rng)
{
    tsk_id_t *buffer[2], *parents, *children, child, left_parent, right_parent;
    double breakpoint;
    int ret, j, t, b;

    buffer[0] = malloc(N * sizeof(tsk_id_t)); /* TODO change to 1 malloc and check value */
    buffer[1] = malloc(N * sizeof(tsk_id_t));
    b = 0;
    parents = buffer[b];

    tables->sequence_length = 1.0;
    for (j = 0; j < N; j++) {
        parents[j] = tsk_node_table_add_row(tables->nodes, TSK_NODE_IS_SAMPLE, T,
                TSK_NULL, TSK_NULL, NULL, 0);
        check_error(parents[j]);
    }
    for (t = T - 1; t >= 0; t--) {
        children = buffer[(b + 1) % 2];
        for (j = 0; j < N; j++) {
            child = tsk_node_table_add_row(tables->nodes, TSK_NODE_IS_SAMPLE, t,
                    TSK_NULL, TSK_NULL, NULL, 0);
            check_error(child);
            left_parent = parents[gsl_rng_uniform_int(rng, N)];
            right_parent = parents[gsl_rng_uniform_int(rng, N)];
            do {
                breakpoint = gsl_rng_uniform(rng);
            } while (breakpoint == 0); /* tiny proba of breakpoint being 0 */
            ret = tsk_edge_table_add_row(tables->edges, 0, breakpoint, left_parent, child);
            check_error(ret);
            ret = tsk_edge_table_add_row(tables->edges, breakpoint, 1, right_parent, child);
            check_error(ret);
            children[j] = child;
        }
        parents = children;
        b = (b + 1) % 2;
        if (t % simplify_interval == 0) {
            ret = tsk_table_collection_sort(tables, 0, 0); /* FIXME; should take position. */
            check_error(ret);
            ret = tsk_table_collection_simplify(tables, parents, N, 0, NULL);
            check_error(ret);
            for (j = 0; j < N; j++) {
                parents[j] = j;
            }
        }
    }
    free(buffer[0]);
    free(buffer[1]);
}

int
main(int argc, char **argv)
{
    tsk_table_collection_t tables;
    gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);

    if (argc != 5) {
        errx(EXIT_FAILURE, "usage: N T simplify-interval output-file");
    }
    tsk_table_collection_alloc(&tables, 0);
    simulate(&tables, atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), rng);
    tsk_table_collection_dump(&tables, "tmp.trees", 0);
    tsk_table_collection_free(&tables);
    gsl_rng_free(rng);

    return 0;
}
