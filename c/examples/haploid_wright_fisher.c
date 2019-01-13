
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <gsl/gsl_rng.h>

#include "tsk_tables.h"

/* TODO:
 * Add command line args to get N, T and simplify interval
 * Add some error checking.
 */

void
simulate(tsk_tbl_collection_t *tables, int N, int T, int simplify_interval, gsl_rng *rng)
{
    tsk_id_t *buffer[2], *parents, *children, child, left_parent, right_parent;
    double breakpoint;
    int j, t, b;

    assert(T % simplify_interval == 0);
    buffer[0] = malloc(N * sizeof(tsk_id_t));
    buffer[1] = malloc(N * sizeof(tsk_id_t));
    b = 0;
    parents = buffer[b];

    tables->sequence_length = 1.0;
    for (j = 0; j < N; j++) {
        parents[j] = tsk_node_tbl_add_row(tables->nodes, 1, T,
                TSK_NULL, TSK_NULL, NULL, 0);
    }
    for (t = 0; t < T; t++) {
        children = buffer[(b + 1) % 2];
        for (j = 0; j < N; j++) {
            child = tsk_node_tbl_add_row(tables->nodes, 1, T - t - 1,
                TSK_NULL, TSK_NULL, NULL, 0);
            left_parent = parents[gsl_rng_uniform_int(rng, N)];
            right_parent = parents[gsl_rng_uniform_int(rng, N)];
            do {
                breakpoint = gsl_rng_uniform(rng);
            } while (breakpoint == 0); /* tiny proba of breakpoint being 0 */
            tsk_edge_tbl_add_row(tables->edges, 0, breakpoint, left_parent, child);
            tsk_edge_tbl_add_row(tables->edges, breakpoint, 1, right_parent, child);
            children[j] = child;
        }
        parents = children;
        b = (b + 1) % 2;
        if (((t + 1) % simplify_interval) == 0) {
            tsk_tbl_collection_sort(tables, 0, 0); /* FIXME; should take position. */
            tsk_tbl_collection_simplify(tables, parents, N, 0, NULL);
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
    tsk_tbl_collection_t tables;
    gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);

    tsk_tbl_collection_alloc(&tables, 0);
    simulate(&tables, 10, 10, 10, rng);
    tsk_tbl_collection_dump(&tables, "tmp.trees", 0);
    tsk_tbl_collection_free(&tables);
    gsl_rng_free(rng);

    return 0;
}
