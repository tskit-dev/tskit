#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <err.h>
#include <string.h>

#include <pthread.h>
#include <tskit/tables.h>

#define check_tsk_error(val)                                                            \
    if (val < 0) {                                                                      \
        errx(EXIT_FAILURE, "line %d: %s\n", __LINE__, tsk_strerror(val));               \
    }

static void
init_tables(tsk_table_collection_t *tcs, int num_chunks)
{
    int j, ret;

    for (j = 0; j < num_chunks; j++) {
        ret = tsk_table_collection_init(&tcs[j], 0);
        check_tsk_error(ret);
        if (j > 0) {
            tsk_node_table_free(&tcs[j].nodes);
        }
    }
}

static void
free_tables(tsk_table_collection_t *tcs, int num_chunks)
{
    int j;

    for (j = 0; j < num_chunks; j++) {
        if (j > 0) {
            /* Must not double free node table columns. */
            memset(&tcs[j].nodes, 0, sizeof(tcs[j].nodes));
        }
        tsk_table_collection_free(&tcs[j]);
    }
}

static void
join_tables(tsk_table_collection_t *tcs, int num_chunks)
{
    int j, ret;

    for (j = 1; j < num_chunks; j++) {
        ret = tsk_edge_table_extend(
            &tcs[0].edges, &tcs[j].edges, tcs[j].edges.num_rows, NULL, 0);
        check_tsk_error(ret);
    }
    /* Get all the squashable edges next to each other */
    ret = tsk_table_collection_sort(&tcs[0], NULL, 0);
    check_tsk_error(ret);
    ret = tsk_edge_table_squash(&tcs[0].edges);
    check_tsk_error(ret);
    /* We need to sort again after squash */
    ret = tsk_table_collection_sort(&tcs[0], NULL, 0);
    check_tsk_error(ret);
    ret = tsk_table_collection_build_index(&tcs[0], 0);
    check_tsk_error(ret);
}

struct chunk_work {
    int chunk;
    tsk_table_collection_t *tc;
    int *samples;
    int N;
};

void *
simplify_chunk(void *arg)
{
    int ret;
    struct chunk_work *work = (struct chunk_work *) arg;
    tsk_size_t edges_before = work->tc->edges.num_rows;

    ret = tsk_table_collection_sort(work->tc, NULL, 0);
    check_tsk_error(ret);
    ret = tsk_table_collection_simplify(work->tc, work->samples, work->N,
        TSK_SIMPLIFY_NO_FILTER_NODES | TSK_SIMPLIFY_NO_UPDATE_SAMPLE_FLAGS, NULL);
    check_tsk_error(ret);
    /* NOTE: this printf makes helgrind complain */
    printf("\tchunk %d: %lld -> %lld\n", work->chunk, (long long) edges_before,
        (long long) work->tc->edges.num_rows);

    return NULL;
}

void
sort_and_simplify_all(tsk_table_collection_t *tcs, int num_chunks, int *samples, int N)
{
    int j, ret;
    struct chunk_work work[num_chunks];
    pthread_t threads[num_chunks];

    for (j = 1; j < num_chunks; j++) {
        tcs[j].nodes = tcs[0].nodes;
    }

    for (j = 0; j < num_chunks; j++) {
        work[j].chunk = j;
        work[j].tc = &tcs[j];
        work[j].samples = samples;
        work[j].N = N;

        ret = pthread_create(&threads[j], NULL, simplify_chunk, (void *) &work[j]);
        if (ret != 0) {
            errx(EXIT_FAILURE, "Pthread create failed");
        }
        /* simplify_chunk((void *) &work[j]); */
    }
    for (j = 0; j < num_chunks; j++) {
        ret = pthread_join(threads[j], NULL);
        if (ret != 0) {
            errx(EXIT_FAILURE, "Pthread join failed");
        }
    }
}

void
simplify_tables(tsk_table_collection_t *tcs, int num_chunks, int *samples, int N)
{
    int j, k, ret;
    const tsk_size_t num_nodes = tcs[0].nodes.num_rows;
    bool *delete_nodes = malloc(num_nodes * sizeof(*delete_nodes));
    tsk_id_t *node_id_map = malloc(num_nodes * sizeof(*node_id_map));
    tsk_id_t *edge_child, *edge_parent;

    if (delete_nodes == NULL || node_id_map == NULL) {
        errx(EXIT_FAILURE, "Out of memory");
    }

    printf("Simplify %lld nodes\n", (long long) tcs[0].nodes.num_rows);
    sort_and_simplify_all(tcs, num_chunks, samples, N);

    for (j = 0; j < num_nodes; j++) {
        delete_nodes[j] = true;
    }
    for (j = 0; j < N; j++) {
        delete_nodes[samples[j]] = false;
    }

    for (j = 0; j < num_chunks; j++) {
        edge_child = tcs[j].edges.child;
        edge_parent = tcs[j].edges.parent;
        for (k = 0; k < tcs[j].edges.num_rows; k++) {
            delete_nodes[edge_child[k]] = false;
            delete_nodes[edge_parent[k]] = false;
        }
    }
    tsk_node_table_delete_rows(&tcs[0].nodes, delete_nodes, 0, node_id_map);
    printf("\tdone: %lld nodes\n", (long long) tcs[0].nodes.num_rows);

    /* Remap node references */
    for (j = 0; j < num_chunks; j++) {
        edge_child = tcs[j].edges.child;
        edge_parent = tcs[j].edges.parent;
        for (k = 0; k < tcs[j].edges.num_rows; k++) {
            edge_child[k] = node_id_map[edge_child[k]];
            edge_parent[k] = node_id_map[edge_parent[k]];
        }
        ret = tsk_table_collection_check_integrity(&tcs[j], 0);
        check_tsk_error(ret);
    }
    for (j = 0; j < N; j++) {
        samples[j] = node_id_map[samples[j]];
    }
    free(delete_nodes);
    free(node_id_map);
}

void
simulate(
    tsk_table_collection_t *tcs, int num_chunks, int N, int T, int simplify_interval)
{
    tsk_id_t *buffer, *parents, *children, child, left_parent, right_parent;
    double breakpoint, chunk_left, chunk_right;
    int ret, j, t, b, k;

    assert(simplify_interval != 0); // leads to division by zero
    buffer = malloc(2 * N * sizeof(tsk_id_t));
    if (buffer == NULL) {
        errx(EXIT_FAILURE, "Out of memory");
    }
    for (k = 0; k < num_chunks; k++) {
        tcs[k].sequence_length = 1.0;
    }
    parents = buffer;
    for (j = 0; j < N; j++) {
        parents[j]
            = tsk_node_table_add_row(&tcs[0].nodes, 0, T, TSK_NULL, TSK_NULL, NULL, 0);
        check_tsk_error(parents[j]);
    }
    b = 0;
    for (t = T - 1; t >= 0; t--) {
        /* Alternate between using the first and last N values in the buffer */
        parents = buffer + (b * N);
        b = (b + 1) % 2;
        children = buffer + (b * N);
        for (j = 0; j < N; j++) {
            child = tsk_node_table_add_row(
                &tcs[0].nodes, 0, t, TSK_NULL, TSK_NULL, NULL, 0);
            check_tsk_error(child);
            /* NOTE: the use of rand() is discouraged for
             * research code and proper random number generator
             * libraries should be preferred.
             */
            left_parent = parents[(size_t)((rand() / (1. + RAND_MAX)) * N)];
            right_parent = parents[(size_t)((rand() / (1. + RAND_MAX)) * N)];
            do {
                breakpoint = rand() / (1. + RAND_MAX);
            } while (breakpoint == 0); /* tiny proba of breakpoint being 0 */

            chunk_left = 0;
            for (k = 0; k < num_chunks; k++) {
                chunk_right = (k + 1) / (double) num_chunks;
                /* FIXME not certain about the boundaries here, check! */
                if (chunk_right < breakpoint) {
                    ret = tsk_edge_table_add_row(&tcs[k].edges, chunk_left, chunk_right,
                        left_parent, child, NULL, 0);
                    check_tsk_error(ret);
                } else if (breakpoint <= chunk_left) {
                    ret = tsk_edge_table_add_row(&tcs[k].edges, chunk_left, chunk_right,
                        right_parent, child, NULL, 0);
                    check_tsk_error(ret);
                } else {
                    assert(chunk_left < breakpoint && breakpoint < chunk_right);
                    /* Breakpoint is within this chunk so need two edges */
                    ret = tsk_edge_table_add_row(&tcs[k].edges, chunk_left, breakpoint,
                        left_parent, child, NULL, 0);
                    check_tsk_error(ret);
                    ret = tsk_edge_table_add_row(&tcs[k].edges, breakpoint, chunk_right,
                        right_parent, child, NULL, 0);
                    check_tsk_error(ret);
                }
                chunk_left = chunk_right;
            }
            children[j] = child;
        }
        if (t % simplify_interval == 0) {
            simplify_tables(tcs, num_chunks, children, N);
        }
    }
    /* Set the sample flags for final generation */
    for (j = 0; j < N; j++) {
        tcs[0].nodes.flags[children[j]] = TSK_NODE_IS_SAMPLE;
    }
    free(buffer);
}

int
main(int argc, char **argv)
{
    int ret;
    const int num_chunks = 8;
    tsk_table_collection_t tcs[num_chunks];

    if (argc != 6) {
        errx(EXIT_FAILURE, "usage: N T simplify-interval output seed");
    }
    srand((unsigned) atoi(argv[5]));
    init_tables(tcs, num_chunks);
    simulate(tcs, num_chunks, atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
    join_tables(tcs, num_chunks);
    ret = tsk_table_collection_dump(&tcs[0], argv[4], 0);
    check_tsk_error(ret);
    free_tables(tcs, num_chunks);

    return 0;
}
