#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <err.h>

#include <tskit/tables.h>

#define check_tsk_error(val)                                                            \
    if (val < 0) {                                                                      \
        errx(EXIT_FAILURE, "line %d: %s", __LINE__, tsk_strerror(val));                 \
    }

typedef struct {
    double left;
    double right;
    double parent_birth_time;
    tsk_id_t parent;
    tsk_id_t child;
} birth;

int
cmp_birth(const void *lhs, const void *rhs)
{
    const birth *clhs = (const birth *) lhs;
    const birth *crhs = (const birth *) rhs;
    int ret = (clhs->parent_birth_time > crhs->parent_birth_time)
              - (crhs->parent_birth_time > clhs->parent_birth_time);
    if (ret == 0) {
        ret = (clhs->parent > crhs->parent) - (crhs->parent > clhs->parent);
    }
    return ret;
}

typedef struct {
    birth *births;
    tsk_size_t capacity;
    tsk_size_t size;
} edge_buffer;

int
edge_buffer_init(edge_buffer *buffer, tsk_size_t initial_capacity)
{
    int ret = 0;
    if (initial_capacity == 0) {
        ret = -1;
        goto out;
    }
    buffer->births = (birth *) malloc(initial_capacity * sizeof(birth));
    buffer->capacity = initial_capacity;
    buffer->size = 0;
out:
    return ret;
}

void
edge_buffer_realloc(edge_buffer *buffer)
{
    if (buffer->size + 1 >= buffer->capacity) {
        buffer->capacity *= 2;
        buffer->births
            = (birth *) realloc(buffer->births, buffer->capacity * sizeof(birth));
    }
}

void
edge_buffer_free(edge_buffer *buffer)
{
    if (buffer->births != NULL) {
        free(buffer->births);
        buffer->births = NULL;
    }
}

void
edge_buffer_buffer_birth(double left, double right, double parent_birth_time,
    tsk_id_t parent, tsk_id_t child, edge_buffer *buffer)
{
    edge_buffer_realloc(buffer);
    buffer->births[buffer->size].left = left;
    buffer->births[buffer->size].right = right;
    buffer->births[buffer->size].parent_birth_time = parent_birth_time;
    buffer->births[buffer->size].parent = parent;
    buffer->births[buffer->size].child = child;
    buffer->size += 1;
}

void
edge_buffer_prep_for_simplification(edge_buffer *buffer)
{
    qsort(buffer->births, (size_t) buffer->size, sizeof(birth), cmp_birth);
}

void
edge_buffer_clear(edge_buffer *buffer)
{
    buffer->size = 0;
}

void
simulate(tsk_table_collection_t *tables, int N, int T, int simplify_interval)
{
    tsk_id_t *buffer, *parents, *children, child, left_parent, right_parent;
    double breakpoint;
    int ret, j, t, b;
    edge_buffer new_births;
    tsk_modular_simplifier_t simplifier;

    assert(simplify_interval != 0); // leads to division by zero
    ret = edge_buffer_init(&new_births, 1000);
    assert(ret == 0);
    buffer = malloc(2 * N * sizeof(tsk_id_t));
    if (buffer == NULL) {
        errx(EXIT_FAILURE, "Out of memory");
    }
    tables->sequence_length = 1.0;
    parents = buffer;
    for (j = 0; j < N; j++) {
        parents[j]
            = tsk_node_table_add_row(&tables->nodes, 0, T, TSK_NULL, TSK_NULL, NULL, 0);
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
                &tables->nodes, 0, t, TSK_NULL, TSK_NULL, NULL, 0);
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
            edge_buffer_buffer_birth(0., breakpoint, tables->nodes.time[left_parent],
                left_parent, child, &new_births);
            edge_buffer_buffer_birth(breakpoint, 1.0, tables->nodes.time[right_parent],
                right_parent, child, &new_births);
            // ret = tsk_edge_table_add_row(
            //    &tables->edges, 0, breakpoint, left_parent, child, NULL, 0);
            // check_tsk_error(ret);
            // ret = tsk_edge_table_add_row(
            //    &tables->edges, breakpoint, 1, right_parent, child, NULL, 0);
            // check_tsk_error(ret);
            children[j] = child;
        }
        if (t % simplify_interval == 0) {
            printf("Simplify at generation %lld: (%lld nodes %lld edges)", (long long) t,
                (long long) tables->nodes.num_rows, (long long) tables->edges.num_rows);
            edge_buffer_prep_for_simplification(&new_births);
            ret = tsk_modular_simplifier_init(&simplifier, tables, children, N, 0);
            check_tsk_error(ret);
            j = 0;
            while (j < new_births.size) {
                left_parent = new_births.births[j].parent;
                while (
                    j < new_births.size && new_births.births[j].parent == left_parent) {
                    ret = tsk_modular_simplifier_add_edge(&simplifier,
                        new_births.births[j].left, new_births.births[j].right,
                        new_births.births[j].parent, new_births.births[j].child);

                    check_tsk_error(ret);
                    j++;
                }
                ret = tsk_modular_simplifier_merge_ancestors(&simplifier, left_parent);
                check_tsk_error(ret);
            }
            ret = tsk_modular_simplifier_finalise(&simplifier, NULL);
            check_tsk_error(ret);
            ret = tsk_modular_simplifier_free(&simplifier);
            check_tsk_error(ret);
            /* Note: Edges must be sorted for simplify to work, and we use a brute force
             * approach of sorting each time here for simplicity. This is inefficient. */
            // ret = tsk_table_collection_sort(tables, NULL, 0);
            // check_tsk_error(ret);
            // ret = tsk_table_collection_simplify(tables, children, N, 0, NULL);
            // check_tsk_error(ret);
            printf(" -> (%lld nodes %lld edges)\n", (long long) tables->nodes.num_rows,
                (long long) tables->edges.num_rows);
            for (j = 0; j < N; j++) {
                children[j] = j;
            }
            edge_buffer_clear(&new_births);
        }
    }
    free(buffer);
    edge_buffer_free(&new_births);
}

int
main(int argc, char **argv)
{
    int ret;
    tsk_table_collection_t tables;

    if (argc != 6) {
        errx(EXIT_FAILURE, "usage: N T simplify-interval output-file seed");
    }
    ret = tsk_table_collection_init(&tables, 0);
    check_tsk_error(ret);
    srand((unsigned) atoi(argv[5]));
    simulate(&tables, atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
    ret = tsk_table_collection_dump(&tables, argv[4], 0);
    check_tsk_error(ret);

    tsk_table_collection_free(&tables);
    return 0;
}
