#include <cstdio>
#include <cstdlib>
#include <err.h>

#include <gsl/gsl_rng.h>

#include <memory>
#include <functional>
#include <tskitpp/TableCollection.hpp>

using GSLrng = std::unique_ptr<gsl_rng, std::function<void(gsl_rng *)>>;

GSLrng
make_rng(unsigned seed)
{
    GSLrng rv(gsl_rng_alloc(gsl_rng_mt19937),
              [](gsl_rng *r) { gsl_rng_free(r); });
    return rv;
}

#define check_error(val)                                                      \
    if (val < 0)                                                              \
        {                                                                     \
            errx(EXIT_FAILURE, "line %d: %s", __LINE__, tsk_strerror(val));   \
        }

void
simulate(tsk_table_collection_t *tables, int N, int T, int simplify_interval,
         gsl_rng *rng)
{
    // TODO: unique_ptrs for the buffers
    //tsk_id_t *buffer[2], *parents, *children,
    tsk_id_t child, left_parent, right_parent;
    double breakpoint;
    int ret, j, t, b;

    std::unique_ptr<tsk_id_t[]> parents(new tsk_id_t[N]),
        children(new tsk_id_t[N]);

    tables->sequence_length = 1.0;
    for (j = 0; j < N; j++)
        {
            parents[j]
                = tsk_node_table_add_row(tables->nodes, TSK_NODE_IS_SAMPLE, T,
                                         TSK_NULL, TSK_NULL, NULL, 0);
            check_error(parents[j]);
        }
    for (t = T - 1; t >= 0; t--)
        {
            for (j = 0; j < N; j++)
                {
                    child = tsk_node_table_add_row(
                        tables->nodes, TSK_NODE_IS_SAMPLE, t, TSK_NULL,
                        TSK_NULL, NULL, 0);
                    check_error(child);
                    left_parent = parents[gsl_rng_uniform_int(rng, N)];
                    right_parent = parents[gsl_rng_uniform_int(rng, N)];
                    do
                        {
                            breakpoint = gsl_rng_uniform(rng);
                        }
                    /* tiny proba of breakpoint being 0 */
                    while (breakpoint == 0);
                    ret = tsk_edge_table_add_row(tables->edges, 0, breakpoint,
                                                 left_parent, child);
                    check_error(ret);
                    ret = tsk_edge_table_add_row(tables->edges, breakpoint, 1,
                                                 right_parent, child);
                    check_error(ret);
                    children[j] = child;
                }
            children.swap(parents);
            if (t % simplify_interval == 0)
                {
                    ret = tsk_table_collection_sort(
                        tables, 0, 0); /* FIXME; should take position. */
                    check_error(ret);
                    ret = tsk_table_collection_simplify(tables, parents.get(),
                                                        N, 0, NULL);
                    check_error(ret);
                    for (j = 0; j < N; j++)
                        {
                            parents[j] = j;
                        }
                }
        }
}

int
main(int argc, char **argv)
{
    tskit::TableCollection tables;
    auto rng = make_rng(42);

    if (argc != 5)
        {
            errx(EXIT_FAILURE, "usage: N T simplify-interval output-file");
        }
    tsk_table_collection_alloc(tables.get(), 0);
    simulate(tables.get(), atoi(argv[1]), atoi(argv[2]), atoi(argv[3]),
             rng.get());
    tsk_table_collection_dump(tables.get(), "tmp.trees", 0);

    return 0;
}
