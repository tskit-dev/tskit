#include <err.h>
#include <stdlib.h>
#include <tskit/tables.h>
#include <tskit/trees.h>

#define check_tsk_error(val)                                                            \
    if (val < 0) {                                                                      \
        errx(EXIT_FAILURE, "line %d: %s", __LINE__, tsk_strerror(val));                 \
    }

int
main(int argc, char **argv)
{
    tsk_table_collection_t *tables;
    tsk_treeseq_t treeseq;
    int rv;

    tables = malloc(sizeof(*tables));
    rv = tsk_table_collection_init(tables, 0);
    check_tsk_error(rv);

    /* NOTE: you must set sequence length AFTER initialization */
    tables->sequence_length = 1.0;

    /* Do your regular table operations */
    rv = tsk_node_table_add_row(&tables->nodes, 0, 0.0, -1, -1, NULL, 0);
    check_tsk_error(rv);

    /* Initalize the tree sequence, transferring all responsibility
     * for the table collection's memory managment
     */
    rv = tsk_treeseq_init(
        &treeseq, tables, TSK_TS_INIT_BUILD_INDEXES | TSK_TAKE_OWNERSHIP);
    check_tsk_error(rv);

    /* WARNING: calling tsk_table_collection_free is now a memory error! */
    tsk_treeseq_free(&treeseq);
}
