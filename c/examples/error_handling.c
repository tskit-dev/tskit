#include <stdio.h>
#include <stdlib.h>
#include <err.h>

#include "tsk_trees.h"

int
main(int argc, char **argv)
{
    int ret;
    tsk_treeseq_t ts;

    if (argc != 2) {
        errx(EXIT_FAILURE, "usage: <tree sequence file>");
    }
    ret = tsk_treeseq_load(&ts, argv[1], 0);
    if (ret < 0) {
        /* Error condition. Free and exit */
        tsk_treeseq_free(&ts);
        errx(EXIT_FAILURE, "%s", tsk_strerror(ret));
    }
    printf("Loaded tree sequence with %d nodes and %d edges from %s\n",
            tsk_treeseq_get_num_nodes(&ts), tsk_treeseq_get_num_edges(&ts), argv[1]);
    tsk_treeseq_free(&ts);

    return EXIT_SUCCESS;
}
