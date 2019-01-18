#include <stdio.h>
#include <stdlib.h>
#include <err.h>

#include <tskit.h>

#define check_tsk_error(val) if (val < 0) {\
    errx(EXIT_FAILURE, "line %d: %s", __LINE__, tsk_strerror(val));\
}

int
main(int argc, char **argv)
{
    int ret, iter;
    tsk_treeseq_t ts;
    tsk_tree_t tree;

    if (argc != 2) {
        errx(EXIT_FAILURE, "usage: <tree sequence file>");
    }
    ret = tsk_treeseq_load(&ts, argv[1], 0);
    check_tsk_error(ret);
    ret = tsk_tree_init(&tree, &ts, 0);
    check_tsk_error(ret);

    printf("Iterate forwards\n");
    for (iter = tsk_tree_first(&tree); iter == 1; iter = tsk_tree_next(&tree)) {
        printf("\ttree %d has %d roots\n", tsk_tree_get_index(&tree),
                tsk_tree_get_num_roots(&tree));
    }
    check_tsk_error(iter);

    printf("Iterate backwards\n");
    for (iter = tsk_tree_last(&tree); iter == 1; iter = tsk_tree_prev(&tree)) {
        printf("\ttree %d has %d roots\n", tsk_tree_get_index(&tree),
                tsk_tree_get_num_roots(&tree));
    }
    check_tsk_error(iter);

    tsk_tree_free(&tree);
    tsk_treeseq_free(&ts);
    return 0;
}
