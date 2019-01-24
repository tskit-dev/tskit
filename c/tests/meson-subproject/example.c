/* Simple example testing that we compile and link in tskit and kastore
 * when we use meson submodules.
 */
#include <stdio.h>
#include <tskit.h>
#include <assert.h>
#include <string.h>

void
test_kas_strerror()
{
    printf("test_kas_strerror\n");
    const char *str = kas_strerror(KAS_ERR_NO_MEMORY);
    assert(strcmp(str, "Out of memory") == 0);
}

void
test_strerror()
{
    printf("test_strerror\n");
    const char *str = tsk_strerror(TSK_ERR_NO_MEMORY);
    assert(strcmp(str, "Out of memory") == 0);
}

void
test_load_error()
{
    printf("test_open_error\n");
    tsk_treeseq_t ts;
    int ret = tsk_treeseq_load(&ts, "no such file", 0);
    assert(tsk_is_kas_error(ret));
    tsk_treeseq_free(&ts);
}

void
test_table_basics()
{
    printf("test_table_basics\n");
    tsk_table_collection_t tables;
    int ret = tsk_table_collection_init(&tables, 0);
    assert(ret == 0);

    ret = tsk_node_table_add_row(&tables.nodes, 0, 1.0, TSK_NULL, TSK_NULL, NULL, 0);
    assert(ret == 0);
    ret = tsk_node_table_add_row(&tables.nodes, 0, 2.0, TSK_NULL, TSK_NULL, NULL, 0);
    assert(ret == 1);
    assert(tables.nodes.num_rows == 2);

    tsk_table_collection_free(&tables);
}

int main()
{
    test_kas_strerror();
    test_strerror();
    test_load_error();
    test_table_basics();
    return 0;
}
