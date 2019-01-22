/* Minimal tests to make sure that tskit at least compiles and links
 * in a simple C++ program */

#include <iostream>
#include <assert.h>
#include <sstream>

#include <tskit.h>

using namespace std;

void
test_kas_strerror()
{
    std::cout << "test_kas_strerror" << endl;
    std::ostringstream o;
    o << kas_strerror(KAS_ERR_NO_MEMORY);
    assert(std::string("Out of memory").compare(o.str()) == 0);
}

void
test_strerror()
{
    std::cout << "test_strerror" << endl;
    std::ostringstream o;
    o << tsk_strerror(TSK_ERR_NO_MEMORY);
    assert(std::string("Out of memory").compare(o.str()) == 0);
}

void
test_load_error()
{
    std::cout << "test_open_error" << endl;
    tsk_treeseq_t ts;
    int ret = tsk_treeseq_load(&ts, "no such file", 0);
    assert(tsk_is_kas_error(ret));
    tsk_treeseq_free(&ts);
}

void
test_table_basics()
{
    std::cout << "test_table_basics" << endl;
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
