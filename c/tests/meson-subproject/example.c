/*
 * MIT License
 *
 * Copyright (c) 2019-2022 Tskit Developers
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

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
    assert(strcmp(str, "Out of memory. (TSK_ERR_NO_MEMORY)") == 0);
}

void
test_load_error()
{
    printf("test_open_error\n");
    tsk_treeseq_t ts;
    int ret = tsk_treeseq_load(&ts, "no such file", 0);
    assert(ret == TSK_ERR_IO);
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

int
main()
{
    test_kas_strerror();
    test_strerror();
    test_load_error();
    test_table_basics();
    return 0;
}
