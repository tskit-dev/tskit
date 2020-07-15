/*
 * MIT License
 *
 * Copyright (c) 2019 Tskit Developers
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

#include "testlib.h"
#include <tskit/tables.h>

#include <unistd.h>
#include <stdlib.h>

static void
reverse_edges(tsk_table_collection_t *tables)
{
    int ret;
    tsk_edge_table_t edges;
    tsk_edge_t edge;
    tsk_id_t j;

    ret = tsk_edge_table_init(&edges, tables->edges.options);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    for (j = (tsk_id_t) tables->edges.num_rows - 1; j >= 0; j--) {
        ret = tsk_edge_table_get_row(&tables->edges, j, &edge);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        ret = tsk_edge_table_add_row(&edges, edge.left, edge.right, edge.parent,
            edge.child, edge.metadata, edge.metadata_length);
        CU_ASSERT_FATAL(ret >= 0);
    }

    ret = tsk_edge_table_copy(&edges, &tables->edges, TSK_NO_INIT);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    tsk_edge_table_free(&edges);
}

static void
insert_edge_metadata(tsk_table_collection_t *tables)
{
    int ret;
    tsk_edge_table_t edges;
    tsk_edge_t edge;
    tsk_id_t j;
    char metadata[100];

    ret = tsk_edge_table_init(&edges, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    for (j = 0; j < (tsk_id_t) tables->edges.num_rows; j++) {
        ret = tsk_edge_table_get_row(&tables->edges, j, &edge);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        snprintf(metadata, sizeof(metadata), "md_%d\n", j);
        ret = tsk_edge_table_add_row(&edges, edge.left, edge.right, edge.parent,
            edge.child, metadata, (tsk_size_t) strlen(metadata));
        CU_ASSERT_FATAL(ret >= 0);
    }
    ret = tsk_edge_table_copy(&edges, &tables->edges, TSK_NO_INIT);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    tsk_edge_table_free(&edges);
}

static void
test_table_collection_simplify_errors(void)
{
    int ret;
    tsk_table_collection_t tables;
    tsk_id_t samples[] = { 0, 1 };

    ret = tsk_table_collection_init(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tables.sequence_length = 1;

    ret = tsk_site_table_add_row(&tables.sites, 0, "A", 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_site_table_add_row(&tables.sites, 0, "A", 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_simplify(&tables, samples, 0, 0, NULL);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_DUPLICATE_SITE_POSITION);

    /* Out of order positions */
    tables.sites.position[0] = 0.5;
    ret = tsk_table_collection_simplify(&tables, samples, 0, 0, NULL);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_UNSORTED_SITES);

    /* Position out of bounds */
    tables.sites.position[0] = 1.5;
    ret = tsk_table_collection_simplify(&tables, samples, 0, 0, NULL);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_SITE_POSITION);

    /* TODO More tests for this: see
     * https://github.com/tskit-dev/msprime/issues/517 */

    tsk_table_collection_free(&tables);
}

static void
test_table_collection_metadata(void)
{
    int ret;
    tsk_table_collection_t tc1, tc2;

    char example_metadata[100] = "An example of metadata with unicode 🎄🌳🌴🌲🎋";
    char example_metadata_schema[100]
        = "An example of metadata schema with unicode 🎄🌳🌴🌲🎋";
    tsk_size_t example_metadata_length = (tsk_size_t) strlen(example_metadata);
    tsk_size_t example_metadata_schema_length
        = (tsk_size_t) strlen(example_metadata_schema);

    // Test equality
    ret = tsk_table_collection_init(&tc1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_init(&tc2, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_equals(&tc1, &tc2));
    ret = tsk_table_collection_set_metadata(
        &tc1, example_metadata, example_metadata_length);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_FALSE(tsk_table_collection_equals(&tc1, &tc2));
    ret = tsk_table_collection_set_metadata(
        &tc2, example_metadata, example_metadata_length);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_equals(&tc1, &tc2));
    ret = tsk_table_collection_set_metadata_schema(
        &tc1, example_metadata_schema, example_metadata_schema_length);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_FALSE(tsk_table_collection_equals(&tc1, &tc2));
    ret = tsk_table_collection_set_metadata_schema(
        &tc2, example_metadata_schema, example_metadata_schema_length);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_equals(&tc1, &tc2));

    // Test copy
    tsk_table_collection_free(&tc1);
    tsk_table_collection_free(&tc2);
    ret = tsk_table_collection_init(&tc1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_set_metadata(
        &tc1, example_metadata, example_metadata_length);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_copy(&tc1, &tc2, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_equals(&tc1, &tc2));

    ret = tsk_table_collection_set_metadata_schema(
        &tc1, example_metadata_schema, example_metadata_schema_length);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tsk_table_collection_free(&tc2);
    ret = tsk_table_collection_copy(&tc1, &tc2, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_equals(&tc1, &tc2));

    // Test dump and load with empty metadata and schema
    tsk_table_collection_free(&tc1);
    tsk_table_collection_free(&tc2);
    ret = tsk_table_collection_init(&tc1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tc1.sequence_length = 1.0;
    ret = tsk_table_collection_dump(&tc1, _tmp_file_name, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_load(&tc2, _tmp_file_name, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_equals(&tc1, &tc2));

    // Test dump and load with set metadata and schema
    tsk_table_collection_free(&tc1);
    tsk_table_collection_free(&tc2);
    ret = tsk_table_collection_init(&tc1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tc1.sequence_length = 1.0;
    ret = tsk_table_collection_set_metadata(
        &tc1, example_metadata, example_metadata_length);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_set_metadata_schema(
        &tc1, example_metadata_schema, example_metadata_schema_length);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_dump(&tc1, _tmp_file_name, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_load(&tc2, _tmp_file_name, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_equals(&tc1, &tc2));
    tsk_table_collection_free(&tc1);
    tsk_table_collection_free(&tc2);
}

static void
test_node_table(void)
{
    int ret;
    tsk_node_table_t table, table2;
    tsk_node_t node;
    uint32_t num_rows = 100;
    tsk_id_t j;
    uint32_t *flags;
    tsk_id_t *population;
    double *time;
    tsk_id_t *individual;
    char *metadata;
    uint32_t *metadata_offset;
    const char *test_metadata = "test";
    tsk_size_t test_metadata_length = 4;
    char metadata_copy[test_metadata_length + 1];

    metadata_copy[test_metadata_length] = '\0';
    ret = tsk_node_table_init(&table, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tsk_node_table_set_max_rows_increment(&table, 1);
    tsk_node_table_set_max_metadata_length_increment(&table, 1);
    tsk_node_table_print_state(&table, _devnull);
    tsk_node_table_dump_text(&table, _devnull);

    for (j = 0; j < (tsk_id_t) num_rows; j++) {
        ret = tsk_node_table_add_row(
            &table, (tsk_flags_t) j, j, j, j, test_metadata, test_metadata_length);
        CU_ASSERT_EQUAL_FATAL(ret, j);
        CU_ASSERT_EQUAL(table.flags[j], (tsk_flags_t) j);
        CU_ASSERT_EQUAL(table.time[j], j);
        CU_ASSERT_EQUAL(table.population[j], j);
        CU_ASSERT_EQUAL(table.individual[j], j);
        CU_ASSERT_EQUAL(table.num_rows, (tsk_size_t) j + 1);
        CU_ASSERT_EQUAL(
            table.metadata_length, (tsk_size_t)(j + 1) * test_metadata_length);
        CU_ASSERT_EQUAL(table.metadata_offset[j + 1], table.metadata_length);
        /* check the metadata */
        memcpy(metadata_copy, table.metadata + table.metadata_offset[j],
            test_metadata_length);
        CU_ASSERT_NSTRING_EQUAL(metadata_copy, test_metadata, test_metadata_length);
        ret = tsk_node_table_get_row(&table, (tsk_id_t) j, &node);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        CU_ASSERT_EQUAL(node.id, j);
        CU_ASSERT_EQUAL(node.flags, (tsk_size_t) j);
        CU_ASSERT_EQUAL(node.time, j);
        CU_ASSERT_EQUAL(node.population, j);
        CU_ASSERT_EQUAL(node.individual, j);
        CU_ASSERT_EQUAL(node.metadata_length, test_metadata_length);
        CU_ASSERT_NSTRING_EQUAL(node.metadata, test_metadata, test_metadata_length);
    }
    CU_ASSERT_EQUAL(tsk_node_table_get_row(&table, (tsk_id_t) num_rows, &node),
        TSK_ERR_NODE_OUT_OF_BOUNDS);
    tsk_node_table_print_state(&table, _devnull);
    tsk_node_table_dump_text(&table, _devnull);

    tsk_node_table_clear(&table);
    CU_ASSERT_EQUAL(table.num_rows, 0);
    CU_ASSERT_EQUAL(table.metadata_length, 0);

    num_rows *= 2;
    flags = malloc(num_rows * sizeof(uint32_t));
    CU_ASSERT_FATAL(flags != NULL);
    memset(flags, 1, num_rows * sizeof(uint32_t));
    population = malloc(num_rows * sizeof(uint32_t));
    CU_ASSERT_FATAL(population != NULL);
    memset(population, 2, num_rows * sizeof(uint32_t));
    time = malloc(num_rows * sizeof(double));
    CU_ASSERT_FATAL(time != NULL);
    memset(time, 0, num_rows * sizeof(double));
    individual = malloc(num_rows * sizeof(uint32_t));
    CU_ASSERT_FATAL(individual != NULL);
    memset(individual, 3, num_rows * sizeof(uint32_t));
    metadata = malloc(num_rows * sizeof(char));
    memset(metadata, 'a', num_rows * sizeof(char));
    CU_ASSERT_FATAL(metadata != NULL);
    metadata_offset = malloc((num_rows + 1) * sizeof(tsk_size_t));
    CU_ASSERT_FATAL(metadata_offset != NULL);
    for (j = 0; j < (tsk_id_t) num_rows + 1; j++) {
        metadata_offset[j] = (tsk_size_t) j;
    }
    ret = tsk_node_table_set_columns(&table, num_rows, flags, time, population,
        individual, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(table.flags, flags, num_rows * sizeof(uint32_t)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.population, population, num_rows * sizeof(uint32_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.time, time, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.individual, individual, num_rows * sizeof(uint32_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.metadata, metadata, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(memcmp(table.metadata_offset, metadata_offset,
                        (num_rows + 1) * sizeof(tsk_size_t)),
        0);
    CU_ASSERT_EQUAL(table.num_rows, num_rows);
    CU_ASSERT_EQUAL(table.metadata_length, num_rows);
    tsk_node_table_print_state(&table, _devnull);
    tsk_node_table_dump_text(&table, _devnull);

    /* Append another num_rows onto the end */
    ret = tsk_node_table_append_columns(&table, num_rows, flags, time, population,
        individual, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(table.flags, flags, num_rows * sizeof(uint32_t)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.flags + num_rows, flags, num_rows * sizeof(uint32_t)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.population, population, num_rows * sizeof(uint32_t)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.population + num_rows, population, num_rows * sizeof(uint32_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.time, time, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(memcmp(table.time + num_rows, time, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.individual, individual, num_rows * sizeof(uint32_t)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.individual + num_rows, individual, num_rows * sizeof(uint32_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.metadata, metadata, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.metadata + num_rows, metadata, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(table.num_rows, 2 * num_rows);
    CU_ASSERT_EQUAL(table.metadata_length, 2 * num_rows);
    tsk_node_table_print_state(&table, _devnull);
    tsk_node_table_dump_text(&table, _devnull);

    /* Truncate back to the original number of rows. */
    ret = tsk_node_table_truncate(&table, num_rows);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(table.flags, flags, num_rows * sizeof(uint32_t)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.population, population, num_rows * sizeof(uint32_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.time, time, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.individual, individual, num_rows * sizeof(uint32_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.metadata, metadata, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(memcmp(table.metadata_offset, metadata_offset,
                        (num_rows + 1) * sizeof(tsk_size_t)),
        0);
    CU_ASSERT_EQUAL(table.num_rows, num_rows);
    CU_ASSERT_EQUAL(table.metadata_length, num_rows);

    ret = tsk_node_table_truncate(&table, num_rows + 1);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_TABLE_POSITION);

    /* If population is NULL it should be set to -1. If metadata is NULL all metadatas
     * should be set to the empty string. If individual is NULL it should be set to -1.
     */
    num_rows = 10;
    memset(population, 0xff, num_rows * sizeof(uint32_t));
    memset(individual, 0xff, num_rows * sizeof(uint32_t));
    ret = tsk_node_table_set_columns(
        &table, num_rows, flags, time, NULL, NULL, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(table.flags, flags, num_rows * sizeof(uint32_t)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.population, population, num_rows * sizeof(uint32_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.time, time, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.individual, individual, num_rows * sizeof(uint32_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.metadata, metadata, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.metadata_offset, metadata_offset, num_rows * sizeof(uint32_t)), 0);
    CU_ASSERT_EQUAL(table.num_rows, num_rows);
    CU_ASSERT_EQUAL(table.metadata_length, num_rows);

    /* flags and time cannot be NULL */
    ret = tsk_node_table_set_columns(
        &table, num_rows, NULL, time, population, individual, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    ret = tsk_node_table_set_columns(&table, num_rows, flags, NULL, population,
        individual, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    ret = tsk_node_table_set_columns(
        &table, num_rows, flags, time, population, individual, NULL, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    ret = tsk_node_table_set_columns(
        &table, num_rows, flags, time, population, individual, metadata, NULL);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);

    /* if metadata and metadata_offset are both null, all metadatas are zero length */
    num_rows = 10;
    memset(metadata_offset, 0, (num_rows + 1) * sizeof(tsk_size_t));
    ret = tsk_node_table_set_columns(
        &table, num_rows, flags, time, NULL, NULL, NULL, NULL);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(table.flags, flags, num_rows * sizeof(uint32_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.time, time, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(memcmp(table.metadata_offset, metadata_offset,
                        (num_rows + 1) * sizeof(tsk_size_t)),
        0);
    CU_ASSERT_EQUAL(table.num_rows, num_rows);
    CU_ASSERT_EQUAL(table.metadata_length, 0);
    ret = tsk_node_table_append_columns(
        &table, num_rows, flags, time, NULL, NULL, NULL, NULL);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(table.flags, flags, num_rows * sizeof(uint32_t)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.flags + num_rows, flags, num_rows * sizeof(uint32_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.time, time, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(memcmp(table.time + num_rows, time, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(memcmp(table.metadata_offset, metadata_offset,
                        (num_rows + 1) * sizeof(tsk_size_t)),
        0);
    CU_ASSERT_EQUAL(memcmp(table.metadata_offset + num_rows, metadata_offset,
                        num_rows * sizeof(uint32_t)),
        0);
    CU_ASSERT_EQUAL(table.num_rows, 2 * num_rows);
    CU_ASSERT_EQUAL(table.metadata_length, 0);
    tsk_node_table_print_state(&table, _devnull);
    tsk_node_table_dump_text(&table, _devnull);

    ret = tsk_node_table_truncate(&table, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(table.metadata_schema_length, 0);
    CU_ASSERT_EQUAL(table.metadata_schema, NULL);
    const char *example = "An example of metadata schema with unicode 🎄🌳🌴🌲🎋";
    tsk_size_t example_length = (tsk_size_t) strlen(example);
    const char *example2 = "A different example 🎄🌳🌴🌲🎋";
    tsk_size_t example2_length = (tsk_size_t) strlen(example);
    tsk_node_table_set_metadata_schema(&table, example, example_length);
    CU_ASSERT_EQUAL(table.metadata_schema_length, example_length);
    CU_ASSERT_EQUAL(memcmp(table.metadata_schema, example, example_length), 0);

    tsk_node_table_copy(&table, &table2, 0);
    CU_ASSERT_EQUAL(table.metadata_schema_length, table2.metadata_schema_length);
    CU_ASSERT_EQUAL(
        memcmp(table.metadata_schema, table2.metadata_schema, example_length), 0);
    tsk_node_table_set_metadata_schema(&table2, example, example_length);
    CU_ASSERT_TRUE(tsk_node_table_equals(&table, &table2));
    tsk_node_table_set_metadata_schema(&table2, example2, example2_length);
    CU_ASSERT_FALSE(tsk_node_table_equals(&table, &table2));

    tsk_node_table_clear(&table);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(table.num_rows, 0);
    CU_ASSERT_EQUAL(table.metadata_length, 0);

    tsk_node_table_free(&table);
    CU_ASSERT_EQUAL(ret, 0);
    tsk_node_table_free(&table2);
    CU_ASSERT_EQUAL(ret, 0);
    free(flags);
    free(population);
    free(time);
    free(metadata);
    free(metadata_offset);
    free(individual);
}

static void
test_edge_table_with_options(tsk_flags_t options)
{
    int ret;
    tsk_edge_table_t table, table2;
    tsk_size_t num_rows = 100;
    tsk_id_t j;
    tsk_edge_t edge;
    tsk_id_t *parent, *child;
    double *left, *right;
    char *metadata;
    uint32_t *metadata_offset;
    const char *test_metadata = "test";
    tsk_size_t test_metadata_length = 4;
    char metadata_copy[test_metadata_length + 1];

    metadata_copy[test_metadata_length] = '\0';
    ret = tsk_edge_table_init(&table, options);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tsk_edge_table_set_max_rows_increment(&table, 1);
    tsk_edge_table_set_max_metadata_length_increment(&table, 1);
    tsk_edge_table_print_state(&table, _devnull);
    tsk_edge_table_dump_text(&table, _devnull);

    for (j = 0; j < (tsk_id_t) num_rows; j++) {
        if (options & TSK_NO_METADATA) {
            ret = tsk_edge_table_add_row(&table, (double) j, (double) j, j, j,
                test_metadata, test_metadata_length);
            CU_ASSERT_EQUAL(ret, TSK_ERR_METADATA_DISABLED);
            ret = tsk_edge_table_add_row(&table, (double) j, (double) j, j, j, NULL, 0);
        } else {
            ret = tsk_edge_table_add_row(&table, (double) j, (double) j, j, j,
                test_metadata, test_metadata_length);
        }
        CU_ASSERT_EQUAL_FATAL(ret, j);
        CU_ASSERT_EQUAL(table.left[j], j);
        CU_ASSERT_EQUAL(table.right[j], j);
        CU_ASSERT_EQUAL(table.parent[j], j);
        CU_ASSERT_EQUAL(table.child[j], j);
        CU_ASSERT_EQUAL(table.num_rows, (tsk_size_t) j + 1);
        if (options & TSK_NO_METADATA) {
            CU_ASSERT_EQUAL(table.metadata_length, 0);
            CU_ASSERT_EQUAL(table.metadata, NULL);
            CU_ASSERT_EQUAL(table.metadata_offset, NULL);
        } else {
            CU_ASSERT_EQUAL(
                table.metadata_length, (tsk_size_t)(j + 1) * test_metadata_length);
            CU_ASSERT_EQUAL(table.metadata_offset[j + 1], table.metadata_length);
            /* check the metadata */
            memcpy(metadata_copy, table.metadata + table.metadata_offset[j],
                test_metadata_length);
            CU_ASSERT_NSTRING_EQUAL(metadata_copy, test_metadata, test_metadata_length);
        }

        ret = tsk_edge_table_get_row(&table, (tsk_id_t) j, &edge);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        CU_ASSERT_EQUAL(edge.id, j);
        CU_ASSERT_EQUAL(edge.left, j);
        CU_ASSERT_EQUAL(edge.right, j);
        CU_ASSERT_EQUAL(edge.parent, j);
        CU_ASSERT_EQUAL(edge.child, j);
        if (options & TSK_NO_METADATA) {
            CU_ASSERT_EQUAL(edge.metadata_length, 0);
            CU_ASSERT_EQUAL(edge.metadata, NULL);
        } else {
            CU_ASSERT_EQUAL(edge.metadata_length, test_metadata_length);
            CU_ASSERT_NSTRING_EQUAL(edge.metadata, test_metadata, test_metadata_length);
        }
    }
    ret = tsk_edge_table_get_row(&table, (tsk_id_t) num_rows, &edge);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_EDGE_OUT_OF_BOUNDS);
    tsk_edge_table_print_state(&table, _devnull);
    tsk_edge_table_dump_text(&table, _devnull);

    num_rows *= 2;
    left = malloc(num_rows * sizeof(double));
    CU_ASSERT_FATAL(left != NULL);
    memset(left, 0, num_rows * sizeof(double));
    right = malloc(num_rows * sizeof(double));
    CU_ASSERT_FATAL(right != NULL);
    memset(right, 0, num_rows * sizeof(double));
    parent = malloc(num_rows * sizeof(tsk_id_t));
    CU_ASSERT_FATAL(parent != NULL);
    memset(parent, 1, num_rows * sizeof(tsk_id_t));
    child = malloc(num_rows * sizeof(tsk_id_t));
    CU_ASSERT_FATAL(child != NULL);
    memset(child, 1, num_rows * sizeof(tsk_id_t));
    metadata = malloc(num_rows * sizeof(char));
    memset(metadata, 'a', num_rows * sizeof(char));
    CU_ASSERT_FATAL(metadata != NULL);
    metadata_offset = malloc((num_rows + 1) * sizeof(tsk_size_t));
    CU_ASSERT_FATAL(metadata_offset != NULL);
    for (j = 0; j < (tsk_id_t) num_rows + 1; j++) {
        metadata_offset[j] = (tsk_size_t) j;
    }
    if (options & TSK_NO_METADATA) {
        ret = tsk_edge_table_set_columns(
            &table, num_rows, left, right, parent, child, metadata, metadata_offset);
        CU_ASSERT_EQUAL(ret, TSK_ERR_METADATA_DISABLED);
        ret = tsk_edge_table_set_columns(
            &table, num_rows, left, right, parent, child, NULL, NULL);
    } else {
        ret = tsk_edge_table_set_columns(
            &table, num_rows, left, right, parent, child, metadata, metadata_offset);
    }
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(table.left, left, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(memcmp(table.right, right, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(memcmp(table.parent, parent, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.child, child, num_rows * sizeof(tsk_id_t)), 0);
    if (options & TSK_NO_METADATA) {
        CU_ASSERT_EQUAL(table.metadata, NULL);
        CU_ASSERT_EQUAL(table.metadata_offset, NULL);
        CU_ASSERT_EQUAL(table.metadata_length, 0);
    } else {
        CU_ASSERT_EQUAL(memcmp(table.metadata, metadata, num_rows * sizeof(char)), 0);
        CU_ASSERT_EQUAL(memcmp(table.metadata_offset, metadata_offset,
                            (num_rows + 1) * sizeof(tsk_size_t)),
            0);
        CU_ASSERT_EQUAL(table.metadata_length, num_rows);
    }

    CU_ASSERT_EQUAL(table.num_rows, num_rows);

    /* Append another num_rows to the end. */
    if (options & TSK_NO_METADATA) {
        ret = tsk_edge_table_append_columns(
            &table, num_rows, left, right, parent, child, metadata, metadata_offset);
        CU_ASSERT_EQUAL(ret, TSK_ERR_METADATA_DISABLED);
        ret = tsk_edge_table_append_columns(
            &table, num_rows, left, right, parent, child, NULL, NULL);
    } else {
        ret = tsk_edge_table_append_columns(
            &table, num_rows, left, right, parent, child, metadata, metadata_offset);
    }
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(table.left, left, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(memcmp(table.left + num_rows, left, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(memcmp(table.right, right, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(memcmp(table.right + num_rows, right, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(memcmp(table.parent, parent, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.parent + num_rows, parent, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.child, child, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.child + num_rows, child, num_rows * sizeof(tsk_id_t)), 0);
    if (options & TSK_NO_METADATA) {
        CU_ASSERT_EQUAL(table.metadata, NULL);
        CU_ASSERT_EQUAL(table.metadata_offset, NULL);
        CU_ASSERT_EQUAL(table.metadata_length, 0);
    } else {
        CU_ASSERT_EQUAL(memcmp(table.metadata, metadata, num_rows * sizeof(char)), 0);
        CU_ASSERT_EQUAL(
            memcmp(table.metadata + num_rows, metadata, num_rows * sizeof(char)), 0);
        CU_ASSERT_EQUAL(table.metadata_length, 2 * num_rows);
    }

    CU_ASSERT_EQUAL(table.num_rows, 2 * num_rows);

    /* Truncate back to num_rows */
    ret = tsk_edge_table_truncate(&table, num_rows);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(table.left, left, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(memcmp(table.right, right, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(memcmp(table.parent, parent, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.child, child, num_rows * sizeof(tsk_id_t)), 0);
    if (options & TSK_NO_METADATA) {
        CU_ASSERT_EQUAL(table.metadata, NULL);
        CU_ASSERT_EQUAL(table.metadata_offset, NULL);
        CU_ASSERT_EQUAL(table.metadata_length, 0);
    } else {
        CU_ASSERT_EQUAL(memcmp(table.metadata, metadata, num_rows * sizeof(char)), 0);
        CU_ASSERT_EQUAL(memcmp(table.metadata_offset, metadata_offset,
                            (num_rows + 1) * sizeof(tsk_size_t)),
            0);
        CU_ASSERT_EQUAL(table.metadata_length, num_rows);
    }
    CU_ASSERT_EQUAL(table.num_rows, num_rows);

    ret = tsk_edge_table_truncate(&table, num_rows + 1);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_TABLE_POSITION);

    /* Inputs cannot be NULL */
    ret = tsk_edge_table_set_columns(
        &table, num_rows, NULL, right, parent, child, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    ret = tsk_edge_table_set_columns(
        &table, num_rows, left, NULL, parent, child, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    ret = tsk_edge_table_set_columns(
        &table, num_rows, left, right, NULL, child, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    ret = tsk_edge_table_set_columns(
        &table, num_rows, left, right, parent, NULL, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    ret = tsk_edge_table_set_columns(
        &table, num_rows, left, right, parent, child, NULL, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    ret = tsk_edge_table_set_columns(
        &table, num_rows, left, right, parent, child, metadata, NULL);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);

    /* if metadata and metadata_offset are both null, all metadatas are zero length */
    num_rows = 10;
    memset(metadata_offset, 0, (num_rows + 1) * sizeof(tsk_size_t));
    ret = tsk_edge_table_set_columns(
        &table, num_rows, left, right, parent, child, NULL, NULL);
    CU_ASSERT_EQUAL(memcmp(table.left, left, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(memcmp(table.right, right, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(memcmp(table.parent, parent, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.child, child, num_rows * sizeof(tsk_id_t)), 0);
    if (options & TSK_NO_METADATA) {
        CU_ASSERT_EQUAL(table.metadata, NULL);
        CU_ASSERT_EQUAL(table.metadata_offset, NULL);
    } else {
        CU_ASSERT_EQUAL(memcmp(table.metadata_offset, metadata_offset,
                            (num_rows + 1) * sizeof(tsk_size_t)),
            0);
    }
    CU_ASSERT_EQUAL(table.metadata_length, 0);
    CU_ASSERT_EQUAL(table.num_rows, num_rows);
    ret = tsk_edge_table_append_columns(
        &table, num_rows, left, right, parent, child, NULL, NULL);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(table.left, left, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(memcmp(table.left + num_rows, left, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(memcmp(table.right, right, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(memcmp(table.right + num_rows, right, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(memcmp(table.parent, parent, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.parent + num_rows, parent, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.child, child, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.child + num_rows, child, num_rows * sizeof(tsk_id_t)), 0);
    if (options & TSK_NO_METADATA) {
        CU_ASSERT_EQUAL(table.metadata, NULL);
        CU_ASSERT_EQUAL(table.metadata_offset, NULL);
    } else {
        CU_ASSERT_EQUAL(memcmp(table.metadata_offset, metadata_offset,
                            (num_rows + 1) * sizeof(tsk_size_t)),
            0);
        CU_ASSERT_EQUAL(memcmp(table.metadata_offset + num_rows, metadata_offset,
                            num_rows * sizeof(uint32_t)),
            0);
    }
    CU_ASSERT_EQUAL(table.metadata_length, 0);
    CU_ASSERT_EQUAL(table.num_rows, 2 * num_rows);
    tsk_edge_table_print_state(&table, _devnull);
    tsk_edge_table_dump_text(&table, _devnull);

    ret = tsk_edge_table_truncate(&table, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(table.metadata_schema_length, 0);
    CU_ASSERT_EQUAL(table.metadata_schema, NULL);
    const char *example = "An example of metadata schema with unicode 🎄🌳🌴🌲🎋";
    tsk_size_t example_length = (tsk_size_t) strlen(example);
    const char *example2 = "A different example 🎄🌳🌴🌲🎋";
    tsk_size_t example2_length = (tsk_size_t) strlen(example);
    ret = tsk_edge_table_set_metadata_schema(&table, example, example_length);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(table.metadata_schema_length, example_length);
    CU_ASSERT_EQUAL(memcmp(table.metadata_schema, example, example_length), 0);

    ret = tsk_edge_table_init(&table2, options);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_edge_table_copy(&table, &table2, TSK_NO_INIT | options);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(table.metadata_schema_length, table2.metadata_schema_length);
    CU_ASSERT_EQUAL(
        memcmp(table.metadata_schema, table2.metadata_schema, example_length), 0);
    ret = tsk_edge_table_set_metadata_schema(&table2, example, example_length);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_edge_table_equals(&table, &table2));
    ret = tsk_edge_table_set_metadata_schema(&table2, example2, example2_length);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_FALSE(tsk_edge_table_equals(&table, &table2));

    ret = tsk_edge_table_clear(&table);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(table.num_rows, 0);
    CU_ASSERT_EQUAL(table.metadata_length, 0);

    ret = tsk_edge_table_free(&table);
    CU_ASSERT_EQUAL(ret, 0);
    ret = tsk_edge_table_free(&table2);
    CU_ASSERT_EQUAL(ret, 0);
    free(left);
    free(right);
    free(parent);
    free(child);
    free(metadata);
    free(metadata_offset);
}

static void
test_edge_table(void)
{
    test_edge_table_with_options(0);
    test_edge_table_with_options(TSK_NO_METADATA);
}

static void
test_edge_table_copy_semantics(void)
{
    int ret;
    tsk_treeseq_t ts;
    tsk_table_collection_t t1, t2;
    tsk_edge_table_t edges;

    tsk_treeseq_from_text(&ts, 1, single_tree_ex_nodes, single_tree_ex_edges, NULL, NULL,
        NULL, NULL, NULL, 0);
    ret = tsk_treeseq_copy_tables(&ts, &t1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    insert_edge_metadata(&t1);

    /* t1 now has metadata. We should be able to copy to another table with metadata */
    ret = tsk_table_collection_copy(&t1, &t2, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_equals(&t1, &t2));
    tsk_table_collection_free(&t2);

    /* We should not be able to copy into a table with no metadata */
    ret = tsk_table_collection_copy(&t1, &t2, TSK_NO_EDGE_METADATA);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_METADATA_DISABLED);
    tsk_table_collection_free(&t2);

    tsk_table_collection_free(&t1);
    ret = tsk_treeseq_copy_tables(&ts, &t1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    /* t1 has no metadata, but metadata is enabled. We should be able to copy
     * into a table with either metadata enabled or disabled.
     */
    ret = tsk_table_collection_copy(&t1, &t2, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_equals(&t1, &t2));
    tsk_table_collection_free(&t2);

    ret = tsk_table_collection_copy(&t1, &t2, TSK_NO_EDGE_METADATA);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_equals(&t1, &t2));
    tsk_table_collection_free(&t2);

    /* Try copying into a table directly */
    ret = tsk_edge_table_copy(&t1.edges, &edges, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_edge_table_equals(&t1.edges, &edges));
    tsk_edge_table_free(&edges);

    tsk_table_collection_free(&t1);
    tsk_treeseq_free(&ts);
}

static void
test_edge_table_squash(void)
{
    int ret;
    tsk_table_collection_t tables;

    const char *nodes_ex = "1  0       -1   -1\n"
                           "1  0       -1   -1\n"
                           "0  0.253   -1   -1\n";
    const char *edges_ex = "0  2   2   0\n"
                           "2  10  2   0\n"
                           "0  2   2   1\n"
                           "2  10  2   1\n";

    /*
      2
     / \
    0   1
    */
    ret = tsk_table_collection_init(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tables.sequence_length = 10;

    parse_nodes(nodes_ex, &tables.nodes);
    CU_ASSERT_EQUAL_FATAL(tables.nodes.num_rows, 3);
    parse_edges(edges_ex, &tables.edges);
    CU_ASSERT_EQUAL_FATAL(tables.edges.num_rows, 4);

    ret = tsk_edge_table_squash(&tables.edges);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    // Check output.
    CU_ASSERT_EQUAL(tables.edges.num_rows, 2);

    // Free things.
    tsk_table_collection_free(&tables);
}

static void
test_edge_table_squash_multiple_parents(void)
{
    int ret;
    tsk_table_collection_t tables;

    const char *nodes_ex = "1  0.000   -1    -1\n"
                           "1  0.000   -1    -1\n"
                           "1  0.000   -1    -1\n"
                           "1  0.000   -1    -1\n"
                           "0  1.000   -1    -1\n"
                           "0  1.000   -1    -1\n";
    const char *edges_ex = "5  10  5   3\n"
                           "5  10  5   2\n"
                           "0  5   5   3\n"
                           "0  5   5   2\n"
                           "4  10  4   1\n"
                           "0  4   4   1\n"
                           "4  10  4   0\n"
                           "0  4   4   0\n";
    /*
                4       5
               / \     / \
              0   1   2   3
    */
    ret = tsk_table_collection_init(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tables.sequence_length = 10;

    parse_nodes(nodes_ex, &tables.nodes);
    CU_ASSERT_EQUAL_FATAL(tables.nodes.num_rows, 6);
    parse_edges(edges_ex, &tables.edges);
    CU_ASSERT_EQUAL_FATAL(tables.edges.num_rows, 8);

    ret = tsk_edge_table_squash(&tables.edges);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    // Check output.
    CU_ASSERT_EQUAL(tables.edges.num_rows, 4);

    // Free things.
    tsk_table_collection_free(&tables);
}

static void
test_edge_table_squash_empty(void)
{
    int ret;
    tsk_table_collection_t tables;

    const char *nodes_ex = "1  0       -1   -1\n"
                           "1  0       -1   -1\n"
                           "0  0.253   -1   -1\n";
    const char *edges_ex = "";

    ret = tsk_table_collection_init(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tables.sequence_length = 10;

    parse_nodes(nodes_ex, &tables.nodes);
    CU_ASSERT_EQUAL_FATAL(tables.nodes.num_rows, 3);
    parse_edges(edges_ex, &tables.edges);
    CU_ASSERT_EQUAL_FATAL(tables.edges.num_rows, 0);

    ret = tsk_edge_table_squash(&tables.edges);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    // Free things.
    tsk_table_collection_free(&tables);
}

static void
test_edge_table_squash_single_edge(void)
{
    int ret;
    tsk_table_collection_t tables;

    const char *nodes_ex = "1  0   -1   -1\n"
                           "0  0   -1   -1\n";
    const char *edges_ex = "0  1   1   0\n";
    ret = tsk_table_collection_init(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tables.sequence_length = 1;

    parse_nodes(nodes_ex, &tables.nodes);
    CU_ASSERT_EQUAL_FATAL(tables.nodes.num_rows, 2);
    parse_edges(edges_ex, &tables.edges);
    CU_ASSERT_EQUAL_FATAL(tables.edges.num_rows, 1);

    ret = tsk_edge_table_squash(&tables.edges);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    // Free things.
    tsk_table_collection_free(&tables);
}

static void
test_edge_table_squash_bad_intervals(void)
{
    int ret;
    tsk_table_collection_t tables;

    const char *nodes_ex = "1  0   -1   -1\n"
                           "0  0   -1   -1\n";
    const char *edges_ex = "0  0.6   1   0\n"
                           "0.4  1   1   0\n";

    ret = tsk_table_collection_init(&tables, TSK_NO_EDGE_METADATA);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tables.sequence_length = 1;

    parse_nodes(nodes_ex, &tables.nodes);
    CU_ASSERT_EQUAL_FATAL(tables.nodes.num_rows, 2);
    parse_edges(edges_ex, &tables.edges);
    CU_ASSERT_EQUAL_FATAL(tables.edges.num_rows, 2);

    ret = tsk_edge_table_squash(&tables.edges);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_EDGES_CONTRADICTORY_CHILDREN);

    // Free things.
    tsk_table_collection_free(&tables);
}

static void
test_edge_table_squash_metadata(void)
{
    int ret;
    tsk_table_collection_t tables;

    ret = tsk_table_collection_init(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tables.sequence_length = 10;
    ret = tsk_edge_table_add_row(&tables.edges, 0, 0, 1, 1, "metadata", 8);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_edge_table_squash(&tables.edges);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_CANT_PROCESS_EDGES_WITH_METADATA);

    tsk_table_collection_free(&tables);

    ret = tsk_table_collection_init(&tables, TSK_NO_EDGE_METADATA);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tables.sequence_length = 10;
    ret = tsk_edge_table_add_row(&tables.edges, 0, 0, 1, 1, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_edge_table_squash(&tables.edges);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    tsk_table_collection_free(&tables);
}

static void
test_site_table(void)
{
    int ret;
    tsk_site_table_t table, table2;
    tsk_size_t num_rows, j;
    char *ancestral_state;
    char *metadata;
    double *position;
    tsk_site_t site;
    tsk_size_t *ancestral_state_offset;
    tsk_size_t *metadata_offset;

    ret = tsk_site_table_init(&table, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tsk_site_table_set_max_rows_increment(&table, 1);
    tsk_site_table_set_max_metadata_length_increment(&table, 1);
    tsk_site_table_set_max_ancestral_state_length_increment(&table, 1);
    tsk_site_table_print_state(&table, _devnull);
    tsk_site_table_dump_text(&table, _devnull);

    ret = tsk_site_table_add_row(&table, 0, "A", 1, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(table.position[0], 0);
    CU_ASSERT_EQUAL(table.ancestral_state_offset[0], 0);
    CU_ASSERT_EQUAL(table.ancestral_state_offset[1], 1);
    CU_ASSERT_EQUAL(table.ancestral_state_length, 1);
    CU_ASSERT_EQUAL(table.metadata_offset[0], 0);
    CU_ASSERT_EQUAL(table.metadata_offset[1], 0);
    CU_ASSERT_EQUAL(table.metadata_length, 0);
    CU_ASSERT_EQUAL(table.num_rows, 1);

    ret = tsk_site_table_get_row(&table, 0, &site);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(site.position, 0);
    CU_ASSERT_EQUAL(site.ancestral_state_length, 1);
    CU_ASSERT_NSTRING_EQUAL(site.ancestral_state, "A", 1);
    CU_ASSERT_EQUAL(site.metadata_length, 0);

    ret = tsk_site_table_add_row(&table, 1, "AA", 2, "{}", 2);
    CU_ASSERT_EQUAL_FATAL(ret, 1);
    CU_ASSERT_EQUAL(table.position[1], 1);
    CU_ASSERT_EQUAL(table.ancestral_state_offset[2], 3);
    CU_ASSERT_EQUAL(table.metadata_offset[1], 0);
    CU_ASSERT_EQUAL(table.metadata_offset[2], 2);
    CU_ASSERT_EQUAL(table.metadata_length, 2);
    CU_ASSERT_EQUAL(table.num_rows, 2);

    ret = tsk_site_table_get_row(&table, 1, &site);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(site.position, 1);
    CU_ASSERT_EQUAL(site.ancestral_state_length, 2);
    CU_ASSERT_NSTRING_EQUAL(site.ancestral_state, "AA", 2);
    CU_ASSERT_EQUAL(site.metadata_length, 2);
    CU_ASSERT_NSTRING_EQUAL(site.metadata, "{}", 2);

    ret = tsk_site_table_add_row(&table, 2, "A", 1, "metadata", 8);
    CU_ASSERT_EQUAL_FATAL(ret, 2);
    CU_ASSERT_EQUAL(table.position[1], 1);
    CU_ASSERT_EQUAL(table.ancestral_state_offset[3], 4);
    CU_ASSERT_EQUAL(table.ancestral_state_length, 4);
    CU_ASSERT_EQUAL(table.metadata_offset[3], 10);
    CU_ASSERT_EQUAL(table.metadata_length, 10);
    CU_ASSERT_EQUAL(table.num_rows, 3);

    ret = tsk_site_table_get_row(&table, 3, &site);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_SITE_OUT_OF_BOUNDS);

    tsk_site_table_print_state(&table, _devnull);
    tsk_site_table_dump_text(&table, _devnull);
    tsk_site_table_clear(&table);
    CU_ASSERT_EQUAL(table.num_rows, 0);
    CU_ASSERT_EQUAL(table.ancestral_state_length, 0);
    CU_ASSERT_EQUAL(table.metadata_length, 0);
    CU_ASSERT_EQUAL(table.ancestral_state_offset[0], 0);
    CU_ASSERT_EQUAL(table.metadata_offset[0], 0);

    num_rows = 100;
    position = malloc(num_rows * sizeof(double));
    CU_ASSERT_FATAL(position != NULL);
    ancestral_state = malloc(num_rows * sizeof(char));
    CU_ASSERT_FATAL(ancestral_state != NULL);
    ancestral_state_offset = malloc((num_rows + 1) * sizeof(uint32_t));
    CU_ASSERT_FATAL(ancestral_state_offset != NULL);
    metadata = malloc(num_rows * sizeof(char));
    CU_ASSERT_FATAL(metadata != NULL);
    metadata_offset = malloc((num_rows + 1) * sizeof(uint32_t));
    CU_ASSERT_FATAL(metadata_offset != NULL);

    for (j = 0; j < num_rows; j++) {
        position[j] = (double) j;
        ancestral_state[j] = (char) j;
        ancestral_state_offset[j] = (tsk_size_t) j;
        metadata[j] = (char) ('A' + j);
        metadata_offset[j] = (tsk_size_t) j;
    }
    ancestral_state_offset[num_rows] = num_rows;
    metadata_offset[num_rows] = num_rows;

    ret = tsk_site_table_set_columns(&table, num_rows, position, ancestral_state,
        ancestral_state_offset, metadata, metadata_offset);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(table.position, position, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.ancestral_state, ancestral_state, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(table.ancestral_state_length, num_rows);
    CU_ASSERT_EQUAL(memcmp(table.metadata, metadata, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(table.metadata_length, num_rows);
    CU_ASSERT_EQUAL(table.num_rows, num_rows);

    /* Append another num rows */
    ret = tsk_site_table_append_columns(&table, num_rows, position, ancestral_state,
        ancestral_state_offset, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(table.position, position, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.position + num_rows, position, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.ancestral_state, ancestral_state, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(memcmp(table.ancestral_state + num_rows, ancestral_state,
                        num_rows * sizeof(char)),
        0);
    CU_ASSERT_EQUAL(memcmp(table.metadata, metadata, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.metadata + num_rows, metadata, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(table.num_rows, 2 * num_rows);
    CU_ASSERT_EQUAL(table.ancestral_state_length, 2 * num_rows);

    /* truncate back to num_rows */
    ret = tsk_site_table_truncate(&table, num_rows);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(table.position, position, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.ancestral_state, ancestral_state, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(table.ancestral_state_length, num_rows);
    CU_ASSERT_EQUAL(memcmp(table.metadata, metadata, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(table.metadata_length, num_rows);
    CU_ASSERT_EQUAL(table.num_rows, num_rows);

    ret = tsk_site_table_truncate(&table, num_rows + 1);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_TABLE_POSITION);

    /* Inputs cannot be NULL */
    ret = tsk_site_table_set_columns(&table, num_rows, NULL, ancestral_state,
        ancestral_state_offset, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    ret = tsk_site_table_set_columns(&table, num_rows, position, NULL,
        ancestral_state_offset, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    ret = tsk_site_table_set_columns(
        &table, num_rows, position, ancestral_state, NULL, metadata, metadata_offset);
    /* Metadata and metadata_offset must both be null */
    ret = tsk_site_table_set_columns(&table, num_rows, position, ancestral_state,
        ancestral_state_offset, NULL, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    ret = tsk_site_table_set_columns(&table, num_rows, position, ancestral_state,
        ancestral_state_offset, metadata, NULL);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);

    /* Set metadata to NULL */
    ret = tsk_site_table_set_columns(
        &table, num_rows, position, ancestral_state, ancestral_state_offset, NULL, NULL);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    memset(metadata_offset, 0, (num_rows + 1) * sizeof(uint32_t));
    CU_ASSERT_EQUAL(memcmp(table.position, position, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.ancestral_state, ancestral_state, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(table.ancestral_state_length, num_rows);
    CU_ASSERT_EQUAL(memcmp(table.metadata_offset, metadata_offset,
                        (num_rows + 1) * sizeof(uint32_t)),
        0);
    CU_ASSERT_EQUAL(table.metadata_length, 0);
    CU_ASSERT_EQUAL(table.num_rows, num_rows);

    /* Test for bad offsets */
    ancestral_state_offset[0] = 1;
    ret = tsk_site_table_set_columns(&table, num_rows, position, ancestral_state,
        ancestral_state_offset, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_OFFSET);
    ancestral_state_offset[0] = 0;
    ancestral_state_offset[num_rows] = 0;
    ret = tsk_site_table_set_columns(&table, num_rows, position, ancestral_state,
        ancestral_state_offset, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_OFFSET);
    ancestral_state_offset[0] = 0;

    metadata_offset[0] = 0;
    ret = tsk_site_table_set_columns(&table, num_rows, position, ancestral_state,
        ancestral_state_offset, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_OFFSET);
    metadata_offset[0] = 0;
    metadata_offset[num_rows] = 0;
    ret = tsk_site_table_set_columns(&table, num_rows, position, ancestral_state,
        ancestral_state_offset, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_OFFSET);

    ret = tsk_site_table_truncate(&table, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(table.metadata_schema_length, 0);
    CU_ASSERT_EQUAL(table.metadata_schema, NULL);
    const char *example = "An example of metadata schema with unicode 🎄🌳🌴🌲🎋";
    tsk_size_t example_length = (tsk_size_t) strlen(example);
    const char *example2 = "A different example 🎄🌳🌴🌲🎋";
    tsk_size_t example2_length = (tsk_size_t) strlen(example);
    tsk_site_table_set_metadata_schema(&table, example, example_length);
    CU_ASSERT_EQUAL(table.metadata_schema_length, example_length);
    CU_ASSERT_EQUAL(memcmp(table.metadata_schema, example, example_length), 0);

    tsk_site_table_copy(&table, &table2, 0);
    CU_ASSERT_EQUAL(table.metadata_schema_length, table2.metadata_schema_length);
    CU_ASSERT_EQUAL(
        memcmp(table.metadata_schema, table2.metadata_schema, example_length), 0);
    tsk_site_table_set_metadata_schema(&table2, example, example_length);
    CU_ASSERT_TRUE(tsk_site_table_equals(&table, &table2));
    tsk_site_table_set_metadata_schema(&table2, example2, example2_length);
    CU_ASSERT_FALSE(tsk_site_table_equals(&table, &table2));

    ret = tsk_site_table_clear(&table);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(table.num_rows, 0);
    CU_ASSERT_EQUAL(table.ancestral_state_length, 0);
    CU_ASSERT_EQUAL(table.metadata_length, 0);

    tsk_site_table_free(&table);
    CU_ASSERT_EQUAL(ret, 0);
    tsk_site_table_free(&table2);
    CU_ASSERT_EQUAL(ret, 0);

    free(position);
    free(ancestral_state);
    free(ancestral_state_offset);
    free(metadata);
    free(metadata_offset);
}

static void
test_mutation_table(void)
{
    int ret;
    tsk_mutation_table_t table, table2;
    tsk_size_t num_rows = 100;
    tsk_size_t max_len = 20;
    tsk_size_t k, len;
    tsk_id_t j;
    tsk_id_t *node;
    tsk_id_t *parent;
    tsk_id_t *site;
    double *time;
    char *derived_state, *metadata;
    char c[max_len + 1];
    tsk_size_t *derived_state_offset, *metadata_offset;
    tsk_mutation_t mutation;

    for (j = 0; j < (tsk_id_t) max_len; j++) {
        c[j] = (char) ('A' + j);
    }

    ret = tsk_mutation_table_init(&table, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tsk_mutation_table_set_max_rows_increment(&table, 1);
    tsk_mutation_table_set_max_metadata_length_increment(&table, 1);
    tsk_mutation_table_set_max_derived_state_length_increment(&table, 1);
    tsk_mutation_table_print_state(&table, _devnull);
    tsk_mutation_table_dump_text(&table, _devnull);

    len = 0;
    for (j = 0; j < (tsk_id_t) num_rows; j++) {
        k = TSK_MIN((tsk_size_t) j + 1, max_len);
        ret = tsk_mutation_table_add_row(&table, j, j, j, j, c, k, c, k);
        CU_ASSERT_EQUAL_FATAL(ret, j);
        CU_ASSERT_EQUAL(table.site[j], j);
        CU_ASSERT_EQUAL(table.node[j], j);
        CU_ASSERT_EQUAL(table.parent[j], j);
        CU_ASSERT_EQUAL(table.time[j], j);
        CU_ASSERT_EQUAL(table.derived_state_offset[j], len);
        CU_ASSERT_EQUAL(table.metadata_offset[j], len);
        CU_ASSERT_EQUAL(table.num_rows, (tsk_size_t) j + 1);
        len += k;
        CU_ASSERT_EQUAL(table.derived_state_offset[j + 1], len);
        CU_ASSERT_EQUAL(table.derived_state_length, len);
        CU_ASSERT_EQUAL(table.metadata_offset[j + 1], len);
        CU_ASSERT_EQUAL(table.metadata_length, len);

        ret = tsk_mutation_table_get_row(&table, (tsk_id_t) j, &mutation);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        CU_ASSERT_EQUAL(mutation.id, j);
        CU_ASSERT_EQUAL(mutation.site, j);
        CU_ASSERT_EQUAL(mutation.node, j);
        CU_ASSERT_EQUAL(mutation.parent, j);
        CU_ASSERT_EQUAL(mutation.time, j);
        CU_ASSERT_EQUAL(mutation.metadata_length, k);
        CU_ASSERT_NSTRING_EQUAL(mutation.metadata, c, k);
        CU_ASSERT_EQUAL(mutation.derived_state_length, k);
        CU_ASSERT_NSTRING_EQUAL(mutation.derived_state, c, k);
    }
    ret = tsk_mutation_table_get_row(&table, (tsk_id_t) num_rows, &mutation);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_MUTATION_OUT_OF_BOUNDS);
    tsk_mutation_table_print_state(&table, _devnull);
    tsk_mutation_table_dump_text(&table, _devnull);

    num_rows *= 2;
    site = malloc(num_rows * sizeof(tsk_id_t));
    CU_ASSERT_FATAL(site != NULL);
    node = malloc(num_rows * sizeof(tsk_id_t));
    CU_ASSERT_FATAL(node != NULL);
    parent = malloc(num_rows * sizeof(tsk_id_t));
    CU_ASSERT_FATAL(parent != NULL);
    time = malloc(num_rows * sizeof(double));
    CU_ASSERT_FATAL(time != NULL);
    derived_state = malloc(num_rows * sizeof(char));
    CU_ASSERT_FATAL(derived_state != NULL);
    derived_state_offset = malloc((num_rows + 1) * sizeof(tsk_size_t));
    CU_ASSERT_FATAL(derived_state_offset != NULL);
    metadata = malloc(num_rows * sizeof(char));
    CU_ASSERT_FATAL(metadata != NULL);
    metadata_offset = malloc((num_rows + 1) * sizeof(tsk_size_t));
    CU_ASSERT_FATAL(metadata_offset != NULL);

    for (j = 0; j < (tsk_id_t) num_rows; j++) {
        node[j] = j;
        site[j] = j + 1;
        parent[j] = j + 2;
        time[j] = j + 3;
        derived_state[j] = 'Y';
        derived_state_offset[j] = (tsk_size_t) j;
        metadata[j] = 'M';
        metadata_offset[j] = (tsk_size_t) j;
    }

    derived_state_offset[num_rows] = num_rows;
    metadata_offset[num_rows] = num_rows;
    ret = tsk_mutation_table_set_columns(&table, num_rows, site, node, parent, time,
        derived_state, derived_state_offset, metadata, metadata_offset);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(table.site, site, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.node, node, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.parent, parent, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.time, time, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.derived_state, derived_state, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(memcmp(table.metadata, metadata, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(table.num_rows, num_rows);
    CU_ASSERT_EQUAL(table.derived_state_length, num_rows);
    CU_ASSERT_EQUAL(table.metadata_length, num_rows);

    /* Append another num_rows */
    ret = tsk_mutation_table_append_columns(&table, num_rows, site, node, parent, time,
        derived_state, derived_state_offset, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(table.site, site, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.site + num_rows, site, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.node, node, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.node + num_rows, node, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.parent, parent, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.parent + num_rows, parent, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.time, time, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(memcmp(table.time + num_rows, time, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.derived_state, derived_state, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.derived_state, derived_state, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(table.derived_state_length, 2 * num_rows);
    CU_ASSERT_EQUAL(memcmp(table.metadata, metadata, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(memcmp(table.metadata, metadata, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(table.metadata_length, 2 * num_rows);
    CU_ASSERT_EQUAL(table.num_rows, 2 * num_rows);

    /* Truncate back to num_rows */
    ret = tsk_mutation_table_truncate(&table, num_rows);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(table.site, site, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.node, node, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.parent, parent, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.time, time, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.derived_state, derived_state, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(memcmp(table.metadata, metadata, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(table.num_rows, num_rows);
    CU_ASSERT_EQUAL(table.derived_state_length, num_rows);
    CU_ASSERT_EQUAL(table.metadata_length, num_rows);

    ret = tsk_mutation_table_truncate(&table, num_rows + 1);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_TABLE_POSITION);

    /* Check all this again, except with parent == NULL, time == NULL
     * and metadata == NULL. */
    memset(parent, 0xff, num_rows * sizeof(tsk_id_t));
    for (j = 0; j < (tsk_id_t) num_rows; j++) {
        time[j] = TSK_UNKNOWN_TIME;
    }
    memset(metadata_offset, 0, (num_rows + 1) * sizeof(tsk_size_t));
    ret = tsk_mutation_table_set_columns(&table, num_rows, site, node, NULL, NULL,
        derived_state, derived_state_offset, NULL, NULL);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(table.site, site, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.node, node, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.parent, parent, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.time, time, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.derived_state, derived_state, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(memcmp(table.derived_state_offset, derived_state_offset,
                        num_rows * sizeof(tsk_size_t)),
        0);
    CU_ASSERT_EQUAL(memcmp(table.metadata_offset, metadata_offset,
                        (num_rows + 1) * sizeof(tsk_size_t)),
        0);
    CU_ASSERT_EQUAL(table.num_rows, num_rows);
    CU_ASSERT_EQUAL(table.derived_state_length, num_rows);
    CU_ASSERT_EQUAL(table.metadata_length, 0);

    /* Append another num_rows */
    ret = tsk_mutation_table_append_columns(&table, num_rows, site, node, NULL, NULL,
        derived_state, derived_state_offset, NULL, NULL);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(table.site, site, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.site + num_rows, site, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.node, node, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.node + num_rows, node, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.parent, parent, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.parent + num_rows, parent, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.time, time, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(memcmp(table.time + num_rows, time, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.derived_state, derived_state, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.derived_state + num_rows, derived_state, num_rows * sizeof(char)),
        0);
    CU_ASSERT_EQUAL(table.num_rows, 2 * num_rows);
    CU_ASSERT_EQUAL(table.derived_state_length, 2 * num_rows);
    CU_ASSERT_EQUAL(table.metadata_length, 0);

    /* Inputs except parent, time, metadata and metadata_offset cannot be NULL*/
    ret = tsk_mutation_table_set_columns(&table, num_rows, NULL, node, parent, time,
        derived_state, derived_state_offset, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    ret = tsk_mutation_table_set_columns(&table, num_rows, site, NULL, parent, time,
        derived_state, derived_state_offset, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    ret = tsk_mutation_table_set_columns(&table, num_rows, site, node, parent, time,
        NULL, derived_state_offset, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    ret = tsk_mutation_table_set_columns(&table, num_rows, site, node, parent, time,
        derived_state, NULL, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    ret = tsk_mutation_table_set_columns(&table, num_rows, site, node, parent, time,
        derived_state, derived_state_offset, NULL, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    ret = tsk_mutation_table_set_columns(&table, num_rows, site, node, parent, time,
        derived_state, derived_state_offset, metadata, NULL);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);

    /* Inputs except parent, time, metadata and metadata_offset cannot be NULL*/
    ret = tsk_mutation_table_append_columns(&table, num_rows, NULL, node, parent, time,
        derived_state, derived_state_offset, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    ret = tsk_mutation_table_append_columns(&table, num_rows, site, NULL, parent, time,
        derived_state, derived_state_offset, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    ret = tsk_mutation_table_append_columns(&table, num_rows, site, node, parent, time,
        NULL, derived_state_offset, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    ret = tsk_mutation_table_append_columns(&table, num_rows, site, node, parent, time,
        derived_state, NULL, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    ret = tsk_mutation_table_append_columns(&table, num_rows, site, node, parent, time,
        derived_state, derived_state_offset, NULL, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    ret = tsk_mutation_table_append_columns(&table, num_rows, site, node, parent, time,
        derived_state, derived_state_offset, metadata, NULL);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);

    /* Test for bad offsets */
    derived_state_offset[0] = 1;
    ret = tsk_mutation_table_set_columns(&table, num_rows, site, node, parent, time,
        derived_state, derived_state_offset, NULL, NULL);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_OFFSET);
    derived_state_offset[0] = 0;
    derived_state_offset[num_rows] = 0;
    ret = tsk_mutation_table_set_columns(&table, num_rows, site, node, parent, time,
        derived_state, derived_state_offset, NULL, NULL);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_OFFSET);

    ret = tsk_mutation_table_truncate(&table, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(table.metadata_schema_length, 0);
    CU_ASSERT_EQUAL(table.metadata_schema, NULL);
    const char *example = "An example of metadata schema with unicode 🎄🌳🌴🌲🎋";
    tsk_size_t example_length = (tsk_size_t) strlen(example);
    const char *example2 = "A different example 🎄🌳🌴🌲🎋";
    tsk_size_t example2_length = (tsk_size_t) strlen(example);
    tsk_mutation_table_set_metadata_schema(&table, example, example_length);
    CU_ASSERT_EQUAL(table.metadata_schema_length, example_length);
    CU_ASSERT_EQUAL(memcmp(table.metadata_schema, example, example_length), 0);

    tsk_mutation_table_copy(&table, &table2, 0);
    CU_ASSERT_EQUAL(table.metadata_schema_length, table2.metadata_schema_length);
    CU_ASSERT_EQUAL(
        memcmp(table.metadata_schema, table2.metadata_schema, example_length), 0);
    tsk_mutation_table_set_metadata_schema(&table2, example, example_length);
    CU_ASSERT_TRUE(tsk_mutation_table_equals(&table, &table2));
    tsk_mutation_table_set_metadata_schema(&table2, example2, example2_length);
    CU_ASSERT_FALSE(tsk_mutation_table_equals(&table, &table2));

    tsk_mutation_table_clear(&table);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(table.num_rows, 0);
    CU_ASSERT_EQUAL(table.derived_state_length, 0);
    CU_ASSERT_EQUAL(table.metadata_length, 0);

    tsk_mutation_table_free(&table);
    CU_ASSERT_EQUAL(ret, 0);
    tsk_mutation_table_free(&table2);
    CU_ASSERT_EQUAL(ret, 0);
    free(site);
    free(node);
    free(parent);
    free(time);
    free(derived_state);
    free(derived_state_offset);
    free(metadata);
    free(metadata_offset);
}

static void
test_migration_table(void)
{
    int ret;
    tsk_migration_table_t table, table2;
    tsk_size_t num_rows = 100;
    tsk_id_t j;
    tsk_id_t *node;
    tsk_id_t *source, *dest;
    double *left, *right, *time;
    tsk_migration_t migration;
    char *metadata;
    uint32_t *metadata_offset;
    const char *test_metadata = "test";
    tsk_size_t test_metadata_length = 4;
    char metadata_copy[test_metadata_length + 1];

    metadata_copy[test_metadata_length] = '\0';
    ret = tsk_migration_table_init(&table, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tsk_migration_table_set_max_rows_increment(&table, 1);
    tsk_migration_table_print_state(&table, _devnull);
    tsk_migration_table_dump_text(&table, _devnull);

    for (j = 0; j < (tsk_id_t) num_rows; j++) {
        ret = tsk_migration_table_add_row(
            &table, j, j, j, j, j, j, test_metadata, test_metadata_length);
        CU_ASSERT_EQUAL_FATAL(ret, j);
        CU_ASSERT_EQUAL(table.left[j], j);
        CU_ASSERT_EQUAL(table.right[j], j);
        CU_ASSERT_EQUAL(table.node[j], j);
        CU_ASSERT_EQUAL(table.source[j], j);
        CU_ASSERT_EQUAL(table.dest[j], j);
        CU_ASSERT_EQUAL(table.time[j], j);
        CU_ASSERT_EQUAL(table.num_rows, (tsk_size_t) j + 1);
        CU_ASSERT_EQUAL(
            table.metadata_length, (tsk_size_t)(j + 1) * test_metadata_length);
        CU_ASSERT_EQUAL(table.metadata_offset[j + 1], table.metadata_length);
        /* check the metadata */
        memcpy(metadata_copy, table.metadata + table.metadata_offset[j],
            test_metadata_length);
        CU_ASSERT_NSTRING_EQUAL(metadata_copy, test_metadata, test_metadata_length);

        ret = tsk_migration_table_get_row(&table, (tsk_id_t) j, &migration);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        CU_ASSERT_EQUAL(migration.id, j);
        CU_ASSERT_EQUAL(migration.left, j);
        CU_ASSERT_EQUAL(migration.right, j);
        CU_ASSERT_EQUAL(migration.node, j);
        CU_ASSERT_EQUAL(migration.source, j);
        CU_ASSERT_EQUAL(migration.dest, j);
        CU_ASSERT_EQUAL(migration.time, j);
        CU_ASSERT_EQUAL(migration.metadata_length, test_metadata_length);
        CU_ASSERT_NSTRING_EQUAL(migration.metadata, test_metadata, test_metadata_length);
    }
    ret = tsk_migration_table_get_row(&table, (tsk_id_t) num_rows, &migration);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_MIGRATION_OUT_OF_BOUNDS);
    tsk_migration_table_print_state(&table, _devnull);
    tsk_migration_table_dump_text(&table, _devnull);

    num_rows *= 2;
    left = malloc(num_rows * sizeof(double));
    CU_ASSERT_FATAL(left != NULL);
    memset(left, 1, num_rows * sizeof(double));
    right = malloc(num_rows * sizeof(double));
    CU_ASSERT_FATAL(right != NULL);
    memset(right, 2, num_rows * sizeof(double));
    time = malloc(num_rows * sizeof(double));
    CU_ASSERT_FATAL(time != NULL);
    memset(time, 3, num_rows * sizeof(double));
    node = malloc(num_rows * sizeof(tsk_id_t));
    CU_ASSERT_FATAL(node != NULL);
    memset(node, 4, num_rows * sizeof(tsk_id_t));
    source = malloc(num_rows * sizeof(tsk_id_t));
    CU_ASSERT_FATAL(source != NULL);
    memset(source, 5, num_rows * sizeof(tsk_id_t));
    dest = malloc(num_rows * sizeof(tsk_id_t));
    CU_ASSERT_FATAL(dest != NULL);
    memset(dest, 6, num_rows * sizeof(tsk_id_t));
    metadata = malloc(num_rows * sizeof(char));
    memset(metadata, 'a', num_rows * sizeof(char));
    CU_ASSERT_FATAL(metadata != NULL);
    metadata_offset = malloc((num_rows + 1) * sizeof(tsk_size_t));
    CU_ASSERT_FATAL(metadata_offset != NULL);
    for (j = 0; j < (tsk_id_t) num_rows + 1; j++) {
        metadata_offset[j] = (tsk_size_t) j;
    }

    ret = tsk_migration_table_set_columns(&table, num_rows, left, right, node, source,
        dest, time, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(table.left, left, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(memcmp(table.right, right, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(memcmp(table.time, time, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(memcmp(table.node, node, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.source, source, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.dest, dest, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.metadata, metadata, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(memcmp(table.metadata_offset, metadata_offset,
                        (num_rows + 1) * sizeof(tsk_size_t)),
        0);
    CU_ASSERT_EQUAL(table.num_rows, num_rows);
    CU_ASSERT_EQUAL(table.metadata_length, num_rows);

    /* Append another num_rows */
    ret = tsk_migration_table_append_columns(&table, num_rows, left, right, node, source,
        dest, time, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(table.left, left, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(memcmp(table.left + num_rows, left, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(memcmp(table.right, right, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(memcmp(table.right + num_rows, right, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(memcmp(table.time, time, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(memcmp(table.time + num_rows, time, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(memcmp(table.node, node, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.node + num_rows, node, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.source, source, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.source + num_rows, source, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.dest, dest, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.dest + num_rows, dest, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.metadata, metadata, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.metadata + num_rows, metadata, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(table.num_rows, 2 * num_rows);
    CU_ASSERT_EQUAL(table.metadata_length, 2 * num_rows);

    /* Truncate back to num_rows */
    ret = tsk_migration_table_truncate(&table, num_rows);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(table.left, left, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(memcmp(table.right, right, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(memcmp(table.time, time, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(memcmp(table.node, node, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.source, source, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.dest, dest, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.metadata, metadata, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(memcmp(table.metadata_offset, metadata_offset,
                        (num_rows + 1) * sizeof(tsk_size_t)),
        0);
    CU_ASSERT_EQUAL(table.num_rows, num_rows);
    CU_ASSERT_EQUAL(table.metadata_length, num_rows);

    ret = tsk_migration_table_truncate(&table, num_rows + 1);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_TABLE_POSITION);

    /* inputs cannot be NULL */
    ret = tsk_migration_table_set_columns(&table, num_rows, NULL, right, node, source,
        dest, time, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    ret = tsk_migration_table_set_columns(&table, num_rows, left, NULL, node, source,
        dest, time, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    ret = tsk_migration_table_set_columns(&table, num_rows, left, right, NULL, source,
        dest, time, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    ret = tsk_migration_table_set_columns(&table, num_rows, left, right, node, NULL,
        dest, time, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    ret = tsk_migration_table_set_columns(&table, num_rows, left, right, node, source,
        NULL, time, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    ret = tsk_migration_table_set_columns(&table, num_rows, left, right, node, source,
        dest, NULL, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    ret = tsk_migration_table_set_columns(
        &table, num_rows, left, right, node, source, dest, time, NULL, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    ret = tsk_migration_table_set_columns(
        &table, num_rows, left, right, node, source, dest, time, metadata, NULL);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);

    tsk_migration_table_clear(&table);
    CU_ASSERT_EQUAL(table.num_rows, 0);

    /* if metadata and metadata_offset are both null, all metadatas are zero length */
    num_rows = 10;
    memset(metadata_offset, 0, (num_rows + 1) * sizeof(tsk_size_t));
    ret = tsk_migration_table_set_columns(
        &table, num_rows, left, right, node, source, dest, time, NULL, NULL);
    CU_ASSERT_EQUAL(memcmp(table.left, left, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(memcmp(table.right, right, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(memcmp(table.time, time, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(memcmp(table.node, node, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.source, source, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.dest, dest, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.metadata_offset, metadata_offset,
                        (num_rows + 1) * sizeof(tsk_size_t)),
        0);
    CU_ASSERT_EQUAL(table.num_rows, num_rows);
    CU_ASSERT_EQUAL(table.metadata_length, 0);
    ret = tsk_migration_table_append_columns(
        &table, num_rows, left, right, node, source, dest, time, NULL, NULL);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(table.left, left, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(memcmp(table.left + num_rows, left, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(memcmp(table.right, right, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(memcmp(table.right + num_rows, right, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(memcmp(table.time, time, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(memcmp(table.time + num_rows, time, num_rows * sizeof(double)), 0);
    CU_ASSERT_EQUAL(memcmp(table.node, node, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.node + num_rows, node, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.source, source, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.source + num_rows, source, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.dest, dest, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.dest + num_rows, dest, num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.metadata_offset, metadata_offset,
                        (num_rows + 1) * sizeof(tsk_size_t)),
        0);
    CU_ASSERT_EQUAL(memcmp(table.metadata_offset + num_rows, metadata_offset,
                        (num_rows + 1) * sizeof(tsk_size_t)),
        0);
    CU_ASSERT_EQUAL(table.num_rows, 2 * num_rows);
    CU_ASSERT_EQUAL(table.metadata_length, 0);
    tsk_migration_table_print_state(&table, _devnull);
    tsk_migration_table_dump_text(&table, _devnull);

    ret = tsk_migration_table_truncate(&table, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(table.metadata_schema_length, 0);
    CU_ASSERT_EQUAL(table.metadata_schema, NULL);
    const char *example = "An example of metadata schema with unicode 🎄🌳🌴🌲🎋";
    tsk_size_t example_length = (tsk_size_t) strlen(example);
    const char *example2 = "A different example 🎄🌳🌴🌲🎋";
    tsk_size_t example2_length = (tsk_size_t) strlen(example);
    tsk_migration_table_set_metadata_schema(&table, example, example_length);
    CU_ASSERT_EQUAL(table.metadata_schema_length, example_length);
    CU_ASSERT_EQUAL(memcmp(table.metadata_schema, example, example_length), 0);

    tsk_migration_table_copy(&table, &table2, 0);
    CU_ASSERT_EQUAL(table.metadata_schema_length, table2.metadata_schema_length);
    CU_ASSERT_EQUAL(
        memcmp(table.metadata_schema, table2.metadata_schema, example_length), 0);
    tsk_migration_table_set_metadata_schema(&table2, example, example_length);
    CU_ASSERT_TRUE(tsk_migration_table_equals(&table, &table2));
    tsk_migration_table_set_metadata_schema(&table2, example2, example2_length);
    CU_ASSERT_FALSE(tsk_migration_table_equals(&table, &table2));

    tsk_migration_table_clear(&table);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(table.num_rows, 0);
    CU_ASSERT_EQUAL(table.metadata_length, 0);

    tsk_migration_table_free(&table);
    CU_ASSERT_EQUAL(ret, 0);
    tsk_migration_table_free(&table2);
    CU_ASSERT_EQUAL(ret, 0);

    free(left);
    free(right);
    free(time);
    free(node);
    free(source);
    free(dest);
    free(metadata);
    free(metadata_offset);
}

static void
test_individual_table(void)
{
    int ret = 0;
    tsk_individual_table_t table, table2;
    /* tsk_table_collection_t tables, tables2; */
    tsk_size_t num_rows = 100;
    tsk_id_t j;
    tsk_size_t k;
    uint32_t *flags;
    double *location;
    char *metadata;
    tsk_size_t *metadata_offset;
    tsk_size_t *location_offset;
    tsk_individual_t individual;
    const char *test_metadata = "test";
    tsk_size_t test_metadata_length = 4;
    char metadata_copy[test_metadata_length + 1];
    tsk_size_t spatial_dimension = 2;
    double test_location[spatial_dimension];

    for (k = 0; k < spatial_dimension; k++) {
        test_location[k] = (double) k;
    }
    metadata_copy[test_metadata_length] = '\0';
    ret = tsk_individual_table_init(&table, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tsk_individual_table_set_max_rows_increment(&table, 1);
    tsk_individual_table_set_max_metadata_length_increment(&table, 1);
    tsk_individual_table_set_max_location_length_increment(&table, 1);

    tsk_individual_table_print_state(&table, _devnull);

    for (j = 0; j < (tsk_id_t) num_rows; j++) {
        ret = tsk_individual_table_add_row(&table, (tsk_flags_t) j, test_location,
            spatial_dimension, test_metadata, test_metadata_length);
        CU_ASSERT_EQUAL_FATAL(ret, j);
        CU_ASSERT_EQUAL(table.flags[j], (tsk_flags_t) j);
        for (k = 0; k < spatial_dimension; k++) {
            test_location[k] = (double) k;
            CU_ASSERT_EQUAL(
                table.location[spatial_dimension * (size_t) j + k], test_location[k]);
        }
        CU_ASSERT_EQUAL(
            table.metadata_length, (tsk_size_t)(j + 1) * test_metadata_length);
        CU_ASSERT_EQUAL(table.metadata_offset[j + 1], table.metadata_length);
        /* check the metadata */
        memcpy(metadata_copy, table.metadata + table.metadata_offset[j],
            test_metadata_length);
        CU_ASSERT_NSTRING_EQUAL(metadata_copy, test_metadata, test_metadata_length);

        ret = tsk_individual_table_get_row(&table, (tsk_id_t) j, &individual);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        CU_ASSERT_EQUAL(individual.id, j);
        CU_ASSERT_EQUAL(individual.flags, (uint32_t) j);
        CU_ASSERT_EQUAL(individual.location_length, spatial_dimension);
        CU_ASSERT_NSTRING_EQUAL(
            individual.location, test_location, spatial_dimension * sizeof(double));
        CU_ASSERT_EQUAL(individual.metadata_length, test_metadata_length);
        CU_ASSERT_NSTRING_EQUAL(
            individual.metadata, test_metadata, test_metadata_length);
    }
    ret = tsk_individual_table_get_row(&table, (tsk_id_t) num_rows, &individual);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_INDIVIDUAL_OUT_OF_BOUNDS);
    tsk_individual_table_print_state(&table, _devnull);
    tsk_individual_table_clear(&table);
    CU_ASSERT_EQUAL(table.num_rows, 0);
    CU_ASSERT_EQUAL(table.metadata_length, 0);

    num_rows *= 2;
    flags = malloc(num_rows * sizeof(uint32_t));
    CU_ASSERT_FATAL(flags != NULL);
    memset(flags, 1, num_rows * sizeof(uint32_t));
    location = malloc(spatial_dimension * num_rows * sizeof(double));
    CU_ASSERT_FATAL(location != NULL);
    memset(location, 0, spatial_dimension * num_rows * sizeof(double));
    location_offset = malloc((num_rows + 1) * sizeof(tsk_size_t));
    CU_ASSERT_FATAL(location_offset != NULL);
    for (j = 0; j < (tsk_id_t) num_rows + 1; j++) {
        location_offset[j] = (tsk_size_t) j * spatial_dimension;
    }
    metadata = malloc(num_rows * sizeof(char));
    memset(metadata, 'a', num_rows * sizeof(char));
    CU_ASSERT_FATAL(metadata != NULL);
    metadata_offset = malloc((num_rows + 1) * sizeof(tsk_size_t));
    CU_ASSERT_FATAL(metadata_offset != NULL);
    for (j = 0; j < (tsk_id_t) num_rows + 1; j++) {
        metadata_offset[j] = (tsk_size_t) j;
    }
    ret = tsk_individual_table_set_columns(
        &table, num_rows, flags, location, location_offset, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(table.flags, flags, num_rows * sizeof(uint32_t)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.location, location, spatial_dimension * num_rows * sizeof(double)),
        0);
    CU_ASSERT_EQUAL(memcmp(table.location_offset, location_offset,
                        (num_rows + 1) * sizeof(tsk_size_t)),
        0);
    CU_ASSERT_EQUAL(memcmp(table.metadata, metadata, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(memcmp(table.metadata_offset, metadata_offset,
                        (num_rows + 1) * sizeof(tsk_size_t)),
        0);
    CU_ASSERT_EQUAL(table.num_rows, num_rows);
    CU_ASSERT_EQUAL(table.location_length, spatial_dimension * num_rows);
    CU_ASSERT_EQUAL(table.metadata_length, num_rows);
    tsk_individual_table_print_state(&table, _devnull);

    /* Append another num_rows onto the end */
    ret = tsk_individual_table_append_columns(
        &table, num_rows, flags, location, location_offset, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(table.flags, flags, num_rows * sizeof(uint32_t)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.flags + num_rows, flags, num_rows * sizeof(uint32_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.metadata, metadata, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.metadata + num_rows, metadata, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.location, location, spatial_dimension * num_rows * sizeof(double)),
        0);
    CU_ASSERT_EQUAL(memcmp(table.location + spatial_dimension * num_rows, location,
                        spatial_dimension * num_rows * sizeof(double)),
        0);
    CU_ASSERT_EQUAL(table.num_rows, 2 * num_rows);
    CU_ASSERT_EQUAL(table.metadata_length, 2 * num_rows);
    tsk_individual_table_print_state(&table, _devnull);
    tsk_individual_table_dump_text(&table, _devnull);

    /* Truncate back to num_rows */
    ret = tsk_individual_table_truncate(&table, num_rows);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(table.flags, flags, num_rows * sizeof(uint32_t)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.location, location, spatial_dimension * num_rows * sizeof(double)),
        0);
    CU_ASSERT_EQUAL(memcmp(table.location_offset, location_offset,
                        (num_rows + 1) * sizeof(tsk_size_t)),
        0);
    CU_ASSERT_EQUAL(memcmp(table.metadata, metadata, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(memcmp(table.metadata_offset, metadata_offset,
                        (num_rows + 1) * sizeof(tsk_size_t)),
        0);
    CU_ASSERT_EQUAL(table.num_rows, num_rows);
    CU_ASSERT_EQUAL(table.location_length, spatial_dimension * num_rows);
    CU_ASSERT_EQUAL(table.metadata_length, num_rows);
    tsk_individual_table_print_state(&table, _devnull);

    ret = tsk_individual_table_truncate(&table, num_rows + 1);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_TABLE_POSITION);

    /* flags can't be NULL */
    ret = tsk_individual_table_set_columns(
        &table, num_rows, NULL, location, location_offset, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    /* location and location offset must be simultaneously NULL or not */
    ret = tsk_individual_table_set_columns(
        &table, num_rows, flags, location, NULL, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    ret = tsk_individual_table_set_columns(
        &table, num_rows, flags, NULL, location_offset, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    /* metadata and metadata offset must be simultaneously NULL or not */
    ret = tsk_individual_table_set_columns(
        &table, num_rows, flags, location, location_offset, NULL, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    ret = tsk_individual_table_set_columns(
        &table, num_rows, flags, location, location_offset, metadata, NULL);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);

    /* if location and location_offset are both null, all locations are zero length */
    num_rows = 10;
    memset(location_offset, 0, (num_rows + 1) * sizeof(tsk_size_t));
    ret = tsk_individual_table_set_columns(
        &table, num_rows, flags, NULL, NULL, NULL, NULL);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(table.location_offset, location_offset,
                        (num_rows + 1) * sizeof(tsk_size_t)),
        0);
    CU_ASSERT_EQUAL(table.num_rows, num_rows);
    CU_ASSERT_EQUAL(table.location_length, 0);
    ret = tsk_individual_table_append_columns(
        &table, num_rows, flags, NULL, NULL, NULL, NULL);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(table.location_offset, location_offset,
                        (num_rows + 1) * sizeof(tsk_size_t)),
        0);
    CU_ASSERT_EQUAL(memcmp(table.location_offset + num_rows, location_offset,
                        num_rows * sizeof(uint32_t)),
        0);
    CU_ASSERT_EQUAL(table.num_rows, 2 * num_rows);
    CU_ASSERT_EQUAL(table.location_length, 0);
    tsk_individual_table_print_state(&table, _devnull);
    tsk_individual_table_dump_text(&table, _devnull);

    /* if metadata and metadata_offset are both null, all metadatas are zero length */
    num_rows = 10;
    memset(metadata_offset, 0, (num_rows + 1) * sizeof(tsk_size_t));
    ret = tsk_individual_table_set_columns(
        &table, num_rows, flags, location, location_offset, NULL, NULL);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(table.flags, flags, num_rows * sizeof(uint32_t)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.location, location, spatial_dimension * num_rows * sizeof(double)),
        0);
    CU_ASSERT_EQUAL(memcmp(table.metadata_offset, metadata_offset,
                        (num_rows + 1) * sizeof(tsk_size_t)),
        0);
    CU_ASSERT_EQUAL(table.num_rows, num_rows);
    CU_ASSERT_EQUAL(table.metadata_length, 0);
    ret = tsk_individual_table_append_columns(
        &table, num_rows, flags, location, location_offset, NULL, NULL);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(
        memcmp(table.location, location, spatial_dimension * num_rows * sizeof(double)),
        0);
    CU_ASSERT_EQUAL(memcmp(table.location + spatial_dimension * num_rows, location,
                        spatial_dimension * num_rows * sizeof(double)),
        0);
    CU_ASSERT_EQUAL(memcmp(table.metadata_offset, metadata_offset,
                        (num_rows + 1) * sizeof(tsk_size_t)),
        0);
    CU_ASSERT_EQUAL(memcmp(table.metadata_offset + num_rows, metadata_offset,
                        num_rows * sizeof(uint32_t)),
        0);
    CU_ASSERT_EQUAL(table.num_rows, 2 * num_rows);
    CU_ASSERT_EQUAL(table.metadata_length, 0);
    tsk_individual_table_print_state(&table, _devnull);
    tsk_individual_table_dump_text(&table, _devnull);

    ret = tsk_individual_table_truncate(&table, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(table.metadata_schema_length, 0);
    CU_ASSERT_EQUAL(table.metadata_schema, NULL);
    const char *example = "An example of metadata schema with unicode 🎄🌳🌴🌲🎋";
    tsk_size_t example_length = (tsk_size_t) strlen(example);
    const char *example2 = "A different example 🎄🌳🌴🌲🎋";
    tsk_size_t example2_length = (tsk_size_t) strlen(example);
    tsk_individual_table_set_metadata_schema(&table, example, example_length);
    CU_ASSERT_EQUAL(table.metadata_schema_length, example_length);
    CU_ASSERT_EQUAL(memcmp(table.metadata_schema, example, example_length), 0);

    tsk_individual_table_copy(&table, &table2, 0);
    CU_ASSERT_EQUAL(table.metadata_schema_length, table2.metadata_schema_length);
    CU_ASSERT_EQUAL(
        memcmp(table.metadata_schema, table2.metadata_schema, example_length), 0);
    tsk_individual_table_set_metadata_schema(&table2, example, example_length);
    CU_ASSERT_TRUE(tsk_individual_table_equals(&table, &table2));
    tsk_individual_table_set_metadata_schema(&table2, example2, example2_length);
    CU_ASSERT_FALSE(tsk_individual_table_equals(&table, &table2));

    tsk_individual_table_clear(&table);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(table.num_rows, 0);
    CU_ASSERT_EQUAL(table.metadata_length, 0);

    ret = tsk_individual_table_free(&table);
    CU_ASSERT_EQUAL(ret, 0);
    ret = tsk_individual_table_free(&table2);
    CU_ASSERT_EQUAL(ret, 0);
    free(flags);
    free(location);
    free(location_offset);
    free(metadata);
    free(metadata_offset);
}

static void
test_population_table(void)
{
    int ret;
    tsk_population_table_t table, table2;
    tsk_size_t num_rows = 100;
    tsk_size_t max_len = 20;
    tsk_size_t k, len;
    tsk_id_t j;
    char *metadata;
    char c[max_len + 1];
    tsk_size_t *metadata_offset;
    tsk_population_t population;

    for (j = 0; j < (tsk_id_t) max_len; j++) {
        c[j] = (char) ('A' + j);
    }

    ret = tsk_population_table_init(&table, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tsk_population_table_set_max_rows_increment(&table, 1);
    tsk_population_table_set_max_metadata_length_increment(&table, 1);
    tsk_population_table_print_state(&table, _devnull);
    tsk_population_table_dump_text(&table, _devnull);
    /* Adding zero length metadata with NULL should be fine */

    ret = tsk_population_table_add_row(&table, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    CU_ASSERT_EQUAL(table.metadata_length, 0);
    CU_ASSERT_EQUAL(table.num_rows, 1);
    CU_ASSERT_EQUAL(table.metadata_offset[0], 0);
    CU_ASSERT_EQUAL(table.metadata_offset[1], 0);
    tsk_population_table_clear(&table);
    CU_ASSERT_EQUAL(table.num_rows, 0);

    len = 0;
    for (j = 0; j < (tsk_id_t) num_rows; j++) {
        k = TSK_MIN((tsk_size_t) j + 1, max_len);
        ret = tsk_population_table_add_row(&table, c, k);
        CU_ASSERT_EQUAL_FATAL(ret, j);
        CU_ASSERT_EQUAL(table.metadata_offset[j], len);
        CU_ASSERT_EQUAL(table.num_rows, (tsk_size_t) j + 1);
        len += k;
        CU_ASSERT_EQUAL(table.metadata_offset[j + 1], len);
        CU_ASSERT_EQUAL(table.metadata_length, len);

        ret = tsk_population_table_get_row(&table, (tsk_id_t) j, &population);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        CU_ASSERT_EQUAL(population.id, j);
        CU_ASSERT_EQUAL(population.metadata_length, k);
        CU_ASSERT_NSTRING_EQUAL(population.metadata, c, k);
    }
    ret = tsk_population_table_get_row(&table, (tsk_id_t) num_rows, &population);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_POPULATION_OUT_OF_BOUNDS);
    tsk_population_table_print_state(&table, _devnull);
    tsk_population_table_dump_text(&table, _devnull);

    num_rows *= 2;
    metadata = malloc(num_rows * sizeof(char));
    CU_ASSERT_FATAL(metadata != NULL);
    metadata_offset = malloc((num_rows + 1) * sizeof(tsk_size_t));
    CU_ASSERT_FATAL(metadata_offset != NULL);

    for (j = 0; j < (tsk_id_t) num_rows; j++) {
        metadata[j] = 'M';
        metadata_offset[j] = (tsk_size_t) j;
    }

    metadata_offset[num_rows] = num_rows;
    ret = tsk_population_table_set_columns(&table, num_rows, metadata, metadata_offset);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(table.metadata, metadata, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(table.num_rows, num_rows);
    CU_ASSERT_EQUAL(table.metadata_length, num_rows);

    /* Append another num_rows */
    ret = tsk_population_table_append_columns(
        &table, num_rows, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(table.metadata, metadata, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.metadata + num_rows, metadata, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(table.metadata_length, 2 * num_rows);
    CU_ASSERT_EQUAL(table.num_rows, 2 * num_rows);

    /* Truncate back to num_rows */
    ret = tsk_population_table_truncate(&table, num_rows);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(table.metadata, metadata, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(table.num_rows, num_rows);
    CU_ASSERT_EQUAL(table.metadata_length, num_rows);

    ret = tsk_population_table_truncate(&table, num_rows + 1);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_TABLE_POSITION);

    /* Metadata = NULL gives an error */
    ret = tsk_population_table_set_columns(&table, num_rows, NULL, NULL);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    ret = tsk_population_table_set_columns(&table, num_rows, metadata, NULL);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    ret = tsk_population_table_set_columns(&table, num_rows, NULL, metadata_offset);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_PARAM_VALUE);

    /* Test for bad offsets */
    metadata_offset[0] = 1;
    ret = tsk_population_table_set_columns(&table, num_rows, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_OFFSET);
    metadata_offset[0] = 0;
    metadata_offset[num_rows] = 0;
    ret = tsk_population_table_set_columns(&table, num_rows, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_OFFSET);

    ret = tsk_population_table_truncate(&table, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(table.metadata_schema_length, 0);
    CU_ASSERT_EQUAL(table.metadata_schema, NULL);
    const char *example = "An example of metadata schema with unicode 🎄🌳🌴🌲🎋";
    tsk_size_t example_length = (tsk_size_t) strlen(example);
    const char *example2 = "A different example 🎄🌳🌴🌲🎋";
    tsk_size_t example2_length = (tsk_size_t) strlen(example);
    tsk_population_table_set_metadata_schema(&table, example, example_length);
    CU_ASSERT_EQUAL(table.metadata_schema_length, example_length);
    CU_ASSERT_EQUAL(memcmp(table.metadata_schema, example, example_length), 0);

    tsk_population_table_copy(&table, &table2, 0);
    CU_ASSERT_EQUAL(table.metadata_schema_length, table2.metadata_schema_length);
    CU_ASSERT_EQUAL(
        memcmp(table.metadata_schema, table2.metadata_schema, example_length), 0);
    tsk_population_table_set_metadata_schema(&table2, example, example_length);
    CU_ASSERT_TRUE(tsk_population_table_equals(&table, &table2));
    tsk_population_table_set_metadata_schema(&table2, example2, example2_length);
    CU_ASSERT_FALSE(tsk_population_table_equals(&table, &table2));

    tsk_population_table_clear(&table);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(table.num_rows, 0);
    CU_ASSERT_EQUAL(table.metadata_length, 0);

    tsk_population_table_free(&table);
    CU_ASSERT_EQUAL(ret, 0);
    tsk_population_table_free(&table2);
    CU_ASSERT_EQUAL(ret, 0);

    free(metadata);
    free(metadata_offset);
}

static void
test_provenance_table(void)
{
    int ret;
    tsk_provenance_table_t table;
    tsk_size_t num_rows = 100;
    tsk_size_t j;
    char *timestamp;
    uint32_t *timestamp_offset;
    const char *test_timestamp = "2017-12-06T20:40:25+00:00";
    tsk_size_t test_timestamp_length = (tsk_size_t) strlen(test_timestamp);
    char timestamp_copy[test_timestamp_length + 1];
    char *record;
    uint32_t *record_offset;
    const char *test_record = "{\"json\"=1234}";
    tsk_size_t test_record_length = (tsk_size_t) strlen(test_record);
    char record_copy[test_record_length + 1];
    tsk_provenance_t provenance;

    timestamp_copy[test_timestamp_length] = '\0';
    record_copy[test_record_length] = '\0';
    ret = tsk_provenance_table_init(&table, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tsk_provenance_table_set_max_rows_increment(&table, 1);
    tsk_provenance_table_set_max_timestamp_length_increment(&table, 1);
    tsk_provenance_table_set_max_record_length_increment(&table, 1);
    tsk_provenance_table_print_state(&table, _devnull);
    tsk_provenance_table_dump_text(&table, _devnull);

    for (j = 0; j < num_rows; j++) {
        ret = tsk_provenance_table_add_row(&table, test_timestamp, test_timestamp_length,
            test_record, test_record_length);
        CU_ASSERT_EQUAL_FATAL(ret, (int) j);
        CU_ASSERT_EQUAL(table.timestamp_length, (j + 1) * test_timestamp_length);
        CU_ASSERT_EQUAL(table.timestamp_offset[j + 1], table.timestamp_length);
        CU_ASSERT_EQUAL(table.record_length, (j + 1) * test_record_length);
        CU_ASSERT_EQUAL(table.record_offset[j + 1], table.record_length);
        /* check the timestamp */
        memcpy(timestamp_copy, table.timestamp + table.timestamp_offset[j],
            test_timestamp_length);
        CU_ASSERT_NSTRING_EQUAL(timestamp_copy, test_timestamp, test_timestamp_length);
        /* check the record */
        memcpy(record_copy, table.record + table.record_offset[j], test_record_length);
        CU_ASSERT_NSTRING_EQUAL(record_copy, test_record, test_record_length);

        ret = tsk_provenance_table_get_row(&table, (tsk_id_t) j, &provenance);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        CU_ASSERT_EQUAL(provenance.id, (tsk_id_t) j);
        CU_ASSERT_EQUAL(provenance.timestamp_length, test_timestamp_length);
        CU_ASSERT_NSTRING_EQUAL(
            provenance.timestamp, test_timestamp, test_timestamp_length);
        CU_ASSERT_EQUAL(provenance.record_length, test_record_length);
        CU_ASSERT_NSTRING_EQUAL(provenance.record, test_record, test_record_length);
    }
    ret = tsk_provenance_table_get_row(&table, (tsk_id_t) num_rows, &provenance);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_PROVENANCE_OUT_OF_BOUNDS);
    tsk_provenance_table_print_state(&table, _devnull);
    tsk_provenance_table_dump_text(&table, _devnull);
    tsk_provenance_table_clear(&table);
    CU_ASSERT_EQUAL(table.num_rows, 0);
    CU_ASSERT_EQUAL(table.timestamp_length, 0);
    CU_ASSERT_EQUAL(table.record_length, 0);

    num_rows *= 2;
    timestamp = malloc(num_rows * sizeof(char));
    memset(timestamp, 'a', num_rows * sizeof(char));
    CU_ASSERT_FATAL(timestamp != NULL);
    timestamp_offset = malloc((num_rows + 1) * sizeof(tsk_size_t));
    CU_ASSERT_FATAL(timestamp_offset != NULL);
    record = malloc(num_rows * sizeof(char));
    memset(record, 'a', num_rows * sizeof(char));
    CU_ASSERT_FATAL(record != NULL);
    record_offset = malloc((num_rows + 1) * sizeof(tsk_size_t));
    CU_ASSERT_FATAL(record_offset != NULL);
    for (j = 0; j < num_rows + 1; j++) {
        timestamp_offset[j] = j;
        record_offset[j] = j;
    }
    ret = tsk_provenance_table_set_columns(
        &table, num_rows, timestamp, timestamp_offset, record, record_offset);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(table.timestamp, timestamp, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(memcmp(table.timestamp_offset, timestamp_offset,
                        (num_rows + 1) * sizeof(tsk_size_t)),
        0);
    CU_ASSERT_EQUAL(memcmp(table.record, record, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.record_offset, record_offset, (num_rows + 1) * sizeof(tsk_size_t)),
        0);
    CU_ASSERT_EQUAL(table.num_rows, num_rows);
    CU_ASSERT_EQUAL(table.timestamp_length, num_rows);
    CU_ASSERT_EQUAL(table.record_length, num_rows);
    tsk_provenance_table_print_state(&table, _devnull);

    /* Append another num_rows onto the end */
    ret = tsk_provenance_table_append_columns(
        &table, num_rows, timestamp, timestamp_offset, record, record_offset);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(table.timestamp, timestamp, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.timestamp + num_rows, timestamp, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(memcmp(table.record, record, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(memcmp(table.record + num_rows, record, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(table.num_rows, 2 * num_rows);
    CU_ASSERT_EQUAL(table.timestamp_length, 2 * num_rows);
    CU_ASSERT_EQUAL(table.record_length, 2 * num_rows);
    tsk_provenance_table_print_state(&table, _devnull);

    /* Truncate back to num_rows */
    ret = tsk_provenance_table_truncate(&table, num_rows);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(table.timestamp, timestamp, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(memcmp(table.timestamp_offset, timestamp_offset,
                        (num_rows + 1) * sizeof(tsk_size_t)),
        0);
    CU_ASSERT_EQUAL(memcmp(table.record, record, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.record_offset, record_offset, (num_rows + 1) * sizeof(tsk_size_t)),
        0);
    CU_ASSERT_EQUAL(table.num_rows, num_rows);
    CU_ASSERT_EQUAL(table.timestamp_length, num_rows);
    CU_ASSERT_EQUAL(table.record_length, num_rows);
    tsk_provenance_table_print_state(&table, _devnull);

    ret = tsk_provenance_table_truncate(&table, num_rows + 1);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_TABLE_POSITION);

    /* No arguments can be null */
    ret = tsk_provenance_table_set_columns(
        &table, num_rows, NULL, timestamp_offset, record, record_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    ret = tsk_provenance_table_set_columns(
        &table, num_rows, timestamp, NULL, record, record_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    ret = tsk_provenance_table_set_columns(
        &table, num_rows, timestamp, timestamp_offset, NULL, record_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    ret = tsk_provenance_table_set_columns(
        &table, num_rows, timestamp, timestamp_offset, record, NULL);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);

    tsk_provenance_table_free(&table);
    free(timestamp);
    free(timestamp_offset);
    free(record);
    free(record_offset);
}

static void
test_table_size_increments(void)
{
    int ret;
    tsk_table_collection_t tables;
    tsk_size_t default_size = 1024;
    tsk_size_t new_size;

    ret = tsk_table_collection_init(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    CU_ASSERT_EQUAL_FATAL(tables.individuals.max_rows_increment, default_size);
    CU_ASSERT_EQUAL_FATAL(
        tables.individuals.max_metadata_length_increment, default_size);
    CU_ASSERT_EQUAL_FATAL(
        tables.individuals.max_location_length_increment, default_size);
    CU_ASSERT_EQUAL_FATAL(tables.nodes.max_rows_increment, default_size);
    CU_ASSERT_EQUAL_FATAL(tables.nodes.max_metadata_length_increment, default_size);
    CU_ASSERT_EQUAL_FATAL(tables.edges.max_rows_increment, default_size);
    CU_ASSERT_EQUAL_FATAL(tables.edges.max_metadata_length_increment, default_size);
    CU_ASSERT_EQUAL_FATAL(tables.sites.max_rows_increment, default_size);
    CU_ASSERT_EQUAL_FATAL(tables.sites.max_metadata_length_increment, default_size);
    CU_ASSERT_EQUAL_FATAL(
        tables.sites.max_ancestral_state_length_increment, default_size);
    CU_ASSERT_EQUAL_FATAL(tables.mutations.max_rows_increment, default_size);
    CU_ASSERT_EQUAL_FATAL(tables.mutations.max_metadata_length_increment, default_size);
    CU_ASSERT_EQUAL_FATAL(
        tables.mutations.max_derived_state_length_increment, default_size);
    CU_ASSERT_EQUAL_FATAL(tables.migrations.max_rows_increment, default_size);
    CU_ASSERT_EQUAL_FATAL(tables.migrations.max_metadata_length_increment, default_size);
    CU_ASSERT_EQUAL_FATAL(tables.populations.max_rows_increment, default_size);
    CU_ASSERT_EQUAL_FATAL(
        tables.populations.max_metadata_length_increment, default_size);
    CU_ASSERT_EQUAL_FATAL(tables.provenances.max_rows_increment, default_size);
    CU_ASSERT_EQUAL_FATAL(
        tables.provenances.max_timestamp_length_increment, default_size);
    CU_ASSERT_EQUAL_FATAL(tables.provenances.max_record_length_increment, default_size);

    /* Setting to zero sets to the default size */
    new_size = 0;
    ret = tsk_individual_table_set_max_rows_increment(&tables.individuals, new_size);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(tables.individuals.max_rows_increment, default_size);
    ret = tsk_individual_table_set_max_metadata_length_increment(
        &tables.individuals, new_size);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(
        tables.individuals.max_metadata_length_increment, default_size);
    ret = tsk_individual_table_set_max_location_length_increment(
        &tables.individuals, new_size);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(
        tables.individuals.max_location_length_increment, default_size);

    ret = tsk_node_table_set_max_rows_increment(&tables.nodes, new_size);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(tables.nodes.max_rows_increment, default_size);
    ret = tsk_node_table_set_max_metadata_length_increment(&tables.nodes, new_size);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(tables.nodes.max_metadata_length_increment, default_size);

    ret = tsk_edge_table_set_max_rows_increment(&tables.edges, new_size);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(tables.edges.max_rows_increment, default_size);
    ret = tsk_edge_table_set_max_metadata_length_increment(&tables.edges, new_size);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(tables.edges.max_metadata_length_increment, default_size);

    ret = tsk_site_table_set_max_rows_increment(&tables.sites, new_size);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(tables.sites.max_rows_increment, default_size);
    ret = tsk_site_table_set_max_metadata_length_increment(&tables.sites, new_size);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(tables.sites.max_metadata_length_increment, default_size);
    ret = tsk_site_table_set_max_ancestral_state_length_increment(
        &tables.sites, new_size);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(
        tables.sites.max_ancestral_state_length_increment, default_size);

    ret = tsk_mutation_table_set_max_rows_increment(&tables.mutations, new_size);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(tables.mutations.max_rows_increment, default_size);
    ret = tsk_mutation_table_set_max_metadata_length_increment(
        &tables.mutations, new_size);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(tables.mutations.max_metadata_length_increment, default_size);
    ret = tsk_mutation_table_set_max_derived_state_length_increment(
        &tables.mutations, new_size);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(
        tables.mutations.max_derived_state_length_increment, default_size);

    ret = tsk_migration_table_set_max_rows_increment(&tables.migrations, new_size);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(tables.migrations.max_rows_increment, default_size);
    ret = tsk_migration_table_set_max_metadata_length_increment(
        &tables.migrations, new_size);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(tables.migrations.max_metadata_length_increment, default_size);

    ret = tsk_population_table_set_max_rows_increment(&tables.populations, new_size);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(tables.populations.max_rows_increment, default_size);
    ret = tsk_population_table_set_max_metadata_length_increment(
        &tables.populations, new_size);
    CU_ASSERT_EQUAL_FATAL(
        tables.populations.max_metadata_length_increment, default_size);

    ret = tsk_provenance_table_set_max_rows_increment(&tables.provenances, new_size);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(tables.provenances.max_rows_increment, default_size);
    ret = tsk_provenance_table_set_max_timestamp_length_increment(
        &tables.provenances, new_size);
    CU_ASSERT_EQUAL_FATAL(
        tables.provenances.max_timestamp_length_increment, default_size);
    ret = tsk_provenance_table_set_max_record_length_increment(
        &tables.provenances, new_size);
    CU_ASSERT_EQUAL_FATAL(tables.provenances.max_record_length_increment, default_size);

    /* Setting to non-zero sets to thatsize */
    new_size = 1;
    ret = tsk_individual_table_set_max_rows_increment(&tables.individuals, new_size);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(tables.individuals.max_rows_increment, new_size);
    ret = tsk_individual_table_set_max_metadata_length_increment(
        &tables.individuals, new_size);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(tables.individuals.max_metadata_length_increment, new_size);
    ret = tsk_individual_table_set_max_location_length_increment(
        &tables.individuals, new_size);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(tables.individuals.max_location_length_increment, new_size);

    ret = tsk_node_table_set_max_rows_increment(&tables.nodes, new_size);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(tables.nodes.max_rows_increment, new_size);
    ret = tsk_node_table_set_max_metadata_length_increment(&tables.nodes, new_size);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(tables.nodes.max_metadata_length_increment, new_size);

    ret = tsk_edge_table_set_max_rows_increment(&tables.edges, new_size);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(tables.edges.max_rows_increment, new_size);
    ret = tsk_edge_table_set_max_metadata_length_increment(&tables.edges, new_size);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(tables.edges.max_metadata_length_increment, new_size);

    ret = tsk_site_table_set_max_rows_increment(&tables.sites, new_size);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(tables.sites.max_rows_increment, new_size);
    ret = tsk_site_table_set_max_metadata_length_increment(&tables.sites, new_size);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(tables.sites.max_metadata_length_increment, new_size);
    ret = tsk_site_table_set_max_ancestral_state_length_increment(
        &tables.sites, new_size);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(tables.sites.max_ancestral_state_length_increment, new_size);

    ret = tsk_mutation_table_set_max_rows_increment(&tables.mutations, new_size);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(tables.mutations.max_rows_increment, new_size);
    ret = tsk_mutation_table_set_max_metadata_length_increment(
        &tables.mutations, new_size);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(tables.mutations.max_metadata_length_increment, new_size);
    ret = tsk_mutation_table_set_max_derived_state_length_increment(
        &tables.mutations, new_size);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(tables.mutations.max_derived_state_length_increment, new_size);

    ret = tsk_migration_table_set_max_rows_increment(&tables.migrations, new_size);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(tables.migrations.max_rows_increment, new_size);
    ret = tsk_migration_table_set_max_metadata_length_increment(
        &tables.migrations, new_size);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(tables.migrations.max_metadata_length_increment, new_size);

    ret = tsk_population_table_set_max_rows_increment(&tables.populations, new_size);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(tables.populations.max_rows_increment, new_size);
    ret = tsk_population_table_set_max_metadata_length_increment(
        &tables.populations, new_size);
    CU_ASSERT_EQUAL_FATAL(tables.populations.max_metadata_length_increment, new_size);

    ret = tsk_provenance_table_set_max_rows_increment(&tables.provenances, new_size);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(tables.provenances.max_rows_increment, new_size);
    ret = tsk_provenance_table_set_max_timestamp_length_increment(
        &tables.provenances, new_size);
    CU_ASSERT_EQUAL_FATAL(tables.provenances.max_timestamp_length_increment, new_size);
    ret = tsk_provenance_table_set_max_record_length_increment(
        &tables.provenances, new_size);
    CU_ASSERT_EQUAL_FATAL(tables.provenances.max_record_length_increment, new_size);

    tsk_table_collection_free(&tables);
}

static void
test_link_ancestors_input_errors(void)
{
    int ret;
    tsk_treeseq_t ts;
    tsk_table_collection_t tables;
    tsk_edge_table_t result;
    tsk_id_t samples[] = { 0, 1 };
    tsk_id_t ancestors[] = { 4, 6 };

    tsk_treeseq_from_text(&ts, 1, single_tree_ex_nodes, single_tree_ex_edges, NULL, NULL,
        NULL, NULL, NULL, 0);
    ret = tsk_treeseq_copy_tables(&ts, &tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* Add an edge with some metadata */
    ret = tsk_node_table_add_row(&tables.nodes, 0, 0, TSK_NULL, TSK_NULL, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 7);
    ret = tsk_edge_table_add_row(&tables.edges, 0, 1, 7, 6, "metadata", 8);
    CU_ASSERT_FATAL(ret > 0);

    ret = tsk_edge_table_init(&result, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_link_ancestors(
        &tables, NULL, 2, ancestors, 2, 0, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_CANT_PROCESS_EDGES_WITH_METADATA);
    tsk_table_collection_free(&tables);
    tsk_treeseq_free(&ts);
    tsk_edge_table_free(&result);

    tsk_treeseq_from_text(&ts, 1, single_tree_ex_nodes, single_tree_ex_edges, NULL, NULL,
        NULL, NULL, NULL, 0);
    ret = tsk_treeseq_copy_tables(&ts, &tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_edge_table_init(&result, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_table_collection_link_ancestors(
        &tables, NULL, 2, ancestors, 2, 0, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_PARAM_VALUE);

    /* Bad sample IDs */
    samples[0] = -1;
    ret = tsk_table_collection_link_ancestors(
        &tables, samples, 2, ancestors, 2, 0, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_NODE_OUT_OF_BOUNDS);

    /* Bad ancestor IDs */
    samples[0] = 0;
    ancestors[0] = -1;
    ret = tsk_table_collection_link_ancestors(
        &tables, samples, 2, ancestors, 2, 0, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_NODE_OUT_OF_BOUNDS);

    /* Duplicate sample IDs */
    ancestors[0] = 4;
    samples[0] = 1;
    ret = tsk_table_collection_link_ancestors(
        &tables, samples, 2, ancestors, 2, 0, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_DUPLICATE_SAMPLE);

    /* Duplicate sample IDs */
    ancestors[0] = 6;
    samples[0] = 0;
    ret = tsk_table_collection_link_ancestors(
        &tables, samples, 2, ancestors, 2, 0, &result);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_DUPLICATE_SAMPLE);

    /* TODO more tests! */

    tsk_table_collection_free(&tables);
    tsk_treeseq_free(&ts);
    tsk_edge_table_free(&result);
}

static void
test_link_ancestors_single_tree(void)
{
    int ret;
    tsk_treeseq_t ts;
    tsk_table_collection_t tables;
    tsk_edge_table_t result;
    tsk_id_t samples[] = { 0, 1 };
    tsk_id_t ancestors[] = { 4, 6 };
    size_t i;
    double res_left = 0;
    double res_right = 1;

    tsk_treeseq_from_text(&ts, 1, single_tree_ex_nodes, single_tree_ex_edges, NULL, NULL,
        NULL, NULL, NULL, 0);
    ret = tsk_treeseq_copy_tables(&ts, &tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_edge_table_init(&result, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_table_collection_link_ancestors(
        &tables, samples, 2, ancestors, 2, 0, &result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    // Check we get the right result.
    CU_ASSERT_EQUAL(result.num_rows, 3);
    tsk_id_t res_parent[] = { 4, 4, 6 };
    tsk_id_t res_child[] = { 0, 1, 4 };
    for (i = 0; i < result.num_rows; i++) {
        CU_ASSERT_EQUAL(res_parent[i], result.parent[i]);
        CU_ASSERT_EQUAL(res_child[i], result.child[i]);
        CU_ASSERT_EQUAL(res_left, result.left[i]);
        CU_ASSERT_EQUAL(res_right, result.right[i]);
    }

    tsk_table_collection_free(&tables);
    tsk_treeseq_free(&ts);
    tsk_edge_table_free(&result);
}

static void
test_link_ancestors_no_edges(void)
{
    int ret;
    tsk_treeseq_t ts;
    tsk_table_collection_t tables;
    tsk_edge_table_t result;
    tsk_id_t samples[] = { 2 };
    tsk_id_t ancestors[] = { 4 };

    tsk_treeseq_from_text(&ts, 1, single_tree_ex_nodes, single_tree_ex_edges, NULL, NULL,
        NULL, NULL, NULL, 0);
    ret = tsk_treeseq_copy_tables(&ts, &tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_edge_table_init(&result, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_table_collection_link_ancestors(
        &tables, samples, 1, ancestors, 1, 0, &result);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    tsk_table_collection_free(&tables);
    tsk_edge_table_free(&result);
    tsk_treeseq_free(&ts);
}

static void
test_link_ancestors_samples_and_ancestors_overlap(void)
{
    int ret;
    tsk_treeseq_t ts;
    tsk_table_collection_t tables;
    tsk_edge_table_t result;
    tsk_id_t samples[] = { 0, 1, 2, 4 };
    tsk_id_t ancestors[] = { 4 };

    tsk_treeseq_from_text(&ts, 1, single_tree_ex_nodes, single_tree_ex_edges, NULL, NULL,
        NULL, NULL, NULL, 0);
    ret = tsk_treeseq_copy_tables(&ts, &tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_edge_table_init(&result, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_table_collection_link_ancestors(
        &tables, samples, 4, ancestors, 1, 0, &result);

    // tsk_edge_table_print_state(&result, stdout);

    CU_ASSERT_EQUAL_FATAL(ret, 0);
    // Check we get the right result.
    CU_ASSERT_EQUAL(result.num_rows, 2);
    size_t i;
    tsk_id_t res_parent = 4;
    tsk_id_t res_child[] = { 0, 1 };
    double res_left = 0;
    double res_right = 1;
    for (i = 0; i < result.num_rows; i++) {
        CU_ASSERT_EQUAL(res_parent, result.parent[i]);
        CU_ASSERT_EQUAL(res_child[i], result.child[i]);
        CU_ASSERT_EQUAL(res_left, result.left[i]);
        CU_ASSERT_EQUAL(res_right, result.right[i]);
    }

    tsk_table_collection_free(&tables);
    tsk_edge_table_free(&result);
    tsk_treeseq_free(&ts);
}

static void
test_link_ancestors_paper(void)
{
    int ret;
    tsk_treeseq_t ts;
    tsk_table_collection_t tables;
    tsk_edge_table_t result;
    tsk_id_t samples[] = { 0, 1, 2 };
    tsk_id_t ancestors[] = { 5, 6, 7 };

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);
    ret = tsk_treeseq_copy_tables(&ts, &tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_edge_table_init(&result, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_table_collection_link_ancestors(
        &tables, samples, 3, ancestors, 3, 0, &result);

    // tsk_edge_table_print_state(&result, stdout);

    // Check we get the right result.
    CU_ASSERT_EQUAL(result.num_rows, 6);
    size_t i;
    tsk_id_t res_parent[] = { 5, 5, 6, 6, 7, 7 };
    tsk_id_t res_child[] = { 1, 2, 0, 5, 0, 5 };
    double res_left[] = { 0, 2, 0, 0, 7, 7 };
    double res_right[] = { 10, 10, 7, 7, 10, 10 };
    for (i = 0; i < result.num_rows; i++) {
        CU_ASSERT_EQUAL(res_parent[i], result.parent[i]);
        CU_ASSERT_EQUAL(res_child[i], result.child[i]);
        CU_ASSERT_EQUAL(res_left[i], result.left[i]);
        CU_ASSERT_EQUAL(res_right[i], result.right[i]);
    }

    tsk_table_collection_free(&tables);
    tsk_edge_table_free(&result);
    tsk_treeseq_free(&ts);
}

static void
test_link_ancestors_multiple_to_single_tree(void)
{
    int ret;
    tsk_treeseq_t ts;
    tsk_table_collection_t tables;
    tsk_edge_table_t result;
    tsk_id_t samples[] = { 1, 3 };
    tsk_id_t ancestors[] = { 5 };

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);
    ret = tsk_treeseq_copy_tables(&ts, &tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_edge_table_init(&result, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_table_collection_link_ancestors(
        &tables, samples, 2, ancestors, 1, 0, &result);

    // tsk_edge_table_print_state(&result, stdout);

    // Check we get the right result.
    CU_ASSERT_EQUAL(result.num_rows, 2);
    size_t i;
    tsk_id_t res_parent = 5;
    tsk_id_t res_child[] = { 1, 3 };
    double res_left = 0;
    double res_right = 10;
    for (i = 0; i < result.num_rows; i++) {
        CU_ASSERT_EQUAL(res_parent, result.parent[i]);
        CU_ASSERT_EQUAL(res_child[i], result.child[i]);
        CU_ASSERT_EQUAL(res_left, result.left[i]);
        CU_ASSERT_EQUAL(res_right, result.right[i]);
    }

    tsk_table_collection_free(&tables);
    tsk_edge_table_free(&result);
    tsk_treeseq_free(&ts);
}

static void
test_simplify_tables_drops_indexes(void)
{
    int ret;
    tsk_treeseq_t ts;
    tsk_table_collection_t tables;
    tsk_id_t samples[] = { 0, 1 };

    tsk_treeseq_from_text(&ts, 1, single_tree_ex_nodes, single_tree_ex_edges, NULL, NULL,
        NULL, NULL, NULL, 0);
    ret = tsk_treeseq_copy_tables(&ts, &tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    CU_ASSERT_TRUE(tsk_table_collection_has_index(&tables, 0))
    ret = tsk_table_collection_simplify(&tables, samples, 2, 0, NULL);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_FALSE(tsk_table_collection_has_index(&tables, 0))

    tsk_table_collection_free(&tables);
    tsk_treeseq_free(&ts);
}

static void
test_simplify_empty_tables(void)
{
    int ret;
    tsk_table_collection_t tables;

    ret = tsk_table_collection_init(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tables.sequence_length = 1;

    ret = tsk_table_collection_simplify(&tables, NULL, 0, 0, NULL);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(tables.nodes.num_rows, 0);
    CU_ASSERT_EQUAL_FATAL(tables.edges.num_rows, 0);

    tsk_table_collection_free(&tables);
}

static void
test_simplify_metadata(void)
{
    int ret;
    tsk_table_collection_t tables;

    ret = tsk_table_collection_init(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tables.sequence_length = 10;
    tsk_edge_table_add_row(&tables.edges, 0, 0, 1, 1, "metadata", 8);
    ret = tsk_table_collection_simplify(&tables, NULL, 0, 0, NULL);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_CANT_PROCESS_EDGES_WITH_METADATA);

    tsk_table_collection_free(&tables);
}

static void
test_copy_table_collection(void)
{
    int ret;
    tsk_treeseq_t ts;
    tsk_table_collection_t tables, tables_copy;

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);
    ret = tsk_treeseq_copy_tables(&ts, &tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* Add some migrations, population and provenance */
    ret = tsk_migration_table_add_row(&tables.migrations, 0, 1, 2, 3, 4, 5, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_migration_table_add_row(&tables.migrations, 1, 2, 3, 4, 5, 0, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 1);
    ret = tsk_population_table_add_row(&tables.populations, "metadata", 8);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_population_table_add_row(&tables.populations, "other", 5);
    CU_ASSERT_EQUAL_FATAL(ret, 1);
    ret = tsk_provenance_table_add_row(&tables.provenances, "time", 4, "record", 6);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_provenance_table_add_row(&tables.provenances, "time ", 5, "record ", 7);
    CU_ASSERT_EQUAL_FATAL(ret, 1);

    tsk_table_collection_copy(&tables, &tables_copy, 0);
    CU_ASSERT_TRUE(tsk_table_collection_equals(&tables, &tables_copy));

    tsk_table_collection_free(&tables);
    tsk_table_collection_free(&tables_copy);
    tsk_treeseq_free(&ts);
}

static void
test_sort_tables_drops_indexes_with_options(tsk_flags_t tc_options)
{
    int ret;
    tsk_treeseq_t ts;
    tsk_table_collection_t tables;

    tsk_treeseq_from_text(&ts, 1, single_tree_ex_nodes, single_tree_ex_edges, NULL, NULL,
        NULL, NULL, NULL, 0);
    ret = tsk_treeseq_copy_tables(&ts, &tables, tc_options);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    CU_ASSERT_TRUE(tsk_table_collection_has_index(&tables, 0))
    ret = tsk_table_collection_sort(&tables, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_FALSE(tsk_table_collection_has_index(&tables, 0))

    tsk_table_collection_free(&tables);
    tsk_treeseq_free(&ts);
}

static void
test_sort_tables_drops_indexes(void)
{
    test_sort_tables_drops_indexes_with_options(0);
    test_sort_tables_drops_indexes_with_options(TSK_NO_EDGE_METADATA);
}

static void
test_sort_tables_edge_metadata(void)
{
    int ret;
    tsk_treeseq_t ts;
    tsk_table_collection_t t1, t2;

    tsk_treeseq_from_text(&ts, 1, single_tree_ex_nodes, single_tree_ex_edges, NULL, NULL,
        NULL, NULL, NULL, 0);
    ret = tsk_treeseq_copy_tables(&ts, &t1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    insert_edge_metadata(&t1);
    ret = tsk_table_collection_copy(&t1, &t2, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_equals(&t1, &t2));
    reverse_edges(&t1);
    CU_ASSERT_FALSE(tsk_table_collection_equals(&t1, &t2));
    ret = tsk_table_collection_sort(&t1, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_equals(&t1, &t2));

    tsk_table_collection_free(&t1);
    tsk_table_collection_free(&t2);
    tsk_treeseq_free(&ts);
}

static void
test_sort_tables_no_edge_metadata(void)
{
    int ret;
    tsk_treeseq_t ts;
    tsk_table_collection_t t1, t2;

    tsk_treeseq_from_text(&ts, 1, single_tree_ex_nodes, single_tree_ex_edges, NULL, NULL,
        NULL, NULL, NULL, 0);
    ret = tsk_treeseq_copy_tables(&ts, &t1, TSK_NO_EDGE_METADATA);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_FATAL(t1.edges.options & TSK_NO_EDGE_METADATA);
    ret = tsk_table_collection_copy(&t1, &t2, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_FALSE(t2.edges.options & TSK_NO_EDGE_METADATA);
    CU_ASSERT_TRUE(tsk_table_collection_equals(&t1, &t2));
    reverse_edges(&t1);
    CU_ASSERT_FALSE(tsk_table_collection_equals(&t1, &t2));
    ret = tsk_table_collection_sort(&t1, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_equals(&t1, &t2));
    tsk_table_collection_free(&t2);

    ret = tsk_table_collection_copy(&t1, &t2, TSK_NO_EDGE_METADATA);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(t1.edges.options & TSK_NO_EDGE_METADATA);
    CU_ASSERT_TRUE(t2.edges.options & TSK_NO_EDGE_METADATA);
    CU_ASSERT_TRUE(tsk_table_collection_equals(&t1, &t2));
    reverse_edges(&t1);
    CU_ASSERT_FALSE(tsk_table_collection_equals(&t1, &t2));
    ret = tsk_table_collection_sort(&t1, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_equals(&t1, &t2));
    tsk_table_collection_free(&t2);

    tsk_table_collection_free(&t1);
    tsk_treeseq_free(&ts);
}

static void
test_sort_tables_errors(void)
{
    int ret;
    tsk_treeseq_t ts;
    tsk_table_collection_t tables;
    tsk_bookmark_t pos;

    tsk_treeseq_from_text(&ts, 1, single_tree_ex_nodes, single_tree_ex_edges, NULL, NULL,
        NULL, NULL, NULL, 0);
    ret = tsk_treeseq_copy_tables(&ts, &tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    memset(&pos, 0, sizeof(pos));
    /* Everything 0 should be fine */
    ret = tsk_table_collection_sort(&tables, &pos, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* Everything is sorted already */
    pos.edges = tables.edges.num_rows;
    ret = tsk_table_collection_sort(&tables, &pos, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    pos.edges = (tsk_size_t) -1;
    ret = tsk_table_collection_sort(&tables, &pos, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_EDGE_OUT_OF_BOUNDS);

    pos.edges = tables.edges.num_rows + 1;
    ret = tsk_table_collection_sort(&tables, &pos, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_EDGE_OUT_OF_BOUNDS);

    /* Individual, node, population and provenance positions are ignored */
    memset(&pos, 0, sizeof(pos));
    pos.individuals = 1;
    ret = tsk_table_collection_sort(&tables, &pos, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    memset(&pos, 0, sizeof(pos));
    pos.nodes = 1;
    ret = tsk_table_collection_sort(&tables, &pos, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    memset(&pos, 0, sizeof(pos));
    pos.populations = 1;
    ret = tsk_table_collection_sort(&tables, &pos, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    memset(&pos, 0, sizeof(pos));
    pos.provenances = 1;
    ret = tsk_table_collection_sort(&tables, &pos, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* Setting migrations, sites or mutations gives a BAD_PARAM. See
     * github.com/tskit-dev/tskit/issues/101 */
    memset(&pos, 0, sizeof(pos));
    pos.migrations = 1;
    ret = tsk_table_collection_sort(&tables, &pos, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_SORT_OFFSET_NOT_SUPPORTED);

    memset(&pos, 0, sizeof(pos));
    pos.sites = 1;
    ret = tsk_table_collection_sort(&tables, &pos, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_SORT_OFFSET_NOT_SUPPORTED);

    memset(&pos, 0, sizeof(pos));
    pos.mutations = 1;
    ret = tsk_table_collection_sort(&tables, &pos, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_SORT_OFFSET_NOT_SUPPORTED);

    /* Migrations are not supported */
    tsk_migration_table_add_row(&tables.migrations, 0, 1, 0, 0, 0, 0, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(tables.migrations.num_rows, 1);
    ret = tsk_table_collection_sort(&tables, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_SORT_MIGRATIONS_NOT_SUPPORTED);

    tsk_table_collection_free(&tables);
    tsk_treeseq_free(&ts);
}

static void
test_sorter_interface(void)
{
    int ret;
    tsk_treeseq_t ts;
    tsk_table_collection_t tables;
    tsk_table_sorter_t sorter;

    tsk_treeseq_from_text(&ts, 1, single_tree_ex_nodes, single_tree_ex_edges, NULL, NULL,
        NULL, NULL, NULL, 0);
    ret = tsk_treeseq_copy_tables(&ts, &tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    CU_ASSERT_TRUE(tsk_table_collection_equals(ts.tables, &tables));

    /* Nominal case */
    reverse_edges(&tables);
    CU_ASSERT_FALSE(tsk_table_collection_equals(ts.tables, &tables));
    ret = tsk_table_sorter_init(&sorter, &tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_sorter_run(&sorter, NULL);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_equals(ts.tables, &tables));
    CU_ASSERT_EQUAL(sorter.user_data, NULL);
    tsk_table_sorter_free(&sorter);

    /* If we set the sort_edges function to NULL then we should leave the
     * node table as is. */
    reverse_edges(&tables);
    CU_ASSERT_FALSE(tsk_edge_table_equals(&ts.tables->edges, &tables.edges));
    ret = tsk_table_sorter_init(&sorter, &tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    sorter.sort_edges = NULL;
    ret = tsk_table_sorter_run(&sorter, NULL);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_FALSE(tsk_edge_table_equals(&ts.tables->edges, &tables.edges));
    tsk_table_sorter_free(&sorter);

    /* Reversing again should make them equal */
    reverse_edges(&tables);
    CU_ASSERT_TRUE(tsk_edge_table_equals(&ts.tables->edges, &tables.edges));

    /* Do not check integrity before sorting */
    reverse_edges(&tables);
    CU_ASSERT_FALSE(tsk_table_collection_equals(ts.tables, &tables));
    ret = tsk_table_sorter_init(&sorter, &tables, TSK_NO_CHECK_INTEGRITY);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_sorter_run(&sorter, NULL);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_equals(ts.tables, &tables));
    tsk_table_sorter_free(&sorter);

    /* The user_data shouldn't be touched */
    reverse_edges(&tables);
    CU_ASSERT_FALSE(tsk_table_collection_equals(ts.tables, &tables));
    ret = tsk_table_sorter_init(&sorter, &tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    sorter.user_data = (void *) &ts;
    ret = tsk_table_sorter_run(&sorter, NULL);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_equals(ts.tables, &tables));
    CU_ASSERT_EQUAL_FATAL(sorter.user_data, &ts);
    tsk_table_sorter_free(&sorter);

    tsk_table_collection_free(&tables);
    tsk_treeseq_free(&ts);
}

static void
test_dump_unindexed_with_options(tsk_flags_t tc_options)
{
    tsk_table_collection_t tables, loaded;
    int ret;

    ret = tsk_table_collection_init(&tables, tc_options);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    tables.sequence_length = 1;
    parse_nodes(single_tree_ex_nodes, &tables.nodes);
    CU_ASSERT_EQUAL_FATAL(tables.nodes.num_rows, 7);
    parse_edges(single_tree_ex_edges, &tables.edges);
    CU_ASSERT_EQUAL_FATAL(tables.edges.num_rows, 6);
    CU_ASSERT_FALSE(tsk_table_collection_has_index(&tables, 0));
    ret = tsk_table_collection_dump(&tables, _tmp_file_name, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_has_index(&tables, 0));

    ret = tsk_table_collection_load(&loaded, _tmp_file_name, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_has_index(&loaded, 0));
    CU_ASSERT_TRUE(tsk_node_table_equals(&tables.nodes, &loaded.nodes));
    CU_ASSERT_TRUE(tsk_edge_table_equals(&tables.edges, &loaded.edges));

    tsk_table_collection_free(&loaded);
    tsk_table_collection_free(&tables);
}

static void
test_dump_unindexed(void)
{
    test_dump_unindexed_with_options(0);
    test_dump_unindexed_with_options(TSK_NO_EDGE_METADATA);
}

static void
test_dump_load_empty_with_options(tsk_flags_t tc_options)
{
    int ret;
    tsk_table_collection_t t1, t2;

    ret = tsk_table_collection_init(&t1, tc_options);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    t1.sequence_length = 1.0;
    ret = tsk_table_collection_dump(&t1, _tmp_file_name, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_load(&t2, _tmp_file_name, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_equals(&t1, &t2));

    tsk_table_collection_free(&t1);
    tsk_table_collection_free(&t2);
}

static void
test_dump_load_empty(void)
{
    test_dump_load_empty_with_options(0);
    test_dump_load_empty_with_options(TSK_NO_EDGE_METADATA);
}

static void
test_dump_load_unsorted_with_options(tsk_flags_t tc_options)
{
    int ret;
    tsk_table_collection_t t1, t2;
    /* tsk_treeseq_t ts; */

    ret = tsk_table_collection_init(&t1, tc_options);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    t1.sequence_length = 1.0;

    ret = tsk_node_table_add_row(
        &t1.nodes, TSK_NODE_IS_SAMPLE, 0, TSK_NULL, TSK_NULL, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_node_table_add_row(
        &t1.nodes, TSK_NODE_IS_SAMPLE, 0, TSK_NULL, TSK_NULL, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 1);
    ret = tsk_node_table_add_row(
        &t1.nodes, TSK_NODE_IS_SAMPLE, 0, TSK_NULL, TSK_NULL, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 2);
    ret = tsk_node_table_add_row(
        &t1.nodes, TSK_NODE_IS_SAMPLE, 1, TSK_NULL, TSK_NULL, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 3);
    ret = tsk_node_table_add_row(
        &t1.nodes, TSK_NODE_IS_SAMPLE, 2, TSK_NULL, TSK_NULL, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 4);

    ret = tsk_edge_table_add_row(&t1.edges, 0, 1, 3, 0, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_edge_table_add_row(&t1.edges, 0, 1, 4, 3, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 1);
    ret = tsk_edge_table_add_row(&t1.edges, 0, 1, 3, 1, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 2);
    ret = tsk_edge_table_add_row(&t1.edges, 0, 1, 4, 2, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 3);

    /* Verify that it's unsorted */
    ret = tsk_table_collection_check_integrity(
        &t1, TSK_CHECK_OFFSETS | TSK_CHECK_EDGE_ORDERING);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_EDGES_NOT_SORTED_PARENT_TIME);

    /* Indexing should fail */
    ret = tsk_table_collection_build_index(&t1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_EDGES_NOT_SORTED_PARENT_TIME);
    CU_ASSERT_FALSE(tsk_table_collection_has_index(&t1, 0));

    /* Trying to dump without first sorting should also fail */
    ret = tsk_table_collection_dump(&t1, _tmp_file_name, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_EDGES_NOT_SORTED_PARENT_TIME);

    ret = tsk_table_collection_dump(&t1, _tmp_file_name, TSK_NO_BUILD_INDEXES);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_FALSE(tsk_table_collection_has_index(&t1, 0));
    ret = tsk_table_collection_load(&t2, _tmp_file_name, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_equals(&t1, &t2));
    CU_ASSERT_FALSE(tsk_table_collection_has_index(&t1, 0));
    CU_ASSERT_FALSE(tsk_table_collection_has_index(&t2, 0));

    tsk_table_collection_free(&t1);
    tsk_table_collection_free(&t2);
}

static void
test_dump_load_unsorted(void)
{
    test_dump_load_unsorted_with_options(0);
    test_dump_load_unsorted_with_options(TSK_NO_EDGE_METADATA);
}

static void
test_dump_load_metadata_schema(void)
{
    int ret;
    tsk_table_collection_t t1, t2;

    ret = tsk_table_collection_init(&t1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    t1.sequence_length = 1.0;
    char example[100] = "An example of metadata schema with unicode 🎄🌳🌴🌲🎋";
    tsk_size_t example_length = (tsk_size_t) strlen(example) + 4;
    tsk_node_table_set_metadata_schema(
        &t1.nodes, strcat(example, "node"), example_length);
    tsk_edge_table_set_metadata_schema(
        &t1.edges, strcat(example, "edge"), example_length);
    tsk_site_table_set_metadata_schema(
        &t1.sites, strcat(example, "site"), example_length);
    tsk_mutation_table_set_metadata_schema(
        &t1.mutations, strcat(example, "muta"), example_length);
    tsk_migration_table_set_metadata_schema(
        &t1.migrations, strcat(example, "migr"), example_length);
    tsk_individual_table_set_metadata_schema(
        &t1.individuals, strcat(example, "indi"), example_length);
    tsk_population_table_set_metadata_schema(
        &t1.populations, strcat(example, "popu"), example_length);
    ret = tsk_table_collection_dump(&t1, _tmp_file_name, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_load(&t2, _tmp_file_name, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_equals(&t1, &t2));

    tsk_table_collection_free(&t1);
    tsk_table_collection_free(&t2);
}

static void
test_dump_fail_no_file(void)
{
    int ret;
    tsk_table_collection_t t1;

    ret = tsk_table_collection_init(&t1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    t1.sequence_length = 1.0;

    ret = tsk_node_table_add_row(
        &t1.nodes, TSK_NODE_IS_SAMPLE, 0, TSK_NULL, TSK_NULL, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_node_table_add_row(
        &t1.nodes, TSK_NODE_IS_SAMPLE, 0, TSK_NULL, TSK_NULL, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 1);
    ret = tsk_node_table_add_row(
        &t1.nodes, TSK_NODE_IS_SAMPLE, 0, TSK_NULL, TSK_NULL, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 2);
    ret = tsk_node_table_add_row(
        &t1.nodes, TSK_NODE_IS_SAMPLE, 1, TSK_NULL, TSK_NULL, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 3);
    ret = tsk_node_table_add_row(
        &t1.nodes, TSK_NODE_IS_SAMPLE, 2, TSK_NULL, TSK_NULL, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 4);

    ret = tsk_edge_table_add_row(&t1.edges, 0, 1, 3, 0, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_edge_table_add_row(&t1.edges, 0, 1, 4, 3, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 1);
    ret = tsk_edge_table_add_row(&t1.edges, 0, 1, 3, 1, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 2);
    ret = tsk_edge_table_add_row(&t1.edges, 0, 1, 4, 2, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 3);

    /* Verify that it's unsorted */
    ret = tsk_table_collection_check_integrity(
        &t1, TSK_CHECK_OFFSETS | TSK_CHECK_EDGE_ORDERING);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_EDGES_NOT_SORTED_PARENT_TIME);

    /* Make sure the file doesn't exist beforehand. */
    unlink(_tmp_file_name);
    errno = 0;

    /* Trying to dump without first sorting fails */
    ret = tsk_table_collection_dump(&t1, _tmp_file_name, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_EDGES_NOT_SORTED_PARENT_TIME);
    CU_ASSERT_EQUAL(access(_tmp_file_name, F_OK), -1);

    tsk_table_collection_free(&t1);
}

static void
test_load_reindex(void)
{
    int ret;
    tsk_treeseq_t ts;
    tsk_table_collection_t tables;

    tsk_treeseq_from_text(&ts, 1, single_tree_ex_nodes, single_tree_ex_edges, NULL, NULL,
        NULL, NULL, NULL, 0);
    ret = tsk_treeseq_dump(&ts, _tmp_file_name, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_load(&tables, _tmp_file_name, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_drop_index(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_FALSE(tsk_table_collection_has_index(&tables, 0));
    ret = tsk_table_collection_build_index(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_has_index(&tables, 0));

    ret = tsk_table_collection_drop_index(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    /* Dump the unindexed version */
    ret = tsk_table_collection_dump(&tables, _tmp_file_name, TSK_NO_BUILD_INDEXES);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_free(&tables);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_load(&tables, _tmp_file_name, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_FALSE(tsk_table_collection_has_index(&tables, 0));
    ret = tsk_table_collection_build_index(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_has_index(&tables, 0));

    tsk_table_collection_free(&tables);
    tsk_treeseq_free(&ts);
}

static void
test_table_overflow(void)
{
    int ret;
    tsk_table_collection_t tables;
    tsk_size_t max_rows = ((tsk_size_t) INT32_MAX) + 1;

    ret = tsk_table_collection_init(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* Simulate overflows */
    tables.individuals.max_rows = max_rows;
    tables.individuals.num_rows = max_rows;
    ret = tsk_individual_table_add_row(&tables.individuals, 0, 0, 0, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_TABLE_OVERFLOW);

    tables.nodes.max_rows = max_rows;
    tables.nodes.num_rows = max_rows;
    ret = tsk_node_table_add_row(&tables.nodes, 0, 0, 0, 0, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_TABLE_OVERFLOW);

    tables.edges.max_rows = max_rows;
    tables.edges.num_rows = max_rows;
    ret = tsk_edge_table_add_row(&tables.edges, 0, 0, 0, 0, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_TABLE_OVERFLOW);

    tables.migrations.max_rows = max_rows;
    tables.migrations.num_rows = max_rows;
    ret = tsk_migration_table_add_row(&tables.migrations, 0, 0, 0, 0, 0, 0, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_TABLE_OVERFLOW);

    tables.sites.max_rows = max_rows;
    tables.sites.num_rows = max_rows;
    ret = tsk_site_table_add_row(&tables.sites, 0, 0, 0, 0, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_TABLE_OVERFLOW);

    tables.mutations.max_rows = max_rows;
    tables.mutations.num_rows = max_rows;
    ret = tsk_mutation_table_add_row(&tables.mutations, 0, 0, 0, 0, 0, 0, 0, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_TABLE_OVERFLOW);

    tables.provenances.max_rows = max_rows;
    tables.provenances.num_rows = max_rows;
    ret = tsk_provenance_table_add_row(&tables.provenances, 0, 0, 0, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_TABLE_OVERFLOW);

    tables.populations.max_rows = max_rows;
    tables.populations.num_rows = max_rows;
    ret = tsk_population_table_add_row(&tables.populations, 0, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_TABLE_OVERFLOW);

    tsk_table_collection_free(&tables);
}

static void
test_column_overflow(void)
{
    int ret;
    tsk_table_collection_t tables;
    tsk_size_t too_big = ((tsk_size_t) UINT32_MAX);
    double zero = 0;
    char zeros[] = { 0, 0, 0, 0, 0, 0, 0, 0 };

    ret = tsk_table_collection_init(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* We can't trigger a column overflow with one element because the parameter
     * value is 32 bit */
    ret = tsk_individual_table_add_row(&tables.individuals, 0, &zero, 1, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_individual_table_add_row(&tables.individuals, 0, NULL, too_big, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_COLUMN_OVERFLOW);
    ret = tsk_individual_table_add_row(&tables.individuals, 0, NULL, 0, zeros, 1);
    CU_ASSERT_EQUAL_FATAL(ret, 1);
    ret = tsk_individual_table_add_row(&tables.individuals, 0, NULL, 0, NULL, too_big);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_COLUMN_OVERFLOW);

    ret = tsk_node_table_add_row(&tables.nodes, 0, 0, 0, 0, zeros, 1);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_node_table_add_row(&tables.nodes, 0, 0, 0, 0, NULL, too_big);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_COLUMN_OVERFLOW);

    ret = tsk_site_table_add_row(&tables.sites, 0, zeros, 1, zeros, 1);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_site_table_add_row(&tables.sites, 0, NULL, too_big, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_COLUMN_OVERFLOW);
    ret = tsk_site_table_add_row(&tables.sites, 0, NULL, 0, NULL, too_big);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_COLUMN_OVERFLOW);

    ret = tsk_mutation_table_add_row(&tables.mutations, 0, 0, 0, 0, zeros, 1, zeros, 1);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 0, 0, 0, 0, NULL, 0, NULL, too_big);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_COLUMN_OVERFLOW);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 0, 0, 0, 0, NULL, too_big, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_COLUMN_OVERFLOW);

    ret = tsk_provenance_table_add_row(&tables.provenances, zeros, 1, zeros, 1);
    CU_ASSERT_EQUAL_FATAL(ret, 0)
    ret = tsk_provenance_table_add_row(&tables.provenances, NULL, too_big, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_COLUMN_OVERFLOW);
    ret = tsk_provenance_table_add_row(&tables.provenances, NULL, 0, NULL, too_big);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_COLUMN_OVERFLOW);

    ret = tsk_population_table_add_row(&tables.populations, zeros, 1);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_population_table_add_row(&tables.populations, NULL, too_big);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_COLUMN_OVERFLOW);

    tsk_table_collection_free(&tables);
}

static void
test_table_collection_check_integrity_with_options(tsk_flags_t tc_options)
{
    int ret;
    tsk_table_collection_t tables;

    ret = tsk_table_collection_init(&tables, tc_options);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tables.sequence_length = 1;

    /* nodes */
    ret = tsk_node_table_add_row(
        &tables.nodes, TSK_NODE_IS_SAMPLE, INFINITY, TSK_NULL, TSK_NULL, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_TIME_NONFINITE);

    ret = tsk_node_table_clear(&tables.nodes);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_node_table_add_row(
        &tables.nodes, TSK_NODE_IS_SAMPLE, 0.0, 0, TSK_NULL, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_check_integrity(&tables, TSK_NO_CHECK_POPULATION_REFS);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_POPULATION_OUT_OF_BOUNDS);

    ret = tsk_node_table_clear(&tables.nodes);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_node_table_add_row(
        &tables.nodes, TSK_NODE_IS_SAMPLE, 0.0, TSK_NULL, 0, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_INDIVIDUAL_OUT_OF_BOUNDS);

    ret = tsk_node_table_clear(&tables.nodes);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_node_table_add_row(
        &tables.nodes, TSK_NODE_IS_SAMPLE, 0.0, TSK_NULL, TSK_NULL, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_node_table_add_row(
        &tables.nodes, TSK_NODE_IS_SAMPLE, 1.0, TSK_NULL, TSK_NULL, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* edges */
    ret = tsk_edge_table_add_row(&tables.edges, 0.0, 1.0, TSK_NULL, 0, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_NULL_PARENT);

    ret = tsk_edge_table_clear(&tables.edges);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_edge_table_add_row(&tables.edges, 0.0, 1.0, 2, 0, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_NODE_OUT_OF_BOUNDS);

    ret = tsk_edge_table_clear(&tables.edges);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_edge_table_add_row(&tables.edges, 0.0, 1.0, 1, TSK_NULL, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_NULL_CHILD);

    ret = tsk_edge_table_clear(&tables.edges);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_edge_table_add_row(&tables.edges, 0.0, 1.0, 1, 2, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_NODE_OUT_OF_BOUNDS);

    ret = tsk_edge_table_clear(&tables.edges);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_edge_table_add_row(&tables.edges, 0.0, INFINITY, 1, 0, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_GENOME_COORDS_NONFINITE);

    ret = tsk_edge_table_clear(&tables.edges);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_edge_table_add_row(&tables.edges, -1.0, 1.0, 1, 0, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_LEFT_LESS_ZERO);

    ret = tsk_edge_table_clear(&tables.edges);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_edge_table_add_row(&tables.edges, 0.0, 1.1, 1, 0, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_RIGHT_GREATER_SEQ_LENGTH);

    ret = tsk_edge_table_clear(&tables.edges);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_edge_table_add_row(&tables.edges, 0.5, 0.1, 1, 0, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_EDGE_INTERVAL);

    ret = tsk_edge_table_clear(&tables.edges);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_edge_table_add_row(&tables.edges, 0.0, 0.5, 0, 1, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_NODE_TIME_ORDERING);

    ret = tsk_edge_table_clear(&tables.edges);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* sites */
    ret = tsk_site_table_add_row(&tables.sites, INFINITY, "A", 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_SITE_POSITION);

    ret = tsk_site_table_clear(&tables.sites);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_site_table_add_row(&tables.sites, -0.5, "A", 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_SITE_POSITION);

    ret = tsk_site_table_clear(&tables.sites);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_site_table_add_row(&tables.sites, 1.5, "A", 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_SITE_POSITION);

    ret = tsk_site_table_clear(&tables.sites);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_site_table_add_row(&tables.sites, 0.5, "A", 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_site_table_add_row(&tables.sites, 0.5, "A", 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_check_integrity(&tables, TSK_CHECK_SITE_DUPLICATES);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_DUPLICATE_SITE_POSITION);

    ret = tsk_site_table_clear(&tables.sites);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_site_table_add_row(&tables.sites, 0.5, "A", 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_site_table_add_row(&tables.sites, 0.4, "A", 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_check_integrity(&tables, TSK_CHECK_SITE_ORDERING);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_UNSORTED_SITES);

    ret = tsk_site_table_clear(&tables.sites);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_site_table_add_row(&tables.sites, 0.5, "A", 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_site_table_add_row(&tables.sites, 0.6, "A", 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);

    /* mutations */
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 2, 0, TSK_NULL, 0, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_SITE_OUT_OF_BOUNDS);

    ret = tsk_mutation_table_clear(&tables.mutations);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 0, 2, TSK_NULL, 0, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_NODE_OUT_OF_BOUNDS);

    ret = tsk_mutation_table_clear(&tables.mutations);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_mutation_table_add_row(&tables.mutations, 0, 1, 2, 0, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_MUTATION_OUT_OF_BOUNDS);

    ret = tsk_mutation_table_clear(&tables.mutations);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_mutation_table_add_row(&tables.mutations, 0, 1, 0, 0, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_MUTATION_PARENT_EQUAL);

    ret = tsk_mutation_table_clear(&tables.mutations);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_mutation_table_add_row(&tables.mutations, 0, 1, 1, 0, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 0, 1, TSK_NULL, 0, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_check_integrity(&tables, TSK_CHECK_MUTATION_ORDERING);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_MUTATION_PARENT_AFTER_CHILD);
    ret = tsk_table_collection_check_integrity(
        &tables, TSK_CHECK_MUTATION_ORDERING | TSK_NO_CHECK_MUTATION_PARENTS);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_mutation_table_clear(&tables.mutations);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 0, 1, TSK_NULL, 0, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_mutation_table_add_row(&tables.mutations, 1, 1, 0, 0, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_check_integrity(&tables, TSK_CHECK_MUTATION_ORDERING);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_MUTATION_PARENT_DIFFERENT_SITE);
    ret = tsk_table_collection_check_integrity(&tables, TSK_NO_CHECK_MUTATION_PARENTS);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_mutation_table_clear(&tables.mutations);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 1, 1, TSK_NULL, 0, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 0, 1, TSK_NULL, 0, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_check_integrity(&tables, TSK_CHECK_MUTATION_ORDERING);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_UNSORTED_MUTATIONS);

    ret = tsk_mutation_table_clear(&tables.mutations);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 0, 0, TSK_NULL, TSK_UNKNOWN_TIME, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_check_integrity(&tables, TSK_CHECK_MUTATION_TIME);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_mutation_table_clear(&tables.mutations);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 0, 0, TSK_NULL, NAN, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_check_integrity(&tables, TSK_CHECK_MUTATION_TIME);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_TIME_NONFINITE);

    ret = tsk_mutation_table_clear(&tables.mutations);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 0, 0, TSK_NULL, INFINITY, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_check_integrity(&tables, TSK_CHECK_MUTATION_TIME);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_TIME_NONFINITE);

    ret = tsk_mutation_table_clear(&tables.mutations);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 1, 1, TSK_NULL, 0, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_check_integrity(&tables, TSK_CHECK_MUTATION_TIME);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_MUTATION_TIME_YOUNGER_THAN_NODE);

    ret = tsk_mutation_table_clear(&tables.mutations);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 1, 1, TSK_NULL, 1, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_mutation_table_add_row(&tables.mutations, 1, 1, 0, 2, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_check_integrity(&tables, TSK_CHECK_MUTATION_TIME);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_MUTATION_TIME_OLDER_THAN_PARENT_MUTATION);

    /* Note that we do not test for TSK_ERR_MUTATION_TIME_OLDER_THAN_PARENT_NODE as
       that is only checked for tree sequences not tables */

    /* migrations */
    ret = tsk_population_table_add_row(&tables.populations, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_population_table_add_row(&tables.populations, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);

    ret = tsk_migration_table_clear(&tables.migrations);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_migration_table_add_row(
        &tables.migrations, 0.0, 0.5, 2, 0, 1, 1.5, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_NODE_OUT_OF_BOUNDS);

    ret = tsk_migration_table_clear(&tables.migrations);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_migration_table_add_row(
        &tables.migrations, 0.0, 0.5, 1, 2, 1, 1.5, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_POPULATION_OUT_OF_BOUNDS);

    ret = tsk_migration_table_clear(&tables.migrations);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_migration_table_add_row(
        &tables.migrations, 0.0, 0.5, 1, 0, 2, 1.5, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_POPULATION_OUT_OF_BOUNDS);

    ret = tsk_migration_table_clear(&tables.migrations);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_migration_table_add_row(
        &tables.migrations, 0.0, 0.5, 1, 0, 1, INFINITY, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_TIME_NONFINITE);

    ret = tsk_migration_table_clear(&tables.migrations);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_migration_table_add_row(
        &tables.migrations, 0.0, INFINITY, 1, 0, 1, 1.5, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_GENOME_COORDS_NONFINITE);

    ret = tsk_migration_table_clear(&tables.migrations);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_migration_table_add_row(
        &tables.migrations, -0.3, 0.5, 1, 0, 1, 1.5, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_LEFT_LESS_ZERO);

    ret = tsk_migration_table_clear(&tables.migrations);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_migration_table_add_row(
        &tables.migrations, 0.0, 1.5, 1, 0, 1, 1.5, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_RIGHT_GREATER_SEQ_LENGTH);

    ret = tsk_migration_table_clear(&tables.migrations);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_migration_table_add_row(
        &tables.migrations, 0.6, 0.5, 1, 0, 1, 1.5, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_EDGE_INTERVAL);

    tsk_table_collection_free(&tables);
}

static void
test_table_collection_check_integrity(void)
{
    test_table_collection_check_integrity_with_options(0);
    test_table_collection_check_integrity_with_options(TSK_NO_EDGE_METADATA);
}

static void
test_table_collection_subset_with_options(tsk_flags_t options)
{
    int ret;
    tsk_table_collection_t tables;
    tsk_table_collection_t tables_copy;
    int k;
    tsk_id_t nodes[4];

    ret = tsk_table_collection_init(&tables, options);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tables.sequence_length = 1;
    ret = tsk_table_collection_init(&tables_copy, options);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    // does not error on empty tables
    ret = tsk_table_collection_subset(&tables, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    // four nodes from two diploids; the first is from pop 0
    ret = tsk_node_table_add_row(&tables.nodes, TSK_NODE_IS_SAMPLE, 0.0, 0, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_node_table_add_row(&tables.nodes, TSK_NODE_IS_SAMPLE, 0.0, 0, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_node_table_add_row(
        &tables.nodes, TSK_NODE_IS_SAMPLE, 0.0, TSK_NULL, 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_node_table_add_row(
        &tables.nodes, TSK_NODE_IS_SAMPLE, 0.0, TSK_NULL, 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_individual_table_add_row(&tables.individuals, 0, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_individual_table_add_row(&tables.individuals, 0, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_population_table_add_row(&tables.populations, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_edge_table_add_row(&tables.edges, 0.0, 1.0, 1, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_edge_table_add_row(&tables.edges, 0.0, 1.0, 2, 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_site_table_add_row(&tables.sites, 0.2, "A", 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_site_table_add_row(&tables.sites, 0.4, "A", 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 0, 0, TSK_NULL, NAN, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_mutation_table_add_row(&tables.mutations, 0, 0, 0, NAN, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 1, 1, TSK_NULL, NAN, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);

    // empty nodes should get empty tables
    ret = tsk_table_collection_copy(&tables, &tables_copy, TSK_NO_INIT | options);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_subset(&tables_copy, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(tables_copy.nodes.num_rows, 0);
    CU_ASSERT_EQUAL_FATAL(tables_copy.individuals.num_rows, 0);
    CU_ASSERT_EQUAL_FATAL(tables_copy.populations.num_rows, 0);

    // the identity transformation
    for (k = 0; k < 4; k++) {
        nodes[k] = k;
    }
    ret = tsk_table_collection_copy(&tables, &tables_copy, TSK_NO_INIT | options);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_subset(&tables_copy, nodes, 4);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_FATAL(tsk_table_collection_equals(&tables, &tables_copy));

    // reverse twice should get back to the start
    for (k = 0; k < 4; k++) {
        nodes[k] = 3 - k;
    }
    ret = tsk_table_collection_copy(&tables, &tables_copy, TSK_NO_INIT | options);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_subset(&tables_copy, nodes, 4);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_subset(&tables_copy, nodes, 4);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_FATAL(tsk_table_collection_equals(&tables, &tables_copy));

    tsk_table_collection_free(&tables_copy);
    tsk_table_collection_free(&tables);
}

static void
test_table_collection_subset(void)
{
    test_table_collection_subset_with_options(0);
    test_table_collection_subset_with_options(TSK_NO_EDGE_METADATA);
}

static void
test_table_collection_subset_errors(void)
{
    int ret;
    tsk_table_collection_t tables;
    tsk_table_collection_t tables_copy;
    tsk_id_t nodes[4] = { 0, 1, 2, 3 };

    ret = tsk_table_collection_init(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_init(&tables_copy, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    // four nodes from two diploids; the first is from pop 0
    ret = tsk_node_table_add_row(&tables.nodes, TSK_NODE_IS_SAMPLE, 0.0, 0, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_node_table_add_row(&tables.nodes, TSK_NODE_IS_SAMPLE, 0.0, 0, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_node_table_add_row(
        &tables.nodes, TSK_NODE_IS_SAMPLE, 0.0, TSK_NULL, 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_node_table_add_row(
        &tables.nodes, TSK_NODE_IS_SAMPLE, 0.0, TSK_NULL, 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_individual_table_add_row(&tables.individuals, 0, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_individual_table_add_row(&tables.individuals, 0, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_population_table_add_row(&tables.populations, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_edge_table_add_row(&tables.edges, 0.0, 1.0, 1, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);

    /* Migrations are not supported */
    ret = tsk_table_collection_copy(&tables, &tables_copy, TSK_NO_INIT);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tsk_migration_table_add_row(&tables_copy.migrations, 0, 1, 0, 0, 0, 0, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(tables_copy.migrations.num_rows, 1);
    ret = tsk_table_collection_subset(&tables_copy, nodes, 4);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_MIGRATIONS_NOT_SUPPORTED);

    // test out of bounds nodes
    nodes[0] = -1;
    ret = tsk_table_collection_subset(&tables, nodes, 4);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_NODE_OUT_OF_BOUNDS);
    nodes[0] = 6;
    ret = tsk_table_collection_subset(&tables, nodes, 4);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_NODE_OUT_OF_BOUNDS);

    tsk_table_collection_free(&tables);
    tsk_table_collection_free(&tables_copy);
}

int
main(int argc, char **argv)
{
    CU_TestInfo tests[] = {
        { "test_node_table", test_node_table },
        { "test_edge_table", test_edge_table },
        { "test_edge_table_copy_semantics", test_edge_table_copy_semantics },
        { "test_edge_table_squash", test_edge_table_squash },
        { "test_edge_table_squash_multiple_parents",
            test_edge_table_squash_multiple_parents },
        { "test_edge_table_squash_empty", test_edge_table_squash_empty },
        { "test_edge_table_squash_single_edge", test_edge_table_squash_single_edge },
        { "test_edge_table_squash_bad_intervals", test_edge_table_squash_bad_intervals },
        { "test_edge_table_squash_metadata", test_edge_table_squash_metadata },
        { "test_site_table", test_site_table },
        { "test_mutation_table", test_mutation_table },
        { "test_migration_table", test_migration_table },
        { "test_individual_table", test_individual_table },
        { "test_population_table", test_population_table },
        { "test_provenance_table", test_provenance_table },
        { "test_table_size_increments", test_table_size_increments },
        { "test_table_collection_simplify_errors",
            test_table_collection_simplify_errors },
        { "test_table_collection_metadata", test_table_collection_metadata },
        { "test_simplify_tables_drops_indexes", test_simplify_tables_drops_indexes },
        { "test_simplify_empty_tables", test_simplify_empty_tables },
        { "test_simplify_metadata", test_simplify_metadata },
        { "test_link_ancestors_no_edges", test_link_ancestors_no_edges },
        { "test_link_ancestors_input_errors", test_link_ancestors_input_errors },
        { "test_link_ancestors_single_tree", test_link_ancestors_single_tree },
        { "test_link_ancestors_paper", test_link_ancestors_paper },
        { "test_link_ancestors_samples_and_ancestors_overlap",
            test_link_ancestors_samples_and_ancestors_overlap },
        { "test_link_ancestors_multiple_to_single_tree",
            test_link_ancestors_multiple_to_single_tree },
        { "test_sort_tables_drops_indexes", test_sort_tables_drops_indexes },
        { "test_sort_tables_edge_metadata", test_sort_tables_edge_metadata },
        { "test_sort_tables_no_edge_metadata", test_sort_tables_no_edge_metadata },
        { "test_copy_table_collection", test_copy_table_collection },
        { "test_sort_tables_errors", test_sort_tables_errors },
        { "test_sorter_interface", test_sorter_interface },
        { "test_dump_unindexed", test_dump_unindexed },
        { "test_dump_load_empty", test_dump_load_empty },
        { "test_dump_load_unsorted", test_dump_load_unsorted },
        { "test_dump_load_metadata_schema", test_dump_load_metadata_schema },
        { "test_dump_fail_no_file", test_dump_fail_no_file },
        { "test_load_reindex", test_load_reindex },
        { "test_table_overflow", test_table_overflow },
        { "test_column_overflow", test_column_overflow },
        { "test_table_collection_check_integrity",
            test_table_collection_check_integrity },
        { "test_table_collection_subset", test_table_collection_subset },
        { "test_table_collection_subset_errors", test_table_collection_subset_errors },
        { NULL, NULL },
    };

    return test_main(tests, argc, argv);
}
