/*
 * MIT License
 *
 * Copyright (c) 2019-2020 Tskit Developers
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

#include <float.h>
#include <unistd.h>
#include <stdlib.h>

static void
reverse_migrations(tsk_table_collection_t *tables)
{
    int ret;
    tsk_migration_table_t migrations;
    tsk_migration_t migration;
    tsk_id_t j;

    /* Easy way to copy the metadata schema */
    ret = tsk_migration_table_copy(&tables->migrations, &migrations, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_migration_table_clear(&migrations);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    for (j = (tsk_id_t) tables->migrations.num_rows - 1; j >= 0; j--) {
        ret = tsk_migration_table_get_row(&tables->migrations, j, &migration);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        ret = tsk_migration_table_add_row(&migrations, migration.left, migration.right,
            migration.node, migration.source, migration.dest, migration.time,
            migration.metadata, migration.metadata_length);
        CU_ASSERT_FATAL(ret >= 0);
    }

    ret = tsk_migration_table_copy(&migrations, &tables->migrations, TSK_NO_INIT);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    tsk_migration_table_free(&migrations);
}

static void
reverse_edges(tsk_table_collection_t *tables)
{
    int ret;
    tsk_edge_table_t edges;
    tsk_edge_t edge;
    tsk_id_t j;

    /* Easy way to copy the metadata schema */
    ret = tsk_edge_table_copy(&tables->edges, &edges, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_edge_table_clear(&edges);
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
reverse_mutations(tsk_table_collection_t *tables)
{
    int ret;
    tsk_mutation_table_t mutations;
    tsk_mutation_t mutation;
    tsk_id_t j;

    ret = tsk_mutation_table_init(&mutations, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    for (j = (tsk_id_t) tables->mutations.num_rows - 1; j >= 0; j--) {
        ret = tsk_mutation_table_get_row(&tables->mutations, j, &mutation);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        ret = tsk_mutation_table_add_row(&mutations, mutation.site, mutation.node,
            mutation.parent, mutation.time, mutation.derived_state,
            mutation.derived_state_length, mutation.metadata, mutation.metadata_length);
        CU_ASSERT_FATAL(ret >= 0);
    }

    ret = tsk_mutation_table_copy(&mutations, &tables->mutations, TSK_NO_INIT);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    tsk_mutation_table_free(&mutations);
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
test_table_collection_equals_options(void)
{
    int ret;
    tsk_table_collection_t tc1, tc2;

    char example_metadata[100] = "An example of metadata with unicode 🎄🌳🌴🌲🎋";
    char example_metadata_schema[100]
        = "An example of metadata schema with unicode 🎄🌳🌴🌲🎋";
    tsk_size_t example_metadata_length = (tsk_size_t) strlen(example_metadata);
    tsk_size_t example_metadata_schema_length
        = (tsk_size_t) strlen(example_metadata_schema);

    // Test equality empty tables
    ret = tsk_table_collection_init(&tc1, 0);
    CU_ASSERT_EQUAL(ret, 0);
    ret = tsk_table_collection_init(&tc2, 0);
    CU_ASSERT_EQUAL(ret, 0);
    ret = tsk_table_collection_equals(&tc1, &tc2, 0);
    CU_ASSERT_TRUE(ret);

    // Adding some meat to the tables
    ret = tsk_node_table_add_row(&tc1.nodes, TSK_NODE_IS_SAMPLE, 0.0, 0, 0, NULL, 0);
    CU_ASSERT(ret >= 0);
    ret = tsk_node_table_add_row(&tc1.nodes, TSK_NODE_IS_SAMPLE, 1.0, 0, 0, NULL, 0);
    CU_ASSERT(ret >= 0);
    ret = tsk_individual_table_add_row(&tc1.individuals, 0, NULL, 0, NULL, 0, NULL, 0);
    CU_ASSERT(ret >= 0);
    ret = tsk_population_table_add_row(&tc1.populations, NULL, 0);
    CU_ASSERT(ret >= 0);
    ret = tsk_edge_table_add_row(&tc1.edges, 0.0, 1.0, 1, 0, NULL, 0);
    CU_ASSERT(ret >= 0);
    ret = tsk_site_table_add_row(&tc1.sites, 0.2, "A", 1, NULL, 0);
    CU_ASSERT(ret >= 0);
    ret = tsk_mutation_table_add_row(
        &tc1.mutations, 0, 0, TSK_NULL, TSK_UNKNOWN_TIME, NULL, 0, NULL, 0);
    CU_ASSERT(ret >= 0);

    // Equality of empty vs non-empty
    ret = tsk_table_collection_equals(&tc1, &tc2, 0);
    CU_ASSERT_FALSE(ret);
    ret = tsk_table_collection_copy(&tc1, &tc2, TSK_NO_INIT);
    CU_ASSERT_EQUAL(ret, 0);

    // Equivalent except for metadata
    ret = tsk_table_collection_set_metadata(
        &tc1, example_metadata, example_metadata_length);
    CU_ASSERT_EQUAL(ret, 0);
    ret = tsk_table_collection_equals(&tc1, &tc2, TSK_CMP_IGNORE_TS_METADATA);
    CU_ASSERT_TRUE(ret);
    /* TSK_CMP_IGNORE_METADATA implies TSK_CMP_IGNORE_TS_METADATA */
    ret = tsk_table_collection_equals(&tc1, &tc2, TSK_CMP_IGNORE_METADATA);
    CU_ASSERT_TRUE(ret);
    ret = tsk_table_collection_equals(&tc1, &tc2, 0);
    CU_ASSERT_FALSE(ret);
    ret = tsk_table_collection_equals(&tc1, &tc2, TSK_CMP_IGNORE_PROVENANCE);
    CU_ASSERT_FALSE(ret);
    ret = tsk_table_collection_set_metadata(
        &tc2, example_metadata, example_metadata_length);
    CU_ASSERT_EQUAL(ret, 0);
    ret = tsk_table_collection_equals(&tc1, &tc2, 0);
    CU_ASSERT_TRUE(ret);
    ret = tsk_table_collection_set_metadata_schema(
        &tc1, example_metadata_schema, example_metadata_schema_length);
    CU_ASSERT_EQUAL(ret, 0);
    ret = tsk_table_collection_equals(&tc1, &tc2, TSK_CMP_IGNORE_TS_METADATA);
    CU_ASSERT_TRUE(ret);
    ret = tsk_table_collection_equals(&tc1, &tc2, 0);
    CU_ASSERT_FALSE(ret);
    ret = tsk_table_collection_set_metadata_schema(
        &tc2, example_metadata_schema, example_metadata_schema_length);
    CU_ASSERT_EQUAL(ret, 0);
    ret = tsk_table_collection_equals(&tc1, &tc2, 0);
    CU_ASSERT_TRUE(ret);

    // Ignore provenance
    ret = tsk_provenance_table_add_row(&tc1.provenances, "time", 4, "record", 6);
    CU_ASSERT_EQUAL(ret, 0);
    ret = tsk_table_collection_equals(&tc1, &tc2, TSK_CMP_IGNORE_PROVENANCE);
    CU_ASSERT_TRUE(ret);
    ret = tsk_table_collection_equals(&tc1, &tc2, 0);
    CU_ASSERT_FALSE(ret);
    ret = tsk_table_collection_equals(&tc1, &tc2, TSK_CMP_IGNORE_TS_METADATA);
    CU_ASSERT_FALSE(ret);
    ret = tsk_provenance_table_add_row(&tc2.provenances, "time", 4, "record", 6);
    CU_ASSERT_EQUAL(ret, 0);
    ret = tsk_table_collection_equals(&tc1, &tc2, TSK_CMP_IGNORE_PROVENANCE);
    CU_ASSERT_TRUE(ret);
    ret = tsk_table_collection_equals(&tc1, &tc2, 0);
    CU_ASSERT_TRUE(ret);

    // Ignore provenance timestamp
    ret = tsk_provenance_table_add_row(&tc1.provenances, "time", 4, "record", 6);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_provenance_table_add_row(&tc2.provenances, "other", 5, "record", 6);
    CU_ASSERT_FATAL(ret >= 0);
    CU_ASSERT_FALSE(tsk_table_collection_equals(&tc1, &tc2, 0));
    CU_ASSERT_TRUE(tsk_table_collection_equals(&tc1, &tc2, TSK_CMP_IGNORE_PROVENANCE));
    CU_ASSERT_TRUE(tsk_table_collection_equals(&tc1, &tc2, TSK_CMP_IGNORE_TIMESTAMPS));

    // Ignore provenance and top-level metadata.
    ret = tsk_provenance_table_clear(&tc1.provenances);
    CU_ASSERT_EQUAL(ret, 0);
    example_metadata[0] = 'J';
    ret = tsk_table_collection_set_metadata(
        &tc1, example_metadata, example_metadata_length);
    CU_ASSERT_EQUAL(ret, 0);
    ret = tsk_table_collection_equals(&tc1, &tc2, 0);
    CU_ASSERT_FALSE(ret);
    ret = tsk_table_collection_equals(
        &tc1, &tc2, TSK_CMP_IGNORE_TS_METADATA | TSK_CMP_IGNORE_PROVENANCE);
    CU_ASSERT_TRUE(ret);

    tsk_table_collection_free(&tc1);
    tsk_table_collection_free(&tc2);

    // Check what happens when one of the tables just differs by metadata.
    ret = tsk_table_collection_init(&tc1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_init(&tc2, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_population_table_add_row(&tc1.populations, "metadata", 8);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_population_table_add_row(&tc2.populations, "", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_FALSE(tsk_table_collection_equals(&tc1, &tc2, 0));
    CU_ASSERT_TRUE(tsk_table_collection_equals(&tc1, &tc2, TSK_CMP_IGNORE_METADATA));

    tsk_table_collection_free(&tc1);
    tsk_table_collection_free(&tc2);
}

static void
test_table_collection_simplify_errors(void)
{
    int ret;
    tsk_table_collection_t tables;
    tsk_id_t samples[] = { 0, 1 };
    const char *individuals = "1      0.25     -2\n";
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
    tsk_site_table_truncate(&tables.sites, 0);
    tables.sites.position[0] = 0;

    /* Individual out of bounds */
    parse_individuals(individuals, &tables.individuals);
    CU_ASSERT_EQUAL_FATAL(tables.individuals.num_rows, 1);
    ret = tsk_table_collection_simplify(&tables, samples, 0, 0, NULL);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_INDIVIDUAL_OUT_OF_BOUNDS);

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
    CU_ASSERT_TRUE(tsk_table_collection_equals(&tc1, &tc2, 0));
    ret = tsk_table_collection_set_metadata(
        &tc1, example_metadata, example_metadata_length);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_FALSE(tsk_table_collection_equals(&tc1, &tc2, 0));
    ret = tsk_table_collection_set_metadata(
        &tc2, example_metadata, example_metadata_length);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_equals(&tc1, &tc2, 0));
    ret = tsk_table_collection_set_metadata_schema(
        &tc1, example_metadata_schema, example_metadata_schema_length);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_FALSE(tsk_table_collection_equals(&tc1, &tc2, 0));
    ret = tsk_table_collection_set_metadata_schema(
        &tc2, example_metadata_schema, example_metadata_schema_length);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_equals(&tc1, &tc2, 0));

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
    CU_ASSERT_TRUE(tsk_table_collection_equals(&tc1, &tc2, 0));

    ret = tsk_table_collection_set_metadata_schema(
        &tc1, example_metadata_schema, example_metadata_schema_length);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tsk_table_collection_free(&tc2);
    ret = tsk_table_collection_copy(&tc1, &tc2, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_equals(&tc1, &tc2, 0));

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
    CU_ASSERT_TRUE(tsk_table_collection_equals(&tc1, &tc2, 0));

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
    CU_ASSERT_TRUE(tsk_table_collection_equals(&tc1, &tc2, 0));
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
    ret = tsk_node_table_dump_text(&table, _devnull);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

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

    /* Test equality with and without metadata */
    tsk_node_table_copy(&table, &table2, 0);
    CU_ASSERT_TRUE(tsk_node_table_equals(&table, &table2, 0));
    CU_ASSERT_TRUE(tsk_node_table_equals(&table, &table2, TSK_CMP_IGNORE_METADATA));
    /* Change the metadata values */
    table2.metadata[0] = 0;
    CU_ASSERT_FALSE(tsk_node_table_equals(&table, &table2, 0));
    CU_ASSERT_TRUE(tsk_node_table_equals(&table, &table2, TSK_CMP_IGNORE_METADATA));
    /* Change the last metadata entry */
    table2.metadata_offset[table2.num_rows]
        = table2.metadata_offset[table2.num_rows - 1];
    CU_ASSERT_FALSE(tsk_node_table_equals(&table, &table2, 0));
    CU_ASSERT_TRUE(tsk_node_table_equals(&table, &table2, TSK_CMP_IGNORE_METADATA));
    /* Delete all metadata */
    memset(table2.metadata_offset, 0,
        (table2.num_rows + 1) * sizeof(*table2.metadata_offset));
    CU_ASSERT_FALSE(tsk_node_table_equals(&table, &table2, 0));
    CU_ASSERT_TRUE(tsk_node_table_equals(&table, &table2, TSK_CMP_IGNORE_METADATA));
    tsk_node_table_free(&table2);

    CU_ASSERT_EQUAL(tsk_node_table_get_row(&table, (tsk_id_t) num_rows, &node),
        TSK_ERR_NODE_OUT_OF_BOUNDS);
    tsk_node_table_print_state(&table, _devnull);
    ret = tsk_node_table_dump_text(&table, _devnull);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

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
    ret = tsk_node_table_dump_text(&table, _devnull);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

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
    ret = tsk_node_table_dump_text(&table, _devnull);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

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
    ret = tsk_node_table_dump_text(&table, _devnull);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

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
    CU_ASSERT_TRUE(tsk_node_table_equals(&table, &table2, 0));
    tsk_node_table_set_metadata_schema(&table2, example2, example2_length);
    CU_ASSERT_FALSE(tsk_node_table_equals(&table, &table2, 0));
    CU_ASSERT_TRUE(tsk_node_table_equals(&table, &table2, TSK_CMP_IGNORE_METADATA));

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
    ret = tsk_edge_table_dump_text(&table, _devnull);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

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
    ret = tsk_edge_table_dump_text(&table, _devnull);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

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

    /* Test equality with and without metadata */
    tsk_edge_table_copy(&table, &table2, 0);
    CU_ASSERT_TRUE(tsk_edge_table_equals(&table, &table2, 0));
    CU_ASSERT_TRUE(tsk_edge_table_equals(&table, &table2, TSK_CMP_IGNORE_METADATA));
    if (!(options & TSK_NO_METADATA)) {
        /* Change the metadata values */
        table2.metadata[0] = 0;
        CU_ASSERT_FALSE(tsk_edge_table_equals(&table, &table2, 0));
        CU_ASSERT_TRUE(tsk_edge_table_equals(&table, &table2, TSK_CMP_IGNORE_METADATA));
        /* Change the last metadata entry */
        table2.metadata_offset[table2.num_rows]
            = table2.metadata_offset[table2.num_rows - 1];
        CU_ASSERT_FALSE(tsk_edge_table_equals(&table, &table2, 0));
        CU_ASSERT_TRUE(tsk_edge_table_equals(&table, &table2, TSK_CMP_IGNORE_METADATA));
        /* Delete all metadata */
        memset(table2.metadata_offset, 0,
            (table2.num_rows + 1) * sizeof(*table2.metadata_offset));
        CU_ASSERT_FALSE(tsk_edge_table_equals(&table, &table2, 0));
        CU_ASSERT_TRUE(tsk_edge_table_equals(&table, &table2, TSK_CMP_IGNORE_METADATA));
    }
    tsk_edge_table_free(&table2);

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
    ret = tsk_edge_table_dump_text(&table, _devnull);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

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
    CU_ASSERT_TRUE(tsk_edge_table_equals(&table, &table2, 0));
    ret = tsk_edge_table_set_metadata_schema(&table2, example2, example2_length);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_FALSE(tsk_edge_table_equals(&table, &table2, 0));
    CU_ASSERT_TRUE(tsk_edge_table_equals(&table, &table2, TSK_CMP_IGNORE_METADATA));

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
    CU_ASSERT_TRUE(tsk_table_collection_equals(&t1, &t2, 0));
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
    CU_ASSERT_TRUE(tsk_table_collection_equals(&t1, &t2, 0));
    tsk_table_collection_free(&t2);

    ret = tsk_table_collection_copy(&t1, &t2, TSK_NO_EDGE_METADATA);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_equals(&t1, &t2, 0));
    tsk_table_collection_free(&t2);

    /* Try copying into a table directly */
    ret = tsk_edge_table_copy(&t1.edges, &edges, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_edge_table_equals(&t1.edges, &edges, 0));
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
    ret = tsk_site_table_dump_text(&table, _devnull);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

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
    ret = tsk_site_table_dump_text(&table, _devnull);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
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

    /* Test equality with and without metadata */
    tsk_site_table_copy(&table, &table2, 0);
    CU_ASSERT_TRUE(tsk_site_table_equals(&table, &table2, 0));
    CU_ASSERT_TRUE(tsk_site_table_equals(&table, &table2, TSK_CMP_IGNORE_METADATA));
    /* Change the metadata values */
    table2.metadata[0] = 0;
    CU_ASSERT_FALSE(tsk_site_table_equals(&table, &table2, 0));
    CU_ASSERT_TRUE(tsk_site_table_equals(&table, &table2, TSK_CMP_IGNORE_METADATA));
    /* Change the last metadata entry */
    table2.metadata_offset[table2.num_rows]
        = table2.metadata_offset[table2.num_rows - 1];
    CU_ASSERT_FALSE(tsk_site_table_equals(&table, &table2, 0));
    CU_ASSERT_TRUE(tsk_site_table_equals(&table, &table2, TSK_CMP_IGNORE_METADATA));
    /* Delete all metadata */
    memset(table2.metadata_offset, 0,
        (table2.num_rows + 1) * sizeof(*table2.metadata_offset));
    CU_ASSERT_FALSE(tsk_site_table_equals(&table, &table2, 0));
    CU_ASSERT_TRUE(tsk_site_table_equals(&table, &table2, TSK_CMP_IGNORE_METADATA));
    tsk_site_table_free(&table2);

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
    CU_ASSERT_TRUE(tsk_site_table_equals(&table, &table2, 0));
    tsk_site_table_set_metadata_schema(&table2, example2, example2_length);
    CU_ASSERT_FALSE(tsk_site_table_equals(&table, &table2, 0));
    CU_ASSERT_TRUE(tsk_site_table_equals(&table, &table2, TSK_CMP_IGNORE_METADATA));

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
    ret = tsk_mutation_table_dump_text(&table, _devnull);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

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
    ret = tsk_mutation_table_dump_text(&table, _devnull);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

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

    /* Test equality with and without metadata */
    tsk_mutation_table_copy(&table, &table2, 0);
    CU_ASSERT_TRUE(tsk_mutation_table_equals(&table, &table2, 0));
    CU_ASSERT_TRUE(tsk_mutation_table_equals(&table, &table2, TSK_CMP_IGNORE_METADATA));
    /* Change the metadata values */
    table2.metadata[0] = 0;
    CU_ASSERT_FALSE(tsk_mutation_table_equals(&table, &table2, 0));
    CU_ASSERT_TRUE(tsk_mutation_table_equals(&table, &table2, TSK_CMP_IGNORE_METADATA));
    /* Change the last metadata entry */
    table2.metadata_offset[table2.num_rows]
        = table2.metadata_offset[table2.num_rows - 1];
    CU_ASSERT_FALSE(tsk_mutation_table_equals(&table, &table2, 0));
    CU_ASSERT_TRUE(tsk_mutation_table_equals(&table, &table2, TSK_CMP_IGNORE_METADATA));
    /* Delete all metadata */
    memset(table2.metadata_offset, 0,
        (table2.num_rows + 1) * sizeof(*table2.metadata_offset));
    CU_ASSERT_FALSE(tsk_mutation_table_equals(&table, &table2, 0));
    CU_ASSERT_TRUE(tsk_mutation_table_equals(&table, &table2, TSK_CMP_IGNORE_METADATA));
    tsk_mutation_table_free(&table2);

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
    CU_ASSERT_TRUE(tsk_mutation_table_equals(&table, &table2, 0));
    tsk_mutation_table_set_metadata_schema(&table2, example2, example2_length);
    CU_ASSERT_FALSE(tsk_mutation_table_equals(&table, &table2, 0));
    CU_ASSERT_TRUE(tsk_mutation_table_equals(&table, &table2, TSK_CMP_IGNORE_METADATA));

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
    ret = tsk_migration_table_dump_text(&table, _devnull);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

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
    ret = tsk_migration_table_dump_text(&table, _devnull);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

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

    /* Test equality with and without metadata */
    tsk_migration_table_copy(&table, &table2, 0);
    CU_ASSERT_TRUE(tsk_migration_table_equals(&table, &table2, 0));
    CU_ASSERT_TRUE(tsk_migration_table_equals(&table, &table2, TSK_CMP_IGNORE_METADATA));
    /* Change the metadata values */
    table2.metadata[0] = 0;
    CU_ASSERT_FALSE(tsk_migration_table_equals(&table, &table2, 0));
    CU_ASSERT_TRUE(tsk_migration_table_equals(&table, &table2, TSK_CMP_IGNORE_METADATA));
    /* Change the last metadata entry */
    table2.metadata_offset[table2.num_rows]
        = table2.metadata_offset[table2.num_rows - 1];
    CU_ASSERT_FALSE(tsk_migration_table_equals(&table, &table2, 0));
    CU_ASSERT_TRUE(tsk_migration_table_equals(&table, &table2, TSK_CMP_IGNORE_METADATA));
    /* Delete all metadata */
    memset(table2.metadata_offset, 0,
        (table2.num_rows + 1) * sizeof(*table2.metadata_offset));
    CU_ASSERT_FALSE(tsk_migration_table_equals(&table, &table2, 0));
    CU_ASSERT_TRUE(tsk_migration_table_equals(&table, &table2, TSK_CMP_IGNORE_METADATA));
    tsk_migration_table_free(&table2);

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
    ret = tsk_migration_table_dump_text(&table, _devnull);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

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
    CU_ASSERT_TRUE(tsk_migration_table_equals(&table, &table2, 0));
    tsk_migration_table_set_metadata_schema(&table2, example2, example2_length);
    CU_ASSERT_FALSE(tsk_migration_table_equals(&table, &table2, 0));
    CU_ASSERT_TRUE(tsk_migration_table_equals(&table, &table2, TSK_CMP_IGNORE_METADATA));

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
    tsk_size_t num_rows = 100;
    tsk_id_t j;
    tsk_size_t k;
    uint32_t *flags;
    double *location;
    tsk_id_t *parents;
    char *metadata;
    tsk_size_t *metadata_offset;
    tsk_size_t *parents_offset;
    tsk_size_t *location_offset;
    tsk_individual_t individual;
    const char *test_metadata = "test";
    tsk_size_t test_metadata_length = 4;
    char metadata_copy[test_metadata_length + 1];
    tsk_size_t spatial_dimension = 2;
    tsk_size_t num_parents = 2;
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
    tsk_individual_table_set_max_parents_length_increment(&table, 1);

    tsk_individual_table_print_state(&table, _devnull);

    for (j = 0; j < (tsk_id_t) num_rows; j++) {
        ret = tsk_individual_table_add_row(&table, (tsk_flags_t) j, test_location,
            spatial_dimension, NULL, 0, test_metadata, test_metadata_length);
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

    /* Test equality with and without metadata */
    tsk_individual_table_copy(&table, &table2, 0);
    CU_ASSERT_TRUE(tsk_individual_table_equals(&table, &table2, 0));
    CU_ASSERT_TRUE(
        tsk_individual_table_equals(&table, &table2, TSK_CMP_IGNORE_METADATA));
    /* Change the metadata values */
    table2.metadata[0] = 0;
    CU_ASSERT_FALSE(tsk_individual_table_equals(&table, &table2, 0));
    CU_ASSERT_TRUE(
        tsk_individual_table_equals(&table, &table2, TSK_CMP_IGNORE_METADATA));
    /* Change the last metadata entry */
    table2.metadata_offset[table2.num_rows]
        = table2.metadata_offset[table2.num_rows - 1];
    CU_ASSERT_FALSE(tsk_individual_table_equals(&table, &table2, 0));
    CU_ASSERT_TRUE(
        tsk_individual_table_equals(&table, &table2, TSK_CMP_IGNORE_METADATA));
    /* Delete all metadata */
    memset(table2.metadata_offset, 0,
        (table2.num_rows + 1) * sizeof(*table2.metadata_offset));
    CU_ASSERT_FALSE(tsk_individual_table_equals(&table, &table2, 0));
    CU_ASSERT_TRUE(
        tsk_individual_table_equals(&table, &table2, TSK_CMP_IGNORE_METADATA));
    tsk_individual_table_free(&table2);

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
    parents = malloc(num_parents * num_rows * sizeof(tsk_id_t));
    CU_ASSERT_FATAL(parents != NULL);
    memset(parents, 0, num_parents * num_rows * sizeof(tsk_id_t));
    parents_offset = malloc((num_rows + 1) * sizeof(tsk_size_t));
    CU_ASSERT_FATAL(parents_offset != NULL);
    for (j = 0; j < (tsk_id_t) num_rows + 1; j++) {
        parents_offset[j] = (tsk_size_t) j * num_parents;
    }
    metadata = malloc(num_rows * sizeof(char));
    memset(metadata, 'a', num_rows * sizeof(char));
    CU_ASSERT_FATAL(metadata != NULL);
    metadata_offset = malloc((num_rows + 1) * sizeof(tsk_size_t));
    CU_ASSERT_FATAL(metadata_offset != NULL);
    for (j = 0; j < (tsk_id_t) num_rows + 1; j++) {
        metadata_offset[j] = (tsk_size_t) j;
    }
    ret = tsk_individual_table_set_columns(&table, num_rows, flags, location,
        location_offset, parents, parents_offset, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(table.flags, flags, num_rows * sizeof(uint32_t)), 0);
    CU_ASSERT_EQUAL(
        memcmp(table.location, location, spatial_dimension * num_rows * sizeof(double)),
        0);
    CU_ASSERT_EQUAL(memcmp(table.location_offset, location_offset,
                        (num_rows + 1) * sizeof(tsk_size_t)),
        0);
    CU_ASSERT_EQUAL(
        memcmp(table.parents, parents, num_parents * num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.parents_offset, parents_offset,
                        (num_rows + 1) * sizeof(tsk_size_t)),
        0);
    CU_ASSERT_EQUAL(memcmp(table.metadata, metadata, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(memcmp(table.metadata_offset, metadata_offset,
                        (num_rows + 1) * sizeof(tsk_size_t)),
        0);
    CU_ASSERT_EQUAL(table.num_rows, num_rows);
    CU_ASSERT_EQUAL(table.location_length, spatial_dimension * num_rows);
    CU_ASSERT_EQUAL(table.parents_length, num_parents * num_rows);
    CU_ASSERT_EQUAL(table.metadata_length, num_rows);
    tsk_individual_table_print_state(&table, _devnull);

    /* Append another num_rows onto the end */
    ret = tsk_individual_table_append_columns(&table, num_rows, flags, location,
        location_offset, parents, parents_offset, metadata, metadata_offset);
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
    CU_ASSERT_EQUAL(
        memcmp(table.parents, parents, num_parents * num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.parents + num_parents * num_rows, parents,
                        num_parents * num_rows * sizeof(tsk_id_t)),
        0);
    CU_ASSERT_EQUAL(table.num_rows, 2 * num_rows);
    CU_ASSERT_EQUAL(table.metadata_length, 2 * num_rows);
    CU_ASSERT_EQUAL(table.parents_length, 2 * num_parents * num_rows);
    CU_ASSERT_EQUAL(table.location_length, 2 * spatial_dimension * num_rows);
    tsk_individual_table_print_state(&table, _devnull);
    ret = tsk_individual_table_dump_text(&table, _devnull);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

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
    CU_ASSERT_EQUAL(
        memcmp(table.parents, parents, num_parents * num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.parents_offset, parents_offset,
                        (num_rows + 1) * sizeof(tsk_size_t)),
        0);
    CU_ASSERT_EQUAL(memcmp(table.metadata, metadata, num_rows * sizeof(char)), 0);
    CU_ASSERT_EQUAL(memcmp(table.metadata_offset, metadata_offset,
                        (num_rows + 1) * sizeof(tsk_size_t)),
        0);
    CU_ASSERT_EQUAL(table.num_rows, num_rows);
    CU_ASSERT_EQUAL(table.location_length, spatial_dimension * num_rows);
    CU_ASSERT_EQUAL(table.parents_length, num_parents * num_rows);
    CU_ASSERT_EQUAL(table.metadata_length, num_rows);
    tsk_individual_table_print_state(&table, _devnull);

    ret = tsk_individual_table_truncate(&table, num_rows + 1);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_TABLE_POSITION);

    /* flags can't be NULL */
    ret = tsk_individual_table_set_columns(&table, num_rows, NULL, location,
        location_offset, parents, parents_offset, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    /* location and location offset must be simultaneously NULL or not */
    ret = tsk_individual_table_set_columns(&table, num_rows, flags, location, NULL,
        parents, parents_offset, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    ret = tsk_individual_table_set_columns(&table, num_rows, flags, NULL,
        location_offset, NULL, NULL, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    /* parents and parents offset must be simultaneously NULL or not */
    ret = tsk_individual_table_set_columns(&table, num_rows, flags, location,
        location_offset, parents, NULL, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    ret = tsk_individual_table_set_columns(&table, num_rows, flags, location,
        location_offset, NULL, parents_offset, metadata, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    /* metadata and metadata offset must be simultaneously NULL or not */
    ret = tsk_individual_table_set_columns(&table, num_rows, flags, location,
        location_offset, parents, parents_offset, NULL, metadata_offset);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    ret = tsk_individual_table_set_columns(&table, num_rows, flags, location,
        location_offset, parents, parents_offset, metadata, NULL);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);

    /* if location and location_offset are both null, all locations are zero length */
    num_rows = 10;
    memset(location_offset, 0, (num_rows + 1) * sizeof(tsk_size_t));
    ret = tsk_individual_table_set_columns(
        &table, num_rows, flags, NULL, NULL, NULL, NULL, NULL, NULL);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(table.location_offset, location_offset,
                        (num_rows + 1) * sizeof(tsk_size_t)),
        0);
    CU_ASSERT_EQUAL(table.num_rows, num_rows);
    CU_ASSERT_EQUAL(table.location_length, 0);
    ret = tsk_individual_table_append_columns(
        &table, num_rows, flags, NULL, NULL, NULL, NULL, NULL, NULL);
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
    ret = tsk_individual_table_dump_text(&table, _devnull);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* if parents and parents_offset are both null, all parents are zero length */
    num_rows = 10;
    memset(parents_offset, 0, (num_rows + 1) * sizeof(tsk_size_t));
    ret = tsk_individual_table_set_columns(
        &table, num_rows, flags, NULL, NULL, NULL, NULL, NULL, NULL);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(table.parents_offset, parents_offset,
                        (num_rows + 1) * sizeof(tsk_size_t)),
        0);
    CU_ASSERT_EQUAL(table.num_rows, num_rows);
    CU_ASSERT_EQUAL(table.parents_length, 0);
    ret = tsk_individual_table_append_columns(
        &table, num_rows, flags, NULL, NULL, NULL, NULL, NULL, NULL);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(table.parents_offset, parents_offset,
                        (num_rows + 1) * sizeof(tsk_size_t)),
        0);
    CU_ASSERT_EQUAL(memcmp(table.parents_offset + num_rows, parents_offset,
                        num_rows * sizeof(uint32_t)),
        0);
    CU_ASSERT_EQUAL(table.num_rows, 2 * num_rows);
    CU_ASSERT_EQUAL(table.parents_length, 0);
    tsk_individual_table_print_state(&table, _devnull);
    ret = tsk_individual_table_dump_text(&table, _devnull);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* if metadata and metadata_offset are both null, all metadatas are zero length */
    num_rows = 10;
    memset(metadata_offset, 0, (num_rows + 1) * sizeof(tsk_size_t));
    ret = tsk_individual_table_set_columns(
        &table, num_rows, flags, location, location_offset, NULL, NULL, NULL, NULL);
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
    ret = tsk_individual_table_append_columns(&table, num_rows, flags, location,
        location_offset, parents, parents_offset, NULL, NULL);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(
        memcmp(table.location, location, spatial_dimension * num_rows * sizeof(double)),
        0);
    CU_ASSERT_EQUAL(memcmp(table.location + spatial_dimension * num_rows, location,
                        spatial_dimension * num_rows * sizeof(double)),
        0);
    CU_ASSERT_EQUAL(
        memcmp(table.parents, parents, num_parents * num_rows * sizeof(tsk_id_t)), 0);
    CU_ASSERT_EQUAL(memcmp(table.parents + num_parents * num_rows, parents,
                        num_parents * num_rows * sizeof(tsk_id_t)),
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
    CU_ASSERT_TRUE(tsk_individual_table_equals(&table, &table2, 0));
    tsk_individual_table_set_metadata_schema(&table2, example2, example2_length);
    CU_ASSERT_FALSE(tsk_individual_table_equals(&table, &table2, 0));
    CU_ASSERT_TRUE(
        tsk_individual_table_equals(&table, &table2, TSK_CMP_IGNORE_METADATA));

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
    free(parents);
    free(parents_offset);
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
    ret = tsk_population_table_dump_text(&table, _devnull);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
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

    /* Test equality with and without metadata */
    tsk_population_table_copy(&table, &table2, 0);
    CU_ASSERT_TRUE(tsk_population_table_equals(&table, &table2, 0));
    CU_ASSERT_TRUE(
        tsk_population_table_equals(&table, &table2, TSK_CMP_IGNORE_METADATA));
    /* Change the metadata values */
    table2.metadata[0] = 0;
    CU_ASSERT_FALSE(tsk_population_table_equals(&table, &table2, 0));
    CU_ASSERT_TRUE(
        tsk_population_table_equals(&table, &table2, TSK_CMP_IGNORE_METADATA));
    /* Change the last metadata entry */
    table2.metadata_offset[table2.num_rows]
        = table2.metadata_offset[table2.num_rows - 1];
    CU_ASSERT_FALSE(tsk_population_table_equals(&table, &table2, 0));
    CU_ASSERT_TRUE(
        tsk_population_table_equals(&table, &table2, TSK_CMP_IGNORE_METADATA));
    /* Delete all metadata */
    memset(table2.metadata_offset, 0,
        (table2.num_rows + 1) * sizeof(*table2.metadata_offset));
    CU_ASSERT_FALSE(tsk_population_table_equals(&table, &table2, 0));
    CU_ASSERT_TRUE(
        tsk_population_table_equals(&table, &table2, TSK_CMP_IGNORE_METADATA));
    tsk_population_table_free(&table2);

    ret = tsk_population_table_get_row(&table, (tsk_id_t) num_rows, &population);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_POPULATION_OUT_OF_BOUNDS);
    tsk_population_table_print_state(&table, _devnull);
    ret = tsk_population_table_dump_text(&table, _devnull);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

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
    CU_ASSERT_TRUE(tsk_population_table_equals(&table, &table2, 0));
    tsk_population_table_set_metadata_schema(&table2, example2, example2_length);
    CU_ASSERT_FALSE(tsk_population_table_equals(&table, &table2, 0));
    CU_ASSERT_TRUE(
        tsk_population_table_equals(&table, &table2, TSK_CMP_IGNORE_METADATA));

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
    tsk_provenance_table_t table, table2;
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
    ret = tsk_provenance_table_dump_text(&table, _devnull);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

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
    ret = tsk_provenance_table_dump_text(&table, _devnull);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
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

    /* Test equality with and without timestamp */
    tsk_provenance_table_copy(&table, &table2, 0);
    CU_ASSERT_TRUE(tsk_provenance_table_equals(&table, &table2, 0));
    CU_ASSERT_TRUE(
        tsk_provenance_table_equals(&table, &table2, TSK_CMP_IGNORE_TIMESTAMPS));
    /* Change the timestamp values */
    table2.timestamp[0] = 0;
    CU_ASSERT_FALSE(tsk_provenance_table_equals(&table, &table2, 0));
    CU_ASSERT_TRUE(
        tsk_provenance_table_equals(&table, &table2, TSK_CMP_IGNORE_TIMESTAMPS));
    /* Change the last timestamp entry */
    table2.timestamp_offset[table2.num_rows]
        = table2.timestamp_offset[table2.num_rows - 1];
    CU_ASSERT_FALSE(tsk_provenance_table_equals(&table, &table2, 0));
    CU_ASSERT_TRUE(
        tsk_provenance_table_equals(&table, &table2, TSK_CMP_IGNORE_TIMESTAMPS));
    /* Delete all timestamps */
    memset(table2.timestamp_offset, 0,
        (table2.num_rows + 1) * sizeof(*table2.timestamp_offset));
    CU_ASSERT_FALSE(tsk_provenance_table_equals(&table, &table2, 0));
    CU_ASSERT_TRUE(
        tsk_provenance_table_equals(&table, &table2, TSK_CMP_IGNORE_TIMESTAMPS));
    tsk_provenance_table_free(&table2);

    /* Test equality with and without timestamp */
    tsk_provenance_table_copy(&table, &table2, 0);
    table2.record_length = 0;
    CU_ASSERT_FALSE(tsk_provenance_table_equals(&table, &table2, 0));
    tsk_provenance_table_free(&table2);

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
    ret = tsk_individual_table_set_max_parents_length_increment(
        &tables.individuals, new_size);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(tables.individuals.max_parents_length_increment, default_size);

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

/* Helper method for running IBD tests */
static int TSK_WARN_UNUSED
ibd_finder_init_and_run(tsk_ibd_finder_t *ibd_finder, tsk_table_collection_t *tables,
    tsk_id_t *samples, tsk_size_t num_samples, double min_length, double max_time)
{
    int ret = 0;

    ret = tsk_ibd_finder_init(ibd_finder, tables, samples, num_samples);
    if (ret != 0) {
        goto out;
    }
    ret = tsk_ibd_finder_set_min_length(ibd_finder, min_length);
    if (ret != 0) {
        goto out;
    }
    ret = tsk_ibd_finder_set_max_time(ibd_finder, max_time);
    if (ret != 0) {
        goto out;
    }
    ret = tsk_ibd_finder_run(ibd_finder);
    if (ret != 0) {
        goto out;
    }

out:
    return ret;
}

static void
test_ibd_finder(void)
{
    int ret;
    int j, k;
    tsk_treeseq_t ts;
    tsk_table_collection_t tables;
    tsk_id_t samples[] = { 0, 1 };
    tsk_ibd_finder_t ibd_finder;
    double true_left[] = { 0.0 };
    double true_right[] = { 1.0 };
    tsk_id_t true_node[] = { 4 };
    tsk_segment_t *seg = NULL;

    // Read in the tree sequence.
    tsk_treeseq_from_text(&ts, 1, single_tree_ex_nodes, single_tree_ex_edges, NULL, NULL,
        NULL, NULL, NULL, 0);
    ret = tsk_treeseq_copy_tables(&ts, &tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = ibd_finder_init_and_run(&ibd_finder, &tables, samples, 1, 0.0, DBL_MAX);

    // Check the output.
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    for (j = 0; j < (int) ibd_finder.num_pairs; j++) {
        tsk_ibd_finder_get_ibd_segments(&ibd_finder, j, &seg);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        k = 0;
        while (seg != NULL) {
            CU_ASSERT_EQUAL_FATAL(seg->left, true_left[k]);
            CU_ASSERT_EQUAL_FATAL(seg->right, true_right[k]);
            CU_ASSERT_EQUAL_FATAL(seg->node, true_node[k]);
            k++;
            seg = seg->next;
        }
    }
    tsk_ibd_finder_print_state(&ibd_finder, _devnull);

    // Free.
    tsk_ibd_finder_free(&ibd_finder);
    tsk_table_collection_free(&tables);
    tsk_treeseq_free(&ts);
}

static void
test_ibd_finder_multiple_trees(void)
{
    int ret;
    int j, k;
    tsk_treeseq_t ts;
    tsk_table_collection_t tables;
    tsk_id_t samples[] = { 0, 1, 0, 2 };
    tsk_ibd_finder_t ibd_finder;
    double true_left[2][2] = { { 0.0, 0.7 }, { 0.7, 0.0 } };
    double true_right[2][2] = { { 0.7, 1.0 }, { 1.0, 0.7 } };
    double true_node[2][2] = { { 4, 5 }, { 5, 6 } };
    tsk_segment_t *seg = NULL;

    // Read in the tree sequence.
    tsk_treeseq_from_text(&ts, 2, multiple_tree_ex_nodes, multiple_tree_ex_edges, NULL,
        NULL, NULL, NULL, NULL, 0);
    ret = tsk_treeseq_copy_tables(&ts, &tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    // Run ibd_finder.
    ret = ibd_finder_init_and_run(&ibd_finder, &tables, samples, 2, 0.0, DBL_MAX);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    // Check the output.
    for (j = 0; j < (int) ibd_finder.num_pairs; j++) {
        ret = tsk_ibd_finder_get_ibd_segments(&ibd_finder, j, &seg);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        k = 0;
        while (seg != NULL) {
            CU_ASSERT_EQUAL_FATAL(seg->left, true_left[j][k]);
            CU_ASSERT_EQUAL_FATAL(seg->right, true_right[j][k]);
            CU_ASSERT_EQUAL_FATAL(seg->node, true_node[j][k]);
            k++;
            seg = seg->next;
        }
    }

    // Free.
    tsk_ibd_finder_free(&ibd_finder);
    tsk_table_collection_free(&tables);
    tsk_treeseq_free(&ts);
}

static void
test_ibd_finder_empty_result(void)
{
    int ret;
    int j;
    tsk_treeseq_t ts;
    tsk_table_collection_t tables;
    tsk_id_t samples[] = { 0, 1 };
    tsk_ibd_finder_t ibd_finder;
    tsk_segment_t *seg = NULL;

    // Read in the tree sequence.
    tsk_treeseq_from_text(&ts, 1, single_tree_ex_nodes, single_tree_ex_edges, NULL, NULL,
        NULL, NULL, NULL, 0);
    ret = tsk_treeseq_copy_tables(&ts, &tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    // Run ibd_finder.
    ret = ibd_finder_init_and_run(&ibd_finder, &tables, samples, 1, 0.0, 0.5);

    // Check the output.
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    for (j = 0; j < (int) ibd_finder.num_pairs; j++) {
        tsk_ibd_finder_get_ibd_segments(&ibd_finder, j, &seg);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        CU_ASSERT_EQUAL_FATAL(seg, NULL);
    }
    tsk_ibd_finder_print_state(&ibd_finder, _devnull);

    // Free.
    tsk_ibd_finder_free(&ibd_finder);
    tsk_table_collection_free(&tables);
    tsk_treeseq_free(&ts);
}

static void
test_ibd_finder_min_length_max_time(void)
{
    int ret;
    int j, k;
    tsk_treeseq_t ts;
    tsk_table_collection_t tables;
    tsk_id_t samples[] = { 0, 1, 1, 2, 2, 0 };
    tsk_ibd_finder_t ibd_finder;
    double true_left[3][1] = { { 0.0 }, { -1 }, { -1 } };
    double true_right[3][1] = { { 0.7 }, { -1 }, { -1 } };
    double true_node[3][1] = { { 4 }, { -1 }, { -1 } };
    tsk_segment_t *seg = NULL;

    // Read in the tree sequence.
    tsk_treeseq_from_text(&ts, 2, multiple_tree_ex_nodes, multiple_tree_ex_edges, NULL,
        NULL, NULL, NULL, NULL, 0);
    ret = tsk_treeseq_copy_tables(&ts, &tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    // Run ibd_finder.
    ret = ibd_finder_init_and_run(&ibd_finder, &tables, samples, 3, 0.5, 3.0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    // Check the output.
    for (j = 0; j < (int) ibd_finder.num_pairs; j++) {
        ret = tsk_ibd_finder_get_ibd_segments(&ibd_finder, j, &seg);
        CU_ASSERT_TRUE_FATAL((ret == 0) || (ret == -1));
        if (ret == -1) {
            continue;
        }
        k = 0;
        while (seg != NULL) {
            CU_ASSERT_EQUAL_FATAL(seg->left, true_left[j][k]);
            CU_ASSERT_EQUAL_FATAL(seg->right, true_right[j][k]);
            CU_ASSERT_EQUAL_FATAL(seg->node, true_node[j][k]);
            k++;
            seg = seg->next;
        }
    }

    // Free.
    tsk_ibd_finder_free(&ibd_finder);
    tsk_table_collection_free(&tables);
    tsk_treeseq_free(&ts);
}

static void
test_ibd_finder_errors(void)
{
    int ret;
    tsk_treeseq_t ts;
    tsk_table_collection_t tables;
    tsk_id_t samples[] = { 0, 1, 2, 0 };
    tsk_id_t duplicate_samples[] = { 0, 1, 1, 0 };
    tsk_id_t samples2[] = { -1, 1 };
    tsk_id_t samples3[] = { 0 };
    tsk_ibd_finder_t ibd_finder;

    tsk_treeseq_from_text(&ts, 2, multiple_tree_ex_nodes, multiple_tree_ex_edges, NULL,
        NULL, NULL, NULL, NULL, 0);
    ret = tsk_treeseq_copy_tables(&ts, &tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    // Invalid sample IDs
    ret = ibd_finder_init_and_run(&ibd_finder, &tables, samples2, 1, 0.0, DBL_MAX);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_NODE_OUT_OF_BOUNDS);
    tsk_ibd_finder_free(&ibd_finder);

    // Only 1 sample
    ret = ibd_finder_init_and_run(&ibd_finder, &tables, samples3, 0, 0.0, DBL_MAX);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_NO_SAMPLE_PAIRS);
    tsk_ibd_finder_free(&ibd_finder);

    // Bad length or time
    ret = ibd_finder_init_and_run(&ibd_finder, &tables, samples, 2, 0.0, -1);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    tsk_ibd_finder_free(&ibd_finder);
    ret = ibd_finder_init_and_run(&ibd_finder, &tables, samples, 2, -1, 0.0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    tsk_ibd_finder_free(&ibd_finder);

    // Duplicate samples
    ret = ibd_finder_init_and_run(&ibd_finder, &tables, duplicate_samples, 2, 0.0, -1);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_DUPLICATE_SAMPLE_PAIRS);
    tsk_ibd_finder_free(&ibd_finder);

    tsk_table_collection_free(&tables);
    tsk_treeseq_free(&ts);
}

static void
test_ibd_finder_samples_are_descendants(void)
{
    int ret;
    int j, k;
    tsk_treeseq_t ts;
    tsk_table_collection_t tables;
    tsk_id_t samples[] = { 0, 2, 0, 4, 2, 4, 1, 3, 1, 5, 3, 5 };
    tsk_ibd_finder_t ibd_finder;
    double true_left[6][1] = { { 0.0 }, { 0.0 }, { 0.0 }, { 0.0 }, { 0.0 }, { 0.0 } };
    double true_right[6][1] = { { 1.0 }, { 1.0 }, { 1.0 }, { 1.0 }, { 1.0 }, { 1.0 } };
    tsk_id_t true_node[6][1] = { { 2 }, { 4 }, { 4 }, { 3 }, { 5 }, { 5 } };
    tsk_segment_t *seg = NULL;

    // Read in the tree sequence.
    tsk_treeseq_from_text(&ts, 1, multi_root_tree_ex_nodes, multi_root_tree_ex_edges,
        NULL, NULL, NULL, NULL, NULL, 0);
    ret = tsk_treeseq_copy_tables(&ts, &tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    // Run ibd_finder.
    ret = ibd_finder_init_and_run(&ibd_finder, &tables, samples, 6, 0.0, DBL_MAX);

    // Check the output.
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    for (j = 0; j < (int) ibd_finder.num_pairs; j++) {
        tsk_ibd_finder_get_ibd_segments(&ibd_finder, j, &seg);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        k = 0;
        while (seg != NULL) {
            CU_ASSERT_EQUAL_FATAL(seg->left, true_left[j][k]);
            CU_ASSERT_EQUAL_FATAL(seg->right, true_right[j][k]);
            CU_ASSERT_EQUAL_FATAL(seg->node, true_node[j][k]);
            k++;
            seg = seg->next;
        }
    }
    tsk_ibd_finder_print_state(&ibd_finder, _devnull);

    // Free.
    tsk_ibd_finder_free(&ibd_finder);
    tsk_table_collection_free(&tables);
    tsk_treeseq_free(&ts);
}

static void
test_ibd_finder_multiple_ibd_paths(void)
{
    int ret;
    int j, k;
    tsk_treeseq_t ts;
    tsk_table_collection_t tables;
    tsk_id_t samples[] = { 0, 1, 0, 2, 1, 2 };
    tsk_ibd_finder_t ibd_finder;
    double true_left[3][2] = { { 0.2, 0.0 }, { 0.2, 0.0 }, { 0.0, 0.2 } };
    double true_right[3][2] = { { 1.0, 0.2 }, { 1.0, 0.2 }, { 0.2, 1.0 } };
    double true_node[3][2] = { { 4, 5 }, { 3, 5 }, { 4, 4 } };
    tsk_segment_t *seg = NULL;

    // Read in the tree sequence.
    tsk_treeseq_from_text(&ts, 2, multi_path_tree_ex_nodes, multi_path_tree_ex_edges,
        NULL, NULL, NULL, NULL, NULL, 0);
    ret = tsk_treeseq_copy_tables(&ts, &tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    // Run ibd_finder.
    ret = ibd_finder_init_and_run(&ibd_finder, &tables, samples, 3, 0.0, 0.0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    // Check the output.
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    for (j = 0; j < (int) ibd_finder.num_pairs; j++) {
        tsk_ibd_finder_get_ibd_segments(&ibd_finder, j, &seg);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        k = 0;
        while (seg != NULL) {
            CU_ASSERT_EQUAL_FATAL(seg->left, true_left[j][k]);
            CU_ASSERT_EQUAL_FATAL(seg->right, true_right[j][k]);
            CU_ASSERT_EQUAL_FATAL(seg->node, true_node[j][k]);
            k++;
            seg = seg->next;
        }
    }
    tsk_ibd_finder_print_state(&ibd_finder, _devnull);

    // Free.
    tsk_ibd_finder_free(&ibd_finder);
    tsk_table_collection_free(&tables);
    tsk_treeseq_free(&ts);
}

static void
test_ibd_finder_odd_topologies(void)
{
    int ret;
    // int j;
    tsk_treeseq_t ts;
    tsk_table_collection_t tables;
    tsk_id_t samples[] = { 0, 1 };
    tsk_id_t samples1[] = { 0, 2 };
    tsk_ibd_finder_t ibd_finder;

    tsk_treeseq_from_text(
        &ts, 1, odd_tree1_ex_nodes, odd_tree1_ex_edges, NULL, NULL, NULL, NULL, NULL, 0);
    ret = tsk_treeseq_copy_tables(&ts, &tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    // Multiple roots.
    ret = ibd_finder_init_and_run(&ibd_finder, &tables, samples, 1, 0, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tsk_ibd_finder_free(&ibd_finder);

    // // Parent is a sample.
    ret = ibd_finder_init_and_run(&ibd_finder, &tables, samples1, 1, 0, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tsk_ibd_finder_free(&ibd_finder);

    tsk_table_collection_free(&tables);
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
test_edge_update_invalidates_index(void)
{
    int ret;
    tsk_treeseq_t ts;
    tsk_table_collection_t tables;

    tsk_treeseq_from_text(&ts, 1, single_tree_ex_nodes, single_tree_ex_edges, NULL, NULL,
        NULL, NULL, NULL, 0);

    /* Any operations on the edge table should now invalidate the index */
    ret = tsk_treeseq_copy_tables(&ts, &tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_has_index(&tables, 0))
    ret = tsk_edge_table_clear(&tables.edges);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_FALSE(tsk_table_collection_has_index(&tables, 0));
    /* Even though the actual indexes still exist */
    CU_ASSERT_FALSE(tables.indexes.edge_insertion_order == NULL);
    CU_ASSERT_FALSE(tables.indexes.edge_removal_order == NULL);
    CU_ASSERT_EQUAL_FATAL(tables.indexes.num_edges, tsk_treeseq_get_num_edges(&ts));

    ret = tsk_treeseq_copy_tables(&ts, &tables, TSK_NO_INIT);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_has_index(&tables, 0))
    ret = tsk_edge_table_add_row(&tables.edges, 0, 1, 0, 1, NULL, 0);
    CU_ASSERT_TRUE(ret > 0);
    CU_ASSERT_FALSE(tsk_table_collection_has_index(&tables, 0));
    /* Even though the actual indexes still exist */
    CU_ASSERT_FALSE(tables.indexes.edge_insertion_order == NULL);
    CU_ASSERT_FALSE(tables.indexes.edge_removal_order == NULL);
    CU_ASSERT_EQUAL_FATAL(tables.indexes.num_edges, tsk_treeseq_get_num_edges(&ts));

    tsk_table_collection_free(&tables);
    tsk_treeseq_free(&ts);
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
    CU_ASSERT_TRUE(tsk_table_collection_equals(&tables, &tables_copy, 0));

    tsk_table_collection_free(&tables);
    tsk_table_collection_free(&tables_copy);
    tsk_treeseq_free(&ts);
}

static void
test_sort_tables_offsets(void)
{
    int ret;
    tsk_treeseq_t *ts;
    tsk_table_collection_t tables, copy;
    tsk_bookmark_t bookmark;

    ts = caterpillar_tree(10, 5, 5);
    ret = tsk_treeseq_copy_tables(ts, &tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_table_collection_sort(&tables, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    /* Check that setting edge offset = len(edges) does nothing */
    reverse_edges(&tables);
    ret = tsk_table_collection_copy(&tables, &copy, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    memset(&bookmark, 0, sizeof(bookmark));
    bookmark.edges = tables.edges.num_rows;
    ret = tsk_table_collection_sort(&tables, &bookmark, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_FATAL(tsk_table_collection_equals(&tables, &copy, 0));

    ret = tsk_table_collection_sort(&tables, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    /* Check that setting migration offset = len(migrations) does nothing */
    reverse_migrations(&tables);
    ret = tsk_table_collection_copy(&tables, &copy, TSK_NO_INIT);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    memset(&bookmark, 0, sizeof(bookmark));
    bookmark.migrations = tables.migrations.num_rows;
    ret = tsk_table_collection_sort(&tables, &bookmark, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_FATAL(tsk_table_collection_equals(&tables, &copy, 0));

    ret = tsk_table_collection_sort(&tables, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_FATAL(tables.sites.num_rows > 2);
    CU_ASSERT_FATAL(tables.mutations.num_rows > 2);

    /* Check that setting mutation and site offset = to the len
     * of the tables leaves them untouched. */
    reverse_mutations(&tables);
    /* Swap the positions of the first two sites, as a quick way
     * to disorder the site table */
    tables.sites.position[0] = tables.sites.position[1];
    tables.sites.position[1] = 0;
    ret = tsk_table_collection_copy(&tables, &copy, TSK_NO_INIT);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    memset(&bookmark, 0, sizeof(bookmark));
    bookmark.sites = tables.sites.num_rows;
    bookmark.mutations = tables.mutations.num_rows;
    ret = tsk_table_collection_sort(&tables, &bookmark, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_FATAL(tsk_table_collection_equals(&tables, &copy, 0));

    /* Anything other than len(table) leads to an error for sites
     * and mutations, and we can't specify one without the other. */
    memset(&bookmark, 0, sizeof(bookmark));
    bookmark.sites = tables.sites.num_rows;
    ret = tsk_table_collection_sort(&tables, &bookmark, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_SORT_OFFSET_NOT_SUPPORTED);

    memset(&bookmark, 0, sizeof(bookmark));
    bookmark.mutations = tables.mutations.num_rows;
    ret = tsk_table_collection_sort(&tables, &bookmark, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_SORT_OFFSET_NOT_SUPPORTED);

    memset(&bookmark, 0, sizeof(bookmark));
    bookmark.sites = tables.sites.num_rows - 1;
    bookmark.mutations = tables.mutations.num_rows - 1;
    ret = tsk_table_collection_sort(&tables, &bookmark, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_SORT_OFFSET_NOT_SUPPORTED);

    /* Individuals must either all be sorted or all skipped */
    ret = tsk_table_collection_sort(&tables, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    /* Add a parent relation that unsorts the table */
    tables.individuals.parents[0] = 5;
    ret = tsk_table_collection_copy(&tables, &copy, TSK_NO_INIT);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    memset(&bookmark, 0, sizeof(bookmark));
    bookmark.individuals = tables.individuals.num_rows;
    ret = tsk_table_collection_sort(&tables, &bookmark, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_equals(&tables, &copy, 0));

    /* Check that sorting would have had an effect */
    ret = tsk_table_collection_sort(&tables, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_FALSE(tsk_table_collection_equals(&tables, &copy, 0));

    memset(&bookmark, 0, sizeof(bookmark));
    bookmark.individuals = tables.individuals.num_rows - 1;
    ret = tsk_table_collection_sort(&tables, &bookmark, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_SORT_OFFSET_NOT_SUPPORTED);

    tsk_table_collection_free(&tables);
    tsk_table_collection_free(&copy);
    tsk_treeseq_free(ts);
    free(ts);
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
    CU_ASSERT_TRUE(tsk_table_collection_equals(&t1, &t2, 0));
    reverse_edges(&t1);
    CU_ASSERT_FALSE(tsk_table_collection_equals(&t1, &t2, 0));
    ret = tsk_table_collection_sort(&t1, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_equals(&t1, &t2, 0));

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
    CU_ASSERT_TRUE(tsk_table_collection_equals(&t1, &t2, 0));
    reverse_edges(&t1);
    CU_ASSERT_FALSE(tsk_table_collection_equals(&t1, &t2, 0));
    ret = tsk_table_collection_sort(&t1, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_equals(&t1, &t2, 0));
    tsk_table_collection_free(&t2);

    ret = tsk_table_collection_copy(&t1, &t2, TSK_NO_EDGE_METADATA);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(t1.edges.options & TSK_NO_EDGE_METADATA);
    CU_ASSERT_TRUE(t2.edges.options & TSK_NO_EDGE_METADATA);
    CU_ASSERT_TRUE(tsk_table_collection_equals(&t1, &t2, 0));
    reverse_edges(&t1);
    CU_ASSERT_FALSE(tsk_table_collection_equals(&t1, &t2, 0));
    ret = tsk_table_collection_sort(&t1, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_equals(&t1, &t2, 0));
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

    memset(&pos, 0, sizeof(pos));
    pos.migrations = (tsk_size_t) -1;
    ret = tsk_table_collection_sort(&tables, &pos, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_MIGRATION_OUT_OF_BOUNDS);

    pos.migrations = tables.migrations.num_rows + 1;
    ret = tsk_table_collection_sort(&tables, &pos, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_MIGRATION_OUT_OF_BOUNDS);

    /* Node, population and provenance positions are ignored */
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

    /* Setting sites or mutations gives a BAD_PARAM. See
     * github.com/tskit-dev/tskit/issues/101 */
    memset(&pos, 0, sizeof(pos));
    pos.sites = 1;
    ret = tsk_table_collection_sort(&tables, &pos, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_SORT_OFFSET_NOT_SUPPORTED);

    memset(&pos, 0, sizeof(pos));
    pos.mutations = 1;
    ret = tsk_table_collection_sort(&tables, &pos, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_SORT_OFFSET_NOT_SUPPORTED);

    tsk_table_collection_free(&tables);
    tsk_treeseq_free(&ts);
}

static void
test_sort_tables_mutation_times(void)
{
    int ret;
    tsk_treeseq_t ts;
    tsk_table_collection_t tables, t1, t2;
    const char *sites = "0       0\n"
                        "0.1     0\n"
                        "0.2     0\n"
                        "0.3     0\n";
    const char *mutations = "0   0  1  -1  3\n"
                            "1   1  1  -1  3\n"
                            "2   4  1  -1  8\n"
                            "2   1  0  -1   4\n"
                            "2   2  1  -1  3\n"
                            "2   1  1  -1   2\n"
                            "3   6  1  -1  10\n";

    ret = tsk_table_collection_init(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    tables.sequence_length = 1;
    parse_nodes(single_tree_ex_nodes, &tables.nodes);
    CU_ASSERT_EQUAL_FATAL(tables.nodes.num_rows, 7);
    tables.nodes.time[4] = 6;
    tables.nodes.time[5] = 8;
    tables.nodes.time[6] = 10;
    parse_edges(single_tree_ex_edges, &tables.edges);
    CU_ASSERT_EQUAL_FATAL(tables.edges.num_rows, 6);
    parse_sites(sites, &tables.sites);
    parse_mutations(mutations, &tables.mutations);
    CU_ASSERT_EQUAL_FATAL(tables.sites.num_rows, 4);
    CU_ASSERT_EQUAL_FATAL(tables.mutations.num_rows, 7);
    tables.sequence_length = 1.0;

    ret = tsk_table_collection_build_index(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* Check to make sure we have legal mutations */
    ret = tsk_treeseq_init(&ts, &tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_treeseq_copy_tables(&ts, &t1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_table_collection_copy(&t1, &t2, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_equals(&t1, &t2, 0));
    reverse_mutations(&t1);
    CU_ASSERT_FALSE(tsk_table_collection_equals(&t1, &t2, 0));
    ret = tsk_table_collection_sort(&t1, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_equals(&t1, &t2, 0));
    tsk_table_collection_free(&t2);

    tsk_table_collection_free(&t1);
    tsk_table_collection_free(&tables);
    tsk_treeseq_free(&ts);
}

static void
test_sort_tables_canonical_errors(void)
{
    int ret;
    tsk_table_collection_t tables;
    ret = tsk_table_collection_init(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tables.sequence_length = 1;

    ret = tsk_node_table_add_row(&tables.nodes, 0, 0.0, TSK_NULL, TSK_NULL, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_site_table_add_row(&tables.sites, 0.0, "x", 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_mutation_table_add_row(&tables.mutations, 0, 0, 2, 0.0, "a", 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_mutation_table_add_row(&tables.mutations, 0, 0, 3, 0.0, "b", 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_mutation_table_add_row(&tables.mutations, 0, 0, 1, 0.0, "c", 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_mutation_table_add_row(&tables.mutations, 0, 0, 2, 0.0, "d", 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);

    ret = tsk_table_collection_canonicalise(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_MUTATION_PARENT_INCONSISTENT);

    ret = tsk_mutation_table_clear(&tables.mutations);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_mutation_table_add_row(&tables.mutations, 0, 0, 2, 0.0, "a", 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_mutation_table_add_row(&tables.mutations, 0, 0, 3, 0.0, "b", 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_mutation_table_add_row(&tables.mutations, 0, 0, 1, 0.0, "c", 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_mutation_table_add_row(&tables.mutations, 0, 0, -1, 0.0, "d", 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);

    ret = tsk_table_collection_canonicalise(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    tsk_table_collection_free(&tables);
}

static void
test_sort_tables_canonical(void)
{
    int ret;
    tsk_table_collection_t t1, t2;
    // this is single_tree_ex with individuals and populations
    const char *nodes = "1  0   -1    1\n"
                        "1  0    2    3\n"
                        "1  0    0   -1\n"
                        "1  0   -1    3\n"
                        "0  1    2   -1\n"
                        "0  2   -1    2\n"
                        "0  3   -1   -1\n";
    const char *individuals = "0 0.0\n"
                              "0 1.0\n"
                              "0 2.0\n"
                              "0 3.0\n";
    const char *sites = "0       0\n"
                        "0.2     0\n"
                        "0.1     0\n";
    const char *mutations = "0   0  2   3 0.5\n"
                            "2   1  1  -1 0.5\n"
                            "1   4  3  -1   3\n"
                            "0   4  1  -1 2.5\n"
                            "2   2  1  -1   2\n"
                            "1   1  5   7 0.5\n"
                            "1   2  1  -1   2\n"
                            "1   1  4   2 0.5\n"
                            "1   1  6   7 0.5\n";
    const char *nodes_sorted = "1  0   -1    0\n"
                               "1  0    0    1\n"
                               "1  0    1   -1\n"
                               "1  0   -1    1\n"
                               "0  1    0   -1\n"
                               "0  2   -1    2\n"
                               "0  3   -1   -1\n";
    const char *individuals_sorted = "0 1.0\n"
                                     "0 3.0\n"
                                     "0 2.0\n";
    const char *sites_sorted = "0       0\n"
                               "0.1     0\n"
                               "0.2     0\n";
    const char *mutations_sorted = "0   4  1  -1 2.5\n"
                                   "0   0  2   0 0.5\n"
                                   "1   2  1  -1   2\n"
                                   "1   1  1  -1 0.5\n"
                                   "2   4  3  -1   3\n"
                                   "2   2  1  -1   2\n"
                                   "2   1  4   4 0.5\n"
                                   "2   1  5   6 0.5\n"
                                   "2   1  6   6 0.5\n";
    const char *individuals_sorted_kept = "0 1.0\n"
                                          "0 3.0\n"
                                          "0 2.0\n"
                                          "0 0.0\n";

    ret = tsk_table_collection_init(&t1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    t1.sequence_length = 1.0;
    ret = tsk_table_collection_init(&t2, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    t2.sequence_length = 1.0;

    parse_nodes(nodes, &t1.nodes);
    CU_ASSERT_EQUAL_FATAL(t1.nodes.num_rows, 7);
    parse_individuals(individuals, &t1.individuals);
    CU_ASSERT_EQUAL_FATAL(t1.individuals.num_rows, 4);
    tsk_population_table_add_row(&t1.populations, "A", 1);
    tsk_population_table_add_row(&t1.populations, "B", 1);
    tsk_population_table_add_row(&t1.populations, "C", 1);
    parse_edges(single_tree_ex_edges, &t1.edges);
    CU_ASSERT_EQUAL_FATAL(t1.edges.num_rows, 6);
    parse_sites(sites, &t1.sites);
    CU_ASSERT_EQUAL_FATAL(t1.sites.num_rows, 3);
    parse_mutations(mutations, &t1.mutations);
    CU_ASSERT_EQUAL_FATAL(t1.mutations.num_rows, 9);

    ret = tsk_table_collection_canonicalise(&t1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    parse_nodes(nodes_sorted, &t2.nodes);
    tsk_population_table_add_row(&t2.populations, "C", 1);
    tsk_population_table_add_row(&t2.populations, "A", 1);
    CU_ASSERT_EQUAL_FATAL(t2.nodes.num_rows, 7);
    parse_individuals(individuals_sorted, &t2.individuals);
    CU_ASSERT_EQUAL_FATAL(t2.individuals.num_rows, 3);
    parse_edges(single_tree_ex_edges, &t2.edges);
    CU_ASSERT_EQUAL_FATAL(t2.edges.num_rows, 6);
    parse_sites(sites_sorted, &t2.sites);
    parse_mutations(mutations_sorted, &t2.mutations);
    CU_ASSERT_EQUAL_FATAL(t2.sites.num_rows, 3);
    CU_ASSERT_EQUAL_FATAL(t2.mutations.num_rows, 9);

    CU_ASSERT_TRUE(tsk_table_collection_equals(&t1, &t2, 0));

    ret = tsk_table_collection_clear(&t1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_clear(&t2, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    // now with KEEP_UNREFERENCED
    parse_nodes(nodes, &t1.nodes);
    parse_individuals(individuals, &t1.individuals);
    tsk_population_table_add_row(&t1.populations, "A", 1);
    tsk_population_table_add_row(&t1.populations, "B", 1);
    tsk_population_table_add_row(&t1.populations, "C", 1);
    parse_edges(single_tree_ex_edges, &t1.edges);
    parse_sites(sites, &t1.sites);
    parse_mutations(mutations, &t1.mutations);

    ret = tsk_table_collection_canonicalise(&t1, TSK_KEEP_UNREFERENCED);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    parse_nodes(nodes_sorted, &t2.nodes);
    tsk_population_table_add_row(&t2.populations, "C", 1);
    tsk_population_table_add_row(&t2.populations, "A", 1);
    tsk_population_table_add_row(&t2.populations, "B", 1);
    parse_individuals(individuals_sorted_kept, &t2.individuals);
    CU_ASSERT_EQUAL_FATAL(t2.individuals.num_rows, 4);
    parse_edges(single_tree_ex_edges, &t2.edges);
    parse_sites(sites_sorted, &t2.sites);
    parse_mutations(mutations_sorted, &t2.mutations);

    CU_ASSERT_TRUE(tsk_table_collection_equals(&t1, &t2, 0));

    tsk_table_collection_free(&t2);
    tsk_table_collection_free(&t1);
}

static void
test_sort_tables_migrations(void)
{
    int ret;
    tsk_treeseq_t *ts;
    tsk_table_collection_t tables, copy;

    ts = caterpillar_tree(13, 1, 1);
    ret = tsk_treeseq_copy_tables(ts, &tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_FATAL(tables.migrations.num_rows > 0);

    ret = tsk_table_collection_copy(&tables, &copy, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_FATAL(tsk_table_collection_equals(&tables, &copy, 0));

    reverse_migrations(&tables);
    CU_ASSERT_FATAL(!tsk_table_collection_equals(&tables, &copy, 0));
    ret = tsk_table_collection_sort(&tables, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_FATAL(tsk_migration_table_equals(&tables.migrations, &copy.migrations, 0));
    CU_ASSERT_FATAL(tsk_table_collection_equals(&tables, &copy, 0));

    /* Make sure we test the deeper comparison keys. The full key is
     * (time, source, dest, left, node) */
    tsk_migration_table_clear(&tables.migrations);

    /* params = left, right, node, source, dest, time */
    tsk_migration_table_add_row(&tables.migrations, 0, 1, 0, 0, 1, 0, NULL, 0);
    tsk_migration_table_add_row(&tables.migrations, 0, 1, 1, 0, 1, 0, NULL, 0);
    ret = tsk_migration_table_copy(&tables.migrations, &copy.migrations, TSK_NO_INIT);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    reverse_migrations(&tables);
    CU_ASSERT_FATAL(!tsk_table_collection_equals(&tables, &copy, 0));
    ret = tsk_table_collection_sort(&tables, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_FATAL(tsk_migration_table_equals(&tables.migrations, &copy.migrations, 0));

    tsk_table_collection_free(&tables);
    tsk_table_collection_free(&copy);
    tsk_treeseq_free(ts);
    free(ts);
}

static void
test_sort_tables_individuals(void)
{
    int ret;
    tsk_table_collection_t tables, copy;
    const char *individuals = "1      0.25   2,3 0\n"
                              "2      0.5    5,-1  1\n"
                              "3      0.3    -1  2\n"
                              "4      0.3    -1  3\n"
                              "5      0.3    3   4\n"
                              "6      0.3    4   5\n";
    const char *individuals_cycle = "1      0.2    2  0\n"
                                    "2      0.5    0  1\n"
                                    "3      0.3    1  2\n";

    ret = tsk_table_collection_init(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tables.sequence_length = 1.0;
    parse_individuals(individuals, &tables.individuals);
    ret = tsk_table_collection_check_integrity(&tables, TSK_CHECK_INDIVIDUAL_ORDERING);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_UNSORTED_INDIVIDUALS);

    ret = tsk_table_collection_sort(&tables, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_check_integrity(&tables, TSK_CHECK_INDIVIDUAL_ORDERING);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* Check that the sort is stable */
    ret = tsk_table_collection_copy(&tables, &copy, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_FATAL(tsk_table_collection_equals(&tables, &copy, 0));

    ret = tsk_table_collection_sort(&tables, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT(tsk_table_collection_equals(&tables, &copy, 0));

    /* Errors on cycle */
    tsk_individual_table_clear(&tables.individuals);
    parse_individuals(individuals_cycle, &tables.individuals);
    ret = tsk_table_collection_sort(&tables, NULL, 0);
    CU_ASSERT_EQUAL(ret, TSK_ERR_INDIVIDUAL_PARENT_CYCLE);

    tsk_table_collection_free(&tables);
    tsk_table_collection_free(&copy);
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

    CU_ASSERT_TRUE(tsk_table_collection_equals(ts.tables, &tables, 0));

    /* Nominal case */
    reverse_edges(&tables);
    CU_ASSERT_FALSE(tsk_table_collection_equals(ts.tables, &tables, 0));
    ret = tsk_table_sorter_init(&sorter, &tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_sorter_run(&sorter, NULL);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_equals(ts.tables, &tables, 0));
    CU_ASSERT_EQUAL(sorter.user_data, NULL);
    tsk_table_sorter_free(&sorter);

    /* If we set the sort_edges function to NULL then we should leave the
     * node table as is. */
    reverse_edges(&tables);
    CU_ASSERT_FALSE(tsk_edge_table_equals(&ts.tables->edges, &tables.edges, 0));
    ret = tsk_table_sorter_init(&sorter, &tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    sorter.sort_edges = NULL;
    ret = tsk_table_sorter_run(&sorter, NULL);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_FALSE(tsk_edge_table_equals(&ts.tables->edges, &tables.edges, 0));
    tsk_table_sorter_free(&sorter);

    /* Reversing again should make them equal */
    reverse_edges(&tables);
    CU_ASSERT_TRUE(tsk_edge_table_equals(&ts.tables->edges, &tables.edges, 0));

    /* Do not check integrity before sorting */
    reverse_edges(&tables);
    CU_ASSERT_FALSE(tsk_table_collection_equals(ts.tables, &tables, 0));
    ret = tsk_table_sorter_init(&sorter, &tables, TSK_NO_CHECK_INTEGRITY);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_sorter_run(&sorter, NULL);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_equals(ts.tables, &tables, 0));
    tsk_table_sorter_free(&sorter);

    /* The user_data shouldn't be touched */
    reverse_edges(&tables);
    CU_ASSERT_FALSE(tsk_table_collection_equals(ts.tables, &tables, 0));
    ret = tsk_table_sorter_init(&sorter, &tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    sorter.user_data = (void *) &ts;
    ret = tsk_table_sorter_run(&sorter, NULL);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_equals(ts.tables, &tables, 0));
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
    CU_ASSERT_TRUE(tsk_node_table_equals(&tables.nodes, &loaded.nodes, 0));
    CU_ASSERT_TRUE(tsk_edge_table_equals(&tables.edges, &loaded.edges, 0));

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
    CU_ASSERT_TRUE(tsk_table_collection_equals(&t1, &t2, 0));

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
    ret = tsk_table_collection_check_integrity(&t1, TSK_CHECK_EDGE_ORDERING);
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
    CU_ASSERT_TRUE(tsk_table_collection_equals(&t1, &t2, 0));
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
    CU_ASSERT_TRUE(tsk_table_collection_equals(&t1, &t2, 0));

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
    ret = tsk_table_collection_check_integrity(&t1, TSK_CHECK_EDGE_ORDERING);
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
    ret = tsk_individual_table_add_row(&tables.individuals, 0, 0, 0, NULL, 0, NULL, 0);
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
    tsk_id_t id_zeros[] = { 0, 0, 0, 0, 0, 0, 0, 0 };

    ret = tsk_table_collection_init(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* We can't trigger a column overflow with one element because the parameter
     * value is 32 bit */
    ret = tsk_individual_table_add_row(
        &tables.individuals, 0, &zero, 1, NULL, 0, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_individual_table_add_row(
        &tables.individuals, 0, NULL, too_big, NULL, 0, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_COLUMN_OVERFLOW);
    ret = tsk_individual_table_add_row(
        &tables.individuals, 0, NULL, 0, NULL, 0, zeros, 1);
    CU_ASSERT_EQUAL_FATAL(ret, 1);
    ret = tsk_individual_table_add_row(
        &tables.individuals, 0, NULL, 0, NULL, 0, NULL, too_big);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_COLUMN_OVERFLOW);
    ret = tsk_individual_table_add_row(
        &tables.individuals, 0, NULL, 0, id_zeros, 1, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 2);
    ret = tsk_individual_table_add_row(
        &tables.individuals, 0, NULL, 0, NULL, too_big, NULL, 0);
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
    const char *individuals = "1      0.25     -1\n"
                              "2      0.5,0.25 2\n"
                              "3      0.5,0.25 0\n";

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

    /* A mixture of known and unknown times on a site fails */
    ret = tsk_mutation_table_clear(&tables.mutations);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 0, 0, TSK_NULL, TSK_UNKNOWN_TIME, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 0, 0, TSK_NULL, 0, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_check_integrity(&tables, TSK_CHECK_MUTATION_ORDERING);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_MUTATION_TIME_HAS_BOTH_KNOWN_AND_UNKNOWN);

    /* But on different sites, passes */
    ret = tsk_mutation_table_clear(&tables.mutations);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 0, 0, TSK_NULL, TSK_UNKNOWN_TIME, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 1, 0, TSK_NULL, 0, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_check_integrity(&tables, TSK_CHECK_MUTATION_ORDERING);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

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
    ret = tsk_table_collection_check_integrity(&tables, TSK_CHECK_MUTATION_ORDERING);
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
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_mutation_table_clear(&tables.mutations);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 0, 1, TSK_NULL, TSK_UNKNOWN_TIME, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 1, 1, 0, TSK_UNKNOWN_TIME, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_check_integrity(&tables, TSK_CHECK_MUTATION_ORDERING);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_MUTATION_PARENT_DIFFERENT_SITE);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_mutation_table_clear(&tables.mutations);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 1, 1, TSK_NULL, TSK_UNKNOWN_TIME, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 0, 1, TSK_NULL, TSK_UNKNOWN_TIME, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_check_integrity(&tables, TSK_CHECK_MUTATION_ORDERING);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_UNSORTED_MUTATIONS);

    /* Unknown times pass */
    ret = tsk_mutation_table_clear(&tables.mutations);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 0, 0, TSK_NULL, TSK_UNKNOWN_TIME, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 0, 0, TSK_NULL, TSK_UNKNOWN_TIME, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_check_integrity(&tables, TSK_CHECK_MUTATION_ORDERING);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* Correctly ordered times pass */
    ret = tsk_mutation_table_clear(&tables.mutations);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 0, 0, TSK_NULL, 1, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 0, 0, TSK_NULL, 1, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 0, 0, TSK_NULL, 0, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_check_integrity(&tables, TSK_CHECK_MUTATION_ORDERING);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* Incorrectly ordered times fail */
    ret = tsk_mutation_table_clear(&tables.mutations);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 0, 0, TSK_NULL, 0, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 0, 0, TSK_NULL, 1, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_check_integrity(&tables, TSK_CHECK_MUTATION_ORDERING);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_UNSORTED_MUTATIONS);

    /* Putting incorrectly ordered times on diff sites passes */
    ret = tsk_mutation_table_clear(&tables.mutations);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 0, 0, TSK_NULL, 1, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 0, 0, TSK_NULL, 0, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 1, 0, TSK_NULL, 2, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 1, 0, TSK_NULL, 1, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_check_integrity(&tables, TSK_CHECK_MUTATION_ORDERING);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_mutation_table_clear(&tables.mutations);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 0, 0, TSK_NULL, NAN, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_TIME_NONFINITE);

    ret = tsk_mutation_table_clear(&tables.mutations);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 0, 0, TSK_NULL, INFINITY, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_TIME_NONFINITE);

    ret = tsk_mutation_table_clear(&tables.mutations);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 1, 1, TSK_NULL, 0, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_check_integrity(&tables, TSK_CHECK_MUTATION_ORDERING);
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
    ret = tsk_table_collection_check_integrity(&tables, TSK_CHECK_MUTATION_ORDERING);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_MUTATION_TIME_OLDER_THAN_PARENT_MUTATION);

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
    ret = tsk_migration_table_clear(&tables.migrations);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    parse_individuals(individuals, &tables.individuals);
    CU_ASSERT_EQUAL_FATAL(tables.individuals.num_rows, 3);
    ret = tsk_table_collection_check_integrity(&tables, TSK_CHECK_INDIVIDUAL_ORDERING);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_UNSORTED_INDIVIDUALS);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* Check that an individual can't be its own parent */
    tables.individuals.parents[0] = 0;
    tables.individuals.parents[1] = 1;
    tables.individuals.parents[2] = 2;
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_INDIVIDUAL_SELF_PARENT);

    tables.individuals.parents[0] = -2;
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_INDIVIDUAL_OUT_OF_BOUNDS);

    tsk_table_collection_free(&tables);
}

static void
test_table_collection_check_integrity_no_populations(void)
{
    int ret;
    tsk_treeseq_t ts;
    tsk_table_collection_t tables;

    tsk_treeseq_from_text(&ts, 10, paper_ex_nodes, paper_ex_edges, NULL, paper_ex_sites,
        paper_ex_mutations, paper_ex_individuals, NULL, 0);
    ret = tsk_treeseq_copy_tables(&ts, &tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* Add in some bad population references and check that we can use
     * TSK_NO_CHECK_POPULATION_REFS with TSK_CHECK_TREES */
    tables.nodes.population[0] = 10;
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_POPULATION_OUT_OF_BOUNDS);
    ret = tsk_table_collection_check_integrity(&tables, TSK_CHECK_TREES);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_POPULATION_OUT_OF_BOUNDS);
    ret = tsk_table_collection_check_integrity(&tables, TSK_NO_CHECK_POPULATION_REFS);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_check_integrity(
        &tables, TSK_CHECK_TREES | TSK_NO_CHECK_POPULATION_REFS);
    /* CHECK_TREES returns the number of trees */
    CU_ASSERT_EQUAL_FATAL(ret, 3);
    tables.nodes.population[0] = TSK_NULL;

    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_migration_table_add_row(
        &tables.migrations, 0.4, 0.5, 1, 0, 1, 1.5, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_check_integrity(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_POPULATION_OUT_OF_BOUNDS);
    ret = tsk_table_collection_check_integrity(&tables, TSK_CHECK_TREES);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_POPULATION_OUT_OF_BOUNDS);
    ret = tsk_table_collection_check_integrity(&tables, TSK_NO_CHECK_POPULATION_REFS);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_check_integrity(
        &tables, TSK_CHECK_TREES | TSK_NO_CHECK_POPULATION_REFS);
    CU_ASSERT_EQUAL_FATAL(ret, 3);

    tsk_table_collection_free(&tables);
    tsk_treeseq_free(&ts);
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
    ret = tsk_table_collection_subset(&tables, NULL, 0, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    // four nodes from two diploids; the first is from pop 0
    ret = tsk_node_table_add_row(&tables.nodes, TSK_NODE_IS_SAMPLE, 0.0, 0, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_node_table_add_row(&tables.nodes, TSK_NODE_IS_SAMPLE, 1.0, 0, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_node_table_add_row(
        &tables.nodes, TSK_NODE_IS_SAMPLE, 2.0, TSK_NULL, 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_node_table_add_row(
        &tables.nodes, TSK_NODE_IS_SAMPLE, 0.0, TSK_NULL, 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_individual_table_add_row(
        &tables.individuals, 0, NULL, 0, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_individual_table_add_row(
        &tables.individuals, 0, NULL, 0, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    // unused individual
    ret = tsk_individual_table_add_row(
        &tables.individuals, 0, NULL, 0, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_population_table_add_row(&tables.populations, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    // unused population
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
    // unused site
    ret = tsk_site_table_add_row(&tables.sites, 0.5, "C", 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 0, 0, TSK_NULL, TSK_UNKNOWN_TIME, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 0, 0, 0, TSK_UNKNOWN_TIME, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 1, 1, TSK_NULL, TSK_UNKNOWN_TIME, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);

    // empty nodes should get empty tables
    ret = tsk_table_collection_copy(&tables, &tables_copy, TSK_NO_INIT | options);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_subset(&tables_copy, NULL, 0, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(tables_copy.nodes.num_rows, 0);
    CU_ASSERT_EQUAL_FATAL(tables_copy.individuals.num_rows, 0);
    CU_ASSERT_EQUAL_FATAL(tables_copy.populations.num_rows, 0);
    CU_ASSERT_EQUAL_FATAL(tables_copy.sites.num_rows, 0);
    CU_ASSERT_EQUAL_FATAL(tables_copy.mutations.num_rows, 0);

    // unless NO_CHANGE_POPULATIONS is provided
    ret = tsk_table_collection_copy(&tables, &tables_copy, TSK_NO_INIT | options);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_subset(&tables_copy, NULL, 0, TSK_NO_CHANGE_POPULATIONS);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(tables_copy.nodes.num_rows, 0);
    CU_ASSERT_EQUAL_FATAL(tables_copy.individuals.num_rows, 0);
    CU_ASSERT_EQUAL_FATAL(tables_copy.sites.num_rows, 0);
    CU_ASSERT_EQUAL_FATAL(tables_copy.mutations.num_rows, 0);
    CU_ASSERT_FATAL(
        tsk_population_table_equals(&tables.populations, &tables_copy.populations, 0));

    // or KEEP_UNREFERENCED
    ret = tsk_table_collection_copy(&tables, &tables_copy, TSK_NO_INIT | options);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_subset(&tables_copy, NULL, 0, TSK_KEEP_UNREFERENCED);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(tables_copy.nodes.num_rows, 0);
    CU_ASSERT_EQUAL_FATAL(tables_copy.individuals.num_rows, 3);
    CU_ASSERT_EQUAL_FATAL(tables_copy.populations.num_rows, 2);
    CU_ASSERT_EQUAL_FATAL(tables_copy.mutations.num_rows, 0);
    CU_ASSERT_FATAL(tsk_site_table_equals(&tables.sites, &tables_copy.sites, 0));

    // or both
    ret = tsk_table_collection_copy(&tables, &tables_copy, TSK_NO_INIT | options);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_subset(
        &tables_copy, NULL, 0, TSK_KEEP_UNREFERENCED | TSK_NO_CHANGE_POPULATIONS);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(tables_copy.nodes.num_rows, 0);
    CU_ASSERT_EQUAL_FATAL(tables_copy.individuals.num_rows, 3);
    CU_ASSERT_EQUAL_FATAL(tables_copy.mutations.num_rows, 0);
    CU_ASSERT_FATAL(
        tsk_population_table_equals(&tables.populations, &tables_copy.populations, 0));
    CU_ASSERT_FATAL(tsk_site_table_equals(&tables.sites, &tables_copy.sites, 0));

    // the identity transformation, since unused inds/pops are at the end
    for (k = 0; k < 4; k++) {
        nodes[k] = k;
    }
    ret = tsk_table_collection_copy(&tables, &tables_copy, TSK_NO_INIT | options);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_subset(&tables_copy, nodes, 4, TSK_KEEP_UNREFERENCED);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_FATAL(tsk_table_collection_equals(&tables, &tables_copy, 0));

    // or, remove unused things:
    ret = tsk_table_collection_copy(&tables, &tables_copy, TSK_NO_INIT | options);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_subset(&tables_copy, nodes, 4, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_FATAL(tsk_node_table_equals(&tables.nodes, &tables_copy.nodes, 0));
    CU_ASSERT_EQUAL_FATAL(tables_copy.individuals.num_rows, 2);
    CU_ASSERT_EQUAL_FATAL(tables_copy.populations.num_rows, 1);
    CU_ASSERT_EQUAL_FATAL(tables_copy.sites.num_rows, 2);
    CU_ASSERT_FATAL(
        tsk_mutation_table_equals(&tables.mutations, &tables_copy.mutations, 0));

    // reverse twice should get back to the start, since unused inds/pops are at the end
    for (k = 0; k < 4; k++) {
        nodes[k] = 3 - k;
    }
    ret = tsk_table_collection_copy(&tables, &tables_copy, TSK_NO_INIT | options);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_subset(&tables_copy, nodes, 4, TSK_KEEP_UNREFERENCED);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_subset(&tables_copy, nodes, 4, TSK_KEEP_UNREFERENCED);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_FATAL(tsk_table_collection_equals(&tables, &tables_copy, 0));

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
test_table_collection_subset_unsorted(void)
{
    int ret;
    tsk_table_collection_t tables;
    tsk_table_collection_t tables_copy;
    int k;
    tsk_id_t nodes[3];

    ret = tsk_table_collection_init(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tables.sequence_length = 1;
    ret = tsk_table_collection_init(&tables_copy, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    // these tables are a big mess
    ret = tsk_node_table_add_row(
        &tables.nodes, TSK_NODE_IS_SAMPLE, 0.0, TSK_NULL, TSK_NULL, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_node_table_add_row(
        &tables.nodes, TSK_NODE_IS_SAMPLE, 0.5, TSK_NULL, TSK_NULL, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_node_table_add_row(&tables.nodes, 0, 1.0, TSK_NULL, TSK_NULL, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_edge_table_add_row(&tables.edges, 0.0, 0.5, 2, 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_edge_table_add_row(&tables.edges, 0.0, 1.0, 1, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_edge_table_add_row(&tables.edges, 0.5, 1.0, 2, 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_site_table_add_row(&tables.sites, 0.2, "A", 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_site_table_add_row(&tables.sites, 0.4, "A", 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 0, 0, 2, TSK_UNKNOWN_TIME, "B", 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 1, 1, TSK_NULL, TSK_UNKNOWN_TIME, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 0, 0, TSK_NULL, TSK_UNKNOWN_TIME, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);

    // but still, this should leave them unchanged
    for (k = 0; k < 3; k++) {
        nodes[k] = k;
    }
    ret = tsk_table_collection_copy(&tables, &tables_copy, TSK_NO_INIT);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_subset(&tables_copy, nodes, 3, TSK_KEEP_UNREFERENCED);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_FATAL(tsk_table_collection_equals(&tables, &tables_copy, 0));

    tsk_table_collection_free(&tables_copy);
    tsk_table_collection_free(&tables);
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
    tables.sequence_length = 1;
    ret = tsk_table_collection_init(&tables_copy, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    // four nodes from two diploids; the first is from pop 0
    ret = tsk_node_table_add_row(&tables.nodes, TSK_NODE_IS_SAMPLE, 0.0, 0, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_node_table_add_row(&tables.nodes, TSK_NODE_IS_SAMPLE, 1.0, 0, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_node_table_add_row(
        &tables.nodes, TSK_NODE_IS_SAMPLE, 2.0, TSK_NULL, 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_node_table_add_row(
        &tables.nodes, TSK_NODE_IS_SAMPLE, 0.0, TSK_NULL, 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_individual_table_add_row(
        &tables.individuals, 0, NULL, 0, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_individual_table_add_row(
        &tables.individuals, 0, NULL, 0, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_population_table_add_row(&tables.populations, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_edge_table_add_row(&tables.edges, 0.0, 1.0, 1, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_build_index(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* Migrations are not supported */
    ret = tsk_table_collection_copy(&tables, &tables_copy, TSK_NO_INIT);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tsk_migration_table_add_row(&tables_copy.migrations, 0, 1, 0, 0, 0, 0, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(tables_copy.migrations.num_rows, 1);
    ret = tsk_table_collection_subset(&tables_copy, nodes, 4, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_MIGRATIONS_NOT_SUPPORTED);

    // test out of bounds nodes
    ret = tsk_table_collection_copy(&tables, &tables_copy, TSK_NO_INIT);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    nodes[0] = -1;
    ret = tsk_table_collection_subset(&tables_copy, nodes, 4, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_NODE_OUT_OF_BOUNDS);
    nodes[0] = 6;
    ret = tsk_table_collection_copy(&tables, &tables_copy, TSK_NO_INIT);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_subset(&tables_copy, nodes, 4, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_NODE_OUT_OF_BOUNDS);

    // check integrity
    nodes[0] = 0;
    nodes[1] = 1;
    ret = tsk_table_collection_copy(&tables, &tables_copy, TSK_NO_INIT);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_node_table_truncate(&tables_copy.nodes, 3);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_node_table_add_row(
        &tables_copy.nodes, TSK_NODE_IS_SAMPLE, 0.0, -2, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_subset(&tables_copy, nodes, 4, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_POPULATION_OUT_OF_BOUNDS);

    tsk_table_collection_free(&tables);
    tsk_table_collection_free(&tables_copy);
}

static void
test_table_collection_union(void)
{
    int ret;
    tsk_table_collection_t tables;
    tsk_table_collection_t tables_empty;
    tsk_table_collection_t tables_copy;
    tsk_id_t node_mapping[3];
    char example_metadata[100] = "An example of metadata with unicode 🎄🌳🌴🌲🎋";
    tsk_size_t example_metadata_length = (tsk_size_t) strlen(example_metadata);

    memset(node_mapping, 0xff, sizeof(node_mapping));

    ret = tsk_table_collection_init(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tables.sequence_length = 1;
    ret = tsk_table_collection_init(&tables_empty, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tables_empty.sequence_length = 1;
    ret = tsk_table_collection_init(&tables_copy, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    // does not error on empty tables
    ret = tsk_table_collection_union(&tables, &tables_empty, node_mapping, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    // does not error on empty tables but that differ on top level metadata
    ret = tsk_table_collection_set_metadata(
        &tables, example_metadata, example_metadata_length);
    CU_ASSERT_EQUAL(ret, 0);
    ret = tsk_table_collection_union(&tables, &tables_empty, node_mapping, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    // three nodes, two pop, three ind, two edge, two site, two mut
    ret = tsk_node_table_add_row(&tables.nodes, TSK_NODE_IS_SAMPLE, 0.0, 0, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_node_table_add_row(&tables.nodes, TSK_NODE_IS_SAMPLE, 0.0, 1, 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_node_table_add_row(&tables.nodes, TSK_NODE_IS_SAMPLE, 0.5, 1, 2, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_individual_table_add_row(
        &tables.individuals, 0, NULL, 0, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_individual_table_add_row(
        &tables.individuals, 0, NULL, 0, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_individual_table_add_row(
        &tables.individuals, 0, NULL, 0, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_population_table_add_row(&tables.populations, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_population_table_add_row(&tables.populations, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_edge_table_add_row(&tables.edges, 0.0, 1.0, 2, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_edge_table_add_row(&tables.edges, 0.0, 1.0, 2, 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_site_table_add_row(&tables.sites, 0.4, "T", 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_site_table_add_row(&tables.sites, 0.2, "A", 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 0, 0, TSK_NULL, TSK_UNKNOWN_TIME, NULL, 0, NULL, 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 1, 1, TSK_NULL, TSK_UNKNOWN_TIME, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_build_index(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_sort(&tables, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    // union with empty should not change
    // other is empty
    ret = tsk_table_collection_copy(&tables, &tables_copy, TSK_NO_INIT);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_union(
        &tables_copy, &tables_empty, node_mapping, TSK_UNION_NO_CHECK_SHARED);
    CU_ASSERT_FATAL(tsk_table_collection_equals(&tables, &tables_copy, 0));
    // self is empty
    ret = tsk_table_collection_clear(&tables_copy, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_union(
        &tables_copy, &tables, node_mapping, TSK_UNION_NO_CHECK_SHARED);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_FATAL(tsk_table_collection_equals(&tables, &tables_copy, 0));

    // union all shared nodes + subset original nodes = original table
    ret = tsk_table_collection_copy(&tables, &tables_copy, TSK_NO_INIT);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_union(
        &tables_copy, &tables, node_mapping, TSK_UNION_NO_CHECK_SHARED);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    node_mapping[0] = 0;
    node_mapping[1] = 1;
    node_mapping[2] = 2;
    ret = tsk_table_collection_subset(&tables_copy, node_mapping, 3, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_FATAL(tsk_table_collection_equals(&tables, &tables_copy, 0));

    // union with one shared node
    ret = tsk_table_collection_copy(&tables, &tables_copy, TSK_NO_INIT);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    node_mapping[0] = TSK_NULL;
    node_mapping[1] = TSK_NULL;
    node_mapping[2] = 2;
    ret = tsk_table_collection_union(&tables_copy, &tables, node_mapping, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(
        tables_copy.populations.num_rows, tables.populations.num_rows + 2);
    CU_ASSERT_EQUAL_FATAL(
        tables_copy.individuals.num_rows, tables.individuals.num_rows + 2);
    CU_ASSERT_EQUAL_FATAL(tables_copy.nodes.num_rows, tables.nodes.num_rows + 2);
    CU_ASSERT_EQUAL_FATAL(tables_copy.edges.num_rows, tables.edges.num_rows + 2);
    CU_ASSERT_EQUAL_FATAL(tables_copy.sites.num_rows, tables.sites.num_rows);
    CU_ASSERT_EQUAL_FATAL(tables_copy.mutations.num_rows, tables.mutations.num_rows + 2);

    // union with one shared node, but no add pop
    ret = tsk_table_collection_copy(&tables, &tables_copy, TSK_NO_INIT);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    node_mapping[0] = TSK_NULL;
    node_mapping[1] = TSK_NULL;
    node_mapping[2] = 2;
    ret = tsk_table_collection_union(
        &tables_copy, &tables, node_mapping, TSK_UNION_NO_ADD_POP);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(tables_copy.populations.num_rows, tables.populations.num_rows);
    CU_ASSERT_EQUAL_FATAL(
        tables_copy.individuals.num_rows, tables.individuals.num_rows + 2);
    CU_ASSERT_EQUAL_FATAL(tables_copy.nodes.num_rows, tables.nodes.num_rows + 2);
    CU_ASSERT_EQUAL_FATAL(tables_copy.edges.num_rows, tables.edges.num_rows + 2);
    CU_ASSERT_EQUAL_FATAL(tables_copy.sites.num_rows, tables.sites.num_rows);
    CU_ASSERT_EQUAL_FATAL(tables_copy.mutations.num_rows, tables.mutations.num_rows + 2);

    tsk_table_collection_free(&tables_copy);
    tsk_table_collection_free(&tables_empty);
    tsk_table_collection_free(&tables);
}

static void
test_table_collection_union_middle_merge(void)
{
    /* Test ability to have non-shared history both above and below the
     * shared bits. The full genealogy, in `tu`, is:
     *  3   4
     *   \ /
     *    2
     *   / \
     *  0   1
     * and the left lineage is in `ta` and right in `tb` */
    int ret;
    tsk_id_t node_mapping[] = { TSK_NULL, 1, TSK_NULL };
    tsk_id_t node_order[] = { 0, 3, 1, 2, 4 };
    tsk_table_collection_t ta, tb, tu;
    ret = tsk_table_collection_init(&ta, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ta.sequence_length = 1;
    ret = tsk_table_collection_init(&tb, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tb.sequence_length = 1;
    ret = tsk_table_collection_init(&tu, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tu.sequence_length = 1;

    ret = tsk_node_table_add_row(
        &tu.nodes, TSK_NODE_IS_SAMPLE, 0, TSK_NULL, TSK_NULL, NULL, 0); // node u0
    CU_ASSERT(ret >= 0);
    ret = tsk_node_table_add_row(
        &ta.nodes, TSK_NODE_IS_SAMPLE, 0, TSK_NULL, TSK_NULL, NULL, 0); // node a0 = u0
    CU_ASSERT(ret >= 0);
    ret = tsk_node_table_add_row(
        &tu.nodes, TSK_NODE_IS_SAMPLE, 0, TSK_NULL, TSK_NULL, NULL, 0); // node u1
    CU_ASSERT(ret >= 0);
    ret = tsk_node_table_add_row(
        &tb.nodes, TSK_NODE_IS_SAMPLE, 0, TSK_NULL, TSK_NULL, NULL, 0); // node b0 = u1
    CU_ASSERT(ret >= 0);
    ret = tsk_node_table_add_row(
        &tu.nodes, 0, 1, TSK_NULL, TSK_NULL, NULL, 0); // node u2
    CU_ASSERT(ret >= 0);
    ret = tsk_edge_table_add_row(&tu.edges, 0, 1, 2, 0, NULL, 0);
    CU_ASSERT(ret >= 0);
    ret = tsk_edge_table_add_row(&tu.edges, 0, 1, 2, 1, NULL, 0);
    CU_ASSERT(ret >= 0);
    ret = tsk_node_table_add_row(
        &ta.nodes, 0, 1, TSK_NULL, TSK_NULL, NULL, 0); // node a1 = u2
    CU_ASSERT(ret >= 0);
    ret = tsk_edge_table_add_row(&ta.edges, 0, 1, 1, 0, NULL, 0);
    CU_ASSERT(ret >= 0);
    ret = tsk_node_table_add_row(
        &tb.nodes, 0, 1, TSK_NULL, TSK_NULL, NULL, 0); // node b1 = u2
    CU_ASSERT(ret >= 0);
    ret = tsk_edge_table_add_row(&tb.edges, 0, 1, 1, 0, NULL, 0);
    CU_ASSERT(ret >= 0);
    ret = tsk_node_table_add_row(
        &tu.nodes, 0, 2, TSK_NULL, TSK_NULL, NULL, 0); // node u3
    CU_ASSERT(ret >= 0);
    ret = tsk_edge_table_add_row(&tu.edges, 0, 0.5, 3, 2, NULL, 0);
    CU_ASSERT(ret >= 0);
    ret = tsk_node_table_add_row(
        &ta.nodes, 0, 2, TSK_NULL, TSK_NULL, NULL, 0); // node a2 = u3
    CU_ASSERT(ret >= 0);
    ret = tsk_edge_table_add_row(&ta.edges, 0, 0.5, 2, 1, NULL, 0);
    CU_ASSERT(ret >= 0);
    ret = tsk_node_table_add_row(
        &tu.nodes, 0, 2, TSK_NULL, TSK_NULL, NULL, 0); // node u4
    CU_ASSERT(ret >= 0);
    ret = tsk_edge_table_add_row(&tu.edges, 0.5, 1, 4, 2, NULL, 0);
    CU_ASSERT(ret >= 0);
    ret = tsk_node_table_add_row(
        &tb.nodes, 0, 2, TSK_NULL, TSK_NULL, NULL, 0); // node b2 = u4
    CU_ASSERT(ret >= 0);
    ret = tsk_edge_table_add_row(&tb.edges, 0.5, 1, 2, 1, NULL, 0);
    CU_ASSERT(ret >= 0);

    ret = tsk_site_table_add_row(&ta.sites, 0.25, "A", 1, NULL, 0);
    CU_ASSERT(ret >= 0);
    ret = tsk_site_table_add_row(&ta.sites, 0.75, "X", 1, NULL, 0);
    CU_ASSERT(ret >= 0);
    ret = tsk_site_table_add_row(&tb.sites, 0.25, "A", 1, NULL, 0);
    CU_ASSERT(ret >= 0);
    ret = tsk_site_table_add_row(&tb.sites, 0.75, "X", 1, NULL, 0);
    CU_ASSERT(ret >= 0);
    ret = tsk_site_table_add_row(&tu.sites, 0.25, "A", 1, NULL, 0);
    CU_ASSERT(ret >= 0);
    ret = tsk_site_table_add_row(&tu.sites, 0.75, "X", 1, NULL, 0);
    CU_ASSERT(ret >= 0);

    ret = tsk_mutation_table_add_row(
        &tu.mutations, 0, 3, TSK_NULL, 3.5, "B", 1, NULL, 0);
    CU_ASSERT(ret >= 0);
    ret = tsk_mutation_table_add_row(
        &ta.mutations, 0, 2, TSK_NULL, 3.5, "B", 1, NULL, 0);
    CU_ASSERT(ret >= 0);
    ret = tsk_mutation_table_add_row(
        &tu.mutations, 0, 2, TSK_NULL, 1.5, "D", 1, NULL, 0);
    CU_ASSERT(ret >= 0);
    ret = tsk_mutation_table_add_row(
        &ta.mutations, 0, 1, TSK_NULL, 1.5, "D", 1, NULL, 0);
    CU_ASSERT(ret >= 0);
    ret = tsk_mutation_table_add_row(
        &tb.mutations, 0, 1, TSK_NULL, 1.5, "D", 1, NULL, 0);
    CU_ASSERT(ret >= 0);
    ret = tsk_mutation_table_add_row(
        &tu.mutations, 0, 2, TSK_NULL, 1.2, "E", 1, NULL, 0);
    CU_ASSERT(ret >= 0);
    ret = tsk_mutation_table_add_row(
        &ta.mutations, 0, 1, TSK_NULL, 1.2, "E", 1, NULL, 0);
    CU_ASSERT(ret >= 0);
    ret = tsk_mutation_table_add_row(
        &tb.mutations, 0, 1, TSK_NULL, 1.2, "E", 1, NULL, 0);
    CU_ASSERT(ret >= 0);
    ret = tsk_mutation_table_add_row(
        &tu.mutations, 0, 0, TSK_NULL, 0.5, "C", 1, NULL, 0);
    CU_ASSERT(ret >= 0);
    ret = tsk_mutation_table_add_row(
        &ta.mutations, 0, 0, TSK_NULL, 0.5, "C", 1, NULL, 0);
    CU_ASSERT(ret >= 0);
    ret = tsk_mutation_table_add_row(
        &tu.mutations, 1, 4, TSK_NULL, 2.4, "Y", 1, NULL, 0);
    CU_ASSERT(ret >= 0);
    ret = tsk_mutation_table_add_row(
        &tb.mutations, 1, 2, TSK_NULL, 2.4, "Y", 1, NULL, 0);
    CU_ASSERT(ret >= 0);
    ret = tsk_mutation_table_add_row(
        &tu.mutations, 1, 1, TSK_NULL, 0.4, "Z", 1, NULL, 0);
    CU_ASSERT(ret >= 0);
    ret = tsk_mutation_table_add_row(
        &tb.mutations, 1, 0, TSK_NULL, 0.4, "Z", 1, NULL, 0);
    CU_ASSERT(ret >= 0);

    ret = tsk_table_collection_build_index(&ta, 0);
    CU_ASSERT_EQUAL(ret, 0);
    ret = tsk_table_collection_compute_mutation_parents(&ta, 0);
    CU_ASSERT_EQUAL(ret, 0);
    ret = tsk_table_collection_build_index(&tb, 0);
    CU_ASSERT_EQUAL(ret, 0);
    ret = tsk_table_collection_compute_mutation_parents(&tb, 0);
    CU_ASSERT_EQUAL(ret, 0);
    ret = tsk_table_collection_build_index(&tu, 0);
    CU_ASSERT_EQUAL(ret, 0);
    ret = tsk_table_collection_compute_mutation_parents(&tu, 0);
    CU_ASSERT_EQUAL(ret, 0);

    ret = tsk_table_collection_union(&ta, &tb, node_mapping, 0);
    CU_ASSERT_EQUAL(ret, 0);
    ret = tsk_table_collection_subset(&ta, node_order, 5, 0);
    CU_ASSERT_EQUAL(ret, 0);
    ret = tsk_provenance_table_clear(&ta.provenances);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_FATAL(tsk_table_collection_equals(&tu, &ta, 0));

    tsk_table_collection_free(&ta);
    tsk_table_collection_free(&tb);
    tsk_table_collection_free(&tu);
}

static void
test_table_collection_union_errors(void)
{
    int ret;
    tsk_table_collection_t tables;
    tsk_table_collection_t tables_copy;
    tsk_id_t node_mapping[] = { 0, 1 };

    ret = tsk_table_collection_init(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tables.sequence_length = 1;
    ret = tsk_table_collection_init(&tables_copy, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    // two nodes, two pop, two ind, one edge, one site, one mut
    ret = tsk_node_table_add_row(&tables.nodes, TSK_NODE_IS_SAMPLE, 0.0, 0, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_node_table_add_row(&tables.nodes, TSK_NODE_IS_SAMPLE, 0.5, 1, 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_individual_table_add_row(
        &tables.individuals, 0, NULL, 0, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_individual_table_add_row(
        &tables.individuals, 0, NULL, 0, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_population_table_add_row(&tables.populations, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_population_table_add_row(&tables.populations, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_edge_table_add_row(&tables.edges, 0.0, 1.0, 1, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_site_table_add_row(&tables.sites, 0.2, "A", 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 0, 0, TSK_NULL, TSK_UNKNOWN_TIME, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);

    // trigger diff histories error
    ret = tsk_table_collection_copy(&tables, &tables_copy, TSK_NO_INIT);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_mutation_table_add_row(
        &tables_copy.mutations, 0, 1, TSK_NULL, TSK_UNKNOWN_TIME, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_union(&tables_copy, &tables, node_mapping, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_UNION_DIFF_HISTORIES);

    // Migrations are not supported
    ret = tsk_table_collection_copy(&tables, &tables_copy, TSK_NO_INIT);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tsk_migration_table_add_row(&tables_copy.migrations, 0, 1, 0, 0, 0, 0, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(tables_copy.migrations.num_rows, 1);
    ret = tsk_table_collection_union(
        &tables_copy, &tables, node_mapping, TSK_UNION_NO_CHECK_SHARED);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_MIGRATIONS_NOT_SUPPORTED);

    // test out of bounds node_mapping
    node_mapping[0] = -4;
    node_mapping[1] = 6;
    ret = tsk_table_collection_copy(&tables, &tables_copy, TSK_NO_INIT);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_union(&tables_copy, &tables, node_mapping, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_UNION_BAD_MAP);

    // check integrity
    node_mapping[0] = 0;
    node_mapping[1] = 1;
    ret = tsk_node_table_add_row(
        &tables_copy.nodes, TSK_NODE_IS_SAMPLE, 0.0, -2, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_union(&tables_copy, &tables, node_mapping, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_POPULATION_OUT_OF_BOUNDS);
    ret = tsk_table_collection_copy(&tables, &tables_copy, TSK_NO_INIT);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_node_table_add_row(&tables.nodes, TSK_NODE_IS_SAMPLE, 0.0, -2, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_table_collection_union(&tables, &tables_copy, node_mapping, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_POPULATION_OUT_OF_BOUNDS);

    tsk_table_collection_free(&tables_copy);
    tsk_table_collection_free(&tables);
}

static void
test_table_collection_clear_with_options(tsk_flags_t options)
{
    int ret;
    tsk_table_collection_t tables;
    bool clear_provenance = !!(options & TSK_CLEAR_PROVENANCE);
    bool clear_metadata_schemas = !!(options & TSK_CLEAR_METADATA_SCHEMAS);
    bool clear_ts_metadata = !!(options & TSK_CLEAR_TS_METADATA_AND_SCHEMA);
    tsk_bookmark_t num_rows;
    tsk_bookmark_t expected_rows = { .provenances = clear_provenance ? 0 : 1 };
    tsk_size_t expected_len = clear_metadata_schemas ? 0 : 4;
    tsk_size_t expected_len_ts = clear_ts_metadata ? 0 : 4;

    ret = tsk_table_collection_init(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tables.sequence_length = 1;

    ret = tsk_node_table_add_row(&tables.nodes, TSK_NODE_IS_SAMPLE, 0.0, 0, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_node_table_add_row(&tables.nodes, TSK_NODE_IS_SAMPLE, 0.5, 1, 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_individual_table_add_row(
        &tables.individuals, 0, NULL, 0, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_individual_table_add_row(
        &tables.individuals, 0, NULL, 0, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_population_table_add_row(&tables.populations, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_population_table_add_row(&tables.populations, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_edge_table_add_row(&tables.edges, 0.0, 1.0, 1, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_site_table_add_row(&tables.sites, 0.2, "A", 1, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_mutation_table_add_row(
        &tables.mutations, 0, 0, TSK_NULL, TSK_UNKNOWN_TIME, NULL, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);
    ret = tsk_migration_table_add_row(&tables.migrations, 0, 1, 0, 0, 0, 0, NULL, 0);
    CU_ASSERT_FATAL(ret >= 0);

    ret = tsk_table_collection_build_index(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_individual_table_set_metadata_schema(&tables.individuals, "test", 4);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_node_table_set_metadata_schema(&tables.nodes, "test", 4);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_edge_table_set_metadata_schema(&tables.edges, "test", 4);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_migration_table_set_metadata_schema(&tables.migrations, "test", 4);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_site_table_set_metadata_schema(&tables.sites, "test", 4);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_mutation_table_set_metadata_schema(&tables.mutations, "test", 4);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_population_table_set_metadata_schema(&tables.populations, "test", 4);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_table_collection_set_metadata(&tables, "test", 4);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_set_metadata_schema(&tables, "test", 4);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_provenance_table_add_row(&tables.provenances, "today", 5, "test", 4);
    CU_ASSERT_FATAL(ret >= 0);

    ret = tsk_table_collection_clear(&tables, options);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_table_collection_record_num_rows(&tables, &num_rows);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(num_rows.individuals, expected_rows.individuals);
    CU_ASSERT_EQUAL(num_rows.nodes, expected_rows.nodes);
    CU_ASSERT_EQUAL(num_rows.edges, expected_rows.edges);
    CU_ASSERT_EQUAL(num_rows.migrations, expected_rows.migrations);
    CU_ASSERT_EQUAL(num_rows.sites, expected_rows.sites);
    CU_ASSERT_EQUAL(num_rows.mutations, expected_rows.mutations);
    CU_ASSERT_EQUAL(num_rows.populations, expected_rows.populations);
    CU_ASSERT_EQUAL(num_rows.provenances, expected_rows.provenances);

    CU_ASSERT_FALSE(tsk_table_collection_has_index(&tables, 0));

    CU_ASSERT_EQUAL(tables.individuals.metadata_schema_length, expected_len);
    CU_ASSERT_EQUAL(tables.nodes.metadata_schema_length, expected_len);
    CU_ASSERT_EQUAL(tables.edges.metadata_schema_length, expected_len);
    CU_ASSERT_EQUAL(tables.migrations.metadata_schema_length, expected_len);
    CU_ASSERT_EQUAL(tables.sites.metadata_schema_length, expected_len);
    CU_ASSERT_EQUAL(tables.mutations.metadata_schema_length, expected_len);
    CU_ASSERT_EQUAL(tables.populations.metadata_schema_length, expected_len);
    CU_ASSERT_EQUAL(tables.metadata_schema_length, expected_len_ts);
    CU_ASSERT_EQUAL(tables.metadata_length, expected_len_ts);

    tsk_table_collection_free(&tables);
}

static void
test_table_collection_clear(void)
{
    test_table_collection_clear_with_options(0);
    test_table_collection_clear_with_options(TSK_CLEAR_PROVENANCE);
    test_table_collection_clear_with_options(TSK_CLEAR_METADATA_SCHEMAS);
    test_table_collection_clear_with_options(TSK_CLEAR_TS_METADATA_AND_SCHEMA);
    test_table_collection_clear_with_options(
        TSK_CLEAR_PROVENANCE | TSK_CLEAR_METADATA_SCHEMAS);
    test_table_collection_clear_with_options(
        TSK_CLEAR_PROVENANCE | TSK_CLEAR_TS_METADATA_AND_SCHEMA);
    test_table_collection_clear_with_options(
        TSK_CLEAR_METADATA_SCHEMAS | TSK_CLEAR_TS_METADATA_AND_SCHEMA);
    test_table_collection_clear_with_options(TSK_CLEAR_PROVENANCE
                                             | TSK_CLEAR_METADATA_SCHEMAS
                                             | TSK_CLEAR_TS_METADATA_AND_SCHEMA);
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
        { "test_table_collection_equals_options", test_table_collection_equals_options },
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
        { "test_ibd_finder", test_ibd_finder },
        { "test_ibd_finder_multiple_trees", test_ibd_finder_multiple_trees },
        { "test_ibd_finder_empty_result", test_ibd_finder_empty_result },
        { "test_ibd_finder_min_length_max_time", test_ibd_finder_min_length_max_time },
        { "test_ibd_finder_samples_are_descendants",
            test_ibd_finder_samples_are_descendants },
        { "test_ibd_finder_multiple_ibd_paths", test_ibd_finder_multiple_ibd_paths },
        { "test_ibd_finder_odd_topologies", test_ibd_finder_odd_topologies },
        { "test_ibd_finder_errors", test_ibd_finder_errors },
        { "test_sorter_interface", test_sorter_interface },
        { "test_sort_tables_canonical_errors", test_sort_tables_canonical_errors },
        { "test_sort_tables_canonical", test_sort_tables_canonical },
        { "test_sort_tables_drops_indexes", test_sort_tables_drops_indexes },
        { "test_sort_tables_edge_metadata", test_sort_tables_edge_metadata },
        { "test_sort_tables_errors", test_sort_tables_errors },
        { "test_sort_tables_individuals", test_sort_tables_individuals },
        { "test_sort_tables_mutation_times", test_sort_tables_mutation_times },
        { "test_sort_tables_migrations", test_sort_tables_migrations },
        { "test_sort_tables_no_edge_metadata", test_sort_tables_no_edge_metadata },
        { "test_sort_tables_offsets", test_sort_tables_offsets },
        { "test_edge_update_invalidates_index", test_edge_update_invalidates_index },
        { "test_copy_table_collection", test_copy_table_collection },
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
        { "test_table_collection_check_integrity_no_populations",
            test_table_collection_check_integrity_no_populations },
        { "test_table_collection_subset", test_table_collection_subset },
        { "test_table_collection_subset_unsorted",
            test_table_collection_subset_unsorted },
        { "test_table_collection_subset_errors", test_table_collection_subset_errors },
        { "test_table_collection_union", test_table_collection_union },
        { "test_table_collection_union_middle_merge",
            test_table_collection_union_middle_merge },
        { "test_table_collection_union_errors", test_table_collection_union_errors },
        { "test_table_collection_clear", test_table_collection_clear },
        { NULL, NULL },
    };

    return test_main(tests, argc, argv);
}
