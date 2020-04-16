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

typedef struct {
    const char *name;
    void *array;
    tsk_size_t len;
    int type;
} write_table_col_t;

static void
write_table_cols(kastore_t *store, write_table_col_t *write_cols, size_t num_cols)
{
    size_t j;
    int ret;

    for (j = 0; j < num_cols; j++) {
        ret = kastore_puts(store, write_cols[j].name, write_cols[j].array,
            write_cols[j].len, write_cols[j].type, 0);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
    }
}

static void
copy_store_drop_columns(
    tsk_treeseq_t *ts, size_t num_drop_cols, const char **drop_cols, const char *outfile)
{
    int ret = 0;
    char tmpfile[] = "/tmp/tsk_c_test_copy_XXXXXX";
    int fd;
    kastore_t read_store, write_store;
    kaitem_t *item;
    size_t j, k;
    bool keep;

    fd = mkstemp(tmpfile);
    CU_ASSERT_FATAL(fd != -1);
    close(fd);

    ret = tsk_treeseq_dump(ts, tmpfile, 0);
    if (ret != 0) {
        unlink(tmpfile);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
    }

    ret = kastore_open(&read_store, tmpfile, "r", KAS_READ_ALL);
    /* We can now unlink the file as either kastore has read it all, or failed */
    unlink(tmpfile);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_open(&write_store, outfile, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* Note: this API is not a documented part of kastore, so may be subject to
     * change. */
    for (j = 0; j < read_store.num_items; j++) {
        item = &read_store.items[j];
        keep = true;
        for (k = 0; k < num_drop_cols; k++) {
            if (strlen(drop_cols[k]) == item->key_len
                && strncmp(drop_cols[k], item->key, item->key_len) == 0) {
                keep = false;
                break;
            }
        }
        if (keep) {
            ret = kastore_put(&write_store, item->key, item->key_len, item->array,
                item->array_len, item->type, 0);
            CU_ASSERT_EQUAL_FATAL(ret, 0);
        }
    }
    kastore_close(&read_store);
    ret = kastore_close(&write_store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
}

static void
test_format_data_load_errors(void)
{
    size_t uuid_size = 36;
    char uuid[uuid_size];
    char format_name[TSK_FILE_FORMAT_NAME_LENGTH];
    double L[2];
    uint32_t version[2]
        = { TSK_FILE_FORMAT_VERSION_MAJOR, TSK_FILE_FORMAT_VERSION_MINOR };
    write_table_col_t write_cols[] = {
        { "format/name", (void *) format_name, sizeof(format_name), KAS_INT8 },
        { "format/version", (void *) version, 2, KAS_UINT32 },
        { "sequence_length", (void *) L, 1, KAS_FLOAT64 },
        { "uuid", (void *) uuid, (tsk_size_t) uuid_size, KAS_INT8 },
    };
    tsk_table_collection_t tables;
    kastore_t store;
    size_t j;
    int ret;

    L[0] = 1;
    L[1] = 0;
    memcpy(format_name, TSK_FILE_FORMAT_NAME, sizeof(format_name));
    /* Note: this will fail if we ever start parsing the form of the UUID */
    memset(uuid, 0, uuid_size);

    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    write_table_cols(&store, write_cols, sizeof(write_cols) / sizeof(*write_cols));
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_load(&tables, _tmp_file_name, 0);
    /* We've only defined the format headers, so we should fail immediately
     * after with required columns not found */
    CU_ASSERT_FALSE(tsk_is_kas_error(ret));
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_REQUIRED_COL_NOT_FOUND);
    ret = tsk_table_collection_free(&tables);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* Version too old */
    version[0] = TSK_FILE_FORMAT_VERSION_MAJOR - 1;
    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    write_table_cols(&store, write_cols, sizeof(write_cols) / sizeof(*write_cols));
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_load(&tables, _tmp_file_name, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_FILE_VERSION_TOO_OLD);
    ret = tsk_table_collection_free(&tables);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* Version too new */
    version[0] = TSK_FILE_FORMAT_VERSION_MAJOR + 1;
    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    write_table_cols(&store, write_cols, sizeof(write_cols) / sizeof(*write_cols));
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_load(&tables, _tmp_file_name, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_FILE_VERSION_TOO_NEW);
    ret = tsk_table_collection_free(&tables);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    version[0] = TSK_FILE_FORMAT_VERSION_MAJOR;

    /* Bad version length */
    write_cols[1].len = 0;
    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    write_table_cols(&store, write_cols, sizeof(write_cols) / sizeof(*write_cols));
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_load(&tables, _tmp_file_name, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_FILE_FORMAT);
    ret = tsk_table_collection_free(&tables);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    write_cols[1].len = 2;

    /* Bad format name length */
    write_cols[0].len = 0;
    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    write_table_cols(&store, write_cols, sizeof(write_cols) / sizeof(*write_cols));
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_load(&tables, _tmp_file_name, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_FILE_FORMAT);
    ret = tsk_table_collection_free(&tables);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    write_cols[0].len = TSK_FILE_FORMAT_NAME_LENGTH;

    /* Bad format name */
    format_name[0] = 'X';
    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    write_table_cols(&store, write_cols, sizeof(write_cols) / sizeof(*write_cols));
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_load(&tables, _tmp_file_name, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_FILE_FORMAT);
    ret = tsk_table_collection_free(&tables);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    format_name[0] = 't';

    /* Bad type for sequence length. */
    write_cols[2].type = KAS_FLOAT32;
    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    write_table_cols(&store, write_cols, sizeof(write_cols) / sizeof(*write_cols));
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_load(&tables, _tmp_file_name, 0);
    CU_ASSERT_TRUE(tsk_is_kas_error(ret));
    CU_ASSERT_EQUAL_FATAL(ret ^ (1 << TSK_KAS_ERR_BIT), KAS_ERR_TYPE_MISMATCH);
    ret = tsk_table_collection_free(&tables);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    write_cols[2].type = KAS_FLOAT64;

    /* Bad length for sequence length. */
    write_cols[2].len = 2;
    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    write_table_cols(&store, write_cols, sizeof(write_cols) / sizeof(*write_cols));
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_load(&tables, _tmp_file_name, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_FILE_FORMAT);
    ret = tsk_table_collection_free(&tables);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    write_cols[2].len = 1;

    /* Bad value for sequence length. */
    L[0] = -1;
    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    write_table_cols(&store, write_cols, sizeof(write_cols) / sizeof(*write_cols));
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_load(&tables, _tmp_file_name, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_SEQUENCE_LENGTH);
    ret = tsk_table_collection_free(&tables);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    L[0] = 1;

    /* Wrong length for uuid */
    write_cols[3].len = 1;
    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    write_table_cols(&store, write_cols, sizeof(write_cols) / sizeof(*write_cols));
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_load(&tables, _tmp_file_name, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_FILE_FORMAT);
    ret = tsk_table_collection_free(&tables);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    write_cols[3].len = (tsk_size_t) uuid_size;

    /* Missing keys */
    for (j = 0; j < sizeof(write_cols) / sizeof(*write_cols) - 1; j++) {
        ret = kastore_open(&store, _tmp_file_name, "w", 0);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        write_table_cols(&store, write_cols, j);
        ret = kastore_close(&store);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        ret = tsk_table_collection_load(&tables, _tmp_file_name, 0);
        CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_REQUIRED_COL_NOT_FOUND);
        ret = tsk_table_collection_free(&tables);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
    }
}

static void
test_missing_optional_column_pairs(void)
{
    int ret;
    size_t j;
    tsk_treeseq_t *ts = caterpillar_tree(5, 3, 3);
    tsk_table_collection_t t1, t2;
    const char *required_cols[][2] = {
        { "edges/metadata", "edges/metadata_offset" },
        { "migrations/metadata", "migrations/metadata_offset" },
    };
    const char *drop_cols[2];

    ret = tsk_treeseq_copy_tables(ts, &t1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    for (j = 0; j < sizeof(required_cols) / sizeof(*required_cols); j++) {
        drop_cols[0] = required_cols[j][0];
        copy_store_drop_columns(ts, 1, drop_cols, _tmp_file_name);
        ret = tsk_table_collection_load(&t2, _tmp_file_name, 0);
        CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BOTH_COLUMNS_REQUIRED);
        tsk_table_collection_free(&t2);

        drop_cols[0] = required_cols[j][1];
        copy_store_drop_columns(ts, 1, drop_cols, _tmp_file_name);
        ret = tsk_table_collection_load(&t2, _tmp_file_name, 0);
        CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BOTH_COLUMNS_REQUIRED);
        tsk_table_collection_free(&t2);

        drop_cols[0] = required_cols[j][0];
        drop_cols[1] = required_cols[j][1];
        copy_store_drop_columns(ts, 2, drop_cols, _tmp_file_name);
        ret = tsk_table_collection_load(&t2, _tmp_file_name, 0);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        CU_ASSERT_FALSE(tsk_table_collection_equals(&t1, &t2));
        tsk_table_collection_free(&t2);
    }

    tsk_table_collection_free(&t1);
    tsk_treeseq_free(ts);
    free(ts);
}

static void
test_missing_required_column_pairs(void)
{
    int ret;
    size_t j;
    tsk_treeseq_t *ts = caterpillar_tree(5, 3, 3);
    tsk_table_collection_t t;
    const char *required_cols[][2] = {
        { "individuals/location", "individuals/location_offset" },
        { "individuals/metadata", "individuals/metadata_offset" },
        { "mutations/derived_state", "mutations/derived_state_offset" },
        { "mutations/metadata", "mutations/metadata_offset" },
        { "nodes/metadata", "nodes/metadata_offset" },
        { "populations/metadata", "populations/metadata_offset" },
        { "provenances/record", "provenances/record_offset" },
        { "provenances/timestamp", "provenances/timestamp_offset" },
        { "sites/ancestral_state", "sites/ancestral_state_offset" },
        { "sites/metadata", "sites/metadata_offset" },
    };
    const char *drop_cols[2];

    for (j = 0; j < sizeof(required_cols) / sizeof(*required_cols); j++) {
        drop_cols[0] = required_cols[j][0];
        copy_store_drop_columns(ts, 1, drop_cols, _tmp_file_name);
        ret = tsk_table_collection_load(&t, _tmp_file_name, 0);
        CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_REQUIRED_COL_NOT_FOUND);
        tsk_table_collection_free(&t);

        drop_cols[0] = required_cols[j][1];
        copy_store_drop_columns(ts, 1, drop_cols, _tmp_file_name);
        ret = tsk_table_collection_load(&t, _tmp_file_name, 0);
        CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_REQUIRED_COL_NOT_FOUND);
        tsk_table_collection_free(&t);

        copy_store_drop_columns(ts, 2, required_cols[j], _tmp_file_name);
        ret = tsk_table_collection_load(&t, _tmp_file_name, 0);
        CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_REQUIRED_COL_NOT_FOUND);
        tsk_table_collection_free(&t);
    }

    tsk_treeseq_free(ts);
    free(ts);
}

static void
verify_bad_offset_columns(tsk_treeseq_t *ts, const char *offset_col)
{
    int ret = 0;
    kastore_t store;
    tsk_table_collection_t tables;
    tsk_size_t *offset_array, *offset_copy;
    size_t offset_len;
    uint32_t data_len;

    ret = tsk_treeseq_dump(ts, _tmp_file_name, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_open(&store, _tmp_file_name, "r", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = kastore_gets_uint32(&store, offset_col, &offset_array, &offset_len);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    offset_copy = malloc(offset_len * sizeof(*offset_array));
    CU_ASSERT_FATAL(offset_copy != NULL);
    memcpy(offset_copy, offset_array, offset_len * sizeof(*offset_array));
    data_len = offset_array[offset_len - 1];
    CU_ASSERT_TRUE(data_len > 0);
    kastore_close(&store);

    offset_copy[0] = UINT32_MAX;
    copy_store_drop_columns(ts, 1, &offset_col, _tmp_file_name);
    ret = kastore_open(&store, _tmp_file_name, "a", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts(&store, offset_col, offset_copy, offset_len, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_load(&tables, _tmp_file_name, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_OFFSET);
    tsk_table_collection_free(&tables);

    offset_copy[0] = 0;
    offset_copy[offset_len - 1] = 0;
    copy_store_drop_columns(ts, 1, &offset_col, _tmp_file_name);
    ret = kastore_open(&store, _tmp_file_name, "a", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts(&store, offset_col, offset_copy, offset_len, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_load(&tables, _tmp_file_name, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_OFFSET);
    tsk_table_collection_free(&tables);

    offset_copy[offset_len - 1] = data_len + 1;
    copy_store_drop_columns(ts, 1, &offset_col, _tmp_file_name);
    ret = kastore_open(&store, _tmp_file_name, "a", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts(&store, offset_col, offset_copy, offset_len, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_load(&tables, _tmp_file_name, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_OFFSET);
    tsk_table_collection_free(&tables);

    copy_store_drop_columns(ts, 1, &offset_col, _tmp_file_name);
    ret = kastore_open(&store, _tmp_file_name, "a", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts(&store, offset_col, NULL, 0, KAS_UINT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_load(&tables, _tmp_file_name, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_FILE_FORMAT);
    tsk_table_collection_free(&tables);

    free(offset_copy);
}

static void
test_bad_offset_columns(void)
{
    size_t j;
    tsk_treeseq_t *ts = caterpillar_tree(5, 3, 3);
    const char *cols[] = {
        "edges/metadata_offset",
        "migrations/metadata_offset",
        "individuals/location_offset",
        "individuals/metadata_offset",
        "mutations/derived_state_offset",
        "mutations/metadata_offset",
        "nodes/metadata_offset",
        "populations/metadata_offset",
        "provenances/record_offset",
        "provenances/timestamp_offset",
        "sites/ancestral_state_offset",
        "sites/metadata_offset",
    };

    for (j = 0; j < sizeof(cols) / sizeof(*cols); j++) {
        verify_bad_offset_columns(ts, cols[j]);
    }
    tsk_treeseq_free(ts);
    free(ts);
}

static void
test_missing_indexes(void)
{
    int ret;
    tsk_treeseq_t *ts = caterpillar_tree(5, 3, 3);
    tsk_table_collection_t t1, t2;
    const char *cols[]
        = { "indexes/edge_insertion_order", "indexes/edge_removal_order" };
    const char *drop_cols[2];

    ret = tsk_treeseq_copy_tables(ts, &t1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    drop_cols[0] = cols[0];
    copy_store_drop_columns(ts, 1, drop_cols, _tmp_file_name);
    ret = tsk_table_collection_load(&t2, _tmp_file_name, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BOTH_COLUMNS_REQUIRED);
    tsk_table_collection_free(&t2);

    drop_cols[0] = cols[1];
    copy_store_drop_columns(ts, 1, drop_cols, _tmp_file_name);
    ret = tsk_table_collection_load(&t2, _tmp_file_name, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BOTH_COLUMNS_REQUIRED);
    tsk_table_collection_free(&t2);

    copy_store_drop_columns(ts, 2, cols, _tmp_file_name);
    ret = tsk_table_collection_load(&t2, _tmp_file_name, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_equals(&t1, &t2));
    CU_ASSERT_FALSE(tsk_table_collection_has_index(&t2, 0));
    tsk_table_collection_free(&t2);

    tsk_table_collection_free(&t1);
    tsk_treeseq_free(ts);
    free(ts);
}

static void
test_malformed_indexes(void)
{
    int ret;
    tsk_treeseq_t *ts = caterpillar_tree(5, 3, 3);
    tsk_table_collection_t tables;
    tsk_treeseq_t ts2;
    tsk_size_t num_edges = tsk_treeseq_get_num_edges(ts);
    tsk_id_t *bad_index = calloc(num_edges, sizeof(tsk_id_t));
    tsk_id_t *good_index = calloc(num_edges, sizeof(tsk_id_t));
    kastore_t store;
    const char *cols[]
        = { "indexes/edge_insertion_order", "indexes/edge_removal_order" };

    CU_ASSERT_FATAL(bad_index != NULL);
    CU_ASSERT_FATAL(good_index != NULL);

    /* If both columns are not the same length as the number of edges we
     * should raise an error */
    copy_store_drop_columns(ts, 2, cols, _tmp_file_name);
    ret = kastore_open(&store, _tmp_file_name, "a", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts(&store, cols[0], NULL, 0, KAS_INT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts(&store, cols[1], NULL, 0, KAS_INT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_load(&tables, _tmp_file_name, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_FILE_FORMAT);
    tsk_table_collection_free(&tables);

    bad_index[0] = -1;

    copy_store_drop_columns(ts, 2, cols, _tmp_file_name);
    ret = kastore_open(&store, _tmp_file_name, "a", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts(&store, cols[0], good_index, num_edges, KAS_INT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts(&store, cols[1], bad_index, num_edges, KAS_INT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_treeseq_load(&ts2, _tmp_file_name, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_EDGE_OUT_OF_BOUNDS);
    tsk_treeseq_free(&ts2);

    copy_store_drop_columns(ts, 2, cols, _tmp_file_name);
    ret = kastore_open(&store, _tmp_file_name, "a", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts(&store, cols[0], bad_index, num_edges, KAS_INT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_puts(&store, cols[1], good_index, num_edges, KAS_INT32, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_treeseq_load(&ts2, _tmp_file_name, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_EDGE_OUT_OF_BOUNDS);
    tsk_treeseq_free(&ts2);

    free(good_index);
    free(bad_index);
    tsk_treeseq_free(ts);
    free(ts);
}

static void
test_missing_required_columns(void)
{
    int ret;
    size_t j;
    tsk_treeseq_t *ts = caterpillar_tree(5, 3, 3);
    tsk_table_collection_t t;
    const char *required_cols[] = {
        "edges/child",
        "edges/left",
        "edges/parent",
        "edges/right",
        "format/name",
        "format/version",
        "individuals/flags",
        "migrations/dest",
        "migrations/left",
        "migrations/node",
        "migrations/right",
        "migrations/source",
        "migrations/time",
        "mutations/node",
        "mutations/parent",
        "mutations/site",
        "nodes/flags",
        "nodes/individual",
        "nodes/population",
        "nodes/time",
        "sequence_length",
        "sites/position",
        "uuid",
    };
    const char *drop_cols[1];

    for (j = 0; j < sizeof(required_cols) / sizeof(*required_cols); j++) {
        drop_cols[0] = required_cols[j];
        copy_store_drop_columns(ts, 1, drop_cols, _tmp_file_name);
        ret = tsk_table_collection_load(&t, _tmp_file_name, 0);
        CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_REQUIRED_COL_NOT_FOUND);
        tsk_table_collection_free(&t);
    }

    tsk_treeseq_free(ts);
    free(ts);
}

static void
test_metadata_schemas_optional(void)
{
    int ret;
    size_t j;
    tsk_treeseq_t *ts = caterpillar_tree(5, 3, 3);
    tsk_table_collection_t t1, t2;
    const char *cols[] = {
        /* "metadata_schema", FIXME - add when table collection gets this */
        "individuals/metadata_schema",
        "populations/metadata_schema",
        "nodes/metadata_schema",
        "edges/metadata_schema",
        "sites/metadata_schema",
        "mutations/metadata_schema",
        "migrations/metadata_schema",
    };
    const char *drop_cols[1];

    ret = tsk_treeseq_copy_tables(ts, &t1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    for (j = 0; j < sizeof(cols) / sizeof(*cols); j++) {
        drop_cols[0] = cols[j];
        copy_store_drop_columns(ts, 1, drop_cols, _tmp_file_name);
        ret = tsk_table_collection_load(&t2, _tmp_file_name, 0);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        /* metadata schemas are included in data comparisons */
        CU_ASSERT_FALSE(tsk_table_collection_equals(&t1, &t2));
        tsk_table_collection_free(&t2);
    }

    tsk_table_collection_free(&t1);
    tsk_treeseq_free(ts);
    free(ts);
}

static void
test_table_collection_load_errors(void)
{
    tsk_table_collection_t tables;
    int ret;
    const char *str;

    ret = tsk_table_collection_load(&tables, "/", 0);
    CU_ASSERT_TRUE(tsk_is_kas_error(ret));
    CU_ASSERT_EQUAL_FATAL(ret ^ (1 << TSK_KAS_ERR_BIT), KAS_ERR_IO);
    str = tsk_strerror(ret);
    CU_ASSERT_TRUE(strlen(str) > 0);

    tsk_table_collection_free(&tables);
}

static void
test_table_collection_dump_errors(void)
{
    tsk_table_collection_t tables;
    int ret;
    const char *str;

    ret = tsk_table_collection_init(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tables.sequence_length = 1.0;

    ret = tsk_table_collection_dump(&tables, "/", 0);
    CU_ASSERT_TRUE(tsk_is_kas_error(ret));
    CU_ASSERT_EQUAL_FATAL(ret ^ (1 << TSK_KAS_ERR_BIT), KAS_ERR_IO);
    str = tsk_strerror(ret);
    CU_ASSERT_TRUE(strlen(str) > 0);

    /* We'd like to catch close errors also, but it's hard to provoke them
     * without intercepting calls to fclose() */

    tsk_table_collection_free(&tables);
}

/* FIXME these are good tests, but we want to make them more general so that
 * they can be applied to other tables.*/
static void
test_load_node_table_errors(void)
{
    char format_name[TSK_FILE_FORMAT_NAME_LENGTH];
    tsk_size_t uuid_size = 36;
    char uuid[uuid_size];
    double L = 1;
    double time = 0;
    double flags = 0;
    int32_t population = 0;
    int32_t individual = 0;
    int8_t metadata = 0;
    uint32_t metadata_offset[] = { 0, 1 };
    uint32_t version[2]
        = { TSK_FILE_FORMAT_VERSION_MAJOR, TSK_FILE_FORMAT_VERSION_MINOR };
    write_table_col_t write_cols[] = {
        { "nodes/time", (void *) &time, 1, KAS_FLOAT64 },
        { "nodes/flags", (void *) &flags, 1, KAS_UINT32 },
        { "nodes/population", (void *) &population, 1, KAS_INT32 },
        { "nodes/individual", (void *) &individual, 1, KAS_INT32 },
        { "nodes/metadata", (void *) &metadata, 1, KAS_UINT8 },
        { "nodes/metadata_offset", (void *) metadata_offset, 2, KAS_UINT32 },
        { "format/name", (void *) format_name, sizeof(format_name), KAS_INT8 },
        { "format/version", (void *) version, 2, KAS_UINT32 },
        { "uuid", (void *) uuid, uuid_size, KAS_INT8 },
        { "sequence_length", (void *) &L, 1, KAS_FLOAT64 },
    };
    tsk_table_collection_t tables;
    kastore_t store;
    int ret;

    memcpy(format_name, TSK_FILE_FORMAT_NAME, sizeof(format_name));
    /* Note: this will fail if we ever start parsing the form of the UUID */
    memset(uuid, 0, uuid_size);

    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    write_table_cols(&store, write_cols, sizeof(write_cols) / sizeof(*write_cols));
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_load(&tables, _tmp_file_name, 0);
    /* We've only defined the format headers and nodes, so we should fail immediately
     * after with key not found */
    CU_ASSERT_FALSE(tsk_is_kas_error(ret));
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_REQUIRED_COL_NOT_FOUND);
    ret = tsk_table_collection_free(&tables);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* Wrong type for time */
    write_cols[0].type = KAS_INT64;
    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    write_table_cols(&store, write_cols, sizeof(write_cols) / sizeof(*write_cols));
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_load(&tables, _tmp_file_name, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_FILE_FORMAT);
    ret = tsk_table_collection_free(&tables);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    write_cols[0].type = KAS_FLOAT64;

    /* Wrong length for flags */
    write_cols[1].len = 0;
    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    write_table_cols(&store, write_cols, sizeof(write_cols) / sizeof(*write_cols));
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_load(&tables, _tmp_file_name, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_FILE_FORMAT);
    ret = tsk_table_collection_free(&tables);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    write_cols[1].len = 1;

    /* Wrong length for metadata offset */
    write_cols[5].len = 1;
    ret = kastore_open(&store, _tmp_file_name, "w", 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    write_table_cols(&store, write_cols, sizeof(write_cols) / sizeof(*write_cols));
    ret = kastore_close(&store);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_load(&tables, _tmp_file_name, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_FILE_FORMAT);
    ret = tsk_table_collection_free(&tables);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    write_cols[5].len = 2;
}

static void
test_example_round_trip(void)
{
    int ret;
    tsk_treeseq_t *ts = caterpillar_tree(5, 3, 3);
    tsk_table_collection_t t1, t2;

    ret = tsk_treeseq_copy_tables(ts, &t1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_dump(&t1, _tmp_file_name, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_load(&t2, _tmp_file_name, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_equals(&t1, &t2));

    tsk_table_collection_free(&t1);
    tsk_table_collection_free(&t2);
    tsk_treeseq_free(ts);
    free(ts);
}

static void
test_copy_store_drop_columns(void)
{
    int ret;
    tsk_treeseq_t *ts = caterpillar_tree(5, 3, 3);
    tsk_table_collection_t t1, t2;

    ret = tsk_treeseq_copy_tables(ts, &t1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    /* Dropping no columns should have no effect on the data */
    copy_store_drop_columns(ts, 0, NULL, _tmp_file_name);
    ret = tsk_table_collection_load(&t2, _tmp_file_name, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(tsk_table_collection_equals(&t1, &t2));

    tsk_table_collection_free(&t1);
    tsk_table_collection_free(&t2);
    tsk_treeseq_free(ts);
    free(ts);
}

int
main(int argc, char **argv)
{
    CU_TestInfo tests[] = {
        { "test_format_data_load_errors", test_format_data_load_errors },
        { "test_missing_indexes", test_missing_indexes },
        { "test_malformed_indexes", test_malformed_indexes },
        { "test_missing_required_columns", test_missing_required_columns },
        { "test_missing_optional_column_pairs", test_missing_optional_column_pairs },
        { "test_missing_required_column_pairs", test_missing_required_column_pairs },
        { "test_bad_offset_columns", test_bad_offset_columns },
        { "test_metadata_schemas_optional", test_metadata_schemas_optional },
        { "test_load_node_table_errors", test_load_node_table_errors },
        { "test_table_collection_load_errors", test_table_collection_load_errors },
        { "test_table_collection_dump_errors", test_table_collection_dump_errors },
        { "test_example_round_trip", test_example_round_trip },
        { "test_copy_store_drop_columns", test_copy_store_drop_columns },
        { NULL, NULL },
    };

    return test_main(tests, argc, argv);
}
