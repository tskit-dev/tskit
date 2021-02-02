/*
 * MIT License
 *
 * Copyright (c) 2019-2020 Tskit Developers
 * Copyright (c) 2015-2018 University of Oxford
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

/* This file defines the LightweightTableCollection class using the
 * Python-C interface. It is intended to be #include-d and compiled
 * into third-party Python modules that use the tskit C interface.
 * See https://github.com/tskit-dev/tskit/tree/main/python/lwt_interface
 * for details and usage examples.
 */

typedef struct {
    // clang-format off
    PyObject_HEAD
    tsk_table_collection_t *tables;
    // clang-format on
} LightweightTableCollection;

static void
handle_tskit_error(int err)
{
    PyErr_SetString(PyExc_ValueError, tsk_strerror(err));
}

static PyObject *
make_Py_Unicode_FromStringAndLength(const char *str, size_t length)
{
    PyObject *ret = NULL;

    /* Py_BuildValue returns Py_None for zero length, we would rather
       return a zero-length string */
    if (length == 0) {
        ret = PyUnicode_FromString("");
    } else {
        ret = Py_BuildValue("s#", str, length);
    }
    return ret;
}

/*
 * Retrieves the PyObject* corresponding the specified key in the
 * specified dictionary. If required is true, raise a TypeError if the
 * value is None or absent.
 *
 * NB This returns a *borrowed reference*, so don't DECREF it!
 */
static PyObject *
get_table_dict_value(PyObject *dict, const char *key_str, bool required)
{
    PyObject *ret = NULL;

    ret = PyDict_GetItemString(dict, key_str);
    if (ret == NULL) {
        ret = Py_None;
    }
    if (required && ret == Py_None) {
        PyErr_Format(PyExc_TypeError, "'%s' is required", key_str);
        ret = NULL;
    }
    return ret;
}

static PyArrayObject *
table_read_column_array(
    PyObject *input, int npy_type, size_t *num_rows, bool check_num_rows)
{
    PyArrayObject *ret = NULL;
    PyArrayObject *array = NULL;
    npy_intp *shape;

    array = (PyArrayObject *) PyArray_FROMANY(input, npy_type, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (array == NULL) {
        goto out;
    }
    shape = PyArray_DIMS(array);
    if (check_num_rows) {
        if (*num_rows != (size_t) shape[0]) {
            PyErr_SetString(PyExc_ValueError, "Input array dimensions must be equal.");
            goto out;
        }
    } else {
        *num_rows = (size_t) shape[0];
    }
    ret = array;
    array = NULL;
out:
    Py_XDECREF(array);
    return ret;
}

static PyArrayObject *
table_read_offset_array(
    PyObject *input, size_t *num_rows, size_t length, bool check_num_rows)
{
    PyArrayObject *ret = NULL;
    PyArrayObject *array = NULL;
    npy_intp *shape;
    uint32_t *data;

    array
        = (PyArrayObject *) PyArray_FROMANY(input, NPY_UINT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (array == NULL) {
        goto out;
    }
    shape = PyArray_DIMS(array);
    if (!check_num_rows) {
        *num_rows = shape[0];
        if (*num_rows == 0) {
            PyErr_SetString(
                PyExc_ValueError, "Offset arrays must have at least one element");
            goto out;
        }
        *num_rows -= 1;
    }
    if (shape[0] != (npy_intp)(*num_rows + 1)) {
        PyErr_SetString(PyExc_ValueError, "offset columns must have n + 1 rows.");
        goto out;
    }
    data = PyArray_DATA(array);
    if (data[*num_rows] != (uint32_t) length) {
        PyErr_SetString(PyExc_ValueError, "Bad offset column encoding");
        goto out;
    }
    ret = array;
out:
    if (ret == NULL) {
        Py_XDECREF(array);
    }
    return ret;
}

static const char *
parse_metadata_schema_arg(PyObject *arg, Py_ssize_t *metadata_schema_length)
{
    const char *ret = NULL;
    if (arg == NULL) {
        PyErr_Format(PyExc_AttributeError,
            "Cannot del metadata_schema, set to empty string (\"\") to clear.");
        goto out;
    }
    ret = PyUnicode_AsUTF8AndSize(arg, metadata_schema_length);
    if (ret == NULL) {
        goto out;
    }
out:
    return ret;
}

static int
parse_individual_table_dict(
    tsk_individual_table_t *table, PyObject *dict, bool clear_table)
{
    int err;
    int ret = -1;
    size_t num_rows, metadata_length, location_length, parents_length;
    char *metadata_data = NULL;
    double *location_data = NULL;
    tsk_id_t *parents_data = NULL;
    uint32_t *metadata_offset_data = NULL;
    uint32_t *location_offset_data = NULL;
    uint32_t *parents_offset_data = NULL;
    PyObject *flags_input = NULL;
    PyArrayObject *flags_array = NULL;
    PyObject *location_input = NULL;
    PyArrayObject *location_array = NULL;
    PyObject *location_offset_input = NULL;
    PyArrayObject *location_offset_array = NULL;
    PyObject *parents_input = NULL;
    PyArrayObject *parents_array = NULL;
    PyObject *parents_offset_input = NULL;
    PyArrayObject *parents_offset_array = NULL;
    PyObject *metadata_input = NULL;
    PyArrayObject *metadata_array = NULL;
    PyObject *metadata_offset_input = NULL;
    PyArrayObject *metadata_offset_array = NULL;
    PyObject *metadata_schema_input = NULL;
    const char *metadata_schema = NULL;
    Py_ssize_t metadata_schema_length = 0;

    /* Get the input values */
    flags_input = get_table_dict_value(dict, "flags", true);
    if (flags_input == NULL) {
        goto out;
    }
    location_input = get_table_dict_value(dict, "location", false);
    if (location_input == NULL) {
        goto out;
    }
    location_offset_input = get_table_dict_value(dict, "location_offset", false);
    if (location_offset_input == NULL) {
        goto out;
    }
    parents_input = get_table_dict_value(dict, "parents", false);
    if (parents_input == NULL) {
        goto out;
    }
    parents_offset_input = get_table_dict_value(dict, "parents_offset", false);
    if (parents_offset_input == NULL) {
        goto out;
    }
    metadata_input = get_table_dict_value(dict, "metadata", false);
    if (metadata_input == NULL) {
        goto out;
    }
    metadata_offset_input = get_table_dict_value(dict, "metadata_offset", false);
    if (metadata_offset_input == NULL) {
        goto out;
    }
    metadata_schema_input = get_table_dict_value(dict, "metadata_schema", false);
    if (metadata_schema_input == NULL) {
        goto out;
    }

    /* Pull out the arrays */
    flags_array = table_read_column_array(flags_input, NPY_UINT32, &num_rows, false);
    if (flags_array == NULL) {
        goto out;
    }
    if ((location_input == Py_None) != (location_offset_input == Py_None)) {
        PyErr_SetString(
            PyExc_TypeError, "location and location_offset must be specified together");
        goto out;
    }
    if (location_input != Py_None) {
        location_array = table_read_column_array(
            location_input, NPY_FLOAT64, &location_length, false);
        if (location_array == NULL) {
            goto out;
        }
        location_data = PyArray_DATA(location_array);
        location_offset_array = table_read_offset_array(
            location_offset_input, &num_rows, location_length, true);
        if (location_offset_array == NULL) {
            goto out;
        }
        location_offset_data = PyArray_DATA(location_offset_array);
    }
    if ((parents_input == Py_None) != (parents_offset_input == Py_None)) {
        PyErr_SetString(
            PyExc_TypeError, "parents and parents_offset must be specified together");
        goto out;
    }
    if (parents_input != Py_None) {
        parents_array
            = table_read_column_array(parents_input, NPY_INT32, &parents_length, false);
        if (parents_array == NULL) {
            goto out;
        }
        parents_data = PyArray_DATA(parents_array);
        parents_offset_array = table_read_offset_array(
            parents_offset_input, &num_rows, parents_length, true);
        if (parents_offset_array == NULL) {
            goto out;
        }
        parents_offset_data = PyArray_DATA(parents_offset_array);
    }
    if ((metadata_input == Py_None) != (metadata_offset_input == Py_None)) {
        PyErr_SetString(
            PyExc_TypeError, "metadata and metadata_offset must be specified together");
        goto out;
    }
    if (metadata_input != Py_None) {
        metadata_array
            = table_read_column_array(metadata_input, NPY_INT8, &metadata_length, false);
        if (metadata_array == NULL) {
            goto out;
        }
        metadata_data = PyArray_DATA(metadata_array);
        metadata_offset_array = table_read_offset_array(
            metadata_offset_input, &num_rows, metadata_length, true);
        if (metadata_offset_array == NULL) {
            goto out;
        }
        metadata_offset_data = PyArray_DATA(metadata_offset_array);
    }

    if (metadata_schema_input != Py_None) {
        metadata_schema
            = parse_metadata_schema_arg(metadata_schema_input, &metadata_schema_length);
        if (metadata_schema == NULL) {
            goto out;
        }
        err = tsk_individual_table_set_metadata_schema(
            table, metadata_schema, metadata_schema_length);
        if (err != 0) {
            handle_tskit_error(err);
            goto out;
        }
    }

    if (clear_table) {
        err = tsk_individual_table_clear(table);
        if (err != 0) {
            handle_tskit_error(err);
            goto out;
        }
    }
    err = tsk_individual_table_append_columns(table, num_rows, PyArray_DATA(flags_array),
        location_data, location_offset_data, parents_data, parents_offset_data,
        metadata_data, metadata_offset_data);
    if (err != 0) {
        handle_tskit_error(err);
        goto out;
    }
    ret = 0;
out:
    Py_XDECREF(flags_array);
    Py_XDECREF(location_array);
    Py_XDECREF(location_offset_array);
    Py_XDECREF(parents_array);
    Py_XDECREF(parents_offset_array);
    Py_XDECREF(metadata_array);
    Py_XDECREF(metadata_offset_array);
    return ret;
}

static int
parse_node_table_dict(tsk_node_table_t *table, PyObject *dict, bool clear_table)
{
    int err;
    int ret = -1;
    size_t num_rows, metadata_length;
    char *metadata_data = NULL;
    uint32_t *metadata_offset_data = NULL;
    void *population_data = NULL;
    void *individual_data = NULL;
    PyObject *time_input = NULL;
    PyArrayObject *time_array = NULL;
    PyObject *flags_input = NULL;
    PyArrayObject *flags_array = NULL;
    PyObject *population_input = NULL;
    PyArrayObject *population_array = NULL;
    PyObject *individual_input = NULL;
    PyArrayObject *individual_array = NULL;
    PyObject *metadata_input = NULL;
    PyArrayObject *metadata_array = NULL;
    PyObject *metadata_offset_input = NULL;
    PyArrayObject *metadata_offset_array = NULL;
    PyObject *metadata_schema_input = NULL;
    const char *metadata_schema = NULL;
    Py_ssize_t metadata_schema_length = 0;

    /* Get the input values */
    flags_input = get_table_dict_value(dict, "flags", true);
    if (flags_input == NULL) {
        goto out;
    }
    time_input = get_table_dict_value(dict, "time", true);
    if (time_input == NULL) {
        goto out;
    }
    population_input = get_table_dict_value(dict, "population", false);
    if (population_input == NULL) {
        goto out;
    }
    individual_input = get_table_dict_value(dict, "individual", false);
    if (individual_input == NULL) {
        goto out;
    }
    metadata_input = get_table_dict_value(dict, "metadata", false);
    if (metadata_input == NULL) {
        goto out;
    }
    metadata_offset_input = get_table_dict_value(dict, "metadata_offset", false);
    if (metadata_offset_input == NULL) {
        goto out;
    }
    metadata_schema_input = get_table_dict_value(dict, "metadata_schema", false);
    if (metadata_schema_input == NULL) {
        goto out;
    }

    /* Create the arrays */
    flags_array = table_read_column_array(flags_input, NPY_UINT32, &num_rows, false);
    if (flags_array == NULL) {
        goto out;
    }
    time_array = table_read_column_array(time_input, NPY_FLOAT64, &num_rows, true);
    if (time_array == NULL) {
        goto out;
    }
    if (population_input != Py_None) {
        population_array
            = table_read_column_array(population_input, NPY_INT32, &num_rows, true);
        if (population_array == NULL) {
            goto out;
        }
        population_data = PyArray_DATA(population_array);
    }
    if (individual_input != Py_None) {
        individual_array
            = table_read_column_array(individual_input, NPY_INT32, &num_rows, true);
        if (individual_array == NULL) {
            goto out;
        }
        individual_data = PyArray_DATA(individual_array);
    }
    if ((metadata_input == Py_None) != (metadata_offset_input == Py_None)) {
        PyErr_SetString(
            PyExc_TypeError, "metadata and metadata_offset must be specified together");
        goto out;
    }
    if (metadata_input != Py_None) {
        metadata_array
            = table_read_column_array(metadata_input, NPY_INT8, &metadata_length, false);
        if (metadata_array == NULL) {
            goto out;
        }
        metadata_data = PyArray_DATA(metadata_array);
        metadata_offset_array = table_read_offset_array(
            metadata_offset_input, &num_rows, metadata_length, true);
        if (metadata_offset_array == NULL) {
            goto out;
        }
        metadata_offset_data = PyArray_DATA(metadata_offset_array);
    }
    if (metadata_schema_input != Py_None) {
        metadata_schema
            = parse_metadata_schema_arg(metadata_schema_input, &metadata_schema_length);
        if (metadata_schema == NULL) {
            goto out;
        }
        err = tsk_node_table_set_metadata_schema(
            table, metadata_schema, metadata_schema_length);
        if (err != 0) {
            handle_tskit_error(err);
            goto out;
        }
    }

    if (clear_table) {
        err = tsk_node_table_clear(table);
        if (err != 0) {
            handle_tskit_error(err);
            goto out;
        }
    }
    err = tsk_node_table_append_columns(table, num_rows, PyArray_DATA(flags_array),
        PyArray_DATA(time_array), population_data, individual_data, metadata_data,
        metadata_offset_data);
    if (err != 0) {
        handle_tskit_error(err);
        goto out;
    }
    ret = 0;
out:
    Py_XDECREF(flags_array);
    Py_XDECREF(time_array);
    Py_XDECREF(population_array);
    Py_XDECREF(individual_array);
    Py_XDECREF(metadata_array);
    Py_XDECREF(metadata_offset_array);
    return ret;
}

static int
parse_edge_table_dict(tsk_edge_table_t *table, PyObject *dict, bool clear_table)
{
    int ret = -1;
    int err;
    size_t num_rows = 0;
    size_t metadata_length;
    char *metadata_data = NULL;
    uint32_t *metadata_offset_data = NULL;
    PyObject *left_input = NULL;
    PyArrayObject *left_array = NULL;
    PyObject *right_input = NULL;
    PyArrayObject *right_array = NULL;
    PyObject *parent_input = NULL;
    PyArrayObject *parent_array = NULL;
    PyObject *child_input = NULL;
    PyArrayObject *child_array = NULL;
    PyObject *metadata_input = NULL;
    PyArrayObject *metadata_array = NULL;
    PyObject *metadata_offset_input = NULL;
    PyArrayObject *metadata_offset_array = NULL;
    PyObject *metadata_schema_input = NULL;
    const char *metadata_schema = NULL;
    Py_ssize_t metadata_schema_length = 0;

    /* Get the input values */
    left_input = get_table_dict_value(dict, "left", true);
    if (left_input == NULL) {
        goto out;
    }
    right_input = get_table_dict_value(dict, "right", true);
    if (right_input == NULL) {
        goto out;
    }
    parent_input = get_table_dict_value(dict, "parent", true);
    if (parent_input == NULL) {
        goto out;
    }
    child_input = get_table_dict_value(dict, "child", true);
    if (child_input == NULL) {
        goto out;
    }
    metadata_input = get_table_dict_value(dict, "metadata", false);
    if (metadata_input == NULL) {
        goto out;
    }
    metadata_offset_input = get_table_dict_value(dict, "metadata_offset", false);
    if (metadata_offset_input == NULL) {
        goto out;
    }
    metadata_schema_input = get_table_dict_value(dict, "metadata_schema", false);
    if (metadata_schema_input == NULL) {
        goto out;
    }

    /* Create the arrays */
    left_array = table_read_column_array(left_input, NPY_FLOAT64, &num_rows, false);
    if (left_array == NULL) {
        goto out;
    }
    right_array = table_read_column_array(right_input, NPY_FLOAT64, &num_rows, true);
    if (right_array == NULL) {
        goto out;
    }
    parent_array = table_read_column_array(parent_input, NPY_INT32, &num_rows, true);
    if (parent_array == NULL) {
        goto out;
    }
    child_array = table_read_column_array(child_input, NPY_INT32, &num_rows, true);
    if (child_array == NULL) {
        goto out;
    }
    if ((metadata_input == Py_None) != (metadata_offset_input == Py_None)) {
        PyErr_SetString(
            PyExc_TypeError, "metadata and metadata_offset must be specified together");
        goto out;
    }
    if (metadata_input != Py_None) {
        metadata_array
            = table_read_column_array(metadata_input, NPY_INT8, &metadata_length, false);
        if (metadata_array == NULL) {
            goto out;
        }
        metadata_data = PyArray_DATA(metadata_array);
        metadata_offset_array = table_read_offset_array(
            metadata_offset_input, &num_rows, metadata_length, true);
        if (metadata_offset_array == NULL) {
            goto out;
        }
        metadata_offset_data = PyArray_DATA(metadata_offset_array);
    }
    if (metadata_schema_input != Py_None) {
        metadata_schema
            = parse_metadata_schema_arg(metadata_schema_input, &metadata_schema_length);
        if (metadata_schema == NULL) {
            goto out;
        }
        err = tsk_edge_table_set_metadata_schema(
            table, metadata_schema, metadata_schema_length);
        if (err != 0) {
            handle_tskit_error(err);
            goto out;
        }
    }

    if (clear_table) {
        err = tsk_edge_table_clear(table);
        if (err != 0) {
            handle_tskit_error(err);
            goto out;
        }
    }
    err = tsk_edge_table_append_columns(table, num_rows, PyArray_DATA(left_array),
        PyArray_DATA(right_array), PyArray_DATA(parent_array), PyArray_DATA(child_array),
        metadata_data, metadata_offset_data);
    if (err != 0) {
        handle_tskit_error(err);
        goto out;
    }
    ret = 0;
out:
    Py_XDECREF(left_array);
    Py_XDECREF(right_array);
    Py_XDECREF(parent_array);
    Py_XDECREF(child_array);
    Py_XDECREF(metadata_array);
    Py_XDECREF(metadata_offset_array);
    return ret;
}

static int
parse_migration_table_dict(
    tsk_migration_table_t *table, PyObject *dict, bool clear_table)
{
    int err;
    int ret = -1;
    size_t num_rows;
    size_t metadata_length;
    char *metadata_data = NULL;
    uint32_t *metadata_offset_data = NULL;
    PyObject *left_input = NULL;
    PyArrayObject *left_array = NULL;
    PyObject *right_input = NULL;
    PyArrayObject *right_array = NULL;
    PyObject *node_input = NULL;
    PyArrayObject *node_array = NULL;
    PyObject *source_input = NULL;
    PyArrayObject *source_array = NULL;
    PyObject *dest_input = NULL;
    PyArrayObject *dest_array = NULL;
    PyObject *time_input = NULL;
    PyArrayObject *time_array = NULL;
    PyObject *metadata_input = NULL;
    PyArrayObject *metadata_array = NULL;
    PyObject *metadata_offset_input = NULL;
    PyArrayObject *metadata_offset_array = NULL;
    PyObject *metadata_schema_input = NULL;
    const char *metadata_schema = NULL;
    Py_ssize_t metadata_schema_length = 0;

    /* Get the input values */
    left_input = get_table_dict_value(dict, "left", true);
    if (left_input == NULL) {
        goto out;
    }
    right_input = get_table_dict_value(dict, "right", true);
    if (right_input == NULL) {
        goto out;
    }
    node_input = get_table_dict_value(dict, "node", true);
    if (node_input == NULL) {
        goto out;
    }
    source_input = get_table_dict_value(dict, "source", true);
    if (source_input == NULL) {
        goto out;
    }
    dest_input = get_table_dict_value(dict, "dest", true);
    if (dest_input == NULL) {
        goto out;
    }
    time_input = get_table_dict_value(dict, "time", true);
    if (time_input == NULL) {
        goto out;
    }
    metadata_input = get_table_dict_value(dict, "metadata", false);
    if (metadata_input == NULL) {
        goto out;
    }
    metadata_offset_input = get_table_dict_value(dict, "metadata_offset", false);
    if (metadata_offset_input == NULL) {
        goto out;
    }
    metadata_schema_input = get_table_dict_value(dict, "metadata_schema", false);
    if (metadata_schema_input == NULL) {
        goto out;
    }

    /* Build the arrays */
    left_array = table_read_column_array(left_input, NPY_FLOAT64, &num_rows, false);
    if (left_array == NULL) {
        goto out;
    }
    right_array = table_read_column_array(right_input, NPY_FLOAT64, &num_rows, true);
    if (right_array == NULL) {
        goto out;
    }
    node_array = table_read_column_array(node_input, NPY_INT32, &num_rows, true);
    if (node_array == NULL) {
        goto out;
    }
    source_array = table_read_column_array(source_input, NPY_INT32, &num_rows, true);
    if (source_array == NULL) {
        goto out;
    }
    dest_array = table_read_column_array(dest_input, NPY_INT32, &num_rows, true);
    if (dest_array == NULL) {
        goto out;
    }
    time_array = table_read_column_array(time_input, NPY_FLOAT64, &num_rows, true);
    if (time_array == NULL) {
        goto out;
    }
    if ((metadata_input == Py_None) != (metadata_offset_input == Py_None)) {
        PyErr_SetString(
            PyExc_TypeError, "metadata and metadata_offset must be specified together");
        goto out;
    }
    if (metadata_input != Py_None) {
        metadata_array
            = table_read_column_array(metadata_input, NPY_INT8, &metadata_length, false);
        if (metadata_array == NULL) {
            goto out;
        }
        metadata_data = PyArray_DATA(metadata_array);
        metadata_offset_array = table_read_offset_array(
            metadata_offset_input, &num_rows, metadata_length, true);
        if (metadata_offset_array == NULL) {
            goto out;
        }
        metadata_offset_data = PyArray_DATA(metadata_offset_array);
    }
    if (metadata_schema_input != Py_None) {
        metadata_schema
            = parse_metadata_schema_arg(metadata_schema_input, &metadata_schema_length);
        if (metadata_schema == NULL) {
            goto out;
        }
        err = tsk_migration_table_set_metadata_schema(
            table, metadata_schema, metadata_schema_length);
        if (err != 0) {
            handle_tskit_error(err);
            goto out;
        }
    }

    if (clear_table) {
        err = tsk_migration_table_clear(table);
        if (err != 0) {
            handle_tskit_error(err);
            goto out;
        }
    }
    err = tsk_migration_table_append_columns(table, num_rows, PyArray_DATA(left_array),
        PyArray_DATA(right_array), PyArray_DATA(node_array), PyArray_DATA(source_array),
        PyArray_DATA(dest_array), PyArray_DATA(time_array), metadata_data,
        metadata_offset_data);
    if (err != 0) {
        handle_tskit_error(err);
        goto out;
    }
    ret = 0;
out:
    Py_XDECREF(left_array);
    Py_XDECREF(right_array);
    Py_XDECREF(node_array);
    Py_XDECREF(source_array);
    Py_XDECREF(dest_array);
    Py_XDECREF(time_array);
    Py_XDECREF(metadata_array);
    Py_XDECREF(metadata_offset_array);
    return ret;
}

static int
parse_site_table_dict(tsk_site_table_t *table, PyObject *dict, bool clear_table)
{
    int err;
    int ret = -1;
    size_t num_rows = 0;
    size_t ancestral_state_length, metadata_length;
    PyObject *position_input = NULL;
    PyArrayObject *position_array = NULL;
    PyObject *ancestral_state_input = NULL;
    PyArrayObject *ancestral_state_array = NULL;
    PyObject *ancestral_state_offset_input = NULL;
    PyArrayObject *ancestral_state_offset_array = NULL;
    PyObject *metadata_input = NULL;
    PyArrayObject *metadata_array = NULL;
    PyObject *metadata_offset_input = NULL;
    PyArrayObject *metadata_offset_array = NULL;
    char *metadata_data;
    uint32_t *metadata_offset_data;
    PyObject *metadata_schema_input = NULL;
    const char *metadata_schema = NULL;
    Py_ssize_t metadata_schema_length = 0;

    /* Get the input values */
    position_input = get_table_dict_value(dict, "position", true);
    if (position_input == NULL) {
        goto out;
    }
    ancestral_state_input = get_table_dict_value(dict, "ancestral_state", true);
    if (ancestral_state_input == NULL) {
        goto out;
    }
    ancestral_state_offset_input
        = get_table_dict_value(dict, "ancestral_state_offset", true);
    if (ancestral_state_offset_input == NULL) {
        goto out;
    }
    metadata_input = get_table_dict_value(dict, "metadata", false);
    if (metadata_input == NULL) {
        goto out;
    }
    metadata_offset_input = get_table_dict_value(dict, "metadata_offset", false);
    if (metadata_offset_input == NULL) {
        goto out;
    }
    metadata_schema_input = get_table_dict_value(dict, "metadata_schema", false);
    if (metadata_schema_input == NULL) {
        goto out;
    }

    /* Get the arrays */
    position_array
        = table_read_column_array(position_input, NPY_FLOAT64, &num_rows, false);
    if (position_array == NULL) {
        goto out;
    }
    ancestral_state_array = table_read_column_array(
        ancestral_state_input, NPY_INT8, &ancestral_state_length, false);
    if (ancestral_state_array == NULL) {
        goto out;
    }
    ancestral_state_offset_array = table_read_offset_array(
        ancestral_state_offset_input, &num_rows, ancestral_state_length, true);
    if (ancestral_state_offset_array == NULL) {
        goto out;
    }

    metadata_data = NULL;
    metadata_offset_data = NULL;
    if ((metadata_input == Py_None) != (metadata_offset_input == Py_None)) {
        PyErr_SetString(
            PyExc_TypeError, "metadata and metadata_offset must be specified together");
        goto out;
    }
    if (metadata_input != Py_None) {
        metadata_array
            = table_read_column_array(metadata_input, NPY_INT8, &metadata_length, false);
        if (metadata_array == NULL) {
            goto out;
        }
        metadata_data = PyArray_DATA(metadata_array);
        metadata_offset_array = table_read_offset_array(
            metadata_offset_input, &num_rows, metadata_length, false);
        if (metadata_offset_array == NULL) {
            goto out;
        }
        metadata_offset_data = PyArray_DATA(metadata_offset_array);
    }
    if (metadata_schema_input != Py_None) {
        metadata_schema
            = parse_metadata_schema_arg(metadata_schema_input, &metadata_schema_length);
        if (metadata_schema == NULL) {
            goto out;
        }
        err = tsk_site_table_set_metadata_schema(
            table, metadata_schema, metadata_schema_length);
        if (err != 0) {
            handle_tskit_error(err);
            goto out;
        }
    }

    if (clear_table) {
        err = tsk_site_table_clear(table);
        if (err != 0) {
            handle_tskit_error(err);
            goto out;
        }
    }
    err = tsk_site_table_append_columns(table, num_rows, PyArray_DATA(position_array),
        PyArray_DATA(ancestral_state_array), PyArray_DATA(ancestral_state_offset_array),
        metadata_data, metadata_offset_data);
    if (err != 0) {
        handle_tskit_error(err);
        goto out;
    }
    ret = 0;
out:
    Py_XDECREF(position_array);
    Py_XDECREF(ancestral_state_array);
    Py_XDECREF(ancestral_state_offset_array);
    Py_XDECREF(metadata_array);
    Py_XDECREF(metadata_offset_array);
    return ret;
}

static int
parse_mutation_table_dict(tsk_mutation_table_t *table, PyObject *dict, bool clear_table)
{
    int err;
    int ret = -1;
    size_t num_rows = 0;
    size_t derived_state_length = 0;
    size_t metadata_length = 0;
    PyObject *site_input = NULL;
    PyArrayObject *site_array = NULL;
    PyObject *derived_state_input = NULL;
    PyArrayObject *derived_state_array = NULL;
    PyObject *derived_state_offset_input = NULL;
    PyArrayObject *derived_state_offset_array = NULL;
    PyObject *node_input = NULL;
    PyArrayObject *node_array = NULL;
    PyObject *time_input = NULL;
    PyArrayObject *time_array = NULL;
    double *time_data;
    PyObject *parent_input = NULL;
    PyArrayObject *parent_array = NULL;
    tsk_id_t *parent_data;
    PyObject *metadata_input = NULL;
    PyArrayObject *metadata_array = NULL;
    PyObject *metadata_offset_input = NULL;
    PyArrayObject *metadata_offset_array = NULL;
    char *metadata_data;
    uint32_t *metadata_offset_data;
    PyObject *metadata_schema_input = NULL;
    const char *metadata_schema = NULL;
    Py_ssize_t metadata_schema_length = 0;

    /* Get the input values */
    site_input = get_table_dict_value(dict, "site", true);
    if (site_input == NULL) {
        goto out;
    }
    node_input = get_table_dict_value(dict, "node", true);
    if (node_input == NULL) {
        goto out;
    }
    parent_input = get_table_dict_value(dict, "parent", false);
    if (parent_input == NULL) {
        goto out;
    }
    time_input = get_table_dict_value(dict, "time", false);
    if (time_input == NULL) {
        goto out;
    }
    derived_state_input = get_table_dict_value(dict, "derived_state", true);
    if (derived_state_input == NULL) {
        goto out;
    }
    derived_state_offset_input
        = get_table_dict_value(dict, "derived_state_offset", true);
    if (derived_state_offset_input == NULL) {
        goto out;
    }
    metadata_input = get_table_dict_value(dict, "metadata", false);
    if (metadata_input == NULL) {
        goto out;
    }
    metadata_offset_input = get_table_dict_value(dict, "metadata_offset", false);
    if (metadata_offset_input == NULL) {
        goto out;
    }
    metadata_schema_input = get_table_dict_value(dict, "metadata_schema", false);
    if (metadata_schema_input == NULL) {
        goto out;
    }

    /* Get the arrays */
    site_array = table_read_column_array(site_input, NPY_INT32, &num_rows, false);
    if (site_array == NULL) {
        goto out;
    }
    derived_state_array = table_read_column_array(
        derived_state_input, NPY_INT8, &derived_state_length, false);
    if (derived_state_array == NULL) {
        goto out;
    }
    derived_state_offset_array = table_read_offset_array(
        derived_state_offset_input, &num_rows, derived_state_length, true);
    if (derived_state_offset_array == NULL) {
        goto out;
    }
    node_array = table_read_column_array(node_input, NPY_INT32, &num_rows, true);
    if (node_array == NULL) {
        goto out;
    }

    time_data = NULL;
    if (time_input != Py_None) {
        time_array = table_read_column_array(time_input, NPY_FLOAT64, &num_rows, true);
        if (time_array == NULL) {
            goto out;
        }
        time_data = PyArray_DATA(time_array);
    }

    parent_data = NULL;
    if (parent_input != Py_None) {
        parent_array = table_read_column_array(parent_input, NPY_INT32, &num_rows, true);
        if (parent_array == NULL) {
            goto out;
        }
        parent_data = PyArray_DATA(parent_array);
    }

    metadata_data = NULL;
    metadata_offset_data = NULL;
    if ((metadata_input == Py_None) != (metadata_offset_input == Py_None)) {
        PyErr_SetString(
            PyExc_TypeError, "metadata and metadata_offset must be specified together");
        goto out;
    }
    if (metadata_input != Py_None) {
        metadata_array
            = table_read_column_array(metadata_input, NPY_INT8, &metadata_length, false);
        if (metadata_array == NULL) {
            goto out;
        }
        metadata_data = PyArray_DATA(metadata_array);
        metadata_offset_array = table_read_offset_array(
            metadata_offset_input, &num_rows, metadata_length, false);
        if (metadata_offset_array == NULL) {
            goto out;
        }
        metadata_offset_data = PyArray_DATA(metadata_offset_array);
    }
    if (metadata_schema_input != Py_None) {
        metadata_schema
            = parse_metadata_schema_arg(metadata_schema_input, &metadata_schema_length);
        if (metadata_schema == NULL) {
            goto out;
        }
        err = tsk_mutation_table_set_metadata_schema(
            table, metadata_schema, metadata_schema_length);
        if (err != 0) {
            handle_tskit_error(err);
            goto out;
        }
    }

    if (clear_table) {
        err = tsk_mutation_table_clear(table);
        if (err != 0) {
            handle_tskit_error(err);
            goto out;
        }
    }
    err = tsk_mutation_table_append_columns(table, num_rows, PyArray_DATA(site_array),
        PyArray_DATA(node_array), parent_data, time_data,
        PyArray_DATA(derived_state_array), PyArray_DATA(derived_state_offset_array),
        metadata_data, metadata_offset_data);
    if (err != 0) {
        handle_tskit_error(err);
        goto out;
    }
    ret = 0;
out:
    Py_XDECREF(site_array);
    Py_XDECREF(derived_state_array);
    Py_XDECREF(derived_state_offset_array);
    Py_XDECREF(metadata_array);
    Py_XDECREF(metadata_offset_array);
    Py_XDECREF(node_array);
    Py_XDECREF(parent_array);
    Py_XDECREF(time_array);
    return ret;
}

static int
parse_population_table_dict(
    tsk_population_table_t *table, PyObject *dict, bool clear_table)
{
    int err;
    int ret = -1;
    size_t num_rows, metadata_length;
    PyObject *metadata_input = NULL;
    PyArrayObject *metadata_array = NULL;
    PyObject *metadata_offset_input = NULL;
    PyArrayObject *metadata_offset_array = NULL;
    PyObject *metadata_schema_input = NULL;
    const char *metadata_schema = NULL;
    Py_ssize_t metadata_schema_length = 0;

    /* Get the inputs */
    metadata_input = get_table_dict_value(dict, "metadata", true);
    if (metadata_input == NULL) {
        goto out;
    }
    metadata_offset_input = get_table_dict_value(dict, "metadata_offset", true);
    if (metadata_offset_input == NULL) {
        goto out;
    }
    metadata_schema_input = get_table_dict_value(dict, "metadata_schema", false);
    if (metadata_schema_input == NULL) {
        goto out;
    }

    /* Get the arrays */
    metadata_array
        = table_read_column_array(metadata_input, NPY_INT8, &metadata_length, false);
    if (metadata_array == NULL) {
        goto out;
    }
    metadata_offset_array = table_read_offset_array(
        metadata_offset_input, &num_rows, metadata_length, false);
    if (metadata_offset_array == NULL) {
        goto out;
    }
    if (metadata_schema_input != Py_None) {
        metadata_schema
            = parse_metadata_schema_arg(metadata_schema_input, &metadata_schema_length);
        if (metadata_schema == NULL) {
            goto out;
        }
        err = tsk_population_table_set_metadata_schema(
            table, metadata_schema, metadata_schema_length);
        if (err != 0) {
            handle_tskit_error(err);
            goto out;
        }
    }

    if (clear_table) {
        err = tsk_population_table_clear(table);
        if (err != 0) {
            handle_tskit_error(err);
            goto out;
        }
    }
    err = tsk_population_table_append_columns(table, num_rows,
        PyArray_DATA(metadata_array), PyArray_DATA(metadata_offset_array));
    if (err != 0) {
        handle_tskit_error(err);
        goto out;
    }
    ret = 0;
out:
    Py_XDECREF(metadata_array);
    Py_XDECREF(metadata_offset_array);
    return ret;
}

static int
parse_provenance_table_dict(
    tsk_provenance_table_t *table, PyObject *dict, bool clear_table)
{
    int err;
    int ret = -1;
    size_t num_rows, timestamp_length, record_length;
    PyObject *timestamp_input = NULL;
    PyArrayObject *timestamp_array = NULL;
    PyObject *timestamp_offset_input = NULL;
    PyArrayObject *timestamp_offset_array = NULL;
    PyObject *record_input = NULL;
    PyArrayObject *record_array = NULL;
    PyObject *record_offset_input = NULL;
    PyArrayObject *record_offset_array = NULL;

    /* Get the inputs */
    timestamp_input = get_table_dict_value(dict, "timestamp", true);
    if (timestamp_input == NULL) {
        goto out;
    }
    timestamp_offset_input = get_table_dict_value(dict, "timestamp_offset", true);
    if (timestamp_offset_input == NULL) {
        goto out;
    }
    record_input = get_table_dict_value(dict, "record", true);
    if (record_input == NULL) {
        goto out;
    }
    record_offset_input = get_table_dict_value(dict, "record_offset", true);
    if (record_offset_input == NULL) {
        goto out;
    }

    timestamp_array
        = table_read_column_array(timestamp_input, NPY_INT8, &timestamp_length, false);
    if (timestamp_array == NULL) {
        goto out;
    }
    timestamp_offset_array = table_read_offset_array(
        timestamp_offset_input, &num_rows, timestamp_length, false);
    if (timestamp_offset_array == NULL) {
        goto out;
    }
    record_array
        = table_read_column_array(record_input, NPY_INT8, &record_length, false);
    if (record_array == NULL) {
        goto out;
    }
    record_offset_array
        = table_read_offset_array(record_offset_input, &num_rows, record_length, true);
    if (record_offset_array == NULL) {
        goto out;
    }

    if (clear_table) {
        err = tsk_provenance_table_clear(table);
        if (err != 0) {
            handle_tskit_error(err);
            goto out;
        }
    }
    err = tsk_provenance_table_append_columns(table, num_rows,
        PyArray_DATA(timestamp_array), PyArray_DATA(timestamp_offset_array),
        PyArray_DATA(record_array), PyArray_DATA(record_offset_array));
    if (err != 0) {
        handle_tskit_error(err);
        goto out;
    }
    ret = 0;
out:
    Py_XDECREF(timestamp_array);
    Py_XDECREF(timestamp_offset_array);
    Py_XDECREF(record_array);
    Py_XDECREF(record_offset_array);
    return ret;
}

static int
parse_indexes_dict(tsk_table_collection_t *tables, PyObject *dict)
{
    int err;
    int ret = -1;
    size_t insertion_length, removal_length;
    PyObject *insertion_input = NULL;
    PyArrayObject *insertion_array = NULL;
    PyObject *removal_input = NULL;
    PyArrayObject *removal_array = NULL;

    /* Get the inputs */
    insertion_input = get_table_dict_value(dict, "edge_insertion_order", false);
    if (insertion_input == NULL) {
        goto out;
    }
    removal_input = get_table_dict_value(dict, "edge_removal_order", false);
    if (removal_input == NULL) {
        goto out;
    }

    if ((insertion_input == Py_None) != (removal_input == Py_None)) {
        PyErr_SetString(PyExc_TypeError,
            "edge_insertion_order and edge_removal_order must be specified together");
        goto out;
    }

    if (insertion_input != Py_None) {
        insertion_array = table_read_column_array(
            insertion_input, NPY_INT32, &insertion_length, false);
        if (insertion_array == NULL) {
            goto out;
        }
        removal_array
            = table_read_column_array(removal_input, NPY_INT32, &removal_length, false);
        if (removal_array == NULL) {
            goto out;
        }
        if (insertion_length != removal_length) {
            PyErr_SetString(PyExc_ValueError,
                "edge_insertion_order and edge_removal_order must be the same length");
            goto out;
        }
        if (insertion_length != tables->edges.num_rows) {
            PyErr_SetString(PyExc_ValueError,
                "edge_insertion_order and edge_removal_order must be "
                "the same length as the number of edges");
            goto out;
        }
        err = tsk_table_collection_set_indexes(
            tables, PyArray_DATA(insertion_array), PyArray_DATA(removal_array));
        if (err != 0) {
            handle_tskit_error(err);
            goto out;
        }
    }
    ret = 0;
out:
    Py_XDECREF(insertion_array);
    Py_XDECREF(removal_array);
    return ret;
}

static int
parse_table_collection_dict(tsk_table_collection_t *tables, PyObject *tables_dict)
{
    int ret = -1;
    PyObject *value = NULL;
    int err;
    char *metadata = NULL;
    const char *metadata_schema = NULL;
    Py_ssize_t metadata_length, metadata_schema_length;

    value = get_table_dict_value(tables_dict, "sequence_length", true);
    if (value == NULL) {
        goto out;
    }
    if (!PyNumber_Check(value)) {
        PyErr_Format(PyExc_TypeError, "'sequence_length' is not number");
        goto out;
    }
    tables->sequence_length = PyFloat_AsDouble(value);

    /* metadata_schema */
    value = get_table_dict_value(tables_dict, "metadata_schema", false);
    if (value == NULL) {
        goto out;
    }
    if (value != Py_None) {
        if (!PyUnicode_Check(value)) {
            PyErr_Format(PyExc_TypeError, "'metadata_schema' is not a string");
            goto out;
        }
        metadata_schema = parse_metadata_schema_arg(value, &metadata_schema_length);
        if (metadata_schema == NULL) {
            goto out;
        }
        err = tsk_table_collection_set_metadata_schema(
            tables, metadata_schema, metadata_schema_length);
        if (err != 0) {
            handle_tskit_error(err);
            goto out;
        }
    }

    /* metadata */
    value = get_table_dict_value(tables_dict, "metadata", false);
    if (value == NULL) {
        goto out;
    }
    if (value != Py_None) {
        if (!PyBytes_Check(value)) {
            PyErr_Format(PyExc_TypeError, "'metadata' is not bytes");
            goto out;
        }
        err = PyBytes_AsStringAndSize(value, &metadata, &metadata_length);
        if (err != 0) {
            goto out;
        }
        err = tsk_table_collection_set_metadata(tables, metadata, metadata_length);
        if (err != 0) {
            handle_tskit_error(err);
            goto out;
        }
    }

    /* individuals */
    value = get_table_dict_value(tables_dict, "individuals", true);
    if (value == NULL) {
        goto out;
    }
    if (!PyDict_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "not a dictionary");
        goto out;
    }
    if (parse_individual_table_dict(&tables->individuals, value, true) != 0) {
        goto out;
    }

    /* nodes */
    value = get_table_dict_value(tables_dict, "nodes", true);
    if (value == NULL) {
        goto out;
    }
    if (!PyDict_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "not a dictionary");
        goto out;
    }
    if (parse_node_table_dict(&tables->nodes, value, true) != 0) {
        goto out;
    }

    /* edges */
    value = get_table_dict_value(tables_dict, "edges", true);
    if (value == NULL) {
        goto out;
    }
    if (!PyDict_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "not a dictionary");
        goto out;
    }
    if (parse_edge_table_dict(&tables->edges, value, true) != 0) {
        goto out;
    }

    /* migrations */
    value = get_table_dict_value(tables_dict, "migrations", true);
    if (value == NULL) {
        goto out;
    }
    if (!PyDict_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "not a dictionary");
        goto out;
    }
    if (parse_migration_table_dict(&tables->migrations, value, true) != 0) {
        goto out;
    }

    /* sites */
    value = get_table_dict_value(tables_dict, "sites", true);
    if (value == NULL) {
        goto out;
    }
    if (!PyDict_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "not a dictionary");
        goto out;
    }
    if (parse_site_table_dict(&tables->sites, value, true) != 0) {
        goto out;
    }

    /* mutations */
    value = get_table_dict_value(tables_dict, "mutations", true);
    if (value == NULL) {
        goto out;
    }
    if (!PyDict_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "not a dictionary");
        goto out;
    }
    if (parse_mutation_table_dict(&tables->mutations, value, true) != 0) {
        goto out;
    }

    /* populations */
    value = get_table_dict_value(tables_dict, "populations", true);
    if (value == NULL) {
        goto out;
    }
    if (!PyDict_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "not a dictionary");
        goto out;
    }
    if (parse_population_table_dict(&tables->populations, value, true) != 0) {
        goto out;
    }

    /* provenances */
    value = get_table_dict_value(tables_dict, "provenances", true);
    if (value == NULL) {
        goto out;
    }
    if (!PyDict_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "not a dictionary");
        goto out;
    }
    if (parse_provenance_table_dict(&tables->provenances, value, true) != 0) {
        goto out;
    }

    /* indexes */
    value = get_table_dict_value(tables_dict, "indexes", false);
    if (value == NULL) {
        goto out;
    }
    if (value != Py_None) {
        if (!PyDict_Check(value)) {
            PyErr_SetString(PyExc_TypeError, "not a dictionary");
            goto out;
        }
        if (parse_indexes_dict(tables, value) != 0) {
            goto out;
        }
    }

    ret = 0;
out:
    return ret;
}

static int
write_table_arrays(tsk_table_collection_t *tables, PyObject *dict)
{
    struct table_col {
        const char *name;
        void *data;
        npy_intp num_rows;
        int type;
    };
    struct table_desc {
        const char *name;
        struct table_col *cols;
        char *metadata_schema;
        tsk_size_t metadata_schema_length;
    };
    int ret = -1;
    PyObject *array = NULL;
    PyObject *table_dict = NULL;
    size_t j;
    struct table_col *col;

    struct table_col individual_cols[] = {
        { "flags", (void *) tables->individuals.flags, tables->individuals.num_rows,
            NPY_UINT32 },
        { "location", (void *) tables->individuals.location,
            tables->individuals.location_length, NPY_FLOAT64 },
        { "location_offset", (void *) tables->individuals.location_offset,
            tables->individuals.num_rows + 1, NPY_UINT32 },
        { "parents", (void *) tables->individuals.parents,
            tables->individuals.parents_length, NPY_INT32 },
        { "parents_offset", (void *) tables->individuals.parents_offset,
            tables->individuals.num_rows + 1, NPY_UINT32 },
        { "metadata", (void *) tables->individuals.metadata,
            tables->individuals.metadata_length, NPY_INT8 },
        { "metadata_offset", (void *) tables->individuals.metadata_offset,
            tables->individuals.num_rows + 1, NPY_UINT32 },
        { NULL },
    };

    struct table_col node_cols[] = {
        { "time", (void *) tables->nodes.time, tables->nodes.num_rows, NPY_FLOAT64 },
        { "flags", (void *) tables->nodes.flags, tables->nodes.num_rows, NPY_UINT32 },
        { "population", (void *) tables->nodes.population, tables->nodes.num_rows,
            NPY_INT32 },
        { "individual", (void *) tables->nodes.individual, tables->nodes.num_rows,
            NPY_INT32 },
        { "metadata", (void *) tables->nodes.metadata, tables->nodes.metadata_length,
            NPY_INT8 },
        { "metadata_offset", (void *) tables->nodes.metadata_offset,
            tables->nodes.num_rows + 1, NPY_UINT32 },
        { NULL },
    };

    struct table_col edge_cols[] = {
        { "left", (void *) tables->edges.left, tables->edges.num_rows, NPY_FLOAT64 },
        { "right", (void *) tables->edges.right, tables->edges.num_rows, NPY_FLOAT64 },
        { "parent", (void *) tables->edges.parent, tables->edges.num_rows, NPY_INT32 },
        { "child", (void *) tables->edges.child, tables->edges.num_rows, NPY_INT32 },
        { "metadata", (void *) tables->edges.metadata, tables->edges.metadata_length,
            NPY_INT8 },
        { "metadata_offset", (void *) tables->edges.metadata_offset,
            tables->edges.num_rows + 1, NPY_UINT32 },
        { NULL },
    };

    struct table_col migration_cols[] = {
        { "left", (void *) tables->migrations.left, tables->migrations.num_rows,
            NPY_FLOAT64 },
        { "right", (void *) tables->migrations.right, tables->migrations.num_rows,
            NPY_FLOAT64 },
        { "node", (void *) tables->migrations.node, tables->migrations.num_rows,
            NPY_INT32 },
        { "source", (void *) tables->migrations.source, tables->migrations.num_rows,
            NPY_INT32 },
        { "dest", (void *) tables->migrations.dest, tables->migrations.num_rows,
            NPY_INT32 },
        { "time", (void *) tables->migrations.time, tables->migrations.num_rows,
            NPY_FLOAT64 },
        { "metadata", (void *) tables->migrations.metadata,
            tables->migrations.metadata_length, NPY_INT8 },
        { "metadata_offset", (void *) tables->migrations.metadata_offset,
            tables->migrations.num_rows + 1, NPY_UINT32 },

        { NULL },
    };

    struct table_col site_cols[] = {
        { "position", (void *) tables->sites.position, tables->sites.num_rows,
            NPY_FLOAT64 },
        { "ancestral_state", (void *) tables->sites.ancestral_state,
            tables->sites.ancestral_state_length, NPY_INT8 },
        { "ancestral_state_offset", (void *) tables->sites.ancestral_state_offset,
            tables->sites.num_rows + 1, NPY_UINT32 },
        { "metadata", (void *) tables->sites.metadata, tables->sites.metadata_length,
            NPY_INT8 },
        { "metadata_offset", (void *) tables->sites.metadata_offset,
            tables->sites.num_rows + 1, NPY_UINT32 },
        { NULL },
    };

    struct table_col mutation_cols[] = {
        { "site", (void *) tables->mutations.site, tables->mutations.num_rows,
            NPY_INT32 },
        { "node", (void *) tables->mutations.node, tables->mutations.num_rows,
            NPY_INT32 },
        { "time", (void *) tables->mutations.time, tables->mutations.num_rows,
            NPY_FLOAT64 },
        { "parent", (void *) tables->mutations.parent, tables->mutations.num_rows,
            NPY_INT32 },
        { "derived_state", (void *) tables->mutations.derived_state,
            tables->mutations.derived_state_length, NPY_INT8 },
        { "derived_state_offset", (void *) tables->mutations.derived_state_offset,
            tables->mutations.num_rows + 1, NPY_UINT32 },
        { "metadata", (void *) tables->mutations.metadata,
            tables->mutations.metadata_length, NPY_INT8 },
        { "metadata_offset", (void *) tables->mutations.metadata_offset,
            tables->mutations.num_rows + 1, NPY_UINT32 },
        { NULL },
    };

    struct table_col population_cols[] = {
        { "metadata", (void *) tables->populations.metadata,
            tables->populations.metadata_length, NPY_INT8 },
        { "metadata_offset", (void *) tables->populations.metadata_offset,
            tables->populations.num_rows + 1, NPY_UINT32 },
        { NULL },
    };

    struct table_col provenance_cols[] = {
        { "timestamp", (void *) tables->provenances.timestamp,
            tables->provenances.timestamp_length, NPY_INT8 },
        { "timestamp_offset", (void *) tables->provenances.timestamp_offset,
            tables->provenances.num_rows + 1, NPY_UINT32 },
        { "record", (void *) tables->provenances.record,
            tables->provenances.record_length, NPY_INT8 },
        { "record_offset", (void *) tables->provenances.record_offset,
            tables->provenances.num_rows + 1, NPY_UINT32 },
        { NULL },
    };

    struct table_col indexes_cols[] = {
        { "edge_insertion_order", (void *) tables->indexes.edge_insertion_order,
            tables->indexes.num_edges, NPY_INT32 },
        { "edge_removal_order", (void *) tables->indexes.edge_removal_order,
            tables->indexes.num_edges, NPY_INT32 },
        { NULL },
    };

    struct table_col no_indexes_cols[] = {
        { NULL },
    };

    struct table_desc table_descs[] = {
        { "individuals", individual_cols, tables->individuals.metadata_schema,
            tables->individuals.metadata_schema_length },
        { "nodes", node_cols, tables->nodes.metadata_schema,
            tables->nodes.metadata_schema_length },
        { "edges", edge_cols, tables->edges.metadata_schema,
            tables->edges.metadata_schema_length },
        { "migrations", migration_cols, tables->migrations.metadata_schema,
            tables->migrations.metadata_schema_length },
        { "sites", site_cols, tables->sites.metadata_schema,
            tables->sites.metadata_schema_length },
        { "mutations", mutation_cols, tables->mutations.metadata_schema,
            tables->mutations.metadata_schema_length },
        { "populations", population_cols, tables->populations.metadata_schema,
            tables->populations.metadata_schema_length },
        { "provenances", provenance_cols, NULL, 0 },
        /* We don't want to insert empty indexes, return an empty dict if there are none
         */
        { "indexes",
            tsk_table_collection_has_index(tables, 0) ? indexes_cols : no_indexes_cols,
            NULL, 0 },
    };

    for (j = 0; j < sizeof(table_descs) / sizeof(*table_descs); j++) {
        table_dict = PyDict_New();
        if (table_dict == NULL) {
            goto out;
        }
        col = table_descs[j].cols;
        while (col->name != NULL) {
            array = (PyObject *) PyArray_EMPTY(1, &col->num_rows, col->type, 0);
            if (array == NULL) {
                goto out;
            }
            memcpy(PyArray_DATA((PyArrayObject *) array), col->data,
                col->num_rows * PyArray_ITEMSIZE((PyArrayObject *) array));
            if (PyDict_SetItemString(table_dict, col->name, array) != 0) {
                goto out;
            }
            Py_DECREF(array);
            array = NULL;
            col++;
        }
        if (table_descs[j].metadata_schema_length > 0) {
            array = make_Py_Unicode_FromStringAndLength(
                table_descs[j].metadata_schema, table_descs[j].metadata_schema_length);
            if (array == NULL) {
                goto out;
            }
            if (PyDict_SetItemString(table_dict, "metadata_schema", array) != 0) {
                goto out;
            }
            Py_DECREF(array);
            array = NULL;
        }

        if (PyDict_SetItemString(dict, table_descs[j].name, table_dict) != 0) {
            goto out;
        }
        Py_DECREF(table_dict);
        table_dict = NULL;
    }

    ret = 0;
out:
    Py_XDECREF(array);
    Py_XDECREF(table_dict);
    return ret;
}

/* Returns a dictionary encoding of the specified table collection */
static PyObject *
dump_tables_dict(tsk_table_collection_t *tables)
{
    PyObject *ret = NULL;
    PyObject *dict = NULL;
    PyObject *val = NULL;
    int err;

    dict = PyDict_New();
    if (dict == NULL) {
        goto out;
    }

    /* Dict representation version */
    val = Py_BuildValue("ll", 1, 3);
    if (val == NULL) {
        goto out;
    }
    if (PyDict_SetItemString(dict, "encoding_version", val) != 0) {
        goto out;
    }
    Py_DECREF(val);
    val = NULL;

    val = Py_BuildValue("d", tables->sequence_length);
    if (val == NULL) {
        goto out;
    }
    if (PyDict_SetItemString(dict, "sequence_length", val) != 0) {
        goto out;
    }
    Py_DECREF(val);
    val = NULL;

    if (tables->metadata_schema_length > 0) {
        val = make_Py_Unicode_FromStringAndLength(
            tables->metadata_schema, tables->metadata_schema_length);
        if (val == NULL) {
            goto out;
        }
        if (PyDict_SetItemString(dict, "metadata_schema", val) != 0) {
            goto out;
        }
        Py_DECREF(val);
        val = NULL;
    }

    if (tables->metadata_length > 0) {
        val = PyBytes_FromStringAndSize(tables->metadata, tables->metadata_length);
        if (val == NULL) {
            goto out;
        }
        if (PyDict_SetItemString(dict, "metadata", val) != 0) {
            goto out;
        }
        Py_DECREF(val);
        val = NULL;
    }

    err = write_table_arrays(tables, dict);
    if (err != 0) {
        goto out;
    }
    ret = dict;
    dict = NULL;
out:
    Py_XDECREF(dict);
    Py_XDECREF(val);
    return ret;
}

/*===================================================================
 * LightweightTableCollection
 *===================================================================
 */

static int
LightweightTableCollection_check_state(LightweightTableCollection *self)
{
    int ret = 0;
    if (self->tables == NULL) {
        PyErr_SetString(PyExc_SystemError, "LightweightTableCollection not initialised");
        ret = -1;
    }
    return ret;
}

static void
LightweightTableCollection_dealloc(LightweightTableCollection *self)
{
    if (self->tables != NULL) {
        tsk_table_collection_free(self->tables);
        PyMem_Free(self->tables);
        self->tables = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static int
LightweightTableCollection_init(
    LightweightTableCollection *self, PyObject *args, PyObject *kwds)
{
    int ret = -1;
    int err;
    static char *kwlist[] = { "sequence_length", NULL };
    double sequence_length = -1;

    self->tables = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|d", kwlist, &sequence_length)) {
        goto out;
    }
    self->tables = PyMem_Malloc(sizeof(*self->tables));
    if (self->tables == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    err = tsk_table_collection_init(self->tables, 0);
    if (err != 0) {
        handle_tskit_error(err);
        goto out;
    }
    self->tables->sequence_length = sequence_length;
    ret = 0;
out:
    return ret;
}

static PyObject *
LightweightTableCollection_asdict(LightweightTableCollection *self)
{
    PyObject *ret = NULL;

    if (LightweightTableCollection_check_state(self) != 0) {
        goto out;
    }
    ret = dump_tables_dict(self->tables);
out:
    return ret;
}

static PyObject *
LightweightTableCollection_fromdict(LightweightTableCollection *self, PyObject *args)
{
    int err;
    PyObject *ret = NULL;
    PyObject *dict = NULL;

    if (LightweightTableCollection_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "O!", &PyDict_Type, &dict)) {
        goto out;
    }
    err = parse_table_collection_dict(self->tables, dict);
    if (err != 0) {
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyMethodDef LightweightTableCollection_methods[] = {
    { .ml_name = "asdict",
        .ml_meth = (PyCFunction) LightweightTableCollection_asdict,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns the tables encoded as a dictionary." },
    { .ml_name = "fromdict",
        .ml_meth = (PyCFunction) LightweightTableCollection_fromdict,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Populates the internal tables using the specified dictionary." },
    { NULL } /* Sentinel */
};

static PyTypeObject LightweightTableCollectionType = {
    // clang-format off
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "LightweightTableCollection",
    .tp_doc = "Low-level table collection interchange.",
    .tp_basicsize = sizeof(LightweightTableCollection),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_methods = LightweightTableCollection_methods,
    .tp_init = (initproc) LightweightTableCollection_init,
    .tp_dealloc = (destructor) LightweightTableCollection_dealloc,
    // clang-format on
};

static int
register_lwt_class(PyObject *module)
{
    if (PyType_Ready(&LightweightTableCollectionType) < 0) {
        return -1;
    }
    Py_INCREF(&LightweightTableCollectionType);
    PyModule_AddObject(module, "LightweightTableCollection",
        (PyObject *) &LightweightTableCollectionType);
    return 0;
}
