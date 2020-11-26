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

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define TSK_BUG_ASSERT_MESSAGE                                                          \
    "Please open an issue on"                                                           \
    " GitHub, ideally with a reproducible example."                                     \
    " (https://github.com/tskit-dev/tskit/issues)"

#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>
#include <float.h>

#include "kastore.h"
#include "tskit.h"

#define SET_COLS 0
#define APPEND_COLS 1

/* TskitException is the superclass of all exceptions that can be thrown by
 * tskit. We define it here in the low-level library so that exceptions defined
 * here and in the high-level library can inherit from it.
 */
static PyObject *TskitException;
static PyObject *TskitLibraryError;
static PyObject *TskitFileFormatError;
static PyObject *TskitVersionTooOldError;
static PyObject *TskitVersionTooNewError;

#include "tskit_lwt_interface.h"

// clang-format off

/* The XTable classes each have 'lock' attribute, which is used to
 * raise an error if a Python thread attempts to access a table
 * while another Python thread is operating on it. Because tables
 * allocate memory dynamically, we cannot gaurantee safety otherwise.
 * The locks are set before the GIL is released and unset afterwards.
 * Because C code executed here represents atomic Python operations
 * (while the GIL is held), this should be safe */

typedef struct _TableCollection {
    PyObject_HEAD
    tsk_table_collection_t *tables;
} TableCollection;

 /* The table pointer in each of the Table classes either points to locally
  * allocated memory or to the table stored in a tbl_collection_t. If we're
  * using the memory in a tbl_collection_t, we keep a reference to the
  * TableCollection object to ensure that the memory isn't free'd while a
  * reference to the table itself is live. */
typedef struct {
    PyObject_HEAD
    bool locked;
    tsk_individual_table_t *table;
    TableCollection *tables;
} IndividualTable;

typedef struct {
    PyObject_HEAD
    bool locked;
    tsk_node_table_t *table;
    TableCollection *tables;
} NodeTable;

typedef struct {
    PyObject_HEAD
    bool locked;
    tsk_edge_table_t *table;
    TableCollection *tables;
} EdgeTable;

typedef struct {
    PyObject_HEAD
    bool locked;
    tsk_site_table_t *table;
    TableCollection *tables;
} SiteTable;

typedef struct {
    PyObject_HEAD
    bool locked;
    tsk_mutation_table_t *table;
    TableCollection *tables;
} MutationTable;

typedef struct {
    PyObject_HEAD
    bool locked;
    tsk_migration_table_t *table;
    TableCollection *tables;
} MigrationTable;

typedef struct {
    PyObject_HEAD
    bool locked;
    tsk_population_table_t *table;
    TableCollection *tables;
} PopulationTable;

typedef struct {
    PyObject_HEAD
    bool locked;
    tsk_provenance_table_t *table;
    TableCollection *tables;
} ProvenanceTable;

typedef struct {
    PyObject_HEAD
    tsk_treeseq_t *tree_sequence;
} TreeSequence;

typedef struct {
    PyObject_HEAD
    TreeSequence *tree_sequence;
    tsk_tree_t *tree;
} Tree;

typedef struct {
    PyObject_HEAD
    Tree *tree;
    int first;
} TreeIterator;

typedef struct {
    PyObject_HEAD
    TreeSequence *tree_sequence;
    tsk_diff_iter_t *tree_diff_iterator;
} TreeDiffIterator;

typedef struct {
    PyObject_HEAD
    TreeSequence *tree_sequence;
    tsk_vargen_t *variant_generator;
} VariantGenerator;

typedef struct {
    PyObject_HEAD
    TreeSequence *tree_sequence;
    tsk_ld_calc_t *ld_calc;
} LdCalculator;

typedef struct {
    PyObject_HEAD
    TreeSequence *tree_sequence;
    tsk_ls_hmm_t *ls_hmm;
} LsHmm;

typedef struct {
    PyObject_HEAD
    TreeSequence *tree_sequence;
    tsk_compressed_matrix_t *compressed_matrix;
} CompressedMatrix;

typedef struct {
    PyObject_HEAD
    TreeSequence *tree_sequence;
    tsk_viterbi_matrix_t *viterbi_matrix;
} ViterbiMatrix;

/* A named tuple of metadata schemas for a tree sequence */
static PyTypeObject MetadataSchemas;

static PyStructSequence_Field metadata_schemas_fields[] = {
    { "node", "The node metadata schema" },
    { "edge", "The edge metadata schema" },
    { "site", "The site metadata schema" },
    { "mutation", "The mutation metadata schema" },
    { "migration", "The migration metadata schema" },
    { "individual", "The individual metadata schema" },
    { "population", "The population metadata schema" },
    { NULL }
};

static PyStructSequence_Desc metadata_schemas_desc = {
    .name = "MetadataSchemas",
    .doc = "Namedtuple of metadata schemas for this tree sequence",
    .fields = metadata_schemas_fields,
    .n_in_sequence = 7
};

// clang-format on

static void
handle_library_error(int err)
{
    if (tsk_is_kas_error(err)) {
        PyErr_SetString(TskitFileFormatError, tsk_strerror(err));
    } else {
        switch (err) {
            case TSK_ERR_FILE_VERSION_TOO_NEW:
                PyErr_SetString(TskitVersionTooNewError, tsk_strerror(err));
                break;
            case TSK_ERR_FILE_VERSION_TOO_OLD:
                PyErr_SetString(TskitVersionTooOldError, tsk_strerror(err));
                break;
            case TSK_ERR_FILE_FORMAT:
                PyErr_SetString(TskitFileFormatError, tsk_strerror(err));
                break;
            case TSK_ERR_IO:
                PyErr_SetFromErrno(PyExc_OSError);
                break;
            case TSK_ERR_EOF:
                PyErr_Format(PyExc_EOFError, "End of file");
                break;
            default:
                PyErr_SetString(TskitLibraryError, tsk_strerror(err));
        }
    }
}

static PyObject *
convert_node_id_list(tsk_id_t *children, size_t num_children)
{
    PyObject *ret = NULL;
    PyObject *t;
    PyObject *py_int;
    size_t j;

    t = PyTuple_New(num_children);
    if (t == NULL) {
        goto out;
    }
    for (j = 0; j < num_children; j++) {
        py_int = Py_BuildValue("i", (int) children[j]);
        if (py_int == NULL) {
            Py_DECREF(t);
            goto out;
        }
        PyTuple_SET_ITEM(t, j, py_int);
    }
    ret = t;
out:
    return ret;
}

static PyObject *
make_metadata(const char *metadata, Py_ssize_t length)
{
    const char *m = metadata == NULL ? "" : metadata;
    return PyBytes_FromStringAndSize(m, length);
}

static PyObject *
make_mutation(const tsk_mutation_t *mutation)
{
    PyObject *ret = NULL;
    PyObject *metadata = NULL;

    metadata = make_metadata(mutation->metadata, (Py_ssize_t) mutation->metadata_length);
    if (metadata == NULL) {
        goto out;
    }
    ret = Py_BuildValue("iis#iOd", mutation->site, mutation->node,
        mutation->derived_state, (Py_ssize_t) mutation->derived_state_length,
        mutation->parent, metadata, mutation->time);
out:
    Py_XDECREF(metadata);
    return ret;
}

static PyObject *
make_mutation_id_list(const tsk_mutation_t *mutations, size_t length)
{
    PyObject *ret = NULL;
    PyObject *t;
    PyObject *item;
    size_t j;

    t = PyTuple_New(length);
    if (t == NULL) {
        goto out;
    }
    for (j = 0; j < length; j++) {
        item = Py_BuildValue("i", mutations[j].id);
        if (item == NULL) {
            Py_DECREF(t);
            goto out;
        }
        PyTuple_SET_ITEM(t, j, item);
    }
    ret = t;
out:
    return ret;
}

static PyObject *
make_population(const tsk_population_t *population)
{
    PyObject *ret = NULL;
    PyObject *metadata
        = make_metadata(population->metadata, (Py_ssize_t) population->metadata_length);
    if (metadata == NULL) {
        goto out;
    }
    ret = Py_BuildValue("(O)", metadata);
out:
    Py_XDECREF(metadata);
    return ret;
}

static PyObject *
make_provenance(const tsk_provenance_t *provenance)
{
    PyObject *ret = NULL;

    ret = Py_BuildValue("s#s#", provenance->timestamp,
        (Py_ssize_t) provenance->timestamp_length, provenance->record,
        (Py_ssize_t) provenance->record_length);
    return ret;
}

static PyObject *
make_individual_row(const tsk_individual_t *r)
{
    PyObject *ret = NULL;
    PyObject *metadata = make_metadata(r->metadata, (Py_ssize_t) r->metadata_length);
    PyArrayObject *location = NULL;
    npy_intp dims;

    dims = (npy_intp) r->location_length;
    location = (PyArrayObject *) PyArray_SimpleNew(1, &dims, NPY_FLOAT64);
    if (metadata == NULL || location == NULL) {
        goto out;
    }
    memcpy(PyArray_DATA(location), r->location, r->location_length * sizeof(double));
    ret = Py_BuildValue("IOO", (unsigned int) r->flags, location, metadata);
out:
    Py_XDECREF(location);
    Py_XDECREF(metadata);
    return ret;
}

static PyObject *
make_individual_object(const tsk_individual_t *r)
{
    PyObject *ret = NULL;
    PyObject *metadata = make_metadata(r->metadata, (Py_ssize_t) r->metadata_length);
    PyArrayObject *location = NULL;
    PyArrayObject *nodes = NULL;
    npy_intp dims;

    dims = (npy_intp) r->location_length;
    location = (PyArrayObject *) PyArray_SimpleNew(1, &dims, NPY_FLOAT64);
    dims = (npy_intp) r->nodes_length;
    nodes = (PyArrayObject *) PyArray_SimpleNew(1, &dims, NPY_INT32);
    if (metadata == NULL || location == NULL || nodes == NULL) {
        goto out;
    }
    memcpy(PyArray_DATA(location), r->location, r->location_length * sizeof(double));
    memcpy(PyArray_DATA(nodes), r->nodes, r->nodes_length * sizeof(tsk_id_t));
    ret = Py_BuildValue("IOOO", (unsigned int) r->flags, location, metadata, nodes);
out:
    Py_XDECREF(location);
    Py_XDECREF(metadata);
    Py_XDECREF(nodes);
    return ret;
}

static PyObject *
make_node(const tsk_node_t *r)
{
    PyObject *ret = NULL;
    PyObject *metadata = make_metadata(r->metadata, (Py_ssize_t) r->metadata_length);
    if (metadata == NULL) {
        goto out;
    }
    ret = Py_BuildValue("IdiiO", (unsigned int) r->flags, r->time, (int) r->population,
        (int) r->individual, metadata);
out:
    Py_XDECREF(metadata);
    return ret;
}

static PyObject *
make_edge(const tsk_edge_t *edge, bool include_id)
{
    PyObject *ret = NULL;
    PyObject *metadata
        = make_metadata(edge->metadata, (Py_ssize_t) edge->metadata_length);
    if (metadata == NULL) {
        goto out;
    }
    if (include_id) {
        ret = Py_BuildValue("ddiiOi", edge->left, edge->right, (int) edge->parent,
            (int) edge->child, metadata, edge->id);
    } else {
        ret = Py_BuildValue("ddiiO", edge->left, edge->right, (int) edge->parent,
            (int) edge->child, metadata);
    }
out:
    Py_XDECREF(metadata);
    return ret;
}

static PyObject *
make_migration(const tsk_migration_t *r)
{
    int source = r->source == TSK_NULL ? -1 : r->source;
    int dest = r->dest == TSK_NULL ? -1 : r->dest;
    PyObject *ret = NULL;
    PyObject *metadata = make_metadata(r->metadata, (Py_ssize_t) r->metadata_length);
    if (metadata == NULL) {
        goto out;
    }
    ret = Py_BuildValue(
        "ddiiidO", r->left, r->right, (int) r->node, source, dest, r->time, metadata);
out:
    Py_XDECREF(metadata);
    return ret;
}

static PyObject *
make_site_row(const tsk_site_t *site)
{
    PyObject *ret = NULL;
    PyObject *metadata = NULL;

    metadata = make_metadata(site->metadata, (Py_ssize_t) site->metadata_length);
    if (metadata == NULL) {
        goto out;
    }
    ret = Py_BuildValue("ds#O", site->position, site->ancestral_state,
        (Py_ssize_t) site->ancestral_state_length, metadata);
out:
    Py_XDECREF(metadata);
    return ret;
}

static PyObject *
make_site_object(const tsk_site_t *site)
{
    PyObject *ret = NULL;
    PyObject *mutations = NULL;
    PyObject *metadata = NULL;

    metadata = make_metadata(site->metadata, (Py_ssize_t) site->metadata_length);
    if (metadata == NULL) {
        goto out;
    }
    mutations = make_mutation_id_list(site->mutations, site->mutations_length);
    if (mutations == NULL) {
        goto out;
    }
    /* TODO should reorder this tuple, as it's not very logical. */
    ret = Py_BuildValue("ds#OnO", site->position, site->ancestral_state,
        (Py_ssize_t) site->ancestral_state_length, mutations, (Py_ssize_t) site->id,
        metadata);
out:
    Py_XDECREF(mutations);
    Py_XDECREF(metadata);
    return ret;
}

static PyObject *
make_alleles(tsk_variant_t *variant)
{
    PyObject *ret = NULL;
    PyObject *item, *t;
    size_t j;

    t = PyTuple_New(variant->num_alleles + variant->has_missing_data);
    if (t == NULL) {
        goto out;
    }
    for (j = 0; j < variant->num_alleles; j++) {
        item = Py_BuildValue("s#", variant->alleles[j], variant->allele_lengths[j]);
        if (item == NULL) {
            Py_DECREF(t);
            goto out;
        }
        PyTuple_SET_ITEM(t, j, item);
    }
    if (variant->has_missing_data) {
        item = Py_BuildValue("");
        if (item == NULL) {
            Py_DECREF(t);
            goto out;
        }
        PyTuple_SET_ITEM(t, variant->num_alleles, item);
    }
    ret = t;
out:
    return ret;
}

static PyObject *
make_variant(tsk_variant_t *variant, size_t num_samples)
{
    PyObject *ret = NULL;
    npy_intp dims = num_samples;
    PyObject *alleles = make_alleles(variant);
    PyArrayObject *genotypes = (PyArrayObject *) PyArray_SimpleNew(1, &dims, NPY_INT8);

    /* TODO update this to account for 16 bit variants when we provide the
     * high-level interface. */
    if (genotypes == NULL || alleles == NULL) {
        goto out;
    }
    memcpy(PyArray_DATA(genotypes), variant->genotypes.i8, num_samples * sizeof(int8_t));
    ret = Py_BuildValue("iOO", variant->site->id, genotypes, alleles);
out:
    Py_XDECREF(genotypes);
    Py_XDECREF(alleles);
    return ret;
}

static PyObject *
convert_sites(const tsk_site_t *sites, size_t num_sites)
{
    PyObject *ret = NULL;
    PyObject *l = NULL;
    PyObject *py_site = NULL;
    size_t j;

    l = PyList_New(num_sites);
    if (l == NULL) {
        goto out;
    }
    for (j = 0; j < num_sites; j++) {
        py_site = make_site_object(&sites[j]);
        if (py_site == NULL) {
            Py_DECREF(l);
            goto out;
        }
        PyList_SET_ITEM(l, j, py_site);
    }
    ret = l;
out:
    return ret;
}

static PyObject *
convert_transitions(tsk_state_transition_t *transitions, size_t num_transitions)
{
    PyObject *ret = NULL;
    PyObject *l = NULL;
    PyObject *py_transition = NULL;
    size_t j;

    l = PyList_New(num_transitions);
    if (l == NULL) {
        goto out;
    }
    for (j = 0; j < num_transitions; j++) {
        py_transition = Py_BuildValue(
            "iii", transitions[j].node, transitions[j].parent, transitions[j].state);
        if (py_transition == NULL) {
            Py_DECREF(l);
            goto out;
        }
        PyList_SET_ITEM(l, j, py_transition);
    }
    ret = l;
out:
    return ret;
}

/* TODO: this should really be a dict we're returning */
static PyObject *
convert_compressed_matrix_site(tsk_compressed_matrix_t *matrix, unsigned int site)
{
    PyObject *ret = NULL;
    PyObject *list = NULL;
    PyObject *item = NULL;
    size_t j, num_values;

    if (site >= matrix->num_sites) {
        PyErr_SetString(PyExc_ValueError, "Site index out of bounds");
        goto out;
    }

    num_values = matrix->num_transitions[site];
    list = PyList_New(num_values);
    if (list == NULL) {
        goto out;
    }
    for (j = 0; j < num_values; j++) {
        item = Py_BuildValue("id", matrix->nodes[site][j], matrix->values[site][j]);
        if (item == NULL) {
            goto out;
        }
        PyList_SET_ITEM(list, j, item);
        item = NULL;
    }
    ret = list;
    list = NULL;
out:
    Py_XDECREF(item);
    Py_XDECREF(list);
    return ret;
}

static PyObject *
decode_compressed_matrix(tsk_compressed_matrix_t *matrix)
{
    int err;
    PyObject *ret = NULL;
    PyArrayObject *decoded = NULL;
    npy_intp dims[2];

    dims[0] = tsk_treeseq_get_num_sites(matrix->tree_sequence);
    dims[1] = tsk_treeseq_get_num_samples(matrix->tree_sequence);
    decoded = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT64);
    if (decoded == NULL) {
        goto out;
    }
    err = tsk_compressed_matrix_decode(matrix, PyArray_DATA(decoded));
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = (PyObject *) decoded;
    decoded = NULL;
out:
    Py_XDECREF(decoded);
    return ret;
}

static const char **
parse_allele_list(PyObject *allele_tuple)
{
    const char **ret = NULL;
    const char **alleles = NULL;
    PyObject *str;
    Py_ssize_t j, num_alleles;

    if (!PyTuple_Check(allele_tuple)) {
        PyErr_SetString(PyExc_TypeError, "Fixed allele list must be a tuple");
        goto out;
    }

    num_alleles = PyTuple_Size(allele_tuple);
    if (num_alleles == 0) {
        PyErr_SetString(PyExc_ValueError, "Must specify at least one allele");
        goto out;
    }
    /* Leave space for the sentinel, and initialise to NULL */
    alleles = PyMem_Calloc(num_alleles + 1, sizeof(*alleles));
    if (alleles == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    for (j = 0; j < num_alleles; j++) {
        str = PyTuple_GetItem(allele_tuple, j);
        if (str == NULL) {
            goto out;
        }
        if (!PyUnicode_Check(str)) {
            PyErr_SetString(PyExc_TypeError, "alleles must be strings");
            goto out;
        }
        /* PyUnicode_AsUTF8AndSize caches the UTF8 representation of the string
         * within the object, and we're not responsible for freeing it. Thus,
         * once we're sure the string object stays alive for the lifetime of the
         * returned string, we can be sure it's safe. These strings are immediately
         * copied during tsk_vargen_init, so the operation is safe.
         */
        alleles[j] = PyUnicode_AsUTF8AndSize(str, NULL);
        if (alleles[j] == NULL) {
            goto out;
        }
    }
    ret = alleles;
    alleles = NULL;
out:
    PyMem_Free(alleles);
    return ret;
}

static PyObject *
table_get_column_array(size_t num_rows, void *data, int npy_type, size_t element_size)
{
    PyObject *ret = NULL;
    PyArrayObject *array;
    npy_intp dims = (npy_intp) num_rows;

    array = (PyArrayObject *) PyArray_EMPTY(1, &dims, npy_type, 0);
    if (array == NULL) {
        goto out;
    }
    memcpy(PyArray_DATA(array), data, num_rows * element_size);
    ret = (PyObject *) array;
out:
    return ret;
}

static FILE *
make_file(PyObject *fileobj, const char *mode)
{
    FILE *ret = NULL;
    FILE *file = NULL;
    int fileobj_fd, new_fd;

    fileobj_fd = PyObject_AsFileDescriptor(fileobj);
    if (fileobj_fd == -1) {
        goto out;
    }
    new_fd = dup(fileobj_fd);
    if (new_fd == -1) {
        PyErr_SetFromErrno(PyExc_OSError);
        goto out;
    }
    file = fdopen(new_fd, mode);
    if (file == NULL) {
        (void) close(new_fd);
        PyErr_SetFromErrno(PyExc_OSError);
        goto out;
    }
    ret = file;
out:
    return ret;
}

static int
uint32_converter(PyObject *py_obj, uint32_t *uint_out)
{
    long long temp_long;
    int ret = 0;

    if (!PyArg_Parse(py_obj, "L", &temp_long)) {
        goto out;
    }
    if (temp_long > UINT32_MAX) {
        PyErr_SetString(PyExc_OverflowError, "unsigned int32 >= than 2^32");
        goto out;
    }
    if (temp_long < 0) {
        PyErr_SetString(
            PyExc_ValueError, "Can't convert negative value to unsigned int");
        goto out;
    }

    uint_out[0] = (uint32_t) temp_long;
    ret = 1;
out:
    return ret;
}

/*===================================================================
 * IndividualTable
 *===================================================================
 */

static int
IndividualTable_check_state(IndividualTable *self)
{
    int ret = -1;
    if (self->table == NULL) {
        PyErr_SetString(PyExc_SystemError, "IndividualTable not initialised");
        goto out;
    }
    if (self->locked) {
        PyErr_SetString(PyExc_RuntimeError, "IndividualTable in use by other thread.");
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static void
IndividualTable_dealloc(IndividualTable *self)
{
    if (self->tables != NULL) {
        Py_DECREF(self->tables);
    } else if (self->table != NULL) {
        tsk_individual_table_free(self->table);
        PyMem_Free(self->table);
        self->table = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static int
IndividualTable_init(IndividualTable *self, PyObject *args, PyObject *kwds)
{
    int ret = -1;
    int err;
    static char *kwlist[] = { "max_rows_increment", NULL };
    Py_ssize_t max_rows_increment = 0;

    self->table = NULL;
    self->locked = false;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|n", kwlist, &max_rows_increment)) {
        goto out;
    }
    if (max_rows_increment < 0) {
        PyErr_SetString(PyExc_ValueError, "max_rows_increment must be positive");
        goto out;
    }
    self->table = PyMem_Malloc(sizeof(tsk_individual_table_t));
    if (self->table == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    err = tsk_individual_table_init(self->table, 0);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    tsk_individual_table_set_max_rows_increment(self->table, max_rows_increment);
    ret = 0;
out:
    return ret;
}

static PyObject *
IndividualTable_add_row(IndividualTable *self, PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    int err;
    unsigned int flags = 0;
    PyObject *py_metadata = Py_None;
    PyObject *py_location = Py_None;
    PyArrayObject *location_array = NULL;
    double *location_data = NULL;
    tsk_size_t location_length = 0;
    char *metadata = "";
    Py_ssize_t metadata_length = 0;
    npy_intp *shape;
    static char *kwlist[] = { "flags", "location", "metadata", NULL };

    if (IndividualTable_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&OO", kwlist, &uint32_converter,
            &flags, &py_location, &py_metadata)) {
        goto out;
    }
    if (py_metadata != Py_None) {
        if (PyBytes_AsStringAndSize(py_metadata, &metadata, &metadata_length) < 0) {
            goto out;
        }
    }
    if (py_location != Py_None) {
        /* This ensures that only 1D arrays are accepted. */
        location_array = (PyArrayObject *) PyArray_FromAny(py_location,
            PyArray_DescrFromType(NPY_FLOAT64), 1, 1, NPY_ARRAY_IN_ARRAY, NULL);
        if (location_array == NULL) {
            goto out;
        }
        shape = PyArray_DIMS(location_array);
        location_length = (tsk_size_t) shape[0];
        location_data = PyArray_DATA(location_array);
    }
    err = tsk_individual_table_add_row(self->table, (tsk_flags_t) flags, location_data,
        location_length, metadata, metadata_length);
    if (err < 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("i", err);
out:
    Py_XDECREF(location_array);
    return ret;
}

/* Forward declaration */
static PyTypeObject IndividualTableType;

static PyObject *
IndividualTable_equals(IndividualTable *self, PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    IndividualTable *other = NULL;
    tsk_flags_t options = 0;
    int ignore_metadata = false;
    static char *kwlist[] = { "other", "ignore_metadata", NULL };

    if (IndividualTable_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|i", kwlist, &IndividualTableType,
            &other, &ignore_metadata)) {
        goto out;
    }
    if (IndividualTable_check_state(other) != 0) {
        goto out;
    }
    if (ignore_metadata) {
        options |= TSK_CMP_IGNORE_METADATA;
    }
    ret = Py_BuildValue(
        "i", tsk_individual_table_equals(self->table, other->table, options));
out:
    return ret;
}

static PyObject *
IndividualTable_get_row(IndividualTable *self, PyObject *args)
{
    PyObject *ret = NULL;
    int err;
    Py_ssize_t row_id;
    tsk_individual_t individual;

    if (IndividualTable_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "n", &row_id)) {
        goto out;
    }
    err = tsk_individual_table_get_row(self->table, (tsk_id_t) row_id, &individual);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = make_individual_row(&individual);
out:
    return ret;
}

static PyObject *
IndividualTable_parse_dict_arg(IndividualTable *self, PyObject *args, bool clear_table)
{
    int err;
    PyObject *ret = NULL;
    PyObject *dict = NULL;

    if (IndividualTable_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "O!", &PyDict_Type, &dict)) {
        goto out;
    }
    err = parse_individual_table_dict(self->table, dict, clear_table);
    if (err != 0) {
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
IndividualTable_append_columns(IndividualTable *self, PyObject *args)
{
    return IndividualTable_parse_dict_arg(self, args, false);
}

static PyObject *
IndividualTable_set_columns(IndividualTable *self, PyObject *args)
{
    return IndividualTable_parse_dict_arg(self, args, true);
}

static PyObject *
IndividualTable_clear(IndividualTable *self)
{
    PyObject *ret = NULL;
    int err;

    if (IndividualTable_check_state(self) != 0) {
        goto out;
    }
    err = tsk_individual_table_clear(self->table);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
IndividualTable_truncate(IndividualTable *self, PyObject *args)
{
    PyObject *ret = NULL;
    Py_ssize_t num_rows;
    int err;

    if (IndividualTable_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "n", &num_rows)) {
        goto out;
    }
    if (num_rows < 0 || num_rows > (Py_ssize_t) self->table->num_rows) {
        PyErr_SetString(PyExc_ValueError, "num_rows out of bounds");
        goto out;
    }
    err = tsk_individual_table_truncate(self->table, (tsk_size_t) num_rows);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
IndividualTable_get_max_rows_increment(IndividualTable *self, void *closure)
{
    PyObject *ret = NULL;
    if (IndividualTable_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("n", (Py_ssize_t) self->table->max_rows_increment);
out:
    return ret;
}

static PyObject *
IndividualTable_get_num_rows(IndividualTable *self, void *closure)
{
    PyObject *ret = NULL;
    if (IndividualTable_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("n", (Py_ssize_t) self->table->num_rows);
out:
    return ret;
}

static PyObject *
IndividualTable_get_max_rows(IndividualTable *self, void *closure)
{
    PyObject *ret = NULL;
    if (IndividualTable_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("n", (Py_ssize_t) self->table->max_rows);
out:
    return ret;
}

static PyObject *
IndividualTable_get_flags(IndividualTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (IndividualTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(
        self->table->num_rows, self->table->flags, NPY_UINT32, sizeof(uint32_t));
out:
    return ret;
}

static PyObject *
IndividualTable_get_location(IndividualTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (IndividualTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(self->table->location_length, self->table->location,
        NPY_FLOAT64, sizeof(double));
out:
    return ret;
}

static PyObject *
IndividualTable_get_location_offset(IndividualTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (IndividualTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(self->table->num_rows + 1, self->table->location_offset,
        NPY_UINT32, sizeof(uint32_t));
out:
    return ret;
}

static PyObject *
IndividualTable_get_metadata(IndividualTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (IndividualTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(
        self->table->metadata_length, self->table->metadata, NPY_INT8, sizeof(char));
out:
    return ret;
}

static PyObject *
IndividualTable_get_metadata_offset(IndividualTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (IndividualTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(self->table->num_rows + 1, self->table->metadata_offset,
        NPY_UINT32, sizeof(uint32_t));
out:
    return ret;
}

static PyObject *
IndividualTable_get_metadata_schema(IndividualTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (IndividualTable_check_state(self) != 0) {
        goto out;
    }
    ret = make_Py_Unicode_FromStringAndLength(
        self->table->metadata_schema, self->table->metadata_schema_length);
out:
    return ret;
}

static int
IndividualTable_set_metadata_schema(IndividualTable *self, PyObject *arg, void *closure)
{
    int ret = -1;
    int err;
    const char *metadata_schema;
    Py_ssize_t metadata_schema_length;

    if (IndividualTable_check_state(self) != 0) {
        goto out;
    }
    metadata_schema = parse_metadata_schema_arg(arg, &metadata_schema_length);
    if (metadata_schema == NULL) {
        goto out;
    }
    err = tsk_individual_table_set_metadata_schema(
        self->table, metadata_schema, metadata_schema_length);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static PyGetSetDef IndividualTable_getsetters[] = {
    { .name = "max_rows_increment",
        .get = (getter) IndividualTable_get_max_rows_increment,
        .doc = "The size increment" },
    { .name = "num_rows",
        .get = (getter) IndividualTable_get_num_rows,
        .doc = "The number of rows in the table." },
    { .name = "max_rows",
        .get = (getter) IndividualTable_get_max_rows,
        .doc = "The current maximum number of rows in the table." },
    { .name = "flags",
        .get = (getter) IndividualTable_get_flags,
        .doc = "The flags array" },
    { .name = "location",
        .get = (getter) IndividualTable_get_location,
        .doc = "The location array" },
    { .name = "location_offset",
        .get = (getter) IndividualTable_get_location_offset,
        .doc = "The location offset array" },
    { .name = "metadata",
        .get = (getter) IndividualTable_get_metadata,
        .doc = "The metadata array" },
    { .name = "metadata_offset",
        .get = (getter) IndividualTable_get_metadata_offset,
        .doc = "The metadata offset array" },
    { .name = "metadata_schema",
        .get = (getter) IndividualTable_get_metadata_schema,
        .set = (setter) IndividualTable_set_metadata_schema,
        .doc = "The metadata schema" },
    { NULL } /* Sentinel */
};

static PyMethodDef IndividualTable_methods[] = {
    { .ml_name = "add_row",
        .ml_meth = (PyCFunction) IndividualTable_add_row,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Adds a new row to this table." },
    { .ml_name = "get_row",
        .ml_meth = (PyCFunction) IndividualTable_get_row,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns the kth row in this table." },
    { .ml_name = "equals",
        .ml_meth = (PyCFunction) IndividualTable_equals,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Returns true if the specified individual table is equal." },
    { .ml_name = "append_columns",
        .ml_meth = (PyCFunction) IndividualTable_append_columns,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Appends the data in the specified arrays into the columns." },
    { .ml_name = "set_columns",
        .ml_meth = (PyCFunction) IndividualTable_set_columns,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Copies the data in the specified arrays into the columns." },
    { .ml_name = "clear",
        .ml_meth = (PyCFunction) IndividualTable_clear,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Clears this table." },
    { .ml_name = "truncate",
        .ml_meth = (PyCFunction) IndividualTable_truncate,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Truncates this table to the specified number of rows." },
    { NULL } /* Sentinel */
};

static PyTypeObject IndividualTableType = {
    // clang-format off
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_tskit.IndividualTable",
    .tp_basicsize = sizeof(IndividualTable),
    .tp_dealloc = (destructor) IndividualTable_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "IndividualTable objects",
    .tp_methods = IndividualTable_methods,
    .tp_getset = IndividualTable_getsetters,
    .tp_init = (initproc) IndividualTable_init,
    .tp_new = PyType_GenericNew,
    // clang-format on
};

/*===================================================================
 * NodeTable
 *===================================================================
 */

static int
NodeTable_check_state(NodeTable *self)
{
    int ret = -1;
    if (self->table == NULL) {
        PyErr_SetString(PyExc_SystemError, "NodeTable not initialised");
        goto out;
    }
    if (self->locked) {
        PyErr_SetString(PyExc_RuntimeError, "NodeTable in use by other thread.");
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static void
NodeTable_dealloc(NodeTable *self)
{
    if (self->tables != NULL) {
        Py_DECREF(self->tables);
    } else if (self->table != NULL) {
        tsk_node_table_free(self->table);
        PyMem_Free(self->table);
        self->table = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static int
NodeTable_init(NodeTable *self, PyObject *args, PyObject *kwds)
{
    int ret = -1;
    int err;
    static char *kwlist[] = { "max_rows_increment", NULL };
    Py_ssize_t max_rows_increment = 0;

    self->table = NULL;
    self->locked = false;
    self->tables = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|n", kwlist, &max_rows_increment)) {
        goto out;
    }
    if (max_rows_increment < 0) {
        PyErr_SetString(PyExc_ValueError, "max_rows_increment must be positive");
        goto out;
    }
    self->table = PyMem_Malloc(sizeof(tsk_node_table_t));
    if (self->table == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    err = tsk_node_table_init(self->table, 0);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    tsk_node_table_set_max_rows_increment(self->table, max_rows_increment);
    ret = 0;
out:
    return ret;
}

static PyObject *
NodeTable_add_row(NodeTable *self, PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    int err;
    unsigned int flags = 0;
    double time = 0;
    int population = -1;
    int individual = -1;
    PyObject *py_metadata = Py_None;
    char *metadata = "";
    Py_ssize_t metadata_length = 0;
    static char *kwlist[]
        = { "flags", "time", "population", "individual", "metadata", NULL };

    if (NodeTable_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&diiO", kwlist, &uint32_converter,
            &flags, &time, &population, &individual, &py_metadata)) {
        goto out;
    }
    if (py_metadata != Py_None) {
        if (PyBytes_AsStringAndSize(py_metadata, &metadata, &metadata_length) < 0) {
            goto out;
        }
    }
    err = tsk_node_table_add_row(self->table, (tsk_flags_t) flags, time,
        (tsk_id_t) population, individual, metadata, metadata_length);
    if (err < 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("i", err);
out:
    return ret;
}

/* Forward declaration */
static PyTypeObject NodeTableType;

static PyObject *
NodeTable_equals(NodeTable *self, PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    NodeTable *other = NULL;
    tsk_flags_t options = 0;
    int ignore_metadata = false;
    static char *kwlist[] = { "other", "ignore_metadata", NULL };

    if (NodeTable_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "O!|i", kwlist, &NodeTableType, &other, &ignore_metadata)) {
        goto out;
    }
    if (NodeTable_check_state(other) != 0) {
        goto out;
    }
    if (ignore_metadata) {
        options |= TSK_CMP_IGNORE_METADATA;
    }
    ret = Py_BuildValue("i", tsk_node_table_equals(self->table, other->table, options));
out:
    return ret;
}

static PyObject *
NodeTable_get_row(NodeTable *self, PyObject *args)
{
    PyObject *ret = NULL;
    int err;
    Py_ssize_t row_id;
    tsk_node_t node;

    if (NodeTable_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "n", &row_id)) {
        goto out;
    }
    err = tsk_node_table_get_row(self->table, (tsk_id_t) row_id, &node);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = make_node(&node);
out:
    return ret;
}

static PyObject *
NodeTable_parse_dict_arg(NodeTable *self, PyObject *args, bool clear_table)
{
    int err;
    PyObject *ret = NULL;
    PyObject *dict = NULL;

    if (NodeTable_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "O!", &PyDict_Type, &dict)) {
        goto out;
    }
    err = parse_node_table_dict(self->table, dict, clear_table);
    if (err != 0) {
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
NodeTable_append_columns(NodeTable *self, PyObject *args)
{
    return NodeTable_parse_dict_arg(self, args, false);
}

static PyObject *
NodeTable_set_columns(NodeTable *self, PyObject *args)
{
    return NodeTable_parse_dict_arg(self, args, true);
}

static PyObject *
NodeTable_clear(NodeTable *self)
{
    PyObject *ret = NULL;
    int err;

    if (NodeTable_check_state(self) != 0) {
        goto out;
    }
    err = tsk_node_table_clear(self->table);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
NodeTable_truncate(NodeTable *self, PyObject *args)
{
    PyObject *ret = NULL;
    Py_ssize_t num_rows;
    int err;

    if (NodeTable_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "n", &num_rows)) {
        goto out;
    }
    if (num_rows < 0 || num_rows > (Py_ssize_t) self->table->num_rows) {
        PyErr_SetString(PyExc_ValueError, "num_rows out of bounds");
        goto out;
    }
    err = tsk_node_table_truncate(self->table, (tsk_size_t) num_rows);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
NodeTable_get_max_rows_increment(NodeTable *self, void *closure)
{
    PyObject *ret = NULL;
    if (NodeTable_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("n", (Py_ssize_t) self->table->max_rows_increment);
out:
    return ret;
}

static PyObject *
NodeTable_get_num_rows(NodeTable *self, void *closure)
{
    PyObject *ret = NULL;
    if (NodeTable_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("n", (Py_ssize_t) self->table->num_rows);
out:
    return ret;
}

static PyObject *
NodeTable_get_max_rows(NodeTable *self, void *closure)
{
    PyObject *ret = NULL;
    if (NodeTable_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("n", (Py_ssize_t) self->table->max_rows);
out:
    return ret;
}

static PyObject *
NodeTable_get_time(NodeTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (NodeTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(
        self->table->num_rows, self->table->time, NPY_FLOAT64, sizeof(double));
out:
    return ret;
}

static PyObject *
NodeTable_get_flags(NodeTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (NodeTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(
        self->table->num_rows, self->table->flags, NPY_UINT32, sizeof(uint32_t));
out:
    return ret;
}

static PyObject *
NodeTable_get_population(NodeTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (NodeTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(
        self->table->num_rows, self->table->population, NPY_INT32, sizeof(int32_t));
out:
    return ret;
}

static PyObject *
NodeTable_get_individual(NodeTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (NodeTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(
        self->table->num_rows, self->table->individual, NPY_INT32, sizeof(int32_t));
out:
    return ret;
}

static PyObject *
NodeTable_get_metadata(NodeTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (NodeTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(
        self->table->metadata_length, self->table->metadata, NPY_INT8, sizeof(char));
out:
    return ret;
}

static PyObject *
NodeTable_get_metadata_offset(NodeTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (NodeTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(self->table->num_rows + 1, self->table->metadata_offset,
        NPY_UINT32, sizeof(uint32_t));
out:
    return ret;
}

static PyObject *
NodeTable_get_metadata_schema(NodeTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (NodeTable_check_state(self) != 0) {
        goto out;
    }
    ret = make_Py_Unicode_FromStringAndLength(
        self->table->metadata_schema, self->table->metadata_schema_length);
out:
    return ret;
}

static int
NodeTable_set_metadata_schema(NodeTable *self, PyObject *arg, void *closure)
{
    int ret = -1;
    int err;
    const char *metadata_schema;
    Py_ssize_t metadata_schema_length;

    if (NodeTable_check_state(self) != 0) {
        goto out;
    }
    metadata_schema = parse_metadata_schema_arg(arg, &metadata_schema_length);
    if (metadata_schema == NULL) {
        goto out;
    }
    err = tsk_node_table_set_metadata_schema(
        self->table, metadata_schema, metadata_schema_length);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static PyGetSetDef NodeTable_getsetters[] = {
    { .name = "max_rows_increment",
        .get = (getter) NodeTable_get_max_rows_increment,
        .doc = "The size increment" },
    { .name = "num_rows",
        .get = (getter) NodeTable_get_num_rows,
        .doc = "The number of rows in the table." },
    { .name = "max_rows",
        .get = (getter) NodeTable_get_max_rows,
        .doc = "The current maximum number of rows in the table." },
    { .name = "time", .get = (getter) NodeTable_get_time, .doc = "The time array" },
    { .name = "flags", .get = (getter) NodeTable_get_flags, .doc = "The flags array" },
    { .name = "population",
        .get = (getter) NodeTable_get_population,
        .doc = "The population array" },
    { .name = "individual",
        .get = (getter) NodeTable_get_individual,
        .doc = "The individual array" },
    { .name = "metadata",
        .get = (getter) NodeTable_get_metadata,
        .doc = "The metadata array" },
    { .name = "metadata_offset",
        .get = (getter) NodeTable_get_metadata_offset,
        .doc = "The metadata offset array" },
    { .name = "metadata_schema",
        .get = (getter) NodeTable_get_metadata_schema,
        .set = (setter) NodeTable_set_metadata_schema,
        .doc = "The metadata schema" },
    { NULL } /* Sentinel */
};

static PyMethodDef NodeTable_methods[] = {
    { .ml_name = "add_row",
        .ml_meth = (PyCFunction) NodeTable_add_row,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Adds a new row to this table." },
    { .ml_name = "equals",
        .ml_meth = (PyCFunction) NodeTable_equals,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Returns True if the specified NodeTable is equal to this one." },
    { .ml_name = "get_row",
        .ml_meth = (PyCFunction) NodeTable_get_row,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns the kth row in this table." },
    { .ml_name = "append_columns",
        .ml_meth = (PyCFunction) NodeTable_append_columns,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Appends the data in the specified arrays into the columns." },
    { .ml_name = "set_columns",
        .ml_meth = (PyCFunction) NodeTable_set_columns,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Copies the data in the specified arrays into the columns." },
    { .ml_name = "clear",
        .ml_meth = (PyCFunction) NodeTable_clear,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Clears this table." },
    { .ml_name = "truncate",
        .ml_meth = (PyCFunction) NodeTable_truncate,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Truncates this table to the specified number of rows." },
    { NULL } /* Sentinel */
};

static PyTypeObject NodeTableType = {
    // clang-format off
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_tskit.NodeTable",
    .tp_basicsize = sizeof(NodeTable),
    .tp_dealloc = (destructor) NodeTable_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "NodeTable objects",
    .tp_methods = NodeTable_methods,
    .tp_getset = NodeTable_getsetters,
    .tp_init = (initproc) NodeTable_init,
    .tp_new = PyType_GenericNew,
    // clang-format on
};

/*===================================================================
 * EdgeTable
 *===================================================================
 */

static int
EdgeTable_check_state(EdgeTable *self)
{
    int ret = -1;
    if (self->table == NULL) {
        PyErr_SetString(PyExc_SystemError, "EdgeTable not initialised");
        goto out;
    }
    if (self->locked) {
        PyErr_SetString(PyExc_RuntimeError, "EdgeTable in use by other thread.");
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static void
EdgeTable_dealloc(EdgeTable *self)
{
    if (self->tables != NULL) {
        Py_DECREF(self->tables);
    } else if (self->table != NULL) {
        tsk_edge_table_free(self->table);
        PyMem_Free(self->table);
        self->table = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static int
EdgeTable_init(EdgeTable *self, PyObject *args, PyObject *kwds)
{
    int ret = -1;
    int err;
    static char *kwlist[] = { "max_rows_increment", NULL };
    Py_ssize_t max_rows_increment = 0;

    self->table = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|n", kwlist, &max_rows_increment)) {
        goto out;
    }
    if (max_rows_increment < 0) {
        PyErr_SetString(PyExc_ValueError, "max_rows_increment must be positive");
        goto out;
    }
    self->table = PyMem_Malloc(sizeof(tsk_edge_table_t));
    if (self->table == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    err = tsk_edge_table_init(self->table, 0);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    tsk_edge_table_set_max_rows_increment(self->table, max_rows_increment);
    ret = 0;
out:
    return ret;
}

static PyObject *
EdgeTable_add_row(EdgeTable *self, PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    int err;
    double left = 0.0;
    double right = 1.0;
    int parent;
    int child;
    PyObject *py_metadata = Py_None;
    char *metadata = "";
    Py_ssize_t metadata_length = 0;
    static char *kwlist[] = { "left", "right", "parent", "child", "metadata", NULL };

    if (EdgeTable_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ddii|O", kwlist, &left, &right,
            &parent, &child, &py_metadata)) {
        goto out;
    }
    if (py_metadata != Py_None) {
        if (PyBytes_AsStringAndSize(py_metadata, &metadata, &metadata_length) < 0) {
            goto out;
        }
    }
    err = tsk_edge_table_add_row(
        self->table, left, right, parent, child, metadata, metadata_length);
    if (err < 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("i", err);
out:
    return ret;
}

/* Forward declaration */
static PyTypeObject EdgeTableType;

static PyObject *
EdgeTable_equals(EdgeTable *self, PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    EdgeTable *other = NULL;
    tsk_flags_t options = 0;
    int ignore_metadata = false;
    static char *kwlist[] = { "other", "ignore_metadata", NULL };

    if (EdgeTable_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "O!|i", kwlist, &EdgeTableType, &other, &ignore_metadata)) {
        goto out;
    }
    if (EdgeTable_check_state(other) != 0) {
        goto out;
    }
    if (ignore_metadata) {
        options |= TSK_CMP_IGNORE_METADATA;
    }
    ret = Py_BuildValue("i", tsk_edge_table_equals(self->table, other->table, options));
out:
    return ret;
}

static PyObject *
EdgeTable_get_row(EdgeTable *self, PyObject *args)
{
    PyObject *ret = NULL;
    Py_ssize_t row_id;
    int err;
    tsk_edge_t edge;

    if (EdgeTable_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "n", &row_id)) {
        goto out;
    }
    err = tsk_edge_table_get_row(self->table, (tsk_id_t) row_id, &edge);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = make_edge(&edge, false);
out:
    return ret;
}

static PyObject *
EdgeTable_parse_dict_arg(EdgeTable *self, PyObject *args, bool clear_table)
{
    int err;
    PyObject *ret = NULL;
    PyObject *dict = NULL;

    if (EdgeTable_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "O!", &PyDict_Type, &dict)) {
        goto out;
    }
    err = parse_edge_table_dict(self->table, dict, clear_table);
    if (err != 0) {
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
EdgeTable_append_columns(EdgeTable *self, PyObject *args)
{
    return EdgeTable_parse_dict_arg(self, args, false);
}

static PyObject *
EdgeTable_set_columns(EdgeTable *self, PyObject *args)
{
    return EdgeTable_parse_dict_arg(self, args, true);
}

static PyObject *
EdgeTable_clear(EdgeTable *self)
{
    PyObject *ret = NULL;
    int err;

    if (EdgeTable_check_state(self) != 0) {
        goto out;
    }
    err = tsk_edge_table_clear(self->table);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
EdgeTable_truncate(EdgeTable *self, PyObject *args)
{
    PyObject *ret = NULL;
    Py_ssize_t num_rows;
    int err;

    if (EdgeTable_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "n", &num_rows)) {
        goto out;
    }
    if (num_rows < 0 || num_rows > (Py_ssize_t) self->table->num_rows) {
        PyErr_SetString(PyExc_ValueError, "num_rows out of bounds");
        goto out;
    }
    err = tsk_edge_table_truncate(self->table, (tsk_size_t) num_rows);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
EdgeTable_squash(EdgeTable *self)
{
    PyObject *ret = NULL;
    int err;

    if (EdgeTable_check_state(self) != 0) {
        goto out;
    }
    err = tsk_edge_table_squash(self->table);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
EdgeTable_get_max_rows_increment(EdgeTable *self, void *closure)
{
    PyObject *ret = NULL;
    if (EdgeTable_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("n", (Py_ssize_t) self->table->max_rows_increment);
out:
    return ret;
}

static PyObject *
EdgeTable_get_num_rows(EdgeTable *self, void *closure)
{
    PyObject *ret = NULL;
    if (EdgeTable_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("n", (Py_ssize_t) self->table->num_rows);
out:
    return ret;
}

static PyObject *
EdgeTable_get_max_rows(EdgeTable *self, void *closure)
{
    PyObject *ret = NULL;
    if (EdgeTable_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("n", (Py_ssize_t) self->table->max_rows);
out:
    return ret;
}

static PyObject *
EdgeTable_get_left(EdgeTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (EdgeTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(
        self->table->num_rows, self->table->left, NPY_FLOAT64, sizeof(double));
out:
    return ret;
}

static PyObject *
EdgeTable_get_right(EdgeTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (EdgeTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(
        self->table->num_rows, self->table->right, NPY_FLOAT64, sizeof(double));
out:
    return ret;
}

static PyObject *
EdgeTable_get_parent(EdgeTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (EdgeTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(
        self->table->num_rows, self->table->parent, NPY_INT32, sizeof(int32_t));
out:
    return ret;
}

static PyObject *
EdgeTable_get_child(EdgeTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (EdgeTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(
        self->table->num_rows, self->table->child, NPY_INT32, sizeof(int32_t));
out:
    return ret;
}

static PyObject *
EdgeTable_get_metadata(EdgeTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (EdgeTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(
        self->table->metadata_length, self->table->metadata, NPY_INT8, sizeof(char));
out:
    return ret;
}

static PyObject *
EdgeTable_get_metadata_offset(EdgeTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (EdgeTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(self->table->num_rows + 1, self->table->metadata_offset,
        NPY_UINT32, sizeof(uint32_t));
out:
    return ret;
}

static PyObject *
EdgeTable_get_metadata_schema(EdgeTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (EdgeTable_check_state(self) != 0) {
        goto out;
    }
    ret = make_Py_Unicode_FromStringAndLength(
        self->table->metadata_schema, self->table->metadata_schema_length);
out:
    return ret;
}

static int
EdgeTable_set_metadata_schema(EdgeTable *self, PyObject *arg, void *closure)
{
    int ret = -1;
    int err;
    const char *metadata_schema;
    Py_ssize_t metadata_schema_length;

    if (EdgeTable_check_state(self) != 0) {
        goto out;
    }
    metadata_schema = parse_metadata_schema_arg(arg, &metadata_schema_length);
    if (metadata_schema == NULL) {
        goto out;
    }
    err = tsk_edge_table_set_metadata_schema(
        self->table, metadata_schema, metadata_schema_length);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static PyGetSetDef EdgeTable_getsetters[] = {
    { .name = "max_rows_increment",
        .get = (getter) EdgeTable_get_max_rows_increment,
        .doc = "The size increment" },
    { .name = "num_rows",
        .get = (getter) EdgeTable_get_num_rows,
        .doc = "The number of rows in the table." },
    { .name = "max_rows",
        .get = (getter) EdgeTable_get_max_rows,
        .doc = "The current maximum number of rows in the table." },
    { .name = "left", .get = (getter) EdgeTable_get_left, .doc = "The left array" },
    { .name = "right", .get = (getter) EdgeTable_get_right, .doc = "The right array" },
    { .name = "parent",
        .get = (getter) EdgeTable_get_parent,
        .doc = "The parent array" },
    { .name = "child", .get = (getter) EdgeTable_get_child, .doc = "The child array" },
    { .name = "metadata",
        .get = (getter) EdgeTable_get_metadata,
        .doc = "The metadata array" },
    { .name = "metadata_offset",
        .get = (getter) EdgeTable_get_metadata_offset,
        .doc = "The metadata offset array" },
    { .name = "metadata_schema",
        .get = (getter) EdgeTable_get_metadata_schema,
        .set = (setter) EdgeTable_set_metadata_schema,
        .doc = "The metadata schema" },
    { NULL } /* Sentinel */
};

static PyMethodDef EdgeTable_methods[] = {
    { .ml_name = "add_row",
        .ml_meth = (PyCFunction) EdgeTable_add_row,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Adds a new row to this table." },
    { .ml_name = "equals",
        .ml_meth = (PyCFunction) EdgeTable_equals,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Returns True if the specified EdgeTable is equal to this one." },
    { .ml_name = "get_row",
        .ml_meth = (PyCFunction) EdgeTable_get_row,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns the kth row in this table." },
    { .ml_name = "set_columns",
        .ml_meth = (PyCFunction) EdgeTable_set_columns,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Copies the data in the specified arrays into the columns." },
    { .ml_name = "append_columns",
        .ml_meth = (PyCFunction) EdgeTable_append_columns,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Copies the data in the specified arrays into the columns." },
    { .ml_name = "clear",
        .ml_meth = (PyCFunction) EdgeTable_clear,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Clears this table." },
    { .ml_name = "truncate",
        .ml_meth = (PyCFunction) EdgeTable_truncate,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Truncates this table to the specified number of rows." },
    { .ml_name = "squash",
        .ml_meth = (PyCFunction) EdgeTable_squash,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Squashes sets of edges with adjacent L,R and identical P,C values." },
    { NULL } /* Sentinel */
};

static PyTypeObject EdgeTableType = {
    // clang-format off
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_tskit.EdgeTable",
    .tp_basicsize = sizeof(EdgeTable),
    .tp_dealloc = (destructor) EdgeTable_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "EdgeTable objects",
    .tp_methods = EdgeTable_methods,
    .tp_getset = EdgeTable_getsetters,
    .tp_init = (initproc) EdgeTable_init,
    .tp_new = PyType_GenericNew,
    // clang-format on
};

/*===================================================================
 * MigrationTable
 *===================================================================
 */

static int
MigrationTable_check_state(MigrationTable *self)
{
    int ret = -1;
    if (self->table == NULL) {
        PyErr_SetString(PyExc_SystemError, "MigrationTable not initialised");
        goto out;
    }
    if (self->locked) {
        PyErr_SetString(PyExc_RuntimeError, "MigrationTable in use by other thread.");
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static void
MigrationTable_dealloc(MigrationTable *self)
{
    if (self->tables != NULL) {
        Py_DECREF(self->tables);
    } else if (self->table != NULL) {
        tsk_migration_table_free(self->table);
        PyMem_Free(self->table);
        self->table = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static int
MigrationTable_init(MigrationTable *self, PyObject *args, PyObject *kwds)
{
    int ret = -1;
    int err;
    static char *kwlist[] = { "max_rows_increment", NULL };
    Py_ssize_t max_rows_increment = 0;

    self->table = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|n", kwlist, &max_rows_increment)) {
        goto out;
    }
    if (max_rows_increment < 0) {
        PyErr_SetString(PyExc_ValueError, "max_rows_increment must be positive");
        goto out;
    }
    self->table = PyMem_Malloc(sizeof(tsk_migration_table_t));
    if (self->table == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    err = tsk_migration_table_init(self->table, 0);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    tsk_migration_table_set_max_rows_increment(self->table, max_rows_increment);
    ret = 0;
out:
    return ret;
}

static PyObject *
MigrationTable_add_row(MigrationTable *self, PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    int err;
    double left, right, time;
    int node, source, dest;
    PyObject *py_metadata = Py_None;
    char *metadata = "";
    Py_ssize_t metadata_length = 0;
    static char *kwlist[]
        = { "left", "right", "node", "source", "dest", "time", "metadata", NULL };

    if (MigrationTable_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ddiiid|O", kwlist, &left, &right,
            &node, &source, &dest, &time, &py_metadata)) {
        goto out;
    }
    if (py_metadata != Py_None) {
        if (PyBytes_AsStringAndSize(py_metadata, &metadata, &metadata_length) < 0) {
            goto out;
        }
    }
    err = tsk_migration_table_add_row(
        self->table, left, right, node, source, dest, time, metadata, metadata_length);
    if (err < 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("i", err);
out:
    return ret;
}

/* Forward declaration */
static PyTypeObject MigrationTableType;

static PyObject *
MigrationTable_equals(MigrationTable *self, PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    MigrationTable *other = NULL;
    tsk_flags_t options = 0;
    int ignore_metadata = false;
    static char *kwlist[] = { "other", "ignore_metadata", NULL };

    if (MigrationTable_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "O!|i", kwlist, &MigrationTableType, &other, &ignore_metadata)) {
        goto out;
    }
    if (MigrationTable_check_state(other) != 0) {
        goto out;
    }
    if (ignore_metadata) {
        options |= TSK_CMP_IGNORE_METADATA;
    }
    ret = Py_BuildValue(
        "i", tsk_migration_table_equals(self->table, other->table, options));
out:
    return ret;
}

static PyObject *
MigrationTable_get_row(MigrationTable *self, PyObject *args)
{
    PyObject *ret = NULL;
    Py_ssize_t row_id;
    int err;
    tsk_migration_t migration;

    if (MigrationTable_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "n", &row_id)) {
        goto out;
    }
    err = tsk_migration_table_get_row(self->table, (tsk_id_t) row_id, &migration);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = make_migration(&migration);
out:
    return ret;
}

static PyObject *
MigrationTable_parse_dict_arg(MigrationTable *self, PyObject *args, bool clear_table)
{
    int err;
    PyObject *ret = NULL;
    PyObject *dict = NULL;

    if (MigrationTable_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "O!", &PyDict_Type, &dict)) {
        goto out;
    }
    err = parse_migration_table_dict(self->table, dict, clear_table);
    if (err != 0) {
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
MigrationTable_append_columns(MigrationTable *self, PyObject *args)
{
    return MigrationTable_parse_dict_arg(self, args, false);
}

static PyObject *
MigrationTable_set_columns(MigrationTable *self, PyObject *args)
{
    return MigrationTable_parse_dict_arg(self, args, true);
}

static PyObject *
MigrationTable_clear(MigrationTable *self)
{
    PyObject *ret = NULL;
    int err;

    if (MigrationTable_check_state(self) != 0) {
        goto out;
    }
    err = tsk_migration_table_clear(self->table);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
MigrationTable_truncate(MigrationTable *self, PyObject *args)
{
    PyObject *ret = NULL;
    Py_ssize_t num_rows;
    int err;

    if (MigrationTable_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "n", &num_rows)) {
        goto out;
    }
    if (num_rows < 0 || num_rows > (Py_ssize_t) self->table->num_rows) {
        PyErr_SetString(PyExc_ValueError, "num_rows out of bounds");
        goto out;
    }
    err = tsk_migration_table_truncate(self->table, (tsk_size_t) num_rows);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
MigrationTable_get_max_rows_increment(MigrationTable *self, void *closure)
{
    PyObject *ret = NULL;
    if (MigrationTable_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("n", (Py_ssize_t) self->table->max_rows_increment);
out:
    return ret;
}

static PyObject *
MigrationTable_get_num_rows(MigrationTable *self, void *closure)
{
    PyObject *ret = NULL;
    if (MigrationTable_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("n", (Py_ssize_t) self->table->num_rows);
out:
    return ret;
}

static PyObject *
MigrationTable_get_max_rows(MigrationTable *self, void *closure)
{
    PyObject *ret = NULL;
    if (MigrationTable_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("n", (Py_ssize_t) self->table->max_rows);
out:
    return ret;
}

static PyObject *
MigrationTable_get_left(MigrationTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (MigrationTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(
        self->table->num_rows, self->table->left, NPY_FLOAT64, sizeof(double));
out:
    return ret;
}

static PyObject *
MigrationTable_get_right(MigrationTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (MigrationTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(
        self->table->num_rows, self->table->right, NPY_FLOAT64, sizeof(double));
out:
    return ret;
}

static PyObject *
MigrationTable_get_time(MigrationTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (MigrationTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(
        self->table->num_rows, self->table->time, NPY_FLOAT64, sizeof(double));
out:
    return ret;
}

static PyObject *
MigrationTable_get_node(MigrationTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (MigrationTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(
        self->table->num_rows, self->table->node, NPY_INT32, sizeof(uint32_t));
out:
    return ret;
}

static PyObject *
MigrationTable_get_source(MigrationTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (MigrationTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(
        self->table->num_rows, self->table->source, NPY_INT32, sizeof(uint32_t));
out:
    return ret;
}

static PyObject *
MigrationTable_get_dest(MigrationTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (MigrationTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(
        self->table->num_rows, self->table->dest, NPY_INT32, sizeof(uint32_t));
out:
    return ret;
}

static PyObject *
MigrationTable_get_metadata(MigrationTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (MigrationTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(
        self->table->metadata_length, self->table->metadata, NPY_INT8, sizeof(char));
out:
    return ret;
}

static PyObject *
MigrationTable_get_metadata_offset(MigrationTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (MigrationTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(self->table->num_rows + 1, self->table->metadata_offset,
        NPY_UINT32, sizeof(uint32_t));
out:
    return ret;
}

static PyObject *
MigrationTable_get_metadata_schema(MigrationTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (MigrationTable_check_state(self) != 0) {
        goto out;
    }
    ret = make_Py_Unicode_FromStringAndLength(
        self->table->metadata_schema, self->table->metadata_schema_length);
out:
    return ret;
}

static int
MigrationTable_set_metadata_schema(MigrationTable *self, PyObject *arg, void *closure)
{
    int ret = -1;
    int err;
    const char *metadata_schema;
    Py_ssize_t metadata_schema_length;

    if (MigrationTable_check_state(self) != 0) {
        goto out;
    }
    metadata_schema = parse_metadata_schema_arg(arg, &metadata_schema_length);
    if (metadata_schema == NULL) {
        goto out;
    }
    err = tsk_migration_table_set_metadata_schema(
        self->table, metadata_schema, metadata_schema_length);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static PyGetSetDef MigrationTable_getsetters[] = {
    { .name = "max_rows_increment",
        .get = (getter) MigrationTable_get_max_rows_increment,
        .doc = "The size increment" },
    { .name = "num_rows",
        .get = (getter) MigrationTable_get_num_rows,
        .doc = "The number of rows in the table." },
    { .name = "max_rows",
        .get = (getter) MigrationTable_get_max_rows,
        .doc = "The current maximum number of rows in the table." },
    { .name = "left", .get = (getter) MigrationTable_get_left, .doc = "The left array" },
    { .name = "right",
        .get = (getter) MigrationTable_get_right,
        .doc = "The right array" },
    { .name = "node", .get = (getter) MigrationTable_get_node, .doc = "The node array" },
    { .name = "source",
        .get = (getter) MigrationTable_get_source,
        .doc = "The source array" },
    { .name = "dest", .get = (getter) MigrationTable_get_dest, .doc = "The dest array" },
    { .name = "time", .get = (getter) MigrationTable_get_time, .doc = "The time array" },
    { .name = "metadata",
        .get = (getter) MigrationTable_get_metadata,
        .doc = "The metadata array" },
    { .name = "metadata_offset",
        .get = (getter) MigrationTable_get_metadata_offset,
        .doc = "The metadata offset array" },
    { .name = "metadata_schema",
        .get = (getter) MigrationTable_get_metadata_schema,
        .set = (setter) MigrationTable_set_metadata_schema,
        .doc = "The metadata schema" },
    { NULL } /* Sentinel */
};

static PyMethodDef MigrationTable_methods[] = {
    { .ml_name = "add_row",
        .ml_meth = (PyCFunction) MigrationTable_add_row,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Adds a new row to this table." },
    { .ml_name = "equals",
        .ml_meth = (PyCFunction) MigrationTable_equals,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Returns True if the specified MigrationTable is equal to this one." },
    { .ml_name = "get_row",
        .ml_meth = (PyCFunction) MigrationTable_get_row,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns the kth row in this table." },
    { .ml_name = "set_columns",
        .ml_meth = (PyCFunction) MigrationTable_set_columns,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Copies the data in the specified arrays into the columns." },
    { .ml_name = "append_columns",
        .ml_meth = (PyCFunction) MigrationTable_append_columns,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Appends the data in the specified arrays into the columns." },
    { .ml_name = "clear",
        .ml_meth = (PyCFunction) MigrationTable_clear,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Clears this table." },
    { .ml_name = "truncate",
        .ml_meth = (PyCFunction) MigrationTable_truncate,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Truncates this table to the specified number of rows." },
    { NULL } /* Sentinel */
};

static PyTypeObject MigrationTableType = {
    // clang-format off
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_tskit.MigrationTable",
    .tp_basicsize = sizeof(MigrationTable),
    .tp_dealloc = (destructor) MigrationTable_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "MigrationTable objects",
    .tp_methods = MigrationTable_methods,
    .tp_getset = MigrationTable_getsetters,
    .tp_init = (initproc) MigrationTable_init,
    .tp_new = PyType_GenericNew,
    // clang-format on
};

/*===================================================================
 * SiteTable
 *===================================================================
 */

static int
SiteTable_check_state(SiteTable *self)
{
    int ret = -1;
    if (self->table == NULL) {
        PyErr_SetString(PyExc_SystemError, "SiteTable not initialised");
        goto out;
    }
    if (self->locked) {
        PyErr_SetString(PyExc_RuntimeError, "SiteTable in use by other thread.");
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static void
SiteTable_dealloc(SiteTable *self)
{
    if (self->tables != NULL) {
        Py_DECREF(self->tables);
    } else if (self->table != NULL) {
        tsk_site_table_free(self->table);
        PyMem_Free(self->table);
        self->table = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static int
SiteTable_init(SiteTable *self, PyObject *args, PyObject *kwds)
{
    int ret = -1;
    int err;
    static char *kwlist[] = { "max_rows_increment", NULL };
    Py_ssize_t max_rows_increment = 0;

    self->table = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|n", kwlist, &max_rows_increment)) {
        goto out;
    }
    if (max_rows_increment < 0) {
        PyErr_SetString(PyExc_ValueError, "max_rows_increment must be positive");
        goto out;
    }
    self->table = PyMem_Malloc(sizeof(tsk_site_table_t));
    if (self->table == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    err = tsk_site_table_init(self->table, 0);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    tsk_site_table_set_max_rows_increment(self->table, max_rows_increment);
    ret = 0;
out:
    return ret;
}

static PyObject *
SiteTable_add_row(SiteTable *self, PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    int err;
    double position;
    char *ancestral_state = NULL;
    Py_ssize_t ancestral_state_length = 0;
    PyObject *py_metadata = Py_None;
    char *metadata = NULL;
    Py_ssize_t metadata_length = 0;
    static char *kwlist[] = { "position", "ancestral_state", "metadata", NULL };

    if (SiteTable_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ds#|O", kwlist, &position,
            &ancestral_state, &ancestral_state_length, &py_metadata)) {
        goto out;
    }
    if (py_metadata != Py_None) {
        if (PyBytes_AsStringAndSize(py_metadata, &metadata, &metadata_length) < 0) {
            goto out;
        }
    }
    err = tsk_site_table_add_row(self->table, position, ancestral_state,
        ancestral_state_length, metadata, metadata_length);
    if (err < 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("i", err);
out:
    return ret;
}

/* Forward declaration */
static PyTypeObject SiteTableType;

static PyObject *
SiteTable_equals(SiteTable *self, PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    SiteTable *other = NULL;
    tsk_flags_t options = 0;
    int ignore_metadata = false;
    static char *kwlist[] = { "other", "ignore_metadata", NULL };

    if (SiteTable_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "O!|i", kwlist, &SiteTableType, &other, &ignore_metadata)) {
        goto out;
    }
    if (SiteTable_check_state(other) != 0) {
        goto out;
    }
    if (ignore_metadata) {
        options |= TSK_CMP_IGNORE_METADATA;
    }
    ret = Py_BuildValue("i", tsk_site_table_equals(self->table, other->table, options));
out:
    return ret;
}

static PyObject *
SiteTable_get_row(SiteTable *self, PyObject *args)
{
    PyObject *ret = NULL;
    Py_ssize_t row_id;
    int err;
    tsk_site_t site;

    if (SiteTable_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "n", &row_id)) {
        goto out;
    }
    err = tsk_site_table_get_row(self->table, (tsk_id_t) row_id, &site);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = make_site_row(&site);
out:
    return ret;
}

static PyObject *
SiteTable_parse_dict_arg(SiteTable *self, PyObject *args, bool clear_table)
{
    int err;
    PyObject *ret = NULL;
    PyObject *dict = NULL;

    if (SiteTable_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "O!", &PyDict_Type, &dict)) {
        goto out;
    }
    err = parse_site_table_dict(self->table, dict, clear_table);
    if (err != 0) {
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
SiteTable_append_columns(SiteTable *self, PyObject *args)
{
    return SiteTable_parse_dict_arg(self, args, false);
}

static PyObject *
SiteTable_set_columns(SiteTable *self, PyObject *args)
{
    return SiteTable_parse_dict_arg(self, args, true);
}

static PyObject *
SiteTable_clear(SiteTable *self)
{
    PyObject *ret = NULL;
    int err;

    if (SiteTable_check_state(self) != 0) {
        goto out;
    }
    err = tsk_site_table_clear(self->table);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
SiteTable_truncate(SiteTable *self, PyObject *args)
{
    PyObject *ret = NULL;
    Py_ssize_t num_rows;
    int err;

    if (SiteTable_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "n", &num_rows)) {
        goto out;
    }
    if (num_rows < 0 || num_rows > (Py_ssize_t) self->table->num_rows) {
        PyErr_SetString(PyExc_ValueError, "num_rows out of bounds");
        goto out;
    }
    err = tsk_site_table_truncate(self->table, (tsk_size_t) num_rows);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
SiteTable_get_max_rows_increment(SiteTable *self, void *closure)
{
    PyObject *ret = NULL;
    if (SiteTable_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("n", (Py_ssize_t) self->table->max_rows_increment);
out:
    return ret;
}

static PyObject *
SiteTable_get_num_rows(SiteTable *self, void *closure)
{
    PyObject *ret = NULL;
    if (SiteTable_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("n", (Py_ssize_t) self->table->num_rows);
out:
    return ret;
}

static PyObject *
SiteTable_get_max_rows(SiteTable *self, void *closure)
{
    PyObject *ret = NULL;
    if (SiteTable_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("n", (Py_ssize_t) self->table->max_rows);
out:
    return ret;
}

static PyObject *
SiteTable_get_position(SiteTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (SiteTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(
        self->table->num_rows, self->table->position, NPY_FLOAT64, sizeof(double));
out:
    return ret;
}

static PyObject *
SiteTable_get_ancestral_state(SiteTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (SiteTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(self->table->ancestral_state_length,
        self->table->ancestral_state, NPY_INT8, sizeof(char));
out:
    return ret;
}

static PyObject *
SiteTable_get_ancestral_state_offset(SiteTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (SiteTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(self->table->num_rows + 1,
        self->table->ancestral_state_offset, NPY_UINT32, sizeof(uint32_t));
out:
    return ret;
}

static PyObject *
SiteTable_get_metadata(SiteTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (SiteTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(
        self->table->metadata_length, self->table->metadata, NPY_INT8, sizeof(char));
out:
    return ret;
}

static PyObject *
SiteTable_get_metadata_offset(SiteTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (SiteTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(self->table->num_rows + 1, self->table->metadata_offset,
        NPY_UINT32, sizeof(uint32_t));
out:
    return ret;
}

static PyObject *
SiteTable_get_metadata_schema(SiteTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (SiteTable_check_state(self) != 0) {
        goto out;
    }
    ret = make_Py_Unicode_FromStringAndLength(
        self->table->metadata_schema, self->table->metadata_schema_length);
out:
    return ret;
}

static int
SiteTable_set_metadata_schema(SiteTable *self, PyObject *arg, void *closure)
{
    int ret = -1;
    int err;
    const char *metadata_schema;
    Py_ssize_t metadata_schema_length;

    if (SiteTable_check_state(self) != 0) {
        goto out;
    }
    metadata_schema = parse_metadata_schema_arg(arg, &metadata_schema_length);
    if (metadata_schema == NULL) {
        goto out;
    }
    err = tsk_site_table_set_metadata_schema(
        self->table, metadata_schema, metadata_schema_length);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static PyGetSetDef SiteTable_getsetters[] = {
    { .name = "max_rows_increment",
        .get = (getter) SiteTable_get_max_rows_increment,
        .doc = "The size increment" },
    { .name = "num_rows",
        .get = (getter) SiteTable_get_num_rows,
        .doc = "The number of rows in the table." },
    { .name = "max_rows",
        .get = (getter) SiteTable_get_max_rows,
        .doc = "The current maximum number of rows in the table." },
    { .name = "position",
        .get = (getter) SiteTable_get_position,
        .doc = "The position array." },
    { .name = "ancestral_state",
        .get = (getter) SiteTable_get_ancestral_state,
        .doc = "The ancestral state array." },
    { .name = "ancestral_state_offset",
        .get = (getter) SiteTable_get_ancestral_state_offset,
        .doc = "The ancestral state offset array." },
    { .name = "metadata",
        .get = (getter) SiteTable_get_metadata,
        .doc = "The metadata array." },
    { .name = "metadata_offset",
        .get = (getter) SiteTable_get_metadata_offset,
        .doc = "The metadata offset array." },
    { .name = "metadata_schema",
        .get = (getter) SiteTable_get_metadata_schema,
        .set = (setter) SiteTable_set_metadata_schema,
        .doc = "The metadata schema" },
    { NULL } /* Sentinel */
};

static PyMethodDef SiteTable_methods[] = {
    { .ml_name = "add_row",
        .ml_meth = (PyCFunction) SiteTable_add_row,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Adds a new row to this table." },
    { .ml_name = "equals",
        .ml_meth = (PyCFunction) SiteTable_equals,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Returns True if the specified SiteTable is equal to this one." },
    { .ml_name = "get_row",
        .ml_meth = (PyCFunction) SiteTable_get_row,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns the kth row in this table." },
    { .ml_name = "set_columns",
        .ml_meth = (PyCFunction) SiteTable_set_columns,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Copies the data in the specified arrays into the columns." },
    { .ml_name = "append_columns",
        .ml_meth = (PyCFunction) SiteTable_append_columns,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Appends the data in the specified arrays into the columns." },
    { .ml_name = "clear",
        .ml_meth = (PyCFunction) SiteTable_clear,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Clears this table." },
    { .ml_name = "truncate",
        .ml_meth = (PyCFunction) SiteTable_truncate,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Truncates this table to the specified number of rows." },
    { NULL } /* Sentinel */
};

static PyTypeObject SiteTableType = {
    // clang-format off
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_tskit.SiteTable",
    .tp_basicsize = sizeof(SiteTable),
    .tp_dealloc = (destructor) SiteTable_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "SiteTable objects",
    .tp_methods = SiteTable_methods,
    .tp_getset = SiteTable_getsetters,
    .tp_init = (initproc) SiteTable_init,
    .tp_new = PyType_GenericNew,
    // clang-format on
};

/*===================================================================
 * MutationTable
 *===================================================================
 */

static int
MutationTable_check_state(MutationTable *self)
{
    int ret = -1;
    if (self->table == NULL) {
        PyErr_SetString(PyExc_SystemError, "MutationTable not initialised");
        goto out;
    }
    if (self->locked) {
        PyErr_SetString(PyExc_RuntimeError, "MutationTable in use by other thread.");
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static void
MutationTable_dealloc(MutationTable *self)
{
    if (self->tables != NULL) {
        Py_DECREF(self->tables);
    } else if (self->table != NULL) {
        tsk_mutation_table_free(self->table);
        PyMem_Free(self->table);
        self->table = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static int
MutationTable_init(MutationTable *self, PyObject *args, PyObject *kwds)
{
    int ret = -1;
    int err;
    static char *kwlist[] = { "max_rows_increment", NULL };
    Py_ssize_t max_rows_increment = 0;

    self->table = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|n", kwlist, &max_rows_increment)) {
        goto out;
    }
    if (max_rows_increment < 0) {
        PyErr_SetString(PyExc_ValueError, "max_rows_increment must be positive");
        goto out;
    }
    self->table = PyMem_Malloc(sizeof(tsk_mutation_table_t));
    if (self->table == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    err = tsk_mutation_table_init(self->table, 0);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    tsk_mutation_table_set_max_rows_increment(self->table, max_rows_increment);
    ret = 0;
out:
    return ret;
}

static PyObject *
MutationTable_add_row(MutationTable *self, PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    int err;
    int site;
    int node;
    int parent = TSK_NULL;
    double time = TSK_UNKNOWN_TIME;
    char *derived_state;
    Py_ssize_t derived_state_length;
    PyObject *py_metadata = Py_None;
    char *metadata = NULL;
    Py_ssize_t metadata_length = 0;
    static char *kwlist[]
        = { "site", "node", "derived_state", "parent", "metadata", "time", NULL };

    if (MutationTable_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iis#|iOd", kwlist, &site, &node,
            &derived_state, &derived_state_length, &parent, &py_metadata, &time)) {
        goto out;
    }
    if (py_metadata != Py_None) {
        if (PyBytes_AsStringAndSize(py_metadata, &metadata, &metadata_length) < 0) {
            goto out;
        }
    }
    err = tsk_mutation_table_add_row(self->table, (tsk_id_t) site, (tsk_id_t) node,
        (tsk_id_t) parent, time, derived_state, derived_state_length, metadata,
        metadata_length);
    if (err < 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("i", err);
out:
    return ret;
}

/* Forward declaration */
static PyTypeObject MutationTableType;

static PyObject *
MutationTable_equals(MutationTable *self, PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    MutationTable *other = NULL;
    tsk_flags_t options = 0;
    int ignore_metadata = false;
    static char *kwlist[] = { "other", "ignore_metadata", NULL };

    if (MutationTable_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "O!|i", kwlist, &MutationTableType, &other, &ignore_metadata)) {
        goto out;
    }
    if (MutationTable_check_state(other) != 0) {
        goto out;
    }
    if (ignore_metadata) {
        options |= TSK_CMP_IGNORE_METADATA;
    }
    ret = Py_BuildValue(
        "i", tsk_mutation_table_equals(self->table, other->table, options));
out:
    return ret;
}

static PyObject *
MutationTable_get_row(MutationTable *self, PyObject *args)
{
    PyObject *ret = NULL;
    Py_ssize_t row_id;
    int err;
    tsk_mutation_t mutation;

    if (MutationTable_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "n", &row_id)) {
        goto out;
    }
    err = tsk_mutation_table_get_row(self->table, (tsk_id_t) row_id, &mutation);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = make_mutation(&mutation);
out:
    return ret;
}

static PyObject *
MutationTable_parse_dict_arg(MutationTable *self, PyObject *args, bool clear_table)
{
    int err;
    PyObject *ret = NULL;
    PyObject *dict = NULL;

    if (MutationTable_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "O!", &PyDict_Type, &dict)) {
        goto out;
    }
    err = parse_mutation_table_dict(self->table, dict, clear_table);
    if (err != 0) {
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
MutationTable_append_columns(MutationTable *self, PyObject *args)
{
    return MutationTable_parse_dict_arg(self, args, false);
}

static PyObject *
MutationTable_set_columns(MutationTable *self, PyObject *args)
{
    return MutationTable_parse_dict_arg(self, args, true);
}

static PyObject *
MutationTable_clear(MutationTable *self)
{
    PyObject *ret = NULL;
    int err;

    if (MutationTable_check_state(self) != 0) {
        goto out;
    }
    err = tsk_mutation_table_clear(self->table);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
MutationTable_truncate(MutationTable *self, PyObject *args)
{
    PyObject *ret = NULL;
    Py_ssize_t num_rows;
    int err;

    if (MutationTable_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "n", &num_rows)) {
        goto out;
    }
    if (num_rows < 0 || num_rows > (Py_ssize_t) self->table->num_rows) {
        PyErr_SetString(PyExc_ValueError, "num_rows out of bounds");
        goto out;
    }
    err = tsk_mutation_table_truncate(self->table, (tsk_size_t) num_rows);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
MutationTable_get_max_rows_increment(MutationTable *self, void *closure)
{
    PyObject *ret = NULL;
    if (MutationTable_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("n", (Py_ssize_t) self->table->max_rows_increment);
out:
    return ret;
}

static PyObject *
MutationTable_get_num_rows(MutationTable *self, void *closure)
{
    PyObject *ret = NULL;
    if (MutationTable_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("n", (Py_ssize_t) self->table->num_rows);
out:
    return ret;
}

static PyObject *
MutationTable_get_max_rows(MutationTable *self, void *closure)
{
    PyObject *ret = NULL;
    if (MutationTable_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("n", (Py_ssize_t) self->table->max_rows);
out:
    return ret;
}

static PyObject *
MutationTable_get_site(MutationTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (MutationTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(
        self->table->num_rows, self->table->site, NPY_INT32, sizeof(int32_t));
out:
    return ret;
}

static PyObject *
MutationTable_get_node(MutationTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (MutationTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(
        self->table->num_rows, self->table->node, NPY_INT32, sizeof(int32_t));
out:
    return ret;
}

static PyObject *
MutationTable_get_parent(MutationTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (MutationTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(
        self->table->num_rows, self->table->parent, NPY_INT32, sizeof(int32_t));
out:
    return ret;
}

static PyObject *
MutationTable_get_time(MutationTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (MutationTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(
        self->table->num_rows, self->table->time, NPY_FLOAT64, sizeof(double));
out:
    return ret;
}

static PyObject *
MutationTable_get_derived_state(MutationTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (MutationTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(self->table->derived_state_length,
        self->table->derived_state, NPY_INT8, sizeof(char));
out:
    return ret;
}

static PyObject *
MutationTable_get_derived_state_offset(MutationTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (MutationTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(self->table->num_rows + 1,
        self->table->derived_state_offset, NPY_UINT32, sizeof(uint32_t));
out:
    return ret;
}

static PyObject *
MutationTable_get_metadata(MutationTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (MutationTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(
        self->table->metadata_length, self->table->metadata, NPY_INT8, sizeof(char));
out:
    return ret;
}

static PyObject *
MutationTable_get_metadata_offset(MutationTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (MutationTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(self->table->num_rows + 1, self->table->metadata_offset,
        NPY_UINT32, sizeof(uint32_t));
out:
    return ret;
}

static PyObject *
MutationTable_get_metadata_schema(MutationTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (MutationTable_check_state(self) != 0) {
        goto out;
    }
    ret = make_Py_Unicode_FromStringAndLength(
        self->table->metadata_schema, self->table->metadata_schema_length);
out:
    return ret;
}

static int
MutationTable_set_metadata_schema(MutationTable *self, PyObject *arg, void *closure)
{
    int ret = -1;
    int err;
    const char *metadata_schema;
    Py_ssize_t metadata_schema_length;

    if (MutationTable_check_state(self) != 0) {
        goto out;
    }
    metadata_schema = parse_metadata_schema_arg(arg, &metadata_schema_length);
    if (metadata_schema == NULL) {
        goto out;
    }
    err = tsk_mutation_table_set_metadata_schema(
        self->table, metadata_schema, metadata_schema_length);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static PyGetSetDef MutationTable_getsetters[] = {
    { .name = "max_rows_increment",
        .get = (getter) MutationTable_get_max_rows_increment,
        .doc = "The size increment" },
    { .name = "num_rows",
        .get = (getter) MutationTable_get_num_rows,
        .doc = "The number of rows in the table." },
    { .name = "max_rows",
        .get = (getter) MutationTable_get_max_rows,
        .doc = "The curret maximum number of rows in the table." },
    { .name = "site", .get = (getter) MutationTable_get_site, .doc = "The site array" },
    { .name = "node", .get = (getter) MutationTable_get_node, .doc = "The node array" },
    { .name = "parent",
        .get = (getter) MutationTable_get_parent,
        .doc = "The parent array" },
    { .name = "time", .get = (getter) MutationTable_get_time, .doc = "The time array" },
    { .name = "derived_state",
        .get = (getter) MutationTable_get_derived_state,
        .doc = "The derived_state array" },
    { .name = "derived_state_offset",
        .get = (getter) MutationTable_get_derived_state_offset,
        .doc = "The derived_state_offset array" },
    { .name = "metadata",
        .get = (getter) MutationTable_get_metadata,
        .doc = "The metadata array" },
    { .name = "metadata_offset",
        .get = (getter) MutationTable_get_metadata_offset,
        .doc = "The metadata_offset array" },
    { .name = "metadata_schema",
        .get = (getter) MutationTable_get_metadata_schema,
        .set = (setter) MutationTable_set_metadata_schema,
        .doc = "The metadata schema" },
    { NULL } /* Sentinel */
};

static PyMethodDef MutationTable_methods[] = {
    { .ml_name = "add_row",
        .ml_meth = (PyCFunction) MutationTable_add_row,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Adds a new row to this table." },
    { .ml_name = "equals",
        .ml_meth = (PyCFunction) MutationTable_equals,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Returns True if the specified MutationTable is equal to this one." },
    { .ml_name = "get_row",
        .ml_meth = (PyCFunction) MutationTable_get_row,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns the kth row in this table." },
    { .ml_name = "set_columns",
        .ml_meth = (PyCFunction) MutationTable_set_columns,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Copies the data in the specified arrays into the columns." },
    { .ml_name = "append_columns",
        .ml_meth = (PyCFunction) MutationTable_append_columns,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Appends the data in the specified  arrays into the columns." },
    { .ml_name = "clear",
        .ml_meth = (PyCFunction) MutationTable_clear,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Clears this table." },
    { .ml_name = "truncate",
        .ml_meth = (PyCFunction) MutationTable_truncate,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Truncates this table to the specified number of rows." },
    { NULL } /* Sentinel */
};

static PyTypeObject MutationTableType = {
    // clang-format off
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_tskit.MutationTable",
    .tp_basicsize = sizeof(MutationTable),
    .tp_dealloc = (destructor) MutationTable_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "MutationTable objects",
    .tp_methods = MutationTable_methods,
    .tp_getset = MutationTable_getsetters,
    .tp_init = (initproc) MutationTable_init,
    .tp_new = PyType_GenericNew,
    // clang-format on
};

/*===================================================================
 * PopulationTable
 *===================================================================
 */

static int
PopulationTable_check_state(PopulationTable *self)
{
    int ret = -1;
    if (self->table == NULL) {
        PyErr_SetString(PyExc_SystemError, "PopulationTable not initialised");
        goto out;
    }
    if (self->locked) {
        PyErr_SetString(PyExc_RuntimeError, "PopulationTable in use by other thread.");
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static void
PopulationTable_dealloc(PopulationTable *self)
{
    if (self->tables != NULL) {
        Py_DECREF(self->tables);
    } else if (self->table != NULL) {
        tsk_population_table_free(self->table);
        PyMem_Free(self->table);
        self->table = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static int
PopulationTable_init(PopulationTable *self, PyObject *args, PyObject *kwds)
{
    int ret = -1;
    int err;
    static char *kwlist[] = { "max_rows_increment", NULL };
    Py_ssize_t max_rows_increment = 0;

    self->table = NULL;
    self->locked = false;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|n", kwlist, &max_rows_increment)) {
        goto out;
    }
    if (max_rows_increment < 0) {
        PyErr_SetString(PyExc_ValueError, "max_rows_increment must be positive");
        goto out;
    }
    self->table = PyMem_Malloc(sizeof(tsk_population_table_t));
    if (self->table == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    err = tsk_population_table_init(self->table, 0);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    tsk_population_table_set_max_rows_increment(self->table, max_rows_increment);
    ret = 0;
out:
    return ret;
}

static PyObject *
PopulationTable_add_row(PopulationTable *self, PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    int err;
    PyObject *py_metadata = Py_None;
    char *metadata = NULL;
    Py_ssize_t metadata_length = 0;
    static char *kwlist[] = { "metadata", NULL };

    if (PopulationTable_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist, &py_metadata)) {
        goto out;
    }

    if (py_metadata != Py_None) {
        if (PyBytes_AsStringAndSize(py_metadata, &metadata, &metadata_length) < 0) {
            goto out;
        }
    }
    err = tsk_population_table_add_row(self->table, metadata, metadata_length);
    if (err < 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("i", err);
out:
    return ret;
}

/* Forward declaration */
static PyTypeObject PopulationTableType;

static PyObject *
PopulationTable_equals(PopulationTable *self, PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    PopulationTable *other = NULL;
    tsk_flags_t options = 0;
    int ignore_metadata = false;
    static char *kwlist[] = { "other", "ignore_metadata", NULL };

    if (PopulationTable_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|i", kwlist, &PopulationTableType,
            &other, &ignore_metadata)) {
        goto out;
    }
    if (PopulationTable_check_state(other) != 0) {
        goto out;
    }
    if (ignore_metadata) {
        options |= TSK_CMP_IGNORE_METADATA;
    }
    ret = Py_BuildValue(
        "i", tsk_population_table_equals(self->table, other->table, options));
out:
    return ret;
}

static PyObject *
PopulationTable_get_row(PopulationTable *self, PyObject *args)
{
    PyObject *ret = NULL;
    Py_ssize_t row_id;
    int err;
    tsk_population_t population;

    if (PopulationTable_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "n", &row_id)) {
        goto out;
    }
    err = tsk_population_table_get_row(self->table, (tsk_id_t) row_id, &population);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = make_population(&population);
out:
    return ret;
}

static PyObject *
PopulationTable_parse_dict_arg(PopulationTable *self, PyObject *args, bool clear_table)
{
    int err;
    PyObject *ret = NULL;
    PyObject *dict = NULL;

    if (PopulationTable_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "O!", &PyDict_Type, &dict)) {
        goto out;
    }
    err = parse_population_table_dict(self->table, dict, clear_table);
    if (err != 0) {
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
PopulationTable_append_columns(PopulationTable *self, PyObject *args)
{
    return PopulationTable_parse_dict_arg(self, args, false);
}

static PyObject *
PopulationTable_set_columns(PopulationTable *self, PyObject *args)
{
    return PopulationTable_parse_dict_arg(self, args, true);
}

static PyObject *
PopulationTable_clear(PopulationTable *self)
{
    PyObject *ret = NULL;
    int err;

    if (PopulationTable_check_state(self) != 0) {
        goto out;
    }
    err = tsk_population_table_clear(self->table);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
PopulationTable_truncate(PopulationTable *self, PyObject *args)
{
    PyObject *ret = NULL;
    Py_ssize_t num_rows;
    int err;

    if (PopulationTable_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "n", &num_rows)) {
        goto out;
    }
    if (num_rows < 0 || num_rows > (Py_ssize_t) self->table->num_rows) {
        PyErr_SetString(PyExc_ValueError, "num_rows out of bounds");
        goto out;
    }
    err = tsk_population_table_truncate(self->table, (tsk_size_t) num_rows);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
PopulationTable_get_max_rows_increment(PopulationTable *self, void *closure)
{
    PyObject *ret = NULL;
    if (PopulationTable_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("n", (Py_ssize_t) self->table->max_rows_increment);
out:
    return ret;
}

static PyObject *
PopulationTable_get_num_rows(PopulationTable *self, void *closure)
{
    PyObject *ret = NULL;
    if (PopulationTable_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("n", (Py_ssize_t) self->table->num_rows);
out:
    return ret;
}

static PyObject *
PopulationTable_get_max_rows(PopulationTable *self, void *closure)
{
    PyObject *ret = NULL;
    if (PopulationTable_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("n", (Py_ssize_t) self->table->max_rows);
out:
    return ret;
}

static PyObject *
PopulationTable_get_metadata(PopulationTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (PopulationTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(
        self->table->metadata_length, self->table->metadata, NPY_INT8, sizeof(char));
out:
    return ret;
}

static PyObject *
PopulationTable_get_metadata_offset(PopulationTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (PopulationTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(self->table->num_rows + 1, self->table->metadata_offset,
        NPY_UINT32, sizeof(uint32_t));
out:
    return ret;
}

static PyObject *
PopulationTable_get_metadata_schema(PopulationTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (PopulationTable_check_state(self) != 0) {
        goto out;
    }
    ret = make_Py_Unicode_FromStringAndLength(
        self->table->metadata_schema, self->table->metadata_schema_length);
out:
    return ret;
}

static int
PopulationTable_set_metadata_schema(PopulationTable *self, PyObject *arg, void *closure)
{
    int ret = -1;
    int err;
    const char *metadata_schema;
    Py_ssize_t metadata_schema_length;

    if (PopulationTable_check_state(self) != 0) {
        goto out;
    }
    metadata_schema = parse_metadata_schema_arg(arg, &metadata_schema_length);
    if (metadata_schema == NULL) {
        goto out;
    }
    err = tsk_population_table_set_metadata_schema(
        self->table, metadata_schema, metadata_schema_length);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static PyGetSetDef PopulationTable_getsetters[] = {
    { .name = "max_rows_increment",
        .get = (getter) PopulationTable_get_max_rows_increment,
        .doc = "The size increment" },
    { .name = "num_rows",
        .get = (getter) PopulationTable_get_num_rows,
        .doc = "The number of rows in the table." },
    { .name = "max_rows",
        .get = (getter) PopulationTable_get_max_rows,
        .doc = "The current maximum number of rows in the table." },
    { .name = "metadata",
        .get = (getter) PopulationTable_get_metadata,
        .doc = "The metadata array" },
    { .name = "metadata_offset",
        .get = (getter) PopulationTable_get_metadata_offset,
        .doc = "The metadata offset array" },
    { .name = "metadata_schema",
        .get = (getter) PopulationTable_get_metadata_schema,
        .set = (setter) PopulationTable_set_metadata_schema,
        .doc = "The metadata schema" },
    { NULL } /* Sentinel */
};

static PyMethodDef PopulationTable_methods[] = {
    { .ml_name = "add_row",
        .ml_meth = (PyCFunction) PopulationTable_add_row,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Adds a new row to this table." },
    { .ml_name = "equals",
        .ml_meth = (PyCFunction) PopulationTable_equals,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc
        = "Returns True if the specified PopulationTable is equal to this one." },
    { .ml_name = "get_row",
        .ml_meth = (PyCFunction) PopulationTable_get_row,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns the kth row in this table." },
    { .ml_name = "append_columns",
        .ml_meth = (PyCFunction) PopulationTable_append_columns,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Appends the data in the specified arrays into the columns." },
    { .ml_name = "set_columns",
        .ml_meth = (PyCFunction) PopulationTable_set_columns,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Copies the data in the specified arrays into the columns." },
    { .ml_name = "clear",
        .ml_meth = (PyCFunction) PopulationTable_clear,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Clears this table." },
    { .ml_name = "truncate",
        .ml_meth = (PyCFunction) PopulationTable_truncate,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Truncates this table to the specified number of rows." },
    { NULL } /* Sentinel */
};

static PyTypeObject PopulationTableType = {
    // clang-format off
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_tskit.PopulationTable",
    .tp_basicsize = sizeof(PopulationTable),
    .tp_dealloc = (destructor) PopulationTable_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "PopulationTable objects",
    .tp_methods = PopulationTable_methods,
    .tp_getset = PopulationTable_getsetters,
    .tp_init = (initproc) PopulationTable_init,
    .tp_new = PyType_GenericNew,
    // clang-format on
};

/*===================================================================
 * ProvenanceTable
 *===================================================================
 */

static int
ProvenanceTable_check_state(ProvenanceTable *self)
{
    int ret = -1;
    if (self->table == NULL) {
        PyErr_SetString(PyExc_SystemError, "ProvenanceTable not initialised");
        goto out;
    }
    if (self->locked) {
        PyErr_SetString(PyExc_RuntimeError, "ProvenanceTable in use by other thread.");
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static void
ProvenanceTable_dealloc(ProvenanceTable *self)
{
    if (self->tables != NULL) {
        Py_DECREF(self->tables);
    } else if (self->table != NULL) {
        tsk_provenance_table_free(self->table);
        PyMem_Free(self->table);
        self->table = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static int
ProvenanceTable_init(ProvenanceTable *self, PyObject *args, PyObject *kwds)
{
    int ret = -1;
    int err;
    static char *kwlist[] = { "max_rows_increment", NULL };
    Py_ssize_t max_rows_increment = 0;

    self->table = NULL;
    self->locked = false;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|n", kwlist, &max_rows_increment)) {
        goto out;
    }
    if (max_rows_increment < 0) {
        PyErr_SetString(PyExc_ValueError, "max_rows_increment must be positive");
        goto out;
    }
    self->table = PyMem_Malloc(sizeof(tsk_provenance_table_t));
    if (self->table == NULL) {
        PyErr_NoMemory();
        goto out;
    }

    err = tsk_provenance_table_init(self->table, 0);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    tsk_provenance_table_set_max_rows_increment(self->table, max_rows_increment);
    ret = 0;
out:
    return ret;
}
static PyObject *
ProvenanceTable_add_row(ProvenanceTable *self, PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    int err;
    char *timestamp = "";
    Py_ssize_t timestamp_length = 0;
    char *record = "";
    Py_ssize_t record_length = 0;
    static char *kwlist[] = { "timestamp", "record", NULL };

    if (ProvenanceTable_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s#s#", kwlist, &timestamp,
            &timestamp_length, &record, &record_length)) {
        goto out;
    }
    err = tsk_provenance_table_add_row(
        self->table, timestamp, timestamp_length, record, record_length);
    if (err < 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("i", err);
out:
    return ret;
}

/* Forward declaration */
static PyTypeObject ProvenanceTableType;

static PyObject *
ProvenanceTable_equals(ProvenanceTable *self, PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    ProvenanceTable *other = NULL;
    tsk_flags_t options = 0;
    int ignore_timestamps = false;
    static char *kwlist[] = { "other", "ignore_timestamps", NULL };

    if (ProvenanceTable_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|i", kwlist, &ProvenanceTableType,
            &other, &ignore_timestamps)) {
        goto out;
    }
    if (ProvenanceTable_check_state(other) != 0) {
        goto out;
    }
    if (ignore_timestamps) {
        options |= TSK_CMP_IGNORE_TIMESTAMPS;
    }
    ret = Py_BuildValue(
        "i", tsk_provenance_table_equals(self->table, other->table, options));
out:
    return ret;
}

static PyObject *
ProvenanceTable_get_row(ProvenanceTable *self, PyObject *args)
{
    PyObject *ret = NULL;
    Py_ssize_t row_id;
    int err;
    tsk_provenance_t provenance;

    if (ProvenanceTable_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "n", &row_id)) {
        goto out;
    }
    err = tsk_provenance_table_get_row(self->table, (tsk_id_t) row_id, &provenance);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = make_provenance(&provenance);
out:
    return ret;
}

static PyObject *
ProvenanceTable_parse_dict_arg(ProvenanceTable *self, PyObject *args, bool clear_table)
{
    int err;
    PyObject *ret = NULL;
    PyObject *dict = NULL;

    if (ProvenanceTable_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "O!", &PyDict_Type, &dict)) {
        goto out;
    }
    err = parse_provenance_table_dict(self->table, dict, clear_table);
    if (err != 0) {
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
ProvenanceTable_append_columns(ProvenanceTable *self, PyObject *args)
{
    return ProvenanceTable_parse_dict_arg(self, args, false);
}

static PyObject *
ProvenanceTable_set_columns(ProvenanceTable *self, PyObject *args)
{
    return ProvenanceTable_parse_dict_arg(self, args, true);
}

static PyObject *
ProvenanceTable_clear(ProvenanceTable *self)
{
    PyObject *ret = NULL;
    int err;

    if (ProvenanceTable_check_state(self) != 0) {
        goto out;
    }
    err = tsk_provenance_table_clear(self->table);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
ProvenanceTable_truncate(ProvenanceTable *self, PyObject *args)
{
    PyObject *ret = NULL;
    Py_ssize_t num_rows;
    int err;

    if (ProvenanceTable_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "n", &num_rows)) {
        goto out;
    }
    if (num_rows < 0 || num_rows > (Py_ssize_t) self->table->num_rows) {
        PyErr_SetString(PyExc_ValueError, "num_rows out of bounds");
        goto out;
    }
    err = tsk_provenance_table_truncate(self->table, (tsk_size_t) num_rows);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
ProvenanceTable_get_max_rows_increment(ProvenanceTable *self, void *closure)
{
    PyObject *ret = NULL;
    if (ProvenanceTable_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("n", (Py_ssize_t) self->table->max_rows_increment);
out:
    return ret;
}

static PyObject *
ProvenanceTable_get_num_rows(ProvenanceTable *self, void *closure)
{
    PyObject *ret = NULL;
    if (ProvenanceTable_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("n", (Py_ssize_t) self->table->num_rows);
out:
    return ret;
}

static PyObject *
ProvenanceTable_get_max_rows(ProvenanceTable *self, void *closure)
{
    PyObject *ret = NULL;
    if (ProvenanceTable_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("n", (Py_ssize_t) self->table->max_rows);
out:
    return ret;
}

static PyObject *
ProvenanceTable_get_timestamp(ProvenanceTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (ProvenanceTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(
        self->table->timestamp_length, self->table->timestamp, NPY_INT8, sizeof(char));
out:
    return ret;
}

static PyObject *
ProvenanceTable_get_timestamp_offset(ProvenanceTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (ProvenanceTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(self->table->num_rows + 1,
        self->table->timestamp_offset, NPY_UINT32, sizeof(uint32_t));
out:
    return ret;
}

static PyObject *
ProvenanceTable_get_record(ProvenanceTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (ProvenanceTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(
        self->table->record_length, self->table->record, NPY_INT8, sizeof(char));
out:
    return ret;
}

static PyObject *
ProvenanceTable_get_record_offset(ProvenanceTable *self, void *closure)
{
    PyObject *ret = NULL;

    if (ProvenanceTable_check_state(self) != 0) {
        goto out;
    }
    ret = table_get_column_array(self->table->num_rows + 1, self->table->record_offset,
        NPY_UINT32, sizeof(uint32_t));
out:
    return ret;
}

static PyGetSetDef ProvenanceTable_getsetters[] = {
    { .name = "max_rows_increment",
        .get = (getter) ProvenanceTable_get_max_rows_increment,
        .doc = "The size increment" },
    { .name = "num_rows",
        .get = (getter) ProvenanceTable_get_num_rows,
        .doc = "The number of rows in the table." },
    { .name = "max_rows",
        .get = (getter) ProvenanceTable_get_max_rows,
        .doc = "The current maximum number of rows in the table." },
    { .name = "timestamp",
        .get = (getter) ProvenanceTable_get_timestamp,
        .doc = "The timestamp array" },
    { .name = "timestamp_offset",
        .get = (getter) ProvenanceTable_get_timestamp_offset,
        .doc = "The timestamp offset array" },
    { .name = "record",
        .get = (getter) ProvenanceTable_get_record,
        .doc = "The record array" },
    { .name = "record_offset",
        .get = (getter) ProvenanceTable_get_record_offset,
        .doc = "The record offset array" },
    { NULL } /* Sentinel */
};

static PyMethodDef ProvenanceTable_methods[] = {
    { .ml_name = "add_row",
        .ml_meth = (PyCFunction) ProvenanceTable_add_row,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Adds a new row to this table." },
    { .ml_name = "equals",
        .ml_meth = (PyCFunction) ProvenanceTable_equals,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc
        = "Returns True if the specified ProvenanceTable is equal to this one." },
    { .ml_name = "get_row",
        .ml_meth = (PyCFunction) ProvenanceTable_get_row,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns the kth row in this table." },
    { .ml_name = "append_columns",
        .ml_meth = (PyCFunction) ProvenanceTable_append_columns,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Appends the data in the specified arrays into the columns." },
    { .ml_name = "set_columns",
        .ml_meth = (PyCFunction) ProvenanceTable_set_columns,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Copies the data in the specified arrays into the columns." },
    { .ml_name = "clear",
        .ml_meth = (PyCFunction) ProvenanceTable_clear,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Clears this table." },
    { .ml_name = "truncate",
        .ml_meth = (PyCFunction) ProvenanceTable_truncate,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Truncates this table to the specified number of rows." },
    { NULL } /* Sentinel */
};

static PyTypeObject ProvenanceTableType = {
    // clang-format off
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_tskit.ProvenanceTable",
    .tp_basicsize = sizeof(ProvenanceTable),
    .tp_dealloc = (destructor) ProvenanceTable_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "ProvenanceTable objects",
    .tp_methods = ProvenanceTable_methods,
    .tp_getset = ProvenanceTable_getsetters,
    .tp_init = (initproc) ProvenanceTable_init,
    .tp_new = PyType_GenericNew,
    // clang-format on
};

/*===================================================================
 * TableCollection
 *===================================================================
 */

static int
TableCollection_check_state(TableCollection *self)
{
    int ret = 0;
    if (self->tables == NULL) {
        PyErr_SetString(PyExc_SystemError, "TableCollection not initialised");
        ret = -1;
    }
    return ret;
}

static int
TableCollection_alloc(TableCollection *self)
{
    int ret = -1;

    if (self->tables != NULL) {
        tsk_table_collection_free(self->tables);
        PyMem_Free(self->tables);
    }
    self->tables = PyMem_Malloc(sizeof(tsk_table_collection_t));
    if (self->tables == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    memset(self->tables, 0, sizeof(*self->tables));
    ret = 0;
out:
    return ret;
}

static void
TableCollection_dealloc(TableCollection *self)
{
    if (self->tables != NULL) {
        tsk_table_collection_free(self->tables);
        PyMem_Free(self->tables);
        self->tables = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static int
TableCollection_init(TableCollection *self, PyObject *args, PyObject *kwds)
{
    int ret = -1;
    int err;
    static char *kwlist[] = { "sequence_length", NULL };
    double sequence_length = -1;

    self->tables = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|d", kwlist, &sequence_length)) {
        goto out;
    }

    self->tables = PyMem_Malloc(sizeof(tsk_table_collection_t));
    if (self->tables == NULL) {
        PyErr_NoMemory();
    }
    err = tsk_table_collection_init(self->tables, 0);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    self->tables->sequence_length = sequence_length;
    ret = 0;
out:
    return ret;
}

/* The getters for each of the tables returns a new reference which we
 * set up here. These references use a pointer to the table stored in
 * the table collection, so to guard against this memory getting freed
 * we the Python Table classes keep a reference to the TableCollection
 * and INCREF it. We don't keep permanent references to the Table classes
 * in the TableCollection as this gives a circular references which would
 * require implementing support for cyclic garbage collection.
 */

static PyObject *
TableCollection_get_individuals(TableCollection *self, void *closure)
{
    IndividualTable *individuals = NULL;

    if (TableCollection_check_state(self) != 0) {
        goto out;
    }
    individuals = PyObject_New(IndividualTable, &IndividualTableType);
    if (individuals == NULL) {
        goto out;
    }
    individuals->table = &self->tables->individuals;
    individuals->locked = false;
    individuals->tables = self;
    Py_INCREF(self);
out:
    return (PyObject *) individuals;
}

static PyObject *
TableCollection_get_nodes(TableCollection *self, void *closure)
{
    NodeTable *nodes = NULL;

    if (TableCollection_check_state(self) != 0) {
        goto out;
    }
    nodes = PyObject_New(NodeTable, &NodeTableType);
    if (nodes == NULL) {
        goto out;
    }
    nodes->table = &self->tables->nodes;
    nodes->locked = false;
    nodes->tables = self;
    Py_INCREF(self);
out:
    return (PyObject *) nodes;
}

static PyObject *
TableCollection_get_edges(TableCollection *self, void *closure)
{
    EdgeTable *edges = NULL;

    if (TableCollection_check_state(self) != 0) {
        goto out;
    }
    edges = PyObject_New(EdgeTable, &EdgeTableType);
    if (edges == NULL) {
        goto out;
    }
    edges->table = &self->tables->edges;
    edges->locked = false;
    edges->tables = self;
    Py_INCREF(self);
out:
    return (PyObject *) edges;
}

static PyObject *
TableCollection_get_migrations(TableCollection *self, void *closure)
{
    MigrationTable *migrations = NULL;

    if (TableCollection_check_state(self) != 0) {
        goto out;
    }
    migrations = PyObject_New(MigrationTable, &MigrationTableType);
    if (migrations == NULL) {
        goto out;
    }
    migrations->table = &self->tables->migrations;
    migrations->locked = false;
    migrations->tables = self;
    Py_INCREF(self);
out:
    return (PyObject *) migrations;
}

static PyObject *
TableCollection_get_sites(TableCollection *self, void *closure)
{
    SiteTable *sites = NULL;

    if (TableCollection_check_state(self) != 0) {
        goto out;
    }
    sites = PyObject_New(SiteTable, &SiteTableType);
    if (sites == NULL) {
        goto out;
    }
    sites->table = &self->tables->sites;
    sites->locked = false;
    sites->tables = self;
    Py_INCREF(self);
out:
    return (PyObject *) sites;
}

static PyObject *
TableCollection_get_mutations(TableCollection *self, void *closure)
{
    MutationTable *mutations = NULL;

    if (TableCollection_check_state(self) != 0) {
        goto out;
    }
    mutations = PyObject_New(MutationTable, &MutationTableType);
    if (mutations == NULL) {
        goto out;
    }
    mutations->table = &self->tables->mutations;
    mutations->locked = false;
    mutations->tables = self;
    Py_INCREF(self);
out:
    return (PyObject *) mutations;
}

static PyObject *
TableCollection_get_populations(TableCollection *self, void *closure)
{
    PopulationTable *populations = NULL;

    if (TableCollection_check_state(self) != 0) {
        goto out;
    }
    populations = PyObject_New(PopulationTable, &PopulationTableType);
    if (populations == NULL) {
        goto out;
    }
    populations->table = &self->tables->populations;
    populations->locked = false;
    populations->tables = self;
    Py_INCREF(self);
out:
    return (PyObject *) populations;
}

static PyObject *
TableCollection_get_provenances(TableCollection *self, void *closure)
{
    ProvenanceTable *provenances = NULL;

    if (TableCollection_check_state(self) != 0) {
        goto out;
    }
    provenances = PyObject_New(ProvenanceTable, &ProvenanceTableType);
    if (provenances == NULL) {
        goto out;
    }
    provenances->table = &self->tables->provenances;
    provenances->locked = false;
    provenances->tables = self;
    Py_INCREF(self);
out:
    return (PyObject *) provenances;
}

static PyObject *
TableCollection_get_sequence_length(TableCollection *self, void *closure)
{
    PyObject *ret = NULL;

    if (TableCollection_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("f", self->tables->sequence_length);
out:
    return ret;
}

static int
TableCollection_set_sequence_length(
    TableCollection *self, PyObject *value, void *closure)
{
    int ret = -1;

    if (TableCollection_check_state(self) != 0) {
        goto out;
    }
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the sequence_length attribute");
        goto out;
    }
    if (!PyNumber_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "sequence_length must be a number");
        goto out;
    }
    self->tables->sequence_length = PyFloat_AsDouble(value);
    ret = 0;
out:
    return ret;
}

static PyObject *
TableCollection_get_file_uuid(TableCollection *self, void *closure)
{
    PyObject *ret = NULL;

    if (TableCollection_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("s", self->tables->file_uuid);
out:
    return ret;
}

static PyObject *
TableCollection_get_metadata(TableCollection *self, void *closure)
{
    PyObject *ret = NULL;

    if (TableCollection_check_state(self) != 0) {
        goto out;
    }
    ret = PyBytes_FromStringAndSize(
        self->tables->metadata, self->tables->metadata_length);
out:
    return ret;
}

static int
TableCollection_set_metadata(TableCollection *self, PyObject *arg, void *closure)
{
    int ret = -1;
    int err;
    char *metadata;
    Py_ssize_t metadata_length;

    if (TableCollection_check_state(self) != 0) {
        goto out;
    }
    if (arg == NULL) {
        PyErr_Format(PyExc_AttributeError,
            "Cannot del metadata, set to empty string (b\"\") to clear.");
        goto out;
    }
    err = PyBytes_AsStringAndSize(arg, &metadata, &metadata_length);
    if (err != 0) {
        goto out;
    }
    err = tsk_table_collection_set_metadata(self->tables, metadata, metadata_length);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static PyObject *
TableCollection_get_metadata_schema(TableCollection *self, void *closure)
{
    PyObject *ret = NULL;

    if (TableCollection_check_state(self) != 0) {
        goto out;
    }
    ret = make_Py_Unicode_FromStringAndLength(
        self->tables->metadata_schema, self->tables->metadata_schema_length);
out:
    return ret;
}

static int
TableCollection_set_metadata_schema(TableCollection *self, PyObject *arg, void *closure)
{
    int ret = -1;
    int err;
    const char *metadata_schema;
    Py_ssize_t metadata_schema_length;

    if (TableCollection_check_state(self) != 0) {
        goto out;
    }
    metadata_schema = parse_metadata_schema_arg(arg, &metadata_schema_length);
    if (metadata_schema == NULL) {
        goto out;
    }
    err = tsk_table_collection_set_metadata_schema(
        self->tables, metadata_schema, metadata_schema_length);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static PyObject *
TableCollection_simplify(TableCollection *self, PyObject *args, PyObject *kwds)
{
    int err;
    PyObject *ret = NULL;
    PyObject *samples = NULL;
    PyArrayObject *samples_array = NULL;
    PyArrayObject *node_map_array = NULL;
    npy_intp *shape, dims;
    size_t num_samples;
    tsk_flags_t options = 0;
    int filter_sites = true;
    int filter_individuals = false;
    int filter_populations = false;
    int keep_unary = false;
    int keep_input_roots = false;
    int reduce_to_site_topology = false;
    static char *kwlist[]
        = { "samples", "filter_sites", "filter_populations", "filter_individuals",
              "reduce_to_site_topology", "keep_unary", "keep_input_roots", NULL };

    if (TableCollection_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|iiiiii", kwlist, &samples,
            &filter_sites, &filter_populations, &filter_individuals,
            &reduce_to_site_topology, &keep_unary, &keep_input_roots)) {
        goto out;
    }
    samples_array = (PyArrayObject *) PyArray_FROMANY(
        samples, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (samples_array == NULL) {
        goto out;
    }
    shape = PyArray_DIMS(samples_array);
    num_samples = shape[0];
    if (filter_sites) {
        options |= TSK_FILTER_SITES;
    }
    if (filter_individuals) {
        options |= TSK_FILTER_INDIVIDUALS;
    }
    if (filter_populations) {
        options |= TSK_FILTER_POPULATIONS;
    }
    if (reduce_to_site_topology) {
        options |= TSK_REDUCE_TO_SITE_TOPOLOGY;
    }
    if (keep_unary) {
        options |= TSK_KEEP_UNARY;
    }
    if (keep_input_roots) {
        options |= TSK_KEEP_INPUT_ROOTS;
    }

    /* Allocate a new array to hold the node map. */
    dims = self->tables->nodes.num_rows;
    node_map_array = (PyArrayObject *) PyArray_SimpleNew(1, &dims, NPY_INT32);
    if (node_map_array == NULL) {
        goto out;
    }
    err = tsk_table_collection_simplify(self->tables, PyArray_DATA(samples_array),
        num_samples, options, PyArray_DATA(node_map_array));
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = (PyObject *) node_map_array;
    node_map_array = NULL;
out:
    Py_XDECREF(samples_array);
    Py_XDECREF(node_map_array);
    return ret;
}

static PyObject *
TableCollection_link_ancestors(TableCollection *self, PyObject *args, PyObject *kwds)
{
    int err;
    PyObject *ret = NULL;
    PyObject *samples = NULL;
    PyObject *ancestors = NULL;
    PyArrayObject *samples_array = NULL;
    PyArrayObject *ancestors_array = NULL;
    npy_intp *shape;
    size_t num_samples, num_ancestors;
    static char *kwlist[] = { "samples", "ancestors", NULL };
    EdgeTable *result = NULL;
    PyObject *result_args = NULL;

    if (TableCollection_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO", kwlist, &samples, &ancestors)) {
        goto out;
    }

    samples_array = (PyArrayObject *) PyArray_FROMANY(
        samples, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (samples_array == NULL) {
        goto out;
    }
    shape = PyArray_DIMS(samples_array);
    num_samples = shape[0];

    ancestors_array = (PyArrayObject *) PyArray_FROMANY(
        ancestors, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (ancestors_array == NULL) {
        goto out;
    }
    shape = PyArray_DIMS(ancestors_array);
    num_ancestors = shape[0];

    result_args = PyTuple_New(0);
    if (result_args == NULL) {
        goto out;
    }
    result = (EdgeTable *) PyObject_CallObject((PyObject *) &EdgeTableType, result_args);
    if (result == NULL) {
        goto out;
    }
    err = tsk_table_collection_link_ancestors(self->tables, PyArray_DATA(samples_array),
        num_samples, PyArray_DATA(ancestors_array), num_ancestors, 0, result->table);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = (PyObject *) result;
    result = NULL;
out:
    Py_XDECREF(samples_array);
    Py_XDECREF(ancestors_array);
    Py_XDECREF(result);
    Py_XDECREF(result_args);
    return ret;
}

static PyObject *
TableCollection_subset(TableCollection *self, PyObject *args)
{
    int err;
    PyObject *ret = NULL;
    PyObject *nodes = NULL;
    PyArrayObject *nodes_array = NULL;
    npy_intp *shape;
    size_t num_nodes;

    if (TableCollection_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "O", &nodes)) {
        goto out;
    }
    nodes_array
        = (PyArrayObject *) PyArray_FROMANY(nodes, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (nodes_array == NULL) {
        goto out;
    }
    shape = PyArray_DIMS(nodes_array);
    num_nodes = shape[0];

    err = tsk_table_collection_subset(
        self->tables, PyArray_DATA(nodes_array), num_nodes);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    Py_XDECREF(nodes_array);
    return ret;
}

/* Forward declaration */
static PyTypeObject TableCollectionType;

static PyObject *
TableCollection_union(TableCollection *self, PyObject *args, PyObject *kwds)
{
    int err;
    TableCollection *other = NULL;
    PyObject *ret = NULL;
    PyObject *other_node_mapping = NULL;
    PyArrayObject *nmap_array = NULL;
    npy_intp *shape;
    tsk_flags_t options = 0;
    int check_shared = true;
    int add_populations = true;
    static char *kwlist[] = { "other", "other_node_mapping", "check_shared_equality",
        "add_populations", NULL };

    if (TableCollection_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O|ii", kwlist, &TableCollectionType,
            &other, &other_node_mapping, &check_shared, &add_populations)) {
        goto out;
    }
    nmap_array = (PyArrayObject *) PyArray_FROMANY(
        other_node_mapping, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (nmap_array == NULL) {
        goto out;
    }
    shape = PyArray_DIMS(nmap_array);
    if (other->tables->nodes.num_rows != (tsk_size_t) shape[0]) {
        PyErr_SetString(PyExc_ValueError,
            "The length of the node mapping array should be equal to the"
            " number of nodes in the other tree sequence.");
        goto out;
    }
    if (!check_shared) {
        options |= TSK_UNION_NO_CHECK_SHARED;
    }
    if (!add_populations) {
        options |= TSK_UNION_NO_ADD_POP;
    }
    err = tsk_table_collection_union(
        self->tables, other->tables, PyArray_DATA(nmap_array), options);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    Py_XDECREF(nmap_array);
    return ret;
}

static PyObject *
convert_ibd_segments(tsk_ibd_finder_t *ibd_finder, tsk_id_t *pairs, tsk_size_t num_pairs)
{
    PyObject *ret = NULL;
    PyObject *key = NULL;
    PyObject *value = NULL;
    PyArrayObject *left_array = NULL;
    PyArrayObject *right_array = NULL;
    PyArrayObject *node_array = NULL;
    double *left, *right;
    int err;
    tsk_id_t *node;
    tsk_size_t j, seg_index;
    tsk_segment_t *u, *head;
    PyObject *pair_dict = PyDict_New();
    npy_intp num_segments;

    if (pair_dict == NULL) {
        goto out;
    }

    for (j = 0; j < num_pairs; j++) {
        err = tsk_ibd_finder_get_ibd_segments(ibd_finder, j, &head);
        if (err == -1) {
            head = NULL;
        } else if (err != 0) {
            handle_library_error(err);
            goto out;
        }
        num_segments = 0;
        for (u = head; u != NULL; u = u->next) {
            num_segments++;
        }
        /* For each pair we return an array of left, right, node values */
        left_array = (PyArrayObject *) PyArray_SimpleNew(1, &num_segments, NPY_FLOAT64);
        right_array = (PyArrayObject *) PyArray_SimpleNew(1, &num_segments, NPY_FLOAT64);
        node_array = (PyArrayObject *) PyArray_SimpleNew(1, &num_segments, NPY_INT32);
        if (left_array == NULL || right_array == NULL || node_array == NULL) {
            goto out;
        }
        left = (double *) PyArray_DATA(left_array);
        right = (double *) PyArray_DATA(right_array);
        node = (tsk_id_t *) PyArray_DATA(node_array);
        seg_index = 0;
        for (u = head; u != NULL; u = u->next) {
            left[seg_index] = u->left;
            right[seg_index] = u->right;
            node[seg_index] = u->node;
            seg_index++;
        }
        key = Py_BuildValue("(ii)", pairs[2 * j], pairs[2 * j + 1]);
        value = Py_BuildValue("{s:O,s:O,s:O}", "left", left_array, "right", right_array,
            "node", node_array);
        if (key == NULL || value == NULL) {
            goto out;
        }
        if (PyDict_SetItem(pair_dict, key, value) != 0) {
            goto out;
        }
        Py_DECREF(key);
        Py_DECREF(value);
        Py_DECREF(left_array);
        Py_DECREF(right_array);
        Py_DECREF(node_array);
        key = NULL;
        value = NULL;
        left_array = NULL;
        right_array = NULL;
        node_array = NULL;
    }
    ret = pair_dict;
    pair_dict = NULL;
out:
    Py_XDECREF(key);
    Py_XDECREF(value);
    Py_XDECREF(left_array);
    Py_XDECREF(right_array);
    Py_XDECREF(node_array);
    Py_XDECREF(pair_dict);
    return ret;
}

static PyObject *
TableCollection_find_ibd(TableCollection *self, PyObject *args, PyObject *kwds)
{
    int err;
    PyObject *ret = NULL;
    tsk_ibd_finder_t ibd_finder;
    PyObject *samples;
    PyArrayObject *samples_array = NULL;
    double min_length = 0;
    double max_time = DBL_MAX;
    npy_intp *shape;
    static char *kwlist[] = { "samples", "min_length", "max_time", NULL };

    memset(&ibd_finder, 0, sizeof(ibd_finder));
    if (TableCollection_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "O|dd", kwlist, &samples, &min_length, &max_time)) {
        goto out;
    }
    samples_array = (PyArrayObject *) PyArray_FROMANY(
        samples, NPY_INT32, 2, 2, NPY_ARRAY_IN_ARRAY);
    if (samples_array == NULL) {
        goto out;
    }
    shape = PyArray_DIMS(samples_array);
    if (shape[1] != 2) {
        PyErr_SetString(PyExc_ValueError, "sample pairs must have shape (n, 2)");
        goto out;
    }
    err = tsk_ibd_finder_init(
        &ibd_finder, self->tables, PyArray_DATA(samples_array), (tsk_size_t) shape[0]);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    err = tsk_ibd_finder_set_min_length(&ibd_finder, min_length);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    err = tsk_ibd_finder_set_max_time(&ibd_finder, max_time);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    err = tsk_ibd_finder_run(&ibd_finder);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = convert_ibd_segments(
        &ibd_finder, PyArray_DATA(samples_array), (tsk_size_t) shape[0]);
out:
    Py_XDECREF(samples_array);
    tsk_ibd_finder_free(&ibd_finder);
    return ret;
}

static PyObject *
TableCollection_sort(TableCollection *self, PyObject *args, PyObject *kwds)
{
    int err;
    PyObject *ret = NULL;
    Py_ssize_t edge_start = 0;
    tsk_bookmark_t start;
    static char *kwlist[] = { "edge_start", NULL };

    if (TableCollection_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|n", kwlist, &edge_start)) {
        goto out;
    }
    memset(&start, 0, sizeof(start));
    start.edges = edge_start;
    err = tsk_table_collection_sort(self->tables, &start, 0);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
TableCollection_compute_mutation_parents(TableCollection *self)
{
    int err;
    PyObject *ret = NULL;

    if (TableCollection_check_state(self) != 0) {
        goto out;
    }
    err = tsk_table_collection_compute_mutation_parents(self->tables, 0);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
TableCollection_compute_mutation_times(TableCollection *self)
{
    int err;
    PyObject *ret = NULL;

    if (TableCollection_check_state(self) != 0) {
        goto out;
    }
    err = tsk_table_collection_compute_mutation_times(self->tables, NULL, 0);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
TableCollection_deduplicate_sites(TableCollection *self)
{
    int err;
    PyObject *ret = NULL;

    if (TableCollection_check_state(self) != 0) {
        goto out;
    }
    err = tsk_table_collection_deduplicate_sites(self->tables, 0);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
TableCollection_build_index(TableCollection *self)
{
    int err;
    PyObject *ret = NULL;

    if (TableCollection_check_state(self) != 0) {
        goto out;
    }
    err = tsk_table_collection_build_index(self->tables, 0);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
TableCollection_drop_index(TableCollection *self)
{
    int err;
    PyObject *ret = NULL;

    if (TableCollection_check_state(self) != 0) {
        goto out;
    }
    err = tsk_table_collection_drop_index(self->tables, 0);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
TableCollection_get_indexes(TableCollection *self, void *closure)
{
    PyObject *ret = NULL;
    PyObject *indexes_dict = NULL;
    PyObject *insertion = NULL;
    PyObject *removal = NULL;

    if (TableCollection_check_state(self) != 0) {
        goto out;
    }

    indexes_dict = PyDict_New();
    if (indexes_dict == NULL) {
        goto out;
    }

    if (tsk_table_collection_has_index(self->tables, 0)) {
        insertion = table_get_column_array(self->tables->indexes.num_edges,
            self->tables->indexes.edge_insertion_order, NPY_INT32, sizeof(tsk_id_t));
        if (insertion == NULL) {
            goto out;
        }
        removal = table_get_column_array(self->tables->indexes.num_edges,
            self->tables->indexes.edge_removal_order, NPY_INT32, sizeof(tsk_id_t));
        if (removal == NULL) {
            goto out;
        }

        if (PyDict_SetItemString(indexes_dict, "edge_insertion_order", insertion) != 0) {
            goto out;
        }
        if (PyDict_SetItemString(indexes_dict, "edge_removal_order", removal) != 0) {
            goto out;
        }
    }

    ret = indexes_dict;
    indexes_dict = NULL;
out:
    Py_XDECREF(indexes_dict);
    Py_XDECREF(insertion);
    Py_XDECREF(removal);
    return ret;
}

static int
TableCollection_set_indexes(TableCollection *self, PyObject *arg, void *closure)
{
    int err;
    int ret = -1;

    if (TableCollection_check_state(self) != 0) {
        goto out;
    }

    err = parse_indexes_dict(self->tables, arg);
    if (err != 0) {
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static PyObject *
TableCollection_has_index(TableCollection *self)
{
    PyObject *ret = NULL;

    if (TableCollection_check_state(self) != 0) {
        goto out;
    }
    bool has_index = tsk_table_collection_has_index(self->tables, 0);
    ret = Py_BuildValue("i", (int) has_index);
out:
    return ret;
}

static PyObject *
TableCollection_equals(TableCollection *self, PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    TableCollection *other = NULL;
    tsk_flags_t options = 0;
    int ignore_metadata = false;
    int ignore_ts_metadata = false;
    int ignore_provenance = false;
    int ignore_timestamps = true;
    static char *kwlist[] = { "other", "ignore_metadata", "ignore_ts_metadata",
        "ignore_provenance", "ignore_timestamps", NULL };

    if (TableCollection_check_state(self)) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|iiii", kwlist, &TableCollectionType,
            &other, &ignore_metadata, &ignore_ts_metadata, &ignore_provenance,
            &ignore_timestamps)) {
        goto out;
    }
    if (ignore_metadata) {
        options |= TSK_CMP_IGNORE_METADATA;
    }
    if (ignore_ts_metadata) {
        options |= TSK_CMP_IGNORE_TS_METADATA;
    }
    if (ignore_provenance) {
        options |= TSK_CMP_IGNORE_PROVENANCE;
    }
    if (ignore_timestamps) {
        options |= TSK_CMP_IGNORE_TIMESTAMPS;
    }
    if (TableCollection_check_state(other) != 0) {
        goto out;
    }
    ret = Py_BuildValue(
        "i", tsk_table_collection_equals(self->tables, other->tables, options));
out:
    return ret;
}

static PyObject *
TableCollection_clear(TableCollection *self, PyObject *args, PyObject *kwds)
{
    int err;
    PyObject *ret = NULL;
    tsk_flags_t options = 0;
    int clear_provenance = false;
    int clear_metadata_schemas = false;
    int clear_ts_metadata = false;
    static char *kwlist[] = { "clear_provenance", "clear_metadata_schemas",
        "clear_ts_metadata_and_schema", NULL };

    if (TableCollection_check_state(self)) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iii", kwlist, &clear_provenance,
            &clear_metadata_schemas, &clear_ts_metadata)) {
        goto out;
    }
    if (clear_provenance) {
        options |= TSK_CLEAR_PROVENANCE;
    }
    if (clear_metadata_schemas) {
        options |= TSK_CLEAR_METADATA_SCHEMAS;
    }
    if (clear_ts_metadata) {
        options |= TSK_CLEAR_TS_METADATA_AND_SCHEMA;
    }

    err = tsk_table_collection_clear(self->tables, options);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
TableCollection_dump(TableCollection *self, PyObject *args, PyObject *kwds)
{
    int err;
    FILE *file = NULL;
    PyObject *py_file = NULL;
    PyObject *ret = NULL;
    static char *kwlist[] = { "file", NULL };

    if (TableCollection_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &py_file)) {
        goto out;
    }

    file = make_file(py_file, "wb");
    if (file == NULL) {
        goto out;
    }

    err = tsk_table_collection_dumpf(self->tables, file, 0);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    if (file != NULL) {
        (void) fclose(file);
    }
    return ret;
}

static PyObject *
TableCollection_load(TableCollection *self, PyObject *args, PyObject *kwds)
{
    int err;
    PyObject *ret = NULL;
    PyObject *py_file;
    FILE *file = NULL;
    static char *kwlist[] = { "file", NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &py_file)) {
        goto out;
    }
    file = make_file(py_file, "rb");
    if (file == NULL) {
        goto out;
    }
    /* Set unbuffered mode to ensure no more bytes are read than requested.
     * Buffered reads could read beyond the end of the current store in a
     * multi-store file or stream. This data would be discarded when we
     * fclose() the file below, such that attempts to load the next store
     * will fail. */
    if (setvbuf(file, NULL, _IONBF, 0) != 0) {
        PyErr_SetFromErrno(PyExc_OSError);
        goto out;
    }
    err = TableCollection_alloc(self);
    if (err != 0) {
        goto out;
    }
    err = tsk_table_collection_loadf(self->tables, file, 0);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    if (file != NULL) {
        (void) fclose(file);
    }
    return ret;
}

static PyGetSetDef TableCollection_getsetters[] = {
    { .name = "individuals",
        .get = (getter) TableCollection_get_individuals,
        .doc = "The individual table." },
    { .name = "nodes",
        .get = (getter) TableCollection_get_nodes,
        .doc = "The node table." },
    { .name = "edges",
        .get = (getter) TableCollection_get_edges,
        .doc = "The edge table." },
    { .name = "migrations",
        .get = (getter) TableCollection_get_migrations,
        .doc = "The migration table." },
    { .name = "sites",
        .get = (getter) TableCollection_get_sites,
        .doc = "The site table." },
    { .name = "mutations",
        .get = (getter) TableCollection_get_mutations,
        .doc = "The mutation table." },
    { .name = "populations",
        .get = (getter) TableCollection_get_populations,
        .doc = "The population table." },
    { .name = "provenances",
        .get = (getter) TableCollection_get_provenances,
        .doc = "The provenance table." },
    { .name = "indexes",
        .get = (getter) TableCollection_get_indexes,
        .set = (setter) TableCollection_set_indexes,
        .doc = "The indexes." },
    { .name = "sequence_length",
        .get = (getter) TableCollection_get_sequence_length,
        .set = (setter) TableCollection_set_sequence_length,
        .doc = "The sequence length." },
    { .name = "file_uuid",
        .get = (getter) TableCollection_get_file_uuid,
        .doc = "The UUID of the corresponding file." },
    { .name = "metadata",
        .get = (getter) TableCollection_get_metadata,
        .set = (setter) TableCollection_set_metadata,
        .doc = "The metadata." },
    { .name = "metadata_schema",
        .get = (getter) TableCollection_get_metadata_schema,
        .set = (setter) TableCollection_set_metadata_schema,
        .doc = "The metadata schema." },
    { NULL } /* Sentinel */
};

static PyMethodDef TableCollection_methods[] = {
    { .ml_name = "simplify",
        .ml_meth = (PyCFunction) TableCollection_simplify,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Simplifies for a given sample subset." },
    { .ml_name = "link_ancestors",
        .ml_meth = (PyCFunction) TableCollection_link_ancestors,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc
        = "Returns an edge table linking samples to a set of specified ancestors." },
    { .ml_name = "subset",
        .ml_meth = (PyCFunction) TableCollection_subset,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Subsets the table collection to a set of nodes." },
    { .ml_name = "union",
        .ml_meth = (PyCFunction) TableCollection_union,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc
        = "Adds to this table collection the portions of another table collection "
          "that are not shared with this one." },
    { .ml_name = "find_ibd",
        .ml_meth = (PyCFunction) TableCollection_find_ibd,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Returns IBD segments for the specified sample pairs." },
    { .ml_name = "sort",
        .ml_meth = (PyCFunction) TableCollection_sort,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Sorts the tables to satisfy tree sequence requirements." },
    { .ml_name = "equals",
        .ml_meth = (PyCFunction) TableCollection_equals,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc
        = "Returns True if the parameter table collection is equal to this one." },
    { .ml_name = "compute_mutation_parents",
        .ml_meth = (PyCFunction) TableCollection_compute_mutation_parents,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Computes the mutation parents for the tables." },
    { .ml_name = "compute_mutation_times",
        .ml_meth = (PyCFunction) TableCollection_compute_mutation_times,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Computes the mutation times for the tables." },
    { .ml_name = "deduplicate_sites",
        .ml_meth = (PyCFunction) TableCollection_deduplicate_sites,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Removes sites with duplicate positions." },
    { .ml_name = "build_index",
        .ml_meth = (PyCFunction) TableCollection_build_index,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Builds an index on the table collection." },
    { .ml_name = "drop_index",
        .ml_meth = (PyCFunction) TableCollection_drop_index,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Drops indexes." },
    { .ml_name = "has_index",
        .ml_meth = (PyCFunction) TableCollection_has_index,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns True if the TableCollection is indexed." },
    { .ml_name = "clear",
        .ml_meth = (PyCFunction) TableCollection_clear,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Clears table contents, and optionally provenances and metadata" },
    { .ml_name = "dump",
        .ml_meth = (PyCFunction) TableCollection_dump,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Writes the table collection out to the specified file." },
    { .ml_name = "load",
        .ml_meth = (PyCFunction) TableCollection_load,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Loads the table collection out to the specified file." },
    { NULL } /* Sentinel */
};

static PyTypeObject TableCollectionType = {
    // clang-format off
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_tskit.TableCollection",
    .tp_basicsize = sizeof(TableCollection),
    .tp_dealloc = (destructor) TableCollection_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "TableCollection objects",
    .tp_methods = TableCollection_methods,
    .tp_getset = TableCollection_getsetters,
    .tp_init = (initproc) TableCollection_init,
    .tp_new = PyType_GenericNew,
    // clang-format on
};

/*===================================================================
 * TreeSequence
 *===================================================================
 */

static int
TreeSequence_check_state(TreeSequence *self)
{
    int ret = 0;
    if (self->tree_sequence == NULL) {
        PyErr_SetString(PyExc_ValueError, "tree_sequence not initialised");
        ret = -1;
    }
    return ret;
}

static void
TreeSequence_dealloc(TreeSequence *self)
{
    if (self->tree_sequence != NULL) {
        tsk_treeseq_free(self->tree_sequence);
        PyMem_Free(self->tree_sequence);
        self->tree_sequence = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static int
TreeSequence_alloc(TreeSequence *self)
{
    int ret = -1;

    if (self->tree_sequence != NULL) {
        tsk_treeseq_free(self->tree_sequence);
        PyMem_Free(self->tree_sequence);
    }
    self->tree_sequence = PyMem_Malloc(sizeof(tsk_treeseq_t));
    if (self->tree_sequence == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    memset(self->tree_sequence, 0, sizeof(*self->tree_sequence));
    ret = 0;
out:
    return ret;
}

static int
TreeSequence_init(TreeSequence *self, PyObject *args, PyObject *kwds)
{
    self->tree_sequence = NULL;
    return 0;
}

static PyObject *
TreeSequence_dump(TreeSequence *self, PyObject *args, PyObject *kwds)
{
    int err;
    FILE *file = NULL;
    PyObject *py_file = NULL;
    PyObject *ret = NULL;
    static char *kwlist[] = { "file", NULL };

    if (TreeSequence_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &py_file)) {
        goto out;
    }

    file = make_file(py_file, "wb");
    if (file == NULL) {
        goto out;
    }

    err = tsk_treeseq_dumpf(self->tree_sequence, file, 0);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    if (file != NULL) {
        (void) fclose(file);
    }
    return ret;
}

static PyObject *
TreeSequence_load_tables(TreeSequence *self, PyObject *args, PyObject *kwds)
{
    int err;
    PyObject *ret = NULL;
    TableCollection *tables = NULL;
    static char *kwlist[] = { "tables", "build_indexes", NULL };
    int build_indexes = false;
    tsk_flags_t options = 0;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "O!|i", kwlist, &TableCollectionType, &tables, &build_indexes)) {
        goto out;
    }
    err = TreeSequence_alloc(self);
    if (err != 0) {
        goto out;
    }
    if (build_indexes) {
        options |= TSK_BUILD_INDEXES;
    }
    err = tsk_treeseq_init(self->tree_sequence, tables->tables, options);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
TreeSequence_dump_tables(TreeSequence *self, PyObject *args, PyObject *kwds)
{
    int err;
    PyObject *ret = NULL;
    TableCollection *tables = NULL;
    static char *kwlist[] = { "tables", NULL };

    if (TreeSequence_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "O!", kwlist, &TableCollectionType, &tables)) {
        goto out;
    }
    err = tsk_treeseq_copy_tables(self->tree_sequence, tables->tables, TSK_NO_INIT);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
TreeSequence_load(TreeSequence *self, PyObject *args, PyObject *kwds)
{
    int err;
    PyObject *ret = NULL;
    PyObject *py_file;
    FILE *file = NULL;
    static char *kwlist[] = { "file", NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &py_file)) {
        goto out;
    }
    file = make_file(py_file, "rb");
    if (file == NULL) {
        goto out;
    }
    /* Set unbuffered mode to ensure no more bytes are read than requested.
     * Buffered reads could read beyond the end of the current store in a
     * multi-store file or stream. This data would be discarded when we
     * fclose() the file below, such that attempts to load the next store
     * will fail. */
    if (setvbuf(file, NULL, _IONBF, 0) != 0) {
        PyErr_SetFromErrno(PyExc_OSError);
        goto out;
    }
    err = TreeSequence_alloc(self);
    if (err != 0) {
        goto out;
    }
    err = tsk_treeseq_loadf(self->tree_sequence, file, 0);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    if (file != NULL) {
        (void) fclose(file);
    }
    return ret;
}

static PyObject *
TreeSequence_get_node(TreeSequence *self, PyObject *args)
{
    int err;
    PyObject *ret = NULL;
    Py_ssize_t record_index, num_records;
    tsk_node_t record;

    if (TreeSequence_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "n", &record_index)) {
        goto out;
    }
    num_records = (Py_ssize_t) tsk_treeseq_get_num_nodes(self->tree_sequence);
    if (record_index < 0 || record_index >= num_records) {
        PyErr_SetString(PyExc_IndexError, "record index out of bounds");
        goto out;
    }
    err = tsk_treeseq_get_node(self->tree_sequence, (size_t) record_index, &record);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = make_node(&record);
out:
    return ret;
}

static PyObject *
TreeSequence_get_edge(TreeSequence *self, PyObject *args)
{
    int err;
    PyObject *ret = NULL;
    Py_ssize_t record_index, num_records;
    tsk_edge_t record;

    if (TreeSequence_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "n", &record_index)) {
        goto out;
    }
    num_records = (Py_ssize_t) tsk_treeseq_get_num_edges(self->tree_sequence);
    if (record_index < 0 || record_index >= num_records) {
        PyErr_SetString(PyExc_IndexError, "record index out of bounds");
        goto out;
    }
    err = tsk_treeseq_get_edge(self->tree_sequence, (size_t) record_index, &record);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = make_edge(&record, false);
out:
    return ret;
}

static PyObject *
TreeSequence_get_migration(TreeSequence *self, PyObject *args)
{
    int err;
    PyObject *ret = NULL;
    Py_ssize_t record_index, num_records;
    tsk_migration_t record;

    if (TreeSequence_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "n", &record_index)) {
        goto out;
    }
    num_records = (Py_ssize_t) tsk_treeseq_get_num_migrations(self->tree_sequence);
    if (record_index < 0 || record_index >= num_records) {
        PyErr_SetString(PyExc_IndexError, "record index out of bounds");
        goto out;
    }
    err = tsk_treeseq_get_migration(self->tree_sequence, (size_t) record_index, &record);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = make_migration(&record);
out:
    return ret;
}

static PyObject *
TreeSequence_get_site(TreeSequence *self, PyObject *args)
{
    int err;
    PyObject *ret = NULL;
    Py_ssize_t record_index, num_records;
    tsk_site_t record;

    if (TreeSequence_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "n", &record_index)) {
        goto out;
    }
    num_records = (Py_ssize_t) tsk_treeseq_get_num_sites(self->tree_sequence);
    if (record_index < 0 || record_index >= num_records) {
        PyErr_SetString(PyExc_IndexError, "record index out of bounds");
        goto out;
    }
    err = tsk_treeseq_get_site(self->tree_sequence, (tsk_id_t) record_index, &record);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = make_site_object(&record);
out:
    return ret;
}

static PyObject *
TreeSequence_get_metadata(TreeSequence *self)
{
    PyObject *ret = NULL;

    if (TreeSequence_check_state(self) != 0) {
        goto out;
    }
    ret = PyBytes_FromStringAndSize(self->tree_sequence->tables->metadata,
        self->tree_sequence->tables->metadata_length);
out:
    return ret;
}

static PyObject *
TreeSequence_get_metadata_schema(TreeSequence *self)
{
    PyObject *ret = NULL;

    if (TreeSequence_check_state(self) != 0) {
        goto out;
    }
    ret = make_Py_Unicode_FromStringAndLength(
        self->tree_sequence->tables->metadata_schema,
        self->tree_sequence->tables->metadata_schema_length);
out:
    return ret;
}

static PyObject *
TreeSequence_get_table_metadata_schemas(TreeSequence *self)
{
    PyObject *ret = NULL;
    PyObject *value = NULL;
    PyObject *schema = NULL;
    size_t j;
    tsk_table_collection_t *tables;
    struct schema_pair {
        const char *schema;
        tsk_size_t length;
    };

    if (TreeSequence_check_state(self) != 0) {
        goto out;
    }
    tables = self->tree_sequence->tables;
    struct schema_pair schema_pairs[] = {
        { tables->nodes.metadata_schema, tables->nodes.metadata_schema_length },
        { tables->edges.metadata_schema, tables->edges.metadata_schema_length },
        { tables->sites.metadata_schema, tables->sites.metadata_schema_length },
        { tables->mutations.metadata_schema, tables->mutations.metadata_schema_length },
        { tables->migrations.metadata_schema,
            tables->migrations.metadata_schema_length },
        { tables->individuals.metadata_schema,
            tables->individuals.metadata_schema_length },
        { tables->populations.metadata_schema,
            tables->populations.metadata_schema_length },
    };
    value = PyStructSequence_New(&MetadataSchemas);
    if (value == NULL) {
        goto out;
    }
    for (j = 0; j < sizeof(schema_pairs) / sizeof(*schema_pairs); j++) {
        schema = make_Py_Unicode_FromStringAndLength(
            schema_pairs[j].schema, schema_pairs[j].length);
        if (schema == NULL) {
            goto out;
        }
        PyStructSequence_SetItem(value, j, schema);
    }
    ret = value;
    value = NULL;
out:
    Py_XDECREF(value);
    return ret;
}

static PyObject *
TreeSequence_get_mutation(TreeSequence *self, PyObject *args)
{
    int err;
    PyObject *ret = NULL;
    Py_ssize_t record_index, num_records;
    tsk_mutation_t record;

    if (TreeSequence_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "n", &record_index)) {
        goto out;
    }
    num_records = (Py_ssize_t) tsk_treeseq_get_num_mutations(self->tree_sequence);
    if (record_index < 0 || record_index >= num_records) {
        PyErr_SetString(PyExc_IndexError, "record index out of bounds");
        goto out;
    }
    err = tsk_treeseq_get_mutation(self->tree_sequence, (size_t) record_index, &record);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = make_mutation(&record);
out:
    return ret;
}

static PyObject *
TreeSequence_get_individual(TreeSequence *self, PyObject *args)
{
    int err;
    PyObject *ret = NULL;
    Py_ssize_t record_index, num_records;
    tsk_individual_t record;

    if (TreeSequence_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "n", &record_index)) {
        goto out;
    }
    num_records = (Py_ssize_t) tsk_treeseq_get_num_individuals(self->tree_sequence);
    if (record_index < 0 || record_index >= num_records) {
        PyErr_SetString(PyExc_IndexError, "record index out of bounds");
        goto out;
    }
    err = tsk_treeseq_get_individual(
        self->tree_sequence, (size_t) record_index, &record);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = make_individual_object(&record);
out:
    return ret;
}

static PyObject *
TreeSequence_get_population(TreeSequence *self, PyObject *args)
{
    int err;
    PyObject *ret = NULL;
    Py_ssize_t record_index, num_records;
    tsk_population_t record;

    if (TreeSequence_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "n", &record_index)) {
        goto out;
    }
    num_records = (Py_ssize_t) tsk_treeseq_get_num_populations(self->tree_sequence);
    if (record_index < 0 || record_index >= num_records) {
        PyErr_SetString(PyExc_IndexError, "record index out of bounds");
        goto out;
    }
    err = tsk_treeseq_get_population(
        self->tree_sequence, (size_t) record_index, &record);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = make_population(&record);
out:
    return ret;
}

static PyObject *
TreeSequence_get_provenance(TreeSequence *self, PyObject *args)
{
    int err;
    PyObject *ret = NULL;
    Py_ssize_t record_index, num_records;
    tsk_provenance_t record;

    if (TreeSequence_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "n", &record_index)) {
        goto out;
    }
    num_records = (Py_ssize_t) tsk_treeseq_get_num_provenances(self->tree_sequence);
    if (record_index < 0 || record_index >= num_records) {
        PyErr_SetString(PyExc_IndexError, "record index out of bounds");
        goto out;
    }
    err = tsk_treeseq_get_provenance(
        self->tree_sequence, (size_t) record_index, &record);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = make_provenance(&record);
out:
    return ret;
}

static PyObject *
TreeSequence_get_num_edges(TreeSequence *self, PyObject *args)
{
    PyObject *ret = NULL;
    size_t num_records;

    if (TreeSequence_check_state(self) != 0) {
        goto out;
    }
    num_records = tsk_treeseq_get_num_edges(self->tree_sequence);
    ret = Py_BuildValue("n", (Py_ssize_t) num_records);
out:
    return ret;
}

static PyObject *
TreeSequence_get_num_migrations(TreeSequence *self, PyObject *args)
{
    PyObject *ret = NULL;
    size_t num_records;

    if (TreeSequence_check_state(self) != 0) {
        goto out;
    }
    num_records = tsk_treeseq_get_num_migrations(self->tree_sequence);
    ret = Py_BuildValue("n", (Py_ssize_t) num_records);
out:
    return ret;
}

static PyObject *
TreeSequence_get_num_individuals(TreeSequence *self, PyObject *args)
{
    PyObject *ret = NULL;
    size_t num_records;

    if (TreeSequence_check_state(self) != 0) {
        goto out;
    }
    num_records = tsk_treeseq_get_num_individuals(self->tree_sequence);
    ret = Py_BuildValue("n", (Py_ssize_t) num_records);
out:
    return ret;
}

static PyObject *
TreeSequence_get_num_populations(TreeSequence *self, PyObject *args)
{
    PyObject *ret = NULL;
    size_t num_records;

    if (TreeSequence_check_state(self) != 0) {
        goto out;
    }
    num_records = tsk_treeseq_get_num_populations(self->tree_sequence);
    ret = Py_BuildValue("n", (Py_ssize_t) num_records);
out:
    return ret;
}

static PyObject *
TreeSequence_get_num_trees(TreeSequence *self, PyObject *args)
{
    PyObject *ret = NULL;
    size_t num_trees;

    if (TreeSequence_check_state(self) != 0) {
        goto out;
    }
    num_trees = tsk_treeseq_get_num_trees(self->tree_sequence);
    ret = Py_BuildValue("n", (Py_ssize_t) num_trees);
out:
    return ret;
}

static PyObject *
TreeSequence_get_sequence_length(TreeSequence *self)
{
    PyObject *ret = NULL;

    if (TreeSequence_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("d", tsk_treeseq_get_sequence_length(self->tree_sequence));
out:
    return ret;
}

static PyObject *
TreeSequence_get_breakpoints(TreeSequence *self)
{
    PyObject *ret = NULL;
    const double *breakpoints;
    PyArrayObject *array = NULL;
    npy_intp dims;

    if (TreeSequence_check_state(self) != 0) {
        goto out;
    }
    breakpoints = tsk_treeseq_get_breakpoints(self->tree_sequence);
    dims = tsk_treeseq_get_num_trees(self->tree_sequence) + 1;
    array = (PyArrayObject *) PyArray_SimpleNew(1, &dims, NPY_FLOAT64);
    if (array == NULL) {
        goto out;
    }
    memcpy(PyArray_DATA(array), breakpoints, dims * sizeof(*breakpoints));
    ret = (PyObject *) array;
    array = NULL;
out:
    Py_XDECREF(array);
    return ret;
}

static PyObject *
TreeSequence_get_file_uuid(TreeSequence *self)
{
    PyObject *ret = NULL;

    if (TreeSequence_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("s", tsk_treeseq_get_file_uuid(self->tree_sequence));
out:
    return ret;
}

static PyObject *
TreeSequence_get_num_samples(TreeSequence *self)
{
    PyObject *ret = NULL;
    size_t num_samples;

    if (TreeSequence_check_state(self) != 0) {
        goto out;
    }
    num_samples = tsk_treeseq_get_num_samples(self->tree_sequence);
    ret = Py_BuildValue("n", (Py_ssize_t) num_samples);
out:
    return ret;
}

static PyObject *
TreeSequence_get_num_nodes(TreeSequence *self)
{
    PyObject *ret = NULL;
    size_t num_nodes;

    if (TreeSequence_check_state(self) != 0) {
        goto out;
    }
    num_nodes = tsk_treeseq_get_num_nodes(self->tree_sequence);
    ret = Py_BuildValue("n", (Py_ssize_t) num_nodes);
out:
    return ret;
}

static PyObject *
TreeSequence_get_samples(TreeSequence *self)
{
    PyObject *ret = NULL;
    const tsk_id_t *samples;
    PyArrayObject *samples_array = NULL;
    npy_intp dim;

    if (TreeSequence_check_state(self) != 0) {
        goto out;
    }
    dim = tsk_treeseq_get_num_samples(self->tree_sequence);
    samples = tsk_treeseq_get_samples(self->tree_sequence);

    /* TODO it would be nice to return a read-only array that points to the
     * tree sequence's memory and to INCREF ts to ensure the pointer stays
     * alive. The details are tricky though. */
    samples_array = (PyArrayObject *) PyArray_SimpleNew(1, &dim, NPY_INT32);
    if (samples_array == NULL) {
        goto out;
    }
    memcpy(PyArray_DATA(samples_array), samples, dim * sizeof(*samples));
    ret = (PyObject *) samples_array;
    samples_array = NULL;
out:
    Py_XDECREF(samples_array);
    return ret;
}

static PyObject *
TreeSequence_genealogical_nearest_neighbours(
    TreeSequence *self, PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    static char *kwlist[] = { "focal", "reference_sets", NULL };
    const tsk_id_t **reference_sets = NULL;
    size_t *reference_set_size = NULL;
    PyObject *focal = NULL;
    PyObject *reference_sets_list = NULL;
    PyArrayObject *focal_array = NULL;
    PyArrayObject **reference_set_arrays = NULL;
    PyArrayObject *ret_array = NULL;
    npy_intp *shape, dims[2];
    size_t num_focal = 0;
    size_t num_reference_sets = 0;
    size_t j;
    int err;

    if (TreeSequence_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "OO!", kwlist, &focal, &PyList_Type, &reference_sets_list)) {
        goto out;
    }

    /* We're releasing the GIL here so we need to make sure that the memory we
     * pass to the low-level code doesn't change while it's in use. This is
     * why we take copies of the input arrays. */
    focal_array = (PyArrayObject *) PyArray_FROMANY(
        focal, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSURECOPY);
    if (focal_array == NULL) {
        goto out;
    }
    shape = PyArray_DIMS(focal_array);
    num_focal = shape[0];
    num_reference_sets = PyList_Size(reference_sets_list);
    if (num_reference_sets == 0) {
        PyErr_SetString(PyExc_ValueError, "Must have at least one sample set");
        goto out;
    }
    reference_set_size = PyMem_Malloc(num_reference_sets * sizeof(*reference_set_size));
    reference_sets = PyMem_Malloc(num_reference_sets * sizeof(*reference_sets));
    reference_set_arrays
        = PyMem_Malloc(num_reference_sets * sizeof(*reference_set_arrays));
    if (reference_sets == NULL || reference_set_size == NULL
        || reference_set_arrays == NULL) {
        goto out;
    }
    memset(reference_set_arrays, 0, num_reference_sets * sizeof(*reference_set_arrays));
    for (j = 0; j < num_reference_sets; j++) {
        reference_set_arrays[j]
            = (PyArrayObject *) PyArray_FROMANY(PyList_GetItem(reference_sets_list, j),
                NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSURECOPY);
        if (reference_set_arrays[j] == NULL) {
            goto out;
        }
        reference_sets[j] = PyArray_DATA(reference_set_arrays[j]);
        shape = PyArray_DIMS(reference_set_arrays[j]);
        reference_set_size[j] = shape[0];
    }

    /* Allocate the return array */
    dims[0] = num_focal;
    dims[1] = num_reference_sets;
    ret_array = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT64);
    if (ret_array == NULL) {
        goto out;
    }

    Py_BEGIN_ALLOW_THREADS err = tsk_treeseq_genealogical_nearest_neighbours(
        self->tree_sequence, PyArray_DATA(focal_array), num_focal, reference_sets,
        reference_set_size, num_reference_sets, 0, PyArray_DATA(ret_array));
    Py_END_ALLOW_THREADS if (err != 0)
    {
        handle_library_error(err);
        goto out;
    }

    ret = (PyObject *) ret_array;
    ret_array = NULL;
out:
    if (reference_sets != NULL) {
        PyMem_Free(reference_sets);
    }
    if (reference_set_size != NULL) {
        PyMem_Free(reference_set_size);
    }
    if (reference_set_arrays != NULL) {
        for (j = 0; j < num_reference_sets; j++) {
            Py_XDECREF(reference_set_arrays[j]);
        }
        PyMem_Free(reference_set_arrays);
    }
    Py_XDECREF(focal_array);
    Py_XDECREF(ret_array);
    return ret;
}

/* Forward Declaration */
static PyTypeObject TreeSequenceType;

static PyObject *
TreeSequence_get_kc_distance(TreeSequence *self, PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    TreeSequence *other = NULL;
    static char *kwlist[] = { "other", "lambda_", NULL };
    double lambda = 0;
    double result = 0;
    int err;

    if (TreeSequence_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "O!d", kwlist, &TreeSequenceType, &other, &lambda)) {
        goto out;
    }
    err = tsk_treeseq_kc_distance(
        self->tree_sequence, other->tree_sequence, lambda, &result);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("d", result);
out:
    return ret;
}

static PyObject *
TreeSequence_mean_descendants(TreeSequence *self, PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    static char *kwlist[] = { "reference_sets", NULL };
    const tsk_id_t **reference_sets = NULL;
    size_t *reference_set_size = NULL;
    PyObject *reference_sets_list = NULL;
    PyArrayObject **reference_set_arrays = NULL;
    PyArrayObject *ret_array = NULL;
    npy_intp *shape, dims[2];
    size_t num_reference_sets = 0;
    size_t j;
    int err;

    if (TreeSequence_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "O!", kwlist, &PyList_Type, &reference_sets_list)) {
        goto out;
    }

    num_reference_sets = PyList_Size(reference_sets_list);
    if (num_reference_sets == 0) {
        PyErr_SetString(PyExc_ValueError, "Must have at least one sample set");
        goto out;
    }
    reference_set_size = PyMem_Malloc(num_reference_sets * sizeof(*reference_set_size));
    reference_sets = PyMem_Malloc(num_reference_sets * sizeof(*reference_sets));
    reference_set_arrays
        = PyMem_Malloc(num_reference_sets * sizeof(*reference_set_arrays));
    if (reference_sets == NULL || reference_set_size == NULL
        || reference_set_arrays == NULL) {
        goto out;
    }
    memset(reference_set_arrays, 0, num_reference_sets * sizeof(*reference_set_arrays));
    for (j = 0; j < num_reference_sets; j++) {
        /* We're releasing the GIL here so we need to make sure that the memory we
         * pass to the low-level code doesn't change while it's in use. This is
         * why we take copies of the input arrays. */
        reference_set_arrays[j]
            = (PyArrayObject *) PyArray_FROMANY(PyList_GetItem(reference_sets_list, j),
                NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSURECOPY);
        if (reference_set_arrays[j] == NULL) {
            goto out;
        }
        reference_sets[j] = PyArray_DATA(reference_set_arrays[j]);
        shape = PyArray_DIMS(reference_set_arrays[j]);
        reference_set_size[j] = shape[0];
    }

    /* Allocate the return array */
    dims[0] = tsk_treeseq_get_num_nodes(self->tree_sequence);
    dims[1] = num_reference_sets;
    ret_array = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT64);
    if (ret_array == NULL) {
        goto out;
    }

    Py_BEGIN_ALLOW_THREADS err
        = tsk_treeseq_mean_descendants(self->tree_sequence, reference_sets,
            reference_set_size, num_reference_sets, 0, PyArray_DATA(ret_array));
    Py_END_ALLOW_THREADS if (err != 0)
    {
        handle_library_error(err);
        goto out;
    }

    ret = (PyObject *) ret_array;
    ret_array = NULL;
out:
    if (reference_sets != NULL) {
        PyMem_Free(reference_sets);
    }
    if (reference_set_size != NULL) {
        PyMem_Free(reference_set_size);
    }
    if (reference_set_arrays != NULL) {
        for (j = 0; j < num_reference_sets; j++) {
            Py_XDECREF(reference_set_arrays[j]);
        }
        PyMem_Free(reference_set_arrays);
    }
    Py_XDECREF(ret_array);
    return ret;
}

/* Error value returned from summary_func callback if an error occured.
 * This is chosen so that it is not a valid tskit error code and so can
 * never be mistaken for a different error */
#define TSK_PYTHON_CALLBACK_ERROR (-100000)

/* Run the Python callable that takes X as parameter and must return a
 * 1D array of length M that we copy in to the Y array */
static int
general_stat_func(size_t K, const double *X, size_t M, double *Y, void *params)
{
    int ret = TSK_PYTHON_CALLBACK_ERROR;
    PyObject *callable = (PyObject *) params;
    PyObject *arglist = NULL;
    PyObject *result = NULL;
    PyArrayObject *X_array = NULL;
    PyArrayObject *Y_array = NULL;
    npy_intp X_dims = (npy_intp) K;
    npy_intp *Y_dims;

    X_array = (PyArrayObject *) PyArray_SimpleNew(1, &X_dims, NPY_FLOAT64);
    if (X_array == NULL) {
        goto out;
    }
    memcpy(PyArray_DATA(X_array), X, K * sizeof(*X));
    arglist = Py_BuildValue("(O)", X_array);
    if (arglist == NULL) {
        goto out;
    }
    result = PyObject_CallObject(callable, arglist);
    if (result == NULL) {
        goto out;
    }
    Y_array = (PyArrayObject *) PyArray_FromAny(
        result, PyArray_DescrFromType(NPY_FLOAT64), 1, 1, NPY_ARRAY_IN_ARRAY, NULL);
    if (Y_array == NULL) {
        goto out;
    }
    Y_dims = PyArray_DIMS(Y_array);
    if (Y_dims[0] != (npy_intp) M) {
        PyErr_SetString(PyExc_ValueError, "Incorrect callback output dimensions");
        goto out;
    }
    /* Copy the contents of the return Y array into Y */
    memcpy(Y, PyArray_DATA(Y_array), M * sizeof(*Y));
    ret = 0;
out:
    Py_XDECREF(X_array);
    Py_XDECREF(arglist);
    Py_XDECREF(result);
    Py_XDECREF(Y_array);
    return ret;
}

static int
parse_stats_mode(char *mode, tsk_flags_t *ret)
{
    tsk_flags_t value = 0;

    if (mode == NULL) {
        value = TSK_STAT_SITE; /* defaults to site mode */
    } else if (strcmp(mode, "site") == 0) {
        value = TSK_STAT_SITE;
    } else if (strcmp(mode, "branch") == 0) {
        value = TSK_STAT_BRANCH;
    } else if (strcmp(mode, "node") == 0) {
        value = TSK_STAT_NODE;
    } else {
        PyErr_SetString(PyExc_ValueError, "Unrecognised stats mode");
        return -1;
    }
    *ret = value;
    return 0;
}

static int
parse_windows(
    PyObject *windows, PyArrayObject **ret_windows_array, tsk_size_t *ret_num_windows)
{
    int ret = -1;
    tsk_size_t num_windows = 0;
    PyArrayObject *windows_array = NULL;
    npy_intp *shape;

    windows_array = (PyArrayObject *) PyArray_FROMANY(
        windows, NPY_FLOAT64, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (windows_array == NULL) {
        goto out;
    }
    shape = PyArray_DIMS(windows_array);
    if (shape[0] < 2) {
        PyErr_SetString(PyExc_ValueError, "Windows array must have at least 2 elements");
        goto out;
    }
    num_windows = shape[0] - 1;

    ret = 0;
out:
    *ret_num_windows = num_windows;
    *ret_windows_array = windows_array;
    return ret;
}

static PyArrayObject *
TreeSequence_allocate_results_array(
    TreeSequence *self, tsk_flags_t mode, tsk_size_t num_windows, tsk_size_t output_dim)
{
    PyArrayObject *result_array = NULL;
    npy_intp result_shape[3];

    if (mode & TSK_STAT_NODE) {
        result_shape[0] = num_windows;
        result_shape[1] = tsk_treeseq_get_num_nodes(self->tree_sequence);
        result_shape[2] = output_dim;
        result_array = (PyArrayObject *) PyArray_SimpleNew(3, result_shape, NPY_FLOAT64);
        if (result_array == NULL) {
            goto out;
        }
    } else {
        result_shape[0] = num_windows;
        result_shape[1] = output_dim;
        result_array = (PyArrayObject *) PyArray_SimpleNew(2, result_shape, NPY_FLOAT64);
        if (result_array == NULL) {
            goto out;
        }
    }
out:
    return result_array;
}

static PyObject *
TreeSequence_general_stat(TreeSequence *self, PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    static char *kwlist[] = { "weights", "summary_func", "output_dim", "windows", "mode",
        "polarised", "span_normalise", NULL };
    PyObject *weights = NULL;
    PyObject *summary_func = NULL;
    PyObject *windows = NULL;
    PyArrayObject *weights_array = NULL;
    PyArrayObject *windows_array = NULL;
    PyArrayObject *result_array = NULL;
    char *mode = NULL;
    int polarised = 0;
    int span_normalise = 0;
    tsk_size_t num_windows;
    unsigned int output_dim;
    npy_intp *w_shape;
    tsk_flags_t options = 0;
    int err;

    if (TreeSequence_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOIO|sii", kwlist, &weights,
            &summary_func, &output_dim, &windows, &mode, &polarised, &span_normalise)) {
        Py_XINCREF(summary_func);
        goto out;
    }
    Py_INCREF(summary_func);
    if (!PyCallable_Check(summary_func)) {
        PyErr_SetString(PyExc_TypeError, "summary_func must be callable");
        goto out;
    }
    if (parse_stats_mode(mode, &options) != 0) {
        goto out;
    }
    if (polarised) {
        options |= TSK_STAT_POLARISED;
    }
    if (span_normalise) {
        options |= TSK_STAT_SPAN_NORMALISE;
    }
    if (parse_windows(windows, &windows_array, &num_windows) != 0) {
        goto out;
    }

    weights_array = (PyArrayObject *) PyArray_FROMANY(
        weights, NPY_FLOAT64, 2, 2, NPY_ARRAY_IN_ARRAY);
    if (weights_array == NULL) {
        goto out;
    }
    w_shape = PyArray_DIMS(weights_array);
    if (w_shape[0] != tsk_treeseq_get_num_samples(self->tree_sequence)) {
        PyErr_SetString(PyExc_ValueError, "First dimension must be num_samples");
        goto out;
    }
    result_array
        = TreeSequence_allocate_results_array(self, options, num_windows, output_dim);
    if (result_array == NULL) {
        goto out;
    }

    err = tsk_treeseq_general_stat(self->tree_sequence, w_shape[1],
        PyArray_DATA(weights_array), output_dim, general_stat_func, summary_func,
        num_windows, PyArray_DATA(windows_array), PyArray_DATA(result_array), options);
    if (err == TSK_PYTHON_CALLBACK_ERROR) {
        goto out;
    } else if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = (PyObject *) result_array;
    result_array = NULL;
out:
    Py_XDECREF(summary_func);
    Py_XDECREF(weights_array);
    Py_XDECREF(windows_array);
    Py_XDECREF(result_array);
    return ret;
}

static int
parse_sample_sets(PyObject *sample_set_sizes, PyArrayObject **ret_sample_set_sizes_array,
    PyObject *sample_sets, PyArrayObject **ret_sample_sets_array,
    tsk_size_t *ret_num_sample_sets)
{
    int ret = -1;
    PyArrayObject *sample_set_sizes_array = NULL;
    PyArrayObject *sample_sets_array = NULL;
    npy_intp *shape;
    tsk_size_t num_sample_sets = 0;
    tsk_size_t j, sum;
    uint32_t *a;

    sample_set_sizes_array = (PyArrayObject *) PyArray_FROMANY(
        sample_set_sizes, NPY_UINT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (sample_set_sizes_array == NULL) {
        goto out;
    }
    shape = PyArray_DIMS(sample_set_sizes_array);
    num_sample_sets = shape[0];
    /* The sum of the lengths in sample_set_sizes must be equal to the length
     * of the sample_sets array */
    sum = 0;
    a = PyArray_DATA(sample_set_sizes_array);
    for (j = 0; j < num_sample_sets; j++) {
        sum += a[j];
    }

    sample_sets_array = (PyArrayObject *) PyArray_FROMANY(
        sample_sets, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (sample_sets_array == NULL) {
        goto out;
    }
    shape = PyArray_DIMS(sample_sets_array);
    if (sum != (uint32_t) shape[0]) {
        PyErr_SetString(PyExc_ValueError,
            "Sum of sample_set_sizes must equal length of sample_sets array");
        goto out;
    }
    ret = 0;
out:
    *ret_sample_set_sizes_array = sample_set_sizes_array;
    *ret_sample_sets_array = sample_sets_array;
    *ret_num_sample_sets = num_sample_sets;
    return ret;
}

static PyObject *
TreeSequence_one_way_weighted_method(
    TreeSequence *self, PyObject *args, PyObject *kwds, one_way_weighted_method *method)
{
    PyObject *ret = NULL;
    static char *kwlist[]
        = { "weights", "windows", "mode", "polarised", "span_normalise", NULL };
    PyObject *weights = NULL;
    PyObject *windows = NULL;
    PyArrayObject *weights_array = NULL;
    PyArrayObject *windows_array = NULL;
    PyArrayObject *result_array = NULL;
    char *mode = NULL;
    int polarised = 0;
    int span_normalise = 0;
    tsk_size_t num_windows;
    npy_intp *w_shape;
    tsk_flags_t options = 0;
    int err;

    if (TreeSequence_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|sii", kwlist, &weights, &windows,
            &mode, &polarised, &span_normalise)) {
        goto out;
    }
    if (parse_stats_mode(mode, &options) != 0) {
        goto out;
    }
    if (polarised) {
        options |= TSK_STAT_POLARISED;
    }
    if (span_normalise) {
        options |= TSK_STAT_SPAN_NORMALISE;
    }
    if (parse_windows(windows, &windows_array, &num_windows) != 0) {
        goto out;
    }

    weights_array = (PyArrayObject *) PyArray_FROMANY(
        weights, NPY_FLOAT64, 2, 2, NPY_ARRAY_IN_ARRAY);
    if (weights_array == NULL) {
        goto out;
    }
    w_shape = PyArray_DIMS(weights_array);
    if (w_shape[0] != tsk_treeseq_get_num_samples(self->tree_sequence)) {
        PyErr_SetString(PyExc_ValueError, "First dimension must be num_samples");
        goto out;
    }
    result_array
        = TreeSequence_allocate_results_array(self, options, num_windows, w_shape[1]);
    if (result_array == NULL) {
        goto out;
    }

    err = method(self->tree_sequence, w_shape[1], PyArray_DATA(weights_array),
        num_windows, PyArray_DATA(windows_array), PyArray_DATA(result_array), options);
    if (err == TSK_PYTHON_CALLBACK_ERROR) {
        goto out;
    } else if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = (PyObject *) result_array;
    result_array = NULL;
out:
    Py_XDECREF(weights_array);
    Py_XDECREF(windows_array);
    Py_XDECREF(result_array);
    return ret;
}

static PyObject *
TreeSequence_one_way_covariates_method(TreeSequence *self, PyObject *args,
    PyObject *kwds, one_way_covariates_method *method)
{
    PyObject *ret = NULL;
    static char *kwlist[] = { "weights", "covariates", "windows", "mode", "polarised",
        "span_normalise", NULL };
    PyObject *weights = NULL;
    PyObject *covariates = NULL;
    PyObject *windows = NULL;
    PyArrayObject *weights_array = NULL;
    PyArrayObject *covariates_array = NULL;
    PyArrayObject *windows_array = NULL;
    PyArrayObject *result_array = NULL;
    char *mode = NULL;
    int polarised = 0;
    int span_normalise = 0;
    tsk_size_t num_windows;
    npy_intp *w_shape, *z_shape;
    tsk_flags_t options = 0;
    int err;

    if (TreeSequence_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOO|sii", kwlist, &weights,
            &covariates, &windows, &mode, &polarised, &span_normalise)) {
        goto out;
    }
    if (parse_stats_mode(mode, &options) != 0) {
        goto out;
    }
    if (polarised) {
        options |= TSK_STAT_POLARISED;
    }
    if (span_normalise) {
        options |= TSK_STAT_SPAN_NORMALISE;
    }
    if (parse_windows(windows, &windows_array, &num_windows) != 0) {
        goto out;
    }

    weights_array = (PyArrayObject *) PyArray_FROMANY(
        weights, NPY_FLOAT64, 2, 2, NPY_ARRAY_IN_ARRAY);
    if (weights_array == NULL) {
        goto out;
    }
    w_shape = PyArray_DIMS(weights_array);
    if (w_shape[0] != tsk_treeseq_get_num_samples(self->tree_sequence)) {
        PyErr_SetString(
            PyExc_ValueError, "First dimension of weights must be num_samples");
        goto out;
    }
    covariates_array = (PyArrayObject *) PyArray_FROMANY(
        covariates, NPY_FLOAT64, 2, 2, NPY_ARRAY_IN_ARRAY);
    if (covariates_array == NULL) {
        goto out;
    }
    z_shape = PyArray_DIMS(covariates_array);
    if (z_shape[0] != tsk_treeseq_get_num_samples(self->tree_sequence)) {
        PyErr_SetString(
            PyExc_ValueError, "First dimension of covariates must be num_samples");
        goto out;
    }
    result_array
        = TreeSequence_allocate_results_array(self, options, num_windows, w_shape[1]);
    if (result_array == NULL) {
        goto out;
    }

    err = method(self->tree_sequence, w_shape[1], PyArray_DATA(weights_array),
        z_shape[1], PyArray_DATA(covariates_array), num_windows,
        PyArray_DATA(windows_array), PyArray_DATA(result_array), options);
    if (err == TSK_PYTHON_CALLBACK_ERROR) {
        goto out;
    } else if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = (PyObject *) result_array;
    result_array = NULL;
out:
    Py_XDECREF(weights_array);
    Py_XDECREF(covariates_array);
    Py_XDECREF(windows_array);
    Py_XDECREF(result_array);
    return ret;
}

static PyObject *
TreeSequence_one_way_stat_method(TreeSequence *self, PyObject *args, PyObject *kwds,
    one_way_sample_stat_method *method)
{
    PyObject *ret = NULL;
    static char *kwlist[] = { "sample_set_sizes", "sample_sets", "windows", "mode",
        "span_normalise", "polarised", NULL };
    PyObject *sample_set_sizes = NULL;
    PyObject *sample_sets = NULL;
    PyObject *windows = NULL;
    char *mode = NULL;
    PyArrayObject *sample_set_sizes_array = NULL;
    PyArrayObject *sample_sets_array = NULL;
    PyArrayObject *windows_array = NULL;
    PyArrayObject *result_array = NULL;
    tsk_size_t num_windows, num_sample_sets;
    tsk_flags_t options = 0;
    int span_normalise = 1;
    int polarised = 0;
    int err;

    if (TreeSequence_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOO|sii", kwlist, &sample_set_sizes,
            &sample_sets, &windows, &mode, &span_normalise, &polarised)) {
        goto out;
    }
    if (parse_stats_mode(mode, &options) != 0) {
        goto out;
    }
    if (span_normalise) {
        options |= TSK_STAT_SPAN_NORMALISE;
    }
    if (polarised) {
        options |= TSK_STAT_POLARISED;
    }
    if (parse_sample_sets(sample_set_sizes, &sample_set_sizes_array, sample_sets,
            &sample_sets_array, &num_sample_sets)
        != 0) {
        goto out;
    }
    if (parse_windows(windows, &windows_array, &num_windows) != 0) {
        goto out;
    }

    result_array = TreeSequence_allocate_results_array(
        self, options, num_windows, num_sample_sets);
    if (result_array == NULL) {
        goto out;
    }
    err = method(self->tree_sequence, num_sample_sets,
        PyArray_DATA(sample_set_sizes_array), PyArray_DATA(sample_sets_array),
        num_windows, PyArray_DATA(windows_array), PyArray_DATA(result_array), options);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = (PyObject *) result_array;
    result_array = NULL;
out:
    Py_XDECREF(sample_set_sizes_array);
    Py_XDECREF(sample_sets_array);
    Py_XDECREF(windows_array);
    Py_XDECREF(result_array);
    return ret;
}

static PyObject *
TreeSequence_allele_frequency_spectrum(
    TreeSequence *self, PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    static char *kwlist[] = { "sample_set_sizes", "sample_sets", "windows", "mode",
        "span_normalise", "polarised", NULL };
    PyObject *sample_set_sizes = NULL;
    PyObject *sample_sets = NULL;
    PyObject *windows = NULL;
    char *mode = NULL;
    PyArrayObject *sample_set_sizes_array = NULL;
    PyArrayObject *sample_sets_array = NULL;
    PyArrayObject *windows_array = NULL;
    PyArrayObject *result_array = NULL;
    tsk_size_t *sizes;
    npy_intp *shape = NULL;
    tsk_size_t k, num_windows, num_sample_sets;
    tsk_flags_t options = 0;
    int polarised = 0;
    int span_normalise = 1;
    int err;

    if (TreeSequence_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOO|sii", kwlist, &sample_set_sizes,
            &sample_sets, &windows, &mode, &span_normalise, &polarised)) {
        goto out;
    }
    if (parse_stats_mode(mode, &options) != 0) {
        goto out;
    }
    if (span_normalise) {
        options |= TSK_STAT_SPAN_NORMALISE;
    }
    if (polarised) {
        options |= TSK_STAT_POLARISED;
    }
    if (parse_sample_sets(sample_set_sizes, &sample_set_sizes_array, sample_sets,
            &sample_sets_array, &num_sample_sets)
        != 0) {
        goto out;
    }
    if (parse_windows(windows, &windows_array, &num_windows) != 0) {
        goto out;
    }

    shape = PyMem_Malloc((num_sample_sets + 1) * sizeof(*shape));
    if (shape == NULL) {
        goto out;
    }
    sizes = PyArray_DATA(sample_set_sizes_array);
    shape[0] = num_windows;
    for (k = 0; k < num_sample_sets; k++) {
        shape[k + 1] = 1 + sizes[k];
    }
    result_array
        = (PyArrayObject *) PyArray_SimpleNew(1 + num_sample_sets, shape, NPY_FLOAT64);
    if (result_array == NULL) {
        goto out;
    }
    err = tsk_treeseq_allele_frequency_spectrum(self->tree_sequence, num_sample_sets,
        PyArray_DATA(sample_set_sizes_array), PyArray_DATA(sample_sets_array),
        num_windows, PyArray_DATA(windows_array), PyArray_DATA(result_array), options);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = (PyObject *) result_array;
    result_array = NULL;
out:
    PyMem_Free(shape);
    Py_XDECREF(sample_set_sizes_array);
    Py_XDECREF(sample_sets_array);
    Py_XDECREF(windows_array);
    Py_XDECREF(result_array);
    return ret;
}

static PyObject *
TreeSequence_diversity(TreeSequence *self, PyObject *args, PyObject *kwds)
{
    return TreeSequence_one_way_stat_method(self, args, kwds, tsk_treeseq_diversity);
}

static PyObject *
TreeSequence_trait_covariance(TreeSequence *self, PyObject *args, PyObject *kwds)
{
    return TreeSequence_one_way_weighted_method(
        self, args, kwds, tsk_treeseq_trait_covariance);
}

static PyObject *
TreeSequence_trait_correlation(TreeSequence *self, PyObject *args, PyObject *kwds)
{
    return TreeSequence_one_way_weighted_method(
        self, args, kwds, tsk_treeseq_trait_correlation);
}

static PyObject *
TreeSequence_trait_linear_model(TreeSequence *self, PyObject *args, PyObject *kwds)
{
    return TreeSequence_one_way_covariates_method(
        self, args, kwds, tsk_treeseq_trait_linear_model);
}

static PyObject *
TreeSequence_segregating_sites(TreeSequence *self, PyObject *args, PyObject *kwds)
{
    return TreeSequence_one_way_stat_method(
        self, args, kwds, tsk_treeseq_segregating_sites);
}

static PyObject *
TreeSequence_Y1(TreeSequence *self, PyObject *args, PyObject *kwds)
{
    return TreeSequence_one_way_stat_method(self, args, kwds, tsk_treeseq_Y1);
}

static PyObject *
TreeSequence_k_way_stat_method(TreeSequence *self, PyObject *args, PyObject *kwds,
    npy_intp tuple_size, general_sample_stat_method *method)
{
    PyObject *ret = NULL;
    static char *kwlist[] = { "sample_set_sizes", "sample_sets", "indexes", "windows",
        "mode", "span_normalise", "polarised", NULL };
    PyObject *sample_set_sizes = NULL;
    PyObject *sample_sets = NULL;
    PyObject *indexes = NULL;
    PyObject *windows = NULL;
    PyArrayObject *sample_set_sizes_array = NULL;
    PyArrayObject *sample_sets_array = NULL;
    PyArrayObject *indexes_array = NULL;
    PyArrayObject *windows_array = NULL;
    PyArrayObject *result_array = NULL;
    tsk_size_t num_windows, num_sample_sets, num_set_index_tuples;
    npy_intp *shape;
    tsk_flags_t options = 0;
    char *mode = NULL;
    int span_normalise = true;
    int polarised = false;
    int err;

    if (TreeSequence_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOO|sii", kwlist, &sample_set_sizes,
            &sample_sets, &indexes, &windows, &mode, &span_normalise, &polarised)) {
        goto out;
    }
    if (parse_stats_mode(mode, &options) != 0) {
        goto out;
    }
    if (span_normalise) {
        options |= TSK_STAT_SPAN_NORMALISE;
    }
    if (polarised) {
        options |= TSK_STAT_POLARISED;
    }
    if (parse_sample_sets(sample_set_sizes, &sample_set_sizes_array, sample_sets,
            &sample_sets_array, &num_sample_sets)
        != 0) {
        goto out;
    }
    if (parse_windows(windows, &windows_array, &num_windows) != 0) {
        goto out;
    }

    indexes_array = (PyArrayObject *) PyArray_FROMANY(
        indexes, NPY_INT32, 2, 2, NPY_ARRAY_IN_ARRAY);
    if (indexes_array == NULL) {
        goto out;
    }
    shape = PyArray_DIMS(indexes_array);
    if (shape[0] < 1 || shape[1] != tuple_size) {
        PyErr_Format(
            PyExc_ValueError, "indexes must be a k x %d array.", (int) tuple_size);
        goto out;
    }
    num_set_index_tuples = shape[0];

    result_array = TreeSequence_allocate_results_array(
        self, options, num_windows, num_set_index_tuples);
    if (result_array == NULL) {
        goto out;
    }
    err = method(self->tree_sequence, num_sample_sets,
        PyArray_DATA(sample_set_sizes_array), PyArray_DATA(sample_sets_array),
        num_set_index_tuples, PyArray_DATA(indexes_array), num_windows,
        PyArray_DATA(windows_array), PyArray_DATA(result_array), options);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = (PyObject *) result_array;
    result_array = NULL;
out:
    Py_XDECREF(sample_set_sizes_array);
    Py_XDECREF(sample_sets_array);
    Py_XDECREF(indexes_array);
    Py_XDECREF(windows_array);
    Py_XDECREF(result_array);
    return ret;
}

static PyObject *
TreeSequence_divergence(TreeSequence *self, PyObject *args, PyObject *kwds)
{
    return TreeSequence_k_way_stat_method(self, args, kwds, 2, tsk_treeseq_divergence);
}

static PyObject *
TreeSequence_genetic_relatedness(TreeSequence *self, PyObject *args, PyObject *kwds)
{
    return TreeSequence_k_way_stat_method(
        self, args, kwds, 2, tsk_treeseq_genetic_relatedness);
}

static PyObject *
TreeSequence_Y2(TreeSequence *self, PyObject *args, PyObject *kwds)
{
    return TreeSequence_k_way_stat_method(self, args, kwds, 2, tsk_treeseq_Y2);
}

static PyObject *
TreeSequence_f2(TreeSequence *self, PyObject *args, PyObject *kwds)
{
    return TreeSequence_k_way_stat_method(self, args, kwds, 2, tsk_treeseq_f2);
}

static PyObject *
TreeSequence_Y3(TreeSequence *self, PyObject *args, PyObject *kwds)
{
    return TreeSequence_k_way_stat_method(self, args, kwds, 3, tsk_treeseq_Y3);
}

static PyObject *
TreeSequence_f3(TreeSequence *self, PyObject *args, PyObject *kwds)
{
    return TreeSequence_k_way_stat_method(self, args, kwds, 3, tsk_treeseq_f3);
}

static PyObject *
TreeSequence_f4(TreeSequence *self, PyObject *args, PyObject *kwds)
{
    return TreeSequence_k_way_stat_method(self, args, kwds, 4, tsk_treeseq_f4);
}

static PyObject *
TreeSequence_get_num_mutations(TreeSequence *self)
{
    PyObject *ret = NULL;
    size_t num_mutations;

    if (TreeSequence_check_state(self) != 0) {
        goto out;
    }
    num_mutations = tsk_treeseq_get_num_mutations(self->tree_sequence);
    ret = Py_BuildValue("n", (Py_ssize_t) num_mutations);
out:
    return ret;
}

static PyObject *
TreeSequence_get_num_sites(TreeSequence *self)
{
    PyObject *ret = NULL;
    size_t num_sites;

    if (TreeSequence_check_state(self) != 0) {
        goto out;
    }
    num_sites = tsk_treeseq_get_num_sites(self->tree_sequence);
    ret = Py_BuildValue("n", (Py_ssize_t) num_sites);
out:
    return ret;
}

static PyObject *
TreeSequence_get_num_provenances(TreeSequence *self)
{
    PyObject *ret = NULL;
    size_t num_provenances;

    if (TreeSequence_check_state(self) != 0) {
        goto out;
    }
    num_provenances = tsk_treeseq_get_num_provenances(self->tree_sequence);
    ret = Py_BuildValue("n", (Py_ssize_t) num_provenances);
out:
    return ret;
}

static PyObject *
TreeSequence_get_genotype_matrix(TreeSequence *self, PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    static char *kwlist[] = { "isolated_as_missing", "alleles", NULL };
    int err;
    size_t num_sites;
    size_t num_samples;
    npy_intp dims[2];
    PyObject *py_alleles = Py_None;
    PyArrayObject *genotype_matrix = NULL;
    tsk_vargen_t *vg = NULL;
    char *V;
    tsk_variant_t *variant;
    size_t j;
    int isolated_as_missing = 1;
    const char **alleles = NULL;
    tsk_flags_t options = 0;

    if (TreeSequence_check_state(self) != 0) {
        goto out;
    }

    /* TODO add option for 16 bit genotypes */
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "|iO", kwlist, &isolated_as_missing, &py_alleles)) {
        goto out;
    }
    if (!isolated_as_missing) {
        options |= TSK_ISOLATED_NOT_MISSING;
    }

    if (py_alleles != Py_None) {
        alleles = parse_allele_list(py_alleles);
        if (alleles == NULL) {
            goto out;
        }
    }

    num_sites = tsk_treeseq_get_num_sites(self->tree_sequence);
    num_samples = tsk_treeseq_get_num_samples(self->tree_sequence);
    dims[0] = num_sites;
    dims[1] = num_samples;

    genotype_matrix = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_INT8);
    if (genotype_matrix == NULL) {
        goto out;
    }
    V = (char *) PyArray_DATA(genotype_matrix);
    vg = PyMem_Malloc(sizeof(tsk_vargen_t));
    if (vg == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    err = tsk_vargen_init(vg, self->tree_sequence, NULL, 0, alleles, options);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    j = 0;
    while ((err = tsk_vargen_next(vg, &variant)) == 1) {
        memcpy(
            V + (j * num_samples), variant->genotypes.i8, num_samples * sizeof(int8_t));
        j++;
    }
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = (PyObject *) genotype_matrix;
    genotype_matrix = NULL;
out:
    if (vg != NULL) {
        tsk_vargen_free(vg);
        PyMem_Free(vg);
    }
    Py_XDECREF(genotype_matrix);
    PyMem_Free(alleles);
    return ret;
}

static PyMethodDef TreeSequence_methods[] = {
    { .ml_name = "dump",
        .ml_meth = (PyCFunction) TreeSequence_dump,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Writes the tree sequence out to the specified file." },
    { .ml_name = "load",
        .ml_meth = (PyCFunction) TreeSequence_load,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Loads a tree sequence from the specified file." },
    { .ml_name = "load_tables",
        .ml_meth = (PyCFunction) TreeSequence_load_tables,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Loads a tree sequence from the specified set of tables" },
    { .ml_name = "dump_tables",
        .ml_meth = (PyCFunction) TreeSequence_dump_tables,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Dumps the tree sequence to the specified set of tables" },
    { .ml_name = "get_node",
        .ml_meth = (PyCFunction) TreeSequence_get_node,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns the node record at the specified index." },
    { .ml_name = "get_edge",
        .ml_meth = (PyCFunction) TreeSequence_get_edge,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns the edge record at the specified index." },
    { .ml_name = "get_migration",
        .ml_meth = (PyCFunction) TreeSequence_get_migration,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns the migration record at the specified index." },
    { .ml_name = "get_site",
        .ml_meth = (PyCFunction) TreeSequence_get_site,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns the mutation type record at the specified index." },
    { .ml_name = "get_mutation",
        .ml_meth = (PyCFunction) TreeSequence_get_mutation,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns the mutation record at the specified index." },
    { .ml_name = "get_individual",
        .ml_meth = (PyCFunction) TreeSequence_get_individual,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns the individual record at the specified index." },
    { .ml_name = "get_population",
        .ml_meth = (PyCFunction) TreeSequence_get_population,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns the population record at the specified index." },
    { .ml_name = "get_provenance",
        .ml_meth = (PyCFunction) TreeSequence_get_provenance,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns the provenance record at the specified index." },
    { .ml_name = "get_num_edges",
        .ml_meth = (PyCFunction) TreeSequence_get_num_edges,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns the number of coalescence records." },
    { .ml_name = "get_num_migrations",
        .ml_meth = (PyCFunction) TreeSequence_get_num_migrations,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns the number of migration records." },
    { .ml_name = "get_num_populations",
        .ml_meth = (PyCFunction) TreeSequence_get_num_populations,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns the number of population records." },
    { .ml_name = "get_num_individuals",
        .ml_meth = (PyCFunction) TreeSequence_get_num_individuals,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns the number of individual records." },
    { .ml_name = "get_num_trees",
        .ml_meth = (PyCFunction) TreeSequence_get_num_trees,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns the number of trees in the tree sequence." },
    { .ml_name = "get_sequence_length",
        .ml_meth = (PyCFunction) TreeSequence_get_sequence_length,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns the sequence length in bases." },
    { .ml_name = "get_breakpoints",
        .ml_meth = (PyCFunction) TreeSequence_get_breakpoints,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns the tree breakpoints as a numpy array." },
    { .ml_name = "get_file_uuid",
        .ml_meth = (PyCFunction) TreeSequence_get_file_uuid,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns the UUID of the underlying file, if present." },
    { .ml_name = "get_metadata",
        .ml_meth = (PyCFunction) TreeSequence_get_metadata,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns the metadata for the tree sequence" },
    { .ml_name = "get_metadata_schema",
        .ml_meth = (PyCFunction) TreeSequence_get_metadata_schema,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns the metadata schema for the tree sequence metadata" },
    { .ml_name = "get_num_sites",
        .ml_meth = (PyCFunction) TreeSequence_get_num_sites,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns the number of sites" },
    { .ml_name = "get_num_mutations",
        .ml_meth = (PyCFunction) TreeSequence_get_num_mutations,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns the number of mutations" },
    { .ml_name = "get_num_provenances",
        .ml_meth = (PyCFunction) TreeSequence_get_num_provenances,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns the number of provenances" },
    { .ml_name = "get_num_nodes",
        .ml_meth = (PyCFunction) TreeSequence_get_num_nodes,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns the number of unique nodes in the tree sequence." },
    { .ml_name = "get_num_samples",
        .ml_meth = (PyCFunction) TreeSequence_get_num_samples,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns the sample size" },
    { .ml_name = "get_table_metadata_schemas",
        .ml_meth = (PyCFunction) TreeSequence_get_table_metadata_schemas,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns the metadata schemas for the tree sequence tables" },
    { .ml_name = "get_samples",
        .ml_meth = (PyCFunction) TreeSequence_get_samples,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns the samples." },
    { .ml_name = "genealogical_nearest_neighbours",
        .ml_meth = (PyCFunction) TreeSequence_genealogical_nearest_neighbours,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Returns the genealogical nearest neighbours statistic." },
    { .ml_name = "get_kc_distance",
        .ml_meth = (PyCFunction) TreeSequence_get_kc_distance,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Returns the KC distance between this tree sequence and another." },
    { .ml_name = "mean_descendants",
        .ml_meth = (PyCFunction) TreeSequence_mean_descendants,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Returns the mean number of nodes descending from each node." },
    { .ml_name = "general_stat",
        .ml_meth = (PyCFunction) TreeSequence_general_stat,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Runs the general stats algorithm for a given summary function." },
    { .ml_name = "diversity",
        .ml_meth = (PyCFunction) TreeSequence_diversity,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Computes diversity within sample sets." },
    { .ml_name = "allele_frequency_spectrum",
        .ml_meth = (PyCFunction) TreeSequence_allele_frequency_spectrum,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Computes the K-dimensional joint AFS." },
    { .ml_name = "trait_covariance",
        .ml_meth = (PyCFunction) TreeSequence_trait_covariance,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Computes covariance with traits." },
    { .ml_name = "trait_correlation",
        .ml_meth = (PyCFunction) TreeSequence_trait_correlation,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Computes correlation with traits." },
    { .ml_name = "trait_linear_model",
        .ml_meth = (PyCFunction) TreeSequence_trait_linear_model,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Computes coefficients of a linear model for each trait." },
    { .ml_name = "segregating_sites",
        .ml_meth = (PyCFunction) TreeSequence_segregating_sites,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Computes density of segregating sites within sample sets." },
    { .ml_name = "divergence",
        .ml_meth = (PyCFunction) TreeSequence_divergence,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Computes diveregence between sample sets." },
    { .ml_name = "genetic_relatedness",
        .ml_meth = (PyCFunction) TreeSequence_genetic_relatedness,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Computes genetic relatedness between sample sets." },
    { .ml_name = "Y1",
        .ml_meth = (PyCFunction) TreeSequence_Y1,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Computes the Y1 statistic." },
    { .ml_name = "Y2",
        .ml_meth = (PyCFunction) TreeSequence_Y2,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Computes the Y2 statistic." },
    { .ml_name = "f2",
        .ml_meth = (PyCFunction) TreeSequence_f2,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Computes the f2 statistic." },
    { .ml_name = "Y3",
        .ml_meth = (PyCFunction) TreeSequence_Y3,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Computes the Y3 statistic." },
    { .ml_name = "f3",
        .ml_meth = (PyCFunction) TreeSequence_f3,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Computes the f3 statistic." },
    { .ml_name = "f4",
        .ml_meth = (PyCFunction) TreeSequence_f4,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Computes the f4 statistic." },
    { .ml_name = "get_genotype_matrix",
        .ml_meth = (PyCFunction) TreeSequence_get_genotype_matrix,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Returns the genotypes matrix." },
    { NULL } /* Sentinel */
};

static PyTypeObject TreeSequenceType = {
    // clang-format off
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_tskit.TreeSequence",
    .tp_basicsize = sizeof(TreeSequence),
    .tp_dealloc = (destructor) TreeSequence_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "TreeSequence objects",
    .tp_methods = TreeSequence_methods,
    .tp_init = (initproc) TreeSequence_init,
    .tp_new = PyType_GenericNew,
    // clang-format on
};

/*===================================================================
 * Tree
 *===================================================================
 */

static int
Tree_check_state(Tree *self)
{
    int ret = 0;
    if (self->tree == NULL) {
        PyErr_SetString(PyExc_SystemError, "tree not initialised");
        ret = -1;
    }
    return ret;
}

static int
Tree_check_bounds(Tree *self, int node)
{
    int ret = 0;
    if (node < 0 || node >= (int) self->tree->num_nodes) {
        PyErr_SetString(PyExc_ValueError, "Node index out of bounds");
        ret = -1;
    }
    return ret;
}

static void
Tree_dealloc(Tree *self)
{
    if (self->tree != NULL) {
        tsk_tree_free(self->tree);
        PyMem_Free(self->tree);
        self->tree = NULL;
    }
    Py_XDECREF(self->tree_sequence);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static int
Tree_init(Tree *self, PyObject *args, PyObject *kwds)
{
    int ret = -1;
    int err;
    static char *kwlist[] = { "tree_sequence", "options", "tracked_samples", NULL };
    PyObject *py_tracked_samples = NULL;
    TreeSequence *tree_sequence = NULL;
    tsk_id_t *tracked_samples = NULL;
    unsigned int options = 0;
    size_t j, num_tracked_samples, num_nodes;
    PyObject *item;

    self->tree = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|IO!", kwlist, &TreeSequenceType,
            &tree_sequence, &options, &PyList_Type, &py_tracked_samples)) {
        goto out;
    }
    self->tree_sequence = tree_sequence;
    Py_INCREF(self->tree_sequence);
    if (TreeSequence_check_state(tree_sequence) != 0) {
        goto out;
    }
    num_nodes = tsk_treeseq_get_num_nodes(tree_sequence->tree_sequence);
    num_tracked_samples = 0;
    if (py_tracked_samples != NULL) {
        if ((options & TSK_NO_SAMPLE_COUNTS)) {
            PyErr_SetString(PyExc_ValueError,
                "Cannot specified tracked_samples without count_samples flag");
            goto out;
        }
        num_tracked_samples = PyList_Size(py_tracked_samples);
    }
    tracked_samples = PyMem_Malloc(num_tracked_samples * sizeof(tsk_id_t));
    if (tracked_samples == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    for (j = 0; j < num_tracked_samples; j++) {
        item = PyList_GetItem(py_tracked_samples, j);
        if (!PyNumber_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "sample must be a number");
            goto out;
        }
        tracked_samples[j] = (tsk_id_t) PyLong_AsLong(item);
        if (tracked_samples[j] < 0 || tracked_samples[j] >= (tsk_id_t) num_nodes) {
            PyErr_SetString(PyExc_ValueError, "samples must be valid nodes");
            goto out;
        }
    }
    self->tree = PyMem_Malloc(sizeof(tsk_tree_t));
    if (self->tree == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    err = tsk_tree_init(self->tree, tree_sequence->tree_sequence, (tsk_flags_t) options);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    if (!(options & TSK_NO_SAMPLE_COUNTS)) {
        err = tsk_tree_set_tracked_samples(
            self->tree, num_tracked_samples, tracked_samples);
        if (err != 0) {
            handle_library_error(err);
            goto out;
        }
    }
    ret = 0;
out:
    if (tracked_samples != NULL) {
        PyMem_Free(tracked_samples);
    }
    return ret;
}

static PyObject *
Tree_first(Tree *self)
{
    PyObject *ret = NULL;
    int err;

    if (Tree_check_state(self) != 0) {
        goto out;
    }
    err = tsk_tree_first(self->tree);
    if (err < 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
Tree_last(Tree *self)
{
    PyObject *ret = NULL;
    int err;

    if (Tree_check_state(self) != 0) {
        goto out;
    }
    err = tsk_tree_last(self->tree);
    if (err < 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
Tree_next(Tree *self)
{
    PyObject *ret = NULL;
    int err;

    if (Tree_check_state(self) != 0) {
        goto out;
    }
    err = tsk_tree_next(self->tree);
    if (err < 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("i", err == 1);
out:
    return ret;
}

static PyObject *
Tree_prev(Tree *self)
{
    PyObject *ret = NULL;
    int err;

    if (Tree_check_state(self) != 0) {
        goto out;
    }
    err = tsk_tree_prev(self->tree);
    if (err < 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("i", err == 1);
out:
    return ret;
}

static PyObject *
Tree_clear(Tree *self)
{
    PyObject *ret = NULL;
    int err;

    if (Tree_check_state(self) != 0) {
        goto out;
    }
    err = tsk_tree_clear(self->tree);
    if (err < 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
Tree_get_sample_size(Tree *self)
{
    PyObject *ret = NULL;

    if (Tree_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("n", (Py_ssize_t) self->tree->tree_sequence->num_samples);
out:
    return ret;
}

static PyObject *
Tree_get_num_nodes(Tree *self)
{
    PyObject *ret = NULL;

    if (Tree_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("n", (Py_ssize_t) self->tree->num_nodes);
out:
    return ret;
}

static PyObject *
Tree_get_num_roots(Tree *self)
{
    PyObject *ret = NULL;

    if (Tree_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("n", (Py_ssize_t) tsk_tree_get_num_roots(self->tree));
out:
    return ret;
}

static PyObject *
Tree_get_index(Tree *self)
{
    PyObject *ret = NULL;

    if (Tree_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("n", (Py_ssize_t) self->tree->index);
out:
    return ret;
}

static PyObject *
Tree_get_left_root(Tree *self)
{
    PyObject *ret = NULL;

    if (Tree_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("i", (int) self->tree->left_root);
out:
    return ret;
}

static PyObject *
Tree_get_left(Tree *self)
{
    PyObject *ret = NULL;

    if (Tree_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("d", self->tree->left);
out:
    return ret;
}

static PyObject *
Tree_get_right(Tree *self)
{
    PyObject *ret = NULL;

    if (Tree_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("d", self->tree->right);
out:
    return ret;
}

static PyObject *
Tree_get_options(Tree *self)
{
    PyObject *ret = NULL;

    if (Tree_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("i", self->tree->options);
out:
    return ret;
}

static int
Tree_get_node_argument(Tree *self, PyObject *args, int *node)
{
    int ret = -1;
    if (Tree_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "I", node)) {
        goto out;
    }
    if (Tree_check_bounds(self, *node)) {
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static PyObject *
Tree_is_sample(Tree *self, PyObject *args)
{
    PyObject *ret = NULL;
    int node;

    if (Tree_get_node_argument(self, args, &node) != 0) {
        goto out;
    }
    ret = Py_BuildValue("i", tsk_tree_is_sample(self->tree, (tsk_id_t) node));
out:
    return ret;
}

static PyObject *
Tree_is_descendant(Tree *self, PyObject *args)
{
    PyObject *ret = NULL;
    int u, v;

    if (Tree_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "II", &u, &v)) {
        goto out;
    }
    if (Tree_check_bounds(self, (tsk_id_t) u)) {
        goto out;
    }
    if (Tree_check_bounds(self, (tsk_id_t) v)) {
        goto out;
    }
    ret = Py_BuildValue(
        "i", tsk_tree_is_descendant(self->tree, (tsk_id_t) u, (tsk_id_t) v));
out:
    return ret;
}

static PyObject *
Tree_get_parent(Tree *self, PyObject *args)
{
    PyObject *ret = NULL;
    tsk_id_t parent;
    int node;

    if (Tree_get_node_argument(self, args, &node) != 0) {
        goto out;
    }
    parent = self->tree->parent[node];
    ret = Py_BuildValue("i", (int) parent);
out:
    return ret;
}

static PyObject *
Tree_get_population(Tree *self, PyObject *args)
{
    PyObject *ret = NULL;
    tsk_node_t node;
    int node_id, err;

    if (Tree_get_node_argument(self, args, &node_id) != 0) {
        goto out;
    }
    err = tsk_treeseq_get_node(self->tree->tree_sequence, node_id, &node);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("i", (int) node.population);
out:
    return ret;
}

static PyObject *
Tree_get_time(Tree *self, PyObject *args)
{
    PyObject *ret = NULL;
    double time;
    int node, err;

    if (Tree_get_node_argument(self, args, &node) != 0) {
        goto out;
    }
    err = tsk_tree_get_time(self->tree, node, &time);
    if (ret != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("d", time);
out:
    return ret;
}

static PyObject *
Tree_get_left_child(Tree *self, PyObject *args)
{
    PyObject *ret = NULL;
    tsk_id_t child;
    int node;

    if (Tree_get_node_argument(self, args, &node) != 0) {
        goto out;
    }
    child = self->tree->left_child[node];
    ret = Py_BuildValue("i", (int) child);
out:
    return ret;
}

static PyObject *
Tree_get_right_child(Tree *self, PyObject *args)
{
    PyObject *ret = NULL;
    tsk_id_t child;
    int node;

    if (Tree_get_node_argument(self, args, &node) != 0) {
        goto out;
    }
    child = self->tree->right_child[node];
    ret = Py_BuildValue("i", (int) child);
out:
    return ret;
}

static PyObject *
Tree_get_left_sib(Tree *self, PyObject *args)
{
    PyObject *ret = NULL;
    tsk_id_t sib;
    int node;

    if (Tree_get_node_argument(self, args, &node) != 0) {
        goto out;
    }
    sib = self->tree->left_sib[node];
    ret = Py_BuildValue("i", (int) sib);
out:
    return ret;
}

static PyObject *
Tree_get_right_sib(Tree *self, PyObject *args)
{
    PyObject *ret = NULL;
    tsk_id_t sib;
    int node;

    if (Tree_get_node_argument(self, args, &node) != 0) {
        goto out;
    }
    sib = self->tree->right_sib[node];
    ret = Py_BuildValue("i", (int) sib);
out:
    return ret;
}

static PyObject *
Tree_get_children(Tree *self, PyObject *args)
{
    PyObject *ret = NULL;
    int node;
    tsk_id_t u;
    size_t j, num_children;
    tsk_id_t *children = NULL;

    if (Tree_get_node_argument(self, args, &node) != 0) {
        goto out;
    }
    num_children = 0;
    for (u = self->tree->left_child[node]; u != TSK_NULL; u = self->tree->right_sib[u]) {
        num_children++;
    }
    children = PyMem_Malloc(num_children * sizeof(tsk_id_t));
    if (children == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    j = 0;
    for (u = self->tree->left_child[node]; u != TSK_NULL; u = self->tree->right_sib[u]) {
        children[j] = u;
        j++;
    }
    ret = convert_node_id_list(children, num_children);
out:
    if (children != NULL) {
        PyMem_Free(children);
    }
    return ret;
}

static PyObject *
Tree_depth(Tree *self, PyObject *args)
{
    PyObject *ret = NULL;
    tsk_size_t depth;
    int node, err;

    if (Tree_get_node_argument(self, args, &node) != 0) {
        goto out;
    }
    err = tsk_tree_depth(self->tree, node, &depth);
    if (ret != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("I", (unsigned int) depth);
out:
    return ret;
}

static bool
Tree_check_sample_list(Tree *self)
{
    bool ret = tsk_tree_has_sample_lists(self->tree);
    if (!ret) {
        PyErr_SetString(PyExc_ValueError,
            "Sample lists not supported. Please set sample_lists=True.");
    }
    return ret;
}

static PyObject *
Tree_get_right_sample(Tree *self, PyObject *args)
{
    PyObject *ret = NULL;
    tsk_id_t sample_index;
    int node;

    if (Tree_get_node_argument(self, args, &node) != 0) {
        goto out;
    }
    if (!Tree_check_sample_list(self)) {
        goto out;
    }
    sample_index = self->tree->right_sample[node];
    ret = Py_BuildValue("i", (int) sample_index);
out:
    return ret;
}

static PyObject *
Tree_get_left_sample(Tree *self, PyObject *args)
{
    PyObject *ret = NULL;
    tsk_id_t sample_index;
    int node;

    if (Tree_get_node_argument(self, args, &node) != 0) {
        goto out;
    }
    if (!Tree_check_sample_list(self)) {
        goto out;
    }
    sample_index = self->tree->left_sample[node];
    ret = Py_BuildValue("i", (int) sample_index);
out:
    return ret;
}

static PyObject *
Tree_get_next_sample(Tree *self, PyObject *args)
{
    PyObject *ret = NULL;
    tsk_id_t out_index;
    int in_index, num_samples;

    if (Tree_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "I", &in_index)) {
        goto out;
    }
    num_samples = (int) tsk_treeseq_get_num_samples(self->tree->tree_sequence);
    if (in_index < 0 || in_index >= num_samples) {
        PyErr_SetString(PyExc_ValueError, "Sample index out of bounds");
        goto out;
    }
    if (!Tree_check_sample_list(self)) {
        goto out;
    }
    out_index = self->tree->next_sample[in_index];
    ret = Py_BuildValue("i", (int) out_index);
out:
    return ret;
}

static PyObject *
Tree_get_mrca(Tree *self, PyObject *args)
{
    PyObject *ret = NULL;
    int err;
    tsk_id_t mrca;
    int u, v;

    if (Tree_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "ii", &u, &v)) {
        goto out;
    }
    if (Tree_check_bounds(self, u)) {
        goto out;
    }
    if (Tree_check_bounds(self, v)) {
        goto out;
    }
    err = tsk_tree_get_mrca(self->tree, (tsk_id_t) u, (tsk_id_t) v, &mrca);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("i", (int) mrca);
out:
    return ret;
}

static PyObject *
Tree_get_num_children(Tree *self, PyObject *args)
{
    PyObject *ret = NULL;
    unsigned int num_children;
    int node;
    tsk_id_t u;

    if (Tree_get_node_argument(self, args, &node) != 0) {
        goto out;
    }
    num_children = 0;
    for (u = self->tree->left_child[node]; u != TSK_NULL; u = self->tree->right_sib[u]) {
        num_children++;
    }
    ret = Py_BuildValue("I", num_children);
out:
    return ret;
}

static PyObject *
Tree_get_num_samples(Tree *self, PyObject *args)
{
    PyObject *ret = NULL;
    size_t num_samples;
    int err, node;

    if (Tree_get_node_argument(self, args, &node) != 0) {
        goto out;
    }
    err = tsk_tree_get_num_samples(self->tree, (tsk_id_t) node, &num_samples);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("I", (unsigned int) num_samples);
out:
    return ret;
}

static PyObject *
Tree_get_num_tracked_samples(Tree *self, PyObject *args)
{
    PyObject *ret = NULL;
    size_t num_tracked_samples;
    int err, node;

    if (Tree_get_node_argument(self, args, &node) != 0) {
        goto out;
    }
    err = tsk_tree_get_num_tracked_samples(
        self->tree, (tsk_id_t) node, &num_tracked_samples);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("I", (unsigned int) num_tracked_samples);
out:
    return ret;
}

static PyObject *
Tree_get_sites(Tree *self, PyObject *args)
{
    PyObject *ret = NULL;

    if (Tree_check_state(self) != 0) {
        goto out;
    }
    ret = convert_sites(self->tree->sites, self->tree->sites_length);
out:
    return ret;
}

static PyObject *
Tree_get_num_sites(Tree *self)
{
    PyObject *ret = NULL;

    if (Tree_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("n", (Py_ssize_t) self->tree->sites_length);
out:
    return ret;
}

static PyObject *
Tree_get_newick(Tree *self, PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    static char *kwlist[] = { "root", "precision", "buffer_size", NULL };
    int precision = 14;
    /* We have a default bufsize for convenience, but the high-level code
     * should set this by computing an upper bound. */
    Py_ssize_t buffer_size = 1024;
    int root, err;
    char *buffer = NULL;

    if (Tree_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "i|in", kwlist, &root, &precision, &buffer_size)) {
        goto out;
    }
    if (precision < 0 || precision > 16) {
        PyErr_SetString(
            PyExc_ValueError, "Precision must be between 0 and 16, inclusive");
        goto out;
    }
    if (buffer_size <= 0) {
        PyErr_SetString(PyExc_ValueError, "Buffer size must be > 0");
        goto out;
    }
    buffer = PyMem_Malloc(buffer_size);
    if (buffer == NULL) {
        PyErr_NoMemory();
    }
    err = tsk_convert_newick(
        self->tree, (tsk_id_t) root, precision, 0, (size_t) buffer_size, buffer);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = PyBytes_FromString(buffer);
out:
    if (buffer != NULL) {
        PyMem_Free(buffer);
    }
    return ret;
}

static PyObject *
Tree_map_mutations(Tree *self, PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    PyObject *genotypes = NULL;
    PyObject *py_transitions = NULL;
    PyArrayObject *genotypes_array = NULL;
    static char *kwlist[] = { "genotypes", NULL };
    int8_t ancestral_state;
    tsk_state_transition_t *transitions = NULL;
    tsk_size_t num_transitions;
    npy_intp *shape;
    int err;

    if (Tree_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &genotypes)) {
        goto out;
    }
    genotypes_array = (PyArrayObject *) PyArray_FROMANY(
        genotypes, NPY_INT8, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (genotypes_array == NULL) {
        goto out;
    }
    shape = PyArray_DIMS(genotypes_array);
    if (shape[0] != tsk_treeseq_get_num_samples(self->tree->tree_sequence)) {
        PyErr_SetString(
            PyExc_ValueError, "Genotypes array must have 1D (num_samples,) array");
        goto out;
    }

    err = tsk_tree_map_mutations(self->tree, (int8_t *) PyArray_DATA(genotypes_array),
        NULL, 0, &ancestral_state, &num_transitions, &transitions);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    py_transitions = convert_transitions(transitions, num_transitions);
    if (py_transitions == NULL) {
        goto out;
    }
    ret = Py_BuildValue("iO", ancestral_state, py_transitions);
out:
    if (transitions != NULL) {
        free(transitions);
    }
    Py_XDECREF(genotypes_array);
    Py_XDECREF(py_transitions);
    return ret;
}

/* Forward declaration */
static PyTypeObject TreeType;

static PyObject *
Tree_equals(Tree *self, PyObject *args)
{
    PyObject *ret = NULL;
    Tree *other = NULL;

    if (Tree_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "O!", &TreeType, &other)) {
        goto out;
    }
    if (Tree_check_state(other) != 0) {
        goto out;
    }
    ret = Py_BuildValue("i", tsk_tree_equals(self->tree, other->tree));
out:
    return ret;
}

static PyObject *
Tree_copy(Tree *self)
{
    int err;
    PyObject *ret = NULL;
    PyObject *args = NULL;
    Tree *copy = NULL;

    if (Tree_check_state(self) != 0) {
        goto out;
    }
    args = Py_BuildValue("(O,i)", self->tree_sequence, self->tree->options);
    if (args == NULL) {
        goto out;
    }
    copy = (Tree *) PyObject_CallObject((PyObject *) &TreeType, args);
    if (copy == NULL) {
        goto out;
    }
    err = tsk_tree_copy(self->tree, copy->tree, TSK_NO_INIT);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = (PyObject *) copy;
    copy = NULL;
out:
    Py_XDECREF(args);
    Py_XDECREF(copy);
    return ret;
}

static PyObject *
Tree_get_kc_distance(Tree *self, PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    Tree *other = NULL;
    static char *kwlist[] = { "other", "lambda_", NULL };
    double lambda = 0;
    double result;
    int err;

    if (Tree_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "O!d", kwlist, &TreeType, &other, &lambda)) {
        goto out;
    }
    err = tsk_tree_kc_distance(self->tree, other->tree, lambda, &result);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("d", result);
out:
    return ret;
}

static PyObject *
Tree_get_root_threshold(Tree *self)
{
    PyObject *ret = NULL;

    if (Tree_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("I", (unsigned int) tsk_tree_get_root_threshold(self->tree));
out:
    return ret;
}

static PyObject *
Tree_set_root_threshold(Tree *self, PyObject *args)
{
    PyObject *ret = NULL;
    int err;
    unsigned int threshold = 0;

    if (Tree_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "I", &threshold)) {
        goto out;
    }

    err = tsk_tree_set_root_threshold(self->tree, threshold);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyMethodDef Tree_methods[] = {
    { .ml_name = "first",
        .ml_meth = (PyCFunction) Tree_first,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Sets this tree to the first in the sequence." },
    { .ml_name = "last",
        .ml_meth = (PyCFunction) Tree_last,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Sets this tree to the last in the sequence." },
    { .ml_name = "prev",
        .ml_meth = (PyCFunction) Tree_prev,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Sets this tree to the previous one in the sequence." },
    { .ml_name = "next",
        .ml_meth = (PyCFunction) Tree_next,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Sets this tree to the next one in the sequence." },
    { .ml_name = "clear",
        .ml_meth = (PyCFunction) Tree_clear,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Resets this tree back to the cleared null state." },
    { .ml_name = "get_sample_size",
        .ml_meth = (PyCFunction) Tree_get_sample_size,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns the number of samples in this tree." },
    { .ml_name = "get_num_nodes",
        .ml_meth = (PyCFunction) Tree_get_num_nodes,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns the number of nodes in this tree." },
    { .ml_name = "get_num_roots",
        .ml_meth = (PyCFunction) Tree_get_num_roots,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns the number of roots in this tree." },
    { .ml_name = "get_index",
        .ml_meth = (PyCFunction) Tree_get_index,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns the index this tree occupies within the tree sequence." },
    { .ml_name = "get_left_root",
        .ml_meth = (PyCFunction) Tree_get_left_root,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns the root of the tree." },
    { .ml_name = "get_left",
        .ml_meth = (PyCFunction) Tree_get_left,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns the left-most coordinate (inclusive)." },
    { .ml_name = "get_right",
        .ml_meth = (PyCFunction) Tree_get_right,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns the right-most coordinate (exclusive)." },
    { .ml_name = "get_sites",
        .ml_meth = (PyCFunction) Tree_get_sites,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns the list of sites on this tree." },
    { .ml_name = "get_options",
        .ml_meth = (PyCFunction) Tree_get_options,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns the value of the options variable." },
    { .ml_name = "get_num_sites",
        .ml_meth = (PyCFunction) Tree_get_num_sites,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns the number of sites on this tree." },
    { .ml_name = "is_sample",
        .ml_meth = (PyCFunction) Tree_is_sample,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns True if the specified node is a sample." },
    { .ml_name = "is_descendant",
        .ml_meth = (PyCFunction) Tree_is_descendant,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns True if u is a descendant of v." },
    { .ml_name = "depth",
        .ml_meth = (PyCFunction) Tree_depth,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns the depth of node u" },
    { .ml_name = "get_parent",
        .ml_meth = (PyCFunction) Tree_get_parent,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns the parent of node u" },
    { .ml_name = "get_time",
        .ml_meth = (PyCFunction) Tree_get_time,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns the time of node u" },
    { .ml_name = "get_population",
        .ml_meth = (PyCFunction) Tree_get_population,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns the population of node u" },
    { .ml_name = "get_left_child",
        .ml_meth = (PyCFunction) Tree_get_left_child,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns the left-most child of node u" },
    { .ml_name = "get_right_child",
        .ml_meth = (PyCFunction) Tree_get_right_child,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns the right-most child of node u" },
    { .ml_name = "get_left_sib",
        .ml_meth = (PyCFunction) Tree_get_left_sib,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns the left-most sib of node u" },
    { .ml_name = "get_right_sib",
        .ml_meth = (PyCFunction) Tree_get_right_sib,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns the right-most sib of node u" },
    { .ml_name = "get_children",
        .ml_meth = (PyCFunction) Tree_get_children,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns the children of u in left-right order." },
    { .ml_name = "get_left_sample",
        .ml_meth = (PyCFunction) Tree_get_left_sample,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns the index of the left-most sample descending from u." },
    { .ml_name = "get_right_sample",
        .ml_meth = (PyCFunction) Tree_get_right_sample,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns the index of the right-most sample descending from u." },
    { .ml_name = "get_next_sample",
        .ml_meth = (PyCFunction) Tree_get_next_sample,
        .ml_flags = METH_VARARGS,
        .ml_doc
        = "Returns the index of the next sample after the specified sample index." },
    { .ml_name = "get_mrca",
        .ml_meth = (PyCFunction) Tree_get_mrca,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns the MRCA of nodes u and v" },
    { .ml_name = "get_num_children",
        .ml_meth = (PyCFunction) Tree_get_num_children,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns the number of children of node u." },
    { .ml_name = "get_num_samples",
        .ml_meth = (PyCFunction) Tree_get_num_samples,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns the number of samples below node u." },
    { .ml_name = "get_num_tracked_samples",
        .ml_meth = (PyCFunction) Tree_get_num_tracked_samples,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns the number of tracked samples below node u." },
    { .ml_name = "get_newick",
        .ml_meth = (PyCFunction) Tree_get_newick,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Returns the newick representation of this tree." },
    { .ml_name = "map_mutations",
        .ml_meth = (PyCFunction) Tree_map_mutations,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc
        = "Returns a parsimonious state reconstruction for the specified genotypes." },
    { .ml_name = "equals",
        .ml_meth = (PyCFunction) Tree_equals,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns True if this tree is equal to the parameter tree." },
    { .ml_name = "copy",
        .ml_meth = (PyCFunction) Tree_copy,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns a copy of this tree." },
    { .ml_name = "get_kc_distance",
        .ml_meth = (PyCFunction) Tree_get_kc_distance,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Returns the KC distance between this tree and another." },
    { .ml_name = "set_root_threshold",
        .ml_meth = (PyCFunction) Tree_set_root_threshold,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Sets the root threshold to the specified value." },
    { .ml_name = "get_root_threshold",
        .ml_meth = (PyCFunction) Tree_get_root_threshold,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns the root threshold for this tree." },
    { NULL } /* Sentinel */
};

static PyTypeObject TreeType = {
    // clang-format off
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_tskit.Tree",
    .tp_basicsize = sizeof(Tree),
    .tp_dealloc = (destructor) Tree_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Tree objects",
    .tp_methods = Tree_methods,
    .tp_init = (initproc) Tree_init,
    .tp_new = PyType_GenericNew,
    // clang-format on
};

/*===================================================================
 * TreeDiffIterator
 *===================================================================
 */

static int
TreeDiffIterator_check_state(TreeDiffIterator *self)
{
    int ret = 0;
    if (self->tree_diff_iterator == NULL) {
        PyErr_SetString(PyExc_SystemError, "iterator not initialised");
        ret = -1;
    }
    return ret;
}

static void
TreeDiffIterator_dealloc(TreeDiffIterator *self)
{
    if (self->tree_diff_iterator != NULL) {
        tsk_diff_iter_free(self->tree_diff_iterator);
        PyMem_Free(self->tree_diff_iterator);
        self->tree_diff_iterator = NULL;
    }
    Py_XDECREF(self->tree_sequence);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static int
TreeDiffIterator_init(TreeDiffIterator *self, PyObject *args, PyObject *kwds)
{
    int ret = -1;
    int err;
    static char *kwlist[] = { "tree_sequence", "include_terminal", NULL };
    TreeSequence *tree_sequence;
    int include_terminal = 0;
    tsk_flags_t options = 0;

    self->tree_diff_iterator = NULL;
    self->tree_sequence = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|p", kwlist, &TreeSequenceType,
            &tree_sequence, &include_terminal)) {
        goto out;
    }
    if (include_terminal) {
        options |= TSK_INCLUDE_TERMINAL;
    }
    self->tree_sequence = tree_sequence;
    Py_INCREF(self->tree_sequence);
    if (TreeSequence_check_state(self->tree_sequence) != 0) {
        goto out;
    }
    self->tree_diff_iterator = PyMem_Malloc(sizeof(tsk_diff_iter_t));
    if (self->tree_diff_iterator == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    memset(self->tree_diff_iterator, 0, sizeof(tsk_diff_iter_t));
    err = tsk_diff_iter_init(
        self->tree_diff_iterator, self->tree_sequence->tree_sequence, options);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static PyObject *
TreeDiffIterator_next(TreeDiffIterator *self)
{
    PyObject *ret = NULL;
    PyObject *out_list = NULL;
    PyObject *in_list = NULL;
    PyObject *value = NULL;
    int err;
    double left, right;
    tsk_size_t list_size, j;
    tsk_edge_list_node_t *record;
    tsk_edge_list_t records_out, records_in;

    if (TreeDiffIterator_check_state(self) != 0) {
        goto out;
    }
    err = tsk_diff_iter_next(
        self->tree_diff_iterator, &left, &right, &records_out, &records_in);
    if (err < 0) {
        handle_library_error(err);
        goto out;
    }
    if (err == 1) {
        /* out records */
        record = records_out.head;
        list_size = 0;
        while (record != NULL) {
            list_size++;
            record = record->next;
        }
        out_list = PyList_New(list_size);
        if (out_list == NULL) {
            goto out;
        }
        record = records_out.head;
        j = 0;
        while (record != NULL) {
            value = make_edge(&record->edge, true);
            if (value == NULL) {
                goto out;
            }
            PyList_SET_ITEM(out_list, j, value);
            record = record->next;
            j++;
        }
        /* in records */
        record = records_in.head;
        list_size = 0;
        while (record != NULL) {
            list_size++;
            record = record->next;
        }
        in_list = PyList_New(list_size);
        if (in_list == NULL) {
            goto out;
        }
        record = records_in.head;
        j = 0;
        while (record != NULL) {
            value = make_edge(&record->edge, true);
            if (value == NULL) {
                goto out;
            }
            PyList_SET_ITEM(in_list, j, value);
            record = record->next;
            j++;
        }
        ret = Py_BuildValue("(dd)OO", left, right, out_list, in_list);
    }
out:
    Py_XDECREF(out_list);
    Py_XDECREF(in_list);
    return ret;
}

static PyTypeObject TreeDiffIteratorType = {
    // clang-format off
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_tskit.TreeDiffIterator",
    .tp_basicsize = sizeof(TreeDiffIterator),
    .tp_dealloc = (destructor) TreeDiffIterator_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "TreeDiffIterator objects",
    .tp_iter = PyObject_SelfIter,
    .tp_iternext = (iternextfunc) TreeDiffIterator_next,
    .tp_init = (initproc) TreeDiffIterator_init,
    .tp_new = PyType_GenericNew,
    // clang-format on
};

/*===================================================================
 * VariantGenerator
 *===================================================================
 */

static int
VariantGenerator_check_state(VariantGenerator *self)
{
    int ret = 0;
    if (self->variant_generator == NULL) {
        PyErr_SetString(PyExc_SystemError, "converter not initialised");
        ret = -1;
    }
    return ret;
}

static void
VariantGenerator_dealloc(VariantGenerator *self)
{
    if (self->variant_generator != NULL) {
        tsk_vargen_free(self->variant_generator);
        PyMem_Free(self->variant_generator);
        self->variant_generator = NULL;
    }
    Py_XDECREF(self->tree_sequence);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static int
VariantGenerator_init(VariantGenerator *self, PyObject *args, PyObject *kwds)
{
    int ret = -1;
    int err;
    static char *kwlist[]
        = { "tree_sequence", "samples", "isolated_as_missing", "alleles", NULL };
    TreeSequence *tree_sequence = NULL;
    PyObject *samples_input = Py_None;
    PyObject *py_alleles = Py_None;
    PyArrayObject *samples_array = NULL;
    tsk_id_t *samples = NULL;
    size_t num_samples = 0;
    int isolated_as_missing = 1;
    const char **alleles = NULL;
    npy_intp *shape;
    tsk_flags_t options = 0;

    /* TODO add option for 16 bit genotypes */
    self->variant_generator = NULL;
    self->tree_sequence = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|OiO", kwlist, &TreeSequenceType,
            &tree_sequence, &samples_input, &isolated_as_missing, &py_alleles)) {
        goto out;
    }
    if (!isolated_as_missing) {
        options |= TSK_ISOLATED_NOT_MISSING;
    }
    self->tree_sequence = tree_sequence;
    Py_INCREF(self->tree_sequence);
    if (TreeSequence_check_state(self->tree_sequence) != 0) {
        goto out;
    }
    if (samples_input != Py_None) {
        samples_array = (PyArrayObject *) PyArray_FROMANY(
            samples_input, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
        if (samples_array == NULL) {
            goto out;
        }
        shape = PyArray_DIMS(samples_array);
        num_samples = (size_t) shape[0];
        samples = PyArray_DATA(samples_array);
    }
    if (py_alleles != Py_None) {
        alleles = parse_allele_list(py_alleles);
        if (alleles == NULL) {
            goto out;
        }
    }
    self->variant_generator = PyMem_Malloc(sizeof(tsk_vargen_t));
    if (self->variant_generator == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    /* Note: the vargen currently takes a copy of the samples list. If we wanted
     * to avoid this we would INCREF the samples array above and keep a reference
     * to in the object struct */
    err = tsk_vargen_init(self->variant_generator, self->tree_sequence->tree_sequence,
        samples, num_samples, alleles, options);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    PyMem_Free(alleles);
    Py_XDECREF(samples_array);
    return ret;
}

static PyObject *
VariantGenerator_next(VariantGenerator *self)
{
    PyObject *ret = NULL;
    tsk_variant_t *var;
    int err;

    if (VariantGenerator_check_state(self) != 0) {
        goto out;
    }
    err = tsk_vargen_next(self->variant_generator, &var);
    if (err < 0) {
        handle_library_error(err);
        goto out;
    }
    if (err == 1) {
        ret = make_variant(var, self->variant_generator->num_samples);
    }
out:
    return ret;
}

static PyTypeObject VariantGeneratorType = {
    // clang-format off
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_tskit.VariantGenerator",
    .tp_basicsize = sizeof(VariantGenerator),
    .tp_dealloc = (destructor) VariantGenerator_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "VariantGenerator objects",
    .tp_iter = PyObject_SelfIter,
    .tp_iternext = (iternextfunc) VariantGenerator_next,
    .tp_init = (initproc) VariantGenerator_init,
    .tp_new = PyType_GenericNew,
    // clang-format on
};

/*===================================================================
 * LdCalculator
 *===================================================================
 */

static int
LdCalculator_check_state(LdCalculator *self)
{
    int ret = 0;
    if (self->ld_calc == NULL) {
        PyErr_SetString(PyExc_SystemError, "converter not initialised");
        ret = -1;
    }
    return ret;
}

static void
LdCalculator_dealloc(LdCalculator *self)
{
    if (self->ld_calc != NULL) {
        tsk_ld_calc_free(self->ld_calc);
        PyMem_Free(self->ld_calc);
        self->ld_calc = NULL;
    }
    Py_XDECREF(self->tree_sequence);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static int
LdCalculator_init(LdCalculator *self, PyObject *args, PyObject *kwds)
{
    int ret = -1;
    int err;
    static char *kwlist[] = { "tree_sequence", NULL };
    TreeSequence *tree_sequence;

    self->ld_calc = NULL;
    self->tree_sequence = NULL;
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "O!", kwlist, &TreeSequenceType, &tree_sequence)) {
        goto out;
    }
    self->tree_sequence = tree_sequence;
    Py_INCREF(self->tree_sequence);
    if (TreeSequence_check_state(self->tree_sequence) != 0) {
        goto out;
    }
    self->ld_calc = PyMem_Malloc(sizeof(tsk_ld_calc_t));
    if (self->ld_calc == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    memset(self->ld_calc, 0, sizeof(tsk_ld_calc_t));
    err = tsk_ld_calc_init(self->ld_calc, self->tree_sequence->tree_sequence);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static PyObject *
LdCalculator_get_r2(LdCalculator *self, PyObject *args)
{
    int err;
    PyObject *ret = NULL;
    Py_ssize_t a, b;
    double r2;

    if (LdCalculator_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "nn", &a, &b)) {
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS err
        = tsk_ld_calc_get_r2(self->ld_calc, (size_t) a, (size_t) b, &r2);
    Py_END_ALLOW_THREADS if (err != 0)
    {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("d", r2);
out:
    return ret;
}

/* TODO this implementation is brittle and cumbersome. Replace with something that
 * returns a numpy array directly. Passing in the memory is a premature optimisation.
 */
static PyObject *
LdCalculator_get_r2_array(LdCalculator *self, PyObject *args, PyObject *kwds)
{
    int err;
    PyObject *ret = NULL;
    static char *kwlist[]
        = { "dest", "source_index", "direction", "max_mutations", "max_distance", NULL };
    PyObject *dest = NULL;
    Py_buffer buffer;
    Py_ssize_t source_index;
    Py_ssize_t max_mutations = -1;
    double max_distance = DBL_MAX;
    int direction = TSK_DIR_FORWARD;
    tsk_size_t num_r2_values = 0;
    int buffer_acquired = 0;

    if (LdCalculator_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "On|ind", kwlist, &dest, &source_index,
            &direction, &max_mutations, &max_distance)) {
        goto out;
    }
    if (direction != TSK_DIR_FORWARD && direction != TSK_DIR_REVERSE) {
        PyErr_SetString(PyExc_ValueError, "direction must be FORWARD or REVERSE");
        goto out;
    }
    if (max_distance < 0) {
        PyErr_SetString(PyExc_ValueError, "max_distance must be >= 0");
        goto out;
    }
    if (!PyObject_CheckBuffer(dest)) {
        PyErr_SetString(
            PyExc_TypeError, "dest buffer must support the Python buffer protocol.");
        goto out;
    }
    if (PyObject_GetBuffer(dest, &buffer, PyBUF_SIMPLE | PyBUF_WRITABLE) != 0) {
        goto out;
    }
    buffer_acquired = 1;
    if (max_mutations == -1) {
        max_mutations = buffer.len / sizeof(double);
    } else if (max_mutations * sizeof(double) > (size_t) buffer.len) {
        PyErr_SetString(PyExc_BufferError, "dest buffer is too small for the results");
        goto out;
    }

    Py_BEGIN_ALLOW_THREADS err = tsk_ld_calc_get_r2_array(self->ld_calc,
        (tsk_id_t) source_index, direction, (tsk_size_t) max_mutations, max_distance,
        (double *) buffer.buf, &num_r2_values);
    Py_END_ALLOW_THREADS if (err != 0)
    {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("n", (Py_ssize_t) num_r2_values);
out:
    if (buffer_acquired) {
        PyBuffer_Release(&buffer);
    }
    return ret;
}

static PyMethodDef LdCalculator_methods[] = {
    { .ml_name = "get_r2",
        .ml_meth = (PyCFunction) LdCalculator_get_r2,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns the value of the r2 statistic between the specified pair of "
                  "mutation indexes" },
    { .ml_name = "get_r2_array",
        .ml_meth = (PyCFunction) LdCalculator_get_r2_array,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = "Returns r2 statistic for a given mutation over specified range" },
    { NULL } /* Sentinel */
};

static PyTypeObject LdCalculatorType = {
    // clang-format off
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_tskit.LdCalculator",
    .tp_basicsize = sizeof(LdCalculator),
    .tp_dealloc = (destructor) LdCalculator_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "LdCalculator objects",
    .tp_methods = LdCalculator_methods,
    .tp_init = (initproc) LdCalculator_init,
    .tp_new = PyType_GenericNew,
    // clang-format on
};

/*===================================================================
 * CompressedMatrix
 *===================================================================
 */

static int
CompressedMatrix_check_state(CompressedMatrix *self)
{
    int ret = -1;
    if (self->compressed_matrix == NULL) {
        PyErr_SetString(PyExc_SystemError, "CompressedMatrix not initialised");
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static void
CompressedMatrix_dealloc(CompressedMatrix *self)
{
    if (self->compressed_matrix != NULL) {
        tsk_compressed_matrix_free(self->compressed_matrix);
        PyMem_Free(self->compressed_matrix);
        self->compressed_matrix = NULL;
    }
    Py_XDECREF(self->tree_sequence);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static int
CompressedMatrix_init(CompressedMatrix *self, PyObject *args, PyObject *kwds)
{
    int ret = -1;
    int err;
    static char *kwlist[] = { "tree_sequence", "block_size", NULL };
    TreeSequence *tree_sequence = NULL;
    Py_ssize_t block_size = 0;

    self->compressed_matrix = NULL;
    self->tree_sequence = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|n", kwlist, &TreeSequenceType,
            &tree_sequence, &block_size)) {
        goto out;
    }
    self->tree_sequence = tree_sequence;
    Py_INCREF(self->tree_sequence);
    if (TreeSequence_check_state(self->tree_sequence) != 0) {
        goto out;
    }
    self->compressed_matrix = PyMem_Malloc(sizeof(tsk_compressed_matrix_t));
    if (self->compressed_matrix == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    memset(self->compressed_matrix, 0, sizeof(tsk_compressed_matrix_t));

    err = tsk_compressed_matrix_init(
        self->compressed_matrix, self->tree_sequence->tree_sequence, block_size, 0);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static PyObject *
CompressedMatrix_get_num_sites(CompressedMatrix *self, void *closure)
{
    PyObject *ret = NULL;

    if (CompressedMatrix_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("n", (Py_ssize_t) self->compressed_matrix->num_sites);
out:
    return ret;
}

static PyObject *
CompressedMatrix_get_normalisation_factor(CompressedMatrix *self, void *closure)
{
    PyObject *ret = NULL;
    PyArrayObject *array;
    size_t num_sites;
    npy_intp dims;

    if (CompressedMatrix_check_state(self) != 0) {
        goto out;
    }
    num_sites = self->compressed_matrix->num_sites;
    dims = (npy_intp) num_sites;
    array = (PyArrayObject *) PyArray_EMPTY(1, &dims, NPY_FLOAT64, 0);
    if (array == NULL) {
        goto out;
    }
    memcpy(PyArray_DATA(array), self->compressed_matrix->normalisation_factor,
        num_sites * sizeof(*self->compressed_matrix->normalisation_factor));
    ret = (PyObject *) array;
out:
    return ret;
}

static PyObject *
CompressedMatrix_get_num_transitions(CompressedMatrix *self, void *closure)
{
    PyObject *ret = NULL;
    PyArrayObject *array;
    size_t num_sites;
    npy_intp dims;

    if (CompressedMatrix_check_state(self) != 0) {
        goto out;
    }
    num_sites = self->compressed_matrix->num_sites;
    dims = (npy_intp) num_sites;
    array = (PyArrayObject *) PyArray_EMPTY(1, &dims, NPY_UINT32, 0);
    if (array == NULL) {
        goto out;
    }
    memcpy(PyArray_DATA(array), self->compressed_matrix->num_transitions,
        num_sites * sizeof(*self->compressed_matrix->num_transitions));
    ret = (PyObject *) array;
out:
    return ret;
}

static PyObject *
CompressedMatrix_get_site(CompressedMatrix *self, PyObject *args)
{
    PyObject *ret = NULL;
    unsigned int site;

    if (CompressedMatrix_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "I", &site)) {
        goto out;
    }
    ret = convert_compressed_matrix_site(self->compressed_matrix, site);
out:
    return ret;
}

static PyObject *
CompressedMatrix_decode(CompressedMatrix *self)
{
    PyObject *ret = NULL;
    if (CompressedMatrix_check_state(self) != 0) {
        goto out;
    }
    ret = decode_compressed_matrix(self->compressed_matrix);
out:
    return ret;
}

static PyGetSetDef CompressedMatrix_getsetters[] = {
    { .name = "num_sites",
        .get = (getter) CompressedMatrix_get_num_sites,
        .doc = "The number of sites." },
    { .name = "normalisation_factor",
        .get = (getter) CompressedMatrix_get_normalisation_factor,
        .doc = "The per-site normalisation factor." },
    { .name = "num_transitions",
        .get = (getter) CompressedMatrix_get_num_transitions,
        .doc = "The per-site number of transitions in the compressed matrix." },
    { NULL } /* Sentinel */
};

static PyMethodDef CompressedMatrix_methods[] = {
    { .ml_name = "get_site",
        .ml_meth = (PyCFunction) CompressedMatrix_get_site,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns the list of (node, value) tuples for the specified site." },
    { .ml_name = "decode",
        .ml_meth = (PyCFunction) CompressedMatrix_decode,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns the full decoded forward matrix." },
    { NULL } /* Sentinel */
};

static PyTypeObject CompressedMatrixType = {
    // clang-format off
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_tskit.CompressedMatrix",
    .tp_basicsize = sizeof(CompressedMatrix),
    .tp_dealloc = (destructor) CompressedMatrix_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "CompressedMatrix objects",
    .tp_methods = CompressedMatrix_methods,
    .tp_getset = CompressedMatrix_getsetters,
    .tp_init = (initproc) CompressedMatrix_init,
    .tp_new = PyType_GenericNew,
    // clang-format on
};

/*===================================================================
 * ViterbiMatrix
 *===================================================================
 */

static int
ViterbiMatrix_check_state(ViterbiMatrix *self)
{
    int ret = -1;
    if (self->viterbi_matrix == NULL) {
        PyErr_SetString(PyExc_SystemError, "ViterbiMatrix not initialised");
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static void
ViterbiMatrix_dealloc(ViterbiMatrix *self)
{
    if (self->viterbi_matrix != NULL) {
        tsk_viterbi_matrix_free(self->viterbi_matrix);
        PyMem_Free(self->viterbi_matrix);
        self->viterbi_matrix = NULL;
    }
    Py_XDECREF(self->tree_sequence);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static int
ViterbiMatrix_init(ViterbiMatrix *self, PyObject *args, PyObject *kwds)
{
    int ret = -1;
    int err;
    static char *kwlist[] = { "tree_sequence", "num_records", NULL };
    TreeSequence *tree_sequence = NULL;
    Py_ssize_t num_records = 0;

    self->viterbi_matrix = NULL;
    self->tree_sequence = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|n", kwlist, &TreeSequenceType,
            &tree_sequence, &num_records)) {
        goto out;
    }
    self->tree_sequence = tree_sequence;
    Py_INCREF(self->tree_sequence);
    if (TreeSequence_check_state(self->tree_sequence) != 0) {
        goto out;
    }
    self->viterbi_matrix = PyMem_Malloc(sizeof(tsk_viterbi_matrix_t));
    if (self->viterbi_matrix == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    memset(self->viterbi_matrix, 0, sizeof(tsk_viterbi_matrix_t));

    err = tsk_viterbi_matrix_init(
        self->viterbi_matrix, self->tree_sequence->tree_sequence, num_records, 0);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static PyObject *
ViterbiMatrix_traceback(ViterbiMatrix *self)
{
    PyObject *ret = NULL;
    PyArrayObject *path = NULL;
    npy_intp dims;
    int err;

    if (ViterbiMatrix_check_state(self) != 0) {
        goto out;
    }
    dims = self->viterbi_matrix->matrix.num_sites;
    path = (PyArrayObject *) PyArray_SimpleNew(1, &dims, NPY_INT32);
    if (path == NULL) {
        goto out;
    }

    err = tsk_viterbi_matrix_traceback(self->viterbi_matrix, PyArray_DATA(path), 0);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = (PyObject *) path;
    path = NULL;
out:
    Py_XDECREF(path);
    return ret;
}

static PyObject *
ViterbiMatrix_get_num_sites(ViterbiMatrix *self, void *closure)
{
    PyObject *ret = NULL;

    if (ViterbiMatrix_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("n", (Py_ssize_t) self->viterbi_matrix->matrix.num_sites);
out:
    return ret;
}

/* NOTE: We're doing something pretty ugly here in that we're duplicating the
 * methods from the CompressedMatrix class to provide access to the
 * viterbi_matrix struct's embedded compressed_matrix. It would be more
 * elegant if the ViterbiMatrix class had a CompressedMatrix member,
 * but the memory management is tricky, so it doesn't seem worth the
 * hassle.
 */

static PyObject *
ViterbiMatrix_get_normalisation_factor(ViterbiMatrix *self, void *closure)
{
    PyObject *ret = NULL;
    PyArrayObject *array;
    size_t num_sites;
    npy_intp dims;

    if (ViterbiMatrix_check_state(self) != 0) {
        goto out;
    }
    num_sites = self->viterbi_matrix->matrix.num_sites;
    dims = (npy_intp) num_sites;
    array = (PyArrayObject *) PyArray_EMPTY(1, &dims, NPY_FLOAT64, 0);
    if (array == NULL) {
        goto out;
    }
    memcpy(PyArray_DATA(array), self->viterbi_matrix->matrix.normalisation_factor,
        num_sites * sizeof(*self->viterbi_matrix->matrix.normalisation_factor));
    ret = (PyObject *) array;
out:
    return ret;
}

static PyObject *
ViterbiMatrix_get_num_transitions(ViterbiMatrix *self, void *closure)
{
    PyObject *ret = NULL;
    PyArrayObject *array;
    size_t num_sites;
    npy_intp dims;

    if (ViterbiMatrix_check_state(self) != 0) {
        goto out;
    }
    num_sites = self->viterbi_matrix->matrix.num_sites;
    dims = (npy_intp) num_sites;

    array = (PyArrayObject *) PyArray_EMPTY(1, &dims, NPY_UINT32, 0);
    if (array == NULL) {
        goto out;
    }
    memcpy(PyArray_DATA(array), self->viterbi_matrix->matrix.num_transitions,
        num_sites * sizeof(*self->viterbi_matrix->matrix.num_transitions));
    ret = (PyObject *) array;
out:
    return ret;
}

static PyObject *
ViterbiMatrix_get_site(ViterbiMatrix *self, PyObject *args)
{
    PyObject *ret = NULL;
    unsigned int site;

    if (ViterbiMatrix_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "I", &site)) {
        goto out;
    }
    ret = convert_compressed_matrix_site(&self->viterbi_matrix->matrix, site);
out:
    return ret;
}

static PyObject *
ViterbiMatrix_decode(ViterbiMatrix *self)
{
    PyObject *ret = NULL;
    if (ViterbiMatrix_check_state(self) != 0) {
        goto out;
    }
    ret = decode_compressed_matrix(&self->viterbi_matrix->matrix);
out:
    return ret;
}

static PyGetSetDef ViterbiMatrix_getsetters[] = {
    { .name = "num_sites",
        .get = (getter) ViterbiMatrix_get_num_sites,
        .doc = "The number of sites." },
    { .name = "normalisation_factor",
        .get = (getter) ViterbiMatrix_get_normalisation_factor,
        .doc = "The per-site normalisation factor." },
    { .name = "num_transitions",
        .get = (getter) ViterbiMatrix_get_num_transitions,
        .doc = "The per-site number of transitions in the compressed matrix." },
    { NULL } /* Sentinel */
};

static PyMethodDef ViterbiMatrix_methods[] = {
    { .ml_name = "traceback",
        .ml_meth = (PyCFunction) ViterbiMatrix_traceback,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns a path for a given haplotype." },
    { .ml_name = "get_site",
        .ml_meth = (PyCFunction) ViterbiMatrix_get_site,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns the list of (node, value) tuples for the specified site." },
    { .ml_name = "decode",
        .ml_meth = (PyCFunction) ViterbiMatrix_decode,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns the full decoded forward matrix." },
    { NULL } /* Sentinel */
};

static PyTypeObject ViterbiMatrixType = {
    // clang-format off
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_tskit.ViterbiMatrix",
    .tp_basicsize = sizeof(ViterbiMatrix),
    .tp_dealloc = (destructor) ViterbiMatrix_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "ViterbiMatrix objects",
    .tp_methods = ViterbiMatrix_methods,
    .tp_getset = ViterbiMatrix_getsetters,
    .tp_init = (initproc) ViterbiMatrix_init,
    .tp_new = PyType_GenericNew,
    // clang-format on
};

/*===================================================================
 * LsHmm
 *===================================================================
 */

static int
LsHmm_check_state(LsHmm *self)
{
    int ret = -1;
    if (self->ls_hmm == NULL) {
        PyErr_SetString(PyExc_SystemError, "LsHmm not initialised");
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static void
LsHmm_dealloc(LsHmm *self)
{
    if (self->ls_hmm != NULL) {
        tsk_ls_hmm_free(self->ls_hmm);
        PyMem_Free(self->ls_hmm);
        self->ls_hmm = NULL;
    }
    Py_XDECREF(self->tree_sequence);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static int
LsHmm_init(LsHmm *self, PyObject *args, PyObject *kwds)
{
    int ret = -1;
    int err;
    static char *kwlist[] = { "tree_sequence", "recombination_rate", "mutation_rate",
        "precision", "acgt_alleles", NULL };
    PyObject *recombination_rate = NULL;
    PyArrayObject *recombination_rate_array = NULL;
    PyObject *mutation_rate = NULL;
    PyArrayObject *mutation_rate_array = NULL;
    TreeSequence *tree_sequence = NULL;
    unsigned int precision = 23;
    int acgt_alleles = 0;
    tsk_flags_t options = 0;
    npy_intp *shape, num_sites;

    self->ls_hmm = NULL;
    self->tree_sequence = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!OO|Ii", kwlist, &TreeSequenceType,
            &tree_sequence, &recombination_rate, &mutation_rate, &precision,
            &acgt_alleles)) {
        goto out;
    }
    self->tree_sequence = tree_sequence;
    Py_INCREF(self->tree_sequence);
    if (TreeSequence_check_state(self->tree_sequence) != 0) {
        goto out;
    }
    self->ls_hmm = PyMem_Malloc(sizeof(tsk_ls_hmm_t));
    if (self->ls_hmm == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    memset(self->ls_hmm, 0, sizeof(tsk_ls_hmm_t));

    num_sites = (npy_intp) tsk_treeseq_get_num_sites(self->tree_sequence->tree_sequence);
    recombination_rate_array = (PyArrayObject *) PyArray_FROMANY(
        recombination_rate, NPY_FLOAT64, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (recombination_rate_array == NULL) {
        goto out;
    }
    shape = PyArray_DIMS(recombination_rate_array);
    if (shape[0] != num_sites) {
        PyErr_SetString(PyExc_ValueError,
            "recombination_rate array must have dimension (num_sites,)");
        goto out;
    }
    mutation_rate_array = (PyArrayObject *) PyArray_FROMANY(
        mutation_rate, NPY_FLOAT64, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (mutation_rate_array == NULL) {
        goto out;
    }
    shape = PyArray_DIMS(mutation_rate_array);
    if (shape[0] != num_sites) {
        PyErr_SetString(
            PyExc_ValueError, "mutation_rate array must have dimension (num_sites,)");
        goto out;
    }
    if (acgt_alleles) {
        options |= TSK_ALLELES_ACGT;
    }

    err = tsk_ls_hmm_init(self->ls_hmm, self->tree_sequence->tree_sequence,
        PyArray_DATA(recombination_rate_array), PyArray_DATA(mutation_rate_array),
        options);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    tsk_ls_hmm_set_precision(self->ls_hmm, precision);
    ret = 0;
out:
    Py_XDECREF(recombination_rate_array);
    Py_XDECREF(mutation_rate_array);
    return ret;
}

static PyObject *
LsHmm_forward_matrix(LsHmm *self, PyObject *args)
{
    int err;
    PyObject *ret = NULL;
    PyObject *haplotype = NULL;
    CompressedMatrix *compressed_matrix = NULL;
    PyArrayObject *haplotype_array = NULL;
    npy_intp *shape, num_sites;

    if (LsHmm_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(
            args, "OO!", &haplotype, &CompressedMatrixType, &compressed_matrix)) {
        goto out;
    }
    num_sites = (npy_intp) tsk_treeseq_get_num_sites(self->tree_sequence->tree_sequence);
    haplotype_array = (PyArrayObject *) PyArray_FROMANY(
        haplotype, NPY_INT8, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (haplotype_array == NULL) {
        goto out;
    }
    shape = PyArray_DIMS(haplotype_array);
    if (shape[0] != num_sites) {
        PyErr_SetString(
            PyExc_ValueError, "haplotype array must have dimension (num_sites,)");
        goto out;
    }
    err = tsk_ls_hmm_forward(self->ls_hmm, PyArray_DATA(haplotype_array),
        compressed_matrix->compressed_matrix, TSK_NO_INIT);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    Py_XDECREF(haplotype_array);
    return ret;
}

static PyObject *
LsHmm_viterbi_matrix(LsHmm *self, PyObject *args)
{
    int err;
    PyObject *ret = NULL;
    PyObject *haplotype = NULL;
    ViterbiMatrix *viterbi_matrix = NULL;
    PyArrayObject *haplotype_array = NULL;
    npy_intp *shape, num_sites;

    if (LsHmm_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(
            args, "OO!", &haplotype, &ViterbiMatrixType, &viterbi_matrix)) {
        goto out;
    }
    num_sites = (npy_intp) tsk_treeseq_get_num_sites(self->tree_sequence->tree_sequence);
    haplotype_array = (PyArrayObject *) PyArray_FROMANY(
        haplotype, NPY_INT8, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (haplotype_array == NULL) {
        goto out;
    }
    shape = PyArray_DIMS(haplotype_array);
    if (shape[0] != num_sites) {
        PyErr_SetString(
            PyExc_ValueError, "haplotype array must have dimension (num_sites,)");
        goto out;
    }
    err = tsk_ls_hmm_viterbi(self->ls_hmm, PyArray_DATA(haplotype_array),
        viterbi_matrix->viterbi_matrix, TSK_NO_INIT);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    Py_XDECREF(haplotype_array);
    return ret;
}

static PyMethodDef LsHmm_methods[] = {
    { .ml_name = "forward_matrix",
        .ml_meth = (PyCFunction) LsHmm_forward_matrix,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns the tree encoded forward matrix for a given haplotype" },
    { .ml_name = "viterbi_matrix",
        .ml_meth = (PyCFunction) LsHmm_viterbi_matrix,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns the tree encoded Viterbi matrix for a given haplotype" },
    { NULL } /* Sentinel */
};

static PyTypeObject LsHmmType = {
    // clang-format off
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_tskit.LsHmm",
    .tp_basicsize = sizeof(LsHmm),
    .tp_dealloc = (destructor) LsHmm_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "LsHmm objects",
    .tp_methods = LsHmm_methods,
    .tp_init = (initproc) LsHmm_init,
    .tp_new = PyType_GenericNew,
    // clang-format on
};

/*===================================================================
 * Module level functions
 *===================================================================
 */

static PyObject *
tskit_get_kastore_version(PyObject *self)
{
    return Py_BuildValue("iii", KAS_VERSION_MAJOR, KAS_VERSION_MINOR, KAS_VERSION_PATCH);
}

static PyObject *
tskit_get_tskit_version(PyObject *self)
{
    return Py_BuildValue("iii", TSK_VERSION_MAJOR, TSK_VERSION_MINOR, TSK_VERSION_PATCH);
}

static PyMethodDef tskit_methods[] = {
    { .ml_name = "get_kastore_version",
        .ml_meth = (PyCFunction) tskit_get_kastore_version,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns the version of kastore we have built in." },
    { .ml_name = "get_tskit_version",
        .ml_meth = (PyCFunction) tskit_get_tskit_version,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns the version of the tskit C API we have built in." },
    { NULL } /* Sentinel */
};

static struct PyModuleDef tskitmodule = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_tskit",
    .m_doc = "Low level interface for tskit",
    .m_size = -1,
    .m_methods = tskit_methods,
};

PyObject *
PyInit__tskit(void)
{
    PyObject *module = PyModule_Create(&tskitmodule);
    if (module == NULL) {
        return NULL;
    }
    import_array();

    if (register_lwt_class(module) != 0) {
        return NULL;
    }

    /* IndividualTable type */
    if (PyType_Ready(&IndividualTableType) < 0) {
        return NULL;
    }
    Py_INCREF(&IndividualTableType);
    PyModule_AddObject(module, "IndividualTable", (PyObject *) &IndividualTableType);

    /* NodeTable type */
    if (PyType_Ready(&NodeTableType) < 0) {
        return NULL;
    }
    Py_INCREF(&NodeTableType);
    PyModule_AddObject(module, "NodeTable", (PyObject *) &NodeTableType);

    /* EdgeTable type */
    if (PyType_Ready(&EdgeTableType) < 0) {
        return NULL;
    }
    Py_INCREF(&EdgeTableType);
    PyModule_AddObject(module, "EdgeTable", (PyObject *) &EdgeTableType);

    /* MigrationTable type */
    if (PyType_Ready(&MigrationTableType) < 0) {
        return NULL;
    }
    Py_INCREF(&MigrationTableType);
    PyModule_AddObject(module, "MigrationTable", (PyObject *) &MigrationTableType);

    /* SiteTable type */
    if (PyType_Ready(&SiteTableType) < 0) {
        return NULL;
    }
    Py_INCREF(&SiteTableType);
    PyModule_AddObject(module, "SiteTable", (PyObject *) &SiteTableType);

    /* MutationTable type */
    if (PyType_Ready(&MutationTableType) < 0) {
        return NULL;
    }
    Py_INCREF(&MutationTableType);
    PyModule_AddObject(module, "MutationTable", (PyObject *) &MutationTableType);

    /* PopulationTable type */
    if (PyType_Ready(&PopulationTableType) < 0) {
        return NULL;
    }
    Py_INCREF(&PopulationTableType);
    PyModule_AddObject(module, "PopulationTable", (PyObject *) &PopulationTableType);

    /* ProvenanceTable type */
    if (PyType_Ready(&ProvenanceTableType) < 0) {
        return NULL;
    }
    Py_INCREF(&ProvenanceTableType);
    PyModule_AddObject(module, "ProvenanceTable", (PyObject *) &ProvenanceTableType);

    /* TableCollectionTable type */
    if (PyType_Ready(&TableCollectionType) < 0) {
        return NULL;
    }
    Py_INCREF(&TableCollectionType);
    PyModule_AddObject(module, "TableCollection", (PyObject *) &TableCollectionType);

    /* TreeSequence type */
    if (PyType_Ready(&TreeSequenceType) < 0) {
        return NULL;
    }
    Py_INCREF(&TreeSequenceType);
    PyModule_AddObject(module, "TreeSequence", (PyObject *) &TreeSequenceType);

    /* Tree type */
    if (PyType_Ready(&TreeType) < 0) {
        return NULL;
    }
    Py_INCREF(&TreeType);
    PyModule_AddObject(module, "Tree", (PyObject *) &TreeType);

    /* TreeDiffIterator type */
    if (PyType_Ready(&TreeDiffIteratorType) < 0) {
        return NULL;
    }
    Py_INCREF(&TreeDiffIteratorType);
    PyModule_AddObject(module, "TreeDiffIterator", (PyObject *) &TreeDiffIteratorType);

    /* VariantGenerator type */
    if (PyType_Ready(&VariantGeneratorType) < 0) {
        return NULL;
    }
    Py_INCREF(&VariantGeneratorType);
    PyModule_AddObject(module, "VariantGenerator", (PyObject *) &VariantGeneratorType);

    /* LdCalculator type */
    if (PyType_Ready(&LdCalculatorType) < 0) {
        return NULL;
    }
    Py_INCREF(&LdCalculatorType);
    PyModule_AddObject(module, "LdCalculator", (PyObject *) &LdCalculatorType);

    /* CompressedMatrix type */
    if (PyType_Ready(&CompressedMatrixType) < 0) {
        return NULL;
    }
    Py_INCREF(&CompressedMatrixType);
    PyModule_AddObject(module, "CompressedMatrix", (PyObject *) &CompressedMatrixType);

    /* ViterbiMatrix type */
    if (PyType_Ready(&ViterbiMatrixType) < 0) {
        return NULL;
    }
    Py_INCREF(&ViterbiMatrixType);
    PyModule_AddObject(module, "ViterbiMatrix", (PyObject *) &ViterbiMatrixType);

    /* LsHmm type */
    if (PyType_Ready(&LsHmmType) < 0) {
        return NULL;
    }
    Py_INCREF(&LsHmmType);
    PyModule_AddObject(module, "LsHmm", (PyObject *) &LsHmmType);

    /* Metadata schemas namedtuple type*/
    if (PyStructSequence_InitType2(&MetadataSchemas, &metadata_schemas_desc) < 0) {
        return NULL;
    };
    Py_INCREF(&MetadataSchemas);
    PyModule_AddObject(module, "MetadataSchemas", (PyObject *) &MetadataSchemas);

    /* Errors and constants */
    TskitException = PyErr_NewException("_tskit.TskitException", NULL, NULL);
    Py_INCREF(TskitException);
    PyModule_AddObject(module, "TskitException", TskitException);
    TskitLibraryError = PyErr_NewException("_tskit.LibraryError", TskitException, NULL);
    Py_INCREF(TskitLibraryError);
    PyModule_AddObject(module, "LibraryError", TskitLibraryError);
    TskitFileFormatError = PyErr_NewException("_tskit.FileFormatError", NULL, NULL);
    Py_INCREF(TskitFileFormatError);
    PyModule_AddObject(module, "FileFormatError", TskitFileFormatError);
    TskitVersionTooNewError
        = PyErr_NewException("_tskit.VersionTooNewError", TskitException, NULL);
    Py_INCREF(TskitVersionTooNewError);
    PyModule_AddObject(module, "VersionTooNewError", TskitVersionTooNewError);
    TskitVersionTooOldError
        = PyErr_NewException("_tskit.VersionTooOldError", TskitException, NULL);
    Py_INCREF(TskitVersionTooOldError);
    PyModule_AddObject(module, "VersionTooOldError", TskitVersionTooOldError);

    PyModule_AddIntConstant(module, "NULL", TSK_NULL);
    PyModule_AddIntConstant(module, "MISSING_DATA", TSK_MISSING_DATA);

    PyObject *unknown_time = PyFloat_FromDouble(TSK_UNKNOWN_TIME);
    PyModule_AddObject(module, "UNKNOWN_TIME", unknown_time);

    /* Node flags */
    PyModule_AddIntConstant(module, "NODE_IS_SAMPLE", TSK_NODE_IS_SAMPLE);
    /* Tree flags */
    PyModule_AddIntConstant(module, "NO_SAMPLE_COUNTS", TSK_NO_SAMPLE_COUNTS);
    PyModule_AddIntConstant(module, "SAMPLE_LISTS", TSK_SAMPLE_LISTS);
    /* Directions */
    PyModule_AddIntConstant(module, "FORWARD", TSK_DIR_FORWARD);
    PyModule_AddIntConstant(module, "REVERSE", TSK_DIR_REVERSE);

    return module;
}
