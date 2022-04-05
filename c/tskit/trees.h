/*
 * MIT License
 *
 * Copyright (c) 2019-2022 Tskit Developers
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

/**
 * @file trees.h
 * @brief Tskit core tree sequence operations.
 */
#ifndef TSK_TREES_H
#define TSK_TREES_H

#ifdef __cplusplus
extern "C" {
#endif

#include <tskit/tables.h>

// clang-format off

/*
 * These are both undocumented options for tsk_tree_init
 */
#define TSK_SAMPLE_LISTS            (1 << 1)
#define TSK_NO_SAMPLE_COUNTS        (1 << 2)

#define TSK_STAT_SITE               (1 << 0)
#define TSK_STAT_BRANCH             (1 << 1)
#define TSK_STAT_NODE               (1 << 2)

/* Leave room for other stat types */
#define TSK_STAT_POLARISED               (1 << 10)
#define TSK_STAT_SPAN_NORMALISE          (1 << 11)
#define TSK_STAT_ALLOW_TIME_UNCALIBRATED (1 << 12)

/* Options for map_mutations */
#define TSK_MM_FIXED_ANCESTRAL_STATE (1 << 0)

#define TSK_DIR_FORWARD 1
#define TSK_DIR_REVERSE -1

/* For the edge diff iterator */
#define TSK_INCLUDE_TERMINAL        (1 << 0)
// clang-format on

/**
@brief The tree sequence object.
*/
typedef struct {
    tsk_size_t num_trees;
    tsk_size_t num_samples;
    tsk_id_t *samples;
    /* Does this tree sequence have time_units == "uncalibrated" */
    bool time_uncalibrated;
    /* Are all genome coordinates discrete? */
    bool discrete_genome;
    /* Are all time values discrete? */
    bool discrete_time;
    /* Breakpoints along the sequence, including 0 and L. */
    double *breakpoints;
    /* If a node is a sample, map to its index in the samples list */
    tsk_id_t *sample_index_map;
    /* Map individuals to the list of nodes that reference them */
    tsk_id_t *individual_nodes_mem;
    tsk_id_t **individual_nodes;
    tsk_size_t *individual_nodes_length;
    /* For each tree, a list of sites on that tree */
    tsk_site_t *tree_sites_mem;
    tsk_site_t **tree_sites;
    tsk_size_t *tree_sites_length;
    /* For each site, a list of mutations at that site */
    tsk_mutation_t *site_mutations_mem;
    tsk_mutation_t **site_mutations;
    tsk_size_t *site_mutations_length;
    /** @brief  The table collection underlying this tree sequence, This table
     *  collection must be treated as read-only, and any changes to it will
     *  lead to undefined behaviour. */
    tsk_table_collection_t *tables;
} tsk_treeseq_t;

/**
@brief A single tree in a tree sequence.

@rst
A ``tsk_tree_t`` object has two basic functions:

1. Represent the state of a single tree in a tree sequence;
2. Provide methods to transform this state into different trees in the sequence.

The state of a single tree in the tree sequence is represented using the
quintuply linked encoding: please see the
:ref:`data model <sec_data_model_tree_structure>` section for details on
how this works. The left-to-right ordering of nodes in this encoding
is arbitrary, and may change depending on the order in which trees are
accessed within the sequence. Please see the
:ref:`sec_c_api_examples_tree_traversals` examples for recommended
usage.

On initialisation, a tree is in the :ref:`null state<sec_c_api_trees_null>` and
we must call one of the :ref:`seeking<sec_c_api_trees_seeking>` methods to make
the state of the tree object correspond to a particular tree in the sequence.
Please see the :ref:`sec_c_api_examples_tree_iteration` examples for
recommended usage.

@endrst
 */
typedef struct {
    /**
     * @brief The parent tree sequence.
     */
    const tsk_treeseq_t *tree_sequence;
    /**
     @brief The ID of the "virtual root" whose children are the roots of the
     tree.
     */
    tsk_id_t virtual_root;
    /**
     @brief The parent of node u is parent[u]. Equal to TSK_NULL if node u is a
     root or is not a node in the current tree.
     */
    tsk_id_t *parent;
    /**
     @brief The leftmost child of node u is left_child[u]. Equal to TSK_NULL
     if node u is a leaf or is not a node in the current tree.
     */
    tsk_id_t *left_child;
    /**
     @brief The rightmost child of node u is right_child[u]. Equal to TSK_NULL
     if node u is a leaf or is not a node in the current tree.
     */
    tsk_id_t *right_child;
    /**
     @brief The sibling to the left of node u is left_sib[u]. Equal to TSK_NULL
     if node u has no siblings to its left.
     */
    tsk_id_t *left_sib;
    /**
     @brief The sibling to the right of node u is right_sib[u]. Equal to TSK_NULL
     if node u has no siblings to its right.
     */
    tsk_id_t *right_sib;
    /**
     @brief The total number of edges defining the topology of this tree.
     This is equal to the number of tree sequence edges that intersect with
     the tree's genomic interval.
     */
    tsk_size_t num_edges;
    /**
     @brief Left and right coordinates of the genomic interval that this
     tree covers. The left coordinate is inclusive and the right coordinate
     exclusive.

    @rst

    Example:

    .. code-block:: c

        tsk_tree_t tree;
        int ret;
        // initialise etc
        ret = tsk_tree_first(&tree);
        // Check for error
        assert(ret == TSK_TREE_OK);
        printf("Coordinates covered by first tree are left=%f, right=%f\n",
            tree.interval.left, tree.interval.right);

    @endrst

     */
    struct {
        double left;
        double right;
    } interval;
    /**
     @brief The index of this tree in the tree sequence.

     @rst
     This attribute provides the zero-based index of the tree represented by the
     current state of the struct within the parent tree sequence. For example,
     immediately after we call ``tsk_tree_first(&tree)``, ``tree.index`` will
     be zero, and after we call ``tsk_tree_last(&tree)``, ``tree.index`` will
     be the number of trees - 1 (see :c:func:`tsk_treeseq_get_num_trees`)

     When the tree is in the null state (immediately after initialisation,
     or after, e.g., calling :c:func:`tsk_tree_prev` on the first tree)
     the value of the ``index`` is -1.
     @endrst
     */
    tsk_id_t index;
    /* Attributes below are private and should not be used in client code. */
    tsk_size_t num_nodes;
    tsk_flags_t options;
    tsk_size_t root_threshold;
    const tsk_id_t *samples;
    /* These are involved in the optional sample tracking; num_samples counts
     * all samples below a give node, and num_tracked_samples counts those
     * from a specific subset. By default sample counts are tracked and roots
     * maintained. If TSK_NO_SAMPLE_COUNTS is specified, then neither sample
     * counts or roots are available. */
    tsk_size_t *num_samples;
    tsk_size_t *num_tracked_samples;
    /* These are for the optional sample list tracking. */
    tsk_id_t *left_sample;
    tsk_id_t *right_sample;
    tsk_id_t *next_sample;
    /* The sites on this tree */
    const tsk_site_t *sites;
    tsk_size_t sites_length;
    /* Counters needed for next() and prev() transformations. */
    int direction;
    tsk_id_t left_index;
    tsk_id_t right_index;
} tsk_tree_t;

/* Diff iterator. */
typedef struct _tsk_edge_list_node_t {
    tsk_edge_t edge;
    struct _tsk_edge_list_node_t *next;
    struct _tsk_edge_list_node_t *prev;
} tsk_edge_list_node_t;

typedef struct {
    tsk_edge_list_node_t *head;
    tsk_edge_list_node_t *tail;
} tsk_edge_list_t;

typedef struct {
    tsk_size_t num_nodes;
    tsk_size_t num_edges;
    double tree_left;
    const tsk_treeseq_t *tree_sequence;
    tsk_id_t insertion_index;
    tsk_id_t removal_index;
    tsk_id_t tree_index;
    tsk_id_t last_index;
    tsk_edge_list_node_t *edge_list_nodes;
} tsk_diff_iter_t;

/****************************************************************************/
/* Tree sequence.*/
/****************************************************************************/

/**
@defgroup TREESEQ_API_GROUP Tree sequence API
@{
*/
int tsk_treeseq_init(
    tsk_treeseq_t *self, tsk_table_collection_t *tables, tsk_flags_t options);

int tsk_treeseq_load(tsk_treeseq_t *self, const char *filename, tsk_flags_t options);
int tsk_treeseq_loadf(tsk_treeseq_t *self, FILE *file, tsk_flags_t options);

int tsk_treeseq_dump(
    const tsk_treeseq_t *self, const char *filename, tsk_flags_t options);
int tsk_treeseq_dumpf(const tsk_treeseq_t *self, FILE *file, tsk_flags_t options);
int tsk_treeseq_copy_tables(
    const tsk_treeseq_t *self, tsk_table_collection_t *tables, tsk_flags_t options);

/**
@brief Free the internal memory for the specified tree sequence.

@param self A pointer to an initialised tsk_treeseq_t object.
@return Always returns 0.
*/
int tsk_treeseq_free(tsk_treeseq_t *self);

/**
@brief Print out the state of this tree sequence to the specified stream.

This method is intended for debugging purposes and should not be used
in production code. The format of the output should **not** be depended
on and may change arbitrarily between versions.

@param self A pointer to a tsk_treeseq_t object.
@param out The stream to write the summary to.
*/
void tsk_treeseq_print_state(const tsk_treeseq_t *self, FILE *out);

/**
@brief Return the number of trees in this tree sequence.

@rst
This is a constant time operation.
@endrst

@param self A pointer to a tsk_treeseq_t object.
@return The number of trees in the tree sequence.
*/
tsk_size_t tsk_treeseq_get_num_trees(const tsk_treeseq_t *self);

/** @} */

bool tsk_treeseq_has_reference_sequence(const tsk_treeseq_t *self);
tsk_size_t tsk_treeseq_get_num_nodes(const tsk_treeseq_t *self);
tsk_size_t tsk_treeseq_get_num_edges(const tsk_treeseq_t *self);
tsk_size_t tsk_treeseq_get_num_migrations(const tsk_treeseq_t *self);
tsk_size_t tsk_treeseq_get_num_sites(const tsk_treeseq_t *self);
tsk_size_t tsk_treeseq_get_num_mutations(const tsk_treeseq_t *self);
tsk_size_t tsk_treeseq_get_num_provenances(const tsk_treeseq_t *self);
tsk_size_t tsk_treeseq_get_num_populations(const tsk_treeseq_t *self);
tsk_size_t tsk_treeseq_get_num_individuals(const tsk_treeseq_t *self);

tsk_size_t tsk_treeseq_get_num_samples(const tsk_treeseq_t *self);
const char *tsk_treeseq_get_metadata(const tsk_treeseq_t *self);
tsk_size_t tsk_treeseq_get_metadata_length(const tsk_treeseq_t *self);
const char *tsk_treeseq_get_metadata_schema(const tsk_treeseq_t *self);
tsk_size_t tsk_treeseq_get_metadata_schema_length(const tsk_treeseq_t *self);
const char *tsk_treeseq_get_time_units(const tsk_treeseq_t *self);
tsk_size_t tsk_treeseq_get_time_units_length(const tsk_treeseq_t *self);
const char *tsk_treeseq_get_file_uuid(const tsk_treeseq_t *self);
double tsk_treeseq_get_sequence_length(const tsk_treeseq_t *self);
const double *tsk_treeseq_get_breakpoints(const tsk_treeseq_t *self);
const tsk_id_t *tsk_treeseq_get_samples(const tsk_treeseq_t *self);
const tsk_id_t *tsk_treeseq_get_sample_index_map(const tsk_treeseq_t *self);
bool tsk_treeseq_is_sample(const tsk_treeseq_t *self, tsk_id_t u);
bool tsk_treeseq_get_discrete_genome(const tsk_treeseq_t *self);
bool tsk_treeseq_get_discrete_time(const tsk_treeseq_t *self);

int tsk_treeseq_get_node(const tsk_treeseq_t *self, tsk_id_t index, tsk_node_t *node);
int tsk_treeseq_get_edge(const tsk_treeseq_t *self, tsk_id_t index, tsk_edge_t *edge);
int tsk_treeseq_get_migration(
    const tsk_treeseq_t *self, tsk_id_t index, tsk_migration_t *migration);
int tsk_treeseq_get_site(const tsk_treeseq_t *self, tsk_id_t index, tsk_site_t *site);
int tsk_treeseq_get_mutation(
    const tsk_treeseq_t *self, tsk_id_t index, tsk_mutation_t *mutation);
int tsk_treeseq_get_provenance(
    const tsk_treeseq_t *self, tsk_id_t index, tsk_provenance_t *provenance);
int tsk_treeseq_get_population(
    const tsk_treeseq_t *self, tsk_id_t index, tsk_population_t *population);
int tsk_treeseq_get_individual(
    const tsk_treeseq_t *self, tsk_id_t index, tsk_individual_t *individual);

int tsk_treeseq_simplify(const tsk_treeseq_t *self, const tsk_id_t *samples,
    tsk_size_t num_samples, tsk_flags_t options, tsk_treeseq_t *output,
    tsk_id_t *node_map);

int tsk_treeseq_kc_distance(const tsk_treeseq_t *self, const tsk_treeseq_t *other,
    double lambda_, double *result);

int tsk_treeseq_genealogical_nearest_neighbours(const tsk_treeseq_t *self,
    const tsk_id_t *focal, tsk_size_t num_focal, const tsk_id_t *const *reference_sets,
    const tsk_size_t *reference_set_size, tsk_size_t num_reference_sets,
    tsk_flags_t options, double *ret_array);
int tsk_treeseq_mean_descendants(const tsk_treeseq_t *self,
    const tsk_id_t *const *reference_sets, const tsk_size_t *reference_set_size,
    tsk_size_t num_reference_sets, tsk_flags_t options, double *ret_array);

typedef int general_stat_func_t(tsk_size_t state_dim, const double *state,
    tsk_size_t result_dim, double *result, void *params);

int tsk_treeseq_general_stat(const tsk_treeseq_t *self, tsk_size_t K, const double *W,
    tsk_size_t M, general_stat_func_t *f, void *f_params, tsk_size_t num_windows,
    const double *windows, double *sigma, tsk_flags_t options);

/* One way weighted stats */

typedef int one_way_weighted_method(const tsk_treeseq_t *self, tsk_size_t num_weights,
    const double *weights, tsk_size_t num_windows, const double *windows, double *result,
    tsk_flags_t options);

int tsk_treeseq_trait_covariance(const tsk_treeseq_t *self, tsk_size_t num_weights,
    const double *weights, tsk_size_t num_windows, const double *windows, double *result,
    tsk_flags_t options);
int tsk_treeseq_trait_correlation(const tsk_treeseq_t *self, tsk_size_t num_weights,
    const double *weights, tsk_size_t num_windows, const double *windows, double *result,
    tsk_flags_t options);

/* One way weighted stats with covariates */

typedef int one_way_covariates_method(const tsk_treeseq_t *self, tsk_size_t num_weights,
    const double *weights, tsk_size_t num_covariates, const double *covariates,
    tsk_size_t num_windows, const double *windows, double *result, tsk_flags_t options);

int tsk_treeseq_trait_linear_model(const tsk_treeseq_t *self, tsk_size_t num_weights,
    const double *weights, tsk_size_t num_covariates, const double *covariates,
    tsk_size_t num_windows, const double *windows, double *result, tsk_flags_t options);

/* One way sample set stats */

typedef int one_way_sample_stat_method(const tsk_treeseq_t *self,
    tsk_size_t num_sample_sets, const tsk_size_t *sample_set_sizes,
    const tsk_id_t *sample_sets, tsk_size_t num_windows, const double *windows,
    double *result, tsk_flags_t options);

int tsk_treeseq_diversity(const tsk_treeseq_t *self, tsk_size_t num_sample_sets,
    const tsk_size_t *sample_set_sizes, const tsk_id_t *sample_sets,
    tsk_size_t num_windows, const double *windows, double *result, tsk_flags_t options);
int tsk_treeseq_segregating_sites(const tsk_treeseq_t *self, tsk_size_t num_sample_sets,
    const tsk_size_t *sample_set_sizes, const tsk_id_t *sample_sets,
    tsk_size_t num_windows, const double *windows, double *result, tsk_flags_t options);
int tsk_treeseq_Y1(const tsk_treeseq_t *self, tsk_size_t num_sample_sets,
    const tsk_size_t *sample_set_sizes, const tsk_id_t *sample_sets,
    tsk_size_t num_windows, const double *windows, double *result, tsk_flags_t options);
int tsk_treeseq_allele_frequency_spectrum(const tsk_treeseq_t *self,
    tsk_size_t num_sample_sets, const tsk_size_t *sample_set_sizes,
    const tsk_id_t *sample_sets, tsk_size_t num_windows, const double *windows,
    double *result, tsk_flags_t options);

typedef int general_sample_stat_method(const tsk_treeseq_t *self,
    tsk_size_t num_sample_sets, const tsk_size_t *sample_set_sizes,
    const tsk_id_t *sample_sets, tsk_size_t num_indexes, const tsk_id_t *indexes,
    tsk_size_t num_windows, const double *windows, double *result, tsk_flags_t options);

int tsk_treeseq_divergence(const tsk_treeseq_t *self, tsk_size_t num_sample_sets,
    const tsk_size_t *sample_set_sizes, const tsk_id_t *sample_sets,
    tsk_size_t num_index_tuples, const tsk_id_t *index_tuples, tsk_size_t num_windows,
    const double *windows, double *result, tsk_flags_t options);
int tsk_treeseq_Y2(const tsk_treeseq_t *self, tsk_size_t num_sample_sets,
    const tsk_size_t *sample_set_sizes, const tsk_id_t *sample_sets,
    tsk_size_t num_index_tuples, const tsk_id_t *index_tuples, tsk_size_t num_windows,
    const double *windows, double *result, tsk_flags_t options);
int tsk_treeseq_f2(const tsk_treeseq_t *self, tsk_size_t num_sample_sets,
    const tsk_size_t *sample_set_sizes, const tsk_id_t *sample_sets,
    tsk_size_t num_index_tuples, const tsk_id_t *index_tuples, tsk_size_t num_windows,
    const double *windows, double *result, tsk_flags_t options);
int tsk_treeseq_genetic_relatedness(const tsk_treeseq_t *self,
    tsk_size_t num_sample_sets, const tsk_size_t *sample_set_sizes,
    const tsk_id_t *sample_sets, tsk_size_t num_index_tuples,
    const tsk_id_t *index_tuples, tsk_size_t num_windows, const double *windows,
    double *result, tsk_flags_t options);

/* Three way sample set stats */
int tsk_treeseq_Y3(const tsk_treeseq_t *self, tsk_size_t num_sample_sets,
    const tsk_size_t *sample_set_sizes, const tsk_id_t *sample_sets,
    tsk_size_t num_index_tuples, const tsk_id_t *index_tuples, tsk_size_t num_windows,
    const double *windows, double *result, tsk_flags_t options);
int tsk_treeseq_f3(const tsk_treeseq_t *self, tsk_size_t num_sample_sets,
    const tsk_size_t *sample_set_sizes, const tsk_id_t *sample_sets,
    tsk_size_t num_index_tuples, const tsk_id_t *index_tuples, tsk_size_t num_windows,
    const double *windows, double *result, tsk_flags_t options);

/* Four way sample set stats */
int tsk_treeseq_f4(const tsk_treeseq_t *self, tsk_size_t num_sample_sets,
    const tsk_size_t *sample_set_sizes, const tsk_id_t *sample_sets,
    tsk_size_t num_index_tuples, const tsk_id_t *index_tuples, tsk_size_t num_windows,
    const double *windows, double *result, tsk_flags_t options);

/****************************************************************************/
/* Tree */
/****************************************************************************/

/**
@defgroup TREE_API_LIFECYCLE_GROUP Tree lifecycle
@{
*/

/**
@brief Initialises the tree by allocating internal memory and associating
    with the specified tree sequence.

@rst
This must be called before any operations are performed on the tree.

The specified tree sequence object must be initialised, and must be
valid for the full lifetime of this tree.

See the :ref:`sec_c_api_overview_structure` for details on how objects
are initialised and freed.

The ``options`` parameter is provided to support future expansions
of the API. A number of undocumented internal features are controlled
via this parameter, and it **must** be set to 0 to ensure that operations
work as expected and for compatibility with future versions of tskit.
@endrst

@param self A pointer to an uninitialised tsk_tree_t object.
@param tree_sequence A pointer to an initialised tsk_treeseq_t object.
@param options Allocation time options. Must be 0, or behaviour is undefined.
@return Return 0 on success or a negative value on failure.
*/
int tsk_tree_init(
    tsk_tree_t *self, const tsk_treeseq_t *tree_sequence, tsk_flags_t options);

/**
@brief Free the internal memory for the specified tree.

@param self A pointer to an initialised tsk_tree_t object.
@return Always returns 0.
*/
int tsk_tree_free(tsk_tree_t *self);

/**
@brief Copies the state of this tree into the specified destination.

@rst
By default (``options`` = 0) the method initialises the specified destination
tree by calling :c:func:`tsk_tree_init`. If the destination is already
initialised, the :c:macro:`TSK_NO_INIT` option should be supplied to avoid
leaking memory. If `TSK_NO_INIT` is supplied and the tree sequence associated
with the ``dest`` tree is not equal to the tree sequence associated
with ``self``, an error is raised.

The destination tree will keep a reference to the tree sequence object
associated with the source tree, and this tree sequence must be
valid for the full lifetime of the destination tree.

**Options**

TSK_NO_INIT
    Do **not** initialise the destination tree

If TSK_NO_INIT is not specified, options supplied to :c:func:`tsk_tree_init`
can be provided.

@endrst

@param self A pointer to an initialised tsk_tree_t object.
@param dest A pointer to a tsk_tree_t object. If the TSK_NO_INIT option
    is specified, this must be an initialised tree. If not, it must
    be an uninitialised tree.
@param options Copy and allocation time options. See the notes above for details.
@return Return 0 on success or a negative value on failure.
*/
int tsk_tree_copy(const tsk_tree_t *self, tsk_tree_t *dest, tsk_flags_t options);

/** @} */

/**
@defgroup TREE_API_SEEKING_GROUP Seeking along the sequence
@{
*/

/** @brief Value returned by seeking methods when they have successfully
    seeked to a non-null tree. */
#define TSK_TREE_OK 1

/**
@brief Seek to the first tree in the sequence.

@rst
Set the state of this tree to reflect the first tree in parent
tree sequence.
@endrst

@param self A pointer to an initialised tsk_tree_t object.
@return Return TSK_TREE_OK on success; or a negative value if an error occurs.
*/
int tsk_tree_first(tsk_tree_t *self);

/**
@brief Seek to the last tree in the sequence.

@rst
Set the state of this tree to reflect the last tree in parent
tree sequence.
@endrst

@param self A pointer to an initialised tsk_tree_t object.
@return Return TSK_TREE_OK on success; or a negative value if an error occurs.
*/
int tsk_tree_last(tsk_tree_t *self);

/**
@brief Seek to the next tree in the sequence.

@rst
Set the state of this tree to reflect the next tree in parent
tree sequence. If the index of the current tree is ``j``,
then the after this operation the index will be ``j + 1``.

Calling :c:func:`tsk_tree_next` a tree in the
:ref:`null state<sec_c_api_trees_null>` is equivalent to calling
:c:func:`tsk_tree_first`.

Calling :c:func:`tsk_tree_next` on the last tree in the
sequence will transform it into the
:ref:`null state<sec_c_api_trees_null>` (equivalent to
calling :c:func:`tsk_tree_clear`).

Please see the :ref:`sec_c_api_examples_tree_iteration` examples for
recommended usage.
@endrst

@param self A pointer to an initialised tsk_tree_t object.
@return Return TSK_TREE_OK on successfully transforming to a
non-null tree; 0 on successfully transforming into the null
tree; or a negative value if an error occurs.
*/
int tsk_tree_next(tsk_tree_t *self);

/**
@brief Seek to the previous tree in the sequence.

@rst
Set the state of this tree to reflect the previous tree in parent
tree sequence. If the index of the current tree is ``j``,
then the after this operation the index will be ``j - 1``.

Calling :c:func:`tsk_tree_prev` a tree in the
:ref:`null state<sec_c_api_trees_null>` is equivalent to calling
:c:func:`tsk_tree_last`.

Calling :c:func:`tsk_tree_prev` on the first tree in the
sequence will transform it into the
:ref:`null state<sec_c_api_trees_null>` (equivalent to
calling :c:func:`tsk_tree_clear`).

Please see the :ref:`sec_c_api_examples_tree_iteration` examples for
recommended usage.
@endrst

@param self A pointer to an initialised tsk_tree_t object.
@return Return TSK_TREE_OK on successfully transforming to a
non-null tree; 0 on successfully transforming into the null
tree; or a negative value if an error occurs.
*/
int tsk_tree_prev(tsk_tree_t *self);

/**
@brief Set the tree into the null state.

@rst
Transform this tree into the :ref:`null state<sec_c_api_trees_null>`.
@endrst

@param self A pointer to an initialised tsk_tree_t object.
@return Return 0 on success or a negative value on failure.
*/
int tsk_tree_clear(tsk_tree_t *self);

/**
@brief Seek to a particular position on the genome.

@rst
Set the state of this tree to reflect the tree in parent
tree sequence covering the specified ``position``. That is, on success
we will have ``tree.interval.left <= position`` and
we will have ``position < tree.interval.right``.

Seeking to a position currently covered by the tree is
a constant time operation.

.. warning::
   The current implementation of ``seek`` does **not** provide efficient
   random access to arbitrary positions along the genome. However,
   sequentially seeking in either direction is as efficient as calling
   :c:func:`tsk_tree_next` or :c:func:`tsk_tree_prev` directly.
@endrst

@param self A pointer to an initialised tsk_tree_t object.
@param position The position in genome coordinates
@param options Seek options. Currently unused. Set to 0 for compatibility
    with future versions of tskit.
@return Return 0 on success or a negative value on failure.
*/
int tsk_tree_seek(tsk_tree_t *self, double position, tsk_flags_t options);

/** @} */

/**
@defgroup TREE_API_TREE_QUERY_GROUP Tree Queries
@{
*/

/**
@brief Returns the number of roots in this tree.

@rst
See the :ref:`sec_data_model_tree_roots` section for more information
on how the roots of a tree are defined.
@endrst

@param self A pointer to an initialised tsk_tree_t object.
@return Returns the number roots in this tree.
*/
tsk_size_t tsk_tree_get_num_roots(const tsk_tree_t *self);

/**
@brief Returns the leftmost root in this tree.

@rst
See the :ref:`sec_data_model_tree_roots` section for more information
on how the roots of a tree are defined.

This function is equivalent to ``tree.left_child[tree.virtual_root]``.
@endrst

@param self A pointer to an initialised tsk_tree_t object.
@return Returns the leftmost root in the tree.
*/
tsk_id_t tsk_tree_get_left_root(const tsk_tree_t *self);

/**
@brief Returns the rightmost root in this tree.

@rst
See the :ref:`sec_data_model_tree_roots` section for more information
on how the roots of a tree are defined.

This function is equivalent to ``tree.right_child[tree.virtual_root]``.
@endrst

@param self A pointer to an initialised tsk_tree_t object.
@return Returns the rightmost root in the tree.
*/
tsk_id_t tsk_tree_get_right_root(const tsk_tree_t *self);

int tsk_tree_get_sites(
    const tsk_tree_t *self, const tsk_site_t **sites, tsk_size_t *sites_length);

bool tsk_tree_equals(const tsk_tree_t *self, const tsk_tree_t *other);

/**
@brief Return an upper bound on the number of nodes reachable
    from the roots of this tree.

@rst
This function provides an upper bound on the number of nodes that
can be reached in tree traversals, and is intended to be used
for memory allocation purposes. If ``num_nodes`` is the number
of nodes visited in a tree traversal from the virtual root
(e.g., ``tsk_tree_preorder(tree, tree->virtual_root, nodes,
&num_nodes)``), the bound ``N`` returned here is guaranteed to
be greater than or equal to ``num_nodes``.

.. warning:: The precise value returned is not defined and should
    not be depended on, as it may change from version-to-version.

@endrst

@param self A pointer to a tsk_tree_t object.
@return An upper bound on the number nodes reachable from the roots
    of this tree, or zero if this tree has not been initialised.
*/
tsk_size_t tsk_tree_get_size_bound(const tsk_tree_t *self);

/**
@brief Print out the state of this tree to the specified stream.

This method is intended for debugging purposes and should not be used
in production code. The format of the output should **not** be depended
on and may change arbitrarily between versions.

@param self A pointer to a tsk_tree_t object.
@param out The stream to write the summary to.
*/
void tsk_tree_print_state(const tsk_tree_t *self, FILE *out);

/** @} */

/**
@defgroup TREE_API_NODE_QUERY_GROUP Node Queries
@{
*/

/* Returns true if u is a descendant of v; false otherwise */
bool tsk_tree_is_descendant(const tsk_tree_t *self, tsk_id_t u, tsk_id_t v);
bool tsk_tree_is_sample(const tsk_tree_t *self, tsk_id_t u);

int tsk_tree_get_branch_length(
    const tsk_tree_t *self, tsk_id_t u, double *branch_length);

/**
@brief Returns the sum of the lengths of all branches reachable from
    the specified node, or from all roots if u=TSK_NULL.

@rst
Return the total branch length in a particular subtree or of the
entire tree. If the specified node is TSK_NULL (or the virtual
root) the sum of the lengths of all branches reachable from roots
is returned. Branch length is defined as difference between the time
of a node and its parent. The branch length of a root is zero.

Note that if the specified node is internal its branch length is
*not* included, so that, e.g., the total branch length of a
leaf node is zero.
@endrst

@param self A pointer to a tsk_tree_t object.
@param u The tree node to compute branch length or TSK_NULL to return the
    total branch length of the tree.
@param ret_tbl A double pointer to store the returned total branch length.
@return 0 on success or a negative value on failure.
*/
int tsk_tree_get_total_branch_length(
    const tsk_tree_t *self, tsk_id_t u, double *ret_tbl);

int tsk_tree_get_time(const tsk_tree_t *self, tsk_id_t u, double *t);
int tsk_tree_get_parent(const tsk_tree_t *self, tsk_id_t u, tsk_id_t *parent);
int tsk_tree_get_depth(const tsk_tree_t *self, tsk_id_t u, int *depth);
int tsk_tree_get_mrca(const tsk_tree_t *self, tsk_id_t u, tsk_id_t v, tsk_id_t *mrca);
int tsk_tree_get_num_samples(
    const tsk_tree_t *self, tsk_id_t u, tsk_size_t *num_samples);

/** @} */

/**
@defgroup TREE_API_TRAVERSAL_GROUP Traversal orders.
@{
*/

int tsk_tree_preorder(
    const tsk_tree_t *self, tsk_id_t root, tsk_id_t *nodes, tsk_size_t *num_nodes);
int tsk_tree_postorder(
    const tsk_tree_t *self, tsk_id_t root, tsk_id_t *nodes, tsk_size_t *num_nodes);
int tsk_tree_preorder_samples(
    const tsk_tree_t *self, tsk_id_t root, tsk_id_t *nodes, tsk_size_t *num_nodes);

/** @} */

/* Undocumented for now */

int tsk_tree_set_root_threshold(tsk_tree_t *self, tsk_size_t root_threshold);
tsk_size_t tsk_tree_get_root_threshold(const tsk_tree_t *self);

bool tsk_tree_has_sample_counts(const tsk_tree_t *self);
bool tsk_tree_has_sample_lists(const tsk_tree_t *self);

int tsk_tree_get_num_tracked_samples(
    const tsk_tree_t *self, tsk_id_t u, tsk_size_t *num_tracked_samples);
int tsk_tree_set_tracked_samples(
    tsk_tree_t *self, tsk_size_t num_tracked_samples, const tsk_id_t *tracked_samples);
int tsk_tree_track_descendant_samples(tsk_tree_t *self, tsk_id_t node);

typedef struct {
    tsk_id_t node;
    tsk_id_t parent;
    int32_t state;
} tsk_state_transition_t;

int tsk_tree_map_mutations(tsk_tree_t *self, int32_t *genotypes, double *cost_matrix,
    tsk_flags_t options, int32_t *ancestral_state, tsk_size_t *num_transitions,
    tsk_state_transition_t **transitions);

int tsk_tree_kc_distance(
    const tsk_tree_t *self, const tsk_tree_t *other, double lambda, double *result);

/****************************************************************************/
/* Diff iterator */
/****************************************************************************/

int tsk_diff_iter_init(
    tsk_diff_iter_t *self, const tsk_treeseq_t *tree_sequence, tsk_flags_t options);
int tsk_diff_iter_free(tsk_diff_iter_t *self);
int tsk_diff_iter_next(tsk_diff_iter_t *self, double *left, double *right,
    tsk_edge_list_t *edges_out, tsk_edge_list_t *edges_in);
void tsk_diff_iter_print_state(const tsk_diff_iter_t *self, FILE *out);

#ifdef __cplusplus
}
#endif
#endif
