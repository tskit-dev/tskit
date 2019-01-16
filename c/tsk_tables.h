/**
 * @file tsk_tables.h
 * @brief Tskit Tables API.
 */
#ifndef TSK_TABLES_H
#define TSK_TABLES_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>

#include <kastore.h>

#include "tsk_core.h"

/**
@brief Tskit Object IDs.

@rst
All objects in tskit are referred to by integer IDs corresponding to the 
row they occupy in the relevant table. The ``tsk_id_t`` type should be used
when manipulating these ID values. The reserved value ``TSK_NULL`` (-1) defines
missing data. 
@endrst
*/
typedef int32_t tsk_id_t;

/**
@brief Tskit sizes.

@rst
Sizes in tskit are defined by the ``tsk_size_t`` type. 
@endrst
*/
typedef uint32_t tsk_size_t;

/**
@brief Container for bitwise flags.

@rst
Bitwise flags are used in tskit as a column type and also as a way to 
specify options to API functions. 
@endrst
*/
typedef uint32_t tsk_flags_t;

/****************************************************************************/
/* Definitions for the basic objects */
/****************************************************************************/

/**
@brief A single individual.

@rst
See the :ref:`data model <sec_data_model_definitions>` section for the definition of
an individual and its properties.
@endrst
*/
typedef struct {
    /** @brief The non-negative ID value corresponding to table row. */
    tsk_id_t id;
    /** @brief The bitwise flags. */
    tsk_flags_t flags;
    /** @brief The spatial location. The number of dimensions is defined by
     * ``location_length``. */
    double *location;
    /** @brief The number of spatial dimensions. */
    tsk_size_t location_length;
    /** @brief The metadata. */
    const char *metadata;
    /** @brief The size of the metadata in bytes. */
    tsk_size_t metadata_length;
    tsk_id_t *nodes;
    tsk_size_t nodes_length;
} tsk_individual_t;

/**
@brief A single node.

@rst
See the :ref:`data model <sec_data_model_definitions>` section for the definition of
a node and its properties.
@endrst
*/
typedef struct {
    tsk_id_t id;
    tsk_flags_t flags;
    double time;
    tsk_id_t population;
    tsk_id_t individual;
    const char *metadata;
    tsk_size_t metadata_length;
} tsk_node_t;

/**
@brief A single edge.

@rst
See the :ref:`data model <sec_data_model_definitions>` section for the definition of
an edge and its properties.
@endrst
*/
typedef struct {
    tsk_id_t id;
    tsk_id_t parent;
    tsk_id_t child;
    double left;
    double right;
} tsk_edge_t;

/**
@brief A single mutation.

@rst
See the :ref:`data model <sec_data_model_definitions>` section for the definition of
a mutation and its properties.
@endrst
*/
typedef struct {
    tsk_id_t id;
    tsk_id_t site;
    tsk_id_t node;
    tsk_id_t parent;
    const char *derived_state;
    tsk_size_t derived_state_length;
    const char *metadata;
    tsk_size_t metadata_length;
} tsk_mutation_t;

/**
@brief A single site.

@rst
See the :ref:`data model <sec_data_model_definitions>` section for the definition of
a site and its properties.
@endrst
*/
typedef struct {
    tsk_id_t id;
    double position;
    const char *ancestral_state;
    tsk_size_t ancestral_state_length;
    const char *metadata;
    tsk_size_t metadata_length;
    tsk_mutation_t *mutations;
    tsk_size_t mutations_length;
} tsk_site_t;

/**
@brief A single migration.

@rst
See the :ref:`data model <sec_data_model_definitions>` section for the definition of
a migration and its properties.
@endrst
*/
typedef struct {
    tsk_id_t id;
    tsk_id_t source;
    tsk_id_t dest;
    tsk_id_t node;
    double left;
    double right;
    double time;
} tsk_migration_t;

/**
@brief A single population.

@rst
See the :ref:`data model <sec_data_model_definitions>` section for the definition of
a population and its properties.
@endrst
*/
typedef struct {
    tsk_id_t id;
    const char *metadata;
    tsk_size_t metadata_length;
} tsk_population_t;

/**
@brief A single provenance object.

@rst
See the :ref:`data model <sec_data_model_definitions>` section for the definition of
a provenance object and its properties. See the :ref:`sec_provenance` section
for more information on how provenance records should be structured.
@endrst
*/
typedef struct {
    tsk_id_t id;
    const char *timestamp;
    tsk_size_t timestamp_length;
    const char *record;
    tsk_size_t record_length;
} tsk_provenance_t;

/****************************************************************************/
/* Table definitions */
/****************************************************************************/

/**
@brief The individual table.

@rst
See the :ref:`data model <sec_individual_table_definition>` section for details
of the columns in this table.
@endrst
*/
typedef struct {
    tsk_size_t num_rows;
    tsk_size_t max_rows;
    tsk_size_t max_rows_increment;
    tsk_size_t location_length;
    tsk_size_t max_location_length;
    tsk_size_t max_location_length_increment;
    tsk_size_t metadata_length;
    tsk_size_t max_metadata_length;
    tsk_size_t max_metadata_length_increment;
    tsk_flags_t *flags;
    double *location;
    tsk_size_t *location_offset;
    char *metadata;
    tsk_size_t *metadata_offset;
} tsk_individual_table_t;

/**
@brief The node table.
*/
typedef struct {
    tsk_size_t num_rows;
    tsk_size_t max_rows;
    tsk_size_t max_rows_increment;
    tsk_size_t metadata_length;
    tsk_size_t max_metadata_length;
    tsk_size_t max_metadata_length_increment;
    tsk_flags_t *flags;
    double *time;
    tsk_id_t *population;
    tsk_id_t *individual;
    char *metadata;
    tsk_size_t *metadata_offset;
} tsk_node_table_t;

/**
@brief The site table.
*/
typedef struct {
    tsk_size_t num_rows;
    tsk_size_t max_rows;
    tsk_size_t max_rows_increment;
    tsk_size_t ancestral_state_length;
    tsk_size_t max_ancestral_state_length;
    tsk_size_t max_ancestral_state_length_increment;
    tsk_size_t metadata_length;
    tsk_size_t max_metadata_length;
    tsk_size_t max_metadata_length_increment;
    double *position;
    char *ancestral_state;
    tsk_size_t *ancestral_state_offset;
    char *metadata;
    tsk_size_t *metadata_offset;
} tsk_site_table_t;

/**
@brief The mutation table.
*/
typedef struct {
    tsk_size_t num_rows;
    tsk_size_t max_rows;
    tsk_size_t max_rows_increment;
    tsk_size_t derived_state_length;
    tsk_size_t max_derived_state_length;
    tsk_size_t max_derived_state_length_increment;
    tsk_size_t metadata_length;
    tsk_size_t max_metadata_length;
    tsk_size_t max_metadata_length_increment;
    tsk_id_t *node;
    tsk_id_t *site;
    tsk_id_t *parent;
    char *derived_state;
    tsk_size_t *derived_state_offset;
    char *metadata;
    tsk_size_t *metadata_offset;
} tsk_mutation_table_t;

/**
@brief The edge table.
*/
typedef struct {
    tsk_size_t num_rows;
    tsk_size_t max_rows;
    tsk_size_t max_rows_increment;
    double *left;
    double *right;
    tsk_id_t *parent;
    tsk_id_t *child;
} tsk_edge_table_t;

/**
@brief The migration table.
*/
typedef struct {
    tsk_size_t num_rows;
    tsk_size_t max_rows;
    tsk_size_t max_rows_increment;
    tsk_id_t *source;
    tsk_id_t *dest;
    tsk_id_t *node;
    double *left;
    double *right;
    double *time;
} tsk_migration_table_t;

/**
@brief The population table.
*/
typedef struct {
    tsk_size_t num_rows;
    tsk_size_t max_rows;
    tsk_size_t max_rows_increment;
    tsk_size_t metadata_length;
    tsk_size_t max_metadata_length;
    tsk_size_t max_metadata_length_increment;
    char *metadata;
    tsk_size_t *metadata_offset;
} tsk_population_table_t;

/**
@brief The provenance table.
*/
typedef struct {
    tsk_size_t num_rows;
    tsk_size_t max_rows;
    tsk_size_t max_rows_increment;
    tsk_size_t timestamp_length;
    tsk_size_t max_timestamp_length;
    tsk_size_t max_timestamp_length_increment;
    tsk_size_t record_length;
    tsk_size_t max_record_length;
    tsk_size_t max_record_length_increment;
    char *timestamp;
    tsk_size_t *timestamp_offset;
    char *record;
    tsk_size_t *record_offset;
} tsk_provenance_table_t;

/**
@brief A collection of tables defining the data for a tree sequence.
*/
typedef struct {
    /** @brief The sequence length defining the tree sequence's coordinate space */
    double sequence_length;
    char *file_uuid;
    /** @brief The individual table */
    tsk_individual_table_t *individuals;
    /** @brief The node table */
    tsk_node_table_t *nodes;
    /** @brief The edge table */
    tsk_edge_table_t *edges;
    /** @brief The migration table */
    tsk_migration_table_t *migrations;
    /** @brief The site table */
    tsk_site_table_t *sites;
    /** @brief The mutation table */
    tsk_mutation_table_t *mutations;
    /** @brief The population table */
    tsk_population_table_t *populations;
    /** @brief The provenance table */
    tsk_provenance_table_t *provenances;
    struct {
        tsk_id_t *edge_insertion_order;
        tsk_id_t *edge_removal_order;
        bool malloced_locally;
    } indexes;
    kastore_t *store;
    /* TODO Add in reserved space for future tables. */
} tsk_table_collection_t;

typedef struct {
    tsk_size_t individuals;
    tsk_size_t nodes;
    tsk_size_t edges;
    tsk_size_t migrations;
    tsk_size_t sites;
    tsk_size_t mutations;
    tsk_size_t populations;
    tsk_size_t provenances;
    /* TODO add reserved space for future tables. */
} tsk_table_collection_position_t;


/****************************************************************************/
/* Function signatures */
/****************************************************************************/

/**
@defgroup INDIVIDUAL_TABLE_API_GROUP Individual table API.
@{
*/

/**
@brief Allocate a new individual table.

@rst
The table is allocated with the default size increment.
@endrst

@param self A pointer to an uninitialised tsk_individual_table_t object.
@param options Allocation time options. Currently unused; should be 
    set to zero to ensure compatability with later versions of tskit.
@return Return 0 on success or a negative value on failure.
*/
int tsk_individual_table_alloc(tsk_individual_table_t *self, tsk_flags_t options);


int tsk_individual_table_set_max_rows_increment(tsk_individual_table_t *self, 
        tsk_size_t max_rows_increment);
int tsk_individual_table_set_max_metadata_length_increment(tsk_individual_table_t *self,
        tsk_size_t max_metadata_length_increment);
int tsk_individual_table_set_max_location_length_increment(tsk_individual_table_t *self,
        tsk_size_t max_location_length_increment);

/**
@brief Adds a row to this individual table.

@rst
Add a new individual with the specified ``flags``, ``location`` and ``metadata``
to the table. Copies of the ``location`` and ``metadata`` parameters are taken
immediately.
See the :ref:`data model <sec_individual_table_definition>` section for details
of the columns in this table.
@endrst

@param self A pointer to a tsk_individual_table_t object.
@param flags The bitwise flags for the new individual.
@param location A pointer to a double array representing the spatial location
    of the new individual. Can be ``NULL`` if ``location_length`` is 0.
@param location_length The number of dimensions in the locations position.
    Note this the number of elements in the corresponding double array
    not the number of bytes.
@param metadata The metadata to be associated with the new individual. This
    is a pointer to arbitrary memory. Can be ``NULL`` if ``metadata_length`` is 0.
@param metadata_length The size of the metadata array in bytes.
@return Return the ID of the newly added individual on success,
    or a negative value on failure.
*/
tsk_id_t tsk_individual_table_add_row(tsk_individual_table_t *self, tsk_flags_t flags,
        double *location, tsk_size_t location_length,
        const char *metadata, tsk_size_t metadata_length);
int tsk_individual_table_set_columns(tsk_individual_table_t *self, tsk_size_t num_rows, tsk_flags_t *flags,
        double *location, tsk_size_t *location_length,
        const char *metadata, tsk_size_t *metadata_length);
int tsk_individual_table_append_columns(tsk_individual_table_t *self, tsk_size_t num_rows, tsk_flags_t *flags,
        double *location, tsk_size_t *location_length,
        const char *metadata, tsk_size_t *metadata_length);
int tsk_individual_table_clear(tsk_individual_table_t *self);
int tsk_individual_table_truncate(tsk_individual_table_t *self, tsk_size_t num_rows);
int tsk_individual_table_free(tsk_individual_table_t *self);
int tsk_individual_table_dump_text(tsk_individual_table_t *self, FILE *out);
int tsk_individual_table_copy(tsk_individual_table_t *self, tsk_individual_table_t *dest);
void tsk_individual_table_print_state(tsk_individual_table_t *self, FILE *out);
bool tsk_individual_table_equals(tsk_individual_table_t *self, tsk_individual_table_t *other);
int tsk_individual_table_get_row(tsk_individual_table_t *self, tsk_id_t index,
        tsk_individual_t *row);

/** @} */

/**
@defgroup NODE_TABLE_API_GROUP Node table API.
@{
*/
int tsk_node_table_alloc(tsk_node_table_t *self, tsk_flags_t options);
int tsk_node_table_set_max_rows_increment(tsk_node_table_t *self, tsk_size_t max_rows_increment);
int tsk_node_table_set_max_metadata_length_increment(tsk_node_table_t *self,
        tsk_size_t max_metadata_length_increment);
tsk_id_t tsk_node_table_add_row(tsk_node_table_t *self, tsk_flags_t flags, double time,
        tsk_id_t population, tsk_id_t individual,
        const char *metadata, tsk_size_t metadata_length);
int tsk_node_table_set_columns(tsk_node_table_t *self, tsk_size_t num_rows,
        tsk_flags_t *flags, double *time,
        tsk_id_t *population, tsk_id_t *individual,
        const char *metadata, tsk_size_t *metadata_length);
int tsk_node_table_append_columns(tsk_node_table_t *self, tsk_size_t num_rows,
        tsk_flags_t *flags, double *time,
        tsk_id_t *population, tsk_id_t *individual,
        const char *metadata, tsk_size_t *metadata_length);
int tsk_node_table_clear(tsk_node_table_t *self);
int tsk_node_table_truncate(tsk_node_table_t *self, tsk_size_t num_rows);
int tsk_node_table_free(tsk_node_table_t *self);
int tsk_node_table_dump_text(tsk_node_table_t *self, FILE *out);
int tsk_node_table_copy(tsk_node_table_t *self, tsk_node_table_t *dest);
void tsk_node_table_print_state(tsk_node_table_t *self, FILE *out);
bool tsk_node_table_equals(tsk_node_table_t *self, tsk_node_table_t *other);
int tsk_node_table_get_row(tsk_node_table_t *self, tsk_id_t index, tsk_node_t *row);

/** @} */

/**
@defgroup EDGE_TABLE_API_GROUP Edge table API.
@{
*/
int tsk_edge_table_alloc(tsk_edge_table_t *self, tsk_flags_t options);
int tsk_edge_table_set_max_rows_increment(tsk_edge_table_t *self, tsk_size_t max_rows_increment);
tsk_id_t tsk_edge_table_add_row(tsk_edge_table_t *self, double left, double right, tsk_id_t parent,
        tsk_id_t child);
int tsk_edge_table_set_columns(tsk_edge_table_t *self, tsk_size_t num_rows, double *left,
        double *right, tsk_id_t *parent, tsk_id_t *child);
int tsk_edge_table_append_columns(tsk_edge_table_t *self, tsk_size_t num_rows, double *left,
        double *right, tsk_id_t *parent, tsk_id_t *child);
int tsk_edge_table_clear(tsk_edge_table_t *self);
int tsk_edge_table_truncate(tsk_edge_table_t *self, tsk_size_t num_rows);
int tsk_edge_table_free(tsk_edge_table_t *self);
int tsk_edge_table_dump_text(tsk_edge_table_t *self, FILE *out);
int tsk_edge_table_copy(tsk_edge_table_t *self, tsk_edge_table_t *dest);
void tsk_edge_table_print_state(tsk_edge_table_t *self, FILE *out);
bool tsk_edge_table_equals(tsk_edge_table_t *self, tsk_edge_table_t *other);
int tsk_edge_table_get_row(tsk_edge_table_t *self, tsk_id_t index, tsk_edge_t *row);

/** @} */

/**
@defgroup SITE_TABLE_API_GROUP Site table API.
@{
*/
int tsk_site_table_alloc(tsk_site_table_t *self, tsk_flags_t options);
int tsk_site_table_set_max_rows_increment(tsk_site_table_t *self, tsk_size_t max_rows_increment);
int tsk_site_table_set_max_metadata_length_increment(tsk_site_table_t *self,
        tsk_size_t max_metadata_length_increment);
int tsk_site_table_set_max_ancestral_state_length_increment(tsk_site_table_t *self,
        tsk_size_t max_ancestral_state_length_increment);
tsk_id_t tsk_site_table_add_row(tsk_site_table_t *self, double position,
        const char *ancestral_state, tsk_size_t ancestral_state_length,
        const char *metadata, tsk_size_t metadata_length);
int tsk_site_table_set_columns(tsk_site_table_t *self, tsk_size_t num_rows, double *position,
        const char *ancestral_state, tsk_size_t *ancestral_state_length,
        const char *metadata, tsk_size_t *metadata_length);
int tsk_site_table_append_columns(tsk_site_table_t *self, tsk_size_t num_rows, double *position,
        const char *ancestral_state, tsk_size_t *ancestral_state_length,
        const char *metadata, tsk_size_t *metadata_length);
bool tsk_site_table_equals(tsk_site_table_t *self, tsk_site_table_t *other);
int tsk_site_table_clear(tsk_site_table_t *self);
int tsk_site_table_truncate(tsk_site_table_t *self, tsk_size_t num_rows);
int tsk_site_table_copy(tsk_site_table_t *self, tsk_site_table_t *dest);
int tsk_site_table_free(tsk_site_table_t *self);
int tsk_site_table_dump_text(tsk_site_table_t *self, FILE *out);
void tsk_site_table_print_state(tsk_site_table_t *self, FILE *out);
int tsk_site_table_get_row(tsk_site_table_t *self, tsk_id_t index, tsk_site_t *row);

/** @} */

/**
@defgroup MUTATION_TABLE_API_GROUP Mutation table API.
@{
*/
void tsk_mutation_table_print_state(tsk_mutation_table_t *self, FILE *out);
int tsk_mutation_table_alloc(tsk_mutation_table_t *self, tsk_flags_t options);
int tsk_mutation_table_set_max_rows_increment(tsk_mutation_table_t *self, tsk_size_t max_rows_increment);
int tsk_mutation_table_set_max_metadata_length_increment(tsk_mutation_table_t *self,
        tsk_size_t max_metadata_length_increment);
int tsk_mutation_table_set_max_derived_state_length_increment(tsk_mutation_table_t *self,
        tsk_size_t max_derived_state_length_increment);
tsk_id_t tsk_mutation_table_add_row(tsk_mutation_table_t *self, tsk_id_t site,
        tsk_id_t node, tsk_id_t parent,
        const char *derived_state, tsk_size_t derived_state_length,
        const char *metadata, tsk_size_t metadata_length);
int tsk_mutation_table_set_columns(tsk_mutation_table_t *self, tsk_size_t num_rows,
        tsk_id_t *site, tsk_id_t *node, tsk_id_t *parent,
        const char *derived_state, tsk_size_t *derived_state_length,
        const char *metadata, tsk_size_t *metadata_length);
int tsk_mutation_table_append_columns(tsk_mutation_table_t *self, tsk_size_t num_rows,
        tsk_id_t *site, tsk_id_t *node, tsk_id_t *parent,
        const char *derived_state, tsk_size_t *derived_state_length,
        const char *metadata, tsk_size_t *metadata_length);
bool tsk_mutation_table_equals(tsk_mutation_table_t *self, tsk_mutation_table_t *other);
int tsk_mutation_table_clear(tsk_mutation_table_t *self);
int tsk_mutation_table_truncate(tsk_mutation_table_t *self, tsk_size_t num_rows);
int tsk_mutation_table_copy(tsk_mutation_table_t *self, tsk_mutation_table_t *dest);
int tsk_mutation_table_free(tsk_mutation_table_t *self);
int tsk_mutation_table_dump_text(tsk_mutation_table_t *self, FILE *out);
void tsk_mutation_table_print_state(tsk_mutation_table_t *self, FILE *out);
int tsk_mutation_table_get_row(tsk_mutation_table_t *self, tsk_id_t index, tsk_mutation_t *row);

/** @} */

/**
@defgroup MIGRATION_TABLE_API_GROUP Migration table API.
@{
*/
int tsk_migration_table_alloc(tsk_migration_table_t *self, tsk_flags_t options);
int tsk_migration_table_set_max_rows_increment(tsk_migration_table_t *self, tsk_size_t max_rows_increment);
tsk_id_t tsk_migration_table_add_row(tsk_migration_table_t *self, double left,
        double right, tsk_id_t node, tsk_id_t source,
        tsk_id_t dest, double time);
int tsk_migration_table_set_columns(tsk_migration_table_t *self, tsk_size_t num_rows,
        double *left, double *right, tsk_id_t *node, tsk_id_t *source,
        tsk_id_t *dest, double *time);
int tsk_migration_table_append_columns(tsk_migration_table_t *self, tsk_size_t num_rows,
        double *left, double *right, tsk_id_t *node, tsk_id_t *source,
        tsk_id_t *dest, double *time);
int tsk_migration_table_clear(tsk_migration_table_t *self);
int tsk_migration_table_truncate(tsk_migration_table_t *self, tsk_size_t num_rows);
int tsk_migration_table_free(tsk_migration_table_t *self);
int tsk_migration_table_copy(tsk_migration_table_t *self, tsk_migration_table_t *dest);
int tsk_migration_table_dump_text(tsk_migration_table_t *self, FILE *out);
void tsk_migration_table_print_state(tsk_migration_table_t *self, FILE *out);
bool tsk_migration_table_equals(tsk_migration_table_t *self, tsk_migration_table_t *other);
int tsk_migration_table_get_row(tsk_migration_table_t *self, tsk_id_t index, tsk_migration_t *row);

/** @} */

/**
@defgroup POPULATION_TABLE_API_GROUP Population table API.
@{
*/
int tsk_population_table_alloc(tsk_population_table_t *self, tsk_flags_t options);
int tsk_population_table_set_max_rows_increment(tsk_population_table_t *self, tsk_size_t max_rows_increment);
int tsk_population_table_set_max_metadata_length_increment(tsk_population_table_t *self,
        tsk_size_t max_metadata_length_increment);
tsk_id_t tsk_population_table_add_row(tsk_population_table_t *self,
        const char *metadata, tsk_size_t metadata_length);
int tsk_population_table_set_columns(tsk_population_table_t *self, tsk_size_t num_rows,
        const char *metadata, tsk_size_t *metadata_offset);
int tsk_population_table_append_columns(tsk_population_table_t *self, tsk_size_t num_rows,
        const char *metadata, tsk_size_t *metadata_offset);
int tsk_population_table_clear(tsk_population_table_t *self);
int tsk_population_table_truncate(tsk_population_table_t *self, tsk_size_t num_rows);
int tsk_population_table_copy(tsk_population_table_t *self, tsk_population_table_t *dest);
int tsk_population_table_free(tsk_population_table_t *self);
void tsk_population_table_print_state(tsk_population_table_t *self, FILE *out);
int tsk_population_table_dump_text(tsk_population_table_t *self, FILE *out);
bool tsk_population_table_equals(tsk_population_table_t *self, tsk_population_table_t *other);
int tsk_population_table_get_row(tsk_population_table_t *self, tsk_id_t index, tsk_population_t *row);

/** @} */

/**
@defgroup PROVENANCE_TABLE_API_GROUP Provenance table API.
@{
*/
int tsk_provenance_table_alloc(tsk_provenance_table_t *self, tsk_flags_t options);
int tsk_provenance_table_set_max_rows_increment(tsk_provenance_table_t *self, tsk_size_t max_rows_increment);
int tsk_provenance_table_set_max_timestamp_length_increment(tsk_provenance_table_t *self,
        tsk_size_t max_timestamp_length_increment);
int tsk_provenance_table_set_max_record_length_increment(tsk_provenance_table_t *self,
        tsk_size_t max_record_length_increment);
tsk_id_t tsk_provenance_table_add_row(tsk_provenance_table_t *self,
        const char *timestamp, tsk_size_t timestamp_length,
        const char *record, tsk_size_t record_length);
int tsk_provenance_table_set_columns(tsk_provenance_table_t *self, tsk_size_t num_rows,
       char *timestamp, tsk_size_t *timestamp_offset,
       char *record, tsk_size_t *record_offset);
int tsk_provenance_table_append_columns(tsk_provenance_table_t *self, tsk_size_t num_rows,
        char *timestamp, tsk_size_t *timestamp_offset,
        char *record, tsk_size_t *record_offset);
int tsk_provenance_table_clear(tsk_provenance_table_t *self);
int tsk_provenance_table_truncate(tsk_provenance_table_t *self, tsk_size_t num_rows);
int tsk_provenance_table_copy(tsk_provenance_table_t *self, tsk_provenance_table_t *dest);
int tsk_provenance_table_free(tsk_provenance_table_t *self);
int tsk_provenance_table_dump_text(tsk_provenance_table_t *self, FILE *out);
void tsk_provenance_table_print_state(tsk_provenance_table_t *self, FILE *out);
bool tsk_provenance_table_equals(tsk_provenance_table_t *self, tsk_provenance_table_t *other);
int tsk_provenance_table_get_row(tsk_provenance_table_t *self, tsk_id_t index, tsk_provenance_t *row);

/** @} */

/****************************************************************************/
/* Table collection .*/
/****************************************************************************/

/**
@defgroup TABLE_COLLECTION_API_GROUP Table collection API.
@{
*/

/**
@brief Allocate a new table collection.

@rst
After allocation, each of the consituent tables is allocated with the default size increment.
The sequence length is set to zero.

Once this function in called for a particular instance (irrespective of whether an 
error errors) :c:func:`tsk_table_collection_free` **must** be called on the instance to 
ensure that memory is not leaked. 
@endrst

@param self A pointer to an uninitialised tsk_table_collection_t object.
@param options Allocation time options. Currently unused; should be 
    set to zero to ensure compatability with later versions of tskit.
@return Return 0 on success or a negative value on failure.
*/
int tsk_table_collection_alloc(tsk_table_collection_t *self, tsk_flags_t options);

/**
@brief Load a table collection from file.

@rst
Allocates a new table collection and loads the data from the specified file.
@endrst

@param self A pointer to an uninitialised tsk_table_collection_t object.
@param filename A NULL terminated string containing the filename.
@param options Allocation time options. Currently unused; should be 
    set to zero to ensure compatability with later versions of tskit.
@return Return 0 on success or a negative value on failure.
*/
int tsk_table_collection_load(tsk_table_collection_t *self, const char *filename, 
    tsk_flags_t options);

/**
@brief Free the memory used by a tsk_table_collection_t object.

@param self A pointer to an initialised tsk_table_collection_t object.
@return Always returns 0.
*/
int tsk_table_collection_free(tsk_table_collection_t *self);

int tsk_table_collection_dump(tsk_table_collection_t *tables, const char *filename, 
    tsk_flags_t options);
int tsk_table_collection_copy(tsk_table_collection_t *self, tsk_table_collection_t *dest);
int tsk_table_collection_print_state(tsk_table_collection_t *self, FILE *out);

bool tsk_table_collection_is_indexed(tsk_table_collection_t *self);
int tsk_table_collection_drop_indexes(tsk_table_collection_t *self);
int tsk_table_collection_build_indexes(tsk_table_collection_t *self, tsk_flags_t options);
int tsk_table_collection_simplify(tsk_table_collection_t *self,
    tsk_id_t *samples, tsk_size_t num_samples, tsk_flags_t options, tsk_id_t *node_map);
int tsk_table_collection_sort(tsk_table_collection_t *self, tsk_size_t edge_start, 
    tsk_flags_t options);
int tsk_table_collection_deduplicate_sites(tsk_table_collection_t *tables, tsk_flags_t options);
int tsk_table_collection_compute_mutation_parents(tsk_table_collection_t *self, tsk_flags_t options);
bool tsk_table_collection_equals(tsk_table_collection_t *self, tsk_table_collection_t *other);
int tsk_table_collection_record_position(tsk_table_collection_t *self,
        tsk_table_collection_position_t *position);
int tsk_table_collection_reset_position(tsk_table_collection_t *self,
        tsk_table_collection_position_t *position);
int tsk_table_collection_clear(tsk_table_collection_t *self);
int tsk_table_collection_check_integrity(tsk_table_collection_t *self, tsk_flags_t options);

/** @} */

int tsk_squash_edges(tsk_edge_t *edges, size_t num_edges, size_t *num_output_edges);

#ifdef __cplusplus
}
#endif
#endif
