---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.9.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{currentmodule} tskit
```

(sec_metadata)=

# Metadata

The tree-sequence and all the entities within it (nodes, mutations, edges,  etc.) can
have metadata associated with them. This is intended for storing and passing on
information that tskit itself does not use or interpret, for example information derived
from a VCF INFO field, or administrative information (such as unique identifiers)
relating to samples and populations. Note that provenance information about how a tree
sequence was created should not be stored in metadata, instead the provenance mechanisms
in tskit should be used (see {ref}`sec_provenance`).

The metadata for each entity (e.g. row in a table) is described by a schema for each
entity type (e.g. table). The schemas allow the tskit Python API to encode and decode
metadata automatically and, most importantly, tells downstream users and tools how to
decode and interpret the metadata. For example, the `msprime` schema for populations
requires both a `name` and a `description` for each defined population: these names and
descriptions can assist downstream users in understanding and using `msprime` tree
sequences. It is best practice to populate such metadata fields if your files will be
used by any third party, or if you wish to remember what the rows refer to some time
after making the file!

Technically, schemas describe what information is stored in each metadata record, and
how it is to be encoded, plus some optional rules about the types and ranges of data
that can be stored. Every node's metadata follows the node schema, every mutation's
metadata the mutation schema, and so on. Most users of tree-sequence files will not
need to modify the schemas: typically, as in the example of `msprime` above, schemas are
defined by the software which created the tree-sequence file. The exact metadata stored
depends on the use case; it is also possible for subsequent processes to add or modify
the schemas, if they wish to add to or modify the types (or encoding) of the metadata.

The metadata schemas are in the form of a
[JSON Schema](http://json-schema.org/) (a good guide to JSON Schema is at
[Understanding JSON Schema](https://json-schema.org/understanding-json-schema/)). The
schema must specify an object with properties,
the keys and types of those properties are specified along with optional
long-form names, descriptions and validations such as min/max or regex matching for
strings, see the {ref}`sec_metadata_schema_examples` below.

As a convenience the simplest, permissive JSON schema is available as
{meth}`MetadataSchema.permissive_json()`.

The {ref}`sec_tutorial_metadata` Tutorial shows how to use schemas and access metadata
in the tskit Python API.

Note that the C API simply provides byte-array binary access to the metadata,
leaving the encoding and decoding to the user. The same can be achieved with the Python
API, see {ref}`sec_tutorial_metadata_binary`.


(sec_metadata_examples)=

## Examples

In this section we give some examples of how to define metadata
schemas and how to add metadata to various parts of a tree sequence
using the Python API. For simplicity, these initial examples use the JSON codec
(see {ref}`sec_metadata_codecs`).

(sec_metadata_examples_top_level)=

### Top level

Top level metadata is associated with the tree sequence as a whole, rather than
any specific table. This is used, for example, by programs such as
[SLiM](https://github.com/MesserLab/SLiM) to store information about the sort of
model that was used to generate the tree sequence (but note that detailed information
used to recreate the tree sequence is better stored in {ref}`sec_provenance`).

Here's an example of adding your own top-level metadata to a tree sequence:

```{code-cell}
import tskit
# Define some top-level metadata you might want to add to a tree sequence
top_level_metadata = {
    "taxonomy": {"species": "Arabidopsis lyrata", "subspecies": "petraea"},
    "generation_time": 2,
}

# Generate a simple tree sequence of one random tree.
ts = tskit.Tree.generate_random_binary(8, branch_length=10, random_seed=9).tree_sequence

# To edit a tree sequence, first dump it to tables.
tables = ts.dump_tables()

# Set the metadata schema for the top-level metadata
tables.metadata_schema = tskit.MetadataSchema.permissive_json()  # simplest schema
# Set the metadata itself
tables.metadata = top_level_metadata

ts = tables.tree_sequence()
print(
    "The tree sequence is of",
    ts.metadata["taxonomy"]["species"],
    "subsp.",
    ts.metadata["taxonomy"]["subspecies"],
)
```

In this case, the species and subspecies name are self-explanatory, but
the interpretation of the `generation_time` field is less clear. Setting
a more precise schema will help other users of your tree sequence:

```{code-cell}
schema = {
    "codec": "json",
    "type": "object",
    "properties": {
        "generation_time": {"type": "number", "description": "Generation time in years"},
    },
    "additionalProperties": True,  # optional: True by default anyway
}
tables.metadata_schema = tskit.MetadataSchema(schema)
tables.metadata = top_level_metadata  # put the metadata back in
ts = tables.tree_sequence()
print(ts.metadata)
```

Note that the schema here only describes the `generation_time` field. The
metadata also contains additional fields (such as the species) that are
not in the schema; this is allowed because `additionalProperties` is `True`
(assumed by default in the {ref}`sec_metadata_codecs_json` codec, but shown
above for clarity). 

Explicitly specified fields are *validated* on input, helping to avoid errors.
For example, setting the generation time to a string will now raise an error:

```{code-cell}
:tags: ["raises-exception", "output_scroll"]
tables.metadata = {"generation_time": "two of your earth years"}
```

:::{note}
Although we have stored the generation time in metadata, the
time *units* of a tree sequence should be stored in the 
{attr}`~TreeSequence.time_units` attribute, not in
metadata. For example, we could set `tables.time_units = "generations"`.
:::

(sec_metadata_examples_reference_sequence)=

### Reference sequence

Often a genome will be associated with a
reference sequence for that species. In this case, we might want to
store not just the species name, but also e.g. the build version of
the reference sequence, and possibly the reference sequence itself.
There is built-in support for this in tskit, via the
{attr}`~tskit.ReferenceSequence.metadata` and
{attr}`~tskit.ReferenceSequence.metadata_schema` properties
of the {attr}`TreeSequence.reference_sequence` attribute
(see the {ref}`sec_data_model_reference_sequence` documentation).

:::{todo}
Add examples of reference sequence metadata when the API becomes less
preliminary. This should
include an example where we declare (or better, use on we define
in the library) a standard metadata schema for a species, which
defines and documents accession numbers, genome builds, etc. e.g.

```python
tables.reference_sequence.metadata_schema = standard_schema
tables.reference_sequence.metadata = {...}
ts = tables.tree_sequence()
```
:::

(sec_metadata_examples_tables)=

### Tables

Each table in a tree sequence (apart from the provenance table)
can have its own metadata, and associated metadata schema.

```{code-cell}
tables.individuals.metadata_schema = tskit.MetadataSchema.permissive_json()
tables.individuals.add_row(metadata={"Accession ID": "ABC123"})
ts = tables.tree_sequence()
print(",\n           ".join(str(ts.individual(0)).split(", ")))

```

However, we might want something more descriptive than the default
{meth}`~MetadataSchema.permissive_json()`. schema. We could create
a new schema, or modify the existing one. Modification is useful
if a nontrivial schema has been set already, for example in the
{ref}`case of populations <msprime:sec_demography_populations_metadata>`
when the tree sequence has been generated by
{func}`msprime:msprime.sim_ancestry`.

```{code-cell}
# Modify an existing schema
schema_as_python_dict = tables.individuals.metadata_schema.schema
if "properties" not in schema_as_python_dict:
    schema_as_python_dict["properties"] = {}
schema_as_python_dict["properties"]["Accession ID"] = {
    "type": "string", "description": "An accession ID for this individual"}

# Optional: require an accession id to be specified for all individuals
if "required" not in schema_as_python_dict:
    schema_as_python_dict["required"] = []
schema_as_python_dict["required"].append("Accession ID")

# Set the schema back on the table
tables.individuals.metadata_schema = tskit.MetadataSchema(schema_as_python_dict)

# Put all the metadata back in, using validate_and_encode_row to validate it
tables.individuals.packset_metadata([
    tables.individuals.metadata_schema.validate_and_encode_row(ind.metadata)
    for ind in tables.individuals
])
print("New schema:", tables.individuals.metadata_schema)
```


### Defaults

Since we specified that the `accession_id` property was required in the
example above, the user *always* has to provide it, otherwise it will
fail to validate:

```{code-cell}
:tags: ["raises-exception", "output_scroll"]
tables.individuals.add_row(metadata={"Comment": "This has no accession ID"})
```

However, rather than require a user-specified value, we can provide a
default, which will be returned if the field is absent. In this case the property
should not be marked as `required`.

```{code-cell}
new_schema = {
    "codec": "json",
    "type": "object",
    "properties": {
        "Accession ID": {
            "type": "string",
            "description": "An accession ID for this individual",
            "default": "N/A",  # Default if this property is absent
        },
    },
    "default": {"Accession ID": "N/A"},  # Default if no metadata in this row
}
tables.individuals.metadata_schema = tskit.MetadataSchema(new_schema)
tables.individuals.packset_metadata([
    tables.individuals.metadata_schema.validate_and_encode_row(ind.metadata)
    for ind in tables.individuals
])
tables.individuals.add_row(metadata={"Comment": "This has no accession ID"})
ts = tables.tree_sequence()

print("Newly added individual:")
print(",\n           ".join(str(ts.individual(-1)).split(", ")))
```

:::{note}
In the {ref}`sec_metadata_codecs_json` codec, defaults can only
be set for the shallowest level of the metadata object.
:::

(sec_metadata_codecs)=

## Codecs

The underlying metadata is in raw binary (see
{ref}`data model <sec_metadata_definition>`) and so it
must be encoded and decoded. The C API does not do this, but the Python API will
use the schema to decode the metadata to Python objects.
The encoding for doing this is specified in the top-level schema property `codec`.
Currently the Python API supports the `json` codec which encodes metadata as
[JSON](https://www.json.org/json-en.html), and the `struct` codec which encodes
metadata in an efficient schema-defined binary format using {func}`python:struct.pack` .

(sec_metadata_codecs_json)=

### `json`

When `json` is specified as the `codec` in the schema the metadata is encoded in
the human readable [JSON](https://www.json.org/json-en.html) format. As this format
is human readable and encodes numbers as text it uses more bytes than the `struct`
format. However it is simpler to configure as it doesn't require any format specifier
for each type in the schema. Tskit deviates from standard JSON in that
empty metadata is interpreted as an empty object. This is to allow setting of a schema
to a table with out the need to modify all existing empty rows.

(sec_metadata_codecs_struct)=

### `struct`

When `struct` is specifed as the `codec` in the schema the metadata is encoded
using {func}`python:struct.pack` which results in a compact binary representation which
is much smaller and generally faster to encode/decode than JSON.

This codec places extra restrictions on the schema:

1. Each property must have a `binaryFormat`
    This sets the binary encoding used for the property.

2. All metadata objects must have fixed properties.
    This means that additional properties not listed in the schema are disallowed. Any
    property that does not have a `default` specified in the schema must be present.
    Default values will be encoded.

3. Arrays must be lists of homogeneous objects.
    For example, this is not valid:
    ```
    {"type": "array", "items": [{"type": "number"}, {"type": "string"}]}
    ```

4. Types must be singular and not unions.
    For example, this is not valid:
    ```
    {"type": ["number", "string"]}
    ```
    One exception is that the top-level can be a union of `object` and `null` to
    support the case where some rows do not have metadata.

5. The order that properties are encoded is by default alphabetically by name.
    The order can be overridden by setting an optional numerical `index` on each
    property. This is due to objects being unordered in JSON and Python `dicts`.


#### binaryFormat

To determine the binary encoding of each property in the metadata the `binaryFormat` key is used.
This describes the encoding for each property using `struct`
[format characters](https://docs.python.org/3/library/struct.html#format-characters).
For example an unsigned 8-byte integer can be specified with::

```
{"type": "number", "binaryFormat":"Q"}
```

And a length 10 string with::

```
{"type": "string", "binaryFormat":"10p"}
```

Some of the text below is copied from
[the python docs](https://docs.python.org/3/library/struct.html).


##### Numeric and boolean types

The supported numeric and boolean types are:


```{list-table}
:header-rows: 1
* - Format
  - C Type
  - Python type
  - Numpy type
  - Size in bytes
* - `?`
  - *_Bool*
  - bool
  - bool
  - 1
* - `b`
  - *signed char*
  - integer
  - int8
  - 1
* - `B`
  - *unsigned char*
  - integer
  - uint8
  - 1
* - `h`
  - *short*
  - integer
  - int16
  - 2
* - `H`
  - *unsigned short*
  - integer
  - uint16
  - 2
* - `i`
  - *int*
  - integer
  - int32
  - 4
* - `I`
  - *unsigned int*
  - integer
  - uint32
  - 4
* - `l`
  - `long`
  - integer
  - int32
  - 4
* - `L`
  - *unsigned long*
  - integer
  - uint32
  - 4
* - `q`
  - `long long`
  - integer
  - int64
  - 8
* - `Q`
  - *unsigned long long*
  - integer
  - uint64
  - 8
* - `f`
  - *float*
  - float
  - float32
  - 4
* - `d`
  - *double*
  - float
  - float64
  - 8
```

When attempting to pack a non-integer using any of the integer conversion
codes, if the non-integer has a `__index__` method then that method is
called to convert the argument to an integer before packing.

For the `'f'` and `'d'` conversion codes, the packed
representation uses the IEEE 754 binary32 or binary64 format (for
`'f'` or `'d'` respectively), regardless of the floating-point
format used by the platform.

Note that endian-ness cannot be specified and is fixed at little endian.

When encoding a value using one of the integer formats (`'b'`,
`'B'`, `'h'`, `'H'`, `'i'`, `'I'`, `'l'`, `'L'`,
`'q'`, `'Q'`), if the value is outside the valid range for that format
then {exc}`struct.error` is raised.

For the `'?'` format character, the decoded value will be either `True` or
`False`. When encoding, the truth value of the input is used.


##### Strings

```{list-table}
:header-rows: 1
* - Format
  - C Type
  - Python type
  - Size in bytes
* - `x`
  - pad byte
  - no value
  - as specified
* - `c`
  - *char*
  - string of length 1
  - 1
* - `s`
  - *char[]*
  - string
  - as specified
* - `p`
  - *char[]*
  - string
  - as specified
```

For the `'s'` format character, the number prefixed is interpreted as the length in
bytes, for example,
`'10s'` means a single 10-byte string. For packing, the string is
truncated or padded with null bytes as appropriate to make it fit. For
unpacking, the resulting bytes object always has exactly the specified number
of bytes, unless `nullTerminated` is `true`, in which case it ends at the first
`null`. As a special case, `'0s'` means a single, empty string.

The `'p'` format character encodes a "Pascal string", meaning a short
variable-length string stored in a fixed number of bytes, given by the count.
The first byte stored is the length of the string, or 255, whichever is
smaller.  The bytes of the string follow.  If the string to encode is too long
(longer than the count minus 1), only the leading
`count-1` bytes of the string are stored.  If the string is shorter than
`count-1`, it is padded with null bytes so that exactly count bytes in all
are used.  Note that strings specified with this format cannot be longer than 255.

Strings that are longer than the specified length will be silently truncated,
note that the length is in bytes, not characters.

The string encoding can be set with `stringEncoding` which defaults to `utf-8`.
A list of possible encodings is
[here](https://docs.python.org/3.7/library/codecs.html#standard-encodings).

For most cases, where there are no `null` characters in the metadata
`{"type":"string", "binaryFormat": "1024s", "nullTerminated": True}` is a good option
with the size set to that appropriate for the strings to be encoded.


##### Padding bytes

Unused padding bytes (for compatibility) can be added with a schema entry like:

```
{"type": "null", "binaryFormat":"5x"} # 5 padding bytes
```

##### Arrays

The codec stores the length of the array before the array data. The format used for the
length of the array can be chosen with `arrayLengthFormat` which must be one
of `B`, `H`, `I`, `L` or `Q` which have the same meaning as in the numeric
types above. `L` is the default. As an example:

```
{"type": "array", {"items": {"type":"number", "binaryFormat":"h"}}, "arrayLengthFormat":"B"}
```

Will result in an array of 2 byte integers, prepended by a single-byte array-length.

For arrays with a known fixed size, you can specify the `length` property instead:
```
{"type": "array", "length": 3, "items": {"type":"number", "binaryFormat":"i"}}
```
This creates a fixed-length array of exactly 3 integers, without storing the array length in the encoded data. 
Fixed-length arrays are more space-efficient since they don't need to store the length prefix.

When using fixed-length arrays:
1. The `arrayLengthFormat` property should not be specified
2. Arrays provided for encoding must match the specified length exactly

For dealing with legacy encodings that do not store the
length of the array, setting `noLengthEncodingExhaustBuffer` to `true` will read
elements of the array until the metadata buffer is exhausted. As such an array
with this option must be the last type in the encoded struct.


##### Union typed metadata

As a special case under the `struct` codec, the top-level type of metadata can be a
union of `object` and `null`. Set `"type": ["object", "null"]`. Properties should
be defined as normal, and will be ignored if the metadata is `None`.

(sec_metadata_schema_examples)=

## Schema examples

### Struct codec

As an example here is a schema using the `struct` codec which could apply, for example,
to the individuals in a tree sequence:

```python
complex_struct_schema = {
    "codec": "struct",
    "type": "object",
    "properties": {
        "accession_number": {"type": "integer", "binaryFormat": "i"},
        "collection_date": {
            "description": "Date of sample collection in ISO format",
            "type": "string",
            "binaryFormat": "10p",
            "pattern": "^([1-9][0-9]{3})-(1[0-2]|0[1-9])-(3[01]|0[1-9]|[12][0-9])?$",
        },
        "phenotype": {
            "description": "Phenotypic measurements on this individual",
            "type": "object",
            "properties": {
                "height": {
                    "description": "Height in metres, or NaN if unknown",
                    "type": "number",
                    "binaryFormat": "f",
                    "default": float("NaN"),
                },
                "age": {
                    "description": "Age in years at time of sampling, or -1 if unknown",
                    "type": "number",
                    "binaryFormat": "h",
                    "default": -1,
                },
            },
            "default": {},
        },
    },
    "required": ["accession_number", "collection_date"],
    "additionalProperties": False,
}

# Demonstrate use
tables.individuals.clear()
tables.individuals.metadata_schema = tskit.MetadataSchema(complex_struct_schema)
tables.individuals.add_row(
    metadata={"accession_number": 123, "collection_date": "2011-02-11"}
)
ts = tables.tree_sequence()
print(ts.individual(0).metadata)
```

This schema states that the metadata for each row of the table is an object consisting
of three properties. Property `accession_number` is a number (stored as a 4-byte int).
Property `collection_date` is a string which must satisfy a regex, which checks it is
a valid [ISO8601](https://www.iso.org/iso-8601-date-and-time-format.html) date. Property
`phenotype` is itself an object consisting of the properties `height` (a single precision
floating point number) and age (a 2 byte signed integer).
Because this is a struct codec, and neither of the the first two properties have a
default set, they must be marked as "required" (in the JSON codec if no default is given,
unspecified properties will simply be missing in the returned metadata dictionary).
Also because this is a struct codec, `additionalProperties` must be set to False. This
is assumed by default in the struct codec, but has been shown above for clarity.

(sec_metadata_api_overview)=

## Python Metadata API Overview

Schemas are represented in the Python API by the {class}`tskit.MetadataSchema`
class which can be assigned to, and retrieved from, tables via their `metadata_schema`
attribute (e.g. {attr}`tskit.IndividualTable.metadata_schema`). The schemas
for all tables can be retrieved from a {class}`tskit.TreeSequence` by the
{attr}`tskit.TreeSequence.table_metadata_schemas` attribute.

The top-level tree sequence metadata schema is set via
{attr}`tskit.TableCollection.metadata_schema` and can be accessed via
{attr}`tskit.TreeSequence.metadata_schema`.

Each table's `add_row` method (e.g. {meth}`tskit.IndividualTable.add_row`) will
validate and encode the metadata using the schema. This encoding will also happen when
tree sequence metadata is set (e.g. `table_collection.metadata = {...}`.

Metadata will be lazily decoded if accessed via
`tables.individuals[0].metadata`.  `tree_sequence.individual(0).metadata` or
`tree_sequence.metadata`

In the interests of efficiency the bulk methods of `set_columns`
(e.g. {meth}`tskit.IndividualTable.set_columns`)
and `append_columns` (e.g. {meth}`tskit.IndividualTable.append_columns`) do not
validate or encode metadata. See {ref}`sec_tutorial_metadata_bulk` for how to prepare
metadata for these methods.

Metadata processing can be disabled and raw bytes stored/retrieved. See
{ref}`sec_tutorial_metadata_binary`.

(sec_structured_array_metadata)=

## Structured array metadata

If the `struct` codec is used for metadata then the metadata can be very efficiently
accessed via a `numpy` [structured array](https://numpy.org/doc/stable/user/basics.rec.html) via the `ts.X_metadata` property, e.g. {attr}`TreeSequence.individuals_metadata`. The codec must meet the following requirements for this to work:

1. The metadata for a given object must be of a fixed size. This means that
   variable length arrays are not permitted, such that the `length` property
   must be set for all arrays.

2. Each object for a given table must be present, i.e. at the top-level
   the metadata must be an `object` and not a union of `object` and `null`.

3. Strings must use the `s` format character, as the `p` pascal string format
   is not supported by `numpy`.

As an example, let's make a tree sequence with a large amount of metadata:

```{code-cell} ipython3
import msprime
import tskit
import itertools
import time

ts = msprime.sim_ancestry(1000, recombination_rate=1, sequence_length=1000)
ts = msprime.sim_mutations(ts, rate=20)
tables = ts.dump_tables()
muts = tables.mutations.copy()
tables.mutations.clear()
schema = tskit.MetadataSchema({
             "codec": "struct",
             "type": "object",
             "properties": {
                 "id": {"type": "integer", "binaryFormat": "i"},
                 "name": {"type": "string", "binaryFormat": "15s"},
                 },
             },
)
tables.mutations.metadata_schema = schema
for i, m in enumerate(muts):
             tables.mutations.append(m.replace(metadata={
                 "id":i,
                 "name":f"name_{i}"
             }))
ts = tables.tree_sequence()
print(f"Tree sequence with {ts.num_mutations} mutations")
```

Accessing the metadata row-by-row is slow:

```{code-cell} ipython3
%timeit [m.metadata for m in ts.mutations()]
```

But accessing via the structured array is fast:

```{code-cell} ipython3
%timeit md = ts.mutations_metadata
```

Arrays of a specific key are easily accessed by item.It is also trivial to create a pandas dataframe from the metadata:

```{code-cell} ipython3
print(ts.mutations_metadata["id"][:5])

import pandas as pd
df = pd.DataFrame(ts.mutations_metadata)
print(df.head())
```


(sec_metadata_schema_schema)=

## Full metaschema

The schema for metadata schemas is formally defined using
[JSON Schema](http://json-schema.org/) and given in full here. Any schema passed to
{class}`tskit.MetadataSchema` is validated against this metaschema.

```{eval-rst}
.. literalinclude:: ../python/tskit/metadata_schema.schema.json
    :language: json
```
