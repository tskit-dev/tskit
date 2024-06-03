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
using the Python API.

(sec_metadata_examples_top_level)=

### Top level

```{eval-rst}
.. todo:: Add examples of top-level metadata. One with the ``permissive_json``
  schema first to to show the simplest possible way of doing it. Then
  followed with an example where we describe the metadata also.
```

(sec_metadata_examples_reference_sequence)=

### Reference sequence

```{eval-rst}
.. todo:: Add examples of reference sequence metadata. This should
  include an example where we declare (or better, use on we define
  in the library) a standard metadata schema for a species, which
  defines and documents accession numbers, genome builds, etc.
```

(sec_metadata_examples_tables)=

### Tables

```{eval-rst}
.. todo:: Add examples of adding table-level metadata schemas.
```

(sec_metadata_codecs)=

## Codecs

As the underlying metadata is in raw binary (see
{ref}`data model <sec_metadata_definition>`) it
must be encoded and decoded. The C API does not do this, but the Python API will
use the schema to decode the metadata to Python objects.
The encoding for doing this is specified in the top-level schema property `codec`.
Currently the Python API supports the `json` codec which encodes metadata as
[JSON](https://www.json.org/json-en.html), and the `struct` codec which encodes
metadata in an efficient schema-defined binary format using {func}`python:struct.pack` .


### JSON

When `json` is specified as the `codec` in the schema the metadata is encoded in
the human readable [JSON](https://www.json.org/json-en.html) format. As this format
is human readable and encodes numbers as text it uses more bytes than the `struct`
format. However it is simpler to configure as it doesn't require any format specifier
for each type in the schema. Default values for properties can be specified for only
the shallowest level of the metadata object. Tskit deviates from standard JSON in that
empty metadata is interpreted as an empty object. This is to allow setting of a schema
to a table with out the need to modify all existing empty rows.


### struct

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
schema = tskit.MetadataSchema(
    {
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
        },
        "required": ["accession_number", "collection_date"],
        "additionalProperties": False,
    }
)
```

This schema states that the metadata for each row of the table
is an object consisting of two properties. Property `accession_number` is a number
(stored as a 4-byte int).
Property `collection_date` is a string which must satisfy a regex, which checks it is
a valid [ISO8601](https://www.iso.org/iso-8601-date-and-time-format.html) date.
Both properties are required to be specified (this must always be done for the struct codec,
for the JSON codec properties can be optional).
Any other properties are not allowed (`additionalProperties` is false), this is also needed
when using struct.

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

(sec_metadata_schema_schema)=

## Full metaschema

The schema for metadata schemas is formally defined using
[JSON Schema](http://json-schema.org/) and given in full here. Any schema passed to
{class}`tskit.MetadataSchema` is validated against this metaschema.

```{eval-rst}
.. literalinclude:: ../python/tskit/metadata_schema.schema.json
    :language: json
```
