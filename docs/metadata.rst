.. _sec_metadata:

========
Metadata
========

Every entity (nodes, mutations, edges,  etc.) in a tskit tree sequence can have
metadata associated with it. This is intended for storing and passing on information
that tskit itself does not use or interpret. For example information derived from a VCF
INFO field, or administrative information (such as unique identifiers) relating to
samples and populations. Note that provenance information about how a tree sequence
was created should not be stored in metadata, instead the provenance mechanisms in
tskit should be used. See :ref:`sec_provenance`.

The metadata for each entity is described by a schema for each entity type. This
schema allows the tskit Python API to encode and decode metadata and, most importantly,
tells downstream users and tools how to decode and interpret the metadata. This schema
is in the form of a
`JSON Schema <http://json-schema.org/>`_. A good to guide to creating JSON Schemas is at
`Understanding JSON Schema <https://json-schema.org/understanding-json-schema/>`_.

The metadata schema must specify an object with properties,
the keys and types of those properties are specified along with optional
long-form names, descriptions and validations such as min/max or regex matching for
strings. See :ref:`sec_metadata_example` below. Names and descriptions can assist
downstream users in understanding and using the metadata. It is best practise to
populate these fields if your files will be used by any third-party, or if you wish to
remember what they were some time after making the file!

The :ref:`sec_tutorial_metadata` Tutorial shows how to use schemas and access metadata
in the tskit Python API.

Note that the C API simply provides byte-array binary access to the metadata and
leaves encoding and decoding to the user. The same can be achieved with the Python
API, see :ref:`sec_tutorial_metadata_binary`.

.. _sec_metadata_codecs:

******
Codecs
******

As the underlying metadata is in raw binary (see
:ref:`data model <sec_metadata_definition>`) it
must be encoded and decoded, in the case of the Python API to Python objects.
The method for doing this is specified in the top-level schema property ``codec``.
Currently the Python API supports the ``json`` codec which encodes metadata as
`JSON <https://www.json.org/json-en.html>`_, and the ``struct`` codec which encodes
metadata in an efficient schema-defined binary format using :py:func:`struct.pack` .
It is possible to define a custom codec using :meth:`tskit.register_metadata_codec`,
however this should only be used when necessary as downstream users of the metadata
will not be able to decode it without the custom codec. For an example see
:ref:`sec_tutorial_metadata_custom_codec`.

----
JSON
----

When ``json`` is specified as the ``codec`` in the schema the metadata is encoded in
the human readable `JSON <https://www.json.org/json-en.html>`_ format. As this format
is human readable and encodes numbers as text it uses more bytes than the ``struct``
format. However it is simpler to configure as it doesn't require any format specifiers
in the schema.

------
struct
------

When ``struct`` is specifed as the ``codec`` in the schema the metadata is encoded
using :py:func:`struct.pack` which results in a compact binary representation. This
codec places extra restrictions on the schema:

#. Each property must have a ``binaryFormat`` entry.
    This describes the encoding for that property using ``struct``
    `format characters <https://docs.python.org/3/library/struct.html#format-characters>`_.
    For example an unsigned 8-byte integer can be specifed with::

        {'type': 'number', 'binaryFormat':'Q'}

    And a length 10 string with::

        {'type': 'string', 'binaryFormat':'10p'}

    * Endian-ness is fixed at little endian so cannot be specified.
    * Not all formats are supported - ``binaryFormat`` must match
      ``^[cbB\?hHiIlLqQfd]|\d*[spx]$``.
    * It is recommended to use ``p`` for strings, as ``s`` doesn't store where
      the string ends. Note that ``5p`` can only store a string that when encoded is 4
      bytes or less. If using ``s``, set ``nullTerminated`` to ``true`` if you are
      sure your strings do not have null characters in. This will remove the spare
      bytes at the end of the string.
    * Strings that are longer than the specified length will be silently truncated,
      note that the length is in bytes, not characters.
    * To insert padding bytes use ``null`` for ``type`` and ``x`` for ``binaryFormat``.
    * For ``array`` types the format of the length of the array can be chosen with
      ``arrayLengthFormat`` which must be one of ``B``, ``H``, ``I``, ``L`` or ``Q``
      which is the default. For dealing with legacy encodings that do not store the
      length of the array setting ``noLengthEncodingExhaustBuffer`` to true will read
      the buffer to exhaustion. As such an array with this encoding must be the last type
      in the encoded struct.
    * For strings, the encoding can be set with ``stringEncoding`` which defaults to
      ``utf-8``.

#. All metadata objects must have fixed properties.
    This means that they can have no missing properties and have no additional
    properties not listed in the schema.

#. Arrays must be lists of homogeneous objects.
    This is not valid::

    {"type": "array", "items": [{"type": "number"}, {"type": "string"}]}

#. Types must be singular and not unions.
    This is not valid::

    {"type": ["number", "string"]}

#. The order that properties are encoded is by default alphabetically by name.
    The order can be overridden by setting an optional numerical ``index`` on each
    property.


.. _sec_metadata_example:

********
Examples
********


As an example here is a schema using the ``struct`` codec which could apply, for example,
to the individuals in a tree sequence:

.. code-block:: json

    {
      "codec": "json",
      "type": "object",
      "properties": {
        "accession_number": {"type": "number", "binaryFormat": "i"},
        "collection_date": {
          "name": "Collection date",
          "description": "Date of sample collection in ISO format",
          "type": "string",
          "binaryFormat": "10p",
          "pattern": "^([1-9][0-9]{3})-(1[0-2]|0[1-9])-(3[01]|0[1-9]|[12][0-9])?$"
        },
      },
      "required": ["accession_number"],
      "additionalProperties": false,
    }

This schema states that the metadata for the each row of the table
is an object consisting of two properties. Property ``accession_number`` is a number
(stored as a 4-byte int) which must be specified (it is included in the ``required``
list).
Property ``collection_date`` is an optional string which must satisfy a regex,
which checks it is a valid `ISO8601 <https://www.iso.org/iso-8601-date-and-time-format
.html>`_ date.
Any other properties are not allowed (``additionalProperties`` is false).

.. _sec_metadata_api_overview:

****************************
Python Metadata API Overview
****************************

Schemas are represented in the Python API by the :class:`tskit.MetadataSchema`
class which can be assigned to, and retrieved from, tables via their ``metadata_schema``
attribute (e.g. :attr:`tskit.IndividualTable.metadata_schema`). The schemas
for all tables can be retrieved from a :class:`tskit.TreeSequence` by the
:attr:`tskit.TreeSequence.table_metadata_schemas` attribute.

Each table's ``add_row`` method (e.g. :meth:`tskit.IndividualTable.add_row`) will
validate and encode the metadata using the schema.

Metadata will be lazily decoded if accessed via
``tables.individuals[0].metadata`` or ``tree_sequence.individual(0).metadata``.

In the interests of efficiency the bulk methods of ``set_columns``
(e.g. :meth:`tskit.IndividualTable.set_columns`)
and ``append_columns`` (e.g. :meth:`tskit.IndividualTable.append_columns`) do not
validate or encode metadata. See :ref:`sec_tutorial_metadata_bulk` for how to prepare
metadata for these methods.

Metadata processing can be disabled and raw bytes stored/retrived. See
:ref:`sec_tutorial_metadata_binary`.

.. _sec_metadata_schema_schema:

***************
Full metaschema
***************

The schema for metadata schemas is formally defined using
`JSON Schema <http://json-schema.org/>`_ and given in full here. Any schema passed to
:class:`tskit.MetadataSchema` is validated against this metaschema.

.. literalinclude:: ../python/tskit/metadata_schema.schema.json
    :language: json
