# MIT License
#
# Copyright (c) 2018-2025 Tskit Developers
# Copyright (c) 2017 University of Oxford
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Tests for metadata handling.
"""
import collections
import io
import json
import os
import pickle
import pprint
import struct
import tempfile
import unittest
from unittest.mock import patch

import msgpack
import msprime
import numpy as np
import pytest

import tskit
import tskit.exceptions as exceptions
import tskit.metadata as metadata


class TestMetadataRoundTrip(unittest.TestCase):
    """
    Tests that we can encode metadata under various formats.
    """

    def setUp(self):
        fd, self.temp_file = tempfile.mkstemp(prefix="msp_meta_test_")
        os.close(fd)

    def tearDown(self):
        os.unlink(self.temp_file)

    def test_json(self):
        ts = msprime.simulate(10, random_seed=1)
        tables = ts.dump_tables()
        nodes = tables.nodes
        # For each node, we create some Python metadata that can be JSON encoded.
        metadata = [
            {"one": j, "two": 2 * j, "three": list(range(j))} for j in range(len(nodes))
        ]
        encoded, offset = tskit.pack_strings(map(json.dumps, metadata))
        nodes.set_columns(
            flags=nodes.flags,
            time=nodes.time,
            population=nodes.population,
            metadata_offset=offset,
            metadata=encoded,
        )
        assert np.array_equal(nodes.metadata_offset, offset)
        assert np.array_equal(nodes.metadata, encoded)
        ts1 = tables.tree_sequence()
        for j, node in enumerate(ts1.nodes()):
            decoded_metadata = json.loads(node.metadata.decode())
            assert decoded_metadata == metadata[j]
        ts1.dump(self.temp_file)
        ts2 = tskit.load(self.temp_file)
        assert ts1.tables.nodes == ts2.tables.nodes

    def test_pickle(self):
        ts = msprime.simulate(10, random_seed=1)
        tables = ts.dump_tables()
        # For each node, we create some Python metadata that can be pickled
        metadata = [
            {"one": j, "two": 2 * j, "three": list(range(j))}
            for j in range(ts.num_nodes)
        ]
        encoded, offset = tskit.pack_bytes(list(map(pickle.dumps, metadata)))
        tables.nodes.set_columns(
            flags=tables.nodes.flags,
            time=tables.nodes.time,
            population=tables.nodes.population,
            metadata_offset=offset,
            metadata=encoded,
        )
        assert np.array_equal(tables.nodes.metadata_offset, offset)
        assert np.array_equal(tables.nodes.metadata, encoded)
        ts1 = tables.tree_sequence()
        for j, node in enumerate(ts1.nodes()):
            decoded_metadata = pickle.loads(node.metadata)
            assert decoded_metadata == metadata[j]
        ts1.dump(self.temp_file)
        ts2 = tskit.load(self.temp_file)
        assert ts1.tables.nodes == ts2.tables.nodes


class ExampleMetadata:
    """
    Simple class that we can pickle/unpickle in metadata.
    """

    def __init__(self, one=None, two=None):
        self.one = one
        self.two = two


class TestMetadataPickleDecoding:
    """
    Tests in which use pickle.pickle to decode metadata in nodes, sites and mutations.
    """

    def test_nodes(self):
        tables = tskit.TableCollection(sequence_length=1)
        metadata = ExampleMetadata(one="node1", two="node2")
        pickled = pickle.dumps(metadata)
        tables.nodes.add_row(time=0.125, metadata=pickled)
        ts = tables.tree_sequence()
        node = ts.node(0)
        assert node.time == 0.125
        assert node.metadata == pickled
        unpickled = pickle.loads(node.metadata)
        assert unpickled.one == metadata.one
        assert unpickled.two == metadata.two

    def test_sites(self):
        tables = tskit.TableCollection(sequence_length=1)
        metadata = ExampleMetadata(one="node1", two="node2")
        pickled = pickle.dumps(metadata)
        tables.sites.add_row(position=0.1, ancestral_state="A", metadata=pickled)
        ts = tables.tree_sequence()
        site = ts.site(0)
        assert site.position == 0.1
        assert site.ancestral_state == "A"
        assert site.metadata == pickled
        unpickled = pickle.loads(site.metadata)
        assert unpickled.one == metadata.one
        assert unpickled.two == metadata.two

    def test_mutations(self):
        tables = tskit.TableCollection(sequence_length=1)
        metadata = ExampleMetadata(one="node1", two="node2")
        pickled = pickle.dumps(metadata)
        tables.nodes.add_row(time=0)
        tables.sites.add_row(position=0.1, ancestral_state="A")
        tables.mutations.add_row(site=0, node=0, derived_state="T", metadata=pickled)
        ts = tables.tree_sequence()
        mutation = ts.site(0).mutations[0]
        assert mutation.site == 0
        assert mutation.node == 0
        assert mutation.derived_state == "T"
        assert mutation.metadata == pickled
        unpickled = pickle.loads(mutation.metadata)
        assert unpickled.one == metadata.one
        assert unpickled.two == metadata.two


class TestLoadTextMetadata:
    """
    Tests that use the load_text interface.
    """

    def test_individuals(self):
        individuals = io.StringIO(
            """\
        id  flags location     parents  metadata
        0   1     0.0,1.0,0.0  -1,-1    abc
        1   1     1.0,2.0      0,0      XYZ+
        2   0     2.0,3.0,0.0  0,1      !@#$%^&*()
        """
        )
        i = tskit.parse_individuals(
            individuals, strict=False, encoding="utf8", base64_metadata=False
        )
        expected = [
            (1, [0.0, 1.0, 0.0], [-1, -1], "abc"),
            (1, [1.0, 2.0], [0, 0], "XYZ+"),
            (0, [2.0, 3.0, 0.0], [0, 1], "!@#$%^&*()"),
        ]
        for a, b in zip(expected, i):
            assert a[0] == b.flags
            assert len(a[1]) == len(b.location)
            for x, y in zip(a[1], b.location):
                assert x == y
            assert len(a[2]) == len(b.parents)
            for x, y in zip(a[2], b.parents):
                assert x == y
        assert a[3].encode("utf8") == b.metadata

    def test_nodes(self):
        nodes = io.StringIO(
            """\
        id  is_sample   time    metadata
        0   1           0   abc
        1   1           0   XYZ+
        2   0           1   !@#$%^&*()
        """
        )
        n = tskit.parse_nodes(
            nodes, strict=False, encoding="utf8", base64_metadata=False
        )
        expected = ["abc", "XYZ+", "!@#$%^&*()"]
        for a, b in zip(expected, n):
            assert a.encode("utf8") == b.metadata

    def test_sites(self):
        sites = io.StringIO(
            """\
        position    ancestral_state metadata
        0.1 A   abc
        0.5 C   XYZ+
        0.8 G   !@#$%^&*()
        """
        )
        s = tskit.parse_sites(
            sites, strict=False, encoding="utf8", base64_metadata=False
        )
        expected = ["abc", "XYZ+", "!@#$%^&*()"]
        for a, b in zip(expected, s):
            assert a.encode("utf8") == b.metadata

    def test_mutations(self):
        mutations = io.StringIO(
            """\
        site    node    derived_state   metadata
        0   2   C   mno
        0   3   G   )(*&^%$#@!
        """
        )
        m = tskit.parse_mutations(
            mutations, strict=False, encoding="utf8", base64_metadata=False
        )
        expected = ["mno", ")(*&^%$#@!"]
        for a, b in zip(expected, m):
            assert a.encode("utf8") == b.metadata

    def test_populations(self):
        populations = io.StringIO(
            """\
        id    metadata
        0     mno
        1     )(*&^%$#@!
        """
        )
        p = tskit.parse_populations(
            populations, strict=False, encoding="utf8", base64_metadata=False
        )
        expected = ["mno", ")(*&^%$#@!"]
        for a, b in zip(expected, p):
            assert a.encode("utf8") == b.metadata

    @pytest.mark.parametrize(
        "base64_metadata,expected", [(True, ["pop", "gen"]), (False, ["cG9w", "Z2Vu"])]
    )
    def test_migrations(self, base64_metadata, expected):
        migrations = io.StringIO(
            """\
        left    right    node    source    dest    time    metadata
        10    100    0    3    4    123.0    cG9w
        150    360    1    1    2    307.0    Z2Vu
        """
        )
        m = tskit.parse_migrations(
            migrations, strict=False, encoding="utf8", base64_metadata=base64_metadata
        )
        for a, b in zip(expected, m):
            assert a.encode("utf8") == b.metadata


class TestMetadataModule:
    """
    Tests that use the metadata module
    """

    def test_metadata_schema(self):
        # Bad jsonschema
        with pytest.raises(exceptions.MetadataSchemaValidationError):
            metadata.MetadataSchema(
                {"codec": "json", "additionalProperties": "THIS ISN'T RIGHT"},
            )
        # Bad codec
        with pytest.raises(exceptions.MetadataSchemaValidationError):
            metadata.MetadataSchema({"codec": "morse-code"})
        # Missing codec
        with pytest.raises(exceptions.MetadataSchemaValidationError):
            metadata.MetadataSchema({})
        schema = {
            "codec": "json",
            "title": "Example Metadata",
            "type": "object",
            "properties": {"one": {"type": "string"}, "two": {"type": "number"}},
            "required": ["one", "two"],
            "additionalProperties": False,
        }
        ms = metadata.MetadataSchema(schema)
        assert repr(ms) == tskit.canonical_json(schema)
        # Missing required properties
        with pytest.raises(exceptions.MetadataValidationError):
            ms.validate_and_encode_row({})

    def test_schema_str(self):
        schema = {
            "codec": "json",
            "title": "Example Metadata",
            "type": "object",
            "properties": {"one": {"type": "string"}, "two": {"type": "number"}},
            "required": ["one", "two"],
            "additionalProperties": False,
        }
        assert (
            str(metadata.MetadataSchema(schema))
            == f"tskit.MetadataSchema(\n{pprint.pformat(schema)}\n)"
        )

    def test_register_codec(self):
        class TestCodec(metadata.AbstractMetadataCodec):
            pass

        metadata.register_metadata_codec(TestCodec, "test")
        assert TestCodec == metadata.codec_registry["test"]

    def test_parse(self):
        # Empty string gives MetaDataSchema with None codec
        ms = metadata.parse_metadata_schema("")
        assert isinstance(ms, metadata.MetadataSchema)
        assert ms.schema is None
        assert ms.asdict() is None

        # json gives MetaDataSchema with json codec
        ms = metadata.parse_metadata_schema(json.dumps({"codec": "json"}))
        assert isinstance(ms, metadata.MetadataSchema)
        assert ms.schema == {"codec": "json"}
        assert ms.asdict() == {"codec": "json"}
        # check we get a copy
        assert ms.asdict() is not ms._schema

        # Bad JSON gives error
        with pytest.raises(ValueError):
            metadata.parse_metadata_schema(json.dumps({"codec": "json"})[:-1])

    def test_canonical_string(self):
        schema = collections.OrderedDict(
            codec="json",
            title="Example Metadata",
            type="object",
            properties=collections.OrderedDict(
                one={"type": "string"}, two={"type": "number"}
            ),
            required=["one", "two"],
            additionalProperties=False,
        )
        schema2 = collections.OrderedDict(
            type="object",
            properties=collections.OrderedDict(
                two={"type": "number"}, one={"type": "string"}
            ),
            required=["one", "two"],
            additionalProperties=False,
            title="Example Metadata",
            codec="json",
        )
        assert json.dumps(schema) != json.dumps(schema2)
        assert repr(metadata.MetadataSchema(schema)) == repr(
            metadata.MetadataSchema(schema2)
        )

    def test_equality(self):
        schema = metadata.MetadataSchema(
            {
                "codec": "json",
                "title": "Example Metadata",
                "type": "object",
                "properties": {"one": {"type": "string"}, "two": {"type": "number"}},
                "required": ["one", "two"],
                "additionalProperties": False,
            }
        )
        schema_same = metadata.MetadataSchema(
            collections.OrderedDict(
                type="object",
                properties=collections.OrderedDict(
                    two={"type": "number"}, one={"type": "string"}
                ),
                required=["one", "two"],
                additionalProperties=False,
                title="Example Metadata",
                codec="json",
            )
        )
        schema_diff = metadata.MetadataSchema(
            {
                "codec": "json",
                "title": "Example Metadata",
                "type": "object",
                "properties": {"one": {"type": "string"}, "two": {"type": "string"}},
                "required": ["one", "two"],
                "additionalProperties": False,
            }
        )
        assert schema == schema
        assert not (schema != schema)
        assert schema == schema_same
        assert not (schema != schema_same)
        assert schema != schema_diff
        assert not (schema == schema_diff)

    def test_bad_top_level_type(self):
        for bad_type in ["array", "boolean", "integer", "null", "number", "string"]:
            schema = {
                "codec": "json",
                "type": bad_type,
            }
            with pytest.raises(exceptions.MetadataSchemaValidationError):
                metadata.MetadataSchema(schema)

    @pytest.mark.parametrize("codec", ["struct", "json"])
    def test_null_union_top_level(self, codec):
        schema = {
            "codec": f"{codec}",
            "type": ["object", "null"],
            "properties": {
                "one": {
                    "type": "string",
                    "binaryFormat": "1024s",
                    "nullTerminated": True,
                },
                "two": {"type": "number", "binaryFormat": "i"},
            },
        }
        ms = metadata.MetadataSchema(schema)
        row_data = {"one": "tree", "two": 5}
        assert ms.decode_row(ms.validate_and_encode_row(row_data)) == row_data
        assert ms.decode_row(ms.validate_and_encode_row(None)) is None

    def test_null_codec(self):
        ms = metadata.MetadataSchema(None)
        assert repr(ms) == ""
        row = b"Some binary data that tskit can't interpret "
        # Encode/decode are no-ops
        assert row == ms.validate_and_encode_row(row)
        assert row == ms.decode_row(row)
        # Only bytes validate
        with pytest.raises(TypeError):
            ms.validate_and_encode_row({})

    def test_json_codec(self):
        schema = {
            "codec": "json",
            "title": "Example Metadata",
            "type": "object",
            "properties": {"one": {"type": "string"}, "two": {"type": "number"}},
            "required": ["one", "two"],
            "additionalProperties": False,
        }
        ms = metadata.MetadataSchema(schema)
        # Valid row data
        row_data = {"one": "tree", "two": 5}
        assert (
            ms.validate_and_encode_row(row_data)
            == tskit.canonical_json(row_data).encode()
        )
        assert ms.decode_row(json.dumps(row_data).encode()) == row_data
        # Round trip
        assert ms.decode_row(ms.validate_and_encode_row(row_data)) == row_data
        # Test canonical encoding
        row_data = collections.OrderedDict(one="tree", two=5)
        row_data2 = collections.OrderedDict(two=5, one="tree")
        assert json.dumps(row_data) != json.dumps(row_data2)
        assert ms.validate_and_encode_row(row_data) == ms.validate_and_encode_row(
            row_data2
        )

    def test_msgpack_codec(self):
        class MsgPackCodec(metadata.AbstractMetadataCodec):
            def __init__(self, schema):
                pass

            def encode(self, obj):
                return msgpack.dumps(obj)

            def decode(self, encoded):
                return msgpack.loads(encoded)

        metadata.register_metadata_codec(MsgPackCodec, "msgpack")

        schema = {
            "codec": "msgpack",
            "title": "Example Metadata",
            "type": "object",
            "properties": {"one": {"type": "string"}, "two": {"type": "number"}},
            "required": ["one", "two"],
            "additionalProperties": False,
        }
        ms = metadata.MetadataSchema(schema)
        # Valid row data
        row_data = {"one": "tree", "two": 5}
        assert ms.validate_and_encode_row(row_data) == msgpack.dumps(row_data)
        assert ms.decode_row(msgpack.dumps(row_data)) == row_data
        # Round trip
        assert ms.decode_row(ms.validate_and_encode_row(row_data)) == row_data


class TestJSONCodec:
    def test_simple_default(self):
        schema = {
            "codec": "json",
            "type": "object",
            "properties": {"number": {"type": "number", "default": 5}},
        }
        ms = tskit.MetadataSchema(schema)
        assert ms.decode_row(b"") == {"number": 5}
        assert ms.decode_row(ms.validate_and_encode_row({})) == {"number": 5}
        assert ms.decode_row(ms.validate_and_encode_row({"number": 42})) == {
            "number": 42
        }

    def test_nested_default_error(self):
        schema = {
            "codec": "json",
            "type": "object",
            "properties": {
                "obj": {
                    "type": "object",
                    "properties": {
                        "nested_obj_no_default": {
                            "type": "object",
                            "properties": {},
                        },
                        "nested_obj": {
                            "type": "object",
                            "properties": {},
                            "default": {"foo": "bar"},
                        },
                    },
                }
            },
        }
        with pytest.raises(
            tskit.MetadataSchemaValidationError,
            match="Defaults can only be specified at the top level for JSON codec",
        ):
            tskit.MetadataSchema(schema)

    def test_bad_type_error(self):
        ms = tskit.MetadataSchema({"codec": "json"})
        with pytest.raises(
            exceptions.MetadataEncodingError,
            match="Could not encode metadata of type TableCollection",
        ):
            ms.validate_and_encode_row(tskit.TableCollection(1))

    def test_skip_validation(self):
        ms = tskit.MetadataSchema({"codec": "json"})
        assert ms._bypass_validation
        with patch.object(ms, "_validate_row", return_value=True) as mocked_validate:
            ms.validate_and_encode_row({})
            assert mocked_validate.call_count == 0

    def test_dont_skip_validation(self):
        ms = tskit.MetadataSchema({"codec": "json", "properties": {"foo": {}}})
        assert not ms._bypass_validation
        with patch.object(ms, "_validate_row", return_value=True) as mocked_validate:
            ms.validate_and_encode_row({})
            assert mocked_validate.call_count == 1

    def test_dont_skip_validation_other_codecs(self):
        ms = tskit.MetadataSchema(
            {
                "codec": "struct",
                "type": "object",
                "properties": {
                    "int": {"type": "number", "binaryFormat": "i"},
                },
            }
        )
        assert not ms._bypass_validation
        with patch.object(ms, "_validate_row", return_value=True) as mocked_validate:
            ms.validate_and_encode_row({"int": 1})
            assert mocked_validate.call_count == 1

    def test_zero_length(self):
        ms = tskit.MetadataSchema({"codec": "json"})
        assert ms.decode_row(b"") == {}


class TestStructCodec:
    def encode_decode(self, method_name, sub_schema, obj, buffer):
        assert (
            getattr(metadata.StructCodec, f"{method_name}_encode")(sub_schema)(obj)
            == buffer
        )
        assert (
            getattr(metadata.StructCodec, f"{method_name}_decode")(sub_schema)(
                iter(buffer)
            )
            == obj
        )

    def test_order_schema(self):
        # Make a guaranteed-unordered nested, schema
        schema = {
            "codec": "struct",
            "title": "Example Struct-encoded Metadata",
            "type": "object",
            "properties": collections.OrderedDict(
                [
                    ("d", {"type": "number", "binaryFormat": "L"}),
                    ("a", {"type": "string", "binaryFormat": "10s"}),
                    (
                        "f",
                        {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": collections.OrderedDict(
                                    [
                                        (
                                            "m",
                                            {
                                                "type": "number",
                                                "index": 0,
                                                "binaryFormat": "L",
                                            },
                                        ),
                                        (
                                            "n",
                                            {
                                                "type": "string",
                                                "index": -1000,
                                                "binaryFormat": "10s",
                                            },
                                        ),
                                        (
                                            "l",
                                            {
                                                "type": "string",
                                                "index": 1000,
                                                "binaryFormat": "10s",
                                            },
                                        ),
                                    ]
                                ),
                            },
                        },
                    ),
                    ("c", {"type": "string", "binaryFormat": "10s"}),
                    (
                        "h",
                        {
                            "type": "object",
                            "properties": collections.OrderedDict(
                                [
                                    (
                                        "i",
                                        {
                                            "type": "string",
                                            "index": 1000,
                                            "binaryFormat": "10s",
                                        },
                                    ),
                                    (
                                        "j",
                                        {
                                            "type": "string",
                                            "index": 567,
                                            "binaryFormat": "10s",
                                        },
                                    ),
                                    (
                                        "k",
                                        {
                                            "type": "number",
                                            "index": 567.5,
                                            "binaryFormat": "L",
                                        },
                                    ),
                                ]
                            ),
                        },
                    ),
                    ("e", {"type": "string", "binaryFormat": "10s"}),
                    ("g", {"type": "string", "binaryFormat": "10s"}),
                    ("b", {"type": "number", "binaryFormat": "L"}),
                ]
            ),
            "required": ["one", "two"],
            "additionalProperties": False,
        }
        schema_sorted = {
            "codec": "struct",
            "title": "Example Struct-encoded Metadata",
            "type": "object",
            "properties": collections.OrderedDict(
                [
                    ("a", {"type": "string", "binaryFormat": "10s"}),
                    ("b", {"type": "number", "binaryFormat": "L"}),
                    ("c", {"type": "string", "binaryFormat": "10s"}),
                    ("d", {"type": "number", "binaryFormat": "L"}),
                    ("e", {"type": "string", "binaryFormat": "10s"}),
                    (
                        "f",
                        {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": collections.OrderedDict(
                                    [
                                        (
                                            "n",
                                            {
                                                "type": "string",
                                                "index": -1000,
                                                "binaryFormat": "10s",
                                            },
                                        ),
                                        (
                                            "m",
                                            {
                                                "type": "number",
                                                "index": 0,
                                                "binaryFormat": "L",
                                            },
                                        ),
                                        (
                                            "l",
                                            {
                                                "type": "string",
                                                "index": 1000,
                                                "binaryFormat": "10s",
                                            },
                                        ),
                                    ]
                                ),
                            },
                        },
                    ),
                    ("g", {"type": "string", "binaryFormat": "10s"}),
                    (
                        "h",
                        {
                            "type": "object",
                            "properties": collections.OrderedDict(
                                [
                                    (
                                        "j",
                                        {
                                            "type": "string",
                                            "index": 567,
                                            "binaryFormat": "10s",
                                        },
                                    ),
                                    (
                                        "k",
                                        {
                                            "type": "number",
                                            "index": 567.5,
                                            "binaryFormat": "L",
                                        },
                                    ),
                                    (
                                        "i",
                                        {
                                            "type": "string",
                                            "index": 1000,
                                            "binaryFormat": "10s",
                                        },
                                    ),
                                ]
                            ),
                        },
                    ),
                ]
            ),
            "required": ["one", "two"],
            "additionalProperties": False,
        }
        assert metadata.StructCodec.order_by_index(schema) == schema_sorted

    def test_make_encode_and_decode(self):
        self.encode_decode(
            "make",
            {
                "type": "array",
                "arrayLengthFormat": "B",
                "items": {"type": "number", "binaryFormat": "b"},
            },
            list(range(5)),
            b"\x05\x00\x01\x02\x03\x04",
        )
        self.encode_decode(
            "make",
            {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "binaryFormat": "b"},
                    "b": {"type": "string", "binaryFormat": "5p"},
                },
            },
            {"a": 5, "b": "FOO"},
            b"\x05\x03FOO\x00",
        )
        self.encode_decode(
            "make",
            {"type": "string", "binaryFormat": "10p"},
            "FOOBAR",
            b"\x06FOOBAR\x00\x00\x00",
        )
        self.encode_decode("make", {"type": "null"}, None, b"")
        self.encode_decode(
            "make", {"type": "boolean", "binaryFormat": "?"}, True, b"\x01"
        )
        self.encode_decode(
            "make", {"type": "integer", "binaryFormat": "b"}, -128, b"\x80"
        )
        self.encode_decode(
            "make",
            {"type": "number", "binaryFormat": "f"},
            42.424198150634766,
            b"a\xb2)B",
        )

    def test_make_array_encode_and_decode(self):
        # Default array length format is 'L'
        self.encode_decode(
            "make_array",
            {"type": "array", "items": {"type": "number", "binaryFormat": "b"}},
            list(range(5)),
            b"\x05\x00\x00\x00\x00\x01\x02\x03\x04",
        )
        self.encode_decode(
            "make_array",
            {
                "type": "array",
                "arrayLengthFormat": "H",
                "items": {"type": "number", "binaryFormat": "b"},
            },
            list(range(6)),
            b"\x06\x00\x00\x01\x02\x03\x04\x05",
        )
        self.encode_decode(
            "make_array",
            {
                "type": "array",
                "arrayLengthFormat": "B",
                "items": {"type": "number", "binaryFormat": "b"},
            },
            [],
            b"\x00",
        )
        sub_schema = {
            "type": "array",
            "arrayLengthFormat": "B",
            "items": {
                "type": "array",
                "arrayLengthFormat": "B",
                "items": {"type": "number", "binaryFormat": "b"},
            },
        }
        self.encode_decode("make_array", sub_schema, [], b"\x00")
        self.encode_decode("make_array", sub_schema, [[]], b"\x01\x00")
        self.encode_decode(
            "make_array", sub_schema, [[3, 4], [5]], b"\x02\x02\x03\x04\x01\x05"
        )

    def test_make_array_no_length_encoding_exhaust_buffer(self):
        self.encode_decode(
            "make_array",
            {
                "type": "array",
                "noLengthEncodingExhaustBuffer": True,
                "items": {"type": "number", "binaryFormat": "b"},
            },
            list(range(5)),
            b"\x00\x01\x02\x03\x04",
        )

        self.encode_decode(
            "make_array",
            {
                "type": "array",
                "noLengthEncodingExhaustBuffer": True,
                "items": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "binaryFormat": "b"},
                        "b": {"type": "number", "binaryFormat": "Q"},
                        "c": {"type": "number", "binaryFormat": "?"},
                        "d": {"type": "string", "binaryFormat": "5p"},
                    },
                },
            },
            [
                {
                    "a": 5 + i,
                    "b": 18446744073709551615 - i,
                    "c": (i // 2) == 0,
                    "d": "FOO",
                }
                for i in range(10)
            ],
            b"\x05\xff\xff\xff\xff\xff\xff\xff\xff\x01\x03FOO\x00"
            b"\x06\xfe\xff\xff\xff\xff\xff\xff\xff\x01\x03FOO\x00"
            b"\x07\xfd\xff\xff\xff\xff\xff\xff\xff\x00\x03FOO\x00"
            b"\x08\xfc\xff\xff\xff\xff\xff\xff\xff\x00\x03FOO\x00"
            b"\x09\xfb\xff\xff\xff\xff\xff\xff\xff\x00\x03FOO\x00"
            b"\x0a\xfa\xff\xff\xff\xff\xff\xff\xff\x00\x03FOO\x00"
            b"\x0b\xf9\xff\xff\xff\xff\xff\xff\xff\x00\x03FOO\x00"
            b"\x0c\xf8\xff\xff\xff\xff\xff\xff\xff\x00\x03FOO\x00"
            b"\x0d\xf7\xff\xff\xff\xff\xff\xff\xff\x00\x03FOO\x00"
            b"\x0e\xf6\xff\xff\xff\xff\xff\xff\xff\x00\x03FOO\x00",
        )

        # Other struct errors should still be raised
        schema = {
            "type": "array",
            "noLengthEncodingExhaustBuffer": True,
            "items": {"type": "number", "binaryFormat": "I'M NOT VALID"},
        }
        with pytest.raises(struct.error):
            metadata.StructCodec.make_array_encode(schema)(5)
        with pytest.raises(struct.error):
            metadata.StructCodec.make_array_decode(schema)(5)

    def test_make_object_encode_and_decode(self):
        self.encode_decode("make_object", {"type": "object", "properties": {}}, {}, b"")
        self.encode_decode(
            "make_object",
            {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "binaryFormat": "b"},
                    "b": {"type": "number", "binaryFormat": "Q"},
                    "c": {"type": "number", "binaryFormat": "?"},
                    "d": {"type": "string", "binaryFormat": "5p"},
                },
            },
            {"a": 5, "b": 18446744073709551615, "c": True, "d": "FOO"},
            b"\x05\xff\xff\xff\xff\xff\xff\xff\xff\x01\x03FOO\x00",
        )
        self.encode_decode(
            "make_object",
            {
                "type": "object",
                "properties": {
                    "obj": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number", "binaryFormat": "b"},
                            "b": {"type": "number", "binaryFormat": "Q"},
                            "c": {"type": "number", "binaryFormat": "?"},
                            "d": {"type": "string", "binaryFormat": "5p"},
                        },
                    },
                },
            },
            {"obj": {"a": 5, "b": 18446744073709551615, "c": True, "d": "FOO"}},
            b"\x05\xff\xff\xff\xff\xff\xff\xff\xff\x01\x03FOO\x00",
        )

    def test_make_string_encode_and_decode(self):
        # Single byte
        self.encode_decode(
            "make_string", {"type": "string", "binaryFormat": "c"}, "a", b"a"
        )
        # With "s" encoding exactly the right size comes back fine
        self.encode_decode(
            "make_string", {"type": "string", "binaryFormat": "4s"}, "abcd", b"abcd"
        )
        # If too small gets truncated
        assert (
            metadata.StructCodec.make_string_encode(
                {"type": "string", "binaryFormat": "2s"}
            )("abcd")
            == b"ab"
        )
        # If too large gets padded - have to test separately as encode and decode are not
        # inverse of each other in this case
        assert (
            metadata.StructCodec.make_string_encode(
                {"type": "string", "binaryFormat": "6s"}
            )("abcd")
            == b"abcd\x00\x00"
        )
        # Too large getting decoded returns padding
        assert (
            metadata.StructCodec.make_string_decode(
                {"type": "string", "binaryFormat": "6s"}
            )(b"abcd\x00\x00")
            == "abcd\x00\x00"
        )
        assert (
            metadata.StructCodec.make_string_decode(
                {"type": "string", "binaryFormat": "6s", "nullTerminated": False}
            )(b"abcd\x00\x00")
            == "abcd\x00\x00"
        )
        # Unless we specify that the field is null-teminated
        self.encode_decode(
            "make_string",
            {"type": "string", "binaryFormat": "6s", "nullTerminated": True},
            "abcd",
            b"abcd\x00\x00",
        )
        # For "p" the padding is not returned, even if nullTerminated is False
        self.encode_decode(
            "make_string",
            {"type": "string", "binaryFormat": "8p"},
            "abcd",
            b"\x04abcd\x00\x00\x00",
        )

        # Unicode
        self.encode_decode(
            "make_string",
            {"type": "string", "binaryFormat": "6s", "nullTerminated": True},
            "💩",
            b"\xf0\x9f\x92\xa9\x00\x00",
        )
        self.encode_decode(
            "make_string",
            {
                "type": "string",
                "binaryFormat": "8s",
                "nullTerminated": True,
                "stringEncoding": "utf-16",
            },
            "💩",
            b"\xff\xfe=\xd8\xa9\xdc\x00\x00",
        )
        self.encode_decode(
            "make_string",
            {"type": "string", "binaryFormat": "9p", "stringEncoding": "utf-32"},
            "💩",
            b"\x08\xff\xfe\x00\x00\xa9\xf4\x01\x00",
        )

    def test_make_null_encode_and_decode(self):
        self.encode_decode("make_null", {"type": "null"}, None, b"")
        self.encode_decode(
            "make_null", {"type": "null", "binaryFormat": "x"}, None, b"\x00"
        )
        self.encode_decode(
            "make_null", {"type": "null", "binaryFormat": "3x"}, None, b"\x00\x00\x00"
        )

    def test_make_numeric_encode_and_decode(self):
        self.encode_decode(
            "make_numeric",
            {"type": "number", "binaryFormat": "f"},
            42.424198150634766,
            b"a\xb2)B",
        )
        self.encode_decode(
            "make_numeric", {"type": "integer", "binaryFormat": "b"}, 42, b"*"
        )

    def test_null_union_top_level(self):
        # This nested with mutiple values tests that the buffer length check has not
        # caused a list to past to sub-decoders
        schema = {
            "codec": "struct",
            "type": ["object", "null"],
            "properties": {
                "o": {
                    "type": "object",
                    "properties": {"x": {"type": "number", "binaryFormat": "d"}},
                },
                "a": {"type": "number", "binaryFormat": "d"},
                "b": {"type": "number", "binaryFormat": "d"},
            },
        }
        ms = metadata.MetadataSchema(schema)
        row_data = {"o": {"x": 5.5}, "a": 4, "b": 7}
        assert ms.decode_row(ms.validate_and_encode_row(row_data)) == row_data
        assert ms.decode_row(ms.validate_and_encode_row(None)) is None

    def test_default_values(self):
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {
                "int": {"type": "number", "binaryFormat": "b", "default": 42},
                "float": {"type": "number", "binaryFormat": "d"},
            },
        }
        ms = metadata.MetadataSchema(schema)
        row_data = {"float": 5.5}
        assert ms.validate_and_encode_row(row_data) == b"\x00\x00\x00\x00\x00\x00\x16@*"
        assert ms.decode_row(ms.validate_and_encode_row(row_data)) == {
            "float": 5.5,
            "int": 42,
        }

    def test_defaults_object_or_null(self):
        schema = {
            "codec": "struct",
            "type": ["object", "null"],
            "properties": {
                "int": {"type": "number", "binaryFormat": "b", "default": 42},
                "float": {"type": "number", "binaryFormat": "d"},
            },
        }
        ms = metadata.MetadataSchema(schema)
        row_data = {"float": 5.5}
        assert ms.validate_and_encode_row(row_data) == b"\x00\x00\x00\x00\x00\x00\x16@*"
        assert ms.decode_row(ms.validate_and_encode_row(row_data)) == {
            "float": 5.5,
            "int": 42,
        }
        assert ms.validate_and_encode_row(None) == b""
        assert ms.decode_row(b"") is None

    def test_add_property_to_schema(self):
        schema = metadata.MetadataSchema(
            {
                "codec": "struct",
                "type": ["object", "null"],
                "name": "Mutation metadata",
                "properties": {
                    "s": {"type": "number", "binaryFormat": "d"},
                },
                "additionalProperties": False,
            }
        )
        schema_with_additional = schema.schema
        schema_with_additional["properties"]["a"] = {
            "type": "number",
            "binaryFormat": "d",
        }
        metadata.MetadataSchema(schema_with_additional)


class TestStructCodecRoundTrip:
    def round_trip(self, schema, row_data):
        ms = metadata.MetadataSchema(schema)
        assert ms.decode_row(ms.validate_and_encode_row(row_data)) == row_data

    def test_simple_types(self):
        for type_, binaryFormat, value in (
            ("number", "i", 5),
            ("number", "d", 5.5),
            ("string", "10p", "foobar"),
            ("boolean", "?", True),
            ("boolean", "?", False),
            ("null", "10x", None),
        ):
            schema = {
                "codec": "struct",
                "type": "object",
                "properties": {type_: {"type": type_, "binaryFormat": binaryFormat}},
            }
            self.round_trip(schema, {type_: value})

        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {"null": {"type": "null"}},
        }
        self.round_trip(schema, {"null": None})

    def test_flat_object(self):
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {
                "int": {"type": "number", "binaryFormat": "i"},
                "float": {"type": "number", "binaryFormat": "d"},
                "null": {"type": "null", "binaryFormat": "3x"},
                "str": {"type": "string", "binaryFormat": "10p"},
                "bool": {"type": "boolean", "binaryFormat": "?"},
            },
        }
        self.round_trip(
            schema, {"null": None, "bool": True, "float": 5.5, "int": 5, "str": "42"}
        )

    def test_nested_object(self):
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {
                "int": {"type": "number", "binaryFormat": "i"},
                "float": {"type": "number", "binaryFormat": "d"},
                "str": {"type": "string", "binaryFormat": "10p"},
                "bool": {"type": "boolean", "binaryFormat": "?"},
                "obj": {
                    "index": 5,
                    "type": "object",
                    "properties": {
                        "int": {"type": "number", "binaryFormat": "i"},
                        "float": {"type": "number", "binaryFormat": "d"},
                        "str": {"type": "string", "binaryFormat": "5p"},
                        "bool": {"type": "boolean", "binaryFormat": "?"},
                    },
                },
            },
        }
        self.round_trip(
            schema,
            {
                "bool": True,
                "float": 5.5,
                "int": 5,
                "str": "42",
                "obj": {"float": 5.78, "int": 9, "bool": False, "str": "41"},
            },
        )

    def test_flat_array(self):
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {
                "array": {
                    "type": "array",
                    "items": {"type": "number", "binaryFormat": "i"},
                }
            },
        }
        self.round_trip(schema, {"array": []})
        self.round_trip(schema, {"array": [1]})
        self.round_trip(schema, {"array": [1, 6, -900]})

        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {
                "array": {
                    "type": "array",
                    "items": {"type": "number", "binaryFormat": "d"},
                }
            },
        }
        self.round_trip(schema, {"array": []})
        self.round_trip(schema, {"array": [1.5]})
        self.round_trip(schema, {"array": [1.5, 6.7, -900.00001]})

    def test_nested_array(self):
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {
                "array": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number", "binaryFormat": "i"},
                    },
                }
            },
        }
        self.round_trip(schema, {"array": [[]]})
        self.round_trip(schema, {"array": [[], []]})
        self.round_trip(schema, {"array": [[1]]})
        self.round_trip(schema, {"array": [[1, 6, -900]]})
        self.round_trip(schema, {"array": [[0, 987, 234903], [1, 6, -900]]})
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {
                "array": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number", "binaryFormat": "d"},
                    },
                }
            },
        }
        self.round_trip(schema, {"array": [[]]})
        self.round_trip(schema, {"array": [[], []]})
        self.round_trip(schema, {"array": [[1.67]]})
        self.round_trip(schema, {"array": [[1.34, 6.56422, -900.0000006]]})
        self.round_trip(
            schema, {"array": [[0.0, 987.123, 234903.123], [1.1235, 6, -900]]}
        )

    def test_array_of_objects(self):
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {
                "array": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "int": {"type": "number", "binaryFormat": "i"},
                            "float": {"type": "number", "binaryFormat": "d"},
                            "padding": {"type": "null", "binaryFormat": "5x"},
                            "str": {"type": "string", "binaryFormat": "10p"},
                            "bool": {"type": "boolean", "binaryFormat": "?"},
                        },
                    },
                }
            },
        }
        self.round_trip(schema, {"array": []})
        self.round_trip(
            schema,
            {
                "array": [
                    {
                        "padding": None,
                        "float": 5.78,
                        "int": 9,
                        "bool": False,
                        "str": "41",
                    }
                ]
            },
        )
        self.round_trip(
            schema,
            {
                "array": [
                    {
                        "padding": None,
                        "float": 5.78,
                        "int": 9,
                        "bool": False,
                        "str": "41",
                    },
                    {
                        "str": "FOO",
                        "int": 7,
                        "bool": True,
                        "float": 45.7,
                        "padding": None,
                    },
                ],
            },
        )

    def test_object_with_array(self):
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {
                "int": {"type": "number", "binaryFormat": "i"},
                "arr": {
                    "index": 2,
                    "type": "array",
                    "items": {"type": "number", "binaryFormat": "i"},
                },
            },
        }
        self.round_trip(schema, {"int": 5, "arr": []})
        self.round_trip(schema, {"int": 5, "arr": [5]})
        self.round_trip(schema, {"arr": [5, 6, 7], "int": 5})

    def test_array_length_format(self):
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {
                "array": {
                    "type": "array",
                    "arrayLengthFormat": "B",
                    "items": {"type": "number", "binaryFormat": "H"},
                }
            },
        }
        self.round_trip(schema, {"array": []})
        self.round_trip(schema, {"array": [1]})
        self.round_trip(schema, {"array": list(range(255))})

    def test_string_encoding(self):
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {
                "string": {
                    "type": "string",
                    "stringEncoding": "utf-16",
                    "binaryFormat": "40p",
                }
            },
        }
        self.round_trip(schema, {"string": "Test string"})

    def test_ordering_of_fields(self):
        row_data = {
            "null": None,
            "bool": True,
            "float": -1.8440714901698642e18,
            "int": 5,
            "str": "foo",
        }
        alpha_ordered_encoded = b"\x01\xaa\xbb\xcc\xdd\x05\x00\x00\x00\x03foo"
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {
                "null": {"type": "null", "binaryFormat": "3x"},
                "float": {"type": "number", "binaryFormat": "f"},
                "bool": {"type": "boolean", "binaryFormat": "?"},
                "int": {"type": "number", "binaryFormat": "b"},
                "str": {"type": "string", "binaryFormat": "4p"},
            },
        }
        alpha_ordered_encoded = b"\x01\xaa\xbb\xcc\xdd\x05\x00\x00\x00\x03foo"
        ms = metadata.MetadataSchema(schema)
        assert ms.validate_and_encode_row(row_data) == alpha_ordered_encoded
        assert ms.decode_row(alpha_ordered_encoded) == row_data
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {
                "null": {"type": "null", "binaryFormat": "3x", "index": 0},
                "float": {"type": "number", "binaryFormat": "f", "index": 1},
                "bool": {"type": "boolean", "binaryFormat": "?", "index": 2},
                "int": {"type": "number", "binaryFormat": "b", "index": 3},
                "str": {"type": "string", "binaryFormat": "4p", "index": 4},
            },
        }
        index_order_encoded = b"\x00\x00\x00\xaa\xbb\xcc\xdd\x01\x05\x03foo"
        ms = metadata.MetadataSchema(schema)
        assert ms.validate_and_encode_row(row_data) == index_order_encoded
        assert ms.decode_row(index_order_encoded) == row_data

    def test_fixed_length_array(self):
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {
                "array": {
                    "type": "array",
                    "length": 3,
                    "items": {"type": "number", "binaryFormat": "i"},
                }
            },
        }
        self.round_trip(schema, {"array": [1, 2, 3]})

        # Test with complex fixed-length arrays
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {
                "array": {
                    "type": "array",
                    "length": 2,
                    "items": {
                        "type": "object",
                        "properties": {
                            "int": {"type": "number", "binaryFormat": "i"},
                            "float": {"type": "number", "binaryFormat": "d"},
                        },
                    },
                }
            },
        }
        self.round_trip(
            schema, {"array": [{"int": 1, "float": 1.1}, {"int": 2, "float": 2.2}]}
        )

        # Test fixed-length nested arrays
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {
                "array": {
                    "type": "array",
                    "length": 2,
                    "items": {
                        "type": "array",
                        "length": 3,
                        "items": {"type": "number", "binaryFormat": "d"},
                    },
                }
            },
        }
        self.round_trip(schema, {"array": [[1.1, 1.2, 1.3], [2.1, 2.2, 2.3]]})

    def test_mixed_fixed_and_variable_arrays(self):
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {
                "fixed_array": {
                    "type": "array",
                    "length": 3,
                    "items": {"type": "number", "binaryFormat": "i"},
                },
                "variable_array": {
                    "type": "array",
                    "items": {"type": "number", "binaryFormat": "i"},
                },
            },
        }
        self.round_trip(
            schema, {"fixed_array": [1, 2, 3], "variable_array": [4, 5, 6, 7]}
        )
        self.round_trip(schema, {"fixed_array": [1, 2, 3], "variable_array": []})

        # Nested case - array of objects where each object has
        # both fixed and variable-length arrays
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {
                "objects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "fixed": {
                                "type": "array",
                                "length": 2,
                                "items": {"type": "number", "binaryFormat": "d"},
                            },
                            "variable": {
                                "type": "array",
                                "items": {"type": "number", "binaryFormat": "i"},
                            },
                        },
                    },
                }
            },
        }
        self.round_trip(
            schema,
            {
                "objects": [
                    {"fixed": [1.1, 2.2], "variable": [1, 2, 3]},
                    {"fixed": [3.3, 4.4], "variable": [4]},
                    {"fixed": [5.5, 6.6], "variable": []},
                ]
            },
        )

    def test_edge_case_zero_length_array(self):
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {
                "empty_fixed": {
                    "type": "array",
                    "length": 0,
                    "items": {"type": "number", "binaryFormat": "i"},
                }
            },
        }
        self.round_trip(schema, {"empty_fixed": []})

        # Can't provide non-empty array when length=0
        ms = metadata.MetadataSchema(schema)
        with pytest.raises(
            ValueError, match="Array length 1 does not match schema fixed length 0"
        ):
            ms.validate_and_encode_row({"empty_fixed": [1]})

        # Complex object with zero-length array
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {
                "name": {"type": "string", "binaryFormat": "10p"},
                "empty_fixed": {
                    "type": "array",
                    "length": 0,
                    "items": {"type": "number", "binaryFormat": "i"},
                },
                "value": {"type": "number", "binaryFormat": "d"},
            },
        }
        self.round_trip(schema, {"name": "test", "empty_fixed": [], "value": 42.0})


class TestStructCodecErrors:
    def encode(self, schema, row_data):
        ms = metadata.MetadataSchema(schema)
        ms.validate_and_encode_row(row_data)

    def test_missing_and_extra_property(self):
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {
                "int": {"type": "number", "binaryFormat": "i"},
                "float": {"type": "number", "binaryFormat": "d"},
            },
        }
        with pytest.raises(
            exceptions.MetadataValidationError, match="'int' is a required property"
        ):
            self.encode(schema, {"float": 5.5})
        with pytest.raises(
            exceptions.MetadataValidationError,
            match="Additional properties are not allowed",
        ):
            self.encode(
                schema, {"float": 5.5, "int": 9, "extra": "I really shouldn't be here"}
            )

    def test_bad_schema_union_type(self):
        schema = {"codec": "struct", "type": ["object", "number"], "binaryFormat": "d"}
        with pytest.raises(
            exceptions.MetadataSchemaValidationError, match="is not one of"
        ):
            metadata.MetadataSchema(schema)
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {"test": {"type": ["number", "string"], "binaryFormat": "d"}},
        }
        with pytest.raises(
            exceptions.MetadataSchemaValidationError, match="is not one of"
        ):
            metadata.MetadataSchema(schema)

    def test_bad_schema_hetrogeneous_array(self):
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {
                "array": {
                    "type": "array",
                    "items": [{"type": "number"}, {"type": "string"}],
                }
            },
        }
        with pytest.raises(
            exceptions.MetadataSchemaValidationError, match="is not of type 'object'"
        ):
            metadata.MetadataSchema(schema)

    def test_bad_binary_format(self):
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {"int": {"type": "number", "binaryFormat": "int"}},
        }
        with pytest.raises(
            exceptions.MetadataSchemaValidationError, match="does not match"
        ):
            metadata.MetadataSchema(schema)
        # Can't specify endianness
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {"int": {"type": "number", "binaryFormat": ">b"}},
        }
        with pytest.raises(
            exceptions.MetadataSchemaValidationError, match="does not match"
        ):
            metadata.MetadataSchema(schema)
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {"null": {"type": "null", "binaryFormat": "l"}},
        }
        with pytest.raises(
            exceptions.MetadataSchemaValidationError,
            match="null type binaryFormat must be padding",
        ):
            metadata.MetadataSchema(schema)

    def test_bad_array_length_format(self):
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {"array": {"type": "array", "arrayLengthFormat": "b"}},
        }
        with pytest.raises(
            exceptions.MetadataSchemaValidationError, match="does not match"
        ):
            metadata.MetadataSchema(schema)

    def test_missing_binary_format(self):
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {"int": {"type": "number"}},
        }
        with pytest.raises(
            exceptions.MetadataSchemaValidationError,
            match="number type must have binaryFormat set",
        ):
            metadata.MetadataSchema(schema)

    def test_bad_string_encoding(self):
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {
                "string": {
                    "type": "string",
                    "binaryFormat": "5s",
                    "stringEncoding": 58,
                }
            },
        }
        with pytest.raises(
            exceptions.MetadataSchemaValidationError, match="is not of type"
        ):
            metadata.MetadataSchema(schema)

    def test_bad_null_terminated(self):
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {
                "string": {
                    "type": "string",
                    "binaryFormat": "5s",
                    "nullTerminated": 58,
                }
            },
        }
        with pytest.raises(
            exceptions.MetadataSchemaValidationError, match="is not of type"
        ):
            metadata.MetadataSchema(schema)

    def test_bad_no_length_encoding_exhaust_buffer(self):
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {
                "string": {
                    "type": "string",
                    "binaryFormat": "5s",
                    "noLengthEncodingExhaustBuffer": 58,
                }
            },
        }
        with pytest.raises(
            exceptions.MetadataSchemaValidationError, match="is not of type"
        ):
            metadata.MetadataSchema(schema)

    def test_too_long_array(self):
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {
                "array": {
                    "type": "array",
                    "arrayLengthFormat": "B",
                    "items": {"type": "number", "binaryFormat": "I"},
                },
            },
        }
        data = {"array": list(range(255))}
        metadata.MetadataSchema(schema).validate_and_encode_row(data)
        data2 = {"array": list(range(256))}
        with pytest.raises(
            ValueError,
            match="Couldn't pack array size - it is likely too long for the"
            " specified arrayLengthFormat",
        ):
            metadata.MetadataSchema(schema).validate_and_encode_row(data2)

    def test_additional_properties(self):
        schema = {
            "codec": "struct",
            "type": "object",
            "additional_properties": True,
            "properties": {},
        }
        with pytest.raises(
            ValueError, match="Struct codec does not support additional_properties"
        ):
            metadata.MetadataSchema(schema)

    def test_unrequired_property_needs_default(self):
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {
                "int": {"type": "number", "binaryFormat": "i"},
                "float": {"type": "number", "binaryFormat": "d"},
            },
            "required": ["float"],
        }
        with pytest.raises(
            exceptions.MetadataSchemaValidationError,
            match="Optional property 'int' must have a default value",
        ):
            metadata.MetadataSchema(schema)

    def test_no_default_implies_required(self):
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {
                "int": {"type": "number", "binaryFormat": "i", "default": 5},
                "float": {"type": "number", "binaryFormat": "d"},
            },
        }
        self.encode(schema, {"float": 5.5})
        with pytest.raises(
            exceptions.MetadataValidationError, match="'float' is a required property"
        ):
            self.encode(schema, {})

    def test_fixed_length_array_wrong_length(self):
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {
                "array": {
                    "type": "array",
                    "length": 3,
                    "items": {"type": "number", "binaryFormat": "i"},
                },
            },
        }
        ms = metadata.MetadataSchema(schema)

        with pytest.raises(
            ValueError, match="Array length 2 does not match schema fixed length 3"
        ):
            ms.validate_and_encode_row({"array": [1, 2]})

        with pytest.raises(
            ValueError, match="Array length 4 does not match schema fixed length 3"
        ):
            ms.validate_and_encode_row({"array": [1, 2, 3, 4]})

    def test_fixed_length_array_conflicts(self):
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {
                "test": {
                    "type": "array",
                    "length": 3,
                    "noLengthEncodingExhaustBuffer": True,
                    "items": {"type": "number", "binaryFormat": "i"},
                },
            },
        }
        with pytest.raises(
            exceptions.MetadataSchemaValidationError,
            match="test array cannot have both 'length' and "
            "'noLengthEncodingExhaustBuffer' set",
        ):
            metadata.MetadataSchema(schema)

    def test_fixed_length_with_length_format(self):
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {
                "array": {
                    "type": "array",
                    "length": 3,
                    "arrayLengthFormat": "B",
                    "items": {"type": "number", "binaryFormat": "i"},
                },
            },
        }
        with pytest.raises(
            exceptions.MetadataSchemaValidationError,
            match="fixed-length array should not specify 'arrayLengthFormat'",
        ):
            metadata.MetadataSchema(schema)

    def test_negative_fixed_length(self):
        """Test that negative fixed-length values are rejected."""
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {
                "array": {
                    "type": "array",
                    "length": -5,
                    "items": {"type": "number", "binaryFormat": "i"},
                },
            },
        }
        with pytest.raises(exceptions.MetadataSchemaValidationError):
            metadata.MetadataSchema(schema)


class TestSLiMDecoding:
    """
    Test with byte strings copied from a SLiM tree sequence
    """

    def test_node(self):
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {
                "genomeID": {"type": "integer", "binaryFormat": "q", "index": 0},
                "isNull": {"type": "boolean", "binaryFormat": "?", "index": 1},
                "genomeType": {"type": "integer", "binaryFormat": "B", "index": 2},
            },
        }
        for example, expected in [
            (
                b"E,\x00\x00\x00\x00\x00\x00\x00\x01",
                {"genomeID": 11333, "genomeType": 1, "isNull": False},
            ),
            (
                b"\xdd.\x00\x00\x00\x00\x00\x00\x01\x00",
                {"genomeID": 11997, "genomeType": 0, "isNull": True},
            ),
        ]:
            assert metadata.MetadataSchema(schema).decode_row(example) == expected

    def test_individual(self):
        schema = {
            "codec": "struct",
            "type": ["object", "null"],
            "properties": {
                "pedigreeID": {"type": "integer", "binaryFormat": "q", "index": 1},
                "age": {"type": "integer", "binaryFormat": "i", "index": 2},
                "subpopulationID": {
                    "type": "integer",
                    "binaryFormat": "i",
                    "index": 3,
                },
                "sex": {"type": "integer", "binaryFormat": "i", "index": 4},
                "flags": {"type": "integer", "binaryFormat": "I", "index": 5},
            },
        }
        for example, expected in [
            (
                b"\x17\x99\x07\x00\x00\x00\x00\x00\x05\x00\x01\x00\x03\x00\x00\x00\x01"
                b"\x00\x00\x00\x00\x10\x00\x00",
                {
                    "age": 65541,
                    "flags": 4096,
                    "pedigreeID": 497943,
                    "sex": 1,
                    "subpopulationID": 3,
                },
            ),
            (b"", None),
            (
                b"\x18\x99\x07\x00\x00\x00\x00\x00\x05\x00\x00\x00\x01\x00\x00\x00\x01"
                b"\x00\x00\x00\x00\x00\x00\x00",
                {
                    "age": 5,
                    "flags": 0,
                    "pedigreeID": 497944,
                    "sex": 1,
                    "subpopulationID": 1,
                },
            ),
        ]:
            assert metadata.MetadataSchema(schema).decode_row(example) == expected

    def test_mutation(self):
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {
                "stacked_mutation_array": {
                    "type": "array",
                    "noLengthEncodingExhaustBuffer": True,
                    "items": {
                        "type": "object",
                        "properties": {
                            "mutationTypeID": {
                                "type": "integer",
                                "binaryFormat": "i",
                                "index": 1,
                            },
                            "selectionCoeff": {
                                "type": "number",
                                "binaryFormat": "f",
                                "index": 2,
                            },
                            "subpopulationID": {
                                "type": "integer",
                                "binaryFormat": "i",
                                "index": 3,
                            },
                            "originGeneration": {
                                "type": "integer",
                                "binaryFormat": "i",
                                "index": 4,
                            },
                            "nucleotide": {
                                "type": "integer",
                                "binaryFormat": "b",
                                "index": 5,
                            },
                        },
                    },
                }
            },
        }

        for example, expected in [
            (
                b"\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\xd8\x03\x00\x00\xff",
                [
                    {
                        "mutationTypeID": 1,
                        "selectionCoeff": 0.0,
                        "subpopulationID": 1,
                        "originGeneration": 984,
                        "nucleotide": -1,
                    }
                ],
            ),
            (
                b"\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\xc8\x03\x00\x00\xff"
                b"\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x94\x01\x00\x00\xff",
                [
                    {
                        "mutationTypeID": 1,
                        "selectionCoeff": 0.0,
                        "subpopulationID": 1,
                        "originGeneration": 968,
                        "nucleotide": -1,
                    },
                    {
                        "mutationTypeID": 1,
                        "selectionCoeff": 0.0,
                        "subpopulationID": 1,
                        "originGeneration": 404,
                        "nucleotide": -1,
                    },
                ],
            ),
            (
                b"\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\xd1\x03\x00\x00\xff"
                b"\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\xb1\x02\x00\x00\xff"
                b"\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\xdf\x01\x00\x00\xff"
                b"\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\xbc\x00\x00\x00\xff",
                [
                    {
                        "mutationTypeID": 1,
                        "selectionCoeff": 0.0,
                        "subpopulationID": 1,
                        "originGeneration": 977,
                        "nucleotide": -1,
                    },
                    {
                        "mutationTypeID": 1,
                        "selectionCoeff": 0.0,
                        "subpopulationID": 1,
                        "originGeneration": 689,
                        "nucleotide": -1,
                    },
                    {
                        "mutationTypeID": 1,
                        "selectionCoeff": 0.0,
                        "subpopulationID": 1,
                        "originGeneration": 479,
                        "nucleotide": -1,
                    },
                    {
                        "mutationTypeID": 1,
                        "selectionCoeff": 0.0,
                        "subpopulationID": 1,
                        "originGeneration": 188,
                        "nucleotide": -1,
                    },
                ],
            ),
        ]:
            assert (
                metadata.MetadataSchema(schema).decode_row(example)[
                    "stacked_mutation_array"
                ]
                == expected
            )

    def test_population(self):
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {
                "subpopulationID": {
                    "type": "integer",
                    "binaryFormat": "i",
                    "index": 0,
                },
                "femaleCloneFraction": {
                    "type": "number",
                    "binaryFormat": "d",
                    "index": 1,
                },
                "maleCloneFraction": {
                    "type": "number",
                    "binaryFormat": "d",
                    "index": 2,
                },
                "sexRatio": {"type": "number", "binaryFormat": "d", "index": 3},
                "boundsX0": {"type": "number", "binaryFormat": "d", "index": 4},
                "boundsX1": {"type": "number", "binaryFormat": "d", "index": 5},
                "boundsY0": {"type": "number", "binaryFormat": "d", "index": 6},
                "boundsY1": {"type": "number", "binaryFormat": "d", "index": 7},
                "boundsZ0": {"type": "number", "binaryFormat": "d", "index": 8},
                "boundsZ1": {"type": "number", "binaryFormat": "d", "index": 9},
                "migrationRecCount": {
                    "type": "integer",
                    "binaryFormat": "d",
                    "index": 10,
                },
            },
        }
        example = (
            b"\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\xe0?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\xf0?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\xf0?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0"
            b"?\x00\x00\x00\x00"
        )
        expected = {
            "boundsX0": 0.5,
            "boundsX1": 0.0,
            "boundsY0": 1.0,
            "boundsY1": 0.0,
            "boundsZ0": 1.0,
            "boundsZ1": 0.0,
            "femaleCloneFraction": 0.0,
            "maleCloneFraction": 0.0,
            "migrationRecCount": 1.0,
            "sexRatio": 0.0,
            "subpopulationID": 1,
        }
        assert metadata.MetadataSchema(schema).decode_row(example) == expected


class TestTableCollectionEquality:
    def test_equality(self):
        ts = msprime.simulate(10, random_seed=42)
        tables = ts.dump_tables()
        tables2 = ts.dump_tables()
        schema = collections.OrderedDict(
            codec="json",
            title="Example Metadata",
            type="object",
            properties=collections.OrderedDict(
                one={"type": "string"}, two={"type": "number"}
            ),
            required=["one", "two"],
            additionalProperties=False,
        )
        schema2 = collections.OrderedDict(
            type="object",
            properties=collections.OrderedDict(
                two={"type": "number"}, one={"type": "string"}
            ),
            required=["one", "two"],
            additionalProperties=False,
            title="Example Metadata",
            codec="json",
        )
        tables.metadata_schema = metadata.MetadataSchema(schema)
        assert tables != tables2
        tables2.metadata_schema = metadata.MetadataSchema(schema2)
        tables.assert_equals(tables2)
        tables.metadata = collections.OrderedDict(one="tree", two=5)
        assert tables != tables2
        tables2.metadata = collections.OrderedDict(two=5, one="tree")
        tables.assert_equals(tables2)

    def test_fixing_uncanonical(self):
        ts = msprime.simulate(10, random_seed=42)
        tables = ts.dump_tables()
        schema = collections.OrderedDict(
            codec="json",
            title="Example Metadata",
            type="object",
            properties=collections.OrderedDict(
                one={"type": "string"}, two={"type": "number"}
            ),
            required=["one", "two"],
            additionalProperties=False,
        )
        # Set with low-level to emulate loading.
        tables._ll_tables.metadata_schema = json.dumps(schema)
        assert tables._ll_tables.metadata_schema != tskit.canonical_json(schema)
        tables.metadata_schema = tables.metadata_schema
        assert tables._ll_tables.metadata_schema == tskit.canonical_json(schema)


class TestStructuredArrays:
    """
    Tests for the get_numpy_dtype method in StructCodec
    """

    def test_not_implemented_json(self):
        schema = {"codec": "json"}
        with pytest.raises(NotImplementedError):
            metadata.MetadataSchema(schema).numpy_dtype()
        with pytest.raises(NotImplementedError):
            metadata.MetadataSchema(schema).structured_array_from_buffer(b"")

    @pytest.mark.parametrize(
        "type_name, format_code, numpy_type",
        [
            ("integer", "b", "<i1"),
            ("integer", "B", "u1"),
            ("integer", "h", "<i2"),
            ("integer", "H", "<u2"),
            ("integer", "i", "<i4"),
            ("integer", "I", "<u4"),
            ("integer", "q", "<i8"),
            ("integer", "Q", "<u8"),
            ("number", "f", "<f4"),
            ("number", "d", "<f8"),
            ("boolean", "?", "?"),
            ("string", "c", "S1"),
            ("string", "s", "S1"),
            ("string", "10s", "S10"),
            ("null", "x", "V1"),
            ("null", "5x", "V5"),
        ],
    )
    def test_types(self, type_name, format_code, numpy_type):
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {"value": {"type": type_name, "binaryFormat": format_code}},
        }

        schema = metadata.MetadataSchema(schema)
        dtype = schema.numpy_dtype()
        assert dtype.names == ("value",)
        assert dtype["value"] == np.dtype(numpy_type)

        test_arrays = {
            "integer": [{"value": i} for i in range(3)],
            "number": [{"value": i + 0.5} for i in range(3)],
            "boolean": [{"value": i} for i in [True, False, True, True]],
            "string": [
                {"value": str(i) * (1 if format_code in "cs" else 3)} for i in range(3)
            ],
            "null": [{"value": None}, {"value": None}, {"value": None}],
        }
        test_array = test_arrays[type_name]
        encoded = b"".join(schema.validate_and_encode_row(row) for row in test_array)
        struct_array = schema.structured_array_from_buffer(encoded)

        if "S" not in numpy_type and "V" not in numpy_type:
            assert np.array_equal(
                struct_array["value"], [i["value"] for i in test_array]
            )
        elif "S" in numpy_type:
            assert np.array_equal(
                struct_array["value"], [i["value"].encode() for i in test_array]
            )
        else:
            for val in struct_array["value"]:
                assert (
                    str(val) == "b'\\x00'"
                    if numpy_type == "V1"
                    else "b'\\x00\\x00\\x00\\x00\\x00'"
                )

    def test_object_with_multiple_fields(self):
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {
                "id": {"type": "integer", "binaryFormat": "i"},
                "name": {"type": "string", "binaryFormat": "10s"},
                "value": {"type": "number", "binaryFormat": "d"},
                "active": {"type": "boolean", "binaryFormat": "?"},
            },
        }

        schema = metadata.MetadataSchema(schema)
        dtype = schema.numpy_dtype()
        assert dtype.names == ("active", "id", "name", "value")  # Note reordering!
        assert dtype["id"] == np.dtype("<i4")
        assert dtype["name"] == np.dtype("S10")
        assert dtype["value"] == np.dtype("<f8")
        assert dtype["active"] == np.dtype("?")

        # Test array of objects with multiple fields
        test_array = [
            {"id": 1, "name": "test1", "value": 1.5, "active": True},
            {"id": 2, "name": "test2", "value": 2.5, "active": False},
            {"id": 3, "name": "test3", "value": 3.5, "active": True},
        ]
        encoded = b"".join(schema.validate_and_encode_row(row) for row in test_array)
        struct_array = schema.structured_array_from_buffer(encoded)

        assert np.array_equal(struct_array["id"], [1, 2, 3])
        assert np.array_equal(struct_array["value"], [1.5, 2.5, 3.5])
        assert np.array_equal(struct_array["active"], [True, False, True])
        assert np.array_equal(struct_array["name"], [b"test1", b"test2", b"test3"])

    def test_nested_objects(self):
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {
                "id": {"type": "integer", "binaryFormat": "i"},
                "nested": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number", "binaryFormat": "d"},
                        "y": {"type": "number", "binaryFormat": "d"},
                    },
                },
            },
        }

        schema = metadata.MetadataSchema(schema)
        dtype = schema.numpy_dtype()
        assert dtype.names == ("id", "nested")
        assert dtype["id"] == np.dtype("<i4")
        assert dtype["nested"].names == ("x", "y")
        assert dtype["nested"]["x"] == np.dtype("<f8")
        assert dtype["nested"]["y"] == np.dtype("<f8")

        # Test array of objects with nested objects
        test_array = [
            {"id": 1, "nested": {"x": 1.0, "y": 2.0}},
            {"id": 2, "nested": {"x": 3.0, "y": 4.0}},
            {"id": 3, "nested": {"x": 5.0, "y": 6.0}},
        ]
        encoded = b"".join(schema.validate_and_encode_row(row) for row in test_array)
        struct_array = schema.structured_array_from_buffer(encoded)

        assert np.array_equal(struct_array["id"], [1, 2, 3])
        assert np.array_equal(struct_array["nested"]["x"], [1.0, 3.0, 5.0])
        assert np.array_equal(struct_array["nested"]["y"], [2.0, 4.0, 6.0])

    def test_fixed_length_arrays(self):
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {
                "vector": {
                    "type": "array",
                    "length": 3,
                    "items": {"type": "number", "binaryFormat": "d"},
                },
                "matrix": {
                    "type": "array",
                    "length": 2,
                    "items": {
                        "type": "array",
                        "length": 2,
                        "items": {"type": "integer", "binaryFormat": "i"},
                    },
                },
            },
        }

        schema = metadata.MetadataSchema(schema)
        dtype = schema.numpy_dtype()
        assert dtype.names == ("matrix", "vector")  # Note reordering
        assert dtype["vector"].shape == (3,)
        assert dtype["vector"].base == np.dtype("<f8")
        assert dtype["matrix"].shape == (2,)
        assert dtype["matrix"].base == (np.dtype("<i4"), (2,))

        # Test array with fixed-length arrays
        test_array = [
            {"vector": [1.1, 2.2, 3.3], "matrix": [[1, 2], [3, 4]]},
            {"vector": [4.4, 5.5, 6.6], "matrix": [[5, 6], [7, 8]]},
            {"vector": [7.7, 8.8, 9.9], "matrix": [[9, 10], [11, 12]]},
        ]
        encoded = b"".join(schema.validate_and_encode_row(row) for row in test_array)
        struct_array = schema.structured_array_from_buffer(encoded)

        expected_vectors = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9]])
        assert np.allclose(struct_array["vector"], expected_vectors)

        expected_matrices = np.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]
        )
        assert np.array_equal(struct_array["matrix"], expected_matrices)

    def test_complex_nested_structure(self):
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {
                "id": {"type": "integer", "binaryFormat": "i"},
                "data": {
                    "type": "array",
                    "length": 2,
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "binaryFormat": "10s"},
                            "coords": {
                                "type": "array",
                                "length": 3,
                                "items": {"type": "number", "binaryFormat": "f"},
                            },
                        },
                    },
                },
            },
        }

        schema = metadata.MetadataSchema(schema)
        dtype = schema.numpy_dtype()
        assert dtype.names == ("data", "id")  # Note reordering
        assert dtype["id"] == np.dtype("<i4")
        assert dtype["data"].shape == (2,)
        assert dtype["data"].base.names == ("coords", "name")  # Note reordering
        assert dtype["data"].base["name"] == np.dtype("S10")
        assert dtype["data"].base["coords"].shape == (3,)
        assert dtype["data"].base["coords"].base == np.dtype("<f4")

        test_array = [
            {
                "id": 1,
                "data": [
                    {"name": "point1", "coords": [1.0, 2.0, 3.0]},
                    {"name": "point2", "coords": [4.0, 5.0, 6.0]},
                ],
            },
            {
                "id": 2,
                "data": [
                    {"name": "point3", "coords": [7.0, 8.0, 9.0]},
                    {"name": "point4", "coords": [10.0, 11.0, 12.0]},
                ],
            },
            {
                "id": 3,
                "data": [
                    {"name": "point5", "coords": [13.0, 14.0, 15.0]},
                    {"name": "point6", "coords": [16.0, 17.0, 18.0]},
                ],
            },
        ]
        encoded = b"".join(schema.validate_and_encode_row(row) for row in test_array)
        struct_array = schema.structured_array_from_buffer(encoded)

        assert np.array_equal(struct_array["id"], [1, 2, 3])

        expected_names = np.array(
            [[b"point1", b"point2"], [b"point3", b"point4"], [b"point5", b"point6"]]
        )
        assert np.array_equal(struct_array["data"]["name"], expected_names)

        expected_coords = np.array(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
            ]
        )
        assert np.allclose(struct_array["data"]["coords"], expected_coords)

    def test_unsupported_formats(self):
        # Pascal strings not supported
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {"pascal_string": {"type": "string", "binaryFormat": "10p"}},
        }

        with pytest.raises(ValueError, match="Pascal string format"):
            metadata.MetadataSchema(schema).numpy_dtype()

    def test_variable_length_arrays_not_supported(self):
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {
                "var_array": {
                    "type": "array",
                    "items": {"type": "integer", "binaryFormat": "i"},
                }
            },
        }

        with pytest.raises(ValueError, match="Only fixed-length arrays"):
            metadata.MetadataSchema(schema).numpy_dtype()

    def test_null_union_top_level_not_supported(self):
        schema = {
            "codec": "struct",
            "type": ["object", "null"],
            "properties": {
                "id": {"type": "integer", "binaryFormat": "i"},
                "name": {"type": "string", "binaryFormat": "10s"},
            },
        }

        with pytest.raises(
            ValueError, match="Top level object/null union not supported"
        ):
            metadata.MetadataSchema(schema).numpy_dtype()

    def test_explicit_ordering(self):
        schema = {
            "codec": "struct",
            "type": "object",
            "properties": {
                "id": {"type": "integer", "binaryFormat": "i", "index": 1},
                "name": {"type": "string", "binaryFormat": "10s", "index": 2},
                "age": {"type": "integer", "binaryFormat": "i", "index": 3},
            },
            "required": ["id", "name", "age"],
        }

        dtype = metadata.MetadataSchema(schema).numpy_dtype()
        assert dtype.names == ("id", "name", "age")
