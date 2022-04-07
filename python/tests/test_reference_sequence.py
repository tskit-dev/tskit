# MIT License
#
# Copyright (c) 2021-2022 Tskit Developers
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
Tests for reference sequence support.
"""
import pytest

import tskit


class TestTablesProperties:
    def test_initially_not_set(self):
        tables = tskit.TableCollection(1)
        assert not tables.has_reference_sequence()
        tables.reference_sequence.data = "ABCDEF"
        assert tables.reference_sequence.data == "ABCDEF"
        assert tables.has_reference_sequence()

    def test_does_not_have_reference_sequence_if_empty(self):
        tables = tskit.TableCollection(1)
        assert not tables.has_reference_sequence()
        tables.reference_sequence.data = ""
        assert not tables.has_reference_sequence()

    def test_same_object(self):
        tables = tskit.TableCollection(1)
        refseq = tables.reference_sequence
        tables.reference_sequence.data = "asdf"
        assert refseq.data == "asdf"
        # Not clear we want to do this, but keeping the same pattern as the
        # tables for now.
        assert tables.reference_sequence is not refseq

    def test_clear(self, ts_fixture):
        tables = ts_fixture.dump_tables()
        tables.reference_sequence.clear()
        assert not tables.has_reference_sequence()

    def test_write_object_fails_bad_type(self):
        tables = tskit.TableCollection(1)
        with pytest.raises(AttributeError):
            tables.reference_sequence = None

    def test_write_object(self, ts_fixture):
        tables = tskit.TableCollection(1)
        tables.reference_sequence = ts_fixture.reference_sequence
        tables.reference_sequence.assert_equals(ts_fixture.reference_sequence)

    def test_asdict_no_reference(self):
        tables = tskit.TableCollection(1)
        d = tables.asdict()
        assert "reference_sequence" not in d

    def test_asdict_reference_no_metadata(self):
        tables = tskit.TableCollection(1)
        tables.reference_sequence.data = "ABCDEF"
        d = tables.asdict()["reference_sequence"]
        assert d["data"] == "ABCDEF"
        assert d["url"] == ""
        assert "metadata" not in d
        assert "metadata_schema" not in d

    def test_asdict_reference_metadata(self):
        tables = tskit.TableCollection(1)
        tables.reference_sequence.metadata_schema = (
            tskit.MetadataSchema.permissive_json()
        )
        tables.reference_sequence.metadata = {"a": "ABCDEF"}
        d = tables.asdict()["reference_sequence"]
        assert d["data"] == ""
        assert d["url"] == ""
        assert d["metadata_schema"] == '{"codec":"json"}'
        assert d["metadata"] == b'{"a":"ABCDEF"}'

    def test_fromdict_reference_data(self):
        d = tskit.TableCollection(1).asdict()
        d["reference_sequence"] = {"data": "XYZ"}
        tables = tskit.TableCollection.fromdict(d)
        assert tables.has_reference_sequence()
        assert tables.reference_sequence.data == "XYZ"
        assert tables.reference_sequence.url == ""
        assert repr(tables.reference_sequence.metadata_schema) == ""
        assert tables.reference_sequence.metadata == b""

    def test_fromdict_reference_url(self):
        d = tskit.TableCollection(1).asdict()
        d["reference_sequence"] = {"url": "file://file.fasta"}
        tables = tskit.TableCollection.fromdict(d)
        assert tables.has_reference_sequence()
        assert tables.reference_sequence.data == ""
        assert tables.reference_sequence.url == "file://file.fasta"
        assert repr(tables.reference_sequence.metadata_schema) == ""
        assert tables.reference_sequence.metadata == b""

    def test_fromdict_reference_metadata(self):
        tables = tskit.TableCollection(1)
        tables.reference_sequence.metadata_schema = (
            tskit.MetadataSchema.permissive_json()
        )
        tables.reference_sequence.metadata = {"a": "ABCDEF"}
        tables = tskit.TableCollection.fromdict(tables.asdict())
        assert tables.has_reference_sequence()
        assert tables.reference_sequence.data == ""
        assert (
            tables.reference_sequence.metadata_schema
            == tskit.MetadataSchema.permissive_json()
        )
        assert tables.reference_sequence.metadata == {"a": "ABCDEF"}

    def test_fromdict_no_reference(self):
        d = tskit.TableCollection(1).asdict()
        tables = tskit.TableCollection.fromdict(d)
        assert not tables.has_reference_sequence()

    def test_fromdict_all_values_empty(self):
        d = tskit.TableCollection(1).asdict()
        d["reference_sequence"] = dict(
            data="", url="", metadata_schema="", metadata=b""
        )
        tables = tskit.TableCollection.fromdict(d)
        assert not tables.has_reference_sequence()


class TestSummaries:
    def test_repr(self):
        tables = tskit.TableCollection(1)
        refseq = tables.reference_sequence
        # TODO add better tests when summaries are updated
        assert repr(refseq).startswith("ReferenceSequence")


class TestEquals:
    def test_equal_self(self, ts_fixture):
        ts_fixture.reference_sequence.assert_equals(ts_fixture.reference_sequence)
        assert ts_fixture.reference_sequence == ts_fixture.reference_sequence
        assert not ts_fixture.reference_sequence != ts_fixture.reference_sequence
        assert ts_fixture.reference_sequence.equals(ts_fixture.reference_sequence)

    def test_equal_empty(self):
        tables = tskit.TableCollection(1)
        tables.reference_sequence.assert_equals(tables.reference_sequence)
        assert tables.reference_sequence == tables.reference_sequence
        assert tables.reference_sequence.equals(tables.reference_sequence)

    @pytest.mark.parametrize("attr", ["url", "data"])
    def test_unequal_attr_missing(self, ts_fixture, attr):
        t1 = ts_fixture.tables
        d = t1.asdict()
        del d["reference_sequence"][attr]
        t2 = tskit.TableCollection.fromdict(d)
        with pytest.raises(AssertionError, match=attr):
            t1.reference_sequence.assert_equals(t2.reference_sequence)
        assert t1.reference_sequence != t2.reference_sequence
        assert not t1.reference_sequence.equals(t2.reference_sequence)
        with pytest.raises(AssertionError, match=attr):
            t2.reference_sequence.assert_equals(t1.reference_sequence)
        assert t2.reference_sequence != t1.reference_sequence
        assert not t2.reference_sequence.equals(t1.reference_sequence)

    @pytest.mark.parametrize(
        ("attr", "val"),
        [
            ("url", "foo"),
            ("data", "bar"),
            ("metadata", {"json": "runs the world"}),
            ("metadata_schema", tskit.MetadataSchema(None)),
        ],
    )
    def test_different_not_equal(self, ts_fixture, attr, val):
        t1 = ts_fixture.dump_tables()
        t2 = t1.copy()
        setattr(t1.reference_sequence, attr, val)

        with pytest.raises(AssertionError):
            t1.reference_sequence.assert_equals(t2.reference_sequence)
        assert t1.reference_sequence != t2.reference_sequence
        assert not t1.reference_sequence.equals(t2.reference_sequence)
        with pytest.raises(AssertionError):
            t2.reference_sequence.assert_equals(t1.reference_sequence)
        assert t2.reference_sequence != t1.reference_sequence
        assert not t2.reference_sequence.equals(t1.reference_sequence)

    @pytest.mark.parametrize(
        ("attr", "val"),
        [
            ("metadata", {"json": "runs the world"}),
            ("metadata_schema", tskit.MetadataSchema(None)),
        ],
    )
    def test_different_but_ignore(self, ts_fixture, attr, val):
        t1 = ts_fixture.dump_tables()
        t2 = t1.copy()
        setattr(t1.reference_sequence, attr, val)

        with pytest.raises(AssertionError):
            t1.reference_sequence.assert_equals(t2.reference_sequence)
        assert t1.reference_sequence != t2.reference_sequence
        assert not t1.reference_sequence.equals(t2.reference_sequence)
        with pytest.raises(AssertionError):
            t2.reference_sequence.assert_equals(t1.reference_sequence)
        assert t2.reference_sequence != t1.reference_sequence
        assert not t2.reference_sequence.equals(t1.reference_sequence)

        t2.reference_sequence.assert_equals(t1.reference_sequence, ignore_metadata=True)
        assert t2.reference_sequence.equals(t1.reference_sequence, ignore_metadata=True)


class TestTreeSequenceProperties:
    @pytest.mark.parametrize("data", ["abcd", "🎄🌳🌴"])
    def test_data_inherited_from_tables(self, data):
        tables = tskit.TableCollection(1)
        tables.reference_sequence.data = data
        ts = tables.tree_sequence()
        assert ts.reference_sequence.data == data
        assert ts.has_reference_sequence()

    @pytest.mark.parametrize("url", ["http://xyx.z", "file://"])
    def test_url_inherited_from_tables(self, url):
        tables = tskit.TableCollection(1)
        tables.reference_sequence.url = url
        ts = tables.tree_sequence()
        assert ts.reference_sequence.url == url
        assert ts.has_reference_sequence()

    def test_no_reference_sequence(self):
        tables = tskit.TableCollection(1)
        ts = tables.tree_sequence()
        assert not ts.has_reference_sequence()
        assert ts.reference_sequence is None

    def test_write_data_fails(self):
        tables = tskit.TableCollection(1)
        tables.reference_sequence.data = "abc"
        ts = tables.tree_sequence()
        with pytest.raises(AttributeError, match="read-only"):
            ts.reference_sequence.data = "xyz"

    def test_write_url_fails(self):
        tables = tskit.TableCollection(1)
        tables.reference_sequence.data = "abc"
        ts = tables.tree_sequence()
        with pytest.raises(AttributeError, match="read-only"):
            ts.reference_sequence.url = "xyz"

    def test_write_metadata_fails(self):
        tables = tskit.TableCollection(1)
        tables.reference_sequence.data = "abc"
        ts = tables.tree_sequence()
        with pytest.raises(AttributeError, match="read-only"):
            # NOTE: it can be slightly confusing here because we try to encode
            # first, and so we don't get an AttributeError for all inputs.
            ts.reference_sequence.metadata = b"xyz"

    def test_write_metadata_schema_fails(self):
        tables = tskit.TableCollection(1)
        tables.reference_sequence.data = "abc"
        ts = tables.tree_sequence()
        with pytest.raises(AttributeError, match="read-only"):
            ts.reference_sequence.metadata_schema = (
                tskit.MetadataSchema.permissive_json()
            )

    def test_write_object_fails(self, ts_fixture):
        tables = tskit.TableCollection(1)
        ts = tables.tree_sequence()
        with pytest.raises(AttributeError):
            ts.reference_sequence = ts_fixture.reference_sequence
