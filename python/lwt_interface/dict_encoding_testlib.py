# MIT License
#
# Copyright (c) 2018-2022 Tskit Developers
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
Test definitions for the low-level LightweightTableCollection class
defined here. These tests are not intended to be executed directly,
but should be imported into another test module that imports a
compiled module exporting the LightweightTableCollection class.
See the test_example_c_module file for an example.
"""
import copy

import kastore
import msprime
import numpy as np
import pytest

import tskit
import tskit.util as util

lwt_module = None

NON_UTF8_STRING = "\ud861\udd37"


@pytest.fixture(scope="session")
def full_ts():
    """
    A tree sequence with data in all fields - duplicated from tskit's conftest.py
    as other test suites using this file will not have that fixture defined.
    """
    demography = msprime.Demography()
    demography.add_population(initial_size=100, name="A")
    demography.add_population(initial_size=100, name="B")
    demography.add_population(initial_size=100, name="C")
    demography.add_population_split(time=10, ancestral="C", derived=["A", "B"])

    ts = msprime.sim_ancestry(
        {"A": 5, "B": 5},
        demography=demography,
        random_seed=1,
        sequence_length=10,
        record_migrations=True,
    )
    assert ts.num_migrations > 0
    assert ts.num_individuals > 0
    ts = msprime.sim_mutations(ts, rate=0.1, random_seed=2)
    assert ts.num_mutations > 0
    tables = ts.dump_tables()
    tables.individuals.clear()

    for ind in ts.individuals():
        tables.individuals.add_row(flags=0, location=[ind.id, ind.id], parents=[-1, -1])

    for name, table in tables.table_name_map.items():
        if name != "provenances":
            table.metadata_schema = tskit.MetadataSchema({"codec": "json"})
            metadatas = [f"n_{name}_{u}" for u in range(len(table))]
            metadata, metadata_offset = tskit.pack_strings(metadatas)
            table.set_columns(
                **{
                    **table.asdict(),
                    "metadata": metadata,
                    "metadata_offset": metadata_offset,
                }
            )
    tables.metadata_schema = tskit.MetadataSchema({"codec": "json"})
    tables.metadata = {"A": "Test metadata"}

    tables.reference_sequence.data = "A" * int(tables.sequence_length)
    tables.reference_sequence.url = "https://example.com/sequence"
    tables.reference_sequence.metadata_schema = tskit.MetadataSchema.permissive_json()
    tables.reference_sequence.metadata = {"A": "Test metadata"}

    # Add some more provenance so we have enough rows for the offset deletion test.
    for j in range(10):
        tables.provenances.add_row(timestamp="x" * j, record="y" * j)
    return tables.tree_sequence()


# The ts above is used for the whole test session, but our tests need fresh tables to
# modify
@pytest.fixture
def tables(full_ts):
    return full_ts.dump_tables()


def test_check_ts_full(tmp_path, full_ts):
    """
    Check that the example ts has data in all fields
    """
    full_ts.dump(tmp_path / "tables")
    store = kastore.load(tmp_path / "tables")
    for v in store.values():
        assert v.nbytes > 0


class TestEncodingVersion:
    def test_version(self):
        lwt = lwt_module.LightweightTableCollection()
        assert lwt.asdict()["encoding_version"] == (1, 6)


class TestRoundTrip:
    """
    Tests if we can do a simple round trip on simulated data.
    """

    def verify(self, tables):
        lwt = lwt_module.LightweightTableCollection()
        lwt.fromdict(tables.asdict())
        other_tables = tskit.TableCollection.fromdict(lwt.asdict())
        tables.assert_equals(other_tables)

    def test_simple(self):
        ts = msprime.simulate(10, mutation_rate=1, random_seed=2)
        self.verify(ts.tables)

    def test_empty(self):
        tables = tskit.TableCollection(sequence_length=1)
        self.verify(tables)

    def test_individuals(self):
        n = 10
        ts = msprime.simulate(n, mutation_rate=1, random_seed=2)
        tables = ts.dump_tables()
        for j in range(n):
            tables.individuals.add_row(
                flags=j, location=(j, j), parents=(j, j), metadata=b"x" * j
            )
        self.verify(tables)

    def test_sequence_length(self):
        ts = msprime.simulate(
            10, recombination_rate=0.1, mutation_rate=1, length=0.99, random_seed=2
        )
        self.verify(ts.tables)

    def test_migration(self):
        pop_configs = [msprime.PopulationConfiguration(5) for _ in range(2)]
        migration_matrix = [[0, 1], [1, 0]]
        ts = msprime.simulate(
            population_configurations=pop_configs,
            migration_matrix=migration_matrix,
            mutation_rate=1,
            record_migrations=True,
            random_seed=1,
        )
        self.verify(ts.tables)

    def test_example(self, tables):
        tables.metadata_schema = tskit.MetadataSchema(
            {
                "codec": "struct",
                "type": "object",
                "properties": {"top-level": {"type": "string", "binaryFormat": "50p"}},
            }
        )
        tables.metadata = {"top-level": "top-level-metadata"}
        for table in tskit.TABLE_NAMES:
            t = getattr(tables, table)
            if hasattr(t, "metadata_schema"):
                t.packset_metadata([f"{table}-{i}".encode() for i in range(t.num_rows)])
                t.metadata_schema = tskit.MetadataSchema(
                    {
                        "codec": "struct",
                        "type": "object",
                        "properties": {
                            table: {"type": "string", "binaryFormat": "50p"}
                        },
                    }
                )

        self.verify(tables)


class TestMissingData:
    """
    Tests what happens when we have missing data in the encoded dict.
    """

    def test_missing_sequence_length(self, tables):
        d = tables.asdict()
        del d["sequence_length"]
        lwt = lwt_module.LightweightTableCollection()
        with pytest.raises(TypeError):
            lwt.fromdict(d)

    def test_missing_time_units(self, tables):
        assert tables.time_units != ""
        d = tables.asdict()
        del d["time_units"]
        lwt = lwt_module.LightweightTableCollection()
        lwt.fromdict(d)
        tables = tskit.TableCollection.fromdict(lwt.asdict())
        assert tables.time_units == tskit.TIME_UNITS_UNKNOWN

    def test_missing_metadata(self, tables):
        assert tables.metadata != b""
        d = tables.asdict()
        del d["metadata"]
        lwt = lwt_module.LightweightTableCollection()
        lwt.fromdict(d)
        tables = tskit.TableCollection.fromdict(lwt.asdict())
        # Empty byte field still gets interpreted by schema
        assert tables.metadata == {}

    def test_missing_metadata_schema(self, tables):
        assert repr(tables.metadata_schema) != ""
        d = tables.asdict()
        del d["metadata_schema"]
        lwt = lwt_module.LightweightTableCollection()
        lwt.fromdict(d)
        tables = tskit.TableCollection.fromdict(lwt.asdict())
        assert repr(tables.metadata_schema) == ""

    def test_missing_tables(self, tables):
        d = tables.asdict()
        table_names = d.keys() - {
            "sequence_length",
            "time_units",
            "metadata",
            "metadata_schema",
            "encoding_version",
            "indexes",
            "reference_sequence",
        }
        for table_name in table_names:
            d = tables.asdict()
            del d[table_name]
            lwt = lwt_module.LightweightTableCollection()
            with pytest.raises(TypeError):
                lwt.fromdict(d)


class TestBadTypes:
    """
    Tests for setting each column to a type that can't be converted to 1D numpy array.
    """

    def verify_columns(self, value, tables):
        d = tables.asdict()
        table_names = set(d.keys()) - {
            "sequence_length",
            "time_units",
            "metadata",
            "metadata_schema",
            "encoding_version",
            "indexes",
            "reference_sequence",
        }
        for table_name in table_names:
            table_dict = d[table_name]
            for colname in set(table_dict.keys()) - {"metadata_schema"}:
                d_copy = dict(table_dict)
                d_copy[colname] = value
                lwt = lwt_module.LightweightTableCollection()
                d = tables.asdict()
                d[table_name] = d_copy
                with pytest.raises(ValueError):
                    lwt.fromdict(d)

    def test_2d_array(self, tables):
        self.verify_columns([[1, 2], [3, 4]], tables)

    def test_str(self, tables):
        self.verify_columns("aserg", tables)

    def test_bad_top_level_types(self, tables):
        d = tables.asdict()
        for key in set(d.keys()) - {"encoding_version", "indexes"}:
            bad_type_dict = tables.asdict()
            # A list should be a ValueError for both the tables and sequence_length
            bad_type_dict[key] = ["12345"]
            lwt = lwt_module.LightweightTableCollection()
            with pytest.raises(TypeError):
                lwt.fromdict(bad_type_dict)


class TestBadLengths:
    """
    Tests for setting each column to a length incompatible with the table.
    """

    def verify(self, num_rows, tables):
        d = tables.asdict()
        table_names = set(d.keys()) - {
            "sequence_length",
            "time_units",
            "metadata",
            "metadata_schema",
            "encoding_version",
            "indexes",
            "reference_sequence",
        }
        for table_name in sorted(table_names):
            table_dict = d[table_name]
            for colname in set(table_dict.keys()) - {"metadata_schema"}:
                d_copy = dict(table_dict)
                d_copy[colname] = table_dict[colname][:num_rows].copy()
                lwt = lwt_module.LightweightTableCollection()
                d = tables.asdict()
                d[table_name] = d_copy
                with pytest.raises(ValueError):
                    lwt.fromdict(d)

    def test_two_rows(self, tables):
        self.verify(2, tables)

    def test_zero_rows(self, tables):
        self.verify(0, tables)

    def test_bad_index_length(self, tables):
        for col in ("insertion", "removal"):
            d = tables.asdict()
            d["indexes"][f"edge_{col}_order"] = d["indexes"][f"edge_{col}_order"][:-1]
            lwt = lwt_module.LightweightTableCollection()
            with pytest.raises(
                ValueError,
                match="^edge_insertion_order and"
                " edge_removal_order must be the same"
                " length$",
            ):
                lwt.fromdict(d)
        d = tables.asdict()
        for col in ("insertion", "removal"):
            d["indexes"][f"edge_{col}_order"] = d["indexes"][f"edge_{col}_order"][:-1]
        lwt = lwt_module.LightweightTableCollection()
        with pytest.raises(
            ValueError,
            match="^edge_insertion_order and edge_removal_order must be"
            " the same length as the number of edges$",
        ):
            lwt.fromdict(d)


class TestParsingUtilities:
    def test_missing_required(self, tables):
        d = tables.asdict()
        del d["sequence_length"]
        lwt = lwt_module.LightweightTableCollection()
        with pytest.raises(TypeError, match="'sequence_length' is required"):
            lwt.fromdict(d)

    def test_string_bad_type(self, tables):
        d = tables.asdict()
        d["time_units"] = b"sdf"
        lwt = lwt_module.LightweightTableCollection()
        with pytest.raises(TypeError, match="'time_units' is not a string"):
            lwt.fromdict(d)

    def test_bytes_bad_type(self, tables):
        d = tables.asdict()
        d["metadata"] = 1234
        lwt = lwt_module.LightweightTableCollection()
        with pytest.raises(TypeError, match="'metadata' is not bytes"):
            lwt.fromdict(d)

    def test_dict_bad_type(self, tables):
        d = tables.asdict()
        d["nodes"] = b"sdf"
        lwt = lwt_module.LightweightTableCollection()
        with pytest.raises(TypeError, match="'nodes' is not a dict"):
            lwt.fromdict(d)

    def test_bad_strings(self, tables):
        def verify_unicode_error(d):
            lwt = lwt_module.LightweightTableCollection()
            with pytest.raises(UnicodeEncodeError):
                lwt.fromdict(d)

        def verify_bad_string_type(d):
            lwt = lwt_module.LightweightTableCollection()
            with pytest.raises(TypeError):
                lwt.fromdict(d)

        d = tables.asdict()
        for k, v in d.items():
            if isinstance(v, str):
                d_copy = copy.deepcopy(d)
                d_copy[k] = NON_UTF8_STRING
                verify_unicode_error(d_copy)
                d_copy[k] = 12345
                verify_bad_string_type(d_copy)
            if isinstance(v, dict):
                for kp, vp in v.items():
                    if isinstance(vp, str):
                        d_copy = copy.deepcopy(d)
                        d_copy[k][kp] = NON_UTF8_STRING
                        verify_unicode_error(d_copy)
                        d_copy[k][kp] = 12345
                        verify_bad_string_type(d_copy)


class TestRequiredAndOptionalColumns:
    """
    Tests that specifying None for some columns will give the intended
    outcome.
    """

    def verify_required_columns(self, tables, table_name, required_cols):
        d = tables.asdict()
        table_dict = {col: None for col in d[table_name].keys()}
        for col in required_cols:
            table_dict[col] = d[table_name][col]
        lwt = lwt_module.LightweightTableCollection()
        d[table_name] = table_dict
        lwt.fromdict(d)
        other = lwt.asdict()
        for col in required_cols:
            assert np.array_equal(other[table_name][col], table_dict[col])

        # Any one of these required columns as None gives an error.
        for col in required_cols:
            d = tables.asdict()
            d_copy = copy.deepcopy(table_dict)
            d_copy[col] = None
            d[table_name] = d_copy
            lwt = lwt_module.LightweightTableCollection()
            with pytest.raises(TypeError):
                lwt.fromdict(d)

        # Removing any one of these required columns gives an error.
        for col in required_cols:
            d = tables.asdict()
            d_copy = copy.deepcopy(table_dict)
            del d_copy[col]
            d[table_name] = d_copy
            lwt = lwt_module.LightweightTableCollection()
            with pytest.raises(TypeError):
                lwt.fromdict(d)

    def verify_optional_column(self, tables, table_len, table_name, col_name):
        d = tables.asdict()
        table_dict = d[table_name]
        table_dict[col_name] = None
        lwt = lwt_module.LightweightTableCollection()
        lwt.fromdict(d)
        out = lwt.asdict()
        assert np.array_equal(
            out[table_name][col_name], np.zeros(table_len, dtype=np.int32) - 1
        )

    def verify_offset_pair(
        self, tables, table_len, table_name, col_name, required=False
    ):
        offset_col = col_name + "_offset"

        if not required:
            d = tables.asdict()
            table_dict = d[table_name]
            table_dict[col_name] = None
            table_dict[offset_col] = None
            lwt = lwt_module.LightweightTableCollection()
            lwt.fromdict(d)
            out = lwt.asdict()
            assert out[table_name][col_name].shape == (0,)
            assert np.array_equal(
                out[table_name][offset_col],
                np.zeros(table_len + 1, dtype=np.uint32),
            )
            d = tables.asdict()
            table_dict = d[table_name]
            del table_dict[col_name]
            del table_dict[offset_col]
            lwt = lwt_module.LightweightTableCollection()
            lwt.fromdict(d)
            out = lwt.asdict()
            assert out[table_name][col_name].shape == (0,)
            assert np.array_equal(
                out[table_name][offset_col],
                np.zeros(table_len + 1, dtype=np.uint32),
            )

        # Setting one or the other raises a TypeError
        d = tables.asdict()
        table_dict = d[table_name]
        table_dict[col_name] = None
        lwt = lwt_module.LightweightTableCollection()
        with pytest.raises(TypeError):
            lwt.fromdict(d)

        d = tables.asdict()
        table_dict = d[table_name]
        del table_dict[col_name]
        lwt = lwt_module.LightweightTableCollection()
        with pytest.raises(TypeError):
            lwt.fromdict(d)

        d = tables.asdict()
        table_dict = d[table_name]
        table_dict[offset_col] = None
        lwt = lwt_module.LightweightTableCollection()
        with pytest.raises(TypeError):
            lwt.fromdict(d)

        d = tables.asdict()
        table_dict = d[table_name]
        del table_dict[offset_col]
        lwt = lwt_module.LightweightTableCollection()
        with pytest.raises(TypeError):
            lwt.fromdict(d)

        d = tables.asdict()
        table_dict = d[table_name]
        bad_offset = np.zeros_like(table_dict[offset_col])
        bad_offset[:-1] = table_dict[offset_col][:-1][::-1]
        bad_offset[-1] = table_dict[offset_col][-1]
        table_dict[offset_col] = bad_offset
        lwt = lwt_module.LightweightTableCollection()
        with pytest.raises(ValueError):
            lwt.fromdict(d)

    def verify_metadata_schema(self, tables, table_name):
        d = tables.asdict()
        d[table_name]["metadata_schema"] = None
        lwt = lwt_module.LightweightTableCollection()
        lwt.fromdict(d)
        out = lwt.asdict()
        assert "metadata_schema" not in out[table_name]
        tables = tskit.TableCollection.fromdict(out)
        assert repr(getattr(tables, table_name).metadata_schema) == ""

    def test_individuals(self, tables):
        self.verify_required_columns(tables, "individuals", ["flags"])
        self.verify_offset_pair(
            tables, len(tables.individuals), "individuals", "location"
        )
        self.verify_offset_pair(
            tables, len(tables.individuals), "individuals", "parents"
        )
        self.verify_offset_pair(
            tables, len(tables.individuals), "individuals", "metadata"
        )
        self.verify_metadata_schema(tables, "individuals")
        # Verify optional parents column
        d = tables.asdict()
        d["individuals"]["parents"] = None
        d["individuals"]["parents_offset"] = None
        lwt = lwt_module.LightweightTableCollection()
        lwt.fromdict(d)
        out = lwt.asdict()
        assert all(val == [] for val in out["individuals"]["parents"])

    def test_nodes(self, tables):
        self.verify_offset_pair(tables, len(tables.nodes), "nodes", "metadata")
        self.verify_optional_column(tables, len(tables.nodes), "nodes", "population")
        self.verify_optional_column(tables, len(tables.nodes), "nodes", "individual")
        self.verify_required_columns(tables, "nodes", ["flags", "time"])
        self.verify_metadata_schema(tables, "nodes")

    def test_edges(self, tables):
        self.verify_required_columns(
            tables, "edges", ["left", "right", "parent", "child"]
        )
        self.verify_offset_pair(tables, len(tables.edges), "edges", "metadata")
        self.verify_metadata_schema(tables, "edges")

    def test_migrations(self, tables):
        self.verify_required_columns(
            tables, "migrations", ["left", "right", "node", "source", "dest", "time"]
        )
        self.verify_offset_pair(
            tables, len(tables.migrations), "migrations", "metadata"
        )
        self.verify_optional_column(tables, len(tables.nodes), "nodes", "individual")
        self.verify_metadata_schema(tables, "migrations")

    def test_sites(self, tables):
        self.verify_required_columns(
            tables, "sites", ["position", "ancestral_state", "ancestral_state_offset"]
        )
        self.verify_offset_pair(tables, len(tables.sites), "sites", "metadata")
        self.verify_metadata_schema(tables, "sites")

    def test_mutations(self, tables):
        self.verify_required_columns(
            tables,
            "mutations",
            ["site", "node", "derived_state", "derived_state_offset"],
        )
        self.verify_offset_pair(tables, len(tables.mutations), "mutations", "metadata")
        self.verify_metadata_schema(tables, "mutations")
        # Verify optional time column
        d = tables.asdict()
        d["mutations"]["time"] = None
        lwt = lwt_module.LightweightTableCollection()
        lwt.fromdict(d)
        out = lwt.asdict()
        assert all(util.is_unknown_time(val) for val in out["mutations"]["time"])

    def test_populations(self, tables):
        self.verify_required_columns(
            tables, "populations", ["metadata", "metadata_offset"]
        )
        self.verify_metadata_schema(tables, "populations")
        self.verify_offset_pair(tables, len(tables.nodes), "nodes", "metadata", True)

    def test_provenances(self, tables):
        self.verify_required_columns(
            tables,
            "provenances",
            ["record", "record_offset", "timestamp", "timestamp_offset"],
        )

    def test_index(self, tables):
        d = tables.asdict()
        lwt = lwt_module.LightweightTableCollection()
        lwt.fromdict(d)
        other = lwt.asdict()
        assert np.array_equal(
            d["indexes"]["edge_insertion_order"],
            other["indexes"]["edge_insertion_order"],
        )
        assert np.array_equal(
            d["indexes"]["edge_removal_order"], other["indexes"]["edge_removal_order"]
        )

        # index is optional
        d = tables.asdict()
        del d["indexes"]
        lwt = lwt_module.LightweightTableCollection()
        lwt.fromdict(d)
        # and a tc without indexes has empty dict
        assert lwt.asdict()["indexes"] == {}

        # Both columns must be provided, if one is
        for col in ("insertion", "removal"):
            d = tables.asdict()
            del d["indexes"][f"edge_{col}_order"]
            lwt = lwt_module.LightweightTableCollection()
            with pytest.raises(
                TypeError,
                match="^edge_insertion_order and "
                "edge_removal_order must be specified "
                "together$",
            ):
                lwt.fromdict(d)

    def test_index_bad_type(self, tables):
        d = tables.asdict()
        lwt = lwt_module.LightweightTableCollection()
        d["indexes"] = "asdf"
        with pytest.raises(TypeError):
            lwt.fromdict(d)

    def test_reference_sequence(self, tables):
        self.verify_metadata_schema(tables, "reference_sequence")

        def get_refseq(d):
            tables = tskit.TableCollection.fromdict(d)
            return tables.reference_sequence

        d = tables.asdict()
        refseq_dict = d.pop("reference_sequence")
        assert get_refseq(d).is_null()

        # All empty strings is the same thing
        d["reference_sequence"] = dict(
            data="", url="", metadata_schema="", metadata=b""
        )
        assert get_refseq(d).is_null()

        del refseq_dict["metadata_schema"]  # handled above
        for key, value in refseq_dict.items():
            d["reference_sequence"] = {key: value}
            refseq = get_refseq(d)
            assert not refseq.is_null()
            assert getattr(refseq, key) == value

    def test_top_level_time_units(self, tables):
        d = tables.asdict()
        # None should give default value
        d["time_units"] = None
        lwt = lwt_module.LightweightTableCollection()
        lwt.fromdict(d)
        out = lwt.asdict()
        tables = tskit.TableCollection.fromdict(out)
        assert tables.time_units == tskit.TIME_UNITS_UNKNOWN
        # Missing is tested in TestMissingData above
        d = tables.asdict()
        d["time_units"] = NON_UTF8_STRING
        lwt = lwt_module.LightweightTableCollection()
        with pytest.raises(UnicodeEncodeError):
            lwt.fromdict(d)

    def test_top_level_metadata(self, tables):
        d = tables.asdict()
        # None should give default value
        d["metadata"] = None
        lwt = lwt_module.LightweightTableCollection()
        lwt.fromdict(d)
        out = lwt.asdict()
        assert "metadata" not in out
        tables = tskit.TableCollection.fromdict(out)
        assert tables.metadata == {}
        # Missing is tested in TestMissingData above

    def test_top_level_metadata_schema(self, tables):
        d = tables.asdict()
        # None should give default value
        d["metadata_schema"] = None
        lwt = lwt_module.LightweightTableCollection()
        lwt.fromdict(d)
        out = lwt.asdict()
        assert "metadata_schema" not in out
        tables = tskit.TableCollection.fromdict(out)
        assert repr(tables.metadata_schema) == ""
        # Missing is tested in TestMissingData above


class TestLifecycle:
    def test_unassigned_empty(self):
        lwt_dict = lwt_module.LightweightTableCollection().asdict()
        assert tskit.TableCollection.fromdict(lwt_dict) == tskit.TableCollection(-1)

    def test_del_empty(self):
        lwt = lwt_module.LightweightTableCollection()
        lwt_dict = lwt.asdict()
        del lwt
        assert tskit.TableCollection.fromdict(lwt_dict) == tskit.TableCollection(-1)

    def test_del_full(self, tables):
        lwt = lwt_module.LightweightTableCollection()
        lwt.fromdict(tables.asdict())
        lwt_dict = lwt.asdict()
        del lwt
        assert tskit.TableCollection.fromdict(lwt_dict) == tables

    def test_del_lwt_and_tables(self, tables):
        lwt = lwt_module.LightweightTableCollection()
        lwt.fromdict(tables.asdict())
        lwt_dict = lwt.asdict()
        del lwt
        tables2 = tables.copy()
        del tables
        assert tskit.TableCollection.fromdict(lwt_dict) == tables2


class TestForceOffset64:
    def get_offset_columns(self, dict_encoding):
        for table_name, table in dict_encoding.items():
            if isinstance(table, dict):
                for name, array in table.items():
                    if name.endswith("_offset"):
                        yield f"{table_name}/{name}", array

    def test_bad_args(self, tables):
        lwt = lwt_module.LightweightTableCollection()
        lwt.fromdict(tables.asdict())
        for bad_type in [None, {}, "sdf"]:
            with pytest.raises(TypeError):
                lwt.asdict(bad_type)

    def test_off_by_default(self, tables):
        lwt = lwt_module.LightweightTableCollection()
        lwt.fromdict(tables.asdict())
        d = lwt.asdict()
        for _, array in self.get_offset_columns(d):
            assert array.dtype == np.uint32

    def test_types_64(self, tables):
        lwt = lwt_module.LightweightTableCollection()
        lwt.fromdict(tables.asdict())
        d = lwt.asdict(force_offset_64=True)
        for _, array in self.get_offset_columns(d):
            assert array.dtype == np.uint64

    def test_types_32(self, tables):
        lwt = lwt_module.LightweightTableCollection()
        lwt.fromdict(tables.asdict())
        d = lwt.asdict(force_offset_64=False)
        for _, array in self.get_offset_columns(d):
            assert array.dtype == np.uint32

    def test_values_equal(self, tables):
        lwt = lwt_module.LightweightTableCollection()
        lwt.fromdict(tables.asdict())
        d64 = lwt.asdict(force_offset_64=True)
        d32 = lwt.asdict(force_offset_64=False)
        offsets_64 = dict(self.get_offset_columns(d64))
        offsets_32 = dict(self.get_offset_columns(d32))
        for col_name, col_32 in offsets_32.items():
            col_64 = offsets_64[col_name]
            assert col_64.shape == col_32.shape
            assert np.all(col_64 == col_32)


@pytest.mark.parametrize("bad_type", [None, "", []])
def test_fromdict_bad_type(bad_type):
    lwt = lwt_module.LightweightTableCollection()
    with pytest.raises(TypeError):
        lwt.fromdict(bad_type)
