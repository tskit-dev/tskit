# MIT License
#
# Copyright (c) 2022 Tskit Developers
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
Test cases for table transformation operations like trim(), decapitate, etc.
"""
import math

import numpy as np
import pytest

import tests
import tskit
from tests.test_highlevel import get_example_tree_sequences

# ↑ See https://github.com/tskit-dev/tskit/issues/1804 for when
# we can remove this.


def split_edges_definition(ts, time, *, flags=None, population=None, metadata=None):
    tables = ts.dump_tables()
    if ts.num_migrations > 0:
        raise ValueError("Migrations not supported")
    default_population = population is None
    if not default_population:
        # -1 is a valid value
        if population < -1 or population >= ts.num_populations:
            raise ValueError("Population out of bounds")
    flags = 0 if flags is None else flags
    if metadata is None:
        metadata = tables.nodes.metadata_schema.empty_value
    metadata = tables.nodes.metadata_schema.validate_and_encode_row(metadata)
    # This is the easiest way to turn off encoding when calling add_row below
    schema = tables.nodes.metadata_schema
    tables.nodes.metadata_schema = tskit.MetadataSchema(None)

    node_time = tables.nodes.time
    node_population = tables.nodes.population
    tables.edges.clear()
    split_edge = np.full(ts.num_edges, tskit.NULL, dtype=int)
    for edge in ts.edges():
        if node_time[edge.child] < time < node_time[edge.parent]:
            if default_population:
                population = node_population[edge.child]
            u = tables.nodes.add_row(
                flags=flags, time=time, population=population, metadata=metadata
            )
            tables.edges.append(edge.replace(parent=u))
            tables.edges.append(edge.replace(child=u))
            split_edge[edge.id] = u
        else:
            tables.edges.append(edge)
    # Reinstate schema
    tables.nodes.metadata_schema = schema

    tables.mutations.clear()
    for mutation in ts.mutations():
        mapped_node = tskit.NULL
        if mutation.edge != tskit.NULL:
            mapped_node = split_edge[mutation.edge]
        if mapped_node != tskit.NULL and mutation.time >= time:
            mutation = mutation.replace(node=mapped_node)
        tables.mutations.append(mutation)

    tables.sort()
    return tables.tree_sequence()


class TestSplitEdgesSimpleTree:

    # 2.00┊   4   ┊
    #     ┊ ┏━┻┓  ┊
    # 1.00┊ ┃  3  ┊
    #     ┊ ┃ ┏┻┓ ┊
    # 0.00┊ 0 1 2 ┊
    #     0       1
    @tests.cached_example
    def ts(self):
        return tskit.Tree.generate_balanced(3, branch_length=1).tree_sequence

    @pytest.mark.parametrize("time", [0.1, 0.5, 0.9])
    def test_lowest_branches(self, time):
        # 2.00┊   4   ┊    2.00┊   4   ┊
        #     ┊ ┏━┻┓  ┊        ┊ ┏━┻┓  ┊
        # 1.00┊ ┃  3  ┊    1.00┊ ┃  3  ┊
        #     ┊ ┃ ┏┻┓ ┊        ┊ ┃ ┏┻┓ ┊
        #     ┊ ┃ ┃ ┃ ┊      t ┊ 7 5 6 ┊
        #     ┊ ┃ ┃ ┃ ┊ ->     ┊ ┃ ┃ ┃ ┊
        # 0.00┊ 0 1 2 ┊    0.00┊ 0 1 2 ┊
        #     0       1        0       1
        before_ts = self.ts()
        ts = before_ts.split_edges(time)
        assert ts.num_nodes == 8
        assert all(ts.node(u).time == time for u in [5, 6, 7])
        assert ts.num_trees == 1
        assert ts.first().parent_dict == {0: 7, 1: 5, 2: 6, 5: 3, 6: 3, 7: 4, 3: 4}
        ts = ts.simplify()
        ts.tables.assert_equals(before_ts.tables, ignore_provenance=True)

    def test_same_time_as_node(self):
        # 2.00┊   4   ┊    2.00┊   4   ┊
        #     ┊ ┏━┻┓  ┊        ┊ ┏━┻┓  ┊
        # 1.00┊ ┃  3  ┊    1.00┊ 5  3  ┊
        #     ┊ ┃ ┏┻┓ ┊        ┊ ┃ ┏┻┓ ┊
        # 0.00┊ 0 1 2 ┊    0.00┊ 0 1 2 ┊
        #     0       1        0       1
        before_ts = self.ts()
        ts = before_ts.split_edges(1)
        assert ts.num_nodes == 6
        assert ts.node(5).time == 1
        assert ts.num_trees == 1
        assert ts.first().parent_dict == {0: 5, 1: 3, 2: 3, 5: 4, 3: 4}
        ts = ts.simplify()
        ts.tables.assert_equals(before_ts.tables, ignore_provenance=True)

    @pytest.mark.parametrize("time", [1.1, 1.5, 1.9])
    def test_top_branches(self, time):
        # 2.00┊   4   ┊    2.00┊   4   ┊
        #     ┊ ┏━┻┓  ┊        ┊ ┏━┻┓  ┊
        #     ┊ ┃  ┃  ┊      t ┊ 5  6  ┊
        #     ┊ ┃  ┃  ┊ ->     ┊ ┃  ┃  ┊
        # 1.00┊ ┃  3  ┊    1.00┊ ┃  3  ┊
        #     ┊ ┃ ┏┻┓ ┊        ┊ ┃ ┏┻┓ ┊
        # 0.00┊ 0 1 2 ┊    0.00┊ 0 1 2 ┊
        #     0       1        0       1

        before_ts = self.ts()
        ts = before_ts.split_edges(time)
        assert ts.num_nodes == 7
        assert all(ts.node(u).time == time for u in [5, 6])
        assert ts.num_trees == 1
        assert ts.first().parent_dict == {0: 5, 1: 3, 2: 3, 3: 6, 6: 4, 5: 4}
        ts = ts.simplify()
        ts.tables.assert_equals(before_ts.tables, ignore_provenance=True)

    @pytest.mark.parametrize("time", [0, 2])
    def test_at_leaf_or_root_time(self, time):
        split = self.ts().split_edges(time)
        split.tables.assert_equals(self.ts().tables, ignore_provenance=True)

    @pytest.mark.parametrize("time", [-1, 2.1])
    def test_outside_time_scales(self, time):
        split = self.ts().split_edges(time)
        split.tables.assert_equals(self.ts().tables, ignore_provenance=True)


class TestSplitEdgesSimpleTreeMutationExamples:
    def test_single_mutation_no_time(self):
        # 2.00┊   4   ┊
        #     ┊ ┏━┻┓  ┊
        # 1.00┊ ┃  3  ┊
        #     ┊ x ┏┻┓ ┊
        # 0.00┊ 0 1 2 ┊
        #     0       1
        tree = tskit.Tree.generate_balanced(3, branch_length=1)
        tables = tree.tree_sequence.dump_tables()
        tables.sites.add_row(0, "A")
        tables.mutations.add_row(site=0, node=0, derived_state="T", metadata=b"1234")
        ts = tables.tree_sequence()

        ts_split = ts.split_edges(1)
        # 2.00┊   4   ┊
        #     ┊ ┏━┻┓  ┊
        # 1.00┊ 5  3  ┊
        #     ┊ x ┏┻┓ ┊
        # 0.00┊ 0 1 2 ┊
        #     0       1
        assert ts_split.num_nodes == 6
        mut = ts_split.mutation(0)
        assert mut.node == 0
        assert mut.derived_state == "T"
        assert mut.metadata == b"1234"
        assert tskit.is_unknown_time(mut.time)

    def test_single_mutation_split_before_time(self):
        # 2.00┊   4   ┊
        #     ┊ x━┻┓  ┊
        # 1.00┊ ┃  3  ┊
        #     ┊ ┃ ┏┻┓ ┊
        # 0.00┊ 0 1 2 ┊
        #     0       1
        tree = tskit.Tree.generate_balanced(3, branch_length=1)
        tables = tree.tree_sequence.dump_tables()
        tables.sites.add_row(0, "A")
        tables.mutations.add_row(
            site=0, node=0, time=1.5, derived_state="T", metadata=b"1234"
        )
        ts = tables.tree_sequence()

        ts_split = ts.split_edges(1)
        # 2.00┊   4   ┊
        #     ┊ x━┻┓  ┊
        # 1.00┊ 5  3  ┊
        #     ┊ ┃ ┏┻┓ ┊
        # 0.00┊ 0 1 2 ┊
        #     0       1
        assert ts_split.num_nodes == 6
        mut = ts_split.mutation(0)
        assert mut.node == 5
        assert mut.derived_state == "T"
        assert mut.metadata == b"1234"
        assert mut.time == 1.5

    def test_single_mutation_split_at_time(self):
        # 2.00┊   4   ┊
        #     ┊ ┏━┻┓  ┊
        # 1.00┊ x  3  ┊
        #     ┊ ┃ ┏┻┓ ┊
        # 0.00┊ 0 1 2 ┊
        #     0       1
        tree = tskit.Tree.generate_balanced(3, branch_length=1)
        tables = tree.tree_sequence.dump_tables()
        tables.sites.add_row(0, "A")
        tables.mutations.add_row(
            site=0, node=0, time=1, derived_state="T", metadata=b"1234"
        )
        ts = tables.tree_sequence()

        ts_split = ts.split_edges(1)
        # 2.00┊   4   ┊
        #     ┊ ┏━┻┓  ┊
        # 1.00┊ 5x 3  ┊
        #     ┊ ┃ ┏┻┓ ┊
        # 0.00┊ 0 1 2 ┊
        #     0       1
        mut = ts_split.mutation(0)
        assert mut.node == 5
        assert mut.derived_state == "T"
        assert mut.metadata == b"1234"
        assert mut.time == 1.0

    def test_multi_mutation_no_time(self):
        # 2.00┊   4   ┊
        #     ┊ ┏━┻┓  ┊
        # 1.00┊ x  3  ┊
        #     ┊ x ┏┻┓ ┊
        # 0.00┊ 0 1 2 ┊
        #     0       1
        tree = tskit.Tree.generate_balanced(3, branch_length=1)
        tables = tree.tree_sequence.dump_tables()
        tables.sites.add_row(0, "A")
        tables.mutations.add_row(site=0, node=0, derived_state="T")
        tables.mutations.add_row(site=0, node=0, parent=0, derived_state="G")
        ts = tables.tree_sequence()

        ts_split = ts.split_edges(1)
        # 2.00┊   4   ┊
        #     ┊ ┏━┻┓  ┊
        #     ┊ 5  3  ┊
        #     ┊ x  ┃  ┊
        #     ┊ x ┏┻┓ ┊
        # 0.00┊ 0 1 2 ┊
        #     0       1
        ts_split.tables.mutations.assert_equals(tables.mutations)

    def test_multi_mutation_over_sample_time(self):
        # 2.00┊   4   ┊
        #     ┊ x━┻┓  ┊
        # 1.00┊ ┃  3  ┊
        #     ┊ x ┏┻┓ ┊
        # 0.00┊ 0 1 2 ┊
        #     0       1
        tree = tskit.Tree.generate_balanced(3, branch_length=1)
        tables = tree.tree_sequence.dump_tables()
        tables.sites.add_row(0, "A")
        tables.mutations.add_row(site=0, node=0, time=1.01, derived_state="T")
        tables.mutations.add_row(site=0, node=0, time=0.99, parent=0, derived_state="G")
        ts = tables.tree_sequence()

        ts_split = ts.split_edges(1)
        # 2.00┊   4   ┊
        #     ┊ x━┻┓  ┊
        # 1.00┊ 5  3  ┊
        #     ┊ x ┏┻┓ ┊
        # 0.00┊ 0 1 2 ┊
        #     0       1
        assert ts_split.num_mutations == 2

        mut = ts_split.mutation(0)
        assert mut.site == 0
        assert mut.node == 5
        assert mut.time == 1.01
        mut = ts_split.mutation(1)
        assert mut.site == 0
        assert mut.node == 0
        assert mut.time == 0.99

    def test_mutation_not_on_branch(self):
        tables = tskit.TableCollection(1)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.sites.add_row(0, "A")
        tables.mutations.add_row(site=0, node=0, derived_state="T")
        ts = tables.tree_sequence()
        tables.assert_equals(ts.split_edges(0).tables, ignore_provenance=True)


class TestSplitEdgesExamples:
    @pytest.mark.parametrize("ts", get_example_tree_sequences())
    def test_genotypes_round_trip(self, ts):
        time = 0 if ts.num_nodes == 0 else np.median(ts.tables.nodes.time)
        if ts.num_migrations == 0:
            split_ts = ts.split_edges(time)
            assert np.array_equal(split_ts.genotype_matrix(), ts.genotype_matrix())
        else:
            with pytest.raises(tskit.LibraryError):
                ts.split_edges(time)

    @pytest.mark.parametrize("ts", get_example_tree_sequences())
    @pytest.mark.parametrize("population", [-1, None])
    def test_definition(self, ts, population):
        time = 0 if ts.num_nodes == 0 else np.median(ts.tables.nodes.time)
        if ts.num_migrations == 0:
            ts1 = split_edges_definition(ts, time, population=population)
            ts2 = ts.split_edges(time, population=population)
            ts1.tables.assert_equals(ts2.tables, ignore_provenance=True)


class TestSplitEdgesInterface:
    def test_migrations_fail(self, ts_fixture):
        assert ts_fixture.num_migrations > 0
        with pytest.raises(tskit.LibraryError, match="MIGRATIONS_NOT_SUPPORTED"):
            ts_fixture.split_edges(0)

    def test_population_out_of_bounds(self):
        tables = tskit.TableCollection(1)
        ts = tables.tree_sequence()
        with pytest.raises(tskit.LibraryError, match="POPULATION_OUT_OF_BOUNDS"):
            ts.split_edges(0, population=0)

    def test_bad_flags(self):
        ts = tskit.TableCollection(1).tree_sequence()
        with pytest.raises(TypeError):
            ts.split_edges(0, flags="asdf")

    def test_bad_metadata_no_schema(self):
        ts = tskit.TableCollection(1).tree_sequence()
        with pytest.raises(TypeError):
            ts.split_edges(0, metadata="asdf")

    def test_bad_metadata_json_schema(self):
        tables = tskit.TableCollection(1)
        tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
        ts = tables.tree_sequence()
        with pytest.raises(tskit.MetadataEncodingError):
            ts.split_edges(0, metadata=b"bytes")

    @pytest.mark.parametrize("time", [math.inf, np.inf, tskit.UNKNOWN_TIME, np.nan])
    def test_nonfinite_time(self, time):
        tables = tskit.TableCollection(1)
        ts = tables.tree_sequence()
        with pytest.raises(tskit.LibraryError, match="TIME_NONFINITE"):
            ts.split_edges(time)


class TestSplitEdgesNodeValues:
    @tests.cached_example
    def ts(self):
        tables = tskit.TableCollection(1)
        for _ in range(5):
            tables.populations.add_row()
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, population=0, time=0)
        tables.nodes.add_row(time=1)
        tables.edges.add_row(0, 1, 1, 0)
        return tables.tree_sequence()

    @tests.cached_example
    def ts_with_schema(self):
        tables = tskit.TableCollection(1)
        for _ in range(5):
            tables.populations.add_row()
        tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, population=0, time=0)
        tables.nodes.add_row(time=1)
        tables.edges.add_row(0, 1, 1, 0)
        return tables.tree_sequence()

    def test_default_population(self):
        ts = self.ts().split_edges(0.5)
        assert ts.node(2).population == 0

    @pytest.mark.parametrize("population", range(-1, 5))
    def test_specify_population(self, population):
        ts = self.ts().split_edges(0.5, population=population)
        assert ts.node(2).population == population

    def test_default_flags(self):
        ts = self.ts().split_edges(0.5)
        assert ts.node(2).flags == 0

    @pytest.mark.parametrize("flags", range(0, 5))
    def test_specify_flags(self, flags):
        ts = self.ts().split_edges(0.5, flags=flags)
        assert ts.node(2).flags == flags

    def test_default_metadata_no_schema(self):
        ts = self.ts().split_edges(0.5)
        assert ts.node(2).metadata == b""

    @pytest.mark.parametrize("metadata", [b"", b"some bytes"])
    def test_specify_metadata_no_schema(self, metadata):
        ts = self.ts().split_edges(0.5, metadata=metadata)
        assert ts.node(2).metadata == metadata

    def test_default_metadata_with_schema(self):
        ts = self.ts_with_schema().split_edges(0.5)
        assert ts.node(2).metadata == {}

    @pytest.mark.parametrize("metadata", [{}, {"some": "json"}])
    def test_specify_metadata_with_schema(self, metadata):
        ts = self.ts_with_schema().split_edges(0.5, metadata=metadata)
        assert ts.node(2).metadata == metadata
