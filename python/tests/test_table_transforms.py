# MIT License
#
# Copyright (c) 2022-2024 Tskit Developers
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
import decimal
import fractions
import io
import math

import numpy as np
import pytest

import tests
import tskit
import tskit.util as util
from tests.test_highlevel import get_example_tree_sequences

# ↑ See https://github.com/tskit-dev/tskit/issues/1804 for when
# we can remove this.


def delete_older_definition(tables, time):
    node_time = tables.nodes.time
    edges = tables.edges.copy()
    tables.edges.clear()
    for edge in edges:
        if node_time[edge.parent] <= time:
            tables.edges.append(edge)

    mutations = tables.mutations.copy()
    # Map of old ID -> new ID
    mutation_map = np.full(len(mutations), tskit.NULL, dtype=int)
    tables.mutations.clear()
    keep = []
    for j, mutation in enumerate(mutations):
        mutation_time = (
            node_time[mutation.node]
            if util.is_unknown_time(mutation.time)
            else mutation.time
        )
        if mutation_time < time:
            mutation_map[j] = len(keep)
            keep.append(mutation)
    # Not making assumptions about ordering, so do it in two passes.
    for mutation in keep:
        if mutation.parent != tskit.NULL:
            mutation = mutation.replace(parent=mutation_map[mutation.parent])
        tables.mutations.append(mutation)

    migrations = tables.migrations.copy()
    tables.migrations.clear()
    for migration in migrations:
        if migration.time < time:
            tables.migrations.append(migration)


class TestDeleteOlderExamples:
    @pytest.mark.parametrize("ts", get_example_tree_sequences())
    def test_definition(self, ts):
        time = 0 if ts.num_nodes == 0 else np.median(ts.tables.nodes.time)
        tables1 = ts.dump_tables()
        delete_older_definition(tables1, time)
        tables2 = ts.dump_tables()
        tables2.delete_older(time)
        tables1.assert_equals(tables2, ignore_provenance=True)

    @pytest.mark.parametrize("ts", get_example_tree_sequences())
    def test_mutation_parents(self, ts):
        time = 0 if ts.num_nodes == 0 else np.median(ts.tables.nodes.time)
        tables1 = ts.dump_tables()
        tables1.delete_older(time)
        tables2 = tables1.copy()
        tables2.build_index()
        tables2.compute_mutation_parents()
        tables1.assert_equals(tables2, ignore_provenance=True)


class TestDeleteOlderSimpleTree:
    # 2.00┊   4   ┊
    #     ┊ ┏━┻┓  ┊
    # 1.00┊ ┃  3  ┊
    #     ┊ ┃ ┏┻┓ ┊
    # 0.00┊ 0 1 2 ┊
    #     0       1
    def tables(self):
        # Don't cache this because we modify the result!
        tree = tskit.Tree.generate_balanced(3, branch_length=1)
        return tree.tree_sequence.dump_tables()

    @pytest.mark.parametrize("time", [0, -0.5, -100, 0.01, 0.999])
    def test_before_first_internal_node(self, time):
        tables = self.tables()
        before = tables.copy()
        tables.delete_older(time)
        ts = tables.tree_sequence()
        assert ts.num_trees == 1
        tree = ts.first()
        assert tree.num_roots == 3
        assert list(sorted(tree.roots)) == [0, 1, 2]
        assert before.nodes.equals(tables.nodes[: len(before.nodes)])
        assert len(tables.edges) == 0

    @pytest.mark.parametrize("time", [1, 1.01, 1.5, 1.999])
    def test_t1_to_2(self, time):
        #
        # 2.00┊       ┊
        #     ┊       ┊
        # 1.00┊    3  ┊
        #     ┊   ┏┻┓ ┊
        # 0.00┊ 0 1 2 ┊
        #     0       1
        tables = self.tables()
        before = tables.copy()
        tables.delete_older(time)
        ts = tables.tree_sequence()
        assert ts.num_trees == 1
        tree = ts.first()
        assert tree.num_roots == 2
        assert list(sorted(tree.roots)) == [0, 3]
        assert len(tables.nodes) == 5
        assert before.nodes.equals(tables.nodes)

    @pytest.mark.parametrize("time", [2, 2.5, 1e9])
    def test_t2(self, time):
        tables = self.tables()
        before = tables.copy()
        tables.delete_older(time)
        tables.assert_equals(before, ignore_provenance=True)


class TestDeleteOlderSimpleTreeMutationExamples:
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

        tables.delete_older(1)
        # 2.00┊       ┊
        #     ┊       ┊
        # 1.00┊    3  ┊
        #     ┊ x ┏┻┓ ┊
        # 0.00┊ 0 1 2 ┊
        #     0       1
        assert len(tables.nodes) == 5
        mut = tables.mutations[0]
        assert mut.node == 0
        assert mut.derived_state == "T"
        assert mut.metadata == b"1234"
        assert tskit.is_unknown_time(mut.time)

    def test_single_mutation_before_time(self):
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
        tables.delete_older(1)
        # 2.00┊       ┊
        #     ┊       ┊
        # 1.00┊    3  ┊
        #     ┊   ┏┻┓ ┊
        # 0.00┊ 0 1 2 ┊
        #     0       1
        assert len(tables.nodes) == 5
        assert len(tables.mutations) == 0

    def test_single_mutation_at_time(self):
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

        tables.delete_older(1)
        # 2.00┊       ┊
        #     ┊       ┊
        # 1.00┊    3  ┊
        #     ┊   ┏┻┓ ┊
        # 0.00┊ 0 1 2 ┊
        #     0       1
        assert len(tables.nodes) == 5
        assert len(tables.mutations) == 0

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
        before = tables.copy()

        tables.delete_older(1)
        # 2.00┊   4   ┊
        #     ┊       ┊
        #     ┊    3  ┊
        #     ┊ x  ┃  ┊
        #     ┊ x ┏┻┓ ┊
        # 0.00┊ 0 1 2 ┊
        #     0       1
        tables.mutations.assert_equals(before.mutations)

    def test_multi_mutation_out_of_order(self):
        # 2.00┊   4   ┊
        #     ┊ ┏━┻┓  ┊
        # 1.00┊ x  3  ┊
        #     ┊ x ┏┻┓ ┊
        # 0.00┊ 0 1 2 ┊
        #     0       1
        tree = tskit.Tree.generate_balanced(3, branch_length=1)
        tables = tree.tree_sequence.dump_tables()
        tables.sites.add_row(0, "A")
        tables.mutations.add_row(site=0, node=0, parent=1, derived_state="G")
        tables.mutations.add_row(site=0, node=0, derived_state="T")
        before = tables.copy()
        with pytest.raises(tskit.LibraryError, match="PARENT_AFTER_CHILD"):
            tables.tree_sequence()

        tables.delete_older(1)
        # 2.00┊   4   ┊
        #     ┊       ┊
        #     ┊    3  ┊
        #     ┊ x  ┃  ┊
        #     ┊ x ┏┻┓ ┊
        # 0.00┊ 0 1 2 ┊
        #     0       1
        tables.mutations.assert_equals(before.mutations)

    def test_mutation_not_on_branch(self):
        tables = tskit.TableCollection(1)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.sites.add_row(0, "A")
        tables.mutations.add_row(site=0, node=0, derived_state="T")
        before = tables.copy()
        tables.delete_older(0.01)
        tables.assert_equals(before, ignore_provenance=True)


class TestDeleteOlderSimpleTreeMigrationExamples:
    @tests.cached_example
    def ts(self):
        # 2.00┊   4   ┊
        #     ┊ o━┻┓  ┊
        # 1.00┊ o  3  ┊
        #     ┊ o ┏┻┓ ┊
        # 0.00┊ 0 1 2 ┊
        #     0       1
        tree = tskit.Tree.generate_balanced(3, branch_length=1)
        tables = tree.tree_sequence.dump_tables()
        tables.populations.add_row()
        tables.populations.add_row()
        tables.migrations.add_row(source=0, dest=1, node=0, time=0.5, left=0, right=1)
        tables.migrations.add_row(source=1, dest=0, node=0, time=1.0, left=0, right=1)
        tables.migrations.add_row(source=0, dest=1, node=0, time=1.5, left=0, right=1)
        tables.compute_mutation_parents()
        ts = tables.tree_sequence()
        return ts

    def test_t099(self):
        tables = self.ts().dump_tables()
        tables.delete_older(0.99)
        assert len(tables.migrations) == 1
        assert tables.migrations[0].time == 0.5

    def test_t1(self):
        tables = self.ts().dump_tables()
        tables.delete_older(1)
        assert len(tables.migrations) == 1
        assert tables.migrations[0].time == 0.5

    @pytest.mark.parametrize("time", [1.51, 2.0, 2.5])
    def test_older(self, time):
        tables = self.ts().dump_tables()
        before = tables.copy()
        tables.delete_older(time)
        tables.migrations.assert_equals(before.migrations)


def split_edges_definition(ts, time, *, flags=0, population=None, metadata=None):
    population = -1 if population is None else population
    tables = ts.dump_tables()
    if ts.num_migrations > 0:
        raise ValueError("Migrations not supported")

    node_time = tables.nodes.time
    tables.edges.clear()
    split_edge = np.full(ts.num_edges, tskit.NULL, dtype=int)
    for edge in ts.edges():
        if node_time[edge.child] < time < node_time[edge.parent]:
            u = tables.nodes.add_row(
                flags=flags, time=time, population=population, metadata=metadata
            )
            tables.edges.append(edge.replace(parent=u))
            tables.edges.append(edge.replace(child=u))
            split_edge[edge.id] = u
        else:
            tables.edges.append(edge)

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
        assert ts.node(2).population == -1

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


def decapitate_definition(ts, time, *, flags=0, population=None, metadata=None):
    """
    Simple loop implementation of the decapitate operation
    """
    population = -1 if population is None else population
    tables = ts.dump_tables()
    node_time = tables.nodes.time
    tables.edges.clear()
    for edge in ts.edges():
        if node_time[edge.parent] <= time:
            tables.edges.append(edge)
        elif node_time[edge.child] < time:
            new_parent = tables.nodes.add_row(
                time=time, population=population, flags=flags, metadata=metadata
            )
            tables.edges.append(edge.replace(parent=new_parent))

    tables.mutations.clear()
    for mutation in ts.mutations():
        mutation_time = (
            node_time[mutation.node]
            if util.is_unknown_time(mutation.time)
            else mutation.time
        )
        if mutation_time < time:
            tables.mutations.append(mutation.replace(parent=tskit.NULL))

    tables.migrations.clear()
    for migration in ts.migrations():
        if migration.time <= time:
            tables.migrations.append(migration)

    tables.build_index()
    tables.compute_mutation_parents()
    return tables.tree_sequence()


class TestDecapitateExamples:
    @pytest.mark.parametrize("ts", get_example_tree_sequences())
    def test_defaults(self, ts):
        time = 0 if ts.num_nodes == 0 else np.median(ts.tables.nodes.time)
        if ts.num_migrations == 0:
            decap1 = decapitate_definition(ts, time)
            decap2 = ts.decapitate(time)
            decap1.tables.assert_equals(decap2.tables, ignore_provenance=True)

    @pytest.mark.parametrize("ts", get_example_tree_sequences())
    def test_no_population(self, ts):
        time = 0 if ts.num_nodes == 0 else np.median(ts.tables.nodes.time)
        if ts.num_migrations == 0:
            decap1 = decapitate_definition(ts, time, population=-1)
            decap2 = ts.decapitate(time, population=-1)
            decap1.tables.assert_equals(decap2.tables, ignore_provenance=True)


class TestDecapitateSimpleTree:
    # 2.00┊   4   ┊
    #     ┊ ┏━┻┓  ┊
    # 1.00┊ ┃  3  ┊
    #     ┊ ┃ ┏┻┓ ┊
    # 0.00┊ 0 1 2 ┊
    #     0       1
    @tests.cached_example
    def ts(self):
        tree = tskit.Tree.generate_balanced(3, branch_length=1)
        return tree.tree_sequence

    @pytest.mark.parametrize("time", [0, -0.5, -100])
    def test_t0_or_before(self, time):
        before = self.ts()
        ts = before.decapitate(time)
        assert ts.num_trees == 1
        tree = ts.first()
        assert tree.num_roots == 3
        assert list(sorted(tree.roots)) == [0, 1, 2]
        assert before.tables.nodes.equals(ts.tables.nodes[: before.num_nodes])
        assert ts.num_edges == 0

    @pytest.mark.parametrize("time", [0.01, 0.5, 0.999])
    def test_t0_to_1(self, time):
        #
        # 2.00┊       ┊
        #     ┊       ┊
        # 0.99┊ 7 5 6 ┊
        #     ┊ ┃ ┃ ┃ ┊
        # 0.00┊ 0 1 2 ┊
        #     0       1
        before = self.ts()
        ts = before.decapitate(time)
        assert ts.num_trees == 1
        tree = ts.first()
        assert tree.num_roots == 3
        assert list(sorted(tree.roots)) == [5, 6, 7]
        assert ts.num_nodes == 8
        assert ts.tables.nodes[5].time == time
        assert ts.tables.nodes[6].time == time
        assert ts.tables.nodes[7].time == time

    def test_t1(self):
        #
        # 2.00┊       ┊
        #     ┊       ┊
        # 1.00┊ 5  3  ┊
        #     ┊ ┃ ┏┻┓ ┊
        # 0.00┊ 0 1 2 ┊
        #     0       1
        before = self.ts()
        ts = before.decapitate(1)
        assert ts.num_trees == 1
        tree = ts.first()
        assert tree.num_roots == 2
        assert list(sorted(tree.roots)) == [3, 5]
        assert ts.num_nodes == 6
        assert ts.tables.nodes[5].time == 1

    @pytest.mark.parametrize("time", [1.01, 1.5, 1.999])
    def test_t1_to_2(self, time):
        # 2.00┊       ┊
        #     ┊       ┊
        # 1.01┊ 5  6  ┊
        #     ┊ ┃  ┃  ┊
        # 1.00┊ ┃  3  ┊
        #     ┊ ┃ ┏┻┓ ┊
        # 0.00┊ 0 1 2 ┊
        #    0       1
        before = self.ts()
        ts = before.decapitate(time)
        assert ts.num_trees == 1
        tree = ts.first()
        assert tree.num_roots == 2
        assert list(sorted(tree.roots)) == [5, 6]
        assert ts.num_nodes == 7
        assert ts.tables.nodes[5].time == time
        assert ts.tables.nodes[6].time == time

    @pytest.mark.parametrize("time", [2, 2.5, 1e9])
    def test_t2(self, time):
        before = self.ts()
        ts = before.decapitate(time)
        ts.tables.assert_equals(before.tables, ignore_provenance=True)


class TestDecapitateSimpleTreeMutationExamples:
    def test_single_mutation_over_sample(self):
        # 2.00┊   4   ┊
        #     ┊ ┏━┻┓  ┊
        # 1.00┊ ┃  3  ┊
        #     ┊ x ┏┻┓ ┊
        # 0.00┊ 0 1 2 ┊
        #     0       1
        tree = tskit.Tree.generate_balanced(3, branch_length=1)
        tables = tree.tree_sequence.dump_tables()
        tables.sites.add_row(0, "A")
        tables.mutations.add_row(site=0, node=0, derived_state="T")
        before = tables.tree_sequence()

        ts = before.decapitate(1)
        # 2.00┊       ┊
        #     ┊       ┊
        # 1.00┊ 5  3  ┊
        #     ┊ x ┏┻┓ ┊
        # 0.00┊ 0 1 2 ┊
        #     0       1
        before.tables.mutations.assert_equals(ts.tables.mutations)
        assert list(before.alignments()) == list(ts.alignments())

    def test_single_mutation_at_decap_time(self):
        # 2.00┊   4   ┊
        #     ┊ ┏━┻┓  ┊
        # 1.00┊ x  3  ┊
        #     ┊ ┃ ┏┻┓ ┊
        # 0.00┊ 0 1 2 ┊
        #     0       1
        tree = tskit.Tree.generate_balanced(3, branch_length=1)
        tables = tree.tree_sequence.dump_tables()
        tables.sites.add_row(0, "A")
        tables.mutations.add_row(site=0, node=0, time=1, derived_state="T")
        before = tables.tree_sequence()

        # Because the mutation is at exactly the decapitation time, we must
        # remove it, or it would violate the requirement that a mutation must
        # have a time less than that of the parent of the edge that its on.
        ts = before.decapitate(1)
        # 2.00┊       ┊
        #     ┊       ┊
        # 1.00┊ 5  3  ┊
        #     ┊ ┃ ┏┻┓ ┊
        # 0.00┊ 0 1 2 ┊
        #     0       1
        assert ts.num_mutations == 0
        assert list(ts.alignments()) == ["A", "A", "A"]

    def test_multi_mutation_over_sample(self):
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
        before = tables.tree_sequence()

        ts = before.decapitate(1)
        # 2.00┊       ┊
        #     ┊ 5  3  ┊
        #     ┊ x  ┃  ┊
        #     ┊ x ┏┻┓ ┊
        # 0.00┊ 0 1 2 ┊
        #     0       1
        before.tables.mutations.assert_equals(ts.tables.mutations)
        assert list(before.alignments()) == list(ts.alignments())

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
        before = tables.tree_sequence()

        ts = before.decapitate(1)
        # 2.00┊       ┊
        #     ┊ 5  3  ┊
        #     ┊ ┃  ┃  ┊
        #     ┊ x ┏┻┓ ┊
        # 0.00┊ 0 1 2 ┊
        #     0       1
        assert ts.num_mutations == 1
        # Alignments are equal because the ancestral mutation was silent anyway.
        assert list(before.alignments()) == list(ts.alignments())

    def test_multi_mutation_over_root(self):
        #         x
        # 2.00┊   4   ┊
        #     ┊ ┏━┻┓  ┊
        # 1.00┊ ┃  3  ┊
        #     ┊ x ┏┻┓ ┊
        # 0.00┊ 0 1 2 ┊
        #     0       1
        tree = tskit.Tree.generate_balanced(3, branch_length=1)
        tables = tree.tree_sequence.dump_tables()
        tables.sites.add_row(0, "A")
        tables.mutations.add_row(site=0, node=4, derived_state="G")
        tables.mutations.add_row(site=0, node=0, parent=0, derived_state="T")
        before = tables.tree_sequence()

        ts = before.decapitate(1)
        # 2.00┊       ┊
        #     ┊ 5  3  ┊
        #     ┊ ┃  ┃  ┊
        #     ┊ x ┏┻┓ ┊
        # 0.00┊ 0 1 2 ┊
        #     0       1
        assert ts.num_mutations == 1
        assert list(before.alignments()) == ["T", "G", "G"]
        # The states inherited by samples changes because we drop the old mutation
        assert list(ts.alignments()) == ["T", "A", "A"]


class TestDecapitateSimpleTsExample:
    # 9.08┊    9    ┊         ┊         ┊         ┊         ┊
    #     ┊  ┏━┻━┓  ┊         ┊         ┊         ┊         ┊
    # 6.57┊  ┃   ┃  ┊         ┊         ┊         ┊    8    ┊
    #     ┊  ┃   ┃  ┊         ┊         ┊         ┊  ┏━┻━┓  ┊
    # 5.31┊  ┃   ┃  ┊    7    ┊         ┊    7    ┊  ┃   ┃  ┊
    #     ┊  ┃   ┃  ┊  ┏━┻━┓  ┊         ┊  ┏━┻━┓  ┊  ┃   ┃  ┊
    # 1.75┊  ┃   ┃  ┊  ┃   ┃  ┊    6    ┊  ┃   ┃  ┊  ┃   ┃  ┊
    #     ┊  ┃   ┃  ┊  ┃   ┃  ┊  ┏━┻━┓  ┊  ┃   ┃  ┊  ┃   ┃  ┊
    # 1.11┊  ┃   5  ┊  ┃   5  ┊  ┃   5  ┊  ┃   5  ┊  ┃   5  ┊
    #     ┊  ┃  ┏┻┓ ┊  ┃  ┏┻┓ ┊  ┃  ┏┻┓ ┊  ┃  ┏┻┓ ┊  ┃  ┏┻┓ ┊
    # 0.11┊  4  ┃ ┃ ┊  4  ┃ ┃ ┊  4  ┃ ┃ ┊  4  ┃ ┃ ┊  4  ┃ ┃ ┊
    #     ┊ ┏┻┓ ┃ ┃ ┊ ┏┻┓ ┃ ┃ ┊ ┏┻┓ ┃ ┃ ┊ ┏┻┓ ┃ ┃ ┊ ┏┻┓ ┃ ┃ ┊
    # 0.00┊ 0 1 2 3 ┊ 0 1 2 3 ┊ 0 1 2 3 ┊ 0 1 2 3 ┊ 0 1 2 3 ┊
    #   0.00      0.06      0.79      0.91      0.91      1.00

    @tests.cached_example
    def ts(self):
        nodes = io.StringIO(
            """\
        id      is_sample   population      individual      time    metadata
        0       1       0       -1      0
        1       1       0       -1      0
        2       1       0       -1      0
        3       1       0       -1      0
        4       0       0       -1      0.114
        5       0       0       -1      1.110
        6       0       0       -1      1.750
        7       0       0       -1      5.310
        8       0       0       -1      6.573
        9       0       0       -1      9.083
        """
        )
        edges = io.StringIO(
            """\
        id      left            right           parent  child
        0       0.00000000      1.00000000      4       0
        1       0.00000000      1.00000000      4       1
        2       0.00000000      1.00000000      5       2
        3       0.00000000      1.00000000      5       3
        4       0.79258618      0.90634460      6       4
        5       0.79258618      0.90634460      6       5
        6       0.05975243      0.79258618      7       4
        7       0.90634460      0.91029435      7       4
        8       0.05975243      0.79258618      7       5
        9       0.90634460      0.91029435      7       5
        10      0.91029435      1.00000000      8       4
        11      0.91029435      1.00000000      8       5
        12      0.00000000      0.05975243      9       4
        13      0.00000000      0.05975243      9       5
        """
        )
        sites = io.StringIO(
            """\
        position      ancestral_state
        0.05          A
        0.06          0
        0.3           C
        0.5           AAA
        0.91          T
        """
        )
        muts = io.StringIO(
            """\
        site   node    derived_state    parent    time
        0      9       T                -1        15
        0      9       GGG              0         9.1
        0      5       1                1         9
        1      4       C                -1        1.6
        1      4       G                3         1.5
        2      7       G                -1        10
        2      3       C                5         1
        4      3       G                -1        1
        """
        )
        ts = tskit.load_text(nodes, edges, sites=sites, mutations=muts, strict=False)
        return ts

    def test_at_time_of_5(self):
        # NOTE: we don't remember that the edge 4-7 was shared in trees 1 and 3.
        # 1.11┊  14  5  ┊ 11   5  ┊ 10   5  ┊  12  5  ┊  13  5  ┊
        #     ┊  ┃  ┏┻┓ ┊  ┃  ┏┻┓ ┊  ┃  ┏┻┓ ┊  ┃  ┏┻┓ ┊  ┃  ┏┻┓ ┊
        # 0.11┊  4  ┃ ┃ ┊  4  ┃ ┃ ┊  4  ┃ ┃ ┊  4  ┃ ┃ ┊  4  ┃ ┃ ┊
        #     ┊ ┏┻┓ ┃ ┃ ┊ ┏┻┓ ┃ ┃ ┊ ┏┻┓ ┃ ┃ ┊ ┏┻┓ ┃ ┃ ┊ ┏┻┓ ┃ ┃ ┊
        # 0.00┊ 0 1 2 3 ┊ 0 1 2 3 ┊ 0 1 2 3 ┊ 0 1 2 3 ┊ 0 1 2 3 ┊
        #   0.00      0.06      0.79      0.91      0.91      1.00
        ts = self.ts().decapitate(1.110)
        assert ts.num_nodes == 15
        assert ts.num_trees == 5
        # Most mutations are older than this.
        assert ts.num_mutations == 2
        for u in range(10, 15):
            node = ts.node(u)
            assert node.time == 1.110
            assert node.flags == 0
        assert [set(tree.roots) for tree in ts.trees()] == [
            {5, 14},
            {11, 5},
            {10, 5},
            {12, 5},
            {13, 5},
        ]

    def test_at_time6(self):
        # 6   ┊ 12   13 ┊         ┊         ┊         ┊ 10   11 ┊
        # 5.31┊  ┃   ┃  ┊    7    ┊         ┊    7    ┊  ┃   ┃  ┊
        #     ┊  ┃   ┃  ┊  ┏━┻━┓  ┊         ┊  ┏━┻━┓  ┊  ┃   ┃  ┊
        # 1.75┊  ┃   ┃  ┊  ┃   ┃  ┊    6    ┊  ┃   ┃  ┊  ┃   ┃  ┊
        #     ┊  ┃   ┃  ┊  ┃   ┃  ┊  ┏━┻━┓  ┊  ┃   ┃  ┊  ┃   ┃  ┊
        # 1.11┊  ┃   5  ┊  ┃   5  ┊  ┃   5  ┊  ┃   5  ┊  ┃   5  ┊
        #     ┊  ┃  ┏┻┓ ┊  ┃  ┏┻┓ ┊  ┃  ┏┻┓ ┊  ┃  ┏┻┓ ┊  ┃  ┏┻┓ ┊
        # 0.11┊  4  ┃ ┃ ┊  4  ┃ ┃ ┊  4  ┃ ┃ ┊  4  ┃ ┃ ┊  4  ┃ ┃ ┊
        #     ┊ ┏┻┓ ┃ ┃ ┊ ┏┻┓ ┃ ┃ ┊ ┏┻┓ ┃ ┃ ┊ ┏┻┓ ┃ ┃ ┊ ┏┻┓ ┃ ┃ ┊
        # 0.00┊ 0 1 2 3 ┊ 0 1 2 3 ┊ 0 1 2 3 ┊ 0 1 2 3 ┊ 0 1 2 3 ┊
        #   0.00      0.06      0.79      0.91      0.91      1.00
        ts = self.ts().decapitate(6)
        assert ts.num_nodes == 14
        assert ts.num_trees == 5
        assert ts.num_mutations == 4
        for u in range(10, 14):
            node = ts.node(u)
            assert node.time == 6
            assert node.flags == 0
        assert [set(tree.roots) for tree in ts.trees()] == [
            {12, 13},
            {7},
            {6},
            {7},
            {10, 11},
        ]


class TestDecapitateNodeValues:
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
        ts = self.ts().decapitate(0.5)
        assert ts.node(2).population == tskit.NULL

    @pytest.mark.parametrize("population", range(-1, 5))
    def test_specify_population(self, population):
        ts = self.ts().decapitate(0.5, population=population)
        assert ts.node(2).population == population

    def test_default_flags(self):
        ts = self.ts().decapitate(0.5)
        assert ts.node(2).flags == 0

    @pytest.mark.parametrize("flags", range(0, 5))
    def test_specify_flags(self, flags):
        ts = self.ts().decapitate(0.5, flags=flags)
        assert ts.node(2).flags == flags

    def test_default_metadata_no_schema(self):
        ts = self.ts().decapitate(0.5)
        assert ts.node(2).metadata == b""

    @pytest.mark.parametrize("metadata", [b"", b"some bytes"])
    def test_specify_metadata_no_schema(self, metadata):
        ts = self.ts().decapitate(0.5, metadata=metadata)
        assert ts.node(2).metadata == metadata

    def test_default_metadata_with_schema(self):
        ts = self.ts_with_schema().decapitate(0.5)
        assert ts.node(2).metadata == {}

    @pytest.mark.parametrize("metadata", [{}, {"some": "json"}])
    def test_specify_metadata_with_schema(self, metadata):
        ts = self.ts_with_schema().decapitate(0.5, metadata=metadata)
        assert ts.node(2).metadata == metadata


class TestDecapitateInterface:
    @tests.cached_example
    def ts(self):
        tree = tskit.Tree.generate_balanced(3, branch_length=1)
        return tree.tree_sequence

    @pytest.mark.parametrize("bad_type", ["x", "0.1", [], [0.1]])
    def test_bad_types(self, bad_type):
        with pytest.raises(TypeError, match="number"):
            self.ts().decapitate(bad_type)

    @pytest.mark.parametrize(
        "time", [1, 1.0, np.array([1])[0], fractions.Fraction(1, 1), decimal.Decimal(1)]
    )
    def test_number_types(self, time):
        expected = self.ts().decapitate(1)
        got = self.ts().decapitate(time)
        expected.tables.assert_equals(got.tables, ignore_timestamps=True)

    def test_migrations_not_supported(self, ts_fixture):
        with pytest.raises(tskit.LibraryError, match="MIGRATIONS_NOT_SUPPORTED"):
            ts_fixture.decapitate(0)

    def test_population_out_of_bounds(self):
        tables = tskit.TableCollection(1)
        ts = tables.tree_sequence()
        with pytest.raises(tskit.LibraryError, match="POPULATION_OUT_OF_BOUNDS"):
            ts.decapitate(0, population=0)

    def test_bad_flags(self):
        ts = tskit.TableCollection(1).tree_sequence()
        with pytest.raises(TypeError):
            ts.decapitate(0, flags="asdf")

    def test_bad_metadata_no_schema(self):
        ts = tskit.TableCollection(1).tree_sequence()
        with pytest.raises(TypeError):
            ts.decapitate(0, metadata="asdf")

    def test_bad_metadata_json_schema(self):
        tables = tskit.TableCollection(1)
        tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
        ts = tables.tree_sequence()
        with pytest.raises(tskit.MetadataEncodingError):
            ts.decapitate(0, metadata=b"bytes")

    @pytest.mark.parametrize("time", [math.inf, np.inf, tskit.UNKNOWN_TIME, np.nan])
    def test_nonfinite_time(self, time):
        tables = tskit.TableCollection(1)
        ts = tables.tree_sequence()
        with pytest.raises(tskit.LibraryError, match="TIME_NONFINITE"):
            ts.decapitate(time)
