# MIT License
#
# Copyright (c) 2018-2019 Tskit Developers
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
Tests for the newick output feature.
"""
import itertools
import unittest

import msprime
import newick
import pytest
from Bio.Nexus import Nexus

import tskit
from tests import tsutil


class TreeExamples(unittest.TestCase):
    """
    Generates trees for testing the phylo format outputs.
    """

    random_seed = 155

    def get_nonbinary_example(self):
        ts = msprime.simulate(
            sample_size=20,
            recombination_rate=10,
            random_seed=self.random_seed,
            demographic_events=[
                msprime.SimpleBottleneck(time=0.5, population=0, proportion=1)
            ],
        )
        # Make sure this really has some non-binary nodes
        found = False
        for e in ts.edgesets():
            if len(e.children) > 2:
                found = True
                break
        assert found
        return ts

    def get_binary_example(self):
        ts = msprime.simulate(
            sample_size=25, recombination_rate=5, random_seed=self.random_seed
        )
        return ts

    def get_multiroot_example(self):
        ts = msprime.simulate(sample_size=50, random_seed=self.random_seed)
        tables = ts.dump_tables()
        edges = tables.edges
        n = len(edges) // 2
        edges.set_columns(
            left=edges.left[:n],
            right=edges.right[:n],
            parent=edges.parent[:n],
            child=edges.child[:n],
        )
        return tables.tree_sequence()

    def get_single_tree(self):
        return msprime.simulate(10, random_seed=2)


class TestNewick(TreeExamples):
    """
    Tests that the newick output has the properties that we need using
    external Newick parser.
    """

    # 3 methods to return example tree sequences with internal samples:
    # (copied from test_highlevel.py)
    def all_nodes_samples_example(self):
        n = 5
        ts = msprime.simulate(n, random_seed=10, mutation_rate=5)
        assert ts.num_mutations > 0
        tables = ts.dump_tables()
        nodes = tables.nodes
        flags = nodes.flags
        # Set all nodes to be samples.
        flags[:] = tskit.NODE_IS_SAMPLE
        nodes.flags = flags
        return tables.tree_sequence()

    def only_internal_samples_example(self):
        n = 5
        ts = msprime.simulate(n, random_seed=10, mutation_rate=5)
        assert ts.num_mutations > 0
        tables = ts.dump_tables()
        nodes = tables.nodes
        flags = nodes.flags
        # Set just internal nodes to be samples.
        flags[:] = 0
        flags[n:] = tskit.NODE_IS_SAMPLE
        nodes.flags = flags
        return tables.tree_sequence()

    def mixed_node_samples_example(self):
        n = 5
        ts = msprime.simulate(n, random_seed=10, mutation_rate=5)
        assert ts.num_mutations > 0
        tables = ts.dump_tables()
        nodes = tables.nodes
        flags = nodes.flags
        # Set a mixture of internal and leaf samples.
        flags[:] = 0
        flags[n // 2 : n + n // 2] = tskit.NODE_IS_SAMPLE
        nodes.flags = flags
        return tables.tree_sequence()

    def verify_newick_topology(
        self, tree, root=None, node_labels=None, include_branch_lengths=True
    ):
        if root is None:
            root = tree.root
        ns = tree.newick(
            precision=16,
            root=root,
            node_labels=node_labels,
            include_branch_lengths=include_branch_lengths,
        )
        if node_labels is None:
            leaf_labels = {u: str(u + 1) for u in tree.leaves(root)}
        else:
            leaf_labels = {u: node_labels[u] for u in tree.leaves(root)}
        # default newick lib outputs 0.0 if length is None => replace the length_parser
        newick_tree = newick.loads(
            ns, length_parser=lambda x: None if x is None else float(x)
        )[0]
        leaf_names = newick_tree.get_leaf_names()
        assert sorted(leaf_names) == sorted(leaf_labels.values())
        for u in tree.leaves(root):
            name = leaf_labels[u]
            node = newick_tree.get_node(name)
            while u != root:
                branch_len = tree.branch_length(u) if include_branch_lengths else None
                self.assertAlmostEqual(node.length, branch_len)
                node = node.ancestor
                u = tree.parent(u)
            assert node.ancestor is None

    def test_nonbinary_tree(self):
        ts = self.get_nonbinary_example()
        for t in ts.trees():
            self.verify_newick_topology(t)

    def test_binary_tree(self):
        ts = self.get_binary_example()
        for t in ts.trees():
            self.verify_newick_topology(t)

    def test_multiroot(self):
        ts = self.get_multiroot_example()
        t = ts.first()
        with pytest.raises(ValueError):
            t.newick()
        for root in t.roots:
            self.verify_newick_topology(t, root=root)

    def test_all_nodes(self):
        ts = msprime.simulate(10, random_seed=5)
        tree = ts.first()
        for u in tree.nodes():
            self.verify_newick_topology(tree, root=u)

    def test_binary_leaf_labels(self):
        tree = self.get_binary_example().first()
        labels = {u: f"x_{u}" for u in tree.leaves()}
        self.verify_newick_topology(tree, node_labels=labels)

    def test_nonbinary_leaf_labels(self):
        ts = self.get_nonbinary_example()
        for t in ts.trees():
            labels = {u: str(u) for u in t.leaves()}
            self.verify_newick_topology(t, node_labels=labels)

    def test_all_node_labels(self):
        tree = msprime.simulate(5, random_seed=2).first()
        labels = {u: f"x_{u}" for u in tree.nodes()}
        ns = tree.newick(node_labels=labels)
        root = newick.loads(ns)[0]
        assert root.name == labels[tree.root]
        assert sorted([n.name for n in root.walk()]) == sorted(labels.values())

    def test_single_node_label(self):
        tree = msprime.simulate(5, random_seed=2).first()
        labels = {tree.root: "XXX"}
        ns = tree.newick(node_labels=labels)
        root = newick.loads(ns)[0]
        assert root.name == labels[tree.root]
        assert [n.name for n in root.walk()] == [labels[tree.root]] + [
            None for _ in range(len(list(tree.nodes())) - 1)
        ]

    def test_no_lengths(self):
        t = msprime.simulate(5, random_seed=2).first()
        self.verify_newick_topology(t, include_branch_lengths=False)

    def test_samples_differ_from_leaves(self):
        for ts in (
            self.all_nodes_samples_example(),
            self.only_internal_samples_example(),
            self.mixed_node_samples_example(),
        ):
            for t in ts.trees():
                self.verify_newick_topology(t)

    def test_no_lengths_equiv(self):
        for ts in (
            self.all_nodes_samples_example(),
            self.only_internal_samples_example(),
            self.mixed_node_samples_example(),
        ):
            for t in ts.trees():
                newick_nolengths = t.newick(include_branch_lengths=False)
                newick_nolengths = newick.loads(newick_nolengths)[0]
                newick_lengths = t.newick()
                newick_lengths = newick.loads(newick_lengths)[0]
                for node in newick_lengths.walk():
                    node.length = None
                assert newick.dumps(newick_nolengths) == newick.dumps(newick_lengths)


class TestNexus(TreeExamples):
    """
    Tests that the nexus output has the properties that we need using
    external Nexus parser.
    """

    def verify_tree(self, nexus_tree, tsk_tree):
        assert len(nexus_tree.get_terminals()) == tsk_tree.num_samples()

        bio_node_map = {}
        for node_id in nexus_tree.all_ids():
            bio_node = nexus_tree.node(node_id)
            bio_node_map[bio_node.data.taxon] = bio_node

        for u in tsk_tree.nodes():
            node = tsk_tree.tree_sequence.node(u)
            label = f"tsk_{node.id}_{node.flags}"
            bio_node = bio_node_map.pop(label)
            self.assertAlmostEqual(
                bio_node.data.branchlength, tsk_tree.branch_length(u)
            )
            if tsk_tree.parent(u) == tskit.NULL:
                assert bio_node.prev is None
            else:
                bio_node_parent = nexus_tree.node(bio_node.prev)
                parent = tsk_tree.tree_sequence.node(tsk_tree.parent(u))
                assert bio_node_parent.data.taxon == f"tsk_{parent.id}_{parent.flags}"
        assert len(bio_node_map) == 0

    def verify_nexus_topology(self, treeseq):
        nexus = treeseq.to_nexus(precision=16)
        nexus_treeseq = Nexus.Nexus(nexus)
        assert treeseq.num_trees == len(nexus_treeseq.trees)
        for tree, nexus_tree in itertools.zip_longest(
            treeseq.trees(), nexus_treeseq.trees
        ):
            name = nexus_tree.name
            split_name = name.split("_")
            assert len(split_name) == 2
            start = float(split_name[0][4:])
            end = float(split_name[1])
            self.assertAlmostEqual(tree.interval[0], start)
            self.assertAlmostEqual(tree.interval[1], end)

            self.verify_tree(nexus_tree, tree)

    def test_binary_tree(self):
        ts = self.get_binary_example()
        self.verify_nexus_topology(ts)

    def test_nonbinary_example(self):
        ts = self.get_nonbinary_example()
        self.verify_nexus_topology(ts)

    def test_single_tree(self):
        ts = self.get_single_tree()
        self.verify_nexus_topology(ts)

    def test_multiroot(self):
        ts = self.get_multiroot_example()
        with pytest.raises(ValueError):
            ts.to_nexus()

    def test_many_trees(self):
        ts = msprime.simulate(4, recombination_rate=2, random_seed=123)
        self.verify_nexus_topology(ts)

    def test_many_trees_sequence_length(self):
        ts = msprime.simulate(4, length=10, recombination_rate=0.2, random_seed=13)
        self.verify_nexus_topology(ts)

    def test_internal_samples(self):
        ts = msprime.simulate(8, random_seed=2)
        ts = tsutil.jiggle_samples(ts)
        self.verify_nexus_topology(ts)
