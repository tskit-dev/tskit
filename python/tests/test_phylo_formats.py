# MIT License
#
# Copyright (c) 2018-2021 Tskit Developers
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
import dendropy
import msprime
import newick
import pytest

import tskit
from tests import tsutil


class TreeExamples:
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
                if include_branch_lengths:
                    assert node.length == pytest.approx(tree.branch_length(u))
                else:
                    assert node.length is None
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
    an external Nexus parser.
    """

    def verify_tree(self, tsk_tree, dpy_tree):
        """
        Checks that the specified Dendropy tree is equal to the specified
        tskit tree, up to the limits imposed by newick.
        """
        label_map = {}
        for node in dpy_tree:
            label_map[str(node.taxon.label)] = node

        def get_label(u):
            node = tsk_tree.tree_sequence.node(u)
            return f"tsk_{node.id}_{node.flags}"

        for u in tsk_tree.nodes(order="postorder"):
            # Consume the nodes in the dendropy node map one-by-one
            dpy_node = label_map.pop(get_label(u))
            parent = tsk_tree.parent(u)
            if parent == tskit.NULL:
                assert dpy_node.edge_length is None
                assert dpy_node.parent_node is None
            else:
                assert tsk_tree.branch_length(u) == pytest.approx(dpy_node.edge_length)
                assert dpy_node.parent_node is label_map[get_label(parent)]

        assert len(label_map) == 0

    def verify_nexus_topology(self, ts):
        nexus = ts.to_nexus(precision=16)
        tree_list = dendropy.TreeList()
        tree_list.read(
            data=nexus,
            schema="nexus",
            preserve_underscores=True,  # TODO remove this when we update labels
            rooting="default-rooted",  # Remove this when we have root marking, #1815
            suppress_internal_node_taxa=False,
        )
        assert ts.num_trees == len(tree_list)
        for tsk_tree, dpy_tree in zip(ts.trees(), tree_list):
            # https://github.com/tskit-dev/tskit/issues/1815
            # FIXME this label should probably start with "[&R]" and
            # use some other separator than "_" to delimit the
            # left and right coords. Should we use a "weight" instead?
            assert dpy_tree.label.startswith("tree")
            left, right = map(float, dpy_tree.label[4:].split("_"))
            assert tsk_tree.interval.left == pytest.approx(left)
            assert tsk_tree.interval.right == pytest.approx(right)
            self.verify_tree(tsk_tree, dpy_tree)

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
