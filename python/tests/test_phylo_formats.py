# MIT License
#
# Copyright (c) 2018-2021 Tskit Developers
# Copyright (c) 2016-2017 University of Oxford
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
Tests for phylogenetics export functions, newick, nexus, FASTA etc.
"""
import functools
import io
import textwrap

import dendropy
import msprime
import newick
import numpy as np
import pytest
from Bio import SeqIO

import tests
import tskit
from tests.test_highlevel import get_example_tree_sequences

# ↑ See https://github.com/tskit-dev/tskit/issues/1804 for when
# we can remove this. The example_ts here is intended to be the
# basic tree sequence which should give a meaningful result for
# most operations. Probably rename it to ``examples.simple_ts()``
# or something.


@functools.lru_cache(maxsize=100)
def alignment_example(sequence_length, include_reference=True):
    ts = msprime.sim_ancestry(
        samples=5, sequence_length=sequence_length, random_seed=123
    )
    ts = msprime.sim_mutations(ts, rate=0.1, random_seed=1234)
    tables = ts.dump_tables()
    if include_reference:
        tables.reference_sequence.data = tskit.random_nucleotides(
            ts.sequence_length, seed=1234
        )
    ts = tables.tree_sequence()
    assert ts.num_sites > 5
    return ts


@tests.cached_example
def missing_data_example():
    # 2.00┊   4     ┊
    #     ┊ ┏━┻┓    ┊
    # 1.00┊ ┃  3    ┊
    #     ┊ ┃ ┏┻┓   ┊
    # 0.00┊ 0 1 2 5 ┊
    #     0        10
    #      |      |
    #  pos 2      9
    #  anc A      T
    ts = tskit.Tree.generate_balanced(3, span=10).tree_sequence
    tables = ts.dump_tables()
    tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    tables.sites.add_row(2, ancestral_state="A")
    tables.sites.add_row(9, ancestral_state="T")
    tables.mutations.add_row(site=0, node=0, derived_state="G")
    tables.mutations.add_row(site=1, node=3, derived_state="C")
    return tables.tree_sequence()


def alignment_map(ts, **kwargs):
    alignments = ts.alignments(**kwargs)
    return {f"n{u}": alignment for u, alignment in zip(ts.samples(), alignments)}


def assert_fully_labelled_trees_equal(tree, root, node_labels, dpy_tree):
    """
    Checks the the specified fully-labelled tree rooted at the specified
    node is equivalent to the specified Dendropy tree.
    """
    label_map = {}
    for node in dpy_tree:
        label_map[str(node.taxon.label)] = node

    for u in tree.nodes(root, order="postorder"):
        # Consume the nodes in the dendropy node map one-by-one
        dpy_node = label_map.pop(node_labels[u])
        parent = tree.parent(u)
        if parent == tskit.NULL:
            assert dpy_node.edge_length is None
            assert dpy_node.parent_node is None
        else:
            assert tree.branch_length(u) == pytest.approx(dpy_node.edge_length)
            assert dpy_node.parent_node is label_map[node_labels[parent]]
    assert len(label_map) == 0


def assert_sample_labelled_trees_equal(tree, dpy_tree):
    """
    Checks that the specified trees are equivalent, where the dendropy tree
    only has labels identifying the samples.
    """
    for sample in tree.samples():
        dpy_node = dpy_tree.find_node_with_taxon_label(f"n{sample}")
        # Check the branch length paths to root are equal
        p1 = []
        u = sample
        while tree.parent(u) != tskit.NULL:
            p1.append(tree.branch_length(u))
            u = tree.parent(u)
        p2 = []
        while dpy_node.parent_node is not None:
            p2.append(dpy_node.edge_length)
            dpy_node = dpy_node.parent_node
        assert len(p1) == len(p2)
        np.testing.assert_array_almost_equal(p1, p2)


def assert_dpy_tree_list_equal(ts, tree_list):
    """
    Check that the nexus-encoded tree list output from tskit is
    parsed correctly by dendropy.
    """
    assert ts.num_trees == len(tree_list)
    for tsk_tree, dpy_tree in zip(ts.trees(), tree_list):
        # We're specifying that the tree is rooted.
        assert dpy_tree.is_rooted
        assert dpy_tree.label.startswith("t")
        left, right = map(float, dpy_tree.label[1:].split("^"))
        assert tsk_tree.interval.left == pytest.approx(left)
        assert tsk_tree.interval.right == pytest.approx(right)
        assert_sample_labelled_trees_equal(tsk_tree, dpy_tree)


class TestBackendsGiveIdenticalOutput:
    # At the default precision of 17 we should get identical results between
    # the two backends as there's no rounding done. In general, we can't
    # depend on this, though, since rounding may be done differently by the
    # Python and C library implementations.
    @pytest.mark.parametrize("ts", get_example_tree_sequences())
    def test_default_precision(self, ts):
        for tree in ts.trees():
            if tree.has_single_root:
                assert tree.as_newick() == tree.as_newick(
                    node_labels={u: f"n{u}" for u in tree.samples()}
                )


class TestNewickRoundTrip:
    """
    Test that the newick formats can round-trip the data under various
    assumptions.
    """

    @pytest.mark.parametrize("ts", get_example_tree_sequences())
    def test_leaf_labels_newick_lib(self, ts):
        for tree in ts.trees():
            # Multiroot trees raise an error
            for root in tree.roots:
                leaf_labels = {u: f"n{u}" for u in tree.leaves(root)}
                ns = tree.newick(
                    root=root,
                    precision=16,
                    node_labels=leaf_labels,
                )
                newick_tree = newick.loads(
                    ns, length_parser=lambda x: None if x is None else float(x)
                )[0]
                leaf_names = newick_tree.get_leaf_names()
                assert sorted(leaf_names) == sorted(leaf_labels.values())
                for u in tree.leaves(root):
                    name = leaf_labels[u]
                    node = newick_tree.get_node(name)
                    while u != root:
                        assert node.length == pytest.approx(tree.branch_length(u))
                        node = node.ancestor
                        u = tree.parent(u)
                    assert node.ancestor is None

    @pytest.mark.parametrize("ts", get_example_tree_sequences())
    def test_all_node_labels_dendropy(self, ts):
        node_labels = {u: f"n{u}" for u in range(ts.num_nodes)}
        for tree in ts.trees():
            # Multiroot trees raise an error
            for root in tree.roots:
                ns = tree.newick(
                    root=root,
                    precision=16,
                    node_labels=node_labels,
                )
                dpy_tree = dendropy.Tree.get(
                    data=ns, suppress_internal_node_taxa=False, schema="newick"
                )
                assert_fully_labelled_trees_equal(tree, root, node_labels, dpy_tree)


class TestNexusTreeRoundTrip:
    """
    Test that the nexus format can round-trip tree data under various
    assumptions.
    """

    @pytest.mark.parametrize("ts", get_example_tree_sequences())
    def test_dendropy_defaults(self, ts):
        if any(tree.num_roots != 1 for tree in ts.trees()):
            with pytest.raises(ValueError, match="single root"):
                ts.as_nexus(include_alignments=False)
        else:
            nexus = ts.as_nexus(include_alignments=False)
            tree_list = dendropy.TreeList()
            tree_list.read(
                data=nexus,
                schema="nexus",
                suppress_internal_node_taxa=False,
            )
            assert_dpy_tree_list_equal(ts, tree_list)


class TestNexusIncludeSections:
    """
    Test if we include the sections as expected.
    """

    @tests.cached_example
    def ts(self):
        # 2.00┊   4   ┊
        #     ┊ ┏━┻┓  ┊
        # 1.00┊ ┃  3  ┊
        #     ┊ ┃ ┏┻┓ ┊
        # 0.00┊ 0 1 2 ┊
        #     0      10
        #      |    |
        #  pos 2    9
        #  anc A    T
        ts = tskit.Tree.generate_balanced(3, span=10).tree_sequence
        tables = ts.dump_tables()
        tables.sites.add_row(2, ancestral_state="A")
        tables.sites.add_row(9, ancestral_state="T")
        tables.mutations.add_row(site=0, node=0, derived_state="G")
        tables.mutations.add_row(site=1, node=3, derived_state="C")
        return tables.tree_sequence()

    def test_nexus_default(self):
        ref = "ACTGACTGAC"
        expected = textwrap.dedent(
            """\
            #NEXUS
            BEGIN TAXA;
              DIMENSIONS NTAX=3;
              TAXLABELS n0 n1 n2;
            END;
            BEGIN DATA;
              DIMENSIONS NCHAR=10;
              FORMAT DATATYPE=DNA MISSING=?;
              MATRIX
                n0 ACGGACTGAT
                n1 ACAGACTGAC
                n2 ACAGACTGAC
              ;
            END;
            BEGIN TREES;
              TREE t0^10 = [&R] (n0:2,(n1:1,n2:1):1);
            END;
            """
        )
        assert expected == self.ts().as_nexus(reference_sequence=ref)

    def test_nexus_no_trees(self):
        ref = "ACTGACTGAC"
        expected = textwrap.dedent(
            """\
            #NEXUS
            BEGIN TAXA;
              DIMENSIONS NTAX=3;
              TAXLABELS n0 n1 n2;
            END;
            BEGIN DATA;
              DIMENSIONS NCHAR=10;
              FORMAT DATATYPE=DNA MISSING=?;
              MATRIX
                n0 ACGGACTGAT
                n1 ACAGACTGAC
                n2 ACAGACTGAC
              ;
            END;
            """
        )
        assert expected == self.ts().as_nexus(
            reference_sequence=ref, include_trees=False
        )

    def test_nexus_no_alignments(self):
        expected = textwrap.dedent(
            """\
            #NEXUS
            BEGIN TAXA;
              DIMENSIONS NTAX=3;
              TAXLABELS n0 n1 n2;
            END;
            BEGIN TREES;
              TREE t0^10 = [&R] (n0:2,(n1:1,n2:1):1);
            END;
            """
        )
        assert expected == self.ts().as_nexus(include_alignments=False)

    def test_nexus_no_trees_or_alignments(self):
        expected = textwrap.dedent(
            """\
            #NEXUS
            BEGIN TAXA;
              DIMENSIONS NTAX=3;
              TAXLABELS n0 n1 n2;
            END;
            """
        )
        assert expected == self.ts().as_nexus(
            include_trees=False, include_alignments=False
        )


class TestNewickCodePaths:
    """
    Test that the different code paths we use under the hood lead to
    identical results.
    """

    # NOTE this probabably won't work in general because the C and
    # Python code paths using different rounding algorithms.

    @pytest.mark.parametrize("ts", get_example_tree_sequences())
    def test_default_newick(self, ts):
        for tree in ts.trees():
            for root in tree.roots:
                ns1 = tree.newick(root=root)
                node_labels = {u: str(u + 1) for u in tree.leaves()}
                ns2 = tree.newick(root=root, node_labels=node_labels)
                assert ns1 == ns2

    @pytest.mark.parametrize("ts", get_example_tree_sequences())
    def test_default_as_newick(self, ts):
        for tree in ts.trees():
            for root in tree.roots:
                ns1 = tree.as_newick(root=root)
                node_labels = {u: f"n{u}" for u in tree.tree_sequence.samples()}
                ns2 = tree.as_newick(root=root, node_labels=node_labels)
                assert ns1 == ns2


class TestBalancedBinaryExample:
    #   4
    # ┏━┻┓
    # ┃  3
    # ┃ ┏┻┓
    # 0 1 2
    @tests.cached_example
    def tree(self):
        return tskit.Tree.generate_balanced(3)

    def test_newick_default(self):
        s = (
            "(1:2.00000000000000,(2:1.00000000000000,3:1.00000000000000)"
            ":1.00000000000000);"
        )
        assert self.tree().newick() == s

    def test_as_newick_default(self):
        s = "(n0:2,(n1:1,n2:1):1);"
        assert self.tree().as_newick() == s

    def test_newick_zero_precision(self):
        s = "(1:2,(2:1,3:1):1);"
        assert self.tree().newick(precision=0) == s

    def test_as_newick_zero_precision(self):
        s = "(n0:2,(n1:1,n2:1):1);"
        assert self.tree().as_newick(precision=0) == s

    def test_as_newick_precision_1(self):
        s = "(n0:2.0,(n1:1.0,n2:1.0):1.0);"
        assert self.tree().as_newick(precision=1) == s

    def test_as_newick_precision_1_explicit_labels(self):
        tree = self.tree()
        s = "(x0:2.0,(x1:1.0,x2:1.0):1.0);"
        node_labels = {u: f"x{u}" for u in tree.samples()}
        assert tree.as_newick(precision=1, node_labels=node_labels) == s

    def test_newick_no_branch_lengths(self):
        s = "(1,(2,3));"
        assert self.tree().newick(include_branch_lengths=False) == s

    def test_as_newick_no_branch_lengths(self):
        s = "(n0,(n1,n2));"
        assert self.tree().as_newick(include_branch_lengths=False) == s

    def test_newick_all_node_labels(self):
        s = "(0:2,(1:1,2:1)3:1)4;"
        node_labels = {u: str(u) for u in self.tree().nodes()}
        ns = self.tree().newick(precision=0, node_labels=node_labels)
        assert s == ns

    def test_as_newick_all_node_labels(self):
        s = "(0:2,(1:1,2:1)3:1)4;"
        node_labels = {u: str(u) for u in self.tree().nodes()}
        ns = self.tree().as_newick(node_labels=node_labels)
        assert s == ns

    def test_as_newick_variable_length_node_labels(self):
        s = "(:2,(1:1,22:1)333:1)4444;"
        node_labels = {u: str(u) * u for u in self.tree().nodes()}
        ns = self.tree().as_newick(node_labels=node_labels)
        assert s == ns

    def test_as_newick_empty_node_labels(self):
        s = "(:2,(:1,:1):1);"
        ns = self.tree().as_newick(node_labels={})
        assert s == ns

    def test_newick_partial_node_labels(self):
        s = "(0:2,(1:1,2:1)3:1);"
        node_labels = {u: str(u) for u in self.tree().preorder()[1:]}
        ns = self.tree().newick(precision=0, node_labels=node_labels)
        assert s == ns

    def test_newick_root(self):
        s = "(2:1,3:1);"
        assert self.tree().newick(root=3, precision=0) == s

    def test_as_nexus_default(self):
        ts = self.tree().tree_sequence
        expected = textwrap.dedent(
            """\
            #NEXUS
            BEGIN TAXA;
              DIMENSIONS NTAX=3;
              TAXLABELS n0 n1 n2;
            END;
            BEGIN TREES;
              TREE t0^1 = [&R] (n0:2,(n1:1,n2:1):1);
            END;
        """
        )
        assert ts.as_nexus() == expected

    def test_as_nexus_precision_1(self):
        ts = self.tree().tree_sequence
        expected = textwrap.dedent(
            """\
            #NEXUS
            BEGIN TAXA;
              DIMENSIONS NTAX=3;
              TAXLABELS n0 n1 n2;
            END;
            BEGIN TREES;
              TREE t0.0^1.0 = [&R] (n0:2.0,(n1:1.0,n2:1.0):1.0);
            END;
        """
        )
        assert ts.as_nexus(precision=1) == expected


class TestFractionalBranchLengths:
    # 0.67┊   4   ┊
    #     ┊ ┏━┻┓  ┊
    # 0.33┊ ┃  3  ┊
    #     ┊ ┃ ┏┻┓ ┊
    # 0.00┊ 0 1 2 ┊
    #     0       1
    @tests.cached_example
    def tree(self):
        return tskit.Tree.generate_balanced(3, branch_length=1 / 3)

    def test_newick_default(self):
        s = (
            "(1:0.66666666666667,(2:0.33333333333333,3:0.33333333333333)"
            ":0.33333333333333);"
        )
        assert self.tree().newick() == s

    def test_as_newick_default(self):
        s = (
            "(n0:0.66666666666666663,(n1:0.33333333333333331,"
            "n2:0.33333333333333331):0.33333333333333331);"
        )
        assert self.tree().as_newick() == s

    def test_c_and_py_output_equal(self):
        t = self.tree()
        assert t.as_newick() == t.as_newick(
            node_labels={u: f"n{u}" for u in t.samples()}
        )

    def test_as_newick_precision_3(self):
        s = "(n0:0.667,(n1:0.333,n2:0.333):0.333);"
        assert self.tree().as_newick(precision=3) == s

    def test_newick_precision_3(self):
        s = "(1:0.667,(2:0.333,3:0.333):0.333);"
        assert self.tree().newick(precision=3) == s

    def test_as_newick_precision_3_labels(self):
        node_labels = {u: f"n{u}" * 3 for u in self.tree().nodes()}
        s = "(n0n0n0:0.667,(n1n1n1:0.333,n2n2n2:0.333)n3n3n3:0.333)n4n4n4;"
        assert self.tree().as_newick(precision=3, node_labels=node_labels) == s


class TestLargeBranchLengths:
    # 2000000000.00┊   4   ┊
    #              ┊ ┏━┻┓  ┊
    # 1000000000.00┊ ┃  3  ┊
    #              ┊ ┃ ┏┻┓ ┊
    # 0.00         ┊ 0 1 2 ┊
    #              0       1
    @tests.cached_example
    def tree(self):
        return tskit.Tree.generate_balanced(3, branch_length=1e9)

    def test_newick_default(self):
        s = (
            "(1:2000000000.00000000000000,(2:1000000000.00000000000000,"
            "3:1000000000.00000000000000):1000000000.00000000000000);"
        )
        assert self.tree().newick() == s

    def test_as_newick_default(self):
        s = "(n0:2000000000,(n1:1000000000,n2:1000000000):1000000000);"
        assert self.tree().as_newick() == s

    def test_newick_precision_3(self):
        s = "(1:2000000000.000,(2:1000000000.000,3:1000000000.000):1000000000.000);"
        assert self.tree().newick(precision=3) == s

    def test_as_newick_precision_3(self):
        s = "(n0:2000000000.000,(n1:1000000000.000,n2:1000000000.000):1000000000.000);"
        assert self.tree().as_newick(precision=3) == s


class TestInternalSampleExample:
    #   4
    # ┏━┻┓
    # ┃ *3*
    # ┃ ┏┻┓
    # 0 1 2
    # Leaves are samples but 3 is also a sample.
    @tests.cached_example
    def tree(self):
        tables = tskit.Tree.generate_balanced(3).tree_sequence.dump_tables()
        flags = tables.nodes.flags
        flags[3] = 1
        tables.nodes.flags = flags
        return tables.tree_sequence().first()

    def test_newick_default(self):
        # Old newick method doesn't do anything with internal sample
        s = (
            "(1:2.00000000000000,(2:1.00000000000000,3:1.00000000000000)"
            ":1.00000000000000);"
        )
        assert self.tree().newick() == s

    def test_as_newick_default(self):
        # Samples are labelled by default, not leaves.
        s = "(n0:2,(n1:1,n2:1)n3:1);"
        assert self.tree().as_newick() == s

    def test_dendropy_parsing(self):
        dpy_tree = dendropy.Tree.get(
            data=self.tree().as_newick(),
            schema="newick",
            suppress_internal_node_taxa=False,
            rooting="default-rooted",
        )
        # Just check that we can correctly parse out the internal sample.
        # More exhaustive testing of properties is done elsewhere.
        n3 = dpy_tree.find_node_with_taxon_label("n3")
        n1 = dpy_tree.find_node_with_taxon_label("n1")
        assert n1.parent_node is n3
        n2 = dpy_tree.find_node_with_taxon_label("n2")
        assert n2.parent_node is n3


class TestAncientSampleExample:
    #     8
    #  ┏━━┻━┓
    #  5    7
    # ┏┻┓ ┏━┻┓
    # 0 1 ┃  6
    #     ┃ ┏┻┓
    #     2 3 4
    @tests.cached_example
    def tree(self):
        tables = tskit.Tree.generate_balanced(5).tree_sequence.dump_tables()
        time = tables.nodes.time
        time[0] = 1
        time[1] = 1
        time[5] = 2
        tables.nodes.time = time
        tables.sort()
        return tables.tree_sequence().first()

    def test_as_newick(self):
        s = "((n0:1,n1:1):1,(n2:2,(n3:1,n4:1):1):1);"
        assert self.tree().as_newick() == s

    def test_newick(self):
        s = "((1:1,2:1):1,(3:2,(4:1,5:1):1):1);"
        assert self.tree().newick(precision=0) == s


class TestNonSampleLeafExample:
    #   4
    # ┏━┻┓
    # ┃  3
    # ┃ ┏┻┓
    # |0|1 2
    # Leaf 0 is *not* a sample
    @tests.cached_example
    def tree(self):
        tables = tskit.Tree.generate_balanced(3).tree_sequence.dump_tables()
        flags = tables.nodes.flags
        flags[0] = 0
        tables.nodes.flags = flags
        return tables.tree_sequence().first()

    def test_newick(self):
        # newick method doesn't think about samples at all.
        s = "(1:2,(2:1,3:1):1);"
        assert self.tree().newick(precision=0) == s

    def test_as_newick_default(self):
        # We don't label node 0 even though it's a leaf.
        s = "(:2,(n1:1,n2:1):1);"
        assert self.tree().as_newick() == s

    def test_dendropy_parsing(self):
        # This odd topology parses OK with dendropy
        dpy_tree = dendropy.Tree.get(
            data=self.tree().as_newick(),
            schema="newick",
            suppress_internal_node_taxa=False,
            rooting="default-rooted",
        )
        n1 = dpy_tree.find_node_with_taxon_label("n1")
        assert n1 is not None
        n2 = dpy_tree.find_node_with_taxon_label("n2")
        assert n2 is not None
        leaves = dpy_tree.leaf_nodes()
        assert len(leaves) == 3
        leaves = set(leaves)
        leaves.remove(n1)
        leaves.remove(n2)
        n0 = leaves.pop()
        assert n0.taxon is None

    def test_newick_lib_parsing(self):
        newick_tree = newick.loads(self.tree().as_newick())[0]
        leaf_names = newick_tree.get_leaf_names()
        assert len(leaf_names) == 3
        assert "n1" in leaf_names
        assert "n2" in leaf_names
        assert None in leaf_names


class TestNonBinaryExample:
    # 2.00┊        12         ┊
    #     ┊   ┏━━━━━╋━━━━━┓   ┊
    # 1.00┊   9    10    11   ┊
    #     ┊ ┏━╋━┓ ┏━╋━┓ ┏━╋━┓ ┊
    # 0.00┊ 0 1 2 3 4 5 6 7 8 ┊
    #     0                   1
    @tests.cached_example
    def tree(self):
        return tskit.Tree.generate_balanced(9, arity=3)

    def test_as_newick(self):
        s = "((n0:1,n1:1,n2:1):1,(n3:1,n4:1,n5:1):1,(n6:1,n7:1,n8:1):1);"
        assert self.tree().as_newick() == s

    def test_newick(self):
        s = "((1:1,2:1,3:1):1,(4:1,5:1,6:1):1,(7:1,8:1,9:1):1);"
        assert self.tree().newick(precision=0) == s


class TestMultiRootExample:
    #
    # 1.00┊   9    10    11   ┊
    #     ┊ ┏━╋━┓ ┏━╋━┓ ┏━╋━┓ ┊
    # 0.00┊ 0 1 2 3 4 5 6 7 8 ┊
    #     0                   1
    @tests.cached_example
    def tree(self):
        tables = tskit.Tree.generate_balanced(9, arity=3).tree_sequence.dump_tables()
        edges = tables.edges.copy()
        tables.edges.clear()
        for edge in edges:
            if edge.parent != 12:
                tables.edges.append(edge)
        return tables.tree_sequence().first()

    def test_as_newick_fails(self):
        with pytest.raises(ValueError, match="single root"):
            self.tree().as_newick()

    def test_newick_fails(self):
        with pytest.raises(ValueError, match="single root"):
            self.tree().newick()

    def test_as_newick_per_root(self):
        t = self.tree()
        assert t.as_newick(root=9) == "(n0:1,n1:1,n2:1);"
        assert t.as_newick(root=10) == "(n3:1,n4:1,n5:1);"
        assert t.as_newick(root=11) == "(n6:1,n7:1,n8:1);"

    def test_newick_per_root(self):
        t = self.tree()
        assert t.newick(root=9, precision=0) == "(1:1,2:1,3:1);"
        assert t.newick(root=10, precision=0) == "(4:1,5:1,6:1);"
        assert t.newick(root=11, precision=0) == "(7:1,8:1,9:1);"


class TestLineTree:
    # 3.00┊ 3 ┊
    #     ┊ ┃ ┊
    # 2.00┊ 2 ┊
    #     ┊ ┃ ┊
    # 1.00┊ 1 ┊
    #     ┊ ┃ ┊
    # 0.00┊ 0 ┊
    #     0   1

    @tests.cached_example
    def tree(self):
        tables = tskit.TableCollection(1.0)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        for j in range(3):
            tables.nodes.add_row(flags=0, time=j + 1)
            tables.edges.add_row(left=0, right=1, parent=j + 1, child=j)
        tables.sort()
        return tables.tree_sequence().first()

    def test_newick(self):
        s = "(((1:1.00000000000000):1.00000000000000):1.00000000000000);"
        assert s == self.tree().newick()

    def test_as_newick(self):
        s = "(((n0:1):1):1);"
        assert s == self.tree().as_newick()

    def test_dendropy_parsing(self):
        dpy_tree = dendropy.Tree.get(
            data=self.tree().as_newick(),
            schema="newick",
            rooting="default-rooted",
        )
        n0 = dpy_tree.find_node_with_taxon_label("n0")
        assert n0 is not None
        assert n0.edge_length == 1


class TestEmptyTree:
    # The empty tree sequence has no nodes and so there's zero roots.
    # This gets caught by the "has_single_root" error check, which is
    # probably not right (we should just return the empty string).
    # It's not an important corner case though, so probably not worth
    # worrying about.
    def tree(self):
        tables = tskit.TableCollection(1.0)
        return tables.tree_sequence().first()

    def test_newick(self):
        with pytest.raises(ValueError, match="single root"):
            self.tree().newick()

    def test_as_newick(self):
        with pytest.raises(ValueError, match="single root"):
            self.tree().as_newick()

    def test_as_nexus(self):
        with pytest.raises(ValueError, match="single root"):
            self.tree().tree_sequence.as_nexus()


class TestSingleNodeTree:
    def tree(self):
        tables = tskit.TableCollection(1.0)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        return tables.tree_sequence().first()

    def test_newick(self):
        assert self.tree().newick() == "1;"

    def test_as_newick(self):
        assert self.tree().as_newick() == "n0;"

    def test_as_newick_labels(self):
        assert self.tree().as_newick(node_labels={0: "ABCDE"}) == "ABCDE;"


class TestIntegerTreeSequence:
    # 3.00┊   5   ┊       ┊
    #     ┊ ┏━┻┓  ┊       ┊
    # 2.00┊ ┃  4  ┊   4   ┊
    #     ┊ ┃ ┏┻┓ ┊  ┏┻━┓ ┊
    # 1.00┊ ┃ ┃ ┃ ┊  3  ┃ ┊
    #     ┊ ┃ ┃ ┃ ┊ ┏┻┓ ┃ ┊
    # 0.00┊ 0 1 2 ┊ 0 2 1 ┊
    #     0       2      10
    @tests.cached_example
    def ts(self):
        nodes = io.StringIO(
            """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       1           0
        3       0           1
        4       0           2
        5       0           3
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        2.0     10      3       0
        2.0     10      3       2
        0.0     10      4       1
        0.0     2.0     4       2
        2.0     10      4       3
        0.0     2.0     5       0
        0.0     2.0     5       4
        """
        )
        return tskit.load_text(nodes=nodes, edges=edges, strict=False)

    def test_nexus_defaults(self):
        ts = self.ts()
        expected = textwrap.dedent(
            """\
            #NEXUS
            BEGIN TAXA;
              DIMENSIONS NTAX=3;
              TAXLABELS n0 n1 n2;
            END;
            BEGIN TREES;
              TREE t0^2 = [&R] (n0:3,(n1:2,n2:2):1);
              TREE t2^10 = [&R] (n1:2,(n0:1,n2:1):1);
            END;
            """
        )
        assert ts.as_nexus() == expected

    def test_nexus_precision_2(self):
        ts = self.ts()
        expected = textwrap.dedent(
            """\
            #NEXUS
            BEGIN TAXA;
              DIMENSIONS NTAX=3;
              TAXLABELS n0 n1 n2;
            END;
            BEGIN TREES;
              TREE t0.00^2.00 = [&R] (n0:3.00,(n1:2.00,n2:2.00):1.00);
              TREE t2.00^10.00 = [&R] (n1:2.00,(n0:1.00,n2:1.00):1.00);
            END;
            """
        )
        assert ts.as_nexus(precision=2) == expected

    @pytest.mark.parametrize("precision", [None, 0, 1, 3])
    def test_file_version_identical(self, precision):
        ts = self.ts()
        out = io.StringIO()
        ts.write_nexus(out, precision=precision)
        assert out.getvalue() == ts.as_nexus(precision=precision)


class TestFloatTimeTreeSequence:
    # 3.25┊   5   ┊       ┊
    #     ┊ ┏━┻┓  ┊       ┊
    # 2.00┊ ┃  4  ┊   4   ┊
    #     ┊ ┃ ┏┻┓ ┊  ┏┻━┓ ┊
    # 1.00┊ ┃ ┃ ┃ ┊  3  ┃ ┊
    #     ┊ ┃ ┃ ┃ ┊ ┏┻┓ ┃ ┊
    # 0.00┊ 0 1 2 ┊ 0 2 1 ┊
    #     0       2      10
    @tests.cached_example
    def ts(self):
        nodes = io.StringIO(
            """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       1           0
        3       0           1
        4       0           2
        5       0           3.25
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        2.0     10      3       0
        2.0     10      3       2
        0.0     10      4       1
        0.0     2.0     4       2
        2.0     10      4       3
        0.0     2.0     5       0
        0.0     2.0     5       4
        """
        )
        return tskit.load_text(nodes=nodes, edges=edges, strict=False)

    def test_nexus_defaults(self):
        ts = self.ts()
        expected = textwrap.dedent(
            """\
            #NEXUS
            BEGIN TAXA;
              DIMENSIONS NTAX=3;
              TAXLABELS n0 n1 n2;
            END;
            BEGIN TREES;
              TREE t0^2 = [&R] (n0:3.25000000000000000,(n1:2.00000000000000000,n2:2.00000000000000000):1.25000000000000000);
              TREE t2^10 = [&R] (n1:2.00000000000000000,(n0:1.00000000000000000,n2:1.00000000000000000):1.00000000000000000);
            END;
            """  # noqa: B950
        )
        assert ts.as_nexus() == expected

    def test_nexus_precision_2(self):
        ts = self.ts()
        expected = textwrap.dedent(
            """\
            #NEXUS
            BEGIN TAXA;
              DIMENSIONS NTAX=3;
              TAXLABELS n0 n1 n2;
            END;
            BEGIN TREES;
              TREE t0.00^2.00 = [&R] (n0:3.25,(n1:2.00,n2:2.00):1.25);
              TREE t2.00^10.00 = [&R] (n1:2.00,(n0:1.00,n2:1.00):1.00);
            END;
            """
        )
        assert ts.as_nexus(precision=2) == expected


class TestFloatPositionTreeSequence:
    # 3.00┊   5   ┊       ┊
    #     ┊ ┏━┻┓  ┊       ┊
    # 2.00┊ ┃  4  ┊   4   ┊
    #     ┊ ┃ ┏┻┓ ┊  ┏┻━┓ ┊
    # 1.00┊ ┃ ┃ ┃ ┊  3  ┃ ┊
    #     ┊ ┃ ┃ ┃ ┊ ┏┻┓ ┃ ┊
    # 0.00┊ 0 1 2 ┊ 0 2 1 ┊
    #     0      2.5      10
    @tests.cached_example
    def ts(self):
        nodes = io.StringIO(
            """\
        id      is_sample   time
        0       1           0
        1       1           0
        2       1           0
        3       0           1
        4       0           2
        5       0           3
        """
        )
        edges = io.StringIO(
            """\
        left    right   parent  child
        2.5     10      3       0
        2.5     10      3       2
        0.0     10      4       1
        0.0     2.5     4       2
        2.5     10      4       3
        0.0     2.5     5       0
        0.0     2.5     5       4
        """
        )
        return tskit.load_text(nodes=nodes, edges=edges, strict=False)

    def test_nexus_defaults(self):
        ts = self.ts()
        expected = textwrap.dedent(
            """\
            #NEXUS
            BEGIN TAXA;
              DIMENSIONS NTAX=3;
              TAXLABELS n0 n1 n2;
            END;
            BEGIN TREES;
              TREE t0.00000000000000000^2.50000000000000000 = [&R] (n0:3,(n1:2,n2:2):1);
              TREE t2.50000000000000000^10.00000000000000000 = [&R] (n1:2,(n0:1,n2:1):1);
            END;
            """  # noqa: B950
        )
        assert ts.as_nexus() == expected

    def test_nexus_precision_2(self):
        ts = self.ts()
        expected = textwrap.dedent(
            """\
            #NEXUS
            BEGIN TAXA;
              DIMENSIONS NTAX=3;
              TAXLABELS n0 n1 n2;
            END;
            BEGIN TREES;
              TREE t0.00^2.50 = [&R] (n0:3.00,(n1:2.00,n2:2.00):1.00);
              TREE t2.50^10.00 = [&R] (n1:2.00,(n0:1.00,n2:1.00):1.00);
            END;
            """
        )
        assert ts.as_nexus(precision=2) == expected


def test_newick_buffer_too_small_bug():
    nodes = io.StringIO(
        """\
    id  is_sample   population individual time
    0       1       0       -1      0.00000000000000
    1       1       0       -1      0.00000000000000
    2       1       0       -1      0.00000000000000
    3       1       0       -1      0.00000000000000
    4       0       0       -1      0.21204940078588
    5       0       0       -1      0.38445004304611
    6       0       0       -1      0.83130278081275
    """
    )
    edges = io.StringIO(
        """\
    id      left            right           parent  child
    0       0.00000000      1.00000000      4       0
    1       0.00000000      1.00000000      4       2
    2       0.00000000      1.00000000      5       1
    3       0.00000000      1.00000000      5       3
    4       0.00000000      1.00000000      6       4
    5       0.00000000      1.00000000      6       5
    """
    )
    ts = tskit.load_text(nodes, edges, sequence_length=1, strict=False)
    tree = ts.first()
    for precision in range(18):
        newick_c = tree.newick(precision=precision)
        node_labels = {u: str(u + 1) for u in ts.samples()}
        newick_py = tree.newick(precision=precision, node_labels=node_labels)
        assert newick_c == newick_py


class TestWrapText:
    def test_even_split(self):
        example = "ABCDEFGH"
        result = list(tskit.text_formats.wrap_text(example, 4))
        assert result == ["ABCD", "EFGH"]

    def test_non_even_split(self):
        example = "ABCDEFGH"
        result = list(tskit.text_formats.wrap_text(example, 3))
        assert result == ["ABC", "DEF", "GH"]

    def test_width_one(self):
        example = "ABCDEFGH"
        result = list(tskit.text_formats.wrap_text(example, 1))
        assert result == ["A", "B", "C", "D", "E", "F", "G", "H"]

    def test_width_full_length(self):
        example = "ABCDEFGH"
        result = list(tskit.text_formats.wrap_text(example, 8))
        assert result == ["ABCDEFGH"]

    def test_width_more_than_length(self):
        example = "ABCDEFGH"
        result = list(tskit.text_formats.wrap_text(example, 100))
        assert result == ["ABCDEFGH"]

    def test_width_0(self):
        example = "ABCDEFGH"
        result = list(tskit.text_formats.wrap_text(example, 0))
        assert result == ["ABCDEFGH"]

    @pytest.mark.parametrize("width", [-1, -2, -8, -100])
    def test_width_negative(self, width):
        # Just documenting that the current implementation works for negative
        # values fine.
        example = "ABCDEFGH"
        result = list(tskit.text_formats.wrap_text(example, width))
        assert result == ["ABCDEFGH"]


class TestFastaLineLength:
    """
    Tests if the fasta file produced has the correct line lengths for
    default, custom, and no-wrapping options.
    """

    def verify_line_length(self, length, wrap_width=60):
        # set up data
        ts = alignment_example(length)
        output = io.StringIO()
        ts.write_fasta(output, wrap_width=wrap_width)
        output.seek(0)

        # check if length perfectly divisible by wrap_width or not and thus
        # expected line lengths
        no_hanging_line = True
        if wrap_width == 0:
            lines_expect = 1
            # for easier code in testing function, redefine wrap_width as
            # full length, ok as called write already
            wrap_width = length
        elif length % wrap_width == 0:
            lines_expect = length // wrap_width
        else:
            lines_expect = length // wrap_width + 1
            extra_line_length = length % wrap_width
            no_hanging_line = False

        seq_line_counter = 0
        id_lines = 0
        for line in output:
            # testing correct characters per sequence line
            if line[0] != ">":
                seq_line_counter += 1
                line_chars = len(line.strip("\n"))
                # test full default width lines
                if seq_line_counter < lines_expect:
                    assert wrap_width == line_chars
                elif no_hanging_line:
                    assert wrap_width == line_chars
                # test extra line if not perfectly divided by wrap_width
                else:
                    assert extra_line_length == line_chars
            # testing correct number of lines per sequence and correct num sequences
            else:
                id_lines += 1
                if seq_line_counter > 0:
                    assert lines_expect == seq_line_counter
                    seq_line_counter = 0
        assert id_lines == ts.num_samples

    def test_wrap_length_default_easy(self):
        # default wrap width (60) perfectly divides sequence length
        self.verify_line_length(length=300)

    def test_wrap_length_default_harder(self):
        # default wrap_width imperfectly divides sequence length
        self.verify_line_length(length=280)

    def test_wrap_length_custom_easy(self):
        # custom wrap_width, perfectly divides
        self.verify_line_length(length=100, wrap_width=20)

    def test_wrap_length_custom_harder(self):
        # custom wrap_width, imperfectly divides
        self.verify_line_length(length=100, wrap_width=30)

    def test_wrap_length_no_wrap(self):
        # no wrapping set by wrap_width = 0
        self.verify_line_length(length=100, wrap_width=0)

    def test_negative_wrap(self):
        ts = alignment_example(10)
        with pytest.raises(ValueError, match="non-negative integer"):
            ts.as_fasta(wrap_width=-1)

    def test_floating_wrap(self):
        ts = alignment_example(10)
        with pytest.raises(ValueError):
            ts.as_fasta(wrap_width=1.1)

    def test_numpy_wrap(self):
        ts = alignment_example(10)
        x1 = ts.as_fasta(wrap_width=4)
        x2 = ts.as_fasta(wrap_width=np.array([4.0])[0])
        assert x1 == x2


class TestFileTextOutputEqual:
    @tests.cached_example
    def ts(self):
        return alignment_example(20)

    def test_fasta_defaults(self):
        ts = self.ts()
        buff = io.StringIO()
        ts.write_fasta(buff)
        assert buff.getvalue() == ts.as_fasta()

    def test_fasta_wrap_width(self):
        ts = self.ts()
        buff = io.StringIO()
        ts.write_fasta(buff, wrap_width=4)
        assert buff.getvalue() == ts.as_fasta(wrap_width=4)

    def test_nexus_defaults(self):
        ts = self.ts()
        buff = io.StringIO()
        ts.write_nexus(buff)
        assert buff.getvalue() == ts.as_nexus()

    def test_nexus_precision(self):
        ts = self.ts()
        buff = io.StringIO()
        ts.write_nexus(buff, precision=2)
        assert buff.getvalue() == ts.as_nexus(precision=2)


class TestFlexibleFileArgFasta:
    @tests.cached_example
    def ts(self):
        return alignment_example(20)

    def test_pathlib(self, tmp_path):
        path = tmp_path / "file.fa"
        ts = self.ts()
        ts.write_fasta(path)
        with open(path) as f:
            assert f.read() == ts.as_fasta()

    def test_path_str(self, tmp_path):
        path = str(tmp_path / "file.fa")
        ts = self.ts()
        ts.write_fasta(path)
        with open(path) as f:
            assert f.read() == ts.as_fasta()

    def test_fileobj(self, tmp_path):
        path = tmp_path / "file.fa"
        ts = self.ts()
        with open(path, "w") as f:
            ts.write_fasta(f)
        with open(path) as f:
            assert f.read() == ts.as_fasta()


class TestFlexibleFileArgNexus:
    @tests.cached_example
    def ts(self):
        return alignment_example(20)

    def test_pathlib(self, tmp_path):
        path = tmp_path / "file.nex"
        ts = self.ts()
        ts.write_nexus(path)
        with open(path) as f:
            assert f.read() == ts.as_nexus()

    def test_path_str(self, tmp_path):
        path = str(tmp_path / "file.nex")
        ts = self.ts()
        ts.write_nexus(path)
        with open(path) as f:
            assert f.read() == ts.as_nexus()

    def test_fileobj(self, tmp_path):
        path = tmp_path / "file.nex"
        ts = self.ts()
        with open(path, "w") as f:
            ts.write_nexus(f)
        with open(path) as f:
            assert f.read() == ts.as_nexus()


def get_alignment_map(ts, reference_sequence=None):
    alignments = ts.alignments(reference_sequence=reference_sequence)
    return {f"n{u}": alignment for u, alignment in zip(ts.samples(), alignments)}


class TestFastaBioPythonRoundTrip:
    """
    Tests that output from our code is read in by available software packages
    Here test for compatability with biopython processing - Bio.SeqIO
    """

    def verify(self, ts, wrap_width=60, reference_sequence=None):
        text = ts.as_fasta(wrap_width=wrap_width, reference_sequence=reference_sequence)
        bio_map = {
            k: v.seq
            for k, v in SeqIO.to_dict(SeqIO.parse(io.StringIO(text), "fasta")).items()
        }
        assert bio_map == get_alignment_map(ts, reference_sequence)

    def test_equal_lines(self):
        # sequence length perfectly divisible by wrap_width
        ts = alignment_example(300)
        self.verify(ts)

    def test_unequal_lines(self):
        # sequence length not perfectly divisible by wrap_width
        ts = alignment_example(280)
        self.verify(ts)

    def test_unwrapped(self):
        # sequences not wrapped
        ts = alignment_example(300)
        self.verify(ts, wrap_width=0)

    def test_A_reference(self):
        ts = alignment_example(20)
        self.verify(ts, reference_sequence="A" * 20)

    @pytest.mark.skip("Missing data in alignments: #1896")
    def test_missing_data(self):
        self.verify(missing_data_example())


class TestFastaDendropyRoundTrip:
    def parse(self, fasta):
        d = dendropy.DnaCharacterMatrix.get(data=fasta, schema="fasta")
        return {str(k.label): str(v) for k, v in d.items()}

    def test_wrapped(self):
        ts = alignment_example(300)
        text = ts.as_fasta()
        alignment_map = self.parse(text)
        assert get_alignment_map(ts) == alignment_map

    def test_unwrapped(self):
        ts = alignment_example(300)
        text = ts.as_fasta(wrap_width=0)
        alignment_map = self.parse(text)
        assert get_alignment_map(ts) == alignment_map

    def test_no_reference(self):
        ts = alignment_example(100, include_reference=False)
        text = ts.as_fasta()
        alignment_map = self.parse(text)
        assert get_alignment_map(ts) == alignment_map

    @pytest.mark.skip("Missing data in alignments: #1896")
    def test_missing_data(self):
        ts = missing_data_example()
        text = ts.as_fasta()
        alignment_map = self.parse(text)
        assert get_alignment_map(ts) == alignment_map


class TestDendropyMissingReference:
    # 2.00┊   4   ┊
    #     ┊ ┏━┻┓  ┊
    # 1.00┊ ┃  3  ┊
    #     ┊ ┃ ┏┻┓ ┊
    # 0.00┊ 0 1 2 ┊
    #     0       10
    #      |     |
    #  pos 2     9
    #  anc A     T

    def ts(self):
        ts = tskit.Tree.generate_balanced(3, span=10).tree_sequence
        tables = ts.dump_tables()
        tables.sites.add_row(2, ancestral_state="A")
        tables.sites.add_row(9, ancestral_state="T")
        tables.mutations.add_row(site=0, node=0, derived_state="G")
        tables.mutations.add_row(site=1, node=3, derived_state="C")
        return tables.tree_sequence()

    def assert_missing_data_encoded(self, d):
        assert d.sequence_size == 10
        assert str(d["n0"][2]) == "G"
        assert str(d["n0"][9]) == "T"
        assert str(d["n1"][2]) == "A"
        assert str(d["n1"][9]) == "C"
        assert str(d["n2"][2]) == "A"
        assert str(d["n2"][9]) == "C"
        for a in d.values():
            for j in range(d.sequence_size):
                if j in [2, 9]:
                    assert (
                        a[j].state_denomination
                        == dendropy.StateAlphabet.FUNDAMENTAL_STATE
                    )
                else:
                    assert (
                        a[j].state_denomination
                        == dendropy.StateAlphabet.AMBIGUOUS_STATE
                    )

    def test_fasta(self):
        ts = self.ts()
        text = ts.as_fasta()
        d = dendropy.DnaCharacterMatrix.get(data=text, schema="fasta")
        self.assert_missing_data_encoded(d)
        assert str(d["n0"][0]) == "N"

    def test_fasta_missing_question(self):
        ts = self.ts()
        text = ts.as_fasta(missing_data_character="?")
        d = dendropy.DnaCharacterMatrix.get(data=text, schema="fasta")
        self.assert_missing_data_encoded(d)
        assert str(d["n0"][0]) == "?"

    def test_nexus(self):
        ts = self.ts()
        text = ts.as_nexus()
        d = dendropy.DnaCharacterMatrix.get(data=text, schema="nexus")
        self.assert_missing_data_encoded(d)
        assert str(d["n0"][0]) == "?"

    def test_nexus_missing_N(self):
        ts = self.ts()
        text = ts.as_nexus(missing_data_character="N")
        d = dendropy.DnaCharacterMatrix.get(data=text, schema="nexus")
        self.assert_missing_data_encoded(d)
        assert str(d["n0"][0]) == "N"


@pytest.mark.skip("Missing data in alignments: #1896")
class TestDendropyMissingData:
    """
    Test that we detect missing data correctly in dendropy under
    various combinations of options.
    """

    # 2.00┊   4     ┊
    #     ┊ ┏━┻┓    ┊
    # 1.00┊ ┃  3    ┊
    #     ┊ ┃ ┏┻┓   ┊
    # 0.00┊ 0 1 2 5 ┊
    #     0        10
    #      |      |
    #  pos 2      9
    #  anc A      T

    def ts(self):
        ts = tskit.Tree.generate_balanced(3, span=10).tree_sequence
        tables = ts.dump_tables()
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.sites.add_row(2, ancestral_state="A")
        tables.sites.add_row(9, ancestral_state="T")
        tables.mutations.add_row(site=0, node=0, derived_state="G")
        tables.mutations.add_row(site=1, node=3, derived_state="C")
        return tables.tree_sequence()

    def assert_missing_data_encoded_A_ref(self, d):
        assert d.sequence_size == 10
        assert str(d["n0"]) == "AAGAAAAAAT"
        assert str(d["n1"]) == "AAAAAAAAAC"
        assert str(d["n2"]) == "AAAAAAAAAC"
        for a in [d["n0"], d["n1"], d["n2"]]:
            assert all(
                a[j].state_denomination == dendropy.StateAlphabet.FUNDAMENTAL_STATE
                for j in range(10)
            )
        # Do we detect that we have an ambiguous state for the missing sample?

        a5 = d["n5"]
        # a5 is missing along the full length of the genome, so all sites are
        # missing.
        assert all(
            a5.state_denomination == dendropy.StateAlphabet.AMBIGUOUS_STATE
            for j in range(10)
        )

    def test_fasta_defaults_A_ref(self):
        ts = self.ts()
        ref = "A" * int(ts.sequence_length)
        text = ts.as_fasta(reference_sequence=ref)
        d = dendropy.DnaCharacterMatrix.get(data=text, schema="fasta")
        self.assert_missing_data_encoded_A_ref(d)

    def test_nexus_defaults_A_ref(self):
        ts = self.ts()
        ref = "A" * int(ts.sequence_length)
        text = ts.as_nexus(reference_sequence=ref, include_trees=False)
        d = dendropy.DnaCharacterMatrix.get(data=text, schema="nexus")
        self.assert_missing_data_encoded_A_ref(d)
