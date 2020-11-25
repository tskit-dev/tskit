#
# MIT License
#
# Copyright (c) 2020 Tskit Developers
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
Module for ranking and unranking trees. Trees are considered only
leaf-labelled and unordered, so order of children does not influence equality.
"""
import collections
import functools
import heapq
import itertools
import json
import random

import attr
import numpy as np

import tskit


def equal_chunks(lst, k):
    """
    Yield k successive equally sized chunks from lst of size n.

    If k >= n, we return n chunks of size 1.

    Otherwise, we always return k chunks. The first k - 1 chunks will
    contain exactly n // k items, and the last chunk the remainder.
    """
    n = len(lst)
    if k <= 0 or int(k) != k:
        raise ValueError("Number of chunks must be a positive integer")

    if n > 0:
        chunk_size = max(1, n // k)
        offset = 0
        j = 0
        while offset < n - chunk_size and j < k - 1:
            yield lst[offset : offset + chunk_size]
            offset += chunk_size
            j += 1
        yield lst[offset:]


@attr.s(eq=False)
class TreeNode:
    """
    Simple linked tree class used to generate tree topologies.
    """

    parent = attr.ib(default=None)
    children = attr.ib(factory=list)
    label = attr.ib(default=None)

    def as_tables(self, *, num_leaves, span, branch_length):
        """
        Convert the tree rooted at this node into an equivalent
        TableCollection. Internal nodes are allocated in postorder.
        """
        tables = tskit.TableCollection(span)
        for _ in range(num_leaves):
            tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)

        def assign_internal_labels(node):
            if len(node.children) == 0:
                node.time = 0
            else:
                max_child_time = 0
                for child in node.children:
                    assign_internal_labels(child)
                    max_child_time = max(max_child_time, child.time)
                node.time = max_child_time + branch_length
                node.label = tables.nodes.add_row(time=node.time)
                for child in node.children:
                    tables.edges.add_row(0, span, node.label, child.label)

        # Do a postorder traversal to assign the internal node labels and times.
        assign_internal_labels(self)
        tables.sort()
        return tables

    @staticmethod
    def random_binary_tree(leaf_labels, rng):
        """
        Returns a random binary tree where the leaves have the specified
        labels using the specified random.Random instance. The root node
        of this tree is returned.

        Based on the description of Remy's method of generating "decorated"
        random binary trees in TAOCP 7.2.1.6. This is not a direct
        implementation of Algorithm R, because we are interested in
        the leaf node labellings.

        The pre-fascicle text is available here, page 16:
        http://www.cs.utsa.edu/~wagner/knuth/fasc4a.pdf
        """
        nodes = [TreeNode(label=leaf_labels[0])]
        for label in leaf_labels[1:]:
            # Choose a node x randomly and insert a new internal node above
            # it with the (n + 1)th labelled leaf as its sibling.
            x = rng.choice(nodes)
            new_leaf = TreeNode(label=label)
            new_internal = TreeNode(parent=x.parent, children=[x, new_leaf])
            if x.parent is not None:
                index = x.parent.children.index(x)
                x.parent.children[index] = new_internal
            rng.shuffle(new_internal.children)
            x.parent = new_internal
            new_leaf.parent = new_internal
            nodes.extend([new_leaf, new_internal])

        root = nodes[0]
        while root.parent is not None:
            root = root.parent

        # Canonicalise the order of the children within a node. This
        # is given by (num_leaves, min_label). See also the
        # RankTree.canonical_order function for the definition of
        # how these are ordered during rank/unrank.

        def reorder_children(node):
            if len(node.children) == 0:
                return 1, node.label
            keys = [reorder_children(child) for child in node.children]
            if keys[0] > keys[1]:
                node.children = node.children[::-1]
            return (
                sum(leaf_count for leaf_count, _ in keys),
                min(min_label for _, min_label in keys),
            )

        reorder_children(root)
        return root

    @classmethod
    def balanced_tree(cls, leaf_labels, arity):
        """
        Returns a balanced tree of the specified arity. At each node the
        leaf labels are split equally among the arity children using the
        equal_chunks method.
        """
        assert len(leaf_labels) > 0
        if len(leaf_labels) == 1:
            root = cls(label=leaf_labels[0])
        else:
            children = [
                cls.balanced_tree(chunk, arity)
                for chunk in equal_chunks(leaf_labels, arity)
            ]
            root = cls(children=children)
            for child in children:
                child.parent = root
        return root


def generate_star(num_leaves, *, span, branch_length, record_provenance, **kwargs):
    """
    Generate a star tree for the specified number of leaves.

    See the documentation for :meth:`Tree.generate_star` for more details.
    """
    if num_leaves < 2:
        raise ValueError("The number of leaves must be 2 or greater")
    tables = tskit.TableCollection(sequence_length=span)
    tables.nodes.set_columns(
        flags=np.full(num_leaves, tskit.NODE_IS_SAMPLE, dtype=np.uint32),
        time=np.zeros(num_leaves),
    )
    root = tables.nodes.add_row(time=branch_length)
    tables.edges.set_columns(
        left=np.full(num_leaves, 0),
        right=np.full(num_leaves, span),
        parent=np.full(num_leaves, root, dtype=np.int32),
        child=np.arange(num_leaves, dtype=np.int32),
    )
    if record_provenance:
        # TODO replace with a version of https://github.com/tskit-dev/tskit/pull/243
        # TODO also make sure we convert all the arguments so that they are
        # definitely JSON encodable.
        parameters = {"command": "generate_star", "TODO": "add parameters"}
        tables.provenances.add_row(
            record=json.dumps(tskit.provenance.get_provenance_dict(parameters))
        )
    return tables.tree_sequence().first(**kwargs)


def generate_comb(num_leaves, *, span, branch_length, record_provenance, **kwargs):
    """
    Generate a comb tree for the specified number of leaves.

    See the documentation for :meth:`Tree.generate_comb` for more details.
    """
    if num_leaves < 2:
        raise ValueError("The number of leaves must be 2 or greater")
    tables = tskit.TableCollection(sequence_length=span)
    tables.nodes.set_columns(
        flags=np.full(num_leaves, tskit.NODE_IS_SAMPLE, dtype=np.uint32),
        time=np.zeros(num_leaves),
    )
    right_child = num_leaves - 1
    time = branch_length
    for left_child in range(num_leaves - 2, -1, -1):
        parent = tables.nodes.add_row(time=time)
        time += branch_length
        tables.edges.add_row(0, span, parent, left_child)
        tables.edges.add_row(0, span, parent, right_child)
        right_child = parent

    if record_provenance:
        # TODO replace with a version of https://github.com/tskit-dev/tskit/pull/243
        # TODO also make sure we convert all the arguments so that they are
        # definitely JSON encodable.
        parameters = {"command": "generate_comb", "TODO": "add parameters"}
        tables.provenances.add_row(
            record=json.dumps(tskit.provenance.get_provenance_dict(parameters))
        )
    return tables.tree_sequence().first(**kwargs)


def generate_balanced(
    num_leaves, *, arity, span, branch_length, record_provenance, **kwargs
):
    """
    Generate a balanced tree for the specified number of leaves.

    See the documentation for :meth:`Tree.generate_balanced` for more details.
    """
    if num_leaves < 1:
        raise ValueError("The number of leaves must be at least 1")
    if arity < 2:
        raise ValueError("The arity must be at least 2")

    root = TreeNode.balanced_tree(range(num_leaves), arity)
    tables = root.as_tables(
        num_leaves=num_leaves, span=span, branch_length=branch_length
    )

    if record_provenance:
        # TODO replace with a version of https://github.com/tskit-dev/tskit/pull/243
        # TODO also make sure we convert all the arguments so that they are
        # definitely JSON encodable.
        parameters = {"command": "generate_balanced", "TODO": "add parameters"}
        tables.provenances.add_row(
            record=json.dumps(tskit.provenance.get_provenance_dict(parameters))
        )

    return tables.tree_sequence().first(**kwargs)


def generate_random_binary(
    num_leaves, *, span, branch_length, random_seed, record_provenance, **kwargs
):
    """
    Sample a leaf-labelled binary tree uniformly.

    See the documentation for :meth:`Tree.generate_random_binary` for more details.
    """
    if num_leaves < 1:
        raise ValueError("The number of leaves must be at least 1")

    rng = random.Random(random_seed)
    root = TreeNode.random_binary_tree(range(num_leaves), rng)
    tables = root.as_tables(
        num_leaves=num_leaves, span=span, branch_length=branch_length
    )

    if record_provenance:
        # TODO replace with a version of https://github.com/tskit-dev/tskit/pull/243
        # TODO also make sure we convert all the arguments so that they are
        # definitely JSON encodable.
        parameters = {"command": "generate_random_binary", "TODO": "add parameters"}
        tables.provenances.add_row(
            record=json.dumps(tskit.provenance.get_provenance_dict(parameters))
        )
    ts = tables.tree_sequence()
    return ts.first(**kwargs)


def split_polytomies(
    tree,
    *,
    epsilon=None,
    method=None,
    record_provenance=True,
    random_seed=None,
    **kwargs,
):
    """
    Return a new tree where extra nodes and edges have been inserted
    so that any any node with more than two children is resolved into
    a binary tree.

    See the documentation for :meth:`Tree.split_polytomies` for more details.
    """
    allowed_methods = ["random"]
    if method is None:
        method = "random"
    if method not in allowed_methods:
        raise ValueError(f"Method must be chosen from {allowed_methods}")

    tables = tree.tree_sequence.dump_tables()
    tables.keep_intervals([tree.interval], simplify=False)
    tables.edges.clear()
    rng = random.Random(random_seed)

    for u in tree.nodes():
        if tree.num_children(u) > 2:
            root = TreeNode.random_binary_tree(tree.children(u), rng)
            root.label = u
            root_time = tree.time(u)
            stack = [(child, root_time) for child in root.children]
            while len(stack) > 0:
                node, parent_time = stack.pop()
                if node.label is None:
                    if epsilon is None:
                        child_time = np.nextafter(parent_time, -np.inf)
                    else:
                        child_time = parent_time - epsilon
                    node.label = tables.nodes.add_row(time=child_time)
                else:
                    assert len(node.children) == 0
                    # This is a leaf node connecting back into the original tree
                    child_time = tree.time(node.label)
                if parent_time <= child_time:
                    u = root.label
                    min_child_time = min(tree.time(v) for v in tree.children(u))
                    min_time = root_time - min_child_time
                    message = (
                        f"Cannot resolve the degree {tree.num_children(u)} "
                        f"polytomy rooted at node {u} with minimum time difference "
                        f"of {min_time} to the resolved leaves."
                    )
                    if epsilon is None:
                        message += (
                            " The time difference between nodes is so small that "
                            "more nodes cannot be inserted between within the limits "
                            "of floating point precision."
                        )
                    else:
                        # We can also have parent_time == child_time if epsilon is
                        # chosen such that we exactly divide up the branch in the
                        # original tree. We avoid saying this is caused by a
                        # too-small epsilon by noting it can only happen when we
                        # are at leaf node in the randomly generated tree.
                        if parent_time == child_time and len(node.children) > 0:
                            message += (
                                f" The fixed epsilon value of {epsilon} is too small, "
                                "resulting in the parent and child times being equal "
                                "within the limits of numerical precision."
                            )
                        else:
                            message += (
                                f" The fixed epsilon value of {epsilon} is too large, "
                                "resulting in the parent time being less than the child "
                                "time."
                            )
                    raise tskit.LibraryError(message)
                tables.edges.add_row(*tree.interval, node.parent.label, node.label)
                for child in node.children:
                    stack.append((child, child_time))
        else:
            for v in tree.children(u):
                tables.edges.add_row(*tree.interval, u, v)

    if record_provenance:
        parameters = {"command": "split_polytomies"}
        tables.provenances.add_row(
            record=json.dumps(tskit.provenance.get_provenance_dict(parameters))
        )
    try:
        tables.sort()
        ts = tables.tree_sequence()
    except tskit.LibraryError as e:
        msg = str(e)
        # We should have caught all topology time travel above.
        assert not msg.startswith("time[parent] must be greater than time[child]")
        if msg.startswith(
            "A mutation's time must be < the parent node of the edge on which it occurs"
        ):
            if epsilon is not None:
                msg = (
                    f"epsilon={epsilon} not small enough to create new nodes below a "
                    "polytomy, due to the time of a mutation above a child of the "
                    "polytomy."
                )
            else:
                msg = (
                    "Cannot split polytomy: mutation with numerical precision "
                    "of the parent time."
                )
            e.args += (msg,)
        raise e
    return ts.at(tree.interval[0], **kwargs)


def treeseq_count_topologies(ts, sample_sets):
    topology_counter = np.full(ts.num_nodes, None, dtype=object)
    parent = np.full(ts.num_nodes, -1)

    def update_state(tree, u):
        stack = [u]
        while len(stack) > 0:
            v = stack.pop()
            children = []
            for c in tree.children(v):
                if topology_counter[c] is not None:
                    children.append(topology_counter[c])
            if len(children) > 0:
                topology_counter[v] = combine_child_topologies(children)
            else:
                topology_counter[v] = None
            p = parent[v]
            if p != -1:
                stack.append(p)

    for sample_set_index, sample_set in enumerate(sample_sets):
        for u in sample_set:
            if not ts.node(u).is_sample():
                raise ValueError(f"Node {u} in sample_sets is not a sample.")
            topology_counter[u] = TopologyCounter.from_sample(sample_set_index)

    for tree, (_, edges_out, edges_in) in zip(ts.trees(), ts.edge_diffs()):
        # Avoid recomputing anything for the parent until all child edges
        # for that parent are inserted/removed
        for p, sibling_edges in itertools.groupby(edges_out, key=lambda e: e.parent):
            for e in sibling_edges:
                parent[e.child] = -1
            update_state(tree, p)
        for p, sibling_edges in itertools.groupby(edges_in, key=lambda e: e.parent):
            if tree.is_sample(p):
                raise ValueError("Internal samples not supported.")
            for e in sibling_edges:
                parent[e.child] = p
            update_state(tree, p)

        counters = []
        for root in tree.roots:
            if topology_counter[root] is not None:
                counters.append(topology_counter[root])
        yield TopologyCounter.merge(counters)


def tree_count_topologies(tree, sample_sets):
    for u in tree.samples():
        if not tree.is_leaf(u):
            raise ValueError("Internal samples not supported.")

    topology_counter = np.full(tree.tree_sequence.num_nodes, None, dtype=object)
    for sample_set_index, sample_set in enumerate(sample_sets):
        for u in sample_set:
            if not tree.is_sample(u):
                raise ValueError(f"Node {u} in sample_sets is not a sample.")
            topology_counter[u] = TopologyCounter.from_sample(sample_set_index)

    for u in tree.nodes(order="postorder"):
        children = []
        for v in tree.children(u):
            if topology_counter[v] is not None:
                children.append(topology_counter[v])
        if len(children) > 0:
            topology_counter[u] = combine_child_topologies(children)

    counters = []
    for root in tree.roots:
        if topology_counter[root] is not None:
            counters.append(topology_counter[root])
    return TopologyCounter.merge(counters)


def combine_child_topologies(topology_counters):
    """
    Select all combinations of topologies from different
    counters in ``topology_counters`` that are capable of
    being combined into a single topology. This includes
    any combination of at least two topologies, all from
    different children, where no topologies share a
    sample set index.
    """
    partial_topologies = PartialTopologyCounter()
    for tc in topology_counters:
        partial_topologies.add_sibling_topologies(tc)

    return partial_topologies.join_all_combinations()


class TopologyCounter:
    """
    Contains the distributions of embedded topologies for every combination
    of the sample sets used to generate the ``TopologyCounter``. It is
    indexable by a combination of sample set indexes and returns a
    ``collections.Counter`` whose keys are topology ranks
    (see :ref:`sec_tree_ranks`). See :meth:`Tree.count_topologies` for more
    detail on how this structure is used.
    """

    def __init__(self):
        self.topologies = collections.defaultdict(collections.Counter)

    def __getitem__(self, sample_set_indexes):
        k = TopologyCounter._to_key(sample_set_indexes)
        return self.topologies[k]

    def __setitem__(self, sample_set_indexes, counter):
        k = TopologyCounter._to_key(sample_set_indexes)
        self.topologies[k] = counter

    @staticmethod
    def _to_key(sample_set_indexes):
        if not isinstance(sample_set_indexes, collections.abc.Iterable):
            sample_set_indexes = (sample_set_indexes,)
        return tuple(sorted(sample_set_indexes))

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.topologies == other.topologies

    @staticmethod
    def merge(topology_counters):
        """
        Union together independent topology counters into one.
        """
        total = TopologyCounter()
        for tc in topology_counters:
            for k, v in tc.topologies.items():
                total.topologies[k] += v

        return total

    @staticmethod
    def from_sample(sample_set_index):
        """
        Generate the topologies covered by a single sample. This
        is the single-leaf topology representing the single sample
        set.
        """
        rank_tree = RankTree(children=[], label=sample_set_index)
        tc = TopologyCounter()
        tc[sample_set_index][rank_tree.rank()] = 1
        return tc


class PartialTopologyCounter:
    """
    Represents the possible combinations of children under a node in a tree
    and the combinations of embedded topologies that are rooted at the node.
    This allows an efficient way of calculating which unique embedded
    topologies arise by only every storing a given pairing of sibling topologies
    once.
    ``partials`` is a dictionary where a key is a tuple of sample set indexes,
    and the value is a ``collections.Counter`` that counts combinations of
    sibling topologies whose tips represent the sample sets in the key.
    Each element of the counter is a homogeneous tuple where each element represents
    a topology. The topology is itself a tuple of the sample set indexes in that
    topology and the rank.
    """

    def __init__(self):
        self.partials = collections.defaultdict(collections.Counter)

    def add_sibling_topologies(self, topology_counter):
        """
        Combine each topology in the given TopologyCounter with every existing
        combination of topologies whose sample set indexes are disjoint from the
        topology from the counter. This also includes adding the topologies from
        the counter without joining them to any existing combinations.
        """
        merged = collections.defaultdict(collections.Counter)
        for sample_set_indexes, topologies in topology_counter.topologies.items():
            for rank, count in topologies.items():
                topology = ((sample_set_indexes, rank),)
                # Cross with existing topology combinations
                for sibling_sample_set_indexes, siblings in self.partials.items():
                    if isdisjoint(sample_set_indexes, sibling_sample_set_indexes):
                        for sib_topologies, sib_count in siblings.items():
                            merged_topologies = merge_tuple(sib_topologies, topology)
                            merged_sample_set_indexes = merge_tuple(
                                sibling_sample_set_indexes, sample_set_indexes
                            )
                            merged[merged_sample_set_indexes][merged_topologies] += (
                                count * sib_count
                            )
                # Propagate without combining
                merged[sample_set_indexes][topology] += count

        for sample_set_indexes, counter in merged.items():
            self.partials[sample_set_indexes] += counter

    def join_all_combinations(self):
        """
        For each pairing of child topologies, join them together into a new
        tree and count the resulting topologies.
        """
        topology_counter = TopologyCounter()
        for sample_set_indexes, sibling_topologies in self.partials.items():
            for topologies, count in sibling_topologies.items():
                # A node must have at least two children
                if len(topologies) >= 2:
                    rank = PartialTopologyCounter.join_topologies(topologies)
                    topology_counter[sample_set_indexes][rank] += count
                else:
                    # Pass on the single tree without adding a parent
                    for _, rank in topologies:
                        topology_counter[sample_set_indexes][rank] += count

        return topology_counter

    @staticmethod
    def join_topologies(child_topologies):
        children = []
        for sample_set_indexes, rank in child_topologies:
            n = len(sample_set_indexes)
            t = RankTree.unrank(n, rank, list(sample_set_indexes))
            children.append(t)
        children.sort(key=RankTree.canonical_order)
        return RankTree(children).rank()


def all_trees(num_leaves, span=1):
    """
    Generates all unique leaf-labelled trees with ``num_leaves``
    leaves. See :ref:`sec_combinatorics` on the details of this
    enumeration. The leaf labels are selected from the set
    ``[0, num_leaves)``. The times and labels on internal nodes are
    chosen arbitrarily.

    :param int num_leaves: The number of leaves of the tree to generate.
    :param float span: The genomic span of each returned tree.
    :rtype: tskit.Tree
    """
    for rank_tree in RankTree.all_labelled_trees(num_leaves):
        yield rank_tree.to_tsk_tree(span=span)


def all_tree_shapes(num_leaves, span=1):
    """
    Generates all unique shapes of trees with ``num_leaves`` leaves.

    :param int num_leaves: The number of leaves of the tree to generate.
    :param float span: The genomic span of each returned tree.
    :rtype: tskit.Tree
    """
    for rank_tree in RankTree.all_unlabelled_trees(num_leaves):
        default_labelling = rank_tree.label_unrank(0)
        yield default_labelling.to_tsk_tree(span=span)


def all_tree_labellings(tree, span=1):
    """
    Generates all unique labellings of the leaves of a
    :class:`tskit.Tree`. Leaves are labelled from the set
    ``[0, n)`` where ``n`` is the number of leaves of ``tree``.

    :param tskit.Tree tree: The tree used to generate
        labelled trees of the same shape.
    :param float span: The genomic span of each returned tree.
    :rtype: tskit.Tree
    """
    rank_tree = RankTree.from_tsk_tree(tree)
    for labelling in RankTree.all_labellings(rank_tree):
        yield labelling.to_tsk_tree(span=span)


class RankTree:
    """
    A tree class that maintains the topological ranks of each node in the tree.
    This structure can be used to efficiently compute the rank of a tree of
    n labelled leaves and produce a tree given a rank.
    """

    def __init__(self, children, label=None):
        # Children are assumed to be sorted by RankTree.canonical_order
        self.children = children
        if len(children) == 0:
            self.num_leaves = 1
            self.labels = [label]
        else:
            self.num_leaves = sum(c.num_leaves for c in children)
            self.labels = list(heapq.merge(*(c.labels for c in children)))

        self._shape_rank = None
        self._label_rank = None

    def compute_shape_rank(self):
        """
        Mirroring the way in which unlabelled trees are enumerated, we must
        first calculate the number of trees whose partitions of number of leaves
        rank lesser than this tree's partition.

        Once we reach the partition of leaves in this tree, we examine the
        groups of child subtrees assigned to subsequences of the partition.
        For each group of children with the same number of leaves, k, the trees
        in that group were selected according to a combination with replacement
        of those trees from S(k). By finding the rank of that combination,
        we find how many combinations preceded the current one in that group.
        That rank is then multiplied by the total number of arrangements that
        could be made in the following groups, added to the total rank,
        and then we recur on the rest of the group and groups.
        """
        part = self.leaf_partition()
        total = 0
        for prev_part in partitions(self.num_leaves):
            if prev_part == part:
                break
            total += num_tree_pairings(prev_part)

        child_groups = self.group_children_by_num_leaves()
        next_child_idx = 0
        for g in child_groups:
            next_child_idx += len(g)
            k = g[0].num_leaves
            S_k = num_shapes(k)

            child_ranks = [c.shape_rank() for c in g]
            g_rank = Combination.with_replacement_rank(child_ranks, S_k)

            # TODO precompute vector before loop
            rest_part = part[next_child_idx:]
            total_rest = num_tree_pairings(rest_part)

            total += g_rank * total_rest

        return total

    def compute_label_rank(self):
        """
        Again mirroring how we've labeled a particular tree, T, we can rank the
        labelling on T.

        We group the children into symmetric groups. In the context of labelling,
        symmetric groups contain child trees that are of the same shape. Each
        group contains a combination of labels selected from all the labels
        available to T.

        The different variables to consider are:
        1. How to assign a combination of labels to the first group.
        2. Given a combination of labels assigned to the group, how can we
            distribute those labels to each tree in the group.
        3. Given an assignment of the labels to each tree in the group, how many
            distinct ways could all the trees in the group be labelled.

        These steps for generating labelled trees break down the stages of
        ranking them.
        For each group G, we can find the rank of the combination of labels
        assigned to G. This rank times the number of ways the trees in G
        could be labelled, times the number of possible labellings of the
        rest of the trees, gives the number of labellings that precede those with
        the given combination of labels assigned to G. This process repeats and
        breaks down to give the rank of the assignment of labels to trees in G,
        and the label ranks of the trees themselves in G.
        """
        all_labels = self.labels
        child_groups = self.group_children_by_shape()
        total = 0
        for i, g in enumerate(child_groups):
            rest_groups = child_groups[i + 1 :]
            g_labels = list(heapq.merge(*(t.labels for t in g)))
            num_rest_labellings = num_list_of_group_labellings(rest_groups)

            # Preceded by all of the ways to label all the groups
            # with a lower ranking combination given to g.
            comb_rank = Combination.rank(g_labels, all_labels)
            num_g_labellings = num_group_labellings(g)
            preceding_comb = comb_rank * num_g_labellings * num_rest_labellings

            # Preceded then by all the configurations of g ranking less than
            # the current one
            rank_from_g = group_rank(g) * num_rest_labellings

            total += preceding_comb + rank_from_g
            all_labels = set_minus(all_labels, g_labels)

        return total

    # TODO I think this would boost performance if it were a field and not
    # recomputed.
    def num_labellings(self):
        child_groups = self.group_children_by_shape()
        return num_list_of_group_labellings(child_groups)

    def rank(self):
        return self.shape_rank(), self.label_rank()

    def shape_rank(self):
        if self._shape_rank is None:
            self._shape_rank = self.compute_shape_rank()
        return self._shape_rank

    def label_rank(self):
        if self._label_rank is None:
            assert self.shape_rank() is not None
            self._label_rank = self.compute_label_rank()
        return self._label_rank

    @staticmethod
    def unrank(num_leaves, rank, labels=None):
        """
        Produce a ``RankTree`` of the given ``rank`` with ``num_leaves`` leaves,
        labelled with ``labels``. Labels must be sorted, and if ``None`` default
        to ``[0, num_leaves)``.
        """
        shape_rank, label_rank = rank
        if shape_rank < 0 or label_rank < 0:
            raise ValueError("Rank is out of bounds.")
        unlabelled = RankTree.shape_unrank(num_leaves, shape_rank)
        return unlabelled.label_unrank(label_rank, labels)

    @staticmethod
    def shape_unrank(n, shape_rank):
        """
        Generate an unlabelled tree with n leaves with a shape corresponding to
        the `shape_rank`.
        """
        part, child_shape_ranks = children_shape_ranks(shape_rank, n)
        children = [
            RankTree.shape_unrank(k, rk) for k, rk in zip(part, child_shape_ranks)
        ]

        t = RankTree(children=children)
        t._shape_rank = shape_rank
        return t

    def label_unrank(self, label_rank, labels=None):
        """
        Generate a tree with the same shape, whose leaves are labelled
        from ``labels`` with the labelling corresponding to ``label_rank``.
        """
        if labels is None:
            labels = list(range(self.num_leaves))

        if self.is_leaf():
            if label_rank != 0:
                raise ValueError("Rank is out of bounds.")
            return RankTree(children=[], label=labels[0])

        child_groups = self.group_children_by_shape()
        child_labels, child_label_ranks = children_label_ranks(
            child_groups, label_rank, labels
        )

        children = self.children
        labelled_children = [
            RankTree.label_unrank(c, c_rank, c_labels)
            for c, c_rank, c_labels in zip(children, child_label_ranks, child_labels)
        ]

        t = RankTree(children=labelled_children)
        t._shape_rank = self.shape_rank()
        t._label_rank = label_rank
        return t

    @staticmethod
    def canonical_order(c):
        """
        Defines the canonical ordering of sibling subtrees.
        """
        return c.num_leaves, c.shape_rank(), c.min_label()

    @staticmethod
    def from_tsk_tree_node(tree, u):
        if tree.is_leaf(u):
            return RankTree(children=[], label=u)

        if tree.num_children(u) == 1:
            raise ValueError("Cannot rank trees with unary nodes")

        children = list(
            sorted(
                (RankTree.from_tsk_tree_node(tree, c) for c in tree.children(u)),
                key=RankTree.canonical_order,
            )
        )
        return RankTree(children=children)

    @staticmethod
    def from_tsk_tree(tree):
        if tree.num_roots != 1:
            raise ValueError("Cannot rank trees with multiple roots")

        return RankTree.from_tsk_tree_node(tree, tree.root)

    def to_tsk_tree(self, span=1, branch_length=1):
        """
        Convert a ``RankTree`` into the only tree in a new tree sequence. Internal
        nodes and their times are assigned via a postorder traversal of the tree.

        :param float span: The genomic span of the returned tree. The tree will cover
            the interval :math:`[0, span)` and the :attr:`~Tree.tree_sequence` from which
            the tree is taken will have its :attr:`~tskit.TreeSequence.sequence_length`
            equal to ``span``.
        :param float branch_length: The minimum length of a branch in the returned
            tree.
        """
        if set(self.labels) != set(range(self.num_leaves)):
            raise ValueError("Labels set must be equivalent to [0, num_leaves)")

        tables = tskit.TableCollection(span)

        def add_node(node):
            if node.is_leaf():
                assert node.label is not None
                return node.label

            child_ids = [add_node(child) for child in node.children]
            max_child_time = max(tables.nodes.time[c] for c in child_ids)
            parent_id = tables.nodes.add_row(time=max_child_time + branch_length)
            for child_id in child_ids:
                tables.edges.add_row(0, span, parent_id, child_id)

            return parent_id

        for _ in range(self.num_leaves):
            tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        add_node(self)

        # The way in which we're inserting nodes doesn't necessarily
        # adhere to the ordering constraint on edges, so we have
        # to sort.
        tables.sort()
        return tables.tree_sequence().first()

    @staticmethod
    def all_labelled_trees(n):
        """
        Generate all unordered, leaf-labelled trees with n leaves.
        """
        for tree in RankTree.all_unlabelled_trees(n):
            yield from RankTree.all_labellings(tree)

    @staticmethod
    def all_unlabelled_trees(n):
        """
        Generate all tree shapes with n leaves. See :ref:`sec_combinatorics`
        for how tree shapes are enumerated.
        """
        if n == 1:
            yield RankTree(children=[])
        else:
            for part in partitions(n):
                for subtree_pairing in RankTree.all_subtree_pairings(
                    group_partition(part)
                ):
                    yield RankTree(children=subtree_pairing)

    @staticmethod
    def all_subtree_pairings(grouped_part):
        if len(grouped_part) == 0:
            yield []
        else:
            g = grouped_part[0]
            k = g[0]
            all_k_leaf_trees = RankTree.all_unlabelled_trees(k)
            num_k_leaf_trees = len(g)
            g_trees = itertools.combinations_with_replacement(
                all_k_leaf_trees, num_k_leaf_trees
            )
            for first_trees in g_trees:
                for rest in RankTree.all_subtree_pairings(grouped_part[1:]):
                    yield list(first_trees) + rest

    @staticmethod
    def all_labellings(tree, labels=None):
        """
        Given a tree, generate all the unique labellings of that tree.
        See :ref:`sec_combinatorics` for how labellings of a tree are
        enumerated.
        """
        if labels is None:
            labels = list(range(tree.num_leaves))

        if tree.is_leaf():
            assert len(labels) == 1
            yield RankTree(children=[], label=labels[0])
        else:
            groups = tree.group_children_by_shape()
            for labeled_children in RankTree.label_all_groups(groups, labels):
                yield RankTree(children=labeled_children)

    @staticmethod
    def label_all_groups(groups, labels):
        if len(groups) == 0:
            yield []
        else:
            g, rest = groups[0], groups[1:]
            x = len(g)
            k = g[0].num_leaves
            for g_labels in itertools.combinations(labels, x * k):
                rest_labels = set_minus(labels, g_labels)
                for labeled_g in RankTree.label_tree_group(g, g_labels):
                    for labeled_rest in RankTree.label_all_groups(rest, rest_labels):
                        yield labeled_g + labeled_rest

    @staticmethod
    def label_tree_group(trees, labels):
        if len(trees) == 0:
            assert len(labels) == 0
            yield []
        else:
            first, rest = trees[0], trees[1:]
            k = first.num_leaves
            min_label = labels[0]
            for first_other_labels in itertools.combinations(labels[1:], k - 1):
                first_labels = [min_label] + list(first_other_labels)
                rest_labels = set_minus(labels, first_labels)
                for labeled_first in RankTree.all_labellings(first, first_labels):
                    for labeled_rest in RankTree.label_tree_group(rest, rest_labels):
                        yield [labeled_first] + labeled_rest

    def newick(self):
        if self.is_leaf():
            return str(self.label) if self.labelled() else ""
        return "(" + ",".join(c.newick() for c in self.children) + ")"

    @property
    def label(self):
        return self.labels[0]

    def labelled(self):
        return all(label is not None for label in self.labels)

    def min_label(self):
        return self.labels[0]

    def is_leaf(self):
        return len(self.children) == 0

    def leaf_partition(self):
        return [c.num_leaves for c in self.children]

    def group_children_by_num_leaves(self):
        def same_num_leaves(c1, c2):
            return c1.num_leaves == c2.num_leaves

        return group_by(self.children, same_num_leaves)

    def group_children_by_shape(self):
        def same_shape(c1, c2):
            return c1.num_leaves == c2.num_leaves and c1.shape_rank() == c2.shape_rank()

        return group_by(self.children, same_shape)

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False

        if self.is_leaf() and other.is_leaf():
            return self.label == other.label

        if len(self.children) != len(other.children):
            return False

        return all(c1 == c2 for c1, c2 in zip(self.children, other.children))

    def __ne__(self, other):
        return not self.__eq__(other)

    def shape_equal(self, other):
        if self.is_leaf() and other.is_leaf():
            return True

        if len(self.children) != len(other.children):
            return False

        return all(c1.shape_equal(c2) for c1, c2 in zip(self.children, other.children))

    def is_canonical(self):
        if self.is_leaf():
            return True

        children = self.children
        for c1, c2 in zip(children, children[1:]):
            if RankTree.canonical_order(c1) > RankTree.canonical_order(c2):
                return False
        return all(c.is_canonical() for c in children)

    def is_symmetrical(self):
        if self.is_leaf():
            return True

        even_split_leaves = len(set(self.leaf_partition())) == 1
        all_same_rank = len({c.shape_rank() for c in self.children}) == 1

        return even_split_leaves and all_same_rank


# TODO This is called repeatedly in ranking and unranking and has a perfect
# subtructure for DP. It's only every called on n in [0, num_leaves]
# so we should compute a vector of those results up front instead of using
# repeated calls to this function.
# Put an lru_cache on for now as a quick replacement (cuts test time down by 80%)
@functools.lru_cache()
def num_shapes(n):
    """
    The cardinality of the set of unlabelled trees with n leaves,
    up to isomorphism.
    """
    if n <= 1:
        return n
    return sum(num_tree_pairings(part) for part in partitions(n))


def num_tree_pairings(part):
    """
    The number of unique tree shapes that could be assembled from
    a given partition of leaves. If we group the elements of the partition
    by number of leaves, each group can be independently enumerated and the
    cardinalities of each group's pairings can be multiplied. Within a group,
    subsequent trees must have equivalent or greater rank, so the number of
    ways to select trees follows combinations with replacement from the set
    of all possible trees for that group.
    """
    total = 1
    for g in group_partition(part):
        k = g[0]
        total *= Combination.comb_with_replacement(num_shapes(k), len(g))
    return total


def num_labellings(n, shape_rk):
    return RankTree.shape_unrank(n, shape_rk).num_labellings()


def children_shape_ranks(rank, n):
    """
    Return the partition of leaves associated
    with the children of the tree of rank `rank`, and
    the ranks of each child tree.
    """
    part = []
    for prev_part in partitions(n):
        num_trees_with_part = num_tree_pairings(prev_part)
        if rank < num_trees_with_part:
            part = prev_part
            break
        rank -= num_trees_with_part
    else:
        if n != 1:
            raise ValueError("Rank is out of bounds.")

    grouped_part = group_partition(part)
    child_ranks = []
    next_child = 0
    for g in grouped_part:
        next_child += len(g)
        k = g[0]

        # TODO precompute vector up front
        rest_children = part[next_child:]
        rest_num_pairings = num_tree_pairings(rest_children)

        shapes_comb_rank = rank // rest_num_pairings
        g_shape_ranks = Combination.with_replacement_unrank(
            shapes_comb_rank, num_shapes(k), len(g)
        )
        child_ranks += g_shape_ranks
        rank %= rest_num_pairings

    return part, child_ranks


def children_label_ranks(child_groups, rank, labels):
    """
    Produces the subsets of labels assigned to each child
    and the associated label rank of each child.
    """
    child_labels = []
    child_label_ranks = []

    for i, g in enumerate(child_groups):
        k = g[0].num_leaves
        g_num_leaves = k * len(g)
        num_g_labellings = num_group_labellings(g)
        # TODO precompute vector of partial products outside of loop
        rest_groups = child_groups[i + 1 :]
        num_rest_labellings = num_list_of_group_labellings(rest_groups)

        num_labellings_per_label_comb = num_g_labellings * num_rest_labellings
        comb_rank = rank // num_labellings_per_label_comb
        rank_given_label_comb = rank % num_labellings_per_label_comb
        g_rank = rank_given_label_comb // num_rest_labellings

        g_labels = Combination.unrank(comb_rank, labels, g_num_leaves)

        g_child_labels, g_child_ranks = group_label_ranks(g_rank, g, g_labels)
        child_labels += g_child_labels
        child_label_ranks += g_child_ranks

        labels = set_minus(labels, g_labels)
        rank %= num_rest_labellings

    return child_labels, child_label_ranks


def group_rank(g):
    k = g[0].num_leaves
    n = len(g) * k
    # Num ways to label a single one of the trees
    # We can do this once because all the trees in the group
    # are of the same shape rank
    y = g[0].num_labellings()
    all_labels = list(heapq.merge(*(t.labels for t in g)))
    rank = 0
    for i, t in enumerate(g):
        u_labels = t.labels
        curr_trees = len(g) - i
        # Kind of cheating here leaving the selection of min labels implicit
        # because the rank of the comb without min labels is the same
        comb_rank = Combination.rank(u_labels, all_labels)

        # number of ways to distribute labels to rest leaves
        num_rest_combs = 1
        remaining_leaves = n - (i + 1) * k
        for j in range(curr_trees - 1):
            num_rest_combs *= Combination.comb(remaining_leaves - j * k - 1, k - 1)

        preceding_combs = comb_rank * num_rest_combs * (y ** curr_trees)
        curr_comb = t.label_rank() * num_rest_combs * (y ** (curr_trees - 1))
        rank += preceding_combs + curr_comb
        all_labels = set_minus(all_labels, u_labels)
    return rank


# TODO This is only used in a few cases and mostly in a n^2 way. Would
# be easy and useful to do this DP and produce a list of partial products
def num_list_of_group_labellings(groups):
    """
    Given a set of labels and a list of groups, how many unique ways are there
    to assign subsets of labels to each group in the list and subsequently
    label all the trees in all the groups.
    """
    remaining_leaves = sum(len(g) * g[0].num_leaves for g in groups)
    total = 1
    for g in groups:
        k = g[0].num_leaves
        x = len(g)
        num_label_choices = Combination.comb(remaining_leaves, x * k)
        total *= num_label_choices * num_group_labellings(g)
        remaining_leaves -= x * k

    return total


def num_group_labellings(g):
    """
    Given a particular set of labels, how many unique ways are there
    to assign subsets of labels to each tree in the group and subsequently
    label those trees.
    """
    # Shortcut because all the trees are identical and can therefore
    # be labelled in the same ways
    num_tree_labelings = g[0].num_labellings() ** len(g)
    return num_assignments_in_group(g) * num_tree_labelings


def num_assignments_in_group(g):
    """
    Given this group of identical trees, how many unique ways
    are there to divide up a set of n labels?
    """
    n = sum(t.num_leaves for t in g)
    total = 1
    for t in g:
        k = t.num_leaves
        # Choose k - 1 from n - 1 because the minimum label must be
        # assigned to the first tree for a canonical labelling.
        total *= Combination.comb(n - 1, k - 1)
        n -= k
    return total


def group_label_ranks(rank, child_group, labels):
    """
    Given a group of trees of the same shape, a label rank and list of labels,
    produce assignment of label subsets to each tree in the group and the
    label rank of each tree.
    """
    child_labels = []
    child_label_ranks = []

    for i, rank_tree in enumerate(child_group):
        k = rank_tree.num_leaves
        num_t_labellings = rank_tree.num_labellings()
        rest_trees = child_group[i + 1 :]
        num_rest_assignments = num_assignments_in_group(rest_trees)
        num_rest_labellings = num_rest_assignments * (
            num_t_labellings ** len(rest_trees)
        )
        num_labellings_per_label_comb = num_t_labellings * num_rest_labellings

        comb_rank = rank // num_labellings_per_label_comb
        rank_given_comb = rank % num_labellings_per_label_comb
        t_rank = rank_given_comb // num_rest_labellings
        rank %= num_rest_labellings

        min_label = labels[0]
        t_labels = [min_label] + Combination.unrank(comb_rank, labels[1:], k - 1)
        labels = set_minus(labels, t_labels)

        child_labels.append(t_labels)
        child_label_ranks.append(t_rank)

    return child_labels, child_label_ranks


class Combination:
    @staticmethod
    def comb(n, k):
        """
        The number of times you can select k items from
        n items without order and without replacement.

        FIXME: This function will be available in `math` in Python 3.8
        and should be replaced eventually.
        """
        k = min(k, n - k)
        res = 1
        for i in range(1, k + 1):
            res *= n - k + i
            res //= i

        return res

    @staticmethod
    def comb_with_replacement(n, k):
        """
        Also called multichoose, the number of times you can select
        k items from n items without order but *with* replacement.
        """
        return Combination.comb(n + k - 1, k)

    @staticmethod
    def rank(combination, elements):
        """
        Find the combination of k elements from the given set of elements
        with the given rank in a lexicographic ordering.
        """
        indices = [elements.index(x) for x in combination]
        return Combination.from_range_rank(indices, len(elements))

    @staticmethod
    def from_range_rank(combination, n):
        """
        Find the combination of k integers from [0, n)
        with the given rank in a lexicographic ordering.
        """
        k = len(combination)
        if k == 0 or k == n:
            return 0

        j = combination[0]
        combination = [x - 1 for x in combination]
        if j == 0:
            return Combination.from_range_rank(combination[1:], n - 1)

        first_rank = Combination.comb(n - 1, k - 1)
        rest_rank = Combination.from_range_rank(combination, n - 1)
        return first_rank + rest_rank

    @staticmethod
    def unrank(rank, elements, k):
        n = len(elements)
        if k == 0:
            return []
        if len(elements) == 0:
            raise ValueError("Rank is out of bounds.")

        n_rest_combs = Combination.comb(n - 1, k - 1)
        if rank < n_rest_combs:
            return elements[:1] + Combination.unrank(rank, elements[1:], k - 1)

        return Combination.unrank(rank - n_rest_combs, elements[1:], k)

    @staticmethod
    def with_replacement_rank(combination, n):
        """
        Find the rank of ``combination`` in the lexicographic ordering of
        combinations with replacement of integers from [0, n).
        """
        k = len(combination)
        if k == 0:
            return 0
        j = combination[0]
        if k == 1:
            return j

        if j == 0:
            return Combination.with_replacement_rank(combination[1:], n)

        rest = [x - j for x in combination[1:]]
        preceding = 0
        for i in range(j):
            preceding += Combination.comb_with_replacement(n - i, k - 1)
        return preceding + Combination.with_replacement_rank(rest, n - j)

    @staticmethod
    def with_replacement_unrank(rank, n, k):
        """
        Find the combination with replacement of k integers from [0, n)
        with the given rank in a lexicographic ordering.
        """
        if k == 0:
            return []

        i = 0
        preceding = Combination.comb_with_replacement(n, k - 1)
        while rank >= preceding:
            rank -= preceding
            i += 1
            preceding = Combination.comb_with_replacement(n - i, k - 1)

        rest = Combination.with_replacement_unrank(rank, n - i, k - 1)
        return [i] + [x + i for x in rest]


def set_minus(arr, subset):
    return [x for x in arr if x not in set(subset)]


# TODO I think we can use part-count form everywhere. Right now
# there's a janky work-around of grouping the partition when
# we needed in part-count form but it doesn't look like there's any
# place that can't just accept it from the start.
def partitions(n):
    """
    Ascending integer partitions of n, excluding the partition [n].
    Since trees with unary nodes are uncountable, the partition of
    leaves must be at least size two.
    """
    if n > 0:
        # last partition is guaranteed to be length 1.
        yield from itertools.takewhile(lambda a: len(a) > 1, rule_asc(n))


def rule_asc(n):
    """
    Produce the integer partitions of n as ascending compositions.
    See: http://jeromekelleher.net/generating-integer-partitions.html
    """
    a = [0 for _ in range(n + 1)]
    k = 1
    a[1] = n
    while k != 0:
        x = a[k - 1] + 1
        y = a[k] - 1
        k -= 1
        while x <= y:
            a[k] = x
            y -= x
            k += 1
        a[k] = x + y
        yield a[: k + 1]


def group_by(values, equal):
    groups = []
    curr_group = []
    for x in values:
        if len(curr_group) == 0 or equal(x, curr_group[0]):
            curr_group.append(x)
        else:
            groups.append(curr_group)
            curr_group = [x]

    if len(curr_group) != 0:
        groups.append(curr_group)
    return groups


def group_partition(part):
    return group_by(part, lambda x, y: x == y)


def merge_tuple(tup1, tup2):
    return tuple(heapq.merge(tup1, tup2))


def isdisjoint(iterable1, iterable2):
    return set(iterable1).isdisjoint(iterable2)
