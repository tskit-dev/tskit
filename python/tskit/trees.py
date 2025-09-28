# MIT License
#
# Copyright (c) 2018-2025 Tskit Developers
# Copyright (c) 2015-2018 University of Oxford
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
Module responsible for managing trees and tree sequences.
"""
from __future__ import annotations

import base64
import builtins
import collections
import concurrent.futures
import functools
import io
import itertools
import json
import math
import numbers
import warnings
from dataclasses import dataclass
from typing import Any
from typing import NamedTuple

import numpy as np

import _tskit
import tskit
import tskit.combinatorics as combinatorics
import tskit.drawing as drawing
import tskit.metadata as metadata_module
import tskit.provenance as provenance
import tskit.tables as tables
import tskit.text_formats as text_formats
import tskit.util as util
import tskit.vcf as vcf
from tskit import NODE_IS_SAMPLE
from tskit import NULL
from tskit import UNKNOWN_TIME

LEGACY_MS_LABELS = "legacy_ms"


@dataclass
class VcfModelMapping:
    individuals_nodes: np.ndarray
    individuals_name: np.ndarray
    transformed_positions: np.ndarray
    contig_length: int
    contig_id: str
    isolated_as_missing: bool


class CoalescenceRecord(NamedTuple):
    left: float
    right: float
    node: int
    children: np.ndarray
    time: float
    population: int


class Interval(NamedTuple):
    """
    A tuple of 2 numbers, ``[left, right)``, defining an interval over the genome.
    """

    left: float | int
    """
    The left hand end of the interval. By convention this value is included
    in the interval
    """
    right: float | int
    """
    The right hand end of the interval. By convention this value is *not*
    included in the interval, i.e., the interval is half-open.
    """

    @property
    def span(self) -> float | int:
        """
        The span of the genome covered by this interval, simply ``right-left``.
        """
        return self.right - self.left

    @property
    def mid(self) -> float | int:
        """
        The middle point of this interval, simply ``left+(right-left)/2``.
        """
        return self.left + (self.right - self.left) / 2


class EdgeDiff(NamedTuple):
    interval: Interval
    edges_out: list
    edges_in: list


def store_tree_sequence(cls):
    wrapped_init = cls.__init__

    # Intercept the init to record the tree_sequence
    def new_init(self, *args, tree_sequence=None, **kwargs):
        builtins.object.__setattr__(self, "_tree_sequence", tree_sequence)
        wrapped_init(self, *args, **kwargs)

    cls.__init__ = new_init
    return cls


@store_tree_sequence
@metadata_module.lazy_decode()
@dataclass
class Individual(util.Dataclass):
    """
    An :ref:`individual <sec_individual_table_definition>` in a tree sequence.
    Since nodes correspond to genomes, individuals are associated with a collection
    of nodes (e.g., two nodes per diploid). See :ref:`sec_nodes_or_individuals`
    for more discussion of this distinction.

    Modifying the attributes in this class will have **no effect** on the
    underlying tree sequence data.
    """

    __slots__ = [
        "id",
        "flags",
        "location",
        "parents",
        "nodes",
        "metadata",
        "_tree_sequence",
    ]
    id: int  # noqa A003
    """
    The integer ID of this individual. Varies from 0 to
    :attr:`TreeSequence.num_individuals` - 1."""
    flags: int
    """
    The bitwise flags for this individual.
    """
    location: np.ndarray
    """
    The spatial location of this individual as a numpy array. The location is an empty
    array if no spatial location is defined.
    """
    parents: np.ndarray
    """
    The parent individual ids of this individual as a numpy array. The parents is an
    empty array if no parents are defined.
    """
    nodes: np.ndarray
    """
    The IDs of the nodes that are associated with this individual as
    a numpy array (dtype=np.int32). If no nodes are associated with the
    individual this array will be empty.
    """
    metadata: bytes | dict | None
    """
    The :ref:`metadata <sec_metadata_definition>`
    for this individual, decoded if a schema applies.
    """

    @property
    def population(self) -> int:
        populations = {self._tree_sequence.node(n).population for n in self.nodes}
        if len(populations) > 1:
            raise ValueError("Individual has nodes with mis-matched populations")
        if len(populations) == 0:
            return tskit.NULL
        return populations.pop()

    @property
    def time(self) -> int:
        times = {self._tree_sequence.node(n).time for n in self.nodes}
        if len(times) > 1:
            raise ValueError("Individual has nodes with mis-matched times")
        if len(times) == 0:
            return tskit.UNKNOWN_TIME
        return times.pop()

    # Custom eq for the numpy arrays
    def __eq__(self, other):
        return (
            self.id == other.id
            and self.flags == other.flags
            and np.array_equal(self.location, other.location)
            and np.array_equal(self.parents, other.parents)
            and np.array_equal(self.nodes, other.nodes)
            and self.metadata == other.metadata
        )


@metadata_module.lazy_decode()
@dataclass
class Node(util.Dataclass):
    """
    A :ref:`node <sec_node_table_definition>` in a tree sequence, corresponding
    to a single genome. The ``time`` and ``population`` are attributes of the
    ``Node``, rather than the ``Individual``, as discussed in
    :ref:`sec_nodes_or_individuals`.

    Modifying the attributes in this class will have **no effect** on the
    underlying tree sequence data.
    """

    __slots__ = ["id", "flags", "time", "population", "individual", "metadata"]
    id: int  # noqa A003
    """
    The integer ID of this node. Varies from 0 to :attr:`TreeSequence.num_nodes` - 1.
    """
    flags: int
    """
    The bitwise flags for this node.
    """
    time: float
    """
    The birth time of this node.
    """
    population: int
    """
    The integer ID of the population that this node was born in.
    """
    individual: int
    """
    The integer ID of the individual that this node was a part of.
    """
    metadata: bytes | dict | None
    """
    The :ref:`metadata <sec_metadata_definition>` for this node, decoded if a schema
    applies.
    """

    def is_sample(self):
        """
        Returns True if this node is a sample. This value is derived from the
        ``flag`` variable.

        :rtype: bool
        """
        return self.flags & NODE_IS_SAMPLE


@metadata_module.lazy_decode(own_init=True)
@dataclass
class Edge(util.Dataclass):
    """
    An :ref:`edge <sec_edge_table_definition>` in a tree sequence.

    Modifying the attributes in this class will have **no effect** on the
    underlying tree sequence data.
    """

    __slots__ = ["left", "right", "parent", "child", "metadata", "id"]
    left: float
    """
    The left coordinate of this edge.
    """
    right: float
    """
    The right coordinate of this edge.
    """
    parent: int
    """
    The integer ID of the parent node for this edge.
    To obtain further information about a node with a given ID, use
    :meth:`TreeSequence.node`.
    """
    child: int
    """
    The integer ID of the child node for this edge.
    To obtain further information about a node with a given ID, use
    :meth:`TreeSequence.node`.
    """
    metadata: bytes | dict | None
    """
    The :ref:`metadata <sec_metadata_definition>` for this edge, decoded if a schema
    applies.
    """
    id: int  # noqa A003
    """
    The integer ID of this edge. Varies from 0 to
    :attr:`TreeSequence.num_edges` - 1.
    """

    # Custom init to define default values with slots
    def __init__(
        self,
        left,
        right,
        parent,
        child,
        metadata=b"",
        id=None,  # noqa A002
        metadata_decoder=None,
    ):
        self.id = id
        self.left = left
        self.right = right
        self.parent = parent
        self.child = child
        self.metadata = metadata
        self._metadata_decoder = metadata_decoder

    @property
    def span(self):
        """
        Returns the span of this edge, i.e., the right position minus the left position

        :return: The span of this edge.
        :rtype: float
        """
        return self.right - self.left

    @property
    def interval(self):
        """
        Returns the left and right positions of this edge as an :class:`Interval`

        :return: The interval covered by this edge.
        :rtype: :class:`Interval`
        """
        return Interval(self.left, self.right)


@metadata_module.lazy_decode()
@dataclass
class Site(util.Dataclass):
    """
    A :ref:`site <sec_site_table_definition>` in a tree sequence.

    Modifying the attributes in this class will have **no effect** on the
    underlying tree sequence data.
    """

    __slots__ = ["id", "position", "ancestral_state", "mutations", "metadata"]
    id: int  # noqa A003
    """
    The integer ID of this site. Varies from 0 to :attr:`TreeSequence.num_sites` - 1.
    """
    position: float
    """
    The floating point location of this site in genome coordinates.
    Ranges from 0 (inclusive) to :attr:`TreeSequence.sequence_length` (exclusive).
    """
    ancestral_state: str
    """
    The ancestral state at this site (i.e., the state inherited by nodes, unless
    mutations occur).
    """
    mutations: np.ndarray
    """
    The list of mutations at this site. Mutations within a site are returned in the
    order they are specified in the underlying :class:`MutationTable`.
    """
    metadata: bytes | dict | None
    """
    The :ref:`metadata <sec_metadata_definition>` for this site, decoded if a schema
    applies.
    """

    # We need a custom eq for the numpy arrays
    def __eq__(self, other):
        return (
            isinstance(other, Site)
            and self.id == other.id
            and self.position == other.position
            and self.ancestral_state == other.ancestral_state
            and np.array_equal(self.mutations, other.mutations)
            and self.metadata == other.metadata
        )

    @property
    def alleles(self) -> set[str]:
        """
        Return the set of all the alleles defined at this site

        .. note::
            This deliberately returns an (unordered) *set* of the possible allelic
            states (as defined by the site's ancestral allele and its associated
            mutations). If you wish to obtain an (ordered) *list* of alleles, for
            example to translate the numeric genotypes at a site into allelic states,
            you should instead use ``.alleles`` attribute of the :class:`Variant` class,
            which unlike this attribute includes ``None`` as a state when there is
            missing data at a site.
        """
        return {self.ancestral_state} | {m.derived_state for m in self.mutations}


@metadata_module.lazy_decode()
@dataclass
class Mutation(util.Dataclass):
    """
    A :ref:`mutation <sec_mutation_table_definition>` in a tree sequence.

    Modifying the attributes in this class will have **no effect** on the
    underlying tree sequence data.
    """

    __slots__ = [
        "id",
        "site",
        "node",
        "derived_state",
        "parent",
        "metadata",
        "time",
        "edge",
        "inherited_state",
    ]
    id: int  # noqa A003
    """
    The integer ID of this mutation. Varies from 0 to
    :attr:`TreeSequence.num_mutations` - 1.

    Modifying the attributes in this class will have **no effect** on the
    underlying tree sequence data.
    """
    site: int
    """
    The integer ID of the site that this mutation occurs at. To obtain
    further information about a site with a given ID use :meth:`TreeSequence.site`.
    """
    node: int
    """
    The integer ID of the first node that inherits this mutation.
    To obtain further information about a node with a given ID, use
    :meth:`TreeSequence.node`.
    """
    derived_state: str
    """
    The derived state for this mutation. This is the state
    inherited by nodes in the subtree rooted at this mutation's node, unless
    another mutation occurs.
    """
    parent: int
    """
    The integer ID of this mutation's parent mutation. When multiple
    mutations occur at a site along a path in the tree, mutations must
    record the mutation that is immediately above them. If the mutation does
    not have a parent, this is equal to the :data:`NULL` (-1).
    To obtain further information about a mutation with a given ID, use
    :meth:`TreeSequence.mutation`.
    """
    metadata: bytes | dict | None
    """
    The :ref:`metadata <sec_metadata_definition>` for this mutation, decoded if a schema
    applies.
    """
    time: float
    """
    The occurrence time of this mutation.
    """
    edge: int
    """
    The ID of the edge that this mutation is on.
    """
    inherited_state: str
    """
    The inherited state for this mutation. This is the state that existed at the site
    before this mutation occurred. This is either the ancestral state of the site
    (if the mutation has no parent) or the derived state of the mutation's
    parent mutation (if it has a parent).
    """

    # To get default values on slots we define a custom init
    def __init__(
        self,
        id=NULL,  # noqa A003
        site=NULL,
        node=NULL,
        time=UNKNOWN_TIME,
        derived_state=None,
        parent=NULL,
        metadata=b"",
        edge=NULL,
        inherited_state=None,
    ):
        self.id = id
        self.site = site
        self.node = node
        self.time = time
        self.derived_state = derived_state
        self.parent = parent
        self.metadata = metadata
        self.edge = edge
        self.inherited_state = inherited_state

    # We need a custom eq to compare unknown times.
    def __eq__(self, other):
        return (
            isinstance(other, Mutation)
            and self.id == other.id
            and self.site == other.site
            and self.node == other.node
            and self.derived_state == other.derived_state
            and self.parent == other.parent
            and self.edge == other.edge
            and self.metadata == other.metadata
            and (
                self.time == other.time
                or (
                    util.is_unknown_time(self.time) and util.is_unknown_time(other.time)
                )
            )
        )


@metadata_module.lazy_decode()
@dataclass
class Migration(util.Dataclass):
    """
    A :ref:`migration <sec_migration_table_definition>` in a tree sequence.

    Modifying the attributes in this class will have **no effect** on the
    underlying tree sequence data.
    """

    __slots__ = ["left", "right", "node", "source", "dest", "time", "metadata", "id"]
    left: float
    """
    The left end of the genomic interval covered by this
    migration (inclusive).
    """
    right: float
    """
    The right end of the genomic interval covered by this migration
    (exclusive).
    """
    node: int
    """
    The integer ID of the node involved in this migration event.
    To obtain further information about a node with a given ID, use
    :meth:`TreeSequence.node`.
    """
    source: int
    """
    The source population ID.
    """
    dest: int
    """
    The destination population ID.
    """
    time: float
    """
    The time at which this migration occurred at.
    """
    metadata: bytes | dict | None
    """
    The :ref:`metadata <sec_metadata_definition>` for this migration, decoded if a schema
    applies.
    """
    id: int  # noqa A003
    """
    The integer ID of this mutation. Varies from 0 to
    :attr:`TreeSequence.num_mutations` - 1.
    """


@metadata_module.lazy_decode()
@dataclass
class Population(util.Dataclass):
    """
    A :ref:`population <sec_population_table_definition>` in a tree sequence.

    Modifying the attributes in this class will have **no effect** on the
    underlying tree sequence data.
    """

    __slots__ = ["id", "metadata"]
    id: int  # noqa A003
    """
    The integer ID of this population. Varies from 0 to
    :attr:`TreeSequence.num_populations` - 1.
    """
    metadata: bytes | dict | None
    """
    The :ref:`metadata <sec_metadata_definition>` for this population, decoded if a
    schema applies.
    """


@dataclass
class Edgeset(util.Dataclass):
    __slots__ = ["left", "right", "parent", "children"]
    left: int
    right: int
    parent: int
    children: np.ndarray

    # We need a custom eq for the numpy array
    def __eq__(self, other):
        return (
            isinstance(other, Edgeset)
            and self.left == other.left
            and self.right == other.right
            and self.parent == other.parent
            and np.array_equal(self.children, other.children)
        )


@dataclass
class Provenance(util.Dataclass):
    """
    A provenance entry in a tree sequence, detailing how this tree
    sequence was generated, or subsequent operations on it (see :ref:`sec_provenance`).
    """

    __slots__ = ["id", "timestamp", "record"]
    id: int  # noqa A003
    timestamp: str
    """
    The time that this entry was made
    """
    record: str
    """
    A JSON string giving details of the provenance (see :ref:`sec_provenance_example`
    for an example JSON string)
    """


class Tree:
    """
    A single tree in a :class:`TreeSequence`. Please see the
    :ref:`tutorials:sec_processing_trees` section for information
    on how efficiently access trees sequentially or obtain a list
    of individual trees in a tree sequence.

    The ``sample_lists`` parameter controls the features that are enabled
    for this tree. If ``sample_lists`` is True a more efficient algorithm is
    used in the :meth:`Tree.samples` method.

    The ``tracked_samples`` parameter can be used to efficiently count the
    number of samples in a given set that exist in a particular subtree
    using the :meth:`Tree.num_tracked_samples` method.

    The :class:`Tree` class is a state-machine which has a state
    corresponding to each of the trees in the parent tree sequence. We
    transition between these states by using the seek functions like
    :meth:`Tree.first`, :meth:`Tree.last`, :meth:`Tree.seek` and
    :meth:`Tree.seek_index`. There is one more state, the so-called "null"
    or "cleared" state. This is the state that a :class:`Tree` is in
    immediately after initialisation;  it has an index of -1, and no edges. We
    can also enter the null state by calling :meth:`Tree.next` on the last
    tree in a sequence, calling :meth:`Tree.prev` on the first tree in a
    sequence or calling calling the :meth:`Tree.clear` method at any time.

    The high-level TreeSequence seeking and iterations methods (e.g,
    :meth:`TreeSequence.trees`) are built on these low-level state-machine
    seek operations. We recommend these higher level operations for most
    users.

    :param TreeSequence tree_sequence: The parent tree sequence.
    :param list tracked_samples: The list of samples to be tracked and
        counted using the :meth:`Tree.num_tracked_samples` method.
    :param bool sample_lists: If True, provide more efficient access
        to the samples beneath a given node using the
        :meth:`Tree.samples` method.
    :param int root_threshold: The minimum number of samples that a node
        must be ancestral to for it to be in the list of roots. By default
        this is 1, so that isolated samples (representing missing data)
        are roots. To efficiently restrict the roots of the tree to
        those subtending meaningful topology, set this to 2. This value
        is only relevant when trees have multiple roots.
    :param bool sample_counts: Deprecated since 0.2.4.
    """

    def __init__(
        self,
        tree_sequence,
        tracked_samples=None,
        *,
        sample_lists=False,
        root_threshold=1,
        sample_counts=None,
    ):
        options = 0
        if sample_counts is not None:
            warnings.warn(
                "The sample_counts option is not supported since 0.2.4 "
                "and is ignored",
                RuntimeWarning,
                stacklevel=4,
            )
        if sample_lists:
            options |= _tskit.SAMPLE_LISTS
        kwargs = {"options": options}
        if root_threshold <= 0:
            raise ValueError("Root threshold must be greater than 0")
        if tracked_samples is not None:
            # TODO remove this when we allow numpy arrays in the low-level API.
            kwargs["tracked_samples"] = list(tracked_samples)

        self._tree_sequence = tree_sequence
        self._ll_tree = _tskit.Tree(tree_sequence.ll_tree_sequence, **kwargs)
        self._ll_tree.set_root_threshold(root_threshold)
        self._make_arrays()

    def copy(self):
        """
        Returns a deep copy of this tree. The returned tree will have identical state
        to this tree.

        :return: A copy of this tree.
        :rtype: Tree
        """
        copy = type(self).__new__(type(self))
        copy._tree_sequence = self._tree_sequence
        copy._ll_tree = self._ll_tree.copy()
        copy._make_arrays()
        return copy

    # TODO make this method public and document it.
    # Note that this probably does not cover all corner cases correctly
    # https://github.com/tskit-dev/tskit/issues/1908
    def _has_isolated_samples(self):
        # TODO Is this definition correct for a single-node tree sequence?
        for root in self.roots:
            # If the root has no children then it must be a sample
            if self.left_child(root) == NULL:
                return True
        return False

    def _make_arrays(self):
        # Store the low-level arrays for efficiency. There's no real overhead
        # in this, because the refer to the same underlying memory as the
        # tree object.
        self._parent_array = self._ll_tree.parent_array
        self._left_child_array = self._ll_tree.left_child_array
        self._right_child_array = self._ll_tree.right_child_array
        self._left_sib_array = self._ll_tree.left_sib_array
        self._right_sib_array = self._ll_tree.right_sib_array
        self._num_children_array = self._ll_tree.num_children_array
        self._edge_array = self._ll_tree.edge_array

    @property
    def tree_sequence(self):
        """
        Returns the tree sequence that this tree is from.

        :return: The parent tree sequence for this tree.
        :rtype: :class:`TreeSequence`
        """
        return self._tree_sequence

    @property
    def root_threshold(self):
        """
        Returns the minimum number of samples that a node must be an ancestor
        of to be considered a potential root. This can be set, for example, when
        calling the :meth:`TreeSequence.trees` iterator.

        :return: The root threshold.
        :rtype: :class:`TreeSequence`
        """
        return self._ll_tree.get_root_threshold()

    def __eq__(self, other):
        ret = False
        if type(other) is type(self):
            ret = bool(self._ll_tree.equals(other._ll_tree))
        return ret

    def __ne__(self, other):
        return not self.__eq__(other)

    def first(self):
        """
        Seeks to the first tree in the sequence. This can be called whether
        the tree is in the null state or not.
        """
        self._ll_tree.first()

    def last(self):
        """
        Seeks to the last tree in the sequence. This can be called whether
        the tree is in the null state or not.
        """
        self._ll_tree.last()

    def next(self):  # noqa A002
        """
        Seeks to the next tree in the sequence. If the tree is in the initial
        null state we seek to the first tree (equivalent to calling :meth:`~Tree.first`).
        Calling ``next`` on the last tree in the sequence results in the tree
        being cleared back into the null initial state (equivalent to calling
        :meth:`~Tree.clear`). The return value of the function indicates whether the
        tree is in a non-null state, and can be used to loop over the trees::

            # Iterate over the trees from left-to-right
            tree = tskit.Tree(tree_sequence)
            while tree.next()
                # Do something with the tree.
                print(tree.index)
            # tree is now back in the null state.

        :return: True if the tree has been transformed into one of the trees
            in the sequence; False if the tree has been transformed into the
            null state.
        :rtype: bool
        """
        return bool(self._ll_tree.next())

    def prev(self):
        """
        Seeks to the previous tree in the sequence. If the tree is in the initial
        null state we seek to the last tree (equivalent to calling :meth:`~Tree.last`).
        Calling ``prev`` on the first tree in the sequence results in the tree
        being cleared back into the null initial state (equivalent to calling
        :meth:`~Tree.clear`). The return value of the function indicates whether the
        tree is in a non-null state, and can be used to loop over the trees::

            # Iterate over the trees from right-to-left
            tree = tskit.Tree(tree_sequence)
            while tree.prev()
                # Do something with the tree.
                print(tree.index)
            # tree is now back in the null state.

        :return: True if the tree has been transformed into one of the trees
            in the sequence; False if the tree has been transformed into the
            null state.
        :rtype: bool
        """
        return bool(self._ll_tree.prev())

    def clear(self):
        """
        Resets this tree back to the initial null state. Calling this method
        on a tree already in the null state has no effect.
        """
        self._ll_tree.clear()

    def seek_index(self, index, skip=None):
        """
        Sets the state to represent the tree at the specified
        index in the parent tree sequence. Negative indexes following the
        standard Python conventions are allowed, i.e., ``index=-1`` will
        seek to the last tree in the sequence.

        .. include:: substitutions/linear_traversal_warning.rst

        :param int index: The tree index to seek to.
        :raises IndexError: If an index outside the acceptable range is provided.
        """
        num_trees = self.tree_sequence.num_trees
        if index < 0:
            index += num_trees
        if index < 0 or index >= num_trees:
            raise IndexError("Index out of bounds")
        skip = False if skip is None else skip
        self._ll_tree.seek_index(index, skip)

    def seek(self, position, skip=None):
        """
        Sets the state to represent the tree that covers the specified
        position in the parent tree sequence. After a successful return
        of this method we have ``tree.interval.left`` <= ``position``
        < ``tree.interval.right``.

        .. include:: substitutions/linear_traversal_warning.rst

        :param float position: The position along the sequence length to
            seek to.
        :raises ValueError: If 0 < position or position >=
            :attr:`TreeSequence.sequence_length`.
        """
        if position < 0 or position >= self.tree_sequence.sequence_length:
            raise ValueError("Position out of bounds")
        skip = False if skip is None else skip
        self._ll_tree.seek(position, skip)

    def rank(self) -> tskit.Rank:
        """
        Produce the rank of this tree in the enumeration of all leaf-labelled
        trees of n leaves. See the :ref:`sec_tree_ranks` section for
        details on ranking and unranking trees.

        :raises ValueError: If the tree has multiple roots.
        """
        return combinatorics.RankTree.from_tsk_tree(self).rank()

    @staticmethod
    def unrank(num_leaves, rank, *, span=1, branch_length=1) -> Tree:
        """
        Reconstruct the tree of the given ``rank``
        (see :meth:`tskit.Tree.rank`) with ``num_leaves`` leaves.
        The labels and times of internal nodes are assigned by a postorder
        traversal of the nodes, such that the time of each internal node
        is the maximum time of its children plus the specified ``branch_length``.
        The time of each leaf is 0.

        See the :ref:`sec_tree_ranks` section for details on ranking and
        unranking trees and what constitutes valid ranks.

        :param int num_leaves: The number of leaves of the tree to generate.
        :param tuple(int) rank: The rank of the tree to generate.
        :param float span: The genomic span of the returned tree. The tree will cover
            the interval :math:`[0, \\text{span})` and the :attr:`~Tree.tree_sequence`
            from which the tree is taken will have its
            :attr:`~tskit.TreeSequence.sequence_length` equal to ``span``.
        :param: float branch_length: The minimum length of a branch in this tree.
        :raises ValueError: If the given rank is out of bounds for trees
            with ``num_leaves`` leaves.
        """
        rank_tree = combinatorics.RankTree.unrank(num_leaves, rank)
        return rank_tree.to_tsk_tree(span=span, branch_length=branch_length)

    def count_topologies(self, sample_sets=None) -> tskit.TopologyCounter:
        """
        Calculates the distribution of embedded topologies for every combination
        of the sample sets in ``sample_sets``. ``sample_sets`` defaults to all
        samples in the tree grouped by population.

        ``sample_sets`` need not include all samples but must be pairwise disjoint.

        The returned object is a :class:`tskit.TopologyCounter` that contains
        counts of topologies per combination of sample sets. For example::

            topology_counter = tree.count_topologies()
            rank, count = topology_counter[0, 1, 2].most_common(1)[0]

        produces the most common tree topology, with populations 0, 1
        and 2 as its tips, according to the genealogies of those
        populations' samples in this tree.

        The counts for each topology in the :class:`tskit.TopologyCounter`
        are absolute counts that we would get if we were to select all
        combinations of samples from the relevant sample sets.
        For sample sets :math:`[s_0, s_1, ..., s_n]`, the total number of
        topologies for those sample sets is equal to
        :math:`|s_0| * |s_1| * ... * |s_n|`, so the counts in the counter
        ``topology_counter[0, 1, ..., n]`` should sum to
        :math:`|s_0| * |s_1| * ... * |s_n|`.

        To convert the topology counts to probabilities, divide by the total
        possible number of sample combinations from the sample sets in question::

            set_sizes = [len(sample_set) for sample_set in sample_sets]
            p = count / (set_sizes[0] * set_sizes[1] * set_sizes[2])

        .. warning:: The interface for this method is preliminary and may be subject to
            backwards incompatible changes in the near future.

        :param list sample_sets: A list of lists of Node IDs, specifying the
            groups of nodes to compute the statistic with.
            Defaults to all samples grouped by population.
        :raises ValueError: If nodes in ``sample_sets`` are invalid or are
            internal samples.
        """
        if sample_sets is None:
            sample_sets = [
                self.tree_sequence.samples(population=pop.id)
                for pop in self.tree_sequence.populations()
            ]

        return combinatorics.tree_count_topologies(self, sample_sets)

    def get_branch_length(self, u):
        # Deprecated alias for branch_length
        return self.branch_length(u)

    def branch_length(self, u):
        """
        Returns the length of the branch (in units of time) joining the
        specified node to its parent. This is equivalent to::

            tree.time(tree.parent(u)) - tree.time(u)

        The branch length for a node that has no parent (e.g., a root) is
        defined as zero.

        Note that this is not related to the property `.length` which
        is a deprecated alias for the genomic :attr:`~Tree.span` covered by a tree.

        :param int u: The node of interest.
        :return: The branch length from u to its parent.
        :rtype: float
        """
        ret = 0
        parent = self.parent(u)
        if parent != NULL:
            ret = self.time(parent) - self.time(u)
        return ret

    def get_total_branch_length(self):
        # Deprecated alias for total_branch_length
        return self.total_branch_length

    @property
    def total_branch_length(self):
        """
        Returns the sum of all the branch lengths in this tree (in
        units of time). This is equivalent to::

            sum(tree.branch_length(u) for u in tree.nodes())

        Note that the branch lengths for root nodes are defined as zero.

        As this is defined by a traversal of the tree, technically we
        return the sum of all branch lengths that are reachable from
        roots. Thus, this is the total length of all branches that are connected
        to at least one sample. This distinction is only important
        in tree sequences that contain 'dead branches', i.e., those
        that define topology that is not connected to a tree root
        (see :ref:`sec_data_model_tree_dead_leaves_and_branches`)

        :return: The sum of lengths of branches in this tree.
        :rtype: float
        """
        return self._ll_tree.get_total_branch_length()

    def get_mrca(self, u, v):
        # Deprecated alias for mrca
        return self.mrca(u, v)

    def mrca(self, *args):
        """
        Returns the most recent common ancestor of the specified nodes.

        :param int `*args`: input node IDs, at least 2 arguments are required.
        :return: The node ID of the most recent common ancestor of the
            input nodes, or :data:`tskit.NULL` if the nodes do not share
            a common ancestor in the tree.
        :rtype: int
        """
        if len(args) < 2:
            raise ValueError("Must supply at least two arguments")
        mrca = args[0]
        for node in args[1:]:
            mrca = self._ll_tree.get_mrca(mrca, node)
            if mrca == tskit.NULL:
                break
        return mrca

    def get_tmrca(self, u, v):
        # Deprecated alias for tmrca
        return self.tmrca(u, v)

    def tmrca(self, *args):
        """
        Returns the time of the most recent common ancestor of the specified
        nodes. This is equivalent to::

            tree.time(tree.mrca(*args))

        .. note::
            If you are using this method to calculate average tmrca values along the
            genome between pairs of sample nodes, for efficiency reasons you should
            instead consider the ``mode="branch"`` option of the
            :meth:`TreeSequence.divergence` or :meth:`TreeSequence.diversity` methods.
            Since these calculate the average branch length between pairs of sample
            nodes, for samples at time 0 the resulting statistics will be exactly
            twice the tmrca value.

        :param `*args`: input node IDs, at least 2 arguments are required.
        :return: The time of the most recent common ancestor of all the nodes.
        :rtype: float
        :raises ValueError: If the nodes do not share a single common ancestor in this
            tree (i.e., if ``tree.mrca(*args) == tskit.NULL``)
        """
        mrca = self.mrca(*args)
        if mrca == tskit.NULL:
            raise ValueError(f"Nodes {args} do not share a common ancestor in the tree")
        return self.get_time(mrca)

    def get_parent(self, u):
        # Deprecated alias for parent
        return self.parent(u)

    def parent(self, u):
        """
        Returns the parent of the specified node. Returns
        :data:`tskit.NULL` if u is a root or is not a node in
        the current tree.

        :param int u: The node of interest.
        :return: The parent of u.
        :rtype: int
        """
        return self._ll_tree.get_parent(u)

    @property
    def parent_array(self):
        """
        A numpy array (dtype=np.int32) encoding the parent of each node
        in this tree, such that ``tree.parent_array[u] == tree.parent(u)``
        for all ``0 <= u <= ts.num_nodes``. See the :meth:`~.parent`
        method for details on the semantics of tree parents and the
        :ref:`sec_data_model_tree_structure` section for information on the
        quintuply linked tree encoding.

        .. include:: substitutions/virtual_root_array_note.rst

        .. include:: substitutions/tree_array_warning.rst
        """
        return self._parent_array

    def ancestors(self, u):
        """
        Returns an iterator over the ancestors of node ``u`` in this tree
        (i.e. the chain of parents from ``u`` to the root).
        """
        u = self.parent(u)
        while u != -1:
            yield u
            u = self.parent(u)

    # Quintuply linked tree structure.

    def left_child(self, u):
        """
        Returns the leftmost child of the specified node. Returns
        :data:`tskit.NULL` if u is a leaf or is not a node in
        the current tree. The left-to-right ordering of children
        is arbitrary and should not be depended on; see the
        :ref:`data model <sec_data_model_tree_structure>` section
        for details.

        This is a low-level method giving access to the quintuply linked
        tree structure in memory; the :meth:`.children` method is a more
        convenient way to obtain the children of a given node.

        :param int u: The node of interest.
        :return: The leftmost child of u.
        :rtype: int
        """
        return self._ll_tree.get_left_child(u)

    @property
    def left_child_array(self):
        """
        A numpy array (dtype=np.int32) encoding the left child of each node
        in this tree, such that ``tree.left_child_array[u] == tree.left_child(u)``
        for all ``0 <= u <= ts.num_nodes``. See the :meth:`~.left_child`
        method for details on the semantics of tree left_child and the
        :ref:`sec_data_model_tree_structure` section for information on the
        quintuply linked tree encoding.

        .. include:: substitutions/virtual_root_array_note.rst

        .. include:: substitutions/tree_array_warning.rst
        """
        return self._left_child_array

    def right_child(self, u):
        """
        Returns the rightmost child of the specified node. Returns
        :data:`tskit.NULL` if u is a leaf or is not a node in
        the current tree. The left-to-right ordering of children
        is arbitrary and should not be depended on; see the
        :ref:`data model <sec_data_model_tree_structure>` section
        for details.

        This is a low-level method giving access to the quintuply linked
        tree structure in memory; the :meth:`.children` method is a more
        convenient way to obtain the children of a given node.

        :param int u: The node of interest.
        :return: The rightmost child of u.
        :rtype: int
        """
        return self._ll_tree.get_right_child(u)

    @property
    def right_child_array(self):
        """
        A numpy array (dtype=np.int32) encoding the right child of each node
        in this tree, such that ``tree.right_child_array[u] == tree.right_child(u)``
        for all ``0 <= u <= ts.num_nodes``. See the :meth:`~.right_child`
        method for details on the semantics of tree right_child and the
        :ref:`sec_data_model_tree_structure` section for information on the
        quintuply linked tree encoding.

        .. include:: substitutions/virtual_root_array_note.rst

        .. include:: substitutions/tree_array_warning.rst
        """
        return self._right_child_array

    def left_sib(self, u):
        """
        Returns the sibling node to the left of u, or :data:`tskit.NULL`
        if u does not have a left sibling.
        The left-to-right ordering of children
        is arbitrary and should not be depended on; see the
        :ref:`data model <sec_data_model_tree_structure>` section
        for details.

        :param int u: The node of interest.
        :return: The sibling node to the left of u.
        :rtype: int
        """
        return self._ll_tree.get_left_sib(u)

    @property
    def left_sib_array(self):
        """
        A numpy array (dtype=np.int32) encoding the left sib of each node
        in this tree, such that ``tree.left_sib_array[u] == tree.left_sib(u)``
        for all ``0 <= u <= ts.num_nodes``. See the :meth:`~.left_sib`
        method for details on the semantics of tree left_sib and the
        :ref:`sec_data_model_tree_structure` section for information on the
        quintuply linked tree encoding.

        .. include:: substitutions/virtual_root_array_note.rst

        .. include:: substitutions/tree_array_warning.rst
        """
        return self._left_sib_array

    def right_sib(self, u):
        """
        Returns the sibling node to the right of u, or :data:`tskit.NULL`
        if u does not have a right sibling.
        The left-to-right ordering of children
        is arbitrary and should not be depended on; see the
        :ref:`data model <sec_data_model_tree_structure>` section
        for details.

        :param int u: The node of interest.
        :return: The sibling node to the right of u.
        :rtype: int
        """
        return self._ll_tree.get_right_sib(u)

    @property
    def right_sib_array(self):
        """
        A numpy array (dtype=np.int32) encoding the right sib of each node
        in this tree, such that ``tree.right_sib_array[u] == tree.right_sib(u)``
        for all ``0 <= u <= ts.num_nodes``. See the :meth:`~.right_sib`
        method for details on the semantics of tree right_sib and the
        :ref:`sec_data_model_tree_structure` section for information on the
        quintuply linked tree encoding.

        .. include:: substitutions/virtual_root_array_note.rst

        .. include:: substitutions/tree_array_warning.rst
        """
        return self._right_sib_array

    def siblings(self, u):
        """
        Returns the sibling(s) of the specified node ``u`` as a tuple of integer
        node IDs. If ``u`` has no siblings or is not a node in the current tree,
        returns an empty tuple. If ``u`` is the root of a single-root tree,
        returns an empty tuple; if ``u`` is the root of a multi-root tree,
        returns the other roots (note all the roots are related by the virtual root).
        If ``u`` is the virtual root (which has no siblings), returns an empty tuple.
        If ``u`` is an isolated node, whether it has siblings or not depends on
        whether it is a sample or non-sample node; if it is a sample node,
        returns the root(s) of the tree, otherwise, returns an empty tuple.
        The ordering of siblings  is arbitrary and should not be depended on;
        see the :ref:`data model <sec_data_model_tree_structure>` section for details.

        :param int u: The node of interest.
        :return: The siblings of ``u``.
        :rtype: tuple(int)
        """
        if u == self.virtual_root:
            return tuple()
        parent = self.parent(u)
        if self.is_root(u):
            parent = self.virtual_root
        if parent != tskit.NULL:
            return tuple(v for v in self.children(parent) if u != v)
        return tuple()

    @property
    def num_children_array(self):
        """
        A numpy array (dtype=np.int32) encoding the number of children of
        each node in this tree, such that
        ``tree.num_children_array[u] == tree.num_children(u)`` for all
        ``0 <= u <= ts.num_nodes``. See the :meth:`~.num_children`
        method for details on the semantics of tree num_children and the
        :ref:`sec_data_model_tree_structure` section for information on the
        quintuply linked tree encoding.

        .. include:: substitutions/virtual_root_array_note.rst

        .. include:: substitutions/tree_array_warning.rst
        """
        return self._num_children_array

    def edge(self, u):
        """
        Returns the id of the edge encoding the relationship between ``u``
        and its parent, or :data:`tskit.NULL` if ``u`` is a root, virtual root
        or is not a node in the current tree.

        :param int u: The node of interest.
        :return: Id of edge connecting u to its parent.
        :rtype: int
        """
        return self._ll_tree.get_edge(u)

    @property
    def edge_array(self):
        """
        A numpy array (dtype=np.int32) of edge ids encoding the relationship
        between the child node ``u`` and its parent, such that
        ``tree.edge_array[u] == tree.edge(u)`` for all
        ``0 <= u <= ts.num_nodes``. See the :meth:`~.edge`
        method for details on the semantics of tree edge and the
        :ref:`sec_data_model_tree_structure` section for information on the
        quintuply linked tree encoding.

        .. include:: substitutions/virtual_root_array_note.rst

        .. include:: substitutions/tree_array_warning.rst
        """
        return self._edge_array

    # Sample list.

    def left_sample(self, u):
        return self._ll_tree.get_left_sample(u)

    def right_sample(self, u):
        return self._ll_tree.get_right_sample(u)

    def next_sample(self, u):
        return self._ll_tree.get_next_sample(u)

    @property
    def virtual_root(self):
        """
        The ID of the virtual root in this tree. This is equal to
        :attr:`TreeSequence.num_nodes`.

        Please see the :ref:`tree roots <sec_data_model_tree_roots>`
        section for more details.
        """
        return self._ll_tree.get_virtual_root()

    @property
    def num_edges(self):
        """
        The total number of edges in this tree. This is equal to the
        number of tree sequence edges that intersect with this tree's
        genomic interval.

        Note that this may be greater than the number of branches that
        are reachable from the tree's roots, since we can have topology
        that is not associated with any samples.
        """
        return self._ll_tree.get_num_edges()

    @property
    def left_root(self):
        """
        The leftmost root in this tree. If there are multiple roots
        in this tree, they are siblings of this node, and so we can
        use :meth:`.right_sib` to iterate over all roots:

        .. code-block:: python

            u = tree.left_root
            while u != tskit.NULL:
                print("Root:", u)
                u = tree.right_sib(u)

        The left-to-right ordering of roots is arbitrary and should
        not be depended on; see the
        :ref:`data model <sec_data_model_tree_structure>`
        section for details.

        This is a low-level method giving access to the quintuply linked
        tree structure in memory; the :attr:`~Tree.roots` attribute is a more
        convenient way to obtain the roots of a tree. If you are assuming
        that there is a single root in the tree you should use the
        :attr:`~Tree.root` property.

        .. warning:: Do not use this property if you are assuming that there
            is a single root in trees that are being processed. The
            :attr:`~Tree.root` property should be used in this case, as it will
            raise an error when multiple roots exists.

        :rtype: int
        """
        return self.left_child(self.virtual_root)

    @property
    def right_root(self):
        return self.right_child(self.virtual_root)

    def get_children(self, u):
        # Deprecated alias for self.children
        return self.children(u)

    def children(self, u):
        """
        Returns the children of the specified node ``u`` as a tuple of integer node IDs.
        If ``u`` is a leaf, return the empty tuple. The ordering of children
        is arbitrary and should not be depended on; see the
        :ref:`data model <sec_data_model_tree_structure>` section
        for details.

        :param int u: The node of interest.
        :return: The children of ``u`` as a tuple of integers
        :rtype: tuple(int)
        """
        return self._ll_tree.get_children(u)

    def get_time(self, u):
        # Deprecated alias for self.time
        return self.time(u)

    def time(self, u):
        """
        Returns the time of the specified node. This is equivalently
        to ``tree.tree_sequence.node(u).time`` except for the special
        case of the tree's :ref:`virtual root <sec_data_model_tree_roots>`,
        which is defined as positive infinity.

        :param int u: The node of interest.
        :return: The time of u.
        :rtype: float
        """
        return self._ll_tree.get_time(u)

    def depth(self, u):
        """
        Returns the number of nodes on the path from ``u`` to a
        root, not including ``u``. Thus, the depth of a root is
        zero.

        As a special case, the depth of the :ref:`virtual root
        <sec_data_model_tree_roots>` is defined as -1.

        :param int u: The node of interest.
        :return: The depth of u.
        :rtype: int
        """
        return self._ll_tree.depth(u)

    def get_population(self, u):
        # Deprecated alias for self.population
        return self.population(u)

    def population(self, u):
        """
        Returns the population associated with the specified node.
        Equivalent to ``tree.tree_sequence.node(u).population``.

        :param int u: The node of interest.
        :return: The ID of the population associated with node u.
        :rtype: int
        """
        return self._ll_tree.get_population(u)

    def is_internal(self, u):
        """
        Returns True if the specified node is not a leaf. A node is internal
        if it has one or more children in the current tree.

        :param int u: The node of interest.
        :return: True if u is not a leaf node.
        :rtype: bool
        """
        return not self.is_leaf(u)

    def is_leaf(self, u):
        """
        Returns True if the specified node is a leaf. A node :math:`u` is a
        leaf if it has zero children.

        .. note::
            :math:`u` can be any node in the entire tree sequence, including ones
            which are not connected via branches to a root node of the tree (and which
            are therefore not conventionally considered part of the tree). Indeed, if
            there are many trees in the tree sequence, it is common for the majority of
            non-sample nodes to be :meth:`isolated<is_isolated>` in any one
            tree. By the definition above, this method will return ``True`` for such
            a tree when a node of this sort is specified. Such nodes can be thought of
            as "dead leaves", see :ref:`sec_data_model_tree_dead_leaves_and_branches`.

        :param int u: The node of interest.
        :return: True if u is a leaf node.
        :rtype: bool
        """
        return len(self.children(u)) == 0

    def is_isolated(self, u):
        """
        Returns True if the specified node is isolated in this tree: that is
        it has no parents and no children (note that all isolated nodes in the tree
        are therefore also :meth:`leaves<Tree.is_leaf>`). Sample nodes that are isolated
        and have no mutations above them are used to represent
        :ref:`missing data<sec_data_model_missing_data>`.

        :param int u: The node of interest.
        :return: True if u is an isolated node.
        :rtype: bool
        """
        return self.num_children(u) == 0 and self.parent(u) == NULL

    def is_sample(self, u):
        """
        Returns True if the specified node is a sample. A node :math:`u` is a
        sample if it has been marked as a sample in the parent tree sequence.

        :param int u: The node of interest.
        :return: True if u is a sample.
        :rtype: bool
        """
        return bool(self._ll_tree.is_sample(u))

    def is_descendant(self, u, v):
        """
        Returns True if the specified node u is a descendant of node v and False
        otherwise. A node :math:`u` is a descendant of another node :math:`v` if
        :math:`v` is on the path from :math:`u` to root. A node is considered
        to be a descendant of itself, so ``tree.is_descendant(u, u)`` will be
        True for any valid node.

        :param int u: The descendant node.
        :param int v: The ancestral node.
        :return: True if u is a descendant of v.
        :rtype: bool
        :raises ValueError: If u or v are not valid node IDs.
        """
        return bool(self._ll_tree.is_descendant(u, v))

    @property
    def num_nodes(self):
        """
        Returns the number of nodes in the :class:`TreeSequence` this tree is in.
        Equivalent to ``tree.tree_sequence.num_nodes``.

        .. deprecated:: 0.4
            Use :attr:`Tree.tree_sequence.num_nodes<TreeSequence.num_nodes>` if you want
            the number of nodes in the entire tree sequence, or
            ``len(tree.preorder())`` to find the number of nodes that are
            reachable from all roots in this tree.

        :rtype: int

        """
        warnings.warn(
            "This property is a deprecated alias for Tree.tree_sequence.num_nodes "
            "and will be removed in the future. To obtain the number of nodes "
            "in the topology of the current tree (i.e. reachable from the roots) "
            "use len(tree.preorder()).",
            FutureWarning,
            stacklevel=4,
        )
        return self.tree_sequence.num_nodes

    @property
    def num_roots(self):
        """
        The number of roots in this tree, as defined in the :attr:`~Tree.roots`
        attribute.

        Only requires O(number of roots) time.

        :rtype: int
        """
        return self._ll_tree.get_num_roots()

    @property
    def has_single_root(self):
        """
        ``True`` if this tree has a single root, ``False`` otherwise.
        Equivalent to tree.num_roots == 1. This is a O(1) operation.

        :rtype: bool
        """
        root = self.left_root
        if root != NULL and self.right_sib(root) == NULL:
            return True
        return False

    @property
    def has_multiple_roots(self):
        """
        ``True`` if this tree has more than one root, ``False`` otherwise.
        Equivalent to tree.num_roots > 1. This is a O(1) operation.

        :rtype: bool
        """
        root = self.left_root
        if root != NULL and self.right_sib(root) != NULL:
            return True
        return False

    @property
    def roots(self):
        """
        The list of roots in this tree. A root is defined as a unique endpoint of the
        paths starting at samples, subject to the condition that it is connected to at
        least :attr:`root_threshold` samples. We can define the set of roots as follows:

        .. code-block:: python

            roots = set()
            for u in tree_sequence.samples():
                while tree.parent(u) != tskit.NULL:
                    u = tree.parent(u)
                if tree.num_samples(u) >= tree.root_threshold:
                    roots.add(u)
            # roots is now the set of all roots in this tree.
            assert sorted(roots) == sorted(tree.roots)

        The roots of the tree are returned in a list, in no particular order.

        Only requires O(number of roots) time.

        .. note::
            In trees with large amounts of :ref:`sec_data_model_missing_data`,
            for example where a region of the genome lacks any ancestral information,
            there can be a very large number of roots, potentially all the samples
            in the tree sequence.

        :return: The list of roots in this tree.
        :rtype: list
        """
        roots = []
        u = self.left_root
        while u != NULL:
            roots.append(u)
            u = self.right_sib(u)
        return roots

    def get_root(self):
        # Deprecated alias for self.root
        return self.root

    @property
    def root(self):
        """
        The root of this tree. If the tree contains multiple roots, a ValueError is
        raised indicating that the :attr:`~Tree.roots` attribute should be used instead.

        :return: The root node.
        :rtype: int
        :raises ValueError: if this tree contains more than one root.
        """
        if self.has_multiple_roots:
            raise ValueError("More than one root exists. Use tree.roots instead")
        return self.left_root

    def is_root(self, u) -> bool:
        """
        Returns ``True`` if the specified node is a root in this tree (see
        :attr:`~Tree.roots` for the definition of a root). This is exactly equivalent to
        finding the node ID in :attr:`~Tree.roots`, but is more efficient for trees
        with large numbers of roots, such as in regions with extensive
        :ref:`sec_data_model_missing_data`.  Note that ``False`` is returned for all
        other nodes, including :ref:`isolated<sec_data_model_tree_isolated_nodes>`
        non-sample nodes which are not found in the topology of the current tree.

        :param int u: The node of interest.
        :return: ``True`` if u is a root.
        """
        return (
            self.num_samples(u) >= self.root_threshold and self.parent(u) == tskit.NULL
        )

    def get_index(self):
        # Deprecated alias for self.index
        return self.index

    @property
    def index(self):
        """
        Returns the index this tree occupies in the parent tree sequence.
        This index is zero based, so the first tree in the sequence has index 0.

        :return: The index of this tree.
        :rtype: int
        """
        return self._ll_tree.get_index()

    def get_interval(self):
        # Deprecated alias for self.interval
        return self.interval

    @property
    def interval(self):
        """
        Returns the coordinates of the genomic interval that this tree
        represents the history of. The interval is returned as a tuple
        :math:`(l, r)` and is a half-open interval such that the left
        coordinate is inclusive and the right coordinate is exclusive. This
        tree therefore applies to all genomic locations :math:`x` such that
        :math:`l \\leq x < r`.

        :return: A named tuple (l, r) representing the left-most (inclusive)
            and right-most (exclusive) coordinates of the genomic region
            covered by this tree. The coordinates can be accessed by index
            (``0`` or ``1``) or equivalently by name (``.left`` or ``.right``)
        :rtype: Interval
        """
        return Interval(self._ll_tree.get_left(), self._ll_tree.get_right())

    def get_length(self):
        # Deprecated alias for self.span
        return self.length

    @property
    def length(self):
        # Deprecated alias for self.span
        return self.span

    @property
    def span(self):
        """
        Returns the genomic distance that this tree spans.
        This is defined as :math:`r - l`, where :math:`(l, r)` is the genomic
        interval returned by :attr:`~Tree.interval`.

        :return: The genomic distance covered by this tree.
        :rtype: float
        """
        return self.interval.span

    @property
    def mid(self):
        """
        Returns the midpoint of the genomic interval that this tree represents
        the history of. This is defined as :math:`(l + (r - l) / 2)`, where
        :math:`(l, r)` is the genomic interval returned by
        :attr:`~Tree.interval`.

        :return: The genomic distance covered by this tree.
        :rtype: float
        """
        return self.interval.mid

    def get_sample_size(self):
        # Deprecated alias for self.sample_size
        return self.sample_size

    @property
    def sample_size(self):
        # Deliberately undocumented but kept for backwards compatibility.
        # The proper way to access this is via tree.tree_sequence.num_samples
        return self._ll_tree.get_sample_size()

    def draw_text(
        self,
        orientation=None,
        *,
        node_labels=None,
        max_time=None,
        use_ascii=False,
        order=None,
    ):
        """
        Create a text representation of a tree.

        :param str orientation: one of ``"top"``, ``"left"``, ``"bottom"``, or
            ``"right"``, specifying the margin on which the root is placed. Specifying
            ``"left"`` or ``"right"`` will lead to time being shown on the x axis (i.e.
            a "horizontal" tree. If ``None`` (default) use the standard coalescent
            arrangement of a vertical tree with recent nodes at the bottom of the plot
            and older nodes above.
        :param dict node_labels: If specified, show custom labels for the nodes
            that are present in the map. Any nodes not specified in the map will
            not have a node label.
        :param str max_time: If equal to ``"tree"`` (the default), the maximum time
            is set to be that of the oldest root in the tree. If equal to ``"ts"`` the
            maximum time is set to be the time of the oldest root in the tree
            sequence; this is useful when drawing trees from the same tree sequence as it
            ensures that node heights are consistent.
        :param bool use_ascii: If ``False`` (default) then use unicode
            `box drawing characters \
<https://en.wikipedia.org/wiki/Box-drawing_character>`_
            to render the tree. If ``True``, use plain ascii characters, which look
            cruder but are less susceptible to misalignment or font substitution.
            Alternatively, if you are having alignment problems with Unicode, you can try
            out the solution documented `here \
<https://github.com/tskit-dev/tskit/issues/189#issuecomment-499114811>`_.
        :param str order: The left-to-right ordering of child nodes in the drawn tree.
            This can be either: ``"minlex"``, which minimises the differences
            between adjacent trees (see also the ``"minlex_postorder"`` traversal
            order for the :meth:`.nodes` method); or ``"tree"`` which draws trees
            in the left-to-right order defined by the
            :ref:`quintuply linked tree structure <sec_data_model_tree_structure>`.
            If not specified or None, this defaults to ``"minlex"``.

        :return: A text representation of a tree.
        :rtype: str
        """
        orientation = drawing.check_orientation(orientation)
        if orientation in (drawing.LEFT, drawing.RIGHT):
            text_tree = drawing.HorizontalTextTree(
                self,
                orientation=orientation,
                node_labels=node_labels,
                max_time=max_time,
                use_ascii=use_ascii,
                order=order,
            )
        else:
            text_tree = drawing.VerticalTextTree(
                self,
                orientation=orientation,
                node_labels=node_labels,
                max_time=max_time,
                use_ascii=use_ascii,
                order=order,
            )
        return str(text_tree)

    def draw_svg(
        self,
        path=None,
        *,
        size=None,
        time_scale=None,
        tree_height_scale=None,
        title=None,
        max_time=None,
        min_time=None,
        max_tree_height=None,
        node_labels=None,
        mutation_labels=None,
        node_titles=None,
        mutation_titles=None,
        root_svg_attributes=None,
        style=None,
        order=None,
        force_root_branch=None,
        symbol_size=None,
        x_axis=None,
        x_label=None,
        x_regions=None,
        y_axis=None,
        y_label=None,
        y_ticks=None,
        y_gridlines=None,
        all_edge_mutations=None,
        omit_sites=None,
        canvas_size=None,
        preamble=None,
        **kwargs,
    ):
        """
        Return an SVG representation of a single tree. By default, numeric
        labels are drawn beside nodes and mutations: these can be altered using the
        ``node_labels`` and ``mutation_labels`` parameters. See the
        :ref:`visualization tutorial<tutorials:sec_tskit_viz>` for more details.

        :param str path: The path to the file to write the output. If None, do not
            write to file.
        :param tuple(int, int) size: A tuple of (width, height) specifying a target
            drawing size in abstract user units (usually interpreted as pixels on
            initial display). Components of the drawing will be scaled so that the total
            plot including labels etc. normally fits onto a canvas of this size (see
            ``canvas_size`` below). If ``None``, pick a size appropriate for a tree
            with a reasonably small number (i.e. tens) of samples. Default: ``None``
        :type size:
        :param str time_scale: Control how height values for nodes are computed.
            If this is equal to ``"time"`` (the default), node heights are proportional
            to their time values. If this is equal to ``"log_time"``, node heights are
            proportional to their log(time) values. If it is equal to ``"rank"``, node
            heights are spaced equally according to their ranked times.
        :param str tree_height_scale: Deprecated alias for time_scale. (Deprecated in
                0.3.6)
        :param str title: A title string to be included in the SVG output. If ``None``
            (default) no title is shown, which gives more vertical space for the tree.
        :param str,float max_time: The maximum plotted time value in the current
            scaling system (see ``time_scale``). Can be either a string or a
            numeric value. If equal to ``"tree"`` (the default), the maximum time
            is set to be that of the oldest root in the tree. If equal to ``"ts"`` the
            maximum time is set to be the time of the oldest root in the tree
            sequence; this is useful when drawing trees from the same tree sequence as it
            ensures that node heights are consistent. If a numeric value, this is used as
            the maximum plotted time by which to scale other nodes.
        :param str,float min_time: The minimum plotted time value in the current
            scaling system (see ``time_scale``). Can be either a string or a
            numeric value. If equal to ``"tree"`` (the default), the minimum time
            is set to be that of the youngest node in the tree. If equal to ``"ts"`` the
            minimum time is set to be the time of the youngest node in the tree
            sequence; this is useful when drawing trees from the same tree sequence as it
            ensures that node heights are consistent. If a numeric value, this is used as
            the minimum plotted time.
        :param str,float max_tree_height: Deprecated alias for max_time. (Deprecated in
            0.3.6)
        :param node_labels: If specified, show custom labels for the nodes
            (specified by ID) that are present in this map; any nodes not present will
            not have a label. To use a metadata key, for example, use
            ``node_labels={node.id: node.metadata["key"] for node in ts.nodes()}``.
        :type node_labels: dict(int, str)
        :param mutation_labels: If specified, show custom labels for the
            mutations (specified by ID) that are present in the map; any mutations
            not present will not have a label.
        :type mutation_labels: dict(int, str)
        :param dict(int, str) node_titles: If specified, add a ``<title>`` string to
            symbols for each node (specified by ID) present in this map. SVG visualizers
            such as web browsers will commonly display this string on mousing over the
            node symbol.
        :param dict(int, str) mutation_titles: If specified, add a ``<title>`` string to
            symbols for each mutation (specified by ID) present in this map. SVG
            visualizers such as web browsers will commonly display this string on
            mousing over the mutation symbol in the tree and (if show) on the x axis.
        :param dict root_svg_attributes: Additional attributes, such as an id, that will
            be embedded in the root ``<svg>`` tag of the generated drawing.
        :param str style: A
            `css style string <https://www.w3.org/TR/CSS22/syndata.html>`_ that will be
            included in the ``<style>`` tag of the generated svg.
        :param str order: The left-to-right ordering of child nodes in the drawn tree.
            This can be either: ``"minlex"``, which minimises the differences
            between adjacent trees (see also the ``"minlex_postorder"`` traversal
            order for the :meth:`.nodes` method); or ``"tree"`` which draws trees
            in the left-to-right order defined by the
            :ref:`quintuply linked tree structure <sec_data_model_tree_structure>`.
            If not specified or None, this defaults to ``"minlex"``.
        :param bool force_root_branch: If ``True`` always plot a branch (edge) above the
            root(s) in the tree. If ``None`` (default) then only plot such root branches
            if there is a mutation above a root of the tree.
        :param float symbol_size: Change the default size of the node and mutation
            plotting symbols. If ``None`` (default) use a standard size.
        :param bool x_axis: Should the plot have an X axis line, showing the start and
            end position of this tree along the genome. If ``None`` (default) do not
            plot an X axis.
        :param str x_label: Place a label under the plot. If ``None`` (default) and
            there is an X axis, create and place an appropriate label.
        :param dict x_regions: A dictionary mapping (left, right) tuples to names. This
            draws a box, labelled with the name, on the X axis between the left and
            right positions, and can be used for annotating genomic regions (e.g.
            genes) on the X axis. If ``None`` (default) do not plot any regions.
        :param bool y_axis: Should the plot have an Y axis line, showing time (or
            ranked node time if ``time_scale="rank"``). If ``None`` (default)
            do not plot a Y axis.
        :param str y_label: Place a label to the left of the plot. If ``None`` (default)
            and there is a Y axis,  create and place an appropriate label.
        :param Union[list, dict] y_ticks: A list of Y values at which to plot
            tickmarks, or a dictionary mapping Y values to labels (``[]`` gives no
            tickmarks). If ``None`` (default), plot one tickmark for each unique node
            value. Note that if ``time_scale="rank"``, the Y values refer to the
            zero-based rank of the plotted nodes, rather than the node time itself.
        :param bool y_gridlines: Whether to plot horizontal lines behind the tree
            at each y tickmark.
        :param bool all_edge_mutations: The edge on which a mutation occurs may span
            multiple trees. If ``False`` or ``None`` (default) mutations are only drawn
            on an edge if their site position exists within the genomic interval covered
            by this tree. If ``True``, all mutations on each edge of the tree are drawn,
            even if their genomic position is to the left or right of the tree
            itself. Note that this means that independent drawings of different trees
            from the same tree sequence may share some plotted mutations.
        :param bool omit_sites: If True, omit sites and mutations from the drawing.
            Default: False
        :param tuple(int, int) canvas_size: The (width, height) of the SVG canvas.
            This will change the SVG width and height without rescaling graphical
            elements, allowing extra room e.g. for unusually long labels. If ``None``
            take the canvas size to be the same as the target drawing size (see
            ``size``, above). Default: None
        :param str preamble: SVG commands to be included at the start of the returned
            object, immediately after the opening tag. These can include custom svg
            elements such as legends or annotations or even entire ``<svg>`` elements.
            The preamble is not checked for validity, so it is up to the user to
            ensure that it is valid SVG. Default: None

        :return: An SVG representation of a tree.
        :rtype: SVGString
        """
        svgtree = drawing.SvgTree(
            self,
            size,
            time_scale=time_scale,
            tree_height_scale=tree_height_scale,
            title=title,
            max_time=max_time,
            min_time=min_time,
            max_tree_height=max_tree_height,
            node_labels=node_labels,
            mutation_labels=mutation_labels,
            node_titles=node_titles,
            mutation_titles=mutation_titles,
            root_svg_attributes=root_svg_attributes,
            style=style,
            order=order,  # NB undocumented: Tree.draw_svg can also take an iterable here
            force_root_branch=force_root_branch,
            symbol_size=symbol_size,
            x_axis=x_axis,
            x_label=x_label,
            x_regions=x_regions,
            y_axis=y_axis,
            y_label=y_label,
            y_ticks=y_ticks,
            y_gridlines=y_gridlines,
            all_edge_mutations=all_edge_mutations,
            omit_sites=omit_sites,
            canvas_size=canvas_size,
            preamble=preamble,
            **kwargs,
        )
        return svgtree.draw(path)

    def draw(
        self,
        path=None,
        width=None,
        height=None,
        node_labels=None,
        node_colours=None,
        mutation_labels=None,
        mutation_colours=None,
        format=None,  # noqa A002
        edge_colours=None,
        time_scale=None,
        tree_height_scale=None,
        max_time=None,
        min_time=None,
        max_tree_height=None,
        order=None,
        omit_sites=None,
    ):
        """
        Returns a drawing of this tree.

        When working in a Jupyter notebook, use the ``IPython.display.SVG``
        function to display the SVG output from this function inline in the notebook::

            SVG(tree.draw())

        The unicode format uses unicode `box drawing characters
        <https://en.wikipedia.org/wiki/Box-drawing_character>`_ to render the tree.
        This allows rendered trees to be printed out to the terminal::

            print(tree.draw(format="unicode"))
              6
            ┏━┻━┓
            ┃   5
            ┃ ┏━┻┓
            ┃ ┃  4
            ┃ ┃ ┏┻┓
            3 0 1 2

        The ``node_labels`` argument allows the user to specify custom labels
        for nodes, or no labels at all::

            print(tree.draw(format="unicode", node_labels={}))
              ┃
            ┏━┻━┓
            ┃   ┃
            ┃ ┏━┻┓
            ┃ ┃  ┃
            ┃ ┃ ┏┻┓
            ┃ ┃ ┃ ┃

        Note: in some environments such as Jupyter notebooks with Windows or Mac,
        users have observed that the Unicode box drawings can be misaligned. In
        these cases, we recommend using the SVG or ASCII display formats instead.
        If you have a strong preference for aligned Unicode, you can try out the
        solution documented
        `here <https://github.com/tskit-dev/tskit/issues/189#issuecomment-499114811>`_.

        :param str path: The path to the file to write the output. If None, do not
            write to file.
        :param int width: The width of the image in pixels. If not specified, either
            defaults to the minimum size required to depict the tree (text formats)
            or 200 pixels.
        :param int height: The height of the image in pixels. If not specified, either
            defaults to the minimum size required to depict the tree (text formats)
            or 200 pixels.
        :param dict node_labels: If specified, show custom labels for the nodes
            that are present in the map. Any nodes not specified in the map will
            not have a node label.
        :param dict node_colours: If specified, show custom colours for the nodes
            given in the map. Any nodes not specified in the map will take the default
            colour; a value of ``None`` is treated as transparent and hence the node
            symbol is not plotted. (Only supported in the SVG format.)
        :param dict mutation_labels: If specified, show custom labels for the mutations
            (specified by ID) that are present in the map. Any mutations not in the map
            will not have a label. (Showing mutations is currently only supported in the
            SVG format)
        :param dict mutation_colours: If specified, show custom colours for the mutations
            given in the map (specified by ID). As for ``node_colours``, mutations not
            present in the map take the default colour, and those mapping to ``None``
            are not drawn. (Only supported in the SVG format.)
        :param str format: The format of the returned image. Currently supported
            are 'svg', 'ascii' and 'unicode'. Note that the :meth:`Tree.draw_svg`
            method provides more comprehensive functionality for creating SVGs.
        :param dict edge_colours: If specified, show custom colours for the edge
            joining each node in the map to its parent. As for ``node_colours``,
            unspecified edges take the default colour, and ``None`` values result in the
            edge being omitted. (Only supported in the SVG format.)
        :param str time_scale: Control how height values for nodes are computed.
            If this is equal to ``"time"``, node heights are proportional to their time
            values. If this is equal to ``"log_time"``, node heights are proportional to
            their log(time) values. If it is equal to ``"rank"``, node heights are spaced
            equally according to their ranked times. For SVG output the default is
            'time'-scale whereas for text output the default is 'rank'-scale.
            Time scaling is not currently supported for text output.
        :param str tree_height_scale: Deprecated alias for time_scale. (Deprecated in
                0.3.6)
        :param str,float max_time: The maximum time value in the current
            scaling system (see ``time_scale``). Can be either a string or a
            numeric value. If equal to ``"tree"``, the maximum time is set to be
            that of the oldest root in the tree. If equal to ``"ts"`` the maximum
            time is set to be the time of the oldest root in the tree sequence;
            this is useful when drawing trees from the same tree sequence as it ensures
            that node heights are consistent. If a numeric value, this is used as the
            maximum time by which to scale other nodes. This parameter
            is not currently supported for text output.
        :param str,float min_time: The minimum time value in the current
            scaling system (see ``time_scale``). Can be either a string or a
            numeric value. If equal to ``"tree"``, the minimum time is set to be
            that of the youngest node in the tree. If equal to ``"ts"`` the minimum
            time is set to be the time of the youngest node in the tree sequence;
            this is useful when drawing trees from the same tree sequence as it ensures
            that node heights are consistent. If a numeric value, this is used as the
            minimum time to display. This parameter is not currently supported for text
            output.
        :param str max_tree_height: Deprecated alias for max_time. (Deprecated in
                0.3.6)
        :param str order: The left-to-right ordering of child nodes in the drawn tree.
            This can be either: ``"minlex"``, which minimises the differences
            between adjacent trees (see also the ``"minlex_postorder"`` traversal
            order for the :meth:`.nodes` method); or ``"tree"`` which draws trees
            in the left-to-right order defined by the
            :ref:`quintuply linked tree structure <sec_data_model_tree_structure>`.
            If not specified or None, this defaults to ``"minlex"``.
        :param bool omit_sites: If True, omit sites and mutations from the drawing
            (only relevant to the SVG format). Default: False
        :return: A representation of this tree in the requested format.
        :rtype: str
        """
        output = drawing.draw_tree(
            self,
            format=format,
            width=width,
            height=height,
            node_labels=node_labels,
            node_colours=node_colours,
            mutation_labels=mutation_labels,
            mutation_colours=mutation_colours,
            edge_colours=edge_colours,
            time_scale=time_scale,
            tree_height_scale=tree_height_scale,
            max_time=max_time,
            min_time=min_time,
            max_tree_height=max_tree_height,
            order=order,
            omit_sites=omit_sites,
        )
        if path is not None:
            with open(path, "w") as f:
                f.write(output)
        return output

    def get_num_mutations(self):
        return self.num_mutations

    @property
    def num_mutations(self):
        """
        Returns the total number of mutations across all sites on this tree.

        :return: The total number of mutations over all sites on this tree.
        :rtype: int
        """
        return sum(len(site.mutations) for site in self.sites())

    @property
    def num_sites(self):
        """
        Returns the number of sites on this tree.

        :return: The number of sites on this tree.
        :rtype: int
        """
        return self._ll_tree.get_num_sites()

    def sites(self):
        """
        Returns an iterator over all the :ref:`sites <sec_site_table_definition>`
        in this tree. Sites are returned in order of increasing ID
        (and also position). See the :class:`Site` class for details on
        the available fields for each site.

        :return: An iterator over all sites in this tree.
        """
        # TODO change the low-level API to just return the IDs of the sites.
        for ll_site in self._ll_tree.get_sites():
            _, _, _, id_, _ = ll_site
            yield self.tree_sequence.site(id_)

    def mutations(self):
        """
        Returns an iterator over all the
        :ref:`mutations <sec_mutation_table_definition>` in this tree.
        Mutations are returned in their
        :ref:`order in the mutations table<sec_mutation_requirements>`,
        that is, by nondecreasing site ID, and within a site, by decreasing
        mutation time with parent mutations before their children.
        See the :class:`Mutation` class for details on the available fields for
        each mutation.

        The returned iterator is equivalent to iterating over all sites
        and all mutations in each site, i.e.::

            for site in tree.sites():
                for mutation in site.mutations:
                    yield mutation

        :return: An iterator over all :class:`Mutation` objects in this tree.
        :rtype: iter(:class:`Mutation`)
        """
        for site in self.sites():
            yield from site.mutations

    def get_leaves(self, u):
        # Deprecated alias for samples. See the discussion in the get_num_leaves
        # method for why this method is here and why it is semantically incorrect.
        # The 'leaves' iterator below correctly returns the leaves below a given
        # node.
        return self.samples(u)

    def leaves(self, u=None):
        """
        Returns an iterator over all the leaves in this tree that descend from
        the specified node. If :math:`u`  is not specified, return all leaves on
        the tree (i.e. all leaves reachable from the tree root(s), see note below).

        .. note::
            :math:`u` can be any node in the entire tree sequence, including ones
            which are not connected via branches to a root node of the tree. If
            called on such a node, the iterator will return "dead" leaves
            (see :ref:`sec_data_model_tree_dead_leaves_and_branches`) which cannot
            be reached from a root of this tree. However, dead leaves will never be
            returned if :math:`u` is left unspecified.

        :param int u: The node of interest.
        :return: An iterator over all leaves in the subtree rooted at u.
        :rtype: collections.abc.Iterable
        """
        roots = [u]
        if u is None:
            roots = self.roots
        for root in roots:
            for v in self.nodes(root):
                if self.is_leaf(v):
                    yield v

    def _sample_generator(self, u):
        if self._ll_tree.get_options() & _tskit.SAMPLE_LISTS:
            samples = self.tree_sequence.samples()
            index = self.left_sample(u)
            if index != NULL:
                stop = self.right_sample(u)
                while True:
                    yield samples[index]
                    if index == stop:
                        break
                    index = self.next_sample(index)
        else:
            # Fall back on iterating over all nodes in the tree, yielding
            # samples as we see them.
            for v in self.nodes(u):
                if self.is_sample(v):
                    yield v

    def samples(self, u=None):
        """
        Returns an iterator over the numerical IDs of all the sample nodes in
        this tree that are underneath the node with ID ``u``. If ``u`` is a sample,
        it is included in the returned iterator. If ``u`` is not a sample, it is
        possible for the returned iterator to be empty, for example if ``u`` is an
        :meth:`isolated<Tree.is_isolated>` node that is not part of the the current
        topology. If u is not specified, return all sample node IDs in the tree
        (equivalent to all the sample node IDs in the tree sequence).

        If the :meth:`TreeSequence.trees` method is called with
        ``sample_lists=True``, this method uses an efficient algorithm to find
        the sample nodes. If not, a simple traversal based method is used.

        .. note::

            The iterator is *not* guaranteed to return the sample node IDs in
            numerical or any other particular order.

        :param int u: The node of interest.
        :return: An iterator over all sample node IDs in the subtree rooted at u.
        :rtype: collections.abc.Iterable
        """
        roots = [u]
        if u is None:
            roots = self.roots
        for root in roots:
            yield from self._sample_generator(root)

    def num_children(self, u):
        """
        Returns the number of children of the specified
        node (i.e., ``len(tree.children(u))``)

        :param int u: The node of interest.
        :return: The number of immediate children of the node u in this tree.
        :rtype: int
        """
        return self._ll_tree.get_num_children(u)

    def get_num_leaves(self, u):
        # Deprecated alias for num_samples. The method name is inaccurate
        # as this will count the number of tracked _samples_. This is only provided to
        # avoid breaking existing code and should not be used in new code. We could
        # change this method to be semantically correct and just count the
        # number of leaves we hit in the leaves() iterator. However, this would
        # have the undesirable effect of making code that depends on the constant
        # time performance of get_num_leaves many times slower. So, the best option
        # is to leave this method as is, and to slowly deprecate it out. Once this
        # has been removed, we might add in a ``num_leaves`` method that returns the
        # length of the leaves() iterator as one would expect.
        return self.num_samples(u)

    def get_num_samples(self, u=None):
        # Deprecated alias for num_samples.
        return self.num_samples(u)

    def num_samples(self, u=None):
        """
        Returns the number of sample nodes in this tree underneath the specified
        node (including the node itself). If u is not specified return
        the total number of samples in the tree.

        This is a constant time operation.

        :param int u: The node of interest.
        :return: The number of samples in the subtree rooted at u.
        :rtype: int
        """
        u = self.virtual_root if u is None else u
        return self._ll_tree.get_num_samples(u)

    def get_num_tracked_leaves(self, u):
        # Deprecated alias for num_tracked_samples. The method name is inaccurate
        # as this will count the number of tracked _samples_. This is only provided to
        # avoid breaking existing code and should not be used in new code.
        return self.num_tracked_samples(u)

    def get_num_tracked_samples(self, u=None):
        # Deprecated alias for num_tracked_samples
        return self.num_tracked_samples(u)

    def num_tracked_samples(self, u=None):
        """
        Returns the number of samples in the set specified in the
        ``tracked_samples`` parameter of the :meth:`TreeSequence.trees` method
        underneath the specified node. If the input node is not specified,
        return the total number of tracked samples in the tree.

        This is a constant time operation.

        :param int u: The node of interest.
        :return: The number of samples within the set of tracked samples in
            the subtree rooted at u.
        :rtype: int
        """
        u = self.virtual_root if u is None else u
        return self._ll_tree.get_num_tracked_samples(u)

    def preorder(self, u=NULL):
        """
        Returns a numpy array of node ids in `preorder
        <https://en.wikipedia.org/wiki/Tree_traversal#Pre-order_(NLR)>`_. If the node u
        is specified the traversal is rooted at this node (and it will be the first
        element in the returned array). Otherwise, all nodes reachable from the tree
        roots will be returned. See :ref:`tutorials:sec_analysing_trees_traversals` for
        examples.

        :param int u: If specified, return all nodes in the subtree rooted at u
            (including u) in traversal order.
        :return: Array of node ids
        :rtype: numpy.ndarray (dtype=np.int32)
        """
        return self._ll_tree.get_preorder(u)

    def postorder(self, u=NULL):
        """
        Returns a numpy array of node ids in `postorder
        <https://en.wikipedia.org/wiki/Tree_traversal##Post-order_(LRN)>`_. If the node u
        is specified the traversal is rooted at this node (and it will be the last
        element in the returned array). Otherwise, all nodes reachable from the tree
        roots will be returned. See :ref:`tutorials:sec_analysing_trees_traversals` for
        examples.

        :param int u: If specified, return all nodes in the subtree rooted at u
            (including u) in traversal order.
        :return: Array of node ids
        :rtype: numpy.ndarray (dtype=np.int32)
        """
        return self._ll_tree.get_postorder(u)

    def timeasc(self, u=NULL):
        """
        Returns a numpy array of node ids. Starting at `u`, returns the reachable
        descendant nodes in order of increasing time (most recent first), falling back
        to increasing ID if times are equal. Also see
        :ref:`tutorials:sec_analysing_trees_traversals` for examples of how to use
        traversals.

        :param int u: If specified, return all nodes in the subtree rooted at u
            (including u) in traversal order.
        :return: Array of node ids
        :rtype: numpy.ndarray (dtype=np.int32)
        """
        nodes = self.preorder(u)
        is_virtual_root = u == self.virtual_root
        time = self.tree_sequence.nodes_time
        if is_virtual_root:
            # We could avoid creating this array if we wanted to, but
            # it's not that often people will be using this with the
            # virtual_root as an argument, so doesn't seem worth
            # the complexity
            time = np.append(time, [np.inf])
        order = np.lexsort([nodes, time[nodes]])
        return nodes[order]

    def timedesc(self, u=NULL):
        """
        Returns a numpy array of node ids. Starting at `u`, returns the reachable
        descendant nodes in order of decreasing time (least recent first), falling back
        to decreasing ID if times are equal. Also see
        :ref:`tutorials:sec_analysing_trees_traversals` for examples of how to use
        traversals.

        :param int u: If specified, return all nodes in the subtree rooted at u
            (including u) in traversal order.
        :return: Array of node ids
        :rtype: numpy.ndarray (dtype=np.int32)
        """
        return self.timeasc(u)[::-1]

    def _preorder_traversal(self, root):
        # Return Python integers for compatibility
        return map(int, self.preorder(root))

    def _postorder_traversal(self, root):
        # Return Python integers for compatibility
        return map(int, self.postorder(root))

    def _inorder_traversal(self, root):
        # TODO add a nonrecursive version of the inorder traversal.

        def traverse(u):
            children = self.get_children(u)
            mid = len(children) // 2
            for c in children[:mid]:
                yield from traverse(c)
            yield u
            for c in children[mid:]:
                yield from traverse(c)

        roots = self.roots if root == NULL else [root]
        for root in roots:
            yield from traverse(root)

    def _levelorder_traversal(self, root):
        roots = self.roots if root == NULL else [root]
        queue = collections.deque(roots)
        # For perf we store these to avoid lookups in the tight loop
        pop = queue.popleft
        extend = queue.extend
        children = self.children
        # Note: the usual style is to be explicit about what we're testing
        # and use while len(queue) > 0, but this form is slightly faster.
        while queue:
            v = pop()
            extend(children(v))
            yield v

    def _timeasc_traversal(self, root):
        """
        Sorts by increasing time but falls back to increasing ID for equal times.
        """
        return map(int, self.timeasc(root))

    def _timedesc_traversal(self, root):
        """
        The reverse of timeasc.
        """
        return map(int, self.timedesc(root))

    def _minlex_postorder_traversal(self, root):
        """
        Postorder traversal that visits leaves in minimum lexicographic order.

        Minlex stands for minimum lexicographic. We wish to visit a tree in such
        a way that the leaves visited, when their IDs are listed out, have
        minimum lexicographic order. This is a useful ordering for drawing
        multiple Trees of a TreeSequence, as it leads to more consistency
        between adjacent Trees.
        """

        # We compute a dictionary mapping from internal node ID to min leaf ID
        # under the node, using a first postorder traversal
        min_leaf = {}
        for u in self.nodes(root, order="postorder"):
            if self.is_leaf(u):
                min_leaf[u] = u
            else:
                min_leaf[u] = min(min_leaf[v] for v in self.children(u))

        # If we deliberately specify the virtual root, it should also be returned
        is_virtual_root = root == self.virtual_root
        if root == -1:
            root = self.virtual_root

        stack = [(root, False)]
        while len(stack) > 0:
            u, visited = stack.pop()
            if visited:
                if u != self.virtual_root or is_virtual_root:
                    yield u
            else:
                stack.append((u, True))  # Reappend, marking visited
                stack.extend(
                    sorted(
                        ((c, False) for c in self.children(u)),
                        key=lambda v: min_leaf[v[0]],
                        reverse=True,
                    )
                )

    def nodes(self, root=None, order="preorder"):
        """
        Returns an iterator over the node IDs reachable from the specified node in this
        tree in the specified traversal order.

        .. note::
            Unlike the :meth:`TreeSequence.nodes` method, this iterator produces
            integer node IDs, not :class:`Node` objects.

        If the ``root`` parameter is not provided or ``None``, iterate over all
        nodes reachable from the roots (see :attr:`Tree.roots` for details
        on which nodes are considered roots). If the ``root`` parameter
        is provided, only the nodes in the subtree rooted at this node
        (including the specified node) will be iterated over. If the
        :attr:`.virtual_root` is specified as the traversal root, it will
        be included in the traversed nodes in the appropriate position
        for the given ordering. (See the
        :ref:`tree roots <sec_data_model_tree_virtual_root>` section for more
        information on the virtual root.)

        The ``order`` parameter defines the order in which tree nodes are visited
        in the iteration (also see the :ref:`sec_analysing_trees_traversals` section
        in the `tutorials <https://tskit.dev/tutorials>`__). The available orders are:

        - 'preorder': starting at root, yield the current node, then recurse
          and do a preorder on each child of the current node. See also `Wikipedia
          <https://en.wikipedia.org/wiki/Tree_traversal#Pre-order_(NLR)>`__.
        - 'inorder': starting at root, assuming binary trees, recurse and do
          an inorder on the first child, then yield the current node, then
          recurse and do an inorder on the second child. In the case of ``n``
          child nodes (not necessarily 2), the first ``n // 2`` children are
          visited in the first stage, and the remaining ``n - n // 2`` children
          are visited in the second stage. See also `Wikipedia
          <https://en.wikipedia.org/wiki/Tree_traversal#In-order_(LNR)>`__.
        - 'postorder': starting at root, recurse and do a postorder on each
          child of the current node, then yield the current node. See also
          `Wikipedia
          <https://en.wikipedia.org/wiki/Tree_traversal#Post-order_(LRN)>`__.
        - 'levelorder' ('breadthfirst'): visit the nodes under root (including
          the root) in increasing order of their depth from root. See also
          `Wikipedia
          <https://en.wikipedia.org/wiki/Tree_traversal\
#Breadth-first_search_/_level_order>`__.
        - 'timeasc': visits the nodes in order of increasing time, falling back to
          increasing ID if times are equal.
        - 'timedesc': visits the nodes in order of decreasing time, falling back to
          decreasing ID if times are equal.
        - 'minlex_postorder': a usual postorder has ambiguity in the order in
          which children of a node are visited. We constrain this by outputting
          a postorder such that the leaves visited, when their IDs are
          listed out, have minimum `lexicographic order
          <https://en.wikipedia.org/wiki/Lexicographical_order>`__ out of all valid
          traversals. This traversal is useful for drawing multiple trees of
          a ``TreeSequence``, as it leads to more consistency between adjacent
          trees. Note that internal non-leaf nodes are not counted in
          assessing the lexicographic order.

        :param int root: The root of the subtree we are traversing.
        :param str order: The traversal ordering. Currently 'preorder',
            'inorder', 'postorder', 'levelorder' ('breadthfirst'), 'timeasc' and
            'timedesc' and 'minlex_postorder' are supported.
        :return: An iterator over the node IDs in the tree in some traversal order.
        :rtype: collections.abc.Iterable, int
        """
        methods = {
            "preorder": self._preorder_traversal,
            "inorder": self._inorder_traversal,
            "postorder": self._postorder_traversal,
            "levelorder": self._levelorder_traversal,
            "breadthfirst": self._levelorder_traversal,
            "timeasc": self._timeasc_traversal,
            "timedesc": self._timedesc_traversal,
            "minlex_postorder": self._minlex_postorder_traversal,
        }
        try:
            iterator = methods[order]
        except KeyError:
            raise ValueError(f"Traversal ordering '{order}' not supported")

        root = -1 if root is None else root
        return iterator(root)

    def _as_newick_fast(self, *, root, precision, legacy_ms_labels):
        """
        Call into the fast but limited C implementation of the newick conversion.
        """
        root_time = max(1, self.time(root))
        max_label_size = math.ceil(math.log10(self.tree_sequence.num_nodes))
        single_node_size = (
            5 + max_label_size + math.ceil(math.log10(root_time)) + precision
        )
        buffer_size = 1 + single_node_size * self.tree_sequence.num_nodes
        return self._ll_tree.get_newick(
            precision=precision,
            root=root,
            buffer_size=buffer_size,
            legacy_ms_labels=legacy_ms_labels,
        )

    def as_newick(
        self,
        *,
        root=None,
        precision=None,
        node_labels=None,
        include_branch_lengths=None,
    ):
        """
        Returns a `newick encoding
        <https://en.wikipedia.org/wiki/Newick_format>`_ of this tree.
        For example, a binary tree with 3 leaves generated by
        :meth:`Tree.generate_balanced(3)<Tree.generate_balanced>`
        encodes as::

            (n0:2,(n1:1,n2:1):1);

        By default :ref:`sample nodes<sec_data_model_definitions>` are
        labelled using the form ``f"n{node_id}"``, i.e. the sample node's
        ID prefixed with the string ``"n"``. Node labels can be specified
        explicitly using the ``node_labels`` argument, which is a mapping from
        integer node IDs to the corresponding string label. If a node is not
        present in the mapping, no label is associated with the node in
        output.

        .. warning:: Node labels are **not** Newick escaped, so care must be taken
            to provide labels that will not break the encoding.

        .. note:: Specifying a ``node_labels`` dictionary or setting
            ``include_branch_lengths=False`` results in a less efficient
            method being used to generate the newick output. The performance
            difference can be substantial for large trees.

        By default, branch lengths are printed out with sufficient precision
        for them to be recovered exactly in double precision (although note
        that this does not necessarily mean that we can precisely recover the
        corresponding node times, since branch lengths are obtained by
        subtraction). If all times on the tree sequence are discrete, then
        branch lengths are printed as integers. Otherwise, branch lengths are
        printed with 17 digits of precision (i.e., ``"%.17f"`` in
        printf-notation).

        The precision for branch lengths can be specified using the ``precision``
        argument. Branch lengths can be omitted entirely by setting
        ``include_branch_lengths=False``.

        If the ``root`` argument is specified, we return the newick encoding of
        the specified subtree, otherwise the full tree is returned. If the tree
        has :ref:`multiple roots <sec_data_model_tree_roots>` and a root node
        is not explicitly specified, we raise a ``ValueError``. This is because
        most libraries and downstream software consider a newick string that
        contains multiple disconnected subtrees an error, and it is therefore
        best to consider how such topologies should be interchanged on a
        case-by-base basis. A list of the newick strings for each root can be
        obtained by ``[tree.as_newick(root=root) for root in tree.roots]``.

        :param int precision: The numerical precision with which branch lengths are
            printed. If not specified or None default to 0 if the tree sequence
            contains only integer node times, or 17 otherwise.
        :param int root: If specified, return the tree rooted at this node.
        :param dict node_labels: If specified, show custom labels for the nodes
            that are present in the map. Any nodes not specified in the map will
            not have a node label.
        :param include_branch_lengths: If True (default), output branch lengths in the
            Newick string. If False, only output the topology, without branch lengths.
        :return: A newick representation of this tree.
        :rtype: str
        """
        if root is None:
            if not self.has_single_root:
                raise ValueError(
                    "Cannot get newick unless a tree has a single root. Try "
                    "[t.as_newick(root) for root in t.roots] to get a list of "
                    "newick trees, one for each root."
                )
            root = self.root

        if precision is None:
            # 17 decimal digits provides the full precision of an IEEE double,
            # as defined by DBL_DECIMAL_DIG macro. If we have discrete time
            # then write out integer branch lengths.
            precision = 0 if self.tree_sequence.discrete_time else 17
        include_branch_lengths = (
            True if include_branch_lengths is None else include_branch_lengths
        )
        # Can we run this through the fast path?
        if include_branch_lengths and node_labels in [LEGACY_MS_LABELS, None]:
            # Note the LEGACY_MS_LABELS code path is not part of the documented
            # interface and should not be depended on by client code.
            return self._as_newick_fast(
                root=root,
                precision=precision,
                legacy_ms_labels=node_labels == LEGACY_MS_LABELS,
            )

        # No, we have to use the slower Python code.
        if node_labels is None:
            node_labels = {u: f"n{u}" for u in self.tree_sequence.samples()}
        elif node_labels == LEGACY_MS_LABELS:
            # NOTE in the ms format it's the *leaf* nodes we label not
            # necessarily the samples. We keep this behaviour to avoid
            # breaking legacy code that may depend on it.
            node_labels = {u: f"{u + 1}" for u in self.leaves()}
        return text_formats.build_newick(
            self,
            root=root,
            precision=precision,
            node_labels=node_labels,
            include_branch_lengths=include_branch_lengths,
        )

    def newick(
        self,
        precision=14,
        *,
        root=None,
        node_labels=None,
        include_branch_lengths=True,
    ):
        """
        .. warning:: This method is deprecated and may be removed in future
            versions of tskit. Please use the :meth:`.as_newick` method
            in new code.

        This method is a deprecated version of the :meth:`.as_newick` method.
        Functionality is equivalent, except for the default node labels.

        By default, *leaf* nodes are labelled with their numerical ID + 1,
        and internal nodes are not labelled. This default strategy was originally
        used to mimic the output of the ``ms`` simulator. However, the choice
        of labelling leaf nodes rather than samples is problematic, and this
        behaviour is only retained to avoid breaking existing code which may
        rely on it.

        Other parameters behave as documented in the :meth:`.as_newick` method.

        :param int precision: The numerical precision with which branch lengths are
            printed. Defaults to 14.
        :param int root: If specified, return the tree rooted at this node.
        :param dict node_labels: If specified, show custom labels for the nodes
            that are present in the map. Any nodes not specified in the map will
            not have a node label.
        :param include_branch_lengths: If True (default), output branch lengths in the
            Newick string. If False, only output the topology, without branch lengths.
        :return: A newick representation of this tree.
        :rtype: str
        """
        node_labels = LEGACY_MS_LABELS if node_labels is None else node_labels
        return self.as_newick(
            root=root,
            precision=precision,
            node_labels=node_labels,
            include_branch_lengths=include_branch_lengths,
        )

    def as_dict_of_dicts(self):
        """
        Convert tree to dict of dicts for conversion to a
        `networkx graph <https://networkx.github.io/documentation/stable/
        reference/classes/digraph.html>`_.

        For example::

            import networkx as nx
            nx.DiGraph(tree.as_dict_of_dicts())
            # undirected graphs work as well
            nx.Graph(tree.as_dict_of_dicts())

        :return: Dictionary of dictionaries of dictionaries where the first key
            is the source, the second key is the target of an edge, and the
            third key is an edge annotation. At this point the only annotation
            is "branch_length", the length of the branch (in units of time).
        """
        dod = {}
        for parent in self.nodes():
            dod[parent] = {}
            for child in self.children(parent):
                dod[parent][child] = {"branch_length": self.branch_length(child)}
        return dod

    @property
    def parent_dict(self):
        return self.get_parent_dict()

    def get_parent_dict(self):
        pi = {
            u: self.parent(u)
            for u in range(self.tree_sequence.num_nodes)
            if self.parent(u) != NULL
        }
        return pi

    def __str__(self):
        """
        Return a plain text summary of a tree in a tree sequence
        """
        tree_rows = [
            ["Index", f"{util.format_number(self.index, sep=',')}"],
            [
                "Interval",
                f"{util.format_number(self.interval.left, sep=',')}-"
                f"{util.format_number(self.interval.right, sep=',')}"
                f"({util.format_number(self.span, sep=',')})",
            ],
            ["Roots", f"{util.format_number(self.num_roots, sep=',')}"],
            ["Nodes", f"{util.format_number(len(self.preorder()), sep=',')}"],
            ["Sites", f"{util.format_number(self.num_sites, sep=',')}"],
            ["Mutations", f"{util.format_number(self.num_mutations, sep=',')}"],
            [
                "Total Branch Length",
                f"{util.format_number(self.total_branch_length, sep=',')}",
            ],
        ]
        return util.unicode_table(tree_rows, title="Tree")

    def _repr_html_(self):
        """
        Return an html summary of a tree in a tree sequence. Called by jupyter
        notebooks to render trees
        """
        return util.tree_html(self)

    def map_mutations(self, genotypes, alleles, ancestral_state=None):
        """
        Given observations for the samples in this tree described by the specified
        set of genotypes and alleles, return a parsimonious set of state transitions
        explaining these observations. The genotypes array is interpreted as indexes
        into the alleles list in the same manner as described in the
        :meth:`TreeSequence.variants` method. Thus, if sample ``j`` carries the
        allele at index ``k``, then we have ``genotypes[j] = k``.
        Missing observations can be specified for a sample using the value
        ``tskit.MISSING_DATA`` (-1), in which case the state at this sample does not
        influence the ancestral state or the position of mutations returned. At least
        one non-missing observation must be provided. A maximum of 64 alleles are
        supported.

        The current implementation uses the Hartigan parsimony algorithm to determine
        the minimum number of state transitions required to explain the data. In this
        model, transitions between any of the non-missing states is equally likely.

        The returned values correspond directly to the data model for describing
        variation at sites using mutations. See the :ref:`sec_site_table_definition`
        and :ref:`sec_mutation_table_definition` definitions for details and background.

        The state reconstruction is returned as two-tuple, ``(ancestral_state,
        mutations)``, where ``ancestral_state`` is the allele assigned to the
        tree root(s) and ``mutations`` is a list of :class:`Mutation` objects,
        ordered as :ref:`required in a mutation table<sec_mutation_requirements>`.
        For each mutation, ``derived_state`` is the new state after this mutation and
        ``node`` is the tree node immediately beneath the mutation (if there are unary
        nodes between two branch points, hence multiple nodes above which the
        mutation could be parsimoniously placed, the oldest node is used). The
        ``parent`` property contains the index in the returned list of the previous
        mutation on the path to root, or ``tskit.NULL``
        if there are no previous mutations (see the :ref:`sec_mutation_table_definition`
        for more information on the concept of mutation parents). All other attributes
        of the :class:`Mutation` object are undefined and should not be used.

        .. note::
            Sample states observed as missing in the input ``genotypes`` need
            not correspond to samples whose nodes are actually "missing" (i.e.,
            :ref:`isolated<sec_data_model_missing_data>`) in the tree. In this
            case, mapping the mutations returned by this method onto the tree
            will result in these missing observations being imputed to the
            most parsimonious state.

        Because the ``parent`` in the returned list of mutations refers to the index
        in that list, if you are adding mutations to an existing tree sequence, you
        will need to maintain a map of list IDs to the newly added mutations, for
        instance::

            last_tree = ts.last()
            anc_state, parsimonious_muts = last_tree.map_mutations([0, 1, 0], ("A", "T"))
            # Edit the tree sequence, see the "Tables and Editing" tutorial
            tables = ts.dump_tables()
            # add a new site at the end of ts, assumes there isn't one there already
            site_id = tables.sites.add_row(ts.sequence_length - 1, anc_state)

            mut_id_map = {tskit.NULL: tskit.NULL}  # don't change if parent id is -1
            for list_id, mutation in enumerate(parsimonious_muts):
                mut_id_map[list_id] = tables.mutations.append(
                    mutation.replace(site=site_id, parent=mut_id_map[mutation.parent]))
            tables.sort()  # Redundant here, but needed if the site is not the last one
            new_ts = tables.tree_sequence()

        See the :ref:`tutorials:sec_analysing_trees_parsimony` section in the tutorial
        for further examples of how to use this method.

        :param array_like genotypes: The input observations for the samples in this tree.
        :param tuple(str) alleles: The alleles for the specified ``genotypes``. Each
            positive value in the ``genotypes`` array is treated as an index into this
            list of alleles.
        :param ancestral_state: A fixed ancestral state, specified either as a
            non-negative integer less than the number of alleles, or a string which
            must be one of the ``alleles`` provided above. If ``None`` (default) then
            an ancestral state is chosen arbitrarily from among those that provide
            the most parsimonious placement of mutations. Note that if the ancestral
            state is specified, the placement of mutations may not be as parsimonious
            as that which could be achieved by leaving the ancestral state unspecified;
            additionally it may lead to mutations being placed above the root node(s) of
            the tree (for example if all the samples have a genotype of 1 but the
            ancestral state is fixed to be 0).
        :type ancestral_state: Union[int, str]
        :return: The inferred ancestral state and list of mutations on this tree
            that encode the specified observations.
        :rtype: (str, list(tskit.Mutation))
        """
        genotypes = util.safe_np_int_cast(genotypes, np.int8)
        max_alleles = np.max(genotypes)
        if ancestral_state is not None:
            if isinstance(ancestral_state, str):
                # Will raise a ValueError if not in the list
                ancestral_state = alleles.index(ancestral_state)
            if ancestral_state < 0 or ancestral_state >= len(alleles):
                raise ValueError("ancestral_state not between 0 and (num_alleles-1)")
            max_alleles = max(ancestral_state, max_alleles)
        if max_alleles >= 64:
            raise ValueError("A maximum of 64 states is supported")
        ancestral_state, transitions = self._ll_tree.map_mutations(
            genotypes, ancestral_state
        )
        # Translate back into string alleles
        ancestral_state = alleles[ancestral_state]
        mutations = [
            Mutation(
                node=node,
                derived_state=alleles[derived_state],
                parent=parent,
                metadata=self.tree_sequence.table_metadata_schemas.mutation.empty_value,
            )
            for node, parent, derived_state in transitions
        ]
        return ancestral_state, mutations

    def kc_distance(self, other, lambda_=0.0):
        """
        Returns the Kendall-Colijn distance between the specified pair of trees.
        The ``lambda_`` parameter  determines the relative weight of topology
        vs branch lengths in calculating the distance. If ``lambda_`` is 0
        (the default) we only consider topology, and if it is 1 we only
        consider branch lengths. See `Kendall & Colijn (2016)
        <https://academic.oup.com/mbe/article/33/10/2735/2925548>`_ for details.

        The trees we are comparing to must have identical lists of sample
        nodes (i.e., the same IDs in the same order). The metric operates on
        samples, not leaves, so internal samples are treated identically to
        sample tips. Subtrees with no samples do not contribute to the metric.

        :param Tree other: The other tree to compare to.
        :param float lambda_: The KC metric lambda parameter determining the
            relative weight of topology and branch length.
        :return: The computed KC distance between this tree and other.
        :rtype: float
        """
        return self._ll_tree.get_kc_distance(other._ll_tree, lambda_)

    def _get_sample_sets(self):
        ret = {}
        for u in self.nodes(order="postorder"):
            u_sample_set = set()
            if self.is_sample(u):
                u_sample_set.add(u)
            for v in self.children(u):
                u_sample_set |= ret[v]
            ret[u] = frozenset(u_sample_set)
        return ret

    def rf_distance(self, other):
        """
        Returns the (unweighted) Robinson-Foulds distance between the specified pair
        of trees, where corresponding samples between the two trees are identified by
        node ID. The Robinson-Foulds distance (also known as the symmetric difference)
        is defined as the number of bipartitions that are present in one tree but
        not the other (see
        `Robinson & Foulds (1981) <https://doi.org/10.1016/0025-5564(81)90043-2>`_).
        This method returns the unnormalised RF distance: if the
        trees are strictly bifurcating, i.e. binary, the value can be
        normalised by dividing by the maximum, which is $2n-4$ for two rooted
        trees of $n$ samples (however, if the trees contain polytomies, the maximum
        RF distance is less easily defined).

        .. note::
            The RF distance can be sensitive to small changes in topology: in some
            cases, changing the position of a single leaf can result in the maximum
            RF distance. Therefore even if adjacent trees in a tree sequence differ
            by a single subtree-prune-and-regraft operation, the RF distance
            between them can be large.

        :param Tree other: The other tree to compare to. Trees are treated as rooted.
        :return: The unweighted Robinson-Foulds distance between this tree and ``other``.
        :rtype: int
        :raises ValueError: If either tree has multiple roots, or the trees have
            different sample nodes.
        """
        if self.num_roots != 1 or other.num_roots != 1:
            raise ValueError("Trees must have a single root")

        s1 = set(self._get_sample_sets().values())
        s2 = set(other._get_sample_sets().values())

        return len(s1.symmetric_difference(s2))

    def path_length(self, u, v):
        """
        Returns the number of edges on the path in this tree between the two nodes.
        If the two nodes have a most recent common ancestor, then this is defined as
        ``tree.depth(u) + tree.depth(v) - 2 * tree.depth(tree.mrca(u, v))``. If the nodes
        do not have an MRCA (i.e., they are in disconnected subtrees) the path length
        is infinity.

        .. note:: This counts the number of "hops" between two nodes. To find the branch
            length distance between them, in units of time (i.e. the sum of edge lengths
            that separate two nodes) use the :meth:`.distance_between` method instead.

        .. seealso:: See also the :meth:`.depth` method

        :param int u: The first node for path length computation.
        :param int v: The second node for path length computation.
        :return: The number of edges between the two nodes.
        :rtype: int
        """
        mrca = self.mrca(u, v)
        if mrca == -1:
            return math.inf
        return self.depth(u) + self.depth(v) - 2 * self.depth(mrca)

    def distance_between(self, u, v):
        """
        Returns the total distance between two nodes in the tree, expressed as
        the sum of "branch lengths" from both nodes to their most recent common ancestor.

        :param int u: The first node for path length computation.
        :param int v: The second node for path length computation.
        :return: The distance between the two nodes, the sum of "branch lengths" .
        :rtype: float
        """
        tmrca = self.tmrca(u, v)
        return tmrca - self.time(u) + tmrca - self.time(v)

    def b1_index(self):
        """
        Returns the
        `B1 balance index <https://treebalance.wordpress.com/b₁-index/>`_
        for this tree. This is defined as the inverse of the sum of all
        longest paths to leaves for each node besides roots.

        .. seealso:: See `Shao and Sokal (1990)
            <https://www.jstor.org/stable/2992186>`_ for details.

        :return: The B1 balance index.
        :rtype: float
        """
        return self._ll_tree.get_b1_index()

    def b2_index(self, base=10):
        """
        Returns the
        `B2 balance index <https://treebalance.wordpress.com/b₂-index/>`_
        this tree.
        This is defined as the Shannon entropy of the probability
        distribution to reach leaves assuming a random walk
        from a root. The default base is 10, following Shao and Sokal (1990).

        .. seealso:: See `Shao and Sokal (1990)
            <https://www.jstor.org/stable/2992186>`_ for details.

        :param int base: The base used for the logarithm in the
            Shannon entropy computation.
        :return: The B2 balance index.
        :rtype: float
        """
        # Let Python decide if the base is acceptable
        math.log(10, base)
        return self._ll_tree.get_b2_index(base)

    def colless_index(self):
        """
        Returns the
        `Colless imbalance index <https://treebalance.wordpress.com/colless-index/>`_
        for this tree. This is defined as the sum of all differences between
        number of leaves subtended by the left and right child of each node.
        The Colless index is undefined for non-binary trees and trees with
        multiple roots. This method will raise a LibraryError if the tree is
        not singly-rooted and binary.

        .. seealso:: See `Shao and Sokal (1990)
            <https://www.jstor.org/stable/2992186>`_ for details.

        :return: The Colless imbalance index.
        :rtype: int
        """
        return self._ll_tree.get_colless_index()

    def sackin_index(self):
        """
        Returns the
        `Sackin imbalance index <https://treebalance.wordpress.com/sackin-index/>`_
        for this tree. This is defined as the sum of the depths of all leaves
        in the tree. Equivalent to ``sum(tree.depth(u) for u in
        tree.leaves())``

        .. seealso:: See `Shao and Sokal (1990)
            <https://www.jstor.org/stable/2992186>`_ for details.

        :return: The Sackin imbalance index.
        :rtype: int
        """
        return self._ll_tree.get_sackin_index()

    def num_lineages(self, t):
        """
        Returns the number of lineages present in this tree at time ``t``. This
        is defined as the number of branches in this tree (reachable from the
        samples) that intersect with ``t``. Thus, ``tree.num_lineages(t)``
        is equal to 0 for any ``t`` greater than or equal to the time of
        the root in a singly-rooted tree.

        .. note:: Note that this definition means that if a (non root) node
            with three children has time ``t``, then it will count as one lineage,
            not three.

        :param int t: The time to count lineages at.
        :return: The number of lineages in the tree at time t.
        :rtype: int
        """
        return self._ll_tree.get_num_lineages(t)

    def split_polytomies(
        self,
        *,
        epsilon=None,
        method=None,
        record_provenance=True,
        random_seed=None,
        **kwargs,
    ):
        """
        Return a new :class:`.Tree` where extra nodes and edges have been inserted
        so that any any node ``u`` with greater than 2 children --- a multifurcation
        or "polytomy" --- is resolved into successive bifurcations. New nodes are
        inserted at times fractionally less than than the time of node ``u``.
        Times are allocated to different levels of the tree, such that any newly
        inserted sibling nodes will have the same time.

        By default, the times of the newly generated children of a particular
        node are the minimum representable distance in floating point arithmetic
        from their parents (using the `nextafter
        <https://numpy.org/doc/stable/reference/generated/numpy.nextafter.html>`_
        function). Thus, the generated branches have the shortest possible nonzero
        length. A fixed branch length between inserted nodes and their parents
        can also be specified by using the ``epsilon`` parameter.

        .. note::
            A tree sequence :ref:`requires<sec_valid_tree_sequence_requirements>` that
            parents be older than children and that mutations are younger than the
            parent of the edge on which they lie. If a fixed ``epsilon`` is specifed
            and is not small enough compared to the distance between a polytomy and
            its oldest child (or oldest child mutation) these requirements may not
            be met. In this case an error will be raised.

        If the ``method`` is ``"random"`` (currently the only option, and the default
        when no method is specified), then for a node with :math:`n` children, the
        :math:`(2n - 3)! / (2^(n - 2) (n - 2!))` possible binary trees with equal
        probability.

        The returned :class:`.Tree` will have the same genomic span as this tree,
        and node IDs will be conserved (that is, node ``u`` in this tree will
        be the same node in the returned tree). The returned tree is derived from a
        tree sequence that contains only one non-degenerate tree, that is, where
        edges cover only the interval spanned by this tree.

        :param epsilon: If specified, the fixed branch length between inserted
            nodes and their parents. If None (the default), the minimal possible
            nonzero branch length is generated for each node.
        :param str method: The method used to break polytomies. Currently only "random"
            is supported, which can also be specified by ``method=None``
            (Default: ``None``).
        :param bool record_provenance: If True, add details of this operation to the
            provenance information of the returned tree sequence. (Default: True).
        :param int random_seed: The random seed. If this is None, a random seed will
            be automatically generated. Valid random seeds must be between 1 and
            :math:`2^32 − 1`.
        :param \\**kwargs: Further arguments used as parameters when constructing the
            returned :class:`Tree`. For example
            ``tree.split_polytomies(sample_lists=True)`` will
            return a :class:`Tree` created with ``sample_lists=True``.
        :return: A new tree with polytomies split into random bifurcations.
        :rtype: tskit.Tree
        """
        return combinatorics.split_polytomies(
            self,
            epsilon=epsilon,
            method=method,
            record_provenance=record_provenance,
            random_seed=random_seed,
            **kwargs,
        )

    @staticmethod
    def generate_star(
        num_leaves, *, span=1, branch_length=1, record_provenance=True, **kwargs
    ):
        """
        Generate a :class:`Tree` whose leaf nodes all have the same parent (i.e.,
        a "star" tree). The leaf nodes are all at time 0 and are marked as sample nodes.

        The tree produced by this method is identical to
        ``tskit.Tree.unrank(n, (0, 0))``, but generated more efficiently for large ``n``.

        :param int num_leaves: The number of leaf nodes in the returned tree (must be
            2 or greater).
        :param float span: The span of the tree, and therefore the
            :attr:`~TreeSequence.sequence_length` of the :attr:`.tree_sequence`
            property of the returned :class:`Tree`.
        :param float branch_length: The length of every branch in the tree (equivalent
            to the time of the root node).
        :param bool record_provenance: If True, add details of this operation to the
            provenance information of the returned tree sequence. (Default: True).
        :param \\**kwargs: Further arguments used as parameters when constructing the
            returned :class:`Tree`. For example
            ``tskit.Tree.generate_star(sample_lists=True)`` will
            return a :class:`Tree` created with ``sample_lists=True``.
        :return: A star-shaped tree. Its corresponding :class:`TreeSequence` is available
            via the :attr:`.tree_sequence` attribute.
        :rtype: Tree
        """
        return combinatorics.generate_star(
            num_leaves,
            span=span,
            branch_length=branch_length,
            record_provenance=record_provenance,
            **kwargs,
        )

    @staticmethod
    def generate_balanced(
        num_leaves,
        *,
        arity=2,
        span=1,
        branch_length=1,
        record_provenance=True,
        **kwargs,
    ):
        """
        Generate a :class:`Tree` with the specified number of leaves that is maximally
        balanced. By default, the tree returned is binary, such that for each
        node that subtends :math:`n` leaves, the left child will subtend
        :math:`\\lfloor{n / 2}\\rfloor` leaves and the right child the
        remainder. Balanced trees with higher arity can also generated using the
        ``arity`` parameter, where the leaves subtending a node are distributed
        among its children analogously.

        In the returned tree, the leaf nodes are all at time 0, marked as samples,
        and labelled 0 to n from left-to-right. Internal node IDs are assigned
        sequentially from n in a postorder traversal, and the time of an internal
        node is the maximum time of its children plus the specified ``branch_length``.

        :param int num_leaves: The number of leaf nodes in the returned tree (must be
            be 2 or greater).
        :param int arity: The maximum number of children a node can have in the returned
            tree.
        :param float span: The span of the tree, and therefore the
            :attr:`~TreeSequence.sequence_length` of the :attr:`.tree_sequence`
            property of the returned :class:`Tree`.
        :param float branch_length: The minimum length of a branch in the tree (see
            above for details on how internal node times are assigned).
        :param bool record_provenance: If True, add details of this operation to the
            provenance information of the returned tree sequence. (Default: True).
        :param \\**kwargs: Further arguments used as parameters when constructing the
            returned :class:`Tree`. For example
            ``tskit.Tree.generate_balanced(sample_lists=True)`` will
            return a :class:`Tree` created with ``sample_lists=True``.
        :return: A balanced tree. Its corresponding :class:`TreeSequence` is available
            via the :attr:`.tree_sequence` attribute.
        :rtype: Tree
        """
        return combinatorics.generate_balanced(
            num_leaves,
            arity=arity,
            span=span,
            branch_length=branch_length,
            record_provenance=record_provenance,
            **kwargs,
        )

    @staticmethod
    def generate_comb(
        num_leaves, *, span=1, branch_length=1, record_provenance=True, **kwargs
    ):
        """
        Generate a :class:`Tree` in which all internal nodes have two children
        and the left child is a leaf. This is a "comb", "ladder" or "pectinate"
        phylogeny, and also known as a `caterpillar tree
        <https://en.wikipedia.org/wiki/Caterpillar_tree>`_.

        The leaf nodes are all at time 0, marked as samples,
        and labelled 0 to n from left-to-right. Internal node IDs are assigned
        sequentially from n as we ascend the tree, and the time of an internal
        node is the maximum time of its children plus the specified ``branch_length``.

        :param int num_leaves: The number of leaf nodes in the returned tree (must be
            2 or greater).
        :param float span: The span of the tree, and therefore the
            :attr:`~TreeSequence.sequence_length` of the :attr:`.tree_sequence`
            property of the returned :class:`Tree`.
        :param float branch_length: The branch length between each internal node; the
            root node is therefore placed at time ``branch_length * (num_leaves - 1)``.
        :param bool record_provenance: If True, add details of this operation to the
            provenance information of the returned tree sequence. (Default: True).
        :param \\**kwargs: Further arguments used as parameters when constructing the
            returned :class:`Tree`. For example
            ``tskit.Tree.generate_comb(sample_lists=True)`` will
            return a :class:`Tree` created with ``sample_lists=True``.
        :return: A comb-shaped bifurcating tree. Its corresponding :class:`TreeSequence`
            is available via the :attr:`.tree_sequence` attribute.
        :rtype: Tree
        """
        return combinatorics.generate_comb(
            num_leaves,
            span=span,
            branch_length=branch_length,
            record_provenance=record_provenance,
            **kwargs,
        )

    @staticmethod
    def generate_random_binary(
        num_leaves,
        *,
        span=1,
        branch_length=1,
        random_seed=None,
        record_provenance=True,
        **kwargs,
    ):
        """
        Generate a random binary :class:`Tree` with :math:`n` = ``num_leaves``
        leaves with an equal probability of returning any topology and
        leaf label permutation among the :math:`(2n - 3)! / (2^{n - 2} (n - 2)!)`
        leaf-labelled binary trees.

        The leaf nodes are marked as samples, labelled 0 to n, and placed at
        time 0. Internal node IDs are assigned sequentially from n as we ascend
        the tree, and the time of an internal node is the maximum time of its
        children plus the specified ``branch_length``.

        .. note::
            The returned tree has not been created under any explicit model of
            evolution. In order to simulate such trees, additional software
            such as `msprime <https://github.com/tskit-dev/msprime>`` is required.

        :param int num_leaves: The number of leaf nodes in the returned tree (must
            be 2 or greater).
        :param float span: The span of the tree, and therefore the
            :attr:`~TreeSequence.sequence_length` of the :attr:`.tree_sequence`
            property of the returned :class:`Tree`.
        :param float branch_length: The minimum time between parent and child nodes.
        :param int random_seed: The random seed. If this is None, a random seed will
            be automatically generated. Valid random seeds must be between 1 and
            :math:`2^32 − 1`.
        :param bool record_provenance: If True, add details of this operation to the
            provenance information of the returned tree sequence. (Default: True).
        :param \\**kwargs: Further arguments used as parameters when constructing the
            returned :class:`Tree`. For example
            ``tskit.Tree.generate_comb(sample_lists=True)`` will
            return a :class:`Tree` created with ``sample_lists=True``.
        :return: A random binary tree. Its corresponding :class:`TreeSequence` is
            available via the :attr:`.tree_sequence` attribute.
        :rtype: Tree
        """
        return combinatorics.generate_random_binary(
            num_leaves,
            span=span,
            branch_length=branch_length,
            random_seed=random_seed,
            record_provenance=record_provenance,
            **kwargs,
        )


def load(file, *, skip_tables=False, skip_reference_sequence=False):
    """
    Return a :class:`TreeSequence` instance loaded from the specified file object or
    path. The file must be in the
    :ref:`tree sequence file format <sec_tree_sequence_file_format>`
    produced by the :meth:`TreeSequence.dump` method.

    .. warning:: With any of the ``skip_tables`` or ``skip_reference_sequence``
        options set, it is not possible to load data from a non-seekable stream
        (e.g. a socket or STDIN) of multiple tree sequences using consecutive
        calls to :meth:`tskit.load`.

    :param str file: The file object or path of the ``.trees`` file containing the
        tree sequence we wish to load.
    :param bool skip_tables: If True, no tables are read from the ``.trees``
        file and only the top-level information is populated in the tree
        sequence object.
    :param bool skip_reference_sequence: If True, the tree sequence is read
        without loading its reference sequence.
    :return: The tree sequence object containing the information
        stored in the specified file path.
    :rtype: :class:`tskit.TreeSequence`
    """
    return TreeSequence.load(
        file, skip_tables=skip_tables, skip_reference_sequence=skip_reference_sequence
    )


def parse_individuals(
    source, strict=True, encoding="utf8", base64_metadata=True, table=None
):
    """
    Parse the specified file-like object containing a whitespace delimited
    description of an individual table and returns the corresponding
    :class:`IndividualTable` instance. See the :ref:`individual text format
    <sec_individual_text_format>` section for the details of the required
    format and the :ref:`individual table definition
    <sec_individual_table_definition>` section for the required properties of
    the contents.

    See :func:`tskit.load_text` for a detailed explanation of the ``strict``
    parameter.

    :param io.TextIOBase source: The file-like object containing the text.
    :param bool strict: If True, require strict tab delimiting (default). If
        False, a relaxed whitespace splitting algorithm is used.
    :param str encoding: Encoding used for text representation.
    :param bool base64_metadata: If True, metadata is encoded using Base64
        encoding; otherwise, as plain text.
    :param IndividualTable table: If specified write into this table. If not,
        create a new :class:`IndividualTable` instance.
    """
    sep = None
    if strict:
        sep = "\t"
    if table is None:
        table = tables.IndividualTable()
    # Read the header and find the indexes of the required fields.
    header = source.readline().rstrip("\n").split(sep)
    flags_index = header.index("flags")
    location_index = None
    parents_index = None
    metadata_index = None
    try:
        location_index = header.index("location")
    except ValueError:
        pass
    try:
        parents_index = header.index("parents")
    except ValueError:
        pass
    try:
        metadata_index = header.index("metadata")
    except ValueError:
        pass
    for line in source:
        tokens = line.rstrip("\n").split(sep)
        if len(tokens) >= 1:
            flags = int(tokens[flags_index])
            location = ()
            if location_index is not None:
                location_string = tokens[location_index]
                if len(location_string) > 0:
                    location = tuple(map(float, location_string.split(",")))
            parents = ()
            if parents_index is not None:
                parents_string = tokens[parents_index]
                if len(parents_string) > 0:
                    parents = tuple(map(int, parents_string.split(",")))
            metadata = b""
            if metadata_index is not None and metadata_index < len(tokens):
                metadata = tokens[metadata_index].encode(encoding)
                if base64_metadata:
                    metadata = base64.b64decode(metadata)
            table.add_row(
                flags=flags, location=location, parents=parents, metadata=metadata
            )
    return table


def parse_nodes(source, strict=True, encoding="utf8", base64_metadata=True, table=None):
    """
    Parse the specified file-like object containing a whitespace delimited
    description of a node table and returns the corresponding :class:`NodeTable`
    instance. See the :ref:`node text format <sec_node_text_format>` section
    for the details of the required format and the
    :ref:`node table definition <sec_node_table_definition>` section for the
    required properties of the contents.

    See :func:`tskit.load_text` for a detailed explanation of the ``strict``
    parameter.

    :param io.TextIOBase source: The file-like object containing the text.
    :param bool strict: If True, require strict tab delimiting (default). If
        False, a relaxed whitespace splitting algorithm is used.
    :param str encoding: Encoding used for text representation.
    :param bool base64_metadata: If True, metadata is encoded using Base64
        encoding; otherwise, as plain text.
    :param NodeTable table: If specified write into this table. If not,
        create a new :class:`NodeTable` instance.
    """
    sep = None
    if strict:
        sep = "\t"
    if table is None:
        table = tables.NodeTable()
    # Read the header and find the indexes of the required fields.
    header = source.readline().rstrip("\n").split(sep)
    is_sample_index = header.index("is_sample")
    time_index = header.index("time")
    population_index = None
    individual_index = None
    metadata_index = None
    try:
        population_index = header.index("population")
    except ValueError:
        pass
    try:
        individual_index = header.index("individual")
    except ValueError:
        pass
    try:
        metadata_index = header.index("metadata")
    except ValueError:
        pass
    for line in source:
        tokens = line.rstrip("\n").split(sep)
        if len(tokens) >= 2:
            is_sample = int(tokens[is_sample_index])
            time = float(tokens[time_index])
            flags = 0
            if is_sample != 0:
                flags |= NODE_IS_SAMPLE
            population = NULL
            if population_index is not None:
                population = int(tokens[population_index])
            individual = NULL
            if individual_index is not None:
                individual = int(tokens[individual_index])
            metadata = b""
            if metadata_index is not None and metadata_index < len(tokens):
                metadata = tokens[metadata_index].encode(encoding)
                if base64_metadata:
                    metadata = base64.b64decode(metadata)
            table.add_row(
                flags=flags,
                time=time,
                population=population,
                individual=individual,
                metadata=metadata,
            )
    return table


def parse_edges(source, strict=True, table=None):
    """
    Parse the specified file-like object containing a whitespace delimited
    description of a edge table and returns the corresponding :class:`EdgeTable`
    instance. See the :ref:`edge text format <sec_edge_text_format>` section
    for the details of the required format and the
    :ref:`edge table definition <sec_edge_table_definition>` section for the
    required properties of the contents.

    See :func:`tskit.load_text` for a detailed explanation of the ``strict`` parameter.

    :param io.TextIOBase source: The file-like object containing the text.
    :param bool strict: If True, require strict tab delimiting (default). If
        False, a relaxed whitespace splitting algorithm is used.
    :param EdgeTable table: If specified, write the edges into this table. If
        not, create a new :class:`EdgeTable` instance and return.
    """
    sep = None
    if strict:
        sep = "\t"
    if table is None:
        table = tables.EdgeTable()
    header = source.readline().rstrip("\n").split(sep)
    left_index = header.index("left")
    right_index = header.index("right")
    parent_index = header.index("parent")
    children_index = header.index("child")
    for line in source:
        tokens = line.rstrip("\n").split(sep)
        if len(tokens) >= 4:
            left = float(tokens[left_index])
            right = float(tokens[right_index])
            parent = int(tokens[parent_index])
            children = tuple(map(int, tokens[children_index].split(",")))
            for child in children:
                table.add_row(left=left, right=right, parent=parent, child=child)
    return table


def parse_sites(source, strict=True, encoding="utf8", base64_metadata=True, table=None):
    """
    Parse the specified file-like object containing a whitespace delimited
    description of a site table and returns the corresponding :class:`SiteTable`
    instance. See the :ref:`site text format <sec_site_text_format>` section
    for the details of the required format and the
    :ref:`site table definition <sec_site_table_definition>` section for the
    required properties of the contents.

    See :func:`tskit.load_text` for a detailed explanation of the ``strict``
    parameter.

    :param io.TextIOBase source: The file-like object containing the text.
    :param bool strict: If True, require strict tab delimiting (default). If
        False, a relaxed whitespace splitting algorithm is used.
    :param str encoding: Encoding used for text representation.
    :param bool base64_metadata: If True, metadata is encoded using Base64
        encoding; otherwise, as plain text.
    :param SiteTable table: If specified write site into this table. If not,
        create a new :class:`SiteTable` instance.
    """
    sep = None
    if strict:
        sep = "\t"
    if table is None:
        table = tables.SiteTable()
    header = source.readline().rstrip("\n").split(sep)
    position_index = header.index("position")
    ancestral_state_index = header.index("ancestral_state")
    metadata_index = None
    try:
        metadata_index = header.index("metadata")
    except ValueError:
        pass
    for line in source:
        tokens = line.rstrip("\n").split(sep)
        if len(tokens) >= 2:
            position = float(tokens[position_index])
            ancestral_state = tokens[ancestral_state_index]
            metadata = b""
            if metadata_index is not None and metadata_index < len(tokens):
                metadata = tokens[metadata_index].encode(encoding)
                if base64_metadata:
                    metadata = base64.b64decode(metadata)
            table.add_row(
                position=position, ancestral_state=ancestral_state, metadata=metadata
            )
    return table


def parse_mutations(
    source, strict=True, encoding="utf8", base64_metadata=True, table=None
):
    """
    Parse the specified file-like object containing a whitespace delimited
    description of a mutation table and returns the corresponding :class:`MutationTable`
    instance. See the :ref:`mutation text format <sec_mutation_text_format>` section
    for the details of the required format and the
    :ref:`mutation table definition <sec_mutation_table_definition>` section for the
    required properties of the contents. Note that if the ``time`` column is missing its
    entries are filled with ``UNKNOWN_TIME``.

    See :func:`tskit.load_text` for a detailed explanation of the ``strict``
    parameter.

    :param io.TextIOBase source: The file-like object containing the text.
    :param bool strict: If True, require strict tab delimiting (default). If
        False, a relaxed whitespace splitting algorithm is used.
    :param str encoding: Encoding used for text representation.
    :param bool base64_metadata: If True, metadata is encoded using Base64
        encoding; otherwise, as plain text.
    :param MutationTable table: If specified, write mutations into this table.
        If not, create a new :class:`MutationTable` instance.
    """
    sep = None
    if strict:
        sep = "\t"
    if table is None:
        table = tables.MutationTable()
    header = source.readline().rstrip("\n").split(sep)
    site_index = header.index("site")
    node_index = header.index("node")
    try:
        time_index = header.index("time")
    except ValueError:
        time_index = None
    derived_state_index = header.index("derived_state")
    parent_index = None
    parent = NULL
    try:
        parent_index = header.index("parent")
    except ValueError:
        pass
    metadata_index = None
    try:
        metadata_index = header.index("metadata")
    except ValueError:
        pass
    for line in source:
        tokens = line.rstrip("\n").split(sep)
        if len(tokens) >= 3:
            site = int(tokens[site_index])
            node = int(tokens[node_index])
            if time_index is None or tokens[time_index] == tskit.TIME_UNITS_UNKNOWN:
                time = UNKNOWN_TIME
            else:
                time = float(tokens[time_index])
            derived_state = tokens[derived_state_index]
            if parent_index is not None:
                parent = int(tokens[parent_index])
            metadata = b""
            if metadata_index is not None and metadata_index < len(tokens):
                metadata = tokens[metadata_index].encode(encoding)
                if base64_metadata:
                    metadata = base64.b64decode(metadata)
            table.add_row(
                site=site,
                node=node,
                time=time,
                derived_state=derived_state,
                parent=parent,
                metadata=metadata,
            )
    return table


def parse_populations(
    source, strict=True, encoding="utf8", base64_metadata=True, table=None
):
    """
    Parse the specified file-like object containing a whitespace delimited
    description of a population table and returns the corresponding
    :class:`PopulationTable` instance. See the :ref:`population text format
    <sec_population_text_format>` section for the details of the required
    format and the :ref:`population table definition
    <sec_population_table_definition>` section for the required properties of
    the contents.

    See :func:`tskit.load_text` for a detailed explanation of the ``strict``
    parameter.

    :param io.TextIOBase source: The file-like object containing the text.
    :param bool strict: If True, require strict tab delimiting (default). If
        False, a relaxed whitespace splitting algorithm is used.
    :param str encoding: Encoding used for text representation.
    :param bool base64_metadata: If True, metadata is encoded using Base64
        encoding; otherwise, as plain text.
    :param PopulationTable table: If specified write into this table. If not,
        create a new :class:`PopulationTable` instance.
    """
    sep = None
    if strict:
        sep = "\t"
    if table is None:
        table = tables.PopulationTable()
    # Read the header and find the indexes of the required fields.
    header = source.readline().rstrip("\n").split(sep)
    metadata_index = header.index("metadata")
    for line in source:
        tokens = line.rstrip("\n").split(sep)
        if len(tokens) >= 1:
            metadata = tokens[metadata_index].encode(encoding)
            if base64_metadata:
                metadata = base64.b64decode(metadata)
            table.add_row(metadata=metadata)
    return table


def parse_migrations(
    source, strict=True, encoding="utf8", base64_metadata=True, table=None
):
    """
    Parse the specified file-like object containing a whitespace delimited
    description of a migration table and returns the corresponding
    :class:`MigrationTable` instance.

    See the :ref:`migration text format <sec_migration_text_format>` section
    for the details of the required format and the
    :ref:`migration table definition <sec_migration_table_definition>` section
    for the required properties of the contents. Note that if the ``time`` column
    is missing its entries are filled with :data:`UNKNOWN_TIME`.

    See :func:`tskit.load_text` for a detailed explanation of the ``strict``
    parameter.

    :param io.TextIOBase source: The file-like object containing the text.
    :param bool strict: If True, require strict tab delimiting (default). If
        False, a relaxed whitespace splitting algorithm is used.
    :param str encoding: Encoding used for text representation.
    :param bool base64_metadata: If True, metadata is encoded using Base64
        encoding; otherwise, as plain text.
    :param MigrationTable table: If specified, write migrations into this table.
        If not, create a new :class:`MigrationTable` instance.
    """
    sep = None
    if strict:
        sep = "\t"
    if table is None:
        table = tables.MigrationTable()
    header = source.readline().rstrip("\n").split(sep)
    left_index = header.index("left")
    right_index = header.index("right")
    node_index = header.index("node")
    source_index = header.index("source")
    dest_index = header.index("dest")
    time_index = header.index("time")
    metadata_index = None
    try:
        metadata_index = header.index("metadata")
    except ValueError:
        pass
    for line in source:
        tokens = line.rstrip("\n").split(sep)
        if len(tokens) >= 6:
            left = float(tokens[left_index])
            right = float(tokens[right_index])
            node = int(tokens[node_index])
            source = int(tokens[source_index])
            dest = int(tokens[dest_index])
            time = float(tokens[time_index])
            metadata = b""
            if metadata_index is not None and metadata_index < len(tokens):
                metadata = tokens[metadata_index].encode(encoding)
                if base64_metadata:
                    metadata = base64.b64decode(metadata)
            table.add_row(
                left=left,
                right=right,
                node=node,
                source=source,
                dest=dest,
                time=time,
                metadata=metadata,
            )
    return table


def load_text(
    nodes,
    edges,
    sites=None,
    mutations=None,
    individuals=None,
    populations=None,
    migrations=None,
    sequence_length=0,
    strict=True,
    encoding="utf8",
    base64_metadata=True,
):
    """
    Return a :class:`TreeSequence` instance parsed from tabulated text data
    contained in the specified file-like objects. The format
    for these files is documented in the :ref:`sec_text_file_format` section,
    and is produced by the :meth:`TreeSequence.dump_text` method. Further
    properties required for an input tree sequence are described in the
    :ref:`sec_valid_tree_sequence_requirements` section. This method is intended as a
    convenient interface for importing external data into tskit; the binary
    file format using by :meth:`tskit.load` is many times more efficient than
    this text format.

    The ``nodes`` and ``edges`` parameters are mandatory and must be file-like
    objects containing text with whitespace delimited columns,  parsable by
    :func:`parse_nodes` and :func:`parse_edges`, respectively. ``sites``,
    ``individuals``, ``populations``, ``mutations``, and ``migrations`` are optional,
    and must be parsable by :func:`parse_sites`, :func:`parse_individuals`,
    :func:`parse_populations`, :func:`parse_mutations`, and :func:`parse_migrations`,
    respectively. For convenience, if the node table refers to populations,
    but the ``populations`` parameter is not provided, a minimal set of rows are
    added to the population table, so that a valid tree sequence can be returned.

    The ``sequence_length`` parameter determines the
    :attr:`TreeSequence.sequence_length` of the returned tree sequence. If it
    is 0 or not specified, the value is taken to be the maximum right
    coordinate of the input edges. This parameter is useful in degenerate
    situations (such as when there are zero edges), but can usually be ignored.

    The ``strict`` parameter controls the field delimiting algorithm that
    is used. If ``strict`` is True (the default), we require exactly one
    tab character separating each field. If ``strict`` is False, a more relaxed
    whitespace delimiting algorithm is used, such that any run of whitespace
    is regarded as a field separator. In most situations, ``strict=False``
    is more convenient, but it can lead to error in certain situations. For
    example, if a deletion is encoded in the mutation table this will not
    be parseable when ``strict=False``.

    After parsing the tables, :meth:`TableCollection.sort` is called to ensure that
    the loaded tables satisfy the tree sequence :ref:`ordering requirements
    <sec_valid_tree_sequence_requirements>`. Note that this may result in the
    IDs of various entities changing from their positions in the input file.

    :param io.TextIOBase nodes: The file-like object containing text describing a
        :class:`NodeTable`.
    :param io.TextIOBase edges: The file-like object containing text
        describing an :class:`EdgeTable`.
    :param io.TextIOBase sites: The file-like object containing text describing a
        :class:`SiteTable`.
    :param io.TextIOBase mutations: The file-like object containing text
        describing a :class:`MutationTable`.
    :param io.TextIOBase individuals: The file-like object containing text
        describing a :class:`IndividualTable`.
    :param io.TextIOBase populations: The file-like object containing text
        describing a :class:`PopulationTable`.
    :param io.TextIOBase migrations: The file-like object containing text
        describing a :class:`MigrationTable`.
    :param float sequence_length: The sequence length of the returned tree sequence. If
        not supplied or zero this will be inferred from the set of edges.
    :param bool strict: If True, require strict tab delimiting (default). If
        False, a relaxed whitespace splitting algorithm is used.
    :param str encoding: Encoding used for text representation.
    :param bool base64_metadata: If True, metadata is encoded using Base64
        encoding; otherwise, as plain text.
    :return: The tree sequence object containing the information
        stored in the specified file paths.
    :rtype: :class:`tskit.TreeSequence`
    """
    # We need to parse the edges so we can figure out the sequence length, and
    # TableCollection.sequence_length is immutable so we need to create a temporary
    # edge table.
    edge_table = parse_edges(edges, strict=strict)
    if sequence_length == 0 and len(edge_table) > 0:
        sequence_length = edge_table.right.max()
    tc = tables.TableCollection(sequence_length)
    tc.edges.set_columns(
        left=edge_table.left,
        right=edge_table.right,
        parent=edge_table.parent,
        child=edge_table.child,
    )
    parse_nodes(
        nodes,
        strict=strict,
        encoding=encoding,
        base64_metadata=base64_metadata,
        table=tc.nodes,
    )
    if sites is not None:
        parse_sites(
            sites,
            strict=strict,
            encoding=encoding,
            base64_metadata=base64_metadata,
            table=tc.sites,
        )
    if mutations is not None:
        parse_mutations(
            mutations,
            strict=strict,
            encoding=encoding,
            base64_metadata=base64_metadata,
            table=tc.mutations,
        )
    if individuals is not None:
        parse_individuals(
            individuals,
            strict=strict,
            encoding=encoding,
            base64_metadata=base64_metadata,
            table=tc.individuals,
        )
    if populations is None:
        # As a convenience we add any populations referenced in the node table.
        if len(tc.nodes) > 0:
            max_population = tc.nodes.population.max()
            if max_population != NULL:
                for _ in range(max_population + 1):
                    tc.populations.add_row()
    else:
        parse_populations(
            populations,
            strict=strict,
            encoding=encoding,
            base64_metadata=base64_metadata,
            table=tc.populations,
        )
    if migrations is not None:
        parse_migrations(
            migrations,
            strict=strict,
            encoding=encoding,
            base64_metadata=base64_metadata,
            table=tc.migrations,
        )
    tc.sort()
    return tc.tree_sequence()


class TreeIterator:
    """
    Simple class providing forward and backward iteration over a tree sequence.
    """

    def __init__(self, tree):
        self.tree = tree
        self.more_trees = True
        self.forward = True

    def __iter__(self):
        return self

    def __reversed__(self):
        self.forward = False
        return self

    def __next__(self):
        if self.forward:
            self.more_trees = self.more_trees and self.tree.next()
        else:
            self.more_trees = self.more_trees and self.tree.prev()
        if not self.more_trees:
            raise StopIteration()
        return self.tree

    def __len__(self):
        return self.tree.tree_sequence.num_trees


class SimpleContainerSequence:
    """
    Simple wrapper to allow arrays of SimpleContainers (e.g. edges, nodes) that have a
    function allowing access by index (e.g. ts.edge(i), ts.node(i)) to be treated as a
    python sequence, allowing forward and reverse iteration.

    To generate a sequence of items in a different order, the ``order`` parameter allows
    an array of indexes to be passed in, such as returned from np.argsort or np.lexsort.
    """

    def __init__(self, getter, length, order=None):
        if order is None:
            self.getter = getter
        else:
            self.getter = lambda index: getter(order[index])
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.getter(index)


@dataclass(frozen=True)
class TableMetadataSchemas:
    """
    Convenience class for returning the schemas of all the tables in a tree sequence.
    """

    node: metadata_module.MetadataSchema = None
    """
    The metadata schema of the node table.
    """

    edge: metadata_module.MetadataSchema = None
    """
    The metadata schema of the edge table.
    """

    site: metadata_module.MetadataSchema = None
    """
    The metadata schema of the site table.
    """

    mutation: metadata_module.MetadataSchema = None
    """
    The metadata schema of the mutation table.
    """

    migration: metadata_module.MetadataSchema = None
    """
    The metadata schema of the migration table.
    """

    individual: metadata_module.MetadataSchema = None
    """
    The metadata schema of the individual table.
    """

    population: metadata_module.MetadataSchema = None
    """
    The metadata schema of the population table.
    """


class TreeSequence:
    """
    A single tree sequence, as defined by the :ref:`data model <sec_data_model>`.
    A TreeSequence instance can be created from a set of
    :ref:`tables <sec_table_definitions>` using
    :meth:`TableCollection.tree_sequence`, or loaded from a set of text files
    using :func:`tskit.load_text`, or loaded from a native binary file using
    :func:`tskit.load`.

    TreeSequences are immutable. To change the data held in a particular
    tree sequence, first get the table information as a :class:`TableCollection`
    instance (using :meth:`.dump_tables`), edit those tables using the
    :ref:`tables api <sec_tables_api>`, and create a new tree sequence using
    :meth:`TableCollection.tree_sequence`.

    The :meth:`.trees` method iterates over all trees in a tree sequence, and
    the :meth:`.variants` method iterates over all sites and their genotypes.
    """

    def __init__(self, ll_tree_sequence):
        self._ll_tree_sequence = ll_tree_sequence
        self._immutable_tables = None
        metadata_schema_strings = self._ll_tree_sequence.get_table_metadata_schemas()
        metadata_schema_instances = {
            name: metadata_module.parse_metadata_schema(
                getattr(metadata_schema_strings, name)
            )
            for name in vars(TableMetadataSchemas)
            if not name.startswith("_")
        }
        self._table_metadata_schemas = TableMetadataSchemas(**metadata_schema_instances)
        self._individuals_time = None
        self._individuals_population = None
        self._individuals_location = None
        self._individuals_nodes = None
        self._mutations_edge = None
        self._mutations_inherited_state = None
        self._sites_ancestral_state = None
        self._mutations_derived_state = None
        # NOTE: when we've implemented read-only access via the underlying
        # tables we can replace these arrays with reference to the read-only
        # tables here (and remove the low-level boilerplate).
        llts = self._ll_tree_sequence
        self._individuals_flags = llts.individuals_flags
        self._individuals_metadata = llts.individuals_metadata
        self._nodes_time = llts.nodes_time
        self._nodes_flags = llts.nodes_flags
        self._nodes_population = llts.nodes_population
        self._nodes_individual = llts.nodes_individual
        self._nodes_metadata = llts.nodes_metadata
        self._edges_left = llts.edges_left
        self._edges_right = llts.edges_right
        self._edges_parent = llts.edges_parent
        self._edges_child = llts.edges_child
        self._edges_metadata = llts.edges_metadata
        self._sites_position = llts.sites_position
        self._sites_metadata = llts.sites_metadata
        self._mutations_site = llts.mutations_site
        self._mutations_node = llts.mutations_node
        self._mutations_parent = llts.mutations_parent
        self._mutations_time = llts.mutations_time
        self._mutations_metadata = llts.mutations_metadata
        self._migrations_left = llts.migrations_left
        self._migrations_right = llts.migrations_right
        self._migrations_node = llts.migrations_node
        self._migrations_source = llts.migrations_source
        self._migrations_dest = llts.migrations_dest
        self._migrations_time = llts.migrations_time
        self._migrations_metadata = llts.migrations_metadata
        self._populations_metadata = llts.populations_metadata
        self._indexes_edge_insertion_order = llts.indexes_edge_insertion_order
        self._indexes_edge_removal_order = llts.indexes_edge_removal_order

    # Implement the pickle protocol for TreeSequence
    def __getstate__(self):
        return self.dump_tables()

    def __setstate__(self, tc):
        self.__init__(tc.tree_sequence().ll_tree_sequence)

    def __eq__(self, other):
        return self.tables == other.tables

    def equals(
        self,
        other,
        *,
        ignore_metadata=False,
        ignore_ts_metadata=False,
        ignore_provenance=False,
        ignore_timestamps=False,
        ignore_tables=False,
        ignore_reference_sequence=False,
    ):
        """
        Returns True if  `self` and `other` are equal. Uses the underlying table
        equality, see :meth:`TableCollection.equals` for details and options.
        """
        return self.tables.equals(
            other.tables,
            ignore_metadata=ignore_metadata,
            ignore_ts_metadata=ignore_ts_metadata,
            ignore_provenance=ignore_provenance,
            ignore_timestamps=ignore_timestamps,
            ignore_tables=ignore_tables,
            ignore_reference_sequence=ignore_reference_sequence,
        )

    @property
    def ll_tree_sequence(self):
        return self.get_ll_tree_sequence()

    def get_ll_tree_sequence(self):
        return self._ll_tree_sequence

    def aslist(self, **kwargs):
        """
        Returns the trees in this tree sequence as a list. Each tree is
        represented by a different instance of :class:`Tree`. As such, this
        method is inefficient and may use a large amount of memory, and should
        not be used when performance is a consideration. The :meth:`.trees`
        method is the recommended way to efficiently iterate over the trees
        in a tree sequence.

        :param \\**kwargs: Further arguments used as parameters when constructing the
            returned trees. For example ``ts.aslist(sample_lists=True)`` will result
            in a list of :class:`Tree` instances created with ``sample_lists=True``.
        :return: A list of the trees in this tree sequence.
        :rtype: list
        """
        return [tree.copy() for tree in self.trees(**kwargs)]

    @classmethod
    def load(cls, file_or_path, *, skip_tables=False, skip_reference_sequence=False):
        file, local_file = util.convert_file_like_to_open_file(file_or_path, "rb")
        try:
            ts = _tskit.TreeSequence()
            ts.load(
                file,
                skip_tables=skip_tables,
                skip_reference_sequence=skip_reference_sequence,
            )
            return TreeSequence(ts)
        except tskit.FileFormatError as e:
            util.raise_known_file_format_errors(file, e)
        finally:
            if local_file:
                file.close()

    @classmethod
    def load_tables(cls, tables, *, build_indexes=False):
        ts = _tskit.TreeSequence()
        ts.load_tables(tables._ll_tables, build_indexes=build_indexes)
        return TreeSequence(ts)

    def dump(self, file_or_path, zlib_compression=False):
        """
        Writes the tree sequence to the specified path or file object.

        :param str file_or_path: The file object or path to write the TreeSequence to.
        :param bool zlib_compression: This parameter is deprecated and ignored.
        """
        if zlib_compression:
            # Note: the msprime CLI before version 1.0 uses this option, so we need
            # to keep it indefinitely.
            warnings.warn(
                "The zlib_compression option is no longer supported and is ignored",
                RuntimeWarning,
                stacklevel=4,
            )
        file, local_file = util.convert_file_like_to_open_file(file_or_path, "wb")
        try:
            self._ll_tree_sequence.dump(file)
        finally:
            if local_file:
                file.close()

    @property
    def reference_sequence(self):
        """
        The :class:`.ReferenceSequence` associated with this :class:`.TreeSequence`
        if one is defined (see :meth:`.TreeSequence.has_reference_sequence`),
        or None otherwise.
        """
        if self.has_reference_sequence():
            return tables.ReferenceSequence(self._ll_tree_sequence.reference_sequence)
        return None

    def has_reference_sequence(self):
        """
        Returns True if this :class:`.TreeSequence` has an associated
        :ref:`reference sequence<sec_data_model_reference_sequence>`.
        """
        return bool(self._ll_tree_sequence.has_reference_sequence())

    @property
    def tables_dict(self):
        """
        Returns a dictionary mapping names to tables in the
        underlying :class:`.TableCollection`. Equivalent to calling
        ``ts.tables.table_name_map``.
        """
        return self.tables.table_name_map

    @property
    def tables(self):
        """
        Returns an immutable view of the tables underlying this tree sequence.

        This view shares the same data as the TreeSequence (zero-copy).
        Use :meth:`.dump_tables` for a modifiable copy.

        Note that if tskit was built with Numpy 1, this method acts as
        :meth:`.dump_tables` and returns a mutable TableCollection.

        :return: An immutable view of the TableCollection underlying this tree sequence.
        """
        if not _tskit.HAS_NUMPY_2:
            warnings.warn(
                "Immutable table views require tskit to be built against NumPy 2.0 or "
                "newer. Falling back to returning a mutable TableCollection.",
                UserWarning,
                stacklevel=2,
            )
            return self.dump_tables()
        if self._immutable_tables is None:
            self._immutable_tables = tables.ImmutableTableCollection(
                self._ll_tree_sequence
            )
        return self._immutable_tables

    @property
    def nbytes(self):
        """
        Returns the total number of bytes required to store the data
        in this tree sequence. Note that this may not be equal to
        the actual memory footprint.
        """
        return self.tables.nbytes

    def dump_tables(self):
        """
        Returns a modifiable copy of the :class:`tables<TableCollection>` defining
        this tree sequence.

        :return: A :class:`TableCollection` containing all tables underlying
            the tree sequence.
        :rtype: TableCollection
        """
        ll_tables = _tskit.TableCollection(self.sequence_length)
        self._ll_tree_sequence.dump_tables(ll_tables)
        return tables.TableCollection(ll_tables=ll_tables)

    def dump_text(
        self,
        nodes=None,
        edges=None,
        sites=None,
        mutations=None,
        individuals=None,
        populations=None,
        migrations=None,
        provenances=None,
        precision=6,
        encoding="utf8",
        base64_metadata=True,
    ):
        """
        Writes a text representation of the tables underlying the tree sequence
        to the specified connections.

        If Base64 encoding is not used, then metadata will be saved directly, possibly
        resulting in errors reading the tables back in if metadata includes whitespace.

        :param io.TextIOBase nodes: The file-like object (having a .write() method) to
            write the NodeTable to.
        :param io.TextIOBase edges: The file-like object to write the EdgeTable to.
        :param io.TextIOBase sites: The file-like object to write the SiteTable to.
        :param io.TextIOBase mutations: The file-like object to write the
            MutationTable to.
        :param io.TextIOBase individuals: The file-like object to write the
            IndividualTable to.
        :param io.TextIOBase populations: The file-like object to write the
            PopulationTable to.
        :param io.TextIOBase migrations: The file-like object to write the
            MigrationTable to.
        :param io.TextIOBase provenances: The file-like object to write the
            ProvenanceTable to.
        :param int precision: The number of digits of precision.
        :param str encoding: Encoding used for text representation.
        :param bool base64_metadata: Only used if a schema is not present on each table
            being dumped. If True, metadata is encoded using Base64
            encoding; otherwise, as plain text.
        """
        text_formats.dump_text(
            self,
            nodes=nodes,
            edges=edges,
            sites=sites,
            mutations=mutations,
            individuals=individuals,
            populations=populations,
            migrations=migrations,
            provenances=provenances,
            precision=precision,
            encoding=encoding,
            base64_metadata=base64_metadata,
        )

    def __str__(self):
        """
        Return a plain text summary of the contents of a tree sequence
        """
        ts_rows = [
            ["Trees", util.format_number(self.num_trees, sep=",")],
            ["Sequence Length", util.format_number(self.sequence_length, sep=",")],
            ["Time Units", self.time_units],
            ["Sample Nodes", util.format_number(self.num_samples, sep=",")],
            ["Total Size", util.naturalsize(self.nbytes)],
        ]
        header = ["Table", "Rows", "Size", "Has Metadata"]
        table_rows = []
        for name, table in self.tables.table_name_map.items():
            table_rows.append(
                [
                    name.capitalize(),
                    f"{util.format_number(table.num_rows, sep=',')}",
                    util.naturalsize(table.nbytes),
                    (
                        "Yes"
                        if hasattr(table, "metadata") and len(table.metadata) > 0
                        else "No"
                    ),
                ]
            )
        return util.unicode_table(ts_rows, title="TreeSequence") + util.unicode_table(
            table_rows, header=header
        )

    def _repr_html_(self):
        """
        Return an html summary of a tree sequence. Called by jupyter notebooks
        to render a TreeSequence.
        """
        return util.tree_sequence_html(self)

    # num_samples was originally called sample_size, and so we must keep sample_size
    # around as a deprecated alias.
    @property
    def num_samples(self):
        """
        Returns the number of sample nodes in this tree sequence. This is also the
        number of sample nodes in each tree.

        :return: The number of sample nodes in this tree sequence.
        :rtype: int
        """
        return self._ll_tree_sequence.get_num_samples()

    @property
    def table_metadata_schemas(self) -> TableMetadataSchemas:
        """
        The set of metadata schemas for the tables in this tree sequence.
        """
        return self._table_metadata_schemas

    @property
    def sample_size(self):
        # Deprecated alias for num_samples
        return self.num_samples

    def get_sample_size(self):
        # Deprecated alias for num_samples
        return self.num_samples

    @property
    def file_uuid(self):
        return self._ll_tree_sequence.get_file_uuid()

    @property
    def discrete_genome(self):
        """
        Returns True if all genome coordinates in this TreeSequence are
        discrete integer values. This is true iff all the following are true:

        - The sequence length is discrete
        - All site positions are discrete
        - All left and right edge coordinates are discrete
        - All migration left and right coordinates are discrete

        :return: True if this TreeSequence uses discrete genome coordinates.
        :rtype: bool
        """
        return bool(self._ll_tree_sequence.get_discrete_genome())

    @property
    def discrete_time(self):
        """
        Returns True if all time coordinates in this TreeSequence are
        discrete integer values. This is true iff all the following are true:

        - All node times are discrete
        - All mutation times are discrete
        - All migration times are discrete

        Note that ``tskit.UNKNOWN_TIME`` counts as discrete.

        :return: True if this TreeSequence uses discrete time coordinates.
        :rtype: bool
        """
        return bool(self._ll_tree_sequence.get_discrete_time())

    @property
    def min_time(self):
        """
        Returns the min time in this tree sequence. This is the minimum
        of the node times and mutation times.

        Note that mutation times with the value ``tskit.UNKNOWN_TIME``
        are ignored.

        :return: The min time of the nodes and mutations in this tree sequence.
        :rtype: float
        """
        return self._ll_tree_sequence.get_min_time()

    @property
    def max_time(self):
        """
        Returns the max time in this tree sequence. This is the maximum
        of the node times and mutation times.

        Note that mutation times with the value ``tskit.UNKNOWN_TIME``
        are ignored.

        :return: The max time of the nodes and mutations in this tree sequence.
        :rtype: float
        """
        return self._ll_tree_sequence.get_max_time()

    @property
    def sequence_length(self):
        """
        Returns the sequence length in this tree sequence. This defines the
        genomic scale over which tree coordinates are defined. Given a
        tree sequence with a sequence length :math:`L`, the constituent
        trees will be defined over the half-closed interval
        :math:`[0, L)`. Each tree then covers some subset of this
        interval --- see :attr:`tskit.Tree.interval` for details.

        :return: The length of the sequence in this tree sequence in bases.
        :rtype: float
        """
        return self.get_sequence_length()

    def get_sequence_length(self):
        return self._ll_tree_sequence.get_sequence_length()

    @property
    def metadata(self) -> Any:
        """
        The decoded metadata for this TreeSequence.
        """
        return self.metadata_schema.decode_row(self._ll_tree_sequence.get_metadata())

    @property
    def metadata_schema(self) -> metadata_module.MetadataSchema:
        """
        The :class:`tskit.MetadataSchema` for this TreeSequence.
        """
        return metadata_module.parse_metadata_schema(
            self._ll_tree_sequence.get_metadata_schema()
        )

    @property
    def time_units(self) -> str:
        """
        String describing the units of the time dimension for this TreeSequence.
        """
        return self._ll_tree_sequence.get_time_units()

    @property
    def num_edges(self):
        """
        Returns the number of :ref:`edges <sec_edge_table_definition>` in this
        tree sequence.

        :return: The number of edges in this tree sequence.
        :rtype: int
        """
        return self._ll_tree_sequence.get_num_edges()

    def get_num_trees(self):
        # Deprecated alias for self.num_trees
        return self.num_trees

    @property
    def num_trees(self):
        """
        Returns the number of distinct trees in this tree sequence. This
        is equal to the number of trees returned by the :meth:`.trees`
        method.

        :return: The number of trees in this tree sequence.
        :rtype: int
        """
        return self._ll_tree_sequence.get_num_trees()

    def get_num_sites(self):
        # Deprecated alias for self.num_sites
        return self._ll_tree_sequence.get_num_sites()

    @property
    def num_sites(self):
        """
        Returns the number of :ref:`sites <sec_site_table_definition>` in
        this tree sequence.

        :return: The number of sites in this tree sequence.
        :rtype: int
        """
        return self.get_num_sites()

    def get_num_mutations(self):
        # Deprecated alias for self.num_mutations
        return self.num_mutations

    @property
    def num_mutations(self):
        """
        Returns the number of :ref:`mutations <sec_mutation_table_definition>`
        in this tree sequence.

        :return: The number of mutations in this tree sequence.
        :rtype: int
        """
        return self._ll_tree_sequence.get_num_mutations()

    def get_num_nodes(self):
        # Deprecated alias for self.num_nodes
        return self.num_nodes

    @property
    def num_individuals(self):
        """
        Returns the number of :ref:`individuals <sec_individual_table_definition>` in
        this tree sequence.

        :return: The number of individuals in this tree sequence.
        :rtype: int
        """
        return self._ll_tree_sequence.get_num_individuals()

    @property
    def num_nodes(self):
        """
        Returns the number of :ref:`nodes <sec_node_table_definition>` in
        this tree sequence.

        :return: The number of nodes in this tree sequence.
        :rtype: int
        """
        return self._ll_tree_sequence.get_num_nodes()

    @property
    def num_provenances(self):
        """
        Returns the number of :ref:`provenances <sec_provenance_table_definition>`
        in this tree sequence.

        :return: The number of provenances in this tree sequence.
        :rtype: int
        """
        return self._ll_tree_sequence.get_num_provenances()

    @property
    def num_populations(self):
        """
        Returns the number of :ref:`populations <sec_population_table_definition>`
        in this tree sequence.

        :return: The number of populations in this tree sequence.
        :rtype: int
        """
        return self._ll_tree_sequence.get_num_populations()

    @property
    def num_migrations(self):
        """
        Returns the number of :ref:`migrations <sec_migration_table_definition>`
        in this tree sequence.

        :return: The number of migrations in this tree sequence.
        :rtype: int
        """
        return self._ll_tree_sequence.get_num_migrations()

    @property
    def max_root_time(self):
        """
        Returns the time of the oldest root in any of the trees in this tree sequence.
        This is usually equal to ``np.max(ts.tables.nodes.time)`` but may not be
        since there can be non-sample nodes that are not present in any tree. Note that
        isolated samples are also defined as roots (so there can be a max_root_time
        even in a tree sequence with no edges).

        :return: The maximum time of a root in this tree sequence.
        :rtype: float
        :raises ValueError: If there are no samples in the tree, and hence no roots (as
            roots are defined by the ends of the upward paths from the set of samples).
        """
        if self.num_samples == 0:
            raise ValueError(
                "max_root_time is not defined in a tree sequence with 0 samples"
            )
        ret = max(self.nodes_time[u] for u in self.samples())
        if self.num_edges > 0:
            # Edges are guaranteed to be listed in parent-time order, so we can get the
            # last one to get the oldest root
            edge = self.edge(self.num_edges - 1)
            # However, we can have situations where there is a sample older than a
            # 'proper' root
            ret = max(ret, self.nodes_time[edge.parent])
        return ret

    def migrations(self):
        """
        Returns an iterable sequence of all the
        :ref:`migrations <sec_migration_table_definition>` in this tree sequence.

        Migrations are returned in nondecreasing order of the ``time`` value.

        :return: An iterable sequence of all migrations.
        :rtype: Sequence(:class:`.Migration`)
        """
        return SimpleContainerSequence(self.migration, self.num_migrations)

    def individuals(self):
        """
        Returns an iterable sequence of all the
        :ref:`individuals <sec_individual_table_definition>` in this tree sequence.

        :return: An iterable sequence of all individuals.
        :rtype: Sequence(:class:`.Individual`)
        """
        return SimpleContainerSequence(self.individual, self.num_individuals)

    def nodes(self, *, order=None):
        """
        Returns an iterable sequence of all the :ref:`nodes <sec_node_table_definition>`
        in this tree sequence.

        .. note:: Although node ids are commonly ordered by node time, this is not a
            formal tree sequence requirement. If you wish to iterate over nodes in
            time order, you should therefore use ``order="timeasc"`` (and wrap the
            resulting sequence in the standard Python :func:`python:reversed` function
            if you wish to iterate over older nodes before younger ones)

        :param str order: The order in which the nodes should be returned: must be
            one of "id" (default) or "timeasc" (ascending order of time, then by
            ascending node id, matching the first two ordering requirements of
            parent nodes in a :meth:`sorted <TableCollection.sort>` edge table).
        :return: An iterable sequence of all nodes.
        :rtype: Sequence(:class:`.Node`)
        """
        order = "id" if order is None else order
        if order not in ["id", "timeasc"]:
            raise ValueError('order must be "id" or "timeasc"')
        odr = None
        if order == "timeasc":
            odr = np.lexsort((np.arange(self.num_nodes), self.nodes_time))
        return SimpleContainerSequence(self.node, self.num_nodes, order=odr)

    def edges(self):
        """
        Returns an iterable sequence of all the :ref:`edges <sec_edge_table_definition>`
        in this tree sequence. Edges are returned in the order required
        for a :ref:`valid tree sequence <sec_valid_tree_sequence_requirements>`. So,
        edges are guaranteed to be ordered such that (a) all parents with a
        given ID are contiguous; (b) edges are returned in non-decreasing
        order of parent time ago; (c) within the edges for a given parent, edges
        are sorted first by child ID and then by left coordinate.

        :return: An iterable sequence of all edges.
        :rtype: Sequence(:class:`.Edge`)
        """
        return SimpleContainerSequence(self.edge, self.num_edges)

    def edgesets(self):
        # TODO the order that these records are returned in is not well specified.
        # Hopefully this does not matter, and we can just state that the ordering
        # should not be depended on.
        children = collections.defaultdict(set)
        active_edgesets = {}
        for (left, right), edges_out, edges_in in self.edge_diffs():
            # Complete and return any edgesets that are affected by this tree
            # transition
            parents = iter(edge.parent for edge in itertools.chain(edges_out, edges_in))
            for parent in parents:
                if parent in active_edgesets:
                    edgeset = active_edgesets.pop(parent)
                    edgeset.right = left
                    edgeset.children = sorted(children[parent])
                    yield edgeset
            for edge in edges_out:
                children[edge.parent].remove(edge.child)
            for edge in edges_in:
                children[edge.parent].add(edge.child)
            # Update the active edgesets
            for edge in itertools.chain(edges_out, edges_in):
                if (
                    len(children[edge.parent]) > 0
                    and edge.parent not in active_edgesets
                ):
                    active_edgesets[edge.parent] = Edgeset(left, right, edge.parent, [])

        for parent in active_edgesets.keys():
            edgeset = active_edgesets[parent]
            edgeset.right = self.sequence_length
            edgeset.children = sorted(children[edgeset.parent])
            yield edgeset

    def _edge_diffs_forward(self, include_terminal=False):
        metadata_decoder = self.table_metadata_schemas.edge.decode_row
        edge_left = self.edges_left
        edge_right = self.edges_right
        sequence_length = self.sequence_length
        in_order = self.indexes_edge_insertion_order
        out_order = self.indexes_edge_removal_order
        M = self.num_edges
        j = 0
        k = 0
        left = 0.0
        while j < M or left < sequence_length:
            edges_out = []
            edges_in = []
            while k < M and edge_right[out_order[k]] == left:
                edges_out.append(
                    Edge(
                        *self._ll_tree_sequence.get_edge(out_order[k]),
                        id=out_order[k],
                        metadata_decoder=metadata_decoder,
                    )
                )
                k += 1
            while j < M and edge_left[in_order[j]] == left:
                edges_in.append(
                    Edge(
                        *self._ll_tree_sequence.get_edge(in_order[j]),
                        id=in_order[j],
                        metadata_decoder=metadata_decoder,
                    )
                )
                j += 1
            right = sequence_length
            if j < M:
                right = min(right, edge_left[in_order[j]])
            if k < M:
                right = min(right, edge_right[out_order[k]])
            yield EdgeDiff(Interval(left, right), edges_out, edges_in)
            left = right

        if include_terminal:
            edges_out = []
            while k < M:
                edges_out.append(
                    Edge(
                        *self._ll_tree_sequence.get_edge(out_order[k]),
                        id=out_order[k],
                        metadata_decoder=metadata_decoder,
                    )
                )
                k += 1
            yield EdgeDiff(Interval(left, right), edges_out, [])

    def _edge_diffs_reverse(self, include_terminal=False):
        metadata_decoder = self.table_metadata_schemas.edge.decode_row
        edge_left = self.edges_left
        edge_right = self.edges_right
        sequence_length = self.sequence_length
        in_order = self.indexes_edge_removal_order
        out_order = self.indexes_edge_insertion_order
        M = self.num_edges
        j = M - 1
        k = M - 1
        right = sequence_length
        while j >= 0 or right > 0:
            edges_out = []
            edges_in = []
            while k >= 0 and edge_left[out_order[k]] == right:
                edges_out.append(
                    Edge(
                        *self._ll_tree_sequence.get_edge(out_order[k]),
                        id=out_order[k],
                        metadata_decoder=metadata_decoder,
                    )
                )
                k -= 1
            while j >= 0 and edge_right[in_order[j]] == right:
                edges_in.append(
                    Edge(
                        *self._ll_tree_sequence.get_edge(in_order[j]),
                        id=in_order[j],
                        metadata_decoder=metadata_decoder,
                    )
                )
                j -= 1
            left = 0
            if j >= 0:
                left = max(left, edge_right[in_order[j]])
            if k >= 0:
                left = max(left, edge_left[out_order[k]])
            yield EdgeDiff(Interval(left, right), edges_out, edges_in)
            right = left

        if include_terminal:
            edges_out = []
            while k >= 0:
                edges_out.append(
                    Edge(
                        *self._ll_tree_sequence.get_edge(out_order[k]),
                        id=out_order[k],
                        metadata_decoder=metadata_decoder,
                    )
                )
                k -= 1
            yield EdgeDiff(Interval(left, right), edges_out, [])

    def edge_diffs(self, include_terminal=False, *, direction=tskit.FORWARD):
        """
        Returns an iterator over all the :ref:`edges <sec_edge_table_definition>` that
        are inserted and removed to build the trees as we move from left-to-right along
        the tree sequence. Each iteration yields a named tuple consisting of 3 values,
        ``(interval, edges_out, edges_in)``. The first value, ``interval``, is the
        genomic interval ``(left, right)`` covered by the incoming tree
        (see :attr:`Tree.interval`). The second, ``edges_out`` is a list of the edges
        that were just-removed to create the tree covering the interval
        (hence ``edges_out`` will always be empty for the first tree). The last value,
        ``edges_in``, is a list of edges that were just
        inserted to construct the tree covering the current interval.

        The edges returned within each ``edges_in`` list are ordered by ascending
        time of the parent node, then ascending parent id, then ascending child id.
        The edges within each ``edges_out`` list are the reverse order (e.g.
        descending parent time, parent id, then child_id). This means that within
        each list, edges with the same parent appear consecutively.

        The ``direction`` argument can be used to control whether diffs are produced
        in the forward (left-to-right, increasing genome coordinate value)
        or reverse (right-to-left, decreasing genome coordinate value) direction.

        :param bool include_terminal: If False (default), the iterator terminates
            after the final interval in the tree sequence (i.e., it does not
            report a final removal of all remaining edges), and the number
            of iterations will be equal to the number of trees in the tree
            sequence. If True, an additional iteration takes place, with the last
            ``edges_out`` value reporting all the edges contained in the final
            tree (with both ``left`` and ``right`` equal to the sequence length).
        :param int direction: The direction of travel along the sequence for
            diffs. Must be one of :data:`.FORWARD` or :data:`.REVERSE`.
            (Default: :data:`.FORWARD`).
        :return: An iterator over the (interval, edges_out, edges_in) tuples. This
            is a named tuple, so the 3 values can be accessed by position
            (e.g. ``returned_tuple[0]``) or name (e.g. ``returned_tuple.interval``).
        :rtype: :class:`collections.abc.Iterable`
        """
        if direction == _tskit.FORWARD:
            return self._edge_diffs_forward(include_terminal=include_terminal)
        elif direction == _tskit.REVERSE:
            return self._edge_diffs_reverse(include_terminal=include_terminal)
        else:
            raise ValueError("direction must be either tskit.FORWARD or tskit.REVERSE")

    def sites(self):
        """
        Returns an iterable sequence of all the :ref:`sites <sec_site_table_definition>`
        in this tree sequence. Sites are returned in order of increasing ID
        (and also position). See the :class:`Site` class for details on
        the available fields for each site.

        :return: An iterable sequence of all sites.
        :rtype: Sequence(:class:`.Site`)
        """
        return SimpleContainerSequence(self.site, self.num_sites)

    def mutations(self):
        """
        Returns an iterator over all the
        :ref:`mutations <sec_mutation_table_definition>` in this tree sequence.
        Mutations are returned in order of nondecreasing site ID.
        See the :class:`Mutation` class for details on the available fields for
        each mutation.

        The returned iterator is equivalent to iterating over all sites
        and all mutations in each site, i.e.::

            for site in tree_sequence.sites():
                for mutation in site.mutations:
                    yield mutation

        :return: An iterator over all mutations in this tree sequence.
        :rtype: iter(:class:`Mutation`)
        """
        for site in self.sites():
            yield from site.mutations

    def populations(self):
        """
        Returns an iterable sequence of all the
        :ref:`populations <sec_population_table_definition>` in this tree sequence.

        :return: An iterable sequence of all populations.
        :rtype: Sequence(:class:`.Population`)
        """
        return SimpleContainerSequence(self.population, self.num_populations)

    def provenances(self):
        """
        Returns an iterable sequence of all the
        :ref:`provenances <sec_provenance_table_definition>` in this tree sequence.

        :return: An iterable sequence of all provenances.
        :rtype: Sequence(:class:`.Provenance`)
        """
        return SimpleContainerSequence(self.provenance, self.num_provenances)

    def breakpoints(self, as_array=False):
        """
        Returns the breakpoints that separate trees along the chromosome, including the
        two extreme points 0 and L. This is equivalent to::

            iter([0] + [t.interval.right for t in self.trees()])

        By default we return an iterator over the breakpoints as Python float objects;
        if ``as_array`` is True we return them as a numpy array.

        Note that the ``as_array`` form will be more efficient and convenient in most
        cases; the default iterator behaviour is mainly kept to ensure compatibility
        with existing code.

        :param bool as_array: If True, return the breakpoints as a numpy array.
        :return: The breakpoints defined by the tree intervals along the sequence.
        :rtype: collections.abc.Iterable or numpy.ndarray
        """
        breakpoints = self.ll_tree_sequence.get_breakpoints()
        if not as_array:
            # Convert to Python floats for backward compatibility.
            breakpoints = map(float, breakpoints)
        return breakpoints

    def at(self, position, **kwargs):
        """
        Returns the tree covering the specified genomic location. The returned tree
        will have ``tree.interval.left`` <= ``position`` < ``tree.interval.right``.
        See also :meth:`Tree.seek`.

        .. include:: substitutions/linear_traversal_warning.rst

        :param float position: A genomic location.
        :param \\**kwargs: Further arguments used as parameters when constructing the
            returned :class:`Tree`. For example ``ts.at(2.5, sample_lists=True)`` will
            result in a :class:`Tree` created with ``sample_lists=True``.
        :return: A new instance of :class:`Tree` positioned to cover the specified
            genomic location.
        :rtype: Tree
        """
        tree = Tree(self, **kwargs)
        tree.seek(position)
        return tree

    def at_index(self, index, **kwargs):
        """
        Returns the tree at the specified index. See also :meth:`Tree.seek_index`.

        .. include:: substitutions/linear_traversal_warning.rst

        :param int index: The index of the required tree.
        :param \\**kwargs: Further arguments used as parameters when constructing the
            returned :class:`Tree`. For example ``ts.at_index(4, sample_lists=True)``
            will result in a :class:`Tree` created with ``sample_lists=True``.
        :return: A new instance of :class:`Tree` positioned at the specified index.
        :rtype: Tree
        """
        tree = Tree(self, **kwargs)
        tree.seek_index(index)
        return tree

    def first(self, **kwargs):
        """
        Returns the first tree in this :class:`TreeSequence`. To iterate over all
        trees in the sequence, use the :meth:`.trees` method.

        :param \\**kwargs: Further arguments used as parameters when constructing the
            returned :class:`Tree`. For example ``ts.first(sample_lists=True)`` will
            result in a :class:`Tree` created with ``sample_lists=True``.
        :return: The first tree in this tree sequence.
        :rtype: :class:`Tree`.
        """
        tree = Tree(self, **kwargs)
        tree.first()
        return tree

    def last(self, **kwargs):
        """
        Returns the last tree in this :class:`TreeSequence`. To iterate over all
        trees in the sequence, use the :meth:`.trees` method.

        :param \\**kwargs: Further arguments used as parameters when constructing the
            returned :class:`Tree`. For example ``ts.first(sample_lists=True)`` will
            result in a :class:`Tree` created with ``sample_lists=True``.
        :return: The last tree in this tree sequence.
        :rtype: :class:`Tree`.
        """
        tree = Tree(self, **kwargs)
        tree.last()
        return tree

    def trees(
        self,
        tracked_samples=None,
        *,
        sample_lists=False,
        root_threshold=1,
        sample_counts=None,
        tracked_leaves=None,
        leaf_counts=None,
        leaf_lists=None,
    ):
        """
        Returns an iterator over the trees in this tree sequence. Each value
        returned in this iterator is an instance of :class:`Tree`. Upon
        successful termination of the iterator, the tree will be in the
        "cleared" null state.

        The ``sample_lists`` and ``tracked_samples`` parameters are passed
        to the :class:`Tree` constructor, and control
        the options that are set in the returned tree instance.

        .. warning:: Do not store the results of this iterator in a list!
           For performance reasons, the same underlying object is used
           for every tree returned which will most likely lead to unexpected
           behaviour. If you wish to obtain a list of trees in a tree sequence
           please use ``ts.aslist()`` instead.

        :param list tracked_samples: The list of samples to be tracked and
            counted using the :meth:`Tree.num_tracked_samples` method.
        :param bool sample_lists: If True, provide more efficient access
            to the samples beneath a given node using the
            :meth:`Tree.samples` method.
        :param int root_threshold: The minimum number of samples that a node
            must be ancestral to for it to be in the list of roots. By default
            this is 1, so that isolated samples (representing missing data)
            are roots. To efficiently restrict the roots of the tree to
            those subtending meaningful topology, set this to 2. This value
            is only relevant when trees have multiple roots.
        :param bool sample_counts: Deprecated since 0.2.4.
        :return: An iterator over the Trees in this tree sequence.
        :rtype: collections.abc.Iterable, :class:`Tree`
        """
        # tracked_leaves, leaf_counts and leaf_lists are deprecated aliases
        # for tracked_samples, sample_counts and sample_lists respectively.
        # These are left over from an older version of the API when leaves
        # and samples were synonymous.
        if tracked_leaves is not None:
            tracked_samples = tracked_leaves
        if leaf_counts is not None:
            sample_counts = leaf_counts
        if leaf_lists is not None:
            sample_lists = leaf_lists
        tree = Tree(
            self,
            tracked_samples=tracked_samples,
            sample_lists=sample_lists,
            root_threshold=root_threshold,
            sample_counts=sample_counts,
        )
        return TreeIterator(tree)

    def coiterate(self, other, **kwargs):
        """
        Returns an iterator over the pairs of trees for each distinct
        interval in the specified pair of tree sequences.

        :param TreeSequence other: The other tree sequence from which to take trees. The
            sequence length must be the same as the current tree sequence.
        :param \\**kwargs: Further named arguments that will be passed to the
            :meth:`.trees` method when constructing the returned trees.

        :return: An iterator returning successive tuples of the form
            ``(interval, tree_self, tree_other)``. For example, the first item returned
            will consist of an tuple of the initial interval, the first tree of the
            current tree sequence, and the first tree of the ``other`` tree sequence;
            the ``.left`` attribute of the initial interval will be 0 and the ``.right``
            attribute will be the smallest non-zero breakpoint of the 2 tree sequences.
        :rtype: iter(:class:`Interval`, :class:`Tree`, :class:`Tree`)

        """
        if self.sequence_length != other.sequence_length:
            raise ValueError("Tree sequences must be of equal sequence length.")
        L = self.sequence_length
        trees1 = self.trees(**kwargs)
        trees2 = other.trees(**kwargs)
        tree1 = next(trees1)
        tree2 = next(trees2)
        right = 0
        while right != L:
            left = right
            right = min(tree1.interval.right, tree2.interval.right)
            yield Interval(left, right), tree1, tree2
            # Advance
            if tree1.interval.right == right:
                tree1 = next(trees1, None)
            if tree2.interval.right == right:
                tree2 = next(trees2, None)

    def _check_genomic_range(self, left, right, ensure_integer=False):
        if left is None:
            left = 0
        if right is None:
            right = self.sequence_length
        if np.isnan(left) or left < 0 or left >= self.sequence_length:
            raise ValueError(
                "`left` not between zero (inclusive) and sequence length (exclusive)"
            )
        if np.isnan(right) or right <= 0 or right > self.sequence_length:
            raise ValueError(
                "`right` not between zero (exclusive) and sequence length (inclusive)"
            )
        if left >= right:
            raise ValueError("`left` must be less than `right`")
        if ensure_integer:
            if left != int(left) or right != int(right):
                raise ValueError("`left` and `right` must be integers")
            return Interval(int(left), int(right))
        return Interval(left, right)

    def _haplotypes_array(
        self,
        *,
        interval,
        isolated_as_missing=None,
        missing_data_character=None,
        samples=None,
    ):
        # return an array of haplotypes and the first and last site positions
        if missing_data_character is None:
            missing_data_character = "N"

        start_site, stop_site = np.searchsorted(self.sites_position, interval)
        H = np.empty(
            (
                self.num_samples if samples is None else len(samples),
                stop_site - start_site,
            ),
            dtype=np.int8,
        )
        missing_int8 = ord(missing_data_character.encode("ascii"))
        for var in self.variants(
            samples=samples,
            isolated_as_missing=isolated_as_missing,
            left=interval.left,
            right=interval.right,
        ):
            alleles = np.full(len(var.alleles), missing_int8, dtype=np.int8)
            for i, allele in enumerate(var.alleles):
                if allele is not None:
                    if len(allele) != 1:
                        raise TypeError(
                            "Multi-letter allele or deletion detected at site {}".format(
                                var.site.id
                            )
                        )
                    try:
                        ascii_allele = allele.encode("ascii")
                    except UnicodeEncodeError:
                        raise TypeError(
                            "Non-ascii character in allele at site {}".format(
                                var.site.id
                            )
                        )
                    allele_int8 = ord(ascii_allele)
                    if allele_int8 == missing_int8:
                        raise ValueError(
                            "The missing data character '{}' clashes with an "
                            "existing allele at site {}".format(
                                missing_data_character, var.site.id
                            )
                        )
                    alleles[i] = allele_int8
            H[:, var.site.id - start_site] = alleles[var.genotypes]
        return H, (start_site, stop_site - 1)

    def haplotypes(
        self,
        *,
        isolated_as_missing=None,
        missing_data_character=None,
        samples=None,
        left=None,
        right=None,
        impute_missing_data=None,
    ):
        """
        Returns an iterator over the strings of haplotypes that result from
        the trees and mutations in this tree sequence. Each haplotype string
        is guaranteed to be of the same length. A tree sequence with
        :math:`n` samples and with :math:`s` sites lying between ``left`` and
        ``right`` will return a total of :math:`n`
        strings of :math:`s` alleles concatenated together, where an allele
        consists of a single ascii character (tree sequences that include alleles
        which are not a single character in length, or where the character is
        non-ascii, will raise an error). The first string returned is the
        haplotype for the first requested sample, and so on.

        The alleles at each site must be represented by single byte characters,
        (i.e., variants must be single nucleotide polymorphisms, or SNPs), hence
        the strings returned will all be of length :math:`s`. If the ``left``
        position is less than or equal to the position of the first site, for a
        haplotype ``h``, the value of ``h[j]`` will therefore be the observed
        allelic state at site ``j``.

        If ``isolated_as_missing`` is True (the default), isolated samples without
        mutations directly above them will be treated as
        :ref:`missing data<sec_data_model_missing_data>` and will be
        represented in the string by the ``missing_data_character``. If
        instead it is set to False, missing data will be assigned the ancestral state
        (unless they have mutations directly above them, in which case they will take
        the most recent derived mutational state for that node). This was the default
        behaviour in versions prior to 0.2.0. Prior to 0.3.0 the `impute_missing_data`
        argument controlled this behaviour.

        See also the :meth:`.variants` iterator for site-centric access
        to sample genotypes.

        .. warning::
            For large datasets, this method can consume a **very large** amount of
            memory! To output all the sample data, it is more efficient to iterate
            over sites rather than over samples.

        :return: An iterator over the haplotype strings for the samples in
            this tree sequence.
        :param bool isolated_as_missing: If True, the allele assigned to
            missing samples (i.e., isolated samples without mutations) is
            the ``missing_data_character``. If False,
            missing samples will be assigned the ancestral state.
            Default: True.
        :param str missing_data_character: A single ascii character that will
            be used to represent missing data.
            If any normal allele contains this character, an error is raised.
            Default: 'N'.
        :param list[int] samples: The samples for which to output haplotypes. If
            ``None`` (default), return haplotypes for all the samples in the tree
            sequence, in the order given by the :meth:`.samples` method.
        :param int left: Haplotype strings will start with the first site at or after
            this genomic position. If ``None`` (default) start at the first site.
        :param int right: Haplotype strings will end with the last site before this
            position. If ``None`` (default) assume ``right`` is the sequence length
            (i.e. the last character in the string will be the last site in the tree
            sequence).
        :param bool impute_missing_data:
            *Deprecated in 0.3.0. Use ``isolated_as_missing``, but inverting value.
            Will be removed in a future version*
        :rtype: collections.abc.Iterable
        :raises TypeError: if the ``missing_data_character`` or any of the alleles
            at a site are not a single ascii character.
        :raises ValueError: if the ``missing_data_character`` exists in one of the
            alleles
        """
        if impute_missing_data is not None:
            warnings.warn(
                "The impute_missing_data parameter was deprecated in 0.3.0 and will"
                " be removed. Use ``isolated_as_missing=False`` instead of"
                "``impute_missing_data=True``.",
                FutureWarning,
                stacklevel=4,
            )
        # Only use impute_missing_data if isolated_as_missing has the default value
        if isolated_as_missing is None:
            isolated_as_missing = not impute_missing_data
        interval = self._check_genomic_range(left, right)
        H, _ = self._haplotypes_array(
            interval=interval,
            isolated_as_missing=isolated_as_missing,
            missing_data_character=missing_data_character,
            samples=samples,
        )
        for h in H:
            yield h.tobytes().decode("ascii")

    def variants(
        self,
        *,
        samples=None,
        isolated_as_missing=None,
        alleles=None,
        impute_missing_data=None,
        copy=None,
        left=None,
        right=None,
    ):
        """
        Returns an iterator over the variants between the ``left`` (inclusive)
        and ``right`` (exclusive) genomic positions in this tree sequence. Each
        returned :class:`Variant` object has a site, a list of possible allelic
        states and an array of genotypes for the specified ``samples``. The
        ``genotypes`` value is a numpy array containing indexes into the
        ``alleles`` list. By default, this list is generated automatically for
        each site such that the first entry, ``alleles[0]``, is the ancestral
        state and subsequent alleles are listed in no
        particular order. This means that the encoding of alleles in
        terms of genotype values can vary from site-to-site, which is
        sometimes inconvenient. It is possible to specify a fixed mapping
        from allele strings to genotype values using the ``alleles``
        parameter. For example, if we set ``alleles=("A", "C", "G", "T")``,
        this will map allele "A" to 0, "C" to 1 and so on (the
        :data:`ALLELES_ACGT` constant provides a shortcut for this
        common mapping).

        By default, genotypes are generated for all samples. The ``samples``
        parameter allows us to specify the nodes for which genotypes are
        generated; output order of genotypes in the returned variants
        corresponds to the order of the samples in this list. It is also
        possible to provide **non-sample** nodes as an argument here, if you
        wish to generate genotypes for (e.g.) internal nodes. However,
        ``isolated_as_missing`` must be False in this case, as it is not
        possible to detect missing data for non-sample nodes.

        If isolated samples are present at a given site without mutations above them,
        they are interpreted by default as
        :ref:`missing data<sec_data_model_missing_data>`, and the genotypes array
        will contain a special value :data:`MISSING_DATA` (-1) to identify them
        while the ``alleles`` tuple will end with the value ``None`` (note that this
        will be the case whether or not we specify a fixed mapping using the
        ``alleles`` parameter; see the :class:`Variant` class for more details).
        Alternatively, if ``isolated_as_missing`` is set to to False, such isolated
        samples will not be treated as missing, and instead assigned the ancestral
        state (this was the default behaviour in versions prior to 0.2.0). Prior to
        0.3.0 the `impute_missing_data` argument controlled this behaviour.

        :param array_like samples: An array of node IDs for which to generate
            genotypes, or None for all sample nodes. Default: None.
        :param bool isolated_as_missing: If True, the genotype value assigned to
            missing samples (i.e., isolated samples without mutations) is
            :data:`.MISSING_DATA` (-1). If False, missing samples will be
            assigned the allele index for the ancestral state.
            Default: True.
        :param tuple alleles: A tuple of strings defining the encoding of
            alleles as integer genotype values. At least one allele must be provided.
            If duplicate alleles are provided, output genotypes will always be
            encoded as the first occurrence of the allele. If None (the default),
            the alleles are encoded as they are encountered during genotype
            generation.
        :param bool impute_missing_data:
            *Deprecated in 0.3.0. Use ``isolated_as_missing``, but inverting value.
            Will be removed in a future version*
        :param bool copy:
            If False re-use the same Variant object for each site such that any
            references held to it are overwritten when the next site is visited.
            If True return a fresh :class:`Variant` for each site. Default: True.
        :param int left: Start with the first site at or after
            this genomic position. If ``None`` (default) start at the first site.
        :param int right: End with the last site before this position. If ``None``
            (default) assume ``right`` is the sequence length, so that the last
            variant corresponds to the last site in the tree sequence.
        :return: An iterator over all variants in this tree sequence.
        :rtype: iter(:class:`Variant`)
        """
        interval = self._check_genomic_range(left, right)
        if impute_missing_data is not None:
            warnings.warn(
                "The impute_missing_data parameter was deprecated in 0.3.0 and will"
                " be removed. Use ``isolated_as_missing=False`` instead of"
                "``impute_missing_data=True``.",
                FutureWarning,
                stacklevel=4,
            )
        # Only use impute_missing_data if isolated_as_missing has the default value
        if isolated_as_missing is None:
            isolated_as_missing = not impute_missing_data
        if copy is None:
            copy = True
        # See comments for the Variant type for discussion on why the
        # present form was chosen.
        variant = tskit.Variant(
            self,
            samples=samples,
            isolated_as_missing=isolated_as_missing,
            alleles=alleles,
        )
        if left == 0 and right == self.sequence_length:
            start = 0
            stop = self.num_sites
        else:
            start, stop = np.searchsorted(self.sites_position, interval)

        if copy:
            for site_id in range(start, stop):
                variant.decode(site_id)
                yield variant.copy()
        else:
            for site_id in range(start, stop):
                variant.decode(site_id)
                yield variant

    def genotype_matrix(
        self,
        *,
        samples=None,
        isolated_as_missing=None,
        alleles=None,
        impute_missing_data=None,
    ):
        """
        Returns an :math:`m \\times n` numpy array of the genotypes in this
        tree sequence, where :math:`m` is the number of sites and :math:`n`
        the number of samples. The genotypes are the indexes into the array
        of ``alleles``, as described for the :class:`Variant` class.

        If isolated samples are present at a given site without mutations above them,
        they will be interpreted as :ref:`missing data<sec_data_model_missing_data>`
        the genotypes array will contain a special value :data:`MISSING_DATA`
        (-1) to identify these missing samples.

        Such samples are treated as missing data by default, but if
        ``isolated_as_missing`` is set to to False, they will not be treated as missing,
        and so assigned the ancestral state. This was the default behaviour in
        versions prior to 0.2.0. Prior to 0.3.0 the `impute_missing_data`
        argument controlled this behaviour.

        .. warning::
            This method can consume a **very large** amount of memory! If
            all genotypes are not needed at once, it is usually better to
            access them sequentially using the :meth:`.variants` iterator.

        :param array_like samples: An array of node IDs for which to generate
            genotypes, or None for all sample nodes. Default: None.
        :param bool isolated_as_missing: If True, the genotype value assigned to
            missing samples (i.e., isolated samples without mutations) is
            :data:`.MISSING_DATA` (-1). If False, missing samples will be
            assigned the allele index for the ancestral state.
            Default: True.
        :param tuple alleles: A tuple of strings describing the encoding of
            alleles to genotype values. At least one allele must be provided.
            If duplicate alleles are provided, output genotypes will always be
            encoded as the first occurrence of the allele. If None (the default),
            the alleles are encoded as they are encountered during genotype
            generation.
        :param bool impute_missing_data:
            *Deprecated in 0.3.0. Use ``isolated_as_missing``, but inverting value.
            Will be removed in a future version*

        :return: The full matrix of genotypes.
        :rtype: numpy.ndarray (dtype=np.int32)
        """
        if impute_missing_data is not None:
            warnings.warn(
                "The impute_missing_data parameter was deprecated in 0.3.0 and will"
                " be removed. Use ``isolated_as_missing=False`` instead of"
                "``impute_missing_data=True``.",
                FutureWarning,
                stacklevel=4,
            )
        # Only use impute_missing_data if isolated_as_missing has the default value
        if isolated_as_missing is None:
            isolated_as_missing = not impute_missing_data

        variant = tskit.Variant(
            self,
            samples=samples,
            isolated_as_missing=isolated_as_missing,
            alleles=alleles,
        )

        num_samples = self.num_samples if samples is None else len(samples)
        ret = np.zeros(shape=(self.num_sites, num_samples), dtype=np.int32)

        for site_id in range(self.num_sites):
            variant.decode(site_id)
            ret[site_id, :] = variant.genotypes

        return ret

    def alignments(
        self,
        *,
        reference_sequence=None,
        missing_data_character=None,
        samples=None,
        left=None,
        right=None,
    ):
        """
        Returns an iterator over the full sequence alignments for the defined samples
        in this tree sequence. Each alignment ``a`` is a string of length ``L`` where
        the first character is the genomic sequence at the ``start`` position in the
        genome (defaulting to 0) and the last character is the genomic sequence one
        position before the ``stop`` value (defaulting to the :attr:`.sequence_length`
        of this tree sequence, which must have :attr:`.discrete_genome` equal to True).
        By default ``L`` is therefore equal to the :attr:`.sequence_length`,
        and ``a[j]`` is the nucleotide value at genomic position ``j``.

        .. note:: This is inherently a **zero-based** representation of the sequence
            coordinate space. Care will be needed when interacting with other
            libraries and upstream coordinate spaces.


        The :ref:`sites<sec_data_model_definitions_site>` in a tree sequence will
        usually only define the variation for a subset of the ``L`` nucleotide
        positions along the genome, and the remaining positions are filled using
        a :ref:`reference sequence <sec_data_model_reference_sequence>`.
        The reference sequence data is defined either via the
        ``reference_sequence`` parameter to this method, or embedded within
        with the tree sequence itself via the :attr:`.TreeSequence.reference_sequence`.

        Site information from the tree sequence takes precedence over the reference
        sequence so that, for example, at a site with no mutations all samples
        will have the site's ancestral state.

        The reference sequence bases are determined in the following way:

        - If the ``reference_sequence`` parameter is supplied this will be
          used, regardless of whether the tree sequence has an embedded
          reference sequence.
        - Otherwise, if the tree sequence has an embedded reference sequence,
          this will be used.
        - If the ``reference_sequence`` parameter is not specified and
          there is no embedded reference sequence, ``L`` copies of the
          ``missing_data_character`` (which defaults to 'N') are used
          instead.

        .. warning:: The :class:`.ReferenceSequence` API is preliminary and
           some behaviours may change in the future. In particular, a
           tree sequence is currently regarded as having an embedded reference
           sequence even if it only has some metadata defined. In this case
           the ``reference_sequence`` parameter will need to be explicitly set.

        .. note::
            Two common options for setting a reference sequence are:

            - Mark them as missing data, by setting
              ``reference_sequence="N" * int(ts.sequence_length)``
            - Fill the gaps with random nucleotides, by setting
              ``reference_sequence=tskit.random_nucleotides(ts.sequence_length)``.
              See the :func:`.random_nucleotides` function for more information.

        .. warning:: Insertions and deletions are not currently supported and
           the alleles at each site must be represented by
           single byte characters, (i.e., variants must be single nucleotide
           polymorphisms, or SNPs).

        .. warning:: :ref:`Missing data<sec_data_model_missing_data>` is not
           currently supported by this method and it will raise a ValueError
           if called on tree sequences containing isolated samples.
           See https://github.com/tskit-dev/tskit/issues/1896 for more
           information.

        See also the :meth:`.variants` iterator for site-centric access
        to sample genotypes and :meth:`.haplotypes` for access to sample sequences
        at just the sites in the tree sequence.

        :param str reference_sequence: The reference sequence to fill in
            gaps between sites in the alignments.
        :param str missing_data_character: A single ascii character that will
            be used to represent missing data.
            If any normal allele contains this character, an error is raised.
            Default: 'N'.
        :param list[int] samples: The samples for which to output alignments. If
            ``None`` (default), return alignments for all the samples in the tree
            sequence, in the order given by the :meth:`.samples` method.
        :param int left: Alignments will start at this genomic position. If ``None``
            (default) alignments start at 0.
        :param int right: Alignments will stop before this genomic position. If ``None``
            (default) alignments will continue until the end of the tree sequence.
        :return: An iterator over the alignment strings for specified samples in
            this tree sequence, in the order given in ``samples``.
        :rtype: collections.abc.Iterable
        :raises ValueError: if any genome coordinate in this tree sequence is not
            discrete, or if the ``reference_sequence`` is not of the correct length.
        :raises TypeError: if any of the alleles at a site are not a
            single ascii character.
        """
        if not self.discrete_genome:
            raise ValueError("sequence alignments only defined for discrete genomes")
        interval = self._check_genomic_range(left, right, ensure_integer=True)
        missing_data_character = (
            "N" if missing_data_character is None else missing_data_character
        )

        L = interval.span
        a = np.empty(L, dtype=np.int8)
        if reference_sequence is None:
            if self.has_reference_sequence():
                # This may be inefficient - see #1989. However, since we're
                # n copies of the reference sequence anyway, this is a relatively
                # minor tweak. We may also want to recode the below not to use direct
                # access to the .data attribute, e.g. if we allow reference sequences
                # to start at non-zero positions
                reference_sequence = self.reference_sequence.data[
                    interval.left : interval.right
                ]
            else:
                reference_sequence = missing_data_character * L

        if len(reference_sequence) != L:
            if interval.right == int(self.sequence_length):
                raise ValueError(
                    "The reference sequence is shorter than the tree sequence length"
                )
            else:
                raise ValueError(
                    "The reference sequence ends before the requested stop position"
                )
        ref_bytes = reference_sequence.encode("ascii")
        a[:] = np.frombuffer(ref_bytes, dtype=np.int8)

        # To do this properly we'll have to detect the missing data as
        # part of a full implementation of alignments in C. The current
        # definition might not be calling some degenerate cases correctly;
        # see https://github.com/tskit-dev/tskit/issues/1908
        #
        # Note also that this will call the presence of missing data
        # incorrectly if have a sample isolated over the region (a, b],
        # and if we have sites at each position from a to b, and at
        # each site there is a mutation over the isolated sample.
        if any(tree._has_isolated_samples() for tree in self.trees()):
            raise ValueError(
                "Missing data not currently supported in alignments; see "
                "https://github.com/tskit-dev/tskit/issues/1896 for details."
                "The current implementation may also incorrectly identify an "
                "input tree sequence has having missing data."
            )
        H, (first_site_id, last_site_id) = self._haplotypes_array(
            interval=interval,
            missing_data_character=missing_data_character,
            samples=samples,
        )
        site_pos = self.sites_position.astype(np.int64)[
            first_site_id : last_site_id + 1
        ]
        for h in H:
            a[site_pos - interval.left] = h
            yield a.tobytes().decode("ascii")

    @property
    def individuals_population(self):
        """
        Returns the length-``num_individuals`` array containing, for each
        individual, the ``population`` attribute of their nodes, or
        ``tskit.NULL`` for individuals with no nodes. Errors if any individual
        has nodes with inconsistent non-NULL populations.
        """
        if self._individuals_population is None:
            self._individuals_population = (
                self._ll_tree_sequence.get_individuals_population()
            )
        return self._individuals_population

    @property
    def individual_populations(self):
        # Undocumented alias for individuals_population to avoid breaking
        # pre-1.0 pyslim code
        return self.individuals_population

    @property
    def individuals_time(self):
        """
        Returns the length-``num_individuals`` array containing, for each
        individual, the ``time`` attribute of their nodes or ``np.nan`` for
        individuals with no nodes. Errors if any individual has nodes with
        inconsistent times.
        """
        if self._individuals_time is None:
            self._individuals_time = self._ll_tree_sequence.get_individuals_time()
        return self._individuals_time

    @property
    def individual_times(self):
        # Undocumented alias for individuals_time to avoid breaking
        # pre-1.0 pyslim code
        return self.individuals_time

    @property
    def individuals_location(self):
        """
        Convenience method returning the ``num_individuals x n`` array
        whose row k-th row contains the ``location`` property of the k-th
        individual. The method only works if all individuals' locations
        have the same length (which is ``n``), and errors otherwise.
        """
        if self._individuals_location is None:
            individuals = self.tables.individuals
            n = 0
            lens = np.unique(np.diff(individuals.location_offset))
            if len(lens) > 1:
                raise ValueError("Individual locations are not all the same length.")
            if len(lens) > 0:
                n = lens[0]
            self._individuals_location = individuals.location.reshape(
                (self.num_individuals, n)
            )
        return self._individuals_location

    @property
    def individual_locations(self):
        # Undocumented alias for individuals_time to avoid breaking
        # pre-1.0 pyslim code
        return self.individuals_location

    @property
    def individuals_flags(self):
        """
        Efficient access to the bitwise ``flags`` column in the
        :ref:`sec_individual_table_definition` as a numpy array (dtype=np.uint32).
        Equivalent to ``ts.tables.individuals.flags`` (but avoiding the full copy
        of the table data that accessing ``ts.tables`` currently entails).
        """
        return self._individuals_flags

    @property
    def individuals_metadata(self):
        """
        Efficient access to the ``metadata`` column in the
        :ref:`sec_individual_table_definition` as a structured numpy array.
        The returned dtype will depend on the metadata schema used. Only a subset
        of `struct` metadata schemas are supported.
        See :ref:`sec_structured_array_metadata` for more information.
        """
        return self.table_metadata_schemas.individual.structured_array_from_buffer(
            self._individuals_metadata
        )

    @property
    def individuals_nodes(self):
        """
        Return an array of node IDs for each individual in the tree sequence.

        :return: Array of shape (num_individuals, max_ploidy) containing node IDs.
            Values of -1 indicate unused slots for individuals with ploidy
            less than the maximum.
        :rtype: numpy.ndarray (dtype=np.int32)
        """
        if self._individuals_nodes is None:
            self._individuals_nodes = self._ll_tree_sequence.get_individuals_nodes()
        return self._individuals_nodes

    @property
    def nodes_metadata(self):
        """
        Efficient access to the ``metadata`` column in the
        :ref:`sec_node_table_definition` as a structured numpy array.
        The returned dtype will depend on the metadata schema used. Only a subset
        of `struct` metadata schemas are supported.
        See :ref:`sec_structured_array_metadata` for more information.
        """
        return self.table_metadata_schemas.node.structured_array_from_buffer(
            self._nodes_metadata
        )

    @property
    def nodes_time(self):
        """
        Efficient access to the ``time`` column in the
        :ref:`sec_node_table_definition` as a numpy array (dtype=np.float64).
        Equivalent to ``ts.tables.nodes.time`` (but avoiding the full copy
        of the table data that accessing ``ts.tables`` currently entails).
        """
        return self._nodes_time

    @property
    def nodes_flags(self):
        """
        Efficient access to the bitwise ``flags`` column in the
        :ref:`sec_node_table_definition` as a numpy array (dtype=np.uint32).
        Equivalent to ``ts.tables.nodes.flags`` (but avoiding the full copy
        of the table data that accessing ``ts.tables`` currently entails).
        """
        return self._nodes_flags

    @property
    def nodes_population(self):
        """
        Efficient access to the ``population`` column in the
        :ref:`sec_node_table_definition` as a numpy array (dtype=np.int32).
        Equivalent to ``ts.tables.nodes.population`` (but avoiding the full copy
        of the table data that accessing ``ts.tables`` currently entails).
        """
        return self._nodes_population

    @property
    def nodes_individual(self):
        """
        Efficient access to the ``individual`` column in the
        :ref:`sec_node_table_definition` as a numpy array (dtype=np.int32).
        Equivalent to ``ts.tables.nodes.individual`` (but avoiding the full copy
        of the table data that accessing ``ts.tables`` currently entails).
        """
        return self._nodes_individual

    @property
    def edges_left(self):
        """
        Efficient access to the ``left`` column in the
        :ref:`sec_edge_table_definition` as a numpy array (dtype=np.float64).
        Equivalent to ``ts.tables.edges.left`` (but avoiding the full copy
        of the table data that accessing ``ts.tables`` currently entails).
        """
        return self._edges_left

    @property
    def edges_right(self):
        """
        Efficient access to the ``right`` column in the
        :ref:`sec_edge_table_definition` as a numpy array (dtype=np.float64).
        Equivalent to ``ts.tables.edges.right`` (but avoiding the full copy
        of the table data that accessing ``ts.tables`` currently entails).
        """
        return self._edges_right

    @property
    def edges_parent(self):
        """
        Efficient access to the ``parent`` column in the
        :ref:`sec_edge_table_definition` as a numpy array (dtype=np.int32).
        Equivalent to ``ts.tables.edges.parent`` (but avoiding the full copy
        of the table data that accessing ``ts.tables`` currently entails).
        """
        return self._edges_parent

    @property
    def edges_child(self):
        """
        Efficient access to the ``child`` column in the
        :ref:`sec_edge_table_definition` as a numpy array (dtype=np.int32).
        Equivalent to ``ts.tables.edges.child`` (but avoiding the full copy
        of the table data that accessing ``ts.tables`` currently entails).
        """
        return self._edges_child

    @property
    def edges_metadata(self):
        """
        Efficient access to the ``metadata`` column in the
        :ref:`sec_edge_table_definition` as a structured numpy array.
        The returned dtype will depend on the metadata schema used. Only a subset
        of `struct` metadata schemas are supported.
        See :ref:`sec_structured_array_metadata` for more information.
        """
        return self.table_metadata_schemas.edge.structured_array_from_buffer(
            self._edges_metadata
        )

    @property
    def sites_position(self):
        """
        Efficient access to the ``position`` column in the
        :ref:`sec_site_table_definition` as a numpy array (dtype=np.float64).
        Equivalent to ``ts.tables.sites.position`` (but avoiding the full copy
        of the table data that accessing ``ts.tables`` currently entails).
        """
        return self._sites_position

    @property
    def sites_ancestral_state(self):
        """
        The ``ancestral_state`` column in the
        :ref:`sec_site_table_definition` as a numpy array (dtype=StringDtype).
        """
        if not _tskit.HAS_NUMPY_2:
            raise RuntimeError(
                "The sites_ancestral_state property requires numpy 2.0 or later."
            )
        if self._sites_ancestral_state is None:
            self._sites_ancestral_state = (
                self._ll_tree_sequence.sites_ancestral_state_string
            )
        return self._sites_ancestral_state

    @property
    def sites_metadata(self):
        """
        Efficient access to the ``metadata`` column in the
        :ref:`sec_site_table_definition` as a structured numpy array.
        The returned dtype will depend on the metadata schema used. Only a subset
        of `struct` metadata schemas are supported.
        See :ref:`sec_structured_array_metadata` for more information.
        """
        return self.table_metadata_schemas.site.structured_array_from_buffer(
            self._sites_metadata
        )

    @property
    def mutations_site(self):
        """
        Efficient access to the ``site`` column in the
        :ref:`sec_mutation_table_definition` as a numpy array (dtype=np.int32).
        Equivalent to ``ts.tables.mutations.site`` (but avoiding the full copy
        of the table data that accessing ``ts.tables`` currently entails).

        .. note::
            To efficently get an array of the number of mutations per site, you
            can use ``np.bincount(ts.mutations_site, minlength=ts.num_sites)``.
        """
        return self._mutations_site

    @property
    def mutations_node(self):
        """
        Efficient access to the ``node`` column in the
        :ref:`sec_mutation_table_definition` as a numpy array (dtype=np.int32).
        Equivalent to ``ts.tables.mutations.node`` (but avoiding the full copy
        of the table data that accessing ``ts.tables`` currently entails).
        """
        return self._mutations_node

    @property
    def mutations_parent(self):
        """
        Efficient access to the ``parent`` column in the
        :ref:`sec_mutation_table_definition` as a numpy array (dtype=np.int32).
        Equivalent to ``ts.tables.mutations.parent`` (but avoiding the full copy
        of the table data that accessing ``ts.tables`` currently entails).
        """
        return self._mutations_parent

    @property
    def mutations_time(self):
        """
        Efficient access to the ``time`` column in the
        :ref:`sec_mutation_table_definition` as a numpy array (dtype=np.float64).
        Equivalent to ``ts.tables.mutations.time`` (but avoiding the full copy
        of the table data that accessing ``ts.tables`` currently entails).
        """
        return self._mutations_time

    @property
    def mutations_derived_state(self):
        """
        Access to the ``derived_state`` column in the
        :ref:`sec_mutation_table_definition` as a numpy array (dtype=StringDtype).
        """
        if not _tskit.HAS_NUMPY_2:
            raise RuntimeError(
                "The mutations_derived_state property requires numpy 2.0 or later."
            )
        if self._mutations_derived_state is None:
            self._mutations_derived_state = (
                self._ll_tree_sequence.mutations_derived_state_string
            )
        return self._mutations_derived_state

    @property
    def mutations_metadata(self):
        """
        Efficient access to the ``metadata`` column in the
        :ref:`sec_mutation_table_definition` as a structured numpy array.
        The returned dtype will depend on the metadata schema used. Only a subset
        of `struct` metadata schemas are supported.
        See :ref:`sec_structured_array_metadata` for more information.
        """
        return self.table_metadata_schemas.mutation.structured_array_from_buffer(
            self._mutations_metadata
        )

    @property
    def mutations_edge(self):
        """
        Return an array of the ID of the edge each mutation sits on in the tree sequence.

        :return: Array of shape (num_mutations,) containing edge IDs.
        :rtype: numpy.ndarray (dtype=np.int32)
        """
        if self._mutations_edge is None:
            self._mutations_edge = self._ll_tree_sequence.get_mutations_edge()
        return self._mutations_edge

    @property
    def mutations_inherited_state(self):
        """
        Return an array of the inherited state for each mutation in the tree sequence.

        The inherited state for a mutation is the state that existed at the site
        before the mutation occurred. This is either the ancestral state of the site
        (if the mutation has no parent) or the derived state of the mutation's
        parent mutation (if it has a parent).

        :return: Array of shape (num_mutations,) containing inherited states.
        :rtype: numpy.ndarray
        """
        if not _tskit.HAS_NUMPY_2:
            raise RuntimeError(
                "The mutations_inherited_state property requires numpy 2.0 or later."
            )
        if self._mutations_inherited_state is None:
            self._mutations_inherited_state = (
                self._ll_tree_sequence.mutations_inherited_state_string
            )
        return self._mutations_inherited_state

    @property
    def migrations_left(self):
        """
        Efficient access to the ``left`` column in the
        :ref:`sec_migration_table_definition` as a numpy array (dtype=np.float64).
        Equivalent to ``ts.tables.migrations.left`` (but avoiding the full copy
        of the table data that accessing ``ts.tables`` currently entails).
        """
        return self._migrations_left

    @property
    def migrations_right(self):
        """
        Efficient access to the ``right`` column in the
        :ref:`sec_migration_table_definition` as a numpy array (dtype=np.float64).
        Equivalent to ``ts.tables.migrations.right`` (but avoiding the full copy
        of the table data that accessing ``ts.tables`` currently entails).
        """
        return self._migrations_right

    @property
    def migrations_node(self):
        """
        Efficient access to the ``node`` column in the
        :ref:`sec_migration_table_definition` as a numpy array (dtype=np.int32).
        Equivalent to ``ts.tables.migrations.node`` (but avoiding the full copy
        of the table data that accessing ``ts.tables`` currently entails).
        """
        return self._migrations_node

    @property
    def migrations_source(self):
        """
        Efficient access to the ``source`` column in the
        :ref:`sec_migration_table_definition` as a numpy array (dtype=np.int32).
        Equivalent to ``ts.tables.migrations.source`` (but avoiding the full copy
        of the table data that accessing ``ts.tables`` currently entails).
        """
        return self._migrations_source

    @property
    def migrations_dest(self):
        """
        Efficient access to the ``dest`` column in the
        :ref:`sec_migration_table_definition` as a numpy array (dtype=np.int32).
        Equivalent to ``ts.tables.migrations.dest`` (but avoiding the full copy
        of the table data that accessing ``ts.tables`` currently entails).
        """
        return self._migrations_dest

    @property
    def migrations_time(self):
        """
        Efficient access to the ``time`` column in the
        :ref:`sec_migration_table_definition` as a numpy array (dtype=np.float64).
        Equivalent to ``ts.tables.migrations.time`` (but avoiding the full copy
        of the table data that accessing ``ts.tables`` currently entails).
        """
        return self._migrations_time

    @property
    def migrations_metadata(self):
        """
        Efficient access to the ``metadata`` column in the
        :ref:`sec_migration_table_definition` as a structured numpy array.
        The returned dtype will depend on the metadata schema used. Only a subset
        of `struct` metadata schemas are supported.
        See :ref:`sec_structured_array_metadata` for more information.
        """
        return self.table_metadata_schemas.migration.structured_array_from_buffer(
            self._migrations_metadata
        )

    @property
    def populations_metadata(self):
        """
        Efficient access to the ``metadata`` column in the
        :ref:`sec_population_table_definition` as a structured numpy array.
        The returned dtype will depend on the metadata schema used. Only a subset
        of `struct` metadata schemas are supported.
        See :ref:`sec_structured_array_metadata` for more information.
        """
        return self.table_metadata_schemas.population.structured_array_from_buffer(
            self._populations_metadata
        )

    @property
    def indexes_edge_insertion_order(self):
        """
        Efficient access to the ``edge_insertion_order`` column in the
        :ref:`sec_table_indexes` as a numpy array (dtype=np.int32).
        Equivalent to ``ts.tables.indexes.edge_insertion_order`` (but avoiding
        the full copy of the table data that accessing ``ts.tables``
        currently entails).
        """
        return self._indexes_edge_insertion_order

    @property
    def indexes_edge_removal_order(self):
        """
        Efficient access to the ``edge_removal_order`` column in the
        :ref:`sec_table_indexes` as a numpy array (dtype=np.int32).
        Equivalent to ``ts.tables.indexes.edge_removal_order`` (but avoiding
        the full copy of the table data that accessing ``ts.tables``
        currently entails).
        """
        return self._indexes_edge_removal_order

    def individual(self, id_):
        """
        Returns the :ref:`individual <sec_individual_table_definition>`
        in this tree sequence with the specified ID.  As with python lists, negative
        IDs can be used to index backwards from the last individual.

        :rtype: :class:`Individual`
        """
        id_ = self.check_index(id_, self.num_individuals)
        (
            flags,
            location,
            parents,
            metadata,
            nodes,
        ) = self._ll_tree_sequence.get_individual(id_)
        ind = Individual(
            id=id_,
            flags=flags,
            location=location,
            parents=parents,
            metadata=metadata,
            nodes=nodes,
            metadata_decoder=self.table_metadata_schemas.individual.decode_row,
            tree_sequence=self,
        )
        return ind

    def node(self, id_):
        """
        Returns the :ref:`node <sec_node_table_definition>` in this tree sequence
        with the specified ID. As with python lists, negative IDs can be used to
        index backwards from the last node.

        :rtype: :class:`Node`
        """
        id_ = self.check_index(id_, self.num_nodes)
        (
            flags,
            time,
            population,
            individual,
            metadata,
        ) = self._ll_tree_sequence.get_node(id_)
        return Node(
            id=id_,
            flags=flags,
            time=time,
            population=population,
            individual=individual,
            metadata=metadata,
            metadata_decoder=self.table_metadata_schemas.node.decode_row,
        )

    @staticmethod
    def check_index(index, length):
        if not isinstance(index, numbers.Integral):
            raise TypeError(
                f"Index must be of integer type, not '{type(index).__name__}'"
            )
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError("Index out of bounds")
        return index

    def edge(self, id_):
        """
        Returns the :ref:`edge <sec_edge_table_definition>` in this tree sequence
        with the specified ID. As with python lists, negative IDs can be used to
        index backwards from the last edge.

        :rtype: :class:`Edge`
        """
        id_ = self.check_index(id_, self.num_edges)
        left, right, parent, child, metadata = self._ll_tree_sequence.get_edge(id_)
        return Edge(
            id=id_,
            left=left,
            right=right,
            parent=parent,
            child=child,
            metadata=metadata,
            metadata_decoder=self.table_metadata_schemas.edge.decode_row,
        )

    def migration(self, id_):
        """
        Returns the :ref:`migration <sec_migration_table_definition>` in this tree
        sequence with the specified ID. As with python lists, negative IDs can be
        used to index backwards from the last migration.

        :rtype: :class:`.Migration`
        """
        id_ = self.check_index(id_, self.num_migrations)
        (
            left,
            right,
            node,
            source,
            dest,
            time,
            metadata,
        ) = self._ll_tree_sequence.get_migration(id_)
        return Migration(
            id=id_,
            left=left,
            right=right,
            node=node,
            source=source,
            dest=dest,
            time=time,
            metadata=metadata,
            metadata_decoder=self.table_metadata_schemas.migration.decode_row,
        )

    def mutation(self, id_):
        """
        Returns the :ref:`mutation <sec_mutation_table_definition>` in this tree sequence
        with the specified ID. As with python lists, negative IDs can be used to
        index backwards from the last mutation.

        :rtype: :class:`Mutation`
        """
        id_ = self.check_index(id_, self.num_mutations)
        (
            site,
            node,
            derived_state,
            parent,
            metadata,
            time,
            edge,
            inherited_state,
        ) = self._ll_tree_sequence.get_mutation(id_)
        return Mutation(
            id=id_,
            site=site,
            node=node,
            derived_state=derived_state,
            parent=parent,
            metadata=metadata,
            time=time,
            edge=edge,
            inherited_state=inherited_state,
            metadata_decoder=self.table_metadata_schemas.mutation.decode_row,
        )

    def site(self, id_=None, *, position=None):
        """
        Returns the :ref:`site <sec_site_table_definition>` in this tree sequence
        with either the specified ID or position. As with python lists, negative IDs
        can be used to index backwards from the last site.

        When position is specified instead of site ID, a binary search is done
        on the list of positions of the sites to try to find a site
        with the user-specified position.

        :rtype: :class:`Site`
        """
        if id_ is None and position is None:
            raise TypeError("Site id or position must be provided.")
        elif id_ is not None and position is not None:
            raise TypeError("Only one of site id or position needs to be provided.")
        elif id_ is None:
            position = np.array(position)
            if len(position.shape) > 0:
                raise ValueError("Position must be provided as a scalar value.")
            if position < 0 or position >= self.sequence_length:
                raise ValueError(
                    "Position is beyond the coordinates defined by sequence length."
                )
            site_pos = self.sites_position
            id_ = site_pos.searchsorted(position)
            if id_ >= len(site_pos) or site_pos[id_] != position:
                raise ValueError(f"There is no site at position {position}.")
        else:
            id_ = self.check_index(id_, self.num_sites)
        ll_site = self._ll_tree_sequence.get_site(id_)
        pos, ancestral_state, ll_mutations, _, metadata = ll_site
        mutations = [self.mutation(mut_id) for mut_id in ll_mutations]
        return Site(
            id=id_,
            position=pos,
            ancestral_state=ancestral_state,
            mutations=mutations,
            metadata=metadata,
            metadata_decoder=self.table_metadata_schemas.site.decode_row,
        )

    def population(self, id_):
        """
        Returns the :ref:`population <sec_population_table_definition>`
        in this tree sequence with the specified ID.  As with python lists, negative
        IDs can be used to index backwards from the last population.

        :rtype: :class:`Population`
        """
        id_ = self.check_index(id_, self.num_populations)
        (metadata,) = self._ll_tree_sequence.get_population(id_)
        return Population(
            id=id_,
            metadata=metadata,
            metadata_decoder=self.table_metadata_schemas.population.decode_row,
        )

    def provenance(self, id_):
        """
        Returns the :ref:`provenance <sec_provenance_table_definition>`
        in this tree sequence with the specified ID.  As with python lists,
        negative IDs can be used to index backwards from the last provenance.
        """
        id_ = self.check_index(id_, self.num_provenances)
        timestamp, record = self._ll_tree_sequence.get_provenance(id_)
        return Provenance(id=id_, timestamp=timestamp, record=record)

    def get_samples(self, population_id=None):
        # Deprecated alias for samples()
        return self.samples(population_id)

    def samples(self, population=None, *, population_id=None, time=None):
        """
        Returns an array of the sample node IDs in this tree sequence. If
        `population` is specified, only return sample IDs from that population.
        It is also possible to restrict samples by time using the parameter
        `time`. If `time` is a numeric value, only return sample IDs whose node
        time is approximately equal to the specified time. If `time` is a pair
        of values of the form `(min_time, max_time)`, only return sample IDs
        whose node time `t` is in this interval such that `min_time <= t < max_time`.

        :param int population: The population of interest. If None, do not
            filter samples by population.
        :param int population_id: Deprecated alias for ``population``.
        :param float,tuple time: The time or time interval of interest. If
            None, do not filter samples by time.
        :return: A numpy array of the node IDs for the samples of interest,
            listed in numerical order.
        :rtype: numpy.ndarray (dtype=np.int32)
        """
        if population is not None and population_id is not None:
            raise ValueError(
                "population_id and population are aliases. Cannot specify both"
            )
        if population_id is not None:
            population = population_id
        samples = self._ll_tree_sequence.get_samples()
        keep = np.full(shape=samples.shape, fill_value=True)
        if population is not None:
            sample_population = self.nodes_population[samples]
            keep = np.logical_and(keep, sample_population == population)
        if time is not None:
            # ndmin is set so that scalars are converted into 1d arrays
            time = np.array(time, ndmin=1, dtype=float)
            sample_times = self.nodes_time[samples]
            if time.shape == (1,):
                keep = np.logical_and(keep, np.isclose(sample_times, time))
            elif time.shape == (2,):
                if time[1] <= time[0]:
                    raise ValueError("time_interval max is less than or equal to min.")
                keep = np.logical_and(keep, sample_times >= time[0])
                keep = np.logical_and(keep, sample_times < time[1])
            else:
                raise ValueError(
                    "time must be either a single value or a pair of values "
                    "(min_time, max_time)."
                )
        return samples[keep]

    def as_vcf(self, *args, **kwargs):
        """
        Return the result of :meth:`.write_vcf` as a string.
        Keyword parameters are as defined in :meth:`.write_vcf`.

        :return: A VCF encoding of the variants in this tree sequence as a string.
        :rtype: str
        """
        buff = io.StringIO()
        self.write_vcf(buff, *args, **kwargs)
        return buff.getvalue()

    def write_vcf(
        self,
        output,
        ploidy=None,
        *,
        contig_id="1",
        individuals=None,
        individual_names=None,
        position_transform=None,
        site_mask=None,
        sample_mask=None,
        isolated_as_missing=None,
        allow_position_zero=None,
        include_non_sample_nodes=None,
    ):
        """
        Convert the genetic variation data in this tree sequence to Variant
        Call Format and write to the specified file-like object.

        .. seealso: See the :ref:`sec_export_vcf` section for examples
            and explanations of how we map VCF to the tskit data model.

        Multiploid samples in the output VCF are generated either using
        individual information in the data model (see
        :ref:`sec_individual_table_definition`), or by combining genotypes for
        adjacent sample nodes using the ``ploidy`` argument. See the
        :ref:`sec_export_vcf_constructing_gt` section for more details
        and examples.

        If individuals are defined in the
        data model (see :ref:`sec_individual_table_definition`), the genotypes
        for each of the individual's nodes are combined into a phased
        multiploid values at each site. By default, all individuals are
        included with their sample nodes, individuals with no nodes are
        omitted. The ``include_non_sample_nodes`` argument can be used to
        included non-sample nodes in the output VCF.

        Subsets or permutations of the sample individuals may be specified
        using the ``individuals`` argument.

        Mixed-sample individuals (e.g., those associated with one node
        that is a sample and another that is not) in the data model will
        only have the sample nodes output by default. However, non-sample
        nodes can be included using the ``include_non_sample_nodes`` argument.

        If there are no individuals in the tree sequence,
        synthetic individuals are created by combining adjacent samples, and
        the number of samples combined is equal to the ``ploidy`` value (1 by
        default). For example, if we have a ``ploidy`` of 2 and 6 sample nodes,
        then we will have 3 diploid samples in the VCF, consisting of the
        combined genotypes for samples [0, 1], [2, 3] and [4, 5]. If we had
        genotypes 011110 at a particular variant, then we would output the
        diploid genotypes 0|1, 1|1 and 1|0 in VCF.

        Each individual in the output is identified by a string; these are the
        VCF "sample" names. By default, these are of the form ``tsk_0``,
        ``tsk_1`` etc, up to the number of individuals, but can be manually
        specified using the ``individual_names`` argument. We do not check
        for duplicates in this array, or perform any checks to ensure that
        the output VCF is well-formed.

        .. note::
            The default individual names (VCF sample IDs) are always of
            the form ``tsk_0``, ``tsk_1``, ..., ``tsk_{N - 1}``, where
            N is the number of individuals we output. These numbers
            are **not** necessarily the individual IDs.

        The REF value in the output VCF is the ancestral allele for a site
        and ALT values are the remaining alleles. It is important to note,
        therefore, that for real data this means that the REF value for a given
        site **may not** be equal to the reference allele. We also do not
        check that the alleles result in a valid VCF---for example, it is possible
        to use the tab character as an allele, leading to a broken VCF.

        The ID value in the output VCF file is the integer ID of the
        corresponding :ref:`site <sec_site_table_definition>` (``site.id``).
        These ID values can be utilized to match the contents of the VCF file
        to the sites in the tree sequence object.

        .. note::
           Older code often uses the ``ploidy=2`` argument, because old
           versions of msprime did not output individual data. Specifying
           individuals in the tree sequence is more robust, and since tree
           sequences now  typically contain individuals (e.g., as produced by
           ``msprime.sim_ancestry( )``), this is not necessary, and the
           ``ploidy`` argument can safely be removed as part of the process
           of updating from the msprime 0.x legacy API.

        :param io.IOBase output: The file-like object to write the VCF output.
        :param int ploidy: The ploidy of the individuals to be written to
            VCF. This sample size must be evenly divisible by ploidy. Cannot be
            used if there is individual data in the tree sequence.
        :param str contig_id: The value of the CHROM column in the output VCF.
        :param list(int) individuals: A list containing the individual IDs to
            corresponding to the VCF samples. Defaults to all individuals
            associated with sample nodes in the tree sequence.
            See the {ref}`sec_export_vcf_constructing_gt` section for more
            details and examples.
        :param list(str) individual_names: A list of string names to identify
            individual columns in the VCF. In VCF nomenclature, these are the
            sample IDs. If specified, this must be a list of strings of
            length equal to the number of individuals to be output. Note that
            we do not check the form of these strings in any way, so that is
            is possible to output malformed VCF (for example, by embedding a
            tab character within on of the names). The default is to output
            ``tsk_j`` for the jth individual.
            See the :ref:`sec_export_vcf_individual_names` for examples
            and more information.
        :param position_transform: A callable that transforms the
            site position values into integer valued coordinates suitable for
            VCF. The function takes a single positional parameter x and must
            return an integer numpy array the same dimension as x. By default,
            this is set to ``numpy.round()`` which will round values to the
            nearest integer. If the string "legacy" is provided here, the
            pre 0.2.0 legacy behaviour of rounding values to the nearest integer
            (starting from 1) and avoiding the output of identical positions
            by incrementing is used.
            See the :ref:`sec_export_vcf_modifying_coordinates` for examples
            and more information.
        :param site_mask: A numpy boolean array (or something convertable to
            a numpy boolean array) with num_sites elements, used to mask out
            sites in the output. If  ``site_mask[j]`` is True, then this
            site (i.e., the line in the VCF file) will be omitted.
            See the :ref:`sec_export_vcf_masking_output` for examples
            and more information.
        :param sample_mask: A numpy boolean array (or something convertable to
            a numpy boolean array) with num_samples elements, or a callable
            that returns such an array, such that if
            ``sample_mask[j]`` is True, then the genotype for sample ``j``
            will be marked as missing using a ".". If ``sample_mask`` is a
            callable, it must take a single argument and return a boolean
            numpy array. This function will be called for each (unmasked) site
            with the corresponding :class:`.Variant` object, allowing
            for dynamic masks to be generated.
            See the :ref:`sec_export_vcf_masking_output` for examples
            and more information.
        :param bool isolated_as_missing: If True, the genotype value assigned to
            missing samples (i.e., isolated samples without mutations) is "."
            If False, missing samples will be assigned the ancestral allele.
            See :meth:`.variants` for more information. Default: True.
        :param bool allow_position_zero: If True allow sites with position zero to be
            output to the VCF, otherwise if one is present an error will be raised.
            The VCF spec does not allow for sites at position 0. However, in practise
            many tools will be fine with this. Default: False.
        :param bool include_non_sample_nodes: If True, include non-sample nodes
            in the output VCF. By default, only sample nodes are included.
        """
        if allow_position_zero is None:
            allow_position_zero = False
        writer = vcf.VcfWriter(
            self,
            ploidy=ploidy,
            contig_id=contig_id,
            individuals=individuals,
            individual_names=individual_names,
            position_transform=position_transform,
            site_mask=site_mask,
            sample_mask=sample_mask,
            isolated_as_missing=isolated_as_missing,
            allow_position_zero=allow_position_zero,
            include_non_sample_nodes=include_non_sample_nodes,
        )
        writer.write(output)

    def write_fasta(
        self,
        file_or_path,
        *,
        wrap_width=60,
        reference_sequence=None,
        missing_data_character=None,
    ):
        """
        Writes the :meth:`.alignments` for this tree sequence to file in
        `FASTA <https://en.wikipedia.org/wiki/FASTA_format>`__ format.
        Please see the :meth:`.alignments` method for details on how
        reference sequences are handled.

        Alignments are returned for the
        :ref:`sample nodes<sec_data_model_definitions>` in this tree
        sequence, and a sample with node id ``u`` is given the label
        ``f"n{u}"``, following the same convention as the
        :meth:`.write_nexus` and :meth:`Tree.as_newick` methods.

        The ``wrap_width`` parameter controls the maximum width of lines
        of sequence data in the output. By default this is 60
        characters in accordance with fasta standard outputs. To turn off
        line-wrapping of sequences, set ``wrap_width`` = 0.

        Example usage:

        .. code-block:: python

            ts.write_fasta("output.fa")

        .. warning:: :ref:`Missing data<sec_data_model_missing_data>` is not
            currently supported by this method and it will raise a ValueError
            if called on tree sequences containing isolated samples.
            See https://github.com/tskit-dev/tskit/issues/1896 for more
            information.

        :param file_or_path: The file object or path to write the output.
            Paths can be either strings or :class:`python:pathlib.Path` objects.
        :param int wrap_width: The number of sequence
            characters to include on each line in the fasta file, before wrapping
            to the next line for each sequence, or 0 to turn off line wrapping.
            (Default=60).
        :param str reference_sequence: As for the :meth:`.alignments` method.
        :param str missing_data_character: As for the :meth:`.alignments` method.
        """
        text_formats.write_fasta(
            self,
            file_or_path,
            wrap_width=wrap_width,
            reference_sequence=reference_sequence,
            missing_data_character=missing_data_character,
        )

    def as_fasta(self, **kwargs):
        """
        Return the result of :meth:`.write_fasta` as a string.
        Keyword parameters are as defined in :meth:`.write_fasta`.

        :return: A FASTA encoding of the alignments in this tree sequence as a string.
        :rtype: str
        """
        buff = io.StringIO()
        self.write_fasta(buff, **kwargs)
        return buff.getvalue()

    def write_nexus(
        self,
        file_or_path,
        *,
        precision=None,
        include_trees=None,
        include_alignments=None,
        reference_sequence=None,
        missing_data_character=None,
    ):
        """
        Returns a `nexus encoding <https://en.wikipedia.org/wiki/Nexus_file>`_
        of this tree sequence. By default, tree topologies are included
        in the output, and sequence data alignments are included by default
        if this tree sequence has discrete genome coordinates and one or
        more sites. Inclusion of these sections can be controlled manually
        using the ``include_trees`` and ``include_alignments`` parameters.

        Tree topologies and branch lengths are listed
        sequentially in the TREES block and the spatial location of each tree
        encoded within the tree name labels. Specifically, a tree spanning
        the interval :math:`[x, y)`` is given the name ``f"t{x}^{y}"``
        (See below for a description of the precision at which these spatial
        coordinates are printed out).

        The :ref:`sample nodes<sec_data_model_definitions>` in this tree
        sequence are regarded as taxa, and a sample with node id ``u``
        is given the label ``f"n{u}"``, following the same convention
        as the :meth:`Tree.as_newick` method.

        By default, genome positions are printed out with with sufficient
        precision for them to be recovered exactly in double precision.
        If the tree sequence is defined on a :attr:`.discrete_genome`,
        then positions are written out as integers. Otherwise, 17 digits
        of precision is used. Branch length precision defaults are handled
        in the same way as :meth:`.Tree.as_newick`.

        If the ``precision`` argument is provided, genome positions and
        branch lengths are printed out with this many digits of precision.

        For example, here is the nexus encoding of a simple tree sequence
        with integer times and genome coordinates with three samples
        and two trees::

            #NEXUS
            BEGIN TAXA;
              DIMENSIONS NTAX=3;
              TAXLABELS n0 n1 n2;
            END;
            BEGIN TREES;
              TREE t0^2 = [&R] (n0:3,(n1:2,n2:2):1);
              TREE t2^10 = [&R] (n1:2,(n0:1,n2:1):1);
            END;

        If sequence data :meth:`.alignments` are defined for this tree sequence
        and there is at least one site present, sequence alignment data will also
        be included by default (this can be suppressed by setting
        ``include_alignments=False``). For example, this tree sequence has
        a sequence length of 10, two variable sites and no
        :ref:`reference sequence<sec_data_model_reference_sequence>`::

            #NEXUS
            BEGIN TAXA;
              DIMENSIONS NTAX=3;
              TAXLABELS n0 n1 n2;
            END;
            BEGIN DATA;
              DIMENSIONS NCHAR=10;
              FORMAT DATATYPE=DNA MISSING=?;
              MATRIX
                n0 ??G??????T
                n1 ??A??????C
                n2 ??A??????C
              ;
            END;
            BEGIN TREES;
              TREE t0^10 = [&R] (n0:2,(n1:1,n2:1):1);
            END;

        Please see the :meth:`.alignments` method for details on how
        reference sequences are handled.

        .. note:: Note the default ``missing_data_character`` for this method
            is "?" rather then "N", in keeping with common conventions for
            nexus data. This can be changed using the ``missing_data_character``
            parameter.

        .. warning:: :ref:`Missing data<sec_data_model_missing_data>`
            is not supported for encoding tree topology information
            as our convention of using trees with multiple roots
            is not often supported by newick parsers. Thus, the method
            will raise a ValueError if we try to output trees with
            multiple roots. Additionally, missing data
            is not currently supported for alignment data.
            See https://github.com/tskit-dev/tskit/issues/1896 for more
            information.

        .. seealso: See also the :meth:`.as_nexus` method which will
            return this nexus representation as a string.

        :param int precision: The numerical precision with which branch lengths
            and tree positions are printed.
        :param bool include_trees: True if the tree topology information should
            be included; False otherwise (default=True).
        :param bool include_alignments: True if the sequence data alignment information
            should be included; False otherwise (default=True if sequence alignments
            are well-defined and the tree sequence contains at least one site).
        :param str reference_sequence: As for the :meth:`.alignments` method.
        :param str missing_data_character: As for the :meth:`.alignments` method,
            but defaults to "?".
        :return: A nexus representation of this :class:`TreeSequence`
        :rtype: str
        """
        text_formats.write_nexus(
            self,
            file_or_path,
            precision=precision,
            include_trees=include_trees,
            include_alignments=include_alignments,
            reference_sequence=reference_sequence,
            missing_data_character=missing_data_character,
        )

    def as_nexus(self, **kwargs):
        """
        Return the result of :meth:`.write_nexus` as a string.
        Keyword parameters are as defined in :meth:`.write_nexus`.

        :return: A nexus encoding of the alignments in this tree sequence as a string.
        :rtype: str
        """
        buff = io.StringIO()
        self.write_nexus(buff, **kwargs)
        return buff.getvalue()

    # TODO
    # (1) Move the definition to text_formats.py
    # (2) Rename to as_macs and keep to_macs as a deprecated synonym
    def to_macs(self):
        """
        Return a `macs encoding <https://github.com/gchen98/macs>`_
        of this tree sequence.

        :return: The macs representation of this TreeSequence as a string.
        :rtype: str
        """
        n = self.get_sample_size()
        m = self.get_sequence_length()
        output = [f"COMMAND:\tnot_macs {n} {m}"]
        output.append("SEED:\tASEED")
        for variant in self.variants(copy=False):
            if any(len(allele) > 1 for allele in variant.alleles):
                raise ValueError("macs output only supports single letter alleles")
            bytes_genotypes = np.empty(self.num_samples, dtype=np.uint8)
            lookup = np.array([ord(a[0]) for a in variant.alleles], dtype=np.uint8)
            bytes_genotypes[:] = lookup[variant.genotypes]
            genotypes = bytes_genotypes.tobytes().decode()
            output.append(
                f"SITE:\t{variant.index}\t{variant.position / m}\t0.0\t" f"{genotypes}"
            )
        return "\n".join(output) + "\n"

    def simplify(
        self,
        samples=None,
        *,
        map_nodes=False,
        reduce_to_site_topology=False,
        filter_populations=None,
        filter_individuals=None,
        filter_sites=None,
        filter_nodes=None,
        update_sample_flags=None,
        keep_unary=False,
        keep_unary_in_individuals=None,
        keep_input_roots=False,
        record_provenance=True,
        filter_zero_mutation_sites=None,  # Deprecated alias for filter_sites
    ):
        """
        Returns a simplified tree sequence that retains only the history of
        the nodes given in the list ``samples``. If ``map_nodes`` is true,
        also return a numpy array whose ``u``-th element is the ID of the node
        in the simplified tree sequence that corresponds to node ``u`` in the
        original tree sequence, or :data:`tskit.NULL` (-1) if ``u`` is no longer
        present in the simplified tree sequence.

        .. note::
            If you wish to simplify a set of tables that do not satisfy all
            requirements for building a TreeSequence, then use
            :meth:`TableCollection.simplify`.

        If the ``reduce_to_site_topology`` parameter is True, the returned tree
        sequence will contain only topological information that is necessary to
        represent the trees that contain sites. If there are zero sites in this
        tree sequence, this will result in an output tree sequence with zero edges.
        When the number of sites is greater than zero, every tree in the output
        tree sequence will contain at least one site. For a given site, the
        topology of the tree containing that site will be identical
        (up to node ID remapping) to the topology of the corresponding tree
        in the input tree sequence.

        If ``filter_populations``, ``filter_individuals``, ``filter_sites``, or
        ``filter_nodes`` is True, any of the corresponding objects that are not
        referenced elsewhere are filtered out. As this is the default behaviour,
        it is important to realise IDs for these objects may change through
        simplification. By setting these parameters to False, however, the
        corresponding tables can be preserved without changes.

        If ``filter_nodes`` is False, then the output node table will be
        unchanged except for updating the sample status of nodes and any ID
        remappings caused by filtering individuals and populations (if the
        ``filter_individuals`` and ``filter_populations`` options are enabled).
        Nodes that are in the specified list of ``samples`` will be marked as
        samples in the output, and nodes that are currently marked as samples
        in the node table but not in the specified list of ``samples`` will
        have their :data:`tskit.NODE_IS_SAMPLE` flag cleared. Note also that
        the order of the ``samples`` list is not meaningful when
        ``filter_nodes`` is False. In this case, the returned node mapping is
        always the identity mapping, such that ``a[u] == u`` for all nodes.

        Setting the ``update_sample_flags`` parameter to False disables the
        automatic sample status update of nodes (described above) from
        occuring, making it the responsibility of calling code to keep track of
        the ultimate sample status of nodes. This is an advanced option, mostly
        of use when combined with the ``filter_nodes=False``,
        ``filter_populations=False`` and ``filter_individuals=False`` options,
        which then guarantees that the node table will not be altered by
        simplification.

        :param list[int] samples: A list of node IDs to retain as samples. They
            need not be nodes marked as samples in the original tree sequence, but
            will constitute the entire set of samples in the returned tree sequence.
            If not specified or None, use all nodes marked with the IS_SAMPLE flag.
            The list may be provided as a numpy array (or array-like) object
            (dtype=np.int32).
        :param bool map_nodes: If True, return a tuple containing the resulting
            tree sequence and a numpy array mapping node IDs in the current tree
            sequence to their corresponding node IDs in the returned tree sequence.
            If False (the default), return only the tree sequence object itself.
        :param bool reduce_to_site_topology: Whether to reduce the topology down
            to the trees that are present at sites. (Default: False)
        :param bool filter_populations: If True, remove any populations that are
            not referenced by nodes after simplification; new population IDs are
            allocated sequentially from zero. If False, the population table will
            not be altered in any way. (Default: None, treated as True)
        :param bool filter_individuals: If True, remove any individuals that are
            not referenced by nodes after simplification; new individual IDs are
            allocated sequentially from zero. If False, the individual table will
            not be altered in any way. (Default: None, treated as True)
        :param bool filter_sites: If True, remove any sites that are
            not referenced by mutations after simplification; new site IDs are
            allocated sequentially from zero. If False, the site table will not
            be altered in any way. (Default: None, treated as True)
        :param bool filter_nodes: If True, remove any nodes that are
            not referenced by edges after simplification. If False, the only
            potential change to the node table may be to change the node flags
            (if ``samples`` is specified and different from the existing samples).
            (Default: None, treated as True)
        :param bool update_sample_flags: If True, update node flags to so that
            nodes in the specified list of samples have the NODE_IS_SAMPLE
            flag after simplification, and nodes that are not in this list
            do not. (Default: None, treated as True)
        :param bool keep_unary: If True, preserve unary nodes (i.e., nodes with
            exactly one child) that exist on the path from samples to root.
            (Default: False)
        :param bool keep_unary_in_individuals: If True, preserve unary nodes
            that exist on the path from samples to root, but only if they are
            associated with an individual in the individuals table. Cannot be
            specified at the same time as ``keep_unary``. (Default: ``None``,
            equivalent to False)
        :param bool keep_input_roots: Whether to retain history ancestral to the
            MRCA of the samples. If ``False``, no topology older than the MRCAs of the
            samples will be included. If ``True`` the roots of all trees in the returned
            tree sequence will be the same roots as in the original tree sequence.
            (Default: False)
        :param bool record_provenance: If True, record details of this call to
            simplify in the returned tree sequence's provenance information
            (Default: True).
        :param bool filter_zero_mutation_sites: Deprecated alias for ``filter_sites``.
        :return: The simplified tree sequence, or (if ``map_nodes`` is True)
            a tuple consisting of the simplified tree sequence and a numpy array
            mapping source node IDs to their corresponding IDs in the new tree
            sequence.
        :rtype: tskit.TreeSequence or (tskit.TreeSequence, numpy.ndarray)
        """
        tables = self.dump_tables()
        assert tables.sequence_length == self.sequence_length
        node_map = tables.simplify(
            samples=samples,
            reduce_to_site_topology=reduce_to_site_topology,
            filter_populations=filter_populations,
            filter_individuals=filter_individuals,
            filter_sites=filter_sites,
            filter_nodes=filter_nodes,
            update_sample_flags=update_sample_flags,
            keep_unary=keep_unary,
            keep_unary_in_individuals=keep_unary_in_individuals,
            keep_input_roots=keep_input_roots,
            record_provenance=record_provenance,
            filter_zero_mutation_sites=filter_zero_mutation_sites,
        )
        new_ts = tables.tree_sequence()
        assert new_ts.sequence_length == self.sequence_length
        if map_nodes:
            return new_ts, node_map
        else:
            return new_ts

    def delete_sites(self, site_ids, record_provenance=True):
        """
        Returns a copy of this tree sequence with the specified sites (and their
        associated mutations) entirely removed. The site IDs do not need to be in any
        particular order, and specifying the same ID multiple times does not have any
        effect (i.e., calling ``tree_sequence.delete_sites([0, 1, 1])`` has the same
        effect as calling ``tree_sequence.delete_sites([0, 1])``.

        .. note::
            To remove only the mutations associated with a site, but keep the site
            itself, use the :meth:`MutationTable.keep_rows` method.

        :param list[int] site_ids: A list of site IDs specifying the sites to remove.
        :param bool record_provenance: If ``True``, add details of this operation to the
            provenance information of the returned tree sequence. (Default: ``True``).
        """
        tables = self.dump_tables()
        tables.delete_sites(site_ids, record_provenance)
        return tables.tree_sequence()

    def delete_intervals(self, intervals, simplify=True, record_provenance=True):
        """
        Returns a copy of this tree sequence for which information in the
        specified list of genomic intervals has been deleted. Edges spanning these
        intervals are truncated or deleted, and sites and mutations falling within
        them are discarded. Note that it is the information in the intervals that
        is deleted, not the intervals themselves, so in particular, all samples
        will be isolated in the deleted intervals.

        Note that node IDs may change as a result of this operation,
        as by default :meth:`.simplify` is called on the returned tree sequence
        to remove redundant nodes. If you wish to map node IDs onto the same
        nodes before and after this method has been called, specify ``simplify=False``.

        See also :meth:`.keep_intervals`, :meth:`.ltrim`, :meth:`.rtrim`, and
        :ref:`missing data<sec_data_model_missing_data>`.

        :param array_like intervals: A list (start, end) pairs describing the
            genomic intervals to delete. Intervals must be non-overlapping and
            in increasing order. The list of intervals must be interpretable as a
            2D numpy array with shape (N, 2), where N is the number of intervals.
        :param bool simplify: If True, return a simplified tree sequence where nodes
            no longer used are discarded. (Default: True).
        :param bool record_provenance: If ``True``, add details of this operation to the
            provenance information of the returned tree sequence. (Default: ``True``).
        :rtype: tskit.TreeSequence
        """
        tables = self.dump_tables()
        tables.delete_intervals(intervals, simplify, record_provenance)
        return tables.tree_sequence()

    def keep_intervals(self, intervals, simplify=True, record_provenance=True):
        """
        Returns a copy of this tree sequence which includes only information in
        the specified list of genomic intervals. Edges are truncated to lie within
        these intervals, and sites and mutations falling outside these intervals
        are discarded.  Note that it is the information outside the intervals that
        is deleted, not the intervals themselves, so in particular, all samples
        will be isolated outside of the retained intervals.

        Note that node IDs may change as a result of this operation,
        as by default :meth:`.simplify` is called on the returned tree sequence
        to remove redundant nodes. If you wish to map node IDs onto the same
        nodes before and after this method has been called, specify ``simplify=False``.

        See also :meth:`.keep_intervals`, :meth:`.ltrim`, :meth:`.rtrim`, and
        :ref:`missing data<sec_data_model_missing_data>`.

        :param array_like intervals: A list (start, end) pairs describing the
            genomic intervals to keep. Intervals must be non-overlapping and
            in increasing order. The list of intervals must be interpretable as a
            2D numpy array with shape (N, 2), where N is the number of intervals.
        :param bool simplify: If True, return a simplified tree sequence where nodes
            no longer used are discarded. (Default: True).
        :param bool record_provenance: If True, add details of this operation to the
            provenance information of the returned tree sequence.
            (Default: True).
        :rtype: tskit.TreeSequence
        """
        tables = self.dump_tables()
        tables.keep_intervals(intervals, simplify, record_provenance)
        return tables.tree_sequence()

    def ltrim(self, record_provenance=True):
        """
        Returns a copy of this tree sequence with a potentially changed coordinate
        system, such that empty regions (i.e., those not covered by any edge) at the
        start of the tree sequence are trimmed away, and the leftmost edge starts at
        position 0. This affects the reported position of sites and
        edges. Additionally, sites and their associated mutations to the left of
        the new zero point are thrown away.

        :param bool record_provenance: If True, add details of this operation to the
            provenance information of the returned tree sequence. (Default: True).
        """
        tables = self.dump_tables()
        tables.ltrim(record_provenance)
        return tables.tree_sequence()

    def rtrim(self, record_provenance=True):
        """
        Returns a copy of this tree sequence with the ``sequence_length`` property reset
        so that the sequence ends at the end of the rightmost edge. Additionally, sites
        and their associated mutations at positions greater than the new
        ``sequence_length`` are thrown away.

        :param bool record_provenance: If True, add details of this operation to the
            provenance information of the returned tree sequence. (Default: True).
        """
        tables = self.dump_tables()
        tables.rtrim(record_provenance)
        return tables.tree_sequence()

    def trim(self, record_provenance=True):
        """
        Returns a copy of this tree sequence with any empty regions (i.e., those not
        covered by any edge) on the right and left trimmed away. This may reset both the
        coordinate system and the ``sequence_length`` property. It is functionally
        equivalent to :meth:`.rtrim` followed by :meth:`.ltrim`. Sites and their
        associated mutations in the empty regions are thrown away.

        :param bool record_provenance: If True, add details of this operation to the
            provenance information of the returned tree sequence. (Default: True).
        """
        tables = self.dump_tables()
        tables.trim(record_provenance)
        return tables.tree_sequence()

    def shift(self, value, sequence_length=None, record_provenance=True):
        """
        Shift the coordinate system (used by edges and sites) of this TableCollection by
        a given value. Positive values shift the coordinate system to the right, negative
        values to the left. The sequence length of the tree sequence will be changed by
        ``value``, unless ``sequence_length`` is given, in which case this will be used
        for the new sequence length.

        .. note::
            By setting ``value=0``, this method will simply return a tree sequence
            with a new sequence length.

        :param value: The amount by which to shift the coordinate system.
        :param sequence_length: The new sequence length of the tree sequence. If
            ``None`` (default) add ``value`` to the sequence length.
        :raises ValueError: If the new coordinate system is invalid (e.g., if
            shifting the coordinate system results in negative coordinates).
        """
        tables = self.dump_tables()
        tables.shift(
            value=value,
            sequence_length=sequence_length,
            record_provenance=record_provenance,
        )
        return tables.tree_sequence()

    def concatenate(
        self, *args, node_mappings=None, record_provenance=True, add_populations=None
    ):
        r"""
        Concatenate a set of tree sequences to the right of this one, by repeatedly
        calling :meth:`~TreeSequence.union` with an (optional)
        node mapping for each of the ``others``. If any node mapping is ``None``
        only map the sample nodes between the input tree sequence and this one,
        based on the numerical order of sample node IDs.

        .. note::
            To add gaps between the concatenated tables, use :meth:`shift` or
            to remove gaps, use :meth:`trim` before concatenating.

        :param TreeSequence \*args: A list of other tree sequences to append to
            the right of this one.
        :param Union[list, None] node_mappings: An list of node mappings for each
            input tree sequence in ``args``. Each should either be an array of
            integers of the same length as the number of nodes in the equivalent
            input tree sequence (see :meth:`~TreeSequence.union` for details), or
            ``None``. If ``None``, only sample nodes are mapped to each other.
            Default: ``None``, treated as ``[None] * len(args)``.
        :param bool record_provenance: If True (default), record details of this
            call to ``concatenate`` in the returned tree sequence's provenance
            information (Default: True).
        :param bool add_populations: If True (default), nodes new to ``self`` will
            be assigned new population IDs (see :meth:`~TreeSequence.union`)
        """
        if node_mappings is None:
            node_mappings = [None] * len(args)
        if add_populations is None:
            add_populations = True
        if len(node_mappings) != len(args):
            raise ValueError(
                "You must provide the same number of node_mappings as args"
            )

        samples = self.samples()
        tables = self.dump_tables()
        tables.drop_index()

        for node_mapping, other in zip(node_mappings, args):
            if node_mapping is None:
                other_samples = other.samples()
                if len(other_samples) != len(samples):
                    raise ValueError(
                        "each `other` must have the same number of samples as `self`"
                    )
                node_mapping = np.full(other.num_nodes, tskit.NULL, dtype=np.int32)
                node_mapping[other_samples] = samples
            other_tables = other.dump_tables()
            other_tables.shift(tables.sequence_length, record_provenance=False)
            tables.sequence_length = other_tables.sequence_length
            # NB: should we use a different default for add_populations?
            tables.union(
                other_tables,
                node_mapping=node_mapping,
                check_shared_equality=False,  # Else checks fail with internal samples
                record_provenance=False,
                add_populations=add_populations,
            )
        if record_provenance:
            parameters = {
                "command": "concatenate",
                "TODO": "add concatenate parameters",  # tricky as both have provenances
            }
            tables.provenances.add_row(
                record=json.dumps(provenance.get_provenance_dict(parameters))
            )

        return tables.tree_sequence()

    def split_edges(self, time, *, flags=None, population=None, metadata=None):
        """
        Returns a copy of this tree sequence in which we replace any
        edge ``(left, right, parent, child)`` in which
        ``node_time[child] < time < node_time[parent]`` with two edges
        ``(left, right, parent, u)`` and ``(left, right, u, child)``,
        where ``u`` is a newly added node for each intersecting edge.

        If ``metadata``, ``flags``, or ``population`` are specified, newly
        added nodes will be assigned these values. Otherwise, default values
        will be used. The default metadata is an empty dictionary if a metadata
        schema is defined for the node table, and is an empty byte string
        otherwise. The default population for the new node is
        :data:`tskit.NULL`. Newly added have a default ``flags`` value of 0.

        Any metadata associated with a split edge will be copied to the new edge.

        .. warning:: This method currently does not support migrations
            and a error will be raised if the migration table is not
            empty. Future versions may take migrations that intersect with the
            edge into account when determining the default population
            assignments for new nodes.

        Any mutations lying on the edge whose time is >= ``time`` will have
        their node value set to ``u``. Note that the time of the mutation is
        defined as the time of the child node if the mutation's time is
        unknown.

        :param float time: The cutoff time.
        :param int flags: The flags value for newly-inserted nodes. (Default = 0)
        :param int population: The population value for newly inserted nodes.
            Defaults to ``tskit.NULL`` if not specified.
        :param metadata: The metadata for any newly inserted nodes. See
            :meth:`.NodeTable.add_row` for details on how default metadata
            is produced for a given schema (or none).
        :return: A copy of this tree sequence with edges split at the specified time.
        :rtype: tskit.TreeSequence
        """
        population = tskit.NULL if population is None else population
        flags = 0 if flags is None else flags
        schema = self.table_metadata_schemas.node
        if metadata is None:
            metadata = schema.empty_value
        metadata = schema.validate_and_encode_row(metadata)
        ll_ts = self._ll_tree_sequence.split_edges(
            time=time,
            flags=flags,
            population=population,
            metadata=metadata,
        )
        return TreeSequence(ll_ts)

    def decapitate(self, time, *, flags=None, population=None, metadata=None):
        """
        Delete all edge topology and mutational information at least as old
        as the specified time from this tree sequence.

        Removes all edges in which the time of the child is >= the specified
        time ``t``, and breaks edges that intersect with ``t``. For each edge
        intersecting with ``t`` we create a new node with time equal to ``t``,
        and set the parent of the edge to this new node. The node table
        is not altered in any other way. Newly added nodes have values
        for ``flags``, ``population`` and ``metadata`` controlled by parameters
        to this function in the same way as :meth:`.split_edges`.

        .. note::
            Note that each edge is treated independently, so that even if two
            edges that are broken by this operation share the same parent and
            child nodes, there will be two different new parent nodes inserted.

        Any mutation whose time is >= ``t`` will be removed. A mutation's time
        is its associated ``time`` value, or the time of its node if the
        mutation's time was marked as unknown (:data:`UNKNOWN_TIME`).

        Migrations are not supported, and a LibraryError will be raise if
        called on a tree sequence containing migration information.

        .. seealso:: This method is implemented using the :meth:`.split_edges`
            and :meth:`TableCollection.delete_older` functions.

        :param float time: The cutoff time.
        :param int flags: The flags value for newly-inserted nodes. (Default = 0)
        :param int population: The population value for newly inserted nodes.
            Defaults to ``tskit.NULL`` if not specified.
        :param metadata: The metadata for any newly inserted nodes. See
            :meth:`.NodeTable.add_row` for details on how default metadata
            is produced for a given schema (or none).
        :return: A copy of this tree sequence with edges split at the specified time.
        :rtype: tskit.TreeSequence
        """
        split_ts = self.split_edges(
            time, flags=flags, population=population, metadata=metadata
        )
        tables = split_ts.dump_tables()
        del split_ts
        tables.delete_older(time)
        return tables.tree_sequence()

    def extend_haplotypes(self, max_iter=10):
        """
        Returns a new tree sequence in which the span covered by ancestral nodes
        is "extended" to regions of the genome according to the following rule:
        If an ancestral segment corresponding to node `n` has ancestor `p` and
        descendant `c` on some portion of the genome, and on an adjacent segment of
        genome `p` is still an ancestor of `c`, then `n` is inserted into the
        path from `p` to `c`. For instance, if `p` is the parent of `n` and `n`
        is the parent of `c`, then the span of the edges from `p` to `n` and
        `n` to `c` are extended, and the span of the edge from `p` to `c` is
        reduced. Thus, the ancestral haplotype represented by `n` is extended
        to a longer span of the genome. However, any edges whose child node is
        a sample are not modified.

        Since some edges may be removed entirely, this process usually reduces
        the number of edges in the tree sequence.

        The method works by iterating over the genome to look for paths that can
        be extended in this way; the maximum number of such iterations is
        controlled by ``max_iter``.

        The rationale is that we know that `n` carries a portion of the segment
        of ancestral genome inherited by `c` from `p`, and so likely carries
        the *entire* inherited segment (since the implication otherwise would
        be that distinct recombined segments were passed down separately from
        `p` to `c`).

        In the example above, if there was a mutation on the node above `c`
        older than the time of `n` in the span into which `n` was extended,
        then the mutation will now occur above `n`. So, this operation may change
        mutations' nodes (but will not affect genotypes).  This is only
        unambiguous if the mutation's time is known, so the method requires
        known mutation times.  See :meth:`.impute_unknown_mutations_time` if
        mutation times are not known.

        The method will not affect the marginal trees (so, if the original tree
        sequence was simplified, then following up with `simplify` will recover
        the original tree sequence, possibly with edges in a different order).
        It will also not affect the genotype matrix, or any of the tables other
        than the edge table or the node column in the mutation table.

        :param int max_iters: The maximum number of iterations over the tree
            sequence. Defaults to 10.

        :return: A new tree sequence with unary nodes extended.
        :rtype: tskit.TreeSequence
        """
        max_iter = int(max_iter)
        ll_ts = self._ll_tree_sequence.extend_haplotypes(max_iter)
        return TreeSequence(ll_ts)

    def subset(
        self,
        nodes,
        record_provenance=True,
        reorder_populations=True,
        remove_unreferenced=True,
    ):
        """
        Returns a tree sequence containing only information directly
        referencing the provided list of nodes to retain.  The result will
        retain only the nodes whose IDs are listed in ``nodes``, only edges for
        which both parent and child are in ``nodes```, only mutations whose
        node is in ``nodes``, and only individuals that are referred to by one
        of the retained nodes.  Note that this does *not* retain
        the ancestry of these nodes - for that, see :meth:`.simplify`.

        This has the side effect that it may change the order of the nodes,
        individuals, populations, and migrations in the tree sequence: the nodes
        in the new tree sequence will be in the order provided in ``nodes``, and
        both individuals and populations will be ordered by the earliest retained
        node that refers to them. (However, ``reorder_populations`` may be set to
        False to keep the population table unchanged.)

        By default, the method removes all individuals and populations not
        referenced by any nodes, and all sites not referenced by any mutations.
        To retain these unreferenced individuals, populations, and sites, pass
        ``remove_unreferenced=False``. If this is done, the site table will
        remain unchanged, unreferenced individuals will appear at the end of
        the individuals table (and in their original order), and unreferenced
        populations will appear at the end of the population table (unless
        ``reorder_populations=False``).

        .. seealso::

            :meth:`.keep_intervals` for subsetting a given portion of the genome;
            :meth:`.simplify` for retaining the ancestry of a subset of nodes.

        :param list nodes: The list of nodes for which to retain information. This
            may be a numpy array (or array-like) object (dtype=np.int32).
        :param bool record_provenance: Whether to record a provenance entry
            in the provenance table for this operation.
        :param bool reorder_populations: Whether to reorder populations
            (default: True).  If False, the population table will not be altered in
            any way.
        :param bool remove_unreferenced: Whether sites, individuals, and populations
            that are not referred to by any retained entries in the tables should
            be removed (default: True). See the description for details.
        :rtype: tskit.TreeSequence
        """
        tables = self.dump_tables()
        tables.subset(
            nodes,
            record_provenance=record_provenance,
            reorder_populations=reorder_populations,
            remove_unreferenced=remove_unreferenced,
        )
        return tables.tree_sequence()

    def union(
        self,
        other,
        node_mapping,
        check_shared_equality=True,
        add_populations=True,
        record_provenance=True,
    ):
        """
        Returns an expanded tree sequence which contains the node-wise union of
        ``self`` and ``other``, obtained by adding the non-shared portions of
        ``other`` onto ``self``. The "shared" portions are specified using a
        map that specifies which nodes in ``other`` are equivalent to those in
        ``self``: the ``node_mapping`` argument should be an array of length
        equal to the number of nodes in ``other`` and whose entries are the ID
        of the matching node in ``self``, or ``tskit.NULL`` if there is no
        matching node. Those nodes in ``other`` that map to ``tskit.NULL`` will
        be added to ``self``, along with:

        1. Individuals whose nodes are new to ``self``.
        2. Edges whose parent or child are new to ``self``.
        3. Mutations whose nodes are new to ``self``.
        4. Sites which were not present in ``self``, if the site contains a newly
           added mutation.

        This can be thought of as a "node-wise" union: for instance, it can not
        be used to add new edges between two nodes already in ``self`` or new
        mutations above nodes already in ``self``.

        By default, with ``add_populations=True``, populations of all newly added
        nodes are assumed to be new populations, and added to the end of the
        population table as well. This is appropriate if all nodes to be added
        are from distinct populations not already in ``self`` and ordering of
        populations is not important. On the other hand, if
        ``add_populations=False`` then no new populations are added, so any
        populations referred to in ``other`` must already exist in ``self``.
        If some new nodes are in populations already in ``self`` but other new
        nodes are in entirely new populations, then you must set up the
        population table first, and then union with ``add_populations=False``.

        If the resulting tree sequence is invalid (for instance, a node is
        specified to have two distinct parents on the same interval),
        an error will be raised.

        Note that this operation also sorts the resulting tables, so the
        resulting tree sequence may not be equal to ``self`` even if nothing
        new was added (although it would differ only in ordering of the tables).

        :param TableCollection other: Another table collection.
        :param list node_mapping: An array of node IDs that relate nodes in
            ``other`` to nodes in ``self``.
        :param bool check_shared_equality: If True, the shared portions of the
            tree sequences will be checked for equality. It does so by
            running :meth:`TreeSequence.subset` on both ``self`` and ``other``
            for the equivalent nodes specified in ``node_mapping``, and then
            checking for equality of the subsets.
        :param bool add_populations: If True, nodes new to ``self`` will be
            assigned new population IDs.
        :param bool record_provenance: Whether to record a provenance entry
            in the provenance table for this operation.
        """
        tables = self.dump_tables()
        other_tables = other.dump_tables()
        tables.union(
            other_tables,
            node_mapping,
            check_shared_equality=check_shared_equality,
            add_populations=add_populations,
            record_provenance=record_provenance,
        )
        return tables.tree_sequence()

    def draw_svg(
        self,
        path=None,
        *,
        size=None,
        x_scale=None,
        time_scale=None,
        tree_height_scale=None,
        title=None,
        node_labels=None,
        mutation_labels=None,
        node_titles=None,
        mutation_titles=None,
        root_svg_attributes=None,
        style=None,
        order=None,
        force_root_branch=None,
        symbol_size=None,
        x_axis=None,
        x_label=None,
        x_lim=None,
        x_regions=None,
        y_axis=None,
        y_label=None,
        y_ticks=None,
        y_gridlines=None,
        omit_sites=None,
        canvas_size=None,
        max_num_trees=None,
        preamble=None,
        **kwargs,
    ):
        """
        Return an SVG representation of a tree sequence. See the
        :ref:`visualization tutorial<tutorials:sec_tskit_viz>` for more details.

        :param str path: The path to the file to write the output. If None, do not write
            to file.
        :param tuple(int, int) size: A tuple of (width, height) specifying a target
            drawing size in abstract user units (usually interpreted as pixels on
            initial display). Components of the drawing will be scaled so that the total
            plot including labels etc. normally fits onto a canvas of this size (see
            ``canvas_size`` below). If ``None``, chose values such that each tree is
            drawn at a size appropriate for a reasonably small set of samples (this will
            nevertheless result in a very wide drawing if there are many trees to
            display). Default: ``None``
        :param str x_scale: Control how the X axis is drawn. If "physical" (the default)
            the axis scales linearly with physical distance along the sequence,
            background shading is used to indicate the position of the trees along the
            X axis, and sites (with associated mutations) are marked at the
            appropriate physical position on axis line. If "treewise", each axis tick
            corresponds to a tree boundary, which are positioned evenly along the axis,
            so that the X axis is of variable scale, no background scaling is required,
            and site positions are not marked on the axis.
        :param str time_scale: Control how height values for nodes are computed.
            If this is equal to ``"time"``, node heights are proportional to their time
            values (this is the default). If this is equal to ``"log_time"``, node
            heights are proportional to their log(time) values. If it is equal to
            ``"rank"``, node heights are spaced equally according to their ranked times.
        :param str tree_height_scale: Deprecated alias for time_scale. (Deprecated in
            0.3.6)
        :param str title: A title string to be included in the SVG output. If ``None``
            (default) no title is shown, which gives more vertical space for the tree.
        :param node_labels: If specified, show custom labels for the nodes
            (specified by ID) that are present in this map; any nodes not present will
            not have a label.
        :type node_labels: dict(int, str)
        :param mutation_labels: If specified, show custom labels for the
            mutations (specified by ID) that are present in the map; any mutations
            not present will not have a label.
        :param dict(int, str) node_titles: If specified, add a ``<title>`` string to
            symbols for each node (specified by ID) present in this map. SVG visualizers
            such as web browsers will commonly display this string on mousing over
            node symbol.
        :param dict(int, str) mutation_titles: If specified, add a ``<title>`` string to
            symbols for each mutation (specified by ID) present in this map. SVG
            visualizers such as web browsers will commonly display this string on
            mousing over the mutation symbol in the tree and (if show) on the x axis.
        :param dict root_svg_attributes: Additional attributes, such as an id, that will
            be embedded in the root ``<svg>`` tag of the generated drawing.
        :param str style: A `css string <https://www.w3.org/TR/CSS21/syndata.htm>`_
            that will be included in the ``<style>`` tag of the generated svg.
        :param str order: The left-to-right ordering of child nodes in each drawn tree.
            This can be either: ``"minlex"``, which minimises the differences
            between adjacent trees (see also the ``"minlex_postorder"`` traversal
            order for the :meth:`.Tree.nodes` method); or ``"tree"`` which draws trees
            in the left-to-right order defined by the
            :ref:`quintuply linked tree structure <sec_data_model_tree_structure>`.
            If not specified or None, this defaults to ``"minlex"``.
        :param bool force_root_branch: If ``True`` plot a branch (edge) above every tree
            root in the tree sequence. If ``None`` (default) then only plot such
            root branches if any root in the tree sequence has a mutation above it.
        :param float symbol_size: Change the default size of the node and mutation
            plotting symbols. If ``None`` (default) use a standard size.
        :param bool x_axis: Should the plot have an X axis line, showing the positions
            of trees along the genome. The scale used is determined by the ``x_scale``
            parameter. If ``None`` (default) plot an X axis.
        :param str x_label: Place a label under the plot. If ``None`` (default) and
            there is an X axis, create and place an appropriate label.
        :param list x_lim: A list of size two giving the genomic positions between which
            trees should be plotted. If the first is ``None``, then plot from the first
            non-empty region of the tree sequence. If the second is ``None``, then plot
            up to the end of the last non-empty region of the tree sequence. The default
            value ``x_lim=None`` is shorthand for the list [``None``, ``None``]. If
            numerical values are given, then regions outside the interval have all
            information discarded: this means that mutations outside the interval will
            not be shown. To force display of the entire tree sequence, including empty
            flanking regions, specify ``x_lim=[0, ts.sequence_length]``.
        :param dict x_regions: A dictionary mapping (left, right) tuples to names. This
            draws a box, labelled with the name, on the X axis between the left and
            right positions, and can be used for annotating genomic regions (e.g.
            genes) on the X axis. If ``None`` (default) do not plot any regions.
        :param Union[bool, str] y_axis: Should the plot have an Y axis line, showing
            time. If ``False`` do not plot a Y axis. If ``True``, plot the Y axis on
            left hand side of the plot. Can also take the strings ``"left"`` or
            ``"right"``, specifying the side of the plot on which to plot the Y axis.
            Default: ``None``, treated as ``False``.
        :param str y_label: Place a label to the left of the plot. If ``None`` (default)
            and there is a Y axis, create and place an appropriate label.
        :param Union[list, dict] y_ticks: A list of Y values at which to plot
            tickmarks, or a dictionary mapping Y values to labels (``[]`` gives no
            tickmarks). If ``None`` (default), plot one tickmark for each unique node
            value. Note that if ``time_scale="rank"``, the Y values refer to the
            zero-based rank of the plotted nodes, rather than the node time itself.
        :param bool y_gridlines: Whether to plot horizontal lines behind the tree
            at each y tickmark.
        :param bool omit_sites: If True, omit sites and mutations from the drawing.
            Default: False
        :param tuple(int, int) canvas_size: The (width, height) of the SVG canvas.
            This will change the SVG width and height without rescaling graphical
            elements, allowing extra room e.g. for unusually long labels. If ``None``
            take the canvas size to be the same as the target drawing size (see
            ``size``, above). Default: None
        :param int max_num_trees: The maximum number of trees to plot. If there are
            more trees than this in the tree sequence, the middle trees will be skipped
            from the plot and a message "XX trees skipped" displayed in their place.
            If ``None``, all the trees will be plotted: this can produce a very wide
            plot if there are many trees in the tree sequence. Default: None
        :param str preamble: SVG commands to be included at the start of the returned
            object, immediately after the opening tag. These can include custom svg
            elements such as legends or annotations or even entire ``<svg>`` elements.
            The preamble is not checked for validity, so it is up to the user to
            ensure that it is valid SVG. Default: None

        :return: An SVG representation of a tree sequence.
        :rtype: SVGString

        .. note::
            Technically, x_lim[0] specifies a *minimum* value for the start of the X
            axis, and x_lim[1] specifies a *maximum* value for the end. This is only
            relevant if the tree sequence contains "empty" regions with no edges or
            mutations. In this case if x_lim[0] lies strictly within an empty region
            (i.e., ``empty_tree.interval.left < x_lim[0] < empty_tree.interval.right``)
            then that tree will not be plotted on the left hand side, and the X axis
            will start at ``empty_tree.interval.right``. Similarly, if x_lim[1] lies
            strictly within an empty region then that tree will not be plotted on the
            right hand side, and the X axis will end at ``empty_tree.interval.left``
        """
        svgtreesequence = drawing.SvgTreeSequence(
            self,
            size,
            x_scale=x_scale,
            time_scale=time_scale,
            tree_height_scale=tree_height_scale,
            title=title,
            node_labels=node_labels,
            mutation_labels=mutation_labels,
            node_titles=node_titles,
            mutation_titles=mutation_titles,
            root_svg_attributes=root_svg_attributes,
            style=style,
            order=order,
            force_root_branch=force_root_branch,
            symbol_size=symbol_size,
            x_axis=x_axis,
            x_label=x_label,
            x_lim=x_lim,
            x_regions=x_regions,
            y_axis=y_axis,
            y_label=y_label,
            y_ticks=y_ticks,
            y_gridlines=y_gridlines,
            omit_sites=omit_sites,
            canvas_size=canvas_size,
            max_num_trees=max_num_trees,
            preamble=preamble,
            **kwargs,
        )
        return svgtreesequence.draw(path)

    def draw_text(
        self,
        *,
        node_labels=None,
        use_ascii=False,
        time_label_format=None,
        position_label_format=None,
        order=None,
        **kwargs,
    ):
        """
        Create a text representation of a tree sequence.

        :param dict node_labels: If specified, show custom labels for the nodes
            that are present in the map. Any nodes not specified in the map will
            not have a node label.
        :param bool use_ascii: If ``False`` (default) then use unicode
            `box drawing characters \
<https://en.wikipedia.org/wiki/Box-drawing_character>`_
            to render the tree. If ``True``, use plain ascii characters, which look
            cruder but are less susceptible to misalignment or font substitution.
            Alternatively, if you are having alignment problems with Unicode, you can try
            out the solution documented `here \
<https://github.com/tskit-dev/tskit/issues/189#issuecomment-499114811>`_.
        :param str time_label_format: A python format string specifying the format (e.g.
            number of decimal places or significant figures) used to print the numerical
            time values on the time axis. If ``None``, this defaults to ``"{:.2f}"``.
        :param str position_label_format: A python format string specifying the format
            (e.g. number of decimal places or significant figures) used to print genomic
            positions. If ``None``, this defaults to ``"{:.2f}"``.
        :param str order: The left-to-right ordering of child nodes in the drawn tree.
            This can be either: ``"minlex"``, which minimises the differences
            between adjacent trees (see also the ``"minlex_postorder"`` traversal
            order for the :meth:`.Tree.nodes` method); or ``"tree"`` which draws trees
            in the left-to-right order defined by the
            :ref:`quintuply linked tree structure <sec_data_model_tree_structure>`.
            If not specified or None, this defaults to ``"minlex"``.

        :return: A text representation of a tree sequence.
        :rtype: str
        """
        return str(
            drawing.TextTreeSequence(
                self,
                node_labels=node_labels,
                use_ascii=use_ascii,
                time_label_format=time_label_format,
                position_label_format=position_label_format,
                order=order,
            )
        )

    ############################################
    #
    # Statistics computation
    #
    ############################################

    def general_stat(
        self,
        W,
        f,
        output_dim,
        windows=None,
        polarised=False,
        mode=None,
        span_normalise=True,
        strict=True,
    ):
        """
        Compute a windowed statistic from weights and a summary function.
        See the :ref:`statistics interface <sec_stats_interface>` section for details on
        :ref:`windows <sec_stats_windows>`,
        :ref:`mode <sec_stats_mode>`,
        :ref:`span normalise <sec_stats_span_normalise>`,
        and :ref:`return value <sec_stats_output_format>`.
        On each tree, this
        propagates the weights ``W`` up the tree, so that the "weight" of each
        node is the sum of the weights of all samples at or below the node.
        Then the summary function ``f`` is applied to the weights, giving a
        summary for each node in each tree. How this is then aggregated depends
        on ``mode``:

        "site"
            Adds together the total summary value across all alleles in each window.

        "branch"
            Adds together the summary value for each node, multiplied by the
            length of the branch above the node and the span of the tree.

        "node"
            Returns each node's summary value added across trees and multiplied
            by the span of the tree.

        Both the weights and the summary can be multidimensional: if ``W`` has ``k``
        columns, and ``f`` takes a ``k``-vector and returns an ``m``-vector,
        then the output will be ``m``-dimensional for each node or window (depending
        on "mode").

        .. note::
            The summary function ``f`` should return zero when given both 0 and
            the total weight (i.e., ``f(0) = 0`` and ``f(np.sum(W, axis=0)) = 0``),
            unless ``strict=False``.  This is necessary for the statistic to be
            unaffected by parts of the tree sequence ancestral to none or all
            of the samples, respectively.

        :param numpy.ndarray W: An array of values with one row for each sample and one
            column for each weight.
        :param f: A function that takes a one-dimensional array of length
            equal to the number of columns of ``W`` and returns a one-dimensional
            array.
        :param int output_dim: The length of ``f``'s return value.
        :param list windows: An increasing list of breakpoints between the windows
            to compute the statistic in.
        :param bool polarised: Whether to leave the ancestral state out of computations:
            see :ref:`sec_stats` for more details.
        :param str mode: A string giving the "type" of the statistic to be computed
            (defaults to "site").
        :param bool span_normalise: Whether to divide the result by the span of the
            window (defaults to True).
        :param bool strict: Whether to check that f(0) and f(total weight) are zero.
        :return: A ndarray with shape equal to (num windows, num statistics).
        """
        if mode is None:
            mode = "site"
        if strict:
            total_weights = np.sum(W, axis=0)
            for x in [total_weights, total_weights * 0.0]:
                with np.errstate(invalid="ignore", divide="ignore"):
                    fx = np.array(f(x))
                fx[np.isnan(fx)] = 0.0
                if not np.allclose(fx, np.zeros((output_dim,))):
                    raise ValueError(
                        "Summary function does not return zero for both "
                        "zero weight and total weight."
                    )
        return self.__run_windowed_stat(
            windows,
            self.ll_tree_sequence.general_stat,
            W,
            f,
            output_dim,
            polarised=polarised,
            span_normalise=span_normalise,
            mode=mode,
        )

    def sample_count_stat(
        self,
        sample_sets,
        f,
        output_dim,
        windows=None,
        polarised=False,
        mode=None,
        span_normalise=True,
        strict=True,
    ):
        """
        Compute a windowed statistic from sample counts and a summary function.
        This is a wrapper around :meth:`.general_stat` for the common case in
        which the weights are all either 1 or 0, i.e., functions of the joint
        allele frequency spectrum.
        See the :ref:`statistics interface <sec_stats_interface>` section for details on
        :ref:`sample sets <sec_stats_sample_sets>`,
        :ref:`windows <sec_stats_windows>`,
        :ref:`mode <sec_stats_mode>`,
        :ref:`span normalise <sec_stats_span_normalise>`,
        and :ref:`return value <sec_stats_output_format>`.
        If ``sample_sets`` is a list of ``k`` sets of samples, then
        ``f`` should be a function that takes an argument of length ``k`` and
        returns a one-dimensional array. The ``j``-th element of the argument
        to ``f`` will be the number of samples in ``sample_sets[j]`` that lie
        below the node that ``f`` is being evaluated for. See
        :meth:`.general_stat`  for more details.

        Here is a contrived example: suppose that ``A`` and ``B`` are two sets
        of samples with ``nA`` and ``nB`` elements, respectively. Passing these
        as sample sets will give ``f`` an argument of length two, giving the number
        of samples in ``A`` and ``B`` below the node in question. So, if we define


        .. code-block:: python

            def f(x):
                pA = x[0] / nA
                pB = x[1] / nB
                return np.array([pA * pB])

        then if all sites are biallelic,

        .. code-block:: python

            ts.sample_count_stat([A, B], f, 1, windows="sites", polarised=False, mode="site")

        would compute, for each site, the product of the derived allele
        frequencies in the two sample sets, in a (num sites, 1) array.  If
        instead ``f`` returns ``np.array([pA, pB, pA * pB])``, then the
        output would be a (num sites, 3) array, with the first two columns
        giving the allele frequencies in ``A`` and ``B``, respectively.

        .. note::
            The summary function ``f`` should return zero when given both 0 and
            the sample size (i.e., ``f(0) = 0`` and
            ``f(np.array([len(x) for x in sample_sets])) = 0``).  This is
            necessary for the statistic to be unaffected by parts of the tree
            sequence ancestral to none or all of the samples, respectively.

        :param list sample_sets: A list of lists of Node IDs, specifying the
            groups of nodes to compute the statistic with.
        :param f: A function that takes a one-dimensional array of length
            equal to the number of sample sets and returns a one-dimensional array.
        :param int output_dim: The length of ``f``'s return value.
        :param list windows: An increasing list of breakpoints between the windows
            to compute the statistic in.
        :param bool polarised: Whether to leave the ancestral state out of computations:
            see :ref:`sec_stats` for more details.
        :param str mode: A string giving the "type" of the statistic to be computed
            (defaults to "site").
        :param bool span_normalise: Whether to divide the result by the span of the
            window (defaults to True).
        :param bool strict: Whether to check that f(0) and f(total weight) are zero.
        :return: A ndarray with shape equal to (num windows, num statistics).
        """  # noqa: B950
        # helper function for common case where weights are indicators of sample sets
        for U in sample_sets:
            if len(U) != len(set(U)):
                raise ValueError(
                    "Elements of sample_sets must be lists without repeated elements."
                )
            if len(U) == 0:
                raise ValueError("Elements of sample_sets cannot be empty.")
            for u in U:
                if not self.node(u).is_sample():
                    raise ValueError("Not all elements of sample_sets are samples.")

        W = np.array([[float(u in A) for A in sample_sets] for u in self.samples()])
        return self.general_stat(
            W,
            f,
            output_dim,
            windows=windows,
            polarised=polarised,
            mode=mode,
            span_normalise=span_normalise,
            strict=strict,
        )

    def parse_windows(self, windows):
        # Note: need to make sure windows is a string or we try to compare the
        # target with a numpy array elementwise.
        if windows is None:
            windows = [0.0, self.sequence_length]
        elif isinstance(windows, str):
            if windows == "trees":
                windows = self.breakpoints(as_array=True)
            elif windows == "sites":
                # breakpoints are at 0.0 and at the sites and at the end
                windows = np.concatenate(
                    [
                        [] if self.num_sites > 0 else [0.0],
                        self.sites_position,
                        [self.sequence_length],
                    ]
                )
                windows[0] = 0.0
            else:
                raise ValueError(
                    f"Unrecognized window specification {windows}:",
                    "the only allowed strings are 'sites' or 'trees'",
                )
        return np.array(windows)

    def parse_time_windows(self, time_windows):
        if time_windows is None:
            time_windows = [0.0, math.inf]
        return np.array(time_windows)

    def __run_windowed_stat(self, windows, method, *args, **kwargs):
        strip_win = windows is None
        windows = self.parse_windows(windows)
        stat = method(*args, **kwargs, windows=windows)
        if strip_win:
            stat = stat[0]
        return stat

    # only for temporary tw version
    def __run_windowed_stat_tw(self, windows, time_windows, method, *args, **kwargs):
        strip_win = windows is None
        strip_timewin = time_windows is None
        windows = self.parse_windows(windows)
        time_windows = self.parse_time_windows(time_windows)
        stat = method(*args, **kwargs, windows=windows, time_windows=time_windows)
        if strip_win and strip_timewin:
            stat = stat[0, 0, :]
        elif strip_win:
            stat = stat[0, :, :]
        elif strip_timewin:
            stat = stat[:, 0, :]
        return stat

    def __one_way_sample_set_stat(
        self,
        ll_method,
        sample_sets,
        windows=None,
        time_windows=None,
        mode=None,
        span_normalise=True,
        polarised=False,
    ):
        if sample_sets is None:
            sample_sets = self.samples()
        # First try to convert to a 1D numpy array. If it is, then we strip off
        # the corresponding dimension from the output.
        drop_dimension = False
        try:
            sample_sets = np.array(sample_sets, dtype=np.uint64)
        except ValueError:
            pass
        else:
            # If we've successfully converted sample_sets to a 1D numpy array
            # of integers then drop the dimension
            if len(sample_sets.shape) == 1:
                sample_sets = [sample_sets]
                drop_dimension = True
        sample_set_sizes = np.array(
            [len(sample_set) for sample_set in sample_sets], dtype=np.uint32
        )
        if np.any(sample_set_sizes == 0):
            raise ValueError("Sample sets must contain at least one element")

        flattened = util.safe_np_int_cast(np.hstack(sample_sets), np.int32)
        # this next line is temporary, while time windows are implemented
        # in other methods
        use_tw = ll_method.__name__ == "allele_frequency_spectrum"
        if use_tw:
            stat = self.__run_windowed_stat_tw(
                windows,
                time_windows,
                ll_method,
                sample_set_sizes,
                flattened,
                mode=mode,
                span_normalise=span_normalise,
                polarised=polarised,
            )
        else:
            stat = self.__run_windowed_stat(
                windows,
                ll_method,
                sample_set_sizes,
                flattened,
                mode=mode,
                span_normalise=span_normalise,
                polarised=polarised,
            )
        if drop_dimension:
            stat = stat.reshape(stat.shape[:-1])
            if stat.shape == () and windows is None and time_windows is None:
                stat = stat[()]
        return stat

    def parse_sites(self, sites):
        row_sites, col_sites = None, None
        if sites is not None:
            if any(not hasattr(a, "__getitem__") or isinstance(a, str) for a in sites):
                raise ValueError("Sites must be a list of lists, tuples, or ndarrays")
            if len(sites) == 2:
                row_sites, col_sites = sites
            elif len(sites) == 1:
                row_sites = col_sites = sites[0]
            else:
                raise ValueError(
                    f"Sites must be a length 1 or 2 list, got a length {len(sites)} list"
                )
        return row_sites, col_sites

    def parse_positions(self, positions):
        row_positions, col_positions = None, None
        if positions is not None:
            if any(
                not hasattr(a, "__getitem__") or isinstance(a, str) for a in positions
            ):
                raise ValueError(
                    "Positions must be a list of lists, tuples, or ndarrays"
                )
            if len(positions) == 2:
                row_positions, col_positions = positions
            elif len(positions) == 1:
                row_positions = col_positions = positions[0]
            else:
                raise ValueError(
                    "Positions must be a length 1 or 2 list, "
                    f"got a length {len(positions)} list"
                )
        return row_positions, col_positions

    def __two_locus_sample_set_stat(
        self,
        ll_method,
        sample_sets,
        sites=None,
        positions=None,
        mode=None,
    ):
        if sample_sets is None:
            sample_sets = self.samples()
        row_sites, col_sites = self.parse_sites(sites)
        row_positions, col_positions = self.parse_positions(positions)

        # First try to convert to a 1D numpy array. If we succeed, then we strip off
        # the corresponding dimension from the output.
        drop_dimension = False
        try:
            sample_sets = np.array(sample_sets, dtype=np.uint64)
        except ValueError:
            pass
        else:
            # If we've successfully converted sample_sets to a 1D numpy array
            # of integers then drop the dimension
            if len(sample_sets.shape) == 1:
                sample_sets = [sample_sets]
                drop_dimension = True

        sample_set_sizes = np.array(
            [len(sample_set) for sample_set in sample_sets], dtype=np.uint32
        )
        if np.any(sample_set_sizes == 0):
            raise ValueError("Sample sets must contain at least one element")

        flattened = util.safe_np_int_cast(np.hstack(sample_sets), np.int32)

        result = ll_method(
            sample_set_sizes,
            flattened,
            row_sites,
            col_sites,
            row_positions,
            col_positions,
            mode,
        )

        if drop_dimension:
            result = result.reshape(result.shape[:2])
        else:
            # Orient the data so that the first dimension is the sample set.
            # With this orientation, we get one LD matrix per sample set.
            result = result.swapaxes(0, 2).swapaxes(1, 2)

        return result

    def __k_way_two_locus_sample_set_stat(
        self,
        ll_method,
        k,
        sample_sets,
        indexes=None,
        sites=None,
        positions=None,
        mode=None,
    ):
        sample_set_sizes = np.array(
            [len(sample_set) for sample_set in sample_sets], dtype=np.uint32
        )
        if np.any(sample_set_sizes == 0):
            raise ValueError("Sample sets must contain at least one element")
        flattened = util.safe_np_int_cast(np.hstack(sample_sets), np.int32)
        row_sites, col_sites = self.parse_sites(sites)
        row_positions, col_positions = self.parse_positions(positions)
        drop_dimension = False
        indexes = util.safe_np_int_cast(indexes, np.int32)
        if len(indexes.shape) == 1:
            indexes = indexes.reshape((1, indexes.shape[0]))
            drop_dimension = True
        if len(indexes.shape) != 2 or indexes.shape[1] != k:
            raise ValueError(
                "Indexes must be convertable to a 2D numpy array with {} "
                "columns".format(k)
            )
        result = ll_method(
            sample_set_sizes,
            flattened,
            indexes,
            row_sites,
            col_sites,
            row_positions,
            col_positions,
            mode,
        )
        if drop_dimension:
            result = result.reshape(result.shape[:2])
        else:
            # Orient the data so that the first dimension is the sample set.
            # With this orientation, we get one LD matrix per sample set.
            result = result.swapaxes(0, 2).swapaxes(1, 2)
        return result

    def __k_way_sample_set_stat(
        self,
        ll_method,
        k,
        sample_sets,
        indexes=None,
        windows=None,
        mode=None,
        span_normalise=True,
        polarised=False,
        centre=True,
    ):
        sample_set_sizes = np.array(
            [len(sample_set) for sample_set in sample_sets], dtype=np.uint32
        )
        if np.any(sample_set_sizes == 0):
            raise ValueError("Sample sets must contain at least one element")
        flattened = util.safe_np_int_cast(np.hstack(sample_sets), np.int32)
        drop_based_on_index = False
        if indexes is None:
            drop_based_on_index = True
            if len(sample_sets) != k:
                raise ValueError(
                    "Must specify indexes if there are not exactly {} sample "
                    "sets.".format(k)
                )
            indexes = np.arange(k, dtype=np.int32)
        drop_dimension = False
        indexes = util.safe_np_int_cast(indexes, np.int32)
        if len(indexes.shape) == 1:
            indexes = indexes.reshape((1, indexes.shape[0]))
            drop_dimension = True
        if len(indexes.shape) != 2 or indexes.shape[1] != k:
            raise ValueError(
                "Indexes must be convertable to a 2D numpy array with {} "
                "columns".format(k)
            )
        stat = self.__run_windowed_stat(
            windows,
            ll_method,
            sample_set_sizes,
            flattened,
            indexes,
            mode=mode,
            span_normalise=span_normalise,
            polarised=polarised,
            centre=centre,
        )
        if drop_dimension:
            stat = stat.reshape(stat.shape[:-1])
            if stat.shape == () and windows is None and drop_based_on_index:
                stat = stat[()]
        return stat

    def __k_way_weighted_stat(
        self,
        ll_method,
        k,
        W,
        indexes=None,
        windows=None,
        mode=None,
        span_normalise=True,
        polarised=False,
        centre=True,
    ):
        W = np.asarray(W)
        if indexes is None:
            if W.shape[1] != k:
                raise ValueError(
                    "Must specify indexes if there are not exactly {} columns "
                    "in W.".format(k)
                )
            indexes = np.arange(k, dtype=np.int32)
        drop_dimension = False
        indexes = util.safe_np_int_cast(indexes, np.int32)
        if len(indexes.shape) == 1:
            indexes = indexes.reshape((1, indexes.shape[0]))
            drop_dimension = True
        if len(indexes.shape) != 2 or indexes.shape[1] != k:
            raise ValueError(
                "Indexes must be convertable to a 2D numpy array with {} "
                "columns".format(k)
            )
        stat = self.__run_windowed_stat(
            windows,
            ll_method,
            W,
            indexes,
            mode=mode,
            span_normalise=span_normalise,
            polarised=polarised,
            centre=centre,
        )
        if drop_dimension:
            stat = stat.reshape(stat.shape[:-1])
        return stat

    def __weighted_vector_stat(
        self,
        ll_method,
        W,
        windows=None,
        mode=None,
        span_normalise=True,
        centre=True,
        nodes=None,
    ):
        W = np.asarray(W)
        if len(W.shape) == 1:
            W = W.reshape(W.shape[0], 1)
        if nodes is None:
            nodes = list(self.samples())
        else:
            if centre:
                raise ValueError("If `nodes` is provided, must have centre=False.")
        try:
            nodes = util.safe_np_int_cast(nodes, np.int32)
        except Exception:
            raise ValueError("Could not interpret `nodes` as a list of node IDs.")
        stat = self.__run_windowed_stat(
            windows,
            ll_method,
            W,
            mode=mode,
            span_normalise=span_normalise,
            centre=centre,
            nodes=nodes,
        )
        return stat

    ############################################
    # Statistics definitions
    ############################################

    def diversity(
        self, sample_sets=None, windows=None, mode="site", span_normalise=True
    ):
        """
        Computes mean genetic diversity (also known as "pi") in each of the
        sets of nodes from ``sample_sets``.  The statistic is also known as
        "sample heterozygosity"; a common citation for the definition is
        `Nei and Li (1979) <https://doi.org/10.1073/pnas.76.10.5269>`_
        (equation 22), so it is sometimes called called "Nei's pi"
        (but also sometimes "Tajima's pi").

        Please see the :ref:`one-way statistics <sec_stats_sample_sets_one_way>`
        section for details on how the ``sample_sets`` argument is interpreted
        and how it interacts with the dimensions of the output array.
        See the :ref:`statistics interface <sec_stats_interface>` section for details on
        :ref:`windows <sec_stats_windows>`,
        :ref:`mode <sec_stats_mode>`,
        :ref:`span normalise <sec_stats_span_normalise>`,
        and :ref:`return value <sec_stats_output_format>`.

        Note that this quantity can also be computed by the
        :meth:`divergence <.TreeSequence.divergence>` method.

        What is computed depends on ``mode``:

        "site"
            Mean pairwise genetic diversity: the average over all n choose 2 pairs of
            sample nodes, of the density of sites at
            which the two carry different alleles, per unit of chromosome length.

        "branch"
            Mean distance in the tree: the average across over all n choose 2 pairs of
            sample nodes and locations in the window, of the mean distance in
            the tree between the two samples (in units of time).

        "node"
            For each node, the proportion of genome on which the node is an ancestor to
            only one of a pair of sample nodes from the sample set, averaged
            over over all n choose 2 pairs of sample nodes.

        :param list sample_sets: A list of lists of Node IDs, specifying the
            groups of nodes for which the statistic is computed. If any of the
            sample sets contain only a single node, the returned diversity will be
            NaN. If ``None`` (default), average over all n choose 2 pairs of distinct
            sample nodes in the tree sequence.
        :param list windows: An increasing list of breakpoints between the windows
            to compute the statistic in.
        :param str mode: A string giving the "type" of the statistic to be computed
            (defaults to "site").
        :param bool span_normalise: Whether to divide the result by the span of the
            window (defaults to True).
        :return: A numpy array whose length is equal to the number of sample sets.
            If there is one sample set and windows=None, a numpy scalar is returned.
        """
        return self.__one_way_sample_set_stat(
            self._ll_tree_sequence.diversity,
            sample_sets,
            windows=windows,
            mode=mode,
            span_normalise=span_normalise,
        )

    def divergence(
        self, sample_sets, indexes=None, windows=None, mode="site", span_normalise=True
    ):
        r"""
        Computes mean genetic divergence between (and within) pairs of
        sets of nodes from ``sample_sets``.
        This is the "average number of differences", usually referred to as "dxy";
        a common citation for this definition is Nei and Li (1979), who called it
        :math:`\pi_{XY}`. Note that the mean pairwise nucleotide diversity of a
        sample set to itself (computed by passing an index of the form (j,j))
        is its :meth:`diversity <.TreeSequence.diversity>` (see the note below).

        Operates on ``k = 2`` sample sets at a time; please see the
        :ref:`multi-way statistics <sec_stats_sample_sets_multi_way>`
        section for details on how the ``sample_sets`` and ``indexes`` arguments are
        interpreted and how they interact with the dimensions of the output array.
        See the :ref:`statistics interface <sec_stats_interface>` section for details on
        :ref:`windows <sec_stats_windows>`,
        :ref:`mode <sec_stats_mode>`,
        :ref:`span normalise <sec_stats_span_normalise>`,
        and :ref:`return value <sec_stats_output_format>`.

        .. note::
            To avoid unexpected results, sample sets should be nonoverlapping,
            since comparisons of individuals to themselves are not removed when computing
            divergence between distinct sample sets. (However, specifying an index
            ``(j, j)`` computes the :meth:`diversity <.TreeSequence.diversity>`
            of ``sample_set[j]``, which removes self comparisons to provide
            an unbiased estimate.)

        What is computed depends on ``mode``:

        "site"
            Mean pairwise genetic divergence: the average across every possible pair of
            chromosomes (one from each sample set), of the density of sites at which
            the two carry different alleles, per unit of chromosome length.

        "branch"
            Mean distance in the tree: the average across every possible pair of
            chromosomes (one from each sample set) and locations in the window, of
            the mean distance in the tree between the two samples (in units of time).

        "node"
            For each node, the proportion of genome on which the node is an ancestor to
            only one of a pair of chromosomes from the sample set, averaged
            over all possible pairs.

        :param list sample_sets: A list of lists of Node IDs, specifying the
            groups of nodes to compute the statistic with.
        :param list indexes: A list of 2-tuples, or None.
        :param list windows: An increasing list of breakpoints between the windows
            to compute the statistic in.
        :param str mode: A string giving the "type" of the statistic to be computed
            (defaults to "site").
        :param bool span_normalise: Whether to divide the result by the span of the
            window (defaults to True).
        :return: A ndarray with shape equal to (num windows, num statistics).
            If there is one pair of sample sets and windows=None, a numpy scalar is
            returned.

        """
        return self.__k_way_sample_set_stat(
            self._ll_tree_sequence.divergence,
            2,
            sample_sets,
            indexes=indexes,
            windows=windows,
            mode=mode,
            span_normalise=span_normalise,
        )

    ############################################
    # Pairwise sample x sample statistics
    ############################################

    def _chunk_sequence_by_tree(self, num_chunks):
        """
        Return list of (left, right) genome interval tuples that contain
        approximately equal numbers of trees as a 2D numpy array. A
        maximum of self.num_trees single-tree intervals can be returned.
        """
        if num_chunks <= 0 or int(num_chunks) != num_chunks:
            raise ValueError("Number of chunks must be an integer > 0")
        num_chunks = min(self.num_trees, num_chunks)
        breakpoints = self.breakpoints(as_array=True)[:-1]
        splits = np.array_split(breakpoints, num_chunks)
        chunks = []
        for j in range(num_chunks - 1):
            chunks.append((splits[j][0], splits[j + 1][0]))
        chunks.append((splits[-1][0], self.sequence_length))
        return chunks

    @staticmethod
    def _chunk_windows(windows, num_chunks):
        """
        Returns a list of (at most) num_chunks windows, which represent splitting
        up the specified list of windows into roughly equal work.

        Currently this is implemented by just splitting up into roughly equal
        numbers of windows in each chunk.
        """
        if num_chunks <= 0 or int(num_chunks) != num_chunks:
            raise ValueError("Number of chunks must be an integer > 0")
        num_chunks = min(len(windows) - 1, num_chunks)
        splits = np.array_split(windows[:-1], num_chunks)
        chunks = []
        for j in range(num_chunks - 1):
            chunk = np.append(splits[j], splits[j + 1][0])
            chunks.append(chunk)
        chunk = np.append(splits[-1], windows[-1])
        chunks.append(chunk)
        return chunks

    def _parallelise_divmat_by_tree(self, num_threads, span_normalise, **kwargs):
        """
        No windows were specified, so we can chunk up the whole genome by
        tree, and do a simple sum of the results. This means that we have to
        handle span_normalise specially, though.
        """

        def worker(interval):
            return self._ll_tree_sequence.divergence_matrix(interval, **kwargs)

        work = self._chunk_sequence_by_tree(num_threads)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
            results = pool.map(worker, work)
        total = sum(results)
        if span_normalise:
            total /= self.sequence_length
        return total

    def _parallelise_divmat_by_window(self, windows, num_threads, **kwargs):
        """
        We assume we have a number of windows that's >= to the number
        of threads available, and let each thread have a chunk of the
        windows. There will definitely cases where this leads to
        pathological behaviour, so we may need a more sophisticated
        strategy at some point.
        """

        def worker(sub_windows):
            return self._ll_tree_sequence.divergence_matrix(sub_windows, **kwargs)

        work = self._chunk_windows(windows, num_threads)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, sub_windows) for sub_windows in work]
            concurrent.futures.wait(futures)
        return np.vstack([future.result() for future in futures])

    @staticmethod
    def _parse_stat_matrix_sample_sets(ids):
        """
        Returns a flattened list of sets of IDs. If ids is a 1D list,
        interpret as n one-element sets. Otherwise, it must be a sequence
        of ID lists.
        """
        id_dtype = np.int32
        size_dtype = np.uint64
        # Exclude some types that could be specified accidentally, and
        # we may want to reserve for future use.
        if isinstance(ids, (str, bytes, collections.abc.Mapping, numbers.Number)):
            raise TypeError(f"ID specification cannot be a {type(ids)}")
        if len(ids) == 0:
            return np.array([], dtype=id_dtype), np.array([], dtype=size_dtype)
        if isinstance(ids[0], numbers.Number):
            # Interpret as a 1D array
            flat = util.safe_np_int_cast(ids, id_dtype)
            sizes = np.ones(len(flat), dtype=size_dtype)
        else:
            set_lists = []
            sizes = []
            for id_list in ids:
                a = util.safe_np_int_cast(id_list, id_dtype)
                if len(a.shape) != 1:
                    raise ValueError("ID sets must be 1D integer arrays")
                set_lists.append(a)
                sizes.append(len(a))
            flat = np.hstack(set_lists)
            sizes = np.array(sizes, dtype=size_dtype)
        return flat, sizes

    # def divergence_matrix(self, sample_sets, windows=None, mode="site"):
    #     """
    #     Finds the mean divergence  between pairs of samples from each set of
    #     samples and in each window. Returns a numpy array indexed by (window,
    #     sample_set, sample_set).  Diagonal entries are corrected so that the
    #     value gives the mean divergence for *distinct* samples, but it is not
    #     checked whether the sample_sets are disjoint (so offdiagonals are not
    #     corrected).  For this reason, if an element of `sample_sets` has only
    #     one element, the corresponding diagonal will be NaN.

    #     The mean divergence between two samples is defined to be the mean: (as
    #     a TreeStat) length of all edges separating them in the tree, or (as a
    #     SiteStat) density of segregating sites, at a uniformly chosen position
    #     on the genome.

    #     :param list sample_sets: A list of sets of IDs of samples.
    #     :param iterable windows: The breakpoints of the windows (including start
    #         and end, so has one more entry than number of windows).
    #     :return: A list of the upper triangle of mean TMRCA values in row-major
    #         order, including the diagonal.
    #     """
    #     ns = len(sample_sets)
    #     indexes = [(i, j) for i in range(ns) for j in range(i, ns)]
    #     x = self.divergence(sample_sets, indexes, windows, mode=mode)
    #     nw = len(windows) - 1
    #     A = np.ones((nw, ns, ns), dtype=float)
    #     for w in range(nw):
    #         k = 0
    #         for i in range(ns):
    #             for j in range(i, ns):
    #                 A[w, i, j] = A[w, j, i] = x[w][k]
    #                 k += 1
    #     return A
    # NOTE: see older definition of divmat here, which may be useful when documenting
    # this function. See https://github.com/tskit-dev/tskit/issues/2781

    # NOTE for documentation of sample_sets. We *must* use samples currently because
    # the normalisation for non-sample nodes is tricky. Do we normalise by the
    # total span of the ts where the node is 'present' in the tree? We avoid this
    # by insisting on sample nodes.

    # NOTE for documentation of num_threads. Need to explain that the
    # its best to think of as the number of background *worker* threads.
    # default is to run without any worker threads. If you want to run
    # with all the cores on the machine, use num_threads=os.cpu_count().

    def divergence_matrix(
        self,
        sample_sets=None,
        *,
        windows=None,
        num_threads=0,
        mode=None,
        span_normalise=True,
    ):
        windows_specified = windows is not None
        windows = self.parse_windows(windows)
        mode = "site" if mode is None else mode

        if sample_sets is None:
            sample_sets = self.samples()
            flattened_samples = self.samples()
            sample_set_sizes = np.ones(len(sample_sets), dtype=np.uint32)
        else:
            flattened_samples, sample_set_sizes = self._parse_stat_matrix_sample_sets(
                sample_sets
            )

        # FIXME this logic should be merged into __run_windowed_stat if
        # we generalise the num_threads argument to all stats.
        if num_threads <= 0:
            D = self._ll_tree_sequence.divergence_matrix(
                windows,
                sample_sets=flattened_samples,
                sample_set_sizes=sample_set_sizes,
                mode=mode,
                span_normalise=span_normalise,
            )
        else:
            if windows_specified:
                D = self._parallelise_divmat_by_window(
                    windows,
                    num_threads,
                    sample_sets=flattened_samples,
                    sample_set_sizes=sample_set_sizes,
                    mode=mode,
                    span_normalise=span_normalise,
                )
            else:
                D = self._parallelise_divmat_by_tree(
                    num_threads,
                    span_normalise=span_normalise,
                    sample_sets=flattened_samples,
                    sample_set_sizes=sample_set_sizes,
                    mode=mode,
                )

        if not windows_specified:
            # Drop the windows dimension
            D = D[0]
        return D

    def genetic_relatedness(
        self,
        sample_sets,
        indexes=None,
        windows=None,
        mode="site",
        span_normalise=True,
        polarised=True,
        proportion=True,
        centre=True,
    ):
        """
        Computes genetic relatedness between (and within) pairs of
        sets of nodes from ``sample_sets``.
        Operates on ``k = 2`` sample sets at a time; please see the
        :ref:`multi-way statistics <sec_stats_sample_sets_multi_way>`
        section for details on how the ``sample_sets`` and ``indexes`` arguments are
        interpreted and how they interact with the dimensions of the output array.
        See the :ref:`statistics interface <sec_stats_interface>` section for details on
        :ref:`windows <sec_stats_windows>`,
        :ref:`mode <sec_stats_mode>`,
        :ref:`span normalise <sec_stats_span_normalise>`,
        :ref:`polarised <sec_stats_polarisation>`,
        and :ref:`return value <sec_stats_output_format>`.

        What is computed depends on ``mode``:

        "site"
            Frequency of pairwise allelic matches in the window between two
            sample sets relative to the rest of the sample sets. To be precise,
            let `m(u,v)` denote the total number of alleles shared between
            nodes `u` and `v`, and let `m(I,J)` be the average of `m(u,v)` over
            all nodes `u` in sample set `I` and `v` in sample set `J`. Let `S`
            and `T` be independently chosen sample sets. Then, for sample sets
            `I` and `J`, this computes `E[m(I,J) - m(I,S) - m(J,T) + m(S,T)]`
            if centre=True (the default), or `E[m(I,J)]` if centre=False.
            This can also be seen as the covariance of a quantitative trait
            determined by additive contributions from the genomes in each
            sample set. Let each derived allele be associated with an effect
            drawn from a `N(0,1)` distribution, and let the trait value of a
            sample be the sum of its allele effects. Then, this computes
            the covariance between the average trait values of two sample sets.
            For example, to compute covariance between the traits of diploid
            individuals, each sample set would be the pair of genomes of each
            individual, with the trait being the average of the two genomes.
            If ``proportion=True``, this then corresponds to :math:`K_{c0}` in
            `Speed & Balding (2014) <https://www.nature.com/articles/nrg3821>`_,
            multiplied by four (see below).

        "branch"
            Average area of branches in the window ancestral to pairs of samples
            in two sample sets relative to the rest of the sample sets. To be
            precise, let `B(u,v)` denote the total area of all branches
            ancestral to nodes `u` and `v`, and let `B(I,J)` be the average of
            `B(u,v)` over all nodes `u` in sample set `I` and `v` in sample set
            `J`. Let `S` and `T` be two independently chosen sample sets. Then
            for sample sets `I` and `J`, this computes
            `E[B(I,J) - B(I,S) - B(J,T) + B(S,T)]` if centre=True (the default),
            or `E[B(I,J)]` if centre=False.

        "node"
            For each node, the proportion of the window over which pairs of
            samples in two sample sets are descendants, relative to the rest of
            the sample sets. To be precise, for each node `n`, let `N(u,v)`
            denote the proportion of the window over which samples `u` and `v`
            are descendants of `n`, and let and let `N(I,J)` be the average of
            `N(u,v)` over all nodes `u` in sample set `I` and `v` in sample set
            `J`. Let `S` and `T` be two independently chosen sample sets. Then
            for sample sets `I` and `J`, this computes
            `E[N(I,J) - N(I,S) - N(J,T) + N(S,T)]` if centre=True (the default),
            or `E[N(I,J)]` if centre=False.

        *Note:* The default for this statistic - unlike most other statistics - is
        ``polarised=True``. Using the default value ``centre=True``, setting
        ``polarised=False`` will only multiply the result by a factor of two
        for branch-mode, or site-mode if all sites are biallelic. (With
        multiallelic sites the difference is more complicated.) The uncentred
        and unpolarised value is probably not what you are looking for: for
        instance, the unpolarised, uncentred site statistic between two samples
        counts the number of alleles inherited by both *and* the number of
        alleles inherited by neither of the two samples.

        *Note:* Some authors
        (see `Speed & Balding (2014) <https://www.nature.com/articles/nrg3821>`_)
        compute relatedness between `I` and `J` as the total number of all pairwise
        allelic matches between `I` and `J`, rather than the frequency,
        which would define `m(I,J)` as the sum of `m(u,v)` rather than the average
        in the definition of "site" relatedness above. If every sample set is the
        samples of a :math:`k`-ploid individual, this would simply multiply the
        result by :math:`k^2`. However, this definition would make the result not
        useful as a summary statistic of typical relatedness for larger sample
        sets.

        :param list sample_sets: A list of lists of Node IDs, specifying the
            groups of nodes to compute the statistic with.
        :param list indexes: A list of 2-tuples, or None.
        :param list windows: An increasing list of breakpoints between the windows
            to compute the statistic in.
        :param str mode: A string giving the "type" of the statistic to be computed
            (defaults to "site").
        :param bool span_normalise: Whether to divide the result by the span of the
            window (defaults to True). Has no effect if ``proportion`` is True.
        :param bool proportion: Defaults to True.  Whether to divide the result by
            :meth:`.segregating_sites`, called with the same ``windows``,
            ``mode``, and ``span_normalise``. Note that this counts sites
            that are segregating between *any* of the samples of *any* of the
            sample sets (rather than segregating between all of the samples of
            the tree sequence).
        :param bool polarised: Whether to leave the ancestral state out of computations:
            see :ref:`sec_stats` for more details. Defaults to True.
        :param bool centre: Defaults to True. Whether to 'centre' the result, as
            described above (the usual definition is centred).
        :return: A ndarray with shape equal to (num windows, num statistics).
            If there is one pair of sample sets and windows=None, a numpy scalar is
            returned.
        """

        out = self.__k_way_sample_set_stat(
            self._ll_tree_sequence.genetic_relatedness,
            2,
            sample_sets,
            indexes=indexes,
            windows=windows,
            mode=mode,
            span_normalise=span_normalise,
            polarised=polarised,
            centre=centre,
        )
        if proportion:
            # TODO this should be done in C also
            all_samples = np.array(list({u for s in sample_sets for u in s}))
            denominator = self.segregating_sites(
                sample_sets=all_samples,
                windows=windows,
                mode=mode,
                span_normalise=span_normalise,
            )
            # the shapes of out and denominator should be the same except that
            # out may have an extra dimension if indexes is not None
            if indexes is not None and not isinstance(denominator, float):
                oshape = list(out.shape)
                oshape[-1] = 1
                denominator = denominator.reshape(oshape)
            with np.errstate(divide="ignore", invalid="ignore"):
                out /= denominator

        return out

    def genetic_relatedness_matrix(
        self,
        sample_sets=None,
        *,
        windows=None,
        num_threads=0,
        mode=None,
        span_normalise=True,
    ):
        """
        Computes the full matrix of pairwise genetic relatedness values
        between (and within) pairs of sets of nodes from ``sample_sets``.
        *Warning:* this does not compute exactly the same thing as
        :meth:`.genetic_relatedness`: see below for more details.

        If `mode="branch"`, then the value obtained is the same as that from
        :meth:`.genetic_relatedness`, using the options `centre=True` and
        `proportion=False`. The same is true if `mode="site"` and all sites have
        at most one mutation.

        However, if some sites have more than one mutation, the value may differ.
        The reason is that this function (for efficiency) computes relatedness
        using :meth:`.divergence` and the following relationship.
        "Relatedness" measures the number of *shared* alleles (or branches),
        while "divergence" measures the number of *non-shared* alleles (or branches).
        Let :math:`T_i` be the total distance from sample :math:`i` up to the root;
        then if :math:`D_{ij}` is the divergence between :math:`i` and :math:`j`
        and :math:`R_{ij}` is the relatedness between :math:`i` and :math:`j`, then
        :math:`T_i + T_j = D_{ij} + 2 R_{ij}.`
        So, for any samples :math:`I`, :math:`J`, :math:`S`, :math:`T`
        (that may now be random choices),
        :math:`R_{IJ}-R_{IS}-R_{JT}+R_{ST} = (D_{IJ}-D_{IS}-D_{JT}+D_{ST})/ (-2)`.
        Note, however, that this relationship only holds for `mode="site"`
        if we can treat "number of differing alleles" as distances on the tree;
        this is not necessarily the case in the presence of multiple mutations.

        Another caveat in the above relationship between :math:`R` and :math:`D`
        is that :meth:`.divergence` of a sample set to itself does not include
        the "self" comparisons (so as to provide an unbiased estimator of a
        population quantity), while the usual definition of genetic relatedness
        *does* include such comparisons (to provide, for instance, an appropriate
        value for prospective results beginning with only a given set of
        individuals).

        :param list sample_sets: A list of lists of Node IDs, specifying the
            groups of nodes to compute the statistic with.
        :param list windows: An increasing list of breakpoints between the windows
            to compute the statistic in.
        :param str mode: A string giving the "type" of the statistic to be computed
            (defaults to "site").
        :param bool span_normalise: Whether to divide the result by the span of the
            window (defaults to True). Has no effect if ``proportion`` is True.
        :return: A ndarray with shape equal to (num windows, num statistics).
            If there is one pair of sample sets and windows=None, a numpy scalar is
            returned.
        """
        D = self.divergence_matrix(
            sample_sets,
            windows=windows,
            num_threads=num_threads,
            mode=mode,
            span_normalise=span_normalise,
        )

        if sample_sets is None:
            n = np.ones(self.num_samples)
        else:
            n = np.array([len(x) for x in sample_sets])

        def _normalise(B):
            if len(B) == 0:
                return B
            # correct for lack of self comparisons in divergence
            np.fill_diagonal(B, np.diag(B) * (n - 1) / n)
            K = B + np.mean(B)
            y = np.mean(B, axis=0)
            X = y[:, np.newaxis] + y[np.newaxis, :]
            K -= X
            return K / -2

        if windows is None:
            return _normalise(D)
        else:
            for j in range(D.shape[0]):
                D[j] = _normalise(D[j])
        return D

    def genetic_relatedness_weighted(
        self,
        W,
        indexes=None,
        windows=None,
        mode="site",
        span_normalise=True,
        polarised=False,
        centre=True,
    ):
        r"""
        Computes weighted genetic relatedness. If the :math:`k` th pair of indices
        is (i, j) then the :math:`k` th column of output will be
        :math:`\sum_{a,b} W_{ai} W_{bj} C_{ab}`,
        where :math:`W` is the matrix of weights, and :math:`C_{ab}` is the
        :meth:`genetic_relatedness <.TreeSequence.genetic_relatedness>` between sample
        a and sample b, summing over all pairs of samples in the tree sequence.

        *Note:* the genetic relatedness matrix :math:`C` here is as returned by
        :meth:`.genetic_relatedness`, rather than by :meth:`.genetic_relatedness_matrix`
        (see the latter's documentation for the difference).

        :param numpy.ndarray W: An array of values with one row for each sample node and
            one column for each set of weights.
        :param list indexes: A list of 2-tuples, or None (default). Note that if
            indexes = None, then W must have exactly two columns and this is equivalent
            to indexes = [(0,1)].
        :param list windows: An increasing list of breakpoints between the windows
            to compute the statistic in.
        :param str mode: A string giving the "type" of the statistic to be computed
            (defaults to "site").
        :param bool span_normalise: Whether to divide the result by the span of the
            window (defaults to True).
        :param bool polarised: Whether to leave the ancestral state out of computations:
            see :ref:`sec_stats` for more details. Defaults to True.
        :param bool centre: Defaults to True. Whether to 'centre' the result, as
            described above (the usual definition is centred).
        :return: A ndarray with shape equal to (num windows, num statistics).
        """
        if len(W) != self.num_samples:
            raise ValueError(
                "First trait dimension must be equal to number of samples."
            )
        return self.__k_way_weighted_stat(
            self._ll_tree_sequence.genetic_relatedness_weighted,
            2,
            W,
            indexes=indexes,
            windows=windows,
            mode=mode,
            span_normalise=span_normalise,
            polarised=polarised,
            centre=centre,
        )

    def genetic_relatedness_vector(
        self,
        W,
        windows=None,
        mode="site",
        span_normalise=True,
        centre=True,
        nodes=None,
    ):
        r"""
        Computes the product of the genetic relatedness matrix and a vector of weights
        (one per sample). The output is a (num windows) x (num samples) x (num weights)
        array whose :math:`(w,i,j)`-th element is :math:`\sum_{b} W_{bj} C_{ib}`,
        where :math:`W` is the matrix of weights, and :math:`C_{ab}` is the
        :meth:`genetic_relatedness <.TreeSequence.genetic_relatedness>` between sample
        `a` and sample `b` in window `w`, and the sum is over all samples in the tree
        sequence.  Like other statistics, if windows is None, the first dimension in
        the output is dropped.

        The relatedness used here corresponds to `polarised=True`; no unpolarised option
        is available for this method.

        Optionally, you may provide a list of focal nodes that modifies the behavior
        as follows. If `nodes` is a list of `n` node IDs (that do not need to be
        samples), then the output will have dimension (num windows) x n x (num weights),
        and the matrix :math:`C` used in the definition above is the rectangular matrix
        with :math:`C_{ij}` the relatedness between `nodes[i]` and `samples[j]`. This
        can only be used with `centre=False`; if relatedness between uncentred nodes
        and centred samples is desired, then simply subtract column means from `W` first.
        The default is `nodes=None`, which is equivalent to setting `nodes` equal to
        `ts.samples()`.

        :param numpy.ndarray W: An array of values with one row for each sample node and
            one column for each set of weights.
        :param list windows: An increasing list of breakpoints between the windows
            to compute the statistic in.
        :param str mode: A string giving the "type" of the statistic to be computed
            (defaults to "site").
        :param bool span_normalise: Whether to divide the result by the span of the
            window (defaults to True).
        :param bool centre: Whether to use the *centred* relatedness matrix or not:
            see :meth:`genetic_relatedness <.TreeSequence.genetic_relatedness>`.
        :param list nodes: Optionally, a list of focal nodes as described above
            (default: None).
        :return: A ndarray with shape equal to (num windows, num samples, num weights),
            or (num samples, num weights) if windows is None.
        """
        if (not hasattr(W, "__len__")) or (len(W) != self.num_samples):
            raise ValueError(
                "First weight dimension must be equal to number of samples."
            )

        out = self.__weighted_vector_stat(
            self._ll_tree_sequence.genetic_relatedness_vector,
            W,
            windows=windows,
            mode=mode,
            span_normalise=span_normalise,
            centre=centre,
            nodes=nodes,
        )
        return out

    def _expand_indices(self, x: np.ndarray, indices: np.ndarray) -> np.ndarray:
        y = np.zeros((self.num_samples, x.shape[1]))
        y[indices] = x

        return y

    def _genetic_relatedness_vector_node(
        self,
        arr: np.ndarray,
        indices: np.ndarray,
        mode: str,
        centre: bool = True,
        windows=None,
    ) -> np.ndarray:
        x = arr - arr.mean(axis=0) if centre else arr
        x = self._expand_indices(x, indices)
        x = self.genetic_relatedness_vector(
            W=x,
            windows=windows,
            mode=mode,
            centre=False,
            nodes=indices,
        )[0]
        x = x - x.mean(axis=0) if centre else x

        return x

    def _genetic_relatedness_vector_individual(
        self,
        arr: np.ndarray,
        indices: np.ndarray,
        mode: str,
        centre: bool = True,
        windows=None,
    ) -> np.ndarray:
        ij = np.vstack(
            [[n, k] for k, i in enumerate(indices) for n in self.individual(i).nodes]
        )
        samples, sample_individuals = (
            ij[:, 0],
            ij[:, 1],
        )
        ploidy = np.bincount(sample_individuals)
        x = arr - arr.mean(axis=0) if centre else arr
        x = x[sample_individuals] / ploidy[sample_individuals, np.newaxis]
        x = self._expand_indices(x, samples)
        x = self.genetic_relatedness_vector(
            W=x,
            windows=windows,
            mode=mode,
            centre=False,
            nodes=samples,
        )[0]

        def bincount_fn(w):
            return np.bincount(sample_individuals, w) / ploidy

        x = np.apply_along_axis(bincount_fn, axis=0, arr=x)
        x = x - x.mean(axis=0) if centre else x  # centering within index in cols

        return x

    def pca(
        self,
        num_components: int,
        windows: list = None,
        samples: np.ndarray = None,
        individuals: np.ndarray = None,
        time_windows: np.ndarray = None,
        mode: str = "branch",
        centre: bool = True,
        num_iterations: int = 5,
        num_oversamples: int = None,
        random_seed: int = None,
        range_sketch: np.ndarray = None,
    ) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Performs principal component analysis (PCA) for a given set of samples or
        individuals (default: all samples). The principal components are the
        eigenvectors of the genetic relatedness matrix, which are obtained by a
        randomized singular value decomposition (rSVD) algorithm.

        Concretely, if :math:`M` is the matrix of genetic relatedness values, with
        :math:`M_{ij}` the output of
        :meth:`genetic_relatedness <.TreeSequence.genetic_relatedness>`
        between sample :math:`i` and sample :math:`j`, then by default this returns
        the top ``num_components`` eigenvectors of :math:`M`, so that
        ``output.factors[i,k]`` is the position of sample `i` on the `k` th PC.
        If ``samples`` or ``individuals`` are provided, then this does the same thing,
        except with :math:`M_{ij}` either the relatedness between ``samples[i]``
        and ``samples[j]`` or the nodes of ``individuals[i]`` and ``individuals[j]``,
        respectively.

        The parameters ``centre`` and ``mode`` are passed to
        :meth:`genetic_relatedness <.TreeSequence.genetic_relatedness>`;
        if ``windows`` are provided then PCA is carried out separately in each window.
        If ``time_windows`` is provided, then genetic relatedness is measured using only
        ancestral material within the given time window (see
        :meth:`decapitate <.TreeSequence.decapitate>` for how this is defined).

        So that the method scales to large tree sequences, the underlying method
        relies on a randomized SVD algorithm, using
        :meth:`genetic_relatedness_vector <.TreeSequence.genetic_relatedness_vector>`).
        Larger values of ``num_iterations`` and
        ``num_oversamples`` should produce better approximations to the true eigenvalues,
        at the cost of greater compute times and/or memory usage. The method relies on
        constructing ``range_sketch``, a low-dimensional approximation to the range
        of :math:`M`, so that the result of a previous call to ``pca()`` may be passed
        in.

        To check for convergence, compare
        ``pc1 = ts.pca()`` and ``pc2 = ts.pca(range_sketch=pc1.range_sketch)``; the
        difference between ``pc1.factors`` and ``pc2.factors`` provides a
        diagnostic of the convergence of the algorithm (i.e., if they are close
        then it has likely converged). Alternatively, the output value of ``error_bound``
        gives an approximate upper bound for the spectral norm of the difference
        between :math:`M` and the projection of :math:`M` into the space spanned by
        the columns of ``range_sketch``.
        Algorithms are based on Algorithms 8
        and 9 in Martinsson and Tropp, https://arxiv.org/pdf/2002.01387 .

        :param int num_components: Number of principal components to return.
        :param list windows: An increasing list of breakpoints between the windows
            to compute the statistic in (default: the entire genome).
        :param numpy.ndarray samples: Samples to perform PCA with (default: all samples).
        :param numpy.ndarray individuals: Individuals to perform PCA with. Cannot specify
            both ``samples`` and ``individuals``.
        :param numpy.ndarray time_windows: The time interval on which to apply PCA:
            currently, this must be either None (default, covers all time)
            or a single interval.
        :param str mode: A string giving the "type" of relatedness to be computed
            (defaults to "branch"; see
            :meth:`genetic_relatedness_vector
            <.TreeSequence.genetic_relatedness_vector>`).
        :param bool centre: Whether to centre the genetic relatedness matrix.
        :param int num_iterations: Number of power iterations used in the range finding
            algorithm.
        :param int num_oversamples: Number of additional test vectors (default: 10).
            Cannot specify along with range_sketch.
        :param int random_seed: The random seed. If this is None, a random seed will
            be automatically generated. Valid random seeds are between 1 and
            :math:`2^32 − 1`. Only used if `range_sketch` is not provided.
        :param numpy.ndarray range_sketch: Sketch matrix for each window. Default is
            randomly generated; cannot specify along with num_oversamples.
        :return: A :class:`PCAResult` object, containing estimated principal components,
            eigenvalues, and other information:
            the principal component loadings are in PCAResult.factors
            and the principal values are in PCAResult.eigenvalues.
        """

        if (not isinstance(num_iterations, int)) or num_iterations < 1:
            raise ValueError("num_iterations should be a positive integer.")

        if samples is None and individuals is None:
            samples = self.samples()

        if samples is not None and individuals is not None:
            raise ValueError("Samples and individuals cannot be used at the same time")
        elif samples is not None:
            output_type = "node"
            dim = len(samples)
        else:
            assert individuals is not None
            output_type = "individual"
            dim = len(individuals)

        if time_windows is None:
            tree_sequence_low, tree_sequence_high = None, self
        else:
            assert (
                time_windows[0] < time_windows[1]
            ), "The second argument should be larger."
            tree_sequence_low, tree_sequence_high = (
                self.decapitate(time_windows[0]),
                self.decapitate(time_windows[1]),
            )

        drop_windows = windows is None
        windows = self.parse_windows(windows)
        num_windows = len(windows) - 1
        if num_windows < 1:
            raise ValueError("Must have at least one window.")

        if num_components > dim:
            raise ValueError(
                "Number of components must be less than or equal to "
                "the number of samples (or individuals, if specified)."
            )

        if num_oversamples is not None and range_sketch is not None:
            raise ValueError("Cannot specify both num_oversamples and range_sketch.")

        if range_sketch is None:
            if num_oversamples is None:
                num_oversamples = min(10, dim - num_components)

            rng = np.random.default_rng(random_seed)
            range_sketch = rng.normal(
                size=(num_windows, dim, num_components + num_oversamples)
            )
        else:
            if drop_windows:
                range_sketch = np.expand_dims(range_sketch, 0)
            if range_sketch.shape[-1] < num_components:
                raise ValueError(
                    "range_sketch must have at least as many columns as num_components"
                )
            num_oversamples = range_sketch.shape[-1] - num_components

        num_vectors = num_components + num_oversamples
        if num_vectors > dim:
            raise ValueError(
                "Number of columns in range_sketch "
                "(num_components + num_oversamples) must be less"
                " than or equal to the number of samples"
                " (or individuals, if specified)."
            )
        rs_exp_dims = (num_windows, dim, num_vectors)
        rs_obs_dims = range_sketch.shape
        if rs_obs_dims != rs_exp_dims:
            if drop_windows:
                rs_obs_dims = rs_obs_dims[1:]
                rs_exp_dims = rs_exp_dims[1:]
            raise ValueError(
                "Incorrect shape of range_sketch:"
                f" expected {rs_exp_dims}; got {rs_obs_dims}."
            )

        def _rand_pow_range_finder(
            operator,
            operator_dim: int,
            rank: int,
            depth: int,
            num_vectors: int,
            Q: np.ndarray,
        ) -> np.ndarray:
            """
            Algorithm 9 in https://arxiv.org/pdf/2002.01387
            """
            assert (
                num_vectors >= rank > 0
            ), "num_vectors should not be smaller than rank"
            for _ in range(depth):
                Q = np.linalg.qr(Q)[0]
                Q = operator(Q)
            Q = np.linalg.qr(Q)[0]
            return Q

        def _rand_svd(
            operator,
            operator_dim: int,
            rank: int,
            depth: int,
            num_vectors: int,
            range_sketch: np.ndarray,
        ) -> (np.ndarray, np.ndarray, np.ndarray, float):
            """
            Algorithm 8 in https://arxiv.org/pdf/2002.01387
            """
            assert num_vectors >= rank > 0
            Q = _rand_pow_range_finder(
                operator,
                operator_dim,
                rank=num_vectors,
                depth=depth,
                num_vectors=num_vectors,
                Q=range_sketch,
            )
            C = operator(Q).T
            U_hat, D, _ = np.linalg.svd(C, full_matrices=False)
            U = Q @ U_hat

            error_factor = np.power(
                1 + 4 * np.sqrt(2 * operator_dim / max(1, (rank - 1))),
                1 / (2 * depth + 1),
            )
            error_bound = D[rank] * (1 + error_factor)
            return U[:, :rank], D[:rank], Q, error_bound

        _f_high = (
            tree_sequence_high._genetic_relatedness_vector_node
            if output_type == "node"
            else tree_sequence_high._genetic_relatedness_vector_individual
        )
        if time_windows is not None:
            _f_low = (
                tree_sequence_low._genetic_relatedness_vector_node
                if output_type == "node"
                else tree_sequence_low._genetic_relatedness_vector_individual
            )
        indices = samples if output_type == "node" else individuals

        U = np.empty((num_windows, dim, num_components))
        D = np.empty((num_windows, num_components))
        Q = np.empty((num_windows, dim, num_vectors))
        E = np.empty(num_windows)
        for i in range(num_windows):

            def _G(x, i=i):
                high = _f_high(
                    arr=x,
                    indices=indices,
                    mode=mode,
                    centre=centre,
                    windows=windows[i : i + 2],
                )
                if time_windows is None:
                    return high
                else:
                    low = _f_low(
                        arr=x,
                        indices=indices,
                        mode=mode,
                        centre=centre,
                        windows=windows[i : i + 2],
                    )
                    return high - low

            U[i], D[i], Q[i], E[i] = _rand_svd(
                operator=_G,
                operator_dim=dim,
                rank=num_components,
                depth=num_iterations,
                num_vectors=num_vectors,
                range_sketch=range_sketch[i],
            )

        if drop_windows:
            U, D, Q, E = U[0], D[0], Q[0], E[0]

        pca_result = PCAResult(factors=U, eigenvalues=D, range_sketch=Q, error_bound=E)

        return pca_result

    def trait_covariance(self, W, windows=None, mode="site", span_normalise=True):
        """
        Computes the mean squared covariances between each of the columns of ``W``
        (the "phenotypes") and inheritance along the tree sequence.
        See the :ref:`statistics interface <sec_stats_interface>` section for details on
        :ref:`windows <sec_stats_windows>`,
        :ref:`mode <sec_stats_mode>`,
        :ref:`span normalise <sec_stats_span_normalise>`,
        and :ref:`return value <sec_stats_output_format>`.
        Operates on all samples in the tree sequence.

        Concretely, if `g` is a binary vector that indicates inheritance from an allele,
        branch, or node and `w` is a column of W, normalised to have mean zero,
        then the covariance of `g` and `w` is :math:`\\sum_i g_i w_i`, the sum of the
        weights corresponding to entries of `g` that are `1`. Since weights sum to
        zero, this is also equal to the sum of weights whose entries of `g` are 0.
        So, :math:`cov(g,w)^2 = ((\\sum_i g_i w_i)^2 + (\\sum_i (1-g_i) w_i)^2)/2`.

        What is computed depends on ``mode``:

        "site"
            The sum of squared covariances between presence/absence of each allele and
            phenotypes, divided by length of the window (if ``span_normalise=True``).
            This is computed as sum_a (sum(w[a])^2 / 2), where
            w is a column of W with the average subtracted off,
            and w[a] is the sum of all entries of w corresponding to samples
            carrying allele "a", and the first sum is over all alleles.

        "branch"
            The sum of squared covariances between the split induced by each branch and
            phenotypes, multiplied by branch length, averaged across trees in
            the window. This is computed as above: a branch with total weight
            w[b] below b contributes (branch length) * w[b]^2 to the total
            value for a tree. (Since the sum of w is zero, the total weight
            below b and not below b are equal, canceling the factor of 2
            above.)

        "node"
            For each node, the squared covariance between the property of
            inheriting from this node and phenotypes, computed as in "branch".

        :param numpy.ndarray W: An array of values with one row for each sample and one
            column for each "phenotype".
        :param list windows: An increasing list of breakpoints between the windows
            to compute the statistic in.
        :param str mode: A string giving the "type" of the statistic to be computed
            (defaults to "site").
        :param bool span_normalise: Whether to divide the result by the span of the
            window (defaults to True).
        :return: A ndarray with shape equal to (num windows, num statistics).
            If windows=None and W is a single column, a numpy scalar is returned.
        """
        if W.shape[0] != self.num_samples:
            raise ValueError(
                "First trait dimension must be equal to number of samples."
            )
        return self.__run_windowed_stat(
            windows,
            self._ll_tree_sequence.trait_covariance,
            W,
            mode=mode,
            span_normalise=span_normalise,
        )

    def trait_correlation(self, W, windows=None, mode="site", span_normalise=True):
        """
        Computes the mean squared correlations between each of the columns of ``W``
        (the "phenotypes") and inheritance along the tree sequence.
        See the :ref:`statistics interface <sec_stats_interface>` section for details on
        :ref:`windows <sec_stats_windows>`,
        :ref:`mode <sec_stats_mode>`,
        :ref:`span normalise <sec_stats_span_normalise>`,
        and :ref:`return value <sec_stats_output_format>`.
        Operates on all samples in the tree sequence.

        This is computed as squared covariance in
        :meth:`trait_covariance <.TreeSequence.trait_covariance>`,
        but divided by :math:`p (1-p)`, where `p` is the proportion of samples
        inheriting from the allele, branch, or node in question.

        What is computed depends on ``mode``:

        "site"
            The sum of squared correlations between presence/absence of each allele and
            phenotypes, divided by length of the window (if ``span_normalise=True``).
            This is computed as the
            :meth:`trait_covariance <.TreeSequence.trait_covariance>`
            divided by the variance of the relevant column of W
            and by :math:`p * (1 - p)`, where :math:`p` is the allele frequency.

        "branch"
            The sum of squared correlations between the split induced by each branch and
            phenotypes, multiplied by branch length, averaged across trees in
            the window. This is computed as the
            :meth:`trait_covariance <.TreeSequence.trait_covariance>`,
            divided by the variance of the column of w
            and by :math:`p * (1 - p)`, where :math:`p` is the proportion of
            the samples lying below the branch.

        "node"
            For each node, the squared correlation between the property of
            inheriting from this node and phenotypes, computed as in "branch".

        Note that above we divide by the **sample** variance, which for a
        vector x of length n is ``np.var(x) * n / (n-1)``.

        :param numpy.ndarray W: An array of values with one row for each sample and one
            column for each "phenotype". Each column must have positive standard
            deviation.
        :param list windows: An increasing list of breakpoints between the windows
            to compute the statistic in.
        :param str mode: A string giving the "type" of the statistic to be computed
            (defaults to "site").
        :param bool span_normalise: Whether to divide the result by the span of the
            window (defaults to True).
        :return: A ndarray with shape equal to (num windows, num statistics).
            If windows=None and W is a single column, a numpy scalar is returned.
        """
        if W.shape[0] != self.num_samples:
            raise ValueError(
                "First trait dimension must be equal to number of samples."
            )
        sds = np.std(W, axis=0)
        if np.any(sds == 0):
            raise ValueError(
                "Weight columns must have positive variance", "to compute correlation."
            )
        return self.__run_windowed_stat(
            windows,
            self._ll_tree_sequence.trait_correlation,
            W,
            mode=mode,
            span_normalise=span_normalise,
        )

    def trait_regression(self, *args, **kwargs):
        """
        Deprecated synonym for
        :meth:`trait_linear_model <.TreeSequence.trait_linear_model>`.
        """
        warnings.warn(
            "This is deprecated: please use trait_linear_model( ) instead.",
            FutureWarning,
            stacklevel=4,
        )
        return self.trait_linear_model(*args, **kwargs)

    def trait_linear_model(
        self,
        W,
        Z=None,
        windows=None,
        mode="site",
        span_normalise=True,
    ):
        """
        Finds the relationship between trait and genotype after accounting for
        covariates.  Concretely, for each trait w (i.e., each column of W),
        this does a least-squares fit of the linear model :math:`w \\sim g + Z`,
        where :math:`g` is inheritance in the tree sequence (e.g., genotype)
        and the columns of :math:`Z` are covariates, and returns the squared
        coefficient of :math:`g` in this linear model.
        See the :ref:`statistics interface <sec_stats_interface>` section for details on
        :ref:`windows <sec_stats_windows>`,
        :ref:`mode <sec_stats_mode>`,
        :ref:`span normalise <sec_stats_span_normalise>`,
        and :ref:`return value <sec_stats_output_format>`.
        Operates on all samples in the tree sequence.

        To do this, if `g` is a binary vector that indicates inheritance from an allele,
        branch, or node and `w` is a column of W, there are :math:`k` columns of
        :math:`Z`, and the :math:`k+2`-vector :math:`b` minimises
        :math:`\\sum_i (w_i - b_0 - b_1 g_i - b_2 z_{2,i} - ... b_{k+2} z_{k+2,i})^2`
        then this returns the number :math:`b_1^2`. If :math:`g` lies in the linear span
        of the columns of :math:`Z`, then :math:`b_1` is set to 0. To fit the
        linear model without covariates (only the intercept), set `Z = None`.

        What is computed depends on ``mode``:

        "site"
            Computes the sum of :math:`b_1^2/2` for each allele in the window,
            as above with :math:`g` indicating presence/absence of the allele,
            then divided by the length of the window if ``span_normalise=True``.
            (For biallelic loci, this number is the same for both alleles, and so summing
            over each cancels the factor of two.)

        "branch"
            The squared coefficient :math:`b_1^2`, computed for the split induced by each
            branch (i.e., with :math:`g` indicating inheritance from that branch),
            multiplied by branch length and tree span, summed over all trees
            in the window, and divided by the length of the window if
            ``span_normalise=True``.

        "node"
            For each node, the squared coefficient :math:`b_1^2`, computed for
            the property of inheriting from this node, as in "branch".

        :param numpy.ndarray W: An array of values with one row for each sample and one
            column for each "phenotype".
        :param numpy.ndarray Z: An array of values with one row for each sample and one
            column for each "covariate", or `None`. Columns of `Z` must be linearly
            independent.
        :param list windows: An increasing list of breakpoints between the windows
            to compute the statistic in.
        :param str mode: A string giving the "type" of the statistic to be computed
            (defaults to "site").
        :param bool span_normalise: Whether to divide the result by the span of the
            window (defaults to True).
        :return: A ndarray with shape equal to (num windows, num statistics).
            If windows=None and W is a single column, a numpy scalar is returned.
        """
        if W.shape[0] != self.num_samples:
            raise ValueError(
                "First trait dimension must be equal to number of samples."
            )
        if Z is None:
            Z = np.ones((self.num_samples, 1))
        else:
            tZ = np.column_stack([Z, np.ones((Z.shape[0], 1))])
            if np.linalg.matrix_rank(tZ) == tZ.shape[1]:
                Z = tZ
        if Z.shape[0] != self.num_samples:
            raise ValueError("First dimension of Z must equal the number of samples.")
        if np.linalg.matrix_rank(Z) < Z.shape[1]:
            raise ValueError("Matrix of covariates is computationally singular.")
        # numpy returns a lower-triangular cholesky
        K = np.linalg.cholesky(np.matmul(Z.T, Z)).T
        Z = np.matmul(Z, np.linalg.inv(K))
        return self.__run_windowed_stat(
            windows,
            self._ll_tree_sequence.trait_linear_model,
            W,
            Z,
            mode=mode,
            span_normalise=span_normalise,
        )

    def segregating_sites(
        self, sample_sets=None, windows=None, mode="site", span_normalise=True
    ):
        """
        Computes the density of segregating sites for each of the sets of nodes
        from ``sample_sets``, and related quantities.
        Please see the :ref:`one-way statistics <sec_stats_sample_sets_one_way>`
        section for details on how the ``sample_sets`` argument is interpreted
        and how it interacts with the dimensions of the output array.
        See the :ref:`statistics interface <sec_stats_interface>` section for details on
        :ref:`windows <sec_stats_windows>`, :ref:`mode <sec_stats_mode>`,
        :ref:`span normalise <sec_stats_span_normalise>`,
        and :ref:`return value <sec_stats_output_format>`.

        What is computed depends on ``mode``. For a sample set ``A``, computes:

        "site"
            The sum over sites of [the number of alleles found in ``A`` at each site
            minus one], per unit of chromosome length.
            If all sites have at most two alleles in ``A``,
            this is the density of segregating/polymorphic sites in ``A``
            (since the "minus one" reduces the sum for monoallelic sites).
            For sites with more than two alleles, the sum is increased by
            one for each additional allele segregating in ``A``.
            To get the **number** of segregating alleles in ``A``,
            use ``span_normalise=False``.

        "branch"
            The total length of all branches in the tree subtended by the samples in
            ``A``, averaged across the window.

        "node"
            The proportion of the window on which the node is ancestral to some,
            but not all, of the samples in ``A``.

        :param list sample_sets: A list of lists of Node IDs, specifying the
            groups of nodes to compute the statistic with.
        :param list windows: An increasing list of breakpoints between the windows
            to compute the statistic in.
        :param str mode: A string giving the "type" of the statistic to be computed
            (defaults to "site").
        :param bool span_normalise: Whether to divide the result by the span of the
            window (defaults to True).
        :return: A ndarray with shape equal to (num windows, num statistics).
            If there is one sample set and windows=None, a numpy scalar is returned.
        """
        return self.__one_way_sample_set_stat(
            self._ll_tree_sequence.segregating_sites,
            sample_sets,
            windows=windows,
            mode=mode,
            span_normalise=span_normalise,
        )

    def allele_frequency_spectrum(
        self,
        sample_sets=None,
        windows=None,
        time_windows=None,
        mode="site",
        span_normalise=True,
        polarised=False,
    ):
        """
        Computes the allele frequency spectrum (AFS) in windows across the genome for
        with respect to the specified ``sample_sets``.
        See the :ref:`statistics interface <sec_stats_interface>` section for details on
        :ref:`sample sets <sec_stats_sample_sets>`,
        :ref:`windows <sec_stats_windows>`,
        :ref:`mode <sec_stats_mode>`,
        :ref:`span normalise <sec_stats_span_normalise>`,
        :ref:`polarised <sec_stats_polarisation>`,
        and :ref:`return value <sec_stats_output_format>`.
        and see :ref:`sec_tutorial_afs` for examples of how to use this method.

        Similar to other windowed stats, the first dimension in the returned array
        corresponds to windows, such that ``result[i]`` is the AFS in the ith
        window. The AFS in each window is a k-dimensional numpy array, where k is
        the number of input sample sets, such that ``result[i, j0, j1, ...]`` is the
        value associated with frequency ``j0`` in ``sample_sets[0]``, ``j1`` in
        ``sample_sets[1]``, etc, in window ``i``. From here, we will assume that
        ``afs`` corresponds to the result in a single window, i.e.,
        ``afs = result[i]``.

        If a single sample set is specified, the allele frequency spectrum within
        this set is returned, such that ``afs[j]`` is the value associated with
        frequency ``j``. Thus, singletons are counted in ``afs[1]``, doubletons in
        ``afs[2]``, and so on. The zeroth entry counts alleles or branches not
        seen in the samples but that are polymorphic among the rest of the samples
        of the tree sequence; likewise, the last entry counts alleles fixed in
        the sample set but polymorphic in the entire set of samples. Please see
        the :ref:`sec_tutorial_afs_zeroth_entry` for an illustration.

        .. warning:: Please note that singletons are **not** counted in the initial
            entry in each AFS array (i.e., ``afs[0]``), but in ``afs[1]``.

        If ``sample_sets`` is None (the default), the allele frequency spectrum
        for all samples in the tree sequence is returned. For convenience, if
        there is only a single sample set, the outer list may be omitted (so that,
        unlike other statistics, ``sample_sets=[0,1,2]`` is equivalent to
        ``sample_sets=[[0,1,2]]``).

        If more than one sample set is specified, the **joint** allele frequency
        spectrum within windows is returned. For example, if we set
        ``sample_sets = [S0, S1]``, then afs[1, 2] counts the number of sites that
        are singletons within S0 and doubletons in S1. The dimensions of the
        output array will be ``[num_windows] + [1 + len(S) for S in sample_sets]``.

        If ``polarised`` is False (the default) the AFS will be *folded*, so that
        the counts do not depend on knowing which allele is ancestral. If folded,
        the frequency spectrum for a single sample set ``S`` has ``afs[j] = 0`` for
        all ``j > len(S) / 2``, so that alleles at frequency ``j`` and ``len(S) - j``
        both add to the same entry. If there is more than one sample set, the
        returned array is "lower triangular" in a similar way. For more details,
        especially about handling of multiallelic sites, see :ref:`sec_stats_notes_afs`.

        What is computed depends on ``mode``:

        "site"
            The number of alleles at a given frequency within the specified sample
            sets for each window, per unit of sequence length. To obtain the total
            number of alleles, set ``span_normalise`` to False.

        "branch"
            The total length of branches in the trees subtended by subsets of the
            specified sample sets, per unit of sequence length. To obtain the
            total, set ``span_normalise`` to False.

        "node"
            Not supported for this method (raises a ValueError).

        For example, suppose that `S0` is a list of 5 sample IDs, and `S1` is
        a list of 3 other sample IDs. Then `afs = ts.allele_frequency_spectrum([S0, S1],
        mode="site", span_normalise=False)` will be a 5x3 numpy array, and if
        there are six alleles that are present in only one sample of `S0` but
        two samples of `S1`, then `afs[1,2]` will be equal to 6.  Similarly,
        `branch_afs = ts.allele_frequency_spectrum([S0, S1], mode="branch",
        span_normalise=False)` will also be a 5x3 array, and `branch_afs[1,2]`
        will be the total area (i.e., length times span) of all branches that
        are above exactly one sample of `S0` and two samples of `S1`.

        :param list sample_sets: A list of lists of Node IDs, specifying the
            groups of samples to compute the joint allele frequency.
        :param list windows: An increasing list of breakpoints between windows
            along the genome.
        :param str mode: A string giving the "type" of the statistic to be computed
            (defaults to "site").
        :param bool span_normalise: Whether to divide the result by the span of the
            window (defaults to True).
        :return: A (k + 1) dimensional numpy array, where k is the number of sample
            sets specified.
            If there is one sample set and windows=None, a 1 dimensional array is
            returned.
        """
        if sample_sets is None:
            sample_sets = [self.samples()]
        try:
            # this also happens in __one_way_sample_set_stat, but we need to do
            # slightly different pre-processing here to allow for the case that
            # sample sets is a single list of IDs (for most stats, this would mean
            # dropping a dimension, but not for the AFS)
            sample_sets = np.array(sample_sets, dtype=np.uint64)
        except ValueError:
            pass
        else:
            if len(sample_sets.shape) == 1:
                sample_sets = [sample_sets]
        return self.__one_way_sample_set_stat(
            self._ll_tree_sequence.allele_frequency_spectrum,
            sample_sets,
            windows=windows,
            time_windows=time_windows,
            mode=mode,
            span_normalise=span_normalise,
            polarised=polarised,
        )

    def Tajimas_D(self, sample_sets=None, windows=None, mode="site"):
        """
        Computes Tajima's D of sets of nodes from ``sample_sets`` in windows.
        Please see the :ref:`one-way statistics <sec_stats_sample_sets_one_way>`
        section for details on how the ``sample_sets`` argument is interpreted
        and how it interacts with the dimensions of the output array.
        See the :ref:`statistics interface <sec_stats_interface>` section for details on
        :ref:`windows <sec_stats_windows>`, :ref:`mode <sec_stats_mode>`,
        and :ref:`return value <sec_stats_output_format>`.
        Operates on ``k = 1`` sample sets at a
        time. For a sample set ``X`` of ``n`` nodes, if and ``T`` is the mean
        number of pairwise differing sites in ``X`` and ``S`` is the number of
        sites segregating in ``X`` (computed with :meth:`diversity
        <.TreeSequence.diversity>` and :meth:`segregating sites
        <.TreeSequence.segregating_sites>`, respectively, both not span
        normalised), then Tajima's D is

        .. code-block:: python

            D = (T - S / h) / sqrt(a * S + (b / c) * S * (S - 1))
            h = 1 + 1 / 2 + ... + 1 / (n - 1)
            g = 1 + 1 / 2**2 + ... + 1 / (n - 1) ** 2
            a = (n + 1) / (3 * (n - 1) * h) - 1 / h**2
            b = 2 * (n**2 + n + 3) / (9 * n * (n - 1)) - (n + 2) / (h * n) + g / h**2
            c = h**2 + g

        What is computed for diversity and divergence depends on ``mode``;
        see those functions for more details.

        :param list sample_sets: A list of lists of Node IDs, specifying the
            groups of nodes to compute the statistic with.
        :param list indexes: A list of 2-tuples, or None.
        :param list windows: An increasing list of breakpoints between the windows
            to compute the statistic in.
        :param str mode: A string giving the "type" of the statistic to be computed
            (defaults to "site").
        :return: A ndarray with shape equal to (num windows, num statistics).
            If there is one sample set and windows=None, a numpy scalar is returned.
        """

        # TODO this should be done in C as we'll want to support this method there.
        def tjd_func(sample_set_sizes, flattened, **kwargs):
            n = sample_set_sizes
            T = self.ll_tree_sequence.diversity(n, flattened, **kwargs)
            S = self.ll_tree_sequence.segregating_sites(n, flattened, **kwargs)
            h = np.array([np.sum(1 / np.arange(1, nn)) for nn in n])
            g = np.array([np.sum(1 / np.arange(1, nn) ** 2) for nn in n])
            with np.errstate(invalid="ignore", divide="ignore"):
                a = (n + 1) / (3 * (n - 1) * h) - 1 / h**2
                b = (
                    2 * (n**2 + n + 3) / (9 * n * (n - 1))
                    - (n + 2) / (h * n)
                    + g / h**2
                )
                D = (T - S / h) / np.sqrt(a * S + (b / (h**2 + g)) * S * (S - 1))
            return D

        return self.__one_way_sample_set_stat(
            tjd_func, sample_sets, windows=windows, mode=mode, span_normalise=False
        )

    def Fst(
        self, sample_sets, indexes=None, windows=None, mode="site", span_normalise=True
    ):
        """
        Computes "windowed" Fst between pairs of sets of nodes from ``sample_sets``.
        Operates on ``k = 2`` sample sets at a time; please see the
        :ref:`multi-way statistics <sec_stats_sample_sets_multi_way>`
        section for details on how the ``sample_sets`` and ``indexes`` arguments are
        interpreted and how they interact with the dimensions of the output array.
        See the :ref:`statistics interface <sec_stats_interface>` section for details on
        :ref:`windows <sec_stats_windows>`,
        :ref:`mode <sec_stats_mode>`,
        :ref:`span normalise <sec_stats_span_normalise>`,
        and :ref:`return value <sec_stats_output_format>`.

        For sample sets ``X`` and ``Y``, if ``d(X, Y)`` is the
        :meth:`divergence <.TreeSequence.divergence>`
        between ``X`` and ``Y``, and ``d(X)`` is the
        :meth:`diversity <.TreeSequence.diversity>` of ``X``, then what is
        computed is

        .. code-block:: python

            Fst = 1 - 2 * (d(X) + d(Y)) / (d(X) + 2 * d(X, Y) + d(Y))

        What is computed for diversity and divergence depends on ``mode``;
        see those functions for more details.

        :param list sample_sets: A list of lists of Node IDs, specifying the
            groups of nodes to compute the statistic with.
        :param list indexes: A list of 2-tuples.
        :param list windows: An increasing list of breakpoints between the windows
            to compute the statistic in.
        :param str mode: A string giving the "type" of the statistic to be computed
            (defaults to "site").
        :param bool span_normalise: Whether to divide the result by the span of the
            window (defaults to True).
        :return: A ndarray with shape equal to (num windows, num statistics).
            If there is one pair of sample sets and windows=None, a numpy scalar is
            returned.
        """
        # TODO this should really be implemented in C (presumably C programmers will want
        # to compute Fst too), but in the mean time implementing using the low-level
        # calls has two advantages: (a) we automatically change dimensions like the other
        # two-way stats and (b) it's a bit more efficient because we're not messing
        # around with indexes and samples sets twice.

        def fst_func(
            sample_set_sizes,
            flattened,
            indexes,
            windows,
            mode,
            span_normalise,
            polarised,
            centre,
        ):
            # note: this is kinda hacky - polarised and centre are not used here -
            # but this seems necessary to use our __k_way_sample_set_stat framework
            divergences = self._ll_tree_sequence.divergence(
                sample_set_sizes,
                flattened,
                indexes=indexes,
                windows=windows,
                mode=mode,
                span_normalise=span_normalise,
                polarised=polarised,
                centre=centre,
            )
            diversities = self._ll_tree_sequence.diversity(
                sample_set_sizes,
                flattened,
                windows=windows,
                mode=mode,
                span_normalise=span_normalise,
                polarised=polarised,
            )

            orig_shape = divergences.shape
            # "node" statistics produce a 3D array
            if len(divergences.shape) == 2:
                divergences.shape = (divergences.shape[0], 1, divergences.shape[1])
                diversities.shape = (diversities.shape[0], 1, diversities.shape[1])

            fst = np.repeat(1.0, np.prod(divergences.shape))
            fst.shape = divergences.shape
            for i, (u, v) in enumerate(indexes):
                denom = (
                    diversities[:, :, u]
                    + diversities[:, :, v]
                    + 2 * divergences[:, :, i]
                )
                with np.errstate(divide="ignore", invalid="ignore"):
                    fst[:, :, i] -= (
                        2 * (diversities[:, :, u] + diversities[:, :, v]) / denom
                    )
            fst.shape = orig_shape
            return fst

        return self.__k_way_sample_set_stat(
            fst_func,
            2,
            sample_sets,
            indexes=indexes,
            windows=windows,
            mode=mode,
            span_normalise=span_normalise,
        )

    def Y3(
        self, sample_sets, indexes=None, windows=None, mode="site", span_normalise=True
    ):
        """
        Computes the 'Y' statistic between triples of sets of nodes from ``sample_sets``.
        Operates on ``k = 3`` sample sets at a time; please see the
        :ref:`multi-way statistics <sec_stats_sample_sets_multi_way>`
        section for details on how the ``sample_sets`` and ``indexes`` arguments are
        interpreted and how they interact with the dimensions of the output array.
        See the :ref:`statistics interface <sec_stats_interface>` section for details on
        :ref:`windows <sec_stats_windows>`,
        :ref:`mode <sec_stats_mode>`,
        :ref:`span normalise <sec_stats_span_normalise>`,
        and :ref:`return value <sec_stats_output_format>`.

        What is computed depends on ``mode``. Each is an average across every
        combination of trios of samples ``(a, b, c)``, one chosen from each sample set:

        "site"
            The average density of sites at which ``a`` differs from ``b`` and
            ``c``, per unit of chromosome length.

        "branch"
            The average length of all branches that separate ``a`` from ``b``
            and ``c`` (in units of time).

        "node"
            For each node, the average proportion of the window on which ``a``
            inherits from that node but ``b`` and ``c`` do not, or vice-versa.

        :param list sample_sets: A list of lists of Node IDs, specifying the
            groups of nodes to compute the statistic with.
        :param list indexes: A list of 3-tuples, or None.
        :param list windows: An increasing list of breakpoints between the windows
            to compute the statistic in.
        :param str mode: A string giving the "type" of the statistic to be computed
            (defaults to "site").
        :param bool span_normalise: Whether to divide the result by the span of the
            window (defaults to True).
        :return: A ndarray with shape equal to (num windows, num statistics).
            If there is one triple of sample sets and windows=None, a numpy scalar is
            returned.
        """
        return self.__k_way_sample_set_stat(
            self._ll_tree_sequence.Y3,
            3,
            sample_sets,
            indexes=indexes,
            windows=windows,
            mode=mode,
            span_normalise=span_normalise,
        )

    def Y2(
        self, sample_sets, indexes=None, windows=None, mode="site", span_normalise=True
    ):
        """
        Computes the 'Y2' statistic between pairs of sets of nodes from ``sample_sets``.
        Operates on ``k = 2`` sample sets at a time; please see the
        :ref:`multi-way statistics <sec_stats_sample_sets_multi_way>`
        section for details on how the ``sample_sets`` and ``indexes`` arguments are
        interpreted and how they interact with the dimensions of the output array.
        See the :ref:`statistics interface <sec_stats_interface>` section for details on
        :ref:`windows <sec_stats_windows>`,
        :ref:`mode <sec_stats_mode>`,
        :ref:`span normalise <sec_stats_span_normalise>`,
        and :ref:`return value <sec_stats_output_format>`.

        What is computed depends on ``mode``. Each is computed exactly as
        ``Y3``, except that the average is across every possible trio of samples
        ``(a, b1, b2)``, where ``a`` is chosen from the first sample set, and
        ``b1, b2`` are chosen (without replacement) from the second sample set.
        See :meth:`Y3 <.TreeSequence.Y3>` for more details.

        :param list sample_sets: A list of lists of Node IDs, specifying the
            groups of nodes to compute the statistic with.
        :param list indexes: A list of 2-tuples, or None.
        :param list windows: An increasing list of breakpoints between the windows
            to compute the statistic in.
        :param str mode: A string giving the "type" of the statistic to be computed
            (defaults to "site").
        :param bool span_normalise: Whether to divide the result by the span of the
            window (defaults to True).
        :return: A ndarray with shape equal to (num windows, num statistics).
            If there is one pair of sample sets and windows=None, a numpy scalar is
            returned.
        """
        return self.__k_way_sample_set_stat(
            self._ll_tree_sequence.Y2,
            2,
            sample_sets,
            indexes=indexes,
            windows=windows,
            mode=mode,
            span_normalise=span_normalise,
        )

    def Y1(self, sample_sets, windows=None, mode="site", span_normalise=True):
        """
        Computes the 'Y1' statistic within each of the sets of nodes given by
        ``sample_sets``.
        Please see the :ref:`one-way statistics <sec_stats_sample_sets_one_way>`
        section for details on how the ``sample_sets`` argument is interpreted
        and how it interacts with the dimensions of the output array.
        See the :ref:`statistics interface <sec_stats_interface>` section for details on
        :ref:`windows <sec_stats_windows>`, :ref:`mode <sec_stats_mode>`,
        :ref:`span normalise <sec_stats_span_normalise>`,
        and :ref:`return value <sec_stats_output_format>`.
        Operates on ``k = 1`` sample set at a time.

        What is computed depends on ``mode``. Each is computed exactly as
        ``Y3``, except that the average is across every possible trio of samples
        samples ``(a1, a2, a3)`` all chosen without replacement from the same
        sample set. See :meth:`Y3 <.TreeSequence.Y3>` for more details.

        :param list sample_sets: A list of lists of Node IDs, specifying the
            groups of nodes to compute the statistic with.
        :param list windows: An increasing list of breakpoints between the windows
            to compute the statistic in.
        :param str mode: A string giving the "type" of the statistic to be computed
            (defaults to "site").
        :param bool span_normalise: Whether to divide the result by the span of the
            window (defaults to True).
        :return: A ndarray with shape equal to (num windows, num statistics).
            If there is one sample set and windows=None, a numpy scalar is returned.
        """
        return self.__one_way_sample_set_stat(
            self._ll_tree_sequence.Y1,
            sample_sets,
            windows=windows,
            mode=mode,
            span_normalise=span_normalise,
        )

    def f4(
        self, sample_sets, indexes=None, windows=None, mode="site", span_normalise=True
    ):
        """
        Computes Patterson's f4 statistic between four groups of nodes from
        ``sample_sets``.
        Operates on ``k = 4`` sample sets at a time; please see the
        :ref:`multi-way statistics <sec_stats_sample_sets_multi_way>`
        section for details on how the ``sample_sets`` and ``indexes`` arguments are
        interpreted and how they interact with the dimensions of the output array.
        See the :ref:`statistics interface <sec_stats_interface>` section for details on
        :ref:`windows <sec_stats_windows>`,
        :ref:`mode <sec_stats_mode>`,
        :ref:`span normalise <sec_stats_span_normalise>`,
        and :ref:`return value <sec_stats_output_format>`.

        What is computed depends on ``mode``. Each is an average across every possible
        combination of four samples ``(a, b; c, d)``, one chosen from each sample set:

        "site"
            The average density of sites at which ``a`` and ``c`` agree but
            differs from ``b`` and ``d``, minus the average density of sites at
            which ``a`` and ``d`` agree but differs from ``b`` and ``c``, per
            unit of chromosome length.

        "branch"
            The average length of all branches that separate ``a`` and ``c``
            from ``b`` and ``d``, minus the average length of all branches that
            separate ``a`` and ``d`` from ``b`` and ``c`` (in units of time).

        "node"
            For each node, the average proportion of the window on which ``a`` and ``c``
            inherit from that node but ``b`` and ``d`` do not, or vice-versa,
            minus the average proportion of the window on which ``a`` and ``d``
            inherit from that node but ``b`` and ``c`` do not, or vice-versa.

        :param list sample_sets: A list of lists of Node IDs, specifying the
            groups of nodes to compute the statistic with.
        :param list indexes: A list of 4-tuples, or None.
        :param list windows: An increasing list of breakpoints between the windows
            to compute the statistic in.
        :param str mode: A string giving the "type" of the statistic to be computed
            (defaults to "site").
        :param bool span_normalise: Whether to divide the result by the span of the
            window (defaults to True).
        :return: A ndarray with shape equal to (num windows, num statistics).
            If there are four sample sets and windows=None, a numpy scalar is returned.
        """
        return self.__k_way_sample_set_stat(
            self._ll_tree_sequence.f4,
            4,
            sample_sets,
            indexes=indexes,
            windows=windows,
            mode=mode,
            span_normalise=span_normalise,
        )

    def f3(
        self, sample_sets, indexes=None, windows=None, mode="site", span_normalise=True
    ):
        r"""
        Computes Patterson's f3 statistic between three groups of nodes from
        ``sample_sets``.
        Note that the order of the arguments of f3 differs across the literature:
        here, ``f3([A, B, C])`` for sample sets ``A``, ``B``, and ``C``
        will estimate
        :math:`f_3(A; B, C) = \mathbb{E}[(p_A - p_B) (p_A - p_C)]`,
        where :math:`p_A` is the allele frequency in ``A``.
        When used as a test for admixture, the putatively admixed population
        is usually placed as population ``A`` (see
        `Peter (2016) <https://doi.org/10.1534/genetics.115.183913>`_
        for more discussion).

        Operates on ``k = 3`` sample sets at a time; please see the
        :ref:`multi-way statistics <sec_stats_sample_sets_multi_way>`
        section for details on how the ``sample_sets`` and ``indexes`` arguments are
        interpreted and how they interact with the dimensions of the output array.
        See the :ref:`statistics interface <sec_stats_interface>` section for details on
        :ref:`windows <sec_stats_windows>`,
        :ref:`mode <sec_stats_mode>`,
        :ref:`span normalise <sec_stats_span_normalise>`,
        and :ref:`return value <sec_stats_output_format>`.

        What is computed depends on ``mode``. Each works exactly as
        :meth:`f4 <.TreeSequence.f4>`, except the average is across every possible
        combination of four samples ``(a1, b; a2, c)`` where `a1` and `a2` have both
        been chosen (without replacement) from the first sample set. See
        :meth:`f4 <.TreeSequence.f4>` for more details.

        :param list sample_sets: A list of lists of Node IDs, specifying the
            groups of nodes to compute the statistic with.
        :param list indexes: A list of 3-tuples, or None.
        :param list windows: An increasing list of breakpoints between the windows
            to compute the statistic in.
        :param str mode: A string giving the "type" of the statistic to be computed
            (defaults to "site").
        :param bool span_normalise: Whether to divide the result by the span of the
            window (defaults to True).
        :return: A ndarray with shape equal to (num windows, num statistics).
            If there are three sample sets and windows=None, a numpy scalar is returned.
        """
        return self.__k_way_sample_set_stat(
            self._ll_tree_sequence.f3,
            3,
            sample_sets,
            indexes=indexes,
            windows=windows,
            mode=mode,
            span_normalise=span_normalise,
        )

    def f2(
        self, sample_sets, indexes=None, windows=None, mode="site", span_normalise=True
    ):
        """
        Computes Patterson's f2 statistic between two groups of nodes from
        ``sample_sets``.
        Operates on ``k = 2`` sample sets at a time; please see the
        :ref:`multi-way statistics <sec_stats_sample_sets_multi_way>`
        section for details on how the ``sample_sets`` and ``indexes`` arguments are
        interpreted and how they interact with the dimensions of the output array.
        See the :ref:`statistics interface <sec_stats_interface>` section for details on
        :ref:`windows <sec_stats_windows>`,
        :ref:`mode <sec_stats_mode>`,
        :ref:`span normalise <sec_stats_span_normalise>`,
        and :ref:`return value <sec_stats_output_format>`.

        What is computed depends on ``mode``. Each works exactly as
        :meth:`f4 <.TreeSequence.f4>`, except the average is across every possible
        combination of four samples ``(a1, b1; a2, b2)`` where `a1` and `a2` have
        both been chosen (without replacement) from the first sample set, and ``b1``
        and ``b2`` have both been chosen (without replacement) from the second
        sample set. See :meth:`f4 <.TreeSequence.f4>` for more details.

        :param list sample_sets: A list of lists of Node IDs, specifying the
            groups of nodes to compute the statistic with.
        :param list indexes: A list of 2-tuples, or None.
        :param list windows: An increasing list of breakpoints between the windows
            to compute the statistic in.
        :param str mode: A string giving the "type" of the statistic to be computed
            (defaults to "site").
        :param bool span_normalise: Whether to divide the result by the span of the
            window (defaults to True).
        :return: A ndarray with shape equal to (num windows, num statistics).
            If there is one pair of sample sets and windows=None, a numpy scalar is
            returned.
        """
        return self.__k_way_sample_set_stat(
            self._ll_tree_sequence.f2,
            2,
            sample_sets,
            indexes=indexes,
            windows=windows,
            mode=mode,
            span_normalise=span_normalise,
        )

    def mean_descendants(self, sample_sets):
        """
        Computes for every node the mean number of samples in each of the
        `sample_sets` that descend from that node, averaged over the
        portions of the genome for which the node is ancestral to *any* sample.
        The output is an array, `C[node, j]`, which reports the total span of
        all genomes in `sample_sets[j]` that inherit from `node`, divided by
        the total span of the genome on which `node` is an ancestor to any
        sample in the tree sequence.

        .. warning:: The interface for this method is preliminary and may be subject to
            backwards incompatible changes in the near future. The long-term stable
            API for this method will be consistent with other :ref:`sec_stats`.
            In particular, the normalization by proportion of the genome that `node`
            is an ancestor to anyone may not be the default behaviour in the future.

        :param list sample_sets: A list of lists of node IDs.
        :return: An array with dimensions (number of nodes in the tree sequence,
            number of reference sets)
        """
        return self._ll_tree_sequence.mean_descendants(sample_sets)

    def genealogical_nearest_neighbours(self, focal, sample_sets, num_threads=0):
        """
        Return the genealogical nearest neighbours (GNN) proportions for the given
        focal nodes, with reference to two or more sets of interest, averaged over all
        trees in the tree sequence.

        The GNN proportions for a focal node in a single tree are given by first finding
        the most recent common ancestral node :math:`a` between the focal node and any
        other node present in the reference sets. The GNN proportion for a specific
        reference set, :math:`S` is the number of nodes in :math:`S` that descend from
        :math:`a`, as a proportion of the total number of descendant nodes in any of the
        reference sets.

        For example, consider a case with 2 sample sets, :math:`S_1` and :math:`S_2`.
        For a given tree, :math:`a` is the node that includes at least one descendant in
        :math:`S_1` or :math:`S_2` (not including the focal node). If the descendants of
        :math:`a` include some nodes in :math:`S_1` but no nodes in :math:`S_2`, then the
        GNN proportions for that tree will be 100% :math:`S_1` and 0% :math:`S_2`, or
        :math:`[1.0, 0.0]`.

        For a given focal node, the GNN proportions returned by this function are an
        average of the GNNs for each tree, weighted by the genomic distance spanned by
        that tree.

        For an precise mathematical definition of GNN, see https://doi.org/10.1101/458067

        .. note:: The reference sets need not include all the samples, hence the most
            recent common ancestral node of the reference sets, :math:`a`, need not be
            the immediate ancestor of the focal node. If the reference sets only comprise
            sequences from relatively distant individuals, the GNN statistic may end up
            as a measure of comparatively distant ancestry, even for tree sequences that
            contain many closely related individuals.

        .. warning:: The interface for this method is preliminary and may be subject to
            backwards incompatible changes in the near future. The long-term stable
            API for this method will be consistent with other :ref:`sec_stats`.

        :param list focal: A list of :math:`n` nodes whose GNNs should be calculated.
        :param list sample_sets: A list of :math:`m` lists of node IDs.
        :return: An :math:`n`  by :math:`m` array of focal nodes by GNN proportions.
            Every focal node corresponds to a row. The numbers in each
            row corresponding to the GNN proportion for each of the passed-in reference
            sets. Rows therefore sum to one.
        :rtype: numpy.ndarray
        """
        # TODO add windows=None option: https://github.com/tskit-dev/tskit/issues/193
        if num_threads <= 0:
            return self._ll_tree_sequence.genealogical_nearest_neighbours(
                focal, sample_sets
            )
        else:
            worker = functools.partial(
                self._ll_tree_sequence.genealogical_nearest_neighbours,
                reference_sets=sample_sets,
            )
            focal = util.safe_np_int_cast(focal, np.int32)
            splits = np.array_split(focal, num_threads)
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
                arrays = pool.map(worker, splits)
            return np.vstack(list(arrays))

    def kc_distance(self, other, lambda_=0.0):
        """
        Returns the average :meth:`Tree.kc_distance` between pairs of trees along
        the sequence whose intervals overlap. The average is weighted by the
        fraction of the sequence on which each pair of trees overlap.

        :param TreeSequence other: The other tree sequence to compare to.
        :param float lambda_: The KC metric lambda parameter determining the
            relative weight of topology and branch length.
        :return: The computed KC distance between this tree sequence and other.
        :rtype: float
        """
        return self._ll_tree_sequence.get_kc_distance(other._ll_tree_sequence, lambda_)

    def count_topologies(self, sample_sets=None):
        """
        Returns a generator that produces the same distribution of topologies as
        :meth:`Tree.count_topologies` but sequentially for every tree in a tree
        sequence. For use on a tree sequence this method is much faster than
        computing the result independently per tree.

        .. warning:: The interface for this method is preliminary and may be subject to
            backwards incompatible changes in the near future.

        :param list sample_sets: A list of lists of Node IDs, specifying the
            groups of nodes to compute the statistic with.
        :rtype: iter(:class:`tskit.TopologyCounter`)
        :raises ValueError: If nodes in ``sample_sets`` are invalid or are
            internal samples.
        """
        if sample_sets is None:
            sample_sets = [
                self.samples(population=pop.id) for pop in self.populations()
            ]

        yield from combinatorics.treeseq_count_topologies(self, sample_sets)

    def ibd_segments(
        self,
        *,
        within=None,
        between=None,
        max_time=None,
        min_span=None,
        store_pairs=None,
        store_segments=None,
    ):
        """
        Finds pairs of samples that are identical by descent (IBD) and returns
        the result as an :class:`.IdentitySegments` instance. The information
        stored in this object is controlled by the ``store_pairs`` and
        ``store_segments`` parameters. By default only total counts and other
        statistics of the IBD segments are stored (i.e.,
        ``store_pairs=False``), since storing pairs and segments has a
        substantial CPU and memory overhead. Please see the
        :ref:`sec_identity` section for more details on how to access the
        information stored in the :class:`.IdentitySegments`.

        If ``within`` is specified, only IBD segments for pairs of nodes within
        that set will be recorded. If ``between`` is specified, only IBD
        segments from pairs that are in one or other of the specified sample
        sets will be reported. Note that ``within`` and ``between`` are
        mutually exclusive.

        A pair of nodes ``(u, v)`` has an IBD segment with a left and right
        coordinate ``[left, right)`` and ancestral node ``a`` iff the most
        recent common ancestor of the segment ``[left, right)`` in nodes ``u``
        and ``v`` is ``a``, and the segment has been inherited along the same
        genealogical path (ie. it has not been broken by recombination). The
        segments returned are the longest possible ones.

        Note that this definition is purely genealogical --- allelic states
        *are not* considered here. If used without time or length thresholds, the
        segments returned for a given pair will partition the span of the contig
        represented by the tree sequence.

        :param list within: A list of node IDs defining set of nodes that
            we finding IBD segments for. If not specified, this defaults to
            all samples in the tree sequence.
        :param list[list] between: A list of lists of sample node IDs. Given
            two sample sets A and B, only IBD segments will be returned such
            that one of the samples is an element of A and the other is
            an element of B. Cannot be specified with ``within``.
        :param float max_time: Only segments inherited from common
            ancestors whose node times are more recent than the specified time
            will be returned. Specifying a maximum time is strongly recommended when
            working with large tree sequences.
        :param float min_span: Only segments in which the difference between
            the right and left genome coordinates (i.e., the span of the
            segment) is greater than this value will be included. (Default=0)
        :param bool store_pairs: If True store information separately for each
            pair of samples ``(a, b)`` that are found to be IBD. Otherwise
            store summary information about all sample apirs. (Default=False)
        :param bool store_segments: If True store each IBD segment
            ``(left, right, c)`` and associate it with the corresponding
            sample pair ``(a, b)``. If True, implies ``store_pairs``.
            (Default=False).
        :return: An :class:`.IdentitySegments` object containing the recorded
            IBD information.
        :rtype: IdentitySegments
        """
        return self.dump_tables().ibd_segments(
            within=within,
            between=between,
            max_time=max_time,
            min_span=min_span,
            store_segments=store_segments,
            store_pairs=store_pairs,
        )

    def pair_coalescence_counts(
        self,
        sample_sets=None,
        indexes=None,
        windows=None,
        span_normalise=True,
        pair_normalise=False,
        time_windows="nodes",
    ):
        """
        Calculate the number of coalescing sample pairs per node, summed over
        trees and weighted by tree span.

        The number of coalescing pairs may be calculated within or between the
        non-overlapping lists of samples contained in `sample_sets`. In the
        latter case, pairs are counted if they have exactly one member in each
        of two sample sets. If `sample_sets` is omitted, a single set
        containing all samples is assumed.

        The argument `indexes` may be used to specify which pairs of sample
        sets to compute the statistic between, and in what order. If
        `indexes=None`, then `indexes` is assumed to equal `[(0,0)]` for a
        single sample set and `[(0,1)]` for two sample sets. For more than two
        sample sets, `indexes` must be explicitly passed.

        The argument `time_windows` may be used to count coalescence
        events within time intervals (if an array of breakpoints is supplied)
        rather than for individual nodes (the default).

        The output array has dimension `(windows, indexes, nodes)` with
        dimensions dropped when the corresponding argument is set to None.

        :param list sample_sets: A list of lists of Node IDs, specifying the
            groups of nodes to compute the statistic with, or None.
        :param list indexes: A list of 2-tuples, or None.
        :param list windows: An increasing list of breakpoints between the
            sequence windows to compute the statistic in, or None.
        :param bool span_normalise: Whether to divide the result by the span of
            non-missing sequence in the window (defaults to True).
        :param bool pair_normalise: Whether to divide the result by the total
            number of pairs for a given index (defaults to False).
        :param time_windows: Either a string "nodes" or an increasing
            list of breakpoints between time intervals.
        """

        if sample_sets is None:
            sample_sets = [list(self.samples())]

        drop_middle_dimension = False
        if indexes is None:
            drop_middle_dimension = True
            if len(sample_sets) == 1:
                indexes = [(0, 0)]
            elif len(sample_sets) == 2:
                indexes = [(0, 1)]
            else:
                raise ValueError(
                    "Must specify indexes if there are more than two sample sets"
                )
        num_indexes = len(indexes)

        drop_left_dimension = False
        if windows is None:
            drop_left_dimension = True
            windows = np.array([0.0, self.sequence_length])
        num_windows = len(windows) - 1

        if isinstance(time_windows, str) and time_windows == "nodes":
            num_time_windows = self.num_nodes
            node_bin_map = np.arange(num_time_windows, dtype=np.int32)
        else:
            if self.time_units == tskit.TIME_UNITS_UNCALIBRATED:
                raise ValueError("Time windows require calibrated node times")
            num_time_windows = len(time_windows) - 1
            node_bin_map = np.digitize(self.nodes_time, time_windows) - 1
            node_bin_map[node_bin_map == num_time_windows] = tskit.NULL
            node_bin_map = node_bin_map.astype(np.int32)
        num_bins = node_bin_map.max() + 1

        sample_set_sizes = np.array([len(s) for s in sample_sets], dtype=np.uint32)
        sample_sets = util.safe_np_int_cast(np.hstack(sample_sets), np.int32)

        coalescing_pairs = np.zeros((num_windows, num_indexes, num_time_windows))
        coalescing_pairs[..., :num_bins] = (
            self.ll_tree_sequence.pair_coalescence_counts(
                sample_sets=sample_sets,
                sample_set_sizes=sample_set_sizes,
                windows=windows,
                indexes=indexes,
                node_bin_map=node_bin_map,
                span_normalise=span_normalise,
                pair_normalise=pair_normalise,
            )
        )

        if drop_middle_dimension:
            coalescing_pairs = np.squeeze(coalescing_pairs, axis=1)
        if drop_left_dimension:
            coalescing_pairs = np.squeeze(coalescing_pairs, axis=0)

        return coalescing_pairs

    def pair_coalescence_quantiles(
        self,
        quantiles,
        sample_sets=None,
        indexes=None,
        windows=None,
    ):
        """
        Estimate quantiles of pair coalescence times by inverting the empirical
        CDF. This is equivalent to the "inverted_cdf" method of
        `numpy.quantile` applied to node times, with weights proportional to
        the number of coalescing pairs per node (averaged over trees, see
        `TreeSequence.pair_coalescence_counts`).

        Quantiles of pair coalescence times may be calculated within or
        between the non-overlapping lists of samples contained in `sample_sets`. In
        the latter case, pairs are counted if they have exactly one member in each
        of two sample sets. If `sample_sets` is omitted, a single set containing
        all samples is assumed.

        The argument `indexes` may be used to specify which pairs of sample sets to
        compute coalescences between, and in what order. If `indexes=None`, then
        `indexes` is assumed to equal `[(0,0)]` for a single sample set and
        `[(0,1)]` for two sample sets. For more than two sample sets, `indexes`
        must be explicitly passed.

        The output array has dimension `(windows, indexes, quantiles)` with
        dimensions dropped when the corresponding argument is set to None.

        :param quantiles: A list of increasing breakpoints between [0, 1].
        :param list sample_sets: A list of lists of Node IDs, specifying the
            groups of nodes to compute the statistic with, or None.
        :param list indexes: A list of 2-tuples, or None.
        :param list windows: An increasing list of breakpoints between the
            sequence windows to compute the statistic in, or None.
        """

        if sample_sets is None:
            sample_sets = [list(self.samples())]

        drop_middle_dimension = False
        if indexes is None:
            drop_middle_dimension = True
            if len(sample_sets) == 1:
                indexes = [(0, 0)]
            elif len(sample_sets) == 2:
                indexes = [(0, 1)]
            else:
                raise ValueError(
                    "Must specify indexes if there are more than two sample sets"
                )

        if self.time_units == tskit.TIME_UNITS_UNCALIBRATED:
            raise ValueError("Pair coalescence quantiles require calibrated node times")

        drop_left_dimension = False
        if windows is None:
            drop_left_dimension = True
            windows = np.array([0.0, self.sequence_length])

        sample_set_sizes = np.array([len(s) for s in sample_sets], dtype=np.uint32)
        sample_sets = util.safe_np_int_cast(np.hstack(sample_sets), np.int32)
        _, node_bin_map = np.unique(self.nodes_time, return_inverse=True)
        node_bin_map = util.safe_np_int_cast(node_bin_map, np.int32)

        coalescence_times = self.ll_tree_sequence.pair_coalescence_quantiles(
            sample_sets=sample_sets,
            sample_set_sizes=sample_set_sizes,
            windows=windows,
            indexes=indexes,
            node_bin_map=node_bin_map,
            quantiles=quantiles,
        )

        if drop_middle_dimension:
            coalescence_times = np.squeeze(coalescence_times, axis=1)
        if drop_left_dimension:
            coalescence_times = np.squeeze(coalescence_times, axis=0)

        return coalescence_times

    def pair_coalescence_rates(
        self,
        time_windows,
        sample_sets=None,
        indexes=None,
        windows=None,
    ):
        """
        Estimate the rate at which pairs of samples coalesce within time
        windows, using the empirical cumulative distribution function (ecdf) of
        pair coalescence times.  Assuming that pair coalescence events follow a
        nonhomogeneous Poisson process, the empirical rate for a time window
        :math:`[a, b)` where :math:`ecdf(b) < 1` is,

        ..math:

            log(1 - \\frac{ecdf(b) - ecdf(a)}{1 - ecdf(a)}) / (a - b)

        If the last coalescence event is within :math:`[a, b)`, so that
        :math:`ecdf(b) = 1`, then an estimate of the empirical rate is

        ..math:

            (\\mathbb{E}[t | t > a] - a)^{-1}

        where :math:`\\mathbb{E}[t | t < a]` is the average pair coalescence time
        conditional on coalescence after the start of the last epoch.

        The first breakpoint in `time_windows` must start at the age of the
        samples, and the last must end at infinity. In the output array, any
        time windows where all pairs have coalesced by start of the window will
        contain `NaN` values.

        Pair coalescence rates may be calculated within or between the
        non-overlapping lists of samples contained in `sample_sets`. In the
        latter case, pairs are counted if they have exactly one member in each
        of two sample sets. If `sample_sets` is omitted, a single group
        containing all samples is assumed.

        The argument `indexes` may be used to specify which pairs of sample
        sets to compute the statistic between, and in what order. If
        `indexes=None`, then `indexes` is assumed to equal `[(0,0)]` for a
        single sample set and `[(0,1)]` for two sample sets. For more than two
        sample sets, `indexes` must be explicitly passed.

        The output array has dimension `(windows, indexes, time_windows)` with
        dimensions dropped when the corresponding argument is set to None.

        :param time_windows: An increasing list of breakpoints between time
            intervals, starting at the age of the samples and ending at
            infinity.
        :param list sample_sets: A list of lists of Node IDs, specifying the
            groups of nodes to compute the statistic with, or None.
        :param list indexes: A list of 2-tuples, or None.
        :param list windows: An increasing list of breakpoints between the
            sequence windows to compute the statistic in, or None.
        """

        if sample_sets is None:
            sample_sets = [list(self.samples())]

        drop_middle_dimension = False
        if indexes is None:
            drop_middle_dimension = True
            if len(sample_sets) == 1:
                indexes = [(0, 0)]
            elif len(sample_sets) == 2:
                indexes = [(0, 1)]
            else:
                raise ValueError(
                    "Must specify indexes if there are more than two sample sets"
                )

        if self.time_units == tskit.TIME_UNITS_UNCALIBRATED:
            raise ValueError("Pair coalescence rates require calibrated node times")

        drop_left_dimension = False
        if windows is None:
            drop_left_dimension = True
            windows = np.array([0.0, self.sequence_length])

        sample_set_sizes = np.array([len(s) for s in sample_sets], dtype=np.uint32)
        sample_sets = util.safe_np_int_cast(np.hstack(sample_sets), np.int32)
        node_bin_map = np.digitize(self.nodes_time, time_windows) - 1
        node_bin_map[node_bin_map == time_windows.size - 1] = tskit.NULL
        node_bin_map = node_bin_map.astype(np.int32)

        coalescence_rates = self.ll_tree_sequence.pair_coalescence_rates(
            sample_sets=sample_sets,
            sample_set_sizes=sample_set_sizes,
            windows=windows,
            indexes=indexes,
            node_bin_map=node_bin_map,
            time_windows=time_windows,
        )

        if drop_middle_dimension:
            coalescence_rates = np.squeeze(coalescence_rates, axis=1)
        if drop_left_dimension:
            coalescence_rates = np.squeeze(coalescence_rates, axis=0)

        return coalescence_rates

    def impute_unknown_mutations_time(
        self,
        method=None,
    ):
        """
        Returns an array of mutation times, where any unknown times are
        imputed from the times of associated nodes. Not to be confused with
        :meth:`TableCollection.compute_mutation_times`, which modifies the
        ``time`` column of the mutations table in place.

        :param str method: The method used to impute the unknown mutation times.
            Currently only "min" is supported, which uses the time of the node
            below the mutation as the mutation time. The "min" method can also
            be specified by ``method=None`` (Default: ``None``).
        :return: An array of length equal to the number of mutations in the
            tree sequence.
        """
        allowed_methods = ["min"]
        if method is None:
            method = "min"
        if method not in allowed_methods:
            raise ValueError(
                f"Mutations time imputation method must be chosen from {allowed_methods}"
            )
        if method == "min":
            mutations_time = self.mutations_time.copy()
            unknown = tskit.is_unknown_time(mutations_time)
            mutations_time[unknown] = self.nodes_time[self.mutations_node[unknown]]
            return mutations_time

    def ld_matrix(
        self,
        sample_sets=None,
        sites=None,
        positions=None,
        mode="site",
        stat="r2",
        indexes=None,
    ):
        one_way_stats = {
            "D": self._ll_tree_sequence.D_matrix,
            "D2": self._ll_tree_sequence.D2_matrix,
            "r2": self._ll_tree_sequence.r2_matrix,
            "D_prime": self._ll_tree_sequence.D_prime_matrix,
            "r": self._ll_tree_sequence.r_matrix,
            "Dz": self._ll_tree_sequence.Dz_matrix,
            "pi2": self._ll_tree_sequence.pi2_matrix,
            "Dz_unbiased": self._ll_tree_sequence.Dz_unbiased_matrix,
            "D2_unbiased": self._ll_tree_sequence.D2_unbiased_matrix,
            "pi2_unbiased": self._ll_tree_sequence.pi2_unbiased_matrix,
        }
        two_way_stats = {
            "D2": self._ll_tree_sequence.D2_ij_matrix,
            "D2_unbiased": self._ll_tree_sequence.D2_ij_unbiased_matrix,
            "r2": self._ll_tree_sequence.r2_ij_matrix,
        }
        stats = one_way_stats if indexes is None else two_way_stats
        try:
            stat_func = stats[stat]
        except KeyError:
            raise ValueError(
                f"Unknown two-locus statistic '{stat}', we support: {list(stats.keys())}"
            )

        if indexes is not None:
            return self.__k_way_two_locus_sample_set_stat(
                stat_func,
                2,
                sample_sets,
                indexes=indexes,
                sites=sites,
                positions=positions,
                mode=mode,
            )

        return self.__two_locus_sample_set_stat(
            stat_func, sample_sets, sites=sites, positions=positions, mode=mode
        )

    def sample_nodes_by_ploidy(self, ploidy):
        """
        Returns an 2D array of node IDs, where each row has length `ploidy`.
        This is useful when individuals are not defined in the tree sequence
        so `TreeSequence.individuals_nodes` cannot be used. The samples are
        placed in the array in the order which they are found in the node
        table. The number of sample nodes must be a multiple of ploidy.

        :param int ploidy: The number of samples per individual.
        :return: A 2D array of node IDs, where each row has length `ploidy`.
        :rtype: numpy.ndarray
        """
        if ploidy <= 0 or ploidy != int(ploidy):
            raise ValueError("Ploidy must be a positive integer")
        sample_node_ids = np.flatnonzero(self.nodes_flags & tskit.NODE_IS_SAMPLE)
        num_samples = len(sample_node_ids)
        if num_samples == 0:
            raise ValueError("No sample nodes in tree sequence")
        if num_samples % ploidy != 0:
            raise ValueError(
                f"Number of sample nodes {num_samples} is not a multiple "
                f"of ploidy {ploidy}"
            )
        num_samples_per_individual = num_samples // ploidy
        sample_node_ids = sample_node_ids.reshape((num_samples_per_individual, ploidy))
        return sample_node_ids

    def map_to_vcf_model(
        self,
        individuals=None,
        ploidy=None,
        name_metadata_key=None,
        individual_names=None,
        include_non_sample_nodes=None,
        position_transform=None,
        contig_id=None,
        isolated_as_missing=None,
    ):
        """
        Maps the sample nodes in this tree sequence to a representation suitable for
        VCF output, using the individuals if present.

        Creates a VcfModelMapping object that contains a nodes-to-individual
        mapping as a 2D array of (individuals, nodes), the individual names and VCF
        compatible site positions and contig length. The
        mapping is created by first checking if the tree sequence contains individuals.
        If it does, the mapping is created using the individuals in the tree sequence.
        By default only the sample nodes of the individuals are included in the mapping,
        unless `include_non_sample_nodes` is set to True, in which case all nodes
        belonging to the individuals are included. Any individuals without any nodes
        will have no nodes in their row of the mapping, being essentially of zero ploidy.
        If no individuals are present, the mapping is created using only the sample nodes
        and the specified ploidy.

        As the tskit data model allows non-integer positions, site positions and contig
        length are transformed to integer values suitable for VCF output. The
        transformation is done using the `position_transform` function, which must
        return an integer numpy array the same dimension as the input. By default,
        this is set to ``numpy.round()`` which will round values to the nearest integer.

        If neither `name_metadata_key` nor `individual_names` is not specified, the
        individual names are set to "tsk_{individual_id}" for each individual. If
        no individuals are present, the individual names are set to "tsk_{i}" with
        `0 <= i < num_sample_nodes/ploidy`.

        A Warning are emmitted if any sample nodes do not have an individual ID.

        :param list individuals: Specific individual IDs to include in the VCF. If not
            specified and the tree sequence contains individuals, all individuals are
            included at least one node.
        :param int ploidy: The ploidy, or number of nodes per individual. Only used when
            the tree sequence does not contain individuals. Cannot be used if the tree
            sequence contains individuals. Defaults to 1 if not specified.
        :param str name_metadata_key: The key in the individual metadata to use
            for individual names. Cannot be specified simultaneously with
            individual_names.
        :param list individual_names: The names to use for each individual. Cannot
            be specified simultaneously with name_metadata_key.
        :param bool include_non_sample_nodes: If True, include all nodes belonging to
            the individuals in the mapping. If False, only include sample nodes.
            Defaults to False.
        :param position_transform: A callable that transforms the
            site position values into integer valued coordinates suitable for
            VCF. The function takes a single positional parameter x and must
            return an integer numpy array the same dimension as x. By default,
            this is set to ``numpy.round()`` which will round values to the
            nearest integer. If the string "legacy" is provided here, the
            pre 0.2.0 legacy behaviour of rounding values to the nearest integer
            (starting from 1) and avoiding the output of identical positions
            by incrementing is used.
            See the :ref:`sec_export_vcf_modifying_coordinates` for examples
            and more information.
        :param str contig_id: The ID of the contig to use in the VCF output.
            Defaults to "1" if not specified.
        :param bool isolated_as_missing: If True, isolated samples without mutations
            will be considered as missing data in the VCF output. If False, these samples
            will have the ancestral state in the VCF output.
            Default: True.
        :return: A VcfModelMapping containing the node-to-individual mapping,
            individual names, transformed positions, and transformed contig length.
        :raises ValueError: If both name_metadata_key and individual_names are specified,
            if ploidy is specified when individuals are present, if an invalid individual
            ID is specified, if a specified individual has no nodes, or if the number of
            individuals doesn't match the number of names.
        """
        if include_non_sample_nodes is None:
            include_non_sample_nodes = False

        if contig_id is None:
            contig_id = "1"

        if isolated_as_missing is None:
            isolated_as_missing = True

        if name_metadata_key is not None and individual_names is not None:
            raise ValueError(
                "Cannot specify both name_metadata_key and individual_names"
            )

        if self.num_individuals > 0 and ploidy is not None:
            raise ValueError(
                "Cannot specify ploidy when individuals are present in the tree sequence"
            )

        if self.num_individuals == 0 and include_non_sample_nodes:
            raise ValueError(
                "Cannot include non-sample nodes when individuals are not present in "
                "the tree sequence"
            )

        if self.num_individuals > 0 and np.any(
            np.logical_and(
                self.nodes_individual == tskit.NULL,
                self.nodes_flags & tskit.NODE_IS_SAMPLE,
            )
        ):
            warnings.warn(
                "At least one sample node does not have an individual ID.", stacklevel=1
            )

        if self.num_individuals == 0 and individuals is None:
            if ploidy is None:
                ploidy = 1
            individuals_nodes = self.sample_nodes_by_ploidy(ploidy)
            if individual_names is None:
                individual_names = [f"tsk_{i}" for i in range(len(individuals_nodes))]
        else:
            if individuals is None:
                individuals = np.arange(self.num_individuals, dtype=np.int32)
            if len(individuals) == 0:
                raise ValueError("No individuals specified")
            if min(individuals) < 0 or max(individuals) >= self.num_individuals:
                raise ValueError("Invalid individual ID")

            individuals_nodes = self.individuals_nodes[individuals]
            non_sample_nodes = np.logical_not(
                self.nodes_flags[individuals_nodes] & tskit.NODE_IS_SAMPLE
            )
            if np.any(non_sample_nodes) and not include_non_sample_nodes:
                individuals_nodes[non_sample_nodes] = -1
                rows_to_reorder = np.any(non_sample_nodes, axis=1)
                for i in np.where(rows_to_reorder)[0]:
                    row = individuals_nodes[i]
                    individuals_nodes[i] = np.concatenate(
                        [row[row != -1], row[row == -1]]
                    )

            if individual_names is None:
                if name_metadata_key is not None:
                    individual_names = [
                        self.individual(i).metadata[name_metadata_key]
                        for i in individuals
                    ]
                else:
                    individual_names = [f"tsk_{i}" for i in individuals]

        individual_names = np.array(individual_names, dtype=object)

        if len(individuals_nodes) != len(individual_names):
            raise ValueError(
                "The number of individuals does not match the number of names"
            )

        def legacy_position_transform(positions):
            """
            Transforms positions in the tree sequence into VCF coordinates under
            the pre 0.2.0 legacy rule.
            """
            last_pos = 0
            transformed = []
            for pos in positions:
                pos = int(round(pos))
                if pos <= last_pos:
                    pos = last_pos + 1
                transformed.append(pos)
                last_pos = pos
            return transformed

        if position_transform is None:
            position_transform = np.round
        elif position_transform == "legacy":
            position_transform = legacy_position_transform
        transformed_positions = np.array(
            position_transform(self.sites_position), dtype=int
        )
        if transformed_positions.shape != (self.num_sites,):
            raise ValueError(
                "Position transform must return an array of the same length"
            )
        contig_length = max(1, int(position_transform([self.sequence_length])[0]))

        return VcfModelMapping(
            individuals_nodes,
            individual_names,
            transformed_positions,
            contig_length,
            contig_id,
            isolated_as_missing,
        )

    ############################################
    #
    # Deprecated APIs. These are either already unsupported, or will be unsupported in a
    # later release.
    #
    ############################################

    def get_pairwise_diversity(self, samples=None):
        # Deprecated alias for self.pairwise_diversity
        return self.pairwise_diversity(samples)

    def pairwise_diversity(self, samples=None):
        """
        Returns the pairwise nucleotide site diversity, the average number of sites
        that differ between a every possible pair of distinct samples.  If `samples` is
        specified, calculate the diversity within this set.

         .. deprecated:: 0.2.0
             please use :meth:`.diversity` instead. Since version 0.2.0 the error
             semantics have also changed slightly. It is no longer an error
             when there is one sample and a tskit.LibraryError is raised
             when non-sample IDs are provided rather than a ValueError. It is
             also no longer an error to compute pairwise diversity at sites
             with multiple mutations.

        :param list samples: The set of samples within which we calculate
            the diversity. If None, calculate diversity within the entire sample.
        :return: The pairwise nucleotide site diversity.
        :rtype: float
        """
        if samples is None:
            samples = self.samples()
        return float(
            self.diversity(
                [samples], windows=[0, self.sequence_length], span_normalise=False
            )[0][0]
        )

    def get_time(self, u):
        # Deprecated. Use ts.node(u).time
        if u < 0 or u >= self.get_num_nodes():
            raise ValueError("ID out of bounds")
        node = self.node(u)
        return node.time

    def get_population(self, u):
        # Deprecated. Use ts.node(u).population
        if u < 0 or u >= self.get_num_nodes():
            raise ValueError("ID out of bounds")
        node = self.node(u)
        return node.population

    def records(self):
        # Deprecated. Use either ts.edges() or ts.edgesets().
        t = [node.time for node in self.nodes()]
        pop = [node.population for node in self.nodes()]
        for e in self.edgesets():
            yield CoalescenceRecord(
                e.left, e.right, e.parent, e.children, t[e.parent], pop[e.parent]
            )

    # Unsupported old methods.

    def get_num_records(self):
        raise NotImplementedError(
            "This method is no longer supported. Please use the "
            "TreeSequence.num_edges if possible to work with edges rather "
            "than coalescence records. If not, please use len(list(ts.edgesets())) "
            "which should return the number of coalescence records, as previously "
            "defined. Please open an issue on GitHub if this is "
            "important for your workflow."
        )

    def diffs(self):
        raise NotImplementedError(
            "This method is no longer supported. Please use the "
            "TreeSequence.edge_diffs() method instead"
        )

    def newick_trees(self, precision=3, breakpoints=None, Ne=1):
        raise NotImplementedError(
            "This method is no longer supported. Please use the Tree.newick"
            " method instead"
        )

    def to_nexus(self, precision=14):
        raise NotImplementedError(
            "This method is no longer supported since 0.4.0. Please use the as_nexus "
            "or write_nexus methods instead"
        )


# TODO move to "text_formats.py"
def write_ms(
    tree_sequence,
    output,
    print_trees=False,
    precision=4,
    num_replicates=1,
    write_header=True,
):
    """
    Write ``ms`` formatted output from the genotypes of a tree sequence
    or an iterator over tree sequences. Usage:

    .. code-block:: python

        import tskit as ts

        tree_sequence = msprime.simulate(
            sample_size=sample_size,
            Ne=Ne,
            length=length,
            mutation_rate=mutation_rate,
            recombination_rate=recombination_rate,
            random_seed=random_seed,
            num_replicates=num_replicates,
        )
        with open("output.ms", "w") as ms_file:
            ts.write_ms(tree_sequence, ms_file)

    :param ts tree_sequence: The tree sequence (or iterator over tree sequences) to
        write to ms file
    :param io.IOBase output: The file-like object to write the ms-style output
    :param bool print_trees: Boolean parameter to write out newick format trees
        to output [optional]
    :param int precision: Numerical precision with which to write the ms
        output [optional]
    :param bool write_header: Boolean parameter to write out the header. [optional]
    :param int num_replicates: Number of replicates simulated [required if
        num_replicates used in simulation]

    The first line of this ms-style output file written has two arguments which
    are sample size and number of replicates. The second line has a 0 as a substitute
    for the random seed.
    """
    if not isinstance(tree_sequence, collections.abc.Iterable):
        tree_sequence = [tree_sequence]

    i = 0
    for tree_seq in tree_sequence:
        if i > 0:
            write_header = False
        i = i + 1

        if write_header is True:
            print(
                f"ms {tree_seq.sample_size} {num_replicates}",
                file=output,
            )
            print("0", file=output)

        print(file=output)
        print("//", file=output)
        if print_trees is True:
            """
            Print out the trees in ms-format from the specified tree sequence.
            """
            if len(tree_seq.trees()) == 1:
                tree = next(tree_seq.trees())
                newick = tree.newick(precision=precision)
                print(newick, file=output)
            else:
                for tree in tree_seq.trees():
                    newick = tree.newick(precision=precision)
                    print(f"[{tree.span:.{precision}f}]", newick, file=output)

        else:
            s = tree_seq.get_num_sites()
            print("segsites:", s, file=output)
            if s != 0:
                print("positions: ", end="", file=output)
                positions = [
                    variant.position / (tree_seq.sequence_length)
                    for variant in tree_seq.variants()
                ]
                for position in positions:
                    print(
                        f"{position:.{precision}f}",
                        end=" ",
                        file=output,
                    )
                print(file=output)

                genotypes = tree_seq.genotype_matrix()
                for k in range(tree_seq.num_samples):
                    tmp_str = "".join(map(str, genotypes[:, k]))
                    if set(tmp_str).issubset({"0", "1", "-"}):
                        print(tmp_str, file=output)
                    else:
                        raise ValueError(
                            "This tree sequence contains non-biallelic"
                            "SNPs and is incompatible with the ms format!"
                        )
            else:
                print(file=output)


@dataclass
class PCAResult:
    """
    The result of a call to TreeSequence.pca() capturing the output values
    and algorithm convergence details.


    """

    factors: np.ndarray
    """
    The principal component factors (or scores).
    Columns are orthogonal, with one entry per sample
    or individual (see :meth:`pca <.TreeSequence.pca>`).
    This is the same as the loadings because the GRM is symmetric.
    """
    eigenvalues: np.ndarray
    """
    Eigenvalues of the genetic relatedness matrix.
    """
    range_sketch: np.ndarray
    """
    Range sketch matrix. Can be used as an input for
    :meth:`pca <.TreeSequence.pca>` option to further improve precision.
    """
    error_bound: np.ndarray
    """
    An estimate of the error resulting from the randomized algorithm (experimental).
    Eigenvalues should be correct to within (roughly) this additive factor,
    and factors should be correct to within (roughly) this factor divided by the
    next-largest eigenvalue in the Euclidean norm. These estimates are obtained from
    a bound on the expected L2 operator norm between the true GRM and its
    low-dimensional approximation, from equation 1.11 in
    https://arxiv.org/pdf/0909.4061 .
    """
