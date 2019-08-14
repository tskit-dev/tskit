#
# MIT License
#
# Copyright (c) 2018-2019 Tskit Developers
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
import collections
import itertools
import json
import base64
import warnings
import functools
import concurrent.futures
import textwrap

import numpy as np

import _tskit
import tskit.drawing as drawing
import tskit.exceptions as exceptions
import tskit.provenance as provenance
import tskit.tables as tables
import tskit.formats as formats
import tskit.util as util
import tskit.vcf as vcf

from tskit import NODE_IS_SAMPLE
from tskit import NULL


CoalescenceRecord = collections.namedtuple(
    "CoalescenceRecord",
    ["left", "right", "node", "children", "time", "population"])


# TODO this interface is rubbish. Should have much better printing options.
# TODO we should be use __slots__ here probably.
class SimpleContainer(object):

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return repr(self.__dict__)


class Individual(SimpleContainer):
    """
    An :ref:`individual <sec_individual_table_definition>` in a tree sequence.
    Since nodes correspond to genomes, individuals are associated with a collection
    of nodes (e.g., two nodes per diploid). See :ref:`sec_nodes_or_individuals`
    for more discussion of this distinction.

    Modifying the attributes in this class will have **no effect** on the
    underlying tree sequence data.

    :ivar id: The integer ID of this individual. Varies from 0 to
        :attr:`TreeSequence.num_individuals` - 1.
    :vartype id: int
    :ivar flags: The bitwise flags for this individual.
    :vartype flags: int
    :ivar location: The spatial location of this individual as a numpy array. The
        location is an empty array if no spatial location is defined.
    :vartype location: numpy.ndarray
    :ivar nodes: The IDs of the nodes that are associated with this individual as
        a numpy array (dtype=np.int32). If no nodes are associated with the
        individual this array will be empty.
    :vartype location: numpy.ndarray
    :ivar metadata: The :ref:`metadata <sec_metadata_definition>` for this individual.
    :vartype metadata: bytes
    """
    def __init__(self, id_=None, flags=0, location=None, nodes=None, metadata=""):
        self.id = id_
        self.flags = flags
        self.location = location
        self.metadata = metadata
        self.nodes = nodes

    def __eq__(self, other):
        return (
            self.id == other.id and
            self.flags == other.flags and
            self.metadata == other.metadata and
            np.array_equal(self.nodes, other.nodes) and
            np.array_equal(self.location, other.location))


class Node(SimpleContainer):
    """
    A :ref:`node <sec_node_table_definition>` in a tree sequence, corresponding
    to a single genome. The ``time`` and ``population`` are attributes of the
    ``Node``, rather than the ``Individual``, as discussed in
    :ref:`sec_nodes_or_individuals`.

    Modifying the attributes in this class will have **no effect** on the
    underlying tree sequence data.

    :ivar id: The integer ID of this node. Varies from 0 to
        :attr:`TreeSequence.num_nodes` - 1.
    :vartype id: int
    :ivar flags: The bitwise flags for this node.
    :vartype flags: int
    :ivar time: The birth time of this node.
    :vartype time: float
    :ivar population: The integer ID of the population that this node was born in.
    :vartype population: int
    :ivar individual: The integer ID of the individual that this node was a part of.
    :vartype individual: int
    :ivar metadata: The :ref:`metadata <sec_metadata_definition>` for this node.
    :vartype metadata: bytes
    """
    def __init__(
            self, id_=None, flags=0, time=0, population=NULL,
            individual=NULL, metadata=""):
        self.id = id_
        self.time = time
        self.population = population
        self.individual = individual
        self.metadata = metadata
        self.flags = flags

    def is_sample(self):
        """
        Returns True if this node is a sample. This value is derived from the
        ``flag`` variable.

        :rtype: bool
        """
        return self.flags & NODE_IS_SAMPLE


class Edge(SimpleContainer):
    """
    An :ref:`edge <sec_edge_table_definition>` in a tree sequence.

    Modifying the attributes in this class will have **no effect** on the
    underlying tree sequence data.

    :ivar left: The left coordinate of this edge.
    :vartype left: float
    :ivar right: The right coordinate of this edge.
    :vartype right: float
    :ivar parent: The integer ID of the parent node for this edge.
        To obtain further information about a node with a given ID, use
        :meth:`TreeSequence.node`.
    :vartype parent: int
    :ivar child: The integer ID of the child node for this edge.
        To obtain further information about a node with a given ID, use
        :meth:`TreeSequence.node`.
    :vartype child: int
    :ivar id: The integer ID of this edge. Varies from 0 to
        :attr:`TreeSequence.num_edges` - 1.
    :vartype id: int
    """
    def __init__(self, left, right, parent, child, id_=None):
        self.id = id_
        self.left = left
        self.right = right
        self.parent = parent
        self.child = child

    def __repr__(self):
        return "{{left={:.3f}, right={:.3f}, parent={}, child={}, id={}}}".format(
            self.left, self.right, self.parent, self.child, self.id)

    @property
    def span(self):
        """
        Returns the span of this edge, i.e. the right position minus the left position

        :return: The span of this edge.
        :rtype: float
        """
        return self.right - self.left


class Site(SimpleContainer):
    """
    A :ref:`site <sec_site_table_definition>` in a tree sequence.

    Modifying the attributes in this class will have **no effect** on the
    underlying tree sequence data.

    :ivar id: The integer ID of this site. Varies from 0 to
        :attr:`TreeSequence.num_sites` - 1.
    :vartype id: int
    :ivar position: The floating point location of this site in genome coordinates.
        Ranges from 0 (inclusive) to :attr:`TreeSequence.sequence_length`
        (exclusive).
    :vartype position: float
    :ivar ancestral_state: The ancestral state at this site (i.e., the state
        inherited by nodes, unless mutations occur).
    :vartype ancestral_state: str
    :ivar metadata: The :ref:`metadata <sec_metadata_definition>` for this site.
    :vartype metadata: bytes
    :ivar mutations: The list of mutations at this site. Mutations
        within a site are returned in the order they are specified in the
        underlying :class:`MutationTable`.
    :vartype mutations: list[:class:`Mutation`]
    """
    def __init__(self, id_, position, ancestral_state, mutations, metadata):
        self.id = id_
        self.position = position
        self.ancestral_state = ancestral_state
        self.mutations = mutations
        self.metadata = metadata


class Mutation(SimpleContainer):
    """
    A :ref:`mutation <sec_mutation_table_definition>` in a tree sequence.

    Modifying the attributes in this class will have **no effect** on the
    underlying tree sequence data.

    :ivar id: The integer ID of this mutation. Varies from 0 to
        :attr:`TreeSequence.num_mutations` - 1.
    :vartype id: int
    :ivar site: The integer ID of the site that this mutation occurs at. To obtain
        further information about a site with a given ID use
        :meth:`TreeSequence.site`.
    :vartype site: int
    :ivar node: The integer ID of the first node that inherits this mutation.
        To obtain further information about a node with a given ID, use
        :meth:`TreeSequence.node`.
    :vartype node: int
    :ivar derived_state: The derived state for this mutation. This is the state
        inherited by nodes in the subtree rooted at this mutation's node, unless
        another mutation occurs.
    :vartype derived_state: str
    :ivar parent: The integer ID of this mutation's parent mutation. When multiple
        mutations occur at a site along a path in the tree, mutations must
        record the mutation that is immediately above them. If the mutation does
        not have a parent, this is equal to the :data:`NULL` (-1).
        To obtain further information about a mutation with a given ID, use
        :meth:`TreeSequence.mutation`.
    :vartype parent: int
    :ivar metadata: The :ref:`metadata <sec_metadata_definition>` for this site.
    :vartype metadata: bytes
    """
    def __init__(
            self, id_=NULL, site=NULL, node=NULL, derived_state=None, parent=NULL,
            metadata=None):
        self.id = id_
        self.site = site
        self.node = node
        self.derived_state = derived_state
        self.parent = parent
        self.metadata = metadata


class Migration(SimpleContainer):
    """
    A :ref:`migration <sec_migration_table_definition>` in a tree sequence.

    Modifying the attributes in this class will have **no effect** on the
    underlying tree sequence data.

    :ivar left: The left end of the genomic interval covered by this
        migration (inclusive).
    :vartype left: float
    :ivar right: The right end of the genomic interval covered by this migration
        (exclusive).
    :vartype right: float
    :ivar node: The integer ID of the node involved in this migration event.
        To obtain further information about a node with a given ID, use
        :meth:`TreeSequence.node`.
    :vartype node: int
    :ivar source: The source population ID.
    :vartype source: int
    :ivar dest: The destination population ID.
    :vartype dest: int
    :ivar time: The time at which this migration occured at.
    :vartype time: float
    """
    def __init__(self, left, right, node, source, dest, time, id_=None):
        self.id = id_
        self.left = left
        self.right = right
        self.node = node
        self.source = source
        self.dest = dest
        self.time = time


class Population(SimpleContainer):
    """
    A :ref:`population <sec_population_table_definition>` in a tree sequence.

    Modifying the attributes in this class will have **no effect** on the
    underlying tree sequence data.

    :ivar id: The integer ID of this population. Varies from 0 to
        :attr:`TreeSequence.num_populations` - 1.
    :vartype id: int
    :ivar metadata: The :ref:`metadata <sec_metadata_definition>` for this population.
    :vartype metadata: bytes
    """
    def __init__(self, id_, metadata=""):
        self.id = id_
        self.metadata = metadata


class Variant(SimpleContainer):
    """
    A variant represents the observed variation among samples
    for a given site. A variant consists (a) of a reference to the
    :class:`Site` instance in question; (b) the **alleles** that may be
    observed at the samples for this site; and (c) the **genotypes**
    mapping sample IDs to the observed alleles.

    Each element in the ``alleles`` tuple is a string, representing the
    actual observed state for a given sample. The ``alleles`` tuple is
    generated in one of two ways. The first (and default) way is for
    ``tskit`` to generate the encoding on the fly as alleles are encountered
    while generating genotypes. In this case, the first element of this
    tuple is guaranteed to be the same as the site's ``ancestral_state`` value
    and the list of alleles is also guaranteed not to contain any duplicates.
    Note that allelic values may be listed that are not referred to by any
    samples. For example, if we have a site that is fixed for the derived state
    (i.e., we have a mutation over the tree root), all genotypes will be 1, but
    the alleles list will be equal to ``('0', '1')``. Other than the
    ancestral state being the first allele, the alleles are listed in
    no particular order, and the ordering should not be relied upon
    (but see the notes on missing data below).

    The second way is for the user to define the mapping between
    genotype values and allelic state strings using the
    ``alleles`` parameter to the :meth:`TreeSequence.variants` method.
    In this case, there is no indication of which allele is the ancestral state,
    as the ordering is determined by the user.

    The ``genotypes`` represent the observed allelic states for each sample,
    such that ``var.alleles[var.genotypes[j]]`` gives the string allele
    for sample ID ``j``. Thus, the elements of the genotypes array are
    indexes into the ``alleles`` list. The genotypes are provided in this
    way via a numpy array to enable efficient calculations.

    When :ref:`missing data<sec_data_model_missing_data>` is present at a given
    site boolean flag ``has_missing_data`` will be True, at least one element
    of the ``genotypes`` array will be equal to ``tskit.MISSING_DATA``, and the
    last element of the ``alleles`` array will be ``None``. Note that in this
    case ``variant.num_alleles`` will **not** be equal to
    ``len(variant.alleles)``. The rationale for adding ``None`` to the end of
    the ``alleles`` list is to help code that does not handle missing data
    correctly fail early rather than introducing subtle and hard-to-find bugs.
    As ``tskit.MISSING_DATA`` is equal to -1, code that decodes genotypes into
    allelic values without taking missing data into account would otherwise
    output the last allele in the list rather missing data.

    Modifying the attributes in this class will have **no effect** on the
    underlying tree sequence data.

    :ivar site: The site object for this variant.
    :vartype site: :class:`Site`
    :ivar alleles: A tuple of the allelic values that may be observed at the
        samples at the current site. The first element of this tuple is always
        the site's ancestral state.
    :vartype alleles: tuple(str)
    :ivar genotypes: An array of indexes into the list ``alleles``, giving the
        state of each sample at the current site.
    :ivar has_missing_data: True if there is missing data for any of the
        samples at the current site.
    :vartype has_missing_data: bool
    :ivar num_alleles: The number of distinct alleles at this site. Note that
        this may be greater than the number of distinct values in the genotypes
        array.
    :vartype num_alleles: int
    :vartype genotypes: numpy.ndarray
    """
    def __init__(self, site, alleles, genotypes):
        self.site = site
        self.alleles = alleles
        self.has_missing_data = alleles[-1] is None
        self.num_alleles = len(alleles) - self.has_missing_data
        self.genotypes = genotypes
        # Deprecated aliases to avoid breaking existing code.
        self.position = site.position
        self.index = site.id

    def __eq__(self, other):
        return (
            self.site == other.site and
            self.alleles == other.alleles and
            np.array_equal(self.genotypes, other.genotypes))


class Edgeset(SimpleContainer):
    def __init__(self, left, right, parent, children):
        self.left = left
        self.right = right
        self.parent = parent
        self.children = children

    def __repr__(self):
        return "{{left={:.3f}, right={:.3f}, parent={}, children={}}}".format(
            self.left, self.right, self.parent, self.children)


class Provenance(SimpleContainer):
    def __init__(self, id_=None, timestamp=None, record=None):
        self.id = id_
        self.timestamp = timestamp
        self.record = record


def add_deprecated_mutation_attrs(site, mutation):
    """
    Add in attributes for the older deprecated way of defining
    mutations. These attributes will be removed in future releases
    and are deliberately undocumented in tskit v0.2.2.
    """
    mutation.position = site.position
    mutation.index = site.id
    return mutation


class Tree(object):
    """
    A single tree in a :class:`TreeSequence`. Please see the
    :ref:`sec_tutorial_moving_along_a_tree_sequence` section for information
    on how efficiently access trees sequentially or obtain a list
    of individual trees in a tree sequence.

    The ``sample_counts`` and ``sample_lists`` parameters control the
    features that are enabled for this tree. If ``sample_counts``
    is True, then it is possible to count the number of samples underneath
    a particular node in constant time using the :meth:`num_samples`
    method. If ``sample_lists`` is True a more efficient algorithm is
    used in the :meth:`Tree.samples` method.

    The ``tracked_samples`` parameter can be used to efficiently count the
    number of samples in a given set that exist in a particular subtree
    using the :meth:`Tree.num_tracked_samples` method. It is an
    error to use the ``tracked_samples`` parameter when the ``sample_counts``
    flag is False.

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
    :param bool sample_counts: If True, support constant time sample counts
        via the :meth:`Tree.num_samples` and
        :meth:`Tree.num_tracked_samples` methods.
    :param bool sample_lists: If True, provide more efficient access
        to the samples beneath a give node using the
        :meth:`Tree.samples` method.
    """
    def __init__(
            self, tree_sequence,
            tracked_samples=None, sample_counts=True, sample_lists=False):
        options = 0
        if sample_counts:
            options |= _tskit.SAMPLE_COUNTS
        elif tracked_samples is not None:
            raise ValueError("Cannot set tracked_samples without sample_counts")
        if sample_lists:
            options |= _tskit.SAMPLE_LISTS
        kwargs = {"options": options}
        if tracked_samples is not None:
            # TODO remove this when we allow numpy arrays in the low-level API.
            kwargs["tracked_samples"] = list(tracked_samples)

        self._tree_sequence = tree_sequence
        self._ll_tree = _tskit.Tree(tree_sequence.ll_tree_sequence, **kwargs)

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
        return copy

    @property
    def tree_sequence(self):
        """
        Returns the tree sequence that this tree is from.

        :return: The parent tree sequence for this tree.
        :rtype: :class:`TreeSequence`
        """
        return self._tree_sequence

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

    def next(self):
        """
        Seeks to the next tree in the sequence. If the tree is in the initial
        null state we seek to the first tree (equivalent to calling :meth:`.first`).
        Calling ``next`` on the last tree in the sequence results in the tree
        being cleared back into the null initial state (equivalent to calling
        :meth:`clear`). The return value of the function indicates whether the
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
        null state we seek to the last tree (equivalent to calling :meth:`.last`).
        Calling ``prev`` on the first tree in the sequence results in the tree
        being cleared back into the null initial state (equivalent to calling
        :meth:`clear`). The return value of the function indicates whether the
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

    def seek_index(self, index):
        """
        Sets the state to represent the tree at the specified
        index in the parent tree sequence. Negative indexes following the
        standard Python conventions are allowed, i.e., ``index=-1`` will
        seek to the last tree in the sequence.

        :param int index: The tree index to seek to.
        :raises IndexError: If an index outside the acceptable range is provided.
        """
        num_trees = self.tree_sequence.num_trees
        if index < 0:
            index += num_trees
        if index < 0 or index >= num_trees:
            raise IndexError("Index out of bounds")
        # This should be implemented in C efficiently using the indexes.
        # No point in complicating the current implementation by trying
        # to seek from the correct direction.
        self.first()
        while self.index != index:
            self.next()

    def seek(self, position):
        """
        Sets the state to represent the tree that covers the specified
        position in the parent tree sequence. After a successful return
        of this method we have ``tree.interval[0]`` <= ``position``
        < ``tree.interval[1]``.

        :param float position: The position along the sequence length to
            seek to.
        :raises ValueError: If 0 < position or position >=
            :attr:`TreeSequence.sequence_length`.
        """
        if position < 0 or position >= self.tree_sequence.sequence_length:
            raise ValueError("Position out of bounds")
        # This should be implemented in C efficiently using the indexes.
        # No point in complicating the current implementation by trying
        # to seek from the correct direction.
        self.first()
        while self.interval[1] <= position:
            self.next()

    def get_branch_length(self, u):
        # Deprecated alias for branch_length
        return self.branch_length(u)

    def branch_length(self, u):
        """
        Returns the length of the branch (in generations) joining the
        specified node to its parent. This is equivalent to

        >>> tree.time(tree.parent(u)) - tree.time(u)

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
        units of generations). This is equivalent to

        >>> sum(tree.branch_length(u) for u in tree.nodes())

        Note that the branch lengths for root nodes are defined as zero.

        As this is defined by a traversal of the tree, technically we
        return the sum of all branch lengths that are reachable from
        roots. Thus, this is the sum of all branches that are ancestral
        to at least one sample. This distinction is only important
        in tree sequences that contain 'dead branches', i.e., those
        that define topology not ancestral to any samples.

        :return: The sum of lengths of branches in this tree.
        :rtype: float
        """
        return sum(self.branch_length(u) for u in self.nodes())

    def get_mrca(self, u, v):
        # Deprecated alias for mrca
        return self.mrca(u, v)

    def mrca(self, u, v):
        """
        Returns the most recent common ancestor of the specified nodes.

        :param int u: The first node.
        :param int v: The second node.
        :return: The most recent common ancestor of u and v.
        :rtype: int
        """
        return self._ll_tree.get_mrca(u, v)

    def get_tmrca(self, u, v):
        # Deprecated alias for tmrca
        return self.tmrca(u, v)

    def tmrca(self, u, v):
        """
        Returns the time of the most recent common ancestor of the specified
        nodes. This is equivalent to::

        >>> tree.time(tree.mrca(u, v))

        :param int u: The first node.
        :param int v: The second node.
        :return: The time of the most recent common ancestor of u and v.
        :rtype: float
        """
        return self.get_time(self.get_mrca(u, v))

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

    # Sample list.

    def left_sample(self, u):
        return self._ll_tree.get_left_sample(u)

    def right_sample(self, u):
        return self._ll_tree.get_right_sample(u)

    def next_sample(self, u):
        return self._ll_tree.get_next_sample(u)

    # TODO do we also have right_root?
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
        return self._ll_tree.get_left_root()

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
        Returns the time of the specified node in generations.
        Equivalent to ``tree.tree_sequence.node(u).time``.

        :param int u: The node of interest.
        :return: The time of u.
        :rtype: float
        """
        return self._ll_tree.get_time(u)

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

        :param int u: The node of interest.
        :return: True if u is a leaf node.
        :rtype: bool
        """
        return len(self.children(u)) == 0

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
        Equivalent to ``tree.tree_sequence.num_nodes``. To find the number of
        nodes that are reachable from all roots use ``len(list(tree.nodes()))``.

        :rtype: int
        """
        return self._ll_tree.get_num_nodes()

    @property
    def num_roots(self):
        """
        The number of roots in this tree, as defined in the :attr:`~Tree.roots`
        attribute.

        Requires O(number of roots) time.

        :rtype: int
        """
        return self._ll_tree.get_num_roots()

    @property
    def roots(self):
        """
        The list of roots in this tree. A root is defined as a unique endpoint of
        the paths starting at samples. We can define the set of roots as follows:

        .. code-block:: python

            roots = set()
            for u in tree_sequence.samples():
                while tree.parent(u) != tskit.NULL:
                    u = tree.parent(u)
                roots.add(u)
            # roots is now the set of all roots in this tree.
            assert sorted(roots) == sorted(tree.roots)

        The roots of the tree are returned in a list, in no particular order.

        Requires O(number of roots) time.

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
        :raises: :class:`ValueError` if this tree contains more than one root.
        """
        root = self.left_root
        if root != NULL and self.right_sib(root) != NULL:
            raise ValueError("More than one root exists. Use tree.roots instead")
        return root

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

        :return: A tuple (l, r) representing the left-most (inclusive)
            and right-most (exclusive) coordinates of the genomic region
            covered by this tree.
        :rtype: tuple
        """
        return self._ll_tree.get_left(), self._ll_tree.get_right()

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
        left, right = self.get_interval()
        return right - left

    # The sample_size (or num_samples) is really a property of the tree sequence,
    # and so we should provide access to this via a tree.tree_sequence.num_samples
    # property access. However, we can't just remove the method as a lot of code
    # may depend on it. To complicate things a bit more, sample_size has been
    # changed to num_samples elsewhere for consistency. We can't do this here
    # because there is already a num_samples method which returns the number of
    # samples below a particular node. The best thing to do is probably to
    # undocument the sample_size property, but keep it around for ever.

    def get_sample_size(self):
        # Deprecated alias for self.sample_size
        return self.sample_size

    @property
    def sample_size(self):
        """
        Returns the sample size for this tree. This is the number of sample
        nodes in the tree.

        :return: The number of sample nodes in the tree.
        :rtype: int
        """
        return self._ll_tree.get_sample_size()

    def draw_text(self, orientation=None, **kwargs):
        orientation = drawing.check_orientation(orientation)
        if orientation in (drawing.LEFT, drawing.RIGHT):
            text_tree = drawing.HorizontalTextTree(
                self, orientation=orientation, **kwargs)
        else:
            text_tree = drawing.VerticalTextTree(self, orientation=orientation, **kwargs)
        return str(text_tree)

    def draw_svg(self, path=None, **kwargs):
        # Experimental drawing code. This aims to replace or at least be a more
        # powerful SVG driven interface for the code below.
        draw = drawing.SvgTree(self, **kwargs)
        output = draw.drawing.tostring()
        if path is not None:
            # TODO: removed the pretty here when this is stable.
            draw.drawing.saveas(path, pretty=True)
        return output

    def draw(
            self, path=None, width=None, height=None,
            node_labels=None, node_colours=None,
            mutation_labels=None, mutation_colours=None,
            format=None, edge_colours=None, tree_height_scale=None,
            max_tree_height=None):
        """
        Returns a drawing of this tree.

        When working in a Jupyter notebook, use the ``IPython.display.SVG``
        function to display the SVG output from this function inline in the notebook::

            >>> SVG(tree.draw())

        The unicode format uses unicode `box drawing characters
        <https://en.wikipedia.org/wiki/Box-drawing_character>`_ to render the tree.
        This allows rendered trees to be printed out to the terminal::

            >>> print(tree.draw(format="unicode"))
              6
            ┏━┻━┓
            ┃   5
            ┃ ┏━┻┓
            ┃ ┃  4
            ┃ ┃ ┏┻┓
            3 0 1 2

        The ``node_labels`` argument allows the user to specify custom labels
        for nodes, or no labels at all::

            >>> print(tree.draw(format="unicode", node_labels={}))
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
            are 'svg', 'ascii' and 'unicode'.
        :param dict edge_colours: If specified, show custom colours for the edge
            joining each node in the map to its parent. As for ``node_colours``,
            unspecified edges take the default colour, and ``None`` values result in the
            edge being omitted. (Only supported in the SVG format.)
        :param str tree_height_scale: Control how height values for nodes are computed.
            If this is equal to ``"time"``, node heights are proportional to their time
            values. If this is equal to ``"log_time"``, node heights are proportional to
            their log(time) values. If it is equal to ``"rank"``, node heights are spaced
            equally according to their ranked times. For SVG output the default is
            'time'-scale whereas for text output the default is 'rank'-scale.
            Time scaling is not currently supported for text output.
        :param str,float max_tree_height: The maximum tree height value in the current
            scaling system (see ``tree_height_scale``). Can be either a string or a
            numeric value. If equal to ``"tree"``, the maximum tree height is set to be
            that of the oldest root in the tree. If equal to ``"ts"`` the maximum
            height is set to be the height of the oldest root in the tree sequence;
            this is useful when drawing trees from the same tree sequence as it ensures
            that node heights are consistent. If a numeric value, this is used as the
            maximum tree height by which to scale other nodes. This parameters
            is not currently supported for text output.
        :return: A representation of this tree in the requested format.
        :rtype: str
        """
        output = drawing.draw_tree(
            self, format=format, width=width, height=height,
            node_labels=node_labels, node_colours=node_colours,
            mutation_labels=mutation_labels, mutation_colours=mutation_colours,
            edge_colours=edge_colours, tree_height_scale=tree_height_scale,
            max_tree_height=max_tree_height)
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
        Mutations are returned in order of nondecreasing site ID.
        See the :class:`Mutation` class for details on the available fields for
        each mutation.

        The returned iterator is equivalent to iterating over all sites
        and all mutations in each site, i.e.::

            >>> for site in tree.sites():
            >>>     for mutation in site.mutations:
            >>>         yield mutation

        :return: An iterator over all :class:`Mutation` objects in this tree.
        :rtype: iter(:class:`Mutation`)
        """
        for site in self.sites():
            for mutation in site.mutations:
                yield add_deprecated_mutation_attrs(site, mutation)

    def get_leaves(self, u):
        # Deprecated alias for samples. See the discussion in the get_num_leaves
        # method for why this method is here and why it is semantically incorrect.
        # The 'leaves' iterator below correctly returns the leaves below a given
        # node.
        return self.samples(u)

    def leaves(self, u=None):
        """
        Returns an iterator over all the leaves in this tree that are
        underneath the specified node. If u is not specified, return all leaves
        in the tree.

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
        Returns an iterator over all the samples in this tree that are
        underneath the specified node. If u is a sample, it is included in the
        returned iterator. If u is not specified, return all samples in the tree.

        If the :meth:`TreeSequence.trees` method is called with
        ``sample_lists=True``, this method uses an efficient algorithm to find
        the samples. If not, a simple traversal based method is used.

        :param int u: The node of interest.
        :return: An iterator over all samples in the subtree rooted at u.
        :rtype: collections.abc.Iterable
        """
        roots = [u]
        if u is None:
            roots = self.roots
        for root in roots:
            for v in self._sample_generator(root):
                yield v

    def num_children(self, u):
        """
        Returns the number of children of the specified
        node (i.e. ``len(tree.children(u))``)

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
        Returns the number of samples in this tree underneath the specified
        node (including the node itself). If u is not specified return
        the total number of samples in the tree.

        If the :meth:`TreeSequence.trees` method is called with
        ``sample_counts=True`` this method is a constant time operation. If not,
        a slower traversal based algorithm is used to count the samples.

        :param int u: The node of interest.
        :return: The number of samples in the subtree rooted at u.
        :rtype: int
        """
        if u is None:
            return sum(self._ll_tree.get_num_samples(u) for u in self.roots)
        else:
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
        :raises RuntimeError: if the :meth:`TreeSequence.trees`
            method is not called with ``sample_counts=True``.
        """
        roots = [u]
        if u is None:
            roots = self.roots
        if not (self._ll_tree.get_options() & _tskit.SAMPLE_COUNTS):
            raise RuntimeError(
                "The get_num_tracked_samples method is only supported "
                "when sample_counts=True.")
        return sum(self._ll_tree.get_num_tracked_samples(root) for root in roots)

    def _preorder_traversal(self, u):
        stack = collections.deque([u])
        # For perf we store these to avoid lookups in the tight loop
        pop = stack.pop
        extend = stack.extend
        get_children = self.children
        # Note: the usual style is to be explicit about what we're testing
        # and use while len(stack) > 0, but this form is slightly faster.
        while stack:
            v = pop()
            extend(reversed(get_children(v)))
            yield v

    def _postorder_traversal(self, u):
        stack = collections.deque([u])
        parent = NULL
        # For perf we store these to avoid lookups in the tight loop
        pop = stack.pop
        extend = stack.extend
        get_children = self.children
        get_parent = self.get_parent
        # Note: the usual style is to be explicit about what we're testing
        # and use while len(stack) > 0, but this form is slightly faster.
        while stack:
            v = stack[-1]
            children = [] if v == parent else get_children(v)
            if children:
                extend(reversed(children))
            else:
                parent = get_parent(v)
                yield pop()

    def _inorder_traversal(self, u):
        # TODO add a nonrecursive version of the inorder traversal.
        children = self.get_children(u)
        mid = len(children) // 2
        for c in children[:mid]:
            for v in self._inorder_traversal(c):
                yield v
        yield u
        for c in children[mid:]:
            for v in self._inorder_traversal(c):
                yield v

    def _levelorder_traversal(self, u):
        queue = collections.deque([u])
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

    def _timeasc_traversal(self, u):
        yield from sorted(self.nodes(u, order="levelorder"), key=self.time)

    def _timedesc_traversal(self, u):
        yield from sorted(self.nodes(u, order="levelorder"), key=self.time, reverse=True)

    def nodes(self, root=None, order="preorder"):
        """
        Returns an iterator over the node IDs in this tree. If the root parameter
        is provided, iterate over the node IDs in the subtree rooted at this
        node. If this is None, iterate over all node IDs. If the order parameter
        is provided, iterate over the nodes in required tree traversal order.

        .. note::
            Unlike the :meth:`TreeSequence.nodes` method, this iterator produces
            integer node IDs, not :class:`Node` objects.

        :param int root: The root of the subtree we are traversing.
        :param str order: The traversal ordering. Currently 'preorder',
            'inorder', 'postorder', 'levelorder' ('breadthfirst'), 'timeasc' and
            'timedesc' are supported.
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
            "timedesc": self._timedesc_traversal
        }
        try:
            iterator = methods[order]
        except KeyError:
            raise ValueError("Traversal ordering '{}' not supported".format(order))
        roots = [root]
        if root is None:
            roots = self.roots
        for u in roots:
            for v in iterator(u):
                yield v

    # TODO make this a bit less embarrassing by using an iterative method.
    def __build_newick(self, node, precision, node_labels):
        """
        Simple recursive version of the newick generator used when non-default
        node labels are needed.
        """
        label = node_labels.get(node, "")
        if self.is_leaf(node):
            s = "{}".format(label)
        else:
            s = "("
            for child in self.children(node):
                branch_length = self.branch_length(child)
                subtree = self.__build_newick(child, precision, node_labels)
                s += subtree + ":{0:.{1}f},".format(branch_length, precision)
            s = s[:-1] + "){}".format(label)
        return s

    def newick(self, precision=14, root=None, node_labels=None):
        """
        Returns a `newick encoding <https://en.wikipedia.org/wiki/Newick_format>`_
        of this tree. If the ``root`` argument is specified, return a representation
        of the specified subtree, otherwise the full tree is returned. If the tree
        has multiple roots then seperate newick strings for each rooted subtree
        must be found (i.e., we do not attempt to concatenate the different trees).

        By default, leaf nodes are labelled with their numerical ID + 1,
        and internal nodes are not labelled. Arbitrary node labels can be specified
        using the ``node_labels`` argument, which maps node IDs to the desired
        labels.

        .. warning:: Node labels are **not** Newick escaped, so care must be taken
            to provide labels that will not break the encoding.

        :param int precision: The numerical precision with which branch lengths are
            printed.
        :param int root: If specified, return the tree rooted at this node.
        :param dict node_labels: If specified, show custom labels for the nodes
            that are present in the map. Any nodes not specified in the map will
            not have a node label.
        :return: A newick representation of this tree.
        :rtype: str
        """
        if root is None:
            if self.num_roots > 1:
                raise ValueError(
                    "Cannot get newick for multiroot trees. Try "
                    "[t.newick(root) for root in t.roots] to get a list of "
                    "newick trees, one for each root.")
            root = self.root
        if node_labels is None:
            s = self._ll_tree.get_newick(precision=precision, root=root)
            s = s.decode()
        else:
            return self.__build_newick(root, precision, node_labels) + ";"
        return s

    @property
    def parent_dict(self):
        return self.get_parent_dict()

    def get_parent_dict(self):
        pi = {
            u: self.parent(u) for u in range(self.num_nodes)
            if self.parent(u) != NULL}
        return pi

    def __str__(self):
        return str(self.get_parent_dict())

    def map_mutations(self, genotypes, alleles):
        """
        Given observations for the samples in this tree described by the specified
        set of genotypes and alleles, return a parsimonious set of state transitions
        explaining these observations. The genotypes array is interpreted as indexes
        into the alleles list in the same manner as described in the
        :meth:`TreeSequence.variants` method. Thus, if sample ``j`` carries the
        allele at index ``k``, then we have ``genotypes[j] = k``.
        Missing data can be specified for a sample using the value
        ``tskit.MISSING_DATA`` (-1). At least one non-missing observation must be
        provided. A maximum of 63 alleles are supported.

        The current implementation uses the Fitch parsimony algorithm to determine
        the minimum number of state transitions required to explain the data. In this
        model, transitions between any of the non-missing states is equally likely.

        The returned values correspond directly to the data model for describing
        variation at sites using mutations. See the :ref:`sec_site_table_definition`
        and :ref:`sec_mutation_table_definition` definitions for details and background.

        The state reconstruction is returned as two-tuple, ``(ancestral_state,
        mutations)``, where ``ancestral_state`` is the allele assigned to the
        tree root(s) and ``mutations`` is a list of :class:`Mutation` objects.
        For each mutation, ``node`` is the tree node at the bottom of the branch
        on which the transition occurred, and ``derived_state`` is the new state
        after this mutation. When multiple mutations are returned the ``parent``
        property contains the index of the previous mutation on the path to root
        (see the :ref:`sec_mutation_table_definition` for more information on the
        concept of mutation parents). All other attributes of the :class:`Mutation`
        object are undefined and should not be used.

        See the :ref:`sec_tutorial_parsimony` section in the tutorial for examples
        of how to use this method.

        :param array_like genotypes: The input observations for the samples in this tree.
        :return: The inferred ancestral state and list of mutations on this tree
            that encode the specified observations.
        :rtype: (str, list(tskit.Mutation))
        """
        genotypes = util.safe_np_int_cast(genotypes, np.int8)
        if np.max(genotypes) >= 64:
            raise ValueError("A maximum of 64 states is supported")
        ancestral_state, transitions = self._ll_tree.map_mutations(genotypes)
        # Translate back into string alleles
        ancestral_state = alleles[ancestral_state]
        mutations = [
            Mutation(node=node, derived_state=alleles[derived_state], parent=parent)
            for node, parent, derived_state in transitions]
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
        nodes (i.e., the same IDs in the same order).

        :param Tree other: The other tree to compare to.
        :param float lambda_: The KC metric lambda parameter determining the
            relative weight of topology and branch length.
        :return: The computed KC distance between this tree and other.
        :rtype: float
        """
        return self._ll_tree.get_kc_distance(other._ll_tree, lambda_)


def load(path):
    """
    Loads a tree sequence from the specified file path. This file must be in the
    :ref:`tree sequence file format <sec_tree_sequence_file_format>` produced by the
    :meth:`TreeSequence.dump` method.

    :param str path: The file path of the ``.trees`` file containing the
        tree sequence we wish to load.
    :return: The tree sequence object containing the information
        stored in the specified file path.
    :rtype: :class:`tskit.TreeSequence`
    """
    try:
        return TreeSequence.load(path)
    except exceptions.FileFormatError as e:
        formats.raise_hdf5_format_error(path, e)


def parse_individuals(
        source, strict=True, encoding='utf8', base64_metadata=True, table=None):
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
    header = source.readline().strip("\n").split(sep)
    flags_index = header.index("flags")
    location_index = None
    metadata_index = None
    try:
        location_index = header.index("location")
    except ValueError:
        pass
    try:
        metadata_index = header.index("metadata")
    except ValueError:
        pass
    for line in source:
        tokens = line.split(sep)
        if len(tokens) >= 1:
            flags = int(tokens[flags_index])
            location = ()
            if location_index is not None:
                location_string = tokens[location_index]
                if len(location_string) > 0:
                    location = tuple(map(float, location_string.split(",")))
            metadata = b''
            if metadata_index is not None and metadata_index < len(tokens):
                metadata = tokens[metadata_index].encode(encoding)
                if base64_metadata:
                    metadata = base64.b64decode(metadata)
            table.add_row(
                flags=flags, location=location, metadata=metadata)
    return table


def parse_nodes(
        source, strict=True, encoding='utf8', base64_metadata=True, table=None):
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
    header = source.readline().strip("\n").split(sep)
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
        tokens = line.split(sep)
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
            metadata = b''
            if metadata_index is not None and metadata_index < len(tokens):
                metadata = tokens[metadata_index].encode(encoding)
                if base64_metadata:
                    metadata = base64.b64decode(metadata)
            table.add_row(
                flags=flags, time=time, population=population,
                individual=individual, metadata=metadata)
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
    header = source.readline().strip("\n").split(sep)
    left_index = header.index("left")
    right_index = header.index("right")
    parent_index = header.index("parent")
    children_index = header.index("child")
    for line in source:
        tokens = line.split(sep)
        if len(tokens) >= 4:
            left = float(tokens[left_index])
            right = float(tokens[right_index])
            parent = int(tokens[parent_index])
            children = tuple(map(int, tokens[children_index].split(",")))
            for child in children:
                table.add_row(left=left, right=right, parent=parent, child=child)
    return table


def parse_sites(
        source, strict=True, encoding='utf8', base64_metadata=True, table=None):
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
    header = source.readline().strip("\n").split(sep)
    position_index = header.index("position")
    ancestral_state_index = header.index("ancestral_state")
    metadata_index = None
    try:
        metadata_index = header.index("metadata")
    except ValueError:
        pass
    for line in source:
        tokens = line.split(sep)
        if len(tokens) >= 2:
            position = float(tokens[position_index])
            ancestral_state = tokens[ancestral_state_index]
            metadata = b''
            if metadata_index is not None and metadata_index < len(tokens):
                metadata = tokens[metadata_index].encode(encoding)
                if base64_metadata:
                    metadata = base64.b64decode(metadata)
            table.add_row(
                position=position, ancestral_state=ancestral_state, metadata=metadata)
    return table


def parse_mutations(
        source, strict=True, encoding='utf8', base64_metadata=True, table=None):
    """
    Parse the specified file-like object containing a whitespace delimited
    description of a mutation table and returns the corresponding :class:`MutationTable`
    instance. See the :ref:`mutation text format <sec_mutation_text_format>` section
    for the details of the required format and the
    :ref:`mutation table definition <sec_mutation_table_definition>` section for the
    required properties of the contents.

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
    header = source.readline().strip("\n").split(sep)
    site_index = header.index("site")
    node_index = header.index("node")
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
        tokens = line.split(sep)
        if len(tokens) >= 3:
            site = int(tokens[site_index])
            node = int(tokens[node_index])
            derived_state = tokens[derived_state_index]
            if parent_index is not None:
                parent = int(tokens[parent_index])
            metadata = b''
            if metadata_index is not None and metadata_index < len(tokens):
                metadata = tokens[metadata_index].encode(encoding)
                if base64_metadata:
                    metadata = base64.b64decode(metadata)
            table.add_row(
                site=site, node=node, derived_state=derived_state, parent=parent,
                metadata=metadata)
    return table


def parse_populations(
        source, strict=True, encoding='utf8', base64_metadata=True, table=None):
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
    header = source.readline().strip("\n").split(sep)
    metadata_index = header.index("metadata")
    for line in source:
        tokens = line.split(sep)
        if len(tokens) >= 1:
            metadata = tokens[metadata_index].encode(encoding)
            if base64_metadata:
                metadata = base64.b64decode(metadata)
            table.add_row(metadata=metadata)
    return table


def load_text(nodes, edges, sites=None, mutations=None, individuals=None,
              populations=None, sequence_length=0, strict=True,
              encoding='utf8', base64_metadata=True):
    """
    Parses the tree sequence data from the specified file-like objects, and
    returns the resulting :class:`TreeSequence` object. The format
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
    ``mutations``, ``individuals`` and ``populations`` are optional, and must
    be parsable by :func:`parse_sites`, :func:`parse_individuals`,
    :func:`parse_populations`, and :func:`parse_mutations`, respectively.

    TODO: there is no method to parse the remaining tables at present, so
    only tree sequences not requiring Population and Individual tables can
    be loaded. This will be fixed: https://github.com/tskit-dev/msprime/issues/498

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
        left=edge_table.left, right=edge_table.right, parent=edge_table.parent,
        child=edge_table.child)
    parse_nodes(
        nodes, strict=strict, encoding=encoding, base64_metadata=base64_metadata,
        table=tc.nodes)
    # We need to add populations any referenced in the node table.
    if len(tc.nodes) > 0:
        max_population = tc.nodes.population.max()
        if max_population != NULL:
            for _ in range(max_population + 1):
                tc.populations.add_row()
    if sites is not None:
        parse_sites(
            sites, strict=strict, encoding=encoding, base64_metadata=base64_metadata,
            table=tc.sites)
    if mutations is not None:
        parse_mutations(
            mutations, strict=strict, encoding=encoding,
            base64_metadata=base64_metadata, table=tc.mutations)
    if individuals is not None:
        parse_individuals(
            individuals, strict=strict, encoding=encoding,
            base64_metadata=base64_metadata, table=tc.individuals)
    if populations is not None:
        parse_populations(
            populations, strict=strict, encoding=encoding,
            base64_metadata=base64_metadata, table=tc.populations)
    tc.sort()
    return tc.tree_sequence()


class TreeIterator(object):
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


class SimpleContainerSequence(object):
    """
    Simple wrapper to allow arrays of SimpleContainers (e.g. edges, nodes) that have a
    function allowing access by index (e.g. ts.edge(i), ts.node(i)) to be treated as a
    python sequence, allowing forward and reverse iteration.
    """
    def __init__(self, getter, length):
        self.getter = getter
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Index out of bounds")
        return self.getter(index)


class TreeSequence(object):
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

    @property
    def ll_tree_sequence(self):
        return self.get_ll_tree_sequence()

    def get_ll_tree_sequence(self):
        return self._ll_tree_sequence

    def aslist(self):
        """
        Returns the trees in this tree sequence as a list. Each tree is
        represented by a different instance of :class:`Tree`. As such, this
        method is inefficient and may use a large amount of memory, and should
        not be used when performance is a consideration. The :meth:`.trees`
        method is the recommended way to efficiently iterate over the trees
        in a tree sequence.

        :return: A list of the trees in this tree sequence.
        :rtype: list
        """
        return [tree.copy() for tree in self.trees()]

    @classmethod
    def load(cls, path):
        ts = _tskit.TreeSequence()
        ts.load(str(path))
        return TreeSequence(ts)

    @classmethod
    def load_tables(cls, tables):
        ts = _tskit.TreeSequence()
        ts.load_tables(tables.ll_tables)
        return TreeSequence(ts)

    def dump(self, path, zlib_compression=False):
        """
        Writes the tree sequence to the specified file path.

        :param str path: The file path to write the TreeSequence to.
        :param bool zlib_compression: This parameter is deprecated and ignored.
        """
        if zlib_compression:
            warnings.warn(
                "The zlib_compression option is no longer supported and is ignored",
                RuntimeWarning)
        # Convert the path to str to allow us use Pathlib inputs
        self._ll_tree_sequence.dump(str(path))

    @property
    def tables(self):
        """
        A copy of the tables underlying this tree sequence. See also
        :meth:`.dump_tables`.

        .. warning:: This propery currently returns a copy of the tables
            underlying a tree sequence but it may return a read-only
            **view** in the future. Thus, if the tables will subsequently be
            updated, please use the :meth:`.dump_tables` method instead as
            this will always return a new copy of the TableCollection.

        :return: A :class:`TableCollection` containing all a copy of the
            tables underlying this tree sequence.
        :rtype: TableCollection
        """
        return self.dump_tables()

    def dump_tables(self):
        """
        A copy of the tables defining this tree sequence.

        :return: A :class:`TableCollection` containing all tables underlying
            the tree sequence.
        :rtype: TableCollection
        """
        t = tables.TableCollection(sequence_length=self.sequence_length)
        self._ll_tree_sequence.dump_tables(t.ll_tables)
        return t

    def dump_text(
            self, nodes=None, edges=None, sites=None, mutations=None, individuals=None,
            populations=None, provenances=None, precision=6, encoding='utf8',
            base64_metadata=True):
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
        :param io.TextIOBase provenances: The file-like object to write the
            ProvenanceTable to.
        :param int precision: The number of digits of precision.
        :param str encoding: Encoding used for text representation.
        :param bool base64_metadata: If True, metadata is encoded using Base64
            encoding; otherwise, as plain text.
        """

        if nodes is not None:
            print(
                "id", "is_sample", "time", "population", "individual", "metadata",
                sep="\t", file=nodes)
            for node in self.nodes():
                metadata = node.metadata
                if base64_metadata:
                    metadata = base64.b64encode(metadata).decode(encoding)
                row = (
                    "{id:d}\t"
                    "{is_sample:d}\t"
                    "{time:.{precision}f}\t"
                    "{population:d}\t"
                    "{individual:d}\t"
                    "{metadata}").format(
                        precision=precision, id=node.id,
                        is_sample=node.is_sample(), time=node.time,
                        population=node.population,
                        individual=node.individual,
                        metadata=metadata)
                print(row, file=nodes)

        if edges is not None:
            print("left", "right", "parent", "child", sep="\t", file=edges)
            for edge in self.edges():
                row = (
                    "{left:.{precision}f}\t"
                    "{right:.{precision}f}\t"
                    "{parent:d}\t"
                    "{child:d}").format(
                        precision=precision, left=edge.left, right=edge.right,
                        parent=edge.parent, child=edge.child)
                print(row, file=edges)

        if sites is not None:
            print("position", "ancestral_state", "metadata", sep="\t", file=sites)
            for site in self.sites():
                metadata = site.metadata
                if base64_metadata:
                    metadata = base64.b64encode(metadata).decode(encoding)
                row = (
                    "{position:.{precision}f}\t"
                    "{ancestral_state}\t"
                    "{metadata}").format(
                        precision=precision, position=site.position,
                        ancestral_state=site.ancestral_state,
                        metadata=metadata)
                print(row, file=sites)

        if mutations is not None:
            print(
                "site", "node", "derived_state", "parent", "metadata",
                sep="\t", file=mutations)
            for site in self.sites():
                for mutation in site.mutations:
                    metadata = mutation.metadata
                    if base64_metadata:
                        metadata = base64.b64encode(metadata).decode(encoding)
                    row = (
                        "{site}\t"
                        "{node}\t"
                        "{derived_state}\t"
                        "{parent}\t"
                        "{metadata}").format(
                            site=mutation.site, node=mutation.node,
                            derived_state=mutation.derived_state,
                            parent=mutation.parent,
                            metadata=metadata)
                    print(row, file=mutations)

        if individuals is not None:
            print(
                "id", "flags", "location", "metadata",
                sep="\t", file=individuals)
            for individual in self.individuals():
                metadata = individual.metadata
                if base64_metadata:
                    metadata = base64.b64encode(metadata).decode(encoding)
                location = ",".join(map(str, individual.location))
                row = (
                    "{id}\t"
                    "{flags}\t"
                    "{location}\t"
                    "{metadata}").format(
                        id=individual.id, flags=individual.flags,
                        location=location, metadata=metadata)
                print(row, file=individuals)

        if populations is not None:
            print(
                "id", "metadata",
                sep="\t", file=populations)
            for population in self.populations():
                metadata = population.metadata
                if base64_metadata:
                    metadata = base64.b64encode(metadata).decode(encoding)
                row = (
                    "{id}\t"
                    "{metadata}").format(id=population.id, metadata=metadata)
                print(row, file=populations)

        if provenances is not None:
            print("id", "timestamp", "record", sep="\t", file=provenances)
            for provenance in self.provenances():
                row = (
                    "{id}\t"
                    "{timestamp}\t"
                    "{record}\t").format(
                        id=provenance.id,
                        timestamp=provenance.timestamp,
                        record=provenance.record)
                print(row, file=provenances)

    # num_samples was originally called sample_size, and so we must keep sample_size
    # around as a deprecated alias.
    @property
    def num_samples(self):
        """
        Returns the number of samples in this tree sequence. This is the number
        of sample nodes in each tree.

        :return: The number of sample nodes in this tree sequence.
        :rtype: int
        """
        return self._ll_tree_sequence.get_num_samples()

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
        Returns time of the oldest root in any of the trees in this tree sequence.
        This is usually equal to ``np.max(ts.tables.nodes.time)`` but may not be
        since there can be nodes that are not present in any tree. Consistent
        with the definition of tree roots, if there are no edges in the tree
        sequence we return the time of the oldest sample.

        :return: The maximum time of a root in this tree sequence.
        :rtype: float
        """
        ret = max(self.node(u).time for u in self.samples())
        if self.num_edges > 0:
            # Edges are guaranteed to be listed in parent-time order, so we can get the
            # last one to get the oldest root.
            edge = self.edge(self.num_edges - 1)
            # However, we can have situations where there is a sample older than a
            # 'proper' root
            ret = max(ret, self.node(edge.parent).time)
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

    def nodes(self):
        """
        Returns an iterable sequence of all the :ref:`nodes <sec_node_table_definition>`
        in this tree sequence.

        :return: An iterable sequence of all nodes.
        :rtype: Sequence(:class:`.Node`)
        """
        return SimpleContainerSequence(self.node, self.num_nodes)

    def edges(self):
        """
        Returns an iterable sequence of all the :ref:`edges <sec_edge_table_definition>`
        in this tree sequence. Edges are returned in the order required
        for a :ref:`valid tree sequence <sec_valid_tree_sequence_requirements>`. So,
        edges are guaranteed to be ordered such that (a) all parents with a
        given ID are contiguous; (b) edges are returned in non-descreasing
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
                if len(children[edge.parent]) > 0 and edge.parent not in active_edgesets:
                    active_edgesets[edge.parent] = Edgeset(left, right, edge.parent, [])

        for parent in active_edgesets.keys():
            edgeset = active_edgesets[parent]
            edgeset.right = self.sequence_length
            edgeset.children = sorted(children[edgeset.parent])
            yield edgeset

    def edge_diffs(self):
        """
        Returns an iterator over all the edges that are inserted and removed to
        build the trees as we move from left-to-right along the tree sequence.
        The iterator yields a sequence of 3-tuples, ``(interval, edges_out,
        edges_in)``. The ``interval`` is a pair ``(left, right)`` representing
        the genomic interval (see :attr:`Tree.interval`). The ``edges_out``
        value is a tuple of the edges that were just-removed to create the tree
        covering the interval (hence ``edges_out`` will always be empty for the
        first tree). The ``edges_in`` value is a tuple of edges that were just
        inserted to construct the tree convering the current interval. The
        iterator terminates after the final interval in the tree sequence (i.e.
        it does not report a final removal of all remaining edges); the number
        of iterations is thus always equal to the number of trees in the tree
        sequence.

        :return: An iterator over the (interval, edges_out, edges_in) tuples.
        :rtype: :class:`collections.abc.Iterable`
        """
        iterator = _tskit.TreeDiffIterator(self._ll_tree_sequence)
        for interval, edge_tuples_out, edge_tuples_in in iterator:
            edges_out = [Edge(*e) for e in edge_tuples_out]
            edges_in = [Edge(*e) for e in edge_tuples_in]
            yield interval, edges_out, edges_in

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

            >>> for site in tree_sequence.sites():
            >>>     for mutation in site.mutations:
            >>>         yield mutation

        :return: An iterator over all mutations in this tree sequence.
        :rtype: iter(:class:`Mutation`)
        """
        for site in self.sites():
            for mutation in site.mutations:
                yield add_deprecated_mutation_attrs(site, mutation)

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
        Returns the breakpoints along the chromosome, including the two extreme points
        0 and L. This is equivalent to

        >>> iter([0] + [t.interval[1] for t in self.trees()])

        By default we return an iterator over the breakpoints as Python float objects;
        if ``as_array`` is True we return them as a numpy array.

        Note that the ``as_array`` form will be more efficient and convenient in most
        cases; the default iterator behaviour is mainly kept to ensure compatability
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

    def at(self, position):
        """
        Returns the tree covering the specified genomic location. The returned tree
        will have ``tree.interval[0]`` <= ``position`` < ``tree.interval[1]``.
        See also :meth:`Tree.seek`.

        :return: A new instance of :class:`Tree` positioned to cover the specified
            position.
        :rtype: Tree
        """
        tree = Tree(self)
        tree.seek(position)
        return tree

    def at_index(self, index):
        """
        Returns the tree at the specified index. See also :meth:`Tree.seek_index`.

        :return: A new instance of :class:`Tree` positioned at the specified index.
        :rtype: Tree
        """
        tree = Tree(self)
        tree.seek_index(index)
        return tree

    def first(self):
        """
        Returns the first tree in this :class:`TreeSequence`. To iterate over all
        trees in the sequence, use the :meth:`.trees` method.

        :return: The first tree in this tree sequence.
        :rtype: :class:`Tree`.
        """
        tree = Tree(self)
        tree.first()
        return tree

    def last(self):
        """
        Returns the last tree in this :class:`TreeSequence`. To iterate over all
        trees in the sequence, use the :meth:`.trees` method.

        :return: The last tree in this tree sequence.
        :rtype: :class:`Tree`.
        """
        tree = Tree(self)
        tree.last()
        return tree

    def trees(
            self, tracked_samples=None, sample_counts=True, sample_lists=False,
            tracked_leaves=None, leaf_counts=None, leaf_lists=None):
        """
        Returns an iterator over the trees in this tree sequence. Each value
        returned in this iterator is an instance of :class:`Tree`. Upon
        successful termination of the iterator, the tree will be in the
        "cleared" null state.

        The ``sample_counts``, ``sample_lists`` and ``tracked_samples``
        parameters are passed to the :class:`Tree` constructor, and control
        the options that are set in the returned tree instance.

        :warning: Do not store the results of this iterator in a list!
           For performance reasons, the same underlying object is used
           for every tree returned which will most likely lead to unexpected
           behaviour. If you wish to obtain a list of trees in a tree sequence
           please use ``ts.aslist()`` instead.

        :param list tracked_samples: The list of samples to be tracked and
            counted using the :meth:`Tree.num_tracked_samples` method.
        :param bool sample_counts: If True, support constant time sample counts
            via the :meth:`Tree.num_samples` and
            :meth:`Tree.num_tracked_samples` methods.
        :param bool sample_lists: If True, provide more efficient access
            to the samples beneath a give node using the
            :meth:`Tree.samples` method.
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
            self, tracked_samples=tracked_samples, sample_counts=sample_counts,
            sample_lists=sample_lists)
        return TreeIterator(tree)

    def haplotypes(self, impute_missing_data=False, missing_data_character="-"):
        """
        Returns an iterator over the strings of haplotypes that result from
        the trees and mutations in this tree sequence. Each haplotype string
        is guaranteed to be of the same length. A tree sequence with
        :math:`n` samples and :math:`s` sites will return a total of :math:`n`
        strings of :math:`s` alleles concatenated together, where an allele
        consists of a single ascii character (tree sequences that include alleles
        which are not a single character in length, or where the character is
        non-ascii, will raise an error). The first string returned is the
        haplotype for sample ``0``, and so on.

        The alleles at each site must be represented by single byte characters,
        (i.e. variants must be single nucleotide polymorphisms, or SNPs), hence
        the strings returned will all be of length :math:`s`, and for a haplotype
        ``h``, the value of ``h[j]`` will be the observed allelic state
        at site ``j``.

        If ``impute_missing_data`` is False,
        :ref:`missing data<sec_data_model_missing_data>` will be represented
        in the string by the ``missing_data_character``. If
        instead it is set to True, missing data will be imputed such that all
        isolated samples are assigned the ancestral state (unless they have
        mutations directly above them, in which case they will take the most recent
        derived  mutational state for that node). This was the default
        behaviour in versions prior to 0.2.0.

        See also the :meth:`.variants` iterator for site-centric access
        to sample genotypes.

        .. warning::
            For large datasets, this method can consume a **very large** amount of
            memory! To output all the sample data, it is more efficient to iterate
            over sites rather than over samples. If you have a large dataset but only
            want to output the haplotypes for a subset of samples, it may be worth
            calling :meth:`.simplify` to reduce tree sequence down to the required
            samples before outputting haplotypes.

        :return: An iterator over the haplotype strings for the samples in
            this tree sequence.
        :param bool impute_missing_data: If True, the allele assigned to any
            isolated samples is the ancestral state; that is, we impute
            missing data as the ancestral state. Default: False.
        :param str missing_data_character: A single ascii character that will
            be used to represent missing data.
            If any normal allele contains this character, an error is raised.
            Default: '-'.
        :rtype: collections.abc.Iterable
        :raises: TypeError if the ``missing_data_character`` or any of the alleles
            at a site or the are not a single ascii character.
        :raises: ValueError
            if the ``missing_data_character`` exists in one of the alleles
        """
        H = np.empty((self.num_samples, self.num_sites), dtype=np.int8)
        missing_int8 = ord(missing_data_character.encode('ascii'))
        for var in self.variants(impute_missing_data=impute_missing_data):
            alleles = np.full(len(var.alleles), missing_int8, dtype=np.int8)
            for i, allele in enumerate(var.alleles):
                if allele is not None:
                    if len(allele) != 1:
                        raise TypeError(
                            "Multi-letter allele or deletion detected at site {}"
                            .format(var.site.id))
                    try:
                        ascii_allele = allele.encode('ascii')
                    except UnicodeEncodeError:
                        raise TypeError(
                            "Non-ascii character in allele at site {}"
                            .format(var.site.id))
                    allele_int8 = ord(ascii_allele)
                    if allele_int8 == missing_int8:
                        raise ValueError(
                            "The missing data character '{}' clashes with an "
                            "existing allele at site {}"
                            .format(missing_data_character, var.site.id))
                    alleles[i] = allele_int8
            H[:, var.site.id] = alleles[var.genotypes]
        for h in H:
            yield h.tostring().decode('ascii')

    def variants(
            self, as_bytes=False, samples=None, impute_missing_data=False,
            alleles=None):
        """
        Returns an iterator over the variants in this tree sequence. See the
        :class:`Variant` class for details on the fields of each returned
        object. The ``genotypes`` for the variants are numpy arrays,
        corresponding to indexes into the ``alleles`` attribute in the
        :class:`Variant` object. By default, the ``alleles`` for each
        site are generated automatically, such that the ancestral state
        is at the zeroth index and subsequent alleles are listed in no
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
        ``impute_missing_data`` must be True in this case, as it is not
        possible to detect missing data for non-sample nodes.

        If :ref:`missing data<sec_data_model_missing_data>` is present
        at a given site, the genotypes array will contain a special value
        :data:`MISSING_DATA` (-1) to identify these missing samples,
        and the ``alleles`` tuple will end with the value ``None``
        (note that this is true whether we specify a fixed mapping
        using the ``alleles`` parameter or not). See the :class:`Variant`
        class for more details on how missing data is reported.

        Missing data is reported by default, but if ``impute_missing_data``
        is set to to True, missing data will be imputed such that all
        isolated samples are assigned the ancestral state (unless they have
        mutations directly above them, in which case they will take the most
        recent derived mutational state for that node). This was the
        default behaviour in versions prior to 0.2.0.

        .. note::
            The ``as_bytes`` parameter is kept as a compatibility
            option for older code. It is not the recommended way of
            accessing variant data, and will be deprecated in a later
            release.

        :param bool as_bytes: If True, the genotype values will be returned
            as a Python bytes object. Legacy use only.
        :param array_like samples: An array of sample IDs for which to generate
            genotypes, or None for all samples. Default: None.
        :param bool impute_missing_data: If True, the allele assigned to any
            isolated samples is the ancestral state; that is, we impute
            missing data as the ancestral state. Default: False.
        :param tuple alleles: A tuple of strings defining the encoding of
            alleles as integer genotype values. At least one allele must be provided.
            If duplicate alleles are provided, output genotypes will always be
            encoded as the first occurance of the allele. If None (the default),
            the alleles are encoded as they are encountered during genotype
            generation.
        :return: An iterator of all variants this tree sequence.
        :rtype: iter(:class:`Variant`)
        """
        # See comments for the Variant type for discussion on why the
        # present form was chosen.
        iterator = _tskit.VariantGenerator(
            self._ll_tree_sequence, samples=samples,
            impute_missing_data=impute_missing_data, alleles=alleles)
        for site_id, genotypes, alleles in iterator:
            site = self.site(site_id)
            if as_bytes:
                if any(len(allele) > 1 for allele in alleles):
                    raise ValueError(
                        "as_bytes only supported for single-letter alleles")
                bytes_genotypes = np.empty(self.num_samples, dtype=np.uint8)
                lookup = np.array([ord(a[0]) for a in alleles], dtype=np.uint8)
                bytes_genotypes[:] = lookup[genotypes]
                genotypes = bytes_genotypes.tobytes()
            yield Variant(site, alleles, genotypes)

    def genotype_matrix(self, impute_missing_data=False, alleles=None):
        """
        Returns an :math:`m \\times n` numpy array of the genotypes in this
        tree sequence, where :math:`m` is the number of sites and :math:`n`
        the number of samples. The genotypes are the indexes into the array
        of ``alleles``, as described for the :class:`Variant` class.

        If there is :ref:`missing data<sec_data_model_missing_data>` present
        the genotypes array will contain a special value
        ``tskit.MISSING_DATA`` (-1) to identify these missing samples.
        Missing data is reported by default, but if ``impute_missing_data``
        is set to to True, missing data will be imputed such that all
        isolated samples are assigned the ancestral state. This was the
        default behaviour in versions prior to 0.2.0.

        .. warning::
            This method can consume a **very large** amount of memory! If
            all genotypes are not needed at once, it is usually better to
            access them sequentially using the :meth:`.variants` iterator.

        :param bool impute_missing_data: If True, the allele assigned to any
            isolated samples is the ancestral state; that is, we impute
            missing data as the ancestral state. Default: False.
        :param tuple alleles: A tuple of strings describing the encoding of
            alleles to genotype values. At least one allele must be provided.
            If duplicate alleles are provided, output genotypes will always be
            encoded as the first occurance of the allele. If None (the default),
            the alleles are encoded as they are encountered during genotype
            generation.
        :return: The full matrix of genotypes.
        :rtype: numpy.ndarray (dtype=np.int8)
        """
        return self._ll_tree_sequence.get_genotype_matrix(
            impute_missing_data=impute_missing_data, alleles=alleles)

    def individual(self, id_):
        """
        Returns the :ref:`individual <sec_individual_table_definition>`
        in this tree sequence with the specified ID.

        :rtype: :class:`Individual`
        """
        flags, location, metadata, nodes = self._ll_tree_sequence.get_individual(id_)
        return Individual(
            id_=id_, flags=flags, location=location, metadata=metadata, nodes=nodes)

    def node(self, id_):
        """
        Returns the :ref:`node <sec_node_table_definition>` in this tree sequence
        with the specified ID.

        :rtype: :class:`Node`
        """
        (flags, time, population, individual,
         metadata) = self._ll_tree_sequence.get_node(id_)
        return Node(
            id_=id_, flags=flags, time=time, population=population,
            individual=individual, metadata=metadata)

    def edge(self, id_):
        """
        Returns the :ref:`edge <sec_edge_table_definition>` in this tree sequence
        with the specified ID.

        :rtype: :class:`Edge`
        """
        left, right, parent, child = self._ll_tree_sequence.get_edge(id_)
        return Edge(id_=id_, left=left, right=right, parent=parent, child=child)

    def migration(self, id_):
        """
        Returns the :ref:`migration <sec_migration_table_definition>` in this tree
        sequence with the specified ID.

        :rtype: :class:`.Migration`
        """
        left, right, node, source, dest, time = self._ll_tree_sequence.get_migration(id_)
        return Migration(
            id_=id_, left=left, right=right, node=node, source=source, dest=dest,
            time=time)

    def mutation(self, id_):
        """
        Returns the :ref:`mutation <sec_mutation_table_definition>` in this tree sequence
        with the specified ID.

        :rtype: :class:`Mutation`
        """
        (site, node, derived_state, parent,
         metadata) = self._ll_tree_sequence.get_mutation(id_)
        return Mutation(
            id_=id_, site=site, node=node, derived_state=derived_state, parent=parent,
            metadata=metadata)

    def site(self, id_):
        """
        Returns the :ref:`site <sec_site_table_definition>` in this tree sequence
        with the specified ID.

        :rtype: :class:`Site`
        """
        ll_site = self._ll_tree_sequence.get_site(id_)
        pos, ancestral_state, ll_mutations, _, metadata = ll_site
        mutations = [self.mutation(mut_id) for mut_id in ll_mutations]
        return Site(
            id_=id_, position=pos, ancestral_state=ancestral_state,
            mutations=mutations, metadata=metadata)

    def population(self, id_):
        """
        Returns the :ref:`population <sec_population_table_definition>`
        in this tree sequence with the specified ID.

        :rtype: :class:`Population`
        """
        metadata, = self._ll_tree_sequence.get_population(id_)
        return Population(id_=id_, metadata=metadata)

    def provenance(self, id_):
        timestamp, record = self._ll_tree_sequence.get_provenance(id_)
        return Provenance(id_=id_, timestamp=timestamp, record=record)

    def get_samples(self, population_id=None):
        # Deprecated alias for samples()
        return self.samples(population_id)

    def samples(self, population=None, population_id=None):
        """
        Returns an array of the sample node IDs in this tree sequence. If the
        ``population`` parameter is specified, only return sample IDs from this
        population.

        :param int population: The population of interest. If None,
            return all samples.
        :param int population_id: Deprecated alias for ``population``.
        :return: A numpy array of the node IDs for the samples of interest.
        :rtype: numpy.ndarray (dtype=np.int32)
        """
        if population is not None and population_id is not None:
            raise ValueError(
                "population_id and population are aliases. Cannot specify both")
        if population_id is not None:
            population = population_id
        samples = self._ll_tree_sequence.get_samples()
        if population is not None:
            sample_population = self.tables.nodes.population[samples]
            samples = samples[sample_population == population]
        return samples

    def write_fasta(self, output, sequence_ids=None, wrap_width=60):
        ""
        # suppress fasta visibility pending https://github.com/tskit-dev/tskit/issues/353
        """
        Writes haplotype data for samples in FASTA format to the
        specified file-like object.

        Default `sequence_ids` (i.e. the text immediately following ">") are
        "tsk_{sample_number}" e.g. "tsk_0", "tsk_1" etc. They can be set by providing
        a list of strings to the `sequence_ids` argument, which must equal the length
        of the number of samples. Please ensure that these are unique and compatible with
        fasta standards, since we do not check this.
        Default `wrap_width` for sequences is 60 characters in accordance with fasta
        standard outputs, but this can be specified. In order to avoid any line-wrapping
        of sequences, set `wrap_width = 0`.

        Example usage:

        .. code-block:: python

            with open("output.fasta", "w") as fasta_file:
                ts.write_fasta(fasta_file)

        This can also be achieved on the command line use the ``tskit fasta`` command,
        e.g.:

        .. code-block:: bash

            $ tskit fasta example.trees > example.fasta

        :param io.IOBase output: The file-like object to write the fasta output.
        :param list(str) sequence_ids: A list of string names to uniquely identify
            each of the sequences in the fasta file. If specified, this must be a
            list of strings of length equal to the number of samples which are output.
            Note that we do not check the form of these strings in any way, so that it
            is possible to output bad fasta IDs (for example, by including spaces
            before the unique identifying part of the string).
            The default is to output ``tsk_j`` for the jth individual.
        :param int wrap_width: This parameter specifies the number of sequence
            characters to include on each line in the fasta file, before wrapping
            to the next line for each sequence. Defaults to 60 characters in
            accordance with fasta standard outputs. To avoid any line-wrapping of
            sequences, set `wrap_width = 0`. Otherwise, supply any positive integer.
        """
        # if not specified, IDs default to sample index
        if sequence_ids is None:
            sequence_ids = ["tsk_{}".format(j) for j in self.samples()]
        if len(sequence_ids) != self.num_samples:
            raise ValueError(
                "sequence_ids must have length equal to the number of samples.")

        wrap_width = int(wrap_width)
        if wrap_width < 0:
            raise ValueError("wrap_width must be a non-negative integer. "
                             "You may specify `wrap_width=0` "
                             "if you do not want any wrapping.")

        for j, hap in enumerate(self.haplotypes()):
            print(">", sequence_ids[j], sep="", file=output)
            if wrap_width == 0:
                print(hap, file=output)
            else:
                for hap_wrap in textwrap.wrap(hap, wrap_width):
                    print(hap_wrap, file=output)

    def write_vcf(
            self, output, ploidy=None, contig_id="1", individuals=None,
            individual_names=None, position_transform=None):
        """
        Writes a VCF formatted file to the specified file-like object.
        If there is individual information present in the tree sequence
        (see :ref:`sec_individual_table_definition`), the values for
        sample nodes associated with these individuals are combined
        into phased multiploid individuals and output.

        If there is no individual data present in the tree sequence, synthetic
        individuals are created by combining adjacent samples, and the number
        of samples combined is equal to the specified ploidy value (1 by
        default). For example, if we have a ploidy of 2 and a sample of size 6,
        then we will have 3 diploid samples in the output, consisting of the
        combined genotypes for samples [0, 1], [2, 3] and [4, 5]. If we had
        genotypes 011110 at a particular variant, then we would output the
        diploid genotypes 0|1, 1|1 and 1|0 in VCF.

        Each individual in the output is identified by a string; these are the
        VCF "sample" names. By default, these are of the form ``tsk_0``,
        ``tsk_1`` etc, up to the number of individuals, but can be manually
        specified using the ``individual_names`` argument. We do not check
        for duplicates in this array, or perform any checks to ensure that
        the output VCF is well-formed.

        The REF value in the output VCF is the ancestral allele for a site
        and ALT values are the remaining alleles. It is important to note,
        therefore, that for real data this means that the REF value for a given
        site **may not** be equal to the reference allele. We also do not
        check that the alleles result in a valid VCF---for example, it is possible
        to use the tab character as an allele, leading to a broken VCF.

        The ``position_transform`` argument provides a way to flexibly translate
        the genomic location of sites in tskit to the appropriate value in VCF.
        There are two fundamental differences in the way that tskit and VCF define
        genomic coordinates. The first is that tskit uses floating point values
        to encode positions, whereas VCF uses integers. Thus, if the tree sequence
        contains positions at non-integral locations there is an information loss
        incurred by translating to VCF. By default, we round the site positions
        to the nearest integer, such that there may be several sites with the
        same integer position in the output. The second difference between VCF
        and tskit is that VCF is defined to be a 1-based coordinate system, whereas
        tskit uses 0-based. However, how coordinates are transformed depends
        on the VCF parser, and so we do **not** account for this change in
        coordinate system by default.

        Example usage:

        .. code-block:: python

            with open("output.vcf", "w") as vcf_file:
                tree_sequence.write_vcf(vcf_file, ploidy=2)

        The VCF output can also be compressed using the :mod:`gzip` module, if you wish:

        .. code-block:: python

            import gzip
            with gzip.open("output.vcf.gz", "wt") as f:
                ts.write_vcf(f)

        However, this gzipped VCF may not be fully compatible with downstream tools
        such as tabix, which may require the VCF use the specialised bgzip format.
        A general way to convert VCF data to various formats is to pipe the text
        produced by ``tskit`` into ``bcftools``, as done here:

        .. code-block:: python

            import os
            import subprocess

            read_fd, write_fd = os.pipe()
            write_pipe = os.fdopen(write_fd, "w")
            with open("output.bcf", "w") as bcf_file:
                proc = subprocess.Popen(
                    ["bcftools", "view", "-O", "b"], stdin=read_fd, stdout=bcf_file)
                ts.write_vcf(write_pipe)
                write_pipe.close()
                os.close(read_fd)
                proc.wait()
                if proc.returncode != 0:
                    raise RuntimeError("bcftools failed with status:", proc.returncode)

        This can also be achieved on the command line use the ``tskit vcf`` command,
        e.g.:

        .. code-block:: bash

            $ tskit vcf example.trees | bcftools view -O b > example.bcf

        :param io.IOBase output: The file-like object to write the VCF output.
        :param int ploidy: The ploidy of the individuals to be written to
            VCF. This sample size must be evenly divisible by ploidy.
        :param str contig_id: The value of the CHROM column in the output VCF.
        :param list(int) individuals: A list containing the individual IDs to
            write out to VCF. Defaults to all individuals in the tree sequence.
        :param list(str) individual_names: A list of string names to identify
            individual columns in the VCF. In VCF nomenclature, these are the
            sample IDs. If specified, this must be a list of strings of
            length equal to the number of individuals to be output. Note that
            we do not check the form of these strings in any way, so that is
            is possible to output malformed VCF (for example, by embedding a
            tab character within on of the names). The default is to output
            ``tsk_j`` for the jth individual.
        :param position_transform: A callable that transforms the
            site position values into integer valued coordinates suitable for
            VCF. The function takes a single positional parameter x and must
            return an integer numpy array the same dimension as x. By default,
            this is set to ``numpy.round()`` which will round values to the
            nearest integer. If the string "legacy" is provided here, the
            pre 0.2.0 legacy behaviour of rounding values to the nearest integer
            (starting from 1) and avoiding the output of identical positions
            by incrementing is used.
        """
        writer = vcf.VcfWriter(
            self, ploidy=ploidy, contig_id=contig_id,
            individuals=individuals,
            individual_names=individual_names,
            position_transform=position_transform)
        writer.write(output)

    def simplify(
            self, samples=None,
            filter_zero_mutation_sites=None,  # Deprecated alias for filter_sites
            map_nodes=False,
            reduce_to_site_topology=False,
            filter_populations=True, filter_individuals=True, filter_sites=True,
            record_provenance=True, keep_unary=False):
        """
        Returns a simplified tree sequence that retains only the history of
        the nodes given in the list ``samples``. If ``map_nodes`` is true,
        also return a numpy array mapping the node IDs in this tree sequence to
        their node IDs in the simplified tree tree sequence. If a node ``u`` is not
        present in the new tree sequence, the value of this mapping will be
        :data:`tskit.NULL` (-1).

        In the returned tree sequence, the node with ID ``0`` corresponds to
        ``samples[0]``, node ``1`` corresponds to ``samples[1]``, and so on.
        Besides the samples, node IDs in the returned tree sequence are then
        allocated sequentially in time order.

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

        If ``filter_populations``, ``filter_individuals`` or ``filter_sites`` is
        True, any of the corresponding objects that are not referenced elsewhere
        are filtered out. As this is the default behaviour, it is important to
        realise IDs for these objects may change through simplification. By setting
        these parameters to False, however, the corresponding tables can be preserved
        without changes.

        :param list samples: The list of nodes for which to retain information. This
            may be a numpy array (or array-like) object (dtype=np.int32).
        :param bool filter_zero_mutation_sites: Deprecated alias for ``filter_sites``.
        :param bool map_nodes: If True, return a tuple containing the resulting
            tree sequence and a numpy array mapping node IDs in the current tree
            sequence to their corresponding node IDs in the returned tree sequence.
            If False (the default), return only the tree sequence object itself.
        :param bool reduce_to_site_topology: Whether to reduce the topology down
            to the trees that are present at sites. (Default: False)
        :param bool filter_populations: If True, remove any populations that are
            not referenced by nodes after simplification; new population IDs are
            allocated sequentially from zero. If False, the population table will
            not be altered in any way. (Default: True)
        :param bool filter_individuals: If True, remove any individuals that are
            not referenced by nodes after simplification; new individual IDs are
            allocated sequentially from zero. If False, the individual table will
            not be altered in any way. (Default: True)
        :param bool filter_sites: If True, remove any sites that are
            not referenced by mutations after simplification; new site IDs are
            allocated sequentially from zero. If False, the site table will not
            be altered in any way. (Default: True)
        :param bool record_provenance: If True, record details of this call to
            simplify in the returned tree sequence's provenance information
            (Default: True).
        :param bool keep_unary: If True, any unary nodes (i.e. nodes with exactly
            one child) that exist on the path from samples to root will be preserved
            in the output. (Default: False)
        :return: The simplified tree sequence, or (if ``map_nodes`` is True)
            a tuple consisting of the simplified tree sequence and a numpy array
            mapping source node IDs to their corresponding IDs in the new tree
            sequence.
        :rtype: .TreeSequence or (.TreeSequence, numpy.ndarray)
        """
        tables = self.dump_tables()
        if samples is None:
            samples = self.get_samples()
        assert tables.sequence_length == self.sequence_length
        node_map = tables.simplify(
            samples=samples,
            filter_zero_mutation_sites=filter_zero_mutation_sites,
            reduce_to_site_topology=reduce_to_site_topology,
            filter_populations=filter_populations,
            filter_individuals=filter_individuals,
            filter_sites=filter_sites,
            keep_unary=keep_unary)
        if record_provenance:
            # TODO move this into tables.simplify. See issue #374
            # TODO also make sure we convert all the arguments so that they are
            # definitely JSON encodable.
            parameters = {
                "command": "simplify",
                "TODO": "add simplify parameters"
            }
            tables.provenances.add_row(record=json.dumps(
                provenance.get_provenance_dict(parameters)))
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
        them are discarded.

        Note that node IDs may change as a result of this operation,
        as by default :meth:`.simplify` is called on the returned tree sequence
        to remove redundant nodes. If you wish to map node IDs onto the same
        nodes before and after this method has been called, specify ``simplify=False``.

        See also :meth:`.keep_intervals`.

        :param array_like intervals: A list (start, end) pairs describing the
            genomic intervals to delete. Intervals must be non-overlapping and
            in increasing order. The list of intervals must be interpretable as a
            2D numpy array with shape (N, 2), where N is the number of intervals.
        :param bool simplify: If True, return a simplified tree sequence where nodes
            no longer used are discarded. (Default: True).
        :param bool record_provenance: If ``True``, add details of this operation to the
            provenance information of the returned tree sequence. (Default: ``True``).
        :rtype: .TreeSequence
        """
        tables = self.dump_tables()
        tables.delete_intervals(intervals, simplify, record_provenance)
        return tables.tree_sequence()

    def keep_intervals(self, intervals, simplify=True, record_provenance=True):
        """
        Returns a copy of this tree sequence which includes only information in
        the specified list of genomic intervals. Edges are truncated to lie within
        these intervals, and sites and mutations falling outside these intervals
        are discarded.

        Note that node IDs may change as a result of this operation,
        as by default :meth:`.simplify` is called on the returned tree sequence
        to remove redundant nodes. If you wish to map node IDs onto the same
        nodes before and after this method has been called, specify ``simplify=False``.

        See also :meth:`.delete_intervals`.

        :param array_like intervals: A list (start, end) pairs describing the
            genomic intervals to keep. Intervals must be non-overlapping and
            in increasing order. The list of intervals must be interpretable as a
            2D numpy array with shape (N, 2), where N is the number of intervals.
        :param bool simplify: If True, return a simplified tree sequence where nodes
            no longer used are discarded. (Default: True).
        :param bool record_provenance: If True, add details of this operation to the
            provenance information of the returned tree sequence.
            (Default: True).
        :rtype: .TreeSequence
        """
        tables = self.dump_tables()
        tables.keep_intervals(intervals, simplify, record_provenance)
        return tables.tree_sequence()

    def ltrim(self, record_provenance=True):
        """
        Returns a copy of this tree sequence with a potentially changed coordinate
        system, such that empty regions (i.e. those not covered by any edge) at the start
        of the tree sequence are trimmed away, and the leftmost edge starts at position
        0. This affects the reported position of sites and edges. Additionally, sites and
        their associated mutations to the left of the new zero point are thrown away.

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
        Returns a copy of this tree sequence with any empty regions (i.e. those not
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

    def draw_svg(self, path=None, **kwargs):
        # TODO document this method, including semantic details of the
        # returned SVG object.
        draw = drawing.SvgTreeSequence(self, **kwargs)
        output = draw.drawing.tostring()
        if path is not None:
            # TODO remove the 'pretty' when we are done debugging this.
            draw.drawing.saveas(path, pretty=True)
        return output

    def draw_text(self, **kwargs):
        # TODO document this method.
        return str(drawing.TextTreeSequence(self, **kwargs))

    ############################################
    #
    # Statistics computation
    #
    ############################################

    def general_stat(self, W, f, output_dim, windows=None, polarised=False, mode=None,
                     span_normalise=True, strict=True):
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
                with np.errstate(invalid='ignore', divide='ignore'):
                    fx = np.array(f(x))
                fx[np.isnan(fx)] = 0.0
                if not np.allclose(fx, np.zeros((output_dim, ))):
                    raise ValueError("Summary function does not return zero for both "
                                     "zero weight and total weight.")
        return self.__run_windowed_stat(
            windows, self.ll_tree_sequence.general_stat,
            W, f, output_dim, polarised=polarised,
            span_normalise=span_normalise, mode=mode)

    def sample_count_stat(
            self, sample_sets, f, output_dim, windows=None, polarised=False, mode=None,
            span_normalise=True, strict=True):
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

            ts.sample_count_stat([A, B], f, windows="site",
                                 polarised=False, mode="site")

        would compute, for each site, the product of the derived allele
        frequencies in the two sample sets, in a (num sites, 1) array.  If
        instead ``f`` returns ``np.array([pA, pB, pA * pB])``, then the
        output would be a (num sites, 3) array, with the first two columns
        giving the allele frequencies in ``A`` and ``B``, respectively.

        .. note::
            The summary function ``f`` should return zero when given both 0 and
            the sample size (i.e., ``f(0) = 0`` and
            ``f(np.array([len(x) for x in sample_sets]) = 0``).  This is
            necessary for the statistic to be unaffected by parts of the tree
            sequence ancestral to none or all of the samples, respectively.

        :param list sample_sets: A list of lists of Node IDs, specifying the
            groups of individuals to compute the statistic with.
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
        """
        # helper function for common case where weights are indicators of sample sets
        for U in sample_sets:
            if len(U) != len(set(U)):
                raise ValueError(
                    "Elements of sample_sets must be lists without repeated elements.")
            if len(U) == 0:
                raise ValueError("Elements of sample_sets cannot be empty.")
            for u in U:
                if not self.node(u).is_sample():
                    raise ValueError("Not all elements of sample_sets are samples.")

        W = np.array([[float(u in A) for A in sample_sets] for u in self.samples()])
        return self.general_stat(W, f, output_dim, windows=windows, polarised=polarised,
                                 mode=mode, span_normalise=span_normalise,
                                 strict=strict)

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
                windows = np.concatenate([
                                [] if self.num_sites > 0 else [0.0],
                                self.tables.sites.position,
                                [self.sequence_length]])
                windows[0] = 0.0
            else:
                raise ValueError("Unrecognized window specification {}:".format(windows),
                                 "the only allowed strings are 'sites' or 'trees'")
        return np.array(windows)

    def __run_windowed_stat(self, windows, method, *args, **kwargs):
        strip_dim = windows is None
        windows = self.parse_windows(windows)
        stat = method(*args, **kwargs, windows=windows)
        if strip_dim:
            stat = stat[0]
        return stat

    def __one_way_sample_set_stat(
            self, ll_method, sample_sets, windows=None, mode=None, span_normalise=True,
            polarised=False):
        if sample_sets is None:
            sample_sets = self.samples()

        # First try to convert to a 1D numpy array. If it is, then we strip off
        # the corresponding dimension from the output.
        drop_dimension = False
        try:
            sample_sets = np.array(sample_sets, dtype=np.int32)
        except ValueError:
            pass
        else:
            # If we've successfully converted sample_sets to a 1D numpy array
            # of integers then drop the dimension
            if len(sample_sets.shape) == 1:
                sample_sets = [sample_sets]
                drop_dimension = True

        sample_set_sizes = np.array(
            [len(sample_set) for sample_set in sample_sets], dtype=np.uint32)
        if np.any(sample_set_sizes == 0):
            raise ValueError("Sample sets must contain at least one element")

        flattened = util.safe_np_int_cast(np.hstack(sample_sets), np.int32)
        stat = self.__run_windowed_stat(
            windows, ll_method, sample_set_sizes, flattened,
            mode=mode, span_normalise=span_normalise, polarised=polarised)
        if drop_dimension:
            stat = stat.reshape(stat.shape[:-1])
        return stat

    def __k_way_sample_set_stat(
            self, ll_method, k, sample_sets, indexes=None, windows=None,
            mode=None, span_normalise=True):
        sample_set_sizes = np.array(
            [len(sample_set) for sample_set in sample_sets], dtype=np.uint32)
        if np.any(sample_set_sizes == 0):
            raise ValueError("Sample sets must contain at least one element")
        flattened = util.safe_np_int_cast(np.hstack(sample_sets), np.int32)
        if indexes is None:
            if len(sample_sets) != k:
                raise ValueError(
                    "Must specify indexes if there are not exactly {} sample "
                    "sets.".format(k))
            indexes = np.arange(k, dtype=np.int32)
        drop_dimension = False
        indexes = util.safe_np_int_cast(indexes, np.int32)
        if len(indexes.shape) == 1:
            indexes = indexes.reshape((1, indexes.shape[0]))
            drop_dimension = True
        if len(indexes.shape) != 2 or indexes.shape[1] != k:
            raise ValueError(
                "Indexes must be convertable to a 2D numpy array with {} "
                "columns".format(k))
        stat = self.__run_windowed_stat(
            windows, ll_method, sample_set_sizes, flattened, indexes,
            mode=mode, span_normalise=span_normalise)
        if drop_dimension:
            stat = stat.reshape(stat.shape[:-1])
        return stat

    ############################################
    # Statistics definitions
    ############################################

    def diversity(self, sample_sets=None, windows=None, mode="site",
                  span_normalise=True):
        """
        Computes mean genetic diversity (also knowns as "Tajima's pi") in each of the
        sets of nodes from ``sample_sets``.
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
            Mean pairwise genetic diversity: the average across distinct,
            randomly chosen pairs of chromosomes, of the density of sites at
            which the two carry different alleles, per unit of chromosome length.

        "branch"
            Mean distance in the tree: the average across distinct, randomly chosen pairs
            of chromsomes and locations in the window, of the mean distance in the tree
            between the two samples (in units of time).

        "node"
            For each node, the proportion of genome on which the node is an ancestor to
            only one of a random pair from the sample set, averaged over choices of pair.

        :param list sample_sets: A list of lists of Node IDs, specifying the
            groups of individuals to compute the statistic with.
        :param list windows: An increasing list of breakpoints between the windows
            to compute the statistic in.
        :param str mode: A string giving the "type" of the statistic to be computed
            (defaults to "site").
        :param bool span_normalise: Whether to divide the result by the span of the
            window (defaults to True).
        :return: A numpy array.
        """
        return self.__one_way_sample_set_stat(
            self._ll_tree_sequence.diversity, sample_sets, windows=windows,
            mode=mode, span_normalise=span_normalise)

    def divergence(self, sample_sets, indexes=None, windows=None, mode="site",
                   span_normalise=True):
        """
        Computes mean genetic divergence between (and within) pairs of
        sets of nodes from ``sample_sets``.
        Operates on ``k = 2`` sample sets at a time; please see the
        :ref:`multi-way statistics <sec_stats_sample_sets_multi_way>`
        section for details on how the ``sample_sets`` and ``indexes`` arguments are
        interpreted and how they interact with the dimensions of the output array.
        See the :ref:`statistics interface <sec_stats_interface>` section for details on
        :ref:`windows <sec_stats_windows>`,
        :ref:`mode <sec_stats_mode>`,
        :ref:`span normalise <sec_stats_span_normalise>`,
        and :ref:`return value <sec_stats_output_format>`.

        As a special case, an index ``(j, j)`` will compute the
        :meth:`diversity <.TreeSequence.diversity>` of ``sample_set[j]``.

        What is computed depends on ``mode``:

        "site"
            Mean pairwise genetic divergence: the average across distinct,
            randomly chosen pairs of chromosomes (one from each sample set), of
            the density of sites at which the two carry different alleles, per
            unit of chromosome length.

        "branch"
            Mean distance in the tree: the average across distinct, randomly
            chosen pairs of chromsomes (one from each sample set) and locations
            in the window, of the mean distance in the tree between the two
            samples (in units of time).

        "node"
            For each node, the proportion of genome on which the node is an ancestor to
            only one of a random pair (one from each sample set), averaged over
            choices of pair.

        :param list sample_sets: A list of lists of Node IDs, specifying the
            groups of individuals to compute the statistic with.
        :param list indexes: A list of 2-tuples, or None.
        :param list windows: An increasing list of breakpoints between the windows
            to compute the statistic in.
        :param str mode: A string giving the "type" of the statistic to be computed
            (defaults to "site").
        :param bool span_normalise: Whether to divide the result by the span of the
            window (defaults to True).
        :return: A ndarray with shape equal to (num windows, num statistics).
        """
        return self.__k_way_sample_set_stat(
            self._ll_tree_sequence.divergence, 2, sample_sets, indexes=indexes,
            windows=windows, mode=mode, span_normalise=span_normalise)

    # JK: commenting this out for now to get the other methods well tested.
    # Issue: https://github.com/tskit-dev/tskit/issues/201
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
        """
        if W.shape[0] != self.num_samples:
            raise ValueError("First trait dimension must be equal to number of samples.")
        return self.__run_windowed_stat(
            windows, self._ll_tree_sequence.trait_covariance, W, mode=mode,
            span_normalise=span_normalise)

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
            and by ;math:`p * (1 - p)`, where :math:`p` is the allele frequency.

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
        """
        if W.shape[0] != self.num_samples:
            raise ValueError("First trait dimension must be equal to number of samples.")
        sds = np.std(W, axis=0)
        if np.any(sds == 0):
            raise ValueError("Weight columns must have positive variance",
                             "to compute correlation.")
        return self.__run_windowed_stat(
            windows, self._ll_tree_sequence.trait_correlation, W, mode=mode,
            span_normalise=span_normalise)

    def trait_regression(self, W, Z=None, windows=None, mode="site",
                         span_normalise=True):
        """
        For each trait w (i.e., each column of W), performs the least-squares
        linear regression :math:`w ~ g + Z`,
        where :math:`g` is inheritance in the tree sequence and the columns of :math:`Z`
        are covariates, and computes the squared coefficient of :math:`g` in this
        regression.
        See the :ref:`statistics interface <sec_stats_interface>` section for details on
        :ref:`windows <sec_stats_windows>`,
        :ref:`mode <sec_stats_mode>`,
        :ref:`span normalise <sec_stats_span_normalise>`,
        and :ref:`return value <sec_stats_output_format>`.
        Operates on all samples in the tree sequence.

        Concretely, if `g` is a binary vector that indicates inheritance from an allele,
        branch, or node and `w` is a column of W, there are :math:`k` columns of
        :math:`Z`, and the :math:`k+2`-vector :math:`b` minimises
        :math:`\\sum_i (w_i - b_0 - b_1 g_i - b_2 z_{2,i} - ... b_{k+2} z_{k+2,i})^2`
        then this returns the number :math:`b_1^2`. If :math:`g` lies in the linear span
        of the columns of :math:`Z`, then :math:`b_1` is set to 0. To perform the
        regression without covariates (only the intercept), set `Z = None`.

        What is computed depends on ``mode``:

        "site"
            Computes the sum of :math:`b_1^2/2` for each allele in the window,
            as above with :math:`g` indicating presence/absence of the allele,
            then divided by the length of the window if ``span_normalise=True``.
            (For biallelic loci, this number is the same for both alleles, and so summing
            over each cancels the factor of two.)

        "branch"
            The squared coefficient `b_1^2`, computed for the split induced by each
            branch (i.e., with :math:`g` indicating inheritance from that branch),
            multiplied by branch length and tree span, summed over all trees
            in the window, and divided by the length of the window if
            ``span_normalise=True``.

        "node"
            For each node, the squared coefficient `b_1^2`, computed for the property of
            inheriting from this node, as in "branch".

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
        """
        if W.shape[0] != self.num_samples:
            raise ValueError("First trait dimension must be equal to number of samples.")
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
            windows, self._ll_tree_sequence.trait_regression,
            W, Z, mode=mode, span_normalise=span_normalise)

    def segregating_sites(self, sample_sets=None, windows=None, mode="site",
                          span_normalise=True):
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
            The sum over sites of the number of alleles found in ``A`` at each site
            minus one, per unit of chromosome length.
            If all sites have at most two alleles, this is the density of sites
            that are polymorphic in ``A``. To get the **number** of segregating minor
            alleles per window, pass ``span_normalise=False``.

        "branch"
            The total length of all branches in the tree subtended by the samples in
            ``A``, averaged across the window.

        "node"
            The proportion of the window on which the node is ancestral to some,
            but not all, of the samples in ``A``.

        :param list sample_sets: A list of lists of Node IDs, specifying the
            groups of individuals to compute the statistic with.
        :param list windows: An increasing list of breakpoints between the windows
            to compute the statistic in.
        :param str mode: A string giving the "type" of the statistic to be computed
            (defaults to "site").
        :param bool span_normalise: Whether to divide the result by the span of the
            window (defaults to True).
        :return: A ndarray with shape equal to (num windows, num statistics).
        """
        return self.__one_way_sample_set_stat(
            self._ll_tree_sequence.segregating_sites, sample_sets, windows=windows,
            mode=mode, span_normalise=span_normalise)

    def allele_frequency_spectrum(
            self, sample_sets=None, windows=None, mode="site", span_normalise=True,
            polarised=False):
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
        for all samples in the tree sequence is returned.

        If more than one sample set is specified, the **joint** allele frequency
        spectrum within windows is returned. For example, if we set
        ``sample_sets = [S0, S1]``, then afs[1, 2] counts the number of sites that
        at singletons within S0 and doubletons in S1. The dimensions of the
        output array will be ``[num_windows] + [1 + len(S) for S in sample_sets]``.

        If ``polarised`` is False (the default) the AFS will be *folded*, so that
        the counts do not depend on knowing which allele is ancestral. If folded,
        the frequency spectrum for a single sample set ``S`` has ``afs[j] = 0`` for
        all ``j > len(S) / 2``, so that alleles at frequency ``j`` and ``len(S) - j``
        both add to the same entry. If there is more than one sample set, the
        returned array is "lower triangular" in a similar way.

        What is computed depends on ``mode``:

        "site"
            The number of sites at a given frequency within the specified sample
            sets for each window, per unit of sequence length. To obtain the total
            number of sites, set ``span_normalise`` to False.

        "branch"
            The total length of branches in the trees subtended by subsets of the
            specified sample sets, per unit of sequence length. To obtain the
            total, set ``span_normalise`` to False.

        "node"
            Not supported for this method (raises a ValueError).

        :param list sample_sets: A list of lists of Node IDs, specifying the
            groups of samples to compute the joint allele frequency
        :param list windows: An increasing list of breakpoints between windows
            along the genome.
        :param str mode: A string giving the "type" of the statistic to be computed
            (defaults to "site").
        :param bool span_normalise: Whether to divide the result by the span of the
            window (defaults to True).
        :return: A (k + 1) dimensional numpy array, where k is the number of sample
            sets specified.
        """
        # TODO should we allow a single sample_set to be specified here as a 1D array?
        # This won't change the output dimensions like the other stats.
        if sample_sets is None:
            sample_sets = [self.samples()]
        return self.__one_way_sample_set_stat(
            self._ll_tree_sequence.allele_frequency_spectrum,
            sample_sets, windows=windows, mode=mode, span_normalise=span_normalise,
            polarised=polarised)

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

            D = (T - S/h) / sqrt(a*S + (b/c)*S*(S-1))
            h = 1 + 1/2 + ... + 1/(n-1)
            g = 1 + 1/2^2 + ... + 1/(n-1)^2
            a = (n+1)/(3*(n-1)*h) - 1/h^2
            b = 2*(n^2 + n + 3)/(9*n*(n-1)) - (n+2)/(h*n) + g/h^2
            c = h^2 + g

        What is computed for diversity and divergence depends on ``mode``;
        see those functions for more details.

        :param list sample_sets: A list of lists of Node IDs, specifying the
            groups of individuals to compute the statistic with.
        :param list indexes: A list of 2-tuples, or None.
        :param list windows: An increasing list of breakpoints between the windows
            to compute the statistic in.
        :param str mode: A string giving the "type" of the statistic to be computed
            (defaults to "site").
        :return: A ndarray with shape equal to (num windows, num statistics).
        """
        # TODO this should be done in C as we'll want to support this method there.
        def tjd_func(sample_set_sizes, flattened, **kwargs):
            n = sample_set_sizes
            T = self.ll_tree_sequence.diversity(n, flattened, **kwargs)
            S = self.ll_tree_sequence.segregating_sites(n, flattened, **kwargs)
            h = np.array([np.sum(1/np.arange(1, nn)) for nn in n])
            g = np.array([np.sum(1/np.arange(1, nn)**2) for nn in n])
            with np.errstate(invalid='ignore', divide='ignore'):
                a = (n + 1) / (3 * (n - 1) * h) - 1 / h**2
                b = 2 * (n**2 + n + 3) / (9 * n * (n - 1)) - (n + 2) / (h * n) + g / h**2
                D = (T - S/h) / np.sqrt(a * S + (b / (h**2 + g)) * S * (S - 1))
            return D

        return self.__one_way_sample_set_stat(
            tjd_func, sample_sets, windows=windows, mode=mode, span_normalise=False)

    def Fst(self, sample_sets, indexes=None, windows=None, mode="site",
            span_normalise=True):
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
            groups of individuals to compute the statistic with.
        :param list indexes: A list of 2-tuples.
        :param list windows: An increasing list of breakpoints between the windows
            to compute the statistic in.
        :param str mode: A string giving the "type" of the statistic to be computed
            (defaults to "site").
        :param bool span_normalise: Whether to divide the result by the span of the
            window (defaults to True).
        :return: A ndarray with shape equal to (num windows, num statistics).
        """
        # TODO this should really be implemented in C (presumably C programmers will want
        # to compute Fst too), but in the mean time implementing using the low-level
        # calls has two advantages: (a) we automatically change dimensions like the other
        # two-way stats and (b) it's a bit more efficient because we're not messing
        # around with indexes and samples sets twice.

        def fst_func(sample_set_sizes, flattened, indexes, **kwargs):
            diversities = self._ll_tree_sequence.diversity(
                sample_set_sizes, flattened, **kwargs)
            divergences = self._ll_tree_sequence.divergence(
                sample_set_sizes, flattened, indexes, **kwargs)

            orig_shape = divergences.shape
            # "node" statistics produce a 3D array
            if len(divergences.shape) == 2:
                divergences.shape = (divergences.shape[0], 1, divergences.shape[1])
                diversities.shape = (diversities.shape[0], 1, diversities.shape[1])

            fst = np.repeat(1.0, np.product(divergences.shape))
            fst.shape = divergences.shape
            for i, (u, v) in enumerate(indexes):
                denom = (
                    diversities[:, :, u] + diversities[:, :, v]
                    + 2 * divergences[:, :, i])
                with np.errstate(divide='ignore', invalid='ignore'):
                    fst[:, :, i] -= 2 * (
                        diversities[:, :, u] + diversities[:, :, v]) / denom
            fst.shape = orig_shape
            return fst

        return self.__k_way_sample_set_stat(
            fst_func, 2, sample_sets, indexes=indexes,
            windows=windows, mode=mode, span_normalise=span_normalise)

    def Y3(self, sample_sets, indexes=None, windows=None, mode="site",
           span_normalise=True):
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

        What is computed depends on ``mode``. Each is an average across
        randomly chosen trios of samples ``(a, b, c)``, one from each sample set:

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
            groups of individuals to compute the statistic with.
        :param list indexes: A list of 3-tuples, or None.
        :param list windows: An increasing list of breakpoints between the windows
            to compute the statistic in.
        :param str mode: A string giving the "type" of the statistic to be computed
            (defaults to "site").
        :param bool span_normalise: Whether to divide the result by the span of the
            window (defaults to True).
        :return: A ndarray with shape equal to (num windows, num statistics).
        """
        return self.__k_way_sample_set_stat(
            self._ll_tree_sequence.Y3, 3, sample_sets, indexes=indexes, windows=windows,
            mode=mode, span_normalise=span_normalise)

    def Y2(self, sample_sets, indexes=None, windows=None, mode="site",
           span_normalise=True):
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
        ``Y3``, except that the average across randomly chosen trios of samples
        ``(a, b1, b2)``, where ``a`` is chosen from the first sample set, and
        ``b1, b2`` are chosen (without replacement) from the second sample set.
        See :meth:`Y3 <.TreeSequence.Y3>` for more details.

        :param list sample_sets: A list of lists of Node IDs, specifying the
            groups of individuals to compute the statistic with.
        :param list indexes: A list of 2-tuples, or None.
        :param list windows: An increasing list of breakpoints between the windows
            to compute the statistic in.
        :param str mode: A string giving the "type" of the statistic to be computed
            (defaults to "site").
        :param bool span_normalise: Whether to divide the result by the span of the
            window (defaults to True).
        :return: A ndarray with shape equal to (num windows, num statistics).
        """
        return self.__k_way_sample_set_stat(
            self._ll_tree_sequence.Y2, 2, sample_sets, indexes=indexes, windows=windows,
            mode=mode, span_normalise=span_normalise)

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
        ``Y3``, except that the average is across a randomly chosen trio of
        samples ``(a1, a2, a3)`` all chosen without replacement from the same
        sample set. See :meth:`Y3 <.TreeSequence.Y3>` for more details.

        :param list sample_sets: A list of lists of Node IDs, specifying the
            groups of individuals to compute the statistic with.
        :param list windows: An increasing list of breakpoints between the windows
            to compute the statistic in.
        :param str mode: A string giving the "type" of the statistic to be computed
            (defaults to "site").
        :param bool span_normalise: Whether to divide the result by the span of the
            window (defaults to True).
        :return: A ndarray with shape equal to (num windows, num statistics).
        """
        return self.__one_way_sample_set_stat(
            self._ll_tree_sequence.Y1, sample_sets, windows=windows,
            mode=mode, span_normalise=span_normalise)

    def f4(self, sample_sets, indexes=None, windows=None, mode="site",
           span_normalise=True):
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

        What is computed depends on ``mode``. Each is an average across
        randomly chosen set of four samples ``(a, b; c, d)``, one from each sample set:

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
            minus the average proportion of the window on which ``a`` anc ``d``
            inherit from that node but ``b`` and ``c`` do not, or vice-versa.

        :param list sample_sets: A list of lists of Node IDs, specifying the
            groups of individuals to compute the statistic with.
        :param list indexes: A list of 4-tuples, or None.
        :param list windows: An increasing list of breakpoints between the windows
            to compute the statistic in.
        :param str mode: A string giving the "type" of the statistic to be computed
            (defaults to "site").
        :param bool span_normalise: Whether to divide the result by the span of the
            window (defaults to True).
        :return: A ndarray with shape equal to (num windows, num statistics).
        """
        return self.__k_way_sample_set_stat(
            self._ll_tree_sequence.f4, 4, sample_sets, indexes=indexes, windows=windows,
            mode=mode, span_normalise=span_normalise)

    def f3(self, sample_sets, indexes=None, windows=None, mode="site",
           span_normalise=True):
        """
        Computes Patterson's f3 statistic between three groups of nodes from
        ``sample_sets``.
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
        :meth:`f4 <.TreeSequence.f4>`, except the average is across randomly
        chosen set of four samples ``(a1, b; a2, c)``, with `a1` and `a2` both
        chosen (without replacement) from the first sample set. See
        :meth:`f4 <.TreeSequence.f4>` for more details.

        :param list sample_sets: A list of lists of Node IDs, specifying the
            groups of individuals to compute the statistic with.
        :param list indexes: A list of 3-tuples, or None.
        :param list windows: An increasing list of breakpoints between the windows
            to compute the statistic in.
        :param str mode: A string giving the "type" of the statistic to be computed
            (defaults to "site").
        :param bool span_normalise: Whether to divide the result by the span of the
            window (defaults to True).
        :return: A ndarray with shape equal to (num windows, num statistics).
        """
        return self.__k_way_sample_set_stat(
            self._ll_tree_sequence.f3, 3, sample_sets, indexes=indexes, windows=windows,
            mode=mode, span_normalise=span_normalise)

    def f2(self, sample_sets, indexes=None, windows=None, mode="site",
           span_normalise=True):
        """
        Computes Patterson's f3 statistic between two groups of nodes from
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
        :meth:`f4 <.TreeSequence.f4>`, except the average is across randomly
        chosen set of four samples ``(a1, b1; a2, b2)``, with `a1` and `a2`
        both chosen (without replacement) from the first sample set and ``b1``
        and ``b2`` chosen randomly without replacement from the second sample
        set. See :meth:`f4 <.TreeSequence.f4>` for more details.

        :param list sample_sets: A list of lists of Node IDs, specifying the
            groups of individuals to compute the statistic with.
        :param list indexes: A list of 2-tuples, or None.
        :param list windows: An increasing list of breakpoints between the windows
            to compute the statistic in.
        :param str mode: A string giving the "type" of the statistic to be computed
            (defaults to "site").
        :param bool span_normalise: Whether to divide the result by the span of the
            window (defaults to True).
        :return: A ndarray with shape equal to (num windows, num statistics).
        """
        return self.__k_way_sample_set_stat(
            self._ll_tree_sequence.f2, 2, sample_sets, indexes=indexes, windows=windows,
            mode=mode, span_normalise=span_normalise)

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
                focal, sample_sets)
        else:
            worker = functools.partial(
                self._ll_tree_sequence.genealogical_nearest_neighbours,
                reference_sets=sample_sets)
            focal = util.safe_np_int_cast(focal, np.int32)
            splits = np.array_split(focal, num_threads)
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
                arrays = pool.map(worker, splits)
            return np.vstack(list(arrays))

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
        that differ between a randomly chosen pair of samples.  If `samples` is
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
        return float(self.diversity(
            [samples], windows=[0, self.sequence_length], span_normalise=False)[0])

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
                e.left, e.right, e.parent, e.children, t[e.parent], pop[e.parent])

    # Unsupported old methods.

    def get_num_records(self):
        raise NotImplementedError(
            "This method is no longer supported. Please use the "
            "TreeSequence.num_edges if possible to work with edges rather "
            "than coalescence records. If not, please use len(list(ts.edgesets())) "
            "which should return the number of coalescence records, as previously "
            "defined. Please open an issue on GitHub if this is "
            "important for your workflow.")

    def diffs(self):
        raise NotImplementedError(
            "This method is no longer supported. Please use the "
            "TreeSequence.edge_diffs() method instead")

    def newick_trees(self, precision=3, breakpoints=None, Ne=1):
        raise NotImplementedError(
            "This method is no longer supported. Please use the Tree.newick"
            " method instead")
