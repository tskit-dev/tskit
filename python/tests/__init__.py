# MIT License
#
# Copyright (c) 2018-2023 Tskit Developers
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
import base64

import tskit
from . import tsutil
from .simplify import *  # NOQA

# TODO remove this code and refactor elsewhere.


class PythonTree:
    """
    Presents the same interface as the Tree object for testing. This
    is tightly coupled with the PythonTreeSequence object below which updates
    the internal structures during iteration.
    """

    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.parent = [tskit.NULL for _ in range(num_nodes)]
        self.left_child = [tskit.NULL for _ in range(num_nodes)]
        self.right_child = [tskit.NULL for _ in range(num_nodes)]
        self.left_sib = [tskit.NULL for _ in range(num_nodes)]
        self.right_sib = [tskit.NULL for _ in range(num_nodes)]
        self.num_children = [0 for _ in range(num_nodes)]
        self.edge = [tskit.NULL for _ in range(num_nodes)]
        self.left = 0
        self.right = 0
        self.index = -1
        self.left_root = -1
        # We need a sites function, so this name is taken.
        self.site_list = []

    @classmethod
    def from_tree(cls, tree):
        ret = PythonTree(tree.tree_sequence.num_nodes)
        ret.left, ret.right = tree.get_interval()
        ret.site_list = list(tree.sites())
        ret.index = tree.get_index()
        ret.left_root = tree.left_root
        ret.tree = tree
        for u in range(ret.num_nodes):
            ret.parent[u] = tree.parent(u)
            ret.left_child[u] = tree.left_child(u)
            ret.right_child[u] = tree.right_child(u)
            ret.left_sib[u] = tree.left_sib(u)
            ret.right_sib[u] = tree.right_sib(u)
            ret.num_children[u] = tree.num_children(u)
            ret.edge[u] = tree.edge(u)
        assert ret == tree
        return ret

    @property
    def roots(self):
        u = self.left_root
        roots = []
        while u != tskit.NULL:
            roots.append(u)
            u = self.right_sib[u]
        return roots

    def children(self, u):
        v = self.left_child[u]
        ret = []
        while v != tskit.NULL:
            ret.append(v)
            v = self.right_sib[v]
        return ret

    def get_interval(self):
        return self.left, self.right

    def get_parent(self, node):
        return self.parent[node]

    def get_children(self, node):
        return self.children[node]

    def get_index(self):
        return self.index

    def get_parent_dict(self):
        d = {
            u: self.parent[u]
            for u in range(self.num_nodes)
            if self.parent[u] != tskit.NULL
        }
        return d

    def sites(self):
        return iter(self.site_list)

    def __eq__(self, other):
        return (
            self.get_parent_dict() == other.get_parent_dict()
            and self.get_interval() == other.get_interval()
            and self.roots == other.roots
            and self.get_index() == other.get_index()
            and list(self.sites()) == list(other.sites())
        )

    def __ne__(self, other):
        return not self.__eq__(other)


class PythonTreeSequence:
    """
    A python implementation of the TreeSequence object.

    TODO this class is of limited use now and should be factored out as
    part of a drive towards more modular versions of the tests currently
    in tests_highlevel.py.
    """

    def __init__(self, tree_sequence, breakpoints=None):
        self._tree_sequence = tree_sequence
        self._sites = []
        # TODO this code here is expressed in terms of the low-level
        # tree sequence for legacy reasons. It probably makes more sense
        # to describe it in terms of the tables now if we want to have an
        # independent implementation.
        ll_ts = self._tree_sequence._ll_tree_sequence

        def make_mutation(id_):
            (
                site,
                node,
                derived_state,
                parent,
                metadata,
                time,
                edge,
            ) = ll_ts.get_mutation(id_)
            return tskit.Mutation(
                id=id_,
                site=site,
                node=node,
                time=time,
                derived_state=derived_state,
                parent=parent,
                metadata=metadata,
                edge=edge,
                metadata_decoder=tskit.metadata.parse_metadata_schema(
                    ll_ts.get_table_metadata_schemas().mutation
                ).decode_row,
            )

        for j in range(tree_sequence.num_sites):
            pos, ancestral_state, ll_mutations, id_, metadata = ll_ts.get_site(j)
            self._sites.append(
                tskit.Site(
                    id=id_,
                    position=pos,
                    ancestral_state=ancestral_state,
                    mutations=[make_mutation(ll_mut) for ll_mut in ll_mutations],
                    metadata=metadata,
                    metadata_decoder=tskit.metadata.parse_metadata_schema(
                        ll_ts.get_table_metadata_schemas().site
                    ).decode_row,
                )
            )

    def trees(self):
        pt = PythonTree(self._tree_sequence.get_num_nodes())
        pt.index = 0
        for (left, right), rtt in tsutil.algorithm_R(self._tree_sequence):
            pt.parent[:] = rtt.parent
            pt.left_child[:] = rtt.left_child
            pt.right_child[:] = rtt.right_child
            pt.left_sib[:] = rtt.left_sib
            pt.right_sib[:] = rtt.right_sib
            pt.num_children[:] = rtt.num_children
            pt.edge[:] = rtt.edge
            pt.left_root = rtt.left_child[-1]
            pt.left = left
            pt.right = right
            # Add in all the sites
            pt.site_list = [
                site for site in self._sites if left <= site.position < right
            ]
            yield pt
            pt.index += 1
        pt.index = -1


class MRCACalculator:
    """
    Class to that allows us to compute the nearest common ancestor of arbitrary
    nodes in an oriented forest.

    This is an implementation of Schieber and Vishkin's nearest common ancestor
    algorithm from TAOCP volume 4A, pg.164-167 [K11]_. Preprocesses the
    input tree into a sideways heap in O(n) time and processes queries for the
    nearest common ancestor between an arbitary pair of nodes in O(1) time.

    :param oriented_forest: the input oriented forest
    :type oriented_forest: list of integers
    """

    LAMBDA = 0

    def __init__(self, oriented_forest):
        # We turn this oriened forest into a 1 based array by adding 1
        # to everything
        converted = [0] + [x + 1 for x in oriented_forest]
        self.__preprocess(converted)

    def __preprocess(self, oriented_forest):
        """
        Preprocess the oriented forest, so that we can answer mrca queries
        in constant time.
        """
        n = len(oriented_forest)
        child = [self.LAMBDA for i in range(n)]
        parent = [self.LAMBDA for i in range(n)]
        sib = [self.LAMBDA for i in range(n)]
        self.__lambda = [0 for i in range(n)]
        self.__pi = [0 for i in range(n)]
        self.__tau = [0 for i in range(n)]
        self.__beta = [0 for i in range(n)]
        self.__alpha = [0 for i in range(n)]
        for u in range(n):
            v = oriented_forest[u]
            sib[u] = child[v]
            child[v] = u
            parent[u] = v
        p = child[self.LAMBDA]
        n = 0
        self.__lambda[0] = -1
        while p != self.LAMBDA:
            notDone = True
            while notDone:
                n += 1
                self.__pi[p] = n
                self.__tau[n] = self.LAMBDA
                self.__lambda[n] = 1 + self.__lambda[n >> 1]
                if child[p] != self.LAMBDA:
                    p = child[p]
                else:
                    notDone = False
            self.__beta[p] = n
            notDone = True
            while notDone:
                self.__tau[self.__beta[p]] = parent[p]
                if sib[p] != self.LAMBDA:
                    p = sib[p]
                    notDone = False
                else:
                    p = parent[p]
                    if p != self.LAMBDA:
                        h = self.__lambda[n & -self.__pi[p]]
                        self.__beta[p] = ((n >> h) | 1) << h
                    else:
                        notDone = False
        # Begin the second traversal
        self.__lambda[0] = self.__lambda[n]
        self.__pi[self.LAMBDA] = 0
        self.__beta[self.LAMBDA] = 0
        self.__alpha[self.LAMBDA] = 0
        p = child[self.LAMBDA]
        while p != self.LAMBDA:
            notDone = True
            while notDone:
                a = self.__alpha[parent[p]] | (self.__beta[p] & -self.__beta[p])
                self.__alpha[p] = a
                if child[p] != self.LAMBDA:
                    p = child[p]
                else:
                    notDone = False
            notDone = True
            while notDone:
                if sib[p] != self.LAMBDA:
                    p = sib[p]
                    notDone = False
                else:
                    p = parent[p]
                    notDone = p != self.LAMBDA

    def get_mrca(self, x, y):
        """
        Returns the most recent common ancestor of the nodes x and y,
        or -1 if the nodes belong to different trees.

        :param x: the first node
        :param y: the second node
        :return: the MRCA of nodes x and y
        """
        # WE need to rescale here because SV expects 1-based arrays.
        return self._sv_mrca(x + 1, y + 1) - 1

    def _sv_mrca(self, x, y):
        if self.__beta[x] <= self.__beta[y]:
            h = self.__lambda[self.__beta[y] & -self.__beta[x]]
        else:
            h = self.__lambda[self.__beta[x] & -self.__beta[y]]
        k = self.__alpha[x] & self.__alpha[y] & -(1 << h)
        h = self.__lambda[k & -k]
        j = ((self.__beta[x] >> h) | 1) << h
        if j == self.__beta[x]:
            xhat = x
        else:
            ell = self.__lambda[self.__alpha[x] & ((1 << h) - 1)]
            xhat = self.__tau[((self.__beta[x] >> ell) | 1) << ell]
        if j == self.__beta[y]:
            yhat = y
        else:
            ell = self.__lambda[self.__alpha[y] & ((1 << h) - 1)]
            yhat = self.__tau[((self.__beta[y] >> ell) | 1) << ell]
        if self.__pi[xhat] <= self.__pi[yhat]:
            z = xhat
        else:
            z = yhat
        return z


def base64_encode(metadata):
    """
    Returns the specified metadata bytes object encoded as an ASCII-safe
    string.
    """
    return base64.b64encode(metadata).decode("utf8")


def cached_example(ts_func):
    """
    Utility decorator to cache the result of a single function call
    returning a tree sequence example.
    """
    cache = None

    def f(*args):
        nonlocal cache
        if cache is None:
            cache = ts_func(*args)
        return cache

    return f
