# MIT License
#
# Copyright (c) 2019-2020 Tskit Developers
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
Python implementation of the Li and Stephens algorithms.
"""
import itertools
import unittest

import msprime
import numpy as np
import pytest

import _tskit  # TMP
import tskit
from tests import tsutil


def in_sorted(values, j):
    # Take advantage of the fact that the numpy array is sorted.
    ret = False
    index = np.searchsorted(values, j)
    if index < values.shape[0]:
        ret = values[index] == j
    return ret


def ls_forward_matrix_naive(h, alleles, G, rho, mu):
    """
    Simple matrix based method for LS forward algorithm using Python loops.
    """
    assert rho[0] == 0
    m, n = G.shape
    alleles = check_alleles(alleles, m)
    F = np.zeros((m, n))
    S = np.zeros(m)
    f = np.zeros(n) + 1 / n

    for el in range(0, m):
        for j in range(n):
            # NOTE Careful with the difference between this expression and
            # the Viterbi algorithm below. This depends on the different
            # normalisation approach.
            p_t = f[j] * (1 - rho[el]) + rho[el] / n
            p_e = mu[el]
            if G[el, j] == h[el] or h[el] == tskit.MISSING_DATA:
                p_e = 1 - (len(alleles[el]) - 1) * mu[el]
            f[j] = p_t * p_e
        S[el] = np.sum(f)
        # TODO need to handle the 0 case.
        assert S[el] > 0
        f /= S[el]
        F[el] = f
    return F, S


def ls_viterbi_naive(h, alleles, G, rho, mu):
    """
    Simple matrix based method for LS Viterbi algorithm using Python loops.
    """
    assert rho[0] == 0
    m, n = G.shape
    alleles = check_alleles(alleles, m)
    L = np.ones(n)
    T = [set() for _ in range(m)]
    T_dest = np.zeros(m, dtype=int)

    for el in range(m):
        # The calculation below is undefined otherwise.
        if len(alleles[el]) > 1:
            assert mu[el] <= 1 / (len(alleles[el]) - 1)
        L_next = np.zeros(n)
        for j in range(n):
            # NOTE Careful with the difference between this expression and
            # the Forward algorithm above. This depends on the different
            # normalisation approach.
            p_no_recomb = L[j] * (1 - rho[el] + rho[el] / n)
            p_recomb = rho[el] / n
            if p_no_recomb > p_recomb:
                p_t = p_no_recomb
            else:
                p_t = p_recomb
                T[el].add(j)
            p_e = mu[el]
            if G[el, j] == h[el] or h[el] == tskit.MISSING_DATA:
                p_e = 1 - (len(alleles[el]) - 1) * mu[el]
            L_next[j] = p_t * p_e
        L = L_next
        j = np.argmax(L)
        T_dest[el] = j
        if L[j] == 0:
            assert mu[el] == 0
            raise ValueError(
                "Trying to match non-existent allele with zero mutation rate"
            )
        L /= L[j]

    P = np.zeros(m, dtype=int)
    P[m - 1] = T_dest[m - 1]
    for el in range(m - 1, 0, -1):
        j = P[el]
        if j in T[el]:
            j = T_dest[el - 1]
        P[el - 1] = j
    return P


def ls_viterbi_vectorised(h, alleles, G, rho, mu):
    # We must have a non-zero mutation rate, or we'll end up with
    # division by zero problems.
    # assert np.all(mu > 0)

    m, n = G.shape
    alleles = check_alleles(alleles, m)
    V = np.ones(n)
    T = [None for _ in range(m)]
    max_index = np.zeros(m, dtype=int)

    for site in range(m):
        # Transition
        p_neq = rho[site] / n
        p_t = (1 - rho[site] + rho[site] / n) * V
        recombinations = np.where(p_neq > p_t)[0]
        p_t[recombinations] = p_neq
        T[site] = recombinations
        # Emission
        p_e = np.zeros(n) + mu[site]
        index = G[site] == h[site]
        if h[site] == tskit.MISSING_DATA:
            # Missing data is considered equal to everything
            index[:] = True
        p_e[index] = 1 - (len(alleles[site]) - 1) * mu[site]
        V = p_t * p_e
        # Normalise
        max_index[site] = np.argmax(V)
        # print(site, ":", V)
        if V[max_index[site]] == 0:
            assert mu[site] == 0
            raise ValueError(
                "Trying to match non-existent allele with zero mutation rate"
            )
        V /= V[max_index[site]]

    # Traceback
    P = np.zeros(m, dtype=int)
    site = m - 1
    P[site] = max_index[site]
    while site > 0:
        j = P[site]
        if in_sorted(T[site], j):
            j = max_index[site - 1]
        P[site - 1] = j
        site -= 1
    return P


def check_alleles(alleles, num_sites):
    """
    Checks the specified allele list and returns a list of lists
    of alleles of length num_sites.

    If alleles is a 1D list of strings, assume that this list is used
    for each site and return num_sites copies of this list.

    Otherwise, raise a ValueError if alleles is not a list of length
    num_sites.
    """
    if isinstance(alleles[0], str):
        return [alleles for _ in range(num_sites)]
    if len(alleles) != num_sites:
        raise ValueError("Malformed alleles list")
    return alleles


def ls_forward_matrix(h, alleles, G, rho, mu):
    """
    Simple matrix based method for LS forward algorithm using numpy vectorisation.
    """
    assert rho[0] == 0
    m, n = G.shape
    alleles = check_alleles(alleles, m)
    F = np.zeros((m, n))
    S = np.zeros(m)
    f = np.zeros(n) + 1 / n
    p_e = np.zeros(n)

    for el in range(0, m):
        p_t = f * (1 - rho[el]) + rho[el] / n
        eq = G[el] == h[el]
        if h[el] == tskit.MISSING_DATA:
            # Missing data is equal to everything
            eq[:] = True
        p_e[:] = mu[el]
        p_e[eq] = 1 - (len(alleles[el]) - 1) * mu[el]
        f = p_t * p_e
        S[el] = np.sum(f)
        # TODO need to handle the 0 case.
        assert S[el] > 0
        f /= S[el]
        F[el] = f
    return F, S


def forward_matrix_log_proba(F, S):
    """
    Given the specified forward matrix and scaling factor array, return the
    overall log probability of the input haplotype.
    """
    return np.sum(np.log(S)) - np.log(np.sum(F[-1]))


def ls_forward_matrix_unscaled(h, alleles, G, rho, mu):
    """
    Simple matrix based method for LS forward algorithm.
    """
    assert rho[0] == 0
    m, n = G.shape
    alleles = check_alleles(alleles, m)
    F = np.zeros((m, n))
    f = np.zeros(n) + 1 / n

    for el in range(0, m):
        s = np.sum(f)
        for j in range(n):
            p_t = f[j] * (1 - rho[el]) + s * rho[el] / n
            p_e = mu[el]
            if G[el, j] == h[el] or h[el] == tskit.MISSING_DATA:
                p_e = 1 - (len(alleles[el]) - 1) * mu[el]
            f[j] = p_t * p_e
        F[el] = f
    return F


# TODO change this to use the log_proba function below.
def ls_path_probability(h, path, G, rho, mu):
    """
    Returns the probability of the specified path through the genotypes for the
    specified haplotype.
    """
    # Assuming num_alleles = 2
    assert rho[0] == 0
    m, n = G.shape
    # TODO It's not entirely clear why we're starting with a proba of 1 / n for the
    # model. This was done because it made it easier to compare with an existing
    # HMM implementation. Need to figure this one out when writing up.
    proba = 1 / n
    for site in range(0, m):
        pe = mu[site]
        if h[site] == G[site, path[site]] or h[site] == tskit.MISSING_DATA:
            pe = 1 - mu[site]
        pt = rho[site] / n
        if site == 0 or path[site] == path[site - 1]:
            pt = 1 - rho[site] + rho[site] / n
        proba *= pt * pe
    return proba


def ls_path_log_probability(h, path, alleles, G, rho, mu):
    """
    Returns the log probability of the specified path through the genotypes for the
    specified haplotype.
    """
    assert rho[0] == 0
    m, n = G.shape
    alleles = check_alleles(alleles, m)
    # TODO It's not entirely clear why we're starting with a proba of 1 / n for the
    # model. This was done because it made it easier to compare with an existing
    # HMM implementation. Need to figure this one out when writing up.
    log_proba = np.log(1 / n)
    for site in range(0, m):
        if len(alleles[site]) > 1:
            assert mu[site] <= 1 / (len(alleles[site]) - 1)
        pe = mu[site]
        if h[site] == G[site, path[site]] or h[site] == tskit.MISSING_DATA:
            pe = 1 - (len(alleles[site]) - 1) * mu[site]
        assert 0 <= pe <= 1
        pt = rho[site] / n
        if site == 0 or path[site] == path[site - 1]:
            pt = 1 - rho[site] + rho[site] / n
        assert 0 <= pt <= 1
        log_proba += np.log(pt) + np.log(pe)
    return log_proba


def ls_forward_tree(h, alleles, ts, rho, mu, precision=30, use_lib=True):
    """
    Forward matrix computation based on a tree sequence.
    """
    if use_lib:
        acgt_alleles = tuple(alleles) == tskit.ALLELES_ACGT
        ls_hmm = _tskit.LsHmm(
            ts.ll_tree_sequence,
            recombination_rate=rho,
            mutation_rate=mu,
            precision=precision,
            acgt_alleles=acgt_alleles,
        )
        cm = _tskit.CompressedMatrix(ts.ll_tree_sequence)
        ls_hmm.forward_matrix(h, cm)
        return cm
    else:
        fa = ForwardAlgorithm(ts, rho, mu, alleles, precision=precision)
        return fa.run(h)


def ls_viterbi_tree(h, alleles, ts, rho, mu, precision=30, use_lib=True):
    """
    Viterbi path computation based on a tree sequence.
    """
    if use_lib:
        acgt_alleles = tuple(alleles) == tskit.ALLELES_ACGT
        ls_hmm = _tskit.LsHmm(
            ts.ll_tree_sequence,
            recombination_rate=rho,
            mutation_rate=mu,
            precision=precision,
            acgt_alleles=acgt_alleles,
        )
        vm = _tskit.ViterbiMatrix(ts.ll_tree_sequence)
        ls_hmm.viterbi_matrix(h, vm)
        return vm
    else:
        va = ViterbiAlgorithm(ts, rho, mu, alleles, precision=precision)
        return va.run(h)


class ValueTransition:
    """
    Simple struct holding value transition values.
    """

    def __init__(self, tree_node=-1, value=-1, value_index=-1):
        self.tree_node = tree_node
        self.value = value
        self.value_index = value_index

    def copy(self):
        return ValueTransition(self.tree_node, self.value, self.value_index)

    def __repr__(self):
        return repr(self.__dict__)

    def __str__(self):
        return repr(self)


class LsHmmAlgorithm:
    """
    Abstract superclass of Li and Stephens HMM algorithm.
    """

    def __init__(self, ts, rho, mu, alleles, precision=10):
        self.ts = ts
        self.mu = mu
        self.rho = rho
        self.alleles = check_alleles(alleles, ts.num_sites)
        self.precision = precision
        # The array of ValueTransitions.
        self.T = []
        # indexes in to the T array for each node.
        self.T_index = np.zeros(ts.num_nodes, dtype=int) - 1
        # The number of nodes underneath each element in the T array.
        self.N = np.zeros(ts.num_nodes, dtype=int)
        # Efficiently compute the allelic state at a site
        self.allelic_state = np.zeros(ts.num_nodes, dtype=int) - 1
        # Diffs so we can can update T and T_index between trees.
        self.edge_diffs = self.ts.edge_diffs()
        self.parent = np.zeros(self.ts.num_nodes, dtype=int) - 1
        self.tree = tskit.Tree(self.ts)
        self.output = None

    def check_integrity(self):
        M = [st.tree_node for st in self.T if st.tree_node != -1]
        assert np.all(self.T_index[M] >= 0)
        index = np.ones_like(self.T_index, dtype=bool)
        index[M] = 0
        assert np.all(self.T_index[index] == -1)
        for j, st in enumerate(self.T):
            if st.tree_node != -1:
                assert j == self.T_index[st.tree_node]

    def compress(self):
        tree = self.tree
        T = self.T
        T_index = self.T_index

        values = np.unique(list(st.value if st.tree_node != -1 else 1e200 for st in T))
        for st in T:
            if st.tree_node != -1:
                st.value_index = np.searchsorted(values, st.value)

        child = np.zeros(len(values), dtype=int)
        num_values = len(values)
        value_count = np.zeros(num_values, dtype=int)

        def compute(u, parent_state):
            value_count[:] = 0
            for v in tree.children(u):
                child[:] = optimal_set[v]
                # If the set for a given child is empty, then we know it inherits
                # directly from the parent state and must be a singleton set.
                if np.sum(child) == 0:
                    child[parent_state] = 1
                for j in range(num_values):
                    value_count[j] += child[j]
            max_value_count = np.max(value_count)
            # NOTE: we need to set the set to zero here because we actually
            # visit some nodes more than once during the postorder traversal.
            # This would seem to be wasteful, so we should revisit this when
            # cleaning up the algorithm logic.
            optimal_set[u, :] = 0
            optimal_set[u, value_count == max_value_count] = 1

        optimal_set = np.zeros((tree.tree_sequence.num_nodes, len(values)), dtype=int)
        t_node_time = [
            -1 if st.tree_node == -1 else tree.time(st.tree_node) for st in T
        ]
        order = np.argsort(t_node_time)
        for j in order:
            st = T[j]
            u = st.tree_node
            if u != -1:
                # Compute the value at this node
                state = st.value_index
                if tree.is_internal(u):
                    compute(u, state)
                else:
                    # A[u, state] = 1
                    optimal_set[u, state] = 1
                # Find parent state
                v = tree.parent(u)
                if v != -1:
                    while T_index[v] == -1:
                        v = tree.parent(v)
                    parent_state = T[T_index[v]].value_index
                    v = tree.parent(u)
                    while T_index[v] == -1:
                        compute(v, parent_state)
                        v = tree.parent(v)

        T_old = [st.copy() for st in T]
        T.clear()
        T_parent = []

        old_state = T_old[T_index[tree.root]].value_index
        new_state = np.argmax(optimal_set[tree.root])

        T.append(ValueTransition(tree_node=tree.root, value=values[new_state]))
        T_parent.append(-1)
        stack = [(tree.root, old_state, new_state, 0)]
        while len(stack) > 0:
            u, old_state, new_state, t_parent = stack.pop()
            for v in tree.children(u):
                old_child_state = old_state
                if T_index[v] != -1:
                    old_child_state = T_old[T_index[v]].value_index
                if np.sum(optimal_set[v]) > 0:
                    new_child_state = new_state
                    child_t_parent = t_parent

                    if optimal_set[v, new_state] == 0:
                        new_child_state = np.argmax(optimal_set[v])
                        child_t_parent = len(T)
                        T_parent.append(t_parent)
                        T.append(
                            ValueTransition(tree_node=v, value=values[new_child_state])
                        )
                    stack.append((v, old_child_state, new_child_state, child_t_parent))
                else:
                    if old_child_state != new_state:
                        T_parent.append(t_parent)
                        T.append(
                            ValueTransition(tree_node=v, value=values[old_child_state])
                        )

        for st in T_old:
            if st.tree_node != -1:
                T_index[st.tree_node] = -1
        for j, st in enumerate(T):
            T_index[st.tree_node] = j
            self.N[j] = tree.num_samples(st.tree_node)
        for j in range(len(T)):
            if T_parent[j] != -1:
                self.N[T_parent[j]] -= self.N[j]

    def update_tree(self):
        """
        Update the internal data structures to move on to the next tree.
        """
        parent = self.parent
        T_index = self.T_index
        T = self.T
        _, edges_out, edges_in = next(self.edge_diffs)

        for edge in edges_out:
            u = edge.child
            if T_index[u] == -1:
                # Make sure the subtree we're detaching has an T_index-value at the root.
                while T_index[u] == -1:
                    u = parent[u]
                    assert u != -1
                T_index[edge.child] = len(T)
                T.append(
                    ValueTransition(tree_node=edge.child, value=T[T_index[u]].value)
                )
            parent[edge.child] = -1

        for edge in edges_in:
            parent[edge.child] = edge.parent
            u = edge.parent
            if parent[edge.parent] == -1:
                # Grafting onto a new root.
                if T_index[edge.parent] == -1:
                    T_index[edge.parent] = len(T)
                    T.append(
                        ValueTransition(
                            tree_node=edge.parent, value=T[T_index[edge.child]].value
                        )
                    )
            else:
                # Grafting into an existing subtree.
                while T_index[u] == -1:
                    u = parent[u]
                    assert u != -1
            assert T_index[u] != -1 and T_index[edge.child] != -1
            if T[T_index[u]].value == T[T_index[edge.child]].value:
                st = T[T_index[edge.child]]
                # Mark the lower ValueTransition as unused.
                st.value = -1
                st.tree_node = -1
                T_index[edge.child] = -1

        # We can have values left over still pointing to old roots. Remove
        for root in self.tree.roots:
            if T_index[root] != -1:
                # Use a special marker here to designate the real roots.
                T[T_index[root]].value_index = -2
        for vt in T:
            if vt.tree_node != -1:
                if parent[vt.tree_node] == -1 and vt.value_index != -2:
                    T_index[vt.tree_node] = -1
                    vt.tree_node = -1
                vt.value_index = -1

    def update_probabilities(self, site, haplotype_state):
        tree = self.tree
        T_index = self.T_index
        T = self.T
        alleles = self.alleles[site.id]
        allelic_state = self.allelic_state

        # Set the allelic_state for this site.
        allelic_state[tree.root] = alleles.index(site.ancestral_state)
        for mutation in site.mutations:
            u = mutation.node
            allelic_state[u] = alleles.index(mutation.derived_state)
            if T_index[u] == -1:
                while T_index[u] == tskit.NULL:
                    u = tree.parent(u)
                T_index[mutation.node] = len(T)
                T.append(
                    ValueTransition(tree_node=mutation.node, value=T[T_index[u]].value)
                )

        for st in T:
            u = st.tree_node
            if u != -1:
                # Get the allelic_state at u. TODO we can cache these states to
                # avoid some upward traversals.
                v = u
                while allelic_state[v] == -1:
                    v = tree.parent(v)
                    assert v != -1
                match = (
                    haplotype_state == tskit.MISSING_DATA
                    or haplotype_state == allelic_state[v]
                )
                st.value = self.compute_next_probability(site.id, st.value, match, u)

        # Unset the states
        allelic_state[tree.root] = -1
        for mutation in site.mutations:
            allelic_state[mutation.node] = -1

    def process_site(self, site, haplotype_state):
        # print(site.id, "num_transitions=", len(self.T))
        self.update_probabilities(site, haplotype_state)
        # FIXME We don't want to call compress here.
        # What we really want to do is just call compress after
        # the values have been normalised and rounded. However, we can't
        # compute the normalisation factor in the forwards algorithm without
        # the N counts (number of samples directly below each value transition
        # in T), and these are currently computed during compress. So to make
        # things work for now we call compress before and put up with having
        # a slightly less than optimally compressed output matrix. It might
        # end up that this makes no difference and compressing the
        # pre-rounded values is basically the same thing.
        self.compress()
        s = self.compute_normalisation_factor()
        for st in self.T:
            if st.tree_node != tskit.NULL:
                st.value /= s
                st.value = round(st.value, self.precision)
        # *This* is where we want to compress (and can, for viterbi).
        # self.compress()
        self.output.store_site(site.id, s, [(st.tree_node, st.value) for st in self.T])

    def run(self, h):
        n = self.ts.num_samples
        for u in self.ts.samples():
            self.T_index[u] = len(self.T)
            self.T.append(ValueTransition(tree_node=u, value=1 / n))
        while self.tree.next():
            self.update_tree()
            for site in self.tree.sites():
                self.process_site(site, h[site.id])
        return self.output

    def compute_normalisation_factor(self):
        raise NotImplementedError()

    def compute_next_probability(self, site_id, p_last, is_match, node):
        raise NotImplementedError()


class CompressedMatrix:
    """
    Class representing a num_samples x num_sites matrix compressed by a
    tree sequence. Each site is represented by a set of (node, value)
    pairs, which act as "mutations", i.e., any sample that descends
    from a particular node will inherit that value (unless any other
    values are on the path).
    """

    def __init__(self, ts):
        self.ts = ts
        self.num_sites = ts.num_sites
        self.num_samples = ts.num_samples
        self.value_transitions = [None for _ in range(self.num_sites)]
        self.normalisation_factor = np.zeros(self.num_sites)

    def store_site(self, site, normalisation_factor, value_transitions):
        self.normalisation_factor[site] = normalisation_factor
        self.value_transitions[site] = value_transitions

    # Expose the same API as the low-level classes

    @property
    def num_transitions(self):
        a = [len(self.value_transitions[j]) for j in range(self.num_sites)]
        return np.array(a, dtype=np.int32)

    def get_site(self, site):
        return self.value_transitions[site]

    def decode(self):
        """
        Decodes the tree encoding of the values into an explicit
        matrix.
        """
        A = np.zeros((self.num_sites, self.num_samples))
        for tree in self.ts.trees():
            for site in tree.sites():
                f = dict(self.value_transitions[site.id])
                for j, u in enumerate(self.ts.samples()):
                    while u not in f:
                        u = tree.parent(u)
                    A[site.id, j] = f[u]
        return A


class ForwardMatrix(CompressedMatrix):
    """
    Class representing a compressed forward matrix.
    """


class ForwardAlgorithm(LsHmmAlgorithm):
    """
    Runs the Li and Stephens forward algorithm.
    """

    def __init__(self, ts, rho, mu, alleles, precision=10):
        super().__init__(ts, rho, mu, alleles, precision)
        self.output = ForwardMatrix(ts)

    def compute_normalisation_factor(self):
        s = 0
        for j, st in enumerate(self.T):
            assert st.tree_node != tskit.NULL
            assert self.N[j] > 0
            s += self.N[j] * st.value
        return s

    def compute_next_probability(self, site_id, p_last, is_match, node):
        rho = self.rho[site_id]
        mu = self.mu[site_id]
        alleles = self.alleles[site_id]
        n = self.ts.num_samples

        p_t = p_last * (1 - rho) + rho / n
        p_e = mu
        if is_match:
            p_e = 1 - (len(alleles) - 1) * mu
        return p_t * p_e


class ViterbiMatrix(CompressedMatrix):
    """
    Class representing the compressed Viterbi matrix.
    """

    def __init__(self, ts):
        super().__init__(ts)
        self.recombination_required = [(-1, 0, False)]

    def add_recombination_required(self, site, node, required):
        self.recombination_required.append((site, node, required))

    def choose_sample(self, site_id, tree):
        max_value = -1
        u = -1
        for node, value in self.value_transitions[site_id]:
            if value > max_value:
                max_value = value
                u = node
        assert u != -1

        transition_nodes = [u for (u, _) in self.value_transitions[site_id]]
        while not tree.is_sample(u):
            for v in tree.children(u):
                if v not in transition_nodes:
                    u = v
                    break
            else:
                raise AssertionError("could not find path")
        return u

    def traceback(self):
        # Run the traceback.
        m = self.ts.num_sites
        match = np.zeros(m, dtype=int)
        recombination_tree = np.zeros(self.ts.num_nodes, dtype=int) - 1
        tree = tskit.Tree(self.ts)
        tree.last()
        current_node = -1

        rr_index = len(self.recombination_required) - 1
        for site in reversed(self.ts.sites()):
            while tree.interval[0] > site.position:
                tree.prev()
            assert tree.interval[0] <= site.position < tree.interval[1]

            # Fill in the recombination tree
            j = rr_index
            while self.recombination_required[j][0] == site.id:
                u, required = self.recombination_required[j][1:]
                recombination_tree[u] = required
                j -= 1

            if current_node == -1:
                current_node = self.choose_sample(site.id, tree)
            match[site.id] = current_node

            # Now traverse up the tree from the current node. The first marked node
            # we meet tells us whether we need to recombine.
            u = current_node
            while u != -1 and recombination_tree[u] == -1:
                u = tree.parent(u)

            assert u != -1
            if recombination_tree[u] == 1:
                # Need to switch at the next site.
                current_node = -1
            # Reset the nodes in the recombination tree.
            j = rr_index
            while self.recombination_required[j][0] == site.id:
                u, required = self.recombination_required[j][1:]
                recombination_tree[u] = -1
                j -= 1
            rr_index = j

        return match


class ViterbiAlgorithm(LsHmmAlgorithm):
    """
    Runs the Li and Stephens Viterbi algorithm.
    """

    def __init__(self, ts, rho, mu, alleles, precision=10):
        super().__init__(ts, rho, mu, alleles, precision)
        self.output = ViterbiMatrix(ts)

    def compute_normalisation_factor(self):
        max_st = ValueTransition(value=-1)
        for st in self.T:
            assert st.tree_node != tskit.NULL
            if st.value > max_st.value:
                max_st = st
        if max_st.value == 0:
            raise ValueError(
                "Trying to match non-existent allele with zero mutation rate"
            )
        return max_st.value

    def compute_next_probability(self, site_id, p_last, is_match, node):
        rho = self.rho[site_id]
        mu = self.mu[site_id]
        alleles = self.alleles[site_id]
        n = self.ts.num_samples

        p_no_recomb = p_last * (1 - rho + rho / n)
        p_recomb = rho / n
        recombination_required = False
        if p_no_recomb > p_recomb:
            p_t = p_no_recomb
        else:
            p_t = p_recomb
            recombination_required = True
        self.output.add_recombination_required(site_id, node, recombination_required)
        p_e = mu
        if is_match:
            p_e = 1 - (len(alleles) - 1) * mu
        return p_t * p_e


################################################################
# Tests
################################################################


class LiStephensBase:
    """
    Superclass of Li and Stephens tests.
    """

    def assertCompressedMatricesEqual(self, cm1, cm2):
        """
        Checks that the specified compressed matrices contain the same data.
        """
        A1 = cm1.decode()
        A2 = cm2.decode()
        assert np.allclose(A1, A2)
        assert A1.shape == A2.shape
        assert cm1.num_sites == cm2.num_sites
        nf1 = cm1.normalisation_factor
        nf2 = cm1.normalisation_factor
        assert np.allclose(nf1, nf2)
        assert nf1.shape == nf2.shape
        # It seems that we can't rely on the number of transitions in the two
        # implementations being equal, which seems odd given that we should
        # be doing things identically. Still, once the decoded matrices are the
        # same then it seems highly likely to be correct.

        # if not np.array_equal(cm1.num_transitions, cm2.num_transitions):
        #     print()
        #     print(cm1.num_transitions)
        #     print(cm2.num_transitions)
        # self.assertTrue(np.array_equal(cm1.num_transitions, cm2.num_transitions))
        # for j in range(cm1.num_sites):
        #     s1 = dict(cm1.get_site(j))
        #     s2 = dict(cm2.get_site(j))
        #     self.assertEqual(set(s1.keys()), set(s2.keys()))
        #     for key in s1.keys():
        #         self.assertAlmostEqual(s1[key], s2[key])

    def example_haplotypes(self, ts, alleles, num_random=10, seed=2):
        rng = np.random.RandomState(seed)
        H = ts.genotype_matrix(alleles=alleles).T
        haplotypes = [H[0], H[-1]]
        for _ in range(num_random):
            # Choose a random path through H
            p = rng.randint(0, ts.num_samples, ts.num_sites)
            h = H[p, np.arange(ts.num_sites)]
            haplotypes.append(h)
        h = H[0].copy()
        h[-1] = tskit.MISSING_DATA
        haplotypes.append(h)
        h = H[0].copy()
        h[ts.num_sites // 2] = tskit.MISSING_DATA
        haplotypes.append(h)
        # All missing is OK tool
        h = H[0].copy()
        h[:] = tskit.MISSING_DATA
        haplotypes.append(h)
        return haplotypes

    def example_parameters(self, ts, alleles, seed=1):
        """
        Returns an iterator over combinations of haplotype, recombination and mutation
        rates.
        """
        rng = np.random.RandomState(seed)
        haplotypes = self.example_haplotypes(ts, alleles, seed=seed)

        # This is the exact matching limit.
        rho = np.zeros(ts.num_sites) + 0.01
        mu = np.zeros(ts.num_sites)
        rho[0] = 0
        for h in haplotypes:
            yield h, rho, mu

        # Here we have equal mutation and recombination
        rho = np.zeros(ts.num_sites) + 0.01
        mu = np.zeros(ts.num_sites) + 0.01
        rho[0] = 0
        for h in haplotypes:
            yield h, rho, mu

        # Mixture of random and extremes
        rhos = [
            np.zeros(ts.num_sites) + 0.999,
            np.zeros(ts.num_sites) + 1e-6,
            rng.uniform(0, 1, ts.num_sites),
        ]
        # mu can't be more than 1 / 3 if we have 4 alleles
        mus = [
            np.zeros(ts.num_sites) + 0.33,
            np.zeros(ts.num_sites) + 1e-6,
            rng.uniform(0, 0.33, ts.num_sites),
        ]
        for h, rho, mu in itertools.product(haplotypes, rhos, mus):
            rho[0] = 0
            yield h, rho, mu

    def assertAllClose(self, A, B):
        assert np.allclose(A, B)

    def test_simple_n_4_no_recombination(self):
        ts = msprime.simulate(4, recombination_rate=0, mutation_rate=0.5, random_seed=1)
        assert ts.num_sites > 3
        self.verify(ts)

    def test_simple_n_3(self):
        ts = msprime.simulate(3, recombination_rate=2, mutation_rate=7, random_seed=2)
        assert ts.num_sites > 5
        self.verify(ts)

    def test_simple_n_7(self):
        ts = msprime.simulate(7, recombination_rate=2, mutation_rate=5, random_seed=2)
        assert ts.num_sites > 5
        self.verify(ts)

    def test_simple_n_8_high_recombination(self):
        ts = msprime.simulate(8, recombination_rate=20, mutation_rate=5, random_seed=2)
        assert ts.num_trees > 15
        assert ts.num_sites > 5
        self.verify(ts)

    def test_simple_n_15(self):
        ts = msprime.simulate(15, recombination_rate=2, mutation_rate=5, random_seed=2)
        assert ts.num_sites > 5
        self.verify(ts)

    def test_jukes_cantor_n_3(self):
        ts = msprime.simulate(3, mutation_rate=2, random_seed=2)
        ts = tsutil.jukes_cantor(ts, num_sites=10, mu=10, seed=4)
        self.verify(ts, tskit.ALLELES_ACGT)

    def test_jukes_cantor_n_8_high_recombination(self):
        ts = msprime.simulate(8, recombination_rate=20, random_seed=2)
        ts = tsutil.jukes_cantor(ts, num_sites=20, mu=5, seed=4)
        self.verify(ts, tskit.ALLELES_ACGT)

    def test_jukes_cantor_n_15(self):
        ts = msprime.simulate(15, mutation_rate=2, random_seed=2)
        ts = tsutil.jukes_cantor(ts, num_sites=10, mu=0.1, seed=10)
        self.verify(ts, tskit.ALLELES_ACGT)

    def test_jukes_cantor_balanced_ternary(self):
        ts = tskit.Tree.generate_balanced(27, arity=3).tree_sequence
        ts = tsutil.jukes_cantor(ts, num_sites=10, mu=0.1, seed=10)
        self.verify(ts, tskit.ALLELES_ACGT)

    @pytest.mark.skip(reason="Not supporting internal samples yet")
    def test_ancestors_n_3(self):
        ts = msprime.simulate(3, recombination_rate=2, mutation_rate=7, random_seed=2)
        assert ts.num_sites > 5
        tables = ts.dump_tables()
        print(tables.nodes)
        tables.nodes.flags = np.ones_like(tables.nodes.flags)
        print(tables.nodes)
        ts = tables.tree_sequence()
        self.verify(ts)


@pytest.mark.slow
class ForwardAlgorithmBase(LiStephensBase):
    """
    Base for forward algorithm tests.
    """


class TestNumpyMatrixMethod(ForwardAlgorithmBase):
    """
    Tests that we compute the same values from the numpy matrix method as
    the naive algorithm.
    """

    def verify(self, ts, alleles=tskit.ALLELES_01):
        G = ts.genotype_matrix(alleles=alleles)
        for h, rho, mu in self.example_parameters(ts, alleles):
            F1, S1 = ls_forward_matrix(h, alleles, G, rho, mu)
            F2, S2 = ls_forward_matrix_naive(h, alleles, G, rho, mu)
            self.assertAllClose(F1, F2)
            self.assertAllClose(S1, S2)


class ViterbiAlgorithmBase(LiStephensBase):
    """
    Base for viterbi algoritm tests.
    """


class TestExactMatchViterbi(ViterbiAlgorithmBase):
    def verify(self, ts, alleles=tskit.ALLELES_01):
        G = ts.genotype_matrix(alleles=alleles)
        H = G.T
        # print(H)
        rho = np.zeros(ts.num_sites) + 0.1
        mu = np.zeros(ts.num_sites)
        rho[0] = 0
        for h in H:
            p1 = ls_viterbi_naive(h, alleles, G, rho, mu)
            p2 = ls_viterbi_vectorised(h, alleles, G, rho, mu)
            cm1 = ls_viterbi_tree(h, alleles, ts, rho, mu, use_lib=True)
            p3 = cm1.traceback()
            cm2 = ls_viterbi_tree(h, alleles, ts, rho, mu, use_lib=False)
            p4 = cm1.traceback()
            self.assertCompressedMatricesEqual(cm1, cm2)

            assert len(np.unique(p1)) == 1
            assert len(np.unique(p2)) == 1
            assert len(np.unique(p3)) == 1
            assert len(np.unique(p4)) == 1
            m1 = H[p1, np.arange(H.shape[1])]
            assert np.array_equal(m1, h)
            m2 = H[p2, np.arange(H.shape[1])]
            assert np.array_equal(m2, h)
            m3 = H[p3, np.arange(H.shape[1])]
            assert np.array_equal(m3, h)
            m4 = H[p3, np.arange(H.shape[1])]
            assert np.array_equal(m4, h)


@pytest.mark.slow
class TestGeneralViterbi(ViterbiAlgorithmBase, unittest.TestCase):
    def verify(self, ts, alleles=tskit.ALLELES_01):
        # np.set_printoptions(linewidth=20000)
        # np.set_printoptions(threshold=20000000)
        G = ts.genotype_matrix(alleles=alleles)
        # m, n = G.shape
        for h, rho, mu in self.example_parameters(ts, alleles):
            # print("h = ", h)
            # print("rho=", rho)
            # print("mu = ", mu)
            p1 = ls_viterbi_vectorised(h, alleles, G, rho, mu)
            p2 = ls_viterbi_naive(h, alleles, G, rho, mu)
            cm1 = ls_viterbi_tree(h, alleles, ts, rho, mu, use_lib=True)
            p3 = cm1.traceback()
            cm2 = ls_viterbi_tree(h, alleles, ts, rho, mu, use_lib=False)
            p4 = cm1.traceback()
            self.assertCompressedMatricesEqual(cm1, cm2)
            # print()
            # m1 = H[p1, np.arange(m)]
            # m2 = H[p2, np.arange(m)]
            # m3 = H[p3, np.arange(m)]
            # count = np.unique(p1).shape[0]
            # print()
            # print("\tp1 = ", p1)
            # print("\tp2 = ", p2)
            # print("\tp3 = ", p3)
            # print("\tm1 = ", m1)
            # print("\tm2 = ", m2)
            # print("\t h = ", h)
            proba1 = ls_path_log_probability(h, p1, alleles, G, rho, mu)
            proba2 = ls_path_log_probability(h, p2, alleles, G, rho, mu)
            proba3 = ls_path_log_probability(h, p3, alleles, G, rho, mu)
            proba4 = ls_path_log_probability(h, p4, alleles, G, rho, mu)
            # print("\t P = ", proba1, proba2)
            self.assertAlmostEqual(proba1, proba2, places=6)
            self.assertAlmostEqual(proba1, proba3, places=6)
            self.assertAlmostEqual(proba1, proba4, places=6)


class TestMissingHaplotypes(LiStephensBase):
    def verify(self, ts, alleles=tskit.ALLELES_01):
        G = ts.genotype_matrix(alleles=alleles)
        H = G.T

        rho = np.zeros(ts.num_sites) + 0.1
        rho[0] = 0
        mu = np.zeros(ts.num_sites) + 0.001

        # When everything is missing data we should have no recombinations.
        h = H[0].copy()
        h[:] = tskit.MISSING_DATA
        path = ls_viterbi_vectorised(h, alleles, G, rho, mu)
        assert np.all(path == 0)
        cm = ls_viterbi_tree(h, alleles, ts, rho, mu, use_lib=True)
        # For the tree base algorithm it's not simple which particular sample
        # gets chosen.
        path = cm.traceback()
        assert len(set(path)) == 1

        # TODO Not clear what else we can check about missing data.


class TestForwardMatrixScaling(ForwardAlgorithmBase, unittest.TestCase):
    """
    Tests that we get the correct values from scaling version of the matrix
    algorithm works correctly.
    """

    def verify(self, ts, alleles=tskit.ALLELES_01):
        G = ts.genotype_matrix(alleles=alleles)
        computed_log_proba = False
        for h, rho, mu in self.example_parameters(ts, alleles):
            F_unscaled = ls_forward_matrix_unscaled(h, alleles, G, rho, mu)
            F, S = ls_forward_matrix(h, alleles, G, rho, mu)
            column = np.atleast_2d(np.cumprod(S)).T
            F_scaled = F * column
            self.assertAllClose(F_scaled, F_unscaled)
            log_proba1 = forward_matrix_log_proba(F, S)
            psum = np.sum(F_unscaled[-1])
            # If the computed probability is close to zero, there's no point in
            # computing.
            if psum > 1e-20:
                computed_log_proba = True
                log_proba2 = np.log(psum)
                self.assertAlmostEqual(log_proba1, log_proba2)
        assert computed_log_proba


class TestForwardTree(ForwardAlgorithmBase):
    """
    Tests that the tree algorithm computes the same forward matrix as the
    simple method.
    """

    def verify(self, ts, alleles=tskit.ALLELES_01):
        G = ts.genotype_matrix(alleles=alleles)
        for h, rho, mu in self.example_parameters(ts, alleles):
            F, S = ls_forward_matrix(h, alleles, G, rho, mu)
            cm1 = ls_forward_tree(h, alleles, ts, rho, mu, use_lib=True)
            cm2 = ls_forward_tree(h, alleles, ts, rho, mu, use_lib=False)
            self.assertCompressedMatricesEqual(cm1, cm2)
            Ft = cm1.decode()
            self.assertAllClose(S, cm1.normalisation_factor)
            self.assertAllClose(F, Ft)


class TestAllPaths(unittest.TestCase):
    """
    Tests that we compute the correct forward probablities if we sum over all
    possible paths through the genotype matrix.
    """

    def verify(self, G, h):
        m, n = G.shape
        rho = np.zeros(m) + 0.1
        mu = np.zeros(m) + 0.01
        rho[0] = 0
        proba = 0
        for path in itertools.product(range(n), repeat=m):
            proba += ls_path_probability(h, path, G, rho, mu)

        alleles = [["0", "1"] for _ in range(m)]
        F = ls_forward_matrix_unscaled(h, alleles, G, rho, mu)
        forward_proba = np.sum(F[-1])
        self.assertAlmostEqual(proba, forward_proba)

    def test_n3_m4(self):
        G = np.array(
            [
                # fmt: off
                [1, 0, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 1],
                # fmt: on
            ]
        )
        self.verify(G, [0, 0, 0, 0])
        self.verify(G, [1, 1, 1, 1])
        self.verify(G, [1, 1, 0, 0])

    def test_n4_m5(self):
        G = np.array(
            [
                # fmt: off
                [1, 0, 0, 0],
                [0, 0, 1, 1],
                [1, 0, 1, 1],
                [0, 1, 1, 0],
                # fmt: on
            ]
        )
        self.verify(G, [0, 0, 0, 0, 0])
        self.verify(G, [1, 1, 1, 1, 1])
        self.verify(G, [1, 1, 0, 0, 0])

    def test_n5_m5(self):
        G = np.zeros((5, 5), dtype=int)
        np.fill_diagonal(G, 1)
        self.verify(G, [0, 0, 0, 0, 0])
        self.verify(G, [1, 1, 1, 1, 1])
        self.verify(G, [1, 1, 0, 0, 0])


class TestBasicViterbi:
    """
    Very simple tests of the Viterbi algorithm.
    """

    def verify_exact_match(self, G, h, path):
        m, n = G.shape
        rho = np.zeros(m) + 1e-9
        mu = np.zeros(m)  # Set mu to zero exact match
        rho[0] = 0
        alleles = [["0", "1"] for _ in range(m)]
        path1 = ls_viterbi_naive(h, alleles, G, rho, mu)
        path2 = ls_viterbi_vectorised(h, alleles, G, rho, mu)
        assert list(path1) == path
        assert list(path2) == path

    def test_n2_m6_exact(self):
        G = np.array(
            [
                # fmt: off
                [1, 0],
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
                [0, 1],
                # fmt: on
            ]
        )
        self.verify_exact_match(G, [1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1])
        self.verify_exact_match(G, [0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0])
        self.verify_exact_match(G, [0, 0, 0, 1, 1, 1], [1, 1, 1, 1, 1, 1])
        self.verify_exact_match(G, [0, 0, 0, 1, 1, 0], [1, 1, 1, 1, 1, 0])
        self.verify_exact_match(G, [0, 0, 0, 0, 1, 0], [1, 1, 1, 0, 1, 0])

    def test_n3_m6_exact(self):
        G = np.array(
            [
                # fmt: off
                [1, 0, 1],
                [1, 0, 0],
                [1, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [0, 1, 0],
                # fmt: on
            ]
        )
        self.verify_exact_match(G, [1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1])
        self.verify_exact_match(G, [0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0])
        self.verify_exact_match(G, [0, 0, 0, 1, 1, 1], [1, 1, 1, 1, 1, 1])
        self.verify_exact_match(G, [1, 0, 1, 0, 1, 0], [2, 2, 2, 2, 2, 2])

    def test_n3_m6(self):
        G = np.array(
            [
                # fmt: off
                [1, 0, 1],
                [1, 0, 0],
                [1, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [0, 1, 0],
                # fmt: on
            ]
        )

        m, n = G.shape
        rho = np.zeros(m) + 1e-2
        mu = np.zeros(m)
        rho[0] = 0
        alleles = [["0", "1"] for _ in range(m)]
        h = np.ones(m, dtype=int)
        path1 = ls_viterbi_naive(h, alleles, G, rho, mu)

        # Add in mutation at a very low rate.
        mu[:] = 1e-8
        path2 = ls_viterbi_naive(h, alleles, G, rho, mu)
        path3 = ls_viterbi_vectorised(h, alleles, G, rho, mu)
        assert np.array_equal(path1, path2)
        assert np.array_equal(path2, path3)
