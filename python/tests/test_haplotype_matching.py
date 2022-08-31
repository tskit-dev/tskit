# MIT License
#
# Copyright (c) 2019-2023 Tskit Developers
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
Python implementation of the Li and Stephens forwards and backwards algorithms.
"""
import itertools

import lshmm as ls
import msprime
import numpy as np

import tskit

MISSING = -1


def check_alleles(alleles, m):
    """
    Checks the specified allele list and returns a list of lists
    of alleles of length num_sites.
    If alleles is a 1D list of strings, assume that this list is used
    for each site and return num_sites copies of this list.
    Otherwise, raise a ValueError if alleles is not a list of length
    num_sites.
    """
    if isinstance(alleles[0], str):
        return [alleles for _ in range(m)], np.int8([len(alleles) for _ in range(m)])
    if len(alleles) != m:
        raise ValueError("Malformed alleles list")
    n_alleles = np.int8([(len(alleles_site)) for alleles_site in alleles])
    return alleles, n_alleles


def mirror_coordinates(ts):
    """
    Returns a copy of the specified tree sequence in which all
    coordinates x are transformed into L - x.
    """
    L = ts.sequence_length
    tables = ts.dump_tables()
    left = tables.edges.left
    right = tables.edges.right
    tables.edges.left = L - right
    tables.edges.right = L - left
    tables.sites.position = L - tables.sites.position  # + 1
    # TODO migrations.
    tables.sort()
    return tables.tree_sequence()


class ValueTransition:
    """Simple struct holding value transition values."""

    def __init__(self, tree_node=-1, value=-1, value_index=-1):
        self.tree_node = tree_node
        self.value = value
        self.value_index = value_index

    def copy(self):
        return ValueTransition(
            self.tree_node,
            self.value,
            self.value_index,
        )

    def __repr__(self):
        return repr(self.__dict__)

    def __str__(self):
        return repr(self)


class LsHmmAlgorithm:
    """
    Abstract superclass of Li and Stephens HMM algorithm.
    """

    def __init__(
        self, ts, rho, mu, alleles, n_alleles, precision=10, scale_mutation=False
    ):
        self.ts = ts
        self.mu = mu
        self.rho = rho
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
        # Vector of the number of alleles at each site
        self.n_alleles = n_alleles
        self.alleles = alleles
        self.scale_mutation_based_on_n_alleles = scale_mutation

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
                    haplotype_state == MISSING or haplotype_state == allelic_state[v]
                )
                st.value = self.compute_next_probability(site.id, st.value, match, u)

        # Unset the states
        allelic_state[tree.root] = -1
        for mutation in site.mutations:
            allelic_state[mutation.node] = -1

    def process_site(self, site, haplotype_state, forwards=True):
        if forwards:
            # Forwards algorithm, or forwards pass in Viterbi
            self.update_probabilities(site, haplotype_state)
            self.compress()
            s = self.compute_normalisation_factor()
            for st in self.T:
                if st.tree_node != tskit.NULL:
                    st.value /= s
                    st.value = round(st.value, self.precision)
            self.output.store_site(
                site.id, s, [(st.tree_node, st.value) for st in self.T]
            )
        else:
            # Backwards algorithm
            self.output.store_site(
                site.id,
                self.output.normalisation_factor[site.id],
                [(st.tree_node, st.value) for st in self.T],
            )
            self.update_probabilities(site, haplotype_state)
            self.compress()
            b_last_sum = self.compute_normalisation_factor()
            s = self.output.normalisation_factor[site.id]
            for st in self.T:
                if st.tree_node != tskit.NULL:
                    st.value = (
                        self.rho[site.id] / self.ts.num_samples
                    ) * b_last_sum + (1 - self.rho[site.id]) * st.value
                    st.value /= s
                    st.value = round(st.value, self.precision)

    def run_forward(self, h):
        n = self.ts.num_samples
        self.tree.clear()
        for u in self.ts.samples():
            self.T_index[u] = len(self.T)
            self.T.append(ValueTransition(tree_node=u, value=1 / n))
        while self.tree.next():
            self.update_tree()
            for site in self.tree.sites():
                self.process_site(site, h[site.id])
        return self.output

    def run_backward(self, h):
        self.tree.clear()
        for u in self.ts.samples():
            self.T_index[u] = len(self.T)
            self.T.append(ValueTransition(tree_node=u, value=1))
        while self.tree.next():
            self.update_tree()
            for site in self.tree.sites():
                self.process_site(site, h[site.id], forwards=False)
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

    def __init__(self, ts, normalisation_factor=None):
        self.ts = ts
        self.num_sites = ts.num_sites
        self.num_samples = ts.num_samples
        self.value_transitions = [None for _ in range(self.num_sites)]
        if normalisation_factor is None:
            self.normalisation_factor = np.zeros(self.num_sites)
        else:
            self.normalisation_factor = normalisation_factor
            assert len(self.normalisation_factor) == self.num_sites

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
    """Class representing a compressed forward matrix."""


class BackwardMatrix(CompressedMatrix):
    """Class representing a compressed backward matrix."""


class ViterbiMatrix(CompressedMatrix):
    """
    Class representing the compressed Viterbi matrix.
    """

    def __init__(self, ts):
        super().__init__(ts)
        # Tuple containing the site, the node in the tree, and whether
        # recombination is required
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
            while tree.interval.left > site.position:
                tree.prev()
            assert tree.interval.left <= site.position < tree.interval.right

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


class ForwardAlgorithm(LsHmmAlgorithm):
    """Runs the Li and Stephens forward algorithm."""

    def __init__(
        self, ts, rho, mu, alleles, n_alleles, scale_mutation=False, precision=10
    ):
        super().__init__(
            ts,
            rho,
            mu,
            alleles,
            n_alleles,
            precision=precision,
            scale_mutation=scale_mutation,
        )
        self.output = ForwardMatrix(ts)

    def compute_normalisation_factor(self):
        s = 0
        for j, st in enumerate(self.T):
            assert st.tree_node != tskit.NULL
            assert self.N[j] > 0
            s += self.N[j] * st.value
        return s

    def compute_next_probability(
        self, site_id, p_last, is_match, node
    ):  # Note node only used in Viterbi
        rho = self.rho[site_id]
        mu = self.mu[site_id]
        n = self.ts.num_samples
        n_alleles = self.n_alleles[site_id]

        if self.scale_mutation_based_on_n_alleles:
            if is_match:
                # Scale mutation based on the number of alleles
                # - so the mutation rate is the mutation rate to one of the
                # alleles. The overall mutation rate is then
                # (n_alleles - 1) * mutation_rate.
                p_e = 1 - (n_alleles - 1) * mu
            else:
                p_e = mu - mu * (n_alleles == 1)
                # Added boolean in case we're at an invariant site
        else:
            # No scaling based on the number of alleles
            #  - so the mutation rate is the mutation rate to anything.
            # This means that we must rescale the mutation rate to a different
            # allele, by the number of alleles.
            if n_alleles == 1:  # In case we're at an invariant site
                if is_match:
                    p_e = 1
                else:
                    p_e = 0
            else:
                if is_match:
                    p_e = 1 - mu
                else:
                    p_e = mu / (n_alleles - 1)

        p_t = p_last * (1 - rho) + rho / n
        return p_t * p_e


class BackwardAlgorithm(LsHmmAlgorithm):
    """Runs the Li and Stephens backward algorithm."""

    def __init__(
        self,
        ts,
        rho,
        mu,
        alleles,
        n_alleles,
        normalisation_factor,
        scale_mutation=False,
        precision=10,
    ):
        super().__init__(
            ts,
            rho,
            mu,
            alleles,
            n_alleles,
            precision=precision,
            scale_mutation=scale_mutation,
        )
        self.output = BackwardMatrix(ts, normalisation_factor)

    def compute_normalisation_factor(self):
        s = 0
        for j, st in enumerate(self.T):
            assert st.tree_node != tskit.NULL
            assert self.N[j] > 0
            s += self.N[j] * st.value
        return s

    def compute_next_probability(
        self, site_id, p_next, is_match, node
    ):  # Note node only used in Viterbi
        mu = self.mu[site_id]
        n_alleles = self.n_alleles[site_id]

        if self.scale_mutation_based_on_n_alleles:
            if is_match:
                p_e = 1 - (n_alleles - 1) * mu
            else:
                p_e = mu - mu * (n_alleles == 1)
        else:
            if n_alleles == 1:
                if is_match:
                    p_e = 1
                else:
                    p_e = 0
            else:
                if is_match:
                    p_e = 1 - mu
                else:
                    p_e = mu / (n_alleles - 1)
        return p_next * p_e


class ViterbiAlgorithm(LsHmmAlgorithm):
    """
    Runs the Li and Stephens Viterbi algorithm.
    """

    def __init__(
        self, ts, rho, mu, alleles, n_alleles, scale_mutation=False, precision=10
    ):
        super().__init__(
            ts,
            rho,
            mu,
            alleles,
            n_alleles,
            precision=precision,
            scale_mutation=scale_mutation,
        )
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
        n = self.ts.num_samples
        n_alleles = self.n_alleles[site_id]

        p_no_recomb = p_last * (1 - rho + rho / n)
        p_recomb = rho / n
        recombination_required = False
        if p_no_recomb > p_recomb:
            p_t = p_no_recomb
        else:
            p_t = p_recomb
            recombination_required = True
        self.output.add_recombination_required(site_id, node, recombination_required)

        if self.scale_mutation_based_on_n_alleles:
            if is_match:
                # Scale mutation based on the number of alleles
                # - so the mutation rate is the mutation rate to one of the
                # alleles. The overall mutation rate is then
                # (n_alleles - 1) * mutation_rate.
                p_e = 1 - (n_alleles - 1) * mu
            else:
                p_e = mu - mu * (n_alleles == 1)
                # Added boolean in case we're at an invariant site
        else:
            # No scaling based on the number of alleles
            #  - so the mutation rate is the mutation rate to anything.
            # This means that we must rescale the mutation rate to a different
            # allele, by the number of alleles.
            if n_alleles == 1:  # In case we're at an invariant site
                if is_match:
                    p_e = 1
                else:
                    p_e = 0
            else:
                if is_match:
                    p_e = 1 - mu
                else:
                    p_e = mu / (n_alleles - 1)

        return p_t * p_e


def ls_forward_tree(
    h, ts, rho, mu, precision=30, alleles=None, scale_mutation_based_on_n_alleles=False
):
    if alleles is None:
        n_alleles = np.int8(
            [
                len(np.unique(np.append(ts.genotype_matrix()[j, :], h[j])))
                for j in range(ts.num_sites)
            ]
        )
        alleles = tskit.ALLELES_ACGT
        if len(set(alleles).intersection(next(ts.variants()).alleles)) == 0:
            alleles = tskit.ALLELES_01
            if len(set(alleles).intersection(next(ts.variants()).alleles)) == 0:
                raise ValueError(
                    """Alleles list could not be identified.
                    Please pass a list of lists of alleles of length m,
                    or a list of alleles (e.g. tskit.ALLELES_ACGT)"""
                )
        alleles = [alleles for _ in range(ts.num_sites)]
    else:
        alleles, n_alleles = check_alleles(alleles, ts.num_sites)

    """Forward matrix computation based on a tree sequence."""
    fa = ForwardAlgorithm(
        ts,
        rho,
        mu,
        alleles,
        n_alleles,
        precision=precision,
        scale_mutation=scale_mutation_based_on_n_alleles,
    )
    return fa.run_forward(h)


def ls_backward_tree(
    h, ts_mirror, rho, mu, normalisation_factor, precision=30, alleles=None
):
    if alleles is None:
        n_alleles = np.int8(
            [
                len(np.unique(np.append(ts_mirror.genotype_matrix()[j, :], h[j])))
                for j in range(ts_mirror.num_sites)
            ]
        )
        alleles = tskit.ALLELES_ACGT
        if len(set(alleles).intersection(next(ts_mirror.variants()).alleles)) == 0:
            alleles = tskit.ALLELES_01
            if len(set(alleles).intersection(next(ts_mirror.variants()).alleles)) == 0:
                raise ValueError(
                    """Alleles list could not be identified.
                    Please pass a list of lists of alleles of length m,
                    or a list of alleles (e.g. tskit.ALLELES_ACGT)"""
                )
        alleles = [alleles for _ in range(ts_mirror.num_sites)]
    else:
        alleles, n_alleles = check_alleles(alleles, ts_mirror.num_sites)

    """Backward matrix computation based on a tree sequence."""
    ba = BackwardAlgorithm(
        ts_mirror,
        rho,
        mu,
        alleles,
        n_alleles,
        normalisation_factor,
        precision=precision,
    )
    return ba.run_backward(h)


def ls_viterbi_tree(
    h, ts, rho, mu, precision=30, alleles=None, scale_mutation_based_on_n_alleles=False
):
    if alleles is None:
        n_alleles = np.int8(
            [
                len(np.unique(np.append(ts.genotype_matrix()[j, :], h[j])))
                for j in range(ts.num_sites)
            ]
        )
        alleles = tskit.ALLELES_ACGT
        if len(set(alleles).intersection(next(ts.variants()).alleles)) == 0:
            alleles = tskit.ALLELES_01
            if len(set(alleles).intersection(next(ts.variants()).alleles)) == 0:
                raise ValueError(
                    """Alleles list could not be identified.
                    Please pass a list of lists of alleles of length m,
                    or a list of alleles (e.g. tskit.ALLELES_ACGT)"""
                )
        alleles = [alleles for _ in range(ts.num_sites)]
    else:
        alleles, n_alleles = check_alleles(alleles, ts.num_sites)
    """
    Viterbi path computation based on a tree sequence.
    """
    va = ViterbiAlgorithm(
        ts,
        rho,
        mu,
        alleles,
        n_alleles,
        precision=precision,
        scale_mutation=scale_mutation_based_on_n_alleles,
    )
    return va.run_forward(h)


class LSBase:
    """Superclass of Li and Stephens tests."""

    def example_haplotypes(self, ts):

        H = ts.genotype_matrix()
        s = H[:, 0].reshape(1, H.shape[0])
        H = H[:, 1:]

        haplotypes = [
            s,
            H[:, -1].reshape(1, H.shape[0]),
        ]
        s_tmp = s.copy()
        s_tmp[0, -1] = MISSING
        haplotypes.append(s_tmp)
        s_tmp = s.copy()
        s_tmp[0, ts.num_sites // 2] = MISSING
        haplotypes.append(s_tmp)
        s_tmp = s.copy()
        s_tmp[0, :] = MISSING
        haplotypes.append(s_tmp)

        return H, haplotypes

    def example_parameters_haplotypes(self, ts, seed=42):
        """Returns an iterator over combinations of haplotype,
        recombination and mutation rates."""
        np.random.seed(seed)
        H, haplotypes = self.example_haplotypes(ts)
        n = H.shape[1]
        m = ts.get_num_sites()

        # Here we have equal mutation and recombination
        r = np.zeros(m) + 0.01
        mu = np.zeros(m) + 0.01
        r[0] = 0

        for s in haplotypes:
            yield n, H, s, r, mu

        # Mixture of random and extremes
        rs = [np.zeros(m) + 0.999, np.zeros(m) + 1e-6, np.random.rand(m)]
        mus = [np.zeros(m) + 0.33, np.zeros(m) + 1e-6, np.random.rand(m) * 0.33]

        for s, r, mu in itertools.product(haplotypes, rs, mus):
            r[0] = 0
            yield n, H, s, r, mu

    def assertAllClose(self, A, B):
        """Assert that all entries of two matrices are 'close'"""
        assert np.allclose(A, B, rtol=1e-5, atol=1e-8)

    # Define a bunch of very small tree-sequences for testing a collection
    # of parameters on
    def test_simple_n_10_no_recombination(self):
        ts = msprime.simulate(
            10, recombination_rate=0, mutation_rate=0.5, random_seed=42
        )
        assert ts.num_sites > 3
        self.verify(ts)

    def test_simple_n_10_no_recombination_high_mut(self):
        ts = msprime.simulate(10, recombination_rate=0, mutation_rate=3, random_seed=42)
        assert ts.num_sites > 3
        self.verify(ts)

    def test_simple_n_10_no_recombination_higher_mut(self):
        ts = msprime.simulate(20, recombination_rate=0, mutation_rate=3, random_seed=42)
        assert ts.num_sites > 3
        self.verify(ts)

    def test_simple_n_6(self):
        ts = msprime.simulate(6, recombination_rate=2, mutation_rate=7, random_seed=42)
        assert ts.num_sites > 5
        self.verify(ts)

    def test_simple_n_8(self):
        ts = msprime.simulate(8, recombination_rate=2, mutation_rate=5, random_seed=42)
        assert ts.num_sites > 5
        self.verify(ts)

    def test_simple_n_8_high_recombination(self):
        ts = msprime.simulate(8, recombination_rate=20, mutation_rate=5, random_seed=42)
        assert ts.num_trees > 15
        assert ts.num_sites > 5
        self.verify(ts)

    def test_simple_n_16(self):
        ts = msprime.simulate(16, recombination_rate=2, mutation_rate=5, random_seed=42)
        assert ts.num_sites > 5
        self.verify(ts)

    # # Define a bunch of very small tree-sequences for testing a collection
    # # of parameters on
    # def test_simple_n_10_no_recombination_blah(self):
    #     ts = msprime.sim_ancestry(
    #         samples=10,
    #         recombination_rate=0,
    #         random_seed=42,
    #         sequence_length=10,
    #         population_size=10000,
    #     )
    #     ts = msprime.sim_mutations(ts, rate=1e-5, random_seed=42)
    #     assert ts.num_sites > 3
    #     self.verify(ts)

    # def test_simple_n_6_blah(self):
    # ts = msprime.sim_ancestry(
    #     samples=6,
    #     recombination_rate=1e-4,
    #     random_seed=42,
    #     sequence_length=40,
    #     population_size=10000,
    # )
    # ts = msprime.sim_mutations(ts, rate=1e-3, random_seed=42)
    #     assert ts.num_sites > 5
    #     self.verify(ts)

    # def test_simple_n_8_blah(self):
    #     ts = msprime.sim_ancestry(
    #         samples=8,
    #         recombination_rate=1e-4,
    #         random_seed=42,
    #         sequence_length=20,
    #         population_size=10000,
    #     )
    #     ts = msprime.sim_mutations(ts, rate=1e-4, random_seed=42)
    #     assert ts.num_sites > 5
    #     assert ts.num_trees > 15
    #     self.verify(ts)

    # def test_simple_n_16_blah(self):
    #     ts = msprime.sim_ancestry(
    #         samples=16,
    #         recombination_rate=1e-2,
    #         random_seed=42,
    #         sequence_length=20,
    #         population_size=10000,
    #     )
    #     ts = msprime.sim_mutations(ts, rate=1e-4, random_seed=42)
    #     assert ts.num_sites > 5
    #     self.verify(ts)

    def verify(self, ts):
        raise NotImplementedError()


class FBAlgorithmBase(LSBase):
    """Base for forwards backwards algorithm tests."""


class VitAlgorithmBase(LSBase):
    """Base for viterbi algoritm tests."""


class TestMirroringHap(FBAlgorithmBase):
    """Tests that mirroring the tree sequence and running forwards and backwards
    algorithms gives the same log-likelihood of observing the data."""

    def verify(self, ts):
        for n, H, s, r, mu in self.example_parameters_haplotypes(ts):
            # Note, need to remove the first sample from the ts, and ensure that
            # invariant sites aren't removed.
            ts_check = ts.simplify(range(1, n + 1), filter_sites=False)
            cm = ls_forward_tree(s[0, :], ts_check, r, mu)
            ll_tree = np.sum(np.log10(cm.normalisation_factor))

            ts_check_mirror = mirror_coordinates(ts_check)
            r_flip = np.insert(np.flip(r)[:-1], 0, 0)
            cm_mirror = ls_forward_tree(
                np.flip(s[0, :]), ts_check_mirror, r_flip, np.flip(mu)
            )
            ll_mirror_tree = np.sum(np.log10(cm_mirror.normalisation_factor))
            self.assertAllClose(ll_tree, ll_mirror_tree)

            # Ensure that the decoded matrices are the same
            F_mirror_matrix, c, ll = ls.forwards(
                np.flip(H, axis=0),
                np.flip(s, axis=1),
                r_flip,
                mutation_rate=np.flip(mu),
                scale_mutation_based_on_n_alleles=False,
            )

            self.assertAllClose(F_mirror_matrix, cm_mirror.decode())
            self.assertAllClose(ll, ll_tree)


class TestForwardHapTree(FBAlgorithmBase):
    """Tests that the tree algorithm computes the same forward matrix as the
    simple method."""

    def verify(self, ts):
        for n, H, s, r, mu in self.example_parameters_haplotypes(ts):
            for scale_mutation in [False, True]:
                F, c, ll = ls.forwards(
                    H,
                    s,
                    r,
                    mutation_rate=mu,
                    scale_mutation_based_on_n_alleles=scale_mutation,
                )
                # Note, need to remove the first sample from the ts, and ensure
                # that invariant sites aren't removed.
                ts_check = ts.simplify(range(1, n + 1), filter_sites=False)
                cm = ls_forward_tree(
                    s[0, :],
                    ts_check,
                    r,
                    mu,
                    scale_mutation_based_on_n_alleles=scale_mutation,
                )
                self.assertAllClose(cm.decode(), F)
                ll_tree = np.sum(np.log10(cm.normalisation_factor))
                self.assertAllClose(ll, ll_tree)


class TestForwardBackwardTree(FBAlgorithmBase):
    """Tests that the tree algorithm computes the same forward matrix as the
    simple method."""

    def verify(self, ts):
        for n, H, s, r, mu in self.example_parameters_haplotypes(ts):
            F, c, ll = ls.forwards(
                H, s, r, mutation_rate=mu, scale_mutation_based_on_n_alleles=False
            )
            B = ls.backwards(
                H,
                s,
                c,
                r,
                mutation_rate=mu,
                scale_mutation_based_on_n_alleles=False,
            )

            # Note, need to remove the first sample from the ts, and ensure that
            # invariant sites aren't removed.
            ts_check = ts.simplify(range(1, n + 1), filter_sites=False)
            c_f = ls_forward_tree(s[0, :], ts_check, r, mu)
            ll_tree = np.sum(np.log10(c_f.normalisation_factor))

            ts_check_mirror = mirror_coordinates(ts_check)
            r_flip = np.flip(r)
            c_b = ls_backward_tree(
                np.flip(s[0, :]),
                ts_check_mirror,
                r_flip,
                np.flip(mu),
                np.flip(c_f.normalisation_factor),
            )
            B_tree = np.flip(c_b.decode(), axis=0)
            F_tree = c_f.decode()

            self.assertAllClose(B, B_tree)
            self.assertAllClose(F, F_tree)
            self.assertAllClose(ll, ll_tree)


class TestTreeViterbiHap(VitAlgorithmBase):
    """Test that we have the same log-likelihood between tree and matrix
    implementations"""

    def verify(self, ts):
        for n, H, s, r, mu in self.example_parameters_haplotypes(ts):
            path, ll = ls.viterbi(
                H, s, r, mutation_rate=mu, scale_mutation_based_on_n_alleles=False
            )
            ts_check = ts.simplify(range(1, n + 1), filter_sites=False)
            cm = ls_viterbi_tree(s[0, :], ts_check, r, mu)
            ll_tree = np.sum(np.log10(cm.normalisation_factor))
            self.assertAllClose(ll, ll_tree)

            # Now, need to ensure that the likelihood of the preferred path is
            # the same as ll_tree (and ll).
            path_tree = cm.traceback()
            ll_check = ls.path_ll(
                H,
                s,
                path_tree,
                r,
                mutation_rate=mu,
                scale_mutation_based_on_n_alleles=False,
            )
            self.assertAllClose(ll, ll_check)
