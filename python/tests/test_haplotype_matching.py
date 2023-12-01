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
import io
import warnings

import lshmm as ls
import msprime
import numpy as np
import numpy.testing as nt
import pytest

import _tskit
import tskit
from tests import tsutil

MISSING = -1


# For debugging
np.set_printoptions(linewidth=1000, precision=3)


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
        self,
        ts,
        rho,
        mu,
        alleles,
        n_alleles,
        precision=10,
        scale_mutation=False,
        match_all_nodes=False,
    ):
        self.ts = ts
        self.mu = mu
        self.rho = rho
        self.precision = precision
        # The array of ValueTransitions.
        self.T = []
        # indexes in to the T array for each node.
        self.T_index = np.zeros(ts.num_nodes, dtype=int) - 1
        # Efficiently compute the allelic state at a site
        self.allelic_state = np.zeros(ts.num_nodes, dtype=int) - 1
        # TreePosition so we can can update T and T_index between trees.
        self.tree_pos = tsutil.TreePosition(ts)
        self.parent = np.zeros(self.ts.num_nodes, dtype=int) - 1
        self.tree = tskit.Tree(self.ts)
        self.output = None
        # Vector of the number of alleles at each site
        self.n_alleles = n_alleles
        self.alleles = alleles
        self.scale_mutation_based_on_n_alleles = scale_mutation
        self.match_all_nodes = match_all_nodes

    def node_values(self):
        """
        Return the current mapping of node->value for each node in the
        tree.
        """
        d = {}
        mapping = {st.tree_node: st.value for st in self.T if st.tree_node != -1}
        for u in self.tree.nodes():
            v = u
            while v not in mapping:
                assert v != -1
                v = self.tree.parent(v)
            d[u] = mapping[v]
        return d

    def print_state(self):
        print("LsHMM state")
        print("match_all_nodes =", self.match_all_nodes)
        print("Tree = ", self.tree.index, self.tree.interval)
        node_labels = {}
        for u, value in self.node_values().items():
            label = f"{u}"
            if self.tree.is_sample(u):
                label = f"*{u}*"
            label += f":{value:.2g}"
            node_labels[u] = label
        print(self.tree.draw_text(node_labels=node_labels))
        print("T =")
        for vt in self.T:
            print("\t", vt)
        print("T_index:")
        for u in range(self.ts.num_nodes):
            print(f"\t{u}\t{self.T_index[u]}")

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
        if self.match_all_nodes:
            self._compress_tsinfer()
        else:
            self._compress_parsimony()
        # self.print_state()
        self.check_integrity()

    def _compress_tsinfer(self):
        tree = self.tree
        T = self.T
        T_index = self.T_index

        T_old = [st.copy() for st in T]
        T.clear()

        for st in T_old:
            u = st.tree_node
            if u != -1:
                # We need to find the likelihood of the parent of u. If this is
                # the same as u, we can delete it.
                v = tree.parent(u)
                while v != -1 and T_index[v] == -1:
                    v = tree.parent(v)
                keep = True
                if v != -1:
                    if st.value == T_old[T_index[v]].value:
                        keep = False
                if keep:
                    T.append(st)
                T_index[u] = -1

        # Sort by decreasing time to ensure postorder. This is used by the
        # compressed matrix, downstream
        self.T.sort(key=lambda st: -tree.time(st.tree_node))

        for j, st in enumerate(self.T):
            self.T_index[st.tree_node] = j

    def _compress_parsimony(self):
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
        # Removeing T_parent as it's not needed currently, see note on N[j] below
        # T_parent = []

        old_state = T_old[T_index[tree.root]].value_index
        new_state = np.argmax(optimal_set[tree.root])

        T.append(ValueTransition(tree_node=tree.root, value=values[new_state]))
        # T_parent.append(-1)
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
                        # T_parent.append(t_parent)
                        T.append(
                            ValueTransition(tree_node=v, value=values[new_child_state])
                        )
                    stack.append((v, old_child_state, new_child_state, child_t_parent))
                else:
                    if old_child_state != new_state:
                        # T_parent.append(t_parent)
                        T.append(
                            ValueTransition(tree_node=v, value=values[old_child_state])
                        )

        for st in T_old:
            if st.tree_node != -1:
                T_index[st.tree_node] = -1
        for j, st in enumerate(T):
            T_index[st.tree_node] = j

        # NOTE: we only use the N values in the forward matrix at the moment,
        # so simplifying here by calculating them on the fly where needed.
        # self.N[j] = tree.num_samples(st.tree_node)
        # for j in range(len(T)):
        # if T_parent[j] != -1:
        #     self.N[T_parent[j]] -= self.N[j]

    def update_tree(self, direction=tskit.FORWARD):
        """
        Update the internal data structures to move on to the next tree.
        """
        parent = self.parent
        T_index = self.T_index
        T = self.T
        if direction == tskit.FORWARD:
            self.tree_pos.next()
        else:
            self.tree_pos.prev()
        assert self.tree_pos.index == self.tree.index

        for j in range(
            self.tree_pos.out_range.start, self.tree_pos.out_range.stop, direction
        ):
            e = self.tree_pos.out_range.order[j]
            edge = self.ts.edge(e)
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

        for j in range(
            self.tree_pos.in_range.start, self.tree_pos.in_range.stop, direction
        ):
            e = self.tree_pos.in_range.order[j]
            edge = self.ts.edge(e)
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
                is_match = (
                    haplotype_state == MISSING or haplotype_state == allelic_state[v]
                )
                # Note that the node u is used only by Viterbi
                st.value = self.compute_next_probability(site.id, st.value, is_match, u)

        # Unset the states
        allelic_state[tree.root] = -1
        for mutation in site.mutations:
            allelic_state[mutation.node] = -1

    def process_site(self, site, haplotype_state):
        self.update_probabilities(site, haplotype_state)
        # d1 = self.node_values()
        # print("PRE")
        # self.print_state()
        self.compress()
        # d2 = self.node_values()
        # assert d1 == d2
        # print("AFTER COMPRESS")
        # self.print_state()
        s = self.compute_normalisation_factor()
        for st in self.T:
            assert st.tree_node != tskit.NULL
            # if st.tree_node != tskit.NULL:
            st.value /= s
            st.value = round(st.value, self.precision)
        self.output.store_site(site.id, s, [(st.tree_node, st.value) for st in self.T])

    def compute_emission_proba(self, site_id, is_match):
        mu = self.mu[site_id]
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
        return p_e

    def initialise(self, value):
        self.tree.clear()
        for u in self.ts.samples():
            j = len(self.T)
            self.T_index[u] = j
            self.T.append(ValueTransition(tree_node=u, value=value))

    def run(self, h):
        n = self.ts.num_samples
        self.initialise(1 / n)
        while self.tree.next():
            self.update_tree()
            # if self.tree.index != 0:
            #     print("AFTER UPDATE TREE")
            #     self.print_state()
            for site in self.tree.sites():
                self.process_site(site, h[site.id])
            # print("BEFORE UPDATE TREE")
            # self.print_state()
        return self.output

    def compute_normalisation_factor(self):
        raise NotImplementedError()

    def compute_next_probability(self, site_id, p_last, is_match, node):
        raise NotImplementedError()


class ForwardAlgorithm(LsHmmAlgorithm):
    """
    The Li and Stephens forward algorithm.
    """

    def __init__(self, ts, *args, **kwargs):
        super().__init__(ts, *args, **kwargs)
        self.output = CompressedMatrix(ts)

    def compute_normalisation_factor(self):
        d = {st.tree_node: st for st in self.T}
        N = np.zeros(self.ts.num_nodes, dtype=int)
        node_count = np.zeros(self.ts.num_nodes, dtype=int)
        if self.match_all_nodes:
            # When matching all nodes we need to count the full
            # number of nodes in that subtree
            for u in self.tree.nodes(order="postorder"):
                node_count[u] += 1
                for v in self.tree.children(u):
                    node_count[u] += node_count[v]

        else:
            # When matching on samples we just count the samples. This
            # is a shortcut so we can share the same code below
            for u in d:
                node_count[u] = self.tree.num_samples(u)

        for u in self.tree.nodes(order="preorder"):
            if u in d:
                N[u] = node_count[u]
                # Subtract this value from everything above
                v = self.tree.parent(u)
                while v != -1 and v not in d:
                    v = self.tree.parent(v)
                if v != -1:
                    N[v] -= N[u]
        s = 0
        for st in self.T:
            assert st.tree_node != tskit.NULL
            assert N[st.tree_node] > 0
            s += N[st.tree_node] * st.value
        return s

    def compute_next_probability(self, site_id, p_last, is_match, node):
        rho = self.rho[site_id]
        n = self.ts.num_samples
        p_e = self.compute_emission_proba(site_id, is_match)
        p_t = p_last * (1 - rho) + rho / n
        return p_t * p_e


class BackwardAlgorithm(ForwardAlgorithm):
    """
    The Li and Stephens backward algorithm.
    """

    def compute_next_probability(self, site_id, p_next, is_match, node):
        p_e = self.compute_emission_proba(site_id, is_match)
        return p_next * p_e

    def process_site(self, site, haplotype_state, s):
        # FIXME see nodes in the C code for why we have two calls to
        # compress
        # https://github.com/tskit-dev/tskit/issues/2803
        self.compress()
        self.output.store_site(
            site.id,
            s,
            [(st.tree_node, st.value) for st in self.T],
        )
        self.update_probabilities(site, haplotype_state)
        # FIXME see nodes in the C code for why we have two calls to
        # compress
        self.compress()
        b_last_sum = self.compute_normalisation_factor()
        n = self.ts.num_samples
        rho = self.rho[site.id]
        for st in self.T:
            if st.tree_node != tskit.NULL:
                st.value = rho * b_last_sum / n + (1 - rho) * st.value
                st.value /= s
                st.value = round(st.value, self.precision)

    def run(self, h, normalisation_factor):
        self.initialise(value=1)
        while self.tree.prev():
            self.update_tree(direction=tskit.REVERSE)
            for site in reversed(list(self.tree.sites())):
                self.process_site(site, h[site.id], normalisation_factor[site.id])
        return self.output


class ViterbiAlgorithm(LsHmmAlgorithm):
    """
    Runs the Li and Stephens Viterbi algorithm.
    """

    def __init__(self, ts, *args, **kwargs):
        super().__init__(ts, *args, **kwargs)
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

        p_e = self.compute_emission_proba(site_id, is_match)
        return p_t * p_e


def assert_compressed_matrices_equal(cm1, cm2):
    nt.assert_array_almost_equal(cm1.normalisation_factor, cm2.normalisation_factor)

    for j in range(cm1.num_sites):
        site1 = cm1.get_site(j)
        site2 = cm2.get_site(j)
        assert len(site1) == len(site2)
        site1 = dict(site1)
        site2 = dict(site2)

        assert set(site1.keys()) == set(site2.keys())
        for node in site1.keys():
            # TODO  the precision value should be used as a parameter here
            nt.assert_allclose(site1[node], site2[node], rtol=1e-5, atol=1e-8)


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
        assert all(u >= 0 for u, _ in value_transitions)
        self.normalisation_factor[site] = normalisation_factor
        self.value_transitions[site] = value_transitions

    def print_state(self):
        print("Compressed matrix state")
        for site in range(self.num_sites):
            print(
                site,
                self.normalisation_factor[site],
                self.value_transitions[site],
                sep="\t",
            )

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
        sample_index_map = np.zeros(self.ts.num_nodes, dtype=int) - 1
        sample_index_map[self.ts.samples()] = np.arange(self.ts.num_samples)
        A = np.zeros((self.num_sites, self.num_samples))
        for tree in self.ts.trees():
            for site in tree.sites():
                for node, value in self.value_transitions[site.id]:
                    for u in tree.samples(node):
                        j = sample_index_map[u]
                        A[site.id, j] = value
        return A


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

    def choose_switch_node(self, site_id, tree, match_all_nodes):
        max_value = -1
        u = -1
        for node, value in self.value_transitions[site_id]:
            if value > max_value:
                max_value = value
                u = node
        assert u != -1

        if not match_all_nodes:
            transition_nodes = [u for (u, _) in self.value_transitions[site_id]]
            while not tree.is_sample(u):
                for v in tree.children(u):
                    if v not in transition_nodes:
                        u = v
                        break
                else:
                    raise AssertionError("could not find path")
        return u

    def traceback(self, match_all_nodes=False):
        # Run the traceback.
        m = self.ts.num_sites
        matched = np.zeros(m, dtype=int)
        recombination_tree = np.zeros(self.ts.num_nodes, dtype=int) - 1
        tree = tskit.Tree(self.ts)
        tree.last()
        current_node = -1

        # self.print_state()

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
                current_node = self.choose_switch_node(
                    site.id, tree, match_all_nodes=match_all_nodes
                )
            matched[site.id] = current_node

            # Now traverse up the tree from the current node. The first marked node
            # we meet tells us whether we need to recombine.
            u = current_node
            while u != -1 and recombination_tree[u] == -1:
                u = tree.parent(u)

            assert u != -1
            if recombination_tree[u] == 1:
                # print("recomb_tree = ", recombination_tree)
                # print("SWITCHING AT ", site)
                # Need to switch at the next site.
                current_node = -1
            # Reset the nodes in the recombination tree.
            j = rr_index
            while self.recombination_required[j][0] == site.id:
                u, required = self.recombination_required[j][1:]
                recombination_tree[u] = -1
                j -= 1
            rr_index = j

        # print("MATCHED = ", matched)
        return matched


def get_site_alleles(ts, h, alleles):
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
    return alleles, n_alleles


def ls_forward_tree(
    h,
    ts,
    rho,
    mu,
    precision=30,
    alleles=None,
    scale_mutation_based_on_n_alleles=False,
    match_all_nodes=False,
):
    alleles, n_alleles = get_site_alleles(ts, h, alleles)
    fa = ForwardAlgorithm(
        ts,
        rho,
        mu,
        alleles,
        n_alleles,
        precision=precision,
        scale_mutation=scale_mutation_based_on_n_alleles,
        match_all_nodes=match_all_nodes,
    )
    return fa.run(h)


def ls_backward_tree(
    h,
    ts,
    rho,
    mu,
    normalisation_factor,
    precision=30,
    alleles=None,
    match_all_nodes=False,
):
    alleles, n_alleles = get_site_alleles(ts, h, alleles)
    ba = BackwardAlgorithm(
        ts,
        rho,
        mu,
        alleles,
        n_alleles,
        precision=precision,
        match_all_nodes=match_all_nodes,
    )
    return ba.run(h, normalisation_factor)


def ls_viterbi_tree(
    h,
    ts,
    rho,
    mu,
    precision=30,
    alleles=None,
    scale_mutation_based_on_n_alleles=False,
    match_all_nodes=False,
):
    alleles, n_alleles = get_site_alleles(ts, h, alleles)
    va = ViterbiAlgorithm(
        ts,
        rho,
        mu,
        alleles,
        n_alleles,
        precision=precision,
        scale_mutation=scale_mutation_based_on_n_alleles,
        match_all_nodes=match_all_nodes,
    )
    return va.run(h)


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

        # FIXME removing these as tests are abominably slow.
        # We'll be refactoring all this to use pytest anyway, so let's not
        # worry too much about coverage for now.
        # # Mixture of random and extremes
        # rs = [np.zeros(m) + 0.999, np.zeros(m) + 1e-6, np.random.rand(m)]
        # mus = [np.zeros(m) + 0.33, np.zeros(m) + 1e-6, np.random.rand(m) * 0.33]

        # import itertools
        # for s, r, mu in itertools.product(haplotypes, rs, mus):
        #     r[0] = 0
        #     yield n, H, s, r, mu

    def assertAllClose(self, A, B):
        np.testing.assert_allclose(A, B, rtol=1e-5, atol=1e-8)

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
                p_mutation=np.flip(mu),
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
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Warning from lshmm:
                    # Passed a vector of mutation rates, but rescaling each mutation
                    # rate conditional on the number of alleles
                    F, c, ll = ls.forwards(
                        H,
                        s,
                        r,
                        p_mutation=mu,
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
                H, s, r, p_mutation=mu, scale_mutation_based_on_n_alleles=False
            )
            B = ls.backwards(
                H,
                s,
                c,
                r,
                p_mutation=mu,
                scale_mutation_based_on_n_alleles=False,
            )

            # Note, need to remove the first sample from the ts, and ensure that
            # invariant sites aren't removed.
            ts_check = ts.simplify(range(1, n + 1), filter_sites=False)
            c_f = ls_forward_tree(s[0, :], ts_check, r, mu)
            ll_tree = np.sum(np.log10(c_f.normalisation_factor))

            c_b = ls_backward_tree(
                s[0, :],
                ts_check,
                r,
                mu,
                c_f.normalisation_factor,
            )
            B_tree = c_b.decode()

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
                H, s, r, p_mutation=mu, scale_mutation_based_on_n_alleles=False
            )
            ts_check = ts.simplify(range(1, n + 1), filter_sites=False)
            cm = ls_viterbi_tree(s[0, :], ts_check, r, mu)
            ll_tree = np.sum(np.log10(cm.normalisation_factor))
            self.assertAllClose(ll, ll_tree)

            # Now, need to ensure that the likelihood of the preferred path is
            # the same as ll_tree (and ll).
            path_tree = cm.traceback()
            # print(path)
            # print(path_tree)
            ll_check = ls.path_ll(
                H,
                s,
                path_tree,
                r,
                p_mutation=mu,
                scale_mutation_based_on_n_alleles=False,
            )
            self.assertAllClose(ll, ll_check)


def check_viterbi(
    ts,
    h,
    recombination=None,
    mutation=None,
    match_all_nodes=False,
    compare_fm_ll=False,
    compare_lib=True,
    compare_lshmm=None,
):
    h = np.array(h).astype(np.int8)
    m = ts.num_sites
    assert len(h) == m
    if recombination is None:
        recombination = np.zeros(ts.num_sites) + 1e-9
    if mutation is None:
        mutation = np.zeros(ts.num_sites)
    precision = 22

    if compare_lshmm is None:
        # By default don't compare LSHMM with results from match_all_nodes because
        # it doesn't support missing data in the ref panel.
        if match_all_nodes:
            compare_lshmm = False
        else:
            compare_lshmm = True

    cm = ls_viterbi_tree(
        h, ts, rho=recombination, mu=mutation, match_all_nodes=match_all_nodes
    )
    # cm.print_state()
    path_tree = cm.traceback(match_all_nodes=match_all_nodes)
    ll_tree = np.sum(np.log10(cm.normalisation_factor))
    assert np.isscalar(ll_tree)
    # print("path tree = ", path_tree)

    if compare_fm_ll:
        # Compare the log-likelihood of the Viterbi path (ll_tree)
        # with the log-likelihood of the most likely path from
        # the forward matrix.

        # This is not always true. If the query haplotype is one
        # of the actual sample haplotypes it is *almost* always
        # true, but not quite. So, a useful check for development
        # but not all that useful in general
        fm = ls_forward_tree(
            h,
            ts,
            recombination,
            mutation,
            scale_mutation_based_on_n_alleles=False,
            match_all_nodes=match_all_nodes,
        )
        ll_fm = np.sum(np.log10(fm.normalisation_factor))
        # print()
        # print("vit ll", ll_tree)
        # print("FMLL", ll_fm)
        np.testing.assert_allclose(ll_tree, ll_fm)

    if compare_lshmm:
        # Check that the likelihood of the preferred path is
        # the same as ll_tree (and ll).
        # Missing haplotypes not supported in lshmm yet
        G = ts.genotype_matrix()
        path, ll = ls.viterbi(
            G,
            h.reshape(1, m),
            recombination,
            p_mutation=mutation,
            scale_mutation_based_on_n_alleles=False,
        )
        assert np.isscalar(ll)
        # This is the log likelihood returned by viterbi alg
        nt.assert_allclose(ll_tree, ll)
        # print()
        # print("ls path = ", path)
        ll_check = ls.path_ll(
            G,
            h.reshape(1, m),
            path_tree,
            recombination,
            p_mutation=mutation,
            scale_mutation_based_on_n_alleles=False,
        )
        # This is the log-likelihood of the path itself, computed
        # different way
        nt.assert_allclose(ll_tree, ll_check)

    if compare_lib:
        nt.assert_allclose(ll_check, ll)
        ll_ts = ts._ll_tree_sequence
        ls_hmm = _tskit.LsHmm(ll_ts, recombination, mutation, precision=precision)
        cm_lib = _tskit.ViterbiMatrix(ll_ts)
        ls_hmm.viterbi_matrix(h, cm_lib)
        path_lib = cm_lib.traceback()

        # Not true in general, but let's see how far it goes
        nt.assert_array_equal(path_lib, path_tree)

        nt.assert_allclose(cm_lib.normalisation_factor, cm.normalisation_factor)

    return path_tree


def check_forward_matrix(
    ts,
    h,
    recombination=None,
    mutation=None,
    match_all_nodes=False,
    compare_lib=True,
    compare_lshmm=None,
):
    precision = 22
    h = np.array(h).astype(np.int8)
    n = ts.num_samples
    m = ts.num_sites
    assert len(h) == m
    if recombination is None:
        recombination = np.zeros(ts.num_sites) + 1e-9
    if mutation is None:
        mutation = np.zeros(ts.num_sites)

    if compare_lshmm is None:
        # By default don't compare LSHMM with results from match_all_nodes because
        # it doesn't support missing data in the ref panel.
        if match_all_nodes:
            compare_lshmm = False
        else:
            compare_lshmm = True

    cm = ls_forward_tree(
        h,
        ts,
        recombination,
        mutation,
        scale_mutation_based_on_n_alleles=False,
        match_all_nodes=match_all_nodes,
    )
    F2 = cm.decode()
    ll_tree = np.sum(np.log10(cm.normalisation_factor))

    if compare_lshmm:
        G = ts.genotype_matrix()
        F, c, ll = ls.forwards(
            G,
            h.reshape(1, m),
            recombination,
            p_mutation=mutation,
            scale_mutation_based_on_n_alleles=False,
        )
        assert F.shape == (m, n)
        assert c.shape == (m,)
        assert np.isscalar(ll)

        # print(ll_tree)
        # print("lshmm fm ll:", ll)
        # print(F)
        # print(F2)
        nt.assert_allclose(F, F2)
        nt.assert_allclose(c, cm.normalisation_factor)
        nt.assert_allclose(ll_tree, ll)

    if compare_lib:
        ll_ts = ts._ll_tree_sequence
        ls_hmm = _tskit.LsHmm(ll_ts, recombination, mutation, precision=precision)
        cm_lib = _tskit.CompressedMatrix(ll_ts)
        ls_hmm.forward_matrix(h, cm_lib)
        F3 = cm_lib.decode()

        assert_compressed_matrices_equal(cm, cm_lib)

        nt.assert_allclose(F, F3)
        nt.assert_allclose(c, cm_lib.normalisation_factor)
    return cm


def check_backward_matrix(
    ts,
    h,
    forward_cm,
    recombination=None,
    mutation=None,
    match_all_nodes=False,
    compare_lib=True,
    compare_lshmm=None,
):
    precision = 22
    h = np.array(h).astype(np.int8)
    m = ts.num_sites
    assert len(h) == m
    if recombination is None:
        recombination = np.zeros(ts.num_sites) + 1e-9
    if mutation is None:
        mutation = np.zeros(ts.num_sites)

    if compare_lshmm is None:
        # By default don't compare LSHMM with results from match_all_nodes because
        # it doesn't support missing data in the ref panel.
        if match_all_nodes:
            compare_lshmm = False
        else:
            compare_lshmm = True

    backward_cm = ls_backward_tree(
        h,
        ts,
        recombination,
        mutation,
        forward_cm.normalisation_factor,
        precision=precision,
        match_all_nodes=match_all_nodes,
    )

    if compare_lshmm:
        G = ts.genotype_matrix()
        B = ls.backwards(
            G,
            h.reshape(1, m),
            forward_cm.normalisation_factor,
            recombination,
            p_mutation=mutation,
            scale_mutation_based_on_n_alleles=False,
        )
        nt.assert_array_equal(
            backward_cm.normalisation_factor, forward_cm.normalisation_factor
        )
    if compare_lib:
        ll_ts = ts._ll_tree_sequence
        ls_hmm = _tskit.LsHmm(ll_ts, recombination, mutation, precision=precision)
        cm_lib = _tskit.CompressedMatrix(ll_ts)
        ls_hmm.backward_matrix(h, forward_cm.normalisation_factor, cm_lib)

        assert_compressed_matrices_equal(backward_cm, cm_lib)

        B_lib = cm_lib.decode()
        B_tree = backward_cm.decode()
        nt.assert_allclose(B_tree, B_lib)
        nt.assert_allclose(B, B_lib)

    return backward_cm


def add_unique_node_mutations(ts, start=0, nodes=None):
    """
    Adds a mutation for each of the samples at equally spaced locations
    along the genome.
    """
    if nodes is None:
        nodes = ts.samples()
    tables = ts.dump_tables()
    L = int(ts.sequence_length)
    n = len(nodes)
    assert L % n == 0
    gap = L // n
    x = start
    for u in nodes:
        site = tables.sites.add_row(position=x, ancestral_state="0")
        tables.mutations.add_row(site=site, derived_state="1", node=u)
        x += gap
    return tables.tree_sequence()


class TestSingleBalancedTreeExample:
    # 3.00┊    6    ┊
    #     ┊  ┏━┻━┓  ┊
    # 2.00┊  4   5  ┊
    #     ┊ ┏┻┓ ┏┻┓ ┊
    # 1.00┊ 0 1 2 3 ┊
    #     0         8

    @staticmethod
    def ts():
        return add_unique_node_mutations(
            tskit.Tree.generate_balanced(4, span=8).tree_sequence,
            start=1,
        )

    @pytest.mark.parametrize("j", [0, 1, 2, 3])
    def test_match_sample(self, j):
        ts = self.ts()
        h = np.zeros(4)
        h[j] = 1
        path = check_viterbi(ts, h)
        nt.assert_array_equal([j, j, j, j], path)
        check_fb_matrices(ts, h)

    @pytest.mark.parametrize("j", [1, 2])
    def test_match_sample_missing_flanks(self, j):
        ts = self.ts()
        h = np.zeros(4)
        h[0] = -1
        h[-1] = -1
        h[j] = 1
        path = check_viterbi(ts, h)
        nt.assert_array_equal([j, j, j, j], path)
        check_fb_matrices(ts, h)

    def test_switch_each_sample(self):
        ts = self.ts()
        h = np.ones(4)
        path = check_viterbi(ts, h)
        nt.assert_array_equal([0, 1, 2, 3], path)
        check_fb_matrices(ts, h)

    def test_switch_each_sample_missing_flanks(self):
        ts = self.ts()
        h = np.ones(4)
        h[0] = -1
        h[-1] = -1
        path = check_viterbi(ts, h)
        nt.assert_array_equal([1, 1, 2, 2], path)
        check_fb_matrices(ts, h)

    def test_switch_each_sample_missing_middle(self):
        ts = self.ts()
        h = np.ones(4)
        h[1:3] = -1
        path = check_viterbi(ts, h)
        # Implementation of Viterbi switches at right-most position
        nt.assert_array_equal([0, 0, 0, 3], path)
        check_fb_matrices(ts, h)


class TestSingleBalancedTreeAllSamplesExample:
    # 3.00┊    6    ┊
    #     ┊  ┏━┻━┓  ┊
    # 2.00┊  4   5  ┊
    #     ┊ ┏┻┓ ┏┻┓ ┊
    # 1.00┊ 0 1 2 3 ┊
    #     0         8

    @staticmethod
    def ts():
        tables = tskit.Tree.generate_balanced(4, span=14).tree_sequence.dump_tables()
        flags = tables.nodes.flags
        flags[:] = 1
        tables.nodes.flags = flags
        return add_unique_node_mutations(tables.tree_sequence(), start=1)

    @pytest.mark.parametrize(
        ("u", "h"),
        [
            (0, [1, 0, 0, 0, 1, 0, 1]),
            (1, [0, 1, 0, 0, 1, 0, 1]),
            (2, [0, 0, 1, 0, 0, 1, 1]),
            (3, [0, 0, 0, 1, 0, 1, 1]),
            (4, [0, 0, 0, 0, 1, 0, 1]),
            (5, [0, 0, 0, 0, 0, 1, 1]),
            (6, [0, 0, 0, 0, 0, 0, 1]),
        ],
    )
    def test_match_sample(self, u, h):
        ts = self.ts()
        path = check_viterbi(
            ts, h, match_all_nodes=True, compare_lib=False, compare_lshmm=True
        )
        nt.assert_array_equal([u] * 7, path)
        fm = check_forward_matrix(
            ts, h, match_all_nodes=True, compare_lib=False, compare_lshmm=True
        )
        bm = check_backward_matrix(
            ts, h, fm, match_all_nodes=True, compare_lib=False, compare_lshmm=True
        )
        check_fb_matrix_integrity(fm, bm)


def check_fb_matrix_integrity(fm, bm):
    """
    Validate properties of the forward and backward matrices.
    """
    F = fm.decode()
    B = bm.decode()
    assert F.shape == B.shape
    for j in range(len(F)):
        s = np.sum(B[j] * F[j])
        np.testing.assert_allclose(s, 1)


def check_fb_matrices(ts, h):
    fm = check_forward_matrix(ts, h)
    bm = check_backward_matrix(ts, h, fm)
    check_fb_matrix_integrity(fm, bm)


def validate_match_all_nodes(ts, h, expected_path):
    # path = check_viterbi(
    #     ts, h, match_all_nodes=True, compare_lib=False, compare_lshmm=False
    # )
    # nt.assert_array_equal(expected_path, path)
    fm = check_forward_matrix(
        ts, h, match_all_nodes=True, compare_lib=False, compare_lshmm=False
    )
    F = fm.decode()
    # print(cm.decode())
    # cm.print_state()
    bm = check_backward_matrix(
        ts, h, fm, match_all_nodes=True, compare_lib=False, compare_lshmm=False
    )
    print("sites = ", ts.num_sites)
    B = bm.decode()
    print(F)
    for j in range(ts.num_sites):
        print(j, np.sum(B[j] * F[j]))

    # sum(B[variant,:] * F[variant,:]) = 1


class TestSingleBalancedTreeAllNodesExample:
    # 3.00┊    6    ┊
    #     ┊  ┏━┻━┓  ┊
    # 2.00┊  4   5  ┊
    #     ┊ ┏┻┓ ┏┻┓ ┊
    # 1.00┊ 0 1 2 3 ┊
    #     0         8

    @staticmethod
    def ts():
        tables = tskit.Tree.generate_balanced(4, span=12).tree_sequence.dump_tables()
        return add_unique_node_mutations(
            tables.tree_sequence(), start=1, nodes=np.arange(len(tables.nodes) - 1)
        )

    @pytest.mark.parametrize(
        ("h", "expected_path"),
        [
            # Just samples
            ([1, 0, 0, 0, 1, 0], [0] * 6),
            ([0, 1, 0, 0, 1, 0], [1] * 6),
            ([0, 0, 1, 0, 0, 1], [2] * 6),
            ([0, 0, 0, 1, 0, 1], [3] * 6),
            # Switching between samples
            ([1, 1, 0, 0, 1, 0], [0] + [1] * 5),
            ([1, 1, 1, 0, 0, 1], [0] + [1] + [2] * 4),
            # Just internal
            ([0, 0, 0, 0, 1, 0], [4] * 6),
            ([0, 0, 0, 0, 0, 1], [5] * 6),
            ([0, 0, 0, 0, 0, 0], [6] * 6),
        ],
    )
    def test_exact_match(self, h, expected_path):
        validate_match_all_nodes(self.ts(), h, expected_path)


class TestMultiTreeExample:
    # 0.84┊     7   ┊    7    ┊
    #     ┊   ┏━┻━┓ ┊  ┏━┻━┓  ┊
    # 0.42┊   ┃   ┃ ┊  6   ┃  ┊
    #     ┊   ┃   ┃ ┊ ┏┻┓  ┃  ┊
    # 0.05┊   5   ┃ ┊ ┃ ┃  ┃  ┊
    #     ┊ ┏━┻┓  ┃ ┊ ┃ ┃  ┃  ┊
    # 0.04┊ ┃  4  ┃ ┊ ┃ ┃  4  ┊
    #     ┊ ┃ ┏┻┓ ┃ ┊ ┃ ┃ ┏┻┓ ┊
    # 0.00┊ 0 1 2 3 ┊ 0 3 1 2 ┊
    #     0         6         7
    @staticmethod
    def ts():
        nodes = """\
        is_sample       time
        1       0.000000
        1       0.000000
        1       0.000000
        1       0.000000
        0       0.041304
        0       0.045967
        0       0.416719
        0       0.838075
        """
        edges = """\
        left    right   parent  child
        0.000000        7.000000       4       1
        0.000000        7.000000       4       2
        0.000000        6.000000       5       0
        0.000000        6.000000       5       4
        6.000000        7.000000       6       0
        6.000000        7.000000       6       3
        0.000000        6.000000       7       3
        6.000000        7.000000       7       4
        0.000000        6.000000       7       5
        6.000000        7.000000       7       6
        """
        ts = tskit.load_text(
            nodes=io.StringIO(nodes), edges=io.StringIO(edges), strict=False
        )
        return add_unique_node_mutations(ts, nodes=range(7))

    # 0.84┊     7   ┊    7    ┊
    #     ┊   ┏━┻━┓ ┊  ┏━┻━┓  ┊
    # 0.42┊   ┃   ┃ ┊  6   ┃  ┊
    #     ┊   ┃   ┃ ┊ ┏┻┓  ┃  ┊
    # 0.05┊   5   ┃ ┊ ┃ ┃  ┃  ┊
    #     ┊ ┏━┻┓  ┃ ┊ ┃ ┃  ┃  ┊
    # 0.04┊ ┃  4  ┃ ┊ ┃ ┃  4  ┊
    #     ┊ ┃ ┏┻┓ ┃ ┊ ┃ ┃ ┏┻┓ ┊
    # 0.00┊ 0 1 2 3 ┊ 0 3 1 2 ┊
    #     0         6         7

    @pytest.mark.parametrize(
        ("h", "expected_path"),
        [
            # Just samples
            ([1, 0, 0, 0, 0, 1, 1], [0] * 7),
            # ([0, 1, 0, 0, 1, 1, 0], [1] * 7),
            # ([0, 0, 1, 0, 1, 1, 0], [2] * 7),
            # ([0, 0, 0, 1, 0, 0, 1], [3] * 7),
            # # Match root
            # ([0, 0, 0, 0, 0, 0, 0], [7] * 7),
        ],
    )
    def test_match_all_nodes(self, h, expected_path):
        # print()
        # print(self.ts().draw_text())
        # with open("tmp.svg", "w") as f:
        #     f.write(self.ts().draw_svg())
        validate_match_all_nodes(self.ts(), h, expected_path)

    @pytest.mark.parametrize(
        ("h", "expected_path"),
        [
            ([1, 0, 0, 0, 0, 1, 1], [0] * 7),
            ([0, 1, 0, 0, 1, 1, 0], [1] * 7),
            ([0, 0, 1, 0, 1, 1, 0], [2] * 7),
            ([0, 0, 0, 1, 0, 0, 1], [3] * 7),
            # Switch between each of the samples
            ([1, 1, 1, 1, 0, 0, 1], [0, 1, 2, 3, 3, 3, 3]),
        ],
    )
    def test_match_samples(self, h, expected_path):
        ts = self.ts()
        path = check_viterbi(ts, h)
        nt.assert_array_equal(expected_path, path)
        cm = check_forward_matrix(ts, h)
        check_backward_matrix(ts, h, cm)


class TestSimulationExamples:
    @pytest.mark.parametrize("n", [3, 10, 50])
    @pytest.mark.parametrize("L", [1, 10, 100])
    def test_continuous_genome(self, n, L):
        ts = msprime.simulate(
            n, length=L, recombination_rate=1, mutation_rate=1, random_seed=42
        )
        h = ts.genotype_matrix(samples=[0])[:, 0].T
        # NOTE this is a bit slow at the moment but we can disable the Python
        # implementation once testing has been improved on smaller examples.
        # Add ``compare_py=False``to these calls.
        check_viterbi(ts, h)
        cm = check_forward_matrix(ts, h)
        check_backward_matrix(ts, h, cm)
