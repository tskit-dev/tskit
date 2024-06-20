import copy
import itertools

import lshmm as ls
import msprime
import numpy as np
import pytest

import tskit

EQUAL_BOTH_HOM = 4
UNEQUAL_BOTH_HOM = 0
BOTH_HET = 7
REF_HOM_OBS_HET = 1
REF_HET_OBS_HOM = 2

MISSING = -1


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

    def __init__(self, tree_node=-1, inner_summation=-1, value_list=-1, value_index=-1):
        self.tree_node = tree_node
        self.value_list = value_list
        self.inner_summation = inner_summation
        self.value_index = value_index

    def copy(self):
        return ValueTransition(
            self.tree_node,
            self.inner_summation,
            self.value_list,
            self.value_index,
        )

    def __repr__(self):
        return repr(self.__dict__)

    def __str__(self):
        return repr(self)


class InternalValueTransition:
    """Simple struct holding the internal value transition values."""

    def __init__(self, tree_node=-1, value=-1, inner_summation=-1):
        self.tree_node = tree_node
        self.value = value
        self.inner_summation = inner_summation

    def __repr__(self):
        return repr(self.__dict__)

    def __str__(self):
        return repr(self)


class LsHmmAlgorithm:
    """
    Abstract superclass of Li and Stephens HMM algorithm.
    """

    def __init__(self, ts, rho, mu, precision=30):
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

    def decode_site_dict(self):
        """
        Decodes the tree encoding of the values into an explicit
        matrix.
        """
        A = np.zeros((self.ts.num_samples, self.ts.num_samples))
        # To look at the inner summations too.
        B = np.zeros((self.ts.num_samples, self.ts.num_samples))
        f = {st.tree_node: st for st in self.T}

        for j1, u1 in enumerate(self.ts.samples()):
            while u1 not in f:
                u1 = self.tree.parent(u1)
            f1 = {st.tree_node: st for st in f[u1].value_list}
            for j2, u2 in enumerate(self.ts.samples()):
                while u2 not in f1:
                    u2 = self.tree.parent(u2)
                A[j1, j2] = f1[u2].value
                B[j1, j2] = f1[u2].inner_summation
        return A, B

    def check_integrity(self):
        M = [st.tree_node for st in self.T if st.tree_node != -1]
        assert np.all(self.T_index[M] >= 0)
        index = np.ones_like(self.T_index, dtype=bool)
        index[M] = 0
        assert np.all(self.T_index[index] == -1)
        for j, st in enumerate(self.T):
            if st.tree_node != -1:
                assert j == self.T_index[st.tree_node]

    def stupid_compress_dict(self):
        """
        Duncan created a compression that just runs parsimony so is
        guaranteed to work.
        """
        tree = self.tree
        T = self.T
        alleles_string_vec = np.zeros(tree.num_samples()).astype("object")
        genotypes = np.zeros(tree.num_samples(), dtype=int)
        genotype_index = 0
        mapping_back = {}

        node_map = {st.tree_node: st for st in self.T}

        for st1 in T:
            if st1.tree_node != -1:
                alleles_string_tmp = [
                    f"{st2.tree_node}:{st2.value:.16f}" for st2 in st1.value_list
                ]
                alleles_string = ",".join(alleles_string_tmp)
                # Add an extra element that tells me the alleles_string there.
                st1.alleles_string = alleles_string
                st1.genotype_index = genotype_index
                # assert alleles_string not in mapping_back
                if alleles_string not in mapping_back:
                    mapping_back[alleles_string] = {
                        "tree_node": st1.tree_node,
                        "value_list": st1.value_list,
                        "inner_summation": st1.inner_summation,
                    }
                genotype_index += 1

        for leaf in tree.samples():
            u = leaf
            while u not in node_map:
                u = tree.parent(u)
            genotypes[leaf] = node_map[u].genotype_index

        alleles_string_vec = []
        for st in T:
            if st.tree_node != -1:
                alleles_string_vec.append(st.alleles_string)

        ancestral_allele, mutations = tree.map_mutations(genotypes, alleles_string_vec)

        # Retain the old T_index, because the internal T that's passed up the tree will
        # retain this ordering.
        old_T_index = copy.deepcopy(self.T_index)
        self.T_index = np.zeros(tree.tree_sequence.num_nodes, dtype=int) - 1
        self.N = np.zeros(tree.tree_sequence.num_nodes, dtype=int)
        self.T.clear()

        # First, create T root.
        self.T_index[tree.root] = 0
        self.T.append(
            ValueTransition(
                tree_node=tree.root,
                value_list=[
                    InternalValueTransition(
                        tree_node=tree.root,
                        value=mapping_back[ancestral_allele]["value_list"][
                            old_T_index[mapping_back[ancestral_allele]["tree_node"]]
                        ].value,
                    )
                ],
            )
        )

        # Then create the rest of T, adding the root each time to value_list
        for i, mut in enumerate(mutations):
            self.T_index[mut.node] = i + 1
            self.T.append(
                ValueTransition(
                    tree_node=mut.node,
                    value_list=[
                        InternalValueTransition(
                            tree_node=tree.root,
                            value=mapping_back[mut.derived_state]["value_list"][
                                old_T_index[mapping_back[ancestral_allele]["tree_node"]]
                            ].value,
                        )
                    ],
                )
            )

        # First add to the root
        for mut in mutations:
            self.T[self.T_index[tree.root]].value_list.append(
                InternalValueTransition(
                    tree_node=mut.node,
                    value=mapping_back[ancestral_allele]["value_list"][
                        old_T_index[mapping_back[mut.derived_state]["tree_node"]]
                    ].value,
                )
            )

        # Then add the rest of T_internal to each internal T.
        for mut1 in mutations:
            for mut2 in mutations:
                self.T[self.T_index[mut1.node]].value_list.append(
                    InternalValueTransition(
                        tree_node=mut2.node,
                        value=mapping_back[mut1.derived_state]["value_list"][
                            old_T_index[mapping_back[mut2.derived_state]["tree_node"]]
                        ].value,
                    )
                )

        # General approach here is to use
        # mapping_back[mut.derived_state]['value_list'][
        #   old_T_index[mapping_back[mut2.derived_state]["tree_node"]
        # ] and append this to the T_inner.

        node_map = {st.tree_node: st for st in self.T}

        for u in tree.samples():
            while u not in node_map:
                u = tree.parent(u)
            self.N[self.T_index[u]] += 1

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
                    ValueTransition(
                        tree_node=edge.child,
                        value_list=copy.deepcopy(T[T_index[u]].value_list),
                    )
                )
                # Add on this extra node to each of the internal lists
                for st in T:
                    if not (st.value_list == tskit.NULL):
                        st.value_list.append(
                            InternalValueTransition(
                                tree_node=edge.child,
                                value=st.value_list.copy()[T_index[u]].value,
                            )
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
                            tree_node=edge.parent,
                            value_list=copy.deepcopy(T[T_index[edge.child]].value_list),
                        )
                    )
                    # Add on this extra node to each of the internal lists
                    for st in T:
                        if not (st.value_list == tskit.NULL):
                            st.value_list.append(
                                InternalValueTransition(
                                    tree_node=edge.parent,
                                    value=st.value_list.copy()[
                                        T_index[edge.child]
                                    ].value,
                                )
                            )
            else:
                # Grafting into an existing subtree.
                while T_index[u] == -1:
                    u = parent[u]
                    assert u != -1
            assert T_index[u] != -1 and T_index[edge.child] != -1
            if (
                T[T_index[u]].value_list == T[T_index[edge.child]].value_list
            ):  # DEV: is this fine?
                st = T[T_index[edge.child]]
                # Mark the lower ValueTransition as unused.
                st.value_list = -1
                # Also need to mark the corresponding InternalValueTransition as
                # unused for the remaining states
                for st2 in T:
                    if not (st2.value_list == tskit.NULL):
                        st2.value_list[T_index[edge.child]].value = -1
                        st2.value_list[T_index[edge.child]].tree_node = -1

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
                    # Also need to mark the corresponding InternalValueTransition as
                    # unused for the remaining states
                    for st2 in T:
                        if not (st2.value_list == tskit.NULL):
                            st2.value_list[T_index[vt.tree_node]].value = -1
                            st2.value_list[T_index[vt.tree_node]].tree_node = -1
                    T_index[vt.tree_node] = -1
                    vt.tree_node = -1
                vt.value_index = -1

        self.N = np.zeros(self.tree.tree_sequence.num_nodes, dtype=int)
        node_map = {st.tree_node: st for st in self.T}

        for u in self.tree.samples():
            while u not in node_map:
                u = self.tree.parent(u)
            self.N[self.T_index[u]] += 1

    def update_probabilities(self, site, genotype_state):
        tree = self.tree
        T_index = self.T_index
        T = self.T
        alleles = ["0", "1"]
        allelic_state = self.allelic_state
        # Set the allelic_state for this site.
        allelic_state[self.tree.root] = alleles.index(site.ancestral_state)
        normalisation_factor_inner = {}

        for st1 in T:
            if st1.tree_node != -1:
                normalisation_factor_inner[st1.tree_node] = (
                    self.compute_normalisation_factor_inner_dict(st1.tree_node)
                )

        for st1 in T:
            if st1.tree_node != -1:
                for st2 in st1.value_list:
                    if st2.tree_node != -1:
                        self.T[self.T_index[st1.tree_node]].value_list[
                            self.T_index[st2.tree_node]
                        ].inner_summation = self.inner_summation_evaluation(
                            normalisation_factor_inner,
                            st1.tree_node,
                            st2.tree_node,
                        )
                        # (
                        #     normalisation_factor_inner[st1.tree_node]
                        #     + normalisation_factor_inner[st2.tree_node]
                        # )

        for mutation in site.mutations:
            u = mutation.node
            allelic_state[u] = alleles.index(mutation.derived_state)
            if T_index[u] == -1:
                while T_index[u] == tskit.NULL:
                    u = tree.parent(u)
                T_index[mutation.node] = len(T)
                T.append(
                    ValueTransition(
                        tree_node=mutation.node,
                        value_list=copy.deepcopy(T[T_index[u]].value_list),
                    )  # DEV: is it possible to not use deepcopies?
                )
                for st in T:
                    if not (st.value_list == tskit.NULL):
                        st.value_list.append(
                            InternalValueTransition(
                                tree_node=mutation.node,
                                value=st.value_list.copy()[T_index[u]].value,
                                inner_summation=st.value_list.copy()[
                                    T_index[u]
                                ].inner_summation,
                            )
                        )

        # Get the allelic state at the leaves.
        allelic_state[: tree.num_samples()] = tree.tree_sequence.genotype_matrix()[
            site.id, :
        ]

        query_is_het = genotype_state == 1
        query_is_missing = genotype_state == MISSING

        for st1 in T:
            u1 = st1.tree_node

            if u1 != -1:
                # Get the allelic_state at u. TODO we can cache these states to
                # avoid some upward traversals.
                v1 = u1
                while allelic_state[v1] == -1:
                    v1 = tree.parent(v1)
                    assert v1 != -1

                for st2 in st1.value_list:
                    u2 = st2.tree_node
                    if u2 != -1:
                        # Get the allelic_state at u. TODO we can cache these states to
                        # avoid some upward traversals.
                        v2 = u2
                        while allelic_state[v2] == -1:
                            v2 = tree.parent(v2)
                            assert v2 != -1

                        genotype_template_state = allelic_state[v1] + allelic_state[v2]
                        match = genotype_state == genotype_template_state
                        template_is_het = genotype_template_state == 1
                        # Fill in the value at the combination of states: (s1, s2)
                        st2.value = self.compute_next_probability_dict(
                            site.id,
                            st2.value,
                            st2.inner_summation,
                            match,
                            template_is_het,
                            query_is_het,
                            query_is_missing,
                            u1,
                            u2,
                            # Last two are not used by forward-backward
                            # but required by Viterbi
                        )

                # This will ensure that allelic_state[:n] is filled
                genotype_template_state = (
                    allelic_state[v1] + allelic_state[: tree.num_samples()]
                )
                # These are vectors of length n (at internal nodes).
                match = genotype_state == genotype_template_state
                template_is_het = genotype_template_state == 1

        # Unset the states
        allelic_state[tree.root] = -1
        for mutation in site.mutations:
            allelic_state[mutation.node] = -1

    def process_site(
        self, site, genotype_state, forwards=True
    ):  # Note: forwards turned on for Viterbi
        if forwards:
            # Forwards algorithm
            self.update_probabilities(site, genotype_state)
            self.stupid_compress_dict()
            s1 = self.compute_normalisation_factor_dict()
            T = self.T

            for st in T:
                if st.tree_node != tskit.NULL:
                    # Need to loop through value copy, and normalise
                    for st2 in st.value_list:
                        st2.value /= s1
                        st2.value = np.round(st2.value, self.precision)

            self.output.store_site(
                site.id, s1, [(st.tree_node, st.value_list) for st in self.T]
            )
        else:
            # Backwards algorithm
            self.output.store_site(
                site.id,
                self.output.normalisation_factor[site.id],
                [(st.tree_node, st.value_list) for st in self.T],
            )
            self.update_probabilities(site, genotype_state)
            self.stupid_compress_dict()
            b_last_sum = self.compute_normalisation_factor_dict()  # (Double sum)

            normalisation_factor_inner = {}

            for st1 in self.T:
                if st1.tree_node != -1:
                    normalisation_factor_inner[st1.tree_node] = (
                        self.compute_normalisation_factor_inner_dict(st1.tree_node)
                    )

            for st1 in self.T:
                if st1.tree_node != -1:
                    for st2 in st1.value_list:
                        if st2.tree_node != -1:
                            self.T[self.T_index[st1.tree_node]].value_list[
                                self.T_index[st2.tree_node]
                            ].inner_summation = (
                                normalisation_factor_inner[st1.tree_node]
                                + normalisation_factor_inner[st2.tree_node]
                            )

            s = self.output.normalisation_factor[site.id]
            for st1 in self.T:
                if st1.tree_node != tskit.NULL:
                    for st2 in st1.value_list:
                        st2.value = (
                            ((self.rho[site.id] / self.ts.num_samples) ** 2)
                            * b_last_sum
                            + (1 - self.rho[site.id])
                            * (self.rho[site.id] / self.ts.num_samples)
                            * st2.inner_summation
                            + (1 - self.rho[site.id]) ** 2 * st2.value
                        )
                        st2.value /= s
                        st2.value = np.round(st2.value, self.precision)

    def run_forward(self, g):
        n = self.ts.num_samples
        self.tree.clear()
        for u in self.ts.samples():
            self.T_index[u] = len(self.T)
            self.T.append(ValueTransition(tree_node=u, value_list=[]))
            for v in self.ts.samples():
                self.T[self.T_index[u]].value_list.append(
                    InternalValueTransition(tree_node=v, value=(1 / n) ** 2)
                )

        while self.tree.next():
            self.update_tree()
            for site in self.tree.sites():
                self.process_site(site, g[site.id])

        return self.output

    def run_backward(self, g):
        self.tree.clear()
        for u in self.ts.samples():
            self.T_index[u] = len(self.T)
            self.T.append(ValueTransition(tree_node=u, value_list=[]))
            for v in self.ts.samples():
                self.T[self.T_index[u]].value_list.append(
                    InternalValueTransition(tree_node=v, value=1)
                )

        while self.tree.next():
            self.update_tree()
            for site in self.tree.sites():
                self.process_site(site, g[site.id], forwards=False)
        return self.output

    def compute_normalisation_factor_dict(self):
        raise NotImplementedError()

    def compute_next_probability_dict(
        self,
        site_id,
        p_last,
        inner_summation,
        is_match,
        template_is_het,
        query_is_het,
        query_is_missing,
        node_1,  # Note: these are only used in Viterbi (node_1 and node_2)
        node_2,
    ):
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
        self.value_transitions[site] = copy.deepcopy(value_transitions)

    # Expose the same API as the low-level classes

    @property
    def num_transitions(self):
        a = [len(self.value_transitions[j]) for j in range(self.num_sites)]
        return np.array(a, dtype=np.int32)

    def get_site(self, site):
        return self.value_transitions[site]

    def decode_site_dict(self, tree, site_id):
        """
        Decodes the tree encoding of the values into an explicit
        matrix.
        """
        A = np.zeros((self.num_samples, self.num_samples))
        f = dict(self.value_transitions[site_id])

        for j1, u1 in enumerate(self.ts.samples()):
            while u1 not in f:
                u1 = tree.parent(u1)
            f1 = {st.tree_node: st for st in f[u1]}
            for j2, u2 in enumerate(self.ts.samples()):
                while u2 not in f1:
                    u2 = tree.parent(u2)
                A[j1, j2] = f1[u2].value
        return A

    def decode(self):
        """
        Decodes the tree encoding of the values into an explicit
        matrix.
        """
        A = np.zeros((self.num_sites, self.num_samples, self.num_samples))
        for tree in self.ts.trees():
            for site in tree.sites():
                A[site.id] = self.decode_site_dict(tree, site.id)
        return A


class ForwardMatrix(CompressedMatrix):
    """Class representing a compressed forward matrix."""


class BackwardMatrix(CompressedMatrix):
    """Class representing a compressed forward matrix."""


class ViterbiMatrix(CompressedMatrix):
    """
    Class representing the compressed Viterbi matrix.
    """

    def __init__(self, ts):
        super().__init__(ts)
        # Tuples containing the site, the pair of nodes in the tree,
        # and whether recombination is required
        self.double_recombination_required = [(-1, 0, 0, False)]
        self.single_recombination_required = [(-1, 0, 0, False)]

    def add_single_recombination_required(self, site, node_s1, node_s2, required):
        self.single_recombination_required.append((site, node_s1, node_s2, required))

    def add_double_recombination_required(self, site, node_s1, node_s2, required):
        self.double_recombination_required.append((site, node_s1, node_s2, required))

    def choose_sample_double(self, site_id, tree):
        max_value = -1
        u1 = -1
        u2 = -1

        for node_s1, value_outer in self.value_transitions[site_id]:
            for value_list in value_outer:
                value_tmp = value_list
                if value_tmp.value > max_value:
                    max_value = value_tmp.value
                    u1 = node_s1
                    u2 = value_tmp.tree_node

        assert u1 != -1
        assert u2 != -1

        transition_nodes = [u_tmp for (u_tmp, _) in self.value_transitions[site_id]]

        while not tree.is_sample(u1):
            for v in tree.children(u1):
                if v not in transition_nodes:
                    u1 = v
                    break
            else:
                raise AssertionError("could not find path")

        while not tree.is_sample(u2):
            for v in tree.children(u2):
                if v not in transition_nodes:
                    u2 = v
                    break
            else:
                raise AssertionError("could not find path")

        return (u1, u2)

    def choose_sample_single(self, site_id, tree, current_nodes):
        # I want to find which is the max between any choice if I switch just u1,
        # and any choice if I switch just u2.
        node_map = {st[0]: st for st in self.value_transitions[site_id]}
        to_compute = (
            np.zeros(2, dtype=int) - 1
        )  # We have two to compute - one for each single switch set of possibilities.

        for i, v in enumerate(current_nodes):  # (u1, u2)
            while v not in node_map:
                v = tree.parent(v)
            to_compute[i] = v

        # Need to go to the (j1 :)th entries, and the (:,j2)the entries,
        # and pick the best.
        T_index = np.zeros(self.ts.num_nodes, dtype=int) - 1
        for j, st in enumerate(self.value_transitions[site_id]):
            T_index[st[0]] = j

        node_single_switch_maxes = np.zeros(2, dtype=int) - 1
        single_switch = np.zeros(2) - 1

        for i, node in enumerate(to_compute):
            value_list = self.value_transitions[site_id][T_index[node]][1]
            s_inner = 0
            for st in value_list:
                j = st.tree_node
                if j != -1:
                    max_st = st.value
                    max_arg = st.tree_node
                    if max_st > s_inner:
                        s_inner = max_st
                        s_arg = max_arg
            node_single_switch_maxes[i] = s_arg
            single_switch[i] = s_inner

        if np.argmax(single_switch) == 0:
            # u1 is fixed, and we switch u2
            u1 = current_nodes[0]
            current_nodes = (u1, node_single_switch_maxes[0])
        else:
            # u2 is fixed, and we switch u1.
            u2 = current_nodes[1]
            current_nodes = (node_single_switch_maxes[1], u2)

        u1 = current_nodes[0]
        u2 = current_nodes[1]

        # Find the collection of transition nodes to use to descend down the tree
        transition_nodes = [u for (u, _) in self.value_transitions[site_id]]

        # Traverse down to find a leaves.
        while not tree.is_sample(u1):
            for v in tree.children(u1):
                if v not in transition_nodes:
                    u1 = v
                    break
            else:
                raise AssertionError("could not find path")

        while not tree.is_sample(u2):
            for v in tree.children(u2):
                if v not in transition_nodes:
                    u2 = v
                    break
            else:
                raise AssertionError("could not find path")

        current_nodes = (u1, u2)

        return current_nodes

    def traceback(self):
        # Run the traceback.
        m = self.ts.num_sites
        match = np.zeros((m, 2), dtype=int)

        single_recombination_tree = (
            np.zeros((self.ts.num_nodes, self.ts.num_nodes), dtype=int) - 1
        )
        double_recombination_tree = (
            np.zeros((self.ts.num_nodes, self.ts.num_nodes), dtype=int) - 1
        )

        tree = tskit.Tree(self.ts)
        tree.last()
        double_switch = True
        current_nodes = (-1, -1)
        current_node_outer = current_nodes[0]

        rr_single_index = len(self.single_recombination_required) - 1
        rr_double_index = len(self.double_recombination_required) - 1

        for site in reversed(self.ts.sites()):
            while tree.interval.left > site.position:
                tree.prev()
            assert tree.interval.left <= site.position < tree.interval.right

            # Fill in the recombination single tree
            j_single = rr_single_index
            # The above starts from the end of all the recombination required
            # information, and includes all the information for the current site.
            while self.single_recombination_required[j_single][0] == site.id:
                u1, u2, required = self.single_recombination_required[j_single][1:]
                single_recombination_tree[u1, u2] = required
                j_single -= 1

            # Fill in the recombination double tree
            j_double = rr_double_index
            # The above starts from the end of all the recombination required
            # information, and includes all the information for the current site.
            while self.double_recombination_required[j_double][0] == site.id:
                u1, u2, required = self.double_recombination_required[j_double][1:]
                double_recombination_tree[u1, u2] = required
                j_double -= 1

            # Note - current nodes are the leaf nodes.
            if current_node_outer == -1:
                if double_switch:
                    current_nodes = self.choose_sample_double(site.id, tree)
                else:
                    current_nodes = self.choose_sample_single(
                        site.id, tree, current_nodes
                    )

            match[site.id, :] = current_nodes

            # Now traverse up the tree from the current node. The first marked node
            # we meet tells us whether we need to recombine.
            current_node_outer = current_nodes[0]
            u1 = current_node_outer
            u2 = current_nodes[1]

            # Just need to move up the tree to evaluate u1 and u2.
            if double_switch:
                while u1 != -1 and double_recombination_tree[u1, u1] == -1:
                    u1 = tree.parent(u1)

                while u2 != -1 and double_recombination_tree[u1, u2] == -1:
                    u2 = tree.parent(u2)
            else:
                while u1 != -1 and single_recombination_tree[u1, u1] == -1:
                    u1 = tree.parent(u1)

                while u2 != -1 and single_recombination_tree[u1, u2] == -1:
                    u2 = tree.parent(u2)

            assert u1 != -1
            assert u2 != -1

            if double_recombination_tree[u1, u2] == 1:
                # Need to double switch at the next site.
                current_node_outer = -1
                double_switch = True
            elif single_recombination_tree[u1, u2] == 1:
                # Need to single switch at the next site
                current_node_outer = -1
                double_switch = False

            # Reset the nodes in the double recombination tree.
            j = rr_single_index
            while self.single_recombination_required[j][0] == site.id:
                u1_tmp, u2_tmp, _ = self.single_recombination_required[j][1:]
                single_recombination_tree[u1_tmp, u2_tmp] = -1
                j -= 1
            rr_single_index = j

            # Reset the nodes in the single recombination tree.
            j = rr_double_index
            while self.double_recombination_required[j][0] == site.id:
                u1_tmp, u2_tmp, _ = self.double_recombination_required[j][1:]
                double_recombination_tree[u1_tmp, u2_tmp] = -1
                j -= 1
            rr_double_index = j

        return match


class ForwardAlgorithm(LsHmmAlgorithm):
    """Runs the Li and Stephens forward algorithm."""

    def __init__(self, ts, rho, mu, precision=30):
        super().__init__(ts, rho, mu, precision)
        self.output = ForwardMatrix(ts)

    def inner_summation_evaluation(
        self, normalisation_factor_inner, st1_tree_node, st2_tree_node
    ):
        return (
            normalisation_factor_inner[st1_tree_node]
            + normalisation_factor_inner[st2_tree_node]
        )

    def compute_normalisation_factor_dict(self):
        s = 0
        for j, st in enumerate(self.T):
            assert st.tree_node != tskit.NULL
            assert self.N[j] > 0
            s += self.N[j] * self.compute_normalisation_factor_inner_dict(st.tree_node)
        return s

    def compute_normalisation_factor_inner_dict(self, node):
        s_inner = 0
        F_previous = self.T[self.T_index[node]].value_list
        for st in F_previous:
            j = st.tree_node
            if j != -1:
                s_inner += self.N[self.T_index[j]] * st.value
        return s_inner

    def compute_next_probability_dict(
        self,
        site_id,
        p_last,
        inner_normalisation_factor,
        is_match,
        template_is_het,
        query_is_het,
        query_is_missing,
        node_1,
        node_2,
    ):
        rho = self.rho[site_id]
        mu = self.mu[site_id]
        n = self.ts.num_samples

        p_t = (
            (rho / n) ** 2
            + ((1 - rho) * (rho / n)) * inner_normalisation_factor
            + (1 - rho) ** 2 * p_last
        )

        if query_is_missing:
            p_e = 1
        else:
            query_is_hom = np.logical_not(query_is_het)
            template_is_hom = np.logical_not(template_is_het)

            equal_both_hom = np.logical_and(
                np.logical_and(is_match, template_is_hom), query_is_hom
            )
            unequal_both_hom = np.logical_and(
                np.logical_and(np.logical_not(is_match), template_is_hom), query_is_hom
            )
            both_het = np.logical_and(template_is_het, query_is_het)
            ref_hom_obs_het = np.logical_and(template_is_hom, query_is_het)
            ref_het_obs_hom = np.logical_and(template_is_het, query_is_hom)

            p_e = (
                equal_both_hom * (1 - mu) ** 2
                + unequal_both_hom * (mu**2)
                + ref_hom_obs_het * (2 * mu * (1 - mu))
                + ref_het_obs_hom * (mu * (1 - mu))
                + both_het * ((1 - mu) ** 2 + mu**2)
            )

        return p_t * p_e


class BackwardAlgorithm(LsHmmAlgorithm):
    """Runs the Li and Stephens forward algorithm."""

    def __init__(self, ts, rho, mu, normalisation_factor, precision=10):
        super().__init__(ts, rho, mu, precision)
        self.output = BackwardMatrix(ts, normalisation_factor)

    def inner_summation_evaluation(
        self, normalisation_factor_inner, st1_tree_node, st2_tree_node
    ):
        return (
            normalisation_factor_inner[st1_tree_node]
            + normalisation_factor_inner[st2_tree_node]
        )

    def compute_normalisation_factor_dict(self):
        s = 0
        for j, st in enumerate(self.T):
            assert st.tree_node != tskit.NULL
            assert self.N[j] > 0
            s += self.N[j] * self.compute_normalisation_factor_inner_dict(st.tree_node)
        return s

    def compute_normalisation_factor_inner_dict(self, node):
        s_inner = 0
        F_previous = self.T[self.T_index[node]].value_list
        for st in F_previous:
            j = st.tree_node
            if j != -1:
                s_inner += self.N[self.T_index[j]] * st.value
        return s_inner

    def compute_next_probability_dict(
        self,
        site_id,
        p_next,
        inner_normalisation_factor,
        is_match,
        template_is_het,
        query_is_het,
        query_is_missing,
        node_1,
        node_2,
    ):
        mu = self.mu[site_id]
        template_is_hom = np.logical_not(template_is_het)

        if query_is_missing:
            p_e = 1
        else:
            query_is_hom = np.logical_not(query_is_het)

            equal_both_hom = np.logical_and(
                np.logical_and(is_match, template_is_hom), query_is_hom
            )
            unequal_both_hom = np.logical_and(
                np.logical_and(np.logical_not(is_match), template_is_hom), query_is_hom
            )
            both_het = np.logical_and(template_is_het, query_is_het)
            ref_hom_obs_het = np.logical_and(template_is_hom, query_is_het)
            ref_het_obs_hom = np.logical_and(template_is_het, query_is_hom)

            p_e = (
                equal_both_hom * (1 - mu) ** 2
                + unequal_both_hom * (mu**2)
                + ref_hom_obs_het * (2 * mu * (1 - mu))
                + ref_het_obs_hom * (mu * (1 - mu))
                + both_het * ((1 - mu) ** 2 + mu**2)
            )

        return p_next * p_e


class ViterbiAlgorithm(LsHmmAlgorithm):
    """
    Runs the Li and Stephens Viterbi algorithm.
    """

    def __init__(self, ts, rho, mu, precision=10):
        super().__init__(ts, rho, mu, precision)
        self.output = ViterbiMatrix(ts)

    def inner_summation_evaluation(
        self, normalisation_factor_inner, st1_tree_node, st2_tree_node
    ):
        return max(
            normalisation_factor_inner[st1_tree_node],
            normalisation_factor_inner[st2_tree_node],
        )

    def compute_normalisation_factor_dict(self):
        s = 0
        for st in self.T:
            assert st.tree_node != tskit.NULL
            max_st = self.compute_normalisation_factor_inner_dict(st.tree_node)
            if max_st > s:
                s = max_st
        if s == 0:
            raise ValueError(
                "Trying to match non-existent allele with zero mutation rate"
            )
        return s

    def compute_normalisation_factor_inner_dict(self, node):
        s_inner = 0
        V_previous = self.T[self.T_index[node]].value_list
        for st in V_previous:
            j = st.tree_node
            if j != -1:
                max_st = st.value
                if max_st > s_inner:
                    s_inner = max_st

        return s_inner

    def compute_next_probability_dict(
        self,
        site_id,
        p_last,
        inner_normalisation_factor,
        is_match,
        template_is_het,
        query_is_het,
        query_is_missing,
        node_1,
        node_2,
    ):
        r = self.rho[site_id]
        mu = self.mu[site_id]
        n = self.ts.num_samples
        r_n = r / n

        double_recombination_required = False
        single_recombination_required = False

        if query_is_missing:
            p_e = 1
        else:
            template_is_hom = np.logical_not(template_is_het)
            query_is_hom = np.logical_not(query_is_het)
            equal_both_hom = np.logical_and(
                np.logical_and(is_match, template_is_hom), query_is_hom
            )
            unequal_both_hom = np.logical_and(
                np.logical_and(np.logical_not(is_match), template_is_hom), query_is_hom
            )
            both_het = np.logical_and(template_is_het, query_is_het)
            ref_hom_obs_het = np.logical_and(template_is_hom, query_is_het)
            ref_het_obs_hom = np.logical_and(template_is_het, query_is_hom)

            p_e = (
                equal_both_hom * (1 - mu) ** 2
                + unequal_both_hom * (mu**2)
                + ref_hom_obs_het * (2 * mu * (1 - mu))
                + ref_het_obs_hom * (mu * (1 - mu))
                + both_het * ((1 - mu) ** 2 + mu**2)
            )

        no_switch = (1 - r) ** 2 + 2 * (r_n * (1 - r)) + r_n**2
        single_switch = r_n * (1 - r) + r_n**2
        double_switch = r_n**2

        V_single_switch = inner_normalisation_factor
        p_t = p_last * no_switch
        single_switch_tmp = single_switch * V_single_switch

        if single_switch_tmp > double_switch:
            # Then single switch is the alternative
            if p_t < single_switch * V_single_switch:
                p_t = single_switch * V_single_switch
                single_recombination_required = True
        else:
            # Double switch is the alternative
            if p_t < double_switch:
                p_t = double_switch
                double_recombination_required = True

        self.output.add_single_recombination_required(
            site_id, node_1, node_2, single_recombination_required
        )
        self.output.add_double_recombination_required(
            site_id, node_1, node_2, double_recombination_required
        )

        return p_t * p_e


def ls_forward_tree(g, ts, rho, mu, precision=30):
    """Forward matrix computation based on a tree sequence."""
    fa = ForwardAlgorithm(ts, rho, mu, precision=precision)
    return fa.run_forward(g)


def ls_backward_tree(g, ts_mirror, rho, mu, normalisation_factor, precision=30):
    """Backward matrix computation based on a tree sequence."""
    ba = BackwardAlgorithm(
        ts_mirror, rho, mu, normalisation_factor, precision=precision
    )
    return ba.run_backward(g)


def ls_viterbi_tree(g, ts, rho, mu, precision=30):
    """
    Viterbi path computation based on a tree sequence.
    """
    va = ViterbiAlgorithm(ts, rho, mu, precision=precision)
    return va.run_forward(g)


class LSBase:
    """Superclass of Li and Stephens tests."""

    def genotype_emission(self, mu, m):
        # Define the emission probability matrix
        e = np.zeros((m, 8))
        e[:, EQUAL_BOTH_HOM] = (1 - mu) ** 2
        e[:, UNEQUAL_BOTH_HOM] = mu**2
        e[:, BOTH_HET] = (1 - mu) ** 2 + mu**2
        e[:, REF_HOM_OBS_HET] = 2 * mu * (1 - mu)
        e[:, REF_HET_OBS_HOM] = mu * (1 - mu)

        return e

    def example_genotypes(self, ts):
        H = ts.genotype_matrix()
        s = H[:, 0].reshape(1, H.shape[0]) + H[:, 1].reshape(1, H.shape[0])
        H = H[:, 2:]

        genotypes = [
            s,
            H[:, -1].reshape(1, H.shape[0]) + H[:, -2].reshape(1, H.shape[0]),
        ]

        s_tmp = s.copy()
        s_tmp[0, -1] = MISSING
        genotypes.append(s_tmp)
        s_tmp = s.copy()
        s_tmp[0, ts.num_sites // 2] = MISSING
        genotypes.append(s_tmp)
        s_tmp = s.copy()
        s_tmp[0, :] = MISSING
        genotypes.append(s_tmp)

        m = ts.get_num_sites()
        n = H.shape[1]

        G = np.zeros((m, n, n))
        for i in range(m):
            G[i, :, :] = np.add.outer(H[i, :], H[i, :])

        return H, G, genotypes

    def example_parameters_genotypes(self, ts, seed=42):
        np.random.seed(seed)
        H, G, genotypes = self.example_genotypes(ts)
        n = H.shape[1]
        m = ts.get_num_sites()

        # Here we have equal mutation and recombination
        r = np.zeros(m) + 0.01
        mu = np.zeros(m) + 0.01
        r[0] = 0

        e = self.genotype_emission(mu, m)

        for s in genotypes:
            yield n, m, G, s, e, r, mu

        # Mixture of random and extremes
        rs = [np.zeros(m) + 0.999, np.zeros(m) + 1e-6, np.random.rand(m)]
        mus = [np.zeros(m) + 0.33, np.zeros(m) + 1e-6, np.random.rand(m) * 0.33]

        s = genotypes[0]
        for r, mu in itertools.product(rs, mus):
            r[0] = 0
            e = self.genotype_emission(mu, m)
            yield n, m, G, s, e, r, mu

    def assertAllClose(self, A, B):
        """Assert that all entries of two matrices are 'close'"""
        assert np.allclose(A, B, rtol=1e-5, atol=1e-8)

    # Define a bunch of very small tree-sequences for testing a collection of
    # parameters on
    @pytest.mark.skip(reason="No plans to implement diploid LS HMM yet.")
    def test_simple_n_10_no_recombination(self):
        ts = msprime.simulate(
            10, recombination_rate=0, mutation_rate=0.5, random_seed=42
        )
        assert ts.num_sites > 3
        self.verify(ts)

    @pytest.mark.skip(reason="No plans to implement diploid LS HMM yet.")
    def test_simple_n_6(self):
        ts = msprime.simulate(6, recombination_rate=2, mutation_rate=7, random_seed=42)
        assert ts.num_sites > 5
        self.verify(ts)

    @pytest.mark.skip(reason="No plans to implement diploid LS HMM yet.")
    def test_simple_n_8_high_recombination(self):
        ts = msprime.simulate(8, recombination_rate=20, mutation_rate=5, random_seed=42)
        assert ts.num_trees > 15
        assert ts.num_sites > 5
        self.verify(ts)

    # FIXME Reducing the number of test cases here as they take a long time to run,
    # and we will want to refactor the test infrastructure when implementing these
    # diploid methods in the library.

    # def test_simple_n_10_no_recombination_high_mut(self):
    #     ts = msprime.simulate(
    #         10, recombination_rate=0, mutation_rate=3, random_seed=42)
    #     assert ts.num_sites > 3
    #     self.verify(ts)

    # def test_simple_n_10_no_recombination_higher_mut(self):
    #     ts = msprime.simulate(
    #         20, recombination_rate=0, mutation_rate=3, random_seed=42)
    #     assert ts.num_sites > 3
    #     self.verify(ts)

    # def test_simple_n_8(self):
    #     ts = msprime.simulate(
    #         8, recombination_rate=2, mutation_rate=5, random_seed=42)
    #     assert ts.num_sites > 5
    #     self.verify(ts)

    # def test_simple_n_16(self):
    #     ts = msprime.simulate(
    #         16, recombination_rate=2, mutation_rate=5, random_seed=42)
    #     assert ts.num_sites > 5
    #     self.verify(ts)

    def verify(self, ts):
        raise NotImplementedError()


class FBAlgorithmBase(LSBase):
    """Base for forwards backwards algorithm tests."""


class VitAlgorithmBase(LSBase):
    """Base for viterbi algoritm tests."""


class TestMirroringDipdict(FBAlgorithmBase):
    """Tests that mirroring the tree sequence and running forwards and backwards
    algorithms give the same log-likelihood of observing the data."""

    def verify(self, ts):
        for n, m, _, s, _, r, mu in self.example_parameters_genotypes(ts):
            # Note, need to remove the first sample from the ts, and ensure that
            # invariant sites aren't removed.
            ts_check, mapping = ts.simplify(
                range(1, n + 1), filter_sites=False, map_nodes=True
            )
            H_check = ts_check.genotype_matrix()
            G_check = np.zeros((m, n, n))
            for i in range(m):
                G_check[i, :, :] = np.add.outer(H_check[i, :], H_check[i, :])

            cm_d = ls_forward_tree(s[0, :], ts_check, r, mu)
            ll_tree = np.sum(np.log10(cm_d.normalisation_factor))

            ts_check_mirror = mirror_coordinates(ts_check)
            r_flip = np.insert(np.flip(r)[:-1], 0, 0)
            cm_mirror = ls_forward_tree(
                np.flip(s[0, :]), ts_check_mirror, r_flip, np.flip(mu)
            )
            ll_mirror_tree_dict = np.sum(np.log10(cm_mirror.normalisation_factor))

            self.assertAllClose(ll_tree, ll_mirror_tree_dict)

            # Ensure that the decoded matrices are the same
            flipped_H_check = np.flip(H_check, axis=0)
            flipped_s = np.flip(s, axis=1)

            F_mirror_matrix, c, ll = ls.forwards(
                flipped_H_check,
                flipped_s,
                ploidy=2,
                prob_recombination=r_flip,
                prob_mutation=np.flip(mu),
                scale_mutation_rate=False,
            )

            self.assertAllClose(F_mirror_matrix, cm_mirror.decode())


class TestForwardDipTree(FBAlgorithmBase):
    """Tests that the tree algorithm computes the same forward matrix as the simple
    method."""

    def verify(self, ts):
        for n, m, _, s, _, r, mu in self.example_parameters_genotypes(ts):
            # Note, need to remove the first sample from the ts, and ensure that
            # invariant sites aren't removed.
            ts_check, mapping = ts.simplify(
                range(1, n + 1), filter_sites=False, map_nodes=True
            )
            H_check = ts_check.genotype_matrix()
            G_check = np.zeros((m, n, n))
            for i in range(m):
                G_check[i, :, :] = np.add.outer(H_check[i, :], H_check[i, :])

            F, c, ll = ls.forwards(
                reference_panel=H_check,
                query=s,
                ploidy=2,
                prob_recombination=r,
                prob_mutation=mu,
                scale_mutation_rate=False,
            )
            cm_d = ls_forward_tree(s[0, :], ts_check, r, mu)
            self.assertAllClose(cm_d.decode(), F)
            ll_tree = np.sum(np.log10(cm_d.normalisation_factor))
            self.assertAllClose(ll, ll_tree)


class TestForwardBackwardTree(FBAlgorithmBase):
    """Tests that the tree algorithm computes the same forward matrix as the simple
    method."""

    def verify(self, ts):
        for n, m, _, s, _, r, mu in self.example_parameters_genotypes(ts):
            # Note, need to remove the first sample from the ts, and ensure that
            # invariant sites aren't removed.
            ts_check, mapping = ts.simplify(
                range(1, n + 1), filter_sites=False, map_nodes=True
            )
            H_check = ts_check.genotype_matrix()
            G_check = np.zeros((m, n, n))
            for i in range(m):
                G_check[i, :, :] = np.add.outer(H_check[i, :], H_check[i, :])

            F, c, ll = ls.forwards(
                reference_panel=H_check,
                query=s,
                ploidy=2,
                prob_recombination=r,
                prob_mutation=mu,
                scale_mutation_rate=False,
            )
            B = ls.backwards(
                reference_panel=H_check,
                query=s,
                ploidy=2,
                normalisation_factor_from_forward=c,
                prob_recombination=r,
                prob_mutation=mu,
                scale_mutation_rate=False,
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


class TestTreeViterbiDip(VitAlgorithmBase):
    """
    Test that we have the same log-likelihood between tree and matrix
    implementations
    """

    def verify(self, ts):
        for n, m, _, s, _, r, mu in self.example_parameters_genotypes(ts):
            # Note, need to remove the first sample from the ts, and ensure that
            # invariant sites aren't removed.
            ts_check, mapping = ts.simplify(
                range(1, n + 1), filter_sites=False, map_nodes=True
            )
            H_check = ts_check.genotype_matrix()
            G_check = np.zeros((m, n, n))
            for i in range(m):
                G_check[i, :, :] = np.add.outer(H_check[i, :], H_check[i, :])
            ts_check = ts.simplify(range(1, n + 1), filter_sites=False)

            phased_path, ll = ls.viterbi(
                reference_panel=H_check,
                query=s,
                ploidy=2,
                prob_recombination=r,
                prob_mutation=mu,
                scale_mutation_rate=False,
            )
            path_ll_matrix = ls.path_loglik(
                reference_panel=H_check,
                query=s,
                ploidy=2,
                path=phased_path,
                prob_recombination=r,
                prob_mutation=mu,
                scale_mutation_rate=False,
            )

            c_v = ls_viterbi_tree(s[0, :], ts_check, r, mu)
            ll_tree = np.sum(np.log10(c_v.normalisation_factor))

            # Attempt to get the path
            path_tree_dict = c_v.traceback()
            # Work out the likelihood of the proposed path
            path_ll_tree = ls.path_loglik(
                reference_panel=H_check,
                query=s,
                ploidy=2,
                path=np.transpose(path_tree_dict),
                prob_recombination=r,
                prob_mutation=mu,
                scale_mutation_rate=False,
            )

            self.assertAllClose(ll, ll_tree)
            self.assertAllClose(path_ll_tree, path_ll_matrix)
