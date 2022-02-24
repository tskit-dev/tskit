import numpy as np
import tskit

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
    tables.sites.position = L - tables.sites.position# + 1
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
        return ValueTransition(self.tree_node, self.value, self.value_index)

    def __repr__(self):
        return repr(self.__dict__)

    def __str__(self):
        return repr(self)


class LsHmmAlgorithm:
    """
    Abstract superclass of Li and Stephens HMM algorithm.
    """

    def __init__(self, ts, rho, mu, precision=10):
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
        alleles = ["0", "1"]
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
                st.value = self.compute_next_probability(site.id, st.value, match)

        # Unset the states
        allelic_state[tree.root] = -1
        for mutation in site.mutations:
            allelic_state[mutation.node] = -1

    def process_site(self, site, haplotype_state, forwards=True):
        if forwards:
            # Forwards algorithm
            self.update_probabilities(site, haplotype_state)
            self.compress()
            s = self.compute_normalisation_factor()
            for st in self.T:
                if st.tree_node != tskit.NULL:
                    st.value /= s
                    st.value = round(st.value, self.precision)
            self.output.store_site(site.id, s, [(st.tree_node, st.value) for st in self.T])
        else:
            # Backwards algorithm
            self.output.store_site(site.id, self.output.normalisation_factor[site.id], [(st.tree_node, st.value) for st in self.T])
            self.update_probabilities(site, haplotype_state)
            self.compress()
            b_last_sum = self.compute_normalisation_factor()
            s = self.output.normalisation_factor[site.id]
            for st in self.T:
                if st.tree_node != tskit.NULL:
                    st.value = (self.rho[site.id]/self.ts.num_samples) * b_last_sum + (1 - self.rho[site.id]) * st.value
                    st.value /= s
                    st.value = round(st.value, self.precision)

    def run_forward(self, h):
        n = self.ts.num_samples
        self.tree.clear()
        for u in self.ts.samples():
            self.T_index[u] = len(self.T)
            self.T.append(ValueTransition(tree_node=u, value=1/n))
        while self.tree.next():
            self.update_tree()
            for site in self.tree.sites():
                self.process_site(site, h[site.id])
        return self.output

    def run_backward(self, h):
        n = self.ts.num_samples
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

    def compute_next_probability(self, site_id, p_last, is_match):
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

class ForwardAlgorithm(LsHmmAlgorithm):
    """Runs the Li and Stephens forward algorithm."""

    def __init__(self, ts, rho, mu, precision=10):
        super().__init__(ts, rho, mu, precision)
        self.output = ForwardMatrix(ts)

    def compute_normalisation_factor(self):
        s = 0
        for j, st in enumerate(self.T):
            assert st.tree_node != tskit.NULL
            assert self.N[j] > 0
            s += self.N[j] * st.value
        return s

    def compute_next_probability(self, site_id, p_last, is_match):
        rho = self.rho[site_id]
        mu = self.mu[site_id]
        n = self.ts.num_samples

        p_t = p_last * (1 - rho) + rho / n
        p_e = mu
        if is_match:
            p_e = 1 - mu
        return p_t * p_e

class BackwardAlgorithm(LsHmmAlgorithm):
    """Runs the Li and Stephens forward algorithm."""

    def __init__(self, ts, rho, mu, normalisation_factor, precision=10):
        super().__init__(ts, rho, mu, precision)
        self.output = BackwardMatrix(ts, normalisation_factor)

    def compute_normalisation_factor(self):
        s = 0
        for j, st in enumerate(self.T):
            assert st.tree_node != tskit.NULL
            assert self.N[j] > 0
            s += self.N[j] * st.value
        return s

    def compute_next_probability(self, site_id, p_next, is_match):
        mu = self.mu[site_id]
        e = mu
        if is_match:
            e = 1 - mu
        return p_next * e

def ls_forward_tree(h, ts, rho, mu, precision=30):
    """Forward matrix computation based on a tree sequence."""
    fa = ForwardAlgorithm(ts, rho, mu, precision=precision)
    return fa.run_forward(h)

def ls_backward_tree(h, ts_mirror, rho, mu, normalisation_factor, precision=30):
    """Backward matrix computation based on a tree sequence."""
    ba = BackwardAlgorithm(ts_mirror, rho, mu, normalisation_factor, precision=precision)
    return ba.run_backward(h)
