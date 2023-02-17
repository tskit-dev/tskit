# Efficient computation of the pairwise divergence matrix.
import msprime
import numpy as np

import tskit


def assert_dicts_close(d1, d2):
    if not set(d1.keys()) == set(d2.keys()):
        print("d1:", set(d1.keys()) - set(d2.keys()))
        print("d2:", set(d2.keys()) - set(d1.keys()))
    assert set(d1.keys()) == set(d2.keys())
    for k in d1:
        if not np.isclose(d1[k], d2[k]):
            print(k, d1[k], d2[k])
    for k in d1:
        assert np.isclose(d1[k], d2[k])


class DivergenceMatrix:
    def __init__(
        self,
        num_nodes,
        samples,
        nodes_time,
        edges_left,
        edges_right,
        edges_parent,
        edges_child,
        edge_insertion_order,
        edge_removal_order,
        sequence_length,
        verbosity=0,
        strict=False,
    ):
        # virtual root is at num_nodes; virtual samples are beyond that
        N = num_nodes + 1 + len(samples)
        # Quintuply linked tree
        self.parent = np.full(N, -1, dtype=np.int32)
        self.left_sib = np.full(N, -1, dtype=np.int32)
        self.right_sib = np.full(N, -1, dtype=np.int32)
        self.left_child = np.full(N, -1, dtype=np.int32)
        self.right_child = np.full(N, -1, dtype=np.int32)
        # Sample lists refer to sample *index*
        self.num_samples = np.full(N, 0, dtype=np.int32)
        # Edges and indexes
        self.edges_left = edges_left
        self.edges_right = edges_right
        self.edges_parent = edges_parent
        self.edges_child = edges_child
        self.edge_insertion_order = edge_insertion_order
        self.edge_removal_order = edge_removal_order
        self.sequence_length = sequence_length
        self.nodes_time = nodes_time
        self.samples = samples
        self.position = 0
        self.virtual_root = num_nodes
        self.x = np.zeros(N, dtype=np.float64)
        self.stack = [{} for _ in range(N)]
        self.verbosity = verbosity
        self.strict = strict

        n = samples.shape[0]
        for j in range(n):
            u = samples[j]
            self.num_samples[u] = 1
            self.insert_root(u)
            self.insert_branch(u, num_nodes + 1 + j)

    def print_state(self, msg=""):
        num_nodes = len(self.parent)
        print(f"..........{msg}................")
        print(f"position = {self.position}")
        for j in range(num_nodes):
            st = "NaN" if j >= self.virtual_root else f"{self.nodes_time[j]}"
            pt = (
                "NaN"
                if self.parent[j] == tskit.NULL
                else f"{self.nodes_time[self.parent[j]]}"
            )
            print(
                f"node {j} -> {self.parent[j]}: "
                f"ns = {self.num_samples[j]}, "
                f"z = ({pt} - {st})"
                f" * ({self.position} - {self.x[j]})"
                f" = {self.get_z(j)}"
            )
            for u, z in self.stack[j].items():
                print(f"   {(j, u)}: {z}")
        print(f"Virtual root: {self.virtual_root}")
        roots = []
        u = self.left_child[self.virtual_root]
        while u != tskit.NULL:
            roots.append(u)
            u = self.right_sib[u]
        print("Roots:", roots)
        print("Current state:")
        state = self.current_state()
        for k in state:
            print(f"   {k}: {state[k]}")

    def remove_branch(self, p, c):
        lsib = self.left_sib[c]
        rsib = self.right_sib[c]
        if lsib == -1:
            self.left_child[p] = rsib
        else:
            self.right_sib[lsib] = rsib
        if rsib == -1:
            self.right_child[p] = lsib
        else:
            self.left_sib[rsib] = lsib
        self.parent[c] = -1
        self.left_sib[c] = -1
        self.right_sib[c] = -1

    def insert_branch(self, p, c):
        self.parent[c] = p
        u = self.right_child[p]
        if u == -1:
            self.left_child[p] = c
            self.left_sib[c] = -1
            self.right_sib[c] = -1
        else:
            self.right_sib[u] = c
            self.left_sib[c] = u
            self.right_sib[c] = -1
        self.right_child[p] = c

    def remove_root(self, root):
        self.remove_branch(self.virtual_root, root)

    def insert_root(self, root):
        self.insert_branch(self.virtual_root, root)
        self.parent[root] = -1

    def remove_edge(self, p, c):
        assert p != -1
        self.remove_branch(p, c)
        # check for root changes
        u = p
        while u != tskit.NULL:
            path_end = u
            path_end_was_root = self.num_samples[u] > 0
            self.num_samples[u] -= self.num_samples[c]
            u = self.parent[u]
        if path_end_was_root and (self.num_samples[path_end] == 0):
            self.remove_root(path_end)
        if self.num_samples[c] > 0:
            self.insert_root(c)

    def insert_edge(self, p, c):
        assert p != -1
        assert self.parent[c] == -1, "contradictory edges"
        # check for root changes
        u = p
        while u != tskit.NULL:
            path_end = u
            path_end_was_root = self.num_samples[u] > 0
            self.num_samples[u] += self.num_samples[c]
            u = self.parent[u]
        if self.num_samples[c] > 0:
            self.remove_root(c)
        if (self.num_samples[path_end] > 0) and not path_end_was_root:
            self.insert_root(path_end)
        self.insert_branch(p, c)

    # begin stack stuff

    def add_to_stack(self, u, v, z):
        if z > 0:
            if v not in self.stack[u]:
                self.stack[u][v] = 0.0
                assert u not in self.stack[v]
                self.stack[v][u] = 0.0
            self.stack[u][v] += z
            self.stack[v][u] += z
        # pedantic error checking:
        if self.strict:
            p = u
            while p != tskit.NULL:
                assert p != v
                p = self.parent[p]
            p = v
            while p != tskit.NULL:
                assert p != u
                p = self.parent[p]

    def empty_stack(self, n):
        for w in self.stack[n]:
            assert n in self.stack[w]
            del self.stack[w][n]
        self.stack[n].clear()

    def verify_zero_spine(self, n):
        """
        Verify that there are no contributions along the path
        from n up to the root. (should be true after clear_spine)
        """
        for u, z in self.stack[n].items():
            if z != 0:
                print(f"Uh-oh!: [{n}] : ({u}, {z})")
            assert z == 0
        while n != tskit.NULL:
            assert self.parent[n] == tskit.NULL or self.x[n] == self.position
            n = self.parent[n]

    def get_z(self, u):
        p = self.parent[u]
        return (
            0
            if p == tskit.NULL
            else (
                (self.nodes_time[self.parent[u]] - self.nodes_time[u])
                * (self.position - self.x[u])
            )
        )

    def mrca(self, a, b):
        # just used for `current_state`
        aa = [a]
        while a != tskit.NULL:
            a = self.parent[a]
            aa.append(a)
        while b not in aa:
            b = self.parent[b]
        return b

    def current_state(self):
        """
        Compute the current output, for debugging.
        NOTE that the path back to the roots of disconnected trees
        *still counts* for divergence *between* those trees!
        (In other words, disconnected trees act as if they are
        connected to a virtual root by a branch of length zero.)
        """
        if self.verbosity > 1:
            print("---------------")
        out = {(a, b): 0 for a in self.samples for b in self.samples if a < b}
        for a in self.samples:
            for b in self.samples:
                if a < b:
                    k = tuple(sorted([a, b]))
                    m = self.mrca(a, b)
                    # edges on the path up from a
                    pa = a
                    while pa != m:
                        if self.verbosity > 1:
                            print("edge:", k, pa, self.get_z(pa))
                        out[k] += self.get_z(pa)
                        pa = self.parent[pa]
                    # edges on the path up from b
                    pb = b
                    while pb != m:
                        if self.verbosity > 1:
                            print("edge:", k, pb, self.get_z(pb))
                        out[k] += self.get_z(pb)
                        pb = self.parent[pb]
                    # pairwise stack references along the way
                    pa = a
                    while pa != m:
                        pb = b
                        while pb != m:
                            for w, z in self.stack[pa].items():
                                if w == pb:
                                    if self.verbosity > 1:
                                        print("stack:", k, (pa, pb), z)
                                    out[k] += z
                            pb = self.parent[pb]
                        pa = self.parent[pa]
        if self.verbosity > 1:
            print("---------------")
        return out

    def clear_edge(self, n):
        if self.verbosity > 0:
            print(f"clear_edge({n})")
        if self.strict:
            # this operation should not change the current output
            before_state = self.current_state()
        z = self.get_z(n)
        self.x[n] = self.position
        assert self.get_z(n) == 0
        # iterate over the siblings of the path to the virtual root
        c = n
        p = self.parent[c]
        # how to better include the virtual root in the traversal?
        root = False
        while p != tskit.NULL:
            s = self.left_child[p]
            while s != tskit.NULL:
                if s != c:
                    if self.verbosity > 1:
                        print(f"adding {z} to {(n, s)}")
                    self.add_to_stack(n, s, z)
                s = self.right_sib[s]
            c = p
            p = self.parent[c]
            if not root and p == tskit.NULL:
                p = self.virtual_root
                root = True
        if self.strict:
            after_state = self.current_state()
            assert_dicts_close(before_state, after_state)

    def push_down(self, n):
        """
        Push down references in the stack from n to other nodes
        to the children of n.
        """
        if self.strict:
            # this operation should not change the current output
            before_state = self.current_state()
        if self.verbosity > 0:
            print(f"push_down({n})")
        for w, z in self.stack[n].items():
            c = self.left_child[n]
            while c != tskit.NULL:
                if self.verbosity > 1:
                    print(f"adding {z} to {(w, c)}")
                self.add_to_stack(w, c, z)
                c = self.right_sib[c]
        self.empty_stack(n)
        assert len(self.stack[n]) == 0
        if self.strict:
            after_state = self.current_state()
            assert_dicts_close(before_state, after_state)

    def clear_spine(self, n):
        """
        Clears all nodes on the path from the virtual root down to n
        by pushing the contributions of all their branches to the stack
        and pushing all stack references to their children.
        """
        if self.verbosity > 0:
            print(f"clear_spine({n})")
        if self.strict:
            # this operation should not change the current output
            before_state = self.current_state()
        spine = []
        p = n
        while p != tskit.NULL:
            spine.append(p)
            p = self.parent[p]
        spine.append(self.virtual_root)
        for j in range(len(spine) - 1, -1, -1):
            p = spine[j]
            self.clear_edge(p)
            self.push_down(p)
            self.verify_zero_spine(p)
        self.verify_zero_spine(n)
        if self.strict:
            after_state = self.current_state()
            assert_dicts_close(before_state, after_state)

    def run(self):
        sequence_length = self.sequence_length
        M = self.edges_left.shape[0]
        in_order = self.edge_insertion_order
        out_order = self.edge_removal_order
        edges_left = self.edges_left
        edges_right = self.edges_right
        edges_parent = self.edges_parent
        edges_child = self.edges_child

        j = 0
        k = 0
        # TODO: self.position is redundant with left
        self.position = left = 0

        while k < M and left <= self.sequence_length:
            while k < M and edges_right[out_order[k]] == left:
                p = edges_parent[out_order[k]]
                c = edges_child[out_order[k]]
                self.clear_edge(c)
                self.clear_spine(p)
                assert self.x[c] == self.position
                assert self.parent[p] == tskit.NULL or self.x[p] == self.position
                self.remove_edge(p, c)
                k += 1
                if self.verbosity > 1:
                    self.print_state(f"remove {p, c}")
            while j < M and edges_left[in_order[j]] == left:
                p = edges_parent[in_order[j]]
                c = edges_child[in_order[j]]
                if self.position > 0:
                    self.clear_spine(p)
                assert self.parent[p] == tskit.NULL or self.x[p] == self.position
                self.insert_edge(p, c)
                self.x[c] = self.position
                j += 1
                if self.verbosity > 1:
                    self.print_state(f"add {p, c}")
            right = sequence_length
            if j < M:
                right = min(right, edges_left[in_order[j]])
            if k < M:
                right = min(right, edges_right[out_order[k]])
            self.position = left = right
        assert j == M and left == self.sequence_length
        # clear remaining things down to virtual samples
        for u in self.samples:
            self.push_down(u)
        out = np.zeros((len(self.samples), len(self.samples)))
        for out_i in range(len(self.samples)):
            i = out_i + self.virtual_root + 1
            for j, z in self.stack[i].items():
                assert j > self.virtual_root
                assert j <= self.virtual_root + len(self.samples)
                out_j = j - self.virtual_root - 1
                out[out_i, out_j] = z
        return out


def divergence_matrix(ts, **kwargs):
    dm = DivergenceMatrix(
        ts.num_nodes,
        samples=ts.samples(),
        nodes_time=ts.nodes_time,
        edges_left=ts.edges_left,
        edges_right=ts.edges_right,
        edges_parent=ts.edges_parent,
        edges_child=ts.edges_child,
        edge_insertion_order=ts.indexes_edge_insertion_order,
        edge_removal_order=ts.indexes_edge_removal_order,
        sequence_length=ts.sequence_length,
        **kwargs,
    )
    return dm.run()


def lib_divergence_matrix(ts, mode="branch"):
    out = ts.divergence(
        [[u] for u in ts.samples()],
        [(i, j) for i in range(ts.num_samples) for j in range(ts.num_samples)],
        mode=mode,
        span_normalise=False,
    ).reshape((ts.num_samples, ts.num_samples))
    for i in range(ts.num_samples):
        out[i, i] = 0
    return out


def small_example():
    ts = msprime.sim_ancestry(
        4,
        ploidy=1,
        population_size=10,
        sequence_length=10,
        recombination_rate=0.01,
        random_seed=124,
    )
    print(f"========trees: {ts.num_trees}=============")
    divergence_matrix(ts, verbosity=2)
    print("=====================")


def verify():
    for seed in range(1, 10):
        ts = msprime.sim_ancestry(
            10,
            ploidy=1,
            population_size=20,
            sequence_length=100,
            recombination_rate=0.01,
            random_seed=seed,
        )
        t = ts.tables
        n = t.nodes.add_row(time=-1, flags=tskit.NODE_IS_SAMPLE)
        t.edges.add_row(parent=0, child=n, left=0, right=ts.sequence_length)
        t.sort()
        t.build_index()
        ts = t.tree_sequence()
        D1 = lib_divergence_matrix(ts, mode="branch")
        D2 = divergence_matrix(ts, verbosity=0, strict=False)
        print(f"========{ts.num_trees}=============")
        for i in range(D2.shape[0]):
            for j in range(D2.shape[1]):
                print(i, j, D1[i, j], D2[i, j])
        print("=====================")
        assert np.allclose(D1, D2)


if __name__ == "__main__":

    np.set_printoptions(linewidth=500, precision=4)
    # small_example()
    verify()
    # compare_perf()
