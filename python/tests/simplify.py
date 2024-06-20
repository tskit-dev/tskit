# MIT License
#
# Copyright (c) 2019-2024 Tskit Developers
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
Python implementation of the simplify algorithm.
"""
import sys

import numpy as np
import portion

import tskit


def overlapping_segments(segments):
    """
    Returns an iterator over the (left, right, X) tuples describing the
    distinct overlapping segments in the specified set.
    """
    S = sorted(segments, key=lambda x: x.left)
    n = len(S)
    # Insert a sentinel at the end for convenience.
    S.append(Segment(sys.float_info.max, 0))
    right = S[0].left
    X = []
    j = 0
    while j < n:
        # Remove any elements of X with right <= left
        left = right
        X = [x for x in X if x.right > left]
        if len(X) == 0:
            left = S[j].left
        while j < n and S[j].left == left:
            X.append(S[j])
            j += 1
        j -= 1
        right = min(x.right for x in X)
        right = min(right, S[j + 1].left)
        yield left, right, X
        j += 1

    while len(X) > 0:
        left = right
        X = [x for x in X if x.right > left]
        if len(X) > 0:
            right = min(x.right for x in X)
            yield left, right, X


class Segment:
    """
    A class representing a single segment. Each segment has a left and right,
    denoting the loci over which it spans, a node and a next, giving the next
    in the chain.

    The node it records is the *output* node ID.
    """

    def __init__(self, left=None, right=None, node=None, next_segment=None):
        self.left = left
        self.right = right
        self.node = node
        self.next = next_segment

    def __str__(self):
        s = "({}-{}->{}:next={})".format(
            self.left, self.right, self.node, repr(self.next)
        )
        return s

    def __repr__(self):
        return repr((self.left, self.right, self.node))

    def __lt__(self, other):
        return (self.left, self.right, self.node) < (other.left, other.right, self.node)


class Simplifier:
    """
    Simplifies a tree sequence to its minimal representation given a subset
    of the leaves.
    """

    def __init__(
        self,
        ts,
        sample,
        reduce_to_site_topology=False,
        filter_sites=True,
        filter_populations=True,
        filter_individuals=True,
        keep_unary=False,
        keep_unary_in_individuals=False,
        keep_input_roots=False,
        filter_nodes=True,
        update_sample_flags=True,
    ):
        self.ts = ts
        self.n = len(sample)
        self.reduce_to_site_topology = reduce_to_site_topology
        self.sequence_length = ts.sequence_length
        self.filter_sites = filter_sites
        self.filter_populations = filter_populations
        self.filter_individuals = filter_individuals
        self.filter_nodes = filter_nodes
        self.update_sample_flags = update_sample_flags
        self.keep_unary = keep_unary
        self.keep_unary_in_individuals = keep_unary_in_individuals
        self.keep_input_roots = keep_input_roots
        self.num_mutations = ts.num_mutations
        self.input_sites = list(ts.sites())
        self.A_head = [None for _ in range(ts.num_nodes)]
        self.A_tail = [None for _ in range(ts.num_nodes)]
        self.tables = self.ts.tables.copy()
        self.tables.clear()
        self.edge_buffer = {}
        self.node_id_map = np.zeros(ts.num_nodes, dtype=np.int32) - 1
        self.is_sample = np.zeros(ts.num_nodes, dtype=np.int8)
        self.mutation_node_map = [-1 for _ in range(self.num_mutations)]
        self.samples = set(sample)
        self.sort_offset = -1
        # We keep a map of input nodes to mutations.
        self.mutation_map = [[] for _ in range(ts.num_nodes)]
        position = ts.sites_position
        site = ts.mutations_site
        node = ts.mutations_node
        for mutation_id in range(ts.num_mutations):
            site_position = position[site[mutation_id]]
            self.mutation_map[node[mutation_id]].append((site_position, mutation_id))

        for sample_id in sample:
            self.is_sample[sample_id] = 1

        if not self.filter_nodes:
            # NOTE In the C implementation we would really just not touch the
            # original tables.
            self.tables.nodes.replace_with(self.ts.tables.nodes)
            if self.update_sample_flags:
                flags = self.tables.nodes.flags
                # Zero out other sample flags
                flags = np.bitwise_and(
                    flags, np.uint32(~tskit.NODE_IS_SAMPLE & 0xFFFFFFFF)
                )
                flags[sample] |= tskit.NODE_IS_SAMPLE
                self.tables.nodes.flags = flags.astype(np.uint32)

            self.node_id_map[:] = np.arange(ts.num_nodes)
            for sample_id in sample:
                self.add_ancestry(sample_id, 0, self.sequence_length, sample_id)
        else:
            for sample_id in sample:
                output_id = self.record_node(sample_id)
                self.add_ancestry(sample_id, 0, self.sequence_length, output_id)

        self.position_lookup = None
        if self.reduce_to_site_topology:
            self.position_lookup = np.hstack([[0], position, [self.sequence_length]])

    def record_node(self, input_id):
        """
        Adds a new node to the output table corresponding to the specified input
        node ID.
        """
        node = self.ts.node(input_id)
        flags = node.flags
        if self.update_sample_flags:
            # Need to zero out the sample flag
            flags &= ~tskit.NODE_IS_SAMPLE
            if self.is_sample[input_id]:
                flags |= tskit.NODE_IS_SAMPLE
        output_id = self.tables.nodes.append(node.replace(flags=flags))
        self.node_id_map[input_id] = output_id
        return output_id

    def rewind_node(self, input_id, output_id):
        """
        Remove the mapping for the specified input and output node pair. This is
        done because there are no edges referring to the node.
        """
        assert output_id == len(self.tables.nodes) - 1
        assert output_id == self.node_id_map[input_id]
        self.tables.nodes.truncate(output_id)
        self.node_id_map[input_id] = -1

    def flush_edges(self):
        """
        Flush the edges to the output table after sorting and squashing
        any redundant records.
        """
        num_edges = 0
        for child in sorted(self.edge_buffer.keys()):
            for edge in self.edge_buffer[child]:
                self.tables.edges.append(edge)
                num_edges += 1
        self.edge_buffer.clear()
        return num_edges

    def record_edge(self, left, right, parent, child):
        """
        Adds an edge to the output list.
        """
        if self.reduce_to_site_topology:
            X = self.position_lookup
            left_index = np.searchsorted(X, left)
            right_index = np.searchsorted(X, right)
            # Find the smallest site position index greater than or equal to left
            # and right, i.e., slide each endpoint of an interval to the right
            # until they hit a site position. If both left and right map to the
            # the same position then we discard this edge. We also discard an
            # edge if left = 0 and right is less than the first site position.
            if left_index == right_index or (left_index == 0 and right_index == 1):
                return
            # Remap back to zero if the left end maps to the first site.
            if left_index == 1:
                left_index = 0
            left = X[left_index]
            right = X[right_index]
        if child not in self.edge_buffer:
            self.edge_buffer[child] = [tskit.Edge(left, right, parent, child)]
        else:
            last = self.edge_buffer[child][-1]
            if last.right == left:
                last.right = right
            else:
                self.edge_buffer[child].append(tskit.Edge(left, right, parent, child))

    def print_state(self):
        print(".................")
        print("Ancestors: ")
        num_nodes = len(self.A_tail)
        for j in range(num_nodes):
            print("\t", j, "->", end="")
            x = self.A_head[j]
            while x is not None:
                print(f"({x.left}-{x.right}->{x.node})", end="")
                x = x.next
            print()
        print("Mutation map:")
        for u in range(len(self.mutation_map)):
            v = self.mutation_map[u]
            if len(v) > 0:
                print("\t", u, "->", v)
        print("Node ID map: (input->output)")
        for input_id, output_id in enumerate(self.node_id_map):
            print("\t", input_id, "->", output_id)
        print("Mutation node map")
        for j in range(self.num_mutations):
            print("\t", j, "->", self.mutation_node_map[j])
        print("Output:")
        print(self.tables)
        self.check_state()

    def map_mutations(self, left, right, input_id, output_id):
        """
        Map any mutations for the input node ID on the
        interval to its output ID.
        """
        assert output_id != -1
        # TODO we should probably remove these as they are used.
        # Or else, binary search the list so it's quick.
        for x, mutation_id in self.mutation_map[input_id]:
            if left <= x < right:
                self.mutation_node_map[mutation_id] = output_id

    def add_ancestry(self, input_id, left, right, node):
        tail = self.A_tail[input_id]
        if tail is None:
            x = Segment(left, right, node)
            self.A_head[input_id] = x
            self.A_tail[input_id] = x
        else:
            if tail.right == left and tail.node == node:
                tail.right = right
            else:
                x = Segment(left, right, node)
                tail.next = x
                self.A_tail[input_id] = x

        self.map_mutations(left, right, input_id, node)

    def merge_labeled_ancestors(self, S, input_id):
        """
        All ancestry segments in S come together into a new parent.
        The new parent must be assigned and any overlapping segments coalesced.
        """
        output_id = self.node_id_map[input_id]
        is_sample = self.is_sample[input_id]
        if is_sample:
            # Free up the existing ancestry mapping.
            x = self.A_tail[input_id]
            assert x.left == 0 and x.right == self.sequence_length
            self.A_tail[input_id] = None
            self.A_head[input_id] = None

        prev_right = 0
        for left, right, X in overlapping_segments(S):
            if len(X) == 1:
                ancestry_node = X[0].node
                if is_sample:
                    self.record_edge(left, right, output_id, ancestry_node)
                    ancestry_node = output_id
                elif self.keep_unary or (
                    self.keep_unary_in_individuals
                    and self.ts.node(input_id).individual >= 0
                ):
                    if output_id == -1:
                        output_id = self.record_node(input_id)
                    self.record_edge(left, right, output_id, ancestry_node)
            else:
                if output_id == -1:
                    output_id = self.record_node(input_id)
                ancestry_node = output_id
                for x in X:
                    self.record_edge(left, right, output_id, x.node)
            if is_sample and left != prev_right:
                # Fill in any gaps in the ancestry for the sample
                self.add_ancestry(input_id, prev_right, left, output_id)
            if self.keep_unary or (
                self.keep_unary_in_individuals
                and self.ts.node(input_id).individual >= 0
            ):
                ancestry_node = output_id
            self.add_ancestry(input_id, left, right, ancestry_node)
            prev_right = right

        if is_sample and prev_right != self.sequence_length:
            # If a trailing gap exists in the sample ancestry, fill it in.
            self.add_ancestry(input_id, prev_right, self.sequence_length, output_id)
        if output_id != -1:
            num_edges = self.flush_edges()
            if self.filter_nodes and num_edges == 0 and not is_sample:
                self.rewind_node(input_id, output_id)

    def extract_ancestry(self, edge):
        S = []
        x = self.A_head[edge.child]

        x_head = None
        x_prev = None
        while x is not None:
            if x.right > edge.left and edge.right > x.left:
                y = Segment(max(x.left, edge.left), min(x.right, edge.right), x.node)
                # print("snip", y)
                S.append(y)
                assert x.left <= y.left
                assert x.right >= y.right
                seg_left = None
                seg_right = None
                if x.left != y.left:
                    seg_left = Segment(x.left, y.left, x.node)
                    if x_prev is None:
                        x_head = seg_left
                    else:
                        x_prev.next = seg_left
                    x_prev = seg_left
                if x.right != y.right:
                    x.left = y.right
                    seg_right = x
                else:
                    # Free x
                    seg_right = x.next
                if x_prev is None:
                    x_head = seg_right
                else:
                    x_prev.next = seg_right
                x = seg_right
            else:
                if x_prev is None:
                    x_head = x
                x_prev = x
                x = x.next
        # Note - we had some code to defragment segments in the output
        # chain here, but couldn't find an example where it needed to
        # be called. So, looks like squashing isn't necessary here.
        self.A_head[edge.child] = x_head
        self.A_tail[edge.child] = x_prev
        return S

    def process_parent_edges(self, edges):
        """
        Process all of the edges for a given parent.
        """
        assert len({e.parent for e in edges}) == 1
        parent = edges[0].parent
        S = []
        for edge in edges:
            S.extend(self.extract_ancestry(edge))
        self.merge_labeled_ancestors(S, parent)
        self.check_state()

    def finalise_sites(self):
        # Build a map from the old mutation IDs to new IDs. Any mutation that
        # has not been mapped to a node in the new tree sequence will be removed.
        mutation_id_map = [-1 for _ in range(self.num_mutations)]
        num_output_mutations = 0

        for site in self.ts.sites():
            num_output_site_mutations = 0
            for mut in site.mutations:
                mapped_node = self.mutation_node_map[mut.id]
                mapped_parent = -1
                if mut.parent != -1:
                    mapped_parent = mutation_id_map[mut.parent]
                if mapped_node != -1:
                    mutation_id_map[mut.id] = num_output_mutations
                    num_output_mutations += 1
                    num_output_site_mutations += 1
            output_site = True
            if self.filter_sites and num_output_site_mutations == 0:
                output_site = False

            if output_site:
                for mut in site.mutations:
                    if mutation_id_map[mut.id] != -1:
                        mapped_parent = -1
                        if mut.parent != -1:
                            mapped_parent = mutation_id_map[mut.parent]
                        self.tables.mutations.append(
                            mut.replace(
                                site=len(self.tables.sites),
                                node=self.mutation_node_map[mut.id],
                                parent=mapped_parent,
                            )
                        )
                self.tables.sites.append(site)

    def finalise_references(self):
        input_populations = self.ts.tables.populations
        population_id_map = np.arange(len(input_populations) + 1, dtype=np.int32)
        # Trick to ensure the null population gets mapped to itself.
        population_id_map[-1] = -1
        input_individuals = self.ts.tables.individuals
        individual_id_map = np.arange(len(input_individuals) + 1, dtype=np.int32)
        # Trick to ensure the null individual gets mapped to itself.
        individual_id_map[-1] = -1

        population_ref_count = np.ones(len(input_populations), dtype=int)
        if self.filter_populations:
            population_ref_count[:] = 0
            population_id_map[:] = -1
        individual_ref_count = np.ones(len(input_individuals), dtype=int)
        if self.filter_individuals:
            individual_ref_count[:] = 0
            individual_id_map[:] = -1

        for node in self.tables.nodes:
            if self.filter_populations and node.population != tskit.NULL:
                population_ref_count[node.population] += 1
            if self.filter_individuals and node.individual != tskit.NULL:
                individual_ref_count[node.individual] += 1

        for input_id, count in enumerate(population_ref_count):
            if count > 0:
                row = input_populations[input_id]
                output_id = self.tables.populations.append(row)
                population_id_map[input_id] = output_id
        for input_id, count in enumerate(individual_ref_count):
            if count > 0:
                row = input_individuals[input_id]
                output_id = self.tables.individuals.append(row)
                individual_id_map[input_id] = output_id

        # Remap the population ID references for nodes.
        nodes = self.tables.nodes
        nodes.set_columns(
            flags=nodes.flags,
            time=nodes.time,
            metadata=nodes.metadata,
            metadata_offset=nodes.metadata_offset,
            individual=individual_id_map[nodes.individual],
            population=population_id_map[nodes.population],
        )

        # Remap the parent ids of individuals
        individuals_copy = self.tables.individuals.copy()
        self.tables.individuals.clear()
        for row in individuals_copy:
            mapped_parents = []
            for p in row.parents:
                if p == -1:
                    mapped_parents.append(-1)
                else:
                    mapped_parents.append(individual_id_map[p])
            self.tables.individuals.append(row.replace(parents=mapped_parents))

        # We don't support migrations for now. We'll need to remap these as well.
        assert self.ts.num_migrations == 0

    def insert_input_roots(self):
        youngest_root_time = np.inf
        for input_id in range(len(self.node_id_map)):
            x = self.A_head[input_id]
            if x is not None:
                output_id = self.node_id_map[input_id]
                if output_id == -1:
                    output_id = self.record_node(input_id)
                while x is not None:
                    if x.node != output_id:
                        self.record_edge(x.left, x.right, output_id, x.node)
                        self.map_mutations(x.left, x.right, input_id, output_id)
                    x = x.next
                self.flush_edges()
                root_time = self.tables.nodes.time[output_id]
                if root_time < youngest_root_time:
                    youngest_root_time = root_time
        # We have to sort the edge table from the point where the edges
        # for the youngest root would be inserted.
        # Note: it would be nicer to do the sort here, but we have to
        # wait until the finalise_references method has been called to
        # make sure all the populations etc have been setup.
        node_time = self.tables.nodes.time
        edge_parent = self.tables.edges.parent
        offset = 0
        while (
            offset < len(self.tables.edges)
            and node_time[edge_parent[offset]] < youngest_root_time
        ):
            offset += 1
        self.sort_offset = offset

    def simplify(self):
        if self.ts.num_edges > 0:
            all_edges = list(self.ts.edges())
            edges = all_edges[:1]
            for e in all_edges[1:]:
                if e.parent != edges[0].parent:
                    self.process_parent_edges(edges)
                    edges = []
                edges.append(e)
            self.process_parent_edges(edges)
        if self.keep_input_roots:
            self.insert_input_roots()
        self.finalise_sites()
        self.finalise_references()
        if self.sort_offset != -1:
            self.tables.sort(edge_start=self.sort_offset)
        ts = self.tables.tree_sequence()
        return ts, self.node_id_map

    def check_state(self):
        # print("CHECK_STATE")
        all_ancestry = []
        num_nodes = len(self.A_head)
        for j in range(num_nodes):
            head = self.A_head[j]
            tail = self.A_tail[j]
            if head is None:
                assert tail is None
            else:
                x = head
                while x.next is not None:
                    assert x.right <= x.next.left
                    x = x.next
                assert x == tail
                x = head
                while x is not None:
                    assert x.left < x.right
                    all_ancestry.append(portion.openclosed(x.left, x.right))
                    if x.next is not None:
                        assert x.right <= x.next.left
                        # We should also not have any squashable segments.
                        if x.right == x.next.left:
                            assert x.node != x.next.node
                    x = x.next
        # Make sure we haven't lost ancestry.
        if len(all_ancestry) > 0:
            union = all_ancestry[0]
            for interval in all_ancestry[1:]:
                union = union.union(interval)
            assert union.atomic
            assert union.lower == 0
            assert union.upper == self.sequence_length


class AncestorMap:
    """
    Simplifies a tree sequence to show relationships between
    samples and a designated set of ancestors.
    """

    def __init__(self, ts, sample, ancestors):
        self.ts = ts
        self.samples = set(sample)
        assert (self.samples).issubset(set(range(0, ts.num_nodes)))
        self.ancestors = set(ancestors)
        assert (self.ancestors).issubset(set(range(0, ts.num_nodes)))
        self.table = tskit.EdgeTable()
        self.sequence_length = ts.sequence_length
        self.A_head = [None for _ in range(ts.num_nodes)]
        self.A_tail = [None for _ in range(ts.num_nodes)]
        for sample_id in sample:
            self.add_ancestry(0, self.sequence_length, sample_id, sample_id)
        self.edge_buffer = {}
        self.oldest_ancestor_time = max(ts.nodes_time[u] for u in ancestors)
        self.oldest_sample_time = max(ts.nodes_time[u] for u in sample)
        self.oldest_node_time = max(self.oldest_ancestor_time, self.oldest_sample_time)

    def link_ancestors(self):
        if self.ts.num_edges > 0:
            all_edges = list(self.ts.edges())
            edges = all_edges[:1]
            for e in all_edges[1:]:
                if self.ts.tables.nodes.time[e.parent] > self.oldest_node_time:
                    break
                if e.parent != edges[0].parent:
                    self.process_parent_edges(edges)
                    edges = []
                edges.append(e)
            self.process_parent_edges(edges)
        return self.table

    def process_parent_edges(self, edges):
        """
        Process all of the edges for a given parent.
        """
        assert len({e.parent for e in edges}) == 1
        parent = edges[0].parent
        S = []
        for edge in edges:
            x = self.A_head[edge.child]
            while x is not None:
                if x.right > edge.left and edge.right > x.left:
                    y = Segment(
                        max(x.left, edge.left), min(x.right, edge.right), x.node
                    )
                    S.append(y)
                x = x.next
        self.merge_labeled_ancestors(S, parent)
        self.check_state()

    def merge_labeled_ancestors(self, S, input_id):
        """
        All ancestry segments in S come together into a new parent.
        The new parent must be assigned and any overlapping segments coalesced.
        """
        is_sample = input_id in self.samples
        if is_sample:
            # Free up the existing ancestry mapping.
            x = self.A_tail[input_id]
            assert x.left == 0 and x.right == self.sequence_length
            self.A_tail[input_id] = None
            self.A_head[input_id] = None

        is_ancestor = input_id in self.ancestors
        prev_right = 0
        for left, right, X in overlapping_segments(S):
            if is_ancestor or is_sample:
                for x in X:
                    ancestry_node = x.node
                    self.record_edge(left, right, input_id, ancestry_node)
                self.add_ancestry(left, right, input_id, input_id)

                if is_sample and left != prev_right:
                    # Fill in any gaps in the ancestry for the sample.
                    self.add_ancestry(prev_right, left, input_id, input_id)

            else:
                for x in X:
                    ancestry_node = x.node
                    # Add sample ancestry for the currently-processed segment set.
                    self.add_ancestry(left, right, ancestry_node, input_id)
            prev_right = right

        if is_sample and prev_right != self.sequence_length:
            # If a trailing gap exists in the sample ancestry, fill it in.
            self.add_ancestry(prev_right, self.sequence_length, input_id, input_id)
        if input_id != -1:
            self.flush_edges()

    def record_edge(self, left, right, parent, child):
        """
        Adds an edge to the output list.
        """
        if child not in self.edge_buffer:
            self.edge_buffer[child] = [tskit.Edge(left, right, parent, child)]
        else:
            last = self.edge_buffer[child][-1]
            if last.right == left:
                last.right = right
            else:
                self.edge_buffer[child].append(tskit.Edge(left, right, parent, child))

    def add_ancestry(self, left, right, node, current_node):
        tail = self.A_tail[current_node]
        if tail is None:
            x = Segment(left, right, node)
            self.A_head[current_node] = x
            self.A_tail[current_node] = x
        else:
            if tail.right == left and tail.node == node:
                tail.right = right
            else:
                x = Segment(left, right, node)
                tail.next = x
                self.A_tail[current_node] = x

    def flush_edges(self):
        """
        Flush the edges to the output table after sorting and squashing
        any redundant records.
        """
        num_edges = 0
        for child in sorted(self.edge_buffer.keys()):
            for edge in self.edge_buffer[child]:
                self.table.append(edge)
                num_edges += 1
        self.edge_buffer.clear()
        return num_edges

    def check_state(self):
        num_nodes = len(self.A_head)
        for j in range(num_nodes):
            head = self.A_head[j]
            tail = self.A_tail[j]
            if head is None:
                assert tail is None
            else:
                x = head
                while x.next is not None:
                    x = x.next
                assert x == tail
                x = head.next
                while x is not None:
                    assert x.left < x.right
                    if x.next is not None:
                        if self.ancestors is None:
                            assert x.right <= x.next.left
                        # We should also not have any squashable segments.
                        if x.right == x.next.left:
                            assert x.node != x.next.node
                    x = x.next

    def print_state(self):
        print(".................")
        print("Ancestors: ")
        num_nodes = len(self.A_tail)
        for j in range(num_nodes):
            print("\t", j, "->", end="")
            x = self.A_head[j]
            while x is not None:
                print(f"({x.left}-{x.right}->{x.node})", end="")
                x = x.next
            print()
        print("Output:")
        print(self.table)
        self.check_state()


if __name__ == "__main__":
    # Simple CLI for running simplifier/ancestor mapping above.
    class_to_implement = sys.argv[1]
    assert class_to_implement == "Simplifier" or class_to_implement == "AncestorMap"
    ts = tskit.load(sys.argv[2])

    if class_to_implement == "Simplifier":
        samples = list(map(int, sys.argv[3:]))

        print("When keep_unary = True:")
        s = Simplifier(ts, samples, keep_unary=True)
        # s.print_state()
        tss, _ = s.simplify()
        tables = tss.dump_tables()
        print(tables.nodes)
        print(tables.edges)
        print(tables.sites)
        print(tables.mutations)

        print("\nWhen keep_unary = False")
        s = Simplifier(ts, samples, keep_unary=False)
        # s.print_state()
        tss, _ = s.simplify()
        tables = tss.dump_tables()
        print(tables.nodes)
        print(tables.edges)
        print(tables.sites)
        print(tables.mutations)

        print("\nWhen keep_unary_in_individuals = True")
        s = Simplifier(ts, samples, keep_unary_in_individuals=True)
        # s.print_state()
        tss, _ = s.simplify()
        tables = tss.dump_tables()
        print(tables.nodes)
        print(tables.edges)
        print(tables.sites)
        print(tables.mutations)

    elif class_to_implement == "AncestorMap":
        samples = sys.argv[3]
        samples = samples.split(",")
        samples = list(map(int, samples))

        ancestors = sys.argv[4]
        ancestors = ancestors.split(",")
        ancestors = list(map(int, ancestors))

        s = AncestorMap(ts, samples, ancestors)
        tss = s.link_ancestors()
        # tables = tss.dump_tables()
        # print(tables.nodes)
        print(tss)
