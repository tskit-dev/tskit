# MIT License
#
# Copyright (c) 2018-2019 Tskit Developers
# Copyright (C) 2017 University of Oxford
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
A collection of utilities to edit and construct tree sequences.
"""
import collections
import json
import random
import string

import numpy as np

import tskit
import tskit.provenance as provenance


def random_bytes(max_length):
    """
    Returns a random bytearray of the specified maximum length.
    """
    length = random.randint(0, max_length)
    return bytearray(random.randint(0, 255) for _ in range(length))


def random_strings(max_length):
    """
    Returns a random bytearray of the specified maximum length.
    """
    length = random.randint(0, max_length)
    return "".join(random.choice(string.printable) for _ in range(length))


def add_provenance(provenance_table, method_name):
    d = provenance.get_provenance_dict({"command": f"tsutil.{method_name}"})
    provenance_table.add_row(json.dumps(d))


def subsample_sites(ts, num_sites):
    """
    Returns a copy of the specified tree sequence with a random subsample of the
    specified number of sites.
    """
    t = ts.dump_tables()
    t.sites.reset()
    t.mutations.reset()
    sites_to_keep = set(random.sample(list(range(ts.num_sites)), num_sites))
    for site in ts.sites():
        if site.id in sites_to_keep:
            site_id = len(t.sites)
            t.sites.add_row(
                position=site.position, ancestral_state=site.ancestral_state
            )
            for mutation in site.mutations:
                t.mutations.add_row(
                    site=site_id,
                    derived_state=mutation.derived_state,
                    node=mutation.node,
                    time=mutation.time,
                    parent=mutation.parent,
                )
    add_provenance(t.provenances, "subsample_sites")
    return t.tree_sequence()


def decapitate(ts, num_edges):
    """
    Returns a copy of the specified tree sequence in which the specified number of
    edges have been retained.
    """
    t = ts.dump_tables()
    t.edges.set_columns(
        left=t.edges.left[:num_edges],
        right=t.edges.right[:num_edges],
        parent=t.edges.parent[:num_edges],
        child=t.edges.child[:num_edges],
    )
    add_provenance(t.provenances, "decapitate")
    # Simplify to get rid of any mutations that are lying around above roots.
    t.simplify()
    return t.tree_sequence()


def insert_branch_mutations(ts, mutations_per_branch=1):
    """
    Returns a copy of the specified tree sequence with a mutation on every branch
    in every tree.
    """
    tables = ts.dump_tables()
    tables.sites.clear()
    tables.mutations.clear()
    for tree in ts.trees():
        site = tables.sites.add_row(position=tree.interval[0], ancestral_state="0")
        for root in tree.roots:
            state = {tskit.NULL: 0}
            mutation = {tskit.NULL: -1}
            stack = [root]
            while len(stack) > 0:
                u = stack.pop()
                stack.extend(tree.children(u))
                v = tree.parent(u)
                state[u] = state[v]
                parent = mutation[v]
                for _ in range(mutations_per_branch):
                    state[u] = (state[u] + 1) % 2
                    metadata = f"{len(tables.mutations)}".encode()
                    mutation[u] = tables.mutations.add_row(
                        site=site,
                        node=u,
                        derived_state=str(state[u]),
                        parent=parent,
                        metadata=metadata,
                    )
                    parent = mutation[u]
    add_provenance(tables.provenances, "insert_branch_mutations")
    return tables.tree_sequence()


def insert_branch_sites(ts):
    """
    Returns a copy of the specified tree sequence with a site on every branch
    of every tree.
    """
    tables = ts.dump_tables()
    tables.sites.clear()
    tables.mutations.clear()
    for tree in ts.trees():
        left, right = tree.interval
        delta = (right - left) / len(list(tree.nodes()))
        x = left
        for u in tree.nodes():
            if tree.parent(u) != tskit.NULL:
                site = tables.sites.add_row(position=x, ancestral_state="0")
                tables.mutations.add_row(site=site, node=u, derived_state="1")
                x += delta
    add_provenance(tables.provenances, "insert_branch_sites")
    return tables.tree_sequence()


def insert_multichar_mutations(ts, seed=1, max_len=10):
    """
    Returns a copy of the specified tree sequence with multiple chararacter
    mutations on a randomly chosen branch in every tree.
    """
    rng = random.Random(seed)
    letters = ["A", "C", "T", "G"]
    tables = ts.dump_tables()
    tables.sites.clear()
    tables.mutations.clear()
    for tree in ts.trees():
        ancestral_state = rng.choice(letters) * rng.randint(0, max_len)
        site = tables.sites.add_row(
            position=tree.interval[0], ancestral_state=ancestral_state
        )
        nodes = list(tree.nodes())
        nodes.remove(tree.root)
        u = rng.choice(nodes)
        derived_state = ancestral_state
        while ancestral_state == derived_state:
            derived_state = rng.choice(letters) * rng.randint(0, max_len)
        tables.mutations.add_row(site=site, node=u, derived_state=derived_state)
    add_provenance(tables.provenances, "insert_multichar_mutations")
    return tables.tree_sequence()


def insert_random_ploidy_individuals(ts, max_ploidy=5, max_dimension=3, seed=1):
    """
    Takes random contiguous subsets of the samples an assigns them to individuals.
    Also creates random locations in variable dimensions in the unit interval.
    """
    rng = random.Random(seed)
    samples = np.array(ts.samples(), dtype=int)
    j = 0
    tables = ts.dump_tables()
    tables.individuals.clear()
    individual = tables.nodes.individual[:]
    individual[:] = tskit.NULL
    while j < len(samples):
        ploidy = rng.randint(0, max_ploidy)
        nodes = samples[j : min(j + ploidy, len(samples))]
        dimension = rng.randint(0, max_dimension)
        location = [rng.random() for _ in range(dimension)]
        ind_id = tables.individuals.add_row(location=location)
        individual[nodes] = ind_id
        j += ploidy
    tables.nodes.individual = individual
    return tables.tree_sequence()


def insert_individuals(ts, samples=None, ploidy=1):
    """
    Inserts individuals into the tree sequence using the specified list
    of samples (or all samples if None) with the specified ploidy by combining
    ploidy-sized chunks of the list.
    """
    if samples is None:
        samples = ts.samples()
    if len(samples) % ploidy != 0:
        raise ValueError("number of samples must be divisible by ploidy")
    tables = ts.dump_tables()
    tables.individuals.clear()
    individual = tables.nodes.individual[:]
    individual[:] = tskit.NULL
    j = 0
    while j < len(samples):
        nodes = samples[j : j + ploidy]
        ind_id = tables.individuals.add_row()
        individual[nodes] = ind_id
        j += ploidy
    tables.nodes.individual = individual
    return tables.tree_sequence()


def permute_nodes(ts, node_map):
    """
    Returns a copy of the specified tree sequence such that the nodes are
    permuted according to the specified map.
    """
    tables = ts.dump_tables()
    tables.nodes.clear()
    tables.edges.clear()
    tables.mutations.clear()
    # Mapping from nodes in the new tree sequence back to nodes in the original
    reverse_map = [0 for _ in node_map]
    for j in range(ts.num_nodes):
        reverse_map[node_map[j]] = j
    old_nodes = list(ts.nodes())
    for j in range(ts.num_nodes):
        old_node = old_nodes[reverse_map[j]]
        tables.nodes.add_row(
            flags=old_node.flags,
            metadata=old_node.metadata,
            population=old_node.population,
            time=old_node.time,
        )
    for edge in ts.edges():
        tables.edges.add_row(
            left=edge.left,
            right=edge.right,
            parent=node_map[edge.parent],
            child=node_map[edge.child],
        )
    for site in ts.sites():
        for mutation in site.mutations:
            tables.mutations.add_row(
                site=site.id,
                time=mutation.time,
                derived_state=mutation.derived_state,
                node=node_map[mutation.node],
                metadata=mutation.metadata,
            )
    tables.sort()
    add_provenance(tables.provenances, "permute_nodes")
    return tables.tree_sequence()


def insert_redundant_breakpoints(ts):
    """
    Builds a new tree sequence containing redundant breakpoints.
    """
    tables = ts.dump_tables()
    tables.edges.reset()
    for r in ts.edges():
        x = r.left + (r.right - r.left) / 2
        tables.edges.add_row(left=r.left, right=x, child=r.child, parent=r.parent)
        tables.edges.add_row(left=x, right=r.right, child=r.child, parent=r.parent)
    add_provenance(tables.provenances, "insert_redundant_breakpoints")
    new_ts = tables.tree_sequence()
    assert new_ts.num_edges == 2 * ts.num_edges
    return new_ts


def single_childify(ts):
    """
    Builds a new equivalent tree sequence which contains an extra node in the
    middle of all existing branches.
    """
    tables = ts.dump_tables()

    mutations_above_node = collections.defaultdict(list)
    for mut in tables.mutations:
        mutations_above_node[mut.node].append(mut)

    mutations_on_edge = collections.defaultdict(list)
    for edge_idx, edge in enumerate(tables.edges):
        for mut in mutations_above_node[edge.child]:
            if edge.left <= tables.sites[mut.site].position < edge.right:
                mutations_on_edge[edge_idx].append(mut)

    time = tables.nodes.time[:]
    tables.edges.reset()
    tables.mutations.reset()
    for edge in ts.edges():
        # Insert a new node in between the parent and child.
        t = time[edge.child] + (time[edge.parent] - time[edge.child]) / 2
        u = tables.nodes.add_row(time=t)
        tables.edges.add_row(
            left=edge.left, right=edge.right, parent=u, child=edge.child
        )
        tables.edges.add_row(
            left=edge.left, right=edge.right, parent=edge.parent, child=u
        )
        for mut in mutations_on_edge[edge.id]:
            if mut.time < t:
                tables.mutations.add_row(
                    mut.site,
                    mut.node,
                    mut.time,
                    mut.derived_state,
                    mut.parent,
                    mut.metadata,
                )
            else:
                tables.mutations.add_row(
                    mut.site, u, mut.derived_state, mut.parent, mut.metadata, mut.time
                )
    tables.sort()
    add_provenance(tables.provenances, "insert_redundant_breakpoints")
    return tables.tree_sequence()


def add_random_metadata(ts, seed=1, max_length=10):
    """
    Returns a copy of the specified tree sequence with random metadata assigned
    to the nodes, sites and mutations.
    """
    tables = ts.dump_tables()
    np.random.seed(seed)

    length = np.random.randint(0, max_length, ts.num_nodes)
    offset = np.cumsum(np.hstack(([0], length)), dtype=np.uint32)
    # Older versions of numpy didn't have a dtype argument for randint, so
    # must use astype instead.
    metadata = np.random.randint(-127, 127, offset[-1]).astype(np.int8)
    nodes = tables.nodes
    nodes.set_columns(
        flags=nodes.flags,
        population=nodes.population,
        time=nodes.time,
        metadata_offset=offset,
        metadata=metadata,
        individual=nodes.individual,
    )

    length = np.random.randint(0, max_length, ts.num_sites)
    offset = np.cumsum(np.hstack(([0], length)), dtype=np.uint32)
    metadata = np.random.randint(-127, 127, offset[-1]).astype(np.int8)
    sites = tables.sites
    sites.set_columns(
        position=sites.position,
        ancestral_state=sites.ancestral_state,
        ancestral_state_offset=sites.ancestral_state_offset,
        metadata_offset=offset,
        metadata=metadata,
    )

    length = np.random.randint(0, max_length, ts.num_mutations)
    offset = np.cumsum(np.hstack(([0], length)), dtype=np.uint32)
    metadata = np.random.randint(-127, 127, offset[-1]).astype(np.int8)
    mutations = tables.mutations
    mutations.set_columns(
        site=mutations.site,
        node=mutations.node,
        time=mutations.time,
        parent=mutations.parent,
        derived_state=mutations.derived_state,
        derived_state_offset=mutations.derived_state_offset,
        metadata_offset=offset,
        metadata=metadata,
    )

    length = np.random.randint(0, max_length, ts.num_individuals)
    offset = np.cumsum(np.hstack(([0], length)), dtype=np.uint32)
    metadata = np.random.randint(-127, 127, offset[-1]).astype(np.int8)
    individuals = tables.individuals
    individuals.set_columns(
        flags=individuals.flags,
        location=individuals.location,
        location_offset=individuals.location_offset,
        metadata_offset=offset,
        metadata=metadata,
    )

    length = np.random.randint(0, max_length, ts.num_populations)
    offset = np.cumsum(np.hstack(([0], length)), dtype=np.uint32)
    metadata = np.random.randint(-127, 127, offset[-1]).astype(np.int8)
    populations = tables.populations
    populations.set_columns(metadata_offset=offset, metadata=metadata)

    add_provenance(tables.provenances, "add_random_metadata")
    ts = tables.tree_sequence()
    return ts


def jiggle_samples(ts):
    """
    Returns a copy of the specified tree sequence with the sample nodes switched
    around. The first n / 2 existing samples become non samples, and the last
    n / 2 node become samples.
    """
    tables = ts.dump_tables()
    nodes = tables.nodes
    flags = nodes.flags
    oldest_parent = tables.edges.parent[-1]
    n = ts.sample_size
    flags[: n // 2] = 0
    flags[oldest_parent - n // 2 : oldest_parent] = 1
    nodes.set_columns(flags, nodes.time)
    add_provenance(tables.provenances, "jiggle_samples")
    return tables.tree_sequence()


def generate_site_mutations(
    tree, position, mu, site_table, mutation_table, multiple_per_node=True
):
    """
    Generates mutations for the site at the specified position on the specified
    tree. Mutations happen at rate mu along each branch. The site and mutation
    information are recorded in the specified tables.  Note that this records
    more than one mutation per edge.
    """
    assert tree.interval[0] <= position < tree.interval[1]
    states = ["A", "C", "G", "T"]
    ancestral_state = random.choice(states)
    site_table.add_row(position, ancestral_state)
    site = site_table.num_rows - 1
    for root in tree.roots:
        stack = [(root, ancestral_state, tskit.NULL)]
        while len(stack) != 0:
            u, state, parent = stack.pop()
            if u != root:
                branch_length = tree.branch_length(u)
                x = random.expovariate(mu)
                new_state = state
                while x < branch_length:
                    new_state = random.choice([s for s in states if s != state])
                    if multiple_per_node and (state != new_state):
                        mutation_table.add_row(site, u, new_state, parent)
                        parent = mutation_table.num_rows - 1
                        state = new_state
                    x += random.expovariate(mu)
                else:
                    if (not multiple_per_node) and (state != new_state):
                        mutation_table.add_row(site, u, new_state, parent)
                        parent = mutation_table.num_rows - 1
                        state = new_state
            stack.extend(reversed([(v, state, parent) for v in tree.children(u)]))


def jukes_cantor(ts, num_sites, mu, multiple_per_node=True, seed=None):
    """
    Returns a copy of the specified tree sequence with Jukes-Cantor mutations
    applied at the specfied rate at the specifed number of sites. Site positions
    are chosen uniformly.
    """
    random.seed(seed)
    positions = [ts.sequence_length * random.random() for _ in range(num_sites)]
    positions.sort()
    tables = ts.dump_tables()
    tables.sites.clear()
    tables.mutations.clear()
    trees = ts.trees()
    t = next(trees)
    for position in positions:
        while position >= t.interval[1]:
            t = next(trees)
        generate_site_mutations(
            t,
            position,
            mu,
            tables.sites,
            tables.mutations,
            multiple_per_node=multiple_per_node,
        )
    add_provenance(tables.provenances, "jukes_cantor")
    new_ts = tables.tree_sequence()
    return new_ts


def caterpillar_tree(n, num_sites=0, num_mutations=1):
    """
    Returns caterpillar tree with n samples. For each of the sites and
    path of at most n - 2 mutations are put down along the internal
    nodes. Each site gets exactly the same set of mutations.
    """
    if num_sites > 0 and num_mutations > n - 2:
        raise ValueError("At most n - 2 mutations allowed")
    tables = tskit.TableCollection(1)
    for _ in range(n):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    last_node = 0
    # Add the internal nodes
    for j in range(n - 1):
        u = tables.nodes.add_row(time=j + 1)
        tables.edges.add_row(0, tables.sequence_length, u, last_node)
        tables.edges.add_row(0, tables.sequence_length, u, j + 1)
        last_node = u
    for j in range(num_sites):
        tables.sites.add_row(position=(j + 1) / n, ancestral_state="0")
        node = 2 * n - 3
        state = 0
        for _ in range(num_mutations):
            state = (state + 1) % 2
            tables.mutations.add_row(site=j, derived_state=str(state), node=node)
            node -= 1

    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()
    return tables.tree_sequence()


def compute_mutation_parent(ts):
    """
    Compute the `parent` column of a MutationTable. Correct computation uses
    topological information in the nodes and edges, as well as the fact that
    each mutation must be listed after the mutation on whose background it
    occurred (i.e., its parent).

    :param TreeSequence ts: The tree sequence to compute for.  Need not
        have a valid mutation parent column.
    """
    mutation_parent = np.zeros(ts.num_mutations, dtype=np.int32) - 1
    # Maps nodes to the bottom mutation on each branch
    bottom_mutation = np.zeros(ts.num_nodes, dtype=np.int32) - 1
    for tree in ts.trees():
        for site in tree.sites():
            # Go forward through the mutations creating a mapping from the
            # mutations to the nodes. If we see more than one mutation
            # at a node, then these must be parents since we're assuming
            # they are in order.
            for mutation in site.mutations:
                if bottom_mutation[mutation.node] != tskit.NULL:
                    mutation_parent[mutation.id] = bottom_mutation[mutation.node]
                bottom_mutation[mutation.node] = mutation.id
            # There's no point in checking the first mutation, since this cannot
            # have a parent.
            for mutation in site.mutations[1:]:
                if mutation_parent[mutation.id] == tskit.NULL:
                    v = tree.parent(mutation.node)
                    # Traverse upwards until we find a another mutation or root.
                    while v != tskit.NULL and bottom_mutation[v] == tskit.NULL:
                        v = tree.parent(v)
                    if v != tskit.NULL:
                        mutation_parent[mutation.id] = bottom_mutation[v]
            # Reset the maps for the next site.
            for mutation in site.mutations:
                bottom_mutation[mutation.node] = tskit.NULL
            assert np.all(bottom_mutation == -1)
    return mutation_parent


def py_subset(tables, nodes, record_provenance=True):
    """
    Naive implementation of the TableCollection.subset method using the Python API.
    """
    if np.any(nodes > tables.nodes.num_rows) or np.any(nodes < 0):
        raise ValueError("Nodes out of bounds.")
    full = tables.copy()
    tables.clear()
    # mapping from old to new ids
    node_map = {}
    ind_map = {tskit.NULL: tskit.NULL}
    pop_map = {tskit.NULL: tskit.NULL}
    for old_id in nodes:
        node = full.nodes[old_id]
        if node.individual not in ind_map and node.individual != tskit.NULL:
            ind = full.individuals[node.individual]
            new_ind_id = tables.individuals.add_row(
                ind.flags, ind.location, ind.metadata
            )
            ind_map[node.individual] = new_ind_id
        if node.population not in pop_map and node.population != tskit.NULL:
            pop = full.populations[node.population]
            new_pop_id = tables.populations.add_row(pop.metadata)
            pop_map[node.population] = new_pop_id
        new_id = tables.nodes.add_row(
            node.flags,
            node.time,
            pop_map[node.population],
            ind_map[node.individual],
            node.metadata,
        )
        node_map[old_id] = new_id
    for edge in full.edges:
        if edge.child in nodes and edge.parent in nodes:
            tables.edges.add_row(
                edge.left,
                edge.right,
                node_map[edge.parent],
                node_map[edge.child],
                edge.metadata,
            )
    if full.migrations.num_rows > 0:
        raise ValueError("Migrations are currently not supported in this operation.")
    site_map = {}
    mutation_map = {tskit.NULL: tskit.NULL}
    for i, mut in enumerate(full.mutations):
        if mut.node in nodes:
            if mut.site not in site_map:
                site = full.sites[mut.site]
                new_site = tables.sites.add_row(
                    site.position, site.ancestral_state, site.metadata
                )
                site_map[mut.site] = new_site
            new_mut = tables.mutations.add_row(
                site_map[mut.site],
                node_map[mut.node],
                mut.derived_state,
                mutation_map.get(mut.parent, tskit.NULL),
                mut.metadata,
            )
            mutation_map[i] = new_mut


def py_union(tables, other, nodes, record_provenance=True, add_populations=True):
    """
    Python implementation of TableCollection.union().
    """
    # mappings of id in other to new id in tables
    # the +1 is to take care of mapping tskit.NULL(-1) to tskit.NULL
    pop_map = [tskit.NULL for _ in range(other.populations.num_rows + 1)]
    ind_map = [tskit.NULL for _ in range(other.individuals.num_rows + 1)]
    node_map = [tskit.NULL for _ in range(other.nodes.num_rows + 1)]
    site_map = [tskit.NULL for _ in range(other.sites.num_rows + 1)]
    mut_map = [tskit.NULL for _ in range(other.mutations.num_rows + 1)]
    for other_id, node in enumerate(other.nodes):
        if nodes[other_id] != tskit.NULL:
            node_map[other_id] = nodes[other_id]
        else:
            if ind_map[node.individual] == tskit.NULL and node.individual != tskit.NULL:
                ind = other.individuals[node.individual]
                ind_id = tables.individuals.add_row(
                    flags=ind.flags, location=ind.location, metadata=ind.metadata
                )
                ind_map[node.individual] = ind_id
            if pop_map[node.population] == tskit.NULL and node.population != tskit.NULL:
                if not add_populations:
                    pop_map[node.population] = node.population
                else:
                    pop = other.populations[node.population]
                    pop_id = tables.populations.add_row(metadata=pop.metadata)
                    pop_map[node.population] = pop_id
            node_id = tables.nodes.add_row(
                time=node.time,
                population=pop_map[node.population],
                individual=ind_map[node.individual],
                metadata=node.metadata,
                flags=node.flags,
            )
            node_map[other_id] = node_id
    for edge in other.edges:
        if (nodes[edge.parent] == tskit.NULL) or (nodes[edge.child] == tskit.NULL):
            # can't do this right not because of sorting of mutations
            if (nodes[edge.parent] == tskit.NULL) and (nodes[edge.child] != tskit.NULL):
                raise ValueError("Cannot graft nodes above existing nodes.")
            tables.edges.add_row(
                left=edge.left,
                right=edge.right,
                parent=node_map[edge.parent],
                child=node_map[edge.child],
                metadata=edge.metadata,
            )
    for other_id, mut in enumerate(other.mutations):
        if nodes[mut.node] == tskit.NULL:
            # add site: may already be in tables, but we deduplicate
            if site_map[mut.site] == tskit.NULL:
                site = other.sites[mut.site]
                site_id = tables.sites.add_row(
                    position=site.position,
                    ancestral_state=site.ancestral_state,
                    metadata=site.metadata,
                )
                site_map[mut.site] = site_id
            mut_id = tables.mutations.add_row(
                site=site_map[mut.site],
                node=node_map[mut.node],
                derived_state=mut.derived_state,
                parent=tskit.NULL,
                metadata=mut.metadata,
            )
            mut_map[other_id] = mut_id
    # migration table
    # grafting provenance table
    if record_provenance:
        pass
    # sorting, deduplicating sites, and re-computing mutation parents
    tables.sort()
    tables.deduplicate_sites()
    tables.build_index()
    tables.compute_mutation_parents()


def compute_mutation_times(ts):
    """
    Compute the `time` column of a MutationTable in a TableCollection.
    Finds the set of mutations on an edge that share a site and spreads
    the times evenly over the edge.

    :param TreeSequence ts: The tree sequence to compute for.  Need not
        have a valid mutation time column.
    """
    tables = ts.dump_tables()
    mutations = tables.mutations

    mutations_above_node = collections.defaultdict(list)
    for mut_idx, mut in enumerate(mutations):
        mutations_above_node[mut.node].append((mut_idx, mut))

    mutations_at_site_on_edge = collections.defaultdict(list)
    for edge_idx, edge in enumerate(tables.edges):
        for mut_idx, mut in mutations_above_node[edge.child]:
            if edge.left <= tables.sites[mut.site].position < edge.right:
                mutations_at_site_on_edge[(mut.site, edge_idx)].append(mut_idx)

    edges = tables.edges
    nodes = tables.nodes
    times = np.full(len(mutations), -1, dtype=np.float64)
    for (_, edge_idx), edge_mutations in mutations_at_site_on_edge.items():
        start_time = nodes[edges[edge_idx].child].time
        end_time = nodes[edges[edge_idx].parent].time
        duration = end_time - start_time
        for i, mut_idx in enumerate(edge_mutations):
            times[mut_idx] = end_time - (
                duration * ((i + 1) / (len(edge_mutations) + 1))
            )

    # Mutations not on a edge (i.e. above a root) get given their node's time
    for i in range(len(mutations)):
        if times[i] == -1:
            times[i] = nodes[mutations[i].node].time
    tables.mutations.time = times
    tables.sort()
    return tables.mutations.time


def algorithm_T(ts):
    """
    Simple implementation of algorithm T from the PLOS paper, taking into
    account tree sequences with gaps and other complexities.
    """
    sequence_length = ts.sequence_length
    edges = list(ts.edges())
    M = len(edges)
    time = [ts.node(edge.parent).time for edge in edges]
    in_order = sorted(
        range(M),
        key=lambda j: (edges[j].left, time[j], edges[j].parent, edges[j].child),
    )
    out_order = sorted(
        range(M),
        key=lambda j: (edges[j].right, -time[j], -edges[j].parent, -edges[j].child),
    )
    j = 0
    k = 0
    left = 0
    parent = [-1 for _ in range(ts.num_nodes)]
    while j < M or left < sequence_length:
        while k < M and edges[out_order[k]].right == left:
            edge = edges[out_order[k]]
            parent[edge.child] = -1
            k += 1
        while j < M and edges[in_order[j]].left == left:
            edge = edges[in_order[j]]
            parent[edge.child] = edge.parent
            j += 1
        right = sequence_length
        if j < M:
            right = min(right, edges[in_order[j]].left)
        if k < M:
            right = min(right, edges[out_order[k]].right)
        yield (left, right), parent
        left = right


class SampleListTree:
    """
    Straightforward implementation of the quintuply linked tree for developing
    and testing the sample lists feature.

    NOTE: The interface is pretty awkward; it's not intended for anything other
    than testing.
    """

    def __init__(self, tree_sequence, tracked_samples=None):
        self.tree_sequence = tree_sequence
        num_nodes = tree_sequence.num_nodes
        # Quintuply linked tree.
        self.parent = [-1 for _ in range(num_nodes)]
        self.left_sib = [-1 for _ in range(num_nodes)]
        self.right_sib = [-1 for _ in range(num_nodes)]
        self.left_child = [-1 for _ in range(num_nodes)]
        self.right_child = [-1 for _ in range(num_nodes)]
        self.left_sample = [-1 for _ in range(num_nodes)]
        self.right_sample = [-1 for _ in range(num_nodes)]
        # This is too long, but it's convenient for printing.
        self.next_sample = [-1 for _ in range(num_nodes)]

        self.sample_index_map = [-1 for _ in range(num_nodes)]
        samples = tracked_samples
        if tracked_samples is None:
            samples = list(tree_sequence.samples())
        for j in range(len(samples)):
            u = samples[j]
            self.sample_index_map[u] = j
            self.left_sample[u] = j
            self.right_sample[u] = j

    def __str__(self):
        fmt = "{:<5}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}\n"
        s = fmt.format(
            "node",
            "parent",
            "lsib",
            "rsib",
            "lchild",
            "rchild",
            "nsamp",
            "lsamp",
            "rsamp",
        )
        for u in range(self.tree_sequence.num_nodes):
            s += fmt.format(
                u,
                self.parent[u],
                self.left_sib[u],
                self.right_sib[u],
                self.left_child[u],
                self.right_child[u],
                self.next_sample[u],
                self.left_sample[u],
                self.right_sample[u],
            )
        # Strip off trailing newline
        return s[:-1]

    def remove_edge(self, edge):
        p = edge.parent
        c = edge.child
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

    def insert_edge(self, edge):
        p = edge.parent
        c = edge.child
        assert self.parent[c] == -1, "contradictory edges"
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

    def update_sample_list(self, parent):
        # This can surely be done more efficiently and elegantly. We are iterating
        # up the tree and iterating over all the siblings of the nodes we visit,
        # rebuilding the links as we go. This results in visiting the same nodes
        # over again, which if we have nodes with many siblings will surely be
        # expensive. Another consequence of the current approach is that the
        # next pointer contains an arbitrary value for the rightmost sample of
        # every root. This should point to NULL ideally, but it's quite tricky
        # to do in practise. It's easier to have a slightly uglier iteration
        # over samples.
        #
        # In the future it would be good have a more efficient version of this
        # algorithm using next and prev pointers that we keep up to date at all
        # times, and which we use to patch the lists together more efficiently.
        u = parent
        while u != -1:
            sample_index = self.sample_index_map[u]
            if sample_index != -1:
                self.right_sample[u] = self.left_sample[u]
            else:
                self.right_sample[u] = -1
                self.left_sample[u] = -1
            v = self.left_child[u]
            while v != -1:
                if self.left_sample[v] != -1:
                    assert self.right_sample[v] != -1
                    if self.left_sample[u] == -1:
                        self.left_sample[u] = self.left_sample[v]
                        self.right_sample[u] = self.right_sample[v]
                    else:
                        self.next_sample[self.right_sample[u]] = self.left_sample[v]
                        self.right_sample[u] = self.right_sample[v]
                v = self.right_sib[v]
            u = self.parent[u]

    def sample_lists(self):
        """
        Iterate over the the trees in this tree sequence, yielding the (left, right)
        interval tuples. The tree state is maintained internally.

        See note above about the cruddiness of this interface.
        """
        ts = self.tree_sequence
        sequence_length = ts.sequence_length
        edges = list(ts.edges())
        M = len(edges)
        time = [ts.node(edge.parent).time for edge in edges]
        in_order = sorted(
            range(M),
            key=lambda j: (edges[j].left, time[j], edges[j].parent, edges[j].child),
        )
        out_order = sorted(
            range(M),
            key=lambda j: (edges[j].right, -time[j], -edges[j].parent, -edges[j].child),
        )
        j = 0
        k = 0
        left = 0

        while j < M or left < sequence_length:
            while k < M and edges[out_order[k]].right == left:
                edge = edges[out_order[k]]
                self.remove_edge(edge)
                self.update_sample_list(edge.parent)
                k += 1
            while j < M and edges[in_order[j]].left == left:
                edge = edges[in_order[j]]
                self.insert_edge(edge)
                self.update_sample_list(edge.parent)
                j += 1
            right = sequence_length
            if j < M:
                right = min(right, edges[in_order[j]].left)
            if k < M:
                right = min(right, edges[out_order[k]].right)
            yield left, right
            left = right


class RootThresholdTree:
    """
    Straightforward implementation of the quintuply linked tree for developing
    and testing the root_threshold feature.

    NOTE: The interface is pretty awkward; it's not intended for anything other
    than testing.
    """

    def __init__(self, tree_sequence, root_threshold=1):
        self.tree_sequence = tree_sequence
        self.root_threshold = root_threshold
        num_nodes = tree_sequence.num_nodes
        # Quintuply linked tree.
        self.parent = [-1 for _ in range(num_nodes)]
        self.left_sib = [-1 for _ in range(num_nodes)]
        self.right_sib = [-1 for _ in range(num_nodes)]
        self.left_child = [-1 for _ in range(num_nodes)]
        self.right_child = [-1 for _ in range(num_nodes)]
        self.num_samples = [0 for _ in range(num_nodes)]
        self.left_root = -1
        for u in tree_sequence.samples()[::-1]:
            self.num_samples[u] = 1
            if self.root_threshold == 1:
                self.add_root(u)

    def __str__(self):
        fmt = "{:<5}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}\n"
        s = f"roots = {self.roots()}\n"
        s += fmt.format("node", "parent", "lsib", "rsib", "lchild", "rchild", "nsamp")
        for u in range(self.tree_sequence.num_nodes):
            s += fmt.format(
                u,
                self.parent[u],
                self.left_sib[u],
                self.right_sib[u],
                self.left_child[u],
                self.right_child[u],
                self.num_samples[u],
            )
        # Strip off trailing newline
        return s[:-1]

    def is_root(self, u):
        return self.num_samples[u] >= self.root_threshold

    def roots(self):
        roots = []
        u = self.left_root
        while u != -1:
            roots.append(u)
            u = self.right_sib[u]
        return roots

    def add_root(self, root):
        if self.left_root != tskit.NULL:
            lroot = self.left_sib[self.left_root]
            if lroot != tskit.NULL:
                self.right_sib[lroot] = root
            self.left_sib[root] = lroot
            self.left_sib[self.left_root] = root
        self.right_sib[root] = self.left_root
        self.left_root = root

    def remove_root(self, root):
        lroot = self.left_sib[root]
        rroot = self.right_sib[root]
        self.left_root = tskit.NULL
        if lroot != tskit.NULL:
            self.right_sib[lroot] = rroot
            self.left_root = lroot
        if rroot != tskit.NULL:
            self.left_sib[rroot] = lroot
            self.left_root = rroot
        self.left_sib[root] = tskit.NULL
        self.right_sib[root] = tskit.NULL

    def remove_edge(self, edge):
        p = edge.parent
        c = edge.child
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

        u = edge.parent
        while u != -1:
            path_end = u
            path_end_was_root = self.is_root(u)
            self.num_samples[u] -= self.num_samples[c]
            u = self.parent[u]
        if path_end_was_root and not self.is_root(path_end):
            self.remove_root(path_end)
        if self.is_root(c):
            self.add_root(c)

    def insert_edge(self, edge):
        p = edge.parent
        c = edge.child
        assert self.parent[c] == -1, "contradictory edges"
        self.parent[c] = p
        u = self.right_child[p]
        lsib = self.left_sib[c]
        rsib = self.right_sib[c]
        if u == -1:
            self.left_child[p] = c
            self.left_sib[c] = -1
            self.right_sib[c] = -1
        else:
            self.right_sib[u] = c
            self.left_sib[c] = u
            self.right_sib[c] = -1
        self.right_child[p] = c

        u = edge.parent
        while u != -1:
            path_end = u
            path_end_was_root = self.is_root(u)
            self.num_samples[u] += self.num_samples[c]
            u = self.parent[u]

        if self.is_root(c):
            if path_end_was_root:
                # Remove c from root list.
                # Note: we don't use the remove_root function here because
                # it assumes that the node is at the end of a path
                self.left_root = tskit.NULL
                if lsib != tskit.NULL:
                    self.right_sib[lsib] = rsib
                    self.left_root = lsib
                if rsib != tskit.NULL:
                    self.left_sib[rsib] = lsib
                    self.left_root = rsib
            else:
                # Replace c with path_end in the root list
                if lsib != tskit.NULL:
                    self.right_sib[lsib] = path_end
                if rsib != tskit.NULL:
                    self.left_sib[rsib] = path_end
                self.left_sib[path_end] = lsib
                self.right_sib[path_end] = rsib
                self.left_root = path_end
        else:
            if self.is_root(path_end) and not path_end_was_root:
                self.add_root(path_end)

    def iterate(self):
        """
        Iterate over the the trees in this tree sequence, yielding the (left, right)
        interval tuples. The tree state is maintained internally.
        """
        ts = self.tree_sequence
        sequence_length = ts.sequence_length
        edges = list(ts.edges())
        M = len(edges)
        time = [ts.node(edge.parent).time for edge in edges]
        in_order = sorted(
            range(M),
            key=lambda j: (edges[j].left, time[j], edges[j].parent, edges[j].child),
        )
        out_order = sorted(
            range(M),
            key=lambda j: (edges[j].right, -time[j], -edges[j].parent, -edges[j].child),
        )
        j = 0
        k = 0
        left = 0

        while j < M or left < sequence_length:
            while k < M and edges[out_order[k]].right == left:
                edge = edges[out_order[k]]
                self.remove_edge(edge)
                k += 1
            while j < M and edges[in_order[j]].left == left:
                edge = edges[in_order[j]]
                self.insert_edge(edge)
                j += 1
            while self.left_sib[self.left_root] != tskit.NULL:
                self.left_root = self.left_sib[self.left_root]
            right = sequence_length
            if j < M:
                right = min(right, edges[in_order[j]].left)
            if k < M:
                right = min(right, edges[out_order[k]].right)
            yield left, right
            left = right


def mean_descendants(ts, reference_sets):
    """
    Returns the mean number of nodes from the specified reference sets
    where the node is ancestral to at least one of the reference nodes. Returns a
    ``(ts.num_nodes, len(reference_sets))`` dimensional numpy array.
    """
    # Check the inputs (could be done more efficiently here)
    all_reference_nodes = set()
    for reference_set in reference_sets:
        U = set(reference_set)
        if len(U) != len(reference_set):
            raise ValueError("Cannot have duplicate values within set")
        if len(all_reference_nodes & U) != 0:
            raise ValueError("Sample sets must be disjoint")
        all_reference_nodes |= U

    K = len(reference_sets)
    C = np.zeros((ts.num_nodes, K))
    parent = np.zeros(ts.num_nodes, dtype=int) - 1
    # The -1th element of ref_count is for all nodes in the reference set.
    ref_count = np.zeros((ts.num_nodes, K + 1), dtype=int)
    last_update = np.zeros(ts.num_nodes)
    total_span = np.zeros(ts.num_nodes)

    def update_counts(edge, left, sign):
        # Update the counts and statistics for a given node. Before we change the
        # node counts in the given direction, check to see if we need to update
        # statistics for that node. When a node count changes, we add the
        # accumulated statistic value for the span since that node was last updated.
        v = edge.parent
        while v != -1:
            if last_update[v] != left:
                if ref_count[v, K] > 0:
                    span = left - last_update[v]
                    C[v] += span * ref_count[v, :K]
                    total_span[v] += span
                last_update[v] = left
            ref_count[v] += sign * ref_count[edge.child]
            v = parent[v]

    # Set the intitial conditions.
    for j in range(K):
        ref_count[reference_sets[j], j] = 1
    ref_count[ts.samples(), K] = 1

    for (left, _right), edges_out, edges_in in ts.edge_diffs():
        for edge in edges_out:
            parent[edge.child] = -1
            update_counts(edge, left, -1)
        for edge in edges_in:
            parent[edge.child] = edge.parent
            update_counts(edge, left, +1)

    # Finally, add the stats for the last tree and divide by the total
    # span that each node was an ancestor to > 0 samples.
    for v in range(ts.num_nodes):
        if ref_count[v, K] > 0:
            span = ts.sequence_length - last_update[v]
            total_span[v] += span
            C[v] += span * ref_count[v, :K]
        if total_span[v] != 0:
            C[v] /= total_span[v]
    return C


def genealogical_nearest_neighbours(ts, focal, reference_sets):

    reference_set_map = np.zeros(ts.num_nodes, dtype=int) - 1
    for k, reference_set in enumerate(reference_sets):
        for u in reference_set:
            if reference_set_map[u] != -1:
                raise ValueError("Duplicate value in reference sets")
            reference_set_map[u] = k

    K = len(reference_sets)
    A = np.zeros((len(focal), K))
    L = np.zeros(len(focal))
    parent = np.zeros(ts.num_nodes, dtype=int) - 1
    sample_count = np.zeros((ts.num_nodes, K), dtype=int)

    # Set the initial conditions.
    for j in range(K):
        sample_count[reference_sets[j], j] = 1

    for (left, right), edges_out, edges_in in ts.edge_diffs():
        for edge in edges_out:
            parent[edge.child] = -1
            v = edge.parent
            while v != -1:
                sample_count[v] -= sample_count[edge.child]
                v = parent[v]
        for edge in edges_in:
            parent[edge.child] = edge.parent
            v = edge.parent
            while v != -1:
                sample_count[v] += sample_count[edge.child]
                v = parent[v]

        # Process this tree.
        for j, u in enumerate(focal):
            focal_reference_set = reference_set_map[u]
            delta = int(focal_reference_set != -1)
            p = u
            while p != tskit.NULL:
                total = np.sum(sample_count[p])
                if total > delta:
                    break
                p = parent[p]
            if p != tskit.NULL:
                span = right - left
                L[j] += span
                scale = span / (total - delta)
                for k, _reference_set in enumerate(reference_sets):
                    n = sample_count[p, k] - int(focal_reference_set == k)
                    A[j, k] += n * scale

    # Avoid division by zero
    L[L == 0] = 1
    A /= L.reshape((len(focal), 1))
    return A
