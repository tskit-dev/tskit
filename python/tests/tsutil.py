# MIT License
#
# Copyright (c) 2018-2024 Tskit Developers
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
import dataclasses
import functools
import json
import random
import string
import struct
import typing

import msprime
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
            site_id = t.sites.append(site)
            for mutation in site.mutations:
                t.mutations.append(mutation.replace(site=site_id))
    add_provenance(t.provenances, "subsample_sites")
    return t.tree_sequence()


def insert_branch_mutations(ts, mutations_per_branch=1):
    """
    Returns a copy of the specified tree sequence with a mutation on every branch
    in every tree.
    """
    if mutations_per_branch == 0:
        return ts
    tables = ts.dump_tables()
    tables.sites.clear()
    tables.mutations.clear()
    for tree in ts.trees():
        site = tables.sites.add_row(position=tree.interval.left, ancestral_state="0")
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


def remove_mutation_times(ts):
    tables = ts.tables
    tables.mutations.time = np.full_like(tables.mutations.time, tskit.UNKNOWN_TIME)
    return tables.tree_sequence()


def insert_discrete_time_mutations(ts, num_times=4, num_sites=10):
    """
    Inserts mutations in the tree sequence at regularly-spaced num_sites
    positions, at only a discrete set of times (the same for all trees): at
    num_times times evenly spaced between 0 and the maximum time.
    """
    tables = ts.tables
    tables.sites.clear()
    tables.mutations.clear()
    height = max(t.time(t.roots[0]) for t in ts.trees())
    for j, pos in enumerate(np.linspace(0, tables.sequence_length, num_sites + 1)[:-1]):
        anc = "X" * j
        tables.sites.add_row(position=pos, ancestral_state=anc)
        t = ts.at(pos)
        for k, s in enumerate(np.linspace(0, height, num_times)):
            for n in t.nodes():
                if t.time(n) <= s and (
                    (t.parent(n) == tskit.NULL) or (t.time(t.parent(n)) > s)
                ):
                    tables.mutations.add_row(
                        site=j, node=n, derived_state=anc + str(k), time=s
                    )
                    k += 1
    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()
    return tables.tree_sequence()


def insert_branch_sites(ts, m=1):
    """
    Returns a copy of the specified tree sequence with m sites on every branch
    of every tree.
    """
    if m == 0:
        return ts
    tables = ts.dump_tables()
    tables.sites.clear()
    tables.mutations.clear()
    for tree in ts.trees():
        left, right = tree.interval
        delta = (right - left) / (m * len(list(tree.nodes())))
        x = left
        for u in tree.nodes():
            if tree.parent(u) != tskit.NULL:
                for _ in range(m):
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
            position=tree.interval.left, ancestral_state=ancestral_state
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


def insert_random_ploidy_individuals(
    ts, min_ploidy=0, max_ploidy=5, max_dimension=3, samples_only=True, seed=1
):
    """
    Takes random contiguous subsets of the samples and assigns them to individuals.
    Also creates random locations in variable dimensions in the unit interval,
    and assigns random parents (including NULL parents). Note that resulting
    individuals will often have nodes with inconsistent populations and/or time.
    """
    rng = random.Random(seed)
    if samples_only:
        node_ids = np.array(ts.samples(), dtype="int")
    else:
        node_ids = np.arange(ts.num_nodes)
    j = 0
    tables = ts.dump_tables()
    tables.individuals.clear()
    individual = tables.nodes.individual[:]
    individual[:] = tskit.NULL
    ind_id = -1
    while j < len(node_ids):
        ploidy = rng.randint(min_ploidy, max_ploidy)
        nodes = node_ids[j : min(j + ploidy, len(node_ids))]
        dimension = rng.randint(0, max_dimension)
        location = [rng.random() for _ in range(dimension)]
        parents = rng.sample(range(-1, 1 + ind_id), min(1 + ind_id, rng.randint(0, 3)))
        ind_id = tables.individuals.add_row(location=location, parents=parents)
        individual[nodes] = ind_id
        j += ploidy
    tables.nodes.individual = individual
    return tables.tree_sequence()


def insert_random_consistent_individuals(
    ts, min_ploidy=0, max_ploidy=5, min_dimension=0, max_dimension=3, seed=1
):
    """
    Takes random subsets of nodes having the same time and population and
    assigns them to individuals.  Also creates random locations in variable
    dimensions in the unit interval, and assigns random parents (including NULL
    parents).
    """
    rng = random.Random(seed)
    tables = ts.dump_tables()
    tables.individuals.clear()
    individual = tables.nodes.individual[:]
    individual[:] = tskit.NULL
    ind_id = -1
    pops = np.arange(ts.num_populations)
    for pop in pops:
        n = tables.nodes.population == pop
        times = np.unique(tables.nodes.time[n])
        for t in times:
            nn = np.where(np.logical_and(n, tables.nodes.time == t))[0]
            rng.shuffle(nn)
            j = 0
            while j < len(nn):
                ploidy = rng.randint(min_ploidy, max_ploidy)
                nodes = nn[j : min(j + ploidy, len(nn))]
                dimension = rng.randint(min_dimension, max_dimension)
                location = [rng.random() for _ in range(dimension)]
                parents = rng.sample(
                    range(-1, 1 + ind_id), min(1 + ind_id, rng.randint(0, 3))
                )
                ind_id = tables.individuals.add_row(location=location, parents=parents)
                individual[nodes] = ind_id
                j += ploidy
                j += rng.randint(0, 2)  # skip a random number
    tables.nodes.individual = individual
    return tables.tree_sequence()


def insert_individuals(ts, nodes=None, ploidy=1):
    """
    Inserts individuals into the tree sequence using the specified list
    of node (or use all sample nodes if None) with the specified ploidy by combining
    ploidy-sized chunks of the list. Add metadata to the individuals so we can
    track them
    """
    if nodes is None:
        nodes = ts.samples()
    assert len(nodes) % ploidy == 0  # To allow mixed ploidies we could comment this out
    tables = ts.dump_tables()
    tables.individuals.clear()
    individual = tables.nodes.individual[:]
    individual[:] = tskit.NULL
    j = 0
    while j < len(nodes):
        nodes_in_individual = nodes[j : min(len(nodes), j + ploidy)]
        # should we warn here if nodes[j : j + ploidy] are at different times?
        # probably not, as although this is unusual, it is actually allowed
        ind_id = tables.individuals.add_row(
            metadata=f"orig_id {tables.individuals.num_rows}".encode()
        )
        individual[nodes_in_individual] = ind_id
        j += ploidy
    tables.nodes.individual = individual
    return tables.tree_sequence()


def mark_metadata(ts, table_name, prefix="orig_id:"):
    """
    Add metadata to all rows of the form prefix + row_number
    """
    tables = ts.dump_tables()
    table = getattr(tables, table_name)
    table.packset_metadata([(prefix + str(i)).encode() for i in range(table.num_rows)])
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
        tables.nodes.append(old_node)
    for edge in ts.edges():
        tables.edges.append(
            edge.replace(parent=node_map[edge.parent], child=node_map[edge.child])
        )
    for site in ts.sites():
        for mutation in site.mutations:
            tables.mutations.append(
                mutation.replace(site=site.id, node=node_map[mutation.node])
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
        tables.edges.append(r.replace(right=x))
        tables.edges.append(r.replace(left=x))
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
        tables.edges.append(edge.replace(parent=u))
        tables.edges.append(edge.replace(child=u))
        for mut in mutations_on_edge[edge.id]:
            if mut.time < t:
                tables.mutations.append(mut)
            else:
                tables.mutations.append(mut.replace(node=u))
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
        parents=individuals.parents,
        parents_offset=individuals.parents_offset,
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
    assert tree.interval.left <= position < tree.interval.right
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
                    new_state = random.choice(states)
                    if multiple_per_node:
                        mutation_table.add_row(site, u, new_state, parent)
                        parent = mutation_table.num_rows - 1
                        state = new_state
                    x += random.expovariate(mu)
                else:
                    if not multiple_per_node:
                        mutation_table.add_row(site, u, new_state, parent)
                        parent = mutation_table.num_rows - 1
                        state = new_state
            stack.extend(reversed([(v, state, parent) for v in tree.children(u)]))


def jukes_cantor(ts, num_sites, mu, multiple_per_node=True, seed=None):
    """
    Returns a copy of the specified tree sequence with Jukes-Cantor mutations
    applied at the specified rate at the specified number of sites. Site positions
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
        while position >= t.interval.right:
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


def py_subset(
    tables,
    nodes,
    record_provenance=True,
    reorder_populations=True,
    remove_unreferenced=True,
):
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
    if not reorder_populations:
        for j, pop in enumerate(full.populations):
            pop_map[j] = j
            tables.populations.append(pop)
    # first build individual map
    if not remove_unreferenced:
        keep_ind = [True for _ in full.individuals]
    else:
        keep_ind = [False for _ in full.individuals]
        for old_id in nodes:
            i = full.nodes[old_id].individual
            if i != tskit.NULL:
                keep_ind[i] = True
    new_ind_id = 0
    for j, k in enumerate(keep_ind):
        if k:
            ind_map[j] = new_ind_id
            new_ind_id += 1
    # now the individual table
    for j, k in enumerate(keep_ind):
        if k:
            ind = full.individuals[j]
            new_ind_id = tables.individuals.append(
                ind.replace(parents=[ind_map[i] for i in ind.parents if i in ind_map])
            )

            assert new_ind_id == ind_map[j]

    for old_id in nodes:
        node = full.nodes[old_id]
        if node.population not in pop_map and node.population != tskit.NULL:
            pop = full.populations[node.population]
            new_pop_id = tables.populations.append(pop)
            pop_map[node.population] = new_pop_id
        new_id = tables.nodes.append(
            node.replace(
                population=pop_map[node.population],
                individual=ind_map[node.individual],
            )
        )
        node_map[old_id] = new_id
    if not remove_unreferenced:
        for j, ind in enumerate(full.populations):
            if j not in pop_map:
                pop_map[j] = tables.populations.append(ind)
    for edge in full.edges:
        if edge.child in nodes and edge.parent in nodes:
            tables.edges.append(
                edge.replace(parent=node_map[edge.parent], child=node_map[edge.child])
            )
    if full.migrations.num_rows > 0:
        raise ValueError("Migrations are currently not supported in this operation.")
    site_map = {}
    if not remove_unreferenced:
        for j, site in enumerate(full.sites):
            site_map[j] = tables.sites.append(site)
    mutation_map = {tskit.NULL: tskit.NULL}
    for i, mut in enumerate(full.mutations):
        if mut.node in nodes:
            if mut.site not in site_map:
                site = full.sites[mut.site]
                new_site = tables.sites.append(site)
                site_map[mut.site] = new_site
            new_mut = tables.mutations.append(
                mut.replace(
                    site=site_map[mut.site],
                    node=node_map[mut.node],
                    parent=mutation_map.get(mut.parent, tskit.NULL),
                )
            )
            mutation_map[i] = new_mut

    tables.sort()


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
    original_num_individuals = tables.individuals.num_rows

    for other_id, node in enumerate(other.nodes):
        if nodes[other_id] != tskit.NULL and node.individual != tskit.NULL:
            ind_map[node.individual] = tables.nodes[nodes[other_id]].individual

    for other_id, node in enumerate(other.nodes):
        if nodes[other_id] != tskit.NULL:
            node_map[other_id] = nodes[other_id]
        else:
            if ind_map[node.individual] == tskit.NULL and node.individual != tskit.NULL:
                ind = other.individuals[node.individual]
                ind_id = tables.individuals.append(ind)
                ind_map[node.individual] = ind_id
            if pop_map[node.population] == tskit.NULL and node.population != tskit.NULL:
                if not add_populations:
                    pop_map[node.population] = node.population
                else:
                    pop = other.populations[node.population]
                    pop_id = tables.populations.append(pop)
                    pop_map[node.population] = pop_id
            node_id = tables.nodes.append(
                node.replace(
                    population=pop_map[node.population],
                    individual=ind_map[node.individual],
                )
            )
            node_map[other_id] = node_id
    individuals = tables.individuals
    new_parents = individuals.parents
    for i in range(
        individuals.parents_offset[original_num_individuals], len(individuals.parents)
    ):
        new_parents[i] = ind_map[individuals.parents[i]]
    individuals.parents = new_parents
    for edge in other.edges:
        if (nodes[edge.parent] == tskit.NULL) or (nodes[edge.child] == tskit.NULL):
            tables.edges.append(
                edge.replace(parent=node_map[edge.parent], child=node_map[edge.child])
            )
    for other_id, mut in enumerate(other.mutations):
        if nodes[mut.node] == tskit.NULL:
            # add site: may already be in tables, but we deduplicate
            if site_map[mut.site] == tskit.NULL:
                site = other.sites[mut.site]
                site_id = tables.sites.append(site)
                site_map[mut.site] = site_id
            mut_id = tables.mutations.append(
                mut.replace(
                    site=site_map[mut.site],
                    node=node_map[mut.node],
                    parent=tskit.NULL,
                )
            )
            mut_map[other_id] = mut_id
    # migration table
    # grafting provenance table
    if record_provenance:
        pass
    # sorting, deduplicating sites, and re-computing mutation parents
    tables.sort()
    tables.deduplicate_sites()
    # need to sort again since after deduplicating sites, mutations may not be
    # sorted by time within sites
    tables.sort()
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


def shuffle_tables(
    tables,
    seed,
    shuffle_edges=True,
    shuffle_populations=True,
    shuffle_individuals=True,
    shuffle_sites=True,
    shuffle_mutations=True,
    shuffle_migrations=True,
    keep_mutation_parent_order=False,
):
    """
    Randomizes the order of rows in (possibly) all except the Node table.  Note
    that if mutations are completely shuffled, then TableCollection.sort() will
    not necessarily produce valid tables (unless all mutation times are present
    and distinct), since currently only canonicalise puts parent mutations
    before children.  However, setting keep_mutation_parent_order to True will
    maintain the order of mutations within each site.

    :param TableCollection tables: The table collection that is shuffled (in place).
    """
    rng = random.Random(seed)
    orig = tables.copy()
    tables.nodes.clear()
    tables.individuals.clear()
    tables.populations.clear()
    tables.edges.clear()
    tables.sites.clear()
    tables.mutations.clear()
    tables.drop_index()
    # populations
    randomised_pops = list(enumerate(orig.populations))
    if shuffle_populations:
        rng.shuffle(randomised_pops)
    pop_id_map = {tskit.NULL: tskit.NULL}
    for j, p in randomised_pops:
        pop_id_map[j] = tables.populations.append(p)
    # individuals
    randomised_inds = list(enumerate(orig.individuals))
    if shuffle_individuals:
        rng.shuffle(randomised_inds)
    ind_id_map = {tskit.NULL: tskit.NULL}
    for j, i in randomised_inds:
        ind_id_map[j] = tables.individuals.append(i)
    tables.individuals.parents = [
        tskit.NULL if i == tskit.NULL else ind_id_map[i]
        for i in tables.individuals.parents
    ]
    # nodes (same order, but remapped populations and individuals)
    for n in orig.nodes:
        tables.nodes.append(
            n.replace(
                population=pop_id_map[n.population],
                individual=ind_id_map[n.individual],
            )
        )
    # edges
    randomised_edges = list(orig.edges)
    if shuffle_edges:
        rng.shuffle(randomised_edges)
    for e in randomised_edges:
        tables.edges.append(e)
    # migrations
    randomised_migrations = list(orig.migrations)
    if shuffle_migrations:
        rng.shuffle(randomised_migrations)
    for m in randomised_migrations:
        tables.migrations.append(
            m.replace(source=pop_id_map[m.source], dest=pop_id_map[m.dest])
        )
    # sites
    randomised_sites = list(enumerate(orig.sites))
    if shuffle_sites:
        rng.shuffle(randomised_sites)
    site_id_map = {}
    for j, s in randomised_sites:
        site_id_map[j] = tables.sites.append(s)
    # mutations
    randomised_mutations = list(enumerate(orig.mutations))
    if shuffle_mutations:
        if keep_mutation_parent_order:
            # randomise *except* keeping parent mutations before children
            mut_site_order = [mut.site for mut in orig.mutations]
            rng.shuffle(mut_site_order)
            mut_by_site = {s: [] for s in mut_site_order}
            for j, m in enumerate(orig.mutations):
                mut_by_site[m.site].insert(0, (j, m))
            randomised_mutations = []
            for s in mut_site_order:
                randomised_mutations.append(mut_by_site[s].pop())
        else:
            rng.shuffle(randomised_mutations)
    mut_id_map = {tskit.NULL: tskit.NULL}
    for j, (k, _) in enumerate(randomised_mutations):
        mut_id_map[k] = j
    for _, m in randomised_mutations:
        tables.mutations.append(
            m.replace(site=site_id_map[m.site], parent=mut_id_map[m.parent])
        )
    if keep_mutation_parent_order:
        assert np.all(tables.mutations.parent < np.arange(tables.mutations.num_rows))
    return tables


def cmp_site(i, j, tables):
    ret = tables.sites.position[i] - tables.sites.position[j]
    if ret == 0:
        ret = i - j
    return ret


def cmp_mutation_canonical(i, j, tables, site_order, num_descendants=None):
    site_i = tables.mutations.site[i]
    site_j = tables.mutations.site[j]
    ret = site_order[site_i] - site_order[site_j]
    if (
        ret == 0
        and (not tskit.is_unknown_time(tables.mutations.time[i]))
        and (not tskit.is_unknown_time(tables.mutations.time[j]))
    ):
        ret = tables.mutations.time[j] - tables.mutations.time[i]
    if ret == 0:
        ret = num_descendants[j] - num_descendants[i]
    if ret == 0:
        ret = tables.mutations.node[i] - tables.mutations.node[j]
    if ret == 0:
        ret = i - j
    return ret


def cmp_mutation(i, j, tables, site_order):
    site_i = tables.mutations.site[i]
    site_j = tables.mutations.site[j]
    ret = site_order[site_i] - site_order[site_j]
    if (
        ret == 0
        and (not tskit.is_unknown_time(tables.mutations.time[i]))
        and (not tskit.is_unknown_time(tables.mutations.time[j]))
    ):
        ret = tables.mutations.time[j] - tables.mutations.time[i]
    if ret == 0:
        ret = i - j
    return ret


def cmp_edge(i, j, tables):
    ret = (
        tables.nodes.time[tables.edges.parent[i]]
        - tables.nodes.time[tables.edges.parent[j]]
    )
    if ret == 0:
        ret = tables.edges.parent[i] - tables.edges.parent[j]
    if ret == 0:
        ret = tables.edges.child[i] - tables.edges.child[j]
    if ret == 0:
        ret = tables.edges.left[i] - tables.edges.left[j]
    return ret


def cmp_migration(i, j, tables):
    ret = tables.migrations.time[i] - tables.migrations.time[j]
    if ret == 0:
        ret = tables.migrations.source[i] - tables.migrations.source[j]
    if ret == 0:
        ret = tables.migrations.dest[i] - tables.migrations.dest[j]
    if ret == 0:
        ret = tables.migrations.left[i] - tables.migrations.left[j]
    if ret == 0:
        ret = tables.migrations.node[i] - tables.migrations.node[j]
    return ret


def cmp_individual_canonical(i, j, tables, num_descendants):
    ret = num_descendants[j] - num_descendants[i]
    if ret == 0:
        node_i = node_j = tables.nodes.num_rows
        ni = np.where(tables.nodes.individual == i)[0]
        if len(ni) > 0:
            node_i = np.min(ni)
        nj = np.where(tables.nodes.individual == j)[0]
        if len(nj) > 0:
            node_j = np.min(nj)
        ret = node_i - node_j
    if ret == 0:
        ret = i - j
    return ret


def compute_mutation_num_descendants(tables):
    mutations = tables.mutations
    num_descendants = np.zeros(mutations.num_rows)
    for p in mutations.parent:
        while p != tskit.NULL:
            num_descendants[p] += 1
            p = mutations.parent[p]
    return num_descendants


def compute_individual_num_descendants(tables):
    # adapted from sort_individual_table
    individuals = tables.individuals
    num_individuals = individuals.num_rows
    num_descendants = np.zeros((num_individuals,), np.int64)

    # First find the set of individuals that have no children
    # by creating an array of incoming edge counts
    incoming_edge_count = np.zeros((num_individuals,), np.int64)
    for parent in individuals.parents:
        if parent != tskit.NULL:
            incoming_edge_count[parent] += 1
    todo = np.full((num_individuals + 1,), -1, np.int64)
    current_todo = 0
    todo_insertion_point = 0
    for individual, num_edges in enumerate(incoming_edge_count):
        if num_edges == 0:
            todo[todo_insertion_point] = individual
            todo_insertion_point += 1

    # Now process individuals from the set that have no children, updating their
    # parents' information as we go, and adding their parents to the list if
    # this was their last child
    while todo[current_todo] != -1:
        individual = todo[current_todo]
        current_todo += 1
        for parent in individuals.parents[
            individuals.parents_offset[individual] : individuals.parents_offset[
                individual + 1
            ]
        ]:
            if parent != tskit.NULL:
                incoming_edge_count[parent] -= 1
                num_descendants[parent] += 1 + num_descendants[individual]
                if incoming_edge_count[parent] == 0:
                    todo[todo_insertion_point] = parent
                    todo_insertion_point += 1

    if num_individuals > 0:
        assert np.min(incoming_edge_count) >= 0
        if np.max(incoming_edge_count) > 0:
            raise ValueError("Individual pedigree has cycles")
    return num_descendants


def py_canonicalise(tables, remove_unreferenced=True):
    tables.subset(
        np.arange(tables.nodes.num_rows),
        record_provenance=False,
        remove_unreferenced=remove_unreferenced,
    )
    py_sort(tables, canonical=True)


def py_sort(tables, canonical=False):
    copy = tables.copy()
    tables.edges.clear()
    tables.sites.clear()
    tables.mutations.clear()
    tables.migrations.clear()
    edge_key = functools.cmp_to_key(lambda a, b: cmp_edge(a, b, tables=copy))
    sorted_edges = sorted(range(copy.edges.num_rows), key=edge_key)
    site_key = functools.cmp_to_key(lambda a, b: cmp_site(a, b, tables=copy))
    sorted_sites = sorted(range(copy.sites.num_rows), key=site_key)
    site_id_map = {k: j for j, k in enumerate(sorted_sites)}
    site_order = np.argsort(sorted_sites)
    if canonical:
        mut_num_descendants = compute_mutation_num_descendants(copy)
        mut_key = functools.cmp_to_key(
            lambda a, b: cmp_mutation_canonical(
                a,
                b,
                tables=copy,
                site_order=site_order,
                num_descendants=mut_num_descendants,
            )
        )
    else:
        mut_key = functools.cmp_to_key(
            lambda a, b: cmp_mutation(
                a,
                b,
                tables=copy,
                site_order=site_order,
            )
        )
    sorted_muts = sorted(range(copy.mutations.num_rows), key=mut_key)
    mut_id_map = {k: j for j, k in enumerate(sorted_muts)}
    mut_id_map[tskit.NULL] = tskit.NULL
    mig_key = functools.cmp_to_key(lambda a, b: cmp_migration(a, b, tables=copy))
    sorted_migs = sorted(range(copy.migrations.num_rows), key=mig_key)
    for edge_id in sorted_edges:
        tables.edges.append(copy.edges[edge_id])
    for site_id in sorted_sites:
        tables.sites.append(copy.sites[site_id])
    for mut_id in sorted_muts:
        tables.mutations.append(
            copy.mutations[mut_id].replace(
                site=site_id_map[copy.mutations[mut_id].site],
                parent=mut_id_map[copy.mutations[mut_id].parent],
            )
        )
    for mig_id in sorted_migs:
        tables.migrations.append(copy.migrations[mig_id])

    # individuals
    if canonical:
        tables.individuals.clear()
        ind_num_descendants = compute_individual_num_descendants(copy)
        ind_key = functools.cmp_to_key(
            lambda a, b: cmp_individual_canonical(
                a,
                b,
                tables=copy,
                num_descendants=ind_num_descendants,
            )
        )
        sorted_inds = sorted(range(copy.individuals.num_rows), key=ind_key)
        ind_id_map = {k: j for j, k in enumerate(sorted_inds)}
        ind_id_map[tskit.NULL] = tskit.NULL
        for ind_id in sorted_inds:
            tables.individuals.append(
                copy.individuals[ind_id].replace(
                    parents=[ind_id_map[p] for p in copy.individuals[ind_id].parents],
                )
            )
        tables.nodes.individual = [ind_id_map[i] for i in tables.nodes.individual]


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


class QuintuplyLinkedTree:
    def __init__(self, n, root_threshold=1):
        self.root_threshold = root_threshold
        self.parent = np.zeros(n + 1, dtype=np.int32) - 1
        self.left_child = np.zeros(n + 1, dtype=np.int32) - 1
        self.right_child = np.zeros(n + 1, dtype=np.int32) - 1
        self.left_sib = np.zeros(n + 1, dtype=np.int32) - 1
        self.right_sib = np.zeros(n + 1, dtype=np.int32) - 1
        self.num_samples = np.zeros(n + 1, dtype=np.int32)
        self.num_edges = 0
        self.num_children = np.zeros(n + 1, dtype=np.int32)
        self.edge = np.zeros(n + 1, dtype=np.int32) - 1

    def __str__(self):
        s = "id\tparent\tlchild\trchild\tlsib\trsib\tnsamp\tnchild\tedge\n"
        for j in range(len(self.parent)):
            s += (
                f"{j}\t{self.parent[j]}\t"
                f"{self.left_child[j]}\t{self.right_child[j]}\t"
                f"{self.left_sib[j]}\t{self.right_sib[j]}\t"
                f"{self.num_samples[j]}\t"
                f"{self.num_children[j]}\t"
                f"{self.edge[j]}\n"
            )
        return s

    def roots(self):
        roots = []
        u = self.left_child[-1]
        while u != -1:
            roots.append(u)
            u = self.right_sib[u]
        return roots

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
        self.num_children[p] -= 1

    def insert_branch(self, p, c):
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
        self.num_children[p] += 1

    def is_potential_root(self, u):
        return self.num_samples[u] >= self.root_threshold

    # Note we cheat a bit here and use the -1 == last element semantics from Python.
    # We could use self.insert_branch(N, root) and then set self.parent[root] = -1.
    def insert_root(self, root):
        self.insert_branch(-1, root)

    def remove_root(self, root):
        self.remove_branch(-1, root)

    def remove_edge(self, edge):
        self.remove_branch(edge.parent, edge.child)
        self.num_edges -= 1
        self.edge[edge.child] = -1

        u = edge.parent
        while u != -1:
            path_end = u
            path_end_was_root = self.is_potential_root(u)
            self.num_samples[u] -= self.num_samples[edge.child]
            u = self.parent[u]

        if path_end_was_root and not self.is_potential_root(path_end):
            self.remove_root(path_end)
        if self.is_potential_root(edge.child):
            self.insert_root(edge.child)

    def insert_edge(self, edge):
        u = edge.parent
        while u != -1:
            path_end = u
            path_end_was_root = self.is_potential_root(u)
            self.num_samples[u] += self.num_samples[edge.child]
            u = self.parent[u]

        if self.is_potential_root(edge.child):
            self.remove_root(edge.child)
        if self.is_potential_root(path_end) and not path_end_was_root:
            self.insert_root(path_end)

        self.insert_branch(edge.parent, edge.child)
        self.num_edges += 1
        self.edge[edge.child] = edge.id


def algorithm_R(ts, root_threshold=1):
    """
    Quintuply linked tree with root tracking.
    """
    sequence_length = ts.sequence_length
    N = ts.num_nodes
    M = ts.num_edges
    tree = QuintuplyLinkedTree(N, root_threshold=root_threshold)
    edges = list(ts.edges())
    in_order = ts.tables.indexes.edge_insertion_order
    out_order = ts.tables.indexes.edge_removal_order

    # Initialise the tree
    for u in ts.samples():
        tree.num_samples[u] = 1
        if tree.is_potential_root(u):
            tree.insert_root(u)

    j = 0
    k = 0
    left = 0
    while j < M or left < sequence_length:
        while k < M and edges[out_order[k]].right == left:
            tree.remove_edge(edges[out_order[k]])
            k += 1
        while j < M and edges[in_order[j]].left == left:
            tree.insert_edge(edges[in_order[j]])
            j += 1
        right = sequence_length
        if j < M:
            right = min(right, edges[in_order[j]].left)
        if k < M:
            right = min(right, edges[out_order[k]].right)
        yield (left, right), tree
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
        in_order = ts.tables.indexes.edge_insertion_order
        out_order = ts.tables.indexes.edge_removal_order
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


class LegacyRootThresholdTree:
    """
    Implementation of the quintuply linked tree with root tracking using the
    pre C 1.0/Python 0.4.0 algorithm. We keep this version around to make sure
    that we can be clear what the differences in the semantics of the new
    and old versions are.

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
        in_order = ts.tables.indexes.edge_insertion_order
        out_order = ts.tables.indexes.edge_removal_order
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
            if self.left_root != tskit.NULL:
                while self.left_sib[self.left_root] != tskit.NULL:
                    self.left_root = self.left_sib[self.left_root]
            right = sequence_length
            if j < M:
                right = min(right, edges[in_order[j]].left)
            if k < M:
                right = min(right, edges[out_order[k]].right)
            yield left, right
            left = right


FORWARD = 1
REVERSE = -1


@dataclasses.dataclass
class Interval:
    left: float
    right: float

    def __iter__(self):
        yield self.left
        yield self.right


@dataclasses.dataclass
class EdgeRange:
    start: int
    stop: int
    order: typing.List


class TreePosition:
    def __init__(self, ts):
        self.ts = ts
        self.index = -1
        self.direction = 0
        self.interval = Interval(0, 0)
        self.in_range = EdgeRange(0, 0, None)
        self.out_range = EdgeRange(0, 0, None)

    def __str__(self):
        s = f"index: {self.index}\ninterval: {self.interval}\n"
        s += f"direction: {self.direction}\n"
        s += f"in_range: {self.in_range}\n"
        s += f"out_range: {self.out_range}\n"
        return s

    def assert_equal(self, other):
        assert self.index == other.index
        assert self.direction == other.direction
        assert self.interval == other.interval

    def set_null(self):
        self.index = -1
        self.interval.left = 0
        self.interval.right = 0

    def next(self):  # NOQA: A003
        M = self.ts.num_edges
        breakpoints = self.ts.breakpoints(as_array=True)
        left_coords = self.ts.edges_left
        left_order = self.ts.indexes_edge_insertion_order
        right_coords = self.ts.edges_right
        right_order = self.ts.indexes_edge_removal_order

        if self.index == -1:
            self.interval.right = 0
            self.out_range.stop = 0
            self.in_range.stop = 0
            self.direction = FORWARD

        if self.direction == FORWARD:
            left_current_index = self.in_range.stop
            right_current_index = self.out_range.stop
        else:
            left_current_index = self.out_range.stop + 1
            right_current_index = self.in_range.stop + 1

        left = self.interval.right

        j = right_current_index
        self.out_range.start = j
        while j < M and right_coords[right_order[j]] == left:
            j += 1
        self.out_range.stop = j
        self.out_range.order = right_order

        j = left_current_index
        self.in_range.start = j
        while j < M and left_coords[left_order[j]] == left:
            j += 1
        self.in_range.stop = j
        self.in_range.order = left_order

        self.direction = FORWARD
        self.index += 1
        if self.index == self.ts.num_trees:
            self.set_null()
        else:
            self.interval.left = left
            self.interval.right = breakpoints[self.index + 1]
        return self.index != -1

    def prev(self):
        M = self.ts.num_edges
        breakpoints = self.ts.breakpoints(as_array=True)
        right_coords = self.ts.edges_right
        right_order = self.ts.indexes_edge_removal_order
        left_coords = self.ts.edges_left
        left_order = self.ts.indexes_edge_insertion_order

        if self.index == -1:
            self.index = self.ts.num_trees
            self.interval.left = self.ts.sequence_length
            self.in_range.stop = M - 1
            self.out_range.stop = M - 1
            self.direction = REVERSE

        if self.direction == REVERSE:
            left_current_index = self.out_range.stop
            right_current_index = self.in_range.stop
        else:
            left_current_index = self.in_range.stop - 1
            right_current_index = self.out_range.stop - 1

        right = self.interval.left

        j = left_current_index
        self.out_range.start = j
        while j >= 0 and left_coords[left_order[j]] == right:
            j -= 1
        self.out_range.stop = j
        self.out_range.order = left_order

        j = right_current_index
        self.in_range.start = j
        while j >= 0 and right_coords[right_order[j]] == right:
            j -= 1
        self.in_range.stop = j
        self.in_range.order = right_order

        self.direction = REVERSE
        self.index -= 1
        if self.index == -1:
            self.set_null()
        else:
            self.interval.left = breakpoints[self.index]
            self.interval.right = right
        return self.index != -1

    def seek_forward(self, index):
        # NOTE this is still in development and not fully tested.
        assert index >= self.index and index < self.ts.num_trees
        M = self.ts.num_edges
        breakpoints = self.ts.breakpoints(as_array=True)
        left_coords = self.ts.edges_left
        left_order = self.ts.indexes_edge_insertion_order
        right_coords = self.ts.edges_right
        right_order = self.ts.indexes_edge_removal_order

        if self.index == -1:
            self.interval.right = 0
            self.out_range.stop = 0
            self.in_range.stop = 0
            self.direction = FORWARD

        if self.direction == FORWARD:
            left_current_index = self.in_range.stop
            right_current_index = self.out_range.stop
        else:
            left_current_index = self.out_range.stop + 1
            right_current_index = self.in_range.stop + 1

        self.direction = FORWARD
        left = breakpoints[index]

        # The range of edges we need consider for removal starts
        # at the current right index and ends at the first edge
        # where the right coordinate is equal to the new tree's
        # left coordinate.
        j = right_current_index
        self.out_range.start = j
        # TODO This could be done with binary search
        while j < M and right_coords[right_order[j]] <= left:
            j += 1
        self.out_range.stop = j

        if self.index == -1:
            # No edges, so out_range should be empty
            self.out_range.start = self.out_range.stop

        # The range of edges we need to consider for the new tree
        # must have right coordinate > left
        j = left_current_index
        while j < M and right_coords[left_order[j]] <= left:
            j += 1
        self.in_range.start = j
        # TODO this could be done with a binary search
        while j < M and left_coords[left_order[j]] <= left:
            j += 1
        self.in_range.stop = j

        self.interval.left = left
        self.interval.right = breakpoints[index + 1]
        self.out_range.order = right_order
        self.in_range.order = left_order
        self.index = index

    def seek_backward(self, index):
        # NOTE this is still in development and not fully tested.
        assert index >= 0
        M = self.ts.num_edges
        breakpoints = self.ts.breakpoints(as_array=True)
        left_coords = self.ts.edges_left
        left_order = self.ts.indexes_edge_insertion_order
        right_coords = self.ts.edges_right
        right_order = self.ts.indexes_edge_removal_order

        if self.index == -1:
            assert index < self.ts.num_trees
            self.index = self.ts.num_trees
            self.interval.left = self.ts.sequence_length
            self.in_range.stop = M - 1
            self.out_range.stop = M - 1
            self.direction = REVERSE
        else:
            assert index <= self.index

        if self.direction == REVERSE:
            left_current_index = self.out_range.stop
            right_current_index = self.in_range.stop
        else:
            left_current_index = self.in_range.stop - 1
            right_current_index = self.out_range.stop - 1

        self.direction = REVERSE
        right = breakpoints[index + 1]

        # The range of edges we need consider for removal starts
        # at the current left index and ends at the first edge
        # where the left coordinate is equal to the new tree's
        # right coordinate.
        j = left_current_index
        self.out_range.start = j
        # TODO This could be done with binary search
        while j >= 0 and left_coords[left_order[j]] >= right:
            j -= 1
        self.out_range.stop = j

        if self.index == self.ts.num_trees:
            # No edges, so out_range should be empty
            self.out_range.start = self.out_range.stop

        # The range of edges we need to consider for the new tree
        # must have left coordinate < right
        j = right_current_index
        while j >= 0 and left_coords[right_order[j]] >= right:
            j -= 1
        self.in_range.start = j
        # We stop at the first edge with right coordinate < right
        while j >= 0 and right_coords[right_order[j]] >= right:
            j -= 1
        self.in_range.stop = j

        self.interval.right = right
        self.interval.left = breakpoints[index]
        self.out_range.order = left_order
        self.in_range.order = right_order
        self.index = index

    def step(self, direction):
        if direction == FORWARD:
            return self.next()
        elif direction == REVERSE:
            return self.prev()
        else:
            raise ValueError("Direction must be FORWARD (+1) or REVERSE (-1)")


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


def sort_individual_table(tables):
    """
    Sorts the individual table by parents-before-children.
    """

    individuals = tables.individuals
    num_individuals = individuals.num_rows

    # First find the set of individuals that have no children
    # by creating an array of incoming edge counts
    incoming_edge_count = np.zeros((num_individuals,), np.int64)
    for parent in individuals.parents:
        if parent != tskit.NULL:
            incoming_edge_count[parent] += 1

    todo = collections.deque()
    sorted_order = []
    for individual, num_edges in reversed(list(enumerate(incoming_edge_count))):
        if num_edges == 0:
            todo.append(individual)
            sorted_order.append(individual)
    # Now emit individuals from the set that have no children, removing their edges
    # as we go adding new individuals to the no children set.
    while len(todo) > 0:
        individual = todo.popleft()
        for parent in individuals[individual].parents:
            if parent != tskit.NULL:
                incoming_edge_count[parent] -= 1
                if incoming_edge_count[parent] == 0:
                    todo.append(parent)
                    sorted_order.append(parent)

    if np.sum(incoming_edge_count) > 0:
        raise ValueError("Individual pedigree has cycles")

    ind_id_map = {tskit.NULL: tskit.NULL}

    individuals_copy = tables.copy().individuals
    tables.individuals.clear()
    for row in reversed(sorted_order):
        ind_id_map[row] = tables.individuals.append(individuals_copy[row])
    tables.individuals.parents = [ind_id_map[i] for i in tables.individuals.parents]
    tables.nodes.individual = [ind_id_map[i] for i in tables.nodes.individual]

    return tables


def insert_unique_metadata(tables, table=None, offset=0):
    if isinstance(tables, tskit.TreeSequence):
        tables = tables.dump_tables()
    else:
        tables = tables.copy()
    if table is None:
        table = [t for t in tskit.TABLE_NAMES if t != "provenances"]
    for t in table:
        getattr(tables, t).packset_metadata(
            [struct.pack("I", offset + i) for i in range(getattr(tables, t).num_rows)]
        )
    return tables.tree_sequence()


def metadata_map(tables):
    # builds a mapping from metadata (as produced by insert_unique_metadata)
    # to ID for all the tables (except provenance)
    if isinstance(tables, tskit.TreeSequence):
        tables = tables.dump_tables()
    out = {}
    for t in [t for t in tskit.TABLE_NAMES if t != "provenances"]:
        out[t] = {}
        for j, x in enumerate(getattr(tables, t)):
            out[t][x.metadata] = j
    return out


@functools.lru_cache(maxsize=None)
def all_trees_ts(n):
    """
    Generate a tree sequence that corresponds to the lexicographic listing
    of all trees with n leaves (i.e. from tskit.all_trees(n)).

    Note: it would be nice to include a version of this in the combinatorics
    module at some point but the implementation is quite inefficient. Also
    it's not entirely clear that the way we're allocating node times is
    guaranteed to work.
    """
    tables = tskit.TableCollection(0)
    for _ in range(n):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    for j in range(1, n):
        tables.nodes.add_row(flags=0, time=j)

    L = 0
    for tree in tskit.all_trees(n):
        for u in tree.preorder()[1:]:
            tables.edges.add_row(L, L + 1, tree.parent(u), u)
        L += 1
    tables.sequence_length = L
    tables.sort()
    tables.simplify()
    return tables.tree_sequence()


def all_fields_ts(edge_metadata=True, migrations=True):
    """
    A tree sequence with data in all fields (except edge metadata is not set if
    edge_metadata is False and migrations are not defined if migrations is False
    (this is needed to test simplify, which doesn't allow either)

    """
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=10_000)
    demography.add_population(name="B", initial_size=5_000)
    demography.add_population(name="C", initial_size=1_000)
    demography.add_population(name="D", initial_size=500)
    demography.add_population(name="E", initial_size=100)
    demography.add_population_split(time=1000, derived=["A", "B"], ancestral="C")
    ts = msprime.sim_ancestry(
        samples={"A": 10, "B": 10},
        demography=demography,
        sequence_length=5,
        random_seed=42,
        recombination_rate=1,
        record_migrations=migrations,
        record_provenance=True,
    )
    ts = msprime.sim_mutations(ts, rate=0.001, random_seed=42)
    tables = ts.dump_tables()
    # Add locations to individuals
    individuals_copy = tables.individuals.copy()
    tables.individuals.clear()
    for i, individual in enumerate(individuals_copy):
        tables.individuals.append(
            individual.replace(flags=i, location=[i, i + 1], parents=[i - 1, i - 1])
        )
    # Ensure all columns have unique values
    nodes_copy = tables.nodes.copy()
    tables.nodes.clear()
    for i, node in enumerate(nodes_copy):
        tables.nodes.append(
            node.replace(
                flags=i,
                time=node.time + 0.00001 * i,
                individual=i % len(tables.individuals),
                population=i % len(tables.populations),
            )
        )
    if migrations:
        tables.migrations.add_row(left=0, right=1, node=21, source=1, dest=3, time=1001)

    # Add metadata
    for name, table in tables.table_name_map.items():
        if name == "provenances":
            continue
        if name == "migrations" and not migrations:
            continue
        if name == "edges" and not edge_metadata:
            continue
        table.metadata_schema = tskit.MetadataSchema.permissive_json()
        metadatas = [f'{{"foo":"n_{name}_{u}"}}' for u in range(len(table))]
        metadata, metadata_offset = tskit.pack_strings(metadatas)
        table.set_columns(
            **{
                **table.asdict(),
                "metadata": metadata,
                "metadata_offset": metadata_offset,
            }
        )
    tables.metadata_schema = tskit.MetadataSchema.permissive_json()
    tables.metadata = "Test metadata"
    tables.time_units = "Test time units"

    tables.reference_sequence.metadata_schema = tskit.MetadataSchema.permissive_json()
    tables.reference_sequence.metadata = "Test reference metadata"
    tables.reference_sequence.data = "A" * int(ts.sequence_length)
    # NOTE: it's unclear whether we'll want to have this set at the same time as
    # 'data', but it's useful to have something in all columns for now.
    tables.reference_sequence.url = "http://example.com/a_reference"

    # Add some more rows to provenance to have enough for testing.
    for i in range(3):
        tables.provenances.add_row(record="A", timestamp=str(i))

    return tables.tree_sequence()
