"""
Examples for the tutorial.
"""
import os
import sys
sys.path.insert(0, os.path.abspath('../python'))

import numpy as np
import msprime
import tskit

def moving_along_tree_sequence():
    ts = msprime.simulate(5, recombination_rate=1, random_seed=42)

    print("Tree sequence has {} trees".format(ts.num_trees))
    print()
    for tree in ts.trees():
        print("Tree {} covers [{:.2f}, {:.2f}); TMRCA = {:.4f}".format(
            tree.index, *tree.interval, tree.time(tree.root)))

    print()
    for tree in reversed(ts.trees()):
        print("Tree {} covers [{:.2f}, {:.2f}); TMRCA = {:.4f}".format(

            tree.index, *tree.interval, tree.time(tree.root)))

    print()
    for tree in list(ts.trees()):
        print("Tree {} covers [{:.2f}, {:.2f}): id={:x}".format(
            tree.index, *tree.interval, id(tree)))

    print()
    for tree in ts.aslist():
        print("Tree {} covers [{:.2f}, {:.2f}): id={:x}".format(
            tree.index, *tree.interval, id(tree)))

    print()
    tree = ts.at(0.5)
    print("Tree {} covers [{:.2f}, {:.2f}): id={:x}".format(
        tree.index, *tree.interval, id(tree)))
    tree = ts.at_index(0)
    print("Tree {} covers [{:.2f}, {:.2f}): id={:x}".format(
        tree.index, *tree.interval, id(tree)))
    tree = ts.at_index(-1)
    print("Tree {} covers [{:.2f}, {:.2f}): id={:x}".format(
        tree.index, *tree.interval, id(tree)))

    print()
    tree = tskit.Tree(ts)
    tree.seek_index(ts.num_trees // 2)
    print("Tree {} covers [{:.2f}, {:.2f})".format(tree.index, *tree.interval))
    tree.seek(0.95)
    print("Tree {} covers [{:.2f}, {:.2f})".format(tree.index, *tree.interval))

    print()
    tree.prev()
    print("Tree {} covers [{:.2f}, {:.2f})".format(tree.index, *tree.interval))
    tree.next()
    print("Tree {} covers [{:.2f}, {:.2f})".format(tree.index, *tree.interval))

    print()
    tree = tskit.Tree(ts)
    print("Tree {}: parent_dict = {}".format(tree.index, tree.parent_dict))
    tree.first()
    print("Tree {}: parent_dict = {}".format(tree.index, tree.parent_dict))
    tree.prev()
    print("Tree {}: parent_dict = {}".format(tree.index, tree.parent_dict))

    tree = tskit.Tree(ts)
    while tree.next():
        print("Tree {} covers [{:.2f}, {:.2f})".format(tree.index, *tree.interval))
    print("After loop: tree index =", tree.index)
    while tree.prev():
        print("Tree {} covers [{:.2f}, {:.2f})".format(tree.index, *tree.interval))


def parsimony():

    tree = msprime.simulate(6, random_seed=42).first()
    alleles = ["red", "blue", "green"]
    genotypes = [0, 0, 0, 0, 1, 2]
    node_colours = {j: alleles[g] for j, g in enumerate(genotypes)}
    ancestral_state, mutations = tree.map_mutations(genotypes, alleles)
    print("Ancestral state = ", ancestral_state)
    for mut in mutations:
        print(f"Mutation: node = {mut.node} derived_state = {mut.derived_state}")
    tree.draw("_static/parsimony1.svg", node_colours=node_colours)


    ts = msprime.simulate(6, random_seed=23)
    ts = msprime.mutate(
        ts, rate=3, model=msprime.InfiniteSites(msprime.NUCLEOTIDES), random_seed=2)

    tree = ts.first()
    tables = ts.dump_tables()
    # Reinfer the sites and mutations from the variants.
    tables.sites.clear()
    tables.mutations.clear()
    for var in ts.variants():
        ancestral_state, mutations = tree.map_mutations(var.genotypes, var.alleles)
        tables.sites.add_row(var.site.position, ancestral_state=ancestral_state)
        parent_offset = len(tables.mutations)
        for mutation in mutations:
            parent = mutation.parent
            if parent != tskit.NULL:
                parent += parent_offset
            tables.mutations.add_row(
                var.index, node=mutation.node, parent=parent,
                derived_state=mutation.derived_state)

    assert tables.sites == ts.tables.sites
    assert tables.mutations == ts.tables.mutations
    print(tables.sites)
    print(tables.mutations)

    tree = msprime.simulate(6, random_seed=42).first()
    alleles = ["red", "blue", "green", "white"]
    genotypes = [-1, 0, 0, 0, 1, 2]
    node_colours = {j: alleles[g] for j, g in enumerate(genotypes)}
    ancestral_state, mutations = tree.map_mutations(genotypes, alleles)
    print("Ancestral state = ", ancestral_state)
    for mut in mutations:
        print(f"Mutation: node = {mut.node} derived_state = {mut.derived_state}")
    tree.draw("_static/parsimony2.svg", node_colours=node_colours)

    tree = msprime.simulate(6, random_seed=42).first()
    alleles = ["red", "blue", "white"]
    genotypes = [1, -1, 0, 0, 0, 0]
    node_colours = {j: alleles[g] for j, g in enumerate(genotypes)}
    ancestral_state, mutations = tree.map_mutations(genotypes, alleles)
    print("Ancestral state = ", ancestral_state)
    for mut in mutations:
        print(f"Mutation: node = {mut.node} derived_state = {mut.derived_state}")
    tree.draw("_static/parsimony3.svg", node_colours=node_colours)


def allele_frequency_spectra():

    ts = msprime.simulate(6, mutation_rate=1, random_seed=47)
    tree = ts.first()
    tree.draw("_static/afs1.svg")

    print(ts.tables.sites)

    afs = ts.allele_frequency_spectrum(polarised=True)
    print(afs)

    afs = ts.allele_frequency_spectrum(
        windows=[0, 0.5, 1], span_normalise=False, polarised=True)
    print(afs)

    node_colours = {
        0: "blue", 2: "blue", 3: "blue",
        1: "green", 4: "green", 5: "green"}
    tree.draw("_static/afs2.svg", node_colours=node_colours)

    afs = ts.allele_frequency_spectrum([[0, 2, 3], [1, 4, 5]], polarised=True)
    print(afs)

    afs = ts.allele_frequency_spectrum(mode="branch", polarised=True)
    print(afs)

    afs = ts.allele_frequency_spectrum([[0, 1, 2]], mode="branch", polarised=True)
    print(afs)
    print("sum afs          = ", np.sum(afs))
    print("total branch len = ", tree.total_branch_length)

def missing_data():

    ts = msprime.simulate(4, random_seed=2)
    tables = ts.dump_tables()
    tables.nodes.add_row(time=0, flags=1)
    tables.simplify()
    ts = tables.tree_sequence()
    tree = ts.first()
    tree.draw("_static/missing_data1.svg")


def stats():
    ts = msprime.simulate(
        10**4, Ne=10**4, recombination_rate=1e-8, mutation_rate=1e-8, length=10**7,
        random_seed=42)
    print("num_trees = ", ts.num_trees, ", num_sites = ", ts.num_sites, sep="")

    x = ts.diversity()
    print("Average diversity per unit sequence length = {:.3G}".format(x))

    windows = np.linspace(0, ts.sequence_length, num=5)
    x = ts.diversity(windows=windows)
    print(windows)
    print(x)

    A = ts.samples()[:100]
    x = ts.diversity(sample_sets=A)
    print(x)

    B = ts.samples()[100:200]
    C = ts.samples()[200:300]
    x = ts.diversity(sample_sets=[A, B, C])
    print(x)

    x = ts.diversity(sample_sets=[A, B, C], windows=windows)
    print("shape = ", x.shape)
    print(x)

    A = ts.samples()[:100]
    B = ts.samples()[:100]
    x = ts.divergence([A, B])
    print(x)

    x = ts.divergence([A, B], windows=windows)
    print(x)

    x = ts.divergence([A, B, C], indexes=[(0, 1), (0, 2)])
    print(x)

    x = ts.divergence([A, B, C], indexes=(0, 1))
    print(x)

    x = ts.divergence([A, B, C], indexes=[(0, 1)])
    print(x)

    x = ts.divergence([A, B, C], indexes=[(0, 1), (0, 2)], windows=windows)
    print(x)

# moving_along_tree_sequence()
# parsimony()
# allele_frequency_spectra()
# missing_data()

stats()

