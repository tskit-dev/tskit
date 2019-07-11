"""
Examples for the tutorial.
"""
import os
import sys
sys.path.insert(0, os.path.abspath('../python'))

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
    colours = ["red", "blue", "green"]
    genotypes = [0, 0, 0, 0, 1, 2]
    node_colours = {j: colours[g] for j, g in enumerate(genotypes)}
    ancestral_state, transitions = tree.map_mutations(genotypes)
    print("Ancestral state = ", ancestral_state)
    for transition in transitions:
        print("\t", transition)
    tree.draw("_static/parsimony1.svg", node_colours=node_colours)

    tree = msprime.simulate(6, random_seed=42).first()
    colours = ["red", "blue", "green", "white"]
    genotypes = [-1, 0, 0, 0, 1, 2]
    node_colours = {j: colours[g] for j, g in enumerate(genotypes)}
    ancestral_state, transitions = tree.map_mutations(genotypes)
    print("Ancestral state = ", ancestral_state)
    for transition in transitions:
        print("\t", transition)
    tree.draw("_static/parsimony2.svg", node_colours=node_colours)

    tree = msprime.simulate(6, random_seed=42).first()
    colours = ["red", "blue", "white"]
    genotypes = [1, -1, 0, 0, 0, 0]
    node_colours = {j: colours[g] for j, g in enumerate(genotypes)}
    ancestral_state, transitions = tree.map_mutations(genotypes)
    print("Ancestral state = ", ancestral_state)
    for transition in transitions:
        print("\t", transition)
    tree.draw("_static/parsimony3.svg", node_colours=node_colours)




# moving_along_tree_sequence()
parsimony()
