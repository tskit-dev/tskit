"""
Examples for the tutorial.
"""
import os
import sys
sys.path.insert(0, os.path.abspath('../python'))

import msprime

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
    for tree in list(ts):
        print("Tree {} covers [{:.2f}, {:.2f}): id={:x}".format(
            tree.index, *tree.interval, id(tree)))

    print()
    tree = ts[0]
    print("Tree {} covers [{:.2f}, {:.2f}): id={:x}".format(
        tree.index, *tree.interval, id(tree)))
    tree = ts[-1]
    print("Tree {} covers [{:.2f}, {:.2f}): id={:x}".format(
        tree.index, *tree.interval, id(tree)))


moving_along_tree_sequence()
