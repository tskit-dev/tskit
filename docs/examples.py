"""
Examples for the tutorial.
"""
import io
import os
import sys

sys.path.insert(0, os.path.abspath("../python"))

import numpy as np  # noqa: E402
import msprime  # noqa: E402

import tskit  # noqa: E402


def moving_along_tree_sequence():
    ts = msprime.simulate(5, recombination_rate=1, random_seed=42)

    print(f"Tree sequence has {ts.num_trees} trees")
    print()
    for tree in ts.trees():
        print(
            "Tree {} covers [{:.2f}, {:.2f}); TMRCA = {:.4f}".format(
                tree.index, *tree.interval, tree.time(tree.root)
            )
        )

    print()
    for tree in reversed(ts.trees()):
        print(
            "Tree {} covers [{:.2f}, {:.2f}); TMRCA = {:.4f}".format(
                tree.index, *tree.interval, tree.time(tree.root)
            )
        )

    print()
    for tree in list(ts.trees()):
        print(
            "Tree {} covers [{:.2f}, {:.2f}): id={:x}".format(
                tree.index, *tree.interval, id(tree)
            )
        )

    print()
    for tree in ts.aslist():
        print(
            "Tree {} covers [{:.2f}, {:.2f}): id={:x}".format(
                tree.index, *tree.interval, id(tree)
            )
        )

    print()
    tree = ts.at(0.5)
    print(
        "Tree {} covers [{:.2f}, {:.2f}): id={:x}".format(
            tree.index, *tree.interval, id(tree)
        )
    )
    tree = ts.at_index(0)
    print(
        "Tree {} covers [{:.2f}, {:.2f}): id={:x}".format(
            tree.index, *tree.interval, id(tree)
        )
    )
    tree = ts.at_index(-1)
    print(
        "Tree {} covers [{:.2f}, {:.2f}): id={:x}".format(
            tree.index, *tree.interval, id(tree)
        )
    )

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
    print(f"Tree {tree.index}: parent_dict = {tree.parent_dict}")
    tree.first()
    print(f"Tree {tree.index}: parent_dict = {tree.parent_dict}")
    tree.prev()
    print(f"Tree {tree.index}: parent_dict = {tree.parent_dict}")

    tree = tskit.Tree(ts)
    while tree.next():
        print("Tree {} covers [{:.2f}, {:.2f})".format(tree.index, *tree.interval))
    print("After loop: tree index =", tree.index)
    while tree.prev():
        print("Tree {} covers [{:.2f}, {:.2f})".format(tree.index, *tree.interval))


def stats():
    ts = msprime.simulate(
        10 ** 4,
        Ne=10 ** 4,
        recombination_rate=1e-8,
        mutation_rate=1e-8,
        length=10 ** 7,
        random_seed=42,
    )
    print("num_trees = ", ts.num_trees, ", num_sites = ", ts.num_sites, sep="")

    x = ts.diversity()
    print(f"Average diversity per unit sequence length = {x:.3G}")

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


def tree_traversal():
    nodes = """\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           0
    3       1           0
    4       1           0
    5       0           1
    6       0           2
    7       0           3
    """
    edges = """\
    left    right   parent  child
    0       1       5       0,1,2
    0       1       6       3,4
    0       1       7       5,6
    """
    # NB same tree as used above, and we're using the same diagram.
    ts = tskit.load_text(
        nodes=io.StringIO(nodes), edges=io.StringIO(edges), strict=False
    )
    tree = ts.first()

    for order in ["preorder", "inorder", "postorder"]:
        print(f"{order}:\t", list(tree.nodes(order=order)))

    total_branch_length = sum(tree.branch_length(u) for u in tree.nodes())
    print(total_branch_length, tree.total_branch_length)

    for u in tree.samples():
        path = []
        v = u
        while v != tskit.NULL:
            path.append(v)
            v = tree.parent(v)
        print(u, "->", path)

    def preorder_dist(tree):
        for root in tree.roots:
            stack = [(root, 0)]
            while len(stack) > 0:
                u, distance = stack.pop()
                yield u, distance
                for v in tree.children(u):
                    stack.append((v, distance + 1))

    print(list(preorder_dist(tree)))


# moving_along_tree_sequence()
# stats()
tree_traversal()
