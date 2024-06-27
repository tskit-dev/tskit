# MIT License
#
# Copyright (c) 2021-2024 Tskit Developers
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
A Python version of the C AVL tree code for development purposes.

Based on Knuth's AVL tree code in TAOCP volume 3, adapted from
https://commandlinefanatic.com/cgi-bin/showarticle.cgi?article=art070

Note there is a bug in that Python translation which is missing
P.B = 0 at the end of A9.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest


# The nodes of the tree are assumed to contain KEY, LLINK, and RLINK fields.
# We also have a new field
#
# B(P) = balance factor of NODE(P)
#
# the height of the right subtree minus the height of the left subtree; this field
# always contains either +1, 0, or -1.  A special header node also appears at the top
# of the tree, in location HEAD; the value of RLINK(HEAD) is a pointer to the root
# of the tree, and LLINK(HEAD) is used to keep track of the overall height of the tree.
# We assume that the tree is nonempty, namely that RLINK(HEAD) != ^.


@dataclasses.dataclass(eq=False)
class Node:
    key: Any = None
    llink: Node = None
    rlink: Node = None
    balance: int = 0

    def __str__(self):
        llink = None if self.llink is None else self.llink.key
        rlink = None if self.rlink is None else self.rlink.key
        return (
            f"Node(key={self.key}, balance={self.balance}, "
            f"llink={llink}, rlink={rlink})"
        )


# For convenience in description, the algorithm uses the notation LINK(a,P)
# as a synonym for LLINK(P) if a = -1, and for RLINK(P) if a = +1.


def get_link(a, P):
    if a == -1:
        return P.llink
    else:
        return P.rlink


def set_link(a, P, val):
    if a == -1:
        P.llink = val
    else:
        P.rlink = val


class AvlTree:
    def __init__(self):
        self.head = Node()
        self.size = 0
        self.height = 0

    @property
    def root(self):
        return self.head.rlink

    def __str__(self):
        stack = [(self.head, 0)]
        s = f"size = {self.size} height = {self.height}\n"
        while len(stack) > 0:
            node, depth = stack.pop()
            s += ("  " * depth) + f"KEY={node.key} B={node.balance}\n"
            for child in [node.llink, node.rlink]:
                if child is not None:
                    stack.append((child, depth + 1))
        return s

    def ordered_keys(self):
        """
        Return the keys in sorted order. This is done by an in-order
        traversal of the nodes.
        """

        def inorder(node):
            if node is not None:
                yield from inorder(node.llink)
                yield node.key
                yield from inorder(node.rlink)

        yield from inorder(self.root)

    def search(self, key):
        P = self.root
        while P is not None:
            if key == P.key:
                break
            elif key < P.key:
                P = P.llink
            else:
                P = P.rlink
        return P

    def __insert_empty(self, key):
        self.head.rlink = Node()
        self.head.rlink.key = key
        self.size = 1
        self.height = 1
        return self.head.rlink

    def __insert(self, K):
        # A1. [Initialize.] Set T <- HEAD, S <- P <- RLINK(HEAD).
        # (The pointer variable P will move down the tree; S will point
        # to the place where rebalancing may be necessary, and
        # T always points to the parent of S.)
        T = self.head
        S = P = self.head.rlink

        # A2. [Compare.] If K < KEY(P), go to A3; if K > KEY(P), go to A4; and if
        # K = KEY(P), the search terminates successfully.
        while True:
            if K == P.key:
                return P
            elif K < P.key:
                # A3. [Move left.] Set Q <- LLINK(P). If Q = ^, set Q <= AVAIL and
                # LLINK(P) <- Q and go to step A5. Otherwise if B(Q) != 0, set T <-
                # P and S <- Q. Finally set P <- Q and return to step A2.
                Q = P.llink
                if Q is None:
                    Q = Node()
                    P.llink = Q
                    break
            # A4. [Move right.] Set Q <- RLINK(P). If Q = ^, set Q <= AVAIL and
            # RLINK(P) <- Q and go to step A5. Otherwise if B(Q) != 0, set T <- P
            # and S <- Q. Finally set P <- Q and return to step A2.
            elif K > P.key:
                Q = P.rlink
                if Q is None:
                    Q = Node()
                    P.rlink = Q
                    break
            if Q.balance != 0:
                T = P
                S = Q
            P = Q
        # A5. [Insert.] Set KEY(Q) <- K, LLINK(Q) <- RLINK(Q) <- ^, and B(Q) <- 0.
        Q.key = K
        Q.llink = Q.rlink = None
        Q.balance = 0

        # A6. [Adjust balance factors.] If K < KEY(S) set a <- -1, otherwise set a
        # <- +1. Then set R <- P <- LINK(a,S), and repeatedly do the following
        # operations zero or more times until P = Q: If K < KEY(P) set B(P) <- -1
        # and P <- LLINK(P); if K > KEY(P), set B(P) <- +1 and P <- RLINK(P).
        if K < S.key:
            a = -1
        else:
            a = 1
        R = P = get_link(a, S)
        while P != Q:
            if K < P.key:
                P.balance = -1
                P = P.llink
            elif K > P.key:
                P.balance = 1
                P = P.rlink

        # A7. [Balancing act.] Several cases now arise:
        #
        #  i) If B(S) = 0, set B(S) <- a, LLINK(HEAD) <- LLINK(HEAD) + 1, and
        #  terminate the algorithm.
        #
        if S.balance == 0:
            S.balance = a
            self.height += 1

        # ii) If B(S) = -a, set B(S) <- 0 and terminate the algorithm.

        elif S.balance == -a:
            S.balance = 0

        # iii) If B(S) = a, go to step A8 if B(R) = a, to A9 if B(R) = -a.
        else:
            if R.balance == a:
                # A8. [Single rotation.] Set P <- R, LINK(a,S) <- LINK(-a,R),
                # LINK(-a,R) <- S,B(S) <- B(R) <- 0. Go to A10.
                P = R
                set_link(a, S, get_link(-a, R))
                set_link(-a, R, S)
                S.balance = R.balance = 0
            elif R.balance == -a:
                # A9. [Double rotation.] Set P <- LINK(-a,R),
                #  LINK(-a,R) <- LINK(a,P),LINK(a,P) <- R, LINK(a,S)
                #  <- LINK(-a,P), LINK(-a,P) <- S. Now set
                #
                #               { (-a,0), if B(P) =  a;
                #  (B(S),B(R))<-{ ( 0,0), if B(P) =  0;
                #               { ( 0,a), if B(P) = -a;
                #
                #  and then set B(P) <- 0
                P = get_link(-a, R)
                set_link(-a, R, get_link(a, P))
                set_link(a, P, R)
                set_link(a, S, get_link(-a, P))
                set_link(-a, P, S)
                if P.balance == a:
                    S.balance = -a
                    R.balance = 0
                elif P.balance == 0:
                    S.balance = 0
                    R.balance = 0
                else:
                    S.balance = 0
                    R.balance = a
                P.balance = 0

            # A10. [Finishing touch.] If S = RLINK(T) then set RLINK(T) <- P,
            # otherwise set LLINK(T) <- P.
            if S == T.rlink:
                T.rlink = P
            else:
                T.llink = P

        return Q

    def insert(self, key):
        if self.size == 0:
            return self.__insert_empty(key)
        return self.__insert(key)


class TestAvlTree:
    def verify_tree(self, tree):
        """
        Check that the tree fits the AVL tree properties.
        """
        # The height of a node is its maximum distance to a leaf
        node_height = {}

        def compute_height(node):
            if node is None:
                return 0
            val = 1 + max([compute_height(node.llink), compute_height(node.rlink)])
            node_height[node] = val
            return val

        compute_height(tree.head.rlink)
        assert tree.height == max(node_height.values())
        assert tree.height == node_height[tree.head.rlink]
        # print(tree)

        # The balance factor B is the height of the right subtree
        # minus the height of the left subtree
        stack = [tree.head.rlink]
        while len(stack) > 0:
            node = stack.pop()
            # print(node, node_height[node])
            assert node.balance in [-1, 0, 1]
            lheight = None
            if node.llink is not None:
                lheight = node_height[node.llink]
                stack.append(node.llink)
            rheight = None
            if node.rlink is not None:
                rheight = node_height[node.rlink]
                stack.append(node.rlink)
            if lheight is not None and rheight is not None:
                balance_factor = rheight - lheight
                assert node.balance == balance_factor
            elif lheight is None and rheight is None:
                assert node_height[node] == 1
                assert node.balance == 0
            else:
                # if one child is None, the height of this node must be 2
                assert node_height[node] == 2
                if lheight is None:
                    assert node.balance == 1
                else:
                    assert node.balance == -1

    def verify(self, keys):
        tree = AvlTree()
        key_set = set()
        for k in keys:
            node = tree.search(k)
            if k in key_set:
                assert node is not None
                assert node.key == k
            else:
                assert node is None
            node = tree.insert(k)
            key_set.add(k)
            self.verify_tree(tree)
            assert tree.search(k) is node
        for k in range(100):
            node = tree.search(k)
            if k in key_set:
                assert node is not None
                assert node.key == k
            else:
                assert node is None
        ordered_keys = list(tree.ordered_keys())
        assert ordered_keys == list(sorted(set(keys)))

        # Implement the inorder on an existing list to mimic C algorithm
        l2 = [None for _ in ordered_keys]

        def visit(node, index, out):
            if node is None:
                return index
            index = visit(node.llink, index, out)
            out[index] = node.key
            return visit(node.rlink, index + 1, out)

        visit(tree.root, 0, l2)
        assert l2 == ordered_keys

    @pytest.mark.parametrize("n", [0, 1, 10, 33, 64, 127, 133])
    def test_sequential(self, n):
        self.verify(range(n))

    @pytest.mark.parametrize("n", [0, 1, 10, 33, 64, 127, 133])
    def test_sequential_reversed(self, n):
        self.verify(range(n)[::-1])

    @pytest.mark.parametrize("n", [0, 1, 10, 33, 64, 127, 133])
    def test_random_integers(self, n):
        rng = np.random.RandomState(42)
        values = rng.randint(-100, 100, size=n)
        self.verify(values)

    @pytest.mark.parametrize("n", [0, 1, 10, 33, 64, 127, 133])
    def test_random_floats(self, n):
        rng = np.random.RandomState(42)
        values = rng.random(size=n)
        self.verify(values)
