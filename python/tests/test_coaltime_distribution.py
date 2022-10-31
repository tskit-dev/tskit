# MIT License
#
# Copyright (c) 2018-2023 Tskit Developers
# Copyright (C) 2016 University of Oxford
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
Test cases for coalescence time distribution objects in tskit.
"""
import msprime
import numpy as np
import pytest

import tests
import tskit


class TestCoalescenceTimeDistribution:
    """
    Tree sequences used in tests of classes `CoalescenceTimeTable` and
    `CoalescenceTimeDistribution`
    """

    @tests.cached_example
    def ts_multimerger_six_leaves(self):
        """
        29.00┊    9        ┊
             ┊ ┏━━┻━━┓     ┊
        8.00 ┊ ┃     8     ┊
             ┊ ┃   ┏━┻━━┓  ┊
        5.00 ┊ ┃   7    ┃  ┊
             ┊ ┃ ┏━╋━┓  ┃  ┊
        1.00 ┊ ┃ ┃ ┃ ┃  6  ┊
             ┊ ┃ ┃ ┃ ┃ ┏┻┓ ┊
        0.00 ┊ 0 1 2 4 3 5 ┊
             0            100
        """
        tables = tskit.TableCollection(sequence_length=100)
        tables.nodes.set_columns(
            time=np.array([0, 0, 0, 0, 0, 0, 1, 5, 8, 29]),
            flags=np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype="uint32"),
        )
        tables.edges.set_columns(
            left=np.repeat([0.0], 9),
            right=np.repeat([100.0], 9),
            parent=np.array([6, 6, 7, 7, 7, 8, 8, 9, 9], dtype="int32"),
            child=np.array([3, 5, 1, 2, 4, 6, 7, 0, 8], dtype="int32"),
        )
        return tables.tree_sequence()

    @tests.cached_example
    def ts_multimerger_eight_leaves(self):
        """
        10.00┊         13      ┊
             ┊       ┏━━┻━━┓   ┊
        8.00 ┊      12     ┃   ┊
             ┊     ┏━┻━┓   ┃   ┊
        6.00 ┊    11   ┃   ┃   ┊
             ┊  ┏━━╋━┓ ┃   ┃   ┊
        2.00 ┊ 10  ┃ ┃ ┃   9   ┊
             ┊ ┏┻┓ ┃ ┃ ┃  ┏┻━┓ ┊
        1.00 ┊ ┃ ┃ ┃ ┃ ┃  8  ┃ ┊
             ┊ ┃ ┃ ┃ ┃ ┃ ┏┻┓ ┃ ┊
        0.00 ┊ 0 7 4 5 6 1 2 3 ┊
             0                100
        """
        tables = tskit.TableCollection(sequence_length=100)
        tables.nodes.set_columns(
            time=np.array([0] * 8 + [1, 2, 2, 6, 8, 10]),
            flags=np.repeat([1, 0], [8, 6]).astype("uint32"),
        )
        tables.edges.set_columns(
            left=np.repeat([0], 13),
            right=np.repeat([100], 13),
            parent=np.array(
                [8, 8, 9, 9, 10, 10, 11, 11, 11, 12, 12, 13, 13], dtype="int32"
            ),
            child=np.array([1, 2, 3, 8, 0, 7, 4, 5, 10, 6, 11, 9, 12], dtype="int32"),
        )
        return tables.tree_sequence()

    @tests.cached_example
    def ts_two_trees_four_leaves(self):
        """
        1.74┊   7     ┊   7     ┊
            ┊ ┏━┻━┓   ┊ ┏━┻━┓   ┊
        0.73┊ ┃   6   ┊ ┃   ┃   ┊
            ┊ ┃ ┏━┻┓  ┊ ┃   ┃   ┊
        0.59┊ ┃ ┃  5  ┊ ┃   5   ┊
            ┊ ┃ ┃ ┏┻┓ ┊ ┃  ┏┻━┓ ┊
        0.54┊ ┃ ┃ ┃ ┃ ┊ ┃  4  ┃ ┊
            ┊ ┃ ┃ ┃ ┃ ┊ ┃ ┏┻┓ ┃ ┊
        0.00┊ 0 1 2 3 ┊ 0 1 2 3 ┊
          0.00      0.88      1.00
        """
        tables = tskit.TableCollection(sequence_length=1)
        tables.nodes.set_columns(
            time=np.array([0, 0, 0, 0, 0.54, 0.59, 0.73, 1.74]),
            flags=np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype="uint32"),
        )
        tables.edges.set_columns(
            left=np.array([0.88, 0.88, 0, 0, 0.88, 0, 0, 0, 0.88, 0]),
            right=np.array([1, 1, 0.88, 1, 1, 0.88, 0.88, 1, 1, 0.88]),
            parent=np.array([4, 4, 5, 5, 5, 6, 6, 7, 7, 7], dtype="int32"),
            child=np.array([1, 2, 2, 3, 4, 1, 5, 0, 5, 6], dtype="int32"),
        )
        return tables.tree_sequence()

    @tests.cached_example
    def ts_five_trees_three_leaves(self):
        tables = tskit.TableCollection(sequence_length=1)
        tables.nodes.set_columns(
            time=np.array([0.0, 0.0, 0.0, 0.05, 0.15, 1.13, 4.21, 7.53]),
            flags=np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype="uint32"),
        )
        tables.edges.set_columns(
            left=np.array([0.4, 0.4, 0, 0, 0.4, 0, 0, 0.2, 0.2, 0.1, 0.3, 0.1, 0.3]),
            right=np.array([1, 1, 1, 0.4, 1, 0.1, 0.1, 0.3, 0.3, 0.2, 0.4, 0.2, 0.4]),
            parent=np.array([3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 7, 7, 7], dtype="int32"),
            child=np.array([0, 2, 1, 2, 3, 0, 4, 0, 4, 0, 0, 4, 4], dtype="int32"),
        )
        return tables.tree_sequence()

    @tests.cached_example
    def ts_eight_trees_two_leaves(self):
        tables = tskit.TableCollection(sequence_length=8)
        tables.nodes.set_columns(
            time=np.array([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
            flags=np.array([1, 1, 0, 0, 0, 0, 0, 0, 0], dtype="uint32"),
        )
        tables.edges.set_columns(
            left=np.array([6, 6, 7, 7, 5, 5, 1, 1, 0, 2, 0, 2, 3, 3, 4, 4]),
            right=np.array([7, 7, 8, 8, 6, 6, 2, 2, 1, 3, 1, 3, 4, 4, 5, 5]),
            parent=np.array(
                [2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 8, 8],
                dtype="int32",
            ),
            child=np.array(
                [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1],
                dtype="int32",
            ),
        )
        return tables.tree_sequence()

    @tests.cached_example
    def ts_two_trees_ten_leaves(self):
        tables = tskit.TableCollection(sequence_length=2)
        tables.nodes.set_columns(
            time=np.array([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
            flags=np.array([1, 1, 0, 0, 0, 0, 0, 0, 0], dtype="uint32"),
        )
        tables.edges.set_columns(
            left=np.array([6, 6, 7, 7, 5, 5, 1, 1, 0, 2, 0, 2, 3, 3, 4, 4]),
            right=np.array([7, 7, 8, 8, 6, 6, 2, 2, 1, 3, 1, 3, 4, 4, 5, 5]),
            parent=np.array(
                [2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 8, 8],
                dtype="int32",
            ),
            child=np.array(
                [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1],
                dtype="int32",
            ),
        )
        return tables.tree_sequence()

    @tests.cached_example
    def ts_many_edge_diffs(self):
        ts = msprime.sim_ancestry(
            samples=80,
            ploidy=1,
            sequence_length=4,
            recombination_rate=10,
            random_seed=1234,
        )
        return ts


class TestUnweightedCoalescenceTimeTable(TestCoalescenceTimeDistribution):
    """
                         Block Weight C.Wght Quant
                         -------------------------
    29.00┊    9        ┊    0     1        4  1.00
         ┊ ┏━━┻━━┓     ┊
    8.00 ┊ ┃     8     ┊    0     1        3  0.75
         ┊ ┃   ┏━┻━━┓  ┊
    5.00 ┊ ┃   7    ┃  ┊    0     1        2  0.50
         ┊ ┃ ┏━╋━┓  ┃  ┊
    1.00 ┊ ┃ ┃ ┃ ┃  6  ┊    0     1        1  0.25
         ┊ ┃ ┃ ┃ ┃ ┏┻┓ ┊
    0.00 ┊ 0 1 2 4 3 5 ┊    0     0        0  0.00 < to catch OOR
         0            100
    Uniform weights on nodes
    """

    def coalescence_time_distribution(self):
        ts = self.ts_multimerger_six_leaves()
        distr = ts.coalescence_time_distribution(span_normalise=False)
        return distr

    def test_time(self):
        t = np.array([0, 1, 5, 8, 29])
        distr = self.coalescence_time_distribution()
        tt = distr.tables[0].time
        np.testing.assert_allclose(t, tt)

    def test_block(self):
        b = np.array([0, 0, 0, 0, 0])
        distr = self.coalescence_time_distribution()
        tb = distr.tables[0].block
        np.testing.assert_allclose(b, tb)

    def test_weights(self):
        w = np.array([[0, 1, 1, 1, 1]]).T
        distr = self.coalescence_time_distribution()
        tw = distr.tables[0].weights
        np.testing.assert_allclose(w, tw)

    def test_cum_weights(self):
        c = np.array([[0, 1, 2, 3, 4]]).T
        distr = self.coalescence_time_distribution()
        tc = distr.tables[0].cum_weights
        np.testing.assert_allclose(c, tc)

    def test_quantile(self):
        q = np.array([[0, 0.25, 0.50, 0.75, 1]]).T
        distr = self.coalescence_time_distribution()
        tq = distr.tables[0].quantile
        np.testing.assert_allclose(q, tq)


class TestPairWeightedCoalescenceTimeTable(TestCoalescenceTimeDistribution):
    """
                                       Weights
                         (A,A) (A,B) (A,C) (B,B) (B,C) (C,C)
                         -----------------------------------
    29.00┊    9        ┊    1     2     2     0     0     0
         ┊ ┏━━┻━━┓     ┊
    8.00 ┊ ┃     8     ┊    0     1     1     1     2     1
         ┊ ┃   ┏━┻━━┓  ┊
    5.00 ┊ ┃   7    ┃  ┊    0     1     1     0     1     0
         ┊ ┃ ┏━╋━┓  ┃  ┊
    1.00 ┊ ┃ ┃ ┃ ┃  6  ┊    0     0     0     0     1     0
         ┊ ┃ ┃ ┃ ┃ ┏┻┓ ┊
    0.00 ┊ 0 1 2 4 3 5 ┊
         0            100
     Pop.┊ A A B C B C ┊
    Weights are number of pairs of a given population labelling that coalesce
    in node
    """

    def coalescence_time_distribution(self):
        ts = self.ts_multimerger_six_leaves()
        sample_sets = [[0, 1], [2, 3], [4, 5]]
        distr = ts.coalescence_time_distribution(
            sample_sets=sample_sets,
            weight_func="pair_coalescence_events",
            span_normalise=False,
        )
        return distr

    def test_time(self):
        t = np.array([0, 1, 5, 8, 29])
        distr = self.coalescence_time_distribution()
        tt = distr.tables[0].time
        np.testing.assert_allclose(t, tt)

    def test_block(self):
        b = np.array([0, 0, 0, 0, 0])
        distr = self.coalescence_time_distribution()
        tb = distr.tables[0].block
        np.testing.assert_allclose(b, tb)

    def test_weights(self):
        w = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 1, 1, 0, 1, 0],
                [0, 1, 1, 1, 2, 1],
                [1, 2, 2, 0, 0, 0],
            ]
        )
        distr = self.coalescence_time_distribution()
        tw = distr.tables[0].weights
        np.testing.assert_allclose(w, tw)

    def test_cum_weights(self):
        c = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 1, 1, 0, 2, 0],
                [0, 2, 2, 1, 4, 1],
                [1, 4, 4, 1, 4, 1],
            ]
        )
        distr = self.coalescence_time_distribution()
        tc = distr.tables[0].cum_weights
        np.testing.assert_allclose(c, tc)

    def test_quantile(self):
        q = np.array(
            [
                [0.0, 0.00, 0.00, 0.00, 0.00, 0.0],
                [0.0, 0.00, 0.00, 0.00, 0.25, 0.0],
                [0.0, 0.25, 0.25, 0.00, 0.50, 0.0],
                [0.0, 0.50, 0.50, 1.00, 1.00, 1.0],
                [1.0, 1.00, 1.00, 1.00, 1.00, 1.0],
            ]
        )
        distr = self.coalescence_time_distribution()
        tq = distr.tables[0].quantile
        np.testing.assert_allclose(q, tq)


class TestTrioFirstWeightedCoalescenceTimeTable(TestCoalescenceTimeDistribution):
    """
                                             Weights
                               AAA AAB AAC ABA ABB ABC ACA ACB ACC
                               -----------------------------------
    10.00┊         13      ┊     0   0   0   0   0   0   0   0   0
         ┊       ┏━━┻━━┓   ┊
    8.00 ┊      12     ┃   ┊     0   0   0   0   0   0   2   1   0
         ┊     ┏━┻━┓   ┃   ┊
    6.00 ┊    11   ┃   ┃   ┊     0   0   0   4   2   2   0   0   0
         ┊  ┏━━╋━┓ ┃   ┃   ┊
    2.00 ┊ 10  ┃ ┃ ┃   9   ┊(10) 0   0   0   0   0   0   2   3   1
         ┊ ┏┻┓ ┃ ┃ ┃  ┏┻━┓ ┊( 9) 0   0   0   2   4   4   0   0   0
    1.00 ┊ ┃ ┃ ┃ ┃ ┃  8  ┃ ┊     1   3   2   0   0   0   0   0   0
         ┊ ┃ ┃ ┃ ┃ ┃ ┏┻┓ ┃ ┊
    0.00 ┊ 0 7 4 5 6 1 2 3 ┊
         0                100  BBA BBB BBC BCA BCB BCC CCA CCB CCC
     Pop.┊ A C B B C A A B ┊   -----------------------------------
    10.00┊         13      ┊     0   0   0   0   0   0   0   0   0 <- removed
         ┊       ┏━━┻━━┓   ┊
    8.00 ┊      12     ┃   ┊     0   0   0   4   2   0   2   1   0
         ┊     ┏━┻━┓   ┃   ┊
    6.00 ┊    11   ┃   ┃   ┊     2   1   1   4   2   2   0   0   0
         ┊  ┏━━╋━┓ ┃   ┃   ┊
    2.00 ┊ 10  ┃ ┃ ┃   9   ┊(10) 0   0   0   0   0   0   0   0   0
         ┊ ┏┻┓ ┃ ┃ ┃  ┏┻━┓ ┊( 9) 0   0   0   0   0   0   0   0   0
    1.00 ┊ ┃ ┃ ┃ ┃ ┃  8  ┃ ┊     0   0   0   0   0   0   0   0   0
         ┊ ┃ ┃ ┃ ┃ ┃ ┏┻┓ ┃ ┊                                     ^empty
    0.00 ┊ 0 7 4 5 6 1 2 3 ┊
    Pop. ┊ A C B B C A A B ┊
    Weights are number of trios of a given population labelling with first coalescence
    in node; shorthand in table columns for newick is ABC = ((A,B):node,C)
    """

    def coalescence_time_distribution(self):
        ts = self.ts_multimerger_eight_leaves()
        sample_sets = [[0, 1, 2], [3, 4, 5], [6, 7]]
        distr = ts.coalescence_time_distribution(
            sample_sets=sample_sets,
            weight_func="trio_first_coalescence_events",
            span_normalise=False,
        )
        return distr

    def test_time(self):
        t = np.array([0.0, 1.0, 2.0, 2.0, 6.0, 8.00])
        distr = self.coalescence_time_distribution()
        tt = distr.tables[0].time
        np.testing.assert_allclose(t, tt)

    def test_block(self):
        b = np.array([0, 0, 0, 0, 0, 0])
        distr = self.coalescence_time_distribution()
        tb = distr.tables[0].block
        np.testing.assert_allclose(b, tb)

    def test_weights(self):
        w = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 2, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 2, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 4, 2, 2, 0, 0, 0, 2, 1, 1, 4, 2, 2, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 4, 2, 0, 2, 1, 0],
            ]
        )
        distr = self.coalescence_time_distribution()
        tw = distr.tables[0].weights
        np.testing.assert_allclose(w, tw)

    def test_cum_weights(self):
        c = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 3, 2, 2, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 3, 2, 2, 4, 4, 2, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 3, 2, 6, 6, 6, 2, 3, 1, 2, 1, 1, 4, 2, 2, 0, 0, 0],
                [1, 3, 2, 6, 6, 6, 4, 4, 1, 2, 1, 1, 8, 4, 2, 2, 1, 0],
            ]
        )
        distr = self.coalescence_time_distribution()
        tc = distr.tables[0].cum_weights
        np.testing.assert_allclose(c, tc)

    def test_quantile(self):
        q = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 3, 2, 2, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 3, 2, 2, 4, 4, 2, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 3, 2, 6, 6, 6, 2, 3, 1, 2, 1, 1, 4, 2, 2, 0, 0],
                [1, 3, 2, 6, 6, 6, 4, 4, 1, 2, 1, 1, 8, 4, 2, 2, 1],
            ],
            dtype="float",
        )
        q /= q[-1, :]
        distr = self.coalescence_time_distribution()
        tq = distr.tables[0].quantile
        np.testing.assert_allclose(q, tq[:, :-1]) and np.all(np.isnan(tq[:, -1]))


class TestSingleBlockCoalescenceTimeTable(TestCoalescenceTimeDistribution):
    """
                                     Cum.
                               Wght  Wght  Qntl
                              -----------------
    1.74┊   7     ┊   7     ┊     2     6  1.00
        ┊ ┏━┻━┓   ┊ ┏━┻━┓   ┊
    0.73┊ ┃   6   ┊ ┃   ┃   ┊     1     4  0.67
        ┊ ┃ ┏━┻┓  ┊ ┃   ┃   ┊
    0.59┊ ┃ ┃  5  ┊ ┃   5   ┊     2     3  0.50
        ┊ ┃ ┃ ┏┻┓ ┊ ┃  ┏┻━┓ ┊
    0.54┊ ┃ ┃ ┃ ┃ ┊ ┃  4  ┃ ┊     1     1  0.17
        ┊ ┃ ┃ ┃ ┃ ┊ ┃ ┏┻┓ ┃ ┊
    0.00┊ 0 1 2 3 ┊ 0 1 2 3 ┊
      0.00      0.88      1.00
    Uniform weights on nodes summed over trees
    """

    def coalescence_time_distribution(self):
        ts = self.ts_two_trees_four_leaves()
        distr = ts.coalescence_time_distribution(
            span_normalise=False,
        )
        return distr

    def test_time(self):
        t = np.array([0.0, 0.54, 0.59, 0.73, 1.74])
        distr = self.coalescence_time_distribution()
        tt = distr.tables[0].time
        np.testing.assert_allclose(t, tt)

    def test_block(self):
        b = np.array([0, 0, 0, 0, 0])
        distr = self.coalescence_time_distribution()
        tb = distr.tables[0].block
        np.testing.assert_allclose(b, tb)

    def test_weights(self):
        w = np.array([[0, 1, 2, 1, 2]]).T
        distr = self.coalescence_time_distribution()
        tw = distr.tables[0].weights
        np.testing.assert_allclose(w, tw)

    def test_cum_weights(self):
        c = np.array([[0, 1, 3, 4, 6]]).T
        distr = self.coalescence_time_distribution()
        tc = distr.tables[0].cum_weights
        np.testing.assert_allclose(c, tc)
        np.testing.assert_allclose(c, tc)

    def test_quantile(self):
        q = np.array([[0.0, 1 / 6, 3 / 6, 4 / 6, 1.0]]).T
        distr = self.coalescence_time_distribution()
        tq = distr.tables[0].quantile
        np.testing.assert_allclose(q, tq)


class TestWindowedCoalescenceTimeTable(TestCoalescenceTimeDistribution):
    """
      0.00  0.50          1.00    Window 0          Window 1
    Wndw┊  0  ┊        1    ┊ Time  Wght  Blck  Time  Wght  Blck
    Blck┊  0  ┊ 0 ┊    1    ┊ ----------------  ----------------
    1.74┊   7     ┊   7     ┊ 1.74     1     0  1.74     1     1
        ┊ ┏━┻━┓   ┊ ┏━┻━┓   ┊                   1.74     1     0
    0.73┊ ┃   6   ┊ ┃   ┃   ┊ 0.73     1     0  0.73     1     0
        ┊ ┃ ┏━┻┓  ┊ ┃   ┃   ┊
    0.59┊ ┃ ┃  5  ┊ ┃   5   ┊ 0.59     1     0  0.59     1     1
        ┊ ┃ ┃ ┏┻┓ ┊ ┃  ┏┻━┓ ┊                   0.59     1     0
    0.54┊ ┃ ┃ ┃ ┃ ┊ ┃  4  ┃ ┊                   0.54     1     1
        ┊ ┃ ┃ ┃ ┃ ┊ ┃ ┏┻┓ ┃ ┊
    0.00┊ 0 1 2 3 ┊ 0 1 2 3 ┊ 0.00     0     0  0.00     0     0 <to catch OOR
      0.00      0.88      1.00
    Uniform weights on nodes summed over trees in blocks
    """

    @tests.cached_example
    def coalescence_time_distribution(self):
        ts = self.ts_two_trees_four_leaves()
        gen_breaks = np.array([0.0, 0.5, 1.0])
        distr = ts.coalescence_time_distribution(
            window_breaks=gen_breaks,
            blocks_per_window=2,
            span_normalise=False,
        )
        return distr

    def test_time(self):
        t1 = np.array([0.0, 0.59, 0.73, 1.74])
        t2 = np.array([0.0, 0.54, 0.59, 0.59, 0.73, 1.74, 1.74])
        distr = self.coalescence_time_distribution()
        tt1 = distr.tables[0].time
        tt2 = distr.tables[1].time
        np.testing.assert_allclose(t1, tt1)
        np.testing.assert_allclose(t2, tt2)

    def test_block(self):
        b1 = np.array([0, 0, 0, 0])
        b2 = np.array([0, 1, 0, 1, 0, 0, 1])
        distr = self.coalescence_time_distribution()
        tb1 = distr.tables[0].block
        tb2 = distr.tables[1].block
        np.testing.assert_allclose(b1, tb1)
        np.testing.assert_allclose(b2, tb2)

    def test_weights(self):
        w1 = np.array([[0, 1, 1, 1]]).T
        w2 = np.array([[0, 1, 1, 1, 1, 1, 1]]).T
        distr = self.coalescence_time_distribution()
        tw1 = distr.tables[0].weights
        tw2 = distr.tables[1].weights
        np.testing.assert_allclose(w1, tw1)
        np.testing.assert_allclose(w2, tw2)

    def test_cum_weights(self):
        c1 = np.array([[0, 1, 2, 3]]).T
        c2 = np.array([[0, 1, 2, 3, 4, 5, 6]]).T
        distr = self.coalescence_time_distribution()
        tc1 = distr.tables[0].cum_weights
        tc2 = distr.tables[1].cum_weights
        np.testing.assert_allclose(c1, tc1)
        np.testing.assert_allclose(c2, tc2)

    def test_quantile(self):
        e1 = np.array([[0.0, 1 / 3, 2 / 3, 1.0]]).T
        e2 = np.array([[0.0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1.0]]).T
        distr = self.coalescence_time_distribution()
        te1 = distr.tables[0].quantile
        te2 = distr.tables[1].quantile
        np.testing.assert_allclose(e1, te1)
        np.testing.assert_allclose(e2, te2)


class TestCoalescenceTimeDistributionPointMethods(TestCoalescenceTimeDistribution):
    """
                                    Cum.
                              Time  Wght  ECDF   Coal.  Uncoal.
                              ---------------------------------
        ┊         ┊         ┊ 2.00        1.00       6        0
    1.74┊   7     ┊   7     ┊ 1.74     6  1.00       6        0
        ┊ ┏━┻━┓   ┊ ┏━┻━┓   ┊ 1.00        0.67       4        2
    0.73┊ ┃   6   ┊ ┃   ┃   ┊ 0.73     4  0.67       4        2
        ┊ ┃ ┏━┻┓  ┊ ┃   ┃   ┊ 0.65        0.50       3        3
    0.59┊ ┃ ┃  5  ┊ ┃   5   ┊ 0.59     3  0.50       3        3
        ┊ ┃ ┃ ┏┻┓ ┊ ┃  ┏┻━┓ ┊ 0.57        0.17       1        5
    0.54┊ ┃ ┃ ┃ ┃ ┊ ┃  4  ┃ ┊ 0.54     1  0.17       1        5
        ┊ ┃ ┃ ┃ ┃ ┊ ┃ ┏┻┓ ┃ ┊ 0.25        0.00       0        6
    0.00┊ 0 1 2 3 ┊ 0 1 2 3 ┊ 0.00        0.00       0        6
      0.00      0.88      1.00
    Uniform weights on nodes summed over trees
    """

    @tests.cached_example
    def coalescence_time_distribution(self):
        ts = self.ts_two_trees_four_leaves()
        distr = ts.coalescence_time_distribution(
            span_normalise=False,
        )
        return distr

    def test_ecdf(self):
        e = np.array(
            [0.0, 0.0, 1 / 6, 1 / 6, 3 / 6, 3 / 6, 4 / 6, 4 / 6, 1.0, 1.0],
        ).reshape(1, 10, 1)
        distr = self.coalescence_time_distribution()
        et = distr.tables[0].time
        t = np.array(
            [0.0, 0.25, et[1], 0.57, et[2], 0.65, et[3], 1.00, et[4], 2.00],
        )
        te = distr.ecdf(t)
        np.testing.assert_allclose(e, te)

    def test_num_coalesced(self):
        c = np.array([0, 0, 1, 1, 3, 3, 4, 4, 6, 6]).reshape(1, 10, 1)
        distr = self.coalescence_time_distribution()
        et = distr.tables[0].time
        t = np.array(
            [0.0, 0.25, et[1], 0.57, et[2], 0.65, et[3], 1.00, et[4], 2.00],
        )
        tc = distr.num_coalesced(t)
        np.testing.assert_allclose(c, tc)

    def test_num_uncoalesced(self):
        u = np.array([6, 6, 5, 5, 3, 3, 2, 2, 0, 0]).reshape(1, 10, 1)
        distr = self.coalescence_time_distribution()
        et = distr.tables[0].time
        t = np.array(
            [0.0, 0.25, et[1], 0.57, et[2], 0.65, et[3], 1.00, et[4], 2.00],
        )
        tu = distr.num_uncoalesced(t)
        np.testing.assert_allclose(u, tu)

    def test_interpolated_quantile(self):
        x = np.array(
            [
                0.54,
                0.558,
                0.576,
                0.5993,
                0.6413,
                0.6833,
                0.7253,
                0.9609,
                1.2206,
                1.4803,
                1.74,
            ]
        ).reshape(1, 11, 1)
        distr = self.coalescence_time_distribution()
        q = np.linspace(0, 1, 11)
        qx = distr.quantile(q).round(4)
        np.testing.assert_allclose(x, qx)


class TestCoalescenceTimeDistributionIntervalMethods(TestCoalescenceTimeDistribution):
    """
                               Time                     Kaplan-Meier
                              Breaks   Coal.Prop.        Coal. Rate
        ┊         ┊         ┊--3.00--  ----------  ----------------------
        ┊         ┊         ┊             NaN               NaN
        ┊         ┊         ┊--2.00--  ----------  ----------------------
    1.74┊   7     ┊   7     ┊             2/2          2/(2*1.74-2*0.73)
        ┊ ┏━┻━┓   ┊ ┏━┻━┓   ┊
    0.73┊ ┃   6   ┊ ┃   ┃   ┊--0.73--  ----------  ----------------------
        ┊ ┃ ┏━┻┓  ┊ ┃   ┃   ┊             3/5      log(1-3/5)/(0.55-0.73)
    0.59┊ ┃ ┃  5  ┊ ┃   5   ┊
        ┊ ┃ ┃ ┏┻┓ ┊ ┃  ┏┻━┓ ┊--0.55--  ----------  ----------------------
    0.54┊ ┃ ┃ ┃ ┃ ┊ ┃  4  ┃ ┊             1/6      log(1-1/6)/(0.00-0.55)
        ┊ ┃ ┃ ┃ ┃ ┊ ┃ ┏┻┓ ┃ ┊
    0.00┊ 0 1 2 3 ┊ 0 1 2 3 ┊--0.00--  ----------  ----------------------
      0.00      0.88      1.00
                               Time   Trunc. Trunc. Trunc.
                             Lo.Bound  Mean   Mean   Mean
        ┊         ┊         ┊-------- ------ ------ ------
        ┊         ┊         ┊                         NaN
        ┊         ┊         ┊--2.00--               ------
    1.74┊   7     ┊   7     ┊
        ┊ ┏━┻━┓   ┊ ┏━┻━┓   ┊
    0.73┊ ┃   6   ┊ ┃   ┃   ┊
        ┊ ┃ ┏━┻┓  ┊ ┃   ┃   ┊                0.4868
    0.59┊ ┃ ┃  5  ┊ ┃   5   ┊--0.59--        ------
        ┊ ┃ ┃ ┏┻┓ ┊ ┃  ┏┻━┓ ┊
    0.54┊ ┃ ┃ ┃ ┃ ┊ ┃  4  ┃ ┊
        ┊ ┃ ┃ ┃ ┃ ┊ ┃ ┏┻┓ ┃ ┊         0.9893
    0.00┊ 0 1 2 3 ┊ 0 1 2 3 ┊--0.00-- ------
      0.00      0.88      1.00
    Uniform weights on nodes summed over trees, in half-closed time intervals
    (lower, upper]
    """

    def coalescence_time_distribution(self):
        ts = self.ts_two_trees_four_leaves()
        distr = ts.coalescence_time_distribution(
            span_normalise=False,
        )
        return distr

    def test_coalescence_probability_in_intervals(self):
        p = np.array([[[1 / 6], [3 / 5], [2 / 2]]])
        distr = self.coalescence_time_distribution()
        et = distr.tables[0].time
        t = np.array([0.00, 0.55, et[3], 2.00])
        tp = distr.coalescence_probability_in_intervals(t)
        np.testing.assert_allclose(p, tp)

    def test_coalescence_probability_in_intervals_oor(self):
        distr = self.coalescence_time_distribution()
        t = np.array([2.00, 3.00])
        tp = distr.coalescence_probability_in_intervals(t)
        assert np.all(np.isnan(tp))

    def test_coalescence_rate_in_intervals(self):
        c = np.array([[[0.3314937], [5.090504], [0.990099]]])
        distr = self.coalescence_time_distribution()
        et = distr.tables[0].time
        t = np.array([0.00, 0.55, et[3], 2.00])
        tc = distr.coalescence_rate_in_intervals(t)
        np.testing.assert_allclose(c, tc, atol=1e-6)

    def test_coalescence_rate_in_intervals_oor(self):
        distr = self.coalescence_time_distribution()
        t = np.array([2.00, 3.00])
        tc = distr.coalescence_rate_in_intervals(t)
        assert np.all(np.isnan(tc))

    def test_mean(self):
        m = np.array([[0.8133333]])
        distr = self.coalescence_time_distribution()
        et = distr.tables[0].time
        tm = distr.mean(et[2])
        np.testing.assert_allclose(m, tm)

    def test_mean_oor(self):
        distr = self.coalescence_time_distribution()
        tm = distr.mean(10.0)
        assert np.all(np.isnan(tm))


class TestCoalescenceTimeDistributionBootstrap(TestCoalescenceTimeDistribution):
    """
      0.00  0.50          1.00
    Wndw┊  0  ┊        1    ┊       Window 0            Window 1
    Blck┊  0  ┊ 0 ┊    1    ┊ Time B.Wt*Wght Blck  Time B.Wt*Wght Blck
    B.Wt┊  1  ┊ 0 ┊    2    ┊ -------------------  -------------------
    1.74┊   7     ┊   7     ┊ 1.74         1    0  1.74         2    1
        ┊ ┏━┻━┓   ┊ ┏━┻━┓   ┊                      1.74         0    0
    0.73┊ ┃   6   ┊ ┃   ┃   ┊ 0.73         1    0  0.73         0    0
        ┊ ┃ ┏━┻┓  ┊ ┃   ┃   ┊
    0.59┊ ┃ ┃  5  ┊ ┃   5   ┊ 0.59         1    0  0.59         2    1
        ┊ ┃ ┃ ┏┻┓ ┊ ┃  ┏┻━┓ ┊                      0.59         0    0
    0.54┊ ┃ ┃ ┃ ┃ ┊ ┃  4  ┃ ┊   Mean time: 1.02    0.54         2    1
        ┊ ┃ ┃ ┃ ┃ ┊ ┃ ┏┻┓ ┃ ┊
    0.00┊ 0 1 2 3 ┊ 0 1 2 3 ┊                        Mean time: 0.96
      0.00      0.88      1.00
    Wndw┊  0  ┊        1    ┊     Window 0        Window 1
    Blck┊  0  ┊ 0 ┊    1    ┊ Time C.Wgt ECDF  Time C.Wgt ECDF
    B.Wt┊  1  ┊ 2 ┊    0    ┊ ---------------  ---------------
    1.74┊   7     ┊   7     ┊ 1.74     3 1.00  1.74     6 1.00
        ┊ ┏━┻━┓   ┊ ┏━┻━┓   ┊                  1.74     4      <- edcf() skips
    0.73┊ ┃   6   ┊ ┃   ┃   ┊ 0.73     2 0.67  0.73     4 0.67
        ┊ ┃ ┏━┻┓  ┊ ┃   ┃   ┊
    0.59┊ ┃ ┃  5  ┊ ┃   5   ┊ 0.59     1 0.33  0.59     4 0.67
        ┊ ┃ ┃ ┏┻┓ ┊ ┃  ┏┻━┓ ┊                  0.59     2      <- edcf() skips
    0.54┊ ┃ ┃ ┃ ┃ ┊ ┃  4  ┃ ┊                  0.54     2 0.33
        ┊ ┃ ┃ ┃ ┃ ┊ ┃ ┏┻┓ ┃ ┊
    0.00┊ 0 1 2 3 ┊ 0 1 2 3 ┊
      0.00      0.88      1.00
    Uniform weights on nodes summed over trees in blocks, reweighted by block
    weights that are multinomial RVs. Only cumulative weights are affected,
    so that bootstrapping a bootstrap replicate is equivalent to bootstrapping
    the original.
    """

    def coalescence_time_distribution_boot(self):
        ts = self.ts_two_trees_four_leaves()
        gen_breaks = np.array([0.0, 0.5, 1.0])
        distr = ts.coalescence_time_distribution(
            window_breaks=gen_breaks,
            blocks_per_window=2,
            span_normalise=False,
        )
        boot_distr = next(distr.block_bootstrap(1, 3))
        return boot_distr

    def test_cum_weights(self):
        w1 = np.array([[0, 1, 2, 3]]).T
        w2 = np.array([[0, 2, 2, 4, 4, 4, 6]]).T
        boot_distr = self.coalescence_time_distribution_boot()
        tw1 = boot_distr.tables[0].cum_weights
        tw2 = boot_distr.tables[1].cum_weights
        np.testing.assert_allclose(w1, tw1)
        np.testing.assert_allclose(w2, tw2)

    def test_ecdf(self):
        e = np.array(
            [
                [0 / 3, 0 / 3, 1 / 3, 1 / 3, 2 / 3, 2 / 3, 3 / 3],
                [2 / 6, 2 / 6, 4 / 6, 4 / 6, 4 / 6, 4 / 6, 6 / 6],
            ]
        ).T.reshape(1, 7, 2)
        boot_distr = self.coalescence_time_distribution_boot()
        t = np.array([0.54, 0.55, 0.59, 0.60, 0.73, 0.74, 1.74])
        te = boot_distr.ecdf(t)
        np.testing.assert_allclose(e, te)

    def test_mean(self):
        m = np.array([[1.02, 0.9566667]])
        boot_distr = self.coalescence_time_distribution_boot()
        tm = boot_distr.mean()
        np.testing.assert_allclose(m, tm)

    def test_boot_of_boot_equivalence(self):
        boot_distr = self.coalescence_time_distribution_boot()
        reboot_distr = next(boot_distr.block_bootstrap(1, 3))
        cw1 = boot_distr.tables[1].cum_weights
        cw2 = reboot_distr.tables[1].cum_weights
        np.testing.assert_allclose(cw1, cw2)


class TestCoalescenceTimeDistributionEmpty(TestCoalescenceTimeDistribution):
    """
    When weights are all zero ECDF table is empty, methods return NaN
    """

    def coalescence_time_distribution(self):
        ts = self.ts_two_trees_four_leaves()

        def null_weight_init(node, sample_sets):
            blank = np.array([[0, 0]], dtype=np.float64)
            return (blank,)

        def null_weight_update(blank):
            blank = np.array([[0, 0]], dtype=np.float64)
            return blank, (blank,)

        distr = ts.coalescence_time_distribution(
            weight_func=(null_weight_init, null_weight_update),
            span_normalise=False,
        )
        return distr

    def test_ecdf(self):
        distr = self.coalescence_time_distribution()
        t = np.array([0.0, 0.5, 1.0])
        te = distr.ecdf(t)
        assert np.all(np.isnan(te))

    def test_num_coalesced(self):
        distr = self.coalescence_time_distribution()
        t = np.array([0.0, 0.5, 1.0])
        tc = distr.num_coalesced(t)
        assert np.all(np.isnan(tc))

    def test_num_uncoalesced(self):
        distr = self.coalescence_time_distribution()
        t = np.array([0.0, 0.5, 1.0])
        tu = distr.num_uncoalesced(t)
        assert np.all(np.isnan(tu))

    def test_mean(self):
        distr = self.coalescence_time_distribution()
        tm = distr.mean()
        assert np.all(np.isnan(tm))

    def test_coalescence_probability_in_intervals(self):
        distr = self.coalescence_time_distribution()
        t = np.array([0.0, 0.5, 1.0])
        tp = distr.coalescence_probability_in_intervals(t)
        assert np.all(np.isnan(tp))

    def test_coalescence_rate_in_intervals(self):
        distr = self.coalescence_time_distribution()
        t = np.array([0.0, 0.5, 1.0])
        tc = distr.coalescence_rate_in_intervals(t)
        assert np.all(np.isnan(tc))

    def test_quantile(self):
        distr = self.coalescence_time_distribution()
        t = np.array([0.0, 0.5, 1.0])
        tq = distr.quantile(t)
        assert np.all(np.isnan(tq))

    def test_resample(self):
        distr = self.coalescence_time_distribution()
        boot_distr = next(distr.block_bootstrap(1, 3))
        assert np.all(boot_distr.tables[0].cum_weights == 0)
        assert np.all(np.isnan(boot_distr.tables[0].quantile))


class TestCoalescenceTimeDistributionNullWeight(TestCoalescenceTimeDistribution):
    """
    Test method return behaviour when a particular weight is all zeros but
    table is not empty. Methods should return NaN only for null weight.
    """

    def coalescence_time_distribution(self):
        ts = self.ts_two_trees_four_leaves()

        def half_empty_init(node, sample_sets):
            blank = np.array([[1, 0]], dtype=np.float64)
            return (blank,)

        def half_empty_update(blank):
            blank = np.array([[1, 0]], dtype=np.float64)
            return blank, (blank,)

        distr = ts.coalescence_time_distribution(
            weight_func=(half_empty_init, half_empty_update),
            span_normalise=False,
        )
        return distr

    def test_ecdf(self):
        distr = self.coalescence_time_distribution()
        t = np.array([0.0, 0.5, 1.0])
        te = distr.ecdf(t)
        assert np.all(np.isnan(te[1, :])) and np.all(~np.isnan(te[0, :]))

    def test_num_coalesced(self):
        distr = self.coalescence_time_distribution()
        t = np.array([0.0, 0.5, 1.0])
        tc = distr.num_coalesced(t)
        assert np.all(np.isnan(tc[1, :])) and np.all(~np.isnan(tc[0, :]))

    def test_num_uncoalesced(self):
        distr = self.coalescence_time_distribution()
        t = np.array([0.0, 0.5, 1.0])
        tu = distr.num_uncoalesced(t)
        assert np.all(np.isnan(tu[1, :])) and np.all(~np.isnan(tu[0, :]))

    def test_mean(self):
        distr = self.coalescence_time_distribution()
        tm = distr.mean()
        assert np.isnan(tm[1]) and ~np.isnan(tm[0])

    def test_coalescence_probability_in_intervals(self):
        distr = self.coalescence_time_distribution()
        t = np.array([0.0, 0.5, 1.0])
        tp = distr.coalescence_probability_in_intervals(t)
        assert np.all(np.isnan(tp[1, :])) and np.all(~np.isnan(tp[0, :]))

    def test_coalescence_rate_in_intervals(self):
        distr = self.coalescence_time_distribution()
        t = np.array([0.0, 0.5, 1.0])
        tr = distr.coalescence_rate_in_intervals(t)
        assert np.all(np.isnan(tr[1, :])) and np.all(~np.isnan(tr[0, :]))

    def test_quantile(self):
        distr = self.coalescence_time_distribution()
        t = np.array([0.0, 0.5, 1.0])
        tq = distr.quantile(t)
        assert np.all(np.isnan(tq[1, :])) and np.all(~np.isnan(tq[0, :]))

    def test_resample(self):
        distr = self.coalescence_time_distribution()
        boot_distr = next(distr.block_bootstrap(1, 3))
        assert np.all(boot_distr.tables[0].cum_weights[:, 1] == 0)
        assert np.all(np.isnan(boot_distr.tables[0].quantile[:, 1]))
        assert np.any(boot_distr.tables[0].cum_weights[:, 0] > 0)
        assert np.all(~np.isnan(boot_distr.tables[0].quantile[:, 0]))


class TestCoalescenceTimeDistributionTableResize(TestCoalescenceTimeDistribution):
    """
    If the initial allocation for the table is exceeded, the number of rows is
    increased.
    """

    def coalescence_time_distribution(self):
        ts = self.ts_five_trees_three_leaves()
        distr = ts.coalescence_time_distribution(
            blocks_per_window=ts.num_trees,
            span_normalise=False,
        )
        return distr

    def test_table_resize(self):
        distr = self.coalescence_time_distribution()
        assert distr.tables[0].num_records > distr.buffer_size + 1


class TestCoalescenceTimeDistributionBlocking(TestCoalescenceTimeDistribution):
    """
    Test assignment of blocks per window and trees per block. If window breaks
    fall on recombination breakpoints, and the number of trees is divisible by
    the number of windows, then there should be an equal number of trees per
    window.
    """

    def coalescence_time_distribution(self):
        # 2 trees/block, 2 blocks/window, 2 windows/ts
        ts = self.ts_eight_trees_two_leaves()
        bk = [t.interval.left for t in ts.trees()][::4] + [ts.sequence_length]

        def count_root_init(node, sample_sets):
            all_samples = [i for s in sample_sets for i in s]
            state = np.array([[node == i for i in all_samples]], dtype=np.float64)
            return (state,)

        def count_root_update(child_state):
            state = np.sum(child_state, axis=0, keepdims=True)
            is_root = np.array([[np.all(state > 0)]], dtype=np.float64)
            return is_root, (state,)

        distr = ts.coalescence_time_distribution(
            weight_func=(count_root_init, count_root_update),
            window_breaks=np.array(bk),
            blocks_per_window=2,
            span_normalise=False,
        )
        return distr

    def test_blocks_per_window(self):
        distr = self.coalescence_time_distribution()
        bpw = np.array([i.num_blocks for i in distr.tables])
        np.testing.assert_allclose(bpw, 2)

    def test_trees_per_window(self):
        distr = self.coalescence_time_distribution()
        tpw = np.array([np.sum(distr.tables[i].weights) for i in range(2)])
        np.testing.assert_allclose(tpw, 4)

    def test_trees_per_block(self):
        distr = self.coalescence_time_distribution()
        tpb = []
        for table in distr.tables:
            for block in range(2):
                tpb += [np.sum(table.weights[table.block == block])]
        np.testing.assert_allclose(tpb, 2)


class TestCoalescenceTimeDistributionBlockedVsUnblocked(
    TestCoalescenceTimeDistribution
):
    """
    Test that methods give the same result regardless of how trees are blocked.
    """

    def coalescence_time_distribution(self, num_blocks=1):
        ts = self.ts_many_edge_diffs()
        sample_sets = [list(range(10)), list(range(20, 40)), list(range(70, 80))]
        distr = ts.coalescence_time_distribution(
            sample_sets=sample_sets,
            weight_func="pair_coalescence_events",
            blocks_per_window=num_blocks,
            span_normalise=True,
        )
        return distr

    def test_ecdf(self):
        distr_noblock = self.coalescence_time_distribution(num_blocks=1)
        distr_block = self.coalescence_time_distribution(num_blocks=10)
        t = np.linspace(0, distr_noblock.tables[0].time[-1] + 1, 5)
        np.testing.assert_allclose(distr_noblock.ecdf(t), distr_block.ecdf(t))

    def test_num_coalesced(self):
        distr_noblock = self.coalescence_time_distribution(num_blocks=1)
        distr_block = self.coalescence_time_distribution(num_blocks=10)
        t = np.linspace(0, distr_noblock.tables[0].time[-1] + 1, 5)
        np.testing.assert_allclose(
            distr_noblock.num_coalesced(t), distr_block.num_coalesced(t)
        )

    def test_num_uncoalesced(self):
        distr_noblock = self.coalescence_time_distribution(num_blocks=1)
        distr_block = self.coalescence_time_distribution(num_blocks=10)
        t = np.linspace(0, distr_noblock.tables[0].time[-1] + 1, 5)
        np.testing.assert_allclose(
            distr_noblock.num_uncoalesced(t), distr_block.num_uncoalesced(t)
        )

    def test_quantile(self):
        distr_noblock = self.coalescence_time_distribution(num_blocks=1)
        distr_block = self.coalescence_time_distribution(num_blocks=10)
        q = np.linspace(0, 1, 11)
        np.testing.assert_allclose(distr_noblock.quantile(q), distr_block.quantile(q))

    def test_mean(self):
        distr_noblock = self.coalescence_time_distribution(num_blocks=1)
        distr_block = self.coalescence_time_distribution(num_blocks=10)
        t = distr_noblock.tables[0].time[-1] / 2
        np.testing.assert_allclose(
            distr_noblock.mean(since=t), distr_block.mean(since=t)
        )

    def test_coalescence_rate_in_intervals(self):
        distr_noblock = self.coalescence_time_distribution(num_blocks=1)
        distr_block = self.coalescence_time_distribution(num_blocks=10)
        t = np.linspace(0, distr_noblock.tables[0].time[-1] + 1, 5)
        np.testing.assert_allclose(
            distr_noblock.coalescence_rate_in_intervals(t),
            distr_block.coalescence_rate_in_intervals(t),
        )

    def test_coalescence_probability_in_intervals(self):
        distr_noblock = self.coalescence_time_distribution(num_blocks=1)
        distr_block = self.coalescence_time_distribution(num_blocks=10)
        t = np.linspace(0, distr_noblock.tables[0].time[-1] + 1, 5)
        np.testing.assert_allclose(
            distr_noblock.coalescence_probability_in_intervals(t),
            distr_block.coalescence_probability_in_intervals(t),
        )


class TestCoalescenceTimeDistributionRunningUpdate(TestCoalescenceTimeDistribution):
    """
    When traversing trees, weights are updated for nodes whose descendant subtree
    has changed. This is done by taking the parents of added edges, and tracing
    ancestors down to the root. This class tests that this "running update"
    scheme produces the correct result.
    """

    def coalescence_time_distribution_running(self, ts, brk, sets=2):
        n = ts.num_samples // sets
        smp_set = [list(range(i, i + n)) for i in range(0, ts.num_samples, n)]
        distr = ts.coalescence_time_distribution(
            sample_sets=smp_set,
            window_breaks=brk,
            weight_func="trio_first_coalescence_events",
            span_normalise=False,
        )
        return distr

    def coalescence_time_distribution_split(self, ts, brk, sets=2):
        n = ts.num_samples // sets
        smp_set = [list(range(i, i + n)) for i in range(0, ts.num_samples, n)]
        distr_by_win = []
        for left, right in zip(brk[:-1], brk[1:]):
            ts_trim = ts.keep_intervals([[left, right]]).trim()
            distr_by_win += [
                ts_trim.coalescence_time_distribution(
                    sample_sets=smp_set,
                    weight_func="trio_first_coalescence_events",
                    span_normalise=False,
                )
            ]
        return distr_by_win

    def test_many_edge_diffs(self):
        """
        Test that ts windowed by tree gives same result as set of single trees.
        """
        ts = self.ts_many_edge_diffs()
        brk = np.array([t.interval.left for t in ts.trees()] + [ts.sequence_length])
        distr = self.coalescence_time_distribution_running(ts, brk)
        distr_win = self.coalescence_time_distribution_split(ts, brk)
        time_breaks = np.array([np.inf])
        updt = distr.num_coalesced(time_breaks)
        sepr = np.zeros(updt.shape)
        for i, d in enumerate(distr_win):
            c = d.num_coalesced(time_breaks)
            sepr[:, :, i] = c.reshape((c.shape[0], 1))
        np.testing.assert_allclose(sepr, updt)

    def test_missing_trees(self):
        """
        Test that ts with half of each tree masked gives same result as unmasked ts.
        """
        ts = self.ts_many_edge_diffs()
        brk = np.array([t.interval.left for t in ts.trees()] + [ts.sequence_length])
        mask = np.array(
            [
                [tr.interval.left, (tr.interval.right + tr.interval.left) / 2]
                for tr in ts.trees()
            ]
        )
        ts_mask = ts.delete_intervals(mask)
        distr = self.coalescence_time_distribution_running(ts, brk)
        distr_mask = self.coalescence_time_distribution_running(ts_mask, brk)
        time_breaks = np.array([np.inf])
        updt = distr.num_coalesced(time_breaks)
        updt_mask = distr_mask.num_coalesced(time_breaks)
        np.testing.assert_allclose(updt, updt_mask)

    def test_unary_nodes(self):
        """
        Test that ts with unary nodes gives same result as ts with unary nodes removed.
        """
        ts = self.ts_many_edge_diffs()
        ts_unary = ts.simplify(
            samples=list(range(ts.num_samples // 2)), keep_unary=True
        )
        ts_nounary = ts.simplify(
            samples=list(range(ts.num_samples // 2)), keep_unary=False
        )
        brk = np.array([t.interval.left for t in ts.trees()] + [ts.sequence_length])
        distr_unary = self.coalescence_time_distribution_running(ts_unary, brk)
        distr_nounary = self.coalescence_time_distribution_running(ts_nounary, brk)
        time_breaks = np.array([np.inf])
        updt_unary = distr_unary.num_coalesced(time_breaks)
        updt_nounary = distr_nounary.num_coalesced(time_breaks)
        np.testing.assert_allclose(updt_unary, updt_nounary)


class TestSpanNormalisedCoalescenceTimeTable(TestCoalescenceTimeDistribution):
    """
                                         Cum.
                                   Wght  Wght
                              ---------------
    1.74┊   7     ┊   7     ┊ 0.88+0.12  3.00
        ┊ ┏━┻━┓   ┊ ┏━┻━┓   ┊
    0.73┊ ┃   6   ┊ ┃   ┃   ┊ 0.88       2.00
        ┊ ┃ ┏━┻┓  ┊ ┃   ┃   ┊
    0.59┊ ┃ ┃  5  ┊ ┃   5   ┊ 0.88+0.12  1.12
        ┊ ┃ ┃ ┏┻┓ ┊ ┃  ┏┻━┓ ┊
    0.54┊ ┃ ┃ ┃ ┃ ┊ ┃  4  ┃ ┊      0.12  0.12
        ┊ ┃ ┃ ┃ ┃ ┊ ┃ ┏┻┓ ┃ ┊
    0.00┊ 0 1 2 3 ┊ 0 1 2 3 ┊
      0.00      0.88      1.00
    SpnW┊   0.88  ┊  0.12   ┊
    Uniform weights on nodes summed over trees, weighted by tree span
    """

    def coalescence_time_distribution(self, mask_half_of_each_tree=False):
        """
        Methods should give the same result if half of each tree is masked,
        because "span weights" are normalised using the accessible (nonmissing)
        portion of the tree sequence.
        """
        ts = self.ts_two_trees_four_leaves()
        if mask_half_of_each_tree:
            mask = np.array(
                [
                    [t.interval.left, (t.interval.right + t.interval.left) / 2]
                    for t in ts.trees()
                ]
            )
            ts = ts.delete_intervals(mask)
        distr = ts.coalescence_time_distribution(
            span_normalise=True,
        )
        return distr

    @pytest.mark.parametrize("with_missing_data", [True, False])
    def test_weights(self, with_missing_data):
        w = np.array([[0, 0.12, 1.0, 0.88, 1.0]]).T
        distr = self.coalescence_time_distribution(with_missing_data)
        tw = distr.tables[0].weights
        np.testing.assert_allclose(w, tw)

    @pytest.mark.parametrize("with_missing_data", [True, False])
    def test_cum_weights(self, with_missing_data):
        c = np.array([[0, 0.12, 1.12, 2.00, 3.00]]).T
        distr = self.coalescence_time_distribution(with_missing_data)
        tc = distr.tables[0].cum_weights
        np.testing.assert_allclose(c, tc)


class TestWindowedSpanNormalisedCoalescenceTimeTable(TestCoalescenceTimeDistribution):
    """
      0.00  0.50           1.00  Window 0    Window 1
    Wndw┊  0  ┊         1    ┊  Time  Wght  Time  Wght
    Blck┊  0  ┊ 0  ┊    1    ┊  ----------  ----------
    1.74┊   7      ┊   7     ┊  1.74     1  1.74  0.24
        ┊ ┏━┻━┓    ┊ ┏━┻━┓   ┊              1.74  0.76
    0.73┊ ┃   6    ┊ ┃   ┃   ┊  0.73     1  0.73  0.76
        ┊ ┃ ┏━┻┓   ┊ ┃   ┃   ┊
    0.59┊ ┃ ┃  5   ┊ ┃   5   ┊  0.59     1  0.59  0.24
        ┊ ┃ ┃ ┏┻┓  ┊ ┃  ┏┻━┓ ┊              0.59  0.76
    0.54┊ ┃ ┃ ┃ ┃  ┊ ┃  4  ┃ ┊              0.54  0.24
        ┊ ┃ ┃ ┃ ┃  ┊ ┃ ┏┻┓ ┃ ┊
    0.00┊ 0 1 2 3  ┊ 0 1 2 3 ┊  0.00     0  0.00     0 <to catch OOR
      0.00       0.88      1.00
    Span┊ 0.5 ┊0.38┊   0.12  ┊
    SpnW┊ 1.0 ┊0.76┊   0.24  ┊
    Uniform weights on nodes summed over trees in blocks, normalised
    by span within windows
    """

    @tests.cached_example
    def coalescence_time_distribution(self, mask_half_of_each_tree=False):
        """
        Methods should give the same result if half of each tree is masked,
        because "span weights" are normalised using the accessible (nonmissing)
        portion of the tree sequence.
        """
        ts = self.ts_two_trees_four_leaves()
        gen_breaks = np.array([0.0, 0.5, 1.0])
        if mask_half_of_each_tree:
            breaks = [i for i in ts.breakpoints()]
            breaks = np.unique(np.concatenate([breaks, gen_breaks]))
            mask = np.array([[a, (a + b) / 2] for a, b in zip(breaks[:-1], breaks[1:])])
            ts = ts.keep_intervals(mask)
        distr = ts.coalescence_time_distribution(
            window_breaks=gen_breaks,
            blocks_per_window=2,
            span_normalise=True,
        )
        return distr

    @pytest.mark.parametrize("with_missing_data", [True, False])
    def test_time(self, with_missing_data):
        t1 = np.array([0.0, 0.59, 0.73, 1.74])
        t2 = np.array([0.0, 0.54, 0.59, 0.59, 0.73, 1.74, 1.74])
        distr = self.coalescence_time_distribution(with_missing_data)
        tt1 = distr.tables[0].time
        tt2 = distr.tables[1].time
        np.testing.assert_allclose(t1, tt1)
        np.testing.assert_allclose(t2, tt2)

    @pytest.mark.parametrize("with_missing_data", [True, False])
    def test_block(self, with_missing_data):
        b1 = np.array([0, 0, 0, 0])
        b2 = np.array([0, 1, 0, 1, 0, 0, 1])
        distr = self.coalescence_time_distribution(with_missing_data)
        tb1 = distr.tables[0].block
        tb2 = distr.tables[1].block
        np.testing.assert_allclose(b1, tb1)
        np.testing.assert_allclose(b2, tb2)

    @pytest.mark.parametrize("with_missing_data", [True, False])
    def test_weights(self, with_missing_data):
        w1 = np.array([[0, 1.0, 1.0, 1.0]]).T
        w2 = np.array([[0, 0.24, 0.76, 0.24, 0.76, 0.76, 0.24]]).T
        distr = self.coalescence_time_distribution(with_missing_data)
        tw1 = distr.tables[0].weights
        tw2 = distr.tables[1].weights
        np.testing.assert_allclose(w1, tw1)
        np.testing.assert_allclose(w2, tw2)
