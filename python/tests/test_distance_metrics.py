# MIT License
#
# Copyright (c) 2023 Tskit Developers
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
Tests for tree distance metrics.
"""
import itertools

import msprime
import pytest

import tskit


class TestDistanceBetween:
    @pytest.mark.parametrize(
        ("u", "v"),
        itertools.combinations([0, 1, 2, 3], 2),
    )
    def test_distance_between_sample(self, u, v):
        ts = msprime.sim_ancestry(
            2, sequence_length=10, recombination_rate=0.1, random_seed=42
        )
        test_tree = ts.first()
        assert test_tree.distance_between(u, v) == pytest.approx(
            ts.diversity([u, v], mode="branch", windows="trees")[0]
        )
    
    def test_distance_between_same_node(self):
        ts = msprime.sim_ancestry(2, sequence_length=10, recombination_rate=0.1, random_seed=42)
        test_tree = ts.first()
        assert test_tree.distance_between(0, 0) == 0

    def test_distance_between_nodes(self):
        # 4.00┊   8       ┊
        #     ┊ ┏━┻━┓     ┊
        # 3.00┊ ┃   7     ┊
        #     ┊ ┃ ┏━┻━┓   ┊
        # 2.00┊ ┃ ┃   6   ┊
        #     ┊ ┃ ┃ ┏━┻┓  ┊
        # 1.00┊ ┃ ┃ ┃  5  ┊
        #     ┊ ┃ ┃ ┃ ┏┻┓ ┊
        # 0.00┊ 0 1 2 3 4 ┊
        #     0           1
        ts = tskit.Tree.generate_comb(5)
        assert ts.distance_between(1, 7) == 3.0
        assert ts.distance_between(6, 8) == 2.0

    def test_distance_between_invalid_nodes(self):
        ts = tskit.Tree.generate_comb(5)
        with pytest.raises(ValueError):
            ts.distance_between(0, 100)