# MIT License
#
# Copyright (c) 2018-2020 Tskit Developers
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
Test cases for the low-level dictionary encoding used to move
data around in C.
"""
import pathlib
import pickle

import _tskit
import lwt_interface.dict_encoding_testlib
import tskit


lwt_interface.dict_encoding_testlib.lwt_module = _tskit
# Bring the tests defined in dict_encoding_testlib into the current namespace
# so pytest will find and execute them.
from lwt_interface.dict_encoding_testlib import *  # noqa


def test_pickled_examples():
    seen_msprime = False
    test_dir = pathlib.Path(__file__).parent / "data/dict-encodings"
    for filename in test_dir.glob("*.pkl"):
        if "msprime" in str(filename):
            seen_msprime = True
        with open(test_dir / filename, "rb") as f:
            d = pickle.load(f)
            lwt = _tskit.LightweightTableCollection()
            lwt.fromdict(d)
            tskit.TableCollection.fromdict(d)
    # Check we've done something
    assert seen_msprime
