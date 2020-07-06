# MIT License
#
# Copyright (c) 2018-2019 Tskit Developers
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
import _tskit

#: Special reserved value representing a null ID.
NULL = _tskit.NULL

#: Special value representing missing data in a genotype array
MISSING_DATA = _tskit.MISSING_DATA

#: Node flag value indicating that it is a sample.
NODE_IS_SAMPLE = _tskit.NODE_IS_SAMPLE

#: Constant representing the forward direction of travel (i.e.,
#: increasing genomic coordinate values).
FORWARD = _tskit.FORWARD

#: Constant representing the reverse direction of travel (i.e.,
#: decreasing genomic coordinate values).
REVERSE = _tskit.REVERSE

#: The allele mapping where the strings "0" and "1" map to genotype
#: values 0 and 1.
ALLELES_01 = ("0", "1")

#: The allele mapping where the four nucleotides A, C, G and T map to
#: the genotype integers 0, 1, 2, and 3, respectively.
ALLELES_ACGT = ("A", "C", "G", "T")

#: Special NAN value used to indicate unknown mutation times
"""
Say what
"""
UNKNOWN_TIME = _tskit.UNKNOWN_TIME

from tskit.provenance import __version__  # NOQA
from tskit.provenance import validate_provenance  # NOQA
from tskit.formats import *  # NOQA
from tskit.trees import *  # NOQA
from tskit.tables import *  # NOQA
from tskit.stats import *  # NOQA
from tskit.combinatorics import (  # NOQA
    all_trees,
    all_tree_shapes,
    all_tree_labellings,
    TopologyCounter,
)
from tskit.exceptions import *  # NOQA
from tskit.util import *  # NOQA
from tskit.metadata import *  # NOQA
