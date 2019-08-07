#
# MIT License
#
# Copyright (c) 2019 Tskit Developers
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
Convert tree sequences to VCF.
"""
import math

import numpy as np

from . import provenance


def write_vcf(ts, output, ploidy=1, contig_id="1"):
    # See TreeSequence.write_vcf for documentation.

    if ploidy < 1:
        raise ValueError("Ploidy must be >= 1")
    if ts.num_samples % ploidy != 0:
        raise ValueError("Sample size must be divisible by ploidy")

    n = ts.sample_size // ploidy
    sample_names = ["msp_{}".format(j) for j in range(n)]

    # This is faster, but not quite behaving the same way as the legacy
    # coordinate conversion.

    # positions = np.round(ts.tables.sites.position).astype(int)
    # zp = (np.diff(positions) == 0)
    # while np.any(zp):
    #     positions[1:] += np.cumsum(zp)
    #     zp = (np.diff(positions) == 0)

    last_pos = 0
    positions = []
    for site in ts.sites():
        pos = int(round(site.position))
        if pos <= last_pos:
            pos = last_pos + 1
        positions.append(pos)
        last_pos = pos
    contig_length = int(math.ceil(ts.sequence_length))
    if len(positions) > 0:
        contig_length = max(positions[-1], contig_length)

    print("##fileformat=VCFv4.2", file=output)
    print("##source=tskit {}".format(provenance.__version__), file=output)
    print('##FILTER=<ID=PASS,Description="All filters passed">', file=output)
    print("##contig=<ID={},length={}>".format(contig_id, contig_length), file=output)
    print('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">', file=output)
    print(
        "#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO",
        "FORMAT", sep="\t", end="", file=output)
    for sample_name in sample_names:
        print("\t", sample_name, sep="", end="", file=output)
    print(file=output)
    gt_array = []
    indexes = []

    # TODO do this less stupidly.
    for j in range(n):
        for j in range(ploidy):
            indexes.append(len(gt_array))
            gt_array.extend([ord("X"), ord("|")])
        gt_array[-1] = ord("\t")
    gt_array[-1] = ord("\n")
    gt_array = np.array(gt_array, dtype=np.int8)
    indexes = np.array(indexes)

    for variant in ts.variants():
        pos = positions[variant.index]
        print(
            contig_id, pos, ".", "A", "T", ".", "PASS", ".", "GT",
            sep="\t", end="\t", file=output)
        # This assumes we're using utf-8. We could get the correct encoding
        # via the file object, presumably.
        variant.genotypes += ord("0")
        gt_array[indexes] = variant.genotypes
        g_bytes = memoryview(gt_array).tobytes()
        g_str = g_bytes.decode()
        print(g_str, end="", file=output)
