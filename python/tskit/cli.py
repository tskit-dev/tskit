#
# MIT License
#
# Copyright (c) 2018-2019 Tskit Developers
# Copyright (c) 2015-2018 University of Oxford
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
Command line utilities for tskit.
"""
import argparse
import json
import os
import signal
import sys

import tskit


def set_sigpipe_handler():
    if os.name == "posix":
        # Set signal handler for SIGPIPE to quietly kill the program.
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)


def exit(message):
    sys.exit(message)


def load_tree_sequence(path):
    try:
        return tskit.load(path)
    except tskit.FileFormatError as e:
        exit("Load error: {}".format(e))


def run_info(args):
    ts = load_tree_sequence(args.tree_sequence)
    print("sequence_length: ", ts.sequence_length)
    print("trees:           ", ts.num_trees)
    print("samples:         ", ts.num_samples)
    print("individuals:     ", ts.num_individuals)
    print("nodes:           ", ts.num_nodes)
    print("edges:           ", ts.num_edges)
    print("sites:           ", ts.num_sites)
    print("mutations:       ", ts.num_mutations)
    print("migrations:      ", ts.num_migrations)
    print("populations:     ", ts.num_populations)
    print("provenances:     ", ts.num_provenances)


def run_upgrade(args):
    try:
        tree_sequence = tskit.load_legacy(args.source, args.remove_duplicate_positions)
        tree_sequence.dump(args.destination)
    except tskit.DuplicatePositionsError:
        exit(
            "Error: Duplicate mutation positions in the source file detected.\n\n"
            "This is not supported in the current file format. Running \"upgrade -d\" "
            "will remove these duplicate positions. However, this will result in loss "
            "of data from the original file!")


def run_dump_nodes(args):
    tree_sequence = load_tree_sequence(args.tree_sequence)
    tree_sequence.dump_text(nodes=sys.stdout, precision=args.precision)


def run_dump_edges(args):
    tree_sequence = load_tree_sequence(args.tree_sequence)
    tree_sequence.dump_text(edges=sys.stdout, precision=args.precision)


def run_dump_sites(args):
    tree_sequence = load_tree_sequence(args.tree_sequence)
    tree_sequence.dump_text(sites=sys.stdout, precision=args.precision)


def run_dump_mutations(args):
    tree_sequence = load_tree_sequence(args.tree_sequence)
    tree_sequence.dump_text(mutations=sys.stdout, precision=args.precision)


def run_dump_provenances(args):
    tree_sequence = load_tree_sequence(args.tree_sequence)
    if args.human:
        for provenance in tree_sequence.provenances():
            d = json.loads(provenance.record)
            print("id={}, timestamp={}, record={}".format(
                provenance.id, provenance.timestamp, json.dumps(d, indent=4)))
    else:
        tree_sequence.dump_text(provenances=sys.stdout)


def run_dump_vcf(args):
    tree_sequence = load_tree_sequence(args.tree_sequence)
    tree_sequence.write_vcf(sys.stdout, args.ploidy)


def add_tree_sequence_argument(parser):
    parser.add_argument(
        "tree_sequence", help="The tskit tree sequence file")


def add_precision_argument(parser):
    parser.add_argument(
        "--precision", "-p", type=int, default=6,
        help="The number of decimal places to print in records")


def get_tskit_parser():
    top_parser = argparse.ArgumentParser(
        description="Command line interface for tskit.")
    top_parser.add_argument(
        "-V", "--version", action='version',
        version='%(prog)s {}'.format(tskit.__version__))
    # This is required to get uniform behaviour in Python2 and Python3
    subparsers = top_parser.add_subparsers(dest="subcommand")
    subparsers.required = True

    parser = subparsers.add_parser(
        "info",
        help="Print summary information about a tree sequence.")
    add_tree_sequence_argument(parser)
    parser.set_defaults(runner=run_info)

    parser = subparsers.add_parser(
        "upgrade",
        help="Upgrade legacy tree sequence files to the latest version.")
    parser.add_argument(
        "source", help="The source tskit tree sequence file in legacy format")
    parser.add_argument(
        "destination", help="The filename of the upgraded copy.")
    parser.add_argument(
        "--remove-duplicate-positions", "-d", action="store_true", default=False,
        help="Remove any duplicated mutation positions in the source file. ")
    parser.set_defaults(runner=run_upgrade)

    parser = subparsers.add_parser(
        "vcf",
        help="Convert the tree sequence genotypes to VCF format.")
    add_tree_sequence_argument(parser)
    parser.add_argument(
        "--ploidy", "-P", type=int, default=1,
        help="The ploidy level of samples")
    parser.set_defaults(runner=run_dump_vcf)

    parser = subparsers.add_parser(
        "nodes",
        help="Output nodes in tabular format.")
    add_tree_sequence_argument(parser)
    add_precision_argument(parser)
    parser.set_defaults(runner=run_dump_nodes)

    parser = subparsers.add_parser(
        "edges",
        help="Output edges in tabular format.")
    add_tree_sequence_argument(parser)
    add_precision_argument(parser)
    parser.set_defaults(runner=run_dump_edges)

    parser = subparsers.add_parser(
        "sites",
        help="Output sites in tabular format.")
    add_tree_sequence_argument(parser)
    add_precision_argument(parser)
    parser.set_defaults(runner=run_dump_sites)

    parser = subparsers.add_parser(
        "mutations",
        help="Output mutations in tabular format.")
    add_tree_sequence_argument(parser)
    add_precision_argument(parser)
    parser.set_defaults(runner=run_dump_mutations)

    parser = subparsers.add_parser(
        "provenances",
        help="Output provenance information in tabular format.")
    add_tree_sequence_argument(parser)
    parser.add_argument(
        "-H", "--human", action="store_true",
        help="Print out the provenances in a human readable format")
    parser.set_defaults(runner=run_dump_provenances)

    return top_parser


def tskit_main(arg_list=None):
    set_sigpipe_handler()
    parser = get_tskit_parser()
    args = parser.parse_args(arg_list)
    args.runner(args)
