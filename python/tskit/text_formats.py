# MIT License
#
# Copyright (c) 2021-2022 Tskit Developers
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
Module responsible for working with text format data.
"""
import base64

import numpy as np

import tskit
from tskit import util


def parse_fam(fam_file):
    """
    Parse PLINK .fam file and convert to tskit IndividualTable.

    Assumes fam file contains five columns: FID, IID, PAT, MAT, SEX

    :param fam_file: PLINK .fam file object
    :param tskit.TableCollection tc: TableCollection with IndividualTable to
        which the individuals will be added
    """
    individuals = np.loadtxt(
        fname=fam_file,
        dtype=str,
        ndmin=2,  # read file as 2-D table
        usecols=(0, 1, 2, 3, 4),  # only keep FID, IID, PAT, MAT, SEX columns
    )  # requires same number of columns in each row, i.e. not ragged

    id_map = {}  # dict for translating PLINK ID to tskit IndividualTable ID
    for tskit_id, (plink_fid, plink_iid, _pat, _mat, _sex) in enumerate(individuals):
        # include space between strings to ensure uniqueness
        plink_id = f"{plink_fid} {plink_iid}"
        if plink_id in id_map:
            raise ValueError("Duplicate PLINK ID: {plink_id}")
        id_map[plink_id] = tskit_id
    id_map["0"] = -1  # -1 is used in tskit to denote "missing"

    tc = tskit.TableCollection(1)
    tb = tc.individuals
    tb.metadata_schema = tskit.MetadataSchema(
        {
            "codec": "json",
            "type": "object",
            "properties": {
                "plink_fid": {"type": "string"},
                "plink_iid": {"type": "string"},
                "sex": {"type": "integer"},
            },
            "required": ["plink_fid", "plink_iid", "sex"],
            "additionalProperties": True,
        }
    )
    for plink_fid, plink_iid, pat, mat, sex in individuals:
        sex = int(sex)
        if not (sex in range(3)):
            raise ValueError(
                "Sex must be one of the following: 0 (unknown), 1 (male), 2 (female)"
            )
        metadata_dict = {"plink_fid": plink_fid, "plink_iid": plink_iid, "sex": sex}
        pat_id = f"{plink_fid} {pat}" if pat != "0" else pat
        mat_id = f"{plink_fid} {mat}" if mat != "0" else mat
        tb.add_row(
            parents=[
                id_map[pat_id],
                id_map[mat_id],
            ],
            metadata=metadata_dict,
        )
    tc.sort()

    return tb


def flexible_file_output(ts_export_func):
    """
    Decorator to support writing to either an open file-like object
    or to a path. Assumes the second argument is the output.
    """

    def f(ts, file_or_path, **kwargs):
        file, local_file = util.convert_file_like_to_open_file(file_or_path, "w")
        try:
            ts_export_func(ts, file, **kwargs)
        finally:
            if local_file:
                file.close()

    return f


@flexible_file_output
def write_nexus(
    ts,
    out,
    *,
    precision,
    include_trees,
    include_alignments,
    reference_sequence,
    missing_data_character,
):
    # See TreeSequence.write_nexus for documentation on parameters.
    if precision is None:
        pos_precision = 0 if ts.discrete_genome else 17
        time_precision = None
    else:
        pos_precision = precision
        time_precision = precision

    indent = "  "
    print("#NEXUS", file=out)
    print("BEGIN TAXA;", file=out)
    print("", f"DIMENSIONS NTAX={ts.num_samples};", sep=indent, file=out)
    taxlabels = " ".join(f"n{u}" for u in ts.samples())
    print("", f"TAXLABELS {taxlabels};", sep=indent, file=out)
    print("END;", file=out)

    if include_alignments is None:
        include_alignments = ts.discrete_genome and ts.num_sites > 0
    if include_alignments:
        missing_data_character = (
            "?" if missing_data_character is None else missing_data_character
        )
        print("BEGIN DATA;", file=out)
        print("", f"DIMENSIONS NCHAR={int(ts.sequence_length)};", sep=indent, file=out)
        print(
            "",
            f"FORMAT DATATYPE=DNA MISSING={missing_data_character};",
            sep=indent,
            file=out,
        )
        print("", "MATRIX", file=out, sep=indent)
        alignments = ts.alignments(
            reference_sequence=reference_sequence,
            missing_data_character=missing_data_character,
        )
        for u, alignment in zip(ts.samples(), alignments):
            print(2 * indent, f"n{u}", " ", alignment, sep="", file=out)
        print("", ";", sep=indent, file=out)
        print("END;", file=out)

    include_trees = True if include_trees is None else include_trees
    if include_trees:
        print("BEGIN TREES;", file=out)
        for tree in ts.trees():
            start_interval = "{0:.{1}f}".format(tree.interval.left, pos_precision)
            end_interval = "{0:.{1}f}".format(tree.interval.right, pos_precision)
            tree_label = f"t{start_interval}^{end_interval}"
            newick = tree.as_newick(precision=time_precision)
            print("", f"TREE {tree_label} = [&R] {newick}", sep=indent, file=out)
        print("END;", file=out)


def wrap_text(text, width):
    """
    Return an iterator over the lines in the specified string of at most the
    specified width. (We could use textwrap.wrap for this, but it uses a
    more complicated algorithm appropriate for blocks of words.)
    """
    width = len(text) if width == 0 else width
    N = len(text) // width
    offset = 0
    for _ in range(N):
        yield text[offset : offset + width]
        offset += width
    if offset != len(text):
        yield text[offset:]


@flexible_file_output
def write_fasta(
    ts,
    output,
    *,
    wrap_width,
    reference_sequence,
    missing_data_character,
):
    # See TreeSequence.write_fasta for documentation
    if wrap_width < 0 or int(wrap_width) != wrap_width:
        raise ValueError(
            "wrap_width must be a non-negative integer. "
            "You may specify `wrap_width=0` "
            "if you do not want any wrapping."
        )
    wrap_width = int(wrap_width)
    alignments = ts.alignments(
        reference_sequence=reference_sequence,
        missing_data_character=missing_data_character,
    )
    for u, alignment in zip(ts.samples(), alignments):
        print(">", f"n{u}", sep="", file=output)
        for line in wrap_text(alignment, wrap_width):
            print(line, file=output)


def _build_newick(tree, *, node, precision, node_labels, include_branch_lengths):
    label = node_labels.get(node, "")
    if tree.is_leaf(node):
        s = f"{label}"
    else:
        s = "("
        for child in tree.children(node):
            branch_length = tree.branch_length(child)
            subtree = _build_newick(
                tree,
                node=child,
                precision=precision,
                node_labels=node_labels,
                include_branch_lengths=include_branch_lengths,
            )
            if include_branch_lengths:
                subtree += ":{0:.{1}f}".format(branch_length, precision)
            s += subtree + ","
        s = s[:-1] + f"){label}"
    return s


def build_newick(tree, *, root, precision, node_labels, include_branch_lengths):
    """
    Simple recursive version of the newick generator used when non-default
    node labels are needed, or when branch lengths are omitted
    """
    s = _build_newick(
        tree,
        node=root,
        precision=precision,
        node_labels=node_labels,
        include_branch_lengths=include_branch_lengths,
    )
    return s + ";"


def dump_text(
    ts,
    *,
    nodes,
    edges,
    sites,
    mutations,
    individuals,
    populations,
    migrations,
    provenances,
    precision,
    encoding,
    base64_metadata,
):
    if nodes is not None:
        print(
            "id",
            "is_sample",
            "time",
            "population",
            "individual",
            "metadata",
            sep="\t",
            file=nodes,
        )
        for node in ts.nodes():
            metadata = text_metadata(base64_metadata, encoding, node)
            row = (
                "{id:d}\t"
                "{is_sample:d}\t"
                "{time:.{precision}f}\t"
                "{population:d}\t"
                "{individual:d}\t"
                "{metadata}"
            ).format(
                precision=precision,
                id=node.id,
                is_sample=node.is_sample(),
                time=node.time,
                population=node.population,
                individual=node.individual,
                metadata=metadata,
            )
            print(row, file=nodes)

    if edges is not None:
        print("left", "right", "parent", "child", "metadata", sep="\t", file=edges)
        for edge in ts.edges():
            metadata = text_metadata(base64_metadata, encoding, edge)
            row = (
                "{left:.{precision}f}\t"
                "{right:.{precision}f}\t"
                "{parent:d}\t"
                "{child:d}\t"
                "{metadata}"
            ).format(
                precision=precision,
                left=edge.left,
                right=edge.right,
                parent=edge.parent,
                child=edge.child,
                metadata=metadata,
            )
            print(row, file=edges)

    if sites is not None:
        print("position", "ancestral_state", "metadata", sep="\t", file=sites)
        for site in ts.sites():
            metadata = text_metadata(base64_metadata, encoding, site)
            row = (
                "{position:.{precision}f}\t" "{ancestral_state}\t" "{metadata}"
            ).format(
                precision=precision,
                position=site.position,
                ancestral_state=site.ancestral_state,
                metadata=metadata,
            )
            print(row, file=sites)

    if mutations is not None:
        print(
            "site",
            "node",
            "time",
            "derived_state",
            "parent",
            "metadata",
            sep="\t",
            file=mutations,
        )
        for site in ts.sites():
            for mutation in site.mutations:
                metadata = text_metadata(base64_metadata, encoding, mutation)
                row = (
                    "{site}\t"
                    "{node}\t"
                    "{time}\t"
                    "{derived_state}\t"
                    "{parent}\t"
                    "{metadata}"
                ).format(
                    site=mutation.site,
                    node=mutation.node,
                    time="unknown"
                    if util.is_unknown_time(mutation.time)
                    else mutation.time,
                    derived_state=mutation.derived_state,
                    parent=mutation.parent,
                    metadata=metadata,
                )
                print(row, file=mutations)

    if individuals is not None:
        print(
            "id",
            "flags",
            "location",
            "parents",
            "metadata",
            sep="\t",
            file=individuals,
        )
        for individual in ts.individuals():
            metadata = text_metadata(base64_metadata, encoding, individual)
            location = ",".join(map(str, individual.location))
            parents = ",".join(map(str, individual.parents))
            row = (
                "{id}\t" "{flags}\t" "{location}\t" "{parents}\t" "{metadata}"
            ).format(
                id=individual.id,
                flags=individual.flags,
                location=location,
                parents=parents,
                metadata=metadata,
            )
            print(row, file=individuals)

    if populations is not None:
        print("id", "metadata", sep="\t", file=populations)
        for population in ts.populations():
            metadata = text_metadata(base64_metadata, encoding, population)
            row = ("{id}\t" "{metadata}").format(id=population.id, metadata=metadata)
            print(row, file=populations)

    if migrations is not None:
        print(
            "left",
            "right",
            "node",
            "source",
            "dest",
            "time",
            "metadata",
            sep="\t",
            file=migrations,
        )
        for migration in ts.migrations():
            metadata = text_metadata(base64_metadata, encoding, migration)
            row = (
                "{left}\t"
                "{right}\t"
                "{node}\t"
                "{source}\t"
                "{dest}\t"
                "{time}\t"
                "{metadata}\t"
            ).format(
                left=migration.left,
                right=migration.right,
                node=migration.node,
                source=migration.source,
                dest=migration.dest,
                time=migration.time,
                metadata=metadata,
            )
            print(row, file=migrations)

    if provenances is not None:
        print("id", "timestamp", "record", sep="\t", file=provenances)
        for provenance in ts.provenances():
            row = ("{id}\t" "{timestamp}\t" "{record}\t").format(
                id=provenance.id,
                timestamp=provenance.timestamp,
                record=provenance.record,
            )
            print(row, file=provenances)


def text_metadata(base64_metadata, encoding, node):
    metadata = node.metadata
    if isinstance(metadata, bytes) and base64_metadata:
        metadata = base64.b64encode(metadata).decode(encoding)
    else:
        metadata = repr(metadata)
    return metadata
