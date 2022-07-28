# MIT License
#
# Copyright (c) 2018-2022 Tskit Developers
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
Module responsible for various utility functions used in other modules.
"""
import dataclasses
import json
import numbers
import os
from typing import Union

import numpy as np

import tskit
from tskit import UNKNOWN_TIME


# Extra methods for dataclasses
class Dataclass:
    def replace(self, **kwargs):
        """
        Return a new instance of this dataclass, with the specified attributes
        overwritten by new values.

        :return: A new instance of the same type
        """
        return dataclasses.replace(self, **kwargs)


def canonical_json(obj):
    """
    Returns string of encoded JSON with keys sorted and whitespace removed to enable
    byte-level comparison of encoded data.

    :param Any obj: Python object to encode
    :return: The encoded string
    :rtype: str
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def is_unknown_time(time):
    """
    As the default unknown mutation time (:const:`UNKNOWN_TIME`) is a specific NAN value,
    equality always fails (A NAN value is not equal to itself by definition).
    This method compares the bitfield such that unknown times can be detected. Either
    single floats can be passed or lists/arrays.

    Note that NANs are a set of floating-point values. `tskit.UNKNOWN_TIME` is a specific
    value in this set. `np.nan` is a differing value, but both are NAN.
    See https://en.wikipedia.org/wiki/NaN

    This function only returns true for ``tskit.is_unknown_time(tskit.UNKNOWN_TIME)``
    and will return false for ``tskit.is_unknown_time(np.nan)`` or any other NAN or
    non-NAN value.

    :param time: Value or array to check.
    :type time: Union[float, array-like]
    :return: A single boolean or array of booleans the same shape as ``time``.
    :rtype: Union[bool, numpy.ndarray[bool]]
    """
    return np.asarray(time, dtype=np.float64).view(np.uint64) == np.float64(
        UNKNOWN_TIME
    ).view(np.uint64)


def safe_np_int_cast(int_array, dtype, copy=False):
    """
    A few functions require arrays of certain dtypes (e.g. node indices are np.int32,
    genotypes are np.int8, etc. Standard numpy integer arrays are of (dtype=np.int64),
    so need casting. This function casts but checks bounds to avoid wrap-around
    conversion errors. Strangely, numpy seems not to have this functionality built-in.

    If copy=False, and the original array is a numpy array of exactly the same dtype
    required, simply return the original rather than making a copy (same as the numpy
    .astype(copy=...) function)
    """
    if not isinstance(int_array, np.ndarray):
        int_array = np.array(int_array)
        # Since this is a new numpy array anyway, it's always a copy, so economize by
        # setting copy=False
        copy = False
    if int_array.size == 0:
        return int_array.astype(dtype, copy=copy)  # Allow empty arrays of any type
    try:
        return int_array.astype(dtype, casting="safe", copy=copy)
    except TypeError:
        if int_array.dtype == np.dtype("O"):
            # this occurs e.g. if we're passed a list of lists of different lengths
            raise TypeError("Cannot convert to a rectangular array.")
        bounds = np.iinfo(dtype)
        if np.any(int_array < bounds.min) or np.any(int_array > bounds.max):
            raise OverflowError(f"Cannot convert safely to {dtype} type")
        if int_array.dtype.kind == "i" and np.dtype(dtype).kind == "u":
            # Allow casting from int to unsigned int, since we have checked bounds
            casting = "unsafe"
        else:
            # Raise a TypeError when we try to convert from, e.g., a float.
            casting = "same_kind"
        return int_array.astype(dtype, casting=casting, copy=copy)


#
# Pack/unpack lists of data into flattened numpy arrays.
#


def pack_bytes(data):
    """
    Packs the specified list of bytes into a flattened numpy array of 8 bit integers
    and corresponding offsets. See :ref:`sec_encoding_ragged_columns` for details
    of this encoding.

    :param list[bytes] data: The list of bytes values to encode.
    :return: The tuple (packed, offset) of numpy arrays representing the flattened
        input data and offsets.
    :rtype: numpy.ndarray (dtype=np.int8), numpy.ndarray (dtype=np.uint32)
    """
    n = len(data)
    offsets = np.zeros(n + 1, dtype=np.uint32)
    for j in range(n):
        offsets[j + 1] = offsets[j] + len(data[j])
    column = np.zeros(offsets[-1], dtype=np.int8)
    for j, value in enumerate(data):
        column[offsets[j] : offsets[j + 1]] = bytearray(value)
    return column, offsets


def unpack_bytes(packed, offset):
    """
    Unpacks a list of bytes from the specified numpy arrays of packed byte
    data and corresponding offsets. See :ref:`sec_encoding_ragged_columns` for details
    of this encoding.

    :param numpy.ndarray packed: The flattened array of byte values.
    :param numpy.ndarray offset: The array of offsets into the ``packed`` array.
    :return: The list of bytes values unpacked from the parameter arrays.
    :rtype: list[bytes]
    """
    # This could be done a lot more efficiently...
    ret = []
    for j in range(offset.shape[0] - 1):
        raw = packed[offset[j] : offset[j + 1]].tobytes()
        ret.append(raw)
    return ret


def pack_strings(strings, encoding="utf8"):
    """
    Packs the specified list of strings into a flattened numpy array of 8 bit integers
    and corresponding offsets using the specified text encoding.
    See :ref:`sec_encoding_ragged_columns` for details of this encoding of
    columns of variable length data.

    :param list[str] data: The list of strings to encode.
    :param str encoding: The text encoding to use when converting string data
        to bytes. See the :mod:`codecs` module for information on available
        string encodings.
    :return: The tuple (packed, offset) of numpy arrays representing the flattened
        input data and offsets.
    :rtype: numpy.ndarray (dtype=np.int8), numpy.ndarray (dtype=np.uint32)
    """
    return pack_bytes([bytearray(s.encode(encoding)) for s in strings])


def unpack_strings(packed, offset, encoding="utf8"):
    """
    Unpacks a list of strings from the specified numpy arrays of packed byte
    data and corresponding offsets using the specified text encoding.
    See :ref:`sec_encoding_ragged_columns` for details of this encoding of
    columns of variable length data.

    :param numpy.ndarray packed: The flattened array of byte values.
    :param numpy.ndarray offset: The array of offsets into the ``packed`` array.
    :param str encoding: The text encoding to use when converting string data
        to bytes. See the :mod:`codecs` module for information on available
        string encodings.
    :return: The list of strings unpacked from the parameter arrays.
    :rtype: list[str]
    """
    return [b.decode(encoding) for b in unpack_bytes(packed, offset)]


def pack_arrays(list_of_lists, dtype=np.float64):
    """
    Packs the specified list of numberic lists into a flattened numpy array of
    numpy float64 and corresponding offsets. See
    :ref:`sec_encoding_ragged_columns` for details of this encoding of columns
    of variable length data.

    :param list[list] list_of_lists: The list of numeric lists to encode.
    :param dtype: The dtype for the packed array, defualts to float64
    :return: The tuple (packed, offset) of numpy arrays representing the flattened
        input data and offsets.
    :rtype: numpy.array (dtype=dtype), numpy.array (dtype=np.uint32)
    """
    # TODO must be possible to do this more efficiently with numpy
    n = len(list_of_lists)
    offset = np.zeros(n + 1, dtype=np.uint32)
    for j in range(n):
        offset[j + 1] = offset[j] + len(list_of_lists[j])
    data = np.empty(offset[-1], dtype=dtype)
    for j in range(n):
        data[offset[j] : offset[j + 1]] = list_of_lists[j]
    return data, offset


def unpack_arrays(packed, offset):
    """
    Unpacks a list of arrays from the specified numpy arrays of packed
    data and corresponding offsets. See
    :ref:`sec_encoding_ragged_columns` for details of this encoding of columns
    of variable length data.

    :param numpy.ndarray packed: The flattened array of data.
    :param numpy.ndarray offset: The array of offsets into the ``packed`` array.
    :return: The list numpy arrays unpacked from the parameter arrays.
    :rtype: list[numpy.ndarray]
    """
    ret = []
    for j in range(offset.shape[0] - 1):
        ret.append(packed[offset[j] : offset[j + 1]])
    return ret


#
# Interval utilities
#


def intervals_to_np_array(intervals, start, end):
    """
    Converts the specified intervals to a numpy array and checks for
    errors.
    """
    intervals = np.array(intervals, dtype=np.float64)
    # Special case the empty list of intervals
    if len(intervals) == 0:
        intervals = np.zeros((0, 2), dtype=np.float64)
    if len(intervals.shape) != 2:
        raise ValueError("Intervals must be a 2D numpy array")
    if intervals.shape[1] != 2:
        raise ValueError("Intervals array shape must be (N, 2)")
    # TODO do this with numpy operations.
    last_right = start
    for left, right in intervals:
        if left < start or right > end:
            raise ValueError(f"Intervals must be within {start} and {end}")
        if right <= left:
            raise ValueError("Bad interval: right <= left")
        if left < last_right:
            raise ValueError("Intervals must be disjoint.")
        last_right = right
    return intervals


def negate_intervals(intervals, start, end):
    """
    Returns the set of intervals *not* covered by the specified set of
    disjoint intervals in the specified range.
    """
    intervals = intervals_to_np_array(intervals, start, end)
    other_intervals = []
    last_right = start
    for left, right in intervals:
        if left != last_right:
            other_intervals.append((last_right, left))
        last_right = right
    if last_right != end:
        other_intervals.append((last_right, end))
    return np.array(other_intervals)


def naturalsize(value):
    """
    Format a number of bytes like a human readable filesize (e.g. 10 kiB)
    """
    # Taken from https://github.com/jmoiron/humanize
    suffix = ("KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB")
    base = 1024
    format_ = "%.1f"

    bytes_ = float(value)
    abs_bytes = abs(bytes_)

    if abs_bytes == 1:
        return "%d Byte" % bytes_
    elif abs_bytes < base:
        return "%d Bytes" % bytes_

    for i, s in enumerate(suffix):
        unit = base ** (i + 2)
        if abs_bytes < unit:
            return (format_ + " %s") % ((base * bytes_ / unit), s)
    return (format_ + " %s") % ((base * bytes_ / unit), s)


def obj_to_collapsed_html(d, name=None, open_depth=0):
    """
    Recursively make an HTML representation of python objects.

    :param str name: Name for this object
    :param int open_depth: By default sub-sections are collapsed. If this number is
    non-zero the first layers up to open_depth will be opened.
    :return: The HTML as a string
    :rtype: str
    """
    opened = "open" if open_depth > 0 else ""
    open_depth -= 1
    name = str(name) + ":" if name is not None else ""
    if type(d) == dict:
        return f"""
                <div>
                  <span class="tskit-details-label">{name}</span>
                  <details {opened}>
                    <summary>dict</summary>
                    {"".join(f"{obj_to_collapsed_html(val, key, open_depth)}<br/>"
                             for key, val in d.items())}
                  </details>
                </div>
                """
    elif type(d) == list:
        return f"""
                <div>
                  <span class="tskit-details-label">{name}</span>
                  <details {opened}>
                    <summary>list</summary>
                    {"".join(f"{obj_to_collapsed_html(val, None, open_depth)}<br/>"
                             for val in d)}
                  </details>
                </div>
                """
    else:
        return f"{name} {d}"


def truncate_string_end(string, length):
    """
    If a string is longer than "length" then snip out the middle and replace with an
    ellipsis.
    """
    if len(string) <= length:
        return string
    return f"{string[:length-3]}..."


def render_metadata(md, length=40):
    if md == b"":
        return ""
    return truncate_string_end(str(md), length)


def unicode_table(rows, title=None, header=None, row_separator=True):
    """
    Convert a table (list of lists) of strings to a unicode table. If a row contains
    the string "__skipped__NNN" then "skipped N rows" is displayed.

    :param list[list[str]] rows: List of rows, each of which is a list of strings for
    each cell. The first column will be left justified, the others right. Each row must
    have the same number of cells.
    :param str title: If specified the first output row will be a single cell
    containing this string, left-justified. [optional]
    :param list[str] header: Specifies a row above the main rows which will be in double
    lined borders and left justified. Must be same length as each row. [optional]
    :param boolean row_separator: If True add lines between each row. [Default: True]
    :return: The table as a string
    :rtype: str
    """
    if header is not None:
        all_rows = [header] + rows
    else:
        all_rows = rows
    widths = [
        max(len(row[i_col]) for row in all_rows) for i_col in range(len(all_rows[0]))
    ]
    out = []
    inner_width = sum(widths) + len(header or rows[0]) - 1
    if title is not None:
        out += [
            f"╔{'═' * inner_width}╗\n" f"║{title.ljust(inner_width)}║\n",
            f"╠{'╤'.join('═' * w for w in widths)}╣\n",
        ]
    if header is not None:
        out += [
            f"╔{'╤'.join('═' * w for w in widths)}╗\n",
            f"║{'│'.join(cell.ljust(w) for cell,w in zip(header,widths))}║\n",
            f"╠{'╪'.join('═' * w for w in widths)}╣\n",
        ]
    last_skipped = False
    for i, row in enumerate(rows):
        if "__skipped__" in row:
            msg = f"{row[11:]} rows skipped (tskit.set_print_options)"[
                :inner_width
            ].center(inner_width)
            row_str = f"║{msg}║\n"
            if row_separator:
                out += [
                    f"╟{'┴'.join('─' * w for w in widths)}╢\n" + row_str,
                    f"╟{'┬'.join('─' * w for w in widths)}╢\n",
                ]
            else:
                out.append(row_str)
            last_skipped = True
        else:
            if i != 0 and not last_skipped and row_separator:
                out.append(f"╟{'┼'.join('─' * w for w in widths)}╢\n")
            out.append(
                f"║{row[0].ljust(widths[0])}│"
                f"{'│'.join(cell.rjust(w) for cell, w in zip(row[1:], widths[1:]))}║\n"
            )
            last_skipped = False

    out.append(f"╚{'╧'.join('═' * w for w in widths)}╝\n")
    return "".join(out)


def tree_sequence_html(ts):
    table_rows = "".join(
        f"""
                  <tr>
                    <td>{name.capitalize()}</td>
                      <td>{table.num_rows}</td>
                      <td>{naturalsize(table.nbytes)}</td>
                      <td style="text-align: center;">
                        {'✅' if hasattr(table, "metadata") and len(table.metadata) > 0
        else ''}
                      </td>
                    </tr>
                """
        for name, table in ts.tables.table_name_map.items()
    )
    return f"""
            <div>
              <style>
                .tskit-table thead tr th {{text-align: left;padding: 0.5em 0.5em;}}
                .tskit-table tbody tr td {{padding: 0.5em 0.5em;}}
                .tskit-table tbody tr td:first-of-type {{text-align: left;}}
                .tskit-details-label {{vertical-align: top; padding-right:5px;}}
                .tskit-table-set {{display: inline-flex;flex-wrap: wrap;margin: -12px 0 0 -12px;width: calc(100% + 12px);}}
                .tskit-table-set-table {{margin: 12px 0 0 12px;}}
                details {{display: inline-block;}}
                summary {{cursor: pointer; outline: 0; display: list-item;}}
              </style>
              <div class="tskit-table-set">
                <div class="tskit-table-set-table">
                  <table class="tskit-table">
                    <thead>
                      <tr>
                        <th style="padding:0;line-height:21px;">
                          <img style="height: 32px;display: inline-block;padding: 3px 5px 3px 0;" src="https://raw.githubusercontent.com/tskit-dev/administrative/main/tskit_logo.svg"/>
                          <a target="_blank" href="https://tskit.dev/tskit/docs/latest/python-api.html#the-treesequence-class"> Tree Sequence </a>
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr><td>Trees</td><td>{ts.num_trees}</td></tr>
                      <tr><td>Sequence Length</td><td>{ts.sequence_length}</td></tr>
                      <tr><td>Time Units</td><td>{ts.time_units}</td></tr>
                      <tr><td>Sample Nodes</td><td>{ts.num_samples}</td></tr>
                      <tr><td>Total Size</td><td>{naturalsize(ts.nbytes)}</td></tr>
                      <tr>
                        <td>Metadata</td><td style="text-align: left;">{obj_to_collapsed_html(ts.metadata, None, 1) if len(ts.tables.metadata_bytes) > 0 else "No Metadata"}</td></tr>
                    </tbody>
                  </table>
                </div>
                <div class="tskit-table-set-table">
                  <table class="tskit-table">
                    <thead>
                      <tr>
                        <th style="line-height:21px;">Table</th>
                        <th>Rows</th>
                        <th>Size</th>
                        <th>Has Metadata</th>
                      </tr>
                    </thead>
                    <tbody>
                    {table_rows}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
            """  # noqa: B950


def tree_html(tree):
    return f"""
            <div>
              <style>
                .tskit-table thead tr th {{text-align: left;padding: 0.5em 0.5em;}}
                .tskit-table tbody tr td {{padding: 0.5em 0.5em;}}
                .tskit-table tbody tr td:first-of-type {{text-align: left;}}
                .tskit-details-label {{vertical-align: top; padding-right:5px;}}
                .tskit-table-set {{display: inline-flex;flex-wrap: wrap;margin: -12px 0 0 -12px;width: calc(100% + 12px);}}
                .tskit-table-set-table {{margin: 12px 0 0 12px;}}
                details {{display: inline-block;}}
                summary {{cursor: pointer; outline: 0; display: list-item;}}
              </style>
              <div class="tskit-table-set">
                <div class="tskit-table-set-table">
                  <table class="tskit-table">
                    <thead>
                      <tr>
                        <th style="padding:0;line-height:21px;">
                          <img style="height: 32px;display: inline-block;padding: 3px 5px 3px 0;" src="https://raw.githubusercontent.com/tskit-dev/administrative/main/tskit_logo.svg"/>
                          <a target="_blank" href="https://tskit.dev/tskit/docs/latest/python-api.html#the-tree-class"> Tree </a>
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr><td>Index</td><td>{tree.index}</td></tr>
                      <tr><td>Interval</td><td>{tree.interval.left:.8g}-{tree.interval.right:.8g} ({tree.span:.8g})</td></tr>
                      <tr><td>Roots</td><td>{tree.num_roots}</td></tr>
                      <tr><td>Nodes</td><td>{len(tree.preorder())}</td></tr>
                      <tr><td>Sites</td><td>{tree.num_sites}</td></tr>
                      <tr><td>Mutations</td><td>{tree.num_mutations}</td></tr>
                      <tr><td>Total Branch Length</td><td>{tree.total_branch_length:.8g}</td></tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
            """  # noqa: B950


def variant_html(variant):
    class_type = "Variant"

    url_tskit_logo = (
        "https://raw.githubusercontent.com/tskit-dev/administrative/main/tskit_logo.svg"
    )
    url_variant_class_doc = (
        "https://tskit.dev/tskit/docs/latest/python-api.html#the-variant-class"
    )

    html_body_head = f"""
        <div>
            <style>
                .tskit-table thead tr th {{text-align: left;padding: 0.5em 0.5em;}}
                .tskit-table tbody tr td {{padding: 0.5em 0.5em;}}
                .tskit-table tbody tr td:first-of-type {{text-align: left;}}
                .tskit-details-label {{vertical-align: top; padding-right:5px;}}
                .tskit-table-set {{display: inline-flex;flex-wrap: wrap;margin: -12px 0 0 -12px;width: calc(100% + 12px);}}
                .tskit-table-set-table {{margin: 12px 0 0 12px;}}
                details {{display: inline-block;}}
                summary {{cursor: pointer; outline: 0; display: list-item;}}
            </style>
            <div class="tskit-table-set">
                <div class="tskit-table-set-table">
                <table class="tskit-table">
                    <thead>
                    <tr>
                        <th style="padding:0;line-height:21px;">
                        <img style="height: 32px;display: inline-block;padding: 3px 5px 3px 0;" src="{url_tskit_logo}"/>
                        <a target="_blank" href="{url_variant_class_doc}"> {class_type} </a>
                        </th>
                    </tr>
                    </thead>
                    <tbody>
        """  # noqa: B950

    html_body_tail = """
                    </tbody>
            </table>
            </div>
        </div>
        </div>
        """

    try:
        variant.site

        site_id = variant.site.id
        site_position = variant.site.position
        num_samples = len(variant.samples)
        num_alleles = variant.num_alleles
        has_missing_data = str(variant.has_missing_data)
        isolated_as_missing = str(bool(variant.isolated_as_missing))

        counts = variant.counts()
        freqs = variant.frequencies()

        return (
            html_body_head
            + f"""
                <tr><td>Site Id</td><td>{site_id}</td></tr>
                <tr><td>Site Position</td><td>{site_position}</td></tr>
                <tr><td>Number of Samples</td><td>{num_samples}</td></tr>
                <tr><td>Number of Alleles</td><td>{num_alleles}</td></tr>
            """
            + "\n".join(
                [
                    f"""<tr><td>Samples with Allele {'missing' if k is None
                        else "'" + k + "'"}</td><td>"""
                    + f"{counts[k]}"
                    + " "
                    + f"({freqs[k] * 100:.2g}%)"
                    + "</td></tr>"
                    for k in variant.alleles
                ]
            )
            + f"""
                <tr><td>Has Missing Data</td><td>{has_missing_data}</td></tr>
                <tr><td>Isolated as Missing</td><td>{isolated_as_missing}</td></tr>
            """
            + html_body_tail
        )
    except ValueError as err:
        return (
            html_body_head
            + f"""
                        <tr><td>Error</td><td>{str(err)}</td></tr>
            """
            + html_body_tail
        )


def convert_file_like_to_open_file(file_like, mode):
    # Get ourselves a local version of the file. The semantics here are complex
    # because need to support a range of inputs and the free behaviour is
    # slightly different on each.
    _file = None
    local_file = True
    try:
        # First, see if we can interpret the argument as a pathlike object.
        path = os.fspath(file_like)
        _file = open(path, mode)
    except TypeError:
        pass
    if _file is None:
        # Now we try to open file. If it's not a pathlike object, it could be
        # an integer fd or object with a fileno method. In this case we
        # must make sure that close is **not** called on the fd.
        try:
            _file = open(file_like, mode, closefd=False, buffering=0)
        except TypeError:
            pass
    if _file is None:
        # Assume that this is a file **but** we haven't opened it, so we must
        # not close it.
        if mode == "wb" and not hasattr(file_like, "write"):
            raise TypeError("file object must have a write method")
        _file = file_like
        local_file = False
    return _file, local_file


def set_print_options(*, max_lines=40):
    """
    Set the options for printing to strings and HTML

    :param integer max_lines: The maximum number of lines to print from a table, beyond
    this number the middle of the table will be skipped.
    """
    tskit._print_options = {"max_lines": max_lines}


def random_nucleotides(length: numbers.Number, *, seed: Union[int, None] = None) -> str:
    """
    Returns a random string of nucleotides of the specified length. Characters
    are drawn uniformly from the alphabet "ACTG".

    :param int length: The length of the random sequence.
    :return: A string of the specified length consisting of random nucleotide
       characters.
    :rtype: str
    """
    if int(length) != length:
        raise ValueError("length must be an integer")
    rng = np.random.RandomState(seed)
    encoded_nucleotides = np.array(list(map(ord, "ACTG")), dtype=np.int8)
    a = rng.choice(encoded_nucleotides, size=int(length))
    return a.tobytes().decode("ascii")
