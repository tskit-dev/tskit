# MIT License
#
# Copyright (c) 2018-2024 Tskit Developers
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
Tests for functions in util.py
"""
import collections
import itertools
import math
import pickle
import textwrap

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import tests.tsutil as tsutil
import tskit
import tskit.util as util
from tskit import UNKNOWN_TIME


class TestCanonicalJSON:
    def test_canonical_json(self):
        assert util.canonical_json([3, 2, 1]) == "[3,2,1]"
        assert (
            util.canonical_json(collections.OrderedDict(c=3, b=2, a=1))
            == '{"a":1,"b":2,"c":3}'
        )
        assert (
            util.canonical_json(
                collections.OrderedDict(
                    c="3",
                    b=collections.OrderedDict(
                        {
                            "b": 1,
                            "z": {},
                            " space": 42,
                            "1": "number",
                            "_": "underscore",
                        }
                    ),
                    a="1",
                )
            )
            == '{"a":"1","b":{" space":42,"1":"number",'
            '"_":"underscore","b":1,"z":{}},"c":"3"}'
        )


class TestUnknownTime:
    def test_unknown_time_bad_types(self):
        with pytest.raises(ValueError):
            util.is_unknown_time("bad")
        with pytest.raises(ValueError):
            util.is_unknown_time(np.array(["bad"]))
        with pytest.raises(ValueError):
            util.is_unknown_time(["bad"])

    def test_unknown_time_scalar(self):
        assert math.isnan(UNKNOWN_TIME)
        assert util.is_unknown_time(UNKNOWN_TIME)
        assert not util.is_unknown_time(math.nan)
        assert not util.is_unknown_time(np.nan)
        assert not util.is_unknown_time(0)
        assert not util.is_unknown_time(math.inf)
        assert not util.is_unknown_time(1)
        assert not util.is_unknown_time(None)
        assert not util.is_unknown_time([None])

    def test_unknown_time_array(self):
        test_arrays = (
            [],
            [True],
            [False],
            [True, False] * 5,
            [[True], [False]],
            [[[True, False], [True, False]], [[False, True], [True, False]]],
        )
        for spec in test_arrays:
            spec = np.asarray(spec, dtype=bool)
            array = np.zeros(shape=spec.shape)
            array[spec] = UNKNOWN_TIME
            assert_array_equal(spec, util.is_unknown_time(array))

        weird_array = [0, UNKNOWN_TIME, np.nan, 1, math.inf]
        assert_array_equal(
            [False, True, False, False, False], util.is_unknown_time(weird_array)
        )


class TestNumpyArrayCasting:
    """
    Tests that the safe_np_int_cast() function works.
    """

    dtypes_to_test = [np.int32, np.uint32, np.int8, np.uint8]

    def test_basic_arrays(self):
        # Simple array
        for dtype in self.dtypes_to_test:
            target = np.array([0, 1], dtype=dtype)
            for test_array in [[0, 1], (0, 1), np.array([0, 1]), target]:
                converted = util.safe_np_int_cast(test_array, dtype=dtype)
                # Use pickle to test exact equality including dtype
                assert pickle.dumps(converted) == pickle.dumps(target)
            # Nested array
            target = np.array([[0, 1], [2, 3]], dtype=dtype)
            for test_array in [[[0, 1], [2, 3]], np.array([[0, 1], [2, 3]]), target]:
                converted = util.safe_np_int_cast(test_array, dtype=dtype)
                assert pickle.dumps(converted) == pickle.dumps(target)

    def test_copy(self):
        # Check that a copy is not returned if copy=False & the original matches
        # the specs
        for dtype in self.dtypes_to_test:
            for orig in (np.array([0, 1], dtype=dtype), np.array([], dtype=dtype)):
                converted = util.safe_np_int_cast(orig, dtype=dtype, copy=True)
                assert id(orig) != id(converted)
                converted = util.safe_np_int_cast(orig, dtype=dtype, copy=False)
                assert id(orig) == id(converted)
        for dtype in [d for d in self.dtypes_to_test if d != np.int64]:
            # non numpy arrays, or arrays of a different dtype don't get converted
            for orig in ([0, 1], np.array([0, 1], dtype=np.int64)):
                converted = util.safe_np_int_cast(orig, dtype=dtype, copy=False)
                assert id(orig) != id(converted)

    def test_empty_arrays(self):
        # Empty arrays of any type (including float) should be allowed
        for dtype in self.dtypes_to_test:
            target = np.array([], dtype=dtype)
            converted = util.safe_np_int_cast([], dtype=dtype)
            assert pickle.dumps(converted) == pickle.dumps(target)
            target = np.array([[]], dtype=dtype)
            converted = util.safe_np_int_cast([[]], dtype=dtype)
            assert pickle.dumps(converted) == pickle.dumps(target)

    def test_bad_types(self):
        # Shouldn't be able to convert a float (possibility of rounding error)
        for dtype in self.dtypes_to_test:
            for bad_type in [
                [0.1],
                ["str"],
                {},
                [{}],
                np.array([0, 1], dtype=float),
            ]:
                with pytest.raises(TypeError):
                    util.safe_np_int_cast(bad_type, dtype)

    def test_overflow(self):
        for dtype in self.dtypes_to_test:
            for bad_node in [np.iinfo(dtype).min - 1, np.iinfo(dtype).max + 1]:
                with pytest.raises(OverflowError):
                    util.safe_np_int_cast([0, bad_node], dtype)
                with pytest.raises(OverflowError):
                    util.safe_np_int_cast(np.array([0, bad_node]), dtype)
            for good_node in [np.iinfo(dtype).min, np.iinfo(dtype).max]:
                target = np.array([good_node], dtype=dtype)
                assert pickle.dumps(target) == pickle.dumps(
                    util.safe_np_int_cast([good_node], dtype)
                )
                assert pickle.dumps(target) == pickle.dumps(
                    util.safe_np_int_cast(np.array([good_node]), dtype)
                )

    def test_nonrectangular_input(self):
        bad_inputs = [
            [0, 1, [2]],
            [[0, 1, 2], []],
            [(0, 1, 2), [2, 3]],
            [(0, 1, 2), tuple()],
            [(0, 1, 2), (2,)],
            [(0, 1, 2), [2, 3]],
        ]
        for dtype in self.dtypes_to_test:
            for bad_input in bad_inputs:
                # On some platforms and Python / numpy versions, a ValueError
                # occurs instead
                with pytest.raises((TypeError, ValueError)):
                    util.safe_np_int_cast(bad_input, dtype)


class TestIntervalOps:
    """
    Test cases for the interval operations used in masks and slicing operations.
    """

    def test_bad_intervals(self):
        for bad_type in [{}, Exception]:
            with pytest.raises(TypeError):
                util.intervals_to_np_array(bad_type, 0, 1)
        for bad_depth in [[[[]]]]:
            with pytest.raises(ValueError):
                util.intervals_to_np_array(bad_depth, 0, 1)
        for bad_shape in [[[0], [0]], [[[0, 1, 2], [0, 1]]]]:
            with pytest.raises(ValueError):
                util.intervals_to_np_array(bad_shape, 0, 1)

        # Out of bounds
        with pytest.raises(ValueError):
            util.intervals_to_np_array([[-1, 0]], 0, 1)
        with pytest.raises(ValueError):
            util.intervals_to_np_array([[0, 1]], 1, 2)
        with pytest.raises(ValueError):
            util.intervals_to_np_array([[0, 1]], 0, 0.5)

        # Overlapping intervals
        with pytest.raises(ValueError):
            util.intervals_to_np_array([[0, 1], [0.9, 2.0]], 0, 10)

        # Empty intervals
        for bad_interval in [[0, 0], [1, 0]]:
            with pytest.raises(ValueError):
                util.intervals_to_np_array([bad_interval], 0, 10)

    def test_empty_interval_list(self):
        intervals = util.intervals_to_np_array([], 0, 10)
        assert len(intervals) == 0

    def test_negate_intervals(self):
        L = 10
        cases = [
            ([], [[0, L]]),
            ([[0, 5], [6, L]], [[5, 6]]),
            ([[0, 5]], [[5, L]]),
            ([[5, L]], [[0, 5]]),
            ([[0, 1], [2, 3], [3, 4], [5, 6]], [[1, 2], [4, 5], [6, L]]),
        ]
        for source, dest in cases:
            assert np.array_equal(util.negate_intervals(source, 0, L), dest)


class TestStringPacking:
    """
    Tests the code for packing and unpacking unicode string data into numpy arrays.
    """

    def test_simple_string_case(self):
        strings = ["hello", "world"]
        packed, offset = util.pack_strings(strings)
        assert list(offset) == [0, 5, 10]
        assert packed.shape == (10,)
        returned = util.unpack_strings(packed, offset)
        assert returned == strings

    def verify_packing(self, strings):
        packed, offset = util.pack_strings(strings)
        assert packed.dtype == np.int8
        assert offset.dtype == np.uint32
        assert packed.shape[0] == offset[-1]
        returned = util.unpack_strings(packed, offset)
        assert strings == returned

    def test_regular_cases(self):
        for n in range(10):
            strings = ["a" * j for j in range(n)]
            self.verify_packing(strings)

    def test_random_cases(self):
        for n in range(100):
            strings = [tsutil.random_strings(10) for _ in range(n)]
            self.verify_packing(strings)

    def test_unicode(self):
        self.verify_packing(["abcdé", "€"])


class TestBytePacking:
    """
    Tests the code for packing and unpacking binary data into numpy arrays.
    """

    def test_simple_string_case(self):
        strings = [b"hello", b"world"]
        packed, offset = util.pack_bytes(strings)
        assert list(offset) == [0, 5, 10]
        assert packed.shape == (10,)
        returned = util.unpack_bytes(packed, offset)
        assert returned == strings

    def verify_packing(self, data):
        packed, offset = util.pack_bytes(data)
        assert packed.dtype == np.int8
        assert offset.dtype == np.uint32
        assert packed.shape[0] == offset[-1]
        returned = util.unpack_bytes(packed, offset)
        assert data == returned
        return returned

    def test_random_cases(self):
        for n in range(100):
            data = [tsutil.random_bytes(10) for _ in range(n)]
            self.verify_packing(data)

    def test_pickle_packing(self):
        data = [list(range(j)) for j in range(10)]
        # Pickle each of these in turn
        pickled = [pickle.dumps(d) for d in data]
        unpacked = self.verify_packing(pickled)
        unpickled = [pickle.loads(p) for p in unpacked]
        assert data == unpickled


class TestArrayPacking:
    """
    Tests the code for packing and unpacking numpy data into numpy arrays.
    """

    def test_simple_case(self):
        lists = [[0], [1.125, 1.25]]
        packed, offset = util.pack_arrays(lists)
        assert list(offset) == [0, 1, 3]
        assert list(packed) == [0, 1.125, 1.25]
        returned = util.unpack_arrays(packed, offset)
        for a1, a2 in itertools.zip_longest(lists, returned):
            assert a1 == list(a2)

    def verify_packing(self, data):
        packed, offset = util.pack_arrays(data)
        assert packed.dtype == np.float64
        assert offset.dtype == np.uint32
        assert packed.shape[0] == offset[-1]
        returned = util.unpack_arrays(packed, offset)
        for a1, a2 in itertools.zip_longest(data, returned):
            assert np.array_equal(a1, a2)
        return returned

    def test_regular_cases(self):
        for n in range(100):
            data = [np.arange(n) for _ in range(n)]
            self.verify_packing(data)
            data = [1 / (1 + np.arange(n)) for _ in range(n)]
            self.verify_packing(data)


@pytest.mark.parametrize(
    "value, expected",
    [
        (0, "0 Bytes"),
        (1, "1 Byte"),
        (300, "300 Bytes"),
        (3000, "2.9 KiB"),
        (3000000, "2.9 MiB"),
        (10**26 * 30, "2481.5 YiB"),
    ],
)
def test_naturalsize(value, expected):
    assert util.naturalsize(value) == expected
    if value != 0:
        assert util.naturalsize(-value) == "-" + expected
    else:
        assert util.naturalsize(-value) == expected


def test_format_number():
    assert util.format_number(0) == "0"
    assert util.format_number("1.23") == "1.23"
    assert util.format_number(3216546.34) == "3 216 546.3"
    assert util.format_number(3216546.34, 9) == "3 216 546.34"
    assert util.format_number(-3456.23) == "-3 456.23"
    assert util.format_number(-3456.23, sep=",") == "-3,456.23"

    with pytest.raises(TypeError) as e_info:
        util.format_number("bad")
        assert str(e_info.value) == "The string cannot be converted to a number"


@pytest.mark.parametrize(
    "obj, expected",
    [
        (0, "Test: 0"),
        (
            {"a": 1},
            '<div><span class="tskit-details-label">Test:</span><details open>'
            "<summary>dict</summary>a: 1<br/></details></div>",
        ),
        (
            {"b": [1, 2, 3]},
            '<div><span class="tskit-details-label">Test:</span><details open>'
            '<summary>dict</summary><div><span class="tskit-details-label">b:'
            "</span><details ><summary>list</summary> 1<br/> 2<br/> 3<br/></"
            "details></div><br/></details></div>",
        ),
        (
            {"b": [1, 2, {"c": 1}]},
            '<div><span class="tskit-details-label">Test:</span><details open>'
            '<summary>dict</summary><div><span class="tskit-details-label">b:'
            "</span><details ><summary>list</summary> 1<br/> 2<br/><div><span"
            ' class="tskit-details-label"></span><details ><summary>dict</'
            "summary>c: 1<br/></details></div><br/></details></div><br/></"
            "details></div>",
        ),
        (
            {"a": "1", "b": "2"},
            '<div><span class="tskit-details-label">Test:</span><details open>'
            "<summary>dict</summary>a: 1<br/>b: 2<br/></details></div>",
        ),
        (
            {"a": "a very long thing that is broken in the output"},
            '<div><span class="tskit-details-label">Test:</span><details open>'
            "<summary>dict</summary>a: a very long thing that is<br/>broken in"
            " the output<br/></details></div>",
        ),
    ],
    ids=[
        "integer",
        "simple_dict",
        "dict_with_list",
        "nested_dict_list",
        "dict_with_strings",
        "dict_with_multiline_strings",
    ],
)
def test_obj_to_collapsed_html(obj, expected):
    assert (
        util.obj_to_collapsed_html(obj, "Test", 1).replace("  ", "").replace("\n", "")
        == expected
    )


def test_truncate_string_end():
    assert util.truncate_string_end("testing", 40) == "testing"
    assert util.truncate_string_end("testing", 7) == "testing"
    assert util.truncate_string_end("testing", 5) == "te..."


def test_render_metadata():
    assert util.render_metadata({}) == "{}"
    assert util.render_metadata("testing") == "testing"
    assert util.render_metadata(b"testing") == "b'testing'"
    assert util.render_metadata(b"testing", 6) == "b't..."
    assert util.render_metadata(b"") == ""


def test_unicode_table():
    assert (
        util.unicode_table(
            [["5", "6", "7", "8"], ["90", "10", "11", "12"]],
            header=["1", "2", "3", "4"],
        )
        == textwrap.dedent(
            """
           ╔══╤══╤══╤══╗
           ║1 │2 │3 │4 ║
           ╠══╪══╪══╪══╣
           ║5 │ 6│ 7│ 8║
           ╟──┼──┼──┼──╢
           ║90│10│11│12║
           ╚══╧══╧══╧══╝
        """
        )[1:]
    )

    assert (
        util.unicode_table(
            [
                ["1", "2", "3", "4"],
                ["5", "6", "7", "8"],
                "__skipped__",
                ["90", "10", "11", "12"],
            ],
            title="TITLE",
        )
        == textwrap.dedent(
            """
           ╔═══════════╗
           ║TITLE      ║
           ╠══╤══╤══╤══╣
           ║1 │ 2│ 3│ 4║
           ╟──┼──┼──┼──╢
           ║5 │ 6│ 7│ 8║
           ╟──┴──┴──┴──╢
           ║ rows skipp║
           ╟──┬──┬──┬──╢
           ║90│10│11│12║
           ╚══╧══╧══╧══╝
        """
        )[1:]
    )

    assert (
        util.unicode_table(
            [["1", "2", "3", "4"], ["5", "6", "7", "8"], ["90", "10", "11", "12"]],
            title="TITLE",
            row_separator=False,
        )
        == textwrap.dedent(
            """
           ╔═══════════╗
           ║TITLE      ║
           ╠══╤══╤══╤══╣
           ║1 │ 2│ 3│ 4║
           ║5 │ 6│ 7│ 8║
           ║90│10│11│12║
           ╚══╧══╧══╧══╝
        """
        )[1:]
    )


def test_unicode_table_column_alignments():
    assert (
        util.unicode_table(
            [["5", "6", "7", "8"], ["90", "10", "11", "12"]],
            header=["1", "2", "3", "4"],
            column_alignments="<>><",
        )
        == textwrap.dedent(
            """
           ╔══╤══╤══╤══╗
           ║1 │2 │3 │4 ║
           ╠══╪══╪══╪══╣
           ║5 │ 6│ 7│8 ║
           ╟──┼──┼──┼──╢
           ║90│10│11│12║
           ╚══╧══╧══╧══╝
        """
        )[1:]
    )


def test_set_printoptions():
    assert tskit._print_options == {"max_lines": 40}
    util.set_print_options(max_lines=None)
    assert tskit._print_options == {"max_lines": None}
    util.set_print_options(max_lines=40)
    assert tskit._print_options == {"max_lines": 40}
    with pytest.raises(TypeError):
        util.set_print_options(40)


class TestRandomNuceotides:
    @pytest.mark.parametrize("length", [0, 1, 10, 10.0, np.array([10])[0]])
    def test_length(self, length):
        s = tskit.random_nucleotides(length, seed=42)
        assert len(s) == length
        assert isinstance(s, str)

    def test_default_alphabet(self):
        s = tskit.random_nucleotides(100, seed=42)
        assert "".join(sorted(set(s))) == "ACGT"

    def test_length_keyword(self):
        s1 = tskit.random_nucleotides(length=10, seed=42)
        s2 = tskit.random_nucleotides(length=10, seed=42)
        assert s1 == s2

    def test_length_required(self):
        with pytest.raises(TypeError, match="required positional"):
            tskit.random_nucleotides()

    def test_seed_keyword_only(self):
        with pytest.raises(TypeError, match="1 positional"):
            tskit.random_nucleotides(10, 42)

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_seed_equality(self, seed):
        s1 = tskit.random_nucleotides(10, seed=seed)
        s2 = tskit.random_nucleotides(10, seed=seed)
        assert s1 == s2

    def test_different_seed_not_equal(self):
        s1 = tskit.random_nucleotides(20, seed=1)
        s2 = tskit.random_nucleotides(20, seed=2)
        assert s1 != s2

    def test_no_seed_different_values(self):
        s1 = tskit.random_nucleotides(20)
        s2 = tskit.random_nucleotides(20)
        assert s1 != s2

    @pytest.mark.parametrize("length", ["0", 0.1, np.array([1.1])[0]])
    def test_length_bad_value(self, length):
        with pytest.raises(ValueError, match="must be an integer"):
            tskit.random_nucleotides(length)

    @pytest.mark.parametrize("length", [{}, None])
    def test_length_bad_type(self, length):
        with pytest.raises(TypeError, match="argument must be a string"):
            tskit.random_nucleotides(length)
