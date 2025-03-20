# MIT License
#
# Copyright (c) 2023-2025 Tskit Developers
# Copyright (C) 2020-2021 University of Oxford
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
#
"""
Utilities for working with intervals and interval maps.
"""
from __future__ import annotations

import collections.abc
import numbers

import numpy as np

import tskit
import tskit.util as util


class RateMap(collections.abc.Mapping):
    """
    A class mapping a non-negative rate value to a set of non-overlapping intervals
    along the genome. Intervals for which the rate is unknown (i.e., missing data)
    are encoded by NaN values in the ``rate`` array.

    :param list position: A list of :math:`n+1` positions, starting at 0, and ending
        in the sequence length over which the RateMap will apply.
    :param list rate: A list of :math:`n` positive rates that apply between each
        position. Intervals with missing data are encoded by NaN values.
    """

    # The args are marked keyword only to give us some flexibility in how we
    # create class this in the future.
    def __init__(
        self,
        *,
        position,
        rate,
    ):
        # Making the arrays read-only guarantees rate and cumulative mass stay in sync
        # We prevent the arrays themselves being overwritten by making self.position,
        # etc properties.

        # TODO we always coerce the position type to float here, but we may not
        # want to do this. int32 is a perfectly good choice a lot of the time.
        self._position = np.array(position, dtype=float)
        self._position.flags.writeable = False
        self._rate = np.array(rate, dtype=float)
        self._rate.flags.writeable = False
        size = len(self._position)
        if size < 2:
            raise ValueError("Must have at least two positions")
        if len(self._rate) != size - 1:
            raise ValueError(
                "Rate array must have one less entry than the position array"
            )
        if self._position[0] != 0:
            raise ValueError("First position must be zero")

        span = self.span
        if np.any(span <= 0):
            bad_pos = np.where(span <= 0)[0] + 1
            raise ValueError(
                f"Position values not strictly increasing at indexes {bad_pos}"
            )
        if np.any(self._rate < 0):
            bad_rates = np.where(self._rate < 0)[0]
            raise ValueError(f"Rate values negative at indexes {bad_rates}")
        self._missing = np.isnan(self.rate)
        self._num_missing_intervals = np.sum(self._missing)
        if self._num_missing_intervals == len(self.rate):
            raise ValueError("All intervals are missing data")
        # We don't expose the cumulative mass array as a part of the array
        # API is it's not quite as obvious how it lines up for each interval.
        # It's really the sum of the mass up to but not including the current
        # interval, which is a bit confusing. Probably best to just leave
        # it as a function, so that people can sample at regular positions
        # along the genome anyway, emphasising that it's a continuous function,
        # not a step function like the other interval attributes.
        self._cumulative_mass = np.insert(np.nancumsum(self.mass), 0, 0)
        assert self._cumulative_mass[0] == 0
        self._cumulative_mass.flags.writeable = False

    @property
    def left(self):
        """
        The left position of each interval (inclusive).
        """
        return self._position[:-1]

    @property
    def right(self):
        """
        The right position of each interval (exclusive).
        """
        return self._position[1:]

    @property
    def mid(self):
        """
        Returns the midpoint of each interval.
        """
        mid = self.left + self.span / 2
        mid.flags.writeable = False
        return mid

    @property
    def span(self):
        """
        Returns the span (i.e., ``right - left``) of each of the intervals.
        """
        span = self.right - self.left
        span.flags.writeable = False
        return span

    @property
    def position(self):
        """
        The breakpoint positions between intervals. This is equal to the
        :attr:`~.RateMap.left` array with the :attr:`sequence_length`
        appended.
        """
        return self._position

    @property
    def rate(self):
        """
        The rate associated with each interval. Missing data is encoded
        by NaN values.
        """
        return self._rate

    @property
    def mass(self):
        r"""
        The "mass" of each interval, defined as the :attr:`~.RateMap.rate`
        :math:`\times` :attr:`~.RateMap.span`. This is NaN for intervals
        containing missing data.
        """
        return self._rate * self.span

    @property
    def missing(self):
        """
        A boolean array encoding whether each interval contains missing data.
        Equivalent to ``np.isnan(rate_map.rate)``
        """
        return self._missing

    @property
    def non_missing(self):
        """
        A boolean array encoding whether each interval contains non-missing data.
        Equivalent to ``np.logical_not(np.isnan(rate_map.rate))``
        """
        return ~self._missing

    #
    # Interval counts
    #

    @property
    def num_intervals(self) -> int:
        """
        The total number of intervals in this map. Equal to
        :attr:`~.RateMap.num_missing_intervals` +
        :attr:`~.RateMap.num_non_missing_intervals`.
        """
        return len(self._rate)

    @property
    def num_missing_intervals(self) -> int:
        """
        Returns the number of missing intervals, i.e., those in which the
        :attr:`~.RateMap.rate` value is a NaN.
        """
        return self._num_missing_intervals

    @property
    def num_non_missing_intervals(self) -> int:
        """
        The number of non missing intervals, i.e., those in which the
        :attr:`~.RateMap.rate` value is not a NaN.
        """
        return self.num_intervals - self.num_missing_intervals

    @property
    def sequence_length(self):
        """
        The sequence length covered by this map
        """
        return self.position[-1]

    @property
    def total_mass(self):
        """
        The cumulative total mass over the entire map.
        """
        return self._cumulative_mass[-1]

    @property
    def mean_rate(self):
        """
        The mean rate over this map weighted by the span covered by each rate.
        Unknown intervals are excluded.
        """
        total_span = np.sum(self.span[self.non_missing])
        return self.total_mass / total_span

    def get_rate(self, x):
        """
        Return the rate at the specified list of positions.

        .. note:: This function will return a NaN value for any positions
            that contain missing data.

        :param numpy.ndarray x: The positions for which to return values.
        :return: An array of rates, the same length as ``x``.
        :rtype: numpy.ndarray
        """
        loc = np.searchsorted(self.position, x, side="right") - 1
        if np.any(loc < 0) or np.any(loc >= len(self.rate)):
            raise ValueError("position out of bounds")
        return self.rate[loc]

    def get_cumulative_mass(self, x):
        """
        Return the cumulative mass of the map up to (but not including) a
        given point for a list of positions along the map. This is equal to
        the integral of the rate from 0 to the point.

        :param numpy.ndarray x: The positions for which to return values.

        :return: An array of cumulative mass values, the same length as ``x``
        :rtype: numpy.ndarray
        """
        x = np.array(x)
        if np.any(x < 0) or np.any(x > self.sequence_length):
            raise ValueError(f"Cannot have positions < 0 or > {self.sequence_length}")
        return np.interp(x, self.position, self._cumulative_mass)

    def find_index(self, x: float) -> int:
        """
        Returns the index of the interval that the specified position falls within,
        such that ``rate_map.left[index] <= x < self.rate_map.right[index]``.

        :param float x: The position to search.
        :return: The index of the interval containing this point.
        :rtype: int
        :raises KeyError: if the position is not contained in any of the intervals.
        """
        if x < 0 or x >= self.sequence_length:
            raise KeyError(f"Position {x} out of bounds")
        index = np.searchsorted(self.position, x, side="left")
        if x < self.position[index]:
            index -= 1
        assert self.left[index] <= x < self.right[index]
        return index

    def missing_intervals(self):
        """
        Returns the left and right coordinates of the intervals containing
        missing data in this map as a 2D numpy array
        with shape (:attr:`~.RateMap.num_missing_intervals`, 2). Each row
        of this returned array is therefore a ``left``, ``right`` tuple
        corresponding to the coordinates of the missing intervals.

        :return: A numpy array of the coordinates of intervals containing
            missing data.
        :rtype: numpy.ndarray
        """
        out = np.empty((self.num_missing_intervals, 2))
        out[:, 0] = self.left[self.missing]
        out[:, 1] = self.right[self.missing]
        return out

    def asdict(self):
        return {"position": self.position, "rate": self.rate}

    #
    # Dunder methods. We implement the Mapping protocol via __iter__, __len__
    # and __getitem__. We have some extra semantics for __getitem__, providing
    # slice notation.
    #

    def __iter__(self):
        # The clinching argument for using mid here is that if we used
        # left instead we would have
        #   RateMap([0, 1], [0.1]) == RateMap([0, 100], [0.1])
        # by the inherited definition of equality since the dictionary items
        # would be equal.
        # Similarly, we only return the midpoints of known intervals
        # because NaN values are not equal, and we would need to do
        # something to work around this. It seems reasonable that
        # this high-level operation returns the *known* values only
        # anyway.
        yield from self.mid[self.non_missing]

    def __len__(self):
        return np.sum(self.non_missing)

    def __getitem__(self, key):
        if isinstance(key, slice):
            if key.step is not None:
                raise TypeError("Only interval slicing is supported")
            return self.slice(key.start, key.stop)
        if isinstance(key, numbers.Number):
            index = self.find_index(key)
            if np.isnan(self.rate[index]):
                # To be consistent with the __iter__ definition above we
                # don't consider these missing positions to be "in" the map.
                raise KeyError(f"Position {key} is within a missing interval")
            return self.rate[index]
        # TODO we could implement numpy array indexing here and call
        # to get_rate. Note we'd need to take care that we return a keyerror
        # if the returned array contains any nans though.
        raise KeyError("Key {key} not in map")

    def _text_header_and_rows(self, limit=None):
        headers = ("left", "right", "mid", "span", "rate")
        num_rows = len(self.left)
        rows = []
        row_indexes = util.truncate_rows(num_rows, limit)
        for j in row_indexes:
            if j == -1:
                rows.append(f"__skipped__{num_rows - limit}")
            else:
                rows.append(
                    [
                        f"{self.left[j]:.10g}",
                        f"{self.right[j]:.10g}",
                        f"{self.mid[j]:.10g}",
                        f"{self.span[j]:.10g}",
                        f"{self.rate[j]:.2g}",
                    ]
                )
        return headers, rows

    def __str__(self):
        header, rows = self._text_header_and_rows(
            limit=tskit._print_options["max_lines"]
        )
        table = util.unicode_table(
            rows=rows,
            header=header,
            column_alignments="<<>>>",
        )
        return table

    def _repr_html_(self):
        header, rows = self._text_header_and_rows(
            limit=tskit._print_options["max_lines"]
        )
        return util.html_table(rows, header=header)

    def __repr__(self):
        return f"RateMap(position={repr(self.position)}, rate={repr(self.rate)})"

    #
    # Methods for building rate maps.
    #

    def copy(self) -> RateMap:
        """
        Returns a deep copy of this RateMap.
        """
        # We take read-only copies of the arrays in the constructor anyway, so
        # no need for copying.
        return RateMap(position=self.position, rate=self.rate)

    def slice(self, left=None, right=None, *, trim=False) -> RateMap:  # noqa: A003
        """
        Returns a subset of this rate map in the specified interval.

        :param float left: The left coordinate (inclusive) of the region to keep.
            If ``None``, defaults to 0.
        :param float right: The right coordinate (exclusive) of the region to keep.
            If ``None``, defaults to the sequence length.
        :param bool trim: If True, remove the flanking regions such that the
            sequence length of the new rate map is ``right`` - ``left``. If ``False``
            (default), do not change the coordinate system and mark the flanking
            regions as "unknown".
        :return: A new RateMap instance
        :rtype: RateMap
        """
        left = 0 if left is None else left
        right = self.sequence_length if right is None else right
        if not (0 <= left < right <= self.sequence_length):
            raise KeyError(f"Invalid slice: left={left}, right={right}")

        i = self.find_index(left)
        j = i + np.searchsorted(self.position[i:], right, side="right")
        if right > self.position[j - 1]:
            j += 1

        position = self.position[i:j].copy()
        rate = self.rate[i : j - 1].copy()
        position[0] = left
        position[-1] = right

        if trim:
            # Return trimmed map with changed coords
            return RateMap(position=position - left, rate=rate)

        # Need to check regions before & after sliced region are filled out:
        if left != 0:
            if np.isnan(rate[0]):
                position[0] = 0  # Extend
            else:
                rate = np.insert(rate, 0, np.nan)  # Prepend
                position = np.insert(position, 0, 0)
        if right != self.position[-1]:
            if np.isnan(rate[-1]):
                position[-1] = self.sequence_length  # Extend
            else:
                rate = np.append(rate, np.nan)  # Append
                position = np.append(position, self.position[-1])
        return RateMap(position=position, rate=rate)

    @staticmethod
    def uniform(sequence_length, rate) -> RateMap:
        """
        Create a uniform rate map
        """
        return RateMap(position=[0, sequence_length], rate=[rate])

    @staticmethod
    def read_hapmap(
        fileobj,
        sequence_length=None,
        *,
        has_header=True,
        position_col=None,
        rate_col=None,
        map_col=None,
    ):
        # Black barfs with an INTERNAL_ERROR trying to reformat this docstring,
        # so we explicitly disable reformatting here.
        # fmt: off
        """
        Parses the specified file in HapMap format and returns a :class:`.RateMap`.
        HapMap files must white-space-delimited, and by default are assumed to
        contain a single header line (which is ignored). Each subsequent line
        then contains a physical position (in base pairs) and either a genetic
        map position (in centiMorgans) or a recombination rate (in centiMorgans
        per megabase). The value in the rate column in a given line gives the
        constant rate between the physical position in that line (inclusive) and the
        physical position on the next line (exclusive).
        By default, the second column of the file is taken
        as the physical position and the fourth column is taken as the genetic
        position, as seen in the following sample of the format::

            Chromosome	Position(bp)  Rate(cM/Mb)  Map(cM)
            chr10       48232         0.1614       0.002664
            chr10       48486         0.1589       0.002705
            chr10       50009         0.159        0.002947
            chr10       52147         0.1574       0.003287
            ...
            chr10	133762002     3.358        181.129345
            chr10	133766368     0.000        181.144008

        In the example above, the first row has a nonzero genetic map position
        (last column, cM), implying a nonzero recombination rate before that
        position, that is assumed to extend to the start of the chromosome
        (at position 0 bp). However, if the first line has a nonzero bp position
        (second column) and a zero genetic map position (last column, cM),
        then the recombination rate before that position is *unknown*, producing
        :ref:`missing data <sec_rate_maps_missing>`.

        .. note::
            The rows are all assumed to come from the same contig, and the
            first column is currently ignored. Therefore if you have a single
            file containing several contigs or chromosomes, you must must split
            it up into multiple files, and pass each one separately to this
            function.

        :param str fileobj: Filename or file to read. This is passed directly
            to :func:`numpy.loadtxt`, so if the filename extension is .gz or .bz2,
            the file is decompressed first
        :param float sequence_length: The total length of the map. If ``None``,
            then assume it is the last physical position listed in the file.
            Otherwise it must be greater then or equal to the last physical
            position in the file, and the region between the last physical position
            and the sequence_length is padded with a rate of zero.
        :param bool has_header: If True (default), assume the file has a header row
            and ignore the first line of the file.
        :param int position_col: The zero-based index of the column in the file
            specifying the physical position in base pairs. If ``None`` (default)
            assume an index of 1 (i.e. the second column).
        :param int rate_col: The zero-based index of the column in the file
            specifying the rate in cM/Mb. If ``None`` (default) do not use the rate
            column, but calculate rates using the genetic map positions, as
            specified in ``map_col``. If the rate column is used, the
            interval from 0 to first physical position in the file is marked as
            unknown, and the last value in the rate column must be zero.
        :param int map_col: The zero-based index of the column in the file
            specifying the genetic map position in centiMorgans. If ``None``
            (default), assume an index of 3 (i.e. the fourth column). If the first
            genetic position is 0 the interval from position 0 to the first
            physical position in the file is marked as unknown. Otherwise, act
            as if an additional row, specifying physical position 0 and genetic
            position 0, exists at the start of the file.
        :return: A RateMap object.
        :rtype: RateMap
        """
        # fmt: on
        column_defs = {}  # column definitions passed to np.loadtxt
        if rate_col is None and map_col is None:
            # Default to map_col
            map_col = 3
        elif rate_col is not None and map_col is not None:
            raise ValueError("Cannot specify both rate_col and map_col")
        if map_col is not None:
            column_defs[map_col] = ("map", float)
        else:
            column_defs[rate_col] = ("rate", float)
        position_col = 1 if position_col is None else position_col
        if position_col in column_defs:
            raise ValueError(
                "Cannot specify the same columns for position_col and "
                "rate_col or map_col"
            )
        column_defs[position_col] = ("pos", int)

        column_names = [c[0] for c in column_defs.values()]
        column_data = np.loadtxt(
            fileobj,
            skiprows=1 if has_header else 0,
            dtype=list(column_defs.values()),
            usecols=list(column_defs.keys()),
            unpack=True,
        )
        data = dict(zip(column_names, column_data))

        if "map" not in data:
            assert "rate" in data
            if data["rate"][-1] != 0:
                raise ValueError("The last entry in the 'rate' column must be zero")
            pos_Mb = data["pos"] / 1e6
            map_pos = np.cumsum(data["rate"][:-1] * np.diff(pos_Mb))
            data["map"] = np.insert(map_pos, 0, 0) / 100
        else:
            data["map"] /= 100  # Convert centiMorgans to Morgans
        if len(data["map"]) == 0:
            raise ValueError("Empty hapmap file")

        # TO DO: read in chrom name from col 0 and poss set as .name
        # attribute on the RateMap

        physical_positions = data["pos"]
        genetic_positions = data["map"]
        start = physical_positions[0]
        end = physical_positions[-1]

        if genetic_positions[0] > 0 and start == 0:
            raise ValueError(
                "The map distance at the start of the chromosome must be zero"
            )
        if start > 0:
            physical_positions = np.insert(physical_positions, 0, 0)
            if genetic_positions[0] > 0:
                # Exception for a map that starts > 0cM: include the start rate
                # in the mean
                start = 0
            genetic_positions = np.insert(genetic_positions, 0, 0)

        if sequence_length is not None:
            if sequence_length < end:
                raise ValueError(
                    "The sequence_length cannot be less than the last physical position "
                    f" ({physical_positions[-1]})"
                )
            if sequence_length > end:
                physical_positions = np.append(physical_positions, sequence_length)
                genetic_positions = np.append(genetic_positions, genetic_positions[-1])

        assert genetic_positions[0] == 0
        rate = np.diff(genetic_positions) / np.diff(physical_positions)
        if start != 0:
            rate[0] = np.nan
        if end != physical_positions[-1]:
            rate[-1] = np.nan
        return RateMap(position=physical_positions, rate=rate)
