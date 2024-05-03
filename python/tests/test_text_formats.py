# MIT License
#
# Copyright (c) 2021-2024 Tskit Developers
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
Test cases for converting fam file to tskit
"""
import dataclasses
import tempfile
from dataclasses import asdict

import numpy as np
import pytest

import tskit


@dataclasses.dataclass
class FamEntry:
    fid: str = "0"
    iid: str = "0"
    pat: str = "0"
    mat: str = "0"
    sex: str = "0"
    phen: str = None

    def get_row(self, delimiter="\t"):
        return delimiter.join([x for x in asdict(self).values() if x is not None])


class TestParseFam:
    """
    Tests for the parse_fam function.
    """

    def get_parsed_fam(self, entries, delimiter="\t"):
        content = "\n".join([entry.get_row(delimiter=delimiter) for entry in entries])
        with tempfile.TemporaryFile() as f:
            f.write(bytes(content, "utf-8"))
            f.seek(0)
            return tskit.parse_fam(f)

    def test_empty_file(self):
        entries = []
        with pytest.warns(UserWarning):
            tb = self.get_parsed_fam(entries=entries)
        assert len(tb) == 0

    @pytest.mark.parametrize("iid", ["1", "a", "100", "abc"])
    def test_single_line(self, iid):
        entries = [FamEntry(iid=iid)]
        tb = self.get_parsed_fam(entries=entries)
        assert len(tb) == 1
        assert np.array_equal(tb[0].parents, [-1, -1])
        assert tb[0].metadata["plink_fid"] == "0"
        assert tb[0].metadata["plink_iid"] == str(iid)
        assert tb[0].metadata["sex"] == 0

    @pytest.mark.parametrize("iids", [("1", "2"), ("a", "b")])
    def test_multiple_line_file(self, iids):
        # test both integer and string IIDs
        iid1, iid2 = iids
        entries = [FamEntry(iid=iid1), FamEntry(iid=iid2)]
        tb = self.get_parsed_fam(entries=entries)
        assert len(tb) == 2
        for idx in range(2):
            assert np.array_equal(tb[idx].parents, [-1, -1])
            assert tb[idx].metadata["plink_fid"] == "0"
            assert tb[idx].metadata["plink_iid"] == str(entries[idx].iid)
            assert tb[idx].metadata["sex"] == 0

    @pytest.mark.parametrize("n_cols", range(1, 5))
    def test_insufficient_cols(self, n_cols):
        fields = list(asdict(FamEntry()))
        entry = FamEntry(iid="1")
        for field in fields[n_cols:]:
            entry.__setattr__(field, None)
        with pytest.raises(  # noqa B017
            Exception
        ):  # Have to be non-specific here as numpy 1.23 changed the exception type
            self.get_parsed_fam(entries=[entry])

    def test_unrelated_duplicate_iids(self):
        # Individuals have the same IID, but are in different families
        entries = [FamEntry(iid="1"), FamEntry(fid="1", iid="1")]
        tb = self.get_parsed_fam(entries=entries)
        assert len(tb) == 2
        assert tb[0].metadata["plink_fid"] == "0"
        assert tb[0].metadata["plink_iid"] == "1"
        assert tb[1].metadata["plink_fid"] == "1"
        assert tb[1].metadata["plink_iid"] == "1"

    def test_duplicate_rows(self):
        entries = [FamEntry(iid="1"), FamEntry(iid="1")]
        with pytest.raises(ValueError):
            self.get_parsed_fam(entries=entries)

    def test_space_delimited(self):
        entries = [FamEntry(iid="1")]
        tb = self.get_parsed_fam(entries=entries, delimiter=" ")
        assert np.array_equal(tb[0].parents, [-1, -1])
        assert tb[0].metadata["plink_fid"] == "0"
        assert tb[0].metadata["plink_iid"] == "1"
        assert tb[0].metadata["sex"] == 0

    def test_missing_phen_col(self):
        entries = [FamEntry(iid="1", phen="1")]
        tb = self.get_parsed_fam(entries=entries)

        entries = [FamEntry(iid="1")]  # remove last column (PHEN column)
        tb_missing = self.get_parsed_fam(entries=entries)

        assert tb == tb_missing

    @pytest.mark.parametrize("sex", [-2, 3, "F"])
    def test_bad_sex_value(self, sex):
        entries = [FamEntry(iid="1", sex=str(sex))]
        with pytest.raises(ValueError):
            self.get_parsed_fam(entries=entries)

    def test_empty_sex_value(self):
        entries = [FamEntry(iid="1", sex="")]
        with pytest.raises(  # noqa B017
            Exception
        ):  # Have to be non-specific here as numpy 1.23 changed the exception type
            self.get_parsed_fam(entries=entries)

    def test_single_family_map_parent_ids(self):
        # PAT is mapped if the individual exists in the dataset
        entries = [FamEntry(iid="1"), FamEntry(iid="2", pat="1")]
        tb = self.get_parsed_fam(entries=entries)
        assert np.array_equal(tb[1].parents, [0, -1])

        # MAT is mapped if the individual exists in the dataset
        entries = [FamEntry(iid="1"), FamEntry(iid="2", mat="1")]
        tb = self.get_parsed_fam(entries=entries)
        assert np.array_equal(tb[1].parents, [-1, 0])

        # both parent IDs are remapped if the both parents exist in the dataset
        entries = [
            FamEntry(iid="1"),
            FamEntry(iid="2"),
            FamEntry(iid="3", pat="1", mat="2"),
        ]
        tb = self.get_parsed_fam(entries=entries)
        assert np.array_equal(tb[2].parents, [0, 1])

    def test_missing_parent_id(self):
        # KeyError raised if at least one parent (PAT) does not exist in dataset
        entries = [
            FamEntry(iid="2"),
            FamEntry(iid="3", pat="1", mat="2"),
        ]
        with pytest.raises(KeyError):
            self.get_parsed_fam(entries=entries)

        # KeyError raised if at least one parent (MAT) does not exist in dataset
        entries = [
            FamEntry(iid="1"),
            FamEntry(iid="3", pat="1", mat="2"),
        ]
        with pytest.raises(KeyError):
            self.get_parsed_fam(entries=entries)

        # KeyError raised if both parents do not exist in dataset
        entries = [FamEntry(iid="1", pat="2", mat="3")]
        with pytest.raises(KeyError):
            self.get_parsed_fam(entries=entries)

    def test_multiple_family_map_parent_ids(self):
        # parents mapped correctly when the same parent ID is used in different families
        entries = [
            FamEntry(iid="2"),
            FamEntry(iid="1"),
            FamEntry(fid="1", iid="2"),
            FamEntry(fid="1", iid="1"),
            FamEntry(iid="3", pat="1", mat="2"),
            FamEntry(fid="1", iid="3", pat="1", mat="2"),
        ]
        tb = self.get_parsed_fam(entries=entries)
        for idx in range(4):
            assert np.array_equal(tb[idx].parents, [-1, -1])
        assert np.array_equal(tb[4].parents, [1, 0])
        assert np.array_equal(tb[5].parents, [3, 2])

        # KeyError raised when FID does not match, even if parent ID matches
        entries = [
            FamEntry(iid="2"),
            FamEntry(iid="1"),
            FamEntry(iid="3", pat="1", mat="2"),
            FamEntry(
                fid="1", iid="1", pat="2", mat="3"
            ),  # there is no parent with FID=1, IID=3
            FamEntry(fid="1", iid="2"),
        ]
        with pytest.raises(KeyError):
            self.get_parsed_fam(entries)

    def test_grandparents(self):
        entries = [
            FamEntry(iid="4"),
            FamEntry(iid="3"),
            FamEntry(iid="2"),
            FamEntry(iid="1"),
            FamEntry(iid="6", pat="3", mat="4"),
            FamEntry(iid="5", pat="1", mat="2"),
            FamEntry(iid="7", pat="5", mat="6"),
        ]
        tb = self.get_parsed_fam(entries=entries)
        assert np.array_equal(tb[4].parents, [1, 0])
        assert np.array_equal(tb[5].parents, [3, 2])
        assert np.array_equal(tb[6].parents, [5, 4])

    def test_children_before_parents(self, tmp_path):
        entries = [
            FamEntry(iid="1", pat="2", mat="3"),
            FamEntry(iid="2"),
            FamEntry(iid="3"),
        ]
        content = "\n".join([entry.get_row() for entry in entries])
        fam_path = f"{tmp_path}/test.fam"
        with open(fam_path, "w+") as f:
            f.write(content)
            f.seek(0)
            tb = tskit.parse_fam(f)

        tc = tskit.TableCollection(1)
        # Issue 1489 will make this better
        tc.individuals.metadata_schema = tb.metadata_schema
        for row in tb:
            tc.individuals.append(row)
        tc.tree_sequence()  # creating tree sequence should succeed
