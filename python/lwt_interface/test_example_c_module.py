# flake8: noqa
import os
import sys

import pytest

# Make sure we use the local tskit version.

sys.path.insert(0, os.path.abspath("../"))

# An example of how to run the tests defined in the dict_encoding_testlib.py
# file for a given compiled version of the code.
import dict_encoding_testlib
import example_c_module
import tskit

# The test cases defined in dict_encoding_testlib all use the form
# lwt_module.LightweightTableCollection() to create an instance
# of LightweightTableCollection. So, by setting this variable in
# the module here, we can control which definition of the
# LightweightTableCollection gets used.
dict_encoding_testlib.lwt_module = example_c_module

from dict_encoding_testlib import *


def test_example_receiving():
    # The example_receiving function returns true if the first tree
    # has more than one root
    lwt = example_c_module.LightweightTableCollection()
    tables = tskit.TableCollection(1)
    lwt.fromdict(tables.asdict())
    # Our example function throws an error for an empty table collection
    with pytest.raises(ValueError, match="Table collection must be indexed"):
        example_c_module.example_receiving(lwt)

    # This tree sequence has one root so we get false
    tables = msprime.simulate(10).tables
    lwt.fromdict(tables.asdict())
    assert not example_c_module.example_receiving(lwt)

    # Add a root and we get true
    tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE)
    lwt.fromdict(tables.asdict())
    assert example_c_module.example_receiving(lwt)


def test_example_modifying():
    lwt = example_c_module.LightweightTableCollection()
    # The example_modifying function clears out the table and adds two rows
    tables = msprime.simulate(10, random_seed=42).tables
    assert tables.edges.num_rows == 18
    assert tables.nodes.num_rows == 19
    lwt.fromdict(tables.asdict())
    example_c_module.example_modifying(lwt)
    modified_tables = tskit.TableCollection.fromdict(lwt.asdict())
    assert modified_tables.edges.num_rows == 0
    assert modified_tables.nodes.num_rows == 2
