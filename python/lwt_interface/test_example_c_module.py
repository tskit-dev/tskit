# flake8: noqa
import os
import sys

# Make sure we use the local tskit version.
sys.path.insert(0, os.path.abspath("../"))

# An example of how to run the tests defined in the dict_encoding_testlib.py
# file for a given compiled version of the code.
import dict_encoding_testlib
import example_c_module

# The test cases defined in dict_encoding_testlib all use the form
# lwt_module.LightweightTableCollection() to create an instance
# of LightweightTableCollection. So, by setting this variable in
# the module here, we can control which definition of the
# LightweightTableCollection gets used.
dict_encoding_testlib.lwt_module = example_c_module

from dict_encoding_testlib import *
