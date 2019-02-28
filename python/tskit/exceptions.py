# MIT License
#
# Copyright (c) 2018-2019 Tskit Developers
# Copyright (c) 2017 University of Oxford
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
Exceptions defined in tskit.
"""
from _tskit import TskitException
from _tskit import LibraryError
from _tskit import FileFormatError
from _tskit import VersionTooNewError
from _tskit import VersionTooOldError

# Some exceptions are defined in the low-level module. In particular, the
# superclass of all exceptions for tskit is defined here. We define the
# docstrings here to avoid difficulties with compiling C code on
# readthedocs.

# TODO finalise this when working out the docs structure for tskit on rtd.

TskitException.__doc__ = "Superclass of all exceptions defined in tskit."
LibraryError.__doc__ = "Generic low-level error raised by the C library."
FileFormatError.__doc__ = "An error was detected in the file format."
VersionTooNewError.__doc__ = """
The version of the file is too new and cannot be read by the library.
"""
VersionTooOldError.__doc__ = """
The version of the file is too old and cannot be read by the library.
"""


class DuplicatePositionsError(TskitException):
    """
    Duplicate positions in the list of sites.
    """


class ProvenanceValidationError(TskitException):
    """
    A JSON document did non validate against the provenance schema.
    """
