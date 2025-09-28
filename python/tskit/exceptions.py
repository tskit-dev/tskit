# MIT License
#
# Copyright (c) 2018-2021 Tskit Developers
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
from _tskit import FileFormatError  # noqa: F401
from _tskit import IdentityPairsNotStoredError  # noqa: F401
from _tskit import IdentitySegmentsNotStoredError  # noqa: F401
from _tskit import LibraryError  # noqa: F401
from _tskit import TskitException  # noqa: F401
from _tskit import VersionTooNewError  # noqa: F401
from _tskit import VersionTooOldError  # noqa: F401


class DuplicatePositionsError(TskitException):
    """
    Duplicate positions in the list of sites.
    """


class ProvenanceValidationError(TskitException):
    """
    A JSON document did not validate against the provenance schema.
    """


class MetadataValidationError(TskitException):
    """
    A metadata object did not validate against the provenance schema.
    """


class MetadataSchemaValidationError(TskitException):
    """
    A metadata schema object did not validate against the metaschema.
    """


class MetadataEncodingError(TskitException):
    """
    A metadata object was of a type that could not be encoded
    """


class ImmutableTableError(ValueError):
    """
    Raised when attempting to modify an immutable table view.

    Use TreeSequence.dump_tables() to get a mutable copy.
    """
