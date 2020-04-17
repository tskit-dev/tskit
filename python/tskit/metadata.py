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
Classes for metadata decoding, encoding and validation
"""
import abc
import json
from typing import Any
from typing import Optional

import jsonschema

import tskit.exceptions as exceptions


class MetadataCodec(abc.ABC):
    @abc.abstractmethod
    def decode(self, encoded_bytes: bytes) -> Any:
        pass

    @abc.abstractmethod
    def encode(self, obj: Any) -> bytes:
        pass


class JSONCodec(MetadataCodec):
    name = "json"

    def decode(self, encoded_bytes: bytes) -> Any:
        return json.loads(encoded_bytes.decode())

    def encode(self, obj: Any) -> bytes:
        return json.dumps(obj).encode()


class AbstractMetadataSchema(abc.ABC):
    @abc.abstractmethod
    def to_str(self) -> Optional[str]:
        pass

    @abc.abstractmethod
    def validate_and_encode_row(self, row: Any) -> bytes:
        pass

    @abc.abstractmethod
    def decode_row(self, row: bytes) -> Any:
        pass


class MetadataSchema(AbstractMetadataSchema):
    @classmethod
    def from_str(cls, encoded_schema: Optional[str]) -> AbstractMetadataSchema:
        if encoded_schema == "":
            return NullMetadataSchema()
        else:
            try:
                decoded = json.loads(encoded_schema)
            except json.decoder.JSONDecodeError:
                raise ValueError(f"Metadata schema is not JSON, found {encoded_schema}")
            return cls(decoded["encoding"], decoded["schema"])

    def __init__(self, encoding, schema) -> None:
        self.encoding: str = encoding
        self.schema: str = schema

    def _get_codec(self) -> MetadataCodec:
        try:
            return {codec().name: codec() for codec in MetadataCodec.__subclasses__()}[
                self.encoding
            ]
        except KeyError:
            raise ValueError(f"Unrecognised metadata encoding:{self.encoding}")

    def to_str(self) -> str:
        return json.dumps({"encoding": self.encoding, "schema": self.schema})

    def validate_and_encode_row(self, row: Any) -> bytes:
        try:
            jsonschema.validate(row, self.schema)
        except jsonschema.exceptions.ValidationError as ve:
            raise exceptions.MetadataValidationError from ve
        return self._get_codec().encode(row)

    def decode_row(self, row: bytes) -> Any:
        return self._get_codec().decode(row)


class NullMetadataSchema(AbstractMetadataSchema):
    def to_str(self) -> None:
        return None

    def validate_and_encode_row(self, row: bytes) -> bytes:
        return row

    def decode_row(self, row: bytes) -> bytes:
        return row
