# MIT License
#
# Copyright (c) 2020 Tskit Developers
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
import copy
import json
from typing import Any
from typing import Optional
from typing import Type

import jsonschema

import tskit.exceptions as exceptions


def replace_root_refs(obj):
    if type(obj) == list:
        return [replace_root_refs(j) for j in obj]
    elif type(obj) == dict:
        ret = {k: replace_root_refs(v) for k, v in obj.items()}
        if ret.get("$ref") == "#":
            ret["$ref"] = "#/definitions/root"
        return ret
    else:
        return obj


class TSKITMetadataSchemaValidator(jsonschema.validators.Draft7Validator):
    """
    Our schema is the Draft7Validator schema with added codec information.
    """

    META_SCHEMA: dict = copy.deepcopy(jsonschema.validators.Draft7Validator.META_SCHEMA)
    # We need a top-level only required property so we need to rewite any reference
    # to the top-level schema to a copy in a definition.
    META_SCHEMA = replace_root_refs(META_SCHEMA)
    META_SCHEMA["definitions"]["root"] = copy.deepcopy(META_SCHEMA)
    META_SCHEMA["codec"] = {"type": "string"}
    META_SCHEMA["required"] = ["codec"]


class AbstractMetadataCodec(metaclass=abc.ABCMeta):
    """
    Superclass of all MetadataCodecs.
    """

    def __init__(self, schema: dict) -> None:
        raise NotImplementedError  # pragma: no cover

    @abc.abstractmethod
    def encode(self, obj: Any) -> bytes:
        raise NotImplementedError  # pragma: no cover

    @abc.abstractmethod
    def decode(self, encoded: bytes) -> Any:
        raise NotImplementedError  # pragma: no cover


codec_registry = {}


def register_metadata_codec(
    codec_cls: Type[AbstractMetadataCodec], codec_id: str
) -> None:
    """
    Register a metadata codec class.
    This function maintains a mapping from metadata codec identifiers used in schemas
    to codec classes. When a codec class is registered, it will replace any class
    previously registered under the same codec identifier, if present.
    See :ref:`sec_tutorial_metadata_custom_codec` for example usage.

    :param str codec_id: String to use to refer to the codec in the schema.
    """
    codec_registry[codec_id] = codec_cls


class JSONCodec(AbstractMetadataCodec):
    def __init__(self, schema: dict) -> None:
        pass

    def encode(self, obj: Any) -> bytes:
        return json.dumps(obj).encode()

    def decode(self, encoded: bytes) -> Any:
        return json.loads(encoded.decode())


register_metadata_codec(JSONCodec, "json")


class NOOPCodec(AbstractMetadataCodec):
    def __init__(self, schema: dict) -> None:
        pass

    def encode(self, data: bytes) -> bytes:
        return data

    def decode(self, data: bytes) -> bytes:
        return data


def validate_bytes(data: Optional[bytes]) -> None:
    if data is not None and not isinstance(data, bytes):
        raise TypeError(
            f"If no encoding is set metadata should be bytes, found {type(bytes)}"
        )


class MetadataSchema:
    """
    Class for validating, encoding and decoding metadata.

    :param dict schema: A dict containing a valid JSONSchema object.
    """

    def __init__(self, schema: Optional[dict]) -> None:
        self._schema = schema

        if schema is None:
            self._string = ""
            self._validate_row = validate_bytes
            self.encode_row = NOOPCodec({}).encode
            self.decode_row = NOOPCodec({}).decode
        else:
            try:
                TSKITMetadataSchemaValidator.check_schema(schema)
            except jsonschema.exceptions.SchemaError as ve:
                raise exceptions.MetadataSchemaValidationError from ve
            codec = schema["codec"]
            try:
                codec_instance = codec_registry[codec](schema)
            except KeyError:
                raise exceptions.MetadataSchemaValidationError(
                    f"Unrecognised metadata codec '{schema['codec']}'. "
                    f"Valid options are {str(list(codec_registry.keys()))}."
                )
            self._string = json.dumps(schema)
            self._validate_row = TSKITMetadataSchemaValidator(schema).validate
            self.encode_row = codec_instance.encode
            self.decode_row = codec_instance.decode

    def __str__(self) -> str:
        return self._string

    @property
    def schema(self) -> Optional[dict]:
        # Make schema read-only
        return self._schema

    def validate_and_encode_row(self, row: Any) -> bytes:
        """
        Validate a row of metadata against this schema and return the encoded
        representation.
        """
        try:
            self._validate_row(row)
        except jsonschema.exceptions.ValidationError as ve:
            raise exceptions.MetadataValidationError from ve
        return self.encode_row(row)

    def decode_row(self, row: bytes) -> Any:
        """
        Decode an encoded row of metadata, note that no validation is performed.
        """
        # Set by __init__
        pass  # pragma: no cover

    def encode_row(self, row: bytes) -> Any:
        """
        Encode an encoded row of metadata, note that no validation is performed.
        """
        # Set by __init__
        pass  # pragma: no cover


def parse_metadata_schema(encoded_schema: str) -> MetadataSchema:
    """
    Create a schema object from its string encoding. The exact class returned is
    determined by the ``encoding`` specification in the string.

    :param str encoded_schema: The string encoded schema.
    :return: A subclass of AbstractMetadataSchema.
    """
    if encoded_schema == "":
        return MetadataSchema(schema=None)
    else:
        try:
            decoded = json.loads(encoded_schema)
        except json.decoder.JSONDecodeError:
            raise ValueError(f"Metadata schema is not JSON, found {encoded_schema}")
        return MetadataSchema(decoded)
