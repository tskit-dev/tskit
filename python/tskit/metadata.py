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
import collections
import copy
import json
import pprint
import struct
from itertools import islice
from typing import Any
from typing import Mapping
from typing import Optional
from typing import Type

import jsonschema

import tskit
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


# Our schema is the Draft7Validator schema with added codec information.
TSKITMetadataSchemaValidator = jsonschema.validators.extend(
    jsonschema.validators.Draft7Validator
)
META_SCHEMA: Mapping[str, Any] = copy.deepcopy(TSKITMetadataSchemaValidator.META_SCHEMA)
# We need a top-level only required property so we need to rewrite any reference
# to the top-level schema to a copy in a definition.
META_SCHEMA = replace_root_refs(META_SCHEMA)
META_SCHEMA["definitions"]["root"] = copy.deepcopy(META_SCHEMA)
META_SCHEMA["codec"] = {"type": "string"}
META_SCHEMA["required"] = ["codec"]
# For interoperability reasons, force the top-level to be an object or union
# of object and null
META_SCHEMA["properties"]["type"] = {"enum": ["object", ["object", "null"]]}
TSKITMetadataSchemaValidator.META_SCHEMA = META_SCHEMA


class AbstractMetadataCodec(metaclass=abc.ABCMeta):
    """
    Superclass of all MetadataCodecs.
    """

    def __init__(self, schema: Mapping[str, Any]) -> None:
        raise NotImplementedError  # pragma: no cover

    @classmethod
    def modify_schema(self, schema: Mapping) -> Mapping:
        return schema

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

    :param str codec_id: String to use to refer to the codec in the schema.
    """
    codec_registry[codec_id] = codec_cls


class JSONCodec(AbstractMetadataCodec):
    def __init__(self, schema: Mapping[str, Any]) -> None:
        pass

    def encode(self, obj: Any) -> bytes:
        return tskit.canonical_json(obj).encode()

    def decode(self, encoded: bytes) -> Any:
        return json.loads(encoded.decode())


register_metadata_codec(JSONCodec, "json")


class NOOPCodec(AbstractMetadataCodec):
    def __init__(self, schema: Mapping[str, Any]) -> None:
        pass

    def encode(self, data: bytes) -> bytes:
        return data

    def decode(self, data: bytes) -> bytes:
        return data


def binary_format_validator(validator, types, instance, schema):
    # We're hooking into jsonschemas validaiton code here, which works by creating
    # generators of exceptions, hence the yielding

    # Make sure the normal type validation gets done
    yield from jsonschema._validators.type(validator, types, instance, schema)

    # Non-composite types must have a binaryFormat
    if (
        validator.is_type(instance, "object")
        and (
            instance.get("type")
            not in (None, "object", "array", "null", ["object", "null"])
        )
        and "binaryFormat" not in instance
    ):
        yield jsonschema.ValidationError(
            f"{instance['type']} type must have binaryFormat set"
        )
    # null type must be padding
    if (
        validator.is_type(instance, "object")
        and instance.get("type") == "null"
        and "binaryFormat" in instance
        and instance["binaryFormat"][-1] != "x"
    ):
        yield jsonschema.ValidationError(
            'null type binaryFormat must be padding ("x") if set'
        )


def required_validator(validator, required, instance, schema):
    # Do the normal validation
    yield from jsonschema._validators.required(validator, required, instance, schema)

    # For struct codec if a property is not required, then it must have a default
    for prop, sub_schema in instance["properties"].items():
        if prop not in instance["required"] and "default" not in sub_schema:
            yield jsonschema.ValidationError(
                f"Optional property '{prop}' must have" f" a default value"
            )


StructCodecSchemaValidator = jsonschema.validators.extend(
    TSKITMetadataSchemaValidator,
    {"type": binary_format_validator, "required": required_validator},
)
META_SCHEMA: Mapping[str, Any] = copy.deepcopy(StructCodecSchemaValidator.META_SCHEMA)
# No union types
META_SCHEMA["definitions"]["root"]["properties"]["type"] = {
    "$ref": "#/definitions/simpleTypes"
}
# No hetrogeneous arrays
META_SCHEMA["properties"]["items"] = {"$ref": "#/definitions/root"}
META_SCHEMA["definitions"]["root"]["properties"]["items"] = META_SCHEMA["properties"][
    "items"
]
# binaryFormat matches regex
META_SCHEMA["properties"]["binaryFormat"] = {
    "type": "string",
    "pattern": r"^([cbB\?hHiIlLqQfd]|\d*[spx])$",
}
META_SCHEMA["definitions"]["root"]["properties"]["binaryFormat"] = META_SCHEMA[
    "properties"
]["binaryFormat"]
# arrayLengthFormat matches regex and has default
META_SCHEMA["properties"]["arrayLengthFormat"] = {
    "type": "string",
    "pattern": r"^[BHILQ]$",
    "default": "L",
}
META_SCHEMA["definitions"]["root"]["properties"]["arrayLengthFormat"] = META_SCHEMA[
    "properties"
]["arrayLengthFormat"]
# index is numeric
META_SCHEMA["properties"]["index"] = {"type": "number"}
META_SCHEMA["definitions"]["root"]["properties"]["index"] = META_SCHEMA["properties"][
    "index"
]
# stringEncoding is string and has default
META_SCHEMA["properties"]["stringEncoding"] = {"type": "string", "default": "utf-8"}
META_SCHEMA["definitions"]["root"]["properties"]["stringEncoding"] = META_SCHEMA[
    "properties"
]["stringEncoding"]
# nullTerminated is a boolean
META_SCHEMA["properties"]["nullTerminated"] = {"type": "boolean"}
META_SCHEMA["definitions"]["root"]["properties"]["nullTerminated"] = META_SCHEMA[
    "properties"
]["nullTerminated"]
# noLengthEncodingExhaustBuffer is a boolean
META_SCHEMA["properties"]["noLengthEncodingExhaustBuffer"] = {"type": "boolean"}
META_SCHEMA["definitions"]["root"]["properties"][
    "noLengthEncodingExhaustBuffer"
] = META_SCHEMA["properties"]["noLengthEncodingExhaustBuffer"]
StructCodecSchemaValidator.META_SCHEMA = META_SCHEMA


class StructCodec(AbstractMetadataCodec):
    """
    Codec that encodes data using struct. Note that this codec has extra restrictions
    Namely that object keys must be fixed (all present and no extra); each entry should
    have a binaryFormat; that arrays are homogeneous and that types are not unions.
    """

    @classmethod
    def order_by_index(cls, obj, do_sort=False):
        """
        Take a schema and recursively convert any dict that is under the key
        name ``properties`` to an OrderedDict.
        """
        if isinstance(obj, collections.abc.Mapping):
            items = obj.items()
            if do_sort:
                # Python sort is stable so we can do the sorts in reverse priority
                items = sorted(items, key=lambda k_v: k_v[0])
                items = sorted(items, key=lambda k_v: k_v[1].get("index", 0))
            items = ((k, cls.order_by_index(v, k == "properties")) for k, v in items)
            if do_sort:
                return collections.OrderedDict(items)
            else:
                return dict(items)
        elif isinstance(obj, list) or isinstance(obj, tuple):
            return [cls.order_by_index(v, False) for v in obj]
        else:
            return obj

    @classmethod
    def make_decode(cls, sub_schema):
        """
        Create a function that can decode objects of this schema
        """
        if set(sub_schema["type"]) == {"object", "null"}:
            return StructCodec.make_object_or_null_decode(sub_schema)
        else:
            return {
                "array": StructCodec.make_array_decode,
                "object": StructCodec.make_object_decode,
                "string": StructCodec.make_string_decode,
                "null": StructCodec.make_null_decode,
                "number": StructCodec.make_numeric_decode,
                "integer": StructCodec.make_numeric_decode,
                "boolean": StructCodec.make_numeric_decode,
            }[sub_schema["type"]](sub_schema)

    @classmethod
    def make_array_decode(cls, sub_schema):
        element_decoder = StructCodec.make_decode(sub_schema["items"])
        array_length_f = "<" + sub_schema.get("arrayLengthFormat", "L")
        array_length_size = struct.calcsize(array_length_f)
        exhaust_buffer = sub_schema.get("noLengthEncodingExhaustBuffer", False)

        def array_decode(buffer):
            array_length = struct.unpack(
                array_length_f, bytes(islice(buffer, array_length_size))
            )[0]
            return [element_decoder(buffer) for _ in range(array_length)]

        def array_decode_exhaust(buffer):
            ret = []
            while True:
                try:
                    ret.append(element_decoder(buffer))
                except struct.error as e:
                    if "unpack requires a buffer" in str(e):
                        break
                    else:
                        raise e
            return ret

        if exhaust_buffer:
            return array_decode_exhaust
        else:
            return array_decode

    @classmethod
    def make_object_decode(cls, sub_schema):
        sub_decoders = {
            key: StructCodec.make_decode(prop)
            for key, prop in sub_schema["properties"].items()
        }
        return lambda buffer: {
            key: sub_decoder(buffer) for key, sub_decoder in sub_decoders.items()
        }

    @classmethod
    def make_object_or_null_decode(cls, sub_schema):
        sub_decoders = {
            key: StructCodec.make_decode(prop)
            for key, prop in sub_schema["properties"].items()
        }

        def decode_object_or_null(buffer):
            # We have to check the buffer length for null, as the islices in
            # sub-decoders won't raise StopIteration
            buffer = list(buffer)
            if len(buffer) == 0:
                return None
            else:
                buffer = iter(buffer)
                return {
                    key: sub_decoder(buffer)
                    for key, sub_decoder in sub_decoders.items()
                }

        return decode_object_or_null

    @classmethod
    def make_string_decode(cls, sub_schema):
        f = "<" + sub_schema["binaryFormat"]
        size = struct.calcsize(f)
        encoding = sub_schema.get("stringEncoding", "utf-8")
        null_terminated = sub_schema.get("nullTerminated", False)
        if not null_terminated:
            return lambda buffer: struct.unpack(f, bytes(islice(buffer, size)))[
                0
            ].decode(encoding)
        else:

            def decode_string(buffer):
                s = struct.unpack(f, bytes(islice(buffer, size)))[0].decode(encoding)
                i = s.find("\x00")
                if i == -1:
                    return s
                return s[:i]

            return decode_string

    @classmethod
    def make_null_decode(cls, sub_schema):
        if sub_schema.get("binaryFormat") is not None:
            f = sub_schema["binaryFormat"]
            size = struct.calcsize(f)

            def padding_decode(buffer):
                struct.unpack(f, bytes(islice(buffer, size)))

            return padding_decode
        else:
            return lambda _: None

    @classmethod
    def make_numeric_decode(cls, sub_schema):
        f = "<" + sub_schema["binaryFormat"]
        size = struct.calcsize(f)
        return lambda buffer: struct.unpack(f, bytes(islice(buffer, size)))[0]

    @classmethod
    def make_encode(cls, sub_schema):
        """
        Create a function that can encode objects of this schema
        """
        if set(sub_schema["type"]) == {"object", "null"}:
            return StructCodec.make_object_or_null_encode(sub_schema)
        else:
            return {
                "array": StructCodec.make_array_encode,
                "object": StructCodec.make_object_encode,
                "string": StructCodec.make_string_encode,
                "null": StructCodec.make_null_encode,
                "number": StructCodec.make_numeric_encode,
                "integer": StructCodec.make_numeric_encode,
                "boolean": StructCodec.make_numeric_encode,
            }[sub_schema["type"]](sub_schema)

    @classmethod
    def make_array_encode(cls, sub_schema):
        array_length_f = "<" + sub_schema.get("arrayLengthFormat", "L")
        element_encoder = StructCodec.make_encode(sub_schema["items"])
        exhaust_buffer = sub_schema.get("noLengthEncodingExhaustBuffer", False)
        if exhaust_buffer:
            return lambda array: b"".join(element_encoder(ele) for ele in array)
        else:

            def array_encode_with_length(array):
                try:
                    packed_length = struct.pack(array_length_f, len(array))
                except struct.error:
                    raise ValueError(
                        "Couldn't pack array size - it is likely too long"
                        " for the specified arrayLengthFormat"
                    )
                return packed_length + b"".join(element_encoder(ele) for ele in array)

            return array_encode_with_length

    @classmethod
    def make_object_encode(cls, sub_schema):
        sub_encoders = {
            key: StructCodec.make_encode(prop)
            for key, prop in sub_schema["properties"].items()
        }
        defaults = {
            key: prop["default"]
            for key, prop in sub_schema["properties"].items()
            if "default" in prop
        }

        def object_encode(obj):
            values = []
            for key, sub_encoder in sub_encoders.items():
                try:
                    values.append(sub_encoder(obj[key]))
                except KeyError:
                    values.append(sub_encoder(defaults[key]))
            return b"".join(values)

        return object_encode

    @classmethod
    def make_object_or_null_encode(cls, sub_schema):
        sub_encoders = {
            key: StructCodec.make_encode(prop)
            for key, prop in sub_schema["properties"].items()
        }
        return (
            lambda obj: b""
            if obj is None
            else b"".join(
                sub_encoder(obj[key]) for key, sub_encoder in sub_encoders.items()
            )
        )

    @classmethod
    def make_string_encode(cls, sub_schema):
        encoding = sub_schema.get("stringEncoding", "utf-8")
        return lambda string: struct.pack(
            "<" + sub_schema["binaryFormat"], string.encode(encoding)
        )

    @classmethod
    def make_null_encode(cls, sub_schema):
        return lambda _: struct.pack(sub_schema.get("binaryFormat", "0x"))

    @classmethod
    def make_numeric_encode(cls, sub_schema):
        return struct.Struct("<" + sub_schema["binaryFormat"]).pack

    @classmethod
    def modify_schema(cls, schema: Mapping) -> Mapping:
        # This codec requires that additional properties are
        # not allowed. Rather than get schema authors to repeat that everywhere
        # we add it here, sadly we can't do this in the metaschema as "default" isn't
        # used by the validator.
        def enforce_fixed_properties(obj):
            if type(obj) == list:
                return [enforce_fixed_properties(j) for j in obj]
            elif type(obj) == dict:
                ret = {k: enforce_fixed_properties(v) for k, v in obj.items()}
                if "object" in ret.get("type", []):
                    if ret.get("additional_properties"):
                        raise ValueError(
                            "Struct codec does not support additional_properties"
                        )
                    # To prevent authors having to list required properties the default
                    # is that all without a default are required.
                    if "required" not in ret:
                        ret["required"] = [
                            prop
                            for prop, sub_schema in ret.get("properties", {}).items()
                            if "default" not in sub_schema
                        ]
                    ret["additionalProperties"] = False
                return ret
            else:
                return obj

        schema = enforce_fixed_properties(schema)

        # We also give the schema an explicit ordering
        return StructCodec.order_by_index(schema)

    def __init__(self, schema: Mapping[str, Any]) -> None:
        try:
            StructCodecSchemaValidator.check_schema(schema)
        except jsonschema.exceptions.SchemaError as ve:
            raise exceptions.MetadataSchemaValidationError(str(ve)) from ve

        self.encode = StructCodec.make_encode(schema)
        decoder = StructCodec.make_decode(schema)
        self.decode = lambda buffer: decoder(iter(buffer))

    def encode(self, obj: Any) -> bytes:
        # Set by __init__
        pass  # pragma: nocover

    def decode(self, encoded: bytes) -> Any:
        # Set by __init__
        pass  # pragma: nocover


register_metadata_codec(StructCodec, "struct")


def validate_bytes(data: Optional[bytes]) -> None:
    if data is not None and not isinstance(data, bytes):
        raise TypeError(
            f"If no encoding is set metadata should be bytes, found {type(data)}"
        )


class MetadataSchema:
    """
    Class for validating, encoding and decoding metadata.

    :param dict schema: A dict containing a valid JSONSchema object.
    """

    def __init__(self, schema: Optional[Mapping[str, Any]]) -> None:
        self._schema = schema

        if schema is None:
            self._string = ""
            self._validate_row = validate_bytes
            self.encode_row = NOOPCodec({}).encode
            self.decode_row = NOOPCodec({}).decode
            self.empty_value = b""
        else:
            try:
                TSKITMetadataSchemaValidator.check_schema(schema)
            except jsonschema.exceptions.SchemaError as ve:
                raise exceptions.MetadataSchemaValidationError(str(ve)) from ve
            try:
                codec_cls = codec_registry[schema["codec"]]
            except KeyError:
                raise exceptions.MetadataSchemaValidationError(
                    f"Unrecognised metadata codec '{schema['codec']}'. "
                    f"Valid options are {str(list(codec_registry.keys()))}."
                )
            # Codecs can modify the schema, for example to set defaults as the validator
            # does not.
            schema = codec_cls.modify_schema(schema)
            codec_instance = codec_cls(schema)
            self._string = tskit.canonical_json(schema)
            self._validate_row = TSKITMetadataSchemaValidator(schema).validate
            self.encode_row = codec_instance.encode
            self.decode_row = codec_instance.decode
            self.empty_value = {}

    def __repr__(self) -> str:
        return self._string

    def __str__(self) -> str:
        return pprint.pformat(self._schema)

    def __eq__(self, other) -> bool:
        return self._string == other._string

    @property
    def schema(self) -> Optional[Mapping[str, Any]]:
        # Return a copy to avoid unintentional mutation
        return copy.deepcopy(self._schema)

    def asdict(self) -> Optional[Mapping[str, Any]]:
        """
        Returns a dict representation of this schema. One possible use of this is to
        modify this dict and then pass it to the ``MetadataSchema`` constructor to create
        a similar schema.
        """
        return self.schema

    def validate_and_encode_row(self, row: Any) -> bytes:
        """
        Validate a row (dict) of metadata against this schema and return the encoded
        representation (bytes) using the codec specified in the schema.
        """
        try:
            self._validate_row(row)
        except jsonschema.exceptions.ValidationError as ve:
            raise exceptions.MetadataValidationError(str(ve)) from ve
        return self.encode_row(row)

    def decode_row(self, row: bytes) -> Any:
        """
        Decode an encoded row (bytes) of metadata, using the codec specifed in the schema
        and return a python dict. Note that no validation of the metadata against the
        schema is performed.
        """
        # Set by __init__
        pass  # pragma: no cover

    def encode_row(self, row: Any) -> bytes:
        """
        Encode a row (dict) of metadata to its binary representation (bytes)
        using the codec specified in the schema. Note that unlike
        :meth:`validate_and_encode_row` no validation against the schema is performed.
        This should only be used for performance if a validation check is not needed.
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
            decoded = json.loads(
                encoded_schema, object_pairs_hook=collections.OrderedDict
            )
        except json.decoder.JSONDecodeError:
            raise ValueError(f"Metadata schema is not JSON, found {encoded_schema}")
        return MetadataSchema(decoded)
