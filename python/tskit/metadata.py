# MIT License
#
# Copyright (c) 2020-2022 Tskit Developers
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
from __future__ import annotations

import abc
import builtins
import collections
import copy
import functools
import json
import pprint
import struct
import types
from itertools import islice
from typing import Any
from typing import Mapping

import jsonschema

import tskit
import tskit.exceptions as exceptions

__builtins__object__setattr__ = builtins.object.__setattr__


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
deref_meta_schema: Mapping[str, Any] = copy.deepcopy(
    TSKITMetadataSchemaValidator.META_SCHEMA
)
# We need a top-level only required property so we need to rewrite any reference
# to the top-level schema to a copy in a definition.
deref_meta_schema = replace_root_refs(deref_meta_schema)
deref_meta_schema["definitions"]["root"] = copy.deepcopy(deref_meta_schema)
deref_meta_schema["codec"] = {"type": "string"}
deref_meta_schema["required"] = ["codec"]
# For interoperability reasons, force the top-level to be an object or union
# of object and null
deref_meta_schema["properties"]["type"] = {"enum": ["object", ["object", "null"]]}
# Change the schema URL to avoid jsonschema's cache
deref_meta_schema["$schema"] = "http://json-schema.org/draft-o=07/schema#tskit"
TSKITMetadataSchemaValidator.META_SCHEMA = deref_meta_schema


class AbstractMetadataCodec(metaclass=abc.ABCMeta):
    """
    Superclass of all MetadataCodecs.
    """

    def __init__(self, schema: Mapping[str, Any]) -> None:
        raise NotImplementedError  # pragma: no cover

    @classmethod
    def modify_schema(self, schema: Mapping) -> Mapping:
        return schema

    @classmethod
    def is_schema_trivial(self, schema: Mapping) -> bool:
        return False

    @abc.abstractmethod
    def encode(self, obj: Any) -> bytes:
        raise NotImplementedError  # pragma: no cover

    @abc.abstractmethod
    def decode(self, encoded: bytes) -> Any:
        raise NotImplementedError  # pragma: no cover


codec_registry = {}


def register_metadata_codec(
    codec_cls: type[AbstractMetadataCodec], codec_id: str
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
    def default_validator(validator, types, instance, schema):
        # For json codec defaults must be at the top level
        if validator.is_type(instance, "object"):
            for v in instance.get("properties", {}).values():
                for v2 in v.get("properties", {}).values():
                    if "default" in v2:
                        yield jsonschema.ValidationError(
                            "Defaults can only be specified at the top level"
                            " for JSON codec"
                        )

    schema_validator = jsonschema.validators.extend(
        TSKITMetadataSchemaValidator, {"default": default_validator}
    )

    @classmethod
    def is_schema_trivial(self, schema: Mapping) -> bool:
        return len(schema.get("properties", {})) == 0

    def __init__(self, schema: Mapping[str, Any]) -> None:
        try:
            self.schema_validator.check_schema(schema)
        except jsonschema.exceptions.SchemaError as ve:
            raise exceptions.MetadataSchemaValidationError(str(ve)) from ve

        # Find default values to fill in on decode, top level only
        self.defaults = {
            key: prop["default"]
            for key, prop in schema.get("properties", {}).items()
            if "default" in prop
        }

    def encode(self, obj: Any) -> bytes:
        try:
            return tskit.canonical_json(obj).encode()
        except TypeError as e:
            raise exceptions.MetadataEncodingError(
                f"Could not encode metadata of type {str(e).split()[3]}"
            )

    def decode(self, encoded: bytes) -> Any:
        if len(encoded) == 0:
            result = {}
        else:
            result = json.loads(encoded.decode())

        # Assign default values
        if isinstance(result, dict):
            return dict(self.defaults, **result)
        else:
            return result


register_metadata_codec(JSONCodec, "json")


class NOOPCodec(AbstractMetadataCodec):
    def __init__(self, schema: Mapping[str, Any]) -> None:
        pass

    def encode(self, data: bytes) -> bytes:
        return data

    def decode(self, data: bytes) -> bytes:
        return data


def binary_format_validator(validator, types, instance, schema):
    # We're hooking into jsonschemas validation code here, which works by creating
    # generators of exceptions, hence the yielding

    # Make sure the normal type validation gets done
    yield from jsonschema._validators.type(validator, types, instance, schema)

    # Non-composite types must have a binaryFormat
    if validator.is_type(instance, "object"):
        for v in instance.values():
            if (
                isinstance(v, dict)
                and v.get("type")
                not in (None, "object", "array", "null", ["object", "null"])
                and "binaryFormat" not in v
            ):
                yield jsonschema.ValidationError(
                    f"{v['type']} type must have binaryFormat set"
                )
    # null type must be padding
    if (
        validator.is_type(instance, "object")
        and "null" in instance
        and instance["null"].get("type") == "null"
        and "binaryFormat" in instance["null"]
        and instance["null"]["binaryFormat"][-1] != "x"
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
struct_meta_schema: Mapping[str, Any] = copy.deepcopy(
    StructCodecSchemaValidator.META_SCHEMA
)
# No union types
struct_meta_schema["definitions"]["root"]["properties"]["type"] = {
    "$ref": "#/definitions/simpleTypes"
}
# No hetrogeneous arrays
struct_meta_schema["properties"]["items"] = {"$ref": "#/definitions/root"}
struct_meta_schema["definitions"]["root"]["properties"]["items"] = struct_meta_schema[
    "properties"
]["items"]
# binaryFormat matches regex
struct_meta_schema["properties"]["binaryFormat"] = {
    "type": "string",
    "pattern": r"^([cbB\?hHiIlLqQfd]|\d*[spx])$",
}
struct_meta_schema["definitions"]["root"]["properties"][
    "binaryFormat"
] = struct_meta_schema["properties"]["binaryFormat"]
# arrayLengthFormat matches regex and has default
struct_meta_schema["properties"]["arrayLengthFormat"] = {
    "type": "string",
    "pattern": r"^[BHILQ]$",
    "default": "L",
}
struct_meta_schema["definitions"]["root"]["properties"][
    "arrayLengthFormat"
] = struct_meta_schema["properties"]["arrayLengthFormat"]
# index is numeric
struct_meta_schema["properties"]["index"] = {"type": "number"}
struct_meta_schema["definitions"]["root"]["properties"]["index"] = struct_meta_schema[
    "properties"
]["index"]
# stringEncoding is string and has default
struct_meta_schema["properties"]["stringEncoding"] = {
    "type": "string",
    "default": "utf-8",
}
struct_meta_schema["definitions"]["root"]["properties"][
    "stringEncoding"
] = struct_meta_schema["properties"]["stringEncoding"]
# nullTerminated is a boolean
struct_meta_schema["properties"]["nullTerminated"] = {"type": "boolean"}
struct_meta_schema["definitions"]["root"]["properties"][
    "nullTerminated"
] = struct_meta_schema["properties"]["nullTerminated"]
# noLengthEncodingExhaustBuffer is a boolean
struct_meta_schema["properties"]["noLengthEncodingExhaustBuffer"] = {"type": "boolean"}
struct_meta_schema["definitions"]["root"]["properties"][
    "noLengthEncodingExhaustBuffer"
] = struct_meta_schema["properties"]["noLengthEncodingExhaustBuffer"]
StructCodecSchemaValidator.META_SCHEMA = struct_meta_schema


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
        defaults = {
            key: prop["default"]
            for key, prop in sub_schema["properties"].items()
            if "default" in prop
        }

        def object_encode(obj):
            values = []
            if obj is not None:
                for key, sub_encoder in sub_encoders.items():
                    try:
                        values.append(sub_encoder(obj[key]))
                    except KeyError:
                        values.append(sub_encoder(defaults[key]))
            return b"".join(values)

        return object_encode

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


def validate_bytes(data: bytes | None) -> None:
    if data is not None and not isinstance(data, bytes):
        raise TypeError(
            f"If no encoding is set metadata should be bytes, found {type(data)}"
        )


class MetadataSchema:
    """
    Class for validating, encoding and decoding metadata.

    :param dict schema: A dict containing a valid JSONSchema object.
    """

    def __init__(self, schema: Mapping[str, Any] | None) -> None:
        self._schema = schema
        self._bypass_validation = False

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
            self._bypass_validation = codec_cls.is_schema_trivial(schema)
            self.encode_row = codec_instance.encode
            self.decode_row = codec_instance.decode

            # If None is allowed by the schema as the top-level type, it gets used even
            # in the presence of default and required values.
            if "type" in schema and "null" in schema["type"]:
                self.empty_value = None
            else:
                self.empty_value = {}

    def __repr__(self) -> str:
        return self._string

    def __str__(self) -> str:
        return pprint.pformat(self._schema)

    def __eq__(self, other) -> bool:
        return self._string == other._string

    @property
    def schema(self) -> Mapping[str, Any] | None:
        # Return a copy to avoid unintentional mutation
        return copy.deepcopy(self._schema)

    def asdict(self) -> Mapping[str, Any] | None:
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
        # If the schema is permissive then validation can't fail
        if not self._bypass_validation:
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

    @staticmethod
    def permissive_json():
        """
        The simplest, permissive JSON schema. Only specifies the JSON codec and has
        no constraints on the properties.
        """
        return MetadataSchema({"codec": "json"})


# Often many replicate tree sequences are processed with identical schemas, so cache them
@functools.lru_cache(maxsize=128)
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


class _CachedMetadata:
    """
    Descriptor for lazy decoding of metadata on attribute access.
    """

    def __get__(self, row, owner):
        if row._metadata_decoder is not None:
            # Some classes that use this are frozen so we need to directly setattr.
            __builtins__object__setattr__(
                row, "_metadata", row._metadata_decoder(row._metadata)
            )
            # Decoder being None indicates that metadata is decoded
            __builtins__object__setattr__(row, "_metadata_decoder", None)
        return row._metadata

    def __set__(self, row, value):
        __builtins__object__setattr__(row, "_metadata", value)


def lazy_decode(own_init=False):
    def _lazy_decode(cls):
        """
        Modifies a dataclass such that it lazily decodes metadata, if it is encoded.
        If the metadata passed to the constructor is encoded a `metadata_decoder`
        parameter must be also be passed.
        """
        if not own_init:
            wrapped_init = cls.__init__

            # Intercept the init to record the decoder
            def new_init(self, *args, metadata_decoder=None, **kwargs):
                __builtins__object__setattr__(
                    self, "_metadata_decoder", metadata_decoder
                )
                wrapped_init(self, *args, **kwargs)

            cls.__init__ = new_init

        # Add a descriptor to the class to decode and cache metadata
        cls.metadata = _CachedMetadata()

        # Add slots needed to the class
        slots = cls.__slots__
        slots.extend(["_metadata", "_metadata_decoder"])
        dict_ = dict()
        sloted_members = dict()
        for k, v in cls.__dict__.items():
            if k not in slots:
                dict_[k] = v
            elif not isinstance(v, types.MemberDescriptorType):
                sloted_members[k] = v
        new_cls = type(cls.__name__, cls.__bases__, dict_)
        for k, v in sloted_members.items():
            setattr(new_cls, k, v)
        return new_cls

    return _lazy_decode


class MetadataProvider:
    """
    Abstract superclass of container objects that provide metadata.
    """

    def __init__(self, ll_object):
        self._ll_object = ll_object

    @property
    def metadata_schema(self) -> MetadataSchema:
        """
        The :class:`tskit.MetadataSchema` for this object.
        """
        return parse_metadata_schema(self._ll_object.metadata_schema)

    @metadata_schema.setter
    def metadata_schema(self, schema: MetadataSchema) -> None:
        # Check the schema is a valid schema instance by roundtripping it.
        text_version = repr(schema)
        parse_metadata_schema(text_version)
        self._ll_object.metadata_schema = text_version

    @property
    def metadata(self) -> Any:
        """
        The decoded metadata for this object.
        """
        return self.metadata_schema.decode_row(self.metadata_bytes)

    @metadata.setter
    def metadata(self, metadata: bytes | dict | None) -> None:
        encoded = self.metadata_schema.validate_and_encode_row(metadata)
        self._ll_object.metadata = encoded

    @property
    def metadata_bytes(self) -> Any:
        """
        The raw bytes of metadata for this TableCollection
        """
        return self._ll_object.metadata

    @property
    def nbytes(self) -> int:
        return len(self._ll_object.metadata) + len(self._ll_object.metadata_schema)

    def assert_equals(self, other: MetadataProvider):
        if self.metadata_schema != other.metadata_schema:
            raise AssertionError(
                f"Metadata schemas differ: self={self.metadata_schema} "
                f"other={other.metadata_schema}"
            )
        if self.metadata != other.metadata:
            raise AssertionError(
                f"Metadata differs: self={self.metadata} " f"other={other.metadata}"
            )
