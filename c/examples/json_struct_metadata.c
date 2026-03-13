#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include <string.h>
#include <tskit.h>

// these are properties of the ``json+struct`` codec, documented in tskit
#define JSON_STRUCT_HEADER_SIZE 21

const uint8_t json_struct_codec_magic[4] = { 'J', 'B', 'L', 'B' };
const uint8_t json_struct_codec_version = 1;

// little-endian read of a uint64_t from an address
static uint64_t
load_u64_le(const uint8_t *p)
{
    uint64_t value = (uint64_t) p[0];
    value |= (uint64_t) p[1] << 8;
    value |= (uint64_t) p[2] << 16;
    value |= (uint64_t) p[3] << 24;
    value |= (uint64_t) p[4] << 32;
    value |= (uint64_t) p[5] << 40;
    value |= (uint64_t) p[6] << 48;
    value |= (uint64_t) p[7] << 56;
    return value;
}

// little-endian write of a uint64_t to an address
static void
set_u64_le(uint8_t *dest, uint64_t value)
{
    dest[0] = (uint8_t) (value & 0xFF);
    dest[1] = (uint8_t) ((value >> 8) & 0xFF);
    dest[2] = (uint8_t) ((value >> 16) & 0xFF);
    dest[3] = (uint8_t) ((value >> 24) & 0xFF);
    dest[4] = (uint8_t) ((value >> 32) & 0xFF);
    dest[5] = (uint8_t) ((value >> 40) & 0xFF);
    dest[6] = (uint8_t) ((value >> 48) & 0xFF);
    dest[7] = (uint8_t) ((value >> 56) & 0xFF);
}

// Extract the json and binary payloads from the `json+struct` codec data buffer.
// Note that the output pointers `json` and `binary` reference memory
// inside the `metadata` buffer passed in.
void
json_struct_codec_get_components(uint8_t *metadata, tsk_size_t metadata_length,
    uint8_t **json, tsk_size_t *json_length, uint8_t **binary, tsk_size_t *binary_length)
{
    // check the structure of the codec header and the sizes it specifies
    if (metadata == NULL || json == NULL || json_length == NULL || binary == NULL
        || binary_length == NULL)
        errx(EXIT_FAILURE, "bad parameter value.");
    if (metadata_length < JSON_STRUCT_HEADER_SIZE)
        errx(EXIT_FAILURE, "metadata truncated.");
    if (memcmp(metadata, json_struct_codec_magic, sizeof(json_struct_codec_magic)) != 0)
        errx(EXIT_FAILURE, "bad magic bytes.");

    uint8_t version = metadata[4];
    if (version != json_struct_codec_version)
        errx(EXIT_FAILURE, "bad version number.");

    uint64_t json_length_u64 = load_u64_le(metadata + 5);
    uint64_t binary_length_u64 = load_u64_le(metadata + 13);
    if (json_length_u64 > UINT64_MAX - (uint64_t) JSON_STRUCT_HEADER_SIZE)
        errx(EXIT_FAILURE, "invalid length.");

    // determine the number of padding bytes and do more safety checks
    uint64_t length = (uint64_t) JSON_STRUCT_HEADER_SIZE + json_length_u64;
    uint64_t padding_length = (8 - (length & 0x07)) % 8;
    if (padding_length > UINT64_MAX - length)
        errx(EXIT_FAILURE, "invalid length.");

    length += padding_length;
    if (binary_length_u64 > UINT64_MAX - length)
        errx(EXIT_FAILURE, "invalid length.");

    length += binary_length_u64;
    if ((uint64_t) metadata_length != length)
        errx(EXIT_FAILURE, "unexpected size.");

    uint8_t *padding_start = metadata + JSON_STRUCT_HEADER_SIZE + json_length_u64;
    for (uint64_t j = 0; j < padding_length; ++j)
        if (*(padding_start + j) != 0)
            errx(EXIT_FAILURE, "padding bytes are nonzero.");

    // the structure of the codec data seems valid; return components
    *json = metadata + JSON_STRUCT_HEADER_SIZE;
    *json_length = (tsk_size_t) json_length_u64;

    *binary = metadata + JSON_STRUCT_HEADER_SIZE + json_length_u64 + padding_length;
    *binary_length = (tsk_size_t) binary_length_u64;
}

// malloc and return a data buffer for the `json+struct` codec
// that contains the given components
void
json_struct_codec_create_buffer(const uint8_t *json, tsk_size_t json_length,
    const uint8_t *binary, tsk_size_t binary_length, uint8_t **buffer,
    tsk_size_t *buffer_length)
{
    // figure out the total length of the codec's data and allocate the buffer for it
    tsk_size_t header_length = JSON_STRUCT_HEADER_SIZE;
    tsk_size_t padding_length = (8 - ((header_length + json_length) & 0x07)) % 8;
    tsk_size_t total_length
        = header_length + json_length + padding_length + binary_length;
    uint8_t *bytes = malloc(total_length);
    if (!bytes)
        errx(EXIT_FAILURE, "memory for buffer could not be allocated.");

    // then set up the bytes for the codec header
    memcpy(bytes, json_struct_codec_magic, 4);
    bytes[4] = json_struct_codec_version;
    set_u64_le(bytes + 5, (uint64_t) json_length);
    set_u64_le(bytes + 13, (uint64_t) binary_length);

    // copy in the JSON and binary data, separated by the padding bytes; the goal of the
    // padding bytes is to ensure that the binary data is 8-byte-aligned relative to the
    // start of the buffer
    memcpy(bytes + header_length, json, json_length);
    memset(bytes + header_length + json_length, 0, padding_length);
    memcpy(bytes + header_length + json_length + padding_length, binary, binary_length);

    // return the buffer and its length; the caller takes ownership of the buffer
    *buffer = bytes;
    *buffer_length = total_length;
}

int
main(int argc, char **argv)
{
    // we start with JSON and binary payloads that we encode into a new buffer
    // note that the JSON payload does not have to end with a trailing NULL
    const char json_payload[] = { '{', '"', 'a', '"', ':', '1', '}' };
    const uint8_t binary_payload[] = { 0x01, 0x02, 0x03, 0x04 };
    uint8_t *metadata;
    tsk_size_t metadata_length;

    json_struct_codec_create_buffer((const uint8_t *) json_payload, sizeof(json_payload),
        binary_payload, sizeof(binary_payload), &metadata, &metadata_length);

    // then we decode that buffer to recover the json and binary data
    uint8_t *decoded_json, *decoded_binary;
    tsk_size_t decoded_json_length, decoded_binary_length;

    json_struct_codec_get_components(metadata, metadata_length, &decoded_json,
        &decoded_json_length, &decoded_binary, &decoded_binary_length);

    // print the recovered data to demonstrate that the round-trip worked
    // note that the JSON data is not NULL-terminated unless you put a NULL there!
    printf("JSON: %.*s\n", (int) decoded_json_length, decoded_json);

    printf("Binary data:");
    for (tsk_size_t j = 0; j < decoded_binary_length; j++)
        printf(" %#04x", decoded_binary[j]);
    printf("\n");

    free(metadata);
    return EXIT_SUCCESS;
}
