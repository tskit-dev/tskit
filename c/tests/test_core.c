/*
 * MIT License
 *
 * Copyright (c) 2019 Tskit Developers
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "testlib.h"
#include <tskit/core.h>

#include <unistd.h>

static void
test_strerror(void)
{
    int j;
    const char *msg;
    int max_error_code = 8192; /* totally arbitrary */

    for (j = 0; j < max_error_code; j++) {
        msg = tsk_strerror(-j);
        CU_ASSERT_FATAL(msg != NULL);
        CU_ASSERT(strlen(msg) > 0);
    }
    CU_ASSERT_STRING_EQUAL(tsk_strerror(0),
            "Normal exit condition. This is not an error!");
}

static void
test_strerror_kastore(void)
{
    int kastore_errors[] = {KAS_ERR_NO_MEMORY, KAS_ERR_IO, KAS_ERR_KEY_NOT_FOUND};
    size_t j;
    int err;

    for (j = 0; j < sizeof(kastore_errors) / sizeof(*kastore_errors); j++) {
        err = tsk_set_kas_error(kastore_errors[j]);
        CU_ASSERT_TRUE(tsk_is_kas_error(err));
        CU_ASSERT_STRING_EQUAL(tsk_strerror(err), kas_strerror(kastore_errors[j]));
    }
}

static void
test_generate_uuid(void)
{
    size_t uuid_size = 36;
    char uuid[uuid_size + 1];
    char other_uuid[uuid_size + 1];
    int ret;

    ret = tsk_generate_uuid(uuid, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(strlen(uuid), uuid_size);
    CU_ASSERT_EQUAL(uuid[8], '-');
    CU_ASSERT_EQUAL(uuid[13], '-');
    CU_ASSERT_EQUAL(uuid[18], '-');
    CU_ASSERT_EQUAL(uuid[23], '-');

    ret = tsk_generate_uuid(other_uuid, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(strlen(other_uuid), uuid_size);
    CU_ASSERT_STRING_NOT_EQUAL(uuid, other_uuid);
}

int
main(int argc, char **argv)
{
    CU_TestInfo tests[] = {
        {"test_strerror", test_strerror},
        {"test_strerror_kastore", test_strerror_kastore},
        {"test_generate_uuid", test_generate_uuid},
        {NULL, NULL},
    };

    return test_main(tests, argc, argv);
}
