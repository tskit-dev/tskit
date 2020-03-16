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
#include <math.h>

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
    CU_ASSERT_STRING_EQUAL(
        tsk_strerror(0), "Normal exit condition. This is not an error!");
}

static void
test_strerror_kastore(void)
{
    int kastore_errors[] = { KAS_ERR_NO_MEMORY, KAS_ERR_IO, KAS_ERR_KEY_NOT_FOUND };
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

static void
test_double_round(void)
{
    struct test_case {
        double source;
        unsigned int num_digits;
        double result;
    };
    struct test_case test_cases[] = {
        { 1.555, 3, 1.555 },
        { 1.5555, 2, 1.56 },
        /* catch the halfway between integers case */
        { 1.5555, 3, 1.556 },

        { 1.5111, 3, 1.511 },
        { 1.5112, 3, 1.511 },
        { 3.141592653589793, 0, 3.0 },
        { 3.141592653589793, 1, 3.1 },
        { 3.141592653589793, 2, 3.14 },
        { 3.141592653589793, 3, 3.142 },
        { 3.141592653589793, 4, 3.1416 },
        { 3.141592653589793, 5, 3.14159 },
        { 3.141592653589793, 6, 3.141593 },
        { 3.141592653589793, 7, 3.1415927 },
        { 3.141592653589793, 8, 3.14159265 },
        { 3.141592653589793, 9, 3.141592654 },
        { 3.141592653589793, 10, 3.1415926536 },
        { 3.141592653589793, 11, 3.14159265359 },
        { 3.141592653589793, 12, 3.14159265359 },
        { 3.141592653589793, 13, 3.1415926535898 },
        { 3.141592653589793, 14, 3.14159265358979 },
        { 3.141592653589793, 15, 3.141592653589793 },
        { 3.141592653589793, 16, 3.141592653589793 },
        { 3.141592653589793, 17, 3.141592653589793 },
        { 3.141592653589793, 18, 3.141592653589793 },
        { 3.141592653589793, 19, 3.141592653589793 },
        /* We have tiny differences in precision at k=20; not worth worrying about. */
        { 3.141592653589793, 21, 3.141592653589793 },
        { 3.141592653589793, 22, 3.141592653589793 },
        { 3.141592653589793, 23, 3.141592653589793 },

        { 0.3333333333333333, 0, 0.0 },
        { 0.3333333333333333, 1, 0.3 },
        { 0.3333333333333333, 2, 0.33 },
        { 0.3333333333333333, 3, 0.333 },
        { 0.3333333333333333, 4, 0.3333 },
        { 0.3333333333333333, 5, 0.33333 },
        { 0.3333333333333333, 6, 0.333333 },
        { 0.3333333333333333, 7, 0.3333333 },
        { 0.3333333333333333, 8, 0.33333333 },
        { 0.3333333333333333, 9, 0.333333333 },
        { 0.3333333333333333, 10, 0.3333333333 },
        { 0.3333333333333333, 11, 0.33333333333 },
        { 0.3333333333333333, 12, 0.333333333333 },
        { 0.3333333333333333, 13, 0.3333333333333 },
        { 0.3333333333333333, 14, 0.33333333333333 },
        { 0.3333333333333333, 15, 0.333333333333333 },
        { 0.3333333333333333, 16, 0.3333333333333333 },
        { 0.3333333333333333, 17, 0.3333333333333333 },
        { 0.3333333333333333, 18, 0.3333333333333333 },
        { 0.3333333333333333, 19, 0.3333333333333333 },
        { 0.3333333333333333, 20, 0.3333333333333333 },
        { 0.3333333333333333, 21, 0.3333333333333333 },
        { 0.3333333333333333, 22, 0.3333333333333333 },
        { 0.3333333333333333, 23, 0.3333333333333333 },

        { 0.6666666666666666, 0, 1.0 },
        { 0.6666666666666666, 1, 0.7 },
        { 0.6666666666666666, 2, 0.67 },
        { 0.6666666666666666, 3, 0.667 },
        { 0.6666666666666666, 4, 0.6667 },
        { 0.6666666666666666, 5, 0.66667 },
        { 0.6666666666666666, 6, 0.666667 },
        { 0.6666666666666666, 7, 0.6666667 },
        { 0.6666666666666666, 8, 0.66666667 },
        { 0.6666666666666666, 9, 0.666666667 },
        { 0.6666666666666666, 10, 0.6666666667 },
        { 0.6666666666666666, 11, 0.66666666667 },
        { 0.6666666666666666, 12, 0.666666666667 },
        { 0.6666666666666666, 13, 0.6666666666667 },
        { 0.6666666666666666, 14, 0.66666666666667 },
        { 0.6666666666666666, 15, 0.666666666666667 },
        { 0.6666666666666666, 16, 0.6666666666666666 },
        { 0.6666666666666666, 17, 0.6666666666666666 },
        { 0.6666666666666666, 18, 0.6666666666666666 },
        { 0.6666666666666666, 19, 0.6666666666666666 },
        { 0.6666666666666666, 20, 0.6666666666666666 },
        { 0.6666666666666666, 21, 0.6666666666666666 },
        { 0.6666666666666666, 22, 0.6666666666666666 },
        { 0.6666666666666666, 23, 0.6666666666666666 },

        { 0.07692307692307693, 0, 0.0 },
        { 0.07692307692307693, 1, 0.1 },
        { 0.07692307692307693, 2, 0.08 },
        { 0.07692307692307693, 3, 0.077 },
        { 0.07692307692307693, 4, 0.0769 },
        { 0.07692307692307693, 5, 0.07692 },
        { 0.07692307692307693, 6, 0.076923 },
        { 0.07692307692307693, 7, 0.0769231 },
        { 0.07692307692307693, 8, 0.07692308 },
        { 0.07692307692307693, 9, 0.076923077 },
        { 0.07692307692307693, 10, 0.0769230769 },
        { 0.07692307692307693, 11, 0.07692307692 },
        { 0.07692307692307693, 12, 0.076923076923 },
        { 0.07692307692307693, 13, 0.0769230769231 },
        { 0.07692307692307693, 14, 0.07692307692308 },
        { 0.07692307692307693, 15, 0.076923076923077 },
        { 0.07692307692307693, 16, 0.0769230769230769 },
        { 0.07692307692307693, 17, 0.07692307692307693 },
        { 0.07692307692307693, 18, 0.07692307692307693 },
        { 0.07692307692307693, 19, 0.07692307692307693 },
        { 0.07692307692307693, 20, 0.07692307692307693 },
        /* Tiny difference in precision at k=21 */
        { 0.07692307692307693, 22, 0.07692307692307693 },
        { 0.07692307692307693, 23, 0.07692307692307693 },

        { 1e-21, 0, 0.0 },
        { 1e-21, 1, 0.0 },
        { 1e-21, 2, 0.0 },
        { 1e-21, 3, 0.0 },
        { 1e-21, 4, 0.0 },
        { 1e-21, 5, 0.0 },
        { 1e-21, 6, 0.0 },
        { 1e-21, 7, 0.0 },
        { 1e-21, 8, 0.0 },
        { 1e-21, 9, 0.0 },
        { 1e-21, 10, 0.0 },
        { 1e-21, 11, 0.0 },
        { 1e-21, 12, 0.0 },
        { 1e-21, 13, 0.0 },
        { 1e-21, 14, 0.0 },
        { 1e-21, 15, 0.0 },
        { 1e-21, 16, 0.0 },
        { 1e-21, 17, 0.0 },
        { 1e-21, 18, 0.0 },
        { 1e-21, 19, 0.0 },
        { 1e-21, 20, 0.0 },
        { 1e-21, 21, 1e-21 },
        { 1e-21, 22, 1e-21 },
        { 1e-21, 23, 1e-21 },

        { 1e-10, 0, 0.0 },
        { 1e-10, 1, 0.0 },
        { 1e-10, 2, 0.0 },
        { 1e-10, 3, 0.0 },
        { 1e-10, 4, 0.0 },
        { 1e-10, 5, 0.0 },
        { 1e-10, 6, 0.0 },
        { 1e-10, 7, 0.0 },
        { 1e-10, 8, 0.0 },
        { 1e-10, 9, 0.0 },
        { 1e-10, 10, 1e-10 },
        { 1e-10, 11, 1e-10 },
        { 1e-10, 12, 1e-10 },
        { 1e-10, 13, 1e-10 },
        { 1e-10, 14, 1e-10 },
        { 1e-10, 15, 1e-10 },
        { 1e-10, 16, 1e-10 },
        { 1e-10, 17, 1e-10 },
        { 1e-10, 18, 1e-10 },
        { 1e-10, 19, 1e-10 },
        { 1e-10, 20, 1e-10 },
        { 1e-10, 21, 1e-10 },
        { 1e-10, 22, 1e-10 },
        { 1e-10, 23, 1e-10 },

        { 3.141592653589793e-08, 0, 0.0 },
        { 3.141592653589793e-08, 1, 0.0 },
        { 3.141592653589793e-08, 2, 0.0 },
        { 3.141592653589793e-08, 3, 0.0 },
        { 3.141592653589793e-08, 4, 0.0 },
        { 3.141592653589793e-08, 5, 0.0 },
        { 3.141592653589793e-08, 6, 0.0 },
        { 3.141592653589793e-08, 7, 0.0 },
        { 3.141592653589793e-08, 8, 3e-08 },
        { 3.141592653589793e-08, 9, 3.1e-08 },
        { 3.141592653589793e-08, 10, 3.14e-08 },
        { 3.141592653589793e-08, 11, 3.142e-08 },
        { 3.141592653589793e-08, 12, 3.1416e-08 },
        { 3.141592653589793e-08, 13, 3.14159e-08 },
        { 3.141592653589793e-08, 14, 3.141593e-08 },
        { 3.141592653589793e-08, 15, 3.1415927e-08 },
        { 3.141592653589793e-08, 16, 3.14159265e-08 },
        { 3.141592653589793e-08, 17, 3.141592654e-08 },
        { 3.141592653589793e-08, 18, 3.1415926536e-08 },
        { 3.141592653589793e-08, 19, 3.14159265359e-08 },
        { 3.141592653589793e-08, 20, 3.14159265359e-08 },
        { 3.141592653589793e-08, 21, 3.1415926535898e-08 },
        /* Tiny precision mismatch at k=22 */
        { 3.141592653589793e-08, 23, 3.141592653589793e-08 },

    };
    size_t num_test_cases = sizeof(test_cases) / sizeof(*test_cases);
    size_t j;

    for (j = 0; j < num_test_cases; j++) {
        CU_ASSERT_EQUAL_FATAL(tsk_round(test_cases[j].source, test_cases[j].num_digits),
            test_cases[j].result);
    }
}

static void
test_blkalloc(void)
{
    tsk_blkalloc_t alloc;
    int ret;
    size_t j, block_size;
    void *mem;

    ret = tsk_blkalloc_init(&alloc, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    tsk_blkalloc_free(&alloc);

    for (block_size = 1; block_size < 10; block_size++) {
        ret = tsk_blkalloc_init(&alloc, block_size);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        for (j = 0; j < 10; j++) {
            mem = tsk_blkalloc_get(&alloc, block_size);
            CU_ASSERT_TRUE(mem != NULL);
            CU_ASSERT_EQUAL(alloc.num_chunks, j + 1);
            memset(mem, 0, block_size);
        }

        mem = tsk_blkalloc_get(&alloc, block_size + 1);
        CU_ASSERT_EQUAL(mem, NULL);
        mem = tsk_blkalloc_get(&alloc, block_size + 2);
        CU_ASSERT_EQUAL(mem, NULL);

        tsk_blkalloc_print_state(&alloc, _devnull);
        tsk_blkalloc_free(&alloc);
    }

    /* Allocate awkward sized chunk */
    ret = tsk_blkalloc_init(&alloc, 100);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    mem = tsk_blkalloc_get(&alloc, 90);
    CU_ASSERT_FATAL(mem != NULL);
    memset(mem, 0, 90);
    mem = tsk_blkalloc_get(&alloc, 10);
    CU_ASSERT_FATAL(mem != NULL);
    memset(mem, 0, 10);
    CU_ASSERT_EQUAL(alloc.num_chunks, 1);
    mem = tsk_blkalloc_get(&alloc, 90);
    CU_ASSERT_FATAL(mem != NULL);
    memset(mem, 0, 90);
    CU_ASSERT_EQUAL(alloc.num_chunks, 2);
    mem = tsk_blkalloc_get(&alloc, 11);
    CU_ASSERT_FATAL(mem != NULL);
    memset(mem, 0, 11);
    CU_ASSERT_EQUAL(alloc.num_chunks, 3);

    tsk_blkalloc_free(&alloc);
}

int
main(int argc, char **argv)
{
    CU_TestInfo tests[] = {
        { "test_strerror", test_strerror },
        { "test_strerror_kastore", test_strerror_kastore },
        { "test_generate_uuid", test_generate_uuid },
        { "test_double_round", test_double_round },
        { "test_blkalloc", test_blkalloc },
        { NULL, NULL },
    };

    return test_main(tests, argc, argv);
}
