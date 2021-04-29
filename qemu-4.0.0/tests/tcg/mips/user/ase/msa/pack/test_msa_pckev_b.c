/*
 *  Test program for MSA instruction PCKEV.B
 *
 *  Copyright (C) 2019  Wave Computing, Inc.
 *  Copyright (C) 2019  Aleksandar Markovic <amarkovic@wavecomp.com>
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *`
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 */

#include <sys/time.h>
#include <stdint.h>

#include "../../../../include/wrappers_msa.h"
#include "../../../../include/test_inputs_128.h"
#include "../../../../include/test_utils_128.h"

#define TEST_COUNT_TOTAL (                                                \
            (PATTERN_INPUTS_SHORT_COUNT) * (PATTERN_INPUTS_SHORT_COUNT) + \
            (RANDOM_INPUTS_SHORT_COUNT) * (RANDOM_INPUTS_SHORT_COUNT))


int32_t main(void)
{
    char *instruction_name = "PCKEV.B";
    int32_t ret;
    uint32_t i, j;
    struct timeval start, end;
    double elapsed_time;

    uint64_t b128_result[TEST_COUNT_TOTAL][2];
    uint64_t b128_expect[TEST_COUNT_TOTAL][2] = {
        { 0xffffffffffffffffULL, 0xffffffffffffffffULL, },    /*   0  */
        { 0x0000000000000000ULL, 0xffffffffffffffffULL, },
        { 0xaaaaaaaaaaaaaaaaULL, 0xffffffffffffffffULL, },
        { 0x5555555555555555ULL, 0xffffffffffffffffULL, },
        { 0xccccccccccccccccULL, 0xffffffffffffffffULL, },
        { 0x3333333333333333ULL, 0xffffffffffffffffULL, },
        { 0xe3388ee38ee3388eULL, 0xffffffffffffffffULL, },
        { 0x1cc7711c711cc771ULL, 0xffffffffffffffffULL, },
        { 0xffffffffffffffffULL, 0x0000000000000000ULL, },    /*   8  */
        { 0x0000000000000000ULL, 0x0000000000000000ULL, },
        { 0xaaaaaaaaaaaaaaaaULL, 0x0000000000000000ULL, },
        { 0x5555555555555555ULL, 0x0000000000000000ULL, },
        { 0xccccccccccccccccULL, 0x0000000000000000ULL, },
        { 0x3333333333333333ULL, 0x0000000000000000ULL, },
        { 0xe3388ee38ee3388eULL, 0x0000000000000000ULL, },
        { 0x1cc7711c711cc771ULL, 0x0000000000000000ULL, },
        { 0xffffffffffffffffULL, 0xaaaaaaaaaaaaaaaaULL, },    /*  16  */
        { 0x0000000000000000ULL, 0xaaaaaaaaaaaaaaaaULL, },
        { 0xaaaaaaaaaaaaaaaaULL, 0xaaaaaaaaaaaaaaaaULL, },
        { 0x5555555555555555ULL, 0xaaaaaaaaaaaaaaaaULL, },
        { 0xccccccccccccccccULL, 0xaaaaaaaaaaaaaaaaULL, },
        { 0x3333333333333333ULL, 0xaaaaaaaaaaaaaaaaULL, },
        { 0xe3388ee38ee3388eULL, 0xaaaaaaaaaaaaaaaaULL, },
        { 0x1cc7711c711cc771ULL, 0xaaaaaaaaaaaaaaaaULL, },
        { 0xffffffffffffffffULL, 0x5555555555555555ULL, },    /*  24  */
        { 0x0000000000000000ULL, 0x5555555555555555ULL, },
        { 0xaaaaaaaaaaaaaaaaULL, 0x5555555555555555ULL, },
        { 0x5555555555555555ULL, 0x5555555555555555ULL, },
        { 0xccccccccccccccccULL, 0x5555555555555555ULL, },
        { 0x3333333333333333ULL, 0x5555555555555555ULL, },
        { 0xe3388ee38ee3388eULL, 0x5555555555555555ULL, },
        { 0x1cc7711c711cc771ULL, 0x5555555555555555ULL, },
        { 0xffffffffffffffffULL, 0xccccccccccccccccULL, },    /*  32  */
        { 0x0000000000000000ULL, 0xccccccccccccccccULL, },
        { 0xaaaaaaaaaaaaaaaaULL, 0xccccccccccccccccULL, },
        { 0x5555555555555555ULL, 0xccccccccccccccccULL, },
        { 0xccccccccccccccccULL, 0xccccccccccccccccULL, },
        { 0x3333333333333333ULL, 0xccccccccccccccccULL, },
        { 0xe3388ee38ee3388eULL, 0xccccccccccccccccULL, },
        { 0x1cc7711c711cc771ULL, 0xccccccccccccccccULL, },
        { 0xffffffffffffffffULL, 0x3333333333333333ULL, },    /*  40  */
        { 0x0000000000000000ULL, 0x3333333333333333ULL, },
        { 0xaaaaaaaaaaaaaaaaULL, 0x3333333333333333ULL, },
        { 0x5555555555555555ULL, 0x3333333333333333ULL, },
        { 0xccccccccccccccccULL, 0x3333333333333333ULL, },
        { 0x3333333333333333ULL, 0x3333333333333333ULL, },
        { 0xe3388ee38ee3388eULL, 0x3333333333333333ULL, },
        { 0x1cc7711c711cc771ULL, 0x3333333333333333ULL, },
        { 0xffffffffffffffffULL, 0xe3388ee38ee3388eULL, },    /*  48  */
        { 0x0000000000000000ULL, 0xe3388ee38ee3388eULL, },
        { 0xaaaaaaaaaaaaaaaaULL, 0xe3388ee38ee3388eULL, },
        { 0x5555555555555555ULL, 0xe3388ee38ee3388eULL, },
        { 0xccccccccccccccccULL, 0xe3388ee38ee3388eULL, },
        { 0x3333333333333333ULL, 0xe3388ee38ee3388eULL, },
        { 0xe3388ee38ee3388eULL, 0xe3388ee38ee3388eULL, },
        { 0x1cc7711c711cc771ULL, 0xe3388ee38ee3388eULL, },
        { 0xffffffffffffffffULL, 0x1cc7711c711cc771ULL, },    /*  56  */
        { 0x0000000000000000ULL, 0x1cc7711c711cc771ULL, },
        { 0xaaaaaaaaaaaaaaaaULL, 0x1cc7711c711cc771ULL, },
        { 0x5555555555555555ULL, 0x1cc7711c711cc771ULL, },
        { 0xccccccccccccccccULL, 0x1cc7711c711cc771ULL, },
        { 0x3333333333333333ULL, 0x1cc7711c711cc771ULL, },
        { 0xe3388ee38ee3388eULL, 0x1cc7711c711cc771ULL, },
        { 0x1cc7711c711cc771ULL, 0x1cc7711c711cc771ULL, },
        { 0x675e7b0c6acc6240ULL, 0x675e7b0c6acc6240ULL, },    /*  64  */
        { 0xf71a3ffcbe639308ULL, 0x675e7b0c6acc6240ULL, },
        { 0xd8ff2b145aaacf80ULL, 0x675e7b0c6acc6240ULL, },
        { 0xf1d842a04f4d314eULL, 0x675e7b0c6acc6240ULL, },
        { 0x675e7b0c6acc6240ULL, 0xf71a3ffcbe639308ULL, },
        { 0xf71a3ffcbe639308ULL, 0xf71a3ffcbe639308ULL, },
        { 0xd8ff2b145aaacf80ULL, 0xf71a3ffcbe639308ULL, },
        { 0xf1d842a04f4d314eULL, 0xf71a3ffcbe639308ULL, },
        { 0x675e7b0c6acc6240ULL, 0xd8ff2b145aaacf80ULL, },    /*  72  */
        { 0xf71a3ffcbe639308ULL, 0xd8ff2b145aaacf80ULL, },
        { 0xd8ff2b145aaacf80ULL, 0xd8ff2b145aaacf80ULL, },
        { 0xf1d842a04f4d314eULL, 0xd8ff2b145aaacf80ULL, },
        { 0x675e7b0c6acc6240ULL, 0xf1d842a04f4d314eULL, },
        { 0xf71a3ffcbe639308ULL, 0xf1d842a04f4d314eULL, },
        { 0xd8ff2b145aaacf80ULL, 0xf1d842a04f4d314eULL, },
        { 0xf1d842a04f4d314eULL, 0xf1d842a04f4d314eULL, },
    };

    gettimeofday(&start, NULL);

    for (i = 0; i < PATTERN_INPUTS_SHORT_COUNT; i++) {
        for (j = 0; j < PATTERN_INPUTS_SHORT_COUNT; j++) {
            do_msa_PCKEV_B(b128_pattern[i], b128_pattern[j],
                           b128_result[PATTERN_INPUTS_SHORT_COUNT * i + j]);
        }
    }

    for (i = 0; i < RANDOM_INPUTS_SHORT_COUNT; i++) {
        for (j = 0; j < RANDOM_INPUTS_SHORT_COUNT; j++) {
            do_msa_PCKEV_B(b128_random[i], b128_random[j],
                           b128_result[((PATTERN_INPUTS_SHORT_COUNT) *
                                        (PATTERN_INPUTS_SHORT_COUNT)) +
                                       RANDOM_INPUTS_SHORT_COUNT * i + j]);
        }
    }

    gettimeofday(&end, NULL);

    elapsed_time = (end.tv_sec - start.tv_sec) * 1000.0;
    elapsed_time += (end.tv_usec - start.tv_usec) / 1000.0;

    ret = check_results(instruction_name, TEST_COUNT_TOTAL, elapsed_time,
                        &b128_result[0][0], &b128_expect[0][0]);

    return ret;
}
