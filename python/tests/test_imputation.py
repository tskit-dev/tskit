"""
Tests for interpolation-style genotype imputation.
"""
import io

import numpy as np
import pytest

import tests.beagle_numba
import tskit


# The following toy query haplotypes were imputed from the toy reference haplotypes
# using BEAGLE 4.1 with Ne set to 10.
#
# There are 9 sites, starting from 10,000 to 90,000, with 10,000 increments.
# The REF is A and ALT is C for all the sites in the VCF input to BEAGLE.
# The ancestral state is A and derived state is C for all the sites.
# In this setup, A is encoded as 0 and C as 1 whether the allele encoding is
# ancestral/derived (0/1) or ACGT (0123).
#
# Case 0:
# Reference haplotypes and query haplotypes have 0|0 at all sites.
# First and last reference markers are missing in the query haplotypes.
# fmt: off
toy_ref_0 = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],   # ref_0, haplotype 0
        [0, 0, 0, 0, 0, 0, 0, 0, 0],   # ref_0, haplotype 1
    ], dtype=np.int32
).T
toy_query_0 = np.array(
    [
        [-1, 0, -1, 0, -1, 0, -1, 0, -1],  # query_0, haplotype 0
        [-1, 0, -1, 0, -1, 0, -1, 0, -1],  # query_0, haplotype 1
    ], dtype=np.int32
)
toy_query_0_beagle_imputed = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # query_0, haplotype 0
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # query_0, haplotype 1
    ], dtype=np.int32
)
toy_query_0_beagle_allele_probs = np.array(
    # Ungenotyped markers only
    [
        # query_0, haplotype 0
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        # query_0, haplotype 1
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
    ],
    dtype=np.float32,
)
# fmt: on
assert toy_ref_0.shape == (9, 2)
assert toy_query_0.shape == (2, 9)
assert toy_query_0_beagle_imputed.shape == (2, 9)
assert np.sum(toy_query_0[0] == 0) == 4
assert np.sum(toy_query_0[0] == -1) == 5
assert np.array_equal(toy_query_0[0], toy_query_0[1])
assert np.all(toy_query_0_beagle_imputed[0] == 0)
assert np.all(toy_query_0_beagle_imputed[1] == 0)
assert toy_query_0_beagle_allele_probs.shape == (2, 2, 5)

# Case 1:
# Reference haplotypes and query haplotypes have 1|1 at all sites.
# First and last reference markers are missing in the query haplotypes.
# fmt: off
toy_ref_1 = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],  # ref_0, haplotype 0
        [1, 1, 1, 1, 1, 1, 1, 1, 1],  # ref_0, haplotype 1
    ], dtype=np.int32
).T
toy_query_1 = np.array(
    [
        [-1, 1, -1, 1, -1, 1, -1, 1, -1],  # query_0, haplotype 0
        [-1, 1, -1, 1, -1, 1, -1, 1, -1],  # query_0, haplotype 1
    ], dtype=np.int32
)
toy_query_1_beagle_imputed = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],  # query_0, haplotype 0
        [1, 1, 1, 1, 1, 1, 1, 1, 1],  # query_0, haplotype 1
    ], dtype=np.int32
)
toy_query_1_beagle_allele_probs = np.array(
    # Ungenotyped markers only
    [
        # query_0, haplotype 0
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ],
        # query_0, haplotype 1
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ],
    ],
    dtype=np.float32,
)
# fmt: on
assert toy_ref_1.shape == (9, 2)
assert toy_query_1.shape == (2, 9)
assert toy_query_1_beagle_imputed.shape == (2, 9)
assert np.sum(toy_query_1[0] == 1) == 4
assert np.sum(toy_query_1[0] == -1) == 5
assert np.array_equal(toy_query_1[0], toy_query_1[1])
assert np.all(toy_query_1_beagle_imputed[0] == 1)
assert np.all(toy_query_1_beagle_imputed[1] == 1)
assert toy_query_1_beagle_allele_probs.shape == (2, 2, 5)

# Case 2:
# Reference haplotypes and query haplotypes have 0|0 at all sites.
# First and last reference markers are genotyped in the query haplotypes.
# fmt: off
toy_ref_2 = np.copy(toy_ref_0)
toy_query_2 = np.array(
    [
        [0, 0, -1, 0, -1, 0, -1, 0, 0],  # query_0, haplotype 0
        [0, 0, -1, 0, -1, 0, -1, 0, 0],  # query_0, haplotype 1
    ], dtype=np.int32
)
toy_query_2_beagle_imputed = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # query_0, haplotype 0
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # query_0, haplotype 1
    ], dtype=np.int32
)
toy_query_2_beagle_allele_probs = np.array(
    # Ungenotyped markers only
    [
        # query_0, haplotype 0
        [
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        # query_0, haplotype 1
        [
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
    ],
    dtype=np.float32,
)
# fmt: on
assert toy_ref_2.shape == (9, 2)
assert toy_query_2.shape == (2, 9)
assert toy_query_2_beagle_imputed.shape == (2, 9)
assert np.sum(toy_query_2[0] == 0) == 6
assert np.sum(toy_query_2[0] == -1) == 3
assert np.array_equal(toy_query_2[0], toy_query_2[1])
assert np.all(toy_query_2_beagle_imputed[0] == 0)
assert np.all(toy_query_2_beagle_imputed[1] == 0)
assert toy_query_2_beagle_allele_probs.shape == (2, 2, 3)

# Case 3:
# Reference haplotypes and query haplotypes have 0|1 at all sites.
# fmt: off
toy_ref_3 = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # ref_0, haplotype 0
        [1, 1, 1, 1, 1, 1, 1, 1, 1],  # ref_0, haplotype 1
    ], dtype=np.int32
).T
toy_query_3 = np.array(
    [
        [-1, 0, -1, 0, -1, 0, -1, 0, -1],  # query_0, haplotype 0
        [-1, 1, -1, 1, -1, 1, -1, 1, -1],  # query_0, haplotype 1
    ], dtype=np.int32
)
toy_query_3_beagle_imputed = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # query_0, haplotype 0
        [1, 1, 1, 1, 1, 1, 1, 1, 1],  # query_0, haplotype 1
    ], dtype=np.int32
)
toy_query_3_beagle_allele_probs = np.array(
    # Ungenotyped markers only
    # See comment https://github.com/tskit-dev/tskit/pull/2815#issuecomment-1708178257
    [
        # query_0, haplotype 0
        [
            [0.9999998, 0.9999999, 1.0, 0.9999999, 0.9999998],
            [0.0      , 0.0      , 0.0, 0.0      , 0.0],
        ],
        # query_0, haplotype 1
        [
            [0.0      , 0.0      , 0.0, 0.0      , 0.0],
            [0.9999998, 0.9999999, 1.0, 0.9999999, 0.9999998],
        ],
    ],
    dtype=np.float32,
)
# fmt: on
assert toy_ref_3.shape == (9, 2)
assert toy_query_3.shape == (2, 9)
assert toy_query_3_beagle_imputed.shape == (2, 9)
assert np.sum(toy_query_3[0] == 0) == 4
assert np.sum(toy_query_3[1] == 1) == 4
assert np.sum(toy_query_3[0] == -1) == np.sum(toy_query_3[1] == -1) == 5
assert np.all(
    np.invert(np.equal(toy_query_3_beagle_imputed[0], toy_query_3_beagle_imputed[1]))
)
assert toy_query_3_beagle_allele_probs.shape == (2, 2, 5)

# Case 4:
# Reference haplotypes and query haplotypes have alternating 0|1 and 1|0 genotypes.
# Query haplotypes have 0|1 at all genotyped sites.
# fmt: off
toy_ref_4 = np.array(
    [
        [0, 1, 0, 1, 0, 1, 0, 1, 0],  # ref_0, haplotype 0
        [1, 0, 1, 0, 1, 0, 1, 0, 1],  # ref_0, haplotype 1
    ], dtype=np.int32
).T
toy_query_4 = np.array(
    [
        [-1, 0, -1, 0, -1, 0, -1, 0, -1],  # query_0, haplotype 0
        [-1, 1, -1, 1, -1, 1, -1, 1, -1],  # query_0, haplotype 1
    ], dtype=np.int32
)
toy_query_4_beagle_imputed = np.array(
    [
        [1, 0, 1, 0, 1, 0, 1, 0, 1],  # query_0, haplotype 0
        [0, 1, 0, 1, 0, 1, 0, 1, 0],  # query_0, haplotype 1
    ], dtype=np.int32
)
toy_query_4_beagle_allele_probs = np.array(
    # Ungenotyped markers only
    [
        # query_0, haplotype 0
        [
            [0.0      , 0.0      , 0.0, 0.0      , 0.9999998],
            [0.9999998, 0.9999999, 1.0, 0.9999999, 0.0],
        ],
        # query_0, haplotype 1
        [
            [0.9999998, 0.9999999, 1.0, 0.9999999, 0.0],
            [0.0      , 0.0      , 0.0, 0.0      , 0.9999998],
        ],
    ],
    dtype=np.float32,
)
# fmt: on
assert toy_ref_4.shape == (9, 2)
assert toy_query_4.shape == (2, 9)
assert toy_query_4_beagle_imputed.shape == (2, 9)
assert np.sum(toy_query_4[0] == 0) == 4
assert np.sum(toy_query_4[1] == 1) == 4
assert np.sum(toy_query_4[0] == -1) == np.sum(toy_query_4[1] == -1) == 5
assert np.all(
    np.invert(np.equal(toy_query_4_beagle_imputed[0], toy_query_4_beagle_imputed[1]))
)
assert toy_query_3_beagle_allele_probs.shape == (2, 2, 5)

# Case 5:
# The reference panel has two individuals. The first individual has 0|0 at all sites,
# and the second individual has 1|1 at all sites.
# The query haplotypes have one recombination breakpoint in the middle.
# fmt: off
toy_ref_5 = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # ref_0
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # ref_0
        [1, 1, 1, 1, 1, 1, 1, 1, 1],  # ref_1
        [1, 1, 1, 1, 1, 1, 1, 1, 1],  # ref_1
    ], dtype=np.int32
).T
toy_query_5 = np.array(
    [
        [-1, 0, -1, 0, -1, 0, -1, 1, -1],  # query_0, haplotype 0
        [-1, 1, -1, 1, -1, 1, -1, 0, -1],  # query_0, haplotype 1
    ], dtype=np.int32
)
toy_query_5_beagle_imputed = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 1, 1],  # query_0, haplotype 0
        [1, 1, 1, 1, 1, 1, 1, 0, 0],  # query_0, haplotype 1
    ], dtype=np.int32
)
toy_query_5_beagle_allele_probs = np.array(
    # Ungenotyped markers only
    [
        # query_0, haplotype 0
        [
            [0.9999999, 0.99999994, 0.9999546, 0.5454091, 0.09090912],
            [0.0      , 0.0       , 0.0      , 0.4545909, 0.9090909],
        ],
        # query_0, haplotype 1
        [
            [0.0      , 0.0       , 0.0      , 0.4545909, 0.9090909],
            [0.9999999, 0.99999994, 0.9999546, 0.5454091, 0.09090912],
        ],
    ],
    dtype=np.float32,
)
# fmt: on
assert toy_ref_5.shape == (9, 4)
assert toy_query_5.shape == (2, 9)
assert toy_query_5_beagle_imputed.shape == (2, 9)
assert np.sum(toy_query_5[0] == 0) == 3
assert np.sum(toy_query_5[0] == 1) == 1
assert np.sum(toy_query_5[0] == -1) == 5
assert np.sum(toy_query_5[1] == 0) == 1
assert np.sum(toy_query_5[1] == 1) == 3
assert np.sum(toy_query_5[1] == -1) == 5
assert np.all(
    np.invert(np.equal(toy_query_5_beagle_imputed[0], toy_query_5_beagle_imputed[1]))
)
assert toy_query_5_beagle_allele_probs.shape == (2, 2, 5)

# Case 6:
# The reference panel has two individuals. The first individual has 0|0 at all sites,
# and the second individual has 1|1 at all sites.
# The query haplotypes have two recombination breakpoints in the middle.
# fmt: off
toy_ref_6 = np.copy(toy_ref_5)
toy_query_6 = np.array(
    [
        [-1, 0, -1, 1, -1, 1, -1, 0, -1],  # query_0, haplotype 0
        [-1, 1, -1, 0, -1, 0, -1, 1, -1],  # query_0, haplotype 1
    ], dtype=np.int32
)
toy_query_6_beagle_imputed = np.array(
    [
        [0, 0, 1, 1, 1, 1, 1, 0, 0],  # query_0, haplotype 0
        [1, 1, 0, 0, 0, 0, 0, 1, 1],  # query_0, haplotype 1
    ], dtype=np.int32
)
toy_query_6_beagle_allele_probs = np.array(
    # Ungenotyped markers only
    [
        # query_0, haplotype 0
        [
            [0.90983605, 0.46770614, 0.025481388, 0.46838862, 0.91139066],
            [0.09016396, 0.53229386, 0.97451866 , 0.5316114 , 0.088609315],
        ],
        # query_0, haplotype 1
        [
            [0.09016395, 0.53229386, 0.97451866 , 0.5316114 , 0.0886093],
            [0.90983605, 0.46770614, 0.025481388, 0.46838862, 0.91139066],
        ],
    ],
    dtype=np.float32,
)
# fmt: on
assert toy_ref_6.shape == (9, 4)
assert toy_query_6.shape == (2, 9)
assert toy_query_6_beagle_imputed.shape == (2, 9)
assert np.sum(toy_query_6[0] == 0) == 2
assert np.sum(toy_query_6[0] == 1) == 2
assert np.sum(toy_query_6[0] == -1) == 5
assert np.sum(toy_query_6[1] == 0) == 2
assert np.sum(toy_query_6[1] == 1) == 2
assert np.sum(toy_query_6[1] == -1) == 5
assert np.all(
    np.invert(np.equal(toy_query_6_beagle_imputed[0], toy_query_6_beagle_imputed[1]))
)
assert toy_query_6_beagle_allele_probs.shape == (2, 2, 5)

# Case 7:
# Hand-crafted example.
# fmt: off
toy_ref_7 = np.array(
    [
        [0, 1, 2, 0, 1, 1, 2, 3, 1],    # ref_0
        [1, 1, 2, 3, 1, 1, 2, 3, 1],    # ref_0
        [0, 1, 3, 0, 1, 1, 2, 3, 1],    # ref_1
        [0, 2, 2, 3, 0, 1, 2, 0, 0],    # ref_1
        [0, 1, 2, 3, 0, 2, 3, 0, 0],    # ref_2
        [1, 1, 2, 3, 1, 1, 2, 3, 1],    # ref_2
        [0, 1, 3, 0, 1, 1, 2, 3, 1],    # ref_3
        [0, 1, 2, 3, 0, 1, 2, 3, 1],    # ref_3
        [0, 2, 2, 3, 0, 1, 2, 0, 0],    # ref_4
        [0, 1, 2, 3, 0, 2, 3, 0, 0],    # ref_4
    ], dtype=np.int32
).T
toy_query_7 = np.array(
    [
        [0, 1, -1, 0, -1, 2, -1, 3, 1],   # query_0, haplotype 0
        [1, 2, -1, 3, -1, 1, -1, 0, 0],   # query_0, haplotype 1
    ], dtype=np.int32
)
toy_query_7_beagle_imputed = np.array(
    [
        [0, 1, 3, 0, 1, 2, 2, 3, 1],    # query_0, haplotype 0
        [1, 2, 2, 3, 0, 1, 2, 0, 0],    # query_0, haplotype 1
    ], dtype=np.int32
)
toy_query_7_beagle_allele_probs = np.array(
    # Ungenotyped markers only
    [
        # query_0, haplotype 0
        [
            [0.33317572, 0.0      , 0.99945664],
            [0.66635144, 0.9991437, 0.0],
        ],
        # query_0, haplotype 1
        [
            [0.9995998, 0.99971414, 0.99974835],
            [0.0      , 0.0       , 0.0],
        ],
    ],
    dtype=np.float32,
)
# fmt: on
assert toy_ref_7.shape == (9, 10)
assert toy_query_7.shape == (2, 9)
assert toy_query_7_beagle_imputed.shape == (2, 9)
assert np.sum(toy_query_7[0] == -1) == np.sum(toy_query_7[1] == -1) == 3
assert toy_query_7_beagle_allele_probs.shape == (2, 2, 3)

# Case 8:
# Same as case 7 except the last genotyped marker is missing.
# fmt: off
toy_ref_8 = np.copy(toy_ref_7)
toy_query_8 = np.array(
    [
        [-1, 1, -1, 0, -1, 2, -1, 3, -1],  # query_0, haplotype 0
        [-1, 2, -1, 3, -1, 1, -1, 0, -1],  # query_0, haplotype 1
    ], dtype=np.int32
)
toy_query_8_beagle_imputed = np.array(
    [
        [0, 1, 3, 0, 1, 2, 2, 3, 1],    # query_0, haplotype 0
        [0, 2, 2, 3, 0, 1, 2, 0, 0],    # query_0, haplotype 1
    ], dtype=np.int32
)
toy_query_8_beagle_allele_probs = np.array(
    # Ungenotyped markers only
    [
        # query_0, haplotype 0
        [
            [0.9997735, 0.33310473, 0.0       , 0.9992305, 0.9997736],
            [0.0      , 0.66620946, 0.99893093, 0.0      , 0.0],
        ],
        # query_0, haplotype 1
        [
            [0.9999998, 0.9999998, 0.9999998, 0.99991965, 0.0],
            [0.0      , 0.0      , 0.0      , 0.0       , 0.9999998],
        ],
    ],
    dtype=np.float32,
)
# fmt: on
assert toy_ref_8.shape == (9, 10)
assert toy_query_8.shape == (2, 9)
assert toy_query_8_beagle_imputed.shape == (2, 9)
assert np.sum(toy_query_8[0] == -1) == 5
assert np.sum(toy_query_8[1] == -1) == 5
assert toy_query_8_beagle_allele_probs.shape == (2, 2, 5)


@pytest.mark.parametrize(
    "input_ref,input_query,expected",
    [
        (toy_ref_0, toy_query_0, toy_query_0_beagle_imputed),
        (toy_ref_1, toy_query_1, toy_query_1_beagle_imputed),
        (toy_ref_2, toy_query_2, toy_query_2_beagle_imputed),
        (toy_ref_3, toy_query_3, toy_query_3_beagle_imputed),
        (toy_ref_4, toy_query_4, toy_query_4_beagle_imputed),
        (toy_ref_5, toy_query_5, toy_query_5_beagle_imputed),
        (toy_ref_6, toy_query_6, toy_query_6_beagle_imputed),
        (toy_ref_7, toy_query_7, toy_query_7_beagle_imputed),
        (toy_ref_8, toy_query_8, toy_query_8_beagle_imputed),
    ],
)
def test_beagle_vanilla(input_ref, input_query, expected):
    """Compare imputed alleles from Python BEAGLE implementation and BEAGLE 4.1."""
    assert input_query.shape == expected.shape
    pos = (np.arange(9) + 1) * 1e4
    num_query_haps = input_query.shape[0]
    num_ungenotyped_markers = np.sum(input_query[0] == -1)
    imputed_alleles = np.zeros(
        (num_query_haps, num_ungenotyped_markers), dtype=np.int32
    )
    expected_ungenotyped = expected[:, input_query[0] == -1]
    assert imputed_alleles.shape == expected_ungenotyped.shape
    for i in range(num_query_haps):
        imputed_alleles[i], _ = tests.beagle.run_beagle(
            input_ref, input_query[i], pos, miscall_rate=1e-4, ne=10.0
        )
        np.testing.assert_array_equal(imputed_alleles[i], expected_ungenotyped[i])


@pytest.mark.parametrize(
    "input_ref,input_query",
    [
        (toy_ref_0, toy_query_0),
        (toy_ref_1, toy_query_1),
        (toy_ref_2, toy_query_2),
        (toy_ref_3, toy_query_3),
        (toy_ref_4, toy_query_4),
        (toy_ref_5, toy_query_5),
        (toy_ref_6, toy_query_6),
        (toy_ref_7, toy_query_7),
        (toy_ref_8, toy_query_8),
    ],
)
def test_beagle_numba(input_ref, input_query):
    """Compare imputed alleles from vanilla and numba Python BEAGLE implementations."""
    pos = (np.arange(9) + 1) * 1e4
    num_query_haps = input_query.shape[0]
    for i in range(num_query_haps):
        imputed_alleles_vanilla, _ = tests.beagle.run_beagle(
            input_ref,
            input_query[i],
            pos,
            ne=10.0,
            miscall_rate=1e-4,
        )
        imputed_alleles_numba, _ = tests.beagle_numba.run_interpolation_beagle(
            input_ref,
            input_query[i],
            pos,
            ne=10.0,
            error_rate=1e-4,
        )
        np.testing.assert_array_equal(imputed_alleles_vanilla, imputed_alleles_numba)


# Below is toy data set case 7 in tree sequence format.
toy_ts_nodes_text = """\
id      is_sample       time    population      individual      metadata
0       1       0.000000        0       0
1       1       0.000000        0       0
2       1       0.000000        0       1
3       1       0.000000        0       1
4       1       0.000000        0       2
5       1       0.000000        0       2
6       1       0.000000        0       3
7       1       0.000000        0       3
8       1       0.000000        0       4
9       1       0.000000        0       4
10      0       0.009923        0       -1
11      0       0.038603        0       -1
12      0       0.057935        0       -1
13      0       0.145141        0       -1
14      0       0.238045        0       -1
15      0       0.528344        0       -1
16      0       0.646418        0       -1
17      0       1.462199        0       -1
18      0       2.836600        0       -1
19      0       3.142225        0       -1
20      0       4.056253        0       -1
"""

toy_ts_edges_text = """\
left    right   parent  child   metadata
0.000000        100000.000000   10      1
0.000000        100000.000000   10      5
0.000000        100000.000000   11      3
0.000000        100000.000000   11      8
0.000000        100000.000000   12      2
0.000000        100000.000000   12      6
0.000000        100000.000000   13      0
0.000000        100000.000000   13      12
0.000000        100000.000000   14      10
0.000000        100000.000000   14      13
0.000000        40443.000000    15      9
0.000000        40443.000000    15      14
40443.000000    100000.000000   16      4
40443.000000    100000.000000   16      9
0.000000        40443.000000    17      4
0.000000        100000.000000   17      11
40443.000000    100000.000000   17      16
0.000000        12721.000000    18      7
0.000000        12721.000000    18      17
12721.000000    100000.000000   19      7
40443.000000    100000.000000   19      14
12721.000000    40443.000000    19      15
0.000000        12721.000000    20      15
12721.000000    100000.000000   20      17
0.000000        12721.000000    20      18
12721.000000    100000.000000   20      19
"""

toy_ts_sites_text = """\
position        ancestral_state metadata
10000.000000    A
20000.000000    C
30000.000000    G
40000.000000    T
50000.000000    A
60000.000000    C
70000.000000    G
80000.000000    T
90000.000000    A
"""

toy_ts_mutations_text = """\
site    node    time    derived_state   parent  metadata
0       10      unknown C       -1
1       11      unknown G       -1
2       12      unknown T       -1
3       13      unknown A       -1
4       14      unknown C       -1
5       16      unknown G       -1
6       16      unknown T       -1
7       17      unknown A       -1
8       19      unknown C       -1
"""

toy_ts_individuals_text = """\
flags
0
0
0
0
0
"""


def get_toy_data():
    """Get toy example 7 in tree sequence format."""
    ts = tskit.load_text(
        nodes=io.StringIO(toy_ts_nodes_text),
        edges=io.StringIO(toy_ts_edges_text),
        sites=io.StringIO(toy_ts_sites_text),
        mutations=io.StringIO(toy_ts_mutations_text),
        individuals=io.StringIO(toy_ts_individuals_text),
        strict=False,
    )
    return ts


def parse_matrix(csv_text):
    # TODO: Maybe remove.
    # This returns a record array, which is essentially the same as a
    # pandas dataframe, which we can access via df["m"] etc.
    return np.lib.npyio.recfromcsv(io.StringIO(csv_text))


@pytest.mark.parametrize(
    "input_query,expected",
    [
        (toy_query_7, toy_query_7_beagle_imputed),
        (toy_query_8, toy_query_8_beagle_imputed),
    ],
)
def test_tsimpute(input_query, expected):
    """
    Compare imputed alleles from tsimpute and BEAGLE 4.1.
    """
    toy_ref_ts = get_toy_data()  # Same for both cases
    pos = toy_ref_ts.sites_position
    num_query_haps = input_query.shape[0]
    mu = np.zeros(len(pos), dtype=np.float32) + 1e-8
    rho = np.zeros(len(pos), dtype=np.float32) + 1e-8
    expected_subset = expected[:, input_query[0] == -1]
    for i in range(num_query_haps):
        imputed_alleles, _ = tests.beagle_numba.run_tsimpute(
            toy_ref_ts,
            input_query[i],
            pos,
            rho,
            mu,
        )
        np.testing.assert_array_equal(imputed_alleles, expected_subset[i])


# Tests for helper functions.
@pytest.mark.parametrize(
    "typed_pos,untyped_pos,expected_weights,expected_idx",
    [
        # All ungenotyped markers are between genotyped markers.
        (
            np.array([10, 20]) * 1e6,
            np.array([15]) * 1e6,
            np.array([0.5]),
            np.array([0]),
        ),
        # Same as above, but more genotyped markers.
        (
            np.array([10, 20, 30, 40]) * 1e6,
            np.array([15, 25, 35]) * 1e6,
            np.array([0.5, 0.5, 0.5]),
            np.array([0, 1, 2]),
        ),
        # Ungenotyped markers are left of the first genotyped marker.
        (
            np.array([10, 20]) * 1e6,
            np.array([5, 15]) * 1e6,
            np.array([1.0, 0.5]),
            np.array([0, 0]),
        ),
        # Ungenotyped markers are right of the last genotyped marker.
        (
            np.array([10, 20]) * 1e6,
            np.array([15, 25]) * 1e6,
            np.array([0.5, 0.0]),
            np.array([0, 1]),
        ),
        # Denominator below min threshold.
        (np.array([10, 20]), np.array([15]), np.array([0.001]), np.array([0])),
    ],
)
def test_get_weights(typed_pos, untyped_pos, expected_weights, expected_idx):
    # beagle, no numba
    actual_weights, actual_idx = tests.beagle.get_weights(typed_pos, untyped_pos)
    np.testing.assert_allclose(actual_weights, expected_weights)
    np.testing.assert_array_equal(actual_idx, expected_idx)
    # beagle, numba
    genotyped_cm = tests.beagle_numba.convert_to_genetic_map_positions(typed_pos)
    ungenotyped_cm = tests.beagle_numba.convert_to_genetic_map_positions(untyped_pos)
    actual_weights, actual_idx = tests.beagle_numba.get_weights(
        typed_pos, untyped_pos, genotyped_cm, ungenotyped_cm
    )
    np.testing.assert_allclose(actual_weights, expected_weights)
    np.testing.assert_array_equal(actual_idx, expected_idx)
