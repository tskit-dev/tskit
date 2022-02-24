"""Collection of functions to run forwards and backwards algorithms on diploid genotype data, where the data is structured as samples x variants."""
import numpy as np

EQUAL_BOTH_HOM = 4
UNEQUAL_BOTH_HOM = 0
BOTH_HET = 7
REF_HOM_OBS_HET = 1
REF_HET_OBS_HOM = 2

# https://github.com/numba/numba/issues/1269
def np_apply_along_axis(func1d, axis, arr):
    """Create numpy-like functions for max, sum etc."""
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result


def np_amax(array, axis):
    """Numba implementation of numpy vectorised maximum."""
    return np_apply_along_axis(np.amax, axis, array)


def np_sum(array, axis):
    """Numba implementation of numpy vectorised sum."""
    return np_apply_along_axis(np.sum, axis, array)

  
def np_argmax(array, axis):
    """Numba implementation of numpy vectorised argmax."""
    return np_apply_along_axis(np.argmax, axis, array)


def forwards_ls_dip(n, m, G, s, e, r, norm=True):
    """Matrix based diploid LS forward algorithm using numpy vectorisation."""
    # Initialise the forward tensor
    F = np.zeros((n, n, m))
    F[:, :, 0] = 1 / (n ** 2)
    index = 4 * np.equal(G[:, :, 0], s[0, 0]).astype(np.int64) + 2 * (
        G[:, :, 0] == 1
    ).astype(np.int64)
    if s[0, 0] == 1:
        index += 1
    F[:, :, 0] *= e[index.ravel(), 0].reshape(n, n)
    c = np.ones(m)
    r_n = r / n

    if norm:
        c[0] = np.sum(F[:, :, 0])
        F[:, :, 0] *= 1 / c[0]

        # Forwards
        for l in range(1, m):

            index = 4 * np.equal(G[:, :, l], s[0, l]).astype(np.int64) + 2 * (
                G[:, :, l] == 1
            ).astype(np.int64)

            if s[0, l] == 1:
                index += 1

            # No change in both
            F[:, :, l] = (1 - r[l]) ** 2 * F[:, :, l - 1]

            # Both change
            F[:, :, l] += (r_n[l]) ** 2

            # One changes
            sum_j = np_sum(F[:, :, l - 1], 0).repeat(n).reshape((-1, n))
            F[:, :, l] += ((1 - r[l]) * r_n[l]) * (sum_j + sum_j.T)

            # Emission
            F[:, :, l] *= e[index.ravel(), l].reshape(n, n)
            c[l] = np.sum(F[:, :, l])
            F[:, :, l] *= 1 / c[l]

        ll = np.sum(np.log10(c))
    else:
        # Forwards
        for l in range(1, m):

            index = 4 * np.equal(G[:, :, l], s[0, l]).astype(np.int64) + 2 * (
                G[:, :, l] == 1
            ).astype(np.int64)

            if s[0, l] == 1:
                index += 1

            # No change in both
            F[:, :, l] = (1 - r[l]) ** 2 * F[:, :, l - 1]

            # Both change
            F[:, :, l] += (r_n[l]) ** 2 * np.sum(F[:, :, l - 1])

            # One changes
            sum_j = np_sum(F[:, :, l - 1], 0).repeat(n).reshape((-1, n)).T
            F[:, :, l] += ((1 - r[l]) * r_n[l]) * (sum_j + sum_j.T)

            # Emission
            F[:, :, l] *= e[index.ravel(), l].reshape(n, n)

        ll = np.log10(np.sum(F[:, :, l]))

    return F, c, ll


def backwards_ls_dip(n, m, G, s, e, c, r):
    """Matrix based diploid LS backward algorithm using numpy vectorisation."""
    # Initialise the backward tensor
    B = np.zeros((n, n, m))

    # Initialise
    B[:, :, m - 1] = 1
    r_n = r / n

    # Backwards
    for l in range(m - 2, -1, -1):

        index = (
            4 * np.equal(G[:, :, l + 1], s[0, l + 1]).astype(np.int64)
            + 2 * (G[:, :, l + 1] == 1).astype(np.int64)
            + np.int64(s[0, l + 1] == 1)
        ).ravel()

        # Both change
        B[:, :, l] = r_n[l + 1] ** 2 * np.sum(
            e[index, l + 1].reshape(n, n) * B[:, :, l + 1]
        )

        # No change in both
        B[:, :, l] += (
            (1 - r[l + 1]) ** 2 * B[:, :, l + 1] * e[index, l + 1].reshape(n, n)
        )

        sum_j = (
            np_sum(B[:, :, l + 1] * e[index, l + 1].reshape(n, n), 0)
            .repeat(n)
            .reshape((-1, n))
        )
        B[:, :, l] += ((1 - r[l + 1]) * r_n[l + 1]) * (sum_j + sum_j.T)
        B[:, :, l] *= 1 / c[l + 1]

    return B


def forward_ls_dip_starting_point(n, m, G, s, e, r):
    """Naive implementation of LS diploid forwards algorithm."""
    # Initialise the forward tensor
    F = np.zeros((n, n, m))
    F[:, :, 0] = 1 / (n ** 2)
    index = (
        4 * np.equal(G[:, :, 0], s[0, 0]).astype(np.int64)
        + 2 * (G[:, :, 0] == 1).astype(np.int64)
        + np.int64(s[0, 0] == 1)
    )
    F[:, :, 0] *= e[index.ravel(), 0].reshape(n, n)
    r_n = r / n

    for l in range(1, m):

        # Determine the various components
        F_no_change = np.zeros((n, n))
        F_j1_change = np.zeros(n)
        F_j2_change = np.zeros(n)
        F_both_change = 0

        for j1 in range(n):
            for j2 in range(n):
                F_no_change[j1, j2] = (1 - r[l]) ** 2 * F[j1, j2, l - 1]

        for j1 in range(n):
            for j2 in range(n):
                F_both_change += r_n[l] ** 2 * F[j1, j2, l - 1]

        for j1 in range(n):
            for j2 in range(n):  # This is the variable to sum over - it changes
                F_j2_change[j1] += (1 - r[l]) * r_n[l] * F[j1, j2, l - 1]

        for j2 in range(n):
            for j1 in range(n):  # This is the variable to sum over - it changes
                F_j1_change[j2] += (1 - r[l]) * r_n[l] * F[j1, j2, l - 1]

        F[:, :, l] = F_both_change

        for j1 in range(n):
            F[j1, :, l] += F_j2_change

        for j2 in range(n):
            F[:, j2, l] += F_j1_change

        for j1 in range(n):
            for j2 in range(n):
                F[j1, j2, l] += F_no_change[j1, j2]

        for j1 in range(n):
            for j2 in range(n):
                # What is the emission?
                if s[0, l] == 1:
                    # OBS is het
                    if G[j1, j2, l] == 1:  # REF is het
                        F[j1, j2, l] *= e[BOTH_HET, l]
                    else:  # REF is hom
                        F[j1, j2, l] *= e[REF_HOM_OBS_HET, l]
                else:
                    # OBS is hom
                    if G[j1, j2, l] == 1:  # REF is het
                        F[j1, j2, l] *= e[REF_HET_OBS_HOM, l]
                    else:  # REF is hom
                        if G[j1, j2, l] == s[0, l]:  # Equal
                            F[j1, j2, l] *= e[EQUAL_BOTH_HOM, l]
                        else:  # Unequal
                            F[j1, j2, l] *= e[UNEQUAL_BOTH_HOM, l]

    ll = np.log10(np.sum(F[:, :, l]))
    return F, ll


def backward_ls_dip_starting_point(n, m, G, s, e, r):
    """Naive implementation of LS diploid backwards algorithm."""
    # Backwards
    B = np.zeros((n, n, m))

    # Initialise
    B[:, :, m - 1] = 1
    r_n = r / n

    for l in range(m - 2, -1, -1):

        # Determine the various components
        B_no_change = np.zeros((n, n))
        B_j1_change = np.zeros(n)
        B_j2_change = np.zeros(n)
        B_both_change = 0

        # Evaluate the emission matrix at this site, for all pairs
        e_tmp = np.zeros((n, n))
        for j1 in range(n):
            for j2 in range(n):
                # What is the emission?
                if s[0, l + 1] == 1:
                    # OBS is het
                    if G[j1, j2, l + 1] == 1:  # REF is het
                        e_tmp[j1, j2] = e[BOTH_HET, l + 1]
                    else:  # REF is hom
                        e_tmp[j1, j2] = e[REF_HOM_OBS_HET, l + 1]
                else:
                    # OBS is hom
                    if G[j1, j2, l + 1] == 1:  # REF is het
                        e_tmp[j1, j2] = e[REF_HET_OBS_HOM, l + 1]
                    else:  # REF is hom
                        if G[j1, j2, l + 1] == s[0, l + 1]:  # Equal
                            e_tmp[j1, j2] = e[EQUAL_BOTH_HOM, l + 1]
                        else:  # Unequal
                            e_tmp[j1, j2] = e[UNEQUAL_BOTH_HOM, l + 1]

        for j1 in range(n):
            for j2 in range(n):
                B_no_change[j1, j2] = (
                    (1 - r[l + 1]) ** 2 * B[j1, j2, l + 1] * e_tmp[j1, j2]
                )

        for j1 in range(n):
            for j2 in range(n):
                B_both_change += r_n[l + 1] ** 2 * e_tmp[j1, j2] * B[j1, j2, l + 1]

        for j1 in range(n):
            for j2 in range(n):  # This is the variable to sum over - it changes
                B_j2_change[j1] += (
                    (1 - r[l + 1]) * r_n[l + 1] * B[j1, j2, l + 1] * e_tmp[j1, j2]
                )

        for j2 in range(n):
            for j1 in range(n):  # This is the variable to sum over - it changes
                B_j1_change[j2] += (
                    (1 - r[l + 1]) * r_n[l + 1] * B[j1, j2, l + 1] * e_tmp[j1, j2]
                )

        B[:, :, l] = B_both_change

        for j1 in range(n):
            B[j1, :, l] += B_j2_change

        for j2 in range(n):
            B[:, j2, l] += B_j1_change

        for j1 in range(n):
            for j2 in range(n):
                B[j1, j2, l] += B_no_change[j1, j2]

    return B


def forward_ls_dip_loop(n, m, G, s, e, r, norm=True):
    """LS diploid forwards algoritm without vectorisation."""
    # Initialise the forward tensor
    F = np.zeros((n, n, m))
    F[:, :, 0] = 1 / (n ** 2)
    index = (
        4 * np.equal(G[:, :, 0], s[0, 0]).astype(np.int64)
        + 2 * (G[:, :, 0] == 1).astype(np.int64)
        + np.int64(s[0, 0] == 1)
    )
    F[:, :, 0] *= e[index.ravel(), 0].reshape(n, n)
    r_n = r / n
    c = np.ones(m)

    if norm:

        c[0] = np.sum(F[:, :, 0])
        F[:, :, 0] *= 1 / c[0]

        for l in range(1, m):

            # Determine the various components
            F_no_change = np.zeros((n, n))
            F_j_change = np.zeros(n)

            for j1 in range(n):
                for j2 in range(n):
                    F_no_change[j1, j2] = (1 - r[l]) ** 2 * F[j1, j2, l - 1]
                    F_j_change[j1] += (1 - r[l]) * r_n[l] * F[j2, j1, l - 1]

            F[:, :, l] = r_n[l] ** 2

            for j1 in range(n):
                F[j1, :, l] += F_j_change
                F[:, j1, l] += F_j_change
                for j2 in range(n):
                    F[j1, j2, l] += F_no_change[j1, j2]

            for j1 in range(n):
                for j2 in range(n):
                    # What is the emission?
                    if s[0, l] == 1:
                        # OBS is het
                        if G[j1, j2, l] == 1:  # REF is het
                            F[j1, j2, l] *= e[BOTH_HET, l]
                        else:  # REF is hom
                            F[j1, j2, l] *= e[REF_HOM_OBS_HET, l]
                    else:
                        # OBS is hom
                        if G[j1, j2, l] == 1:  # REF is het
                            F[j1, j2, l] *= e[REF_HET_OBS_HOM, l]
                        else:  # REF is hom
                            if G[j1, j2, l] == s[0, l]:  # Equal
                                F[j1, j2, l] *= e[EQUAL_BOTH_HOM, l]
                            else:  # Unequal
                                F[j1, j2, l] *= e[UNEQUAL_BOTH_HOM, l]

            c[l] = np.sum(F[:, :, l])
            F[:, :, l] *= 1 / c[l]

        ll = np.sum(np.log10(c))

    else:

        for l in range(1, m):

            # Determine the various components
            F_no_change = np.zeros((n, n))
            F_j1_change = np.zeros(n)
            F_j2_change = np.zeros(n)
            F_both_change = 0

            for j1 in range(n):
                for j2 in range(n):
                    F_no_change[j1, j2] = (1 - r[l]) ** 2 * F[j1, j2, l - 1]
                    F_j1_change[j1] += (1 - r[l]) * r_n[l] * F[j2, j1, l - 1]
                    F_j2_change[j1] += (1 - r[l]) * r_n[l] * F[j1, j2, l - 1]
                    F_both_change += r_n[l] ** 2 * F[j1, j2, l - 1]

            F[:, :, l] = F_both_change

            for j1 in range(n):
                F[j1, :, l] += F_j2_change
                F[:, j1, l] += F_j1_change
                for j2 in range(n):
                    F[j1, j2, l] += F_no_change[j1, j2]

            for j1 in range(n):
                for j2 in range(n):
                    # What is the emission?
                    if s[0, l] == 1:
                        # OBS is het
                        if G[j1, j2, l] == 1:  # REF is het
                            F[j1, j2, l] *= e[BOTH_HET, l]
                        else:  # REF is hom
                            F[j1, j2, l] *= e[REF_HOM_OBS_HET, l]
                    else:
                        # OBS is hom
                        if G[j1, j2, l] == 1:  # REF is het
                            F[j1, j2, l] *= e[REF_HET_OBS_HOM, l]
                        else:  # REF is hom
                            if G[j1, j2, l] == s[0, l]:  # Equal
                                F[j1, j2, l] *= e[EQUAL_BOTH_HOM, l]
                            else:  # Unequal
                                F[j1, j2, l] *= e[UNEQUAL_BOTH_HOM, l]

        ll = np.log10(np.sum(F[:, :, l]))

    return F, c, ll


def backward_ls_dip_loop(n, m, G, s, e, c, r):
    """LS diploid backwards algoritm without vectorisation."""
    # Initialise the backward tensor
    B = np.zeros((n, n, m))
    B[:, :, m - 1] = 1
    r_n = r / n

    for l in range(m - 2, -1, -1):

        # Determine the various components
        B_no_change = np.zeros((n, n))
        B_j_change = np.zeros(n)
        B_both_change = 0

        # Evaluate the emission matrix at this site, for all pairs
        e_tmp = np.zeros((n, n))
        for j1 in range(n):
            for j2 in range(n):
                # What is the emission?
                if s[0, l + 1] == 1:
                    # OBS is het
                    if G[j1, j2, l + 1] == 1:  # REF is het
                        e_tmp[j1, j2] = e[BOTH_HET, l + 1]
                    else:  # REF is hom
                        e_tmp[j1, j2] = e[REF_HOM_OBS_HET, l + 1]
                else:
                    # OBS is hom
                    if G[j1, j2, l + 1] == 1:  # REF is het
                        e_tmp[j1, j2] = e[REF_HET_OBS_HOM, l + 1]
                    else:  # REF is hom
                        if G[j1, j2, l + 1] == s[0, l + 1]:  # Equal
                            e_tmp[j1, j2] = e[EQUAL_BOTH_HOM, l + 1]
                        else:  # Unequal
                            e_tmp[j1, j2] = e[UNEQUAL_BOTH_HOM, l + 1]

        for j1 in range(n):
            for j2 in range(n):
                B_no_change[j1, j2] = (
                    (1 - r[l + 1]) ** 2 * B[j1, j2, l + 1] * e_tmp[j1, j2]
                )
                B_j_change[j1] += (
                    (1 - r[l + 1]) * r_n[l + 1] * B[j1, j2, l + 1] * e_tmp[j1, j2]
                )
                B_both_change += r_n[l + 1] ** 2 * e_tmp[j1, j2] * B[j1, j2, l + 1]

        B[:, :, l] = B_both_change

        for j1 in range(n):
            B[:, j1, l] += B_j_change
            B[j1, :, l] += B_j_change

            for j2 in range(n):
                B[j1, j2, l] += B_no_change[j1, j2]

        B[:, :, l] *= 1 / c[l + 1]

    return B
