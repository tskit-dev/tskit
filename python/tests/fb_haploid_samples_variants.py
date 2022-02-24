"""Collection of functions to run forwards and backwards algorithms on haploid genotype data, where the data is structured as samples x variants."""
import numpy as np


def forwards_ls_hap(n, m, H, s, e, r, norm=True):
    """Matrix based haploid LS forward algorithm using numpy vectorisation."""
    # Initialise
    F = np.zeros((n, m))
    c = np.ones(m)
    F[:, 0] = 1 / n * e[np.equal(H[:, 0], s[0, 0]).astype(np.int64), 0]
    r_n = r / n

    if norm:

        c[0] = np.sum(F[:, 0])
        F[:, 0] *= 1 / c[0]

        # Forwards pass
        for l in range(1, m):
            F[:, l] = (
                F[:, l - 1] * (1 - r[l]) + r_n[l]
            )  # Don't need to multiply by F[:,l-1] as we've normalised.
            F[:, l] *= e[np.equal(H[:, l], s[0, l]).astype(np.int64), l]
            c[l] = np.sum(F[:, l])
            F[:, l] *= 1 / c[l]

        ll = np.sum(np.log10(c))

    else:
        # Forwards pass
        for l in range(1, m):
            F[:, l] = F[:, l - 1] * (1 - r[l]) + np.sum(F[:, l - 1]) * r_n[l]
            F[:, l] *= e[np.equal(H[:, l], s[0, l]).astype(np.int64), l]

        ll = np.log10(np.sum(F[:, m - 1]))

    return F, c, ll


def backwards_ls_hap(n, m, H, s, e, c, r):
    """Matrix based haploid LS backward algorithm using numpy vectorisation."""
    # Initialise
    B = np.zeros((n, m))
    B[:, m - 1] = 1
    r_n = r / n

    # Backwards pass
    for l in range(m - 2, -1, -1):
        B_tmp = (
            e[np.equal(H[:, l + 1], s[0, l + 1]).astype(np.int64), l + 1] * B[:, l + 1]
        )
        B[:, l] = r_n[l + 1] * np.sum(B_tmp) + (1 - r[l + 1]) * B_tmp
        B[:, l] *= 1 / c[l + 1]

    return B
