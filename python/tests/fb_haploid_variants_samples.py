"""Collection of functions to run forwards and backwards algorithms on haploid genotype data, where the data is structured as variants x samples."""
import numpy as np

def forwards_ls_hap(n, m, H, s, e, r, norm=True):
    """Matrix based haploid LS forward algorithm using numpy vectorisation."""
    # Initialise
    F = np.zeros((m, n))
    r_n = r / n

    if norm:

        c = np.zeros(m)
        for i in range(n):
            F[0, i] = 1 / n * e[0, np.int64(np.equal(H[0, i], s[0, 0]))]
            c[0] += F[0, i]

        for i in range(n):
            F[0, i] *= 1 / c[0]

        # Forwards pass
        for l in range(1, m):
            for i in range(n):
                F[l, i] = F[l - 1, i] * (1 - r[l]) + r_n[l]
                F[l, i] *= e[l, np.int64(np.equal(H[l, i], s[0, l]))]
                c[l] += F[l, i]

            for i in range(n):
                F[l, i] *= 1 / c[l]

        ll = np.sum(np.log10(c))

    else:

        c = np.ones(m)

        for i in range(n):
            F[0, i] = 1 / n * e[0, np.int64(np.equal(H[0, i], s[0, 0]))]

        # Forwards pass
        for l in range(1, m):
            for i in range(n):
                F[l, i] = F[l - 1, i] * (1 - r[l]) + np.sum(F[l - 1, :]) * r_n[l]
                F[l, i] *= e[l, np.int64(np.equal(H[l, i], s[0, l]))]

        ll = np.log10(np.sum(F[m - 1, :]))

    return F, c, ll

  
def backwards_ls_hap(n, m, H, s, e, c, r):
    """Matrix based haploid LS backward algorithm using numpy vectorisation."""
    # Initialise
    B = np.zeros((m, n))
    for i in range(n):
        B[m - 1, i] = 1
    r_n = r / n

    # Backwards pass
    for l in range(m - 2, -1, -1):
        tmp_B = np.zeros(n)
        tmp_B_sum = 0
        for i in range(n):
            tmp_B[i] = (
                e[l + 1, np.int64(np.equal(H[l + 1, i], s[0, l + 1]))] * B[l + 1, i]
            )
            tmp_B_sum += tmp_B[i]
        for i in range(n):
            B[l, i] = r_n[l + 1] * tmp_B_sum
            B[l, i] += (1 - r[l + 1]) * tmp_B[i]
            B[l, i] *= 1 / c[l + 1]

    return B
