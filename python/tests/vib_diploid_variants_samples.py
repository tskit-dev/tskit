"""Collection of functions to run Viterbi algorithms on dipoid genotype data, where the data is structured as variants x samples."""
import numpy as np


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


def forwards_viterbi_dip_naive(n, m, G, s, e, r):
    """Naive implementation of LS diploid Viterbi algorithm."""
    # Initialise
    V = np.zeros((m, n, n))
    P = np.zeros((m, n, n)).astype(np.int64)
    c = np.ones(m)
    r_n = r / n

    for j1 in range(n):
        for j2 in range(n):
            index_tmp = (
                4 * np.int64(np.equal(G[0, j1, j2], s[0, 0]))
                + 2 * np.int64((G[0, j1, j2] == 1))
                + np.int64(s[0, 0] == 1)
            )
            V[0, j1, j2] = 1 / (n ** 2) * e[0, index_tmp]

    for l in range(1, m):
        index = (
            4 * np.equal(G[l, :, :], s[0, l]).astype(np.int64)
            + 2 * (G[l, :, :] == 1).astype(np.int64)
            + np.int64(s[0, l] == 1)
        )
        for j1 in range(n):
            for j2 in range(n):
                # Get the vector to maximise over
                v = np.zeros((n, n))
                for k1 in range(n):
                    for k2 in range(n):
                        v[k1, k2] = V[l - 1, k1, k2]
                        if (k1 == j1) and (k2 == j2):
                            v[k1, k2] *= (
                                (1 - r[l]) ** 2 + 2 * (1 - r[l]) * r_n[l] + r_n[l] ** 2
                            )
                        elif (k1 == j1) or (k2 == j2):
                            v[k1, k2] *= r_n[l] * (1 - r[l]) + r_n[l] ** 2
                        else:
                            v[k1, k2] *= r_n[l] ** 2
                V[l, j1, j2] = np.amax(v) * e[l, index[j1, j2]]
                P[l, j1, j2] = np.argmax(v)
        c[l] = np.amax(V[l, :, :])
        V[l, :, :] *= 1 / c[l]

    ll = np.sum(np.log10(c))

    return V, P, ll


def forwards_viterbi_dip_naive_low_mem(n, m, G, s, e, r):
    """Naive implementation of LS diploid Viterbi algorithm, with reduced memory."""
    # Initialise
    V = np.zeros((n, n))
    V_previous = np.zeros((n, n))
    P = np.zeros((m, n, n)).astype(np.int64)
    c = np.ones(m)
    r_n = r / n

    for j1 in range(n):
        for j2 in range(n):
            index_tmp = (
                4 * np.int64(np.equal(G[0, j1, j2], s[0, 0]))
                + 2 * np.int64((G[0, j1, j2] == 1))
                + np.int64(s[0, 0] == 1)
            )
            V_previous[j1, j2] = 1 / (n ** 2) * e[0, index_tmp]

    # Take a look at Haploid Viterbi implementation in Jeromes code and see if we can pinch some ideas.
    # Diploid Viterbi, with smaller memory footprint.
    for l in range(1, m):
        index = (
            4 * np.equal(G[l, :, :], s[0, l]).astype(np.int64)
            + 2 * (G[l, :, :] == 1).astype(np.int64)
            + np.int64(s[0, l] == 1)
        )
        for j1 in range(n):
            for j2 in range(n):
                # Get the vector to maximise over
                v = np.zeros((n, n))
                for k1 in range(n):
                    for k2 in range(n):
                        v[k1, k2] = V_previous[k1, k2]
                        if (k1 == j1) and (k2 == j2):
                            v[k1, k2] *= (
                                (1 - r[l]) ** 2 + 2 * (1 - r[l]) * r_n[l] + r_n[l] ** 2
                            )
                        elif (k1 == j1) or (k2 == j2):
                            v[k1, k2] *= r_n[l] * (1 - r[l]) + r_n[l] ** 2
                        else:
                            v[k1, k2] *= r_n[l] ** 2
                V[j1, j2] = np.amax(v) * e[l, index[j1, j2]]
                P[l, j1, j2] = np.argmax(v)
        c[l] = np.amax(V)
        V_previous = np.copy(V) / c[l]

    ll = np.sum(np.log10(c))

    return V, P, ll


def forwards_viterbi_dip_low_mem(n, m, G, s, e, r):
    """LS diploid Viterbi algorithm, with reduced memory."""
    # Initialise
    V = np.zeros((n, n))
    V_previous = np.zeros((n, n))
    P = np.zeros((m, n, n)).astype(np.int64)
    c = np.ones(m)
    r_n = r / n

    for j1 in range(n):
        for j2 in range(n):
            index_tmp = (
                4 * np.int64(np.equal(G[0, j1, j2], s[0, 0]))
                + 2 * np.int64((G[0, j1, j2] == 1))
                + np.int64(s[0, 0] == 1)
            )
            V_previous[j1, j2] = 1 / (n ** 2) * e[0, index_tmp]

    # Diploid Viterbi, with smaller memory footprint, rescaling, and using the structure of the HMM.
    for l in range(1, m):

        index = (
            4 * np.equal(G[l, :, :], s[0, l]).astype(np.int64)
            + 2 * (G[l, :, :] == 1).astype(np.int64)
            + np.int64(s[0, l] == 1)
        )

        c[l] = np.amax(V_previous)
        argmax = np.argmax(V_previous)

        V_previous *= 1 / c[l]
        V_rowcol_max = np_amax(V_previous, 0)
        arg_rowcol_max = np_argmax(V_previous, 0)

        no_switch = (1 - r[l]) ** 2 + 2 * (r_n[l] * (1 - r[l])) + r_n[l] ** 2
        single_switch = r_n[l] * (1 - r[l]) + r_n[l] ** 2
        double_switch = r_n[l] ** 2

        j1_j2 = 0

        for j1 in range(n):
            for j2 in range(n):

                V_single_switch = max(V_rowcol_max[j1], V_rowcol_max[j2])
                P_single_switch = np.argmax(
                    np.array([V_rowcol_max[j1], V_rowcol_max[j2]])
                )

                if P_single_switch == 0:
                    template_single_switch = j1 * n + arg_rowcol_max[j1]
                else:
                    template_single_switch = arg_rowcol_max[j2] * n + j2

                V[j1, j2] = V_previous[j1, j2] * no_switch  # No switch in either
                P[l, j1, j2] = j1_j2

                # Single or double switch?
                single_switch_tmp = single_switch * V_single_switch
                if single_switch_tmp > double_switch:
                    # Then single switch is the alternative
                    if V[j1, j2] < single_switch * V_single_switch:
                        V[j1, j2] = single_switch * V_single_switch
                        P[l, j1, j2] = template_single_switch
                else:
                    # Double switch is the alternative
                    if V[j1, j2] < double_switch:
                        V[j1, j2] = double_switch
                        P[l, j1, j2] = argmax

                V[j1, j2] *= e[l, index[j1, j2]]
                j1_j2 += 1
        V_previous = np.copy(V)

    ll = np.sum(np.log10(c)) + np.log10(np.amax(V))

    return V, P, ll


def forwards_viterbi_dip_naive_vec(n, m, G, s, e, r):
    """Vectorised LS diploid Viterbi algorithm using numpy."""
    # Initialise
    V = np.zeros((m, n, n))
    P = np.zeros((m, n, n)).astype(np.int64)
    c = np.ones(m)
    r_n = r / n

    for j1 in range(n):
        for j2 in range(n):
            index_tmp = (
                4 * np.int64(np.equal(G[0, j1, j2], s[0, 0]))
                + 2 * np.int64((G[0, j1, j2] == 1))
                + np.int64(s[0, 0] == 1)
            )
            V[0, j1, j2] = 1 / (n ** 2) * e[0, index_tmp]

    # Jumped the gun - vectorising.
    for l in range(1, m):

        index = (
            4 * np.equal(G[l, :, :], s[0, l]).astype(np.int64)
            + 2 * (G[l, :, :] == 1).astype(np.int64)
            + np.int64(s[0, l] == 1)
        )

        for j1 in range(n):
            for j2 in range(n):
                v = (r_n[l] ** 2) * np.ones((n, n))
                v[j1, j2] += (1 - r[l]) ** 2
                v[j1, :] += r_n[l] * (1 - r[l])
                v[:, j2] += r_n[l] * (1 - r[l])
                v *= V[l - 1, :, :]
                V[l, j1, j2] = np.amax(v) * e[l, index[j1, j2]]
                P[l, j1, j2] = np.argmax(v)

        c[l] = np.amax(V[l, :, :])
        V[l, :, :] *= 1 / c[l]

    ll = np.sum(np.log10(c))

    return V, P, ll


def forwards_viterbi_dip_naive_full_vec(n, m, G, s, e, r):
    """Fully vectorised naive LS diploid Viterbi algorithm using numpy."""
    char_both = np.eye(n * n).ravel().reshape((n, n, n, n))
    char_col = np.tile(np.sum(np.eye(n * n).reshape((n, n, n, n)), 3), (n, 1, 1, 1))
    char_row = np.copy(char_col).T
    rows, cols = np.ogrid[:n, :n]

    # Initialise
    V = np.zeros((m, n, n))
    P = np.zeros((m, n, n)).astype(np.int64)
    c = np.ones(m)
    index = (
        4 * np.equal(G[0, :, :], s[0, 0]).astype(np.int64)
        + 2 * (G[0, :, :] == 1).astype(np.int64)
        + np.int64(s[0, 0] == 1)
    )
    V[0, :, :] = 1 / (n ** 2) * e[0, index]
    r_n = r / n

    for l in range(1, m):
        index = (
            4 * np.equal(G[l, :, :], s[0, l]).astype(np.int64)
            + 2 * (G[l, :, :] == 1).astype(np.int64)
            + np.int64(s[0, l] == 1)
        )
        v = (
            (r_n[l] ** 2)
            + (1 - r[l]) ** 2 * char_both
            + (r_n[l] * (1 - r[l])) * (char_col + char_row)
        )
        v *= V[l - 1, :, :]
        P[l, :, :] = np.argmax(v.reshape(n, n, -1), 2)  # Have to flatten to use argmax
        V[l, :, :] = v.reshape(n, n, -1)[rows, cols, P[l, :, :]] * e[l, index]
        c[l] = np.amax(V[l, :, :])
        V[l, :, :] *= 1 / c[l]

    ll = np.sum(np.log10(c))

    return V, P, ll


def backwards_viterbi_dip(m, V_last, P):
    """Run a backwards pass to determine the most likely path."""
    assert V_last.ndim == 2
    assert V_last.shape[0] == V_last.shape[1]
    # Initialisation
    path = np.zeros(m).astype(np.int64)
    path[m - 1] = np.argmax(V_last)

    # Backtrace
    for j in range(m - 2, -1, -1):
        path[j] = P[j + 1, :, :].ravel()[path[j + 1]]

    return path


def get_phased_path(n, path):
    """Obtain the phased path."""
    return np.unravel_index(path, (n, n))


def path_ll_dip(n, m, G, phased_path, s, e, r):
    """Evaluate log-likelihood path through a reference panel which results in sequence s."""
    index = (
        4 * np.int64(np.equal(G[0, phased_path[0][0], phased_path[1][0]], s[0, 0]))
        + 2 * np.int64(G[0, phased_path[0][0], phased_path[1][0]] == 1)
        + np.int64(s[0, 0] == 1)
    )
    log_prob_path = np.log10(1 / (n ** 2) * e[0, index])
    old_phase = np.array([phased_path[0][0], phased_path[1][0]])
    r_n = r / n

    for l in range(1, m):

        index = (
            4 * np.int64(np.equal(G[l, phased_path[0][l], phased_path[1][l]], s[0, l]))
            + 2 * np.int64(G[l, phased_path[0][l], phased_path[1][l]] == 1)
            + np.int64(s[0, l] == 1)
        )

        current_phase = np.array([phased_path[0][l], phased_path[1][l]])
        phase_diff = np.sum(~np.equal(current_phase, old_phase))

        if phase_diff == 0:
            log_prob_path += np.log10(
                (1 - r[l]) ** 2 + 2 * (r_n[l] * (1 - r[l])) + r_n[l] ** 2
            )
        elif phase_diff == 1:
            log_prob_path += np.log10(r_n[l] * (1 - r[l]) + r_n[l] ** 2)
        else:
            log_prob_path += np.log10(r_n[l] ** 2)

        log_prob_path += np.log10(e[l, index])
        old_phase = current_phase

    return log_prob_path
