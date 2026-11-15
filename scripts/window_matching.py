"""Sparse one-to-one assignment used for directional window matching."""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import min_weight_full_bipartite_matching


def lap_succ(n, I, J, S, tau=.05):
    """local indices I->J with successor prob S; returns succ (n,) matched successor or -1."""
    m = (S > tau) & (I != J); I, J, W = I[m], J[m], np.log(S[m] / tau)              # positive gains
    # doubling: left = [0..n) windows-as-pred  + [n..2n) copies of windows-as-succ ; right = [0..n) windows-as-succ + [n..2n) copies of windows-as-pred
    rows = np.r_[I, J + n, np.arange(n), np.arange(n) + n]
    cols = np.r_[J, I + n, np.arange(n) + n, np.arange(n)]
    cost = np.r_[-W, np.zeros(len(W)), np.zeros(n), np.zeros(n)] + 1e-6          # tiny offset: explicit zeros must stay in the sparse structure
    G = sp.csr_matrix((cost + 1e3, (rows, cols)), shape=(2 * n, 2 * n))             # shift to positive (perfect matching => constant offset)
    r, c = min_weight_full_bipartite_matching(G)
    succ = np.full(n, -1); real = (r < n) & (c < n); succ[r[real]] = c[real]
    return succ
