"""
Mean and max absolute difference in L2 compromise score between
Heuristics Miner and Inductive Miner, across the full weighting range
pi in [0,1], for each noise type.

For each noise type, eps_{i,h} gives the raw stability value for
algorithm i on intended function h in {F1 (rediscoverability),
F2 (behavior replay)}. We normalize by the anti-ideal (max over all
algorithms, per function per noise type) to get eta_{i,h} in [0,1],
then compute

    L_2(i; pi) = ( pi * eta_{i,F1}^2 + (1-pi) * eta_{i,F2}^2 )^(1/2)

as pi sweeps from 0 to 1, and report:
    - mean_abs_diff = mean_pi | L_2(Heuristics; pi) - L_2(Inductive; pi) |
    - max_abs_diff  = max_pi  | L_2(Heuristics; pi) - L_2(Inductive; pi) |
"""

import numpy as np

# ---------------------------------------------------------------
# Raw data: eps_{i,h} for each noise type
# Rows: Alpha, Heuristics, Inductive
# Columns: F1, F2
# ---------------------------------------------------------------
noise_types = ['ABSENCE', 'INSERTION', 'ORDERING', 'SUBSTITUTION', 'MIXED']

eps_data = {
    'ABSENCE':      np.array([[0.9392612271939864, 0.7572929367661907],
                              [0.5757544504778505, 0.11580801354593395],
                              [0.829591532001652,  0.03247236858551317]]),
    'INSERTION':    np.array([[1.0,                0.8918783578920478],
                              [0.9885604216542798, 0.10772117540682438],
                              [0.994146577631536,  0.007255584711392904]]),
    'ORDERING':     np.array([[1.0,                0.876316851366872],
                              [0.5818937326256086, 0.11595621230311481],
                              [0.8359199369068672, 0.031964143200801806]]),
    'SUBSTITUTION': np.array([[0.9994920657591844, 0.8989617351554324],
                              [0.9902495930679012, 0.11304745329135657],
                              [0.9948541465872376, 0.025177970057192578]]),
    'MIXED':        np.array([[1.0,                0.8980605796312059],
                              [0.9787784138127394, 0.10743838341152083],
                              [0.9898331838304554, 0.008882353862329957]]),
}
algo_row = {'Alpha': 0, 'Heuristics': 1, 'Inductive': 2}

# Fine grid over pi in [0, 1]. 401 points -> step size 0.0025.
pis = np.linspace(0, 1, 401)


def compute_eta(eps_matrix):
    """Normalize each column by its column max (anti-ideal), over ALL algorithms."""
    anti_ideal = eps_matrix.max(axis=0)
    return eps_matrix / anti_ideal


def l2_score(eta_f1, eta_f2, pi):
    return (pi * eta_f1**2 + (1 - pi) * eta_f2**2) ** 0.5


def main():
    header = f"{'Noise':<14}{'Mean |Diff|':>14}{'Max |Diff|':>14}"
    print(header)
    print("-" * len(header))

    for noise in noise_types:
        eta = compute_eta(eps_data[noise])
        eta_f1_h, eta_f2_h = eta[algo_row['Heuristics']]
        eta_f1_i, eta_f2_i = eta[algo_row['Inductive']]

        scores_h = l2_score(eta_f1_h, eta_f2_h, pis)
        scores_i = l2_score(eta_f1_i, eta_f2_i, pis)

        diff = np.abs(scores_h - scores_i)
        mean_abs_diff = diff.mean()
        max_abs_diff = diff.max()

        print(f"{noise.capitalize():<14}{mean_abs_diff:>14.4f}{max_abs_diff:>14.4f}")


if __name__ == "__main__":
    main()