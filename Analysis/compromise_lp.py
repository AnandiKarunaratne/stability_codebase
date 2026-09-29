"""
Compromise-score sensitivity to weighting, p=2 only, per noise type.

For each noise type, we have raw stability values eps_{i,h} for
algorithm i in {Alpha, Heuristics, Inductive} under intended function
h in {F1, F2}. We normalize by the anti-ideal (max over all
algorithms, per function per noise type) to get eta_{i,h} in [0,1],
then compute the compromise score at p=2

    C_2(i; pi) = ( pi * eta_{i,F1}^2 + (1-pi) * eta_{i,F2}^2 )^(1/2)

as pi (the weight on F1) sweeps from 0 to 1.

One standalone figure per noise type (5 total), plus one standalone
legend figure, all saved as PDF.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ---------------------------------------------------------------
# Raw data: eps_{i,h} for each noise type
# Rows: Alpha, Heuristics, Inductive
# Columns: F1, F2
# ---------------------------------------------------------------
noise_types = ['ABSENCE', 'INSERTION', 'ORDERING', 'SUBSTITUTION', 'MIXED']
noise_labels = ['Absence', 'Insertion', 'Ordering', 'Substitution', 'Mixed']

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

algorithms = ['Alpha', 'Heuristics', 'Inductive']
algo_row = {'Alpha': 0, 'Heuristics': 1, 'Inductive': 2}

algo_colors = {
    'Alpha': '#003d5c',
    'Heuristics': '#008b56',
    'Inductive': '#ffa600'
}

pis = np.linspace(0, 1, 401)


def compute_eta(eps_matrix):
    """Normalize each column by its column max (anti-ideal)."""
    anti_ideal = eps_matrix.max(axis=0)
    return eps_matrix / anti_ideal


def l2_score(eta_f1, eta_f2, pi):
    return (pi * eta_f1**2 + (1 - pi) * eta_f2**2) ** 0.5


def plot_for_noise_type(noise_type):
    eta = compute_eta(eps_data[noise_type])

    fig, ax = plt.subplots(figsize=(5, 4))

    for algo in algorithms:
        eta_f1, eta_f2 = eta[algo_row[algo]]
        scores = l2_score(eta_f1, eta_f2, pis)

        ax.plot(
            pis,
            scores,
            color=algo_colors[algo],
            linewidth=2.4
        )

    ax.set_xlabel(r"$\pi$")
    ax.set_ylabel(r"$C_2(\hat{f}_i)$")

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    fig.tight_layout()

    fname = f"weight_sensitivity_p2_{noise_type.lower()}.pdf"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)

    return fname


def create_legend():
    """Create a standalone legend PDF for the three algorithms."""

    handles = [
        Line2D(
            [0], [0],
            color=algo_colors[algo],
            linewidth=2.4,
            label=algo
        )
        for algo in algorithms
    ]

    fig, ax = plt.subplots(figsize=(4.5, 0.8))
    ax.axis("off")

    ax.legend(
        handles=handles,
        loc="center",
        ncol=3,
        frameon=False,
        fontsize=10,
        handlelength=2.5,
        columnspacing=1.8
    )

    fig.savefig(
        "weight_sensitivity_p2_legend.pdf",
        bbox_inches="tight",
        transparent=True
    )

    plt.close(fig)

    return "weight_sensitivity_p2_legend.pdf"


if __name__ == "__main__":

    for nt in noise_types:
        out = plot_for_noise_type(nt)
        print(f"saved: {out}")

    legend = create_legend()
    print(f"saved: {legend}")