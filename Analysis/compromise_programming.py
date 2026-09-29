"""
Compromise-score weight sensitivity: difference plot.

For each noise type, we plot

    Delta(pi) = L_p(Inductive ; pi) - L_p(Heuristics ; pi)

as pi (weight on Intended Function 1) sweeps from 0 to 1. A zero
crossing of Delta(pi) is exactly the weight at which the preferred
algorithm switches: Delta < 0 means Inductive is preferred (lower
score is better), Delta > 0 means Heuristics is preferred.

Alpha is excluded: it is the anti-ideal (worst) algorithm on every
single (intended function, noise type) pair, so eta_Alpha = 1
everywhere and L_p(Alpha; pi) = 1 for all p and all pi.

One figure per p value (p=1, p=2, p=inf), each showing all 5 noise
types (inferno palette) as a single Delta(pi) line.

The legend is saved separately as compromise_score_legend.pdf.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

algo_row = {'Alpha': 0, 'Heuristics': 1, 'Inductive': 2}

cmap = sns.color_palette("inferno", 5)
colors = {noise: cmap[i] for i, noise in enumerate(noise_types)}

pis = np.linspace(0, 1, 401)


def compute_eta(eps_matrix):
    anti_ideal = eps_matrix.max(axis=0)
    return eps_matrix / anti_ideal


def lp_score(eta_f1, eta_f2, pi, p):
    if p == np.inf:
        return np.full_like(
            np.asarray(pi, dtype=float),
            max(eta_f1, eta_f2)
        )

    return (
                   pi * eta_f1**p +
                   (1 - pi) * eta_f2**p
           ) ** (1.0 / p)


def find_crossing(pi_arr, delta_arr):
    """Return pi at first sign change of delta, or None if no crossing."""
    sign_changes = np.where(np.diff(np.sign(delta_arr)) != 0)[0]

    if len(sign_changes) == 0:
        return None

    i = sign_changes[0]

    x0, x1 = pi_arr[i], pi_arr[i + 1]
    y0, y1 = delta_arr[i], delta_arr[i + 1]

    return x0 - y0 * (x1 - x0) / (y1 - y0)


def compute_all_deltas():
    """Precompute delta(pi) for every noise type and p."""
    all_deltas = {}

    for p in [1, 2, np.inf]:
        for noise in noise_types:
            eta = compute_eta(eps_data[noise])

            eta_f1_h, eta_f2_h = eta[algo_row['Heuristics']]
            eta_f1_i, eta_f2_i = eta[algo_row['Inductive']]

            scores_h = lp_score(
                eta_f1_h, eta_f2_h, pis, p
            )
            scores_i = lp_score(
                eta_f1_i, eta_f2_i, pis, p
            )

            all_deltas[(p, noise)] = scores_i - scores_h

    return all_deltas


def make_plot(p, fname, all_deltas, y_range):
    fig, ax = plt.subplots(figsize=(5, 4))

    for noise, label in zip(noise_types, noise_labels):
        color = colors[noise]
        delta = all_deltas[(p, noise)]

        ax.plot(
            pis,
            delta,
            color=color,
            linewidth=2.4,
            alpha=0.8
        )

        crossing = find_crossing(pis, delta)

        if crossing is not None:
            ax.plot(
                crossing,
                0,
                'o',
                color=color,
                markersize=7,
                markeredgecolor='black',
                markeredgewidth=0.8,
                zorder=5,
                alpha=0.8
            )

    ax.axhline(
        0,
        color='black',
        linewidth=1,
        linestyle='-',
        alpha=0.8
    )

    ax.set_xlabel(r"$\pi$")
    ax.set_ylabel(
        r"$\Delta = C_p(\mathrm{Inductive}) - C_p(\mathrm{Heuristics})$"
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(y_range)
    ax.grid(alpha=0.3)

    # No legend in the individual figures.

    fig.tight_layout()

    fig.savefig(
        fname,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


def make_legend(fname):
    """Save the noise-type legend as a separate horizontal PDF."""

    fig, ax = plt.subplots(figsize=(7, 0.6))
    ax.axis("off")

    handles = [
        Line2D(
            [0],
            [0],
            color=colors[noise],
            linewidth=2.4,
            label=label
        )
        for noise, label in zip(noise_types, noise_labels)
    ]

    ax.legend(
        handles=handles,
        loc="center",
        ncol=5,
        frameon=False,
        fontsize=9,
        handlelength=2.4,
        handletextpad=0.5,
        columnspacing=1.2,
        borderaxespad=0
    )

    fig.savefig(
        fname,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05
    )

    plt.close(fig)


if __name__ == "__main__":
    all_deltas = compute_all_deltas()

    y_min = min(d.min() for d in all_deltas.values())
    y_max = max(d.max() for d in all_deltas.values())

    pad = 0.05 * (y_max - y_min)
    y_range = (y_min - pad, y_max + pad)

    make_plot(
        1,
        "crossover_p1.pdf",
        all_deltas,
        y_range
    )

    make_plot(
        2,
        "crossover_p2.pdf",
        all_deltas,
        y_range
    )

    make_plot(
        np.inf,
        "crossover_pinf.pdf",
        all_deltas,
        y_range
    )

    make_legend("compromise_score_legend.pdf")

    print("done")