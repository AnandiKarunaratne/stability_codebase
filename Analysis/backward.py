import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportion_confint

def backward_stability_with_ci(dl_values, dm_values, epsilon_backward, delta, confidence=0.95):
    """
    Estimate P(dm <= delta | dl <= epsilon_backward) with confidence interval
    """
    nearby_mask = dl_values <= epsilon_backward
    nearby_dm = dm_values[nearby_mask]

    if len(nearby_dm) == 0:
        return None, None, None, 0

    n = len(nearby_dm)
    k = np.sum(nearby_dm <= delta)
    p_hat = k / n

    ci_low, ci_high = proportion_confint(k, n, alpha=1-confidence, method='wilson')

    print (str(p_hat) + ',' + str(ci_low) + ',' + str(ci_high))

    return p_hat, ci_low, ci_high, n


def plot_backward_stability_heatmap_old(df, output_dir='plots', delta=0.1):
    """
    Plot backward stability probabilities as heatmaps for each algorithm.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns: ['algorithm', 'noise_type', 'dl', 'dm']
    output_dir : str
        Directory to save plots
    delta : float
        Threshold for acceptable model deviation (default: 0.1)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    epsilons = [0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 1.0]

    algorithms = df['algorithm'].unique()
    noise_types = sorted(df['noisetype'].unique())

    for algorithm in algorithms:
        # Filter data for this algorithm
        alg_data = df[df['algorithm'] == algorithm]

        # Create matrix for heatmap: rows=noise_types, cols=epsilons
        prob_matrix = np.full((len(noise_types), len(epsilons)), np.nan)
        n_matrix = np.zeros((len(noise_types), len(epsilons)), dtype=int)

        for i, noise_type in enumerate(noise_types):
            noise_data = alg_data[alg_data['noisetype'] == noise_type]

            if len(noise_data) == 0:
                continue

            dl_values = noise_data['dl'].values
            dm_values = noise_data['dm'].values

            for j, eps in enumerate(epsilons):
                p, ci_low, ci_high, n = backward_stability_with_ci(
                    dl_values, dm_values, eps, delta
                )

                if p is not None:
                    prob_matrix[i, j] = p
                    n_matrix[i, j] = n

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create annotations with probabilities and sample sizes
        annot = np.empty((len(noise_types), len(epsilons)), dtype=object)
        for i in range(len(noise_types)):
            for j in range(len(epsilons)):
                if not np.isnan(prob_matrix[i, j]):
                    annot[i, j] = f'{prob_matrix[i, j]:.2f}'
                else:
                    annot[i, j] = 'N/A'

        # Plot heatmap
        sns.heatmap(prob_matrix,
                    annot=annot,
                    fmt='',
                    cmap='RdYlGn',
                    vmin=0, vmax=1,
                    cbar_kws={'label': 'Probability'},
                    xticklabels=[f'{eps:.2f}' for eps in epsilons],
                    yticklabels=noise_types,
                    linewidths=0.5,
                    ax=ax)

        ax.set_xlabel('$\\epsilon_{\\mathrm{backward}}$', fontsize=14)
        ax.set_ylabel('Noise Type', fontsize=14)
        ax.set_title(f'Backward Stability: {algorithm.capitalize()} Miner\n' +
                     f'$P(d_m \\leq {delta} \\mid d_l \\leq \\epsilon_{{\\mathrm{{backward}}}})$',
                     fontsize=16, fontweight='bold')

        plt.tight_layout()

        # Save figure
        filename = f'{output_dir}/backward_stability_heatmap_{algorithm}.pdf'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")

        plt.close()


# Alternative: Simpler annotations (just probabilities)
def plot_backward_stability_heatmap_simple(df, output_dir='plots', delta=0.1):
    """
    Plot backward stability heatmaps with simpler annotations (just probabilities).
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    epsilons = [0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 1.0]

    algorithms = df['algorithm'].unique()
    noise_types = sorted([nt for nt in df['noisetype'].unique() if nt is not None and pd.notna(nt)])

    for algorithm in algorithms:
        alg_data = df[df['algorithm'] == algorithm]

        prob_matrix = np.full((len(noise_types), len(epsilons)), np.nan)

        for i, noise_type in enumerate(noise_types):
            noise_data = alg_data[alg_data['noisetype'] == noise_type]

            if len(noise_data) == 0:
                continue

            dl_values = noise_data['dl'].values
            dm_values = noise_data['dm'].values

            for j, eps in enumerate(epsilons):
                p, _, _, _ = backward_stability_with_ci(dl_values, dm_values, eps, delta)
                if p is not None:
                    prob_matrix[i, j] = p

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))

        sns.heatmap(prob_matrix,
                    annot=True,
                    fmt='.2f',
                    cmap='RdYlGn',
                    vmin=0, vmax=1,
                    cbar_kws={'label': 'Probability'},
                    xticklabels=[f'{eps:.2f}' for eps in epsilons],
                    yticklabels=noise_types,
                    linewidths=0.5,
                    ax=ax)

        ax.set_xlabel('$\\epsilon_{\\mathrm{backward}}$', fontsize=14)
        ax.set_ylabel('Noise Type', fontsize=14)
        ax.set_title(f'Backward Stability: {algorithm.capitalize()} Miner\n' +
                     f'$P(d_m \\leq {delta} \\mid d_l \\leq \\epsilon_{{\\mathrm{{backward}}}})$',
                     fontsize=16, fontweight='bold')

        plt.tight_layout()

        filename = f'{output_dir}/backward_stability_heatmap_{algorithm}.pdf'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")

        plt.close()

def plot_backward_stability_heatmap(df, output_dir='plots', delta=0.1):
    """
    Plot backward stability probabilities as heatmaps for each algorithm.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns: ['algorithm', 'noisetype', 'dl', 'dm']
    output_dir : str
        Directory to save plots
    delta : float
        Threshold for acceptable model deviation (default: 0.1)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    epsilons = [0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 1.0]

    algorithms = df['algorithm'].unique()
    df['dm'] = pd.to_numeric(df['d_struct'], errors='coerce')

    # Fixed noise types in order
    noise_types = ['ABSENCE', 'INSERTION', 'ORDERING', 'SUBSTITUTION', 'MIXED']
    noise_labels = ['Absence', 'Insertion', 'Ordering', 'Substitution', 'Mixed']

    for algorithm in algorithms:
        # Filter data for this algorithm (exclude None noise types)
        alg_data = df[(df['algorithm'] == algorithm) & (df['noisetype'].notna())]

        # Create matrix for heatmap: rows=epsilons, cols=noise_types (swapped!)
        prob_matrix = np.full((len(epsilons), len(noise_types)), np.nan)
        n_matrix = np.zeros((len(epsilons), len(noise_types)), dtype=int)

        for j, noise_type in enumerate(noise_types):
            noise_data = alg_data[alg_data['noisetype'] == noise_type]

            if len(noise_data) == 0:
                continue

            dl_values = noise_data['dl'].values
            dm_values = noise_data['dm'].values

            for i, eps in enumerate(epsilons):
                p, ci_low, ci_high, n = backward_stability_with_ci(
                    dl_values, dm_values, eps, delta
                )

                if p is not None:
                    prob_matrix[i, j] = p
                    n_matrix[i, j] = n

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create annotations with probabilities and sample sizes
        annot = np.empty((len(epsilons), len(noise_types)), dtype=object)
        for i in range(len(epsilons)):
            for j in range(len(noise_types)):
                if not np.isnan(prob_matrix[i, j]):
                    annot[i, j] = f'{prob_matrix[i, j]:.2f}'
                else:
                    annot[i, j] = 'N/A'

        # Plot heatmap (red to blue colormap)
        sns.heatmap(prob_matrix,
                    annot=annot,
                    fmt='',
                    cmap='RdBu',  # Red (low) to Blue (high)
                    vmin=0, vmax=1,
                    cbar_kws={'label': 'Probability'},
                    xticklabels=noise_labels,
                    yticklabels=[f'{eps:.2f}' for eps in epsilons],
                    linewidths=0.5,
                    ax=ax)

        ax.invert_yaxis()
        ax.set_xlabel('Noise Type', fontsize=14)
        ax.set_ylabel('$\epsilon_{backward}$', fontsize=14)
        # ax.set_title(f'Backward Stability: {algorithm.capitalize()} Miner\n' +
        #              f'$P(d_m \\leq {delta} \\mid d_l \\leq \\epsilon_{{\\mathrm{{backward}}}})$',
        #              fontsize=16, fontweight='bold')

        plt.tight_layout()

        # Save figure
        filename = f'{output_dir}/backward_stability_heatmap_{algorithm}.pdf'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")

        plt.close()


# Example usage:
# df = pd.DataFrame({
#     'algorithm': ['alpha', 'alpha', 'heuristics', ...],
#     'noise_type': ['ABSENCE', 'INSERTION', 'ORDERING', ...],
#     'dl': [0.05, 0.12, 0.03, ...],
#     'dm': [0.2, 0.5, 0.08, ...]
# })
#
# # With sample sizes in annotations
# plot_backward_stability_heatmap(df, output_dir='plots', delta=0.1)
#
# # Simpler version (just probabilities)
# plot_backward_stability_heatmap_simple(df, output_dir='plots', delta=0.1)