import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def stability(dm_values, confidence=0.95, coverage=0.95, n_bootstrap=10000):
    """
    dm_values: your observed dm values
    confidence: 95% confidence
    coverage: want to cover 95% of future observations
    """
    percentile = coverage * 100  # 95th percentile

    bootstrap_percentiles = []
    n = len(dm_values)

    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = np.random.choice(dm_values, size=n, replace=True)
        # Compute the coverage percentile
        p = np.percentile(sample, percentile)
        bootstrap_percentiles.append(p)

    # Upper prediction bound at confidence level
    upper_bound = np.percentile(bootstrap_percentiles, confidence * 100)

    return upper_bound

def plot_stability(df_final, dm, plot_name):

    # --- Filter and preprocess ---
    df = df_final.copy()
    df['noisetype'] = df['noisetype'].fillna('N/A')
    df['dm'] = pd.to_numeric(df[dm], errors='coerce')
    df = df.dropna(subset=['dm'])
    df = df[df['noisetype'] != "N/A"]

    # Define noise types in desired order
    noise_types = ['ABSENCE', 'INSERTION', 'ORDERING', 'SUBSTITUTION', 'MIXED']
    display_names = ['Absence', 'Insertion', 'Ordering', 'Substitution', 'Mixed']
    palette = sns.color_palette("inferno", 5)
    color_dict = dict(zip(noise_types, palette))

    # --- Plot per algorithm ---
    for algo in df['algorithm'].unique():
        df_algo = df[df['algorithm'] == algo]

        # Compute epsilon per noise type
        eps_dict = {}
        for nt in df_algo['noisetype'].unique():
            sub_nt = df_algo[df_algo['noisetype'] == nt]
            eps_dict[nt] = stability(sub_nt['dm'], confidence=0.95, coverage=0.95)
            # print(algo + ", " + nt)

        plt.figure(figsize=(5, 4))

        # Boxplot with ordered x-axis
        sns.boxplot(
            data=df_algo,
            x='noisetype',
            y='dm',
            order=noise_types,  # enforce order
            palette=color_dict,
            linewidth=0.5
        )

        # Plot epsilon as red diamonds
        print(eps_dict)
        for idx, nt in enumerate(noise_types):
            if nt in eps_dict:
                plt.scatter(
                    idx, eps_dict[nt],
                    color='red',
                    marker='D',
                    s=70,
                    label='ε' if idx == 0 else ""
                )

        plt.ylim(-0.1, 1.1)
        plt.xlabel("Noise Type")
        plt.ylabel("$d_{\mathcal{M}}$")

        noise_patches = [Patch(facecolor=color_dict[nt], label=display_names[i], alpha=0.6)
                         for i, nt in enumerate(noise_types)]
        # ε marker
        epsilon_marker = [Line2D([0], [0], marker='D', color='w', label='$\\epsilon$',
                                 markerfacecolor='red', markersize=8)]

        # plt.legend(
        #     handles=noise_patches + epsilon_marker,
        #     loc='upper center',
        #     bbox_to_anchor=(0.5, 1.15),  # slightly above plot
        #     ncol=len(noise_types)+1,
        #     frameon=False
        # )

        # Replace x-axis tick labels with display names
        plt.xticks(ticks=range(len(noise_types)), labels=display_names)

        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.savefig(f"{plot_name}_{algo}.pdf", bbox_inches='tight')
        plt.close()
