import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np

def basename(path):
    if pd.isna(path):
        return ""
    s = str(path).strip()
    return re.split(r"[\\/]", s)[-1]

clean = pd.read_csv("results_clean_replay.csv")
clean["model"] = clean["model"].apply(basename)
clean = clean.drop_duplicates(subset=['model'], keep='last')
tmp = clean["model"].str.replace(".pnml", "", regex=False)
parts = tmp.str.rsplit("_", n=2, expand=True)
parts.columns = ["system", "logsize", "algorithm"]
clean_data_df = pd.concat([parts, clean["simplicity"]], axis=1)

noisy = pd.read_csv("results_noisy_replay.csv")
noisy["model"] = noisy["model"].apply(basename)
noisy = noisy.drop_duplicates(subset=['model'], keep='last')
tmp = noisy["model"].str.replace(".pnml", "", regex=False)
parts = tmp.str.rsplit("_", n=6, expand=True)
parts.columns = ["system", "logsize", "noisetype","noiselevel", "iteration", "algorithm"]
noisy_data_df = pd.concat([parts,noisy["simplicity"]], axis=1)

# Load jaccard data and create logname for merging
jaccard = pd.read_csv("jaccard.csv", header=None, names=["logname", "sl"])
jaccard = jaccard.drop_duplicates(subset=['logname'], keep='last')
jaccard['dl'] = 1 - jaccard['sl']

# Create logname in noisy_data_df before merging
noisy_data_df["logname"] = (
        noisy_data_df["system"].astype(str) + "_" +
        noisy_data_df["logsize"].astype(str) + "_" +
        noisy_data_df["noisetype"].astype(str) + "_" +
        noisy_data_df["noiselevel"].astype(str) + "_" +
        noisy_data_df["iteration"].astype(str) + ".xes"
)

# Add dl to noisy_data_df
noisy_data_df = noisy_data_df.merge(jaccard, on="logname", how="left")

# Now merge with clean data
merged = noisy_data_df.merge(
    clean_data_df,
    on=["system", "logsize", "algorithm"],
    suffixes=("_noisy", "_clean")
)

merged["dm"] = abs(merged["simplicity_noisy"] - merged["simplicity_clean"]) / merged["simplicity_clean"]
df_final = merged

def assess_conditioning(df, validation_split=0.3, random_state=42):
    """
    Assess algorithmic conditioning from empirical measurements.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns 'dm' (model distance) and 'dl' (log distance)
    validation_split : float
        Fraction of data to hold out for validation
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    dict with conditioning estimates and diagnostics
    """

    # Remove rows where dl is zero (no perturbation)
    df_filtered = df[df['dl'] > 0].copy()

    # Calculate local conditioning numbers
    df_filtered['kappa'] = df_filtered['dm'] / df_filtered['dl']

    # Split into estimation and validation sets
    df_est, df_val = train_test_split(
        df_filtered,
        test_size=validation_split,
        random_state=random_state
    )

    # Step 4: Aggregation on estimation set
    kappa_hat = df_est['kappa'].max()
    kappa_50 = df_est['kappa'].quantile(0.50)
    kappa_90 = df_est['kappa'].quantile(0.90)
    kappa_95 = df_est['kappa'].quantile(0.95)

    # Check if conditioning is input-dependent
    stability_ratio = kappa_95 / kappa_50 if kappa_50 > 0 else np.inf
    is_stable = stability_ratio <= 10

    # Step 5: Validation
    kappa_hat_val = df_val['kappa'].max()
    relative_error = abs(kappa_hat_val - kappa_hat) / kappa_hat if kappa_hat > 0 else np.inf
    validation_passed = relative_error < 0.5

    # Interpretation
    if kappa_hat < 2:
        interpretation = "well-conditioned"
    elif kappa_hat <= 10:
        interpretation = "moderately conditioned"
    else:
        interpretation = "ill-conditioned"

    # Compile results
    results = {
        # Primary estimate
        'kappa_hat': kappa_hat,

        # Distribution statistics
        'kappa_median': kappa_50,
        'kappa_90': kappa_90,
        'kappa_95': kappa_95,
        'stability_ratio': stability_ratio,
        'is_input_stable': is_stable,

        # Validation
        'kappa_hat_validation': kappa_hat_val,
        'relative_error': relative_error,
        'validation_passed': validation_passed,

        # Interpretation
        'interpretation': interpretation,

        # Sample sizes
        'n_estimation': len(df_est),
        'n_validation': len(df_val),
    }

    return results


def print_conditioning_report(results):
    """Pretty print conditioning assessment results."""
    print("=" * 60)
    print("ALGORITHMIC CONDITIONING ASSESSMENT")
    print("=" * 60)

    print(f"\n📊 STEP 4: AGGREGATION")
    print(f"  Conditioning estimate (κ̂): {results['kappa_hat']:.3f}")
    print(f"  Median conditioning (κ₅₀): {results['kappa_median']:.3f}")
    print(f"  90th percentile (κ₉₀):    {results['kappa_90']:.3f}")
    print(f"  95th percentile (κ₉₅):    {results['kappa_95']:.3f}")
    print(f"  Stability ratio (κ₉₅/κ₅₀): {results['stability_ratio']:.2f}")

    if results['is_input_stable']:
        print(f"  ✓ Conditioning is INPUT-STABLE (ratio ≤ 10)")
        print(f"    → Report: κ̂ = {results['kappa_hat']:.3f}")
    else:
        print(f"  ✗ Conditioning is INPUT-DEPENDENT (ratio > 10)")
        print(f"    → Report range: κ₅₀ = {results['kappa_median']:.3f}, κ̂ = {results['kappa_hat']:.3f}")

    print(f"\n🔬 STEP 5: VALIDATION")
    print(f"  Validation estimate:  {results['kappa_hat_validation']:.3f}")
    print(f"  Relative error:       {results['relative_error']:.2%}")
    print(f"  Sample sizes:         n_est={results['n_estimation']}, n_val={results['n_validation']}")

    if results['validation_passed']:
        print(f"  ✓ VALIDATION PASSED (error < 50%)")
    else:
        print(f"  ✗ VALIDATION FAILED (error ≥ 50%)")
        print(f"    → Expand log diversity and re-assess")

    print(f"\n📋 INTERPRETATION")
    print(f"  Algorithm is: {results['interpretation'].upper()}")
    print("=" * 60)

def plot_conditioning_by_noise_type(df, save_path="."):
    """
    Plot local conditioning numbers (kappa vs dl) for different noise types and algorithms.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: 'dl', 'dm', 'noisetype', 'algorithm'
    save_path : str, optional
        If provided, saves plots to this directory
    """

    # Calculate kappa
    df = df.copy()
    df = df[(df['dl'] > 0) & np.isfinite(df['dm']) & np.isfinite(df['dl'])]
    df['kappa'] = df['dm'] / df['dl']
    df = df[np.isfinite(df['kappa'])]

    # Get unique algorithms
    algorithms = sorted(df['algorithm'].unique())

    # Define noise types in specific order
    noise_types = ['ABSENCE', 'INSERTION', 'ORDERING', 'SUBSTITUTION', 'MIXED']
    noise_labels = ['Absence', 'Insertion', 'Ordering', 'Substitution', 'Mixed']

    # Use viridis colormap
    cmap = sns.color_palette("inferno", 5)
    colors = {noise: cmap[i] for i, noise in enumerate(noise_types)}

    # Create a separate plot for each algorithm
    for algo in algorithms:
        fig, ax = plt.subplots(figsize=(4, 4))

        df_algo = df[df['algorithm'] == algo]

        # Plot each noise type in specified order
        for noise, label in zip(noise_types, noise_labels):
            df_noise = df_algo[df_algo['noisetype'] == noise]

            if len(df_noise) > 0:
                # Sort by dl for better visualization
                df_noise = df_noise.sort_values('dl')

                ax.scatter(df_noise['dl'], df_noise['kappa'],
                           alpha=0.2, s=30, label=label, color=colors[noise])

        ax.set_xlabel('Log Distance ($d_{\mathcal{L}}$)', fontsize=12)
        ax.set_ylabel('Conditioning Number ($\kappa$)', fontsize=12)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # Legend at top outside, horizontal
        # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
        #           ncol=len(noise_types), frameon=False)

        plt.tight_layout()

        if save_path:
            plt.savefig(f"{save_path}/{algo}_conditioning.pdf",
                        dpi=300, bbox_inches='tight')

        plt.show()

for algo in df_final['algorithm'].unique():
    for noise_type in df_final['noisetype'].unique():
        df_intermediate = df_final[df_final['algorithm']==algo]
        if noise_type!=None:
            print(algo + "," + noise_type)
            df_intermediate = df_intermediate[df_intermediate['noisetype']==noise_type]
            print_conditioning_report(assess_conditioning(df_intermediate))
plot_conditioning_by_noise_type(df_final)

# Add this at the end of your script, before plot_conditioning_by_noise_type

# Calculate kappa for the final dataframe
df_export = df_final[df_final['dl'] > 0].copy()
df_export['kappa'] = df_export['dm'] / df_export['dl']

# Select relevant columns for analysis
df_export = df_export[['algorithm', 'noisetype', 'dl', 'dm', 'kappa', 'system', 'logsize', 'noiselevel']]

# Sort for easier analysis
df_export = df_export.sort_values(['algorithm', 'noisetype', 'dl'])

# Export to CSV
df_export.to_csv('conditioning_data.csv', index=False)
print(f"\n✓ Exported {len(df_export)} rows to conditioning_data.csv")

# Also create a summary showing max kappa at different dl ranges
summary_rows = []
for algo in df_export['algorithm'].unique():
    for noise in df_export['noisetype'].unique():
        df_subset = df_export[(df_export['algorithm'] == algo) & (df_export['noisetype'] == noise)]
        if len(df_subset) > 0:
            # Bin dl into ranges
            for dl_min, dl_max in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
                df_range = df_subset[(df_subset['dl'] >= dl_min) & (df_subset['dl'] < dl_max)]
                if len(df_range) > 0:
                    summary_rows.append({
                        'algorithm': algo,
                        'noisetype': noise,
                        'dl_range': f'{dl_min}-{dl_max}',
                        'n_points': len(df_range),
                        'max_kappa': df_range['kappa'].max(),
                        'median_kappa': df_range['kappa'].median(),
                        'mean_kappa': df_range['kappa'].mean()
                    })

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv('conditioning_summary_by_dl_range.csv', index=False)
print(f"✓ Exported summary to conditioning_summary_by_dl_range.csv")

plot_conditioning_by_noise_type(df_final)
