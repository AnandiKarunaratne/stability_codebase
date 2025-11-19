import ast

from typing import Dict
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_model_name(model_name: str) -> Dict[str, str]:
    """Parse model filename to extract metadata"""
    name = model_name.replace('.pnml', '')
    parts = name.split('_')

    if len(parts) >= 6:
        return {
            'system': parts[0],
            'logsize': parts[1],
            'noisetype': parts[2],
            'noiselevel': float(parts[3]),
            'iterations': int(parts[4]),
            'algorithm': parts[5]
        }
    else:
        raise ValueError(f"Cannot parse model name: {model_name}")


def parse_similarity_dict(sim_str: str) -> Dict[str, float]:
    """Parse similarity dictionary string"""
    try:
        return ast.literal_eval(sim_str)
    except:
        raise ValueError(f"Cannot parse similarity string: {sim_str}")


def calculate_structural_distance(row) -> float:
    """Calculate d_struct from Jaccard similarities"""
    d_places = 1 - row['place_similarity']
    d_transitions = 1 - row['transition_similarity']
    d_arcs = 1 - row['arc_similarity']
    return (d_places + d_transitions + d_arcs) / 3


def calculate_behavioral_distance(row) -> float:
    """Calculate d_behav from TAR similarity"""
    return 1 - row['tar_similarity']


def load_and_process_data(csv_path: str) -> pd.DataFrame:
    """Load CSV and add calculated columns"""
    # This shows a certain model's structural and behavioral similarity between its clean counterpart
    df = pd.read_csv(csv_path, header=None, names=['model_name', 'structural_sim', 'tar_similarity'])

    # Parse model names
    parsed = df['model_name'].apply(parse_model_name)
    df = pd.concat([df, pd.DataFrame(parsed.tolist())], axis=1)

    # Parse similarity dictionaries
    sim_dicts = df['structural_sim'].apply(parse_similarity_dict)
    df['place_similarity'] = sim_dicts.apply(lambda x: x.get('places', 0))
    df['transition_similarity'] = sim_dicts.apply(lambda x: x.get('transitions', 0))
    df['arc_similarity'] = sim_dicts.apply(lambda x: x.get('arcs', 0))

    # Calculate distances
    df['d_struct'] = df.apply(calculate_structural_distance, axis=1)
    df['d_behav'] = df.apply(calculate_behavioral_distance, axis=1)

    return df

def process_results(results_path: str, jaccard_path: str) -> pd.DataFrame:
    df = load_and_process_data(results_path)
    df = df.drop_duplicates(subset=['model_name'], keep='last')
    parsed = df['model_name'].apply(parse_model_name)
    meta_df = pd.DataFrame(parsed.tolist()).reset_index(drop=True)
    df = pd.concat([meta_df, df[['d_struct', 'd_behav']].reset_index(drop=True)], axis=1)

    # Construct logname for matching
    def make_logname(row):
        if row['noisetype'] is None or row['iterations'] is None:
            return f"{row['system']}_{row['logsize']}.xes"
        else:
            return f"{row['system']}_{row['logsize']}_{row['noisetype']}_{row['noiselevel']}_{int(row['iterations'])}.xes"

    df['logname'] = df.apply(make_logname, axis=1)

    # Read jaccard.csv (no headers)
    jaccard_df = pd.read_csv(jaccard_path, header=None, names=['logname', 'sl'])
    jaccard_df = jaccard_df.drop_duplicates(subset=['logname'], keep='last')
    jaccard_df['dl'] = 1 - jaccard_df['sl']

    df = df.merge(jaccard_df[['logname', 'dl']], on='logname', how='left')
    df['dl'] = df['dl'].fillna(0)
    df = df.drop(columns=['logname'])
    return df

df_final = process_results("results_similarity.csv", "jaccard.csv")

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
    df_filtered['kappa'] = df_filtered['d_struct'] / df_filtered['dl']

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
    df = df[(df['dl'] > 0) & np.isfinite(df['d_struct']) & np.isfinite(df['dl'])]
    df['kappa'] = df['d_struct'] / df['dl']
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
