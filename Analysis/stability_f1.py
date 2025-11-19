import pandas as pd
import re
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import pandas as pd
from scipy.stats import binom
import numpy as np

import backward
import stability


def basename(path):
    if pd.isna(path):
        return ""
    s = str(path).strip()
    return re.split(r"[\\/]", s)[-1]

def parse_log_name(log_name: str) -> Dict[str, str]:
    log_name = basename(log_name)
    name = log_name.replace('.xes', '')
    parts = name.split('_')

    if len(parts) == 5:
        return {
            'system': parts[0],
            'logsize': parts[1],
            'noisetype': parts[2],
            'noiselevel': float(parts[3]),
            'iterations': int(parts[4])
        }
    else: return None

def parse_model_name(model_name: str) -> Dict[str, str]:
    """Parse model filename to extract metadata"""
    model_name = basename(model_name)
    name = model_name.replace('.pnml', '')
    parts = name.split('_')

    if len(parts) == 6:
        return {
            'system': parts[0],
            'logsize': parts[1],
            'noisetype': parts[2],
            'noiselevel': float(parts[3]),
            'iterations': int(parts[4]),
            'algorithm': parts[5]
        }
    else:
        return {
            'system': parts[0],
            'logsize': parts[1],
            'noisetype': None,
            'noiselevel': 0.0,
            'iterations': None,
            'algorithm': parts[2]
        }

def process_results(results_path: str, jaccard_path: str) -> pd.DataFrame:
    df = pd.read_csv(results_path, header=None, names=['system', 'modelname', 'precision', 'recall', 'size'])
    df = df.drop_duplicates(subset=['modelname'], keep='last')
    df = df[['modelname', 'precision', 'recall']].reset_index(drop=True)
    parsed = df['modelname'].apply(parse_model_name)
    meta_df = pd.DataFrame(parsed.tolist()).reset_index(drop=True)
    df = pd.concat([meta_df, df[['precision', 'recall']].reset_index(drop=True)], axis=1)
    df = df.dropna(subset=['system'])

    # Calculate dm = 1 - F1, safely handle division by zero
    df['f1'] = 2 * df['precision'] * df['recall'] / (df['precision'] + df['recall'])
    df['f1'] = df['f1'].fillna(1)
    df['dm'] = 1 - df['f1']
    df = df.drop(columns=['f1', 'precision', 'recall'])

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

df_final = process_results("results_clean_f2.csv", "jaccard.csv")

def calculate_empirical_supremum(dm_values, dl_values):
    """
    Calculates the empirical supremum (maximum) of the stability ratio dm/dl.

    Args:
        dm_values (list or np.array): A list/array of model change values.
        dl_values (list or np.array): A list/array of log change values (perturbations).

    Returns:
        float: The maximum observed ratio (the empirical supremum).
    """

    # 1. Convert to NumPy arrays for efficient calculation
    dm = np.array(dm_values)
    dl = np.array(dl_values)

    # Check for same length
    if len(dm) != len(dl):
        raise ValueError("dm_values and dl_values must have the same length.")

    # 2. Filter out cases where dl is zero (to avoid division by zero)
    # Only consider experiments where a perturbation (dl > 0) was applied.
    valid_indices = dl > 0

    if not np.any(valid_indices):
        print("Warning: No valid experiments found where dl > 0.")
        return 0.0

    # Apply the filter to both arrays
    dm_valid = dm[valid_indices]
    dl_valid = dl[valid_indices]

    # 3. Calculate the ratio dm/dl for all valid experiments
    ratios = dm_valid / dl_valid

    # 4. Find the maximum ratio (the empirical supremum)
    empirical_sup = np.max(ratios)

    return empirical_sup

stability.plot_stability(df_final, "dm", "stability_f1")
