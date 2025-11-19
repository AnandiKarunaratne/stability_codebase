import pandas as pd
import re
from typing import Dict
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
    df = pd.read_csv(results_path)
    df = df.drop_duplicates(subset=['model'], keep='last')
    df = df[['model', 'precision_clean', 'recall_clean', 'precision_noisy', 'recall_noisy']].reset_index(drop=True)
    parsed = df['model'].apply(parse_model_name)
    meta_df = pd.DataFrame(parsed.tolist()).reset_index(drop=True)
    df = pd.concat([meta_df, df[['recall_noisy']].reset_index(drop=True)], axis=1)
    df = df.dropna(subset=['system'])
    df['dm'] = 1 - df['recall_noisy']
    df = df.drop(columns=['recall_noisy'])

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

df_final = process_results("results_noisy_replay.csv", "jaccard.csv")
stability.plot_stability(df_final, "dm", "stability_f2")

