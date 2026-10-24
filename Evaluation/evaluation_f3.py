import os
import csv
from functools import lru_cache

from behavioral_profile import behavioral_profile_from_pnml, behavioral_profile_from_xes
from similarity_metric import m3_metric


@lru_cache(maxsize=None)
def get_model_profile(pnml_path):
    """Parse a model + compute its behavioral profile once, then reuse it
    for every comparison that references this path. This matters a lot
    here: the same clean model gets compared against every noise
    level/iteration for that system+size+algo, and the same system log
    gets compared against every discovered model -- without caching, each
    of those re-parses the file and rebuilds its reachability graph /
    rescans its traces from scratch every single time."""
    return behavioral_profile_from_pnml(pnml_path)


@lru_cache(maxsize=None)
def get_log_profile(xes_path):
    return behavioral_profile_from_xes(xes_path)


def write_to_csv(row, filename):
    """Append a row to filename, writing a header the first time.
    NOTE: if you already have a write_to_csv helper elsewhere in your
    project, delete this and just import that one instead -- this is
    only here so the script runs standalone."""
    file_exists = os.path.isfile(filename)
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["item", "system", "log_size", "algo", "distance", "sim_+", "sim_~>'"])
        writer.writerow(row)


def write_failure(row, filename):
    file_exists = os.path.isfile(filename)
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["item", "system", "log_size", "algo", "error"])
        writer.writerow(row)


def eval_log_vs_system(model_path, system_log_path, model_name, system, log_size, algo,
                        out_csv="log_vs_system.csv", err_csv="log_vs_system_failures.csv"):
    """Task 1: distance between a discovered model and the original,
    full system log it's meant to represent."""
    try:
        profile_log = get_log_profile(system_log_path)
        profile_model = get_model_profile(model_path)
        distance, sims = m3_metric(profile_log, profile_model)
        write_to_csv([model_name, system, log_size, algo, distance,
                      sims.get("+"), sims.get("~>'")], out_csv)
        print(f"\u2713 {model_name} vs {os.path.basename(system_log_path)} \u2192 {distance:.4f}")
    except Exception as e:
        write_failure([model_name, system, log_size, algo, str(e)], err_csv)
        print(f"\u2717 {model_name} vs system log \u2192 ERROR: {e}")


def eval_noisy_vs_clean(noisy_model_path, clean_model_path, model_name, system, log_size, algo,
                         out_csv="noisy_vs_clean.csv", err_csv="noisy_vs_clean_failures.csv"):
    """Task 2: distance between a noisy-log-discovered model and its
    same-system/size/algo clean-log-discovered counterpart."""
    try:
        profile_clean = get_model_profile(clean_model_path)
        profile_noisy = get_model_profile(noisy_model_path)
        distance, sims = m3_metric(profile_clean, profile_noisy)
        write_to_csv([model_name, system, log_size, algo, distance,
                      sims.get("+"), sims.get("~>'")], out_csv)
        print(f"\u2713 {model_name} vs clean baseline \u2192 {distance:.4f}")
    except Exception as e:
        write_failure([model_name, system, log_size, algo, str(e)], err_csv)
        print(f"\u2717 {model_name} vs clean baseline \u2192 ERROR: {e}")


def main():
    systems = ["Sepsis", "RTFMS", "BPIC2012"]
    log_sizes = [1000, 2000, 4000, 10000, 20000, 40000, 100000]
    base_file_path = r"C:\projects\python\Anda\resources"
    csv_file_path = "log_data copy 3.csv"
    algos = ["alpha", "heuristics", "inductive"]

    # Original, full-size system logs -- e.g. .../logs/Sepsis.xes.
    # Update this mapping if your ground-truth logs are named differently.
    system_log_paths = {
        system: os.path.join(base_file_path, "logs", f"{system}.xes")
        for system in systems
    }
    for system, path in system_log_paths.items():
        if not os.path.exists(path):
            print(f"WARNING: original system log not found: {path}")

    # === Task 1a: clean-log-discovered models vs. original system log ===
    for system in systems:
        system_log_path = system_log_paths[system]
        for log_size in log_sizes:
            for algo in algos:
                clean_model_name = f"{system}_{log_size}_{algo}.pnml"
                clean_model_path = os.path.join(base_file_path, "models", clean_model_name)
                if not os.path.exists(clean_model_path):
                    print(f"missing: {clean_model_path}")
                    continue
                eval_log_vs_system(clean_model_path, system_log_path,
                                    clean_model_name, system, log_size, algo)

    # === Task 1b + Task 2: noisy-log-discovered models ===
    with open(csv_file_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            system = row["system"]
            log_size = int(row["logSize"])
            noise_type = row["noiseType"]
            noise_level = float(row["noiseLevel"])
            iteration = int(row["iteration"])
            system_log_path = system_log_paths[system]

            for algo in algos:
                noisy_model_name = f"{system}_{log_size}_{noise_type}_{noise_level}_{iteration}_{algo}.pnml"
                noisy_model_path = os.path.join(base_file_path, "models", noisy_model_name)
                clean_model_name = f"{system}_{log_size}_{algo}.pnml"
                clean_model_path = os.path.join(base_file_path, "models", clean_model_name)

                if not os.path.exists(noisy_model_path):
                    print(f"missing: {noisy_model_path}")
                    continue

                # Task 1b: noisy model vs. original system log
                eval_log_vs_system(noisy_model_path, system_log_path,
                                    noisy_model_name, system, log_size, algo)

                # Task 2: noisy model vs. corresponding clean model
                if not os.path.exists(clean_model_path):
                    print(f"missing clean baseline: {clean_model_path}")
                    continue
                eval_noisy_vs_clean(noisy_model_path, clean_model_path,
                                     noisy_model_name, system, log_size, algo)


if __name__ == "__main__":
    main()
    