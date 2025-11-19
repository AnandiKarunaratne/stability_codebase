import os
import csv
import pm4py
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.log.importer.xes import importer as xes_import

def process_log(file_path, log_name, base_file_path):
    # Read log
    log = xes_import.apply(file_path)
    log = pm4py.convert_to_event_log(log)

    # === Alpha Miner ===
    a_net, a_initial_marking, a_final_marking = alpha_miner.apply(log)
    pm4py.write_pnml(a_net, a_initial_marking, a_final_marking,
                     os.path.join(base_file_path, "models", f"{os.path.splitext(log_name)[0]}_alpha.pnml"))

    # === Heuristics Miner ===
    h_net, h_initial_marking, h_final_marking = heuristics_miner.apply(log)
    pm4py.write_pnml(h_net, h_initial_marking, h_final_marking,
                     os.path.join(base_file_path, "models", f"{os.path.splitext(log_name)[0]}_heuristics.pnml"))

    # === Inductive Miner ===
    i_tree = inductive_miner.apply(log)
    i_net, i_initial_marking, i_final_marking = pm4py.convert_to_petri_net(i_tree)
    pm4py.write_pnml(i_net, i_initial_marking, i_final_marking,
                     os.path.join(base_file_path, "models", f"{os.path.splitext(log_name)[0]}_inductive.pnml"))

def main():
    systems = ["Sepsis", "RTFMS", "BPIC2012"]
    log_sizes = [1000, 2000, 4000, 10000, 20000, 40000, 100000]
    base_file_path = "/Users/anandik/Documents/ExperimentsSNIPStability"
    csv_file_path = "log_data.csv"   # randomized noisy logs

    # === Step 1: Clean logs (deterministic order, because we need this baseline to compare) ===
    for system in systems:
        for log_size in log_sizes:
            print(f"\nProcessing clean log: {system} of size {log_size}")
            clean_log_name = f"{system}_{log_size}.xes"
            process_log(base_file_path + "/logs/" + clean_log_name, clean_log_name, base_file_path)

    # === Step 2: Noisy logs (randomized order) ===
    with open(csv_file_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            system = row["system"]
            log_size = int(row["logSize"])
            noise_type = row["noiseType"]
            noise_level = float(row["noiseLevel"])
            iteration = int(row["iteration"])

            print(f"\nProcessing {system} | Iteration {iteration} | Noise: {noise_type} | Level: {noise_level}")
            noisy_log_name = f"{system}_{log_size}_{noise_type}_{noise_level}_{iteration}.xes"
            process_log(base_file_path + "/logs/" + noisy_log_name, noisy_log_name, base_file_path)

if __name__ == "__main__":
    main()