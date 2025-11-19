import os
import csv
import pm4py
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.log.importer.xes import importer as xes_import

def calculate_simplicity(net, initial_marking=None, final_marking=None):
    # Count places and transitions
    num_places = len(net.places)
    num_transitions = len(net.transitions)
    total_nodes = num_places + num_transitions

    return total_nodes

def write_to_csv(row, results_file="done1.csv"):
    file_exists = os.path.isfile(results_file)
    with open(results_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["model"])
        writer.writerow(row)

def process_log(file_path, log_name, system, log_size, noise_type, noise_level, iteration, output_csv, base_file_path, noisy=True):
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
    base_file_path =  r"C:\projects\python\Anda\resources"
    csv_file_path = "log_data.csv"   # randomized noisy logs

    # === Step 1: Clean logs (deterministic order) ===
    for system in systems:
        for log_size in log_sizes:
            print(f"\nProcessing clean log: {system} of size {log_size}")
            clean_log_name = f"{system}_{log_size}.xes"
            clean_log_path = os.path.join(base_file_path, "logs", clean_log_name)
            process_log(clean_log_path, clean_log_name, system, log_size, "n/a", "n/a", "n/a", "results.csv", base_file_path, False)

    # === Step 2: Noisy logs (randomized order) ===
    with open(csv_file_path, newline="") as f:
        reader = csv.DictReader(f)
        row=[]
        for row in reader:
            system = row["system"]
            log_size = int(row["logSize"])
            noise_type = row["noiseType"]
            noise_level = float(row["noiseLevel"])
            iteration = int(row["iteration"])

            print(f"\nProcessing {system} | Iteration {iteration} | Noise: {noise_type} | Level: {noise_level}")
            noisy_log_name = f"{system}_{log_size}_{noise_type}_{noise_level}_{iteration}.xes"
            row=[noisy_log_name]
            process_log(os.path.join(base_file_path, "logs", noisy_log_name), noisy_log_name, system, log_size, noise_type, noise_level, iteration, "results.csv", base_file_path)
            write_to_csv(row)

if __name__ == "__main__":
    main()