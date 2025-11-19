import csv
import os
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer

from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.replay_fitness import algorithm as recall_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator

def calculate_simplicity(net, initial_marking=None, final_marking=None):
    # Count places and transitions
    num_places = len(net.places)
    num_transitions = len(net.transitions)
    total_nodes = num_places + num_transitions

    return total_nodes

def evaluate(clean_model_path, clean_log_path, results_file = "results_f1.csv"):
    try:
        log = xes_importer.apply(clean_log_path)
        net, im, fm = pnml_importer.apply(clean_model_path)

        prec = precision_evaluator.apply(log, net, im, fm, variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
        rec = recall_evaluator.apply(log, net, im, fm, variant=recall_evaluator.Variants.TOKEN_BASED)['log_fitness']
        simp = calculate_simplicity(net, im, fm)
        
        row = [clean_log_path, clean_model_path, prec, rec, simp]
        write_to_csv(row, results_file)

    except Exception as e:
        fail_row = ["n/a", clean_log_path, clean_model_path, "FAILED", str(e)]
        write_to_csv(fail_row, "failures.csv")

def write_to_csv(row, results_file):
    file_exists = os.path.isfile(results_file)
    with open(results_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            if results_file == "results_clean.csv":
                writer.writerow(["log", "model", "precision", "recall", "simplicity"])
            elif results_file == "results_noisy_replay.csv":
                writer.writerow(["noisy_log", "clean_log", "model", "precision_noisy", "recall_noisy", "precision_clean", "recall_clean", "simplicity"])
            elif results_file == "failures_replay.csv":
                writer.writerow(["noisy_log", "clean_log", "model", "status", "error"])
        writer.writerow(row)

def main():
    systems = ["Sepsis", "RTFMS", "BPIC2012"]
    log_sizes = [1000, 2000, 4000, 10000, 20000, 40000, 100000]
    base_log_file_path =  r"C:\projects\python\Anda\Stability_Java\src\main\resources\systems"
    base_model_file_path =  r"C:\projects\python\Anda\resources"
    csv_file_path = "log_data_f2.csv"
    algos = ["alpha", "heuristics", "inductive"]

    # === Step 1: Clean logs (deterministic order) ===
    for system in systems:
        for log_size in log_sizes:
            print(f"\nProcessing clean log: {system} of size {log_size}")
            clean_log_name = f"{system}.xes"
            clean_log_path = os.path.join(base_log_file_path, clean_log_name)
            
            for algo in algos:
                clean_model_name = f"{system}_{log_size}_{algo}.pnml"
                clean_model_path = os.path.join(base_model_file_path, "models", clean_model_name)
                evaluate(clean_model_path, clean_log_path)

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
            noisy_log_name = f"{system}.xes"
            noisy_log_path = os.path.join(base_log_file_path, noisy_log_name)
            for algo in algos:
                noisy_model_name = f"{system}_{log_size}_{noise_type}_{noise_level}_{iteration}_{algo}.pnml"
                noisy_model_path = os.path.join(base_model_file_path, "models", noisy_model_name)
                evaluate(noisy_model_path, noisy_log_path)

if __name__ == "__main__":
    main()


