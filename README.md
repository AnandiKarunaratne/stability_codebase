# Conditioning & Stability Analysis of Process Discovery Algorithms

This repository contains the full experimental codebase used to analyze the conditioning and stability of process discovery algorithms.
The experiments and methodology implemented here are described in the following papers:

1. Karunaratne, A., Polyvyanyy, A., Moffat, A.: Conditioning and stability in process discovery. In: CAiSE, Springer (2026), pp. 322--337.
2. Karunaratne, A., Polyvyanyy, A., Moffat, A.: Reliability of process discovery algorithms. Manuscript in preparation.

The workflow is organized into four modular sub-projects, each responsible for a key stage of the experimental pipeline.

## Project Structure
### 1. LogGeneration

Generates synthetic event logs for experimentation.

- Uses [SNIP](https://github.com/AnandiKarunaratne/SNIP) to inject controlled noise.
- Supports multiple noise types and parameter configurations.
- Produces input logs for downstream discovery tasks.

To run, update `src/main/resources/systems/` folder with the systems folder in https://doi.org/10.26188/30739082, and run `src/main/java/org/anandi/Main.java`.
The clean samples from the base logs and noisy versions of those clean samples will be generated in `src/main/resources/logs/`.

### 2. ModelDiscovery

Discovers process models for each generated log.

- Implements Alpha Miner, Heuristics Miner, and Inductive Miner.
- Provides consistent interfaces for batch model discovery.
- Outputs models in standardized formats for evaluation.
 
To run, run `model_discovery.py` (if needed, update the base file path to where noisy logs are stored). The code assumesa path `<base_path>/logs/`. 
Models will be generated at `<base_path>/models/`.
The clean models are discovered first, then noisy models are discovered in a random order (according to `log_data.csv`).

### 3. Evaluation

Computes the core measures needed for conditioning and stability analysis.

- Quantifies differences between: logs; discovered models; behavioral perspectives; structural perspectives.
- Produces structured, machine-readable evaluation results.

For rediscoverability analysis, run `evaluation_f1.py`. If needed, update the paths to `systems` and `models`.
For behavior replay analysis, run `evaluation_f2.py`. If needed, update the paths to `logs` and `models`.
For rediscoverability analysis with a metric distance, run `evaluation_m3.py`. This script measures model distances using the [_m3_ metric](https://doi.org/10.1007/978-3-642-23059-2_15).
### 4. Analysis

Analyzes conditioning and stability results using the computed metrics.

- Performs statistical comparisons and robustness assessments.
- Generates all visualizations used in the study (plots, summaries, and tables).
- Includes scripts for reproducing the final figures.

**Conditioning.** To analyze how model complexity changes with log changes, run `conditioning.py`. This generates the analysis report and plots.

**Stability.** To analyze stability, run `stability_f1.py` for rediscoverability or `stability_f2.py` for behavior replay. Each generates the analysis report and plots.

**Stability under multiple goals.** To compare algorithms across both goals using compromise programming, run `compromise_lp.py` for the compromise score against the goal weight, `compromise_programming.py` for the score difference between Inductive and Heuristics Miners and the weights where the preferred algorithm switches, and `mean_max_delta.py` for the mean and maximum difference between them.

**Relating conditioning and stability.** Run `conditioning2.py` for conditioning and `stability_f3.py` for stability, both using the _m3_ metric. This analysis is limited to the Inductive Miner, since _m3_ applies only to bounded Petri nets.
