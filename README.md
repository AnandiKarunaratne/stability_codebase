# Conditioning & Stability Analysis of Process Discovery Algorithms

This repository contains the full experimental codebase used to analyze the conditioning and stability of process discovery algorithms.
The workflow is organized into four modular sub-projects, each responsible for a key stage of the experimental pipeline.

## Project Structure
### 1. LogGeneration

Generates synthetic event logs for experimentation.

- Uses SNIP to inject controlled noise.
- Supports multiple noise types and parameter configurations.
- Produces input logs for downstream discovery tasks.

### 2. ModelDiscovery

Discovers process models for each generated log.

- Implements Alpha Miner, Heuristics Miner, and Inductive Miner.
- Provides consistent interfaces for batch model discovery.
- Outputs models in standardized formats for evaluation.

### 3. Evaluation

Computes the core measures needed for conditioning and stability analysis.

- Quantifies differences between: logs; discovered models; behavioral perspectives; structural perspectives.
- Produces structured, machine-readable evaluation results.

### 4. Analysis

Analyzes conditioning and stability results using the computed metrics.

- Performs statistical comparisons and robustness assessments.
- Generates all visualizations used in the study (plots, summaries, and tables).
- Includes scripts for reproducing final figures.
