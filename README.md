# AI-Assisted Prehospital Dispatch Triage under Resource Constraints

This repository contains the core analytical code and simulation framework for our study on AI-assisted prehospital triage, submitted to *npj Digital Medicine*.

## Overview
This project proposes an NLP-driven dispatch copilot designed to mitigate systemic prehospital undertriage. Instead of evaluating predictive performance in isolation, this framework embeds risk stratification within a resource-constrained operational environment using discrete-event queuing simulations (DES).

## Code Structure
- **`Model_Training_and_Fairness.py`**: Python script for text vectorization (TF-IDF), XGBoost model training, calibration analysis (Brier score), and subgroup fairness evaluation.
- **`DES_Simulation.py`**: Discrete-event queuing simulation (DES) and Monte Carlo iterations to evaluate capacity-aware dispatch policies and compute marginal time-saved benefits.
- **`CrossDomain_Validation.py`**: Script for zero-shot external validation on the independent chest pain cohort and extraction of SHAP values for interpretability analysis.

## Note on Data Availability
Due to strict institutional data security and patient privacy policies, the raw EMS dispatch records containing free-text narratives cannot be made publicly available in this repository. De-identified data may be provided by the corresponding author upon reasonable request and formal data-sharing agreement.
