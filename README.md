# Species Distribution Modelling of Fish in the Baltic Sea
## A Neural Network Approach

## Overview
This repository contains R code for the Master's thesis:
"Multi-species distribution modelling of fish in the Baltic Sea. 
A neural network approach." (Theophil Fröse, SLU Uppsala, 2026)

The project develops and compares multi-species neural network approaches 
for predicting spatial distributions of 25 fish species across the Baltic 
Sea using satellite-derived environmental predictor variables. Two model 
types are implemented: an independent binomial ANN and a joint 
Multivariate Probit (MVP) ANN that accounts for residual covariance 
among species.

## Repository Structure

├── Functions.R                              # Shared functions across all scripts
│
│   --- Preprocessing & Setup ---
├── 00_model_exploration.R                   # Initial exploration and prototyping using a subset of 5 species
├── 01_data_preprocessing_and_split.R        # Data cleaning and predictor preparation (Splitting development data and independent test data)
├── 02_cross_validation_folds.R              # Grouped spatial cross-validation setup (balances presence/absence data in 5-folds and saves fold indices for use across models and scripts)
├── 03_dataset_exploration.R                 # Exploratory dataset analysis (independent of ML pipeline)
│
│   --- Model Tuning ---
├── 04a_binomial_ANN_tuning.R                # Hyperparameter tuning — binomial model: iterative random search with different numbers of configurations due to computational restrictions
├── 04b_MVP_ANN_tuning.R                     # Hyperparameter tuning — MVP model: iterative random search with different numbers of configurations due to computational restrictions
│
│   --- Model Training, Prediction & Evaluation ---
├── 05a_binomial_ANN_final_train_pred_eval.R # Binomial ANN — full pipeline
├── 05b_MVP_ANN_final_train_pred_eval.R      # MVP ANN — full pipeline
├── 05c_gillnet_MVP_model.R                  # Gillnet MVP model (no tuning)
├── 05d_CNN_model.R                          # CNN architecture (experimental, no tuning)
│
│   --- Analysis & Visualisation ---
├── 06_model_analysis.R                      # Model comparison and interpretation
├── 07_species_performance_plot.R            # Species-level performance visualisation
├── 08_ensemble_cv_predictions.R             # Cross-validated ensemble predictions for MVP model
├── 09_joint_community_predictions.R         # Script for experimental Joint composition predictions (Monte Carlo)
│
├── data/                                    # Data directory (not included, see below)
├── results/                                 # Model outputs (weights, predictions) and figures
├── renv/                                    # R environment
├── ANN.Rproj                                # R project file
└── renv.lock                                # Package dependencies (renv)

## Data
Raw and processed data were offered by the PROTECT BALTIC project and are not included due to licensing restrictions.
Environmental predictor variables were derived from satellite products 
and processed as spatial raster data using the terra package in R.
Contact theo.froese@proton.me for further information.

## Reproducing the Analysis
Scripts should be run in numerical order (00 → 09).
Scripts 06-09 are independent of each other and all use 
the trained models from step 05.

Package dependencies are managed with renv.
To restore the environment: renv::restore()

## Key Methods
- Independent binomial ANN and joint MVP ANN for species distribution modelling
- Grouped spatial cross-validation with balanced presences/absences across folds and species
- Hyperparameter tuning using iterative random search for neural network architectures (parallel CPU cores)
- Accumulated local effects (ALE) and permutation importance for interpretability
- Extraction and visualisation of species residual covariance matrix of the MVP model using only the Gillnet data to avoid confounding gear effects
- Joint probability prediction via Monte Carlo sampling from multivariate normal distribution (predict_jointly function)
- Manual weight extraction for MVP model predictions (predict_mvp_manual) implemented to handle torch pointer issues after model reloading


## Dependencies
R version > 4.4
Main R packages: cito, terra, rsample, torch, ggplot2
Full dependency list: renv.lock



## Contact
Theophil Fröse
theo.froese@proton.me
