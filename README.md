# Companion code for "Bayesian logistic regression for online recalibration and revision of risk prediction models with performance guarantees"


# Installation
We use `pip` to install things into a python virtual environment. Refer to `requirements.txt` for package requirements.
We use `nestly` + `SCons` to run simulations.

# File descriptions

`generate_data_single_pop.py` -- Simulate a data stream from a single population following a logistic regression model.
* Inputs:
  - `--simulation`: string for selecting the type of distribution shift. Options for this argument are the keys in `SIM_SETTINGS` in `constants.py`.
* Outputs:
  - `--out-file`: pickle file containing the data stream

`generate_data_two_pop.py` -- Simulate a data stream from two subpopulations, where each are generated using logistic regression models. Similar arguments as `generate_data_single_pop.py`. The percentage split beween the two subpopulations is controlled by the `--subpopulations` argument.
* Outputs:
  - `--out-file`: pickle file containing the data stream

`create_modeler.py` -- Creates a model developer who fits the original prediction model and may propose a continually refitted model at each time point.
* Inputs:
  - `--data-file`: pickle file with the entire data stream
  - `--simulation`: string for selecting the model refitting strategy by the model developer. Options are to keep the model locked (`locked`), refit on all accumulated data (`cumulative_refit`), and refit on the latest observations within some window length (`boxed`, window length specified by `--max-box`). The last two options is to train an ensemble with the original and the `cumulative_refit` models (`combo_refit`) and train an ensemble with the original and the `boxed` models (`combo_boxed`).
* Outputs: 
  - `--out-file`: pickle file containing the modeler

`main.py` -- Given the data and the model developer, run online model recalibration/revision using MarBLR and BLR.
* Inputs: 
  - `--data-file`: pickle file with the entire data stream
  - `--model-file`: pickle file with the model developer
  - `--type-i-regret-factor`: Type I regret will be controlled at the rate of `args.type_i_regret_factor` * (Initial loss of the original model)
  - `--reference-recalibs`: comma-separated string to select which other online model revisers to run. Options are no updating at all `locked`, ADAM `adam`, cumulative logistic regression `cumulativeLR`.
* Outputs:
  - `--obs-scores-file`: csv file containing predicted probabilities and observed outcomes on the data stream
  - `--history-file`: csv file containing the predicted and actual probabilities on a held-out test data stream (only available if the data stream was simulated)
  - `--scores-file`: csv file containing performance measures on a held-out test data stream (only available if the data stream was simulated)
  - `--recalibrators-file`: pickle file containing the history of the online model revisers


# Reproducing simulation results

The `simulation_recalib` folder contains the first set of simulations for online model recalibration. The `simulation_revise` folder contains the second set of simulations where we perform online logistic revision. The `simulation_revise` folder contains the third set of simulations where we perform online ensembling of the original model with a continually refitted model. The `copd_analysis` folder contains code for online model recalibration and revision for the COPD dataset. To reproduce the simulations, run `scons <simulation_folder_name>`.
