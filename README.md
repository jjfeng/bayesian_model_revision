# Companion code for "Bayesian logistic regression for online recalibration and revision of risk prediction models with performance guarantees"


# Installation
We use `pip` to install things into a python virtual environment. Refer to `requirements.txt` for package requirements.
We use `nestly` + `SCons` to run simulations.

# File descriptions

`generate_data_single_pop.py` -- Generate data for a single population.

`generate_data_two_pop.py` -- Generate data for two subpopulations.

`create_modeler.py` -- Creates a model developer who fits the original prediction model and may propose a continually refitted model at each time point.

`main.py` -- Given the data and the model developer, run online model recalibration/revision using MarBLR and BLR. It can also run other online model recalibrations if they are specified using the `--reference-recalibs` argument (e.g. no updating at all `locked`, ADAM `adam`, cumulative logistic regression `cumulativeLR`).

# Reproducing simulation results

The `simulation_recalib` folder contains the first set of simulations for online model recalibration. The `simulation_revise` folder contains the second set of simulations where we perform online logistic revision. The `simulation_revise` folder contains the third set of simulations where we perform online ensembling of the original model with a continually refitted model. The `copd_analysis` folder contains code for online model recalibration and revision for the COPD dataset. To reproduce the simulations, run `scons <simulation_folder_name>`.
