# Robust Approximate Sampling via Stochastic Gradient Barker Dynamics

This repo contains the files containing the necessary R code to replicate all the experiments in the article **Robust Approximate Sampling via Stochastic Gradient Barker Dynamics** (Mauri L., Zanella G., 2024). 

Experiments were run on a laptop with 11th Gen Intel(R) Core(TM) i7-1165G7 2.80 GHz using R version 4.3.1.

The repo is structured as follows:

```
├── ica
    ├── data
    ├── ica_setup.R
    └── ica_simulations.R
├── logistic_regression
    ├── data_hd
    ├── data_sh
    ├── high_dimensional_log_reg_simulatons.R
    ├── logistic_regression_model.STAN  
    ├── log_reg_simulatons.R   
    └── scale_heterogeneity_log_reg_simulatons.R
├── probabilistic_matrix_factorization
    ├── bpmf_setup.R
    ├── bpmf_simulations.R
	└── data
├── toy_models
    ├── toy_models_setup.R
    └── toy_models_simulations.R
└── utils.R
 ```  

`*_simulations.R` files contain the code to replicate the experiments presented in the article. In particular, `toy_model_simulations.R` implements the experiments in Sections 4.1 and S3-4.2. 
`scale_heterogeneity_log_reg_simulations.R` implements the experiments in Sections 4.2 and S4.3, `high_dimensional_log_reg_simulations.R` implements the experiments in Sections S4.1 and S4.4, `bpmf_simulations.R` implements the experiments in Section 4.3, and `ica_simulations.R` implements the experiments in Section 4.4. 
`data*` folders contain the data used for the relative experiments. `utils.R` and `*_setup.R` files contain helper functions and `*.STAN` files contain the STAN code used in the experiments.
