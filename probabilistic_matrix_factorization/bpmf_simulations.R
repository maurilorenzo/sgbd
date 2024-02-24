# BAYESIAN PROBABILISTIC MATRIX FACTORIZATION
# this R script contains the simulations the Bayesian Probabilistic Matrix Factorization (BPMF) (Section 4.3)

setwd('bpmf')
source('bpmf_setup.R')

####################################################################################################
##### LOAD and PREPROCESS DATA
####################################################################################################

# import df
df1 = read.delim("data/u1.base", header=FALSE, col.names = c("uid", "iid", "R", "info"))[c("uid", "iid", "R")]
test1 = read.delim("data/u1.test", header=FALSE, col.names = c("uid", "iid", "R", "info"))[c("uid", "iid", "R")]

head(df1)
head(test1)

# preprocess df
R_mean = mean(df1$R)
print(R_mean)

# center ratings
df1 = preprocess_df_bpmf(df1)

####################################################################################################
##### EXPERIMENTS (sec 4.3)
####################################################################################################

mini_batch = 800
S = 5000 # 50 ephocs
hyperparams_bpmf = list('R_mean'=R_mean, mu_0=0, alpha=3, alpha_0=1, beta=5, 
                        'N'= length(unique(df1$uid)), 'M' = length(unique(df1$iid)))
params_bpmf = initialize_bpfm(df1, hyperparams_bpmf, d=20)

# barker

# v-SGBD
step_size = sqrt(0.0005) # 0.0223
barker.bpmf.v.1 = SGMCMC_pmf(df1, S, mini_batch, step_size, params_bpmf, hyperparams_bpmf, test_performance=TRUE, 
                          test1, method='vanilla', 
                          performance_functions = list('likelihood'=log_lik_bpmf, 
                                                       'error_function'=rmse_bpmf, 'prediction'=predict_bpmf),
                          grad_functions=list('compute_grad'=compute_grad_bpmf, 'update_params'=update_bpmf))
# 1.11, 0.97

# e-SGBD
step_size = 0.005
barker.bpmf.e.1 = SGMCMC_pmf(df1, S, mini_batch, step_size, params_bpmf, hyperparams_bpmf, test_performance=TRUE, 
                                test1, method='extreme', performance_functions = list('likelihood'=log_lik_bpmf, 
                              'error_function'=rmse_bpmf, 'prediction'=predict_bpmf),
                             grad_functions=list('compute_grad'=compute_grad_bpmf, 'update_params'=update_bpmf))


# langevin

# v-SGLD
step_size = 0.011
langevin.bpmf.v.1 = SGMCMC_pmf(df1, S, mini_batch, step_size, params_bpmf, hyperparams_bpmf, test_performance=TRUE, 
                             test1, method='vanilla', performance_functions = list('likelihood'=log_lik_bpmf, 
                                                                                   'error_function'=rmse_bpmf, 'prediction'=predict_bpmf),
                             grad_functions=list('compute_grad'=compute_grad_bpmf, 'update_params'=update_bpmf), barker=FALSE)

# e-SGLD (SGD)
step_size = 0.0105
langevin.bpmf.e.1 = SGMCMC_pmf(df1, S, mini_batch, step_size, params_bpmf, hyperparams_bpmf, test_performance=TRUE, 
                               test1, method='extreme', performance_functions = list('likelihood'=log_lik_bpmf, 
                                                                                     'error_function'=rmse_bpmf, 'prediction'=predict_bpmf),
                               grad_functions=list('compute_grad'=compute_grad_bpmf, 'update_params'=update_bpmf), barker=FALSE)

####################################################################################################
##### PLOTS 
####################################################################################################

# rmse

# Fig. 4a
# sample
par(mar=c(5,6,4,2)+.1)
plot(seq(1, S, 100), barker.bpmf.v.1$error_sample, col='blue', type='l', xlab='iteration', ylab='rMSE', 
     ylim=c(1, 1.5), cex.lab=2, cex.axis=2, lwd=2, lty=3)
lines(seq(1, S, 100), langevin.bpmf.v.1$error_sample, col='red', lty=3, lwd=2)
lines(seq(1, S, 100), barker.bpmf.e.1$error_sample, col='blue4', lty=4, lwd=2)
lines(seq(1, S, 100), langevin.bpmf.e.1$error_sample, col='red4', lty=4, lwd=2)
legend('topright', legend=c("v-SGLD ", "e-SGLD ", "v-SGBD ", "e-SGBD "),
       col=c("red",  "red4", "blue", "blue4"), lty=c(3,4,3,4), cex=1.5)

# Fig 4b
# mcmc (only)
plot(seq(S/2+1, S, 100), barker.bpmf.v.1$error_mcmc, col='blue', type='l', xlab='iteration', 
     ylab='rMSE', cex.lab=2, cex.axis=2, lwd=2, lty=3, ylim=c(0.95, 1.16))
lines(seq(S/2+1, S, 100), langevin.bpmf.v.1$error_mcmc, col='red', lwd=2, lty=3)
lines(seq(S/2+1, S, 100), barker.bpmf.e.1$error_mcmc, col='blue4', lwd=2, lty=4)
lines(seq(S/2+1, S, 100), langevin.bpmf.e.1$error_mcmc, col='red4', lwd=2, lty=4)
legend('topright', legend=c("v-SGLD ", "e-SGLD ", "v-SGBD ", "e-SGBD "),
       col=c("red",  "red4", "blue", "blue4"), lty=c(3,4,3,4))
