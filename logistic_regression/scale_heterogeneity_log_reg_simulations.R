# BINARY REGRESSION WITH SCALE HETEROGENEITY 
# this R script contains the experiments with binary regression with scale heterogeneity (Section 4.2 and S4.3)

source("log_reg_setup.R")
library(coda)

#######################################################################################
# sepsis data preparation
#######################################################################################
sepsis_df = read.csv('data_sh/s41598-020-73558-3_sepsis_survival_primary_cohort.csv')
N = dim(sepsis_df)[1]
set.seed(123)
train_sepsis = sample(1:N, size=as.integer(0.8*N))
sepsis_train = sepsis_df[train_sepsis,]
sepsis_test = sepsis_df[-train_sepsis,]

#######################################################################################
# posterior samples via STAN
#######################################################################################
y_stan_sepsis = sepsis_train$y
x_stan_sepsis = sepsis_train[,-1]
d = dim(sepsis_train)[2] - 1
N = dim(sepsis_train)[1]
betaVar_sepsis = betaPrior(d)

sepsis_data_stan = list("N" = N, "X" = x_stan_sepsis, "y" = y_stan_sepsis, "d" = d, "Sigma0" = betaVar_sepsis)
sepsis_stan =  stan("logistic_regression_model.stan", data = sepsis_data_stan, iter = 10000, chains = 1)
list_of_draws_sepsis <- extract(sepsis_stan)
sepsis_posterior_means = colMeans(list_of_draws_sepsis$beta)
sepsis_posterior_vars = colSds(list_of_draws_sepsis$beta)^2

#######################################################################################
# EXPERIMENTS (section 4.1)
#######################################################################################
# params
S = 200000
n = as.integer(dim(sepsis_train)[1] * 0.01)
# init
set.seed(123)
theta_0 = rnorm(4, 0, 0.1)

sigma = 0.0002
langevin.sepsis.v.1 = SGMCMC(sepsis_train, S, n, sepsis_posterior_means, sigma, grad_L_logistic_regression,
                              grad_p_logistic_regression, method='vanilla', barker=FALSE, compute_performance=TRUE, test_set=sepsis_test, predict_y=TRUE, compute_performance_train=TRUE,
                             performance_functions=list('likelihood'=log_loss_log_reg,
                                                        'error'=function(x,y,type)(0), 
                                                        'predict'=predict_log_reg))
effectiveSize(as.mcmc(langevin.sepsis.v.1$thetas))


sigma = 0.0004
langevin.sepsis.v.2 = SGMCMC(sepsis_train, S, n, sepsis_posterior_means, sigma, grad_L_logistic_regression,
                             grad_p_logistic_regression, method='vanilla', barker=FALSE,compute_performance=TRUE, test_set=sepsis_test, predict_y=TRUE, compute_performance_train=TRUE,
                             performance_functions=list('likelihood'=log_loss_log_reg,
                                                        'error'=function(x,y,type)(0), 
                                                        'predict'=predict_log_reg))
effectiveSize(as.mcmc(langevin.sepsis.v.2.b$thetas))

sigma = 0.00075
barker.sepsis.v.1 = SGMCMC(sepsis_train, S, n, sepsis_posterior_means, sigma, grad_L_logistic_regression,
                           grad_p_logistic_regression, method='vanilla', barker=TRUE, compute_performance=TRUE, test_set=sepsis_test, predict_y=TRUE, compute_performance_train=TRUE,
                           performance_functions=list('likelihood'=log_loss_log_reg,
                                                      'error'=function(x,y,type)(0), 
                                                      'predict'=predict_log_reg))
effectiveSize(as.mcmc(barker.sepsis.v.1$thetas))

sigma = 0.0015
barker.sepsis.v.2 = SGMCMC(sepsis_train, S, n, sepsis_posterior_means, sigma, grad_L_logistic_regression,
                           grad_p_logistic_regression, method='vanilla', barker=TRUE,
                           compute_performance=TRUE, test_set=sepsis_test, predict_y=TRUE, compute_performance_train=TRUE,
                           performance_functions=list('likelihood'=log_loss_log_reg,
                                                      'error'=function(x,y,type)(0), 
                                                      'predict'=predict_log_reg))
effectiveSize(as.mcmc(barker.sepsis.v.2$thetas))



# Fig. 3 in main and Fig. S6
for (idx in c(1,2,3, 4)){
  mean = sepsis_posterior_means[idx]
  sd = sqrt(sepsis_posterior_vars[idx])
  par(mar=c(4.5,5.5,1.5,1))
  plot(langevin.sepsis.v.1$theta[1:200000, idx], col=rgb(red=1, green=0, blue=0, alpha=0.75), lwd=0.3,
       lty=1, main='', xlab='iteration', cex.lab=2, cex.axis=2,
       ylab=bquote(~theta[.(idx)]), type='l', ylim=c(mean-6*sd, mean+6*sd))
  lines(barker.sepsis.v.1$theta[1:200000, idx], col=rgb(red=0, green=0, blue=1, alpha=0.75), lwd=0.3)
  abline(h=mean+2*sd)
  abline(h=mean-2*sd)
  if(idx==1){legend('topleft', legend=c("SGLD", "SGBD"),
                    col=c("red", "blue"), lty=c(1, 1), cex=2)}
}

for (idx in c(1, 2, 3,4)){ 
  mean = sepsis_posterior_means[idx]
  sd = sqrt(sepsis_posterior_vars[idx])
  par(mar=c(5.5,5.5,1.5,1))
  if(idx==1){ylims = c(-0.1, 0.1)}
  else{ylims=c(mean-6*sd, mean+6*sd)}
  plot(langevin.sepsis.v.2$theta[1:200000, idx], col=rgb(red=1, green=0, blue=0, alpha=0.75), lwd=0.3,
       lty=1, main='', xlab='iteration', cex.lab=2, cex.axis=2,
       ylab=bquote(~theta[.(idx)]), type='l', ylim = ylims)
  lines(barker.sepsis.v.2$theta[1:200000, idx], col=rgb(red=0, green=0, blue=1, alpha=0.75), lwd=0.3)
  abline(h=mean+2*sd)
  abline(h=mean-2*sd)
}

# Fig. S5
bws = c(0.0005, 0.008, 0.005, 0.03)
for (idx in 1:4){
  print(idx)
  plot_marginal_distributions_lr(idx, sepsis_stan_samples, barker.sepsis.v.1$thetas, langevin.sepsis.v.1$thetas, 
                                 barker.sepsis.v.2$thetas, langevin.sepsis.v.2.b$thetas, bws=bws,
                                 save=T, legend=T)
}

#######################################################################################
# EXPERIMENTS (supplemental S4.4)
#######################################################################################


S = 200000
sigma = 0.1*sqrt(sepsis_posterior_vars)
langevin.sepsis.v.3 = SGMCMC(sepsis_train, S, n, sepsis_posterior_means, sigma, grad_L_logistic_regression,
                             grad_p_logistic_regression, method='vanilla', barker=FALSE, 
                             compute_performance=TRUE, test_set=sepsis_test, predict_y=TRUE, compute_performance_train=TRUE,
                             performance_functions=list('likelihood'=log_loss_log_reg,
                                                        'error'=function(x,y,type)(0), 
                                                        'predict'=predict_log_reg))
effectiveSize(as.mcmc(langevin.sepsis.v.3$thetas))


sigma = 0.2*sqrt(sepsis_posterior_vars)
langevin.sepsis.v.4 = SGMCMC(sepsis_train, S, n, sepsis_posterior_means, sigma, grad_L_logistic_regression,
                             grad_p_logistic_regression, method='vanilla', barker=FALSE,
                             compute_performance=TRUE, test_set=sepsis_test, predict_y=TRUE, compute_performance_train=TRUE,
                             performance_functions=list('likelihood'=log_loss_log_reg,
                                                        'error'=function(x,y,type)(0), 
                                                        'predict'=predict_log_reg))
effectiveSize(as.mcmc(langevin.sepsis.v.4$thetas))


sigma = 0.1*sqrt(sepsis_posterior_vars)
barker.sepsis.v.3 = SGMCMC(sepsis_train, S, n, sepsis_posterior_means, sigma, grad_L_logistic_regression,
                           grad_p_logistic_regression, method='vanilla', barker=TRUE, 
                           compute_performance=TRUE, test_set=sepsis_test, predict_y=TRUE, compute_performance_train=TRUE,
                           performance_functions=list('likelihood'=log_loss_log_reg,
                                                      'error'=function(x,y,type)(0), 
                                                      'predict'=predict_log_reg))
effectiveSize(as.mcmc(barker.sepsis.v.3$thetas))

sigma = 0.2*sqrt(sepsis_posterior_vars)
barker.sepsis.v.4 = SGMCMC(sepsis_train, S, n, sepsis_posterior_means, sigma, grad_L_logistic_regression,
                           grad_p_logistic_regression, method='vanilla', barker=TRUE,
                           compute_performance=TRUE, test_set=sepsis_test, predict_y=TRUE, compute_performance_train=TRUE,
                           performance_functions=list('likelihood'=log_loss_log_reg,
                                                      'error'=function(x,y,type)(0), 
                                                      'predict'=predict_log_reg))
effectiveSize(as.mcmc(barker.sepsis.v.4$thetas))


# Fig. S7
bws = c(0.0005, 0.009, 0.006, 0.03)
for (idx in 1:4){
  print(idx)
  plot_marginal_distributions_lr(idx, sepsis_stan_samples, barker.sepsis.v.3$thetas, 
                                 langevin.sepsis.v.3$thetas,
                                 barker.sepsis.v.4$thetas, langevin.sepsis.v.4$thetas)
                                 if(idx==1){legend('topleft', legend=c(expression(paste("SGLD-", sigma[1], '  ')), expression(paste("SGLD-", sigma[2],'  ')),
                                           expression(paste("SGBD-", sigma[1], '  ')), expression(paste("SGBD-", sigma[2], '  '))),
                   col=c("red", "red3", "blue", "blue3"), lty=c(3,5,3,5), cex=1.75)}
}


# mamba
N_iter_adaptation <- 20000
S <- 200000
n <- as.integer(dim(sepsis_train)[1] * 0.01)
set.seed(123)
theta_0 <- rnorm(4, 0, 0.1)

sigma_l <- 0.00001
sigma_u <- 0.0005
sigmas <- exp(seq(log(sigma_l), log(sigma_u), length.out=10))
sigmas_candidates <- sigmas
for(i in 0:(log(length(sigmas), 3)-1)){
  S_trial <- as.integer(N_iter_adaptation/(length(sigmas_candidates) * log(length(sigmas), 3)))
  S_trial <- 2*as.integer(S_trial/2)
  samples_sigmas <- lapply(sigmas_candidates, 
                           function(x)
                             (SGMCMC(sepsis_train, S_trial, n, sepsis_posterior_means,
                                     x, grad_L_logistic_regression,grad_p_logistic_regression, 
                                     method='vanilla', barker=FALSE, compute_performance=TRUE,
                                     test_set=sepsis_test, predict_y=TRUE, compute_performance_train=TRUE,
                                     performance_functions=list('likelihood'=log_loss_log_reg,
                                                                'error'=function(x,y,type)(0), 
                                                                'predict'=predict_log_reg))))
  
  fssd_sigmas <- lapply(samples_sigmas, function(x) (compute_FSSD(x$thetas, sepsis_train)))
  fssd_sigmas <- unlist(fssd_sigmas)
  print(sigmas_candidates)
  print(fssd_sigmas)
  sigmas_candidates <- sigmas_candidates[rank(fssd_sigmas) <= length(sigmas_candidates)/3]
  print(sigmas_candidates)
}
sigma.langevin.mamba <- 0.0003237394
langevin.sepsis.v.mamba = SGMCMC(sepsis_train, S, n, sepsis_posterior_means,sigma.langevin.mamba , grad_L_logistic_regression,
                             grad_p_logistic_regression, method='vanilla', barker=FALSE,
                             compute_performance=TRUE, test_set=sepsis_test, predict_y=TRUE, compute_performance_train=TRUE,
                             performance_functions=list('likelihood'=log_loss_log_reg,
                                                        'error'=function(x,y,type)(0), 
                                                        'predict'=predict_log_reg))


sigma_l <- 0.0001
sigma_u <- 0.005
sigmas_barker <- exp(seq(log(sigma_l), log(sigma_u), length.out=10))
sigmas_candidates_barker <- sigmas_barker
for(i in 0:(log(length(sigmas), 3)-1)){
  S_trial <- as.integer(N_iter_adaptation/(length(sigmas_candidates_barker) * log(length(sigmas_barker), 3)))
  S_trial <- 2*as.integer(S_trial/2)
  samples_sigmas_barker <- lapply(sigmas_candidates_barker, 
                           function(x)
                             (SGMCMC(sepsis_train, S_trial, n, sepsis_posterior_means,
                                     x, grad_L_logistic_regression,grad_p_logistic_regression, 
                                     method='vanilla', barker=TRUE, compute_performance=TRUE,
                                     test_set=sepsis_test, predict_y=TRUE, compute_performance_train=TRUE,
                                     performance_functions=list('likelihood'=log_loss_log_reg,
                                                                'error'=function(x,y,type)(0), 
                                                                'predict'=predict_log_reg))))
  
  fssd_sigmas_barker <- lapply(samples_sigmas_barker, function(x) (compute_FSSD(x$thetas, sepsis_train)))
  fssd_sigmas_barker <- unlist(fssd_sigmas_barker)
  print(sigmas_candidates_barker)
  print(fssd_sigmas_barker)
  sigmas_candidates_barker <- sigmas_candidates_barker[rank(fssd_sigmas_barker) < length(sigmas_candidates_barker)/3]
  print(sigmas_candidates_barker)
}

sigma.barker.mamba <- 0.00135720881
barker.sepsis.v.mamba = SGMCMC(sepsis_train, S, n, sepsis_posterior_means, sigma.barker.mamba, grad_L_logistic_regression,
                           grad_p_logistic_regression, method='vanilla', barker=TRUE,
                           compute_performance=TRUE, test_set=sepsis_test, predict_y=TRUE, compute_performance_train=TRUE,
                           performance_functions=list('likelihood'=log_loss_log_reg,
                                                      'error'=function(x,y,type)(0), 
                                                      'predict'=predict_log_reg))





