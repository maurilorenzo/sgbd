# INDEPENDENT COMPONENT ANALYSIS
# this R script contains the simulations with the Independent Component Analysis (ICA) (Section 4.4)
setwd('ica')

# load setup
source('ica_setup.R')

############################################################################################
##### DATA PREPARATION
############################################################################################
MEG_df <- read_csv("data/MEG_data_reduced.csv")
dim(MEG_df)
X = as.matrix(MEG_df)
set.seed(123)
train_index_ica = sample(dim(MEG_df)[2], int(0.8* dim(MEG_df)[2]))
X_train = t(as.matrix(MEG_df_train))
X_means = colMeans(X_train)
X_sds = colSds(X_train)
X_train = scale(X_train)
X_test = t(as.matrix(MEG_df_test))
X_test = scale(X_test, center=X_means, scale=X_sds)

############################################################################################
##### EXPERIMENTS (sec 4.2)
############################################################################################
# run FAST ICA to find a good initialization point
fast_ica = fastICA(X_train, n.comp=10)
names(fast_ica)
fast_ica$W
sum(fast_ica$X != X_train)

log_likelihood_ica_alt(fast_ica$X, fast_ica$W)
log_likelihood_ica_alt(fast_ica$X, t(fast_ica$W))

# set hyperparams
# mini batch size
n = 100
# latent dimensionality
d = 10
# prior precision
lambda = 1
# stating point
theta = fast_ica$W
options(digits=8)

S = 40000 # 40k

# barker

# vanilla
sigma = 160 / dim(X_train)[1] # best
barker.ica.v = SGMCMC(X_train, S, n, theta, sigma, grad_log_likelihood_ica,
                      grad_log_prior_ica, noise='bimodal',method='vanilla',
                      compute_performance=TRUE, test_set=X_test, 
                      performance_functions=list('likelihood'=log_likelihood_ica_alt, 'error'=function(x)(0), 'predict'=function(x)(0)),
                      compute_error=FALSE, compute_performance_train=TRUE)

# corrected
sigma = 120.5 / dim(X_train)[1] # best
barker.ica.c = SGMCMC(X_train, S, n, theta, sigma, grad_log_likelihood_ica,
                      grad_log_prior_ica, noise='bimodal',method='corrected',
                      compute_performance=TRUE, test_set=X_test, 
                      performance_functions=list('likelihood'=log_likelihood_ica_alt, 'error'=function(x)(0), 'predict'=function(x)(0)),
                      compute_error=FALSE, compute_performance_train=TRUE)

# extreme
sigma = 120.0 / dim(X_train)[1] # 
barker.ica.e = SGMCMC(X_train, S, n, theta, sigma, grad_log_likelihood_ica,
                      grad_log_prior_ica, noise='bimodal',method='extreme',
                      compute_performance=TRUE, test_set=X_test, 
                      performance_functions=list('likelihood'=log_likelihood_ica_alt, 'error'=function(x)(0), 'predict'=function(x)(0)),
                      compute_error=FALSE, compute_performance_train=TRUE)


# langevin

# vanilla
sigma = 100 / dim(X_train)[1] 
langevin.ica.v = SGMCMC(X_train, S, n, theta, sigma, grad_log_likelihood_ica,
                        grad_log_prior_ica, noise='bimodal',method='vanilla', barker=FALSE,
                        compute_performance=TRUE, test_set=X_test, 
                        performance_functions=list('likelihood'=log_likelihood_ica_alt, 'error'=function(x)(0), 'predict'=function(x)(0)),
                        compute_error=FALSE, compute_performance_train=TRUE)

# corrected
sigma = 100 / dim(X_train)[1] 
langevin.ica.c = SGMCMC(X_train, S, n, theta, sigma, grad_log_likelihood_ica,
                        grad_log_prior_ica, noise='bimodal',method='corrected', barker=FALSE,
                        compute_performance=TRUE, test_set=X_test, 
                        performance_functions=list('likelihood'=log_likelihood_ica_alt, 'error'=function(x)(0), 'predict'=function(x)(0)),
                        compute_error=FALSE, compute_performance_train=TRUE)

sigma = 100 / dim(X_train)[1] 
langevin.ica.c.2 = SGMCMC(X_train, S, n, theta, sigma, grad_log_likelihood_ica,
                        grad_log_prior_ica, noise='bimodal',method='corrected', barker=FALSE,
                        compute_performance=TRUE, test_set=X_test, 
                        performance_functions=list('likelihood'=log_likelihood_ica_alt, 'error'=function(x)(0), 'predict'=function(x)(0)),
                        compute_error=FALSE, compute_performance_train=TRUE)

# extreme
sigma = 90 / dim(X_train)[1] 
langevin.ica.e = SGMCMC(X_train, S, n, theta, sigma, grad_log_likelihood_ica,
                        grad_log_prior_ica, noise='bimodal',method='extreme', barker=FALSE,
                        compute_performance=TRUE, test_set=X_test, 
                        performance_functions=list('likelihood'=log_likelihood_ica_alt, 'error'=function(x)(0), 'predict'=function(x)(0)),
                        compute_error=FALSE, compute_performance_train=TRUE)

# plot results
lwd=3
# Fig. 5a
# mcmc
plot(seq(S/2+1, S, 100), barker.ica.v$lik_mcmc, col='blue', type='l', xlab='iteration', 
     ylab='log-likelihood', cex.lab=2.5, cex.axis=2.5, lwd=lwd, lty=3, ylim=c(-11.668, -11.659))
lines(seq(S/2+1, S, 100), langevin.ica.v$lik_mcmc, col='red', lty=3, lwd=lwd)
lines(seq(S/2+1, S, 100), barker.ica.c$lik_mcmc, col='blue3', lty=5, lwd=lwd)
lines(seq(S/2+1, S, 100), langevin.ica.c.2$lik_mcmc, col='red3', lty=5, lwd=lwd)
lines(seq(S/2+1, S, 100), langevin.ica.e$lik_mcmc, col='red4', lty=4, lwd=lwd)
lines(seq(S/2+1, S, 100), barker.ica.e$lik_mcmc, col='blue4', lty=4, lwd=lwd)

# Fig. 5b
# samples
plot(seq(1, S, 100), barker.ica.v$lik_sample, col='blue', type='l', xlab='iteration', 
     ylab='log-likelihood', cex.lab=2.5, cex.axis=2.5, lwd=lwd, lty=3, ylim=c(-11.8, -11.67))
lines(seq(1, S, 100), langevin.ica.v$lik_sample, col='red', lty=3, lwd=lwd)
lines(seq(1, S, 100), barker.ica.c$lik_sample, col='blue3', lty=5, lwd=lwd)
lines(seq(1, S, 100), langevin.ica.c.2$lik_sample, col='red3', lty=5, lwd=lwd)
lines(seq(1, S, 100), barker.ica.e$lik_sample, col='blue4',  lty=4, lwd=lwd)
lines(seq(1, S, 100), langevin.ica.e$lik_sample, col='red4', lty=4, lwd=lwd)
legend('bottomright', legend=c("v-SGLD ", "c-SGLD ","e-SGLD ", "v-SGBD ", "c-SGBD ", "e-SGBD "),
       col=c("red", "red3", "red4", "blue", "blue3","blue4"), lty=c(3,5,4,3,5,4), ncol=2, cex=1.7)