# HIGH DIMENSIONAL BINARY REGRESSION
# this script contains the Logistic regression simulations on the Arrhythmia dataset (Section 3.3, S4.1 and S4.4)
setwd('logistic_regression')
source("log_reg_setup.R")

############################################################################################
##### DATA PREPARATION
############################################################################################
# arrhrytmia data preparation
# load df
arrhythmia = read.csv("data_hd/arrhythmia.data", header=FALSE, na.strings="?")

# eliminate null targets
arrhythmia = arrhythmia[arrhythmia['V280'] < 16, ]

# create binary response variable
arrhythmia$y = 1
arrhythmia$y[arrhythmia['V280']==1] = 0
arrhythmia = arrhythmia[names(arrhythmia) != 'V280']
sum(arrhythmia$y==1)
sum(arrhythmia$y==0)
colSums(is.na(arrhythmia))

# eliminate columns with only zeros
arrhythmia = arrhythmia[, ! names(arrhythmia) %in% c('V11', 'V12', 'V13', 'V14', 'V15', 'V22', 'V46', 'V61', 'V86')]
arrhythmia = arrhythmia[, ! names(arrhythmia) %in% c('V20', 'V68', 'V70','V84', 'V205', 'V165', 'V157', 'V158', 'V146', 
                                                     'V152', 'V140', 'V142', 'V144', 'V132', 'V133', 'V275', 'V265', 'V48', 'V116')]
# remove collinear columns
arrhythmia = arrhythmia[, ! names(arrhythmia) %in% c('V51', 'V62', 'V75','V80', 'V85')]


arrhythmia = arrhythmia[,c(ncol(arrhythmia), 1:100)]
rankMatrix(arrhythmia[,-1])
dim(arrhythmia)
head(arrhythmia)

standard scaling
arrhythmia[,-1] <- scale(arrhythmia[,-1])
arrhythmia_2[,-1] <- scale(arrhythmia_2[,-1])

# train-test split
set.seed(123)
train = sample(1:nrow(arrhythmia), nrow(arrhythmia)*0.8) 

train_set = arrhythmia[train, ]
test_set = arrhythmia[-train, ]
 
means_train = colMeans(train_set)
sds_train = matrixStats::colSds(as.matrix(train_set))
# scale train set
train_set[,-1] = scale(train_set[,-1])
colSums(is.na(train_set))
# scale trst set
test_set[,-1] = scale(test_set[,-1], center=means_train[-1], scale=sds_train[-1])
train_set['const'] = 1
test_set['const'] = 1

# glm
glm.1 <- glm(y ~.-1, family=binomial(link='logit'), data=train_set) 
summary(glm.1)
sd(glm.1$coefficients)
############################################################################################
##### EXPERIMENTS - ESS vs Accuracy (Section S4.4)
############################################################################################
S = 100000 # 100k
n = 34 # 10%N
##### ESS vs Accuracy
# SGBD
step_sizes_barker_vanilla = c(0.04, 0.05, 0.07, 0.1, 0.15, 0.275, 0.5, 0.75, 1, 1.2)
barker.vanilla.lr = SGMCMC_logistic_regression(train_set, test_set, step_sizes_barker_vanilla, S, n, 
                                               stan_means=post_means_arrhythmia, stan_vars=post_vars_arrhythmia, saga=FALSE, 
                                               barker=TRUE, noise='bimodal', method='vanilla', map=FALSE, beta=0.9)

step_sizes_barker_corrected = c(0.04, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1)
barker.corrected.lr = SGMCMC_logistic_regression(train_set, test_set, step_sizes_barker_corrected, S, n, 
                                                 stan_means=post_means_arrhythmia, 
                                                 stan_vars=post_vars_arrhythmia, saga=FALSE, 
                                                 barker=TRUE, noise='bimodal', 
                                                 method='corrected', map=FALSE, beta=0.9)


barker.extreme.lr = SGMCMC_logistic_regression(train_set, test_set, step_sizes_barker_corrected, S, n, 
                                               stan_means=post_means_arrhythmia, stan_vars=post_vars_arrhythmia, saga=FALSE, 
                                               barker=TRUE, noise='bimodal', method='extreme', map=FALSE, beta=0.9)

# SGLD
step_sizes_langevin_vanilla = c(0.02, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3)
langevin.vanilla.lr = SGMCMC_logistic_regression(train_set, test_set, step_sizes_langevin_vanilla, S, n, 
                                                 stan_means=post_means_arrhythmia, stan_vars=post_vars_arrhythmia, saga=FALSE, 
                                                 barker=FALSE, noise='bimodal', method='vanilla', map=FALSE, beta=0.9)


step_sizes_langevin_corrected = c(0.02, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.275, 0.29)
langevin.corrected.lr = SGMCMC_logistic_regression(train_set, test_set, step_sizes_langevin_corrected, S, n, 
                                                   stan_means=post_means_arrhythmia, stan_vars=post_vars_arrhythmia, saga=FALSE, 
                                                   barker=FALSE, noise='bimodal', method='corrected', map=FALSE, beta=0.9)


langevin.extreme.lr = SGMCMC_logistic_regression(train_set, test_set, step_sizes_langevin_corrected, S, n, 
                                                 stan_means=post_means_arrhythmia, stan_vars=post_vars_arrhythmia, saga=FALSE, 
                                                 barker=FALSE, noise='bimodal', method='extreme', map=FALSE, beta=0.9)

# Fig. S8
# plots ess vs accuracy
x_lab = 'median_ESS'
plot_lr_ess_accuracy(x_lab, 'bias_mean_rel_mean', langevin.vanilla.lr, barker.vanilla.lr, 
                     langevin.corrected.lr, barker.corrected.lr, 
                     path='../../plots_generic/new/arr_ess_mean.jpeg',
                     save=0, y_lim=c(0, 4), x_lab_plot='ESS', y_lab_plot='bias mean', 
                     legend=T, lwd=3, x_lim=c(100, 4000))


plot_lr_ess_accuracy(x_lab, 'bias_var_rel_mean', langevin.vanilla.lr, barker.vanilla.lr, 
                     langevin.corrected.lr, barker.corrected.lr, 
                     path='../../plots_generic/new/arr_ess_var_x_long.jpeg',
                     save=0, y_lim=c(0, 6), x_lab_plot='ESS', y_lab_plot='bias var', 
                     legend=FALSE, lwd=3, x_lim=c(100, 4000))


# Fig. S9b
y_lab = 'll_test'
y_lab= 'bias_mean_rel_mean'
par(mar=c(4.5,5,1,1))
lwd=3
plot(barker.corrected.lr[,x_lab], barker.corrected.lr[,y_lab], 
     type='l', xlab='ESS', ylab='log-loss', col='blue3', main='', ylim=c(0.44, 0.57) ,
     xlim=c(500, 4000),
     cex.lab=2, cex.axis=2, lwd=lwd, lty=5)
points(barker.vanilla.lr[,x_lab], barker.vanilla.lr[,y_lab], type='l', col='blue', lwd=lwd, lty=3)
points(langevin.vanilla.lr[,x_lab], langevin.vanilla.lr[,y_lab], type='l', col='red', lwd=lwd, lty=3)
points(langevin.corrected.lr[,x_lab], langevin.corrected.lr[,y_lab], type='l', col='red3',lwd=lwd, lty=5)
points(barker.extreme.lr_2[,x_lab], barker.extreme.lr_2[,y_lab], type='l', col='blue4',lwd=lwd, lty=4)
points(langevin.extreme.lr_2[,x_lab], langevin.extreme.lr_2[,y_lab], type='l', col='red4',lwd=lwd, lty=4)
legend('topleft', legend=c("v-SGLD  ", "c-SGLD  ", "e-SGLD  ", "v-SGBD  ", "c-SGBD  ", "e-SGBD  "),
       col=c("red", "red3", "red4", "blue", "blue3",  "blue4"), lty=c(3,5,4,3,5,4), 
       cex=1.5, ncol=2)

############################################################################################
##### EXPERIMENTS - Single Chains vs Pred. Accuracy (Section S4.4)
############################################################################################
# params: S = 100k, n=34 (10%N)
S= 100000
n = 34
# init
set.seed(123)
theta_0 = rnorm(101)

# v-SGBD
sigma = 0.27
barker.log_reg.v = SGMCMC(train_set, S, n, theta_0, sigma, grad_L_logistic_regression,
                          grad_p_logistic_regression, compute_performance=TRUE, 
                          test_set=test_set,
                          predict_y=TRUE, 
                          performance_functions=
                            list('likelihood'=log_loss_log_reg, 
                                 'error'=function(x,y,type)(0), 
                                 'predict'=predict_log_reg))

barker.log_reg.v$lik_mcmc
ess_b_v <- effectiveSize(mcmc(barker.log_reg.v$thetas[(S/2+1):S,]))
median(ess_b_v) # 1048.2342

# c-SGBD
sigma = 0.22
barker.log_reg.c = SGMCMC(train_set, S, n, theta_0, sigma, grad_L_logistic_regression,
                          grad_p_logistic_regression, method='corrected',compute_performance=TRUE, test_set=test_set,
                          predict_y=TRUE, compute_performance_train=TRUE,
                          performance_functions=list('likelihood'=log_loss_log_reg, 'error'=function(x,y,type)(0), 'predict'=predict_log_reg))
ess_b_c<- effectiveSize(mcmc(barker.log_reg.c$thetas[(S/2+1):S,]))
median(ess_b_c)# 1027.8318

# e-SGBD
sigma = 0.07
barker.log_reg.e = SGMCMC(train_set, S, n, theta_0, sigma, grad_L_logistic_regression,
                          grad_p_logistic_regression, method='extreme',
                          compute_performance=TRUE, test_set=test_set,
                          predict_y=TRUE, performance_functions=
                            list('likelihood'=log_loss_log_reg,
                                 'error'=function(x,y,type)(0), 
                                 'predict'=predict_log_reg))
ess_b_e <- effectiveSize(mcmc(barker.log_reg.e$thetas[(S/2+1):S,]))
median(ess_b_e) # 1015.3848

# v-SGLD
sigma = 0.145
langevin.log_reg.v = SGMCMC(train_set, S, n, theta_0, sigma, grad_L_logistic_regression,
                            grad_p_logistic_regression, barker=FALSE, method='vanilla',compute_performance=TRUE, test_set=test_set,
                            predict_y=TRUE, 
                            performance_functions=list('likelihood'=log_loss_log_reg, 
                                                       'error'=function(x,y,type)(0), 
                                                       'predict'=predict_log_reg))
ess_l_v <- effectiveSize(mcmc(langevin.log_reg.v$thetas[(S/2+1):S,]))
median(ess_l_v) # 991.25799


# c-SGLD
sigma = 0.113
langevin.log_reg.c = SGMCMC(train_set, S, n, theta_0, sigma, grad_L_logistic_regression,
                            grad_p_logistic_regression, barker=FALSE, method='corrected',compute_performance=TRUE, test_set=test_set,
                            predict_y=TRUE, 
                            performance_functions=list('likelihood'=log_loss_log_reg, 'error'=function(x,y,type)(0), 'predict'=predict_log_reg))
ess_l_c <- effectiveSize(mcmc(langevin.log_reg.c$thetas[(S/2+1):S,]))
median(ess_l_c) # 1174.0115

# e-SGLD (SGD)
sigma = 0.090
langevin.log_reg.e = SGMCMC(train_set, S, n, theta_0, sigma, grad_L_logistic_regression,
                            grad_p_logistic_regression, barker=FALSE, method='extreme',compute_performance=TRUE, test_set=test_set,
                            predict_y=TRUE, 
                            performance_functions=list('likelihood'=log_loss_log_reg, 'error'=function(x,y,type)(0), 'predict'=predict_log_reg))
ess_l_e <- effectiveSize(mcmc(langevin.log_reg.e$thetas[(S/2+1):S,]))
median(ess_l_e) # 1032.0371


# fig. S9a
par(mar=c(4.5,5,1,1))
# log-lik MCMC
S = 100000
b = 1
lwd=3
plot(seq(S/2+1, S, by=100), b*barker.log_reg.v$lik_mcmc, col='blue', type='l', xlab='iteration', 
     ylab='log-loss', ylim=c(0.44, 0.59),  cex.lab=2, cex.axis=2, lwd=lwd, lty=3)
lines(seq(S/2+1, S, by=100), b*barker.log_reg.c$lik_mcmc, col='blue3', lwd=lwd, lty=5)
lines(seq(S/2+1, S, by=100), b*langevin.log_reg.v$lik_mcmc, col='red', lwd=lwd, lty=3)
lines(seq(S/2+1, S, by=100), b*langevin.log_reg.c$lik_mcmc, col='red3', lwd=lwd, lty=5)
lines(seq(S/2+1, S, by=100), b*langevin.log_reg.e$lik_mcmc, col='red4', lwd=lwd, lty=4)
lines(seq(S/2+1, S, by=100), b*barker.log_reg.e$lik_mcmc, col='blue4', lwd=lwd, lty=4)
legend('topright', legend=c("v-SGLD ", "c-SGLD ","e-SGLD ", "v-SGBD ", "c-SGBD ", "e-SGBD "),
       col=c("red", "red3", "red4", "blue", "blue3","blue4"), 
       lty=c(3,5,4,3,5,4), cex=1.5, ncol=2)

S
###########################################################################################
##### EXPERIMENTS - simulation of E[p_hat] (Sections 3.3 and S4.1)
###########################################################################################
# simulation of E[p_hat]
samples = barker.log_reg.v$thetas[100000,]

# Fig 1 and S3a
i = 63
xlim = 0.3
p_s_34.a = sim_p_hat_LR(1000, 34, i, samples,  train_set, grad_L_logistic_regression, grad_p_logistic_regression, xlim=xlim)

# Fig. S3b
xlim = 0.3
i = 34
p_s_34.b = sim_p_hat_LR(1000, 34, i, samples,  train_set, grad_L_logistic_regression, grad_p_logistic_regression, xlim=xlim)

# Fig S3c
i = 42
xlim = 0.3
p_s_34.c = sim_p_hat_LR(1000, 34, i, samples,  train_set, grad_L_logistic_regression, grad_p_logistic_regression, xlim=xlim)

# Fig S3d 
i = 15
xlim = 0.3
p_s_34.d = sim_p_hat_LR(1000, 34, i, samples,  train_set, grad_L_logistic_regression, grad_p_logistic_regression, xlim=xlim)
