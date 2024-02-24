# TOY EXAMPLES
# this script contains the code to replicate the toy examples (Sections 4.1 and S3-4.2)

setwd('toy_models')
source('toy_models_setup.R')

##################################################################################################
#### TOY EXAMPLE: skew-normal target distribution (Section S4.2)
##################################################################################################

# medium step size and medium noise
step_size_p = 0.1
sigma_p = 1
S = 200000
toy.skew.2.b = skew_normal_simulation(step_size_p, sigma_p, S)
toy.skew.2.b

# large step size and medium noise
step_size_p = 0.5
sigma_p = 1
S = 200000
toy.skew.2.c = skew_normal_simulation(step_size_p, sigma_p, S)
toy.skew.2.c

# combining small and large step sizes
y_var = 'bias_mean_rel'

# fig. 2 in the main
plot_toy_skew_simulation(toy.skew.2.b, toy.skew.2.c, lwd=3)
# fig. S2a in the supp
plot_toy_skew_simulation(toy.skew.2.b, toy.skew.2.c, corrected=TRUE, lwd=3)


# medium step size and large noise
step_size_p = 0.1
sigma_p = 10
S = 200000
toy.skew.3.b = skew_normal_simulation(step_size_p, sigma_p, S)

# large step size and large noise
step_size_p = 0.5
sigma_p = 10
S = 200000
toy.skew.3.c = skew_normal_simulation(step_size_p, sigma_p, S)

plot_toy_skew_simulation(toy.skew.3.b, toy.skew.3.c, lwd=3, legend=FALSE)

# fig. S4 in the supp
plot_toy_skew_simulation(toy.skew.3.b, toy.skew.3.c, lwd=3, corrected=TRUE, legend=FALSE)

## plots distributions (sigma_noise = 1 * sd_target, step_size = 10% and 50% sd target, theta_0 = mean target)
S = 200000
alpha = 5
alpha = 20 # shape parameter (controlls skewness of target)

# target quantities
mean_skew = 1 * alpha / (sqrt(1 + alpha**2)) * sqrt(2 / pi)
var_skew = 1 - 2*alpha**2/((1+alpha**2)*pi)

# hyperparams
step_size_1 = 0.1 * sqrt(var_skew)
step_size_2 = 0.5 * sqrt(var_skew)
sigma = 1 * sqrt(var_skew)
theta_0 = mean_skew

# small step size
barker.skew.v.1 = isotropic_noise_chain(alpha, r_barker, grad_skew_normal, step_size_1, sigma, S, theta_0, method='vanilla')
langevin.skew.v.1 = isotropic_noise_chain(alpha, r_sgld, grad_skew_normal, step_size_1, sigma, S, theta_0, method='vanilla')

# large step size
barker.skew.v.2 = isotropic_noise_chain(alpha, r_barker, grad_skew_normal, step_size_2, sigma, S, theta_0, method='vanilla')
langevin.skew.v.2 = isotropic_noise_chain(alpha, r_sgld, grad_skew_normal, step_size_2, sigma, S, theta_0, method='vanilla')

# target density
x = seq(mean_skew - 4*sqrt(var_skew), mean_skew + 4*sqrt(var_skew), length.out=10000)
den = dsn(x, alpha=alpha)

#5 0.1,  20 0.11
bw = 0.11
# fig. 2b in the main
plot(x, den, type='l', xlab=expression(theta), ylab='density', col='grey', xlim=c(-1, 3.5), cex.lab=3, 
     cex.axis=3)
polygon(x, den, col=adjustcolor("grey", alpha=0.25), border='grey')
lines(density(barker.skew.v.1[(S/2+1):S], bw=bw), col='blue', lty=3, lwd=3)
lines(density(langevin.skew.v.1[(S/2+1):S], bw=bw), col='red', lty=3, lwd=3)
lines(density(barker.skew.v.2[(S/2+1):S], bw=bw), col='blue', lty=5, lwd=3)
lines(density(langevin.skew.v.2[(S/2+1):S], bw=bw), col='red', lty=5, lwd=3)

##################################################################################################
##### TOY EXAMPLE: standard normal target with HT noise (Section S3)
##################################################################################################

# laplace distributed noise
step_size_p = 0.1
S = 200000
toy.ht.1.a = ht_noise_simulation(0.1, S, 'laplace', sigmas)

toy.ht.1.b = ht_noise_simulation(0.5, S, 'laplace', sigmas)

# Fig. S1a
plot_ht_noise_simulation(toy.ht.1.a, toy.ht.1.b, ylab=expression(paste('bias ', 95^(th)-q)),
                         col='bias_q_95', y_log_scale=F, save=F,
                         filename = 'ht_toy_1_q_95.png', legend=T)
# density estimates
sgld.toy.ht.1 <- isotropic_noise_normal_target_chain(r_sgld, 0.1, exp(1.5)-1, 
                                                     200000, 0, method='vanilla', 
                                                     noise='laplace')
sgld.toy.ht.2 <- isotropic_noise_normal_target_chain(r_sgld, 0.5, exp(1.5)-1, 
                                                     200000, 0, method='vanilla', 
                                                     noise='laplace')

sgbd.toy.ht.1 <- isotropic_noise_normal_target_chain(r_barker, 0.1, exp(1.5)-1, 
                                                     200000, 0, method='vanilla', 
                                                     noise='laplace')
sgbd.toy.ht.2 <- isotropic_noise_normal_target_chain(r_barker, 0.5, exp(1.5)-1, 
                                                     200000, 0, method='vanilla', 
                                                     noise='laplace')

bw = 0.1
x <- seq(-4.5, 4.5, length.out=10000)
target_d <- dnorm(x)
# Fig. S1b
par(mar=c(5.5,5.5,1.5,1))
plot(x, target_d, type='l', xlab=expression(theta), ylab='density', col='grey',
     xlim=c(-4, 4), cex.lab=2.5, cex.axis=2.5)
polygon(x, target_d, col=adjustcolor("grey", alpha=0.25), border='grey')
lines(density(sgbd.toy.ht.1[(S/2+1):S], bw=bw), col='blue', lty=3, lwd=3)
lines(density(sgld.toy.ht.1[(S/2+1):S], bw=bw), col='red', lty=3, lwd=3)
lines(density(sgbd.toy.ht.2[(S/2+1):S], bw=bw), col='blue', lty=5, lwd=3)
lines(density(sgld.toy.ht.2[(S/2+1):S], bw=bw), col='red', lty=5, lwd=3)

# cauchy distributed noise

toy.ht.2.a = ht_noise_simulation(0.1, S, 'cauchy')

toy.ht.2.b = ht_noise_simulation(0.5, S, 'cauchy', sigmas)

# Fig. S2a
plot_ht_noise_simulation(toy.ht.2.a, toy.ht.2.b, ylab=expression(paste('bias ', 95^(th)-q)),
                         col='bias_q_95', y_log_scale=F, save=F,
                         filename = 'ht_toy_2_q_95.png', legend=F)

# density estimates
sgld.toy.ht.3 <- isotropic_noise_normal_target_chain(r_sgld, 0.1, exp(1.5)-1, 
                                                     200000, 0, method='vanilla', 
                                                     noise='cauchy')
sgld.toy.ht.4 <- isotropic_noise_normal_target_chain(r_sgld, 0.5, exp(1.5)-1, 
                                                     200000, 0, method='vanilla', 
                                                     noise='cauchy')

sgbd.toy.ht.3 <- isotropic_noise_normal_target_chain(r_barker, 0.1, exp(1.5)-1, 
                                                     200000, 0, method='vanilla', 
                                                     noise='cauchy')
sgbd.toy.ht.4 <- isotropic_noise_normal_target_chain(r_barker, 0.5, exp(1.5)-1, 
                                                     200000, 0, method='vanilla', 
                                                     noise='cauchy')
par(mar=c(5.5,5.5,1.5,1))
# Fig. S2b
plot(x, target_d, type='l', xlab=expression(theta), ylab='density', col='grey', 
     xlim=c(-4, 4), cex.lab=2.5, cex.axis=2.5)
polygon(x, target_d, col=adjustcolor("grey", alpha=0.25), border='grey')
lines(density(sgbd.toy.ht.3[(S/2+1):S], bw='SJ'), col='blue', lty=3, lwd=3)
lines(density(sgld.toy.ht.3[(S/2+1):S]), col='red', lty=3, lwd=3)
lines(density(sgbd.toy.ht.4[(S/2+1):S], bw='SJ'), col='blue', lty=5, lwd=3)
lines(density(sgld.toy.ht.4[(S/2+1):S], bw=0.1), col='red', lty=5, lwd=3)