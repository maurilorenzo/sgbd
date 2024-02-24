# this scirpt contains helper functions for the toy simulations
setwd('toy_models')
library(extraDistr)
library(sn)
source("../utils.R")


#################################################################################################
###### 1d skew normal
#################################################################################################

grad_skew_normal = function(theta, alpha){
  return(-theta + alpha * exp(log(dnorm(alpha * theta)) - log(pnorm(alpha * theta))))
}

isotropic_noise_chain = function(alpha, algorithm, grad_log_target, h, sigma, S, theta, method='vanilla'){
  '
  at each iter gaussian isotropic noise is added to the true gradient of the target
  '
  set.seed(123)
  d = length(theta)
  samples = matrix(NA, ncol=d, nrow=S)
  samples[1,] = theta
  for (i in seq(2, S)){
    grad = grad_log_target(theta, alpha) + sigma * rnorm(d)
    theta = algorithm(theta, grad, h, method=method, tau=sigma)
    samples[i,] = theta
  }
  return(samples)
}



isotropic_noise_simulation = function(algorithm, step_size_p, sigma_p, S, method='vanilla', alpha=1){
  
  mean_skew = 1 * alpha / (sqrt(1 + alpha**2)) * sqrt(2 / pi)
  var_skew = 1 - 2*alpha**2/((1+alpha**2)*pi)
  
  step_size = step_size_p * sqrt(var_skew)
  sigma = sigma_p * sqrt(var_skew)
  
  theta_0 = mean_skew
  
  chain = isotropic_noise_chain(alpha, algorithm, grad_skew_normal, step_size, sigma, S, theta_0, method)
  chain = chain[(floor(S/2)+1):S,] # remove burn-in
  
  mean_samples = mean(chain)
  std_samples = std(chain)
  ess_samples = ESS(chain)
  
  bias_mean = abs(mean_samples - mean_skew)
  bias_mean_rel = abs(mean_samples - mean_skew) / sqrt(var_skew)
  bias_var = abs(std_samples**2 - var_skew)
  bias_var_rel = abs(std_samples**2 - var_skew) / sqrt(var_skew)
  
  return(c(alpha, bias_mean, bias_mean_rel, bias_var, bias_var_rel, ess_samples, step_size, sigma))
}


skew_normal_simulation = function(step_size_p=0.05, noise_p=0.1, S, len=10){
  
  alphas = exp(seq(log(1.01), 4.6, length.out=len))
  
  res_sgld_vanilla = data.frame(t(apply(matrix(alphas), 1, function(x) (isotropic_noise_simulation(
    r_sgld, step_size_p, noise_p, S, method='vanilla', alpha=x)))))
  res_sgld_corrected = data.frame(t(apply(matrix(alphas), 1, function(x) (isotropic_noise_simulation(
    r_sgld, step_size_p, noise_p, S,  method='corrected', alpha=x)))))
  res_barker_vanilla = data.frame(t(apply(matrix(alphas), 1, function(x) (isotropic_noise_simulation(
    r_barker, step_size_p, noise_p, S, method='vanilla', alpha=x)))))
  res_barker_corrected = data.frame(t(apply(matrix(alphas), 1, function(x) (isotropic_noise_simulation(
    r_barker, step_size_p, noise_p, S, method='corrected', alpha=x)))))
 
  
  col_names = c("alpha", "bias_mean", "bias_mean_rel", "bias_var", "bias_var_rel", "ESS", "step_size", "sigma_noise")
  colnames(res_sgld_vanilla) = col_names
  colnames(res_sgld_corrected) = col_names
  colnames(res_barker_vanilla) = col_names
  colnames(res_barker_corrected) = col_names

  
  return(list('sgld.vanilla'=res_sgld_vanilla, 'sgld.corrected'=res_sgld_corrected, 
              'barker.vanilla'=res_barker_vanilla, 'barker.corrected'=res_barker_corrected))
}

ploy_toy_skew_simulation_bias = function(res_simulation, mean=TRUE, var=FALSE){
  if (mean){plot_toy_skew_simulation(res_simulation, col='bias_mean_rel')}
  if (var){plot_toy_skew_simulation(res_simulation, col='bias_var_rel')}
}

plot_toy_skew_simulation = function(res_simulation_1, res_simulation_2, col='bias_mean_rel', 
                                    corrected=FALSE, y_log_scale=FALSE, lwd=3, legend=TRUE){
  alphas = res_simulation_1$barker.vanilla[,'alpha']
  y_fn = function(x){
    if(y_log_scale){
      return(log(1+x))
    }
    return(x)
  }
  par(mar=c(4.5,5.5,1.5,1))
  plot(log(1+alphas), y_fn(res_simulation_1$sgld.vanilla[,col]), col='red', xlab=expression(paste("log(1+", alpha, ")")),
       ylab='bias mean', type='l', ylim=c(0,1), lty=3, cex.lab=2, cex.axis=2, lwd=lwd)
  lines(log(1+alphas), y_fn(res_simulation_1$barker.vanilla[,col]), col='blue', lty=3, lwd=lwd)
  lines(log(1+alphas), y_fn(res_simulation_2$sgld.vanilla[,col]), col='red', lty=5, lwd=lwd)
  lines(log(1+alphas), y_fn(res_simulation_2$barker.vanilla[,col]), col='blue', lty=5, lwd=lwd)
  if(corrected){
    lines(log(1+alphas), res_simulation_1$sgld.corrected[,col], col='red3', lty=3, lwd=lwd)
    lines(log(1+alphas), res_simulation_1$barker.corrected[,col], col='blue3', lty=3, lwd=lwd)
    lines(log(1+alphas), res_simulation_2$sgld.corrected[,col], col='red3', lty=5, lwd=lwd)
    lines(log(1+alphas), res_simulation_2$barker.corrected[,col], col='blue3', lty=5, lwd=lwd)
  }
  if (legend){
    legend('topleft', legend=c(expression(paste("SGLD-", sigma[1], '  ')), expression(paste("SGLD-", sigma[2],'  ')),
                                expression(paste("SGBD-", sigma[1], '  ')), expression(paste("SGBD-", sigma[2], '  '))),
            col=c("red", "red3", "blue", "blue3"), lty=c(3,5,3,5), cex=1.75, ncol=1)
  }
}

gen_data_skewed=function(sigma=1, n=100){
  set.seed(123)
  x = rhcauchy(n, sigma)
  y = rep(1, n)
  data = data.frame(cbind(y, x))
  colnames(data) = c('y', 'x')
  return(data)
}

#################################################################################################
##### 1-d log reg toy
#################################################################################################



grad_log_std_normal = function(theta){
  return(-theta)
}


isotropic_noise_normal_target_chain = function(algorithm, h,
                                               sigma, S, theta, method='vanilla', 
                                               noise='gaussian'){
  '
  at each iter ht noise isotropic noise is added to the true gradient of the target
  '
  set.seed(123)
  d = length(theta)
  samples = matrix(NA, ncol=d, nrow=S)
  samples[1,] = theta
  for (i in seq(2, S)){
    if(noise == 'gaussian'){
      eps = sigma * rnorm(d)
    }
    else if (noise == 'laplace'){
      eps = rlaplace(d, 0, sigma)
    }
    else{
      eps = rcauchy(d, 0, scale=sigma)
    }
    grad = -theta + eps
    theta = algorithm(theta, grad, h, method=method, tau=sigma)
    samples[i,] = theta
  }
  return(samples)
}

ht_isotropic_noise_simulation = function(algorithm, step_size_p, S=100000, 
                                         method='vanilla', sigma=1, noise='laplace'){
  step_size = step_size_p * 1
  theta_0 = 0
  
  chain = isotropic_noise_normal_target_chain(algorithm, step_size, sigma, S, theta_0, 
                                              method, noise=noise)
  chain = chain[(floor(S/2)+1):S,] # remove burn-in
  
  mean_samples = mean(chain)
  std_samples = std(chain)
  
  
  bias_mean = abs(mean_samples)
  bias_var = abs(std_samples**2 - 1)
  bias_q_60 = abs(qnorm(0.6) - quantile(chain, probs=0.6))
  bias_q_95 = abs(qnorm(0.95) - quantile(chain, probs=0.95))
  
  return(c(sigma, bias_mean, bias_var,bias_q_60, bias_q_95, step_size))
  
}

ht_noise_simulation = function(step_size_p=0.05, S=100000, noise='laplace',
                               sigmas=c(0.01, 0.1, 0.5, 1, 2, 5, 10)){
  
  res_sgld_vanilla = data.frame(t(apply(matrix(sigmas), 1, function(x) 
    (ht_isotropic_noise_simulation(
      r_sgld, step_size_p=step_size_p, S=S, method='vanilla', sigma=x, noise=noise)))))
  res_sgld_corrected = data.frame(t(apply(matrix(sigmas), 1, function(x)
    (ht_isotropic_noise_simulation(
      r_sgld, step_size_p,  S,  method='corrected', sigma=x, noise=noise)))))
  res_barker_vanilla = data.frame(t(apply(matrix(sigmas), 1, function(x)
    (ht_isotropic_noise_simulation(
      r_barker, step_size_p,  S, method='vanilla', sigma=x, noise=noise)))))
  res_barker_corrected = data.frame(t(apply(matrix(sigmas), 1, function(x)
    (ht_isotropic_noise_simulation(
      r_barker, step_size_p, S, method='corrected', sigma=x, noise=noise)))))
  
  
  col_names = c("sigma", "bias_mean", "bias_var", 
                "bias_q_60","bias_q_95","step_size")
  colnames(res_sgld_vanilla) = col_names
  colnames(res_sgld_corrected) = col_names
  colnames(res_barker_vanilla) = col_names
  colnames(res_barker_corrected) = col_names
  
  #plot_simulation_2(res_barker, res_barker_alternative, res_sgld, sigma, legend=FALSE)
  
  return(list('sgld.vanilla'=res_sgld_vanilla, 'sgld.corrected'=res_sgld_corrected, 
              'barker.vanilla'=res_barker_vanilla, 'barker.corrected'=res_barker_corrected))
}

ploy_ht_noise_simulation_bias = function(res_simulation, mean=TRUE, var=FALSE){
  if (mean){plot_ht_noise_simulation(res_simulation, col='bias_mean')}
  if (var){plot_ht_noise_simulation(res_simulation, col='bias_var')}
}

plot_ht_noise_simulation = function(res_simulation_1, res_simulation_2, col='bias_var', 
                                    corrected=FALSE, y_log_scale=FALSE, lwd=3, legend=TRUE, 
                                    ylab='bias var', save=F, filename='toy.png'){
  sigma = res_simulation_1$barker.vanilla[,'sigma']
  y_fn = function(x){
    if(y_log_scale){
      return(log(1+x))
    }
    return(x)
  }
  if(save){jpeg(filename, width = 800, height = 533)}
  par(mar=c(5.5, 6, 1.5, 1))
  plot(log(1+sigma), y_fn(res_simulation_1$sgld.vanilla[,col]), col='red', xlab=expression(paste("log(1+", tau[theta], ")")),
       ylab=ylab, type='l', ylim=c(0,3), lty=3, cex.lab=2.5, cex.axis=2.5, lwd=lwd)
  lines(log(1+sigma), y_fn(res_simulation_1$barker.vanilla[,col]), col='blue', lty=3, lwd=lwd)
  lines(log(1+sigma), y_fn(res_simulation_2$sgld.vanilla[,col]), col='red', lty=5, lwd=lwd)
  lines(log(1+sigma), y_fn(res_simulation_2$barker.vanilla[,col]), col='blue', lty=5, lwd=lwd)
  if(corrected){
    lines(log(1+sigma), res_simulation_1$sgld.corrected[,col], col='red3', lty=3, lwd=lwd)
    lines(log(1+sigma), res_simulation_1$barker.corrected[,col], col='blue3', lty=3, lwd=lwd)
    lines(log(1+sigma), res_simulation_2$sgld.corrected[,col], col='red3', lty=5, lwd=lwd)
    lines(log(1+sigma), res_simulation_2$barker.corrected[,col], col='blue3', lty=5, lwd=lwd)
  }
  if (legend){
    legend('topleft', legend=c(expression(paste("SGLD-", sigma[1], '  ')), expression(paste("SGLD-", sigma[2],'  ')),
                               expression(paste("SGBD-", sigma[1], '  ')), expression(paste("SGBD-", sigma[2], '  '))),
           col=c("red", "red3", "blue", "blue3"), lty=c(3,5,3,5), cex=1.75, ncol=1)
  }
  if(save){dev.off()}
}

