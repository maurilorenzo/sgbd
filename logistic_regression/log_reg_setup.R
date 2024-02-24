# this script contains helper functions for experiments with the Logistic Regression Model (Sections 4.2 and S4.1, S4.3, S4.4)
library(coda)
library(DALEX)
library(mvtnorm)
library(sBIC)
library(ISLR)
library(rstan)

source("../utils.R")


# grad log - likelihood
grad_L_logistic_regression = function(data, theta){
  # it returns the gradient of the log likelihood of the logistic regression model 
  # args:
  # theta (vector d * 1) = parameter vector of the model
  # data = dataset: data$y (vector n * 1) = response variable; data$X (matrix n * d) = covariates
  # return: 
  # res (vector d * 1) = gradient of the log likelihood
  #print(dim(as.matrix(data[-1])))
  if (is.null(dim(data))){
    res =  t(data[-1]) * (data[1]- 1/ (1 + exp(-sum(data[-1]*theta))))
    return(res)
  }
  
  res = t(as.matrix(data[,-1])) %*% (as.matrix(data[,1])- 1 
                                    / (1 + exp(-as.matrix(data[,-1]) %*% theta)))
  return(res)
}

grad_L_logistic_regression_alt= function(data, theta){
  # it returns the gradient of the log likelihood of the logistic regression model 
  # args:
  # theta (vector d * 1) = parameter vector of the model
  # data = dataset: data$y (vector n * 1) = response variable; data$X (matrix n * d) = covariates
  # return: 
  # res (vector d * 1) = gradient of the log likelihood
  #data = as.matrix(data_df)
  #print(data)
  print(dim(data[,-1]))
  res = t(data[,-1]) %*% ((data[,1]- exp(data[,-1] %*% theta) 
                                     / (1 + exp(data[,-1] %*% theta))))
  return(res)
}





# grad log - prior
grad_p_logistic_regression = function(theta, d=100, tau=1) {
  
  # it return the gradient log prior (mutlivariate normal with diagonal covariance)
  # the variance of each component is 100/d
  
  # PARAMS:
  # theta (vector K * 1) = parameter vector
  # d - int, factor to normalize variance
  # OUTPUT:
  # gradient of the log prior (vector K * 1)
  
  #c = d/100
  return(-1/tau*theta)
}

# grad log - posterior
gradLogPost = function(data, theta) {
  # it returns the gradient of the log posterior of the logistic regression model with normal prior
  # PARAMS:
  # theta (vector d * 1) = parameter vector of the model
  # data = dataset: data$y (vector n * 1) = response variable; data$X (matrix n * d) = covariates
  # OUTPUT: 
  # res (vector d * 1) = gradient of the log posterior
  res = gradLogP(theta) + gradLogL(data, theta)
  return(res)
}




log_loss_log_reg = function(data, theta, type='sample'){
  # this function compute the log loss of the Logistic Regression
  # PARAMS:
  # data (data.frame N by (d+1)) = held out dataset
  # theta (d-dimesional vector) = parameters
  # OUTPUT:
  # logLoss (scalar)
  #N = dim(data)[1]
  #x_theta = rowSums(t(theta * t(data[names(data)!='y'])))
  #print(length(x_theta))
  #p = (1 + exp(- x_theta)) ^ (-1)
  if (type=='sample'){p_hat = data$pred}
  else{p_hat = data$pred_mcmc}
  log_loss = -mean(data$y*log(p_hat) + (1-data$y)*(log(1-p_hat)))
  return(log_loss)
}

x_theta = function(df, x){
  return(rowSums(t(x * t(df[names(df)!='y']))))
}

log_lossLR = function(data, samples){
  N = dim(data)[1]
  x_t = apply(samples, 1, function(x) x_theta(data, x))
  ps = (1 + exp(- x_t)) ^ (-1)
  print(dim(ps))
  p = rowMeans(ps)
  print(length(p))
  #ll = prod((p ^ (data$y)) * ((1 - p) ^ (1 - data$y)))
  logLoss = -mean(data$y*log(p) + (1-data$y)*(log(1-p)))
  return(logLoss)
}

log_loss_log_reg_sample = function(data, theta){
  p_hat = p_hat_log_reg(data, theta)
  log_loss = -(sum(log(p_hat[data$y==1]))+sum(log(1-p_hat[data$y==0])))/dim(data)[1]
  return(log_loss)
}

p_hat_log_reg = function(df, theta){
  d = length(theta)
  data = as.matrix(df[! names(df)%in% c('y', 'pred', 'pred_mcmc')])
  data = data[, 1:d]
  x_t =  data %*% theta
  p_hat = (1 + exp(- x_t)) ^ (-1)
  return(p_hat)
}

predict_log_reg = function(df, theta){
  df$pred = p_hat_log_reg(df, theta)
  return(df)
}


avgLogLossLR = function(samples, data){
  # this function compute the average log loss using the last 1000 MCMC samples
  # PARAMS:
  # data (data.frame N by (d+1)) = held out dataset
  # samples (matrix N by d) = MCMC samples of the parameters
  # OUTPUT:
  # res (scalar)
  N = dim(samples)[1]
  res = apply(samples, 1, function(x) logLossLR(data, x))
  res = mean(res)
  return(res)
}


sim_logistic_regression = function(dataset, held_out_test, 
                                   sigma_2s_naive, sigma_2s_extreme, sigma_2s_langevin, 
                                   S, n, stan_means=0, stan_vars=0, saga=FALSE, 
                                   sigma_2s_naive_saga=c(0.0001), sigma_2s_extreme_saga=c(0.0001)){
  # helper function used to simulate the sgld and stochastic barker with 
  # different value of the step size and compute the ksd, minimum ess and 
  # the log loss on a held out set
  set.seed(123)
  print(stan_means[1])
  
  d = ncol(dataset) - 1
  # starting point of the algorithm
  theta0 = rnorm(d, 0, 1)
  
  # grad log prior
  grad_prior = function(x){
  return(grad_p_logistic_regression(x, d))
      }
  
  
  res_barker_naive = data.frame(t(apply(matrix(sigma_2s_naive), 1, function(x) 
    (mcmc_logistic_regression(x, S, n, theta0, dataset, held_out_test, 
                              grad_L_logistic_regression, grad_p_logistic_regression, 
                              noise='bimodal', method='naive',
                              stan_means=stan_means, stan_vars=stan_vars, map=FALSE)))))
  
  res_langevin = data.frame(t(apply(matrix(sigma_2s_langevin), 1, function(x) 
    (mcmc_logistic_regression(x, S, n, theta0, dataset, held_out_test, 
                              grad_L_logistic_regression, grad_p_logistic_regression, 
                              noise='bimodal', method='naive',
                              stan_means=stan_means, stan_vars=stan_vars, map=FALSE, barker=FALSE)))))
  
  
  res_barker_extreme = data.frame(t(apply(matrix(sigma_2s_extreme), 1, function(x) 
    (mcmc_logistic_regression(x, S, n, theta0, dataset, held_out_test, 
                              grad_L_logistic_regression, grad_p_logistic_regression,
                              noise='bimodal', method='extreme',
                              stan_means=stan_means, stan_vars=stan_vars, map=FALSE)))))
  
  col_names = c('step_size', 'log_loss_test', 'log_loss_train', 'min_ESS', 
                'median_ESS', 'bias_mean', 'bias_mean_max','bias_mean_relative',
                'bias_var', 'bias_var_max')
  
  colnames(res_barker_naive) = col_names
  colnames(res_barker_langevin) = col_names
  colnames(res_barker_extreme) = col_names
  
  print(res_barker_naive)
  print(res_barker_extreme)
  print(res_langevin)
  res = list("res_langevin"=res_langevin, "res_barker_naive"=res_barker_naive, "res_barker_extreme"=res_barker_extreme)
  if (saga){
    res_barker_naive_saga = data.frame(t(apply(matrix(sigma_2s_naive_saga), 1, function(x) 
      (mcmc_logistic_regression(x, S, n, theta0, dataset, held_out_test, 
                                grad_L_logistic_regression, grad_p_logistic_regression, 
                                noise='bimodal', method='vanilla',
                                stan_means=stan_means, stan_vars=stan_vars, saga=TRUE)))))
    
    
    res_barker_extreme_saga = data.frame(t(apply(matrix(sigma_2s_extreme_saga), 1, function(x) 
      (mcmc_logistic_regression(x, S, n, theta0, dataset, held_out_test, 
                                grad_L_logistic_regression, grad_p_logistic_regression,
                                noise='bimodal', method='extreme',
                                stan_means=stan_means, stan_vars=stan_vars, saga=TRUE)))))
    
    colnames(res_barker_naive_saga) = col_names
    colnames(res_barker_extreme_saga) = col_names
    
    print(res_barker_naive_saga)
    print(res_barker_extreme_saga)
    
    res = list("res_barker_naive"=res_barker_naive, "res_barker_extreme"=res_barker_extreme,
               "res_barker_naive_saga"=res_barker_naive_saga, "res_barker_extreme_saga"=res_barker_extreme_saga)
    
    plot_logistic_regression(res, max_=TRUE, saga=TRUE)
    return(res)
  }

  plot_logistic_regression(res, max_=TRUE)
  return(res)
}


SGMCMC_logistic_regression = function(dataset, held_out_test, step_sizes, S, n, 
                                      stan_means=0, stan_vars=0, saga=FALSE, barker=TRUE, noise='bimodal',
                                      method='vanilla', map=FALSE, beta=0.9, sigma_noise=1,
                                      eta=0.001, tau=1){
  # helper function used to simulate the sgld and stochastic barker with 
  # different value of the step size and compute the ksd, minimum ess and 
  # the log loss on a held out set
  set.seed(123)
  print(stan_means[1])
  
  d = ncol(dataset) - 1
  # starting point of the algorithm
  theta0 = rnorm(d, 0, sigma_noise)
  
  # grad log prior
  grad_prior = function(x){
    return(grad_p_logistic_regression(x, d, tau))
  }
  
  
  res_SGMCMC = data.frame(t(apply(matrix(step_sizes), 1, function(x) 
    (mcmc_logistic_regression(x, S, n, theta0, dataset, held_out_test, 
                              grad_L_logistic_regression, grad_prior, 
                              noise=noise, method=method, stan_means=stan_means,
                              stan_vars=stan_vars, map=map, beta=beta, barker=barker, eta=eta)))))
  
  
  col_names = c('step_size', 'll_test', 'll_train', 'min_ESS', 
                'median_ESS', 'bias_mean', 'bias_mean_max','bias_mean_rel_mean', 
                'bias_mean_rel_median','bias_var', 'bias_var_max', 
                'bias_var_rel_mean', 'bias_var_rel_median')
  
  colnames(res_SGMCMC) = col_names
  print(res_SGMCMC)
  
  return(res_SGMCMC)
}



mcmc_logistic_regression = function(sigma, S, n, theta, dataset, 
                                    held_out_test, grad_L, grad_p, 
                                    noise='bimodal', method='vanilla',
                                    stan_means=0, stan_vars=0, saga=FALSE, map=FALSE, barker=TRUE, beta=0.9, eta=0.001){
  
  options(digits=8)
  print(sigma)
  
  set.seed(123)
  samples = SGMCMC(dataset, S, n, theta, sigma, grad_L, grad_p, noise, 
                         method, saga=saga, map=map, barker=barker, beta=beta, eta=eta)

  
  if (any(is.na(samples))){
    print("NA values")
    return(c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
  }
  # discard burn-in
  if (! map) {
    samples = samples[(S/2+1):S,]
  }
  # log-loss
  LL_test = log_lossLR(held_out_test, samples[seq(100, S/2, by=100),])
  LL_train = log_lossLR(dataset, samples[seq(100, S/2, by=100),])
  
  # effective samples size
  ESS_samples = ESS(samples)
  min_ESS = min(ESS_samples)
  median_ESS = median(ESS_samples)
  
  # samples bias
  means = colMeans(samples)
  vars = colSds(samples)^2
  
  bias_mean_samples= mean(abs(means - stan_means))
  bias_mean_max_samples = max(abs(means - stan_means))
  bias_mean_relative_samples= mean(abs(means - stan_means)/sqrt(stan_vars))
  bias_mean_relative_samples_median = median(abs(means - stan_means)/sqrt(stan_vars))
  bias_var_samples = mean(abs(vars - stan_vars))
  bias_var_max_samples = max(abs(vars - stan_vars))
  bias_var_relative_samples_mean = mean(abs(vars - stan_vars)/sqrt(stan_vars))
  bias_var_relative_samples_median = median(abs(vars - stan_vars)/sqrt(stan_vars))
  
  
  print(c(sigma, LL_test, LL_train, min_ESS, median_ESS, bias_mean_samples, bias_mean_max_samples, 
          bias_mean_relative_samples, bias_mean_relative_samples_median, 
          bias_var_samples, bias_var_max_samples, bias_var_relative_samples_mean, bias_var_relative_samples_median))
  return(c(sigma, LL_test, LL_train, min_ESS, median_ESS, bias_mean_samples, 
           bias_mean_max_samples, bias_mean_relative_samples, bias_mean_relative_samples_median, 
           bias_var_samples, bias_var_max_samples, bias_var_relative_samples_mean, 
           bias_var_relative_samples_median))
}


plot_logistic_regression_bias = function(res, col_x, col_y, x_lab, y_lab, saga=FALSE){
  x_max = max(c(max(res$res_barker_naive[col_x]), max(res$res_barker_extreme[col_x])))
  y_max = max(c(max(res$res_barker_naive[col_y]), max(res$res_barker_extreme[col_y])))
  plot(res$res_barker_naive[,col_x], res$res_barker_naive[,col_y],
       xlab = x_lab, ylab = y_lab, xlim = c(0, x_max), ylim = c(0, y_max), type = 'o')
  points(res$res_barker_extreme[,col_x], res$res_barker_extreme[,col_y], col = 'dark blue', type = 'o')
  points(res$res_langevin[,col_x], res$res_langevin[,col_y], col = 'red', type = 'o')
  
  if (saga) {
    points(res$res_barker_extreme_saga[,col_x], res$res_barker_extreme_saga[,col_y], col = 'red4', type = 'o')
    points(res$res_barker_naive_saga[,col_x], res$res_barker_naive_saga[,col_y], col = 'blue4', type = 'o')
    
  }
} 
                                         

plot_logistic_regression = function(res, max_=FALSE, saga=FALSE){
  
  par(mfrow = c(1,1))
  
  plot_logistic_regression_bias(res,col_x='min_ESS', col_y='bias_mean',
                                x_lab='Min ESS', y_lab='Bias mean', saga)
  
  plot_logistic_regression_bias(res, col_x='median_ESS', col_y='bias_mean',
                                x_lab='Median ESS', y_lab='Bias mean', saga)
    
  plot_logistic_regression_bias(res, col_x='min_ESS', col_y='bias_var',
                                x_lab='Min ESS', y_lab='Bias var', saga)
  
  plot_logistic_regression_bias(res, col_x='median_ESS', col_y='bias_var',
                                x_lab='Median ESS', y_lab='Bias var', saga)
  
  if(max_ == TRUE){
    
    plot_logistic_regression_bias(res, col_x='min_ESS', col_y='bias_mean_max',
                                  x_lab='Min ESS', y_lab='Bias mean (max)', saga)
    
    plot_logistic_regression_bias(res, col_x='median_ESS', col_y='bias_mean_max',
                                  x_lab='Median ESS', y_lab='Bias mean (max)', saga)
    
    plot_logistic_regression_bias(res, col_x='min_ESS', col_y='bias_var_max',
                                  x_lab='Min ESS', y_lab='Bias var (max)', saga)
    
    plot_logistic_regression_bias(res, col_x='median_ESS', col_y='bias_var_max',
                                  x_lab='Median ESS', y_lab='Bias var (max)', saga)
  }
  
}

plot_trace_LR = function(idx, sgldLR, barkerLR, barkerLR_alternative, list_of_draws, stan_means, ylim = c(-10, 10), legend=TRUE){
  par(mfrow = c(1,1))
  plot(barkerLR[,idx], xlab = 'Iteration', ylab = paste(bquote(theta), idx), type = 'l', ylim = ylim)
  lines(barkerLR_alternative[,idx], col='blue')
  lines(sgldLR[,idx], col='red')
  lines(seq(50001, 100000), list_of_draws$beta[,idx], col=rgb(0, 0.39, 0, alpha = 0.5))
  abline(h=stan_means[idx], col = "dark green", lwd = 5)
  if (legend){
    legend('bottomright', legend=c("barker", "barker.2","sgld", "stan"),
         col=c("black", "blue","red", "dark green"), lty=19:19:19:19, cex=0.8)
  }
}

plot_3d = function(samples, idx_1=1, idx_2=2){
  x_c <- cut(samples[,idx_1], 50)
  y_c <- cut(samples[,idx_2], 50)
  z <- table(x_c, y_c)
  image2D(z=z, border="black")
}

histogram_combined_LR = function(idx, stan, sgld, barker, barker_alt, xlim, ylim=c(0,1), S=100000){
  dev.new()
  par(mfrow = c(1,3))
  hist(sgld[(S/2+1):S,idx], breaks=40, probability=TRUE, main = 'SGLD', xlab = 'theta', xlim = xlim, ylim=ylim)
  lines(density(stan$beta[,idx]), col='green')
  
  hist(barker[(S/2+1):S,idx], breaks=40, probability=TRUE, main = 'Barker', xlab = 'theta',  xlim = xlim, ylim=ylim)
  lines(density(stan$beta[,idx]), col='green')
  
  hist(barker_alt[(S/2+1):S,idx], breaks=40, probability=TRUE, main = 'Barker 2', xlab = 'theta',  xlim = xlim, ylim=ylim)
  lines(density(stan$beta[,idx]), col='green')
  par(mfrow = c(1,1))
}



correction_fn = function(sigma, sds){
  res = rep(9999999, length(sds))
  mask = ((sigma^2*sds^2)<3)
  sigma_sds = sigma*sds
  res[mask] = sqrt(3/(3-sigma_sds[mask]^2))
  return(res)
}



sim_p_hat_LR = function(S, mini_batch_size, j, theta, data, gradLogL, 
                        gradLogP, xlim=3, file_name='plot_p_hat', legend=F,
                        breaking_pt=F){
  
  grads = rep(0, S)
  p_matrix = matrix(0, ncol=1000, nrow=S)
  p_corrected_matrix = matrix(0, ncol=1000, nrow=S)
  p_hat_extreme_matrix = matrix(0, ncol=1000, nrow=S)
  grad_vector = rep(0, S)
  z = seq(-xlim, xlim, length.out=1000)
  N = dim(data)[1]
  
  grad_full = gradLogL(data, theta) + gradLogP(theta)
  grad_full = grad_full[j]
  print("Gradient using full dataset:")
  print(grad_full)
  
  p_s = p_hat(z, grad_full)
  
  set.seed(123)
  for(i in seq(1, S)){
    mini_batch = sample(N, mini_batch_size)
    mini_batch = data[mini_batch,]
    grad = N/mini_batch_size * gradLogL(mini_batch, theta) + gradLogP(theta)
    grad = grad[j]
    grad_vector[i] = grad
  }
  
  hist(grad_vector, freq=FALSE, breaks=40, main='', xlab='stochastic gradient')
  
  grad_sd = sd(grad_vector)
  correction = correction_fn(grad_sd, z)
  print("Noise SD")
  print(grad_sd)
  
  for(i in seq(1, S)){
    grad_estimate = grad_vector[i]
    p_hat_s = p_hat(z, grad_estimate)
    p_matrix[i, ] = p_hat_s
    p_hat_corrected_s = p_hat(z*correction, grad_estimate)
    p_corrected_matrix[i, ] = p_hat_corrected_s
    p_hat_extreme = p_hat(z, grad_estimate, method='extreme')
    p_hat_extreme_matrix[i,] = p_hat_extreme
  }
  
  p_hats = colMeans(p_matrix)
  p_hats_corrected = colMeans(p_corrected_matrix)
  p_hats_extreme = colMeans(p_hat_extreme_matrix)
  p_q_1 = colQuantiles(p_matrix, probs=0.25)
  p_q_3 = colQuantiles(p_matrix, probs=0.75)
  print(mean(grad_vector))
  
  jpeg(file_name, width = 800, height = 533)
  par(mar=c(5,5.5,1.5,1))
  plot(z, p_s, main='', col='grey4', type='l', ylab='p', xlim=c(-xlim, xlim), 
       cex.lab=2, cex.axis=2, lwd=3)
  lines(z, p_hats, col='blue', lty=3, lwd=3)
  lines(z, p_hats_corrected, col='blue4', lty=5, lwd=3)
  if(breaking_pt){
    z_tilde <- 1.702/grad_sd
    abline(v=z_tilde, col='red', lwd=2)
    abline(v=-z_tilde, col='red', lwd=2)
  }
  if(legend){
    legend('topright', legend=c("p", expression(paste(bold("E"), ' ',hat('p'), '  ')), 
                                expression(paste(bold("E"), ' ',tilde('p'), '  '))),
           col=c("black", "blue", "blue4"), lty=c(1, 3, 5), cex=2)
  }
  dev.off()
  return(list('p'=p_s, 'p_hat'=p_hats, 'p_hat_corrected'=p_hats_corrected, 'p_hat_extreme'=p_hats_extreme))
}



plot_marginal_distributions_lr = function(idx, stan_samples, barker.1, langevin.1, 
                                          barker.2, langevin.2, name_output=NULL, bws, save=F, legend=F){
  if(save){jpeg(paste('hist_theta_', idx, '.png'), width = 800, height = 533)}
  den = density(stan_samples[,idx], bw=bws[idx])
  par(mar=c(5.5,5.5,1.5,1))
  plot(den, col='grey', main='', xlab=bquote(~theta[.(idx)]), ylab='density',
       cex.lab=2, cex.axis=2)
  polygon(den, col=adjustcolor("grey", alpha=0.25), border='grey')
  lines(density(langevin.1[, idx], bw=bws[idx]), col='red', lty=3, lwd=2)
  lines(density(barker.1[, idx], bw=bws[idx]), col='blue', lty=3, lwd=2)
  lines(density(langevin.2[, idx], bw=bws[idx]), col='red', lty=5, lwd=2)
  lines(density(barker.2[, idx], bw=bws[idx]), col='blue', lty=5, lwd=2)
  if(idx==1 & legend){
    legend('topleft', legend=c(expression(paste("SGLD-", sigma[1], '  ')), 
                               expression(paste("SGLD-", sigma[2],'  ')),
                               expression(paste("SGBD-", sigma[1], '  ')),
                               expression(paste("SGBD-", sigma[2], '  '))),
           col=c("red", "red3", "blue", "blue3"), ncol=2,
           lty=c(3,5,3,5), cex=1.75)
  }
  if(save){dev.off()}
}

trace_plot_lr = function(idx, stan_samples, sg_samples, col='blue', lty=1){
  mean = mean(stan_samples[,idx])
  sd = std(stan_samples[,idx])
  par(mar=c(5,6,4,2)+.1)
  plot(sg_samples[,idx], col=col, lty=lty, main='', xlab='iteration', 
       ylab=bquote(~theta[.(idx)]), type='l', cex.lab=2, cex.axis=2)
  abline(h=mean+2*sd)
  abline(h=mean-2*sd)
}

plot_lr_ess_accuracy = function(x_lab, y_lab, langevin_v, barker_v, langevin_c, barker_c,
                                main='', path='fig1.jpg', y_lim=c(0, 2), 
                                save=TRUE, legend=FALSE, x_lab_plot, y_lab_plot, 
                                lwd=2, x_lim=c(1000, 5000)){
  if (save){jpeg(path, width = 800, height = 533)}
  par(mar=c(5.5,5.5,1.5,1))
  plot(langevin_v[,x_lab], langevin_v[,y_lab], type='l', xlab=x_lab_plot, 
       ylab=y_lab_plot, col='red', main=main, ylim=y_lim, cex.lab=2.5, cex.axis=2.5,
       lwd=lwd, lty=3,xlim=x_lim)
  points(barker_v[,x_lab], barker_v[,y_lab], type='l', col='blue', lty=3, lwd=lwd)
  points(langevin_c[,x_lab], langevin_c[,y_lab], type='l', col='red3', lty=5, lwd=lwd)
  points(barker_c[,x_lab], barker_c[,y_lab], type='l', col='blue3', lty=5, lwd=lwd)
  if (legend) {legend('topleft', legend=c("v-SGLD ", "c-SGLD ", "v-SGBD", "c-SGBD "),
                      col=c("red", "red4", "blue", "blue4"), lty=c(3,5,3,5), cex=1.75, ncol=1)}
  if (save) {dev.off()}
}


plot_univariate_distribution_lr = function(idx, betas_stan, stan_means, stan_vars, barker.v, langevin.v, barker.c, langevin.c){
  par(mfrow=c(1,1))
  mean = stan_means[idx]
  xlim_1 = mean - 3
  xlim_2 = mean + 3
  hist(betas_stan[,idx], freq=FALSE, xlim=c(xlim_1, xlim_2), main=idx, xlab='')
  lines(density(betas_stan[,idx]), col='black')
  lines(seq(-5, 5, by=.01), dnorm(seq(-5, 5, by=.01), stan_means[idx], sqrt(stan_vars[idx])), col='green')
  lines(density(langevin.v[(S/2+1):S, idx]), col='red')
  lines(density(barker.v[(S/2+1):S, idx]), col='blue')
  lines(density(langevin.c[(S/2+1):S, idx]), col='red4')
  lines(density(barker.c[(S/2+1):S, idx]), col='blue4')
}


betaPrior = function(d, tau=1) {
  return(tau*diag(rep(1, d)))
}

# MAMBA-utils
imq_kernel <- function(x, y){
  c <- 1
  beta <- -0.5
  return((c + sum((x-y)^2))^beta)
}

D_imq_kernel <- function(x, y){
  c <- 1
  beta <- -0.5
  return((y-x)*(c + sum((x-y)^2))^(beta-1))
}

FSSD_naive <- function(samples, grads, V){
  N <- dim(samples)[1]
  d <- dim(samples)[2]
  J <- dim(V)[1]
  g_sum <- 0
  for(i in 1:J){
    vi <- V[i,]
    K_s <- apply(samples, 1, function(x) (imq_kernel(x, vi))) # N*d
    K_s <- matrix(rep(K_s, dim(samples)[2]), ncol=dim(samples)[2])
    D_K_s <- t(apply(samples, 1, function(x) (D_imq_kernel(x, vi)))) # N*d
    g_s <- grads * K_s + D_K_s
    g <- colMeans(g_s)
    g_bar <- sum(g^2)/d
    g_sum <- g_sum + g_bar
  }
  g_sum <- g_sum / J
  g_res <- sqrt(g_sum)
  return(g_res)
}



get_test_locations <- function(samples, J){
  d <- dim(samples)[2]
  mean_n <- colMeans(samples)
  cov_n <- cov(samples) + diag(0.00001, d, d)
  vs <- mvrnorm(J, mean_n, cov_n)
  return(vs)
}

compute_grad_log_reg <- function(samples, data){
  grads_l <- t(apply(samples, 1, function(x) (grad_L_logistic_regression(data, x))))
  grads_p <- t(apply(samples, 1, function(x) (grad_p_logistic_regression(x))))
  return(grads_l + grads_p)
}

compute_FSSD <- function(samples, data, J=100){
  grads <- compute_grad_log_reg(samples, data)
  V <- get_test_locations(samples, J)
  fssd <- FSSD_naive(samples, grads, V)
  return(fssd)
}
