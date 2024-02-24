# this script contains helper functions for ICA simulations
library(fastICA)
library(readr)

source("../utils.R")


grad_log_prior_ica = function(W, lambda=1){
  return(-lambda * W)
}

grad_log_likelihood_ica_i = function(x, W, lambda=1){
  # computes the data component of the gradient if the log likelihood of a single instance x
  y = W %*% x# d * 1
  Y = tanh(0.5*y)
  grad_l = Y %*% x # d * d
  return(matrix(grad_l))
}

grad_log_likelihood_ica = function(X, W, lambda=1, d=10){
  # computes gradient of log likelihood
  if (is.null(dim(X))){
    return(t(inv(W)) - matrix(grad_log_likelihood_ica_i(X, W, lambda), ncol=d, byrow=FALSE))
  }
  grads_i = t(apply(X, 1, function(x_i) (grad_log_likelihood_ica_i(x_i, W, lambda))))
  grads_i = colSums(grads_i)
  
  return(dim(X)[1]*t(inv(W)) - matrix(grads_i, ncol=d, byrow=FALSE))
}


likelihood_ica_i = function(x_i, W){
  # unused atm
  y = W %*% x_i # d * 1
  p_y_i = 1/(4*cosh(0.5*y)^2) # d * 1
  p_y = prod(p_y_i) # scalar
  return(p_y)
}


log_prior_ica = function(W, lambda=1){
  # compute log prior density of the parameters (normal with diag precision=1_d)
  return(-lambda/2*t(matrix(W))%*%matrix(W))
}


log_likelihood_ica_alt = function(X, W, d=10, type='sample'){
  # computes avg log likelihood (ICA model)
  if (is.null(dim(X))){
    return(log(abs(det(W)), exp(1)) + log(likelihood_ica_i(X, W), exp(1)))
  }
  S = W %*% t(X) # d * N
  p_S = 1 / (4 * cosh(0.5*S)^2)
  log_p_S = log(p_S)
  # avg log lik
  p = log(abs(det(W)), exp(1)) + sum(log_p_S)/(dim(X)[1])
  return(p)
}


log_posterior_ica = function(X, W){
  # computes log posterior (=N * avg log likelihood + log prior)
  return(dim(X)[1]*log_likelihood_ica_alt(X, W) + log_prior_ica(W))
}

polynomial_decay = function(sigma, iter){
  # implements polynomial deca of the step size
  sigma = sigma * (iter/(iter+1))^(0.55)
  return(sigma)
}