library(coda)
library(mvtnorm)
library(pracma)
library(matrixStats)
library(LaplacesDemon)
library(kernlab)
library(KSD)
library(sn)
library(GMCM)

# stochastic gradent barker proposal kernel
r_barker = function(x, c, sigma, noise="bimodal", method="vanilla", correction=FALSE, tau=0){

  # this function computes the new value of the Barker Proposal Algorithm

  # PARAMS:
  # x - vector, current location
  # c - vector, gradient of the log-density
  # sigma - float, hyperparameter of the algorithm
  # noise - str, type of the distribution of the artificial noise
  # method - str, method to compute the probability of the sign flip of the increment
  # correction - bool, whether to apply bias correction formula (default to FALSE)
  # tau - float, estimate of gradient noise s.d.. Used if correction = TRUE

  # RETURNS:
  # vector, uptated value of the params


  stopifnot(noise=="gaussian"|noise=="bimodal")
  stopifnot(method=="vanilla"|method=="extreme"|method=="corrected"|method=="adaptive")

  if (method=='adaptive'){
    sigma = 1.792/1.233/tau
  }

  # draw increment
  if(noise=="bimodal"){
    z<-rnorm(n=length(c), mean=sigma, sd=0.1*sigma)
  }
  else if(noise=="gaussian"){
    z<-sigma*rnorm(n=length(c))
  }

  # increment sign flip step (inject skweness using the gradient)
  if(method=="vanilla"){

    p_hat <- 1/(1+exp(-c*z))
    if (correction){
      p_hat <- p_hat - tau_2/2*z^2*p_hat*(1-p_hat)*(1-2*p_hat)
    }
    u = runif(n=length(c))
    b <- 2*(u < p_hat) - 1
  }
  else if(method=="corrected"|method=='adaptive'){
    b <- sign(c*z)
    factor = 1.702^2
    mask = (factor > tau^2*z^2)
    correction = factor / (factor - tau^2*z^2)
    if (sum(mask > 0)){
      p_hat <- 1/(1+exp(-c[mask]*z[mask]*sqrt(correction[mask])))
      b[mask] <- 2*(runif(n=sum(mask))< p_hat) -1
    }
  }
  else if(method=="extreme"){
    b <- sign(c*z)
  }
  b[abs(c) == Inf] = sign(z[abs(c) == Inf]*c[abs(c) == Inf])
  return(x + z*b)
}



r_sgld = function(x, c, sigma, method="vanilla", tau=0){

  # this function computes the new value of the Stochastic Gradient Langevin Dynamics Algorithm

  # PARAMS:
  # x - vector, current location
  # c - vector, gradient of the log-density
  # sigma - float or vectort, hyperparameter of the algorithm
  # method - str, method used when generating the artificial noise. If "vanilla" no correction is used, if "corrected" the gradient noise is accounted to correct the artificial noise variance
  # tau - float, estimate of gradient noise s.d.. Used if method='corrected'

  # RETURNS:
  # vector, uptated value of the params
  stopifnot(method=="vanilla" | method=="corrected"|
              method=="extreme" | method=="modified"| method=="adaptive")

  if (method=='adaptive'){
    sigma = 2/tau
  }

  if (method=="vanilla"){
    z = rnorm(n=length(c), mean=0, sd=sigma)
  }
  else if(method=="extreme"){
    return(x = x + sigma^2/2*c)
  }
  else{
    z = rep(0, length(c))
    mask = (tau < sqrt(2)/sigma)
    if (sum(mask) > 0){
      if (method=='corrected'|method=="adaptive"){
        z[mask] = rnorm(n=sum(mask), mean=0, sd=sqrt((sigma^2 - tau^2/4*sigma^4)[mask]))
      }
      else {
        z[mask] = rnorm(n=sum(mask), mean=0, sd=sigma*(1-sigma^2/2*tau[mask]^2))
      }
    }
  }

  x = x + sigma^2/2*c + z

  return(x)
}


# stochastic gradient barker proposal MCMC
SGMCMC = function(data, S, n, theta, sigma, grad_L, grad_p, noise='bimodal', method='vanilla',
                  saga=FALSE, thresh=0, map=FALSE, eps=0.00001, eta=0.01, barker=TRUE,
                  beta=0.9,adaptive_step_size=FALSE,to_df=TRUE,
                  compute_performance=FALSE, compute_performance_train=FALSE, test_set=NA, schedule=NULL,
                  predict_y=FALSE, compute_error=TRUE, performance_functions=list('likelihood'=log_loss_log_reg, 'error'=rmse, 'predict'=predict_log_reg)){

  # this function implements the Stochatic Gradient Barker Proposal Algorithm (SGBD)

  # PARAMS:
  # data - dataset or matrix, data
  # S - int, number of itereations
  # n - int, minibatch size
  # theta - vector, starting point of the chain
  # sigma - float, hyperparameter of the algorithm
  # grad_L - function, gradient of the log-likelihood
  # grad_p - function, gradient of the log-prior
  # noise - str, type of the distribution of the artificial noise
  # method - str, method to compute the probability of the sign flip of the increment
  # thres - float, threshold used when method = 'mixed' (to be implemented)
  # map - bool, whether burn-in is replaced with sgd step to find an approximation of the MAP (default is FASLE)
  # eps - float, precision of SGD. Used when map = TRUE (default is 0.00001)
  # eta - float, learning rate of SGD. Used when map = TRUE (default is 0.001)
  # beta - float, parameter of the exponential average of estimate od the s.d. gradient noise. Used when method = 'corrected' (default is 0.9)

  # RETURNS:
  # thetas - array[S, d], array of the samples produced by the algorithm (array[S//2, d] if map = TRUE)

  set.seed(123)

  stopifnot(noise=="gaussian"|noise=="bimodal")
  stopifnot(method=="vanilla"|method=="extreme"|method=="corrected"|method=='modified'|method=='adaptive')


  if (to_df){
    # transform data to data.frame
    data = data.frame(data)
  }

  # number of data points
  N = dim(data)[1]
  # dimensionality of the parameter
  d = length(as.vector(theta))

  if (map) {
    S = floor(S/2)
  }

  # array to store samples
  thetas = matrix(NA, nrow = S, ncol = d)

  if (compute_performance){
    liks_sample = rep(0, S/100)
    errors_sample = rep(0, S/100)
    liks_mcmc = rep(0, S/200)
    errors_mcmc = rep(0, S/200)
    liks_train_sample = rep(0, S/100)
    liks_train_mcmc = rep(0, S/200)
    likelihood = performance_functions$likelihood
    error = performance_functions$error
    predict = performance_functions$predict
  }

  if (compute_performance_train){
    train_copy = data
  }

  if (map) {
    theta = SGD(data, S, n, theta, eta, grad_L, grad_p, eps)
  }

  modif_step_size = FALSE
  if (! is.null(schedule)){
    modif_step_size = TRUE
  }

  # initialize starting point
  thetas[1,] = as.vector(theta)

  sd_noise_bias_corrected = 0

  if (method == "corrected"| method == 'adaptive' |adaptive_step_size){
    sd_noise = rep(0, d)
  }

  for (i in 2:S) {
    #subsample the minibatch
    mini_batch_index = sample(N, n)
    mini_batch = data[mini_batch_index,]

    # compute grad log likelihood for each data point
    if (method=='corrected'| adaptive_step_size | method == 'adaptive' ){
      grad_mini_batch = t(apply(mini_batch, 1, function(x) (grad_L(x, theta))))
    }
    # estimate gradient
    if (method == 'corrected'|adaptive_step_size | method == 'adaptive' ){
      grad_likelihood =  N/n * colSums(grad_mini_batch)
    }
    else{
      grad_likelihood = N/n * grad_L(mini_batch, theta)
    }
    # compute gradient
    grad_target = grad_p(theta) + grad_likelihood


    if (method=="corrected" | adaptive_step_size | method == 'adaptive' ){
      sd_noise = beta*sd_noise + (1-beta)*N/sqrt(n)*sqrt(N-n)/sqrt(N-1)*colSds(grad_mini_batch)
      sd_noise_bias_corrected = sd_noise / (1 - beta^(i-1))
    }

    if (adaptive_step_size){
      sigma = 1.702/sd_noise_bias_corrected/1.233
    }

    if (modif_step_size & i>S/2){
      sigma = schedule(sigma, i-S/2)
    }

    # update param with suitable MCMC kernel
    if (barker){
      # sgbd
      theta = r_barker(theta, grad_target, sigma, noise=noise, method=method, tau=sd_noise_bias_corrected)
    }
    else{
      # sgld
      theta = r_sgld(theta, grad_target, sigma, method=method, tau=sd_noise_bias_corrected)
    }

    if (i > S/2){
      if (i == (S/2 +1)){
        if (predict_y){
          test_set = predict(test_set, theta)
          test_set$pred_mcmc = test_set$pred
          if (compute_performance_train){
            train_copy = predict(train_copy, theta)
            train_copy$pred_mcmc = train_copy$pred
          }
        }
        theta_mcmc = theta
      }
      else{
        if (predict_y){
          test_set = predict(test_set, theta)
          test_set$pred_mcmc = ((i-1-S/2)*test_set$pred_mcmc + test_set$pred)/(i-S/2)
          if (compute_performance_train){
            train_copy = predict(train_copy, theta)
            train_copy$pred_mcmc = ((i-1-S/2)*train_copy$pred_mcmc + train_copy$pred)/(i-S/2)
          }
        }
        theta_mcmc = (theta_mcmc*(i-1-S/2) + theta)/(i-S/2)
      }
    }

    if (compute_performance & (i%%100 == 0)){
      if (predict_y) {test_set = predict(test_set, theta)}
      liks_sample[i/100] = likelihood(test_set, theta)
      if (compute_error){errors_sample[i/100] = error(test_set, theta)}
      if (compute_performance_train){
        if (predict_y){train_copy = predict(train_copy, theta)}
        liks_train_sample[i/100] = likelihood(train_copy, theta)
      }
      if (i > S/2){
        liks_mcmc[i/100 - S/200] = likelihood(test_set, theta_mcmc, type='mcmc')
        if (compute_error){errors_mcmc[i/100- S/200] = error(test_set, theta_mcmc, type='mcmc')}
        if (compute_performance_train){
          liks_train_mcmc[i/100 - S/200] = likelihood(train_copy, theta_mcmc, type='mcmc')

        }
      }
    }

    theta_to_store = theta
    # store new value
    thetas[i,] = as.vector(theta_to_store)
  }
  print('MCMC run done')
  if (compute_performance){
    res = list('thetas'=thetas, 'lik_sample'=liks_sample, 'error_sample'=errors_sample,
               'lik_mcmc'=liks_mcmc, 'error_mcmc'=errors_mcmc, 'lik_train_sample'=liks_train_sample, 'lik_train_mcmc'=liks_train_mcmc)
  }
  else{res=thetas}

  return(res)
}


# probability keeping the increment sign
p_hat = function(c, z, method='naive', correction=1){
  stopifnot(method=="naive"|method=="extreme"|method=='corrected')
  if (method == 'naive'){
    p = (1+exp(-c*z))^{-1}
  }
  else if(method=='extreme'){
    p = as.numeric((sign(c) == sign(z)))
  }
  else{
    p = (1+exp(-c*z*correction))^{-1}
  }
  return(p)
}


# Stochastic gradient descent
SGD = function(data, S, n, theta, eta, grad_L, grad_p, eps=0.0001){

  N = dim(data)[1]

  decay = eta/S

  for (i in 2:S){

    eta_t = eta / (1+i*decay)
    # subsample the minibatch
    mini_batch_index = sample(N, n)
    mini_batch = data[mini_batch_index,]

    theta_old = theta

    # estimate gradient of posterior
    grad_likelihood = N/n * grad_L(mini_batch, theta)
    grad_target = grad_p(theta) + grad_likelihood

    theta = theta + eta_t*grad_target

    if (max(abs(theta - theta_old)) < eps) {
      cat('SGD converged in', i, 'iterations')
      return(theta)
    }
  }
  print('SGD did not converge')
  cat('last max divergence:', max(abs(theta - theta_old)))
  return(theta)
}
