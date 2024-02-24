# this script contains helper function for the simulations with the BPMF model
source("../utils.R")

library(mvtnorm)
library(zoo)

SGMCMC_pmf = function(df, S, n, step_size, params_pmf, hyperparams_pmf=0, test_performance=TRUE, 
                      test_set, schedule=NULL, barker=TRUE, method='vanilla', thinning=100, 
                      performance_functions = list('likelihood'=log_likelihood_pmf, 'error_function'=rmse_pmf, 'prediction'=prediction_pmf),
                      grad_functions=list('compute_grad'=compute_grad_pmf, 'update_params'=update_pmf)){
  

  set.seed(123)
  
  print(hyperparams_pmf)
  
  # gradient functions
  compute_grad_pmf = grad_functions$compute_grad
  update_pmf = grad_functions$update_params
  
  # compute performance functions
  likelihood = performance_functions$likelihood
  error_function = performance_functions$error_function
  prediction = performance_functions$prediction
  
  # number of data points
  N_data = dim(df)[1]
  # number of users - items
  N = max(df$uid) # users
  M = max(df$iid) # items
  
  # initialize params for mcmc estimate
  params_mcmc = params_pmf
  
  if (test_performance){
    liks_sample = rep(0, S/thinning)
    errors_sample = rep(0, S/thinning)
    liks_mcmc = rep(0, S/thinning/2)
    errors_mcmc = rep(0, S/thinning/2)
  }
  
  modif_step_size = FALSE
  if (! is.null(schedule)){
    modif_step_size = TRUE 
  }
  
  for (i in 1:S) { 
    #print(i)
    
    #subsample the minibatch
    mini_batch_index = sample(N_data, n)
    mini_batch = df[mini_batch_index,]
    
    # compute gradient
    grad_pmf = compute_grad_pmf(mini_batch, params_pmf, hyperparams_pmf, N_data)
    
    if (modif_step_size & i>S/2){
      step_size = schedule(step_size, i-S/2)
    }
    
    # update param with suitable MCMC kernel
    # sgbd
    if (barker){params_pmf = update_pmf(params_pmf, grad_pmf, step_size, kernel=r_barker, method=method)}
    # sgld
    else{params_pmf = update_pmf(params_pmf, grad_pmf, step_size, kernel=r_sgld, method=method)}
  
    if (i > S/2){
      if (i == (S/2 +1)){
        test_set = prediction(test_set, params_pmf, hyperparams_pmf)
        #test_set$pred_clipped = test_set$pred
        test_set$pred_mcmc = test_set$pred
        test_set$pred_mcmc_clipped = test_set$pred_clipped
      }
      else{
        test_set = prediction(test_set, params_pmf, hyperparams_pmf)
        test_set$pred_mcmc = (test_set$pred + test_set$pred_mcmc*(i-1-S/2))/(i-S/2)
        test_set$pred_mcmc_clipped = test_set$pred_mcmc
        test_set$pred_mcmc_clipped[, test_set$pred_mcmc_clipped > 5] = 5
        test_set$pred_mcmc_clipped[, test_set$pred_mcmc_clipped < 1] = 1
      }
    }
    
    if (test_performance){
      if ((i%%thinning == 0)){
        test_set = prediction(test_set, params_pmf, hyperparams_pmf)
        lik = likelihood(test_set, params_pmf, hyperparams_pmf, 'sample')
        err = error_function(test_set, params_mcmc, hyperparams_pmf, 'sample')
        liks_sample[i/thinning] = lik
        errors_sample[i/thinning] = err
        if(i>S/2){
          lik_mcmc = likelihood(test_set, params_mcmc, hyperparams_pmf, 'mcmc')
          err_mcmc =  error_function(test_set, params_mcmc, hyperparams_pmf, 'mcmc')
          liks_mcmc[(i - S/2)/thinning] = lik_mcmc
          errors_mcmc[(i - S/2)/thinning] = err_mcmc
        }            
      }
    }
    
    if ((i%%100 == 0) & test_performance){
      print(i)
      print(lik)
      print(err)
      if (i > S/2){
        print(lik_mcmc)
        print(err_mcmc)
      }
    }
  }
  print('MCMC run done')
  if (test_performance){
    plot(seq(1, S, by=thinning), liks_sample, xlab='iter', ylab='test likelihood', main='', type='l', col='blue')
    lines(seq(S/2+1, S, by=thinning), liks_mcmc, col='blue4')
    plot(seq(1, S, by=thinning), errors_sample, xlab='iter', ylab='test error', main='', type='l', col='blue', ylim=c(0.8,  2))
    lines(seq(S/2+1, S, by=thinning), errors_mcmc, col='blue4')
    res = list('params'=params_pmf, 'params_mcmc'=params_mcmc,'lik_sample'=liks_sample, 'error_sample'=errors_sample, 'lik_mcmc'=liks_mcmc, 'error_mcmc'=errors_mcmc)
  }
  else{res=list('params'=params_pmf, 'params_mcmc'=params_mcmc)}
  return(res)
}


# init parameters 
initialize_bpfm = function(df, hyperparams, d=20){
  set.seed(123)
  # number of users - items
  N = max(df$uid) # users
  M = max(df$iid) # items
  mu0 = hyperparams$mu_0
  lambda_U = rnorm(d, -.1, 1)
  lambda_V = rnorm(d, -.1, .1)
  mu_U = rnorm(d, mu0, 1)
  mu_V = rnorm(d, mu0,1)
  U = t(rmvnorm(N, mu_U, 1 * diag(d)))
  V = t(rmvnorm(M, mu_V, 1 * diag(d)))
  return(list('lambda_U' = lambda_U, 'lambda_V' = lambda_V, 'mu_U' = mu_U, 'mu_V' = mu_V, 'U' = U, 'V' = V))
}
  
update_bpmf = function(params_bpmf, grads_bpmf, step_size, kernel=r_barker, method='vanilla'){
  
  params_bpmf_new = list()
  
  params_bpmf_new$U = kernel(params_bpmf$U, grads_bpmf$U, step_size, method=method)
  params_bpmf_new$V = kernel(params_bpmf$V, grads_bpmf$V, step_size, method=method)
  params_bpmf_new$mu_U = kernel(params_bpmf$mu_U, grads_bpmf$mu_U, step_size, method=method)
  params_bpmf_new$mu_V = kernel(params_bpmf$mu_V, grads_bpmf$mu_V, step_size, method=method)
  params_bpmf_new$lambda_U = kernel(params_bpmf$lambda_U, grads_bpmf$lambda_U, step_size, method=method)
  params_bpmf_new$lambda_V = kernel(params_bpmf$lambda_V, grads_bpmf$lambda_V, step_size, method=method)
  
  return(params_bpmf_new)
}

# define helper function used to compute the gradient
U_t_V_bpmf = function(x, U, V) {return(return(sum(U[,x['uid']] * V[, x['iid']])))}

predict_bpmf = function(df, params_bpmf, hyperparams_pmf){
  
  df['pred'] = apply(df, 1, function(x)(U_t_V_bpmf(x, params_bpmf$U, params_bpmf$V) + hyperparams_pmf$R_mean))
  df$pred_clipped = df$pred
  df[df$pred >5, 'pred_clipped'] = 5
  df[df$pred <1, 'pred_clipped'] = 1
  
  return(df)
}

grad_lik_U = function(x, df,  U, V, mu_U, sigma_U, alpha){
  index = df['iid'][df['uid'] == x]
  if (length(index) == 1){
    return(alpha * t(df['res'][df['uid'] == x] * t(V[,index])))
  }
  return(rowSums(alpha * t(df['res'][df['uid'] == x] * t(V[,index]))))
}

grad_prior_U = function(U, mu_U, sigma_U){
  return(- exp(sigma_U) * (U - mu_U))
}


grad_lik_V = function(x, df, U, V, mu_V, sigma_V, alpha){
  index = df['uid'][df['iid'] == x]
  n_j = length(index)
  if (length(index) == 1){
    return(alpha * t(df['res'][df['iid'] == x] *  t(U[,index])))
  }
  return(rowSums(alpha * t(df['res'][df['iid'] == x] *  t(U[,index]))))
}

grad_prior_V = function(V, mu_V, sigma_V) {
  return(-exp(sigma_V) * (V - mu_V))
}


grad_prior_lambda_U = function(sigma_U, U, mu_U, N, alpha0, beta){
  rs1.u = rowSums((U - mu_U)**2)
  return(((N + 1)/2 + alpha0)  - ((mu_U**2 + rs1.u) / 2 + beta) * exp(sigma_U))
}

grad_prior_lambda_V= function(sigma_V, Vs, mu_V, M, alpha0, beta){
  rs1.v = rowSums((Vs - mu_V)**2)
  return(((M + 1)/2 + alpha0)  - ((mu_V**2 + rs1.v ) / 2 + beta) * exp(sigma_V))
}

grad_prior_mu_U = function(sigma_U, mu_U, Us, N){
  rs2.u = rowSums(Us)
  return(exp(sigma_U) * (rs2.u - (N + 1) * mu_U))
}

grad_prior_mu_V = function(sigma_V, mu_V, Vs, M){
  rs2.v = rowSums(Vs)
  return(exp(sigma_V) * (rs2.v - (M + 1) * mu_V))
}

p = function(z, grad_log_pi){
  return(1/(1+exp(-z*grad_log_pi)))
}


grad_latent_matrix_bpmf = function(df, params_bpmf, hyperparams_bpmf, N_data){
  
  N_batch = dim(df)[1]
  
  df['pred'] = apply(df, 1, function(x) (U_t_V_bpmf(x, params_bpmf$U, params_bpmf$V))) # predictions
  df['res'] = df['R'] - df['pred'] # residuals
  
  # Users latent features
  grad_U = grad_prior_U(params_bpmf$U, params_bpmf$mu_U, params_bpmf$lambda_U)
  uis = sort(unique(df$uid))
  uis.df = data.frame(uis)
  grad_Us = apply(uis.df, 1, function(x) (grad_lik_U(x, df, params_bpmf$U, params_bpmf$V, params_bpmf$mu_U,
                                                     params_bpmf$lambda_U, hyperparams_bpmf$alpha)))
  grad_U[, uis] = grad_U[, uis] + N_data/N_batch * grad_Us
  
  # Items latent features
  grad_V = grad_prior_V(params_bpmf$V, params_bpmf$mu_V, params_bpmf$lambda_V)
  vis = sort(unique(df$iid))
  vis.df = data.frame(vis) 
  grad_Vs = apply(vis.df, 1, function(x) (grad_lik_V(x, df, params_bpmf$U, params_bpmf$V, params_bpmf$mu_V,
                                                     params_bpmf$lambda_V, hyperparams_bpmf$alpha)))
  grad_V[, vis] = grad_V[, vis] + N_data/N_batch * grad_Vs
  
  return(list('grad_U'=grad_U, 'grad_V'=grad_V))
}

grad_hyperparams_bpmf = function(df, params_bpmf, hyperparams_bpmf){
 
  grad_lambda_U = grad_prior_lambda_U(params_bpmf$lambda_U, params_bpmf$U, params_bpmf$mu_U,
                                     hyperparams_bpmf$N, hyperparams_bpmf$alpha_0, hyperparams_bpmf$beta)
  grad_lambda_V = grad_prior_lambda_V(params_bpmf$lambda_V, params_bpmf$V, params_bpmf$mu_V,
                                     hyperparams_bpmf$M, hyperparams_bpmf$alpha_0, hyperparams_bpmf$beta)
  
  grad_mu_U = grad_prior_mu_U(params_bpmf$lambda_U, params_bpmf$mu_U, params_bpmf$U, hyperparams_bpmf$N)
  grad_mu_V =  grad_prior_mu_V(params_bpmf$lambda_V, params_bpmf$mu_V, params_bpmf$V, hyperparams_bpmf$M)
  
  return(list('grad_lambda_U'=grad_lambda_U, 'grad_lambda_V'=grad_lambda_V, 'grad_mu_U'=grad_mu_U, 'grad_mu_V'=grad_mu_V))
}

compute_grad_bpmf = function(df, params_bpmf, hyperparams_bpmf, N_data){
  
  grad_latent_matrix = grad_latent_matrix_bpmf(df, params_bpmf, hyperparams_bpmf, N_data)
  grad_hyperparams = grad_hyperparams_bpmf(df, params_bpmf, hyperparams_bpmf)
  
  return(list('U' = grad_latent_matrix$grad_U, 'V' = grad_latent_matrix$grad_V, 
              'lambda_U' = grad_hyperparams$grad_lambda_U, 'lambda_V' = grad_hyperparams$grad_lambda_V,
              'mu_U' = grad_hyperparams$grad_mu_U, 'mu_V' = grad_hyperparams$grad_mu_V))
}

preprocess_df_bpmf = function(df){
  meanR = mean(df$R)
  df$R = df$R - meanR
  return(df)
}

rmse_bpmf = function(df, type='sample'){
  stopifnot(type=='sample'|type=='mcmc')
  if (type=='sample'){
    rmse = sqrt(mean((df$R-df$pred_clipped)^2))
    return(rmse)
  }
  rmse = sqrt(mean((df$R-df$pred_mcmc_clipped)^2))
  return(rmse)
}

log_lik_bpmf = function(df, params_bpmf, hyperparams_pmf, type='sample'){
  stopifnot(type=='sample'|type=='mcmc')
  if (type=='sample'){
    log_lik = -mean((df$R-df$pred)^2) * hyperparams_bpmf$alpha
    return(log_lik)
  }
  log_lik = -mean((df$R-df$pred_mcmc)^2) * hyperparams_bpmf$alpha
  return(log_lik)
}

BPMF = function(h, df, test, S, d, miniBatchSize = 800, algo = 'sgld',
                version = 'standard', burn_in = 0, grad = TRUE, 
                train_accuracy = FALSE, return_samples=FALSE){
  
  h_1 = h[1]
  h_2 = h[2]
  h_3 = h[3]
  
  
  alpha = 3
  alpha0 = 1
  beta = 5
  
  N = length(unique(df$uid))
  M = length(unique(df$iid))
  
  N_u = max(df$uid)
  N_i = max(df$iid)
  
  d = 20
  N_data = dim(df)[1]
  
  print(d)
  set.seed(123)
  params = initParams_bpfm(N_u, N_i, d, 0, 1, 5)
  U = params$U
  V = params$V
  mu_U = params$mu_U
  mu_V = params$mu_V
  sigma_U = params$sigma_U
  sigma_V = params$sigma_V
  
  rMSEs_avg = rep(NA, S - burn_in)
  rMSEs = rep(NA, S)
  rMSEs_avg_train = rep(NA, S - burn_in)
  rMSEs_train = rep(NA, S)
  
  
  if (grad) {
    grads_U = rep(NA, S)
    grads_V = rep(NA, S)
    grads_mu_U = rep(NA, S)
    grads_mu_V = rep(NA, S)
    grads_lambda_U = rep(NA, S)
    grads_lambda_V = rep(NA, S)
  }
  else{
    grads_U = 0
    grads_V = 0
    grads_mu_U = 0
    grads_mu_V = 0
    grads_lambda_U = 0
    grads_lambda_V = 0
  }
  
  L = dim(df)[1]
  # miniBatchSize = floor(0.1 * L)
  # standardize the rating
  meanR = mean(df$R)
  df$R = df$R - meanR
  
  
  for (i in seq(1, S)){
    minibatch = sample(L, miniBatchSize)
    minibatch = df[minibatch,]
    uis = sort(unique(minibatch$uid))
    vis = sort(unique(minibatch$iid))
    n = length(uis)
    m = length(vis)
    Us = U[, uis]
    Vs = V[, vis]
    uis.df = data.frame(uis)
    vis.df = data.frame(vis)
    # compute R - UTV
    minibatch['res'] = apply(minibatch, 1, function(x) (f1(x, U, V)))
    
    grad_U = matrix(0, ncol = N_u, nrow = d)
    grad_V = matrix(0, ncol = N_i, nrow = d)
    
    grad_sigma_U = grad_lambda_U_f(sigma_U, U, mu_U, N, alpha0, beta)
    grad_sigma_V = grad_lambda_V_f(sigma_V, V, mu_V, M, alpha0, beta)
    
    grad_mu_U = grad_mu_U_f(sigma_U, mu_U, U, N)
    grad_mu_V =  grad_mu_V_f(sigma_V, mu_V, V, M)
    
    grad_Us = apply(uis.df, 1, function(x1) (f2_c(x1, minibatch, U, V, mu_U, sigma_U, alpha, N_data, miniBatchSize)))
    grad_Vs = apply(vis.df, 1, function(x1) (f3_c(x1, minibatch, U, V, mu_V, sigma_V, alpha, N_data, miniBatchSize)))
    
    grad_U[, uis] = grad_Us
    grad_U = f2_b(grad_U, U, mu_U, sigma_U)

    grad_V[, vis] = grad_Vs
    grad_V = f3_b(grad_V, V, mu_V, sigma_V)
    
    
    if (grad){
      grads_lambda_U[i] = max(abs(grad_sigma_U))
      grads_lambda_V[i] = max(abs(grad_sigma_V))
      grads_mu_U[i] = max(abs(grad_mu_U))
      grads_mu_V[i] = max(abs(grad_mu_V))
      grads_U[i] = max(abs(grad_Us))
      grads_V[i] = max(abs(grad_Vs))
    }
    
    if (algo == 'sgld') {
      
      # one step langevin dynamics
      sigma_U = sigma_U + h_3/2 * grad_sigma_U +  sqrt(h_3) * rnorm(d)
      sigma_V = sigma_V + h_3/2 * grad_sigma_V +  sqrt(h_3) * rnorm(d)
      mu_U = mu_U + h_2/2 * grad_mu_U +  sqrt(h_2) * rnorm(d)
      mu_V = mu_V + h_2/2 * grad_mu_V +  sqrt(h_2) * rnorm(d)
      U = U + sqrt(h_1) * rnorm(N_u * d) + h_1/2 * grad_U
      #U[, uis] = U[,uis] + h_1/2 * grad_Us
      V = V + sqrt(h_1) * rnorm(N_i * d) + h_1/2 * grad_V
      #V[, vis] = V[,vis] + h_1/2 * grad_Vs
    }
    
    else {
      # one step barker
      sigma_U = oneStepBarker(sigma_U, h_3, grad_sigma_U, version)
      sigma_V = oneStepBarker(sigma_V, h_3, grad_sigma_V, version)
      mu_U = oneStepBarker(mu_U, h_2, grad_mu_U, version)
      mu_V = oneStepBarker(mu_V, h_2, grad_mu_V, version)
      U = matrix(oneStepBarker(as.vector(U), h_1, as.vector(grad_U), version), ncol = N_u)
      V = matrix(oneStepBarker(as.vector(V), h_1, as.vector(grad_V), version), ncol = N_i)
    }
    
    # predict new data and compute MSE
    current_pred = apply(test, 1, function(x) f4(x, U, V, meanR))
    rMSE = sqrt(mean((test$R - current_pred)**2))
    rMSEs[i] = rMSE
    if(train_accuracy){
      current_pred_train = apply(df, 1, function(x) f4(x, U, V, meanR))
      rMSE_train = sqrt(mean((df$R - current_pred_train)**2))
      rMSEs_train[i] = rMSE_train
    }
    if (i%%100 == 0) {
      cat("iter: ", i, "rMSE: ",rMSE)
      print("")
    }
    if (i > burn_in){
      if (i == burn_in +1) {
        pred = current_pred
        
      }
      else {
        pred = (current_pred + (i - burn_in - 1)*old_pred) / (i - burn_in)
      }
      rMSE = sqrt(mean((test$R - pred)**2))
      #print(rMSE)
      if (i%%100 == 0) {
        cat("iter: ", i, "rMSE avg: ",rMSE)
        print("")
      }
      old_pred = pred
      #print(current_pred)
      rMSEs_avg[i - burn_in] = rMSE
      
      
      if (is.na(rMSE)) {
        print("NaN values")
        
        return(rMSEs)
      }
    }
  }
  if(return_samples){
    samples = list('U'=U, 'V'=V, 'mu_U'=mu_U, 'mu_V'=mu_V, 'lambda_U'=sigma_U, 'lambda_V'=sigma_V)
  }
  else{
    samples = 0
  }
  
  return(list('rMSEs' = rMSEs, 'rMSEs_avg' = rMSEs_avg, 'rMSEs_train' = rMSEs_train, 'grads_U' = grads_U, 'grads_V' = grads_V, 'grads_mu_U' = grads_mu_U, 'grads_mu_V' = grads_mu_V, 'grads_lambda_U' = grads_lambda_U, 'grads_lambda_V' = grads_lambda_V, 'samples'=samples))
}






