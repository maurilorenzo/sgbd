data {
  int<lower=0> N;
  int<lower=0> d;
  matrix[N,d] X;
  int y[N];
  int<lower=0> N_test;
  matrix[N_test,d] X_test;
  int y_test[N_test];
  matrix[d,d] Sigma0;

}

transformed data {
  vector[d] mu0;
  for (j in 1:d)
    mu0[j] = 0;
}

parameters {
  vector[d] beta;
}

transformed parameters {
  real Xbeta[N];
  for (i in 1:N)
    Xbeta[i] = dot_product(row(X, i), beta);
}

model {
  beta ~ multi_normal(mu0, Sigma0);
  y ~ bernoulli_logit(Xbeta);
}

generated quantities {
  real<lower=0, upper=1> p_train[N];
  real<lower=0, upper=1> p_test[N_test];
  real log_lik_test;
  real Xbeta_test[N_test];

  p_train = inv_logit(Xbeta);
  for (i in 1:N_test)
    Xbeta_test[i] = dot_product(row(X_test, i), beta);
  p_test = inv_logit(Xbeta_test);
  log_lik_test = bernoulli_lpmf(y_test | p_test);
}

