//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//


data {
  int<lower=0> N;
  array[N] real y;
  vector[N] SourceSelf;
  vector[N] SourceOther;
}

parameters {
  real<lower=0> sigma;
}

model {
  target += normal_lpdf(sigma | 0, 1) - normal_lccdf(0|0,1);
  target += normal_lpdf(y | logit(SourceSelf)+logit(SourceOther), sigma);
}

generated quantities{
  array[N] real log_lik;
  
  for (n in 1:N){  
    log_lik[n] = normal_lpdf(y[n] | logit(SourceSelf[n])+logit(SourceOther[n]), sigma);
  }
  
}

