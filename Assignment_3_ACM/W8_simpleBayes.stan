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
  array[N] int y;
  vector[N] Source1;
  vector[N] Source2;
}

parameters {
}

model {
}

generated quantities{
  array[N] real log_lik;
  
  for (n in 1:N){  
    log_lik[n] = bernoulli_logit_lpmf(y[n] | logit(Source1[n]) +  logit(Source2[n]));
  }
  
}

