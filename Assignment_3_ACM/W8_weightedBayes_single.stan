//

functions{
  real weight_f(real L_raw, real w_raw) {
    real L;
    real w;
    L = exp(L_raw);
    w = 0.5 + inv_logit(w_raw)/2;
    return log((w * L + 1 - w)./((1 - w) * L + w));
  }
}

data {
  int<lower=0> N;
  array[N] real y;
  vector[N] SourceSelf;
  vector[N] SourceOther;
}

parameters {
  real weight1;
  real weight2;
  real<lower=0> sigma;
}

model {
  target += normal_lpdf(weight1 | 0,1);
  target += normal_lpdf(weight2 | 0,1);
  target += normal_lpdf(sigma | 0, 1) - normal_lccdf(0|0,1);
  
  for (n in 1:N){  
  target += normal_lpdf(y[n] | weight_f(SourceSelf[n], weight1) + weight_f(SourceOther[n], weight2), sigma);
  }
}

generated quantities{
  array[N] real log_lik;
  real w1;
  real w2;
  real w1_prior;
  real w2_prior;
  
  w1_prior = 0.5 + inv_logit(normal_rng(0,1))/2;
  w2_prior = 0.5 + inv_logit(normal_rng(0,1))/2;
  w1 = 0.5 + inv_logit(weight1)/2;
  w2 = 0.5 + inv_logit(weight2)/2;
  
  for (n in 1:N){  
    log_lik[n] = normal_lpdf(y[n] | weight_f(SourceSelf[n], weight1) + weight_f(SourceOther[n], weight2), sigma);
  }
  
}

