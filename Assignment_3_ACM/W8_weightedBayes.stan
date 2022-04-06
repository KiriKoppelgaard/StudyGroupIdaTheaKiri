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
  int<lower=0> trials;//trials
  int<lower=1> participants; //number of participants
  array[trials, participants] real <lower=0> choice; //decisions
  array[trials, participants] real <lower=0> SourceSelf;
  array[trials, participants] real <lower=0> SourceOther;
}

parameters {
  real weight1M;//population level mean weight (Self)
  real weight2M;//population level mean weight (Other)
  real<lower=0> sigma; //expected average error
  vector<lower=0>[2] tau; //sd for population means, weight1M and weight2M (Vector containing 2 numbers)
  matrix[2, participants] z_IDs; //z-scored individual deviation from population weight means (two for each participant)
  cholesky_factor_corr[2] L_u; //weight correlation
}

transformed parameters{
  matrix[participants, 2] IDs; //non-scaled individual deviations from population weight means (two for each participant)
  IDs = (diag_pre_multiply(tau, L_u)*z_IDs)'; //weight multiplication with consideration of correlation between individual deviations within participant
}


model {
  target += normal_lpdf(weight1M | 0,1);//population level mean
  target += normal_lpdf(weight2M | 0,1);//population level mean
  target += normal_lpdf(tau[1]|0, 1.5) - normal_lccdf(0|0, 1.5); //population level sd
  target += normal_lpdf(tau[2]|0, 1.5) - normal_lccdf(0|0, 1.5); //population level sd
  target += normal_lpdf(sigma|0, .3) - normal_lccdf(0|0, .3); //population level sd
  
  target += lkj_corr_cholesky_lpdf(L_u | 2); //lkj-prior ranging from -1 to 1, convient for correlations
  
  target += std_normal_lpdf(to_vector(z_IDs)); //sampling the scaled individual deviations

  for (participant in 1:participants){
    for (trial in 1:trials){  
      target += normal_lpdf(choice[trial, participant] | 
        weight_f(SourceSelf[trial, participant], weight1M + IDs[participant, 1]) + 
        weight_f(SourceOther[trial, participant], weight2M + IDs[participant, 2]) , 
        sigma);
    }
  }
}

generated quantities{
  array[trials, participants] real log_lik;
  array[trials, participants] real<lower=0, upper=1> prior_preds;
  array[trials, participants] real<lower=0, upper=1> posterior_preds;

  real w1;
  real w2;
  real w1_prior;
  real w2_prior;
  

  w1_prior = 0.5 + inv_logit(normal_rng(0,1))/2; //generate prior distribution between (0.5; 1)
  w2_prior = 0.5 + inv_logit(normal_rng(0,1))/2; //generate prior distribution between (0.5; 1)
  w1 = 0.5 + inv_logit(weight1M)/2; //convert posterior back to interpretable space between (0.5; 1)
  w2 = 0.5 + inv_logit(weight2M)/2; //convert posterior back to interpretable space between (0.5; 1)

  
  for (participant in 1:participants){
    for (trial in 1:trials){  
      log_lik[trial, participant] = normal_lpdf(choice[trial, participant] | 
        weight_f(SourceSelf[trial, participant], weight1M + IDs[participant, 1]) + 
        weight_f(SourceOther[trial, participant], weight2M + IDs[participant, 2]), 
        sigma);
      
      prior_preds[trial, participant] = normal_lpdf(choice[trial, participant] | 
        weight_f(SourceSelf[trial, participant], w1_prior + IDs[participant, 1]) + 
        weight_f(SourceOther[trial, participant], w2_prior + IDs[participant, 2]) , 
        sigma);
        
      posterior_preds[trial, participant] = normal_lpdf(choice[trial, participant] | 
        weight_f(SourceSelf[trial, participant], weight1M + IDs[participant, 1]) + 
        weight_f(SourceOther[trial, participant], weight2M + IDs[participant, 2]) , 
        sigma);
    }
  }
}

