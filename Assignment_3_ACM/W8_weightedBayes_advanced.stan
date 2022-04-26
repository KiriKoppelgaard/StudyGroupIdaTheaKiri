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
  array[trials, participants] real<lower=0> choice; //decisions
  array[trials, participants] real<lower=0> SourceSelf;
  array[trials, participants] real<lower=0> SourceOther;
  vector[trials*participants] schizo;
  vector[trials*participants] control;
}

parameters {
  real weight1MC;//population level mean weight (Self)
  real weight2MC;//population level mean weight (Other)
  real weight1MS;//population level mean weight (Self)
  real weight2MS;//population level mean weight (Other)
  real<lower=0> sigmaC; //expected average error
  real<lower=0> sigmaS; //expected average error
  vector<lower=0>[2] tauC; //sd for population means, weight1M and weight2M (Vector containing 2 numbers)
  vector<lower=0>[2] tauS; //sd for population means, weight1M and weight2M (Vector containing 2 numbers)
  matrix[2, participants] z_IDCs; //z-scored individual deviation from population weight means (two for each participant)
  matrix[2, participants] z_IDSs; //z-scored individual deviation from population weight means (two for each participant)
  cholesky_factor_corr[2] L_u; //weight correlation
}

transformed parameters{
  matrix[participants, 2] IDCs; //non-scaled individual deviations from population weight means (two for each participant)
  matrix[participants, 2] IDSs; //non-scaled individual deviations from population weight means (two for each participant)

  IDSs = (diag_pre_multiply(tauS, L_u)*z_IDSs)'; //weight multiplication with consideration of correlation between individual deviations within participant
  IDCs = (diag_pre_multiply(tauC, L_u)*z_IDCs)'; //weight multiplication with consideration of correlation between individual deviations within participant
}


model {
  target += normal_lpdf(weight1MS | 0,1);//population level mean
  target += normal_lpdf(weight2MS | 0,1);//population level mean
  target += normal_lpdf(weight1MC | 0,1);//population level mean
  target += normal_lpdf(weight2MC | 0,1);//population level mean
  
  target += normal_lpdf(tauS[1]|0, .3) - normal_lccdf(0|0, .3); //population level sd
  target += normal_lpdf(tauS[2]|0, .3) - normal_lccdf(0|0, .3); //population level sd
  target += normal_lpdf(tauC[1]|0, .3) - normal_lccdf(0|0, .3); //population level sd
  target += normal_lpdf(tauC[2]|0, .3) - normal_lccdf(0|0, .3); //population level sd
  
  target += normal_lpdf(sigmaS|0, 1) - normal_lccdf(0|0, 1); //population level sd
  target += normal_lpdf(sigmaC|0, 1) - normal_lccdf(0|0, 1); //population level sd
  
  target += lkj_corr_cholesky_lpdf(L_u | 2); //lkj-prior ranging from -1 to 1, convient for correlations
  
  target += std_normal_lpdf(to_vector(z_IDCs)); //sampling the scaled individual deviations
  target += std_normal_lpdf(to_vector(z_IDSs)); //sampling the scaled individual deviations

  for (participant in 1:participants){
    for (trial in 1:trials){  
      target += normal_lpdf(choice[trial, participant] | 
        weight_f(SourceSelf[trial, participant], 
          schizo[participant]*(weight1MS + IDSs[participant, 1]) + 
          control[participant]*(weight1MC + IDCs[participant, 1])) + 
          
        weight_f(SourceOther[trial, participant], 
           schizo[participant]*(weight2MS + IDSs[participant, 2]) + 
           control[participant]*(weight2MC + IDCs[participant, 2])), 
          
        schizo[participant] * sigmaS + control[participant] * sigmaC);
    }
  }
}

generated quantities{
  array[trials, participants] real log_lik;
  array[participants]real<lower=0, upper=1> prior_preds;
  array[participants]real<lower=0, upper=1> posterior_preds;


  real w1S;
  real w1C;
  real w2S;
  real w2C;
  real w1_prior;
  real w2_prior;
  real w1_prior_t;
  real w2_prior_t;
  
  
  w1_prior_t = 0.5 + inv_logit(normal_rng(0,1))/2; //generate prior distribution between (0.5; 1)
  w2_prior_t = 0.5 + inv_logit(normal_rng(0,1))/2; //generate prior distribution between (0.5; 1)
  w1_prior = logit(w1_prior_t); //generate prior distribution between (0.5; 1)
  w2_prior = logit(w2_prior_t); //generate prior distribution between (0.5; 1)
  w1S = 0.5 + inv_logit(weight1MS)/2; //convert posterior back to interpretable space between (0.5; 1)
  w1C = 0.5 + inv_logit(weight1MC)/2; //convert posterior back to interpretable space between (0.5; 1)
  w2S = 0.5 + inv_logit(weight2MS)/2; //convert posterior back to interpretable space between (0.5; 1)
  w2C = 0.5 + inv_logit(weight2MC)/2; //convert posterior back to interpretable space between (0.5; 1)
  
  for (participant in 1:participants){
    for (trial in 1:trials){  
      log_lik[trial, participant] = normal_lpdf(choice[trial, participant] | 
        weight_f(SourceSelf[trial, participant], 
          schizo[participant]*(weight1MS + IDSs[participant, 1]) + 
          control[participant]*(weight1MC + IDCs[participant, 1])) + 
          
        weight_f(SourceOther[trial, participant], 
           schizo[participant]*(weight2MS + IDSs[participant, 2]) + 
           control[participant]*(weight2MC + IDCs[participant, 2])), 
          
        schizo[participant] * sigmaS + control[participant] * sigmaC);
    }
    
    //prior predictions
    prior_preds[participant] =  inv_logit(normal_rng( 
    weight_f(logit(mean(SourceSelf[,participant])), 
    schizo[participant]*(w1_prior + IDSs[participant, 1]) + 
    control[participant]*(w1_prior + IDCs[participant, 1])) + 
          
    weight_f(logit(mean(SourceOther[,participant])), 
    schizo[participant]*(w2_prior + IDSs[participant, 2]) + 
    control[participant]*(w2_prior + IDCs[participant, 2])), 
          
    schizo[participant] * sigmaS + control[participant] * sigmaC));
    
    //posterior predictions
    posterior_preds[participant] =  inv_logit(normal_rng( 
    weight_f(logit(mean(SourceSelf[,participant])), 
    schizo[participant]*(weight1MS + IDSs[participant, 1]) + 
    control[participant]*(weight1MC + IDCs[participant, 1])) + 
          
    weight_f(logit(mean(SourceOther[,participant])), 
    schizo[participant]*(weight2MS + IDSs[participant, 2]) + 
    control[participant]*(weight2MC + IDCs[participant, 2])), 
          
    schizo[participant] * sigmaS + control[participant] * sigmaC));
    
    
  }
  
  
  
  
}

