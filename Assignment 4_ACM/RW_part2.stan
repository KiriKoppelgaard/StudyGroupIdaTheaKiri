// This script models


data {
  int<lower=1> trials;
  array[trials] int<lower=1,upper=2> choice;
  array[trials] int<lower=-1,upper=1> feedback;
  vector[trials] con1; // is 1 if it's condition 1, is 0 otherwise
  vector[trials] con2; // is 1 if it's condition 2, is 0 otherwise
  real prior_alphaMean1;
  real <lower = 0> prior_alphaSD1;
  real prior_alphaMean2;
  real <lower = 0> prior_alphaSD2;
  real prior_temperatureMean;
  real <lower = 0> prior_temperatureSD;
}

transformed data {
  // here we specify that the initial values of the two choices is 0.5 (the same, will be random in softmax)
  vector[2] initValue; // initial values for V
  initValue = rep_vector(0.5, 2);
}


parameters {
  real alpha1; // learning rate for condition 1
  real alpha2; // learning rate for condition 2
  real temperature; // softmax inv.temp. 0.5 in our simulated data
}

model {
  real pe; // prediction error
  vector[2] value; //expected reward/value
  vector[2] theta; //probability of choosing 1
  
  //priors
  target += normal_lpdf(alpha1 | prior_alphaMean1, prior_alphaSD1);
  target += normal_lpdf(alpha2 | prior_alphaMean2, prior_alphaSD2); 
  target += normal_lpdf(temperature | prior_temperatureMean, prior_temperatureSD); 
  

  value = initValue; // at start, value is initial value from transformed data block (just 0.5)
  
  for (t in 1:trials) {
    theta = softmax(inv_logit(temperature)*20 * value); // action prob. computed via softmax
    target += categorical_lpmf(choice[t] | theta); // choice is distributed according to a categorical with a rate of theta
    
    pe = feedback[t] - value[choice[t]]; // compute prediction error for chosen value only
    value[choice[t]] = value[choice[t]] + con1[t]*inv_logit(alpha1)*pe + con2[t]*inv_logit(alpha2) *pe; // update chosen V
    
  }
}

generated quantities {
  real alpha1_prior;
  real alpha2_prior;
  real temperature_prior;
  
  real pe;
  vector[2] value; // expected value of choice 1 and 2 for every trial
  vector[2] theta; // probability of choosing choice 1 and 2 for every trial
  
  real log_lik;
  
  alpha1_prior = normal_rng(prior_alphaMean1,prior_alphaSD1);
  alpha2_prior = normal_rng(prior_alphaMean2,prior_alphaSD2);
  temperature_prior = normal_rng(prior_temperatureMean,prior_temperatureSD);
  
  value = initValue;
  log_lik = 0;
  
  for (t in 1:trials) {
    theta = softmax(inv_logit(temperature) * 20 * value); // action probability computed via softmax
    log_lik = log_lik + categorical_lpmf(choice[t] | theta); // instead of target, we just save it
    
    pe = feedback[t] - value[choice[t]]; // compute prediction error for chosen value only
    value[choice[t]] = value[choice[t]] + con1[t]*inv_logit(alpha1)*pe + con2[t]*inv_logit(alpha2) *pe; // update chosen V
    
    }
  
}





