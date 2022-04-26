// This script does SOMETHING


data {
  int<lower=1> trials;
  array[trials] int<lower=1,upper=2> choice;
  array[trials] int<lower=-1,upper=1> feedback;
  int<lower=0,upper=1> con1; // is 1 if it's condition 1, is 0 otherwise
  int<lower=0,upper=1> con2; // is 1 if it's condition 2, is 0 otherwise
}

transformed data {
  // here we specify that the initial values of the two choices is 0.5 (the same, will be random in softmax)
  vector[2] initValue; // initial values for V
  initValue = rep_vector(0.5, 2);
}


parameters {
  real<lower=0, upper=1> alpha; // learning rate
  real<lower=0> temperature; // softmax inv.temp.
}

model {
  real pe; // prediction error
  vector[2] value;
  vector[2] theta;
  
  target += uniform_lpdf(alpha | 0, 1); // BAD PRIOR!
  target += uniform_lpdf(temperature | 0, 20); // BAD PRIOR!
  
  value = initValue; // at start, value is initial value from transformed data block (just 0)
  
  for (t in 1:trials) {
    theta = softmax(temperature * value); // action prob. computed via softmax
    target += categorical_lpmf(choice[t] | theta); // choice is distributed according to a categorical with a rate of theta
    
    pe = feedback[t] - value[choice[t]]; // compute prediction error for chosen value only
    value[choice[t]] = value[choice[t]] + alpha * pe; // update chosen V
    
  }
}

generated quantities {
  real<lower=0, upper=1> alpha_prior;
  real<lower=0, upper=20> temperature_prior;
  
  real pr;
  vector[2] value; // expected value of choice 1 and 2 for every trial
  vector[2] theta; // probability of choosing choice 1 and 2 for every trial
  
  real log_lik;
  
  alpha_prior = uniform_rng(0,1); // BAD PRIOR!
  temperature_prior = uniform_rng(0,20); // BAD PRIOR!
  
  value = initValue;
  log_lik = 0;
  
  for (t in 1:trials) {
    theta = softmax(temperature * value); // action probability computed via softmax
    log_lik = log_lik + categorical_lpmf(choice[t] | theta); // instead of target, we just save it
    
    pe = feedback[t] - value[choice[t]]; // compute prediction error for chosen value only
    value[choice[t]] = value[choice[t]] + alpha * pr; //update chosen V
    
    
    }
  
}





