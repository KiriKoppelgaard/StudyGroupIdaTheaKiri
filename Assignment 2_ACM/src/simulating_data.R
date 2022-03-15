
#import strategies: bias_detector_agent, ws2ls_agent, copy_agent, anti_agent, random_agent, wsls_agent
source('src/agent_strategies.R')

library(tidyverse)

## bias_detector_agent with random_agent
play <- function(Self_agent, Other_agent, n_trials, n_agents, staybias, leavebias) {

  for (agent in seq(n_agents)){
    
    #Initialise variables
    Self <- rep(NA, n_trials)
    Other <- rep(NA, n_trials)
    past_outcomes_other <- NULL
    past_outcomes_self <- NULL
    old_feedback <- NULL
    
    
    #specify first action, this is random (as implemented with random agent)
    Self[1] <- random_agent(1, 0.5)
    Other[1] <- random_agent(1, 0.5)
    
    if(Self[1] == Other[1]){
       Feedback = 1
     } else {
      Feedback = 0
    }
    
    #Loop through trials after first random trial: 
    for (i in 2:n_trials){
      old_feedback <- Feedback
      
      if (Self[i-1] == Other[i-1]){
        Feedback = 1
      } else {Feedback = 0} #Define feedback: match = 1, mismatch = 0
      
      #Self agent: Define agent functions
      if (Self_agent == 'bias_detector_agent'){
        #Save the other's previous actions (for bias_detector_agent)
        past_outcomes_other <- append(past_outcomes_other, Other[i-1])
        Self[i] <- bias_detector_agent(past_outcomes_other, weighted = FALSE, noise=0.1)
        
      } else if (Self_agent == 'random_agent') {
        Self[i] <- random_agent(noise=0.1) 
        
      } else if (Self_agent == 'ws2ls_agent') {
        Self[i] <- ws2ls_agent(prev_choice = Self[i-1], Feedback, old_feedback, noise=0.1)
      
      } else if (Self_agent == 'copy_agent'){
        Self[i] <- copy_agent(prev_choice = Other[i-1], noise=0.1)
        
      } else if (Self_agent == 'anti_agent'){
        Self[i] <- copy_agent(prev_choice = Other[i-1], noise=0.1)
      
      } else if (Self_agent =='wsls_agent'){
        Self[i] <- wsls_agent(self_prev_choice = Self[i-1], feedback = Feedback, staybias = staybias, leavebias = leavebias, noise=0)
      }
      
      #Other agent: Define agent functions
      if (Other_agent == 'bias_detector_agent'){
        
        past_outcomes_self <- append(past_outcomes_self, Self[i-1])
        Other[i] <- bias_detector_agent(past_outcomes_self, weighted = FALSE, noise=0.1)
        
      } else if (Other_agent == 'random_agent') {
        Other[i] <- random_agent(noise=0.1)  

      } else if (Other_agent == 'ws2ls_agent') {
        Other[i] <- ws2ls_agent(prev_choice = Other[i-1], feedback = 1 - Feedback, old_feedback = 1- old_feedback, noise=0.1)
      
      } else if (Other_agent == 'copy_agent'){
        Other[i] <- copy_agent(prev_choice = Self[i-1], noise=0.1)
        
      } else if (Other_agent == 'anti_agent'){
        Other[i] <- anti_agent(prev_choice = Self[i-1], noise=0.1)
        
      } else if (Other_agent =='wsls_agent'){
        Other[i] <- wsls_agent(self_prev_choice = Other[i-1], feedback = 1 - Feedback, leavebias = leavebias, staybias = staybias, noise=0)
      }
      
      
    }
    
    #Save outcome
    temp <- tibble(Self_agent, Self, Other_agent, Other, trial = seq(n_trials), Feedback_Self = as.numeric(Self==Other),  agent, staybias, leavebias)
  
    #Append after first agent
    if (agent==1 ){df1 <- temp} else {df1 <- bind_rows(df1, temp)}
    
  }
  write.csv(df1, paste('data/', Self_agent, '_', Other_agent, leavebias, '_', staybias, '.csv', sep = ''), row.names = FALSE)
}

agents = c('bias_detector_agent', 'ws2ls_agent', 'copy_agent', 'anti_agent', 'random_agent', 'wsls_agent')

for (leavebias in seq(0, 1, 0.1)){ 
  for (staybias in seq(0, 1, 0.1)){ 
    play(Self_agent = 'wsls_agent', Other_agent = 'random_agent', n_trials = 200, n_agents = 1, staybias = staybias, leavebias = leavebias)
  }
}



for (file in list.files(path = 'data/')){
  temp <- read.csv(paste('data/', file, sep = ''))

  #Recoding variables to stay and leave bias: 1 = heads, -1 = tails
  temp$win[temp$Feedback_Self == 1 & temp$Self == 1] <- 1
  temp$win[temp$Feedback_Self == 1 & temp$Self == 0] <- -1
  temp$win[temp$Feedback_Self == 0] <- 0

  temp$lose[temp$Feedback_Self == 0 & temp$Self == 1] <- -1
  temp$lose[temp$Feedback_Self == 0 & temp$Self == 0] <- 1
  temp$lose[temp$Feedback_Self == 1] <- 0

  if(exists("recovery_df")){recovery_df <- rbind(recovery_df, temp)} else {recovery_df <- temp}
  file.remove(paste('data/', file, sep = ''))
}

write.csv(recovery_df, paste('data/', 'simulated_data.csv', sep = ''), row.names = FALSE)


#for (agent in agents){
#  Self_agent = 'wsls_agent'
#  Other_agent = agent
#  play(Self_agent, Other_agent, n_trials = 120, n_agents = 100)
#}
