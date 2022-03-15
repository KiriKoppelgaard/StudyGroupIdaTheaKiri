#---------------------------------------------------------------------------
#AGENT STRATEGIES - FUNCTIONS FOR PLAYING THE MATCHING PENNIES GAME
#Study Group 7: Kiri, Thea & Ida
#Advanced Cognitive Modeling - Cognitive Science, Aarhus University
#Spring 2022
#---------------------------------------------------------------------------

#---RANDOM
random_agent <- function(bias=0.5, noise=0) { #default bias is 0.5, and no chance of noise
  choice <- rbinom(1, 1, bias)
  
  #'noise' is the probability of disregarding choice and choosing randomly with a 50-50 chance:
  if ( rbinom(1, 1, noise)==1 ) {
    choice <- rbinom(1, 1, 0.5)
  }
  
  return(choice)
}

#---WIN-STAY-LOSE-SHIFT
wsls_agent <- function(self_prev_choice, feedback, staybias, leavebias, noise=0) { 
  #prev_choice = self's previous choice
  #feedback = 0 (no match/loss) or 1 (match/win) from previous round
  
  if (feedback == 1) {
      choice <- ifelse(rbinom(1, 1, staybias) == 1,
      self_prev_choice,
      rbinom(1, 1, 0.5))
  } else if (feedback == 0) {
    choice <- ifelse(rbinom(1, 1, leavebias) == 1,
                     (1 - self_prev_choice),
                     rbinom(1, 1, 0.5))
  }
  
  #noise, chance of choosing randomly:
  if ( rbinom(1, 1, noise)==1 ) {
    choice <- rbinom(1, 1, 0.5)
  }
  
  return(choice)
}

#---WIN-STAY-TWO-LOSE-SHIFT
ws2ls_agent <- function(prev_choice, feedback, old_feedback, noise=0) {
  #prev_choice = opponent's previous choice
  #feedback = 0 (no match/loss) or 1 (match/win) from previous round
  #old_feedback = feedback from TWO rounds ago
  
  if (feedback == 1 || old_feedback == 1) {
    choice <- prev_choice #if at least one of the two last rounds were wins, keep same choice from last round
  }
  else if (feedback == 0 && old_feedback == 0) {
    choice <- 1 - prev_choice #if last two rounds were losses, make opposite choice from last round
  }
  
  #noise, chance of choosing randomly:
  if ( rbinom(1, 1, noise)==1 ) {
    choice <- rbinom(1, 1, 0.5)
  }
  
  return(choice)
}

#---COPYER
copy_agent <- function(prev_choice, noise=0) {
  #prev_choice = opponent's previous choice on the last round

  choice <- prev_choice #simply copy whatever the opponent did last round
  
  #noise, chance of choosing randomly:
  if ( rbinom(1, 1, noise)==1 ) {
    choice <- rbinom(1, 1, 0.5)
  }
  
  return(choice)
}

#---ANTI-COPYER / Rasmus modsat :-)
anti_agent <- function(prev_choice, noise=0) {
  #prev_choice = opponent's previous choice on the last round
  
  choice <- 1 - prev_choice #do the opposite of whatever the opponent did last round
  
  #noise, chance of choosing randomly:
  if ( rbinom(1, 1, noise)==1 ) {
    choice <- rbinom(1, 1, 0.5)
  }
  
  return(choice)
}

#---BIAS DETECTOR
bias_detector_agent <- function(input, weighted=TRUE, noise=0) { #input is a vector of all opponent's past choices (most recent last)
  memory <- 10 #simplified: memory is now always 10
  weight_list <- c(0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 7, 10) #hard-coded weight list (more recent options weighted more heavily)
  
  #if 10 trials haven't been played yet, just take simple average of all past opponent choices:
  if (length(input) < memory) {
    choice <- ifelse(mean(input) >= 0.5, 1, 0)
  }
  #otherwise, take (weighted) average of their past 10 choices:
  else {
    past <- tail(input, memory) #only look at the last 10
    choice <- ifelse(mean(past) >= 0.5, 1, 0)
    
    if (weighted == TRUE) {
      #take weighted average:
      choice <- ifelse(weighted.mean(past, weight_list) >= 0.5, 1, 0)
    }
  }
  
  #Potential for noise:
  if ( rbinom(1, 1, noise)==1 ) {
    choice <- rbinom(1, 1, 0.5)
  }
  
  return(choice)
}