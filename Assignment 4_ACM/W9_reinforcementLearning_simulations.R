## RL simulation

pacman::p_load(tidyverse, cmdstanr)

agents <- 100
trials <- 120

# Useful functions
softmax <- function(x, tau) {
  outcome = 1 / (1 + exp(-tau * x))
  return(outcome)
}

ValueUpdate = function(value, alpha, choice, feedback) {
  
  PE <- feedback - value
  
  v1 <- value[1] + alpha * (1 - choice) * (feedback - value[1])

  v2 <- value[2] + alpha * (choice) * (feedback - value[2])
  
  updatedValue <- c(v1, v2)
  
  return(updatedValue)
}

value <- c(0,0)
alpha <- 0.9
temperature <- 1
choice <- 0
feedback <- -1
p <- 0.9 # probability that choice 0 gives a prize (1-p is probability that choice 1 gives a prize)

ValueUpdate(value, alpha, choice, feedback)

d <- tibble(trial = rep(NA, trials),
            choice = rep(NA, trials), 
            value1 = rep(NA, trials), 
            value2 = rep(NA, trials), 
            feedback = rep(NA, trials))

Bot <- rbinom(trials, 1, p)

for (i in 1:trials) {
    
    choice <- 1 #rbinom(1, 1, softmax(value[2] - value[1], temperature))
    feedback <- ifelse(Bot[i] == choice, 1, -1)
    value <- ValueUpdate(value, alpha, choice, feedback)
    d$choice[i] <- choice
    d$value1[i] <- value[1]
    d$value2[i] <- value[2]
    d$feedback[i] <- feedback
  
}

d <- d %>% mutate(
  trial = seq(trials),
  prevFeedback = lead(feedback))

ggplot(subset(d, trial < 21)) + 
  geom_line(aes(trial, value1), color = "green") + 
  geom_line(aes(trial, value2), color = "blue") +
  geom_line(aes(trial, prevFeedback), color = "red") +
  theme_bw()

# Let's imagine a situation where the underlying rate is 0.75


alpha <- 0.9
temperature <- 5
choice <- 0
feedback <- -1
p <- 0.75 # probability that choice 0 gives a prize (1-p is probability that choice 1 gives a prize)



df <- NULL
n <- 1
for (temperature in c(0.01, 0.5, 1, 5, 10, 15)) {
  
  for (alpha in seq(0.1, 1, 0.1)) {
    
    value <- c(0,0)
    d <- tibble(trial = rep(NA, trials),
                choice = rep(NA, trials), 
                value1 = rep(NA, trials), 
                value2 = rep(NA, trials), 
                feedback = rep(NA, trials),
                alpha = rep(NA, trials),
                temperature = rep(NA, trials),
                agent = n)
    
    for (i in 1:trials) {
      
      choice <- rbinom(1, 1, softmax(value[2] - value[1], temperature))
      feedback <- ifelse(Bot[i] == choice, 1, -1)
      value <- ValueUpdate(value, alpha, choice, feedback)
      d$trial[i] <- i
      d$choice[i] <- choice
      d$value1[i] <- value[1]
      d$value2[i] <- value[2]
      d$feedback[i] <- feedback
      d$alpha[i] <- alpha
      d$temperature[i] <- temperature
      
    }
    if (exists("df")) {df <- rbind(df, d)} else {df <- d}
    n <- n + 1
  }
}
df <- df %>% group_by(alpha, temperature) %>% mutate(
  prevFeedback = lead(feedback))

d1 <- df %>% subset(trial < 21 & temperature == 0.01)

ggplot() + 
  geom_line(data = subset(d1, alpha == 1), aes(trial, prevFeedback), color = "red") +
  geom_line(data = subset(d1, alpha == 0.9), aes(trial, value2), color = "purple") +
  geom_line(data = subset(d1, alpha == 0.5), aes(trial, value2), color = "blue") +
  geom_line(data = subset(d1, alpha == 0.2), aes(trial, value2), color = "green") +
  theme_bw()

df <- df %>% group_by(alpha, temperature) %>% mutate(
  rate = cumsum(choice) / seq_along(choice),
  performance = cumsum(feedback) / seq_along(feedback)
)

ggplot(subset(df, trial < 41), aes(trial, performance, group = alpha, color = alpha)) +
  geom_line(alpha = 0.5) +
  facet_wrap(.~temperature) +
  theme_bw()

ggplot(subset(df, trial < 41), aes(trial, performance, group = temperature, color = temperature)) +
  geom_line(alpha = 0.5) +
  facet_wrap(.~alpha) +
  theme_bw()


### Now onto model fitting

d <- df %>% subset(alpha == 0.6 & temperature == 5)

data <- list(
  trials = trials,
  choice = d$choice + 1,
  feedback = d$feedback
)

file <- file.path("/Users/au209589/Dropbox/My courses/2022 - AdvancedCognitiveModeling/W9_RL.stan")
mod <- cmdstan_model(file, cpp_options = list(stan_threads = TRUE), pedantic = TRUE)

samples <- mod$sample(
  data = data,
  seed = 123,
  chains = 2,
  parallel_chains = 2,
  threads_per_chain = 2,
  iter_warmup = 2000,
  iter_sampling = 2000,
  refresh = 1000,
  max_treedepth = 20,
  adapt_delta = 0.99,
)

samples$cmdstan_diagnose()
samples$summary() 

draws_df <- as_draws_df(samples$draws()) 

ggplot(draws_df, aes(.iteration, alpha, group = .chain, color = .chain)) +
  geom_line() +
  theme_classic()

ggplot(draws_df, aes(.iteration, temperature, group = .chain, color = .chain)) +
  geom_line() +
  theme_classic()


ggplot(draws_df) +
  geom_density(aes(alpha), fill = "blue", alpha = 0.3) +
  geom_density(aes(alpha_prior), fill = "red", alpha = 0.3) +
  xlab("Learning Rate") +
  ylab("Posterior Density") +
  theme_classic()

ggplot(draws_df) +
  geom_density(aes(temperature), fill = "blue", alpha = 0.3) +
  geom_density(aes(temperature_prior), fill = "red", alpha = 0.3) +
  xlab("(inverse) temperature") +
  ylab("Posterior Density") +
  theme_classic()



## Asymmetric
file <- file.path("/Users/au209589/Dropbox/My courses/2022 - AdvancedCognitiveModeling/W9_RL_asymmetric.stan")
mod <- cmdstan_model(file, cpp_options = list(stan_threads = TRUE), pedantic = TRUE)

samples <- mod$sample(
  data = data,
  seed = 123,
  chains = 2,
  parallel_chains = 2,
  threads_per_chain = 2,
  iter_warmup = 2000,
  iter_sampling = 2000,
  refresh = 1000,
  max_treedepth = 20,
  adapt_delta = 0.99,
)

## Multilevel
agents <- 100
trials <- 120 

df <- NULL
for (agent in 1:agents) {
  
  temperature <- boot::inv.logit(rnorm(1, -2, 0.3))*20
  alpha <- boot::inv.logit(rnorm(1, 1.1, 0.3))  
    
    value <- c(0,0)
    d <- tibble(trial = rep(NA, trials),
                choice = rep(NA, trials), 
                value1 = rep(NA, trials), 
                value2 = rep(NA, trials), 
                feedback = rep(NA, trials),
                alpha = alpha,
                temperature = temperature,
                agent = agent)
    
    for (i in 1:trials) {
      
      choice <- rbinom(1, 1, softmax(value[2] - value[1], temperature))
      feedback <- ifelse(Bot[i] == choice, 1, -1)
      value <- ValueUpdate(value, alpha, choice, feedback)
      d$trial[i] <- i
      d$choice[i] <- choice
      d$value1[i] <- value[1]
      d$value2[i] <- value[2]
      d$feedback[i] <- feedback
      d$alpha[i] <- alpha
      d$temperature[i] <- temperature
      
    }
    if (exists("df")) {df <- rbind(df, d)} else {df <- d}

}

df <- df %>% group_by(alpha, temperature) %>% mutate(
  prevFeedback = lead(feedback))

## Create the data

trials <- trials
agents <- agents

d_choice <- df %>% 
  subset(select = c(agent, choice)) %>% 
  mutate(row = rep(seq(trials),agents)) %>% 
  pivot_wider(names_from = agent, values_from = choice)

d_feedback <- df %>% 
  subset(select = c(agent, feedback)) %>% 
  mutate(row = rep(seq(trials),agents)) %>% 
  pivot_wider(names_from = agent, values_from = feedback)


data <- list(
  trials = trials,
  agents = agents,
  choice = as.matrix(d_choice[,2:(agents + 1)]),
  feedback = as.matrix(d_feedback[,2:(agents + 1)])
)

data$choice <- data$choice + 1

file <- file.path("/Users/au209589/Dropbox/My courses/2022 - AdvancedCognitiveModeling/W9_RL_multilevel.stan")
mod <- cmdstan_model(file, cpp_options = list(stan_threads = TRUE), pedantic = TRUE)

samples <- mod$sample(
  data = data,
  seed = 123,
  chains = 1,
  parallel_chains = 1,
  threads_per_chain = 4,
  iter_warmup = 2000,
  iter_sampling = 2000,
  refresh = 10,
  max_treedepth = 20,
  adapt_delta = 0.99,
)


