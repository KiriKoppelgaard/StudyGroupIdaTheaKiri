Guide to the Matching Pennies Data

There are 3 standardised datasets available: students, schizophrenia and primates. Plus your own data (described at the end)

Common columns:
- Framing indicates whether the game was presented as a challenge with a human, or a slot machine. (nb. Only students got the slot machine)
- ID indicates the participant ID
- BotStrategy indicates the strategy employed by the bot: 
   . "RS" or "-1" is Biased Nash (60% random chance of getting right), # in students and primates 
   . SimpleReversal is 20 trials with 75% e.g. left; then 20 trials with 75% right # in schizophrenia
   . DoubleAlternation is alternating 2 trials 90% e.g. left, and 2 trials 90% right # in schizophrenia
   . WSLS or "-2" is win stay lose shift: if the bot wins, chooses the same as before, otherwise it changes # in students
   . "0-ToM" is a reinforcement learning strategy # in all datasets
   . "1-ToM" and "2-ToM" include representations of how the players infers the bot (see tomsup paper) # in students
- Trial indicates the trial number within that specific experimental block (playing against a specific bot)
- BotDecision: what the bot decided
- Decision: what the player decided
- Payoff: the outcome for the player: -1 is a loss, 1 is a win

Each dataset then has specific columns due to the experimental design.

1. Students 

The dataset also includes 
- Education - which education the students came from
- Year - which year they started their education

2. Schizophrenia

The dataset also includes
- RT, the reaction time in milliseconds
- Noise, the decision making is stochastic: what is the probability that the bot will follow the strategy in this trial? .9 means 90% probability, .75 75%
- Random indicates whether the bot followed the strategy or picked randomly
-  P1 indicates the bots probability of choosing 0
-  P2 indicates the bots probability of choosing 1
- Block indicates the "block" of trials (every time a bot changes strategy, there is a new block)
- points indicates the points accumulated by the player
- Order, indicates whether simple reversal preceded or followed the double alternation for this player 
- Group indicates whether the player was diagnosed with schizophrenia or part of the comparison group

3. Primates

The primates dataset contains many additional columns about the species and ecological niche of the players. Feel free to explore them. Here is only a list (for now?):
                                                                                                                                     - "opponent_level" 
- "date_time"
- "Species"
- "op_payoff"
- "RT"
- "handedness"
- "payoff_dir"
-"stay"
- "leave"
- "two_in_a_row"
- "no_sequence"
- "alt_leave"
- "alt_stay"
- "wsls"
- "prevpayoff"
- "commonName"
- "groupsize_mean"
- "groupsize_std"
- "ECV"
- "neocortex"
- "PFC"
- "telencephalon"
- "common name"
- "EQ"
- "Sex"
- "Age"
- "Lifespan"
- "source"
- "age/lifespan"
- "Agecat"
- "Mean social group size (in the wild)"
- "range of social groupe size (source)"
- "Average max social group size"
- "Max social group size"
- "Origin"
- "Rearing"
- "Social development"
- "Diet"
- "Flexibility Dietary index"
- "Habitus"
- "Score Terrestrial Habitus"
- "Hierarchy score (see here next for score decoding)"
- "Hierarchy score (1=strictly in both sexs, 2 strict but one sex, 3 less strict in one sex and strict in the other, 4 less strict in both)"
- "Hierarchy"
- "AgeLife"
- "eats_insects"
- "eats_anything"
- "Block"              


The students data was collected within the study described in Waade, P. T., Enevoldsen, K. C., Simonsen, A., & Fusaroli, R. Introducing tomsup: Theory of Mind Simulations Using Python.

The schizophrenia data was collected during a currently unpublished study.

The primates data was collected within the study described in Devaine, M., San-Galli, A., Trapanese, C., Bardino, G., Hano, C., Saint Jalme, M., ... & Daunizeau, J. (2017). Reading wild minds: A computational assay of Theory of Mind sophistication across seven primate species. PLoS computational biology, 13(11), e1005833. And enriched with ecological data by Arnault Vermillet.

4 - Your own data

- ID: your secret city ID
- BotStrategy:  
   . -2 = WSLS
   . -1 = Random Bias
   . 0, 1, 2 = respective ToM lvl
- Role: 0 is matcher, 1 mismatcher
- player.tom_role: role of the bot: 0 is matcher, 1 mismatcher
- Choice: 1 is head, 0 is tail
- BothChoice: 1 is head, 0 is tail
- Payoff: -1 is lose, 1 is win
- BotPayoff: -1 is lose, 1 is win
