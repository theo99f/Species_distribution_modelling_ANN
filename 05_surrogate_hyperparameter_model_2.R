################################################################################
#                   Fitting linear surrogate models                            #
#
################################################################################
# Purpose: propose candidate configurations for tuning round 3.
# The surrogate only decides WHICH configs get tested; the final model is chosen
# by measured CV AUC (mean over seeds 1-3), never by surrogate predictions.
# Term rule: keep a term only if it improves BOTH AICc and LOOCV RMSE.
# Limitation: with ~100 points the surrogate extrapolates badly in sparse corners
# (e.g. depth 4 + high alpha, large batch + high lr); predictions there are probes.
# Selection rule: ties within 0.005 AUC go to the simpler config.

library(dplyr)
library(tidyr)

source("Functions.R")

################################################################################
#load hyperparameter configurations from both previous rounds
configurations_round1 <- read.csv("results/tuning/hyperparameter_configurations_round1.csv")
configurations_round2 <- read.csv("results/tuning/hyperparameter_configurations_round2.csv")

#combine data frames of both 
configurations_combined <- rbind(configurations_round1,configurations_round2) %>%
  separate_wider_delim(hidden, delim = ",", 
                       names = c("layer1", "layer2", "layer3","layer4"), 
                       too_few = "align_start") %>%
  mutate(
    # Count number of layers
    n_layers = case_when(
      !is.na(layer4) ~ 4,
      !is.na(layer3) ~ 3,
      !is.na(layer2) ~ 2,
      TRUE ~ 1
    ),
    # Convert layer columns to numeric
    layer1 = as.numeric(layer1),
    layer2 = as.numeric(layer2),
    layer3 = as.numeric(layer3),
    layer4 = as.numeric(layer4),
    converged = ifelse(is.na(mean_AUC), 0, 1)
  )%>%
  rowwise() %>%
  mutate(layer_mean_width = mean(c_across(starts_with("layer")), na.rm = TRUE))%>%
  ungroup() %>%
  arrange(desc(mean_AUC))  


#check mean AUC by layer type
aggregate(mean_AUC ~ n_layers, data = configurations_combined, FUN = mean)




#a list containing the formulas for tested GLMs
#list was expanded while testing based on results 
formulas <- c(
  #best model from previous tuning round with 3-way interaction for lambda alpha and depth
  m0 = mean_AUC~alpha*lambda*n_layers+lr,
  
  ##testing for an optimum in depth
  #model including a quadratic term for depth
  m1 = mean_AUC~alpha*lambda*n_layers+lr+I(n_layers^2),     #better than m0
  
  #model with depth as a factor
  m2 = mean_AUC~alpha*lambda*factor(n_layers)+lr,           #better than m1
  
  #model with depth as a factor without learning rate
  m3 = mean_AUC~alpha*lambda*factor(n_layers),              #rejected, worse than m2
                                                            #keep learning rate
  #model with depth as factor and quadratic lambda term
  m4 = mean_AUC~alpha*lambda*factor(n_layers)+lr+I(lambda^2),  #new best model!
  
  #added batchsize to new best model (m4)
  m5 = mean_AUC~alpha*lambda*factor(n_layers)+lr+I(lambda^2)+batchsize, #no gain over m4
  
  #added width to formula of m4
  m6 = mean_AUC~alpha*lambda*factor(n_layers)+lr+I(lambda^2) + layer_mean_width, #new best!
                                                                                 #better than m4
  #added quadratic term for width
  m7 = mean_AUC~alpha*lambda*factor(n_layers)+lr+I(lambda^2)+layer_mean_width+I(layer_mean_width^2), #rejected, no gain over m6
  
  #added width and batchsize
  m8 = mean_AUC~alpha*lambda*factor(n_layers)+lr+I(lambda^2)+layer_mean_width+batchsize, #better than m6
  
  #added width and quadratic term for batchsize
  m9 = mean_AUC~alpha*lambda*factor(n_layers)+lr+I(lambda^2)+layer_mean_width+batchsize+I(batchsize^2), #no gain over m8
  
  #added width and interaction between batchsize and learning rate
  m10 = mean_AUC~alpha*lambda*factor(n_layers)+lr+I(lambda^2)+layer_mean_width+batchsize*lr, #better than m8
  
  #added four width and interaction between batchsize and width
  m11 = mean_AUC~alpha*lambda*factor(n_layers)*layer_mean_width+I(lambda^2)+batchsize*lr, #no gain over m10
  
  #removed layer width from m10 with 
  m12 = mean_AUC~alpha*lambda*factor(n_layers)+lr+I(lambda^2)+batchsize*lr #no gain over m10
)

#Root mean square error over leave-one-out cross validation
LOOCV_RMSE <- sapply(formulas,leave_one_out_cv, data_fr=configurations_combined)


#fitting models on all 100 configurations
m0 <- glm(formulas$m0, data = configurations_combined)
m1 <- glm(formulas$m1, data = configurations_combined)
m2 <- glm(formulas$m2, data = configurations_combined)
m3 <- glm(formulas$m3, data = configurations_combined)
m4 <- glm(formulas$m4, data = configurations_combined)
m5 <- glm(formulas$m5, data = configurations_combined)
m6 <- glm(formulas$m6, data = configurations_combined)
m7 <- glm(formulas$m7, data = configurations_combined)
m8 <- glm(formulas$m8, data = configurations_combined)
m9 <- glm(formulas$m9, data = configurations_combined)
m10 <- glm(formulas$m10, data = configurations_combined)
m11 <- glm(formulas$m11, data = configurations_combined)
m12 <- glm(formulas$m12, data = configurations_combined)

#AIC calculation
AIC_raw <- AIC(m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12)
#correct AIC for small sample sizes n/k < 40
AICc <- AIC_raw$AIC+(2*AIC_raw$df^2+2*AIC_raw$df)/(nrow(configurations_combined)-AIC_raw$df-1)
names(AICc) <- rownames(AIC_raw)


#check results
cbind(LOOCV_RMSE, AICc)



#best model retains the three-way interaction between alpha, lambda and factor depth
#and the interaction between batchsize and learning rate and the main effect of layer width
final_glm <- glm(mean_AUC~alpha*lambda*factor(n_layers)+lr+I(lambda^2)+layer_mean_width+batchsize*lr, data = configurations_combined)


top_25_range <- configurations_combined[1:25,]

#Draw random 100000 configurations of hyperparameters
set.seed(87654321)
configs <- data.frame(
  #depth
  n_layers = sample(min(top_25_range$n_layers):max(top_25_range$n_layers), 100000, replace = TRUE),
  #width
  layer_mean_width = runif(100000, min(top_25_range$layer_mean_width),max(top_25_range$layer_mean_width)),
  #alpha
  alpha = runif(100000, min(top_25_range$alpha),max(top_25_range$alpha)),
  #log uniform distribution
  lambda = 10^-(runif(100000,-log10(max(top_25_range$lambda)),-log10(min(top_25_range$lambda)))),
  batchsize = sample(min(top_25_range$batchsize):max(top_25_range$batchsize), 100000, replace = TRUE),
  lr = 10^-(runif(100000, -log10(max(top_25_range$lr)), -log10(min(top_25_range$lr))))
)

#fix learning rate and batchsize to not influence the ranking of configurations
pred_data <- configs
pred_data$lr <- 0.05
pred_data$batchsize <- 48

#predict mean AUC with the linear surrogate model
configs$predicted_auc <- predict(final_glm, newdata= pred_data, type ="response")
configs$width_band <- ifelse(configs$layer_mean_width < mean(configs$layer_mean_width), "narrow", "wide")


#rank predicted configurations and pick the best for each depth x width cell with different lambda values                     
sampled_top_predicted_configs <- configs %>%
  group_by(n_layers, width_band) %>%                  #for each layer/ wide vs narrow width
  arrange(desc(predicted_auc), .by_group = TRUE) %>%  #take 100 best predictions
  slice_head(n = 100) %>%
  arrange(lambda, .by_group = TRUE) %>%
  slice(round(seq(1, n(), length.out = 3))) %>%   # 3 picks spread across lambda
  ungroup()%>%
  mutate(
    layer1 = as.integer(round(layer_mean_width)),
    layer2 = as.integer(ifelse(n_layers>=2, layer1,NaN)),
    layer3 = as.integer(ifelse(n_layers>=3, layer1,NaN)),
    layer4 = as.integer(ifelse(n_layers>=4, layer1,NaN)))

#combine sampled top predicted configs with the 5 best actual tested configurations 
round3_configs <-full_join(
  sampled_top_predicted_configs,
  slice_max(configurations_combined, mean_AUC, n=5))%>%
  select(all_of(colnames(sampled_top_predicted_configs)))
  
#replicate every row for three different seeds per configuration
run_table <- tidyr::crossing(round3_configs, seed = 1:3) %>% 
  arrange(seed)%>%
  mutate(run_id = row_number())

write.csv(run_table, "results/tuning/round3_run_table.csv", row.names = FALSE)

configs %>% group_by(n_layers, width_band) %>% arrange(desc(predicted_auc), .by_group = TRUE) %>%slice_head(n=100)%>%
  summarise(across(c(lr, batchsize), list(p10 = ~quantile(.x, .1), p90 = ~quantile(.x, .9))))