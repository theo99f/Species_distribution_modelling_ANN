################################################################################
#                   Fitting linear surrogate models                            #
#
################################################################################
# Purpose: propose candidate configurations for tuning round 2.
# The surrogate only decides WHICH configs get tested; the final model is chosen
# by measured CV AUC (mean over seeds 1-3), never by surrogate predictions.
# Term rule: keep a term only if it improves BOTH AICc and LOOCV RMSE.

library(dplyr)
library(tidyr)

source("Functions.R")
################################################################################
#load hyperparameter configurations
configurations <- read.csv("results/tuning/hyperparameter_configurations_round1.csv")

configurations <- configurations %>%
  separate_wider_delim(hidden, delim = ",", 
                       names = c("layer1", "layer2", "layer3"), 
                       too_few = "align_start") %>%
  mutate(
    # Count number of layers
    n_layers = case_when(
      !is.na(layer3) ~ 3,
      !is.na(layer2) ~ 2,
      TRUE ~ 1
    ),
    # Convert layer columns to numeric
    layer1 = as.numeric(layer1),
    layer2 = as.numeric(layer2),
    layer3 = as.numeric(layer3),
    converged = ifelse(is.na(mean_AUC), 0, 1)
  )%>%
  rowwise() %>%
  mutate(layer_mean_width = mean(c_across(starts_with("layer")), na.rm = TRUE))%>%
  sort_by(configurations$mean_AUC)

################################################################################
#fitting and testing of different surrogate models 

#testing the main effects separately
depth <- glm(mean_AUC~n_layers,data = configurations)
width <- glm(mean_AUC~layer_mean_width,data = configurations)
alpha <- glm(mean_AUC~alpha,data = configurations)
lambda <- glm(mean_AUC~lambda,data = configurations)
lr <- glm(mean_AUC~lr,data = configurations)
batchsize <- glm(mean_AUC~batchsize,data = configurations)

#AIC
AIC <- AIC(depth, width, alpha, lambda, lr, batchsize)
Anova <- anova(depth, alpha, lambda, lr, batchsize)
#corrected AIC for small sample sizes n/k < 40
AICc <- AIC$AIC+(2*AIC$df^2+2*AIC$df)/(nrow(configurations)-AIC$df-1)
names(AICc) <- rownames(AIC)

AICc
#Intermediate result: lambda appears to have the biggest marginal effect

#test quadratic lambda
quadratic_lambda <- glm(mean_AUC~lambda+I(lambda^2),data = configurations)

#test linear models with main effect of lambda and one additional main effect
lambda_alpha <- glm(mean_AUC~lambda+alpha, data = configurations)
lambda_lr <- glm(mean_AUC~lambda+lr, data = configurations)
lambda_batchsize <- glm(mean_AUC~lambda+batchsize, data = configurations)
lambda_depth <- glm(mean_AUC~lambda+n_layers, data = configurations)
lambda_width <- glm(mean_AUC~lambda+layer_mean_width, data = configurations)
#AIC
AIC <- AIC(lambda, quadratic_lambda, lambda_alpha, lambda_lr, lambda_batchsize, lambda_depth, lambda_width)
#corrected AIC for small sample sizes n/k < 40
AICc <- AIC$AIC+(2*AIC$df^2+2*AIC$df)/(nrow(configurations)-AIC$df-1)
names(AICc) <- rownames(AIC)

AICc
#Intermediate result: main effects of alpha and lambda show lowest AIC

#testing interactions between alpha and lambda and quadratic terms
interaction_alpha_lambda <- glm(mean_AUC~lambda+alpha+alpha*lambda, data = configurations)
quadratic_alpha_lambda <- glm(mean_AUC~lambda+I(lambda^2)+alpha+alpha*I(lambda^2), data = configurations)
quadratic_lambda_alpha_interaction <- glm(mean_AUC~lambda+I(lambda^2)+alpha+alpha*lambda, data = configurations)
quadratic_lambda_quadratic_alpha_interaction <- glm(mean_AUC~lambda+alpha+I(lambda^2)+I(alpha^2)+alpha*lambda, data = configurations)

#AIC
AIC <- AIC(lambda_alpha, 
           interaction_alpha_lambda, 
           quadratic_alpha_lambda, 
           quadratic_lambda_alpha_interaction, 
           quadratic_lambda_quadratic_alpha_interaction)
#corrected AIC for small sample sizes n/k < 40
AICc <- AIC$AIC+(2*AIC$df^2+2*AIC$df)/(50-AIC$df-1)
names(AICc) <- rownames(AIC)

AICc

#the interaction between alpha and lambda appears to explain most performance 
#variance with and without a quadratic term for lambda

#keeping interaction between lambda and alpha, and adding pairwise interactions 
#with remaining hyperparameters depth, learning rate and batchsize respectively
interaction_alpha_lambda_depth <- glm(mean_AUC~alpha+lambda+n_layers+alpha*lambda+lambda*n_layers+alpha*n_layers,data = configurations)
interaction_alpha_lambda_lr <- glm(mean_AUC~alpha+lambda+lr+alpha*lambda+lambda*lr+alpha*lr, data = configurations)
interaction_alpha_lambda_batchsize <- glm(mean_AUC~alpha+lambda+batchsize+alpha*lambda+lambda*batchsize+alpha*batchsize, data = configurations)

#AIC
AIC <- AIC(interaction_alpha_lambda_depth,
           interaction_alpha_lambda_lr,
           interaction_alpha_lambda_batchsize)
#corrected AIC for small sample sizes n/k < 40
AICc <- AIC$AIC+(2*AIC$df^2+2*AIC$df)/(50-AIC$df-1)
names(AICc) <- rownames(AIC)
AICc

#testing three way interaction terms with alpha and lambda and batchsize or depth respectively
full_triple_batchsize <- glm(mean_AUC ~ alpha*lambda*batchsize, data = configurations)
full_triple_depth <- glm(mean_AUC ~ alpha*lambda*n_layers, data = configurations)

#three way interaction with alpha, lambda and depth achieves best AICc
#testing main effects batchsize and learning rate in addition
full_triple_depth_batchsize <- glm(mean_AUC ~ alpha*lambda*n_layers + batchsize, data = configurations)
full_triple_depth_lr <- glm(mean_AUC ~ alpha*lambda*n_layers + lr, data = configurations)
full_triple_quadratic_lambda <-  glm(mean_AUC ~ alpha*lambda*n_layers+ I(lambda^2), data = configurations)

factor_depth <- glm(mean_AUC~alpha*lambda*factor(n_layers), data = configurations)

#AIC
AIC <- AIC(factor_depth,
           full_triple_batchsize,
           full_triple_depth,
           full_triple_depth_batchsize,
           full_triple_depth_lr,
           full_triple_quadratic_lambda)
#corrected AIC for small sample sizes n/k < 40
AICc <- AIC$AIC+(2*AIC$df^2+2*AIC$df)/(50-AIC$df-1)
names(AICc) <- rownames(AIC)

AICc



#verify candidate models with cross-validation and anova
leave_one_out_cv(mean_AUC ~ alpha*lambda)
leave_one_out_cv(mean_AUC~alpha+lambda+n_layers+alpha*lambda+lambda*n_layers+alpha*n_layers)
leave_one_out_cv(mean_AUC ~ alpha*lambda*n_layers)
leave_one_out_cv(mean_AUC ~ alpha*lambda*n_layers+lr)
leave_one_out_cv(mean_AUC ~ alpha*lambda*n_layers*lr)

anova(full_triple_depth_lr)
anova(full_triple_depth_batchsize)
anova(interaction_alpha_lambda_depth)
anova(full_triple_depth)
anova(full_triple_batchsize)
anova(full_triple_quadratic_lambda)


#Draw random 10000 configurations of hyperparameters
set.seed(12345678)
configs <- data.frame(
  #depth
  n_layers = sample(1:3, 100000, replace = TRUE),
  #random sampling of hyperparameter
  alpha = runif(100000, 0, 1),
  #log uniform distribution between 0.000001 and 0.1
  lambda = 10^-(runif(100000, 1, 6)),
  lr = 10^-(runif(100000, 1, 3))
)

#predict mean AUC with the linear surrogate model
configs$predicted_auc <- predict(full_triple_depth_lr, newdata= configs, type ="response")
#sort configurations per predicted AUC
top_predicted_configs <- configs %>% 
  sort_by(-configs$predicted_auc)%>% 
  arrange(desc(predicted_auc)) %>% 
  head(100)

saveRDS(top_predicted_configs, file = "results/tuning/top_predicted_configs.rds")

