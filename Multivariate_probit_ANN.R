################################################################################
#                           Multivariate probit ANN                            #
#                                                                              #
#          Script for hyperparameter tuning, training and model selection      #
################################################################################

library(torch)
library(cito)
library(terra)
library(dplyr)
library(tidyr)
library(pROC)

source("Functions.R")

#loading preprocessed development set
directory <- "data/preprocessed_data/final_data_split/development_set/"
train_set_predictors <- read.csv(paste0(directory,"train_set_predictors.csv"))
full_response <- read.csv(paste0(directory,"train_set_response.csv"))

predictor_names <- colnames(train_set_predictors)
species_names <- colnames(full_response)[6:30]

train_set_response <- full_response %>%
  select(all_of(species_names))

##################Hyperparameter Screening######################################
#using standard 3-fold cross-validation with a broad range of hyperparameters
model_hp_screening <- dnn(X=train_set_predictors, 
                          Y=cbind(train_set_response), 
                          loss = "mvp",
                          validation = 0.1,
                          early_stopping = 15,
                          epochs = 200,
                          activation = "prelu",
                          lr_scheduler = config_lr_scheduler("reduce_on_plateau", 
                                                             patience = 5, factor = 0.5),
                          lr =tune(),
                          hidden = tune(lower = c(10L,1L), upper = c(50L,4L)),
                          batchsize = tune(16,64),
                          lambda = tune(0.0001,0.1),
                          alpha =tune(0, 0.5),
                          tuning = config_tuning(CV = 3, steps = 10))          
#Hyperparameter configurations
hp_tuning <- model_hp_screening$tuning
#sort the hyperparameter configuration models by the validation loss and save width and depth of hidden layers in seperate columns
hp_sorted <- hp_tuning %>%
  mutate(
    hidden_width = sapply(hidden, `[`, 1),
    hidden_layers = sapply(hidden, `[`, 2),
    converged = ifelse(is.infinite(test),0,1)
  ) %>%
  select(-hidden)%>%
  sort_by(hp_tuning$test)

# write to CSV
write.csv(hp_sorted, "results/hyperparameter_tuning_mvp.csv", row.names = FALSE)   

#################################################################################

hp_sorted <- read.csv("results/hyperparameter_tuning_mvp.csv")
#no significant effect of learning rate
cor.test(hp_sorted$lr, hp_sorted$test, method = "pearson")
mean(hp_sorted$lr)

#no significant effect of alpha
cor.test(hp_sorted$alpha[!is.infinite(hp_sorted$test)], hp_sorted$test[!is.infinite(hp_sorted$test)], method = "pearson")

#significant effect of lambda 
cor.test(hp_sorted$lambda[!is.infinite(hp_sorted$test)], hp_sorted$test[!is.infinite(hp_sorted$test)], method = "pearson")

#no significant effect of batch size
cor.test(hp_sorted$batchsize[!is.infinite(hp_sorted$test)], hp_sorted$test[!is.infinite(hp_sorted$test)], method = "pearson")

#no significant effect of network width 
cor.test(hp_sorted$hidden_width[!is.infinite(hp_sorted$test)], hp_sorted$test[!is.infinite(hp_sorted$test)], method = "pearson")

#no significant effect of network depth 
cor.test(hp_sorted$hidden_layers[!is.infinite(hp_sorted$test)], hp_sorted$test[!is.infinite(hp_sorted$test)])

#################################################################################

cv_split_folder <- "data/preprocessed_data/final_data_split/development_set/5-fold_grouped_cv_split/"
cv_indices <- readRDS(paste0(cv_split_folder,"grouped_5fold_cv_indeces.rds")) 


#hyperparameter tuning using random search for 10 configuration steps and 5fold grouped cv
for (config in 1:10) {
  configs <- list(
    loss = "binomial",
    hidden = c(sample(10:40, sample(1:2,1))),
    lambda = runif(1, 0.03, 0.1), #lambda was significant negative correlated thus increased
    alpha = runif(1, 0, 0.4),         #marginally significant thus slightly reduced
    lr = runif(1, 0.5, 0.7),          
    batchsize = 32
  )
  #cross-validate each hyperparameter configuration using 5 folds
  model <- cross_validate_ann(configs, train_set_predictors, train_set_response, cv_indices)
  
  #calculate mean AUC and mean TSS over species for each fold
  mean_metrics_fold <- sapply(model$performance, function(x) rowMeans(x))
  #calculate mean AUC and mean TSS over all folds for the whole configuration
  mean_metric_config <- rowMeans(mean_metrics_fold)[1:2]
  names(mean_metric_config) <- c("mean_AUC", "mean_TSS")
  
  #save configurations and their mean AUC and TSS
  configs$hidden <- paste0(configs$hidden, collapse = ",")
  configs_df <- as.data.frame(configs)
  configs_df <- cbind(configs_df, t(mean_metric_config))
  print(paste0(config,". Configuration was tested"))
  if(config>1){
    configuration <- rbind(configuration, configs_df)}
  else{configuration <- configs_df}
}
#save the hyperparameter search results
write.csv(configuration, file = "results/hyperparameter_configurations_mvp.csv")

################################################################################
library(parallel)
library(foreach)
library(doParallel)

# Setup parallel backend
n_cores <- 2  # Or detectCores() - 1
cl <- makeCluster(n_cores)
registerDoParallel(cl)

# Load necessary objects/functions to each worker
clusterExport(cl, c("cross_validate_ann", "train_set_predictors", 
                    "train_set_response", "cv_indices", "species_names",
                    "predictor_names", "evaluate_model"))

# Load required packages on each worker
clusterEvalQ(cl, {
  library(cito)
  library(dplyr)
  library(pROC)
})

# Parallel loop
results_list <- foreach(config = 1:10, .combine = rbind, 
                        .errorhandling = "pass") %dopar% {
                          
                          # Generate config
                          configs <- list(
                            loss = "mvp",
                            hidden = c(sample(10:40, sample(1:2, 1))),
                            lambda = runif(1, 0.03, 0.1),
                            alpha = runif(1, 0, 0.4),
                            lr = runif(1, 0.5, 0.7),
                            batchsize = 32
                          )
                          
                          # Cross-validate
                          model <- cross_validate_ann(configs, train_set_predictors, 
                                                      train_set_response, cv_indices)
                          
                          # Calculate metrics - FIX: handle the structure correctly
                          # model$performance is a list of matrices (one per fold)
                          # Each matrix has species as columns and metrics as rows
                          
                          # Extract AUC and TSS for each fold
                          auc_by_fold <- sapply(model$performance, function(x) x["AUC", ])
                          tss_by_fold <- sapply(model$performance, function(x) x["TSS", ])
                          
                          # Calculate means
                          if(is.matrix(auc_by_fold)) {
                            mean_AUC <- mean(auc_by_fold)
                            mean_TSS <- mean(tss_by_fold)
                          } else {
                            # Single fold case
                            mean_AUC <- mean(auc_by_fold)
                            mean_TSS <- mean(tss_by_fold)
                          }
                          
                          # Format output
                          configs_df <- data.frame(
                            loss = configs$loss,
                            hidden = paste0(configs$hidden, collapse = ","),
                            lambda = configs$lambda,
                            alpha = configs$alpha,
                            lr = configs$lr,
                            batchsize = configs$batchsize,
                            mean_AUC = mean_AUC,
                            mean_TSS = mean_TSS,
                            stringsAsFactors = FALSE
                          )
                          
                          return(configs_df)
                        }

# Stop cluster
stopCluster(cl)

# View results
print(results_list)

# Save results
write.csv(results_list, "results/hyperparameter_configurations_mvp.csv", 
          row.names = FALSE)
