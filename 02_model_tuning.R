################################################################################
#                           Multivariate probit ANN                            #
#                                                                              #
#       Script for hyper-parameter tuning, training and model selection        #
################################################################################

library(torch)
library(cito)
library(terra)
library(dplyr)
library(tidyr)
library(pROC)
library(parallel)
library(foreach)
library(doParallel)

source("Functions.R")

# Random search hyperparameter tuning with 5-fold cross-validation in parallel loop
# Setup parallel backend
n_cores <- detectCores()-1
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

# Parallel loop for random search hyperparameter tuning
results_list <- foreach(config = 1:3, 
                        .errorhandling = "pass") %dopar% {
                          #random sampling of a configuration
                          configs <- list(
                            loss = "mvp",
                            hidden = c(sample(15:45, sample(1:3, 1))),
                            lambda = 10^-(runif(1, 1, 6)),
                            alpha = runif(1, 0, 1),
                            #log uniform distribution between 0.001 and 0.1
                            lr = 10^-(runif(1, 1, 3)),
                            batchsize = sample(16:96, 1)
                          )
                          #
                          result <- tryCatch({
                            model <- cross_validate_ann(configs, train_set_predictors, 
                                                        train_set_response, cv_indices)
                            
                            auc_by_fold <- sapply(model$performance, function(x) mean(x["AUC_species", ]))
                            tss_by_fold <- sapply(model$performance, function(x) mean(x["TSS_species", ]))
                            
                            list(
                              loss = configs$loss,
                              hidden = paste0(configs$hidden, collapse = ","),
                              lambda = configs$lambda,
                              alpha = configs$alpha,
                              lr = configs$lr,
                              batchsize = configs$batchsize,
                              mean_AUC = mean(auc_by_fold),
                              sd_AUC = sd(auc_by_fold),
                              mean_TSS = mean(tss_by_fold),
                              sd_TSS = sd(tss_by_fold),
                              status = "ok"
                            )
                          }, error = function(e) {
                            list(
                              loss = configs$loss,
                              hidden = paste0(configs$hidden, collapse = ","),
                              lambda = configs$lambda,
                              alpha = configs$alpha,
                              lr = configs$lr,
                              batchsize = configs$batchsize,
                              mean_AUC = NA,
                              sd_AUC = NA,
                              mean_TSS = NA,
                              sd_TSS = NA,
                              status = paste("error:", conditionMessage(e))
                            )
                          })
                          
                          return(result)
                        }
# Stop cluster
stopCluster(cl)

# Combine results into data frame AFTER parallel loop
results_df <- do.call(rbind, lapply(results_list, function(x) {
  data.frame(
    loss = x$loss,
    hidden = x$hidden,
    lambda = x$lambda,
    alpha = x$alpha,
    lr = x$lr,
    batchsize = x$batchsize,
    mean_AUC = x$mean_AUC,
    sd_AUC = x$sd_AUC,
    mean_TSS = x$mean_TSS,
    sd_TSS = x$sd_TSS,
    status = x$status,
    stringsAsFactors = FALSE
  )
}))

# View results
print(results_df)

# Save results
write.csv(results_df, "results/hyperparameter_configurations_mvp.csv", 
          row.names = FALSE)
# Results: AUC 0.50-0.82, training instability observed

#load hyperparameter search results and test for correlations
configuration <- read.csv("results/hyperparameter_configurations_mvp.csv")
configuration <- configuration %>%
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
  sort_by(configuration$mean_AUC)



# Exploratory Round 2 (n=10, LR 0.3-0.7, narrowed ranges)  
library(parallel)
library(foreach)
library(doParallel)

# Setup parallel backend
n_cores <- 3
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

# Parallel loop - FIX: use list instead of rbind
results_list <- foreach(config = 1:10, 
                        .errorhandling = "pass") %dopar% {
                          
                          # Generate config
                          configs <- list(
                            loss = "mvp",
                            hidden = c(sample(10:45, sample(1:2, 1))), #still tuned
                            lambda = runif(1, 0.01, 0.05), #adjusted (as significant) to include the 3 best configs
                            alpha = 0.25,   #roughly best model
                            lr = 0.4,       #roughly best model
                            batchsize = 32
                          )
                          
                          # Cross-validate
                          model <- cross_validate_ann(configs, train_set_predictors, 
                                                      train_set_response, cv_indices)
                          
                          # Extract metrics
                          auc_by_fold <- sapply(model$performance, function(x) mean(x["AUC_species", ]))
                          tss_by_fold <- sapply(model$performance, function(x) mean(x["TSS_species", ]))
                          
                          # Calculate means across folds
                          mean_AUC <- mean(auc_by_fold)
                          mean_TSS <- mean(tss_by_fold)
                          
                          # Format output - ensure all values are scalar
                          result <- list(
                            loss = configs$loss,
                            hidden = paste0(configs$hidden, collapse = ","),
                            lambda = configs$lambda,
                            alpha = configs$alpha,
                            lr = configs$lr,
                            batchsize = configs$batchsize,
                            mean_AUC = mean_AUC,
                            mean_TSS = mean_TSS
                          )
                          
                          return(result)
                        }

# Stop cluster
stopCluster(cl)

# Combine results into data frame AFTER parallel loop
results_df <- do.call(rbind, lapply(results_list, function(x) {
  data.frame(
    loss = x$loss,
    hidden = x$hidden,
    lambda = x$lambda,
    alpha = x$alpha,
    lr = x$lr,
    batchsize = x$batchsize,
    mean_AUC = x$mean_AUC,
    mean_TSS = x$mean_TSS,
    stringsAsFactors = FALSE
  )
}))

# View results
print(results_df)

# Save results
write.csv(results_df, "results/hyperparameter_configurations_2_mvp.csv", 
          row.names = FALSE)

configuration2 <- read.csv("results/hyperparameter_configurations_2_mvp.csv")
configuration2 <- configuration2 %>%
  separate_wider_delim(hidden, delim = ",", names = c("layer1", "layer2"), too_few = "align_start")%>%
  mutate(layer= ifelse(is.na(layer2),1,2))%>%
  sort_by(configuration2$mean_AUC)

#lambda not significant
cor.test(configuration2$lambda, configuration2$mean_AUC, method = "pearson")
cor.test(configuration2$lambda, configuration2$mean_TSS, method = "pearson")

#layer width not significant among one hidden layer models
cor.test(as.numeric(configuration2$layer1[!is.na(configuration2$layer2)]), 
         configuration2$mean_AUC[!is.na(configuration2$layer2)], method = "pearson")
cor.test(as.numeric(configuration2$layer1[!is.na(configuration2$layer2)]), 
         configuration2$mean_TSS[!is.na(configuration2$layer2)], method = "pearson")

#significant correlation between number of layers and performance
cor.test(configuration2$layer, configuration2$mean_AUC)
cor.test(configuration2$layer, configuration2$mean_TSS)

#significant difference between one and two layers
wilcox.test(mean_AUC~layer,configuration2)
wilcox.test(mean_TSS~layer,configuration2)


#fitting candidate models
config30_ll <- list(loss = "mvp",
                 hidden = c(30),
                 lambda = 0.025, 
                 alpha = 0.25, 
                 lr = 0.4,
                 batchsize = 32)

config30_hl <- list(loss = "mvp",
                    hidden = c(30),
                    lambda = 0.035, 
                    alpha = 0.25, 
                    lr = 0.4,
                    batchsize = 32)

config35_ll <- list(loss = "mvp",
                    hidden = c(35),
                    lambda = 0.025, 
                    alpha = 0.25, 
                    lr = 0.4,
                    batchsize = 32)

model_mvp_30_ll <- cross_validate_ann(config30_ll, train_set_predictors, train_set_response, cv_indices)
model_mvp_30_hl <- cross_validate_ann(config30_hl, train_set_predictors, train_set_response, cv_indices)
model_mvp_35_ll <- cross_validate_ann(config30_ll, train_set_predictors, train_set_response, cv_indices)

#training instability is observed, loss only decreases when learning rate is reduced

#testing lower learning rate and lower lambda on a promising model from exploration phase
test <- dnn(X=train_set_predictors[cv_indices[[3]]$train,], 
            Y=cbind(train_set_response[cv_indices[[3]]$train,]), 
            loss = "mvp",
            validation = 0.1,
            early_stopping = 15,
            epochs = 200,
            activation = "prelu",
            lr_scheduler = config_lr_scheduler("reduce_on_plateau", 
                                               patience = 5, factor = 0.5),
            lr = 0.01,
            hidden = c(30),
            batchsize = 32,
            lambda = 0.01,
            alpha = 0.25)

test_eval <- evaluate_model(test,val_pred3,val_resp3)
#convergence more stable, lower loss!!
#indicates lower learning rate works better, higher learning rate even when 
#learning rate scheduler can miss loss functions optimum 

################################################################################
#                        MVP Hyperparameter optimization                       #
#                            (Used for final models)                           #
################################################################################


#loading cross validation split indices
cv_split_folder <- "data/preprocessed_data/final_data_split/development_set/5-fold_grouped_cv_split/"
cv_indices <- readRDS(paste0(cv_split_folder,"grouped_5fold_cv_indeces.rds")) 

#Random search hyperparameter tuning with 5-fold cross-validation in parallel loop
library(parallel)
library(foreach)
library(doParallel)

# Setup parallel backend
n_cores <- 3
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
results_list <- foreach(config = 1:20, 
                        .errorhandling = "pass") %dopar% {
                          
                          # Generate config
                          configs <- list(
                            loss = "mvp",
                            hidden = c(sample(10:45, sample(1:3, 1))),
                            lambda = runif(1, 0.0001, 0.1),
                            alpha = runif(1, 0, 0.5),
                            lr = runif(1, 0.001, 0.2),
                            batchsize = sample(16:64, 1)
                          )
                          
                          # Cross-validate
                          model <- cross_validate_ann(configs, train_set_predictors, 
                                                      train_set_response, cv_indices)
                          
                          # Extract metrics
                          auc_by_fold <- sapply(model$performance, function(x) mean(x["AUC_species", ]))
                          tss_by_fold <- sapply(model$performance, function(x) mean(x["TSS_species", ]))
                          
                          # Calculate means across folds
                          mean_AUC <- mean(auc_by_fold)
                          mean_TSS <- mean(tss_by_fold)
                          
                          # Format output - ensure all values are scalar
                          result <- list(
                            loss = configs$loss,
                            hidden = paste0(configs$hidden, collapse = ","),
                            lambda = configs$lambda,
                            alpha = configs$alpha,
                            lr = configs$lr,
                            batchsize = configs$batchsize,
                            mean_AUC = mean_AUC,
                            mean_TSS = mean_TSS
                          )
                          
                          return(result)
                        }

# Stop cluster
stopCluster(cl)

# Combine results into data frame AFTER parallel loop
results_df <- do.call(rbind, lapply(results_list, function(x) {
  data.frame(
    loss = x$loss,
    hidden = x$hidden,
    lambda = x$lambda,
    alpha = x$alpha,
    lr = x$lr,
    batchsize = x$batchsize,
    mean_AUC = x$mean_AUC,
    mean_TSS = x$mean_TSS,
    stringsAsFactors = FALSE
  )
}))

# View results
print(results_df)

# Save results
write.csv(results_df, "results/mvp_hyperparameter_optimization_20_configs.csv", 
          row.names = FALSE)


#load hyperparameter search results and test for correlations
configuration <- read.csv("results/mvp_hyperparameter_optimization_20_configs.csv")
configuration <- configuration %>%
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
  rowwise()%>%
  mutate(
    network_capacity= sum(c(layer1, layer2, layer3), na.rm = T))%>%
  ungroup()%>%
  sort_by(configuration$mean_AUC)

#lambda significant, negative correlated with performance
cor.test(configuration$lambda, configuration$mean_AUC, method = "pearson")
cor.test(configuration$lambda, configuration$mean_TSS, method = "pearson")

#learning rate not significant
cor.test(configuration$lr, configuration$mean_AUC, method = "pearson")
cor.test(configuration$lr, configuration$mean_TSS, method = "pearson")

#layer width not significant among one hidden layer models
cor.test(as.numeric(configuration$layer1[!is.na(configuration$layer2)]), 
         configuration$mean_AUC[!is.na(configuration$layer2)], method = "pearson")
cor.test(as.numeric(configuration$layer1[!is.na(configuration$layer2)]), 
         configuration$mean_TSS[!is.na(configuration$layer2)], method = "pearson")

#significant correlation between number of layers and performance
cor.test(configuration$n_layers, configuration$mean_AUC)
cor.test(configuration$n_layers, configuration$mean_TSS)

#no significant difference between one and two layers p=0.15
wilcox.test(mean_AUC~n_layers,configuration[is.na(configuration$layer3),])
wilcox.test(mean_TSS~n_layers,configuration[is.na(configuration$layer3),])

#alpha not significant
cor.test(configuration$alpha, configuration$mean_AUC, method = "pearson")
cor.test(configuration$alpha, configuration$mean_TSS, method = "pearson")

#batchsize is significant
cor.test(configuration$batchsize, configuration$mean_AUC, method = "pearson")
cor.test(configuration$batchsize, configuration$mean_TSS, method = "pearson")

cor.test(configuration$network_capacity, configuration$mean_AUC, method = "pearson")

plot(configuration$network_capacity, configuration$mean_AUC)

################################################################################
# Setup parallel backend
n_cores <- 3
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
results_list <- foreach(config = 1:10, 
                        .errorhandling = "pass") %dopar% {
                          
                          # Generate config
                          configs <- list(
                            loss = "mvp",
                            hidden = c(sample(10:45, sample(1:2, 1))),
                            lambda = runif(1, 0.0001, 0.031),
                            alpha = 0.15,     #intermediate value roughly best config
                            lr = 0.03,
                            batchsize = sample(32:96, 1)
                          )
                          
                          # Cross-validate
                          model <- cross_validate_ann(configs, train_set_predictors, 
                                                      train_set_response, cv_indices)
                          
                          # Extract metrics
                          auc_by_fold <- sapply(model$performance, function(x) mean(x["AUC_species", ]))
                          tss_by_fold <- sapply(model$performance, function(x) mean(x["TSS_species", ]))
                          
                          # Calculate means across folds
                          mean_AUC <- mean(auc_by_fold)
                          mean_TSS <- mean(tss_by_fold)
                          
                          # Format output - ensure all values are scalar
                          result <- list(
                            loss = configs$loss,
                            hidden = paste0(configs$hidden, collapse = ","),
                            lambda = configs$lambda,
                            alpha = configs$alpha,
                            lr = configs$lr,
                            batchsize = configs$batchsize,
                            mean_AUC = mean_AUC,
                            mean_TSS = mean_TSS
                          )
                          
                          return(result)
                        }
# Stop cluster
stopCluster(cl)

# Combine results into data frame AFTER parallel loop
results_df <- do.call(rbind, lapply(results_list, function(x) {
  data.frame(
    loss = x$loss,
    hidden = x$hidden,
    lambda = x$lambda,
    alpha = x$alpha,
    lr = x$lr,
    batchsize = x$batchsize,
    mean_AUC = x$mean_AUC,
    mean_TSS = x$mean_TSS,
    stringsAsFactors = FALSE
  )
}))

# View results
print(results_df)

# Save results
write.csv(results_df, "results/mvp_hyperparameter_optimization_10_configs.csv", 
          row.names = FALSE)

#load hyperparameter search results and test for correlations
configuration <- read.csv("results/mvp_hyperparameter_optimization_10_configs.csv")
configuration <- configuration %>%
  separate_wider_delim(hidden, delim = ",", 
                       names = c("layer1", "layer2"), 
                       too_few = "align_start") %>%
  mutate(
    # Count number of layers
    n_layers = ifelse(is.na(layer2), 1, 2),
    # Convert layer columns to numeric
    layer1 = as.numeric(layer1),
    layer2 = as.numeric(layer2)
  )%>%
  rowwise()%>%
  mutate(
    network_capacity= sum(c(layer1, layer2), na.rm = T))%>%
  ungroup()%>%
  sort_by(configuration$mean_AUC)

#lambda significant, negative correlated with performance
cor.test(configuration$lambda, configuration$mean_AUC, method = "pearson")
cor.test(configuration$lambda, configuration$mean_TSS, method = "pearson")

#batchsize not significant
cor.test(configuration$batchsize, configuration$mean_AUC, method = "pearson")
cor.test(configuration$batchsize, configuration$mean_TSS, method = "pearson")

#layer width not significant among one hidden layer models
cor.test(as.numeric(configuration$layer1[!is.na(configuration$layer2)]), 
         configuration$mean_AUC[!is.na(configuration$layer2)], method = "pearson")
cor.test(as.numeric(configuration$layer1[!is.na(configuration$layer2)]), 
         configuration$mean_TSS[!is.na(configuration$layer2)], method = "pearson")

#no significant correlation between number of layers and performance
cor.test(configuration$n_layers, configuration$mean_AUC)
cor.test(configuration$n_layers, configuration$mean_TSS)

#no significant difference between one and two layers 
wilcox.test(mean_AUC~n_layers,configuration)
wilcox.test(mean_TSS~n_layers,configuration)


################################################################################
#FINAL model candidates

#testing 3 models with one hidden layer and 20, 25 and 30 neurons respectively
config20 <- list(loss = "mvp",
                 hidden = c(20),
                 lambda =  0.0001, 
                 alpha = 0.15, 
                 lr = 0.03,
                 batchsize = 32)

config25 <- list(loss = "mvp",
                 hidden = c(25),
                 lambda =  0.0001, 
                 alpha = 0.15,         
                 lr = 0.03,          
                 batchsize = 32)

config30 <- list(loss = "mvp",
                 hidden = c(30),
                 lambda =  0.0001, 
                 alpha = 0.15,         
                 lr = 0.03,          
                 batchsize = 32)

config40 <- list(loss = "mvp",
                 hidden = c(40),
                 lambda =  0.0001, 
                 alpha = 0.15,         
                 lr = 0.03,          
                 batchsize = 32)

config50 <- list(loss = "mvp",
                 hidden = c(50),
                 lambda =  0.0001, 
                 alpha = 0.15,         
                 lr = 0.03,          
                 batchsize = 32)

config55 <- list(loss = "mvp",
                 hidden = c(55),
                 lambda =  0.0001, 
                 alpha = 0.15,         
                 lr = 0.03,          
                 batchsize = 32)

model_cv_20 <- cross_validate_ann(config20, train_set_predictors, train_set_response, cv_indices)
saveRDS(model_cv_20, file = "results/candidate_models/mvp_ann/CV_models_hidden_20")

model_cv_25 <- cross_validate_ann(config25, train_set_predictors, train_set_response, cv_indices)
saveRDS(model_cv_25, file = "results/candidate_models/mvp_ann/CV_models_hidden_25")

model_cv_30 <- cross_validate_ann(config30, train_set_predictors, train_set_response, cv_indices)
saveRDS(model_cv_30, file = "results/candidate_models/mvp_ann/CV_models_hidden_30")

model_cv_40 <- cross_validate_ann(config40, train_set_predictors, train_set_response, cv_indices)
saveRDS(model_cv_40, file = "results/candidate_models/mvp_ann/CV_models_hidden_40")

model_cv_50 <- cross_validate_ann(config50, train_set_predictors, train_set_response, cv_indices)
saveRDS(model_cv_50, file = "results/candidate_models/mvp_ann/CV_models_hidden_50")

model_cv_55 <- cross_validate_ann(config55, train_set_predictors, train_set_response, cv_indices)
saveRDS(model_cv_55, file = "results/candidate_models/mvp_ann/CV_models_hidden_55")

#testing 3 models with two hidden layers
config30_25 <- list(loss = "mvp",
                    hidden = c(30,25),
                    lambda =  0.0001, 
                    alpha = 0.15,         
                    lr = 0.03,          
                    batchsize = 32)

config25_30 <- list(loss = "mvp",
                    hidden = c(25,30),
                    lambda =  0.0001, 
                    alpha = 0.15,         
                    lr = 0.03,          
                    batchsize = 32)

config27_27 <- list(loss = "mvp",
                    hidden = c(27,27),
                    lambda =  0.0001, 
                    alpha = 0.15,         
                    lr = 0.03,          
                    batchsize = 32)

model_cv_30_25 <- cross_validate_ann(config30_25, train_set_predictors, train_set_response, cv_indices)
saveRDS(model_cv_30_25, file = "results/candidate_models/mvp_ann/CV_models_hidden_30_25")

model_cv_25_30 <- cross_validate_ann(config25_30, train_set_predictors, train_set_response, cv_indices)
saveRDS(model_cv_25_30, file = "results/candidate_models/mvp_ann/CV_models_hidden_25_30")

model_cv_27_27 <- cross_validate_ann(config27_27, train_set_predictors, train_set_response, cv_indices)
saveRDS(model_cv_27_27, file = "results/candidate_models/mvp_ann/CV_models_hidden_27_27")

################################################################################


#reloading candidate models
model_cv_27_27 <- readRDS("results/candidate_models/mvp_ann/CV_models_hidden_27_27")
model_cv_30_25 <- readRDS("results/candidate_models/mvp_ann/CV_models_hidden_30_25")
model_cv_25_30 <- readRDS(file = "results/candidate_models/mvp_ann/CV_models_hidden_25_30")
model_cv_20 <- readRDS("results/candidate_models/mvp_ann/CV_models_hidden_20")
model_cv_25 <- readRDS("results/candidate_models/mvp_ann/CV_models_hidden_25")
model_cv_30 <- readRDS("results/candidate_models/mvp_ann/CV_models_hidden_30")
model_cv_40 <- readRDS("results/candidate_models/mvp_ann/CV_models_hidden_40")
model_cv_50 <- readRDS("results/candidate_models/mvp_ann/CV_models_hidden_50")
model_cv_55 <- readRDS("results/candidate_models/mvp_ann/CV_models_hidden_55")


AUC_cv_30_25 <- sapply(model_cv_30_25$performance, function(x){mean(x["AUC_species",])})
AUC_cv_25_30 <- sapply(model_cv_25_30$performance, function(x){mean(x["AUC_species",])})
AUC_cv_27_27 <- sapply(model_cv_27_27$performance, function(x){mean(x["AUC_species",])})
AUC_cv_20 <- sapply(model_cv_20$performance, function(x){mean(x["AUC_species",])})
AUC_cv_25 <- sapply(model_cv_25$performance, function(x){mean(x["AUC_species",])})
AUC_cv_30 <- sapply(model_cv_30$performance, function(x){mean(x["AUC_species",])})
AUC_cv_40 <- sapply(model_cv_40$performance, function(x){mean(x["AUC_species",])})
AUC_cv_50 <- sapply(model_cv_50$performance, function(x){mean(x["AUC_species",])})
AUC_cv_55 <- sapply(model_cv_55$performance, function(x){mean(x["AUC_species",])})

TSS_cv_30_25 <- sapply(model_cv_30_25$performance, function(x){mean(x["TSS_species",])})
TSS_cv_25_30 <- sapply(model_cv_25_30$performance, function(x){mean(x["TSS_species",])})
TSS_cv_27_27 <- sapply(model_cv_27_27$performance, function(x){mean(x["TSS_species",])})
TSS_cv_20 <- sapply(model_cv_20$performance, function(x){mean(x["TSS_species",])})
TSS_cv_25 <- sapply(model_cv_25$performance, function(x){mean(x["TSS_species",])})
TSS_cv_30 <- sapply(model_cv_30$performance, function(x){mean(x["TSS_species",])})
TSS_cv_40 <- sapply(model_cv_40$performance, function(x){mean(x["TSS_species",])})
TSS_cv_50 <- sapply(model_cv_50$performance, function(x){mean(x["TSS_species",])})
TSS_cv_55 <- sapply(model_cv_55$performance, function(x){mean(x["TSS_species",])})


# Prepare data
architectures <- c("1 layer: 20 neurons",
                   "1 layer: 25 neurons",
                   "1 layer: 30 neurons",
                   "1 layer: 40 neurons",
                   "1 layer: 50 neurons",
                   "1 layer: 55 neurons",
                   "2 layers: 27-27 neurons",
                   "2 layers: 25-30 neurons",
                   "2 layers: 30-25 neurons")

AUC_cv <- cbind(AUC_cv_20, AUC_cv_25, AUC_cv_30, AUC_cv_40, AUC_cv_50, AUC_cv_55,
                AUC_cv_27_27, AUC_cv_25_30, AUC_cv_30_25)
TSS_cv <- cbind(TSS_cv_20, TSS_cv_25, TSS_cv_30, TSS_cv_40, TSS_cv_50, TSS_cv_55,
                TSS_cv_27_27, TSS_cv_25_30, TSS_cv_30_25)

# Set up plotting area
par(mfrow = c(1, 2), mar = c(4, 10, 3, 2))

# TSS plot
boxplot(TSS_cv,
        horizontal = TRUE,
        names = architectures,
        las = 1,
        col = "lightblue",
        border = "darkblue",
        ylim = c(0.65, 0.71),
        xlab = "True Skill Statistic (TSS)",
        main = "TSS across CV folds")
abline(v = mean(TSS_cv), lty = 2, col = "gray50")

# AUC plot
boxplot(AUC_cv,
        horizontal = TRUE,
        names = architectures,
        las = 1,
        col = "lightcoral",
        border = "darkred",
        ylim = c(0.89, 0.915),
        xlab = "Area Under Curve (AUC)",
        main = "AUC across CV folds")
abline(v = mean(AUC_cv), lty = 2, col = "gray50")

# Reset plotting parameters
par(mfrow = c(1, 1))

colMeans(AUC_cv)
colMeans(TSS_cv)

sd(AUC_cv_30_25)
sd(TSS_cv_30_25)




