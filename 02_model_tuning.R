################################################################################
#                      Hyperparameter tuning random search                     #
#                                                                              #
#                                     Round 1                                  #
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

# Create directory for tuning results and log if it doesn't exist
tuning_result_folder <- file.path("results","tuning")
log_file <- paste0(tuning_result_folder,"/tuning_progress.log")
if (!dir.exists(tuning_result_folder)) {
  dir.create(tuning_result_folder, recursive = TRUE)
}

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
results_list <- foreach(config = 1:50, 
                        .errorhandling = "pass") %dopar% {
                          #initializing a configuration as list
                          configs <- list(
                            loss = "mvp",
                            epochs = 200,
                            #random sampling of hyperparameter
                            hidden = c(sample(15:45, sample(1:3, 1),replace = T)),
                            alpha = runif(1, 0, 1),
                            #log uniform distribution between 0.001 and 0.1
                            lr = 10^-(runif(1, 1, 3)),
                            #log uniform distribution between 0.000001 and 0.1
                            lambda = 10^-(runif(1, 1, 6)),
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
                          #save a print message to a log file
                          try(cat(sprintf("[%s] Config %d finished — AUC: %.3f, status: %s\n",
                                      format(Sys.time(), "%H:%M:%S"), config, 
                                      if(exists("auc_by_fold")) mean(auc_by_fold) else NA, result$status),
                              file = log_file, append = TRUE))
                          
                          
                          

                          
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
# 
# View results
print(results_df)


# Save results
write.csv(results_df, paste0(tuning_result_folder,"/hyperparameter_configurations_round1.csv"), 
          row.names = FALSE)

################################################################################

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




