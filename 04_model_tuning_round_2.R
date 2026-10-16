################################################################################
#                      Hyperparameter tuning random search                     #
#                                                                              #
#                                     Round 2                                  #
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
log_file <- paste0(tuning_result_folder,"/tuning_2_progress.log")
if (!dir.exists(tuning_result_folder)) {
  dir.create(tuning_result_folder, recursive = TRUE)
}

#load the configuration with the best predicted AUC of the surrogate linear model
top_predicted_configs <- readRDS(file = "results/tuning/top_predicted_configs.rds")

#save the ranges of the hyperparameters in the predicted high performing region
range(top_predicted_configs$predicted_auc)
alpha_range <- range(top_predicted_configs$alpha)
lambda_range <- range(top_predicted_configs$lambda)
lr_range <- range(top_predicted_configs$lr)

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
                            hidden = c(sample(15:45, 
                                              sample(2:4, 1),
                                              replace = T)),
                            #uniform distribution of the 
                            alpha = runif(1, 
                                          min = min(alpha_range),
                                          max = max(alpha_range)),
                            #log uniform distribution between 0.001 and 0.1
                            lr = 10^-(runif(1, 
                                            min= min(-log10(lr_range)), 
                                            max= max(-log10(lr_range)))),
                            #log uniform distribution between 0.001 and 0.01
                            lambda = 10^-(runif(1, 
                                                min = min(-log10(lambda_range)),
                                                max = max(-log10(lambda_range)))),
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
write.csv(results_df, paste0(tuning_result_folder,"/hyperparameter_configurations_round2.csv"), 
          row.names = FALSE)

################################################################################
