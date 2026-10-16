################################################################################
#                            Final Tuning Round                                #
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
log_file <- paste0(tuning_result_folder,"/tuning_3_progress.log")
if (!dir.exists(tuning_result_folder)) {
  dir.create(tuning_result_folder, recursive = TRUE)
}

#loading the table with prepared configurations for final round (3 seeds each)
run_table <- read.csv("results/tuning/round3_run_table.csv")


# Random search hyperparameter tuning with 5-fold cross-validation in parallel loop
# Setup parallel backend
n_cores <- detectCores()-1
cl <- makeCluster(n_cores)
registerDoParallel(cl)

# Load necessary objects/functions to each worker
clusterExport(cl, c("cross_validate_ann", "train_set_predictors", 
                    "train_set_response", "cv_indices", "species_names",
                    "predictor_names", "evaluate_model", "run_table"))

# Load required packages on each worker
clusterEvalQ(cl, {
  library(cito)
  library(dplyr)
  library(pROC)
})

# Parallel loop for random search hyperparameter tuning
results_list <- foreach(config = 1:nrow(run_table), 
                        .errorhandling = "pass") %dopar% {
                          
                          #extracting the configs layer widths
                          l1 <- run_table$layer1[config]
                          l2 <- run_table$layer2[config]
                          l3 <- run_table$layer3[config]
                          l4 <- run_table$layer4[config]
                          hidden <- c(l1,l2,l3,l4)
                          
                          #taking a configuration from the prepared configs
                          configs <- list(
                            loss = "mvp",
                            epochs = 200,
                            hidden = hidden[!is.na(hidden)],
                            alpha = run_table$alpha[config],
                            lr = run_table$lr[config],
                            lambda = run_table$lambda[config],
                            batchsize = run_table$batchsize[config],
                            runID = run_table$run_id[config]
                          )
                          #setting the seed for reproducable network weight initilization
                          set.seed(run_table$seed[config])
                          torch::torch_manual_seed(run_table$seed[config])
                          
                          
                          result <- tryCatch({
                            model <- cross_validate_ann(configs, train_set_predictors, 
                                                        train_set_response, cv_indices)
                            
                            auc_by_fold <- sapply(model$performance, function(x) mean(x["AUC_species", ]))
                            tss_by_fold <- sapply(model$performance, function(x) mean(x["TSS_species", ]))
                            
                            list(
                              runID = configs$runID,
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
                              runID = configs$runID,
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
                          #save intermediate result
                          try(saveRDS(result, file.path(tuning_result_folder,
                                                        paste0("round3_run_", run_table$run_id[config], ".rds"))), silent = TRUE)
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
    runID = x$runID,
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
write.csv(results_df, paste0(tuning_result_folder,"/hyperparameter_configurations_round3.csv"), 
          row.names = FALSE)

################################################################################
