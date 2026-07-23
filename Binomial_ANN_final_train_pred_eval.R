################################################################################
#                             Final Binomial ANN                               #
################################################################################
#package loading
library(torch)
library(cito)
library(terra)
library(dplyr)
library(tidyr)
library(pROC)
source("Functions.R")

################################################################################
#train the best model on all the training data
binomial_ann <- dnn(X=train_set_predictors, 
                    Y=cbind(train_set_response), 
                    loss = "binomial",
                    validation = 0.1,
                    early_stopping = 15,
                    epochs = 200,
                    activation = "prelu",
                    lr_scheduler = config_lr_scheduler("reduce_on_plateau", 
                                                       patience = 5, factor = 0.5),
                    lr = 0.6,
                    hidden = c(30,25),
                    batchsize = 32,
                    lambda = 0.0001,
                    alpha = 0.1)

saveRDS(binomial_ann, file = "results/final_trained_models/binomial_ann")

################################################################################
#reload the model
binomial_ann <- readRDS("results/final_trained_models/binomial_ann")


# get the averaged tuned threshold that maximizes TSS from all the cross-validation folds
# load the CV models object
cv_bin_model <- readRDS("results/candidate_models/binomial_ann/CV_models_hidden_30_25")
thr_bin <- get_cv_median_threshold(cv_bin_model)

threshold_matrix_bin <- sapply(cv_bin_model$performance, function(x) x["threshold", ])
write.csv(cbind(threshold_matrix_bin,thr_bin), "results/final_trained_models/threshold_binomial.csv")

################################################################################
# Test set evaluation
################################################################################
test_set_predictors <- read.csv("data/preprocessed_data/final_data_split/test_set/test_set_predictors.csv")
test_set_response <- read.csv("data/preprocessed_data/final_data_split/test_set/test_set_response.csv")[6:30]

#call the model evaluation function inside a printing device to save AUC curves
jpeg("results/plots/AUC_curves_binomial.jpeg", width = 800, height = 700)
evaluation_results_binomial <- evaluate_model_fixed_threshold(model = binomial_ann,
                                              test_pred = test_set_predictors,
                                              test_resp = test_set_response,
                                              fixed_threshold = thr_bin)
dev.off()

#save evaluation results
write.csv(evaluation_results_binomial, "results/final_trained_models/binomial_testset_evaluation.csv")

################################################################################
#performance uncertainty by evaluating each cross validation model on their 
#validation set using the fixed median thresholds
for (fold in 1:length(cv_indices)) {
  model <- cv_bin_model$model[[fold]]
  validation_idx <- cv_indices[[fold]]$val
  
  #get the median of thresholds of the 4 other folds excluding the threshold of the present fold to avoid data leakage
  thr <- apply(threshold_matrix_bin[,-fold], 1, median)
  evaluation_bin_fold <- evaluate_model_fixed_threshold(model = model,
                                                        test_pred = train_set_predictors[validation_idx,],
                                                        test_resp = train_set_response[validation_idx,],
                                                        fixed_threshold = thr,
                                                        default_predict_function = TRUE)
  if (fold==1){
    AUC_cv <- evaluation_bin_fold["AUC_species",]
    TSS_cv <- evaluation_bin_fold["TSS_species",]
  }else{
    AUC_cv <- cbind(AUC_cv, evaluation_bin_fold["AUC_species",])
    TSS_cv <- cbind(TSS_cv, evaluation_bin_fold["TSS_species",])}
}
AUC_cv <- cbind(AUC_cv, rowMeans(AUC_cv),apply(AUC_cv, 1, sd))
TSS_cv <- cbind(TSS_cv, rowMeans(TSS_cv),apply(TSS_cv, 1, sd))
colnames(AUC_cv) <- c(1:5, "mean", "sd")
colnames(TSS_cv) <- c(1:5, "mean", "sd")

write.csv(AUC_cv, 
          "results/final_trained_models/AUC_bin_performance_uncertainty_crossvalidation.csv")
write.csv(TSS_cv, 
          "results/final_trained_models/TSS_bin_performance_uncertainty_crossvalidation.csv")

################################################################################
#saving predictions on the whole data set using the final model for co-occurrence analysis
all_predictors <- rbind(train_set_predictors, test_set_predictors)
predictions_data_set <- predict(binomial_ann, newdata=all_predictors, type = "response")

write.csv(predictions_data_set, "results/community_analysis/predicted_community_binomial.csv")


#removing unnecessary variables for memory capacity
rm(evaluation_results_binomial, test_set_predictors, test_set_response, all_predictors, predictions_data_set)
gc()


#plotting and saving Accumulated local effect plots to determine predictor effects
library(patchwork)
library(ggplot2)
short_names <- sub("^([A-Za-z])[a-z]+_", "\\1. ", species_names)
short_names <- gsub("_", " ", short_names)

for (pred in 1:length(predictor_names)) {
  ale <- ALE(binomial_ann, variable = predictor_names[pred])
  
  plots <- lapply(seq_along(ale), function(i){
    ale[[i]] +
      ggtitle(short_names[i]) +
      theme(
        plot.title = element_text(
          size = 8,
          face = "italic",
          hjust = 0.5
        ),
        legend.position = "none",
        axis.title = element_blank()
      )
  })
  

  ggsave(paste0("results/plots/binomial/",predictor_names[pred],".tiff"),
         wrap_plots(plots, ncol = 5),
         width = 8, height = 10, dpi = 300, compression = "lzw")
}


################################################################################

#saving feature importance
feature_importance_ann <- summary(binomial_ann)
saveRDS(feature_importance_ann, 
        file = "results/final_trained_models/summary_binomial.rds")

rm(feature_importance_ann, model, cv_bin_model)
gc()

################################################################################
# Preparing predictions for the whole map
################################################################################

# Read all predictor rasters
predictor_files <- list.files("data/raw_data/final_raw_dataset/predictors_fish", 
                              pattern = "\\.tif$", 
                              full.names = TRUE)
# Stack them together as raster
predictor_stack <- rast(predictor_files)

# loading the scaling parameters
scaling <- read.csv("data/preprocessed_data/final_data_split/scaling_parameters.csv")
center_predictors <- as.numeric(scaling[1,])
sd_predictors <- as.numeric(scaling[2,])
names(center_predictors) <- predictor_names
names(sd_predictors) <- predictor_names

#scaling the predictors using the mean and standard deviation of the training data
map_predictors <- as.data.frame(predictor_stack, xy = TRUE) 
map_predictors_scaled <- map_predictors %>%
  mutate(across(all_of(predictor_names), ~ (.x - center_predictors[cur_column()])/sd_predictors[cur_column()]))

#cells / rows with all predictors values present
valid_rows <- rowSums(is.na(map_predictors_scaled)) == 0
sum(valid_rows)/length(valid_rows)
#summary(map_predictors_scaled)
#summary(map_predictors)

#saving the coordinates of all the valid grid cells with no missing values
xy_valid_cells <- map_predictors[valid_rows, c("x","y")]
dim(xy_valid_cells)
ANN_predictor_map <- map_predictors_scaled[valid_rows, predictor_names]

#removing unnecessary variables for memory capacity
rm(map_predictors_scaled, map_predictors, predictor_stack, valid_rows, cv_bin_model, full_response)
gc()

################################################################################
# Predict the whole map using probabilities and batches
prediction_map_binomial <- predict_map(binomial_ann, 
                               ANN_predictor_map, 
                               xy_valid_cells,
                               batch_size = 10000) 

#convert from list to SpatRaster
prediction_map_binomial <- rast(prediction_map_binomial)
names(prediction_map_binomial) <- species_names

terra::writeRaster(prediction_map_binomial, 
            "results/prediction_maps/prediction_map_binomial.tif", 
            overwrite = TRUE)


################################################################################
#binary prediction maps
prediction_map_binomial_binary <- predict_map(binomial_ann, 
                                       ANN_predictor_map, 
                                       xy_valid_cells,
                                       presence = T,
                                       threshold =  thr_bin,
                                       batch_size = 30000) 
prediction_map_binomial_binary <- rast(prediction_map_binomial_binary)
names(prediction_map_binomial_binary) <- species_names

terra::writeRaster(prediction_map_binomial_binary, 
                   "results/prediction_maps/prediction_map_binomial_binary.tif", 
                   overwrite = TRUE)

rm(prediction_map_binomial_binary)
gc()
