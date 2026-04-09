################################################################################
#                               Final MVP_ANN                                  #
################################################################################
# 1. Training of the final model
# 2. predictions on the whole data set for later analysis
# 3. creating and saving of Accumulated local effect plots and feature importance
# 4. predictions for the whole map for the probability maps
# 5. Model evaluation
# 6. predictions for binary maps

# NOTE: external pointers to torch do not work after saving and reloading a cito
# dnn, therefore all steps that require the citodnn predict function for the mvp 
# model have to be done in the same session as model training process

# Due to Bug found and fixed in test evaluation after the Rsession was closed, 
# the model with intact external pointers was not available anymore for citodnns
# internal predict function. 
# Model evaluation and binary maps are therefore done with a manual predict function,
# at the end of the script. The manual predict function extracts the networks 
# weights that has been tested to yield the same predictions 
# as citos own function (see "Functions.R")

#package loading
library(torch)
library(cito)
library(terra)
library(dplyr)
library(tidyr)
library(pROC)
source("Functions.R")


################################################################################
#train the best model configuration on all the training data
mvp_ann <- dnn(X=train_set_predictors, 
               Y=cbind(train_set_response), 
               loss = "mvp",
               validation = 0.1,
               early_stopping = 15,
               epochs = 200,
               activation = "prelu",
               lr_scheduler = config_lr_scheduler("reduce_on_plateau", 
                                                  patience = 5, factor = 0.5),
               lr = 0.03,
               hidden = c(27,27),
               batchsize = 32,
               lambda = 0.0001,
               alpha = 0.15)


saveRDS(mvp_ann, file = "results/final_trained_models/mvp_ann")


################################################################################
#saving predictions on the whole data set for cooccurence analysis
all_predictors <- rbind(train_set_predictors, test_set_predictors)
predictions_data_set <- predict(mvp_ann, newdata=all_predictors, type = "response")

write.csv(predictions_data_set, "results/community_analysis/predicted_community_mvp.csv")


#removing unnecessary variables for memory capacity
rm(evaluation_results_mvp, test_set_predictors, test_set_response, all_predictors, predictions_data_set)
gc()

################################################################################
#plotting and saving Accumulated local effect plots to determine predictor effects
for (pred in 1:length(predictor_names)) {
  png(paste0("results/plots/mvp/", predictor_names[pred], "_mvp.png"),
      width = 8000, height = 1200)  # Very wide, less tall
  
  ALE(mvp_ann, variable = predictor_names[pred])
  
  dev.off()
}

#saving feature importance
feature_importance_ann <- summary(mvp_ann)
saveRDS(feature_importance_ann, file = "results/final_trained_models/summary_mvp.rds")

rm(feature_importance_ann)
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

#saving the coordinates of all the valid grid cells with no missing values
xy_valid_cells <- map_predictors[valid_rows, c("x","y")]
dim(xy_valid_cells)
ANN_predictor_map <- map_predictors_scaled[valid_rows, predictor_names]

#removing unnecessary variables for memory capacity
rm(map_predictors_scaled, map_predictors, predictor_stack, valid_rows, scaling)
gc()

################################################################################
# Predictions for whole map with probabilities
prediction_map_mvp <- predict_map(mvp_ann, 
                                  ANN_predictor_map, 
                                  xy_valid_cells,
                                  batch_size = 30000) 
#convert to spatRast
prediction_map_mvp <- rast(prediction_map_mvp)
names(prediction_map_mvp) <- species_names

# Save result
writeRaster(prediction_map_mvp, 
            "results/prediction_maps/prediction_map_mvp.tif", 
            overwrite = TRUE)

################################################################################


################################################################################
# Test set evaluation
################################################################################
test_set_predictors <- read.csv("data/preprocessed_data/final_data_split/test_set/test_set_predictors.csv")
test_set_response <- read.csv("data/preprocessed_data/final_data_split/test_set/test_set_response.csv")[6:30]

# Reload the model
mvp_ann <- readRDS("results/final_trained_models/mvp_ann")


# get the averaged tuned threshold that maximizes TSS from all the cross-validation folds
# load the CV model object
cv_mvp_model <- readRDS("results/candidate_models/mvp_ann/CV_models_hidden_27_27")
thr_mvp <- get_cv_median_threshold(cv_mvp_model)

threshold_matrix_mvp <- sapply(cv_mvp_model$performance, function(x) x["threshold", ])

jpeg("results/plots/mvp/AUC_curves_mvp.jpeg", width = 800, height = 700)
evaluation_results_mvp <- evaluate_model_fixed_threshold(model = mvp_ann,
                                          test_pred = test_set_predictors,
                                          test_resp = test_set_response,
                                          fixed_threshold = thr_mvp,
                                          default_predict_function = FALSE)
dev.off()

write.csv(evaluation_results_mvp, 
          "results/final_trained_models/mvp_testset_evaluation.csv")

#performance uncertainty by evaluating each cross validation model on their 
#validation set using the fixed median thresholds
for (fold in 1:length(cv_indices)) {
  model <- cv_mvp_model$model[[fold]]
  validation_idx <- cv_indices[[fold]]$val
  
  #get the median of thresholds of the 4 other folds excluding the threshold of the present fold to avoid data leakage
  thr <- apply(threshold_matrix_mvp[,-fold], 1, median)
  evaluation_mvp_fold <- evaluate_model_fixed_threshold(model = model,
                                                           test_pred = train_set_predictors[validation_idx,],
                                                           test_resp = train_set_response[validation_idx,],
                                                           fixed_threshold = thr,
                                                           default_predict_function = FALSE)
  if (fold==1){
    AUC_cv <- evaluation_mvp_fold["AUC_species",]
    TSS_cv <- evaluation_mvp_fold["TSS_species",]
  }else{
    AUC_cv <- cbind(AUC_cv, evaluation_mvp_fold["AUC_species",])
    TSS_cv <- cbind(TSS_cv, evaluation_mvp_fold["TSS_species",])}
}
AUC_cv <- cbind(AUC_cv, rowMeans(AUC_cv),apply(AUC_cv, 1, sd))
TSS_cv <- cbind(TSS_cv, rowMeans(TSS_cv),apply(TSS_cv, 1, sd))
colnames(AUC_cv) <- c(1:5, "mean", "sd")
colnames(TSS_cv) <- c(1:5, "mean", "sd")

write.csv(AUC_cv, 
          "results/final_trained_models/AUC_mvp_performance_uncertainty_crossvalidation.csv")
write.csv(TSS_cv, 
          "results/final_trained_models/TSS_mvp_performance_uncertainty_crossvalidation.csv")

rm(evaluation_results_mvp)
gc()
################################################################################
# Predictions for whole map with presence/absence
prediction_map_mvp_binary <- predict_map(mvp_ann, 
                                              ANN_predictor_map, 
                                              xy_valid_cells,
                                              presence = T,
                                              threshold =  thr_mvp,
                                              batch_size = 30000,
                                         default_predict_function = FALSE) 
#convert to SpatRaster
prediction_map_mvp_binary <- rast(prediction_map_mvp_binary)
names(prediction_map_mvp_binary) <- species_names

terra::writeRaster(prediction_map_mvp_binary, 
                   "results/prediction_maps/prediction_map_mvp_binary.tif", 
                   overwrite = TRUE)

################################################################################

rm(prediction_map_mvp, prediction_map_mvp_binary)
gc()
################################################################################
