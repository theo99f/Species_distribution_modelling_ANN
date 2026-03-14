################################################################################
#                               Final MVP_ANN                                  #
################################################################################
# 1. Training of the final model
# 2. Test set evaluation
# 3. predictions on the whole data set for later analysis
# 4. creating and saving of Accumulated local effect plots and feature importance
# 5. predictions for the whole map for binary and probability maps


# NOTE: external pointers to torch do not work after saving and reloading a cito
# dnn, therefore all steps that require 
# the citodnn predict function for the mvp model have to be done in the same 
# session as model training process

#package loading
library(torch)
library(cito)
library(terra)
library(dplyr)
library(tidyr)
library(pROC)
source("Functions.R")

train_set_response <- full_response[6:30]
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
# Test set evaluation
################################################################################
test_set_predictors <- read.csv("data/preprocessed_data/final_data_split/test_set/test_set_predictors.csv")
test_set_response <- read.csv("data/preprocessed_data/final_data_split/test_set/test_set_response.csv")[6:30]

jpeg("results/plots/mvp/AUC_curves_mvp.jpeg", width = 800, height = 700)
evaluation_results_mvp <- evaluate_model(model = mvp_ann,
                                              test_pred = test_set_predictors,
                                              test_resp = test_set_response)
dev.off()

write.csv(evaluation_results_mvp, 
          "results/final_trained_models/mvp_testset_evaluation.csv")


#extracting species-specific thresholds for later presence/absence predictions
thr <- evaluation_results_mvp["threshold",]
################################################################################
#saving predictions on the whole data set for cooccurence analysis
all_predictors <- rbind(train_set_predictors, test_set_predictors)
predictions_data_set <- predict(mvp_ann, newdata=all_predictors, type = "response")

write.csv(predictions_data_set, "results/community_analysis/predicted_community_mvp.csv")


#removing unnecessary variables for memory capacity
rm(evaluation_results_mvp, test_set_predictors, test_set_response, all_predictors, predictions_data_set)
gc()


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
# Predictions for whole map with presence/absence
prediction_map_mvp_binary <- predict_map(mvp_ann, 
                                              ANN_predictor_map, 
                                              xy_valid_cells,
                                              presence = T,
                                              threshold =  thr,
                                              batch_size = 30000) 
#convert to SpatRaster
prediction_map_mvp_binary <- rast(prediction_map_mvp_binary)
names(prediction_map_mvp_binary) <- species_names

terra::writeRaster(prediction_map_mvp_binary, 
                   "results/prediction_maps/prediction_map_mvp_binary.tif", 
                   overwrite = TRUE)

################################################################################

rm(prediction_map_mvp, prediction_map_mvp_binary)
gc()


