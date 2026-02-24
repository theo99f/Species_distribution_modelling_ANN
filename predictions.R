################################################################################
#                                 Predictions                                  #
################################################################################
#package loading
library(torch)
library(cito)
library(terra)
library(dplyr)
library(tidyr)
library(pROC)
source("Functions.R")

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

#removing unnecessary variables for memory
rm(map_predictors_scaled, map_predictors, predictor_stack)
gc()
######################################################################################################################################
#predict_map function with batch processing
predict_map <- function(model, map_predictor_values, xy_valid_cells, 
                        presence = FALSE, threshold = 0.5, batch_size = 10000){
  
  if (!nrow(map_predictor_values) == nrow(xy_valid_cells)){
    print("Number of valid location grid cells does not equal predictions")
    return(NULL)
  }
  
  n_rows <- nrow(map_predictor_values)
  n_batches <- ceiling(n_rows / batch_size)
  
  cat("Predicting in", n_batches, "batches of max", batch_size, "cells\n")
  
  # Predict in batches
  predictions_list <- list()
  
  for(i in 1:n_batches) {
    start_idx <- (i - 1) * batch_size + 1
    end_idx <- min(i * batch_size, n_rows)
    
    cat("Processing batch", i, "of", n_batches, 
        "(rows", start_idx, "to", end_idx, ")\n")
    
    # predictions for batch i
    predictions_list[[i]] <- predict(model, 
                                     newdata = map_predictor_values[start_idx:end_idx, ], 
                                     type = "response")
    
    gc()  # Clean up memory between batches
  }
  
  # Combine all predictions
  cat("All predictions done! Combining batched predictions...\n")
  predicted_map_probabilities <- do.call(rbind, predictions_list)
  
  cat("Predictions combined!")
  # Project the predictions as probabilities or presence/absence
  if (presence){
    predicted_map_probabilities <- ifelse(predicted_map_probabilities > threshold, 1, 0)}
  
  #creating spatial species raster maps for one species at the time
  prediction_map <- list()  
  for (species in 1:length(species_names)) {
    species_name <- species_names[species]
    prediction_map[[species_name]] <- rast(cbind(xy_valid_cells, predicted_map_probabilities[,species]),
                           crs = "EPSG:3035",
                           type = "xyz")
    names(prediction_map[[species_name]]) <- species_name
    cat("Raster map for ",species_name, " created.\n")
    gc()
  }

  
  #names(prediction_map) <- species_names
  return(prediction_map)
}

# Predictions using the median model from the best binomial model configuration
model_cv_30_25 <- readRDS("results/candidate_models/binomial_ann/CV_models_hidden_30_25")
median_best_config_binomial <- model_cv_30_25$model[[5]]

# Predict with batching
prediction_map1 <- predict_map(median_best_config_binomial, 
                               ANN_predictor_map, 
                               xy_valid_cells,
                               batch_size = 10000) 

plot(prediction_map1$Clupea_harengus)

#
binomial_ann <- readRDS("results/candidate_models/binomial_ann")



# Save result
writeRaster(prediction_map1, "results/prediction_maps/multispecies_probabilities.tif", overwrite = TRUE)

