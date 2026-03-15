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

#removing unnecessary variables for memory capacity
rm(map_predictors_scaled, map_predictors, predictor_stack)
gc()


#load binomial model
binomial_ann <- readRDS("results/final_trained_models/binomial_ann")

# Option 2 Fixed: Grid layout for all species
for (pred in 1:length(predictor_names)) {
  png(paste0("results/plots/binomial/", predictor_names[pred], "_binomial.png"),
      width = 5000, height = 4000)
  
  par(mfrow = c(5, 5), mar = c(4, 4, 3, 1))  # Added margins
  
  for (sp in 1:length(species_names)) {
    # Create empty plot first
    plot.new()
    
    # Then add ALE plot
    ale_result <- ALE(binomial_ann, variable = predictor_names[pred], output = sp, 
                      plot = TRUE)
    
    # Add title
    mtext(species_names[sp], side = 3, line = 1, cex = 1.2, font = 2)
  }
  
  dev.off()
}
# Your original approach but MUCH bigger
for (pred in 1:length(predictor_names)) {
  png(paste0("results/plots/binomial/", predictor_names[pred], "_binomial.png"),
      width = 8000, height = 1200)  # Very wide, less tall
  
  ALE(binomial_ann, variable = predictor_names[pred])
  
  dev.off()
}
# Predict with batching to a
prediction_map_binomial <- predict_map(binomial_ann, 
                               ANN_predictor_map, 
                               xy_valid_cells,
                               batch_size = 10000) 

#convert from list to SpatRaster
prediction_map_binomial <- rast(prediction_map_binomial)


terra::writeRaster(prediction_map_binomial, 
            "results/prediction_maps/prediction_map_binomial.tif", 
            overwrite = TRUE)



################################################################################

prediction_map_binomial <- rast("results/prediction_maps/prediction_map_binomial.tif")
names(prediction_map_binomial) <- species_names

plot(prediction_map_binomial$Perca_fluviatilis)
################################################################################
# Test set evaluation
test_set_predictors <- read.csv("data/preprocessed_data/final_data_split/test_set/test_set_predictors.csv")
test_set_response <- read.csv("data/preprocessed_data/final_data_split/test_set/test_set_response.csv")[6:30]

jpeg("results/plots/AUC_curves_binomial.jpeg", width = 800, height = 700)
evaluation_results_binomial <- evaluate_model(model = binomial_ann,
                                         test_pred = test_set_predictors,
                                         test_resp = test_set_response)
dev.off()

write.csv(evaluation_results_binomial, "results/final_trained_models/binomial_testset_evaluation.csv")
################################################################################
#binary prediction maps

#extracting species specific thresholds
thr <- evaluation_results_binomial["threshold",]

prediction_map_binomial_binary <- predict_map(binomial_ann, 
                                       ANN_predictor_map, 
                                       xy_valid_cells,
                                       presence = T,
                                       threshold =  thr,
                                       batch_size = 10000) 
prediction_map_binomial_binary <- rast(prediction_map_binomial_binary)
names(prediction_map_binomial_binary) <- species_names

terra::writeRaster(prediction_map_binomial_binary, 
                   "results/prediction_maps/prediction_map_binomial_binary.tif", 
                   overwrite = TRUE)


plot(prediction_map_binomial$Clupea_harengus)

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