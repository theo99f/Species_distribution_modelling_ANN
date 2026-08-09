################################################################################
# Gillnet ONLY model!


#package loading
library(rsample)
library(terra)
library(dplyr)
library(tidyr)
library(pROC)
library(reshape2)
library(ggplot2)
source("Functions.R")

gillnet_response_data <- full_response %>%
  filter(method== "Gillnet")%>%
  select(all_of(species_names), longitude, latitude)

#project the longitude/latitude coordinates of training data (locations) to meter coordinates
coordinates_xy <- cbind(gillnet_response_data$longitude, gillnet_response_data$latitude)
coordinates_meters <- project(coordinates_xy, "EPSG:4326", "EPSG:3035")
colnames(coordinates_meters) <- c("longitude_meters", "latitude_meters")

#finding rows with complete data in the response variable
complete_rows <- complete.cases(gillnet_response_data)
sum(complete_rows)
#all rows complete


########################Predictor Variables#####################################

# Read all predictor Rasters
predictor_files <- list.files("data/raw_data/final_raw_dataset/predictors_fish", 
                              pattern = "\\.tif$", 
                              full.names = TRUE)

# Stack them together as raster
predictor_stack <- rast(predictor_files)
predictor_names <- names(predictor_stack)

#get the values of environmental predictors at GILLNET fishing locations AND their cell numbers
gillnet_predictor_values <- terra::extract(predictor_stack, y=coordinates_meters, cells=T)

sum(is.na(gillnet_predictor_values))
#no missing values in predictors either

################################################################################
#DATA SPLIT

gillnet_response_data <- gillnet_response_data %>%
  select(all_of(species_names))%>%
  mutate(cell = gillnet_predictor_values$cell)
#train/test split is performed 100 times grouped by grid cells and the split with 
#the most equal presences/absences is kept
#initial arbitrary high value for split metric
best_split_measure <- 1000000
for (variable in 1:100) {
  #splits train and test data by assigning all data points of the same grid cell to the same split
  split <- group_initial_split(gillnet_response_data, group = "cell", prop = 0.8)
  development_data <- training(split)%>%
    select(all_of(species_names))
  test_data <- testing(split)%>%
    select(all_of(species_names))
  
  presence_ratio <- colMeans(development_data)/colMeans(test_data)
  curr_split_measure <- sum((1-presence_ratio)^2)
  if (curr_split_measure<best_split_measure){
    best_split_measure <- curr_split_measure
    final_split <- split
  }
}

#Save the indeces of the train and testing data
index <- 1:nrow(gillnet_response_data)
train_set_index <- final_split$in_id
test_set_index <- index[!(index %in% train_set_index)]


#DEFINE the response variable for train and testing data
gillnet_train_response <- training(final_split)%>%
  select(-cell)
gillnet_test_response <- testing(final_split)%>%
  select(-cell)

#check presence balance in split
colMeans(gillnet_test_response)/colMeans(gillnet_train_response)


#Construct the predictor variables for train and testing data
#First scale the training data, then apply the same scale to the test data
gillnet_train_predictors <- scale(gillnet_predictor_values[train_set_index, predictor_names])

#getting the mean and standard deviations of the training set
center_predictors <- attr(gillnet_train_predictors, "scaled:center")
sd_predictors <- attr(gillnet_train_predictors, "scaled:scale")

#apply the same scale to the test set 
gillnet_test_predictors <- gillnet_predictor_values[test_set_index,predictor_names]%>%
  scale(center = center_predictors, scale = sd_predictors)


write.csv(gillnet_test_response, "data/preprocessed_data/gillnet_only_dataset/test_response.csv")
write.csv(gillnet_train_response, "data/preprocessed_data/gillnet_only_dataset/train_response.csv")
write.csv(gillnet_test_predictors, "data/preprocessed_data/gillnet_only_dataset/test_predictors.csv")
write.csv(gillnet_train_predictors, "data/preprocessed_data/gillnet_only_dataset/train_predictors.csv")


rm(gillnet_response_data)
gc()
################################################################################
#train the best model configuration tuned on the whole data set on all the GILLNET training data
gillnet_mvp_ann <- dnn(X=gillnet_train_predictors, 
               Y=cbind(gillnet_train_response), 
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

saveRDS(gillnet_mvp_ann, file = "results/final_trained_models/gillnet_mvp_ann")


################################################################################
#saving predictions on the whole data set and test set for co-occurence analysis

predictions_data_set <- predict(gillnet_mvp_ann, newdata = gillnet_predictor_values[,predictor_names], type = "response")
predictions_test_set <- predict(gillnet_mvp_ann, newdata = gillnet_test_predictors, type = "response")

write.csv(predictions_data_set, "results/community_analysis/predicted_community_gillnet.csv")
write.csv(predictions_test_set, "results/community_analysis/predicted_community_gillnet_testset.csv")

################################################################################
# Test set evaluation
################################################################################
gillnet_test_predictors <- read.csv("data/preprocessed_data/gillnet_only_dataset/test_predictors.csv", row.names = 1)
gillnet_test_response <- read.csv("data/preprocessed_data/gillnet_only_dataset/test_response.csv", row.names = 1)
gillnet_mvp_ann <- readRDS("results/final_trained_models/gillnet_mvp_ann")


#this function returns max TSS on test set. The TSS results are therefore not valid 
# and only the AUC score is unbiased
jpeg("results/plots/gillnet_mvp_model/AUC_curves_gillnet_model.jpeg", 
     width = 800, height = 700)
evaluation_results <- evaluate_model(model = gillnet_mvp_ann,
                                     test_pred = gillnet_test_predictors,
                                     test_resp = gillnet_test_response)
dev.off()

write.csv(evaluation_results, 
          "results/final_trained_models/gillnet_mvp_testset_evaluation.csv")

cv_mvp_model <- readRDS("results/candidate_models/mvp_ann/CV_models_hidden_27_27")
thr_mvp <- get_cv_median_threshold(cv_mvp_model)

evaluation_results <- evaluate_model_fixed_threshold(model = gillnet_mvp_ann,
                                                     test_pred = gillnet_test_predictors,
                                                     test_resp = gillnet_test_response,
                                                     fixed_threshold = thr_mvp,
                                                     default_predict_function = FALSE)
rowMeans(evaluation_results)
#removing unnecessary variables for memory capacity
rm(evaluation_results_mvp, 
   test_set_predictors, 
   test_set_response, 
   predictions_data_set, 
   predictions_test_set,
   gillnet_test_response,
   gillnet_train_response,
   gillnet_test_predictors,
   gillnet_train_predictors,
   train_set_predictors,
   train_set_response,
   development_data,
   full_response,
   split)
gc()


################################################################################
# Preparing predictions for the whole map
################################################################################

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
prediction_map_mvp <- predict_map(gillnet_mvp_ann, 
                                  ANN_predictor_map, 
                                  xy_valid_cells,
                                  batch_size = 40000) 
#convert to spatRast
prediction_map_mvp <- rast(prediction_map_mvp)
names(prediction_map_mvp) <- species_names

# Save result
writeRaster(prediction_map_mvp, 
            "results/prediction_maps/prediction_map_gillnet.tif", 
            overwrite = TRUE)

rm(prediction_map_mvp)
gc()
################################################################################
# Predictions for whole map with presence/absence
prediction_map_mvp_binary <- predict_map(gillnet_mvp_ann, 
                                         ANN_predictor_map, 
                                         xy_valid_cells,
                                         presence = T,
                                         threshold =  thr,
                                         batch_size = 40000) 
#convert to SpatRaster
prediction_map_mvp_binary <- rast(prediction_map_mvp_binary)
names(prediction_map_mvp_binary) <- species_names

terra::writeRaster(prediction_map_mvp_binary, 
                   "results/prediction_maps/prediction_map_gillnet_binary.tif", 
                   overwrite = TRUE)

################################################################################


