################################################################################
#                Script for preprocessing the full dataset                     #
#             Random grouped balanced sampling of the test set                 #
#

#  DO NOT RUN THE "SAVING FILES" SECTION, TO AVOID OVERWRITING THE USED SPLIT  #

library(rsample)
library(terra)

########################RESPONSE VARIABLE#######################################
# Load fish data - training labels/response
fish_data <- read.csv("data/raw_data/final_raw_dataset/fish_observations_wide.csv")

#project the longitude/latitude coordinates of training data (locations) to meter coordinates
coordinates_xy <- cbind(fish_data$longitude, fish_data$latitude)
coordinates_meters <- project(coordinates_xy, "EPSG:4326", "EPSG:3035")
colnames(coordinates_meters) <- c("longitude_meters", "latitude_meters")

#finding rows with complete data in the response variable
complete_rows <- complete.cases(fish_data)

#saving the label names
species_names <- colnames(fish_data)[6:ncol(fish_data)]
########################Predictor Variables#####################################

# Read all predictor rasters
predictor_files <- list.files("data/raw_data/final_raw_dataset/predictors_fish", 
                              pattern = "\\.tif$", 
                              full.names = TRUE)

# Stack them together as raster
predictor_stack <- rast(predictor_files)
predictor_names <- names(predictor_stack)
plot(predictor_stack)

#get the values of environmental predictors at fishing locations AND their cell numbers
predictor_values <- terra::extract(predictor_stack, y=coordinates_meters, cells=T)


########################Exclusion of missing values#############################

#inspect whether there are missing values in predictors
for (predictor in 1:length(predictor_names)) {
  print(sum(is.na(predictor_values[,predictor])))
}
#Whole Train dataset with excluding observations with NA values for predictors and responses
predictors_fish <- predictor_values[!is.na(predictor_values$depth) & complete_rows, ]
response_fish <- fish_data[!is.na(predictor_values$depth) & complete_rows, ]
response_fish <- cbind(response_fish, cell=predictor_values$cell)


################################################################################
# Randomly splitting the dataset grouped by grid cell numbers and choosing a split
# with balanced presences between train and test set


#maximum repeated cell in the data set was repeated 279 times
max(table(response_fish$cell))

#train/test split is performed 100 times grouped by grid cells and the split with 
#the most equal presences/absences is kept
#initial arbitrary high value for split metric
best_split_measure <- 1000000
for (variable in 1:100) {
  #splits train and test data by assigning all data points of the same grid cell to the same split
  split <- group_initial_split(response_fish, group = "cell", prop = 0.8)
  development_data <- training(split)%>%
    select(all_of(species_names))
  test_data <- testing(split)%>%
    select(all_of(species_names))
  
  presence_ratio <- colMeans(development_data)/colMeans(test_data)
  curr_split_measure <- sum((1-presence_ratio)^2)
    final_split <-split}



#Save the indeces of the train and testing data
index <- 1:nrow(response_fish)
train_set_index <- final_split$in_id
test_set_index <- index[!(index %in% train_set_index)]

length(train_set_index)
length(test_set_index)

#DEFINE the response variable for train and testing data
train_set_response <- training(final_split)
test_set_response <- testing(final_split)

#Construct the predictor variables for train and testing data
#First scale the training data, then apply the same scale to the test data
train_set_predictors <- scale(predictors_fish[train_set_index, predictor_names])

#getting the mean and standard deviations of the training set
center_predictors <- attr(train_set_predictors, "scaled:center")
sd_predictors <- attr(train_set_predictors, "scaled:scale")

#apply the same scale to the test set 
test_set_predictors <- predictors_fish[test_set_index,predictor_names]%>%
  scale(center = center_predictors, scale = sd_predictors)

#############################CHECK BALANCE######################################

train_response <- train_set_response%>%
  select(all_of(species_names))


test_response <- test_set_response%>%
  select(all_of(species_names))

colMeans(train_response)
colMeans(test_response)



################################################################################
#                             SAVING FILES                                     #
################################################################################

#saving the training and testing data as csv
folder <- "data/preprocessed_data/final_data_split/"
write.csv(test_set_predictors, file= paste0(folder,"test_set/test_set_predictors.csv"), 
          row.names = F)
write.csv(test_set_response, file= paste0(folder,"test_set/test_set_response.csv"), 
          row.names = F)
write.csv(train_set_predictors, file= paste0(folder,"development_set/train_set_predictors.csv"), 
          row.names = F)
write.csv(train_set_response, file= paste0(folder,"development_set/train_set_response.csv"), 
          row.names = F)

#save the scaling parameters
write.csv(rbind(center_predictors,sd_predictors), 
          file= paste0(folder,"scaling_parameters.csv"), row.names = F)

#save train and test indices
write.csv(data.frame(index = train_set_index), 
          file = paste0(folder, "development_set/train_indices.csv"), 
          row.names = FALSE)

write.csv(data.frame(index = test_set_index), 
          file = paste0(folder, "test_set/test_indices.csv"), 
          row.names = FALSE)

##################################################################################
# Verify that row order is preserved
cat("\nVerifying data alignment...\n")

# Check that cell numbers match between predictors and response
train_cells_from_response <- train_set_response$cell
train_cells_from_predictors <- predictors_fish$cell[train_set_index]

if (all(train_cells_from_response == train_cells_from_predictors)) {
  cat("Training set: Predictors and response are correctly aligned\n")
} else {
  stop("ERROR: Training set alignment mismatch!")
}

test_cells_from_response <- test_set_response$cell
test_cells_from_predictors <- predictors_fish$cell[test_set_index]

if (all(test_cells_from_response == test_cells_from_predictors)) {
  cat("Test set: Predictors and response are correctly aligned\n")
} else {
  stop("ERROR: Test set alignment mismatch!")
}
