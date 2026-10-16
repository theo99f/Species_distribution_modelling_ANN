################################################################################
#                Script for preprocessing the full dataset                     #
#             Random grouped balanced sampling of the test set                 #
#
################################################################################
#  DO NOT RUN THE "SAVING FILES" SECTION, TO AVOID OVERWRITING THE USED SPLIT  #

source("Functions.R")
library(terra)
library(dplyr)

############################# RESPONSE VARIABLE ################################
# Load fish data - training labels/response
fish_data <- read.csv("data/raw_data/final_raw_dataset/fish_observations_wide.csv")

# Subsetting only the Gillnet data observations
gillnet_data <- fish_data[fish_data$method=="Gillnet",]

#project the longitude/latitude coordinates of training data (locations) to meter coordinates
coordinates_xy <- cbind(gillnet_data$longitude, gillnet_data$latitude)
coordinates_meters <- project(coordinates_xy, "EPSG:4326", "EPSG:3035")
colnames(coordinates_meters) <- c("longitude_meters", "latitude_meters")

#finding rows with complete data in the response variable
complete_rows <- complete.cases(gillnet_data)
sum(complete_rows)
#saving the label names
species_names <- colnames(gillnet_data)[6:ncol(fish_data)]

########################### Predictor Variables ################################

# Read all predictor rasters
predictor_files <- list.files("data/raw_data/final_raw_dataset/predictors_fish", 
                              pattern = "\\.tif$", 
                              full.names = TRUE)

# Stack them together as raster
predictor_stack <- rast(predictor_files)
predictor_names <- names(predictor_stack)
#plot(predictor_stack)

#get the values of environmental predictors at fishing locations AND their cell numbers
predictor_values <- terra::extract(predictor_stack, y=coordinates_meters, cells=T)


######################## Exclusion of missing values ###########################

#inspect whether there are missing values in predictors
for (predictor in 1:length(predictor_names)) {
  print(sum(is.na(predictor_values[,predictor])))
}

#Whole Train dataset with excluding observations with NA values for predictors and responses
predictors_fish <- predictor_values[!is.na(predictor_values$depth) & complete_rows, ]
response_fish <- gillnet_data[!is.na(predictor_values$depth) & complete_rows, ]
response_fish <- cbind(response_fish, cell=predictor_values$cell)

response_data <- response_fish %>%
  select(all_of(c(species_names, "cell")))

################################################################################
# Some basic dataset checks
#maximum repeated cell in the data set was repeated 279 times
max(table(response_fish$cell))

#unique cell numbers: 4128
length(unique(response_fish$cell))

# Prevalence of species
colMeans(response_data)*100

################################################################################
#        Dataset split into development (training) and test data
#        Randomly splitting the dataset grouped by grid cell numbers
################################################################################

#        INITIAL SPLIT

set.seed(987654321)
#train/test split is performed 100 times grouped by grid cells and the split with 
#the most equal presences/absences is kept
initial_split <- balanced_grouped_stratified_split(response_data, 
                                                   type = "initial", 
                                                   n=100,
                                                   prop = 0.8)


#Save the indeces of the train and testing data
index <- 1:nrow(response_data)
train_set_index <- initial_split$in_id
test_set_index <- index[!(index %in% train_set_index)]

length(train_set_index)
length(test_set_index)

#DEFINE the response variable for train and testing data
train_set_response <- training(initial_split)%>%
  select(-cell)
test_set_response <- testing(initial_split)%>%
  select(-cell)

#Construct the predictor variables for train and testing data
#First scale the training data, then apply the same scale to the test data
train_set_predictors <- scale(predictors_fish[train_set_index, predictor_names])

#getting the mean and standard deviations of the training set
center_predictors <- attr(train_set_predictors, "scaled:center")
sd_predictors <- attr(train_set_predictors, "scaled:scale")

#apply the same scale to the test set 
test_set_predictors <- predictors_fish[test_set_index, predictor_names]%>%
  scale(center = center_predictors, scale = sd_predictors)

#############################CHECK BALANCE######################################

colMeans(train_set_response)/colMeans(test_set_response)

sd(colMeans(train_set_response)/colMeans(test_set_response))

################################################################################
#                     CREATE CROSS-VALIDATION FOLDS                            #
################################################################################

#take the training data (roughly 80%) to split into 5 folds for cross-validation
#with approximate balanced presences
cv_folds <- balanced_grouped_stratified_split(response_set = training(initial_split),
                         type = "cv",
                         n=100,
                         k=5)

#create a list with row indeces for each fold 
fold_indices <- tidy(cv_folds)
cv_5_fold_indices <- list(
  list(train=fold_indices$Row[fold_indices$Fold=="Resample1"& fold_indices$Data=="Analysis"],
       val = fold_indices$Row[fold_indices$Fold=="Resample1"& fold_indices$Data=="Assessment"]),
  list(train=fold_indices$Row[fold_indices$Fold=="Resample2"& fold_indices$Data=="Analysis"],
       val = fold_indices$Row[fold_indices$Fold=="Resample2"& fold_indices$Data=="Assessment"]),
  list(train=fold_indices$Row[fold_indices$Fold=="Resample3"& fold_indices$Data=="Analysis"],
       val = fold_indices$Row[fold_indices$Fold=="Resample3"& fold_indices$Data=="Assessment"]),
  list(train=fold_indices$Row[fold_indices$Fold=="Resample4"& fold_indices$Data=="Analysis"],
       val = fold_indices$Row[fold_indices$Fold=="Resample4"& fold_indices$Data=="Assessment"]),
  list(train=fold_indices$Row[fold_indices$Fold=="Resample5"& fold_indices$Data=="Analysis"],
       val = fold_indices$Row[fold_indices$Fold=="Resample5"& fold_indices$Data=="Assessment"])
)

################################################################################
#                             SAVING FILES                                     #
################################################################################


#saving the training and testing data as csv
write.csv(test_set_predictors, file= paste0(folder,"/test_set_predictors.csv"), 
          row.names = F)
write.csv(test_set_response, file= paste0(folder,"/test_set_response.csv"), 
          row.names = F)
write.csv(train_set_predictors, file= paste0(folder,"/train_set_predictors.csv"), 
          row.names = F)
write.csv(train_set_response, file= paste0(folder,"/train_set_response.csv"), 
          row.names = F)

#save the scaling parameters
write.csv(rbind(center_predictors,sd_predictors), 
          file= paste0(folder,"/scaling_parameters.csv"), row.names = F)

#save train and test indices
write.csv(data.frame(index = train_set_index), 
          file = paste0(folder, "/train_indices.csv"), 
          row.names = FALSE)

write.csv(data.frame(index = test_set_index), 
          file = paste0(folder, "/test_indices.csv"), 
          row.names = FALSE)


#save the whole cross validation split object
saveRDS(cv_folds, paste0(cv_split_folder,"/cv_folds_5_grouped.rds")) 

#save cross-validation indices list
saveRDS(cv_5_fold_indices, paste0(cv_split_folder,"/grouped_5fold_cv_indeces.rds")) 

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
