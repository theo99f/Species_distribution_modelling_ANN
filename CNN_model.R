################################################################################
#                                     CNN                                      #
#                                                                              #
#        Script for predictor preprocessing, training and model selection      #
################################################################################


#package loading
library(torch)
library(cito)
library(terra)
library(dplyr)
library(tidyr)
library(pROC)


#loading response with included cell numbers of preprocessed development set
directory <- "data/preprocessed_data/final_data_split/development_set/"
full_response <- read.csv(paste0(directory,"train_set_response.csv"))
train_set_response <- full_response %>%
  select(all_of(species_names))

################################################################################
# creating the training data set for the CNN with patches of predictor values  #
# using the same development split like the other models
################################################################################
# Read all predictor rasters
predictor_files <- list.files("data/raw_data/final_raw_dataset/predictors_fish", 
                              pattern = "\\.tif$", 
                              full.names = TRUE)

# Stack them together as raster
predictor_stack <- rast(predictor_files)
predictor_names <- names(predictor_stack)

#get the cell numbers of the training data
Cells_of_fish_obs_train <- full_response %>%
  select(cell)
RowCol_of_fish_obs_train <-rowColFromCell(predictor_stack,
                                         cell= Cells_of_fish_obs_train$cell)
colnames(RowCol_of_fish_obs_train) <- c("Row", "Column")

# loading the scaling parameters
scaling <- read.csv("data/preprocessed_data/final_data_split/scaling_parameters.csv")
center_predictors <- scaling[1,]
sd_predictors <- scaling[2,]

#
get_Patch <- function(Spatrast_object, Row_Col_of_cell, patch_size){
  
  row <- Row_Col_of_cell[1]
  col <- Row_Col_of_cell[2]
  
  margin <- (patch_size-1)/2
  rows <- (row-margin):(row+margin)
  cols <- (col-margin):(col+margin)
  #get patches cell numbers
  patch_cell_numbers <- cellFromRowColCombine(Spatrast_object, rows, cols)
  
  #extract values for patch
  patch_values <- terra::extract(Spatrast_object, patch_cell_numbers)
  
  a <- sapply(as.list(patch_values), function(x){matrix(nrow= patch_size, ncol = patch_size,byrow = TRUE)}, "array")
  matrix(patch_values, nrow= patch_size, ncol = patch_size,byrow = TRUE)
  return(patch_values)
}


#Setting of the patch size that is used for convolution at each location
patch_size <- 5

#create an empty 4 dimensional array to store the predictor patches of each observation
#1 dim:observations, 2 dim: predictor, 3 dim: row, 4 dim: column
predictor <- array(0, dim = c(nrow(RowCol_of_fish_locations), length(predictor_names), patch_size, patch_size))




dim(aperm(Patch_conv_map, c(3,1,2)))

get_Patch(predictor_stack, RowCol_of_fish_obs_train[1,], patch_size = patch_size)


#4 dimensional array 1 dim:observations, 2 dim: predictor, 3 dim: row, 4 dim: column
predictor <- array(0, dim = c(nrow(RowCol_of_fish_locations), length(predictor_names), patch_size, patch_size))
dimnames(predictor) <- c

pred_obs <- get_Patch(predictor_stack, RowCol_of_fish_locations[1,1], RowCol_of_fish_locations[1,2], patch_size = 7)



predictor[1,,,] <- aperm(pred_obs, c(3,1,2))

dim(predictor[1,])
#predictor: array with dimension 3, 4 or 5 for 1D-, 2D- or 3D-convolutions, respectively. 
#The first dimension are the samples, the second dimension the channels and the third - fifth dimension are the input dimensions

#get row and column number of predictor raster at fishing locations (response)
Cells_of_fish_locations <- cellFromXY(predictor_stack,coordinates_meters)
RowCol_of_fish_locations<-rowColFromCell(predictor_stack,
                                         cell= Cells_of_fish_locations)
#model to predict species patch-wise
architecture <- create_architecture(
  list(
    conv(),
    avgPool(),
    conv(),
    avgPool(),
    linear(n_neurons = n_species
    ),
    default_n_neurons = 10,
    default_n_kernels = 10,
    default_kernel_size = list(conv = 3, maxPool = 2, avgPool = 2),
    default_stride = list(conv = 1, maxPool = NULL, avgPool = NULL),
    default_padding = list(conv = 0, maxPool = 0, avgPool = 0),
    default_dilation = list(conv = 1, maxPool = 1),
    default_bias = list(conv = TRUE, linear = TRUE),
    default_activation = list(conv = "leaky_relu", linear = "relu"),
    default_normalization = list(conv = FALSE, linear = FALSE),
    default_dropout = list(conv = 0, linear = 0)
  ))

?cnn(
  X,
  Y = NULL,
  architecture,
  #use binomial for cnn or custom loss function (binary cross-entropy) or mvp with dnn
  loss = "binomial",
  optimizer = "adam",
  lr = 0.01,
  alpha = 0.5,
  lambda = 0,
  validation = 0.1,
  batchsize = 32L,
  burnin = 10,
  shuffle = TRUE,
  epochs = 100,
  early_stopping = NULL,
  lr_scheduler = NULL,
  custom_parameters = NULL,
  device = c("cpu", "cuda", "mps"),
  plot = TRUE,
  verbose = TRUE
)

