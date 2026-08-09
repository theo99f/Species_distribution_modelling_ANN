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

source("Functions.R")

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
#get the row and column numbers of the whole training data
RowCol_of_fish_obs_train <-rowColFromCell(predictor_stack,
                                         cell= Cells_of_fish_obs_train$cell)
colnames(RowCol_of_fish_obs_train) <- c("Row", "Column")

# loading the scaling parameters
scaling <- read.csv("data/preprocessed_data/final_data_split/scaling_parameters.csv")
center_predictors <- scaling[1,]
sd_predictors <- scaling[2,]

#function to extract the surrounding cells of a single training location for convolution patch
#
#Input a spatial raster object, a double (Row, Column) and an integer with the patch side length
#Outputs a 3 dim array with dimensions Row x Column x predictor (layer)
get_Patch <- function(Spatrast_object, Row_Col_of_cell, patch_size){
  
  row <- Row_Col_of_cell[1]
  col <- Row_Col_of_cell[2]
  
  margin <- (patch_size-1)/2
  rows <- (row-margin):(row+margin)
  cols <- (col-margin):(col+margin)
  #get patches cell numbers (ordered byrow)
  patch_cell_numbers <- cellFromRowColCombine(Spatrast_object, rows, cols)
  
  #extract values for patch
  patch_values <- terra::extract(Spatrast_object, patch_cell_numbers)
  
  #scale the values using mean and standard deviation from the training data cells
  patch_values_scaled <- lapply(names(patch_values), 
                                function(x){(patch_values[[x]]-center_predictors[[x]])/sd_predictors[[x]]})
  names(patch_values_scaled) <- predictor_names
  
  #reshape each layer to represent a 2-dim matrix with the cell in the center and simplify to a 3-dim array
  scaled_patch_array <- sapply(as.list(patch_values_scaled), 
              function(x){
                matrix(x,
                       nrow= patch_size, 
                       ncol = patch_size,
                       byrow = TRUE)},
              simplify = "array")
  return(scaled_patch_array)
}





################################################################################
# create a training data set for convolution with patches of 9x9 grid cells 
# around each fishing observation
# NO NEED TO RERUN can be loaded in next section
patch_size <- 9

#create an empty 4 dimensional array to store the predictor patches of each observation
#1 dim:observations, 2 dim: predictor, 3 dim: row, 4 dim: column
cnn_predictor <- array(0, dim = c(nrow(Cells_of_fish_obs_train), 
                              length(predictor_names), 
                              patch_size, 
                              patch_size))

cnn_predictor[1,,,] <- aperm(get_Patch(predictor_stack, 
                                       RowCol_of_fish_obs_train[1,], 
                                       patch_size = patch_size),
                             c(3,1,2))

for (obs in 1:dim(cnn_predictor)[1]) {
  cnn_predictor[obs,,,] <- aperm(get_Patch(predictor_stack, 
                                         RowCol_of_fish_obs_train[obs,], 
                                         patch_size = patch_size),
                               c(3,1,2))
}

sum(is.na(cnn_predictor))

#replacing missing values with arbitrarily high negative number
cnn_predictor <- ifelse(is.na(cnn_predictor), -9999, cnn_predictor)

saveRDS(cnn_predictor, file = "data/preprocessed_data/final_data_split/development_set/cnn_predictors9x9")
################################################################################
# create a training data set for convolution with patches of 5x5 grid cells 
# around each fishing observation
# NO NEED TO RERUN can be loaded in next section
patch_size <- 5

#create an empty 4 dimensional array to store the predictor patches of each observation
#1 dim:observations, 2 dim: predictor, 3 dim: row, 4 dim: column
cnn_predictor <- array(0, dim = c(nrow(Cells_of_fish_obs_train), 
                                  length(predictor_names), 
                                  patch_size, 
                                  patch_size))

cnn_predictor[1,,,] <- aperm(get_Patch(predictor_stack, 
                                       RowCol_of_fish_obs_train[1,], 
                                       patch_size = patch_size),
                             c(3,1,2))

for (obs in 1:dim(cnn_predictor)[1]) {
  cnn_predictor[obs,,,] <- aperm(get_Patch(predictor_stack, 
                                           RowCol_of_fish_obs_train[obs,], 
                                           patch_size = patch_size),
                                 c(3,1,2))
}

sum(is.na(cnn_predictor))

#replacing missing values with arbitrarily high negative number
cnn_predictor <- ifelse(is.na(cnn_predictor), -9999, cnn_predictor)

saveRDS(cnn_predictor, file = "data/preprocessed_data/final_data_split/development_set/cnn_predictors5x5")
################################################################################
#latest stable cito version 1.1 does not support cnn functionality yet, 
#thus the development version is set up for cnn
library(remotes)

# Create a temp library directory for installing cito development version
dir.create("~/temp_lib", recursive = TRUE)

# Install development cito in the separate library
remotes::install_github("citoverse/cito", lib = "~/temp_lib")

# Detach the old cito version first
detach("package:cito", unload = TRUE)
# Load it separately 
library(cito, lib.loc = "~/temp_lib")

exists("cnn", where = "package:cito")
exists("create_architecture", where = "package:cito")
exists("conv", where = "package:cito")

################################################################################

#reload cnn predictor dataset
cnn_predictor <- readRDS("data/preprocessed_data/final_data_split/development_set/cnn_predictors9x9")

cnn_predictor5x5 <- readRDS("data/preprocessed_data/final_data_split/development_set/cnn_predictors5x5")



architecture <- create_architecture(
                       #15 predictor channels
    conv(n_kernels = 10,
         kernel_size = 3,
         activation = "prelu"),    #

    linear(n_neurons = 30,
           activation = "prelu"),
  
    default_n_neurons = 10,
    default_n_kernels = 10,
    default_kernel_size = list(conv = 3, maxPool = 2, avgPool = 2),
    default_stride = list(conv = 1, maxPool = NULL, avgPool = NULL),
    default_padding = list(conv = 1, maxPool = 0, avgPool = 0),
    default_dilation = list(conv = 1, maxPool = 1),
    default_bias = list(conv = TRUE, linear = TRUE),
    default_normalization = list(conv = TRUE, linear = TRUE),
    default_dropout = list(conv = 0, linear = 0)
  )

train_pred <- cnn_predictor[cv_indices[[2]]$train,,,]
train_resp <- train_set_response[cv_indices[[2]]$train,]

cnn_5_test <- cnn(
  X=train_pred,
  Y = train_resp,
  architecture,
  #use binomial for cnn or custom loss function (binary cross-entropy) 
  loss = "mvp",
  optimizer = "SGD",
  lr = 0.001,
  alpha = 0.5,
  lambda = 0.0001,
  validation = 0.1,
  batchsize = 32L,
  shuffle = TRUE,
  epochs = 30,
  early_stopping = 15,
  lr_scheduler = config_lr_scheduler("reduce_on_plateau", 
                                     patience = 5, factor = 0.5),
  custom_parameters = NULL,
  plot = TRUE,
  verbose = TRUE
)


predict(cnn_9_test, newdata=cnn_predictor[cv_indices[[2]]$val,,,], type="response")
a <- evaluate_model(cnn_9_test, cnn_predictor[cv_indices[[2]]$val,,,],train_set_response[cv_indices[[2]]$val,])
rowMeans(a)


################################################################################
#Random search hyperparameter tuning with 5-fold cross-validation in parallel loop
library(parallel)
library(foreach)
library(doParallel)

# Setup parallel backend
n_cores <- 3
cl <- makeCluster(n_cores)
registerDoParallel(cl)

# Load necessary objects/functions to each worker
clusterExport(cl, c("cross_validate_ann", "train_set_predictors", 
                    "train_set_response", "cv_indices", "species_names",
                    "predictor_names", "evaluate_model"))

# Load required packages on each worker
clusterEvalQ(cl, {
  library(cito)
  library(dplyr)
  library(pROC)
})

# Parallel loop 
results_list <- foreach(config = 1:20, 
                        .errorhandling = "pass") %dopar% {
                          
                          # Generate config
                          configs <- list(
                            loss = "binomial",
                            hidden = c(sample(10:40, sample(1:2, 1))),
                            lambda = runif(1, 0.0001, 0.0006), #very low range 
                            alpha = 0.1, #not significant thus just best config
                            lr = 0.6,    #not significant thus just best config
                            batchsize = 32
                          )
                          
                          # Cross-validate
                          model <- cross_validate_ann(configs, train_set_predictors, 
                                                      train_set_response, cv_indices)
                          
                          # Extract metrics
                          auc_by_fold <- sapply(model$performance, function(x) mean(x["AUC_species", ]))
                          tss_by_fold <- sapply(model$performance, function(x) mean(x["TSS_species", ]))
                          
                          # Calculate means across folds
                          mean_AUC <- mean(auc_by_fold)
                          mean_TSS <- mean(tss_by_fold)
                          
                          # Format output - ensure all values are scalar
                          result <- list(
                            loss = configs$loss,
                            hidden = paste0(configs$hidden, collapse = ","),
                            lambda = configs$lambda,
                            alpha = configs$alpha,
                            lr = configs$lr,
                            batchsize = configs$batchsize,
                            mean_AUC = mean_AUC,
                            mean_TSS = mean_TSS
                          )
                          return(result)
                        }

# Stop cluster
stopCluster(cl)

# Combine results into data frame AFTER parallel loop
results_df <- do.call(rbind, lapply(results_list, function(x) {
  data.frame(
    loss = x$loss,
    hidden = x$hidden,
    lambda = x$lambda,
    alpha = x$alpha,
    lr = x$lr,
    batchsize = x$batchsize,
    mean_AUC = x$mean_AUC,
    mean_TSS = x$mean_TSS,
    stringsAsFactors = FALSE
  )
}))

# View results
print(results_df)

# Save results
write.csv(results_df, "results/cnn_hyperparameter_optimization_20_configs.csv", 
          row.names = FALSE)

cnn_9_test$model_properties
print(architecture)