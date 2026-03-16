################################################################################
#                                Functions                                     #
#                 (and variables needed across scripts)                        #
################################################################################
library(dplyr)
library(torch)

#loading preprocessed development set
directory <- "data/preprocessed_data/final_data_split/development_set/"
train_set_predictors <- read.csv(paste0(directory,"train_set_predictors.csv"))
full_response <- read.csv(paste0(directory,"train_set_response.csv"))

#saving the predictor and species names to make them available across scripts
predictor_names <- colnames(train_set_predictors)
species_names <- colnames(full_response)[6:30]

train_set_response <- full_response %>%
  dplyr::select(all_of(species_names))


#loading the indices of the grouped cross-validation split
cv_split_folder <- "data/preprocessed_data/final_data_split/development_set/5-fold_grouped_cv_split/"
cv_indices <- readRDS(paste0(cv_split_folder,"grouped_5fold_cv_indeces.rds")) 


#function for a binary cito multi-species model evaluation
evaluate_model <- function(model, test_pred, test_resp, model_name = "DEFAULT"){
  #predictions for the test set predictors
  predictions <- predict(model, newdata = test_pred, type="response")
  col_sp <- rainbow(length(species_names))
  if (all(dim(predictions) == dim(test_resp))){
    
    #calculate AUC and TSS per species and plot ROC
    AUC_species <- numeric(length(species_names))
    TSS_species <- numeric(length(species_names))
    threshold <- numeric(length(species_names))
    names(AUC_species) <- species_names
    for (species in 1:length(species_names)) {
      
      ROC_species <- roc(test_resp[,species], predictions[,species])
      AUC_species[species] <- ROC_species$auc
      
      roc_values <- coords(ROC_species)
      TSS_thr <- roc_values$sensitivity+roc_values$specificity-1
      TSS_species[species] <- max(TSS_thr)
      threshold[species] <- roc_values$threshold[TSS_thr==TSS_species[species]]
      
      if (species==1){
        plot(ROC_species, col =col_sp[species])
      }
      else {plot(ROC_species, col =col_sp[species], add= T)}
    }
    legend("bottomright", legend = species_names, 
           col = col_sp,
           lty = 1,
           lwd = 2,
           bty ="n")
    return(rbind(AUC_species,TSS_species,threshold))
    
  }else{
    print("Prediction dimensions do not match the test set response dimensions")
    return(NULL)
  }
}

# Function to cross-validate a cito model including training and evaluation, 
# given a list of model configurations, a separate predictor and response set of
# the same length, and a list with cross-validation indices 

cross_validate_ann <- function(model_configs, predictor_set, response_set, cv_fold_indices){
  model_configs <- model_configs
  k_fold_models <- list(model = list(),
                        performance = list(),
                        configuration = model_configs)
  #loop over the folds
  for (fold in 1:length(cv_indices)) {
    #construct train and validation sets for each fold from index list
    train_pred <- predictor_set[cv_indices[[fold]]$train,]
    train_resp <- response_set[cv_indices[[fold]]$train,]
    val_pred <- predictor_set[cv_indices[[fold]]$val,]
    val_resp <- response_set[cv_indices[[fold]]$val,]
    
    #train the model
    model <- dnn(X=train_pred, 
                 Y=cbind(train_resp),
                 validation = 0.1,
                 early_stopping = 15,
                 epochs = 200,
                 activation = "prelu",
                 lr_scheduler = config_lr_scheduler("reduce_on_plateau", 
                                                    patience = 5, factor = 0.5),
                 #model specific configurations
                 loss = model_configs$loss,
                 lr = model_configs$lr,
                 hidden = model_configs$hidden,
                 batchsize = model_configs$batchsize,
                 lambda = model_configs$lambda,
                 alpha = model_configs$alpha)
    analyze_training(model)
    #check if training was successful
    if (model$successfull){
      #save the folds model
      k_fold_models$model[[fold]] <- model
      print(paste("Successfully trained model for fold",fold,"\n"))
      
      #evaluate the model
      performance <- evaluate_model(model = model, 
                                    test_pred = val_pred, 
                                    test_resp = val_resp)
      k_fold_models$performance[[fold]] <- performance
      print(paste("Models mean AUC", mean(performance[1,]),"and mean TSS",mean(performance[2,]),"\n"))
    }else{
      print(paste("Training was aborted for fold",fold,"\n"))
    }
  }
  return(k_fold_models)
}

######################################################################################################################################
#predict_map function with batch processing
predict_map <- function(model, map_predictor_values, 
                        xy_valid_cells, 
                        presence = FALSE, 
                        threshold = rep(0.5,length(species_names)), #vector of species-specific thresholds
                        batch_size = 10000){
  
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
  
  # Combine all predictions from the list into one matrix
  cat("All predictions done! Combining batched predictions...\n")
  predicted_map_probabilities <- do.call(rbind, predictions_list)
  
  cat("Predictions combined!")
  # Project the predictions as probabilities or presence/absence
  if (presence){
    #checks for every prediction if the respective species is bigger than the species-specific 
    #presence vs. absence threshold
    predicted_map_probabilities <- 1*(sweep(predicted_map_probabilities, 2, threshold,">"))}
  
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
  
  return(prediction_map)
}


predict_mvp_manual <- function(X, model, species_names) {
  X <- as.matrix(X)
  
  # Get weights from saved model
  W1 <- model$weights[[1]]$`0.weight`
  b1 <- model$weights[[1]]$`0.bias`
  alpha1 <- model$weights[[1]]$`1.weight`
  
  W2 <- model$weights[[1]]$`2.weight`
  b2 <- model$weights[[1]]$`2.bias`
  alpha2 <- model$weights[[1]]$`3.weight`
  
  W3 <- model$weights[[1]]$`4.weight`
  b3 <- model$weights[[1]]$`4.bias`
  
  # Forward pass
  z1 <- X %*% t(W1) + matrix(b1, nrow(X), length(b1), byrow = TRUE)
  h1 <- ifelse(z1 > 0, z1, alpha1 * z1)
  
  z2 <- h1 %*% t(W2) + matrix(b2, nrow(h1), length(b2), byrow = TRUE)
  h2 <- ifelse(z2 > 0, z2, alpha2 * z2)
  
  logits <- h2 %*% t(W3) + matrix(b3, nrow(h2), length(b3), byrow = TRUE)
  probs <- 1 / (1 + exp(-logits))
  
  colnames(probs) <- species_names
  return(probs)
}

