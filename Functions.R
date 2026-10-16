################################################################################
#                                Functions                                     #
#                 (and variables needed across scripts)                        #
################################################################################
library(dplyr)
library(torch)
library(cito)

# Define path for saving preprocessed dataset
folder <- file.path("data","preprocessed_split_data")
cv_split_folder <- file.path(folder, "cross_validation_folds")

# Create directory if it doesn't exist
if (!dir.exists(cv_split_folder)) {
  dir.create(cv_split_folder, recursive = TRUE)
}


#loading preprocessed development set and the predictor and species names to
#make them available across scripts if the split was created already
if (file.exists(file.path(folder, "train_set_predictors.csv"))) {
  train_set_predictors <- read.csv(file.path(folder, "train_set_predictors.csv"))
  predictor_names <- colnames(train_set_predictors)
}
if (file.exists(file.path(folder, "train_set_response.csv"))) {
  train_set_response <- read.csv(file.path(folder, "train_set_response.csv"))
  species_names <- colnames(train_set_response)
}
#loading the indices of the grouped cross-validation split
if (file.exists(file.path(cv_split_folder, "grouped_5fold_cv_indeces.rds"))) {
  cv_indices <- readRDS(file.path(cv_split_folder, "grouped_5fold_cv_indeces.rds"))
}

################################################################################
# Balanced Split function for presence/absence data

################################################################################
# function performs a grouped split for initial test/development set or 
# cross-validation folds and returns the most balanced split object (rsample)

# response_set: presence/absence data set that should be split
# type: "initial" for initial split requires prop
#       "cv"  for cross-validation split requires k
# n: times to perform the split
# k: number of cv-folds
# prop: proportion of the initial 
balanced_grouped_stratified_split <- function(response_set,
                                     type = c("initial", "cv"),
                                     n,
                                     k,
                                     prop = 0.8) {
  library(rsample)
  library(broom)
  #prevalence: mean presence of species in the whole data set
  p_s <- colMeans(response_set)
  
  #logic to handle different split types initial vs. cross-validation
  #defines the split function depending on the prompted split type for the use 
  #the main loop. Returns the split and split-summaries for balance calculation 
  #in the main loop
  if (type == "initial") {
    split_function <- function(){
      split <- group_initial_split(response_set, group = "cell", prop = prop)
      
      development_data <- training(split)%>%
        select(all_of(species_names))
      test_data <- testing(split)%>%
        select(all_of(species_names))
      #create a list that stores for each fold k:
      #   n_k   - the number of observations in the fold
      #   p_sk  - mean presence of each species in the fold
      presence_species_folds <- list(
        list(n_k = nrow(development_data), p_sk = colMeans(development_data)),
        list(n_k = nrow(test_data), p_sk = colMeans(test_data))
      )
      return(list(split=split, folds=presence_species_folds))
    }
  }
  else{
    split_function <- function() {
      #perform a split
      split <- group_vfold_cv(response_set, group = "cell" , v = k)
      #create a list that stores for each fold k:
      #   n_k   - the number of observations in the fold
      #   p_sk  - mean presence of each species in the fold
      presence_species_folds <- list()
      for (fold in 1:k) {
        val_set <- assessment(split$splits[[fold]])
        presence_species_folds[[fold]] <- list(n_k = nrow(val_set), p_sk = colMeans(val_set))
      }
      return(list(split=split, folds=presence_species_folds))
    }
  }
  
  #initial metric value to determine presence balance-will be low for good balance
  fold_balance_metric <- Inf
  
  #MAIN LOOP
  for (s in 1:n) {
    #split the data grouped by cell
    curr_split <- split_function()
    #calculate penalty metric for balance:
    #each folds k squared presence deviation from the overall species prevalence
    #normalized by species' presence/absence Bernoulli Variance
    folds_balance <- lapply(curr_split$folds, function(fold) {
      penalty_sk <- (fold$n_k * (fold$p_sk - p_s)^2) / ((1 - p_s) * p_s)
      return(penalty_sk[!names(penalty_sk) %in% "cell"])
      })
      #sum over folds and species
      curr_split_balance <- sum(sapply(folds_balance, sum))
      print(curr_split_balance)
      
      if (curr_split_balance < fold_balance_metric) {
        fold_balance_metric <- curr_split_balance
        balanced_split <- curr_split$split
      }
    }
    return(balanced_split)
  }



################################################################################
#function for a binary cito multi-species model evaluation
################################################################################
#This function should only be used for evaluation in cross-validation, as it 
#returns the maximum TSS with the optimal threshold optimized on the validation set

# to avoid data leakage, for final model evaluation function 
# "evaluate_model_fixed_threshold()" should be used
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
################################################################################
# evaluation function that evaluates TSS for a predefined threshold
# to be used for final model evaluation

#only set defauft_predict_function to FALSE, if predictions should be made for 
#the MVP model after model was reloaded
evaluate_model_fixed_threshold <- function(model, 
                                           test_pred, 
                                           test_resp, 
                                           fixed_threshold,
                                           default_predict_function =TRUE){
  #predictions for the test set predictors
  if (default_predict_function ==TRUE){
    predictions <- predict(model, newdata = test_pred, type="response")}
  else{
    predictions <- predict_mvp_manual(X = test_pred, model = model, species_names = species_names)  
  }
  
  col_sp <- rainbow(length(species_names))
  if (all(dim(predictions) == dim(test_resp))){
    
    #calculate AUC and TSS per species and plot ROC
    AUC_species <- numeric(length(species_names))
    TSS_species <- numeric(length(species_names))
    
    names(AUC_species) <- species_names
    for (species in 1:length(species_names)) {
      
      ROC_species <- roc(test_resp[,species], predictions[,species])
      AUC_species[species] <- ROC_species$auc
      
      # Compute sensitivity and specificity at the fixed threshold
      sp_metrics <- coords(ROC_species, x = fixed_threshold[species], 
                           input = "threshold", ret = c("sensitivity", "specificity"))
      TSS_species[species] <- sp_metrics$sensitivity + sp_metrics$specificity - 1
      
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
    threshold <- fixed_threshold
    names(threshold) <- species_names
    return(rbind(AUC_species,TSS_species,threshold))
    
  }else{
    print("Prediction dimensions do not match the test set response dimensions")
    return(NULL)
  }
}
################################################################################
#function to evaluate a generalized linear model with leave one out cross validation
leave_one_out_cv <- function(formula="mean_AUC~", data_fr=configurations){
  RMSE <-c()
  for (i in 1:nrow(data_fr)) {
    train_data <- data_fr[-i,]
    model <- glm(formula, data = train_data)
    RMSE <- c(RMSE, (predict(model, data_fr[i,])-data_fr$mean_AUC[i])^2)
    
  }
  return(sqrt(sum(RMSE)/nrow(data_fr)))
}
################################################################################
# Helper function to extract species-specific thresholds aggregated across CV folds.
# Takes a CV model object (as returned by cross_validate_ann) and returns
# the median threshold per species across all folds.

get_cv_median_threshold <- function(cv_model){
  # Extract the threshold row from each fold's performance matrix
  threshold_matrix <- sapply(cv_model$performance, function(x) x["threshold", ])
  # threshold_matrix: species (rows) x folds (columns)
  # return the median threshold
  median_threshold <- apply(threshold_matrix, 1, median)
  return(median_threshold)
}



################################################################################
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
                 early_stopping = 10,
                 activation = "prelu",
                 lr_scheduler = config_lr_scheduler("reduce_on_plateau", 
                                                    patience = 5, factor = 0.5),
                 #model specific configurations
                 loss = model_configs$loss,
                 epochs = model_configs$epochs,
                 lr = model_configs$lr,
                 hidden = model_configs$hidden,
                 batchsize = model_configs$batchsize,
                 lambda = model_configs$lambda,
                 alpha = model_configs$alpha)
    
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

#only set defauft_predict_function to FALSE, if predictions should be made for 
#the MVP model after model was reloaded
predict_map <- function(model, map_predictor_values, 
                        xy_valid_cells, 
                        presence = FALSE, 
                        threshold = rep(0.5,length(species_names)), #vector of species-specific thresholds
                        batch_size = 10000,
                        default_predict_function = TRUE){
  
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
    #for mvp model if torch pointer are broken
    if (default_predict_function==FALSE){
      predictions_list[[i]] <- predict_mvp_manual(X = map_predictor_values[start_idx:end_idx, ], 
                                       model = model,
                                       species_names = species_names)
      
      gc()  # Clean up memory between batches
      next
    }
    #standard default case with cito's predict function
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

#create a heatmap from a correlation matrix
create_heatmap <- function(cor_matrix, 
                           title = "Insert a title", 
                           scale = "AC"){
  # Convert matrix to long format
  cor_long <- melt(cor_matrix)
  colnames(cor_long) <- c("Species1", "Species2", "Correlation")
  
  colors = c("darkred", "red", "pink", "white", "lightblue", "blue", "darkblue")
  values = scales::rescale(c(-1, -0.4, -0.2, 0, 0.2, 0.4, 1))  # More colors near 0
  if (scale=="B"){
    values = scales::rescale(c(-1, -0.5, -0.3, 0, 0.3, 0.5, 1))  # less colors near 0
  }
  # Create heatmap
  heatmap <- ggplot(cor_long, aes(x = Species1, y = Species2, fill = Correlation)) +
    geom_tile(color = "white", linewidth = 0.5) +
    scale_fill_gradientn(
      colors = colors,
      values = values,
      limits = c(-1, 1),
      name = "Correlation"
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1, size = 9),
          axis.text.y = element_text(size = 9),
          axis.title = element_blank(),
          #legend.position = "right",
          panel.grid = element_blank()) +
    coord_fixed() +
    labs(title = title)
  
  return(heatmap)
}


#function to manually predict for the MVP model by extracting its saved weights
#function becomes necessary as cito's own predict function fails after reloading
#the model object
predict_mvp_manual <- function(X, model, species_names) {
  X <- as.matrix(X)
  
  # Get weights from saved model
  W1 <- model$weights[[2]]$`0.weight`
  b1 <- model$weights[[2]]$`0.bias`
  alpha1 <- model$weights[[2]]$`1.weight`
  
  W2 <- model$weights[[2]]$`2.weight`
  b2 <- model$weights[[2]]$`2.bias`
  alpha2 <- model$weights[[2]]$`3.weight`
  
  W3 <- model$weights[[2]]$`4.weight`
  b3 <- model$weights[[2]]$`4.bias`
  
  # Forward pass
  z1 <- X %*% t(W1) + matrix(b1, nrow(X), length(b1), byrow = TRUE)
  h1 <- ifelse(z1 > 0, z1, alpha1 * z1)
  
  z2 <- h1 %*% t(W2) + matrix(b2, nrow(h1), length(b2), byrow = TRUE)
  h2 <- ifelse(z2 > 0, z2, alpha2 * z2)
  
  logits <- h2 %*% t(W3) + matrix(b3, nrow(h2), length(b3), byrow = TRUE)
  probs <- 1 / (1 + exp(-logits*1.7012))
  
  colnames(probs) <- species_names
  return(probs)
}

#function to predict probabilities of species compositions (for single locations!!!)
#composition should be a vector of length 25:
#with 1 for species that should be present, 0 for species that should be absent and NA for species that can be ignored
predict_jointly <- function(X, model, composition, species_names) {
  X <- as.matrix(X)
  
  #create the covariance matrix
  M <- model$parameter$parameter
  covariance <- M%*%t(M) + diag(1, length(species_names))
  
  #check how the user choose to input the species composition and convert to a condition
  if(length(composition) == 25 && all(is.element(composition, c(0L, 1L, NA)))){
    condition <- composition
  }else if(is.list(composition)){
    condition <- sapply(species_names, function(s){
      if(s %in% names(composition)){
        composition[[s]]
      }else{NA}
      
    }, simplify = T)
    
  }else{stop("wrong input of composition to predict")}
  
  # Get network weights from saved model
  W1 <- model$weights[[2]]$`0.weight`
  b1 <- model$weights[[2]]$`0.bias`
  alpha1 <- model$weights[[2]]$`1.weight`
  
  W2 <- model$weights[[2]]$`2.weight`
  b2 <- model$weights[[2]]$`2.bias`
  alpha2 <- model$weights[[2]]$`3.weight`
  
  W3 <- model$weights[[2]]$`4.weight`
  b3 <- model$weights[[2]]$`4.bias`
  
  # Forward pass
  z1 <- X %*% t(W1) + matrix(b1, nrow(X), length(b1), byrow = TRUE)
  h1 <- ifelse(z1 > 0, z1, alpha1 * z1)
  
  z2 <- h1 %*% t(W2) + matrix(b2, nrow(h1), length(b2), byrow = TRUE)
  h2 <- ifelse(z2 > 0, z2, alpha2 * z2)
  
  #network output (environmental suitability)
  logits <- h2 %*% t(W3) + matrix(b3, nrow(h2), length(b3), byrow = TRUE)
  
  #create multivariate distribution with the environmental suitability as mean
  draw <- distr_multivariate_normal(torch_tensor(logits), torch_tensor(covariance))
  x <- draw$sample(10000) # draw x monte carlo samples from the multivariate distribution
  
  #convert to presence/absence
  binary_occurrence <- torch_where(x>0, 1,0) #if else for torch
  binary_array <- as.array(torch_squeeze(binary_occurrence))
  
  
  check <- apply(binary_array, 1, function(n){
    #check if specified species presences and absences are matched
    match <- ifelse(is.na(condition), NA, ifelse(n==condition, 1, 0)) #keep NA for species that are not asked for
    
    #check whether the whole joint probability condition is met
    condition_check <- sum(match, na.rm = T)==sum(!is.na(match))
    
    return(condition_check)
    })
  
  #return probability as the proportion of samples that met the composition
  probability <- mean(check)
  return(probability)
}





