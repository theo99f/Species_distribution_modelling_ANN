################################################################################
#                                Functions                                     #
# 

#loading preprocessed development set
directory <- "data/preprocessed_data/final_data_split/development_set/"
train_set_predictors <- read.csv(paste0(directory,"train_set_predictors.csv"))
full_response <- read.csv(paste0(directory,"train_set_response.csv"))

#saving the predictor and species names to make them available across scripts
predictor_names <- colnames(train_set_predictors)
species_names <- colnames(full_response)[6:30]

#function for a binary multi-species model evaluation
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

