

test_set_predictors <- read.csv("data/preprocessed_data/final_data_split/test_set/test_set_predictors.csv")
test_set_response <- read.csv("data/preprocessed_data/final_data_split/test_set/test_set_response.csv")[6:30]

cv_mvp_model <- readRDS("results/candidate_models/mvp_ann/CV_models_hidden_27_27")
thr_mvp <-get_cv_median_threshold(cv_mvp_model)

test_predictions_fold <- list()
for (m in 1:length(cv_indices)) {
  test_predictions_fold[[m]] <- predict_mvp_manual(test_set_predictors, cv_mvp_model$model[[m]], species_names)
}

aggregated_predictions <- Reduce("+", test_predictions_fold)/length(test_predictions_fold)

#calculate AUC and TSS per species and plot ROC
AUC_species <- numeric(length(species_names))
TSS_species <- numeric(length(species_names))

names(AUC_species) <- species_names
names(TSS_species) <- species_names
for (species in 1:length(species_names)) {
  
  ROC_species <- roc(test_set_response[,species], aggregated_predictions[,species])
  AUC_species[species] <- ROC_species$auc
  
  # Compute sensitivity and specificity at the fixed threshold
  sp_metrics <- coords(ROC_species, x = thr_mvp[species], 
                       input = "threshold", ret = c("sensitivity", "specificity"))
  TSS_species[species] <- sp_metrics$sensitivity + sp_metrics$specificity - 1
}
  
mean(AUC_species)
mean(TSS_species)

cv_bin_model <- readRDS("results/candidate_models/binomial_ann/CV_models_hidden_30_25")
thr_bin <-get_cv_median_threshold(cv_bin_model)

test_predictions_fold <- list()
for (m in 1:length(cv_indices)) {
  test_predictions_fold[[m]] <- predict(cv_bin_model$model[[m]], newdata=test_set_predictors, type="response")
}

aggregated_predictions <- Reduce("+", test_predictions_fold)/length(test_predictions_fold)

#calculate AUC and TSS per species and plot ROC
AUC_species <- numeric(length(species_names))
TSS_species <- numeric(length(species_names))

names(AUC_species) <- species_names
names(TSS_species) <- species_names
for (species in 1:length(species_names)) {
  
  ROC_species <- roc(test_set_response[,species], aggregated_predictions[,species])
  AUC_species[species] <- ROC_species$auc
  
  # Compute sensitivity and specificity at the fixed threshold
  sp_metrics <- coords(ROC_species, x = thr_bin[species], 
                       input = "threshold", ret = c("sensitivity", "specificity"))
  TSS_species[species] <- sp_metrics$sensitivity + sp_metrics$specificity - 1
}
