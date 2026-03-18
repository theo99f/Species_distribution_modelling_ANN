################################################################################
#       Script for creating grouped random Cross-Validation folds              #
#             Random grouped balanced sampling of k-folds                      #
#

#    DO NOT RUN THE "SAVING FILES" SECTION, TO AVOID LOSING THE SPLIT USED     #

library(rsample)
library(broom)

#loading preprocessed development set
directory <- "data/preprocessed_data/final_data_split/development_set/"
full_response <- read.csv(paste0(directory,"train_set_response.csv"))

species_names <- colnames(full_response)[6:30]

train_set_response <- full_response %>%
  select(all_of(species_names),cell)


#function performs grouped cross-validation for k folds for n times and returns 
#the split with most balanced species presences across folds

grouped_stratified_k_fold <- function(response_set, k, n){
  #mean presence of species in the data set
  p_s <- colMeans(response_set)
  #initial metric value to determine presence balance-will be low for good balance
  fold_balance_metric <- Inf
  for (split in 1:n) {
    #split the data grouped by cell
    split_current <- group_vfold_cv(response_set, group = "cell" , v=k)
    
    #create a list that stores number of samples in each fold k (n_k) and the 
    #mean presence of each species in each fold p_sk
    presence_species_folds <- list()
    for (fold in 1:k) {
      val_set <-assessment(split_current$splits[[fold]])
      presence_species_folds[[fold]] <- list(n_k = nrow(val_set), 
                                             p_sk = colMeans(val_set))
    }
    #calculate penalty for each folds presence deviation from overall species presence 
    #normalized by species individual standard deviations excluding cell numbers
    folds_balance <- lapply(presence_species_folds, function(k){
      penalty_sk <- (k$n_k*(k$p_sk-p_s)^2)/((1-p_s)*p_s)
      return(penalty_sk[! names(penalty_sk)%in% "cell"])
    })
    #sum over folds and species
    curr_split_balance <- sum(sapply(folds_balance, sum))
    print(curr_split_balance)
    
    if (curr_split_balance<fold_balance_metric){
      fold_balance_metric <- curr_split_balance
      balanced_split <- split_current
    }
  }
  return(balanced_split)
}

#create a 5-fold random species presence balanced and cell-grouped cross-validation split
grouped_5_fold_cv_split <-grouped_stratified_k_fold(train_set_response, k=5, n=100)

############################SAVING FILES########################################
#save the whole split object
cv_split_folder <- "data/preprocessed_data/final_data_split/development_set/5-fold_grouped_cv_split/"
saveRDS(grouped_5_fold_cv_split, paste0(cv_split_folder,"cv_folds_5_grouped.rds")) 

#create a list with row indeces for each fold 
fold_indices <- tidy(grouped_5_fold_cv_split)
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
#save cross-validation indices list
saveRDS(cv_5_fold_indices, paste0(cv_split_folder,"grouped_5fold_cv_indeces.rds")) 


