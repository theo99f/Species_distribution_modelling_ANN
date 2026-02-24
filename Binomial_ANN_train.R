################################################################################
#                                 Binomial ANN                                 #
#                                                                              #
#          Script for hyperparameter tuning, training and model selection      #
################################################################################

library(torch)
library(cito)
library(terra)
library(dplyr)
library(tidyr)
library(pROC)
source("Functions.R")

#loading preprocessed development set
directory <- "data/preprocessed_data/final_data_split/development_set/"
train_set_predictors <- read.csv(paste0(directory,"train_set_predictors.csv"))
full_response <- read.csv(paste0(directory,"train_set_response.csv"))

predictor_names <- colnames(train_set_predictors)
species_names <- colnames(full_response)[6:30]

train_set_response <- full_response %>%
  select(all_of(species_names))

##################Hyperparameter Screening######################################
#using standard 3-fold cross-validation with a broad range of hyperparameters
model_hp_screening <- dnn(X=train_set_predictors, 
                 Y=cbind(train_set_response), 
                 loss = "binomial",
                 validation = 0.1,
                 early_stopping = 15,
                 epochs = 200,
                 activation = "prelu",
                 lr_scheduler = config_lr_scheduler("reduce_on_plateau", 
                                                    patience = 5, factor = 0.5),
                 lr =tune(),
                 hidden = tune(lower = c(10L,1L), upper = c(50L,4L)),
                 batchsize = tune(16,64),
                 lambda = tune(0.0001,0.1),
                 alpha =tune(0, 0.5),
                 tuning = config_tuning(CV = 3, steps = 50))          
#Hyperparameter configurations
hp_tuning <- model_hp_screening$tuning
#sort the hyperparameter configuration models by the validation loss and save width and depth of hidden layers in seperate columns
hp_sorted <- hp_tuning %>%
  mutate(
    hidden_width = sapply(hidden, `[`, 1),
    hidden_layers = sapply(hidden, `[`, 2),
    converged = ifelse(is.infinite(test),0,1)
  ) %>%
  select(-hidden)%>%
  sort_by(hp_tuning$test)

# write to CSV
write.csv(hp_sorted, "results/hyperparameter_tuning_binomial.csv", row.names = FALSE)   


#################################################################################
# Testing hyperparameters effects on convergence (wilcox.test) 
# and model performance (correlation test)

hp_sorted <- read.csv("results/hyperparameter_tuning_binomial.csv")

#Learning rate
#marginally significant effect of the learning rate on convergence
wilcox.test(lr~converged,hp_sorted)
mean(hp_sorted$lr[hp_sorted$converged==1])
mean(hp_sorted$lr[hp_sorted$converged==0])
#no significant effect on performance
cor.test(hp_sorted$lr[!is.infinite(hp_sorted$test)], 
         hp_sorted$test[!is.infinite(hp_sorted$test)], 
         method = "pearson")

#Lambda
wilcox.test(lambda~converged,hp_sorted)
#significant effect of lambda on performance
cor.test(hp_sorted$lambda[!is.infinite(hp_sorted$test)], 
         hp_sorted$test[!is.infinite(hp_sorted$test)], 
         method = "pearson")

wilcox.test(alpha~converged,hp_sorted)
wilcox.test(hidden_width~converged,hp_sorted)
wilcox.test(hidden_layers~converged,hp_sorted)
wilcox.test(batchsize~converged,hp_sorted)



mean(hp_sorted$alpha[hp_sorted$converged==1])
mean(hp_sorted$alpha[hp_sorted$converged==0])

#no significant effect of alpha
cor.test(hp_sorted$alpha[!is.infinite(hp_sorted$test)], hp_sorted$test[!is.infinite(hp_sorted$test)], method = "pearson")


#no significant effect of batch size
cor.test(hp_sorted$batchsize[!is.infinite(hp_sorted$test)], hp_sorted$test[!is.infinite(hp_sorted$test)], method = "pearson")

#significant effect of network width p<0.01
cor.test(hp_sorted$hidden_width[!is.infinite(hp_sorted$test)], hp_sorted$test[!is.infinite(hp_sorted$test)], method = "pearson")

#no significant effect of network depth
cor.test(hp_sorted$hidden_layers[!is.infinite(hp_sorted$test)], hp_sorted$test[!is.infinite(hp_sorted$test)])

mean(hp_sorted$lambda[hp_sorted$converged==1])
mean(hp_sorted$lambda[hp_sorted$converged==0])
#################################################################################





#################################################################################
cv_split_folder <- "data/preprocessed_data/final_data_split/development_set/5-fold_grouped_cv_split/"
cv_indices <- readRDS(paste0(cv_split_folder,"grouped_5fold_cv_indeces.rds")) 



#hyperparameter tuning using random search for 20 configuration steps and 5fold grouped cv
for (config in 1:20) {
  configs <- list(
    loss = "binomial",
    hidden = c(sample(10:40, sample(1:2,1))),
    lambda = runif(1, 0.0001, 0.003), #lambda was significant thus reduced
    alpha = runif(1, 0, 0.4),         #marginally significant thus slightly reduced
    lr = runif(1, 0.5, 0.7),          
    batchsize = 32
  )
  #cross-validate each hyperparameter configuration using 5 folds
  model <- cross_validate_ann(configs, train_set_predictors, train_set_response, cv_indices)
  
  #calculate mean AUC and mean TSS over species for each fold
  mean_metrics_fold <- sapply(model$performance, function(x) rowMeans(x))
  #calculate mean AUC and mean TSS over all folds for the whole configuration
  mean_metric_config <- rowMeans(mean_metrics_fold)[1:2]
  names(mean_metric_config) <- c("mean_AUC", "mean_TSS")
  
  #save configurations and their mean AUC and TSS
  configs$hidden <- paste0(configs$hidden, collapse = ",")
  configs_df <- as.data.frame(configs)
  configs_df <- cbind(configs_df, t(mean_metric_config))
  print(paste0(config,". Configuration was tested"))
  if(config>1){
    configuration <- rbind(configuration, configs_df)}
  else{configuration <- configs_df}
}
#save the hyperparameter search results
write.csv(configuration, file = "results/hyperparameter_configurations_binomial.csv")

#load hyperparameter search results and test for correlations
configuration <- read.csv("results/hyperparameter_configurations_binomial.csv")
configuration <- configuration %>%
  separate_wider_delim(hidden, delim = ",", names = c("layer1", "layer2"), too_few = "align_start")%>%
  mutate(layer= ifelse(is.na(layer2),1,2))

#layer width not significant among one hidden layer models
cor.test(as.numeric(configuration$layer1[!is.na(configuration$layer2)]), 
         configuration$mean_AUC[!is.na(configuration$layer2)], method = "pearson")

#no significant difference between one and two layers
wilcox.test(mean_AUC~layer,configuration)
wilcox.test(mean_TSS~layer,configuration)

#alpha not significant
cor.test(configuration$alpha, configuration$mean_AUC, method = "pearson")
cor.test(configuration$alpha, configuration$mean_TSS, method = "pearson")

#lambda highly significant
cor.test(configuration$lambda, configuration$mean_AUC, method = "pearson")
cor.test(configuration$lambda, configuration$mean_TSS, method = "pearson")

#learning rate not significant
cor.test(configuration$lr, configuration$mean_AUC, method = "pearson")


################################################################################
#FINAL model candidates

#testing 3 models with one hidden layer and 20, 25 and 30 neurons respectively
config20 <- list(loss = "binomial",
                 hidden = c(20),
                 lambda =  0.0001, 
                 alpha = 0.1, 
                 lr = 0.6,
                 batchsize = 32)

config25 <- list(loss = "binomial",
                 hidden = c(25),
                 lambda =  0.0001, 
                 alpha = 0.1,         
                 lr = 0.6,          
                 batchsize = 32)

config30 <- list(loss = "binomial",
                 hidden = c(30),
                 lambda =  0.0001, 
                 alpha = 0.1,         
                 lr = 0.6,          
                 batchsize = 32)

model_cv_20 <- cross_validate_ann(config20, train_set_predictors, train_set_response, cv_indices)
saveRDS(model_cv_20, file = "results/candidate_models/binomial_ann/CV_models_hidden_20")

model_cv_25 <- cross_validate_ann(config25, train_set_predictors, train_set_response, cv_indices)
saveRDS(model_cv_25, file = "results/candidate_models/binomial_ann/CV_models_hidden_25")

model_cv_30 <- cross_validate_ann(config30, train_set_predictors, train_set_response, cv_indices)
saveRDS(model_cv_30, file = "results/candidate_models/binomial_ann/CV_models_hidden_30")



#testing 3 models with two hidden layers
config30_25 <- list(loss = "binomial",
                 hidden = c(30,25),
                 lambda =  0.0001, 
                 alpha = 0.1,         
                 lr = 0.6,          
                 batchsize = 32)

config25_30 <- list(loss = "binomial",
                 hidden = c(25,30),
                 lambda =  0.0001, 
                 alpha = 0.1,         
                 lr = 0.6,          
                 batchsize = 32)

config27_27 <- list(loss = "binomial",
                 hidden = c(27,27),
                 lambda =  0.0001, 
                 alpha = 0.1,         
                 lr = 0.6,          
                 batchsize = 32)

model_cv_30_25 <- cross_validate_ann(config30_25, train_set_predictors, train_set_response, cv_indices)
saveRDS(model_cv_30_25, file = "results/candidate_models/binomial_ann/CV_models_hidden_30_25")

model_cv_25_30 <- cross_validate_ann(config25_30, train_set_predictors, train_set_response, cv_indices)
saveRDS(model_cv_25_30, file = "results/candidate_models/binomial_ann/CV_models_hidden_25_30")

model_cv_27_27 <- cross_validate_ann(config27_27, train_set_predictors, train_set_response, cv_indices)
saveRDS(model_cv_27_27, file = "results/candidate_models/binomial_ann/CV_models_hidden_27_27")

################################################################################

#reloading candidate models
model_cv_27_27 <- readRDS("results/candidate_models/binomial_ann/CV_models_hidden_27_27")
model_cv_30_25 <- readRDS("results/candidate_models/binomial_ann/CV_models_hidden_30_25")
model_cv_25_30 <- readRDS(file = "results/candidate_models/binomial_ann/CV_models_hidden_25_30")
model_cv_20 <- readRDS("results/candidate_models/binomial_ann/CV_models_hidden_20")
model_cv_25 <- readRDS("results/candidate_models/binomial_ann/CV_models_hidden_25")
model_cv_30 <- readRDS("results/candidate_models/binomial_ann/CV_models_hidden_30")


AUC_cv_30_25 <- sapply(model_cv_30_25$performance, function(x){mean(x["AUC_species",])})
AUC_cv_25_30 <- sapply(model_cv_25_30$performance, function(x){mean(x["AUC_species",])})
AUC_cv_27_27 <- sapply(model_cv_27_27$performance, function(x){mean(x["AUC_species",])})
AUC_cv_20 <- sapply(model_cv_20$performance, function(x){mean(x["AUC_species",])})
AUC_cv_25 <- sapply(model_cv_25$performance, function(x){mean(x["AUC_species",])})
AUC_cv_30 <- sapply(model_cv_30$performance, function(x){mean(x["AUC_species",])})

TSS_cv_30_25 <- sapply(model_cv_30_25$performance, function(x){mean(x["TSS_species",])})
TSS_cv_25_30 <- sapply(model_cv_25_30$performance, function(x){mean(x["TSS_species",])})
TSS_cv_27_27 <- sapply(model_cv_27_27$performance, function(x){mean(x["TSS_species",])})
TSS_cv_20 <- sapply(model_cv_20$performance, function(x){mean(x["TSS_species",])})
TSS_cv_25 <- sapply(model_cv_25$performance, function(x){mean(x["TSS_species",])})
TSS_cv_30 <- sapply(model_cv_30$performance, function(x){mean(x["TSS_species",])})


# Prepare data
architectures <- c("1 layer: 20 neurons",
                   "1 layer: 25 neurons",
                   "1 layer: 30 neurons",
                   "2 layers: 27-27 neurons",
                   "2 layers: 25-30 neurons",
                   "2 layers: 30-25 neurons")

AUC_cv <- cbind(AUC_cv_20, AUC_cv_25, AUC_cv_30, 
                AUC_cv_27_27, AUC_cv_25_30, AUC_cv_30_25)
TSS_cv <- cbind(TSS_cv_20, TSS_cv_25, TSS_cv_30, 
                TSS_cv_27_27, TSS_cv_25_30, TSS_cv_30_25)

# Set up plotting area
par(mfrow = c(1, 2), mar = c(4, 10, 3, 2))

# TSS plot
boxplot(TSS_cv,
        horizontal = TRUE,
        names = architectures,
        las = 1,
        col = "lightblue",
        border = "darkblue",
        ylim = c(0.64, 0.69),
        xlab = "True Skill Statistic (TSS)",
        main = "TSS across CV folds")
abline(v = mean(TSS_cv), lty = 2, col = "gray50")

# AUC plot
boxplot(AUC_cv,
        horizontal = TRUE,
        names = architectures,
        las = 1,
        col = "lightcoral",
        border = "darkred",
        ylim = c(0.88, 0.91),
        xlab = "Area Under Curve (AUC)",
        main = "AUC across CV folds")
abline(v = mean(AUC_cv), lty = 2, col = "gray50")

# Reset plotting parameters
par(mfrow = c(1, 1))
################################################################################
#train the best model on all the training data
binomial_ann <- dnn(X=train_set_predictors, 
                    Y=cbind(train_set_response), 
                    loss = "binomial",
                    validation = 0.1,
                    early_stopping = 15,
                    epochs = 200,
                    activation = "prelu",
                    lr_scheduler = config_lr_scheduler("reduce_on_plateau", 
                                                       patience = 5, factor = 0.5),
                    lr = 0.6,
                    hidden = c(30,25),
                    batchsize = 32,
                    lambda = 0.0001,
                    alpha = 0.1)

saveRDS(binomial_ann, file = "results/final_trained_models/binomial_ann")


