#package loading
library(torch)
library(cito)
library(terra)
library(dplyr)
library(tidyr)
library(pROC)

# Load fish data - training labels/response
fish_data <- read.csv("data/raw_data/fish_observations_wide.csv")

#project the longitude/latitude coordinates of training data (locations) to meter coordinates
coordinates_xy <- cbind(fish_data$longitude, fish_data$latitude)
coordinates_meters <- project(coordinates_xy, "EPSG:4326", "EPSG:3035")
colnames(coordinates_meters) <- c("longitude_meters", "latitude_meters")

#load the grid in meter coordinates of predictor values
grid_predictors <- rast("data/raw_data/grid.tif")
plot(grid_predictors)

# Read all predictor rasters
predictor_files <- list.files("data/raw_data/predictors_v2", 
                              pattern = "\\.tif$", 
                              full.names = TRUE)

# Stack them together as raster
predictor_stack <- ?rast(predictor_files)
predictor_names <- names(predictor_stack)
plot(predictor_stack)

#get the values of environmental predictors at fishing locations
predictor_values <- terra::extract(predictor_stack, y=coordinates_meters, cells=T)

#inspect missing values in predictors
for (predictor in 1:length(predictor_names)) {
  print(sum(is.na(predictor_values[,predictor])))
}

#finding rows with complete data in the response variable
complete_rows <- complete.cases(fish_data)

#Whole Train dataset with excluding observations with NA values for predictors O
train_predictors <- predictor_values[!is.na(predictor_values$depth) & complete_rows, predictor_names]

response_abundance <- fish_data[!is.na(predictor_values$depth) & complete_rows, 6:10]
#transforming training labels with abundance data into binary 
response_binary <- ifelse(response_abundance >= 1, 1, 0)

species_names <- colnames(response_abundance)




#function that splits the dataset n times (default=10) and measures overall absence/presence similarity between test and training 
#and returns the test set indexes of the most equal split
index <- 1:nrow(response_binary)
test_train_sampling <- function(response, n=10, test_size=0.2){
    best_split_measure <- 1000
    for (split in 1:n) {
      test_set_index <- sample(index, length(index)*test_size, replace = F)
      train_set_index <- index[!(index %in% test_set_index)]
      
      ratio <-colMeans(response[test_set_index,])/colMeans(response[train_set_index,])
      curr_split_measure <- sum((1-ratio)^2)
      
      if (best_split_measure>curr_split_measure){
        best_split_measure <- curr_split_measure
        final_test_index <-test_set_index
      }
    }
    return(final_test_index)
}
#usage:
test_set_index <- test_train_sampling(response_binary, n=100, test_size = 0.2)
train_set_index <- index[!(index %in% test_set_index)]

summary(train_set_response_binary)
summary(test_set_response_binary)

#train set 90% with the scaled predictors
train_set_predictors <- scale(train_predictors[train_set_index,])
train_set_response_binary <- response_binary[train_set_index,]
train_set_response_abundance <- response_abundance[train_set_index,]

#getting the mean and standard deviations of the training set
center_predictors <- attr(train_set_predictors, "scaled:center")
sd_predictors <- attr(train_set_predictors, "scaled:scale")

#test set 10% by applying the same scale to the test set predictors as present in the training set
test_set_predictors <- train_predictors[test_set_index,] %>%
  scale(center = center_predictors, scale = sd_predictors)
test_set_response_binary <- response_binary[test_set_index,]
test_set_response_abundance <- response_abundance[test_set_index,]
nrow(test_set_response_binary)

#saving the training and testing data as csv
folder <- "data/preprocessed_data/exploration_dataset_ann/"
write.csv(test_set_predictors, file= paste0(folder,"test_set_predictors.csv"), row.names = F)
write.csv(test_set_response_binary, file= paste0(folder,"test_set_response_binary.csv"), row.names = F)
write.csv(train_set_predictors, file= paste0(folder,"train_set_predictors.csv"), row.names = F)
write.csv(train_set_response_binary, file= paste0(folder,"train_set_response_binary.csv"), row.names = F)
write.csv(train_set_response_abundance, file= paste0(folder,"train_set_response_abundance.csv"), row.names = F)
write.csv(test_set_response_abundance, file= paste0(folder,"test_set_response_abundance.csv"), row.names = F)
write.csv(rbind(center_predictors,sd_predictors), file= paste0(folder,"scaling_parameters.csv"), row.names = F)

################################################################################
#reload the same train and test set
folder <- "data/preprocessed_data/exploration_dataset_ann/"
train_set_predictors <- read.csv(paste0(folder,"train_set_predictors.csv"))
train_set_response_binary <- read.csv(paste0(folder,"train_set_response_binary.csv"))
test_set_response_binary <- read.csv(paste0(folder,"test_set_response_binary.csv"))
test_set_predictors <- read.csv(paste0(folder,"test_set_predictors.csv"))
species_names <- colnames(test_set_response_binary)

#model 1 with binomial (binary cross entropy) loss function, allowing for multi-label classification
model1_exploration <- dnn(X=train_set_predictors, 
              Y=cbind(train_set_response_binary), 
              loss = "binomial",
              validation = 0.2,
              early_stopping = 15,
              epochs = 500,
              lr_scheduler = config_lr_scheduler("reduce_on_plateau", patience = 5, factor = 0.5),
              lr = 0.05,
              hidden = c(30L,30L),
              activation = "prelu")

#simple loop to explore model performance on different network depths
#training models with different number of hidden layers
for (layer in 1:5) {
  model1_exploration <- dnn(X=train_set_predictors, 
                            Y=cbind(train_set_response_binary), 
                            loss = "binomial",
                            validation = 0.2,
                            early_stopping = 15,
                            epochs = 500,
                            lr_scheduler = config_lr_scheduler("reduce_on_plateau", patience = 5, factor = 0.5),
                            lr = 0.5,
                            hidden = rep(40, times=layer),
                            activation = "prelu")
  mean_AUC <- mean(evaluate_model(model1_exploration))
  print(paste(layer, ":" ,mean_AUC))
}

#hyperparameter tuning Round 1, with a broad range of hyperparameters
model1_hp <- dnn(X=train_set_predictors, 
              Y=cbind(train_set_response_binary), 
              loss = "binomial",
              validation = 0.2,
              early_stopping = 15,
              epochs = 500,
              activation = "prelu",
              lr_scheduler = config_lr_scheduler("reduce_on_plateau", patience = 5, factor = 0.5),
              lr =tune(0.05, 0.6),
              hidden = tune(lower = c(10L,1L), upper = c(50L,4L)),
              batchsize = tune(10,100),
              lambda = tune(0.0001,0.1),
              alpha =tune(0, 0.5),
              tuning = config_tuning(CV = 3, steps = 50))          

hp_tuning <- model1_hp$tuning
#sort the hyperparameter configuration models by the validation loss and save width and depth of hidden layers in seperate columns
hp_sorted <- hp_tuning %>%
  mutate(
    hidden_width = sapply(hidden, `[`, 1),
    hidden_length = sapply(hidden, `[`, 2)
  ) %>%
  select(-hidden)%>%
  sort_by(hp_tuning$test)

# write to CSV
write.csv(hp_sorted, "results/hyperparameter_tuning_1.csv", row.names = FALSE)   
hp_sorted <- read.csv("results/hyperparameter_tuning_1.csv")

#no significant effect of learning rate but p=0.08
cor.test(hp_sorted$lr[!is.infinite(hp_sorted$test)], hp_sorted$test[!is.infinite(hp_sorted$test)], method = "pearson")

#no significant effect of alpha
cor.test(hp_sorted$alpha[!is.infinite(hp_sorted$test)], hp_sorted$test[!is.infinite(hp_sorted$test)], method = "pearson")

#significant effect of lambda p<0.01: corr= 0.597
cor.test(hp_sorted$lambda[!is.infinite(hp_sorted$test)], hp_sorted$test[!is.infinite(hp_sorted$test)], method = "pearson")

#no significant effect of batch size
cor.test(hp_sorted$batchsize[!is.infinite(hp_sorted$test)], hp_sorted$test[!is.infinite(hp_sorted$test)], method = "pearson")

#no significant effect of network width p=0.3: corr=0.454
cor.test(hp_sorted$hidden_width[!is.infinite(hp_sorted$test)], hp_sorted$test[!is.infinite(hp_sorted$test)])

#significant effect of network depth p<0.03: corr=0.486
cor.test(hp_sorted$hidden_length[!is.infinite(hp_sorted$test)], hp_sorted$test[!is.infinite(hp_sorted$test)])

hp_sorted$mean_auc <- 0

#training hyperparameter 
for (config in 1:length(hp_sorted$test[!is.infinite(hp_sorted$test)])){
  lr <- hp_sorted$lr[config]
  network_depth <- hp_sorted$hidden_length[config]
  layer_width <- hp_sorted$hidden_width[config]
  batchsize <-hp_sorted$batchsize[config]
  lambda <- hp_sorted$lambda[config]
  alpha <- hp_sorted$alpha[config]
  model1 <- dnn(X=train_set_predictors, 
                   Y=cbind(train_set_response_binary), 
                   loss = "binomial",
                   validation = 0.2,
                   early_stopping = 15,
                   epochs = 500,
                   activation = "prelu",
                   lr_scheduler = config_lr_scheduler("reduce_on_plateau", patience = 5, factor = 0.5),
                   lr =lr,
                   hidden = rep(layer_width, times=network_depth),
                   batchsize = batchsize,
                   lambda = lambda,
                   alpha = alpha)
  auc_score <- evaluate_model(model1)
  hp_sorted$mean_auc[config] <- mean(auc_score)
}





residuals(model1)
analyze_training(model1)
summary(model1)
model1_hp$model_properties



#model2 with mvp -loss considering covariance of species occurence
model2 <- dnn(X=train_set_predictors, 
              Y=cbind(train_set_response_binary), 
              hidden = c(30L,30L),
              activation = "relu",
              loss = "mvp",
              validation = 0.2,
              early_stopping = 20,
              epochs = 500,
              lr = 0.01,
              lr_scheduler = config_lr_scheduler("reduce_on_plateau", patience = 5, factor = 0.5))


analyze_training(model2)
summary(model2)
evaluate_model(model2)
#function that tests 
separate_species_model <- function(species){
  idx <- species==species_names
  model <- dnn(X=train_set_predictors, 
      Y=train_set_response_binary[,idx], 
      hidden = c(30L,30L),
      activation = "relu",
      loss = "binomial",
      validation = 0.2,
      early_stopping = 10,
      epochs = 100,
      lr = 0.03,
      lr_scheduler = config_lr_scheduler("reduce_on_plateau", patience = 5, factor = 0.5))
  return(model)
}

pike <-separate_species_model(species_names[4])
evaluate_single_species_model(pike, test_predictors = test_set_predictors, test_response = test_set_response_binary[,4])
mean(test_set_response_binary[,4])

perch <- separate_species_model(species_names[2])
evaluate_single_species_model(pike, test_predictors = test_set_predictors, test_response = test_set_response_binary[,2])

herring <- separate_species_model(species_names[1])
evaluate_single_species_model(pike, test_predictors = test_set_predictors, test_response = test_set_response_binary[,1])

stickleback <- separate_species_model(species_names[5])
evaluate_single_species_model(pike, test_predictors = test_set_predictors, test_response = test_set_response_binary[,5])


analyze_training(pike)
analyze_training(perch)


####################################################################################################
#making model evaluation

#function for a binary multi-species model evaluation
evaluate_model <- function(model, model_name = "DEFAULT"){
  #predictions for the test set predictors
  predictions <- predict(model, newdata = test_set_predictors, type="response")
  col_sp <- rainbow(length(species_names))
  if (all(dim(predictions) == dim(test_set_response_binary))){
    #overall test accuracy
    correct <- ifelse((predictions >= 0.5 & test_set_response_binary==1) | (predictions < 0.5 & test_set_response_binary==0),1,0)
    correct <- correct[!is.na(correct)]
    test_accuracy <- sum(correct)/length(correct)
    print(paste("Overall test accuracy is", test_accuracy))
    
    #calculate AUC per species and plot ROC
    AUC_package <- numeric(length(species_names))
    names(AUC_package) <- species_names
    
    for (species in 1:length(species_names)) {
      ROC_species <- roc(test_set_response_binary[,species], predictions[,species])
      AUC_package[species] <- ROC_species$auc
      print(paste(species_names[species]," - ", ROC_species$auc))
      
      if (species==1){
        plot(ROC_species, col =col_sp[species])
      }
      else {plot(ROC_species, col =col_sp[species], add= T)}
    }
    legend("bottom", legend = species_names, 
           col = col_sp,
           lty = 1,
           lwd = 2,
           bty ="n")
    return(AUC_package)
    
  }else{
    print("Prediction dimensions do not match the test set response dimensions")
    return(NULL)
  }
}


AUC_model1 <- evaluate_model(model1) #c(30,30)
AUC_model2 <- evaluate_model(model2)

#function for evaluating single 
evaluate_single_species_model <- function(model, test_predictors, test_response, model_name = "DEFAULT"){
  #predictions for the test set predictors
  predictions <- predict(model, newdata = test_predictors, type="response")

  if (all(dim(predictions) == dim(test_response))){
    #overall test accuracy
    correct <- ifelse((predictions >= 0.5 & test_response==1) | (predictions < 0.5 & test_response==0),1,0)
    correct <- correct[!is.na(correct)]
    test_accuracy <- sum(correct)/length(correct)
    print(paste("Overall test accuracy is", test_accuracy))
    
    
    #calculate AUC per species and plot ROC
    ROC_species <- roc(test_response, predictions)
    plot(ROC_species)
    print(ROC_species$auc)
    
    return(ROC_species)
    
  }else{
    print("Prediction dimensions do not match the test set response dimensions")
    return(NULL)
  }
}

#check the covariance matrix of the joint species distribution model
M <- model2$parameter$parameter
biotic_association = cov2cor(M%*%t(M) + diag(1, length(species_names)))
biotic_association

#correlation of cooccurence in the train data
cor(train_set_response_binary)

#correlation of cooccurence in model1
pred_binomial <- predict(model1, train_set_predictors, type = "response")
env_cor <- cor(pred_binomial)

#correlation of cooccurence in model2
pred_joint <- predict(model2, train_set_predictors, type = "response")
joint_cor <- cor(pred_joint)





#threshold loop to calculate accuracy with different uncertainty margins
for (uncertainty_margin in 0:8) {
  presence_threshold <- 0.5 + uncertainty_margin/20
  absence_threshold <- 0.5 - uncertainty_margin/20
  print(paste("absence threshold: ", absence_threshold,'-', "presence threshold: ", presence_threshold))
  
  TP <- sum(predictions >= presence_threshold & test_set_response==1, na.rm = T)
  TN <- sum(predictions < absence_threshold & test_set_response==0, na.rm = T)
  FP <- sum(predictions >= presence_threshold & test_set_response==0, na.rm = T)
  FN <- sum(predictions < absence_threshold & test_set_response==1, na.rm = T)
  
  TPR <- TP/(TP+FN)
  
  FPR <- FP/(FP+TN)
  
  print((TP+TN+FP+FN))
  print(paste("Accuracy", (TP+TN)/(TP+TN+FP+FN)))
}



#plotting and saving Accumulated local effect plots to determine predictor effects
for (pred in 1:length(predictor_names)){
  png(paste0("plots/",predictor_names[pred], ".png"),width= 2000, height = 900)
  ALE(model1, variable = predictor_names[pred])
  
  dev.off()
}


#############################################################################################################################
#predict the whole map

#scaling the predictors using the mean and standard deviation of the training data
map_predictors <- as.data.frame(predictor_stack, xy = TRUE) 
map_predictors_scaled <- map_predictors %>%
  mutate(across(all_of(predictor_names),  ~ (.x - center_predictors[cur_column()])/sd_predictors[cur_column()] ))

#cells / rows with all predictors values present
valid_rows <- rowSums(is.na(map_predictors_scaled))==0
sum(valid_rows)/length(valid_rows)

#saving the coordinates of all the valid grid cells with no missing values
xy_valid_rows <- map_predictors[valid_rows,c("x","y")]
dim(xy_valid_rows)

######################################################################################################################################
#projecting the predictions
map_predictors_scaled[valid_rows, predictor_names]

#function that predicts values for the whole map and returns predictions as a SpatRaster
#By default species presence probabilities are returned, alternatively with presence =True predictions for presence absence are returned and can be customed using a threshold
predict_map <- function(model, map_predictors, presence = FALSE, threshold = 0.5){
  #prediction of all grid cells with valid predictor values
  predicted_map_probabilities <- predict(model, 
                                         newdata = map_predictors, 
                                         type="response")
  
  #projecting the predictions as probabilities
  if (!presence){
    prediction_map <- rast(cbind(xy_valid_rows, predicted_map_probabilities),
                           crs = "EPSG:3035",
                           type = "xyz")
  #projecting the predictions as presence/absence
  }else{
    predicted_map_presence <- ifelse(predicted_map_probabilities> threshold, 1,0)
    prediction_map <- rast(cbind(xy_valid_rows, predicted_map_presence),
                           crs = "EPSG:3035",
                           type = "xyz")}

  names(prediction_map) <- species_names
  return(prediction_map)
}

prediction_map1 <- predict_map(model1_hp)
plot(prediction_map1)

prediction_map2 <- predict_map(model2)
plot(prediction_map2)



###################################################################################

#loop that calculates AUC and plots ROC
AUC_per_species <- numeric(5)
AUC_package <- numeric(length(species_names))

names(AUC_package) <- species_names
col_sp <- rainbow(length(species_names))


for (species in 1:length(species_names)) {
  
  thr <- sort(unique(predictions[, species]))
  TPR <- numeric(length(thr))
  FPR <- numeric(length(thr))
  
  for (t in 1:length(TPR)) {
    p <- thr[t]
    TP <- sum(predictions[,species] >= p & test_set_response[,species]==1, na.rm = T)
    TN <- sum(predictions[,species] < p & test_set_response[,species]==0, na.rm = T)
    FP <- sum(predictions[,species] >= p & test_set_response[,species]==0, na.rm = T)
    FN <- sum(predictions[,species] < p & test_set_response[,species]==1, na.rm = T)
    
    TPR[t] <- TP/(TP+FN)
    FPR[t] <- FP/(FP+TN)
    
  }
  # Sort by FPR 
  sorted_idx <- order(FPR)
  FPR_sorted <- FPR[sorted_idx]
  TPR_sorted <- TPR[sorted_idx]
  
  
  AUC_per_species[species] <- sum(diff(FPR_sorted) * (TPR_sorted[-1] + TPR_sorted[-length(TPR_sorted)]) / 2)
  
  ROC_species <- roc(test_set_response[,species], predictions[,species])
  AUC_package[species] <- ROC_species$auc
  print(paste(species_names[species]," - ", ROC_species$auc))
  
  if (species==1){
    plot(ROC_species, col =col_sp[species])
  }
  else {plot(ROC_species, col =col_sp[species], add= T)}
  #lines(ROC_species$specificities,ROC_species$sensitivities,
  # lwd=3, col = col_sp[species])}
}
legend("bottom", legend = species_names, 
       col = col_sp,
       lty = 1,
       lwd = 2,
       bty ="n")
AUC_package
