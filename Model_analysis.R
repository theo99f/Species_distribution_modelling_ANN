################################################################################
#                                Model Analysis                                #
#                                                                              #
source("Functions.R")
library(ggplot2)
library(reshape2)
library(patchwork)

binomial_ann <- readRDS("results/final_trained_models/binomial_ann")
mvp_ann <- readRDS("results/final_trained_models/mvp_ann")
gillnet_mvp_ann <- readRDS("results/final_trained_models/gillnet_mvp_ann")


analyze_training(binomial_ann)
analyze_training(mvp_ann)
analyze_training(gillnet_mvp_ann)

################################################################################
# Look at Test set evaluation results
binomial_test_eval <- read.csv("results/final_trained_models/binomial_testset_evaluation.csv")
rowMeans(binomial_test_eval[,2:26])

mvp_test_eval <- as.matrix(read.csv("results/final_trained_models/mvp_testset_evaluation.csv"))
rowMeans(mvp_test_eval[,2:26])

gillnet_test_eval <- read.csv("results/final_trained_models/gillnet_mvp_testset_evaluation.csv")
rowMeans(test_eval[,2:26])




coef(binomial_ann)
coef(mvp_ann)
coef(gillnet_mvp_ann)


# Compare Correlations of Raw observed data, Correlations in environmental predicted 
# occurences and residual co-occurence patterns
################################################################################
#look at correlations between species in the predicted data
whole_response_dataset <- read.csv("data/raw_data/final_raw_dataset/fish_observations_wide.csv")[,6:30]
dim(whole_response_dataset)
range(whole_response_dataset)

# compute correlations
cor_dataset <- cor(whole_response_dataset)
p1 <- create_heatmap(cor_dataset, title = "A) Observed (Binary Data)")

#################################################################################
#look at correlations between species in the predicted data
predictions_dataset_mvp <- read.csv("results/community_analysis/predicted_community_mvp.csv")[,2:26]
colnames(predictions_dataset_mvp) <- species_names


#creating a correlation map using only the predictions on the test data
predicted_mvp_correlations <- cor(predictions_dataset_mvp[30966:38707,])
p2 <- create_heatmap(predicted_mvp_correlations, title = "B) Environmental (Predicted Probabilities)", scale = "B")

################################################################################
#check the covariance matrix of the joint species distribution model
# cov2cor() scales a covariance matrix by its diagonal to become the correlation matrix.

M <- mvp_ann$parameter$parameter
biotic_association = cov2cor(M%*%t(M) + diag(1, length(species_names)))
colnames(biotic_association) <- species_names
rownames(biotic_association) <- species_names

#create a heatmap for residual correlations
p3 <- create_heatmap(biotic_association, title = "C) Residual Co-occurrence Patterns (Covariance matrix)")

#save a combined plot
ggsave("results/plots/correlation_decomposition.tiff", 
       p1 + p2 + p3, 
       width = 18, 
       height = 6, 
       dpi = 300,
       compression = "lzw")

################################################################################
# Correlation inspection of the raw gillnet data, gillnet model predicted probabilities and residual correlations
gillnet_test_response <- read.csv("data/preprocessed_data/gillnet_only_dataset/test_response.csv")
gillnet_train_response <- read.csv("data/preprocessed_data/gillnet_only_dataset/train_response.csv")
gillnet_response_data <- rbind(gillnet_train_response, gillnet_test_response)[,species_names]

#correlation in gillnet data
cor_gillnet_dataset <- cor(gillnet_response_data)

p1g <- create_heatmap(cor_gillnet_dataset ,title = "A) Observed (Binary Data)")

#correlation in probability predictions of the gillnet model
predictions_test_set <- read.csv("results/community_analysis/predicted_community_gillnet_testset.csv")[,2:26]
colnames(predictions_test_set) <- species_names
gillnet_predicted_cor <- cor(predictions_test_set)

p2g <- create_heatmap(gillnet_predicted_cor, title = "Environmental (Predicted Probabilities)", scale = "B")

#check the covariance matrix of the joint species distribution model
# Mg is the latent factor matrix used to construct the covariance matrix
# cov2cor() scales a covariance matrix by its diagonal to become the correlation matrix.
Mg <- gillnet_mvp_ann$parameter$parameter
biotic_association_gillnet = cov2cor(Mg%*%t(Mg) + diag(1, length(species_names)))
colnames(biotic_association_gillnet) <- species_names
rownames(biotic_association_gillnet) <- species_names

p3g <- create_heatmap(biotic_association_gillnet, 
                      title = "Residual Co-occurence Pattern (Covariance matrix)")


#save a combined plot
ggsave("results/plots/correlation_decomposition_gillnet.tiff", 
       p1g + p2g + p3g, 
       width = 18, 
       height = 6, 
       dpi = 300,
       compression = "lzw")
################################################################################
# Contrary vs. synergistic relationships between environment and residuals
# i.e. that change from positive to negative 
relathionship_cor <- function(res, env, prob = 0, 
                         title = "Correlation relathionships \nEnvironment vs Residuals", 
                         colors = c("orange", "grey", "lightblue")){
  
  res <- ifelse(abs(res) < quantile(abs(res), probs = prob), 0, res)
  env <- ifelse(abs(env) < quantile(abs(env), probs = prob), 0, env)
  
  relationship <- (res * env) / abs(res * env)
  relationship <- ifelse(is.na(relationship), 0, relationship)
  
  cor_long <- melt(relationship)
  colnames(cor_long) <- c("Species1", "Species2", "Correlation")
  cor_long$Correlation <- factor(cor_long$Correlation, levels = c(-1, 0, 1), 
                                 labels = c("Contra", 
                                            paste("weakest\n(<",prob*100,"th percentile)"), 
                                            "Synergy"))
  
  ggplot(cor_long, aes(x = Species1, y = Species2, fill = Correlation)) +
    geom_tile(color = "white", linewidth = 0.5) +
    scale_fill_manual(values = colors, name = NULL) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1, size = 9),
          axis.text.y = element_text(size = 9),
          axis.title = element_blank(),
          panel.grid = element_blank()) +
    coord_fixed() +
    labs(title = title)
}

# Contrary correlation plots
p1_contrary <- relathionship_cor(biotic_association, predicted_mvp_correlations, 
                            prob = 0.25)
p2_contrary <- relathionship_cor(gillnet_predicted_cor, biotic_association_gillnet, 
                            prob = 0.25, 
                            title = "Gillnet: Contrary vs Synergistic")

p1_contrary + p2_contrary
# Save combined plot
ggsave("results/plots/contrary_correlations.tiff",
       p1_contrary + p2_contrary,
       width = 12, height = 6, dpi = 300, compression = "lzw")

################################################################################
#Inspection of correlation values for species combinations 

#saving all correlation matrices
corr_matrix_list <- list(
  Observed_Correlation = cor_dataset,
  Predicted_Correlation = predicted_mvp_correlations,
  Residual_Correlation = biotic_association,
  Observed_Correlation_Gillnet = cor_gillnet_dataset,
  Predicted_Correlation_Gillnet = gillnet_predicted_cor,
  Residual_Correlation_Gillnet = biotic_association_gillnet
)

#function to print all correlation values of a species pair
get_species_pair_correlations <- function(species1, species2,
                                          corr_matrix_list){
  for (matrix_name in names(corr_matrix_list)) {
    matrix <- corr_matrix_list[[matrix_name]]
    print(paste(matrix_name, matrix[species1, species2]))
  }
}



#function to get the n strongest positive and negative correlations with species names
get_top_correlations <- function (n=10, cor_matrix){
  #only values in the lower triangel of the matrix are kept
  #diagonal and duplicate correlations of the upper part are set to 0
  cor <- cor_matrix*lower.tri(cor_matrix)
  
  #extract lowest and highest n correlations
  min_n <- sort(cor)[1:n]
  max_n <- sort(cor, decreasing = T)[1:n]
  
  top_corr <- data.frame(
    species1_positive=numeric(n),
    species2_positive=numeric(n),
    positive_cor=numeric(n),
    species1_negative =numeric(n),
    species2_negative =numeric(n),
    negative_cor = numeric(n)
  )
  for (i in 1:n){
    pos_match <- which(cor==max_n[i], arr.ind = T)
    neg_match <- which(cor==min_n[i], arr.ind = T)
    top_corr$species1_positive[i] <- rownames(cor)[pos_match[1,"row"]]
    top_corr$species2_positive[i] <- colnames(cor)[pos_match[1,"col"]]
    top_corr$positive_cor[i] <- max_n[i]
    top_corr$species1_negative[i] <- rownames(cor)[neg_match[1,"row"]]
    top_corr$species2_negative[i] <- colnames(cor)[pos_match[1,"col"]]
    top_corr$negative_cor[i] <- min_n[i]
  }
  return(top_corr)
}


get_species_pair_correlations("Clupea_harengus","Gasterosteus_aculeatus", corr_matrix_list)
get_species_pair_correlations("Clupea_harengus","Sander_lucioperca", corr_matrix_list)
get_species_pair_correlations("Clupea_harengus", "Perca_fluviatilis", corr_matrix_list)
get_species_pair_correlations("Clupea_harengus", "Gadus_morhua", corr_matrix_list)
get_species_pair_correlations("Clupea_harengus","Esox_lucius", corr_matrix_list)
get_species_pair_correlations("Clupea_harengus","Rutilus_rutilus", corr_matrix_list)
get_species_pair_correlations("Clupea_harengus", "Sprattus_sprattus", corr_matrix_list)

get_species_pair_correlations("Gasterosteus_aculeatus", "Sander_lucioperca", corr_matrix_list)
get_species_pair_correlations("Gasterosteus_aculeatus", "Perca_fluviatilis", corr_matrix_list)
get_species_pair_correlations("Gasterosteus_aculeatus", "Gadus_morhua", corr_matrix_list)
get_species_pair_correlations("Gasterosteus_aculeatus", "Esox_lucius", corr_matrix_list)
get_species_pair_correlations("Gasterosteus_aculeatus", "Sprattus_sprattus", corr_matrix_list)
get_species_pair_correlations("Gasterosteus_aculeatus", "Rutilus_rutilus", corr_matrix_list)

get_species_pair_correlations("Sander_lucioperca", "Perca_fluviatilis", corr_matrix_list)
get_species_pair_correlations("Sander_lucioperca", "Gadus_morhua", corr_matrix_list)
get_species_pair_correlations("Sander_lucioperca", "Esox_lucius", corr_matrix_list)
get_species_pair_correlations("Sander_lucioperca", "Sprattus_sprattus", corr_matrix_list)
get_species_pair_correlations("Sander_lucioperca", "Rutilus_rutilus", corr_matrix_list)

get_species_pair_correlations("Perca_fluviatilis", "Gadus_morhua", corr_matrix_list)
get_species_pair_correlations("Perca_fluviatilis", "Esox_lucius", corr_matrix_list)
get_species_pair_correlations("Perca_fluviatilis", "Sprattus_sprattus", corr_matrix_list)
get_species_pair_correlations("Perca_fluviatilis", "Rutilus_rutilus", corr_matrix_list)

get_species_pair_correlations("Gadus_morhua", "Esox_lucius", corr_matrix_list)
get_species_pair_correlations("Gadus_morhua", "Sprattus_sprattus", corr_matrix_list)
get_species_pair_correlations("Gadus_morhua", "Rutilus_rutilus", corr_matrix_list)

get_species_pair_correlations("Esox_lucius", "Sprattus_sprattus", corr_matrix_list)
get_species_pair_correlations("Esox_lucius", "Rutilus_rutilus", corr_matrix_list)

get_species_pair_correlations("Sprattus_sprattus", "Rutilus_rutilus", corr_matrix_list)

#Interesting residual relationship
get_species_pair_correlations("Osmerus_eperlanus", "Gadus_morhua", corr_matrix_list)
get_species_pair_correlations("Clupea_harengus", "Coregonus_albula", corr_matrix_list)


get_species_pair_correlations("Sander_lucioperca", "Neogobius_melanostomus", corr_matrix_list)


get_top_correlations(n= 15, cor_matrix=biotic_association)
get_top_correlations(n= 15, cor_matrix=biotic_association_gillnet)

get_top_correlations(n= 15, cor_matrix=predicted_mvp_correlations)



################################################################################
#load and inspect prediction maps
library(terra)
mvp_map <- rast("results/prediction_maps/prediction_map_mvp.tif")
mvp_map_binary <- rast("results/prediction_maps/prediction_map_mvp_binary.tif")
gillnet_map <- rast("results/prediction_maps/prediction_map_gillnet.tif")

#
plot(mvp_map$Coregonus_maraena)
plot(mvp_map_binary$Coregonus_maraena)


#Stickleback
plot(mvp_map$Gasterosteus_aculeatus)
plot(mvp_map_binary$Gasterosteus_aculeatus)

#Cod
plot(mvp_map$Gadus_morhua)
plot(mvp_map_binary$Gadus_morhua)
plot(gillnet_map$Gadus_morhua)



################################################################################
#look at correlations between species in the predicted data
predictions_dataset_binomial <- read.csv("results/community_analysis/predicted_community_binomial.csv")[,2:26]
colnames(predictions_dataset_binomial) <- species_names

predicted_binomial_correlations <- cor(predictions_dataset_binomial)
# Convert matrix to long format
cor_long <- melt(predicted_binomial_correlations)
colnames(cor_long) <- c("Species1", "Species2", "Correlation")

# Create heatmap
ggplot(cor_long, aes(x = Species1, y = Species2, fill = Correlation)) +
  geom_tile(color = "white", linewidth = 0.5) +
  scale_fill_gradient2(low = "darkred", 
                       mid = "white", 
                       high = "blue",
                       midpoint = 0,
                       limits = c(-1, 1),
                       name = "Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
        axis.text.y = element_text(size = 8),
        axis.title = element_blank(),
        legend.position = "right",
        panel.grid = element_blank()) +
  coord_fixed() +
  labs(title = "Species correlations in binomial predictions")


################################################################################





mvp_ann$net(as.matrix(train_set_predictors[1,]))
torch_sigmoid(mvp_ann$net$forward(as.matrix(train_set_predictors[1,]))*1.7012)

predict(mvp_ann, train_set_predictors[1,], type= "response")

m <- distr_multivariate_normal(torch_zeros(25), torch_tensor(biotic_association))
m$sample() # normally distributed with mean=`[0,0]` and covariance_matrix=`I`


