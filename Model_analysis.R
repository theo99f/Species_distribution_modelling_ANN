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
binomial_test_eval <- read.csv("results/final_trained_models/binomial_testset_evaluation.csv")
rowMeans(binomial_test_eval[,2:26])

mvp_test_eval <- as.matrix(read.csv("results/final_trained_models/mvp_testset_evaluation.csv"))
rowMeans(mvp_test_eval[,2:26])

gillnet_test_eval <- read.csv("results/final_trained_models/gillnet_mvp_testset_evaluation.csv")
rowMeans(test_eval[,2:26])

thr_mvp <- mvp_test_eval[3 ,species_names]



coef(binomial_ann)
coef(mvp_ann)
coef(gillnet_mvp_ann)

library(igraph)
library(ggraph)
plot(binomial_ann)


# Compare Correlations of Raw observed data, Correlations in environmental predicted 
# occurences and residual co-occurence patterns
################################################################################
#look at correlations between species in the predicted data
whole_response_dataset <- read.csv("data/raw_data/final_raw_dataset/fish_observations_wide.csv")[,6:30]
dim(whole_response_dataset)
range(whole_response_dataset)

# compute correlations
cor_dataset <- cor(whole_response_dataset)
p1 <- create_heatmap(cor_dataset, title = "A) Observed Correlations (Raw Data)")

#################################################################################
#look at correlations between species in the predicted data
predictions_dataset_mvp <- read.csv("results/community_analysis/predicted_community_mvp.csv")[,2:26]
colnames(predictions_dataset_mvp) <- species_names

binary_predictions_mvp <- 1*(sweep(predictions_dataset_mvp, 2, thr_mvp,">"))

binary_predictions_mvp[1,]
predictions_dataset_mvp[1,]

#creating a correlation map using only the predictions on the test data
predicted_mvp_correlations <- cor(binary_predictions_mvp)
predicted_mvp_correlation <- cor(predictions_dataset_mvp)
p2 <- create_heatmap(predicted_mvp_correlations, title = "B) Environmental Component (MVP Predictions)")
p4 <- create_heatmap(predicted_mvp_correlation, title = "B) Environmental Component (MVP Predictions)")

p2 + p4
################################################################################
#check the covariance matrix of the joint species distribution model
# cov2cor() scales a covariance matrix by its diagonal to become the correlation matrix.

M <- mvp_ann$parameter$parameter
biotic_association = cov2cor(M%*%t(M) + diag(1, length(species_names)))
colnames(biotic_association) <- species_names
rownames(biotic_association) <- species_names

#create a heatmap for residual correlations
p3 <- create_heatmap(biotic_association, title = "C) Residual Co-occurrence Patterns (MVP Covariance matrix)")

#save a combined plot
ggsave("results/plots/correlation_decomposition.tiff", 
       p1 + p2 + p3, 
       width = 18, 
       height = 6, 
       dpi = 300,
       compression = "lzw")

################################################################################

gillnet_test_response <- read.csv("data/preprocessed_data/gillnet_only_dataset/test_response.csv")
gillnet_train_response <- read.csv("data/preprocessed_data/gillnet_only_dataset/train_response.csv")
gillnet_response_data <- rbind(gillnet_train_response, gillnet_test_response)[,species_names]

cor_gillnet_dataset <- cor(gillnet_response_data)


p1 <- create_heatmap(cor_gillnet_dataset ,title = "A) Observed Correlations (Raw Gillnet Data)")

predictions_test_set <- read.csv("results/community_analysis/predicted_community_gillnet_testset.csv")[,2:26]
colnames(predictions_test_set) <- species_names
gillnet_predicted_cor <- cor(predictions_test_set)

p2 <- create_heatmap(gillnet_predicted_cor, title = "Environmental Component (MVP Predictions)")

#check the covariance matrix of the joint species distribution model
# M is the latent factor matrix used to construct the covariance matrix
# cov2cor() scales a covariance matrix by its diagonal to become the correlation matrix.

M <- gillnet_mvp_ann$parameter$parameter
biotic_association = cov2cor(M%*%t(M) + diag(1, length(species_names)))
colnames(biotic_association) <- species_names
rownames(biotic_association) <- species_names


p3 <- create_heatmap(biotic_association, title = "Residual Co-occurence Pattern (MVP Covariance matrix)")

p1 + p2 + p3



################################################################################
#load and inspect prediction maps
library(terra)
mvp_map <- rast("results/prediction_maps/prediction_map_mvp.tif")
mvp_map_binary <- rast("results/prediction_maps/prediction_map_mvp_binary.tif")
plot(mvp_map$Coregonus_maraena)
plot(mvp_map_binary$Coregonus_maraena)

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


