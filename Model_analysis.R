################################################################################
#                                Model Analysis                                #
#                                                                              #
source("Functions.R")
library(ggplot2)
library(reshape2)
library(patchwork)

binomial_ann <- readRDS("results/final_trained_models/binomial_ann")
mvp_ann <- readRDS("results/final_trained_models/mvp_ann")

analyze_training(binomial_ann)

coef(binomial_ann)
coef(mvp_ann)

library(igraph)
library(ggraph)
plot(binomial_ann)


# Compare Correlations of Raw observed data, Correlations in environmental predicted 
# occurences and residual co-occurence patterns

################################################################################
#look at correlations between species in the predicted data
whole_response_dataset <- read.csv("data/raw_data/final_raw_dataset/fish_observations_wide.csv")[,6:30]

cor_dataset <- cor(whole_response_dataset)
# Convert matrix to long format
cor_long <- melt(cor_dataset)
colnames(cor_long) <- c("Species1", "Species2", "Correlation")

# Create heatmap
p1 <- ggplot(cor_long, aes(x = Species1, y = Species2, fill = Correlation)) +
  geom_tile(color = "white", linewidth = 0.5) +
  scale_fill_gradient2(low = "darkred", 
                       mid = "white", 
                       high = "blue",
                       midpoint = 0,
                       limits = c(-1, 1),
                       name = "Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, size = 9),
        axis.text.y = element_text(size = 9),
        axis.title = element_blank(),
        legend.position = "right",
        panel.grid = element_blank()) +
  coord_fixed() +
  labs(title = "A) Observed Correlations (Raw Data)")


#################################################################################
#look at correlations between species in the predicted data
predictions_dataset_mvp <- read.csv("results/community_analysis/predicted_community_mvp.csv")[,2:26]
colnames(predictions_dataset_mvp) <- species_names

predicted_mvp_correlations <- cor(predictions_dataset_mvp)
# Convert matrix to long format
cor_long <- melt(predicted_mvp_correlations)
colnames(cor_long) <- c("Species1", "Species2", "Correlation")

# Create heatmap
p2 <- ggplot(cor_long, aes(x = Species1, y = Species2, fill = Correlation)) +
  geom_tile(color = "white", linewidth = 0.5) +
  scale_fill_gradient2(low = "darkred", 
                       mid = "white", 
                       high = "blue",
                       midpoint = 0,
                       limits = c(-1, 1),
                       name = "Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, size = 9),
        axis.text.y = element_text(size = 9),
        axis.title = element_blank(),
        legend.position = "right",
        panel.grid = element_blank()) +
  coord_fixed() +
  labs(title = "B) Environmental Component (MVP Predictions)")




#check the covariance matrix of the joint species distribution model

# cov2cor() scales a covariance matrix by its diagonal
# to become the correlation matrix.

M <- mvp_ann$parameter$parameter
biotic_association = cov2cor(M%*%t(M) + diag(1, length(species_names)))
colnames(biotic_association) <- species_names
rownames(biotic_association) <- species_names


#create a heatmap for residual correlations

# Convert matrix to long format
cor_long <- melt(biotic_association)
colnames(cor_long) <- c("Species1", "Species2", "Correlation")

# Create heatmap
p3 <- ggplot(cor_long, aes(x = Species1, y = Species2, fill = Correlation)) +
  geom_tile(color = "white", linewidth = 0.5) +
  scale_fill_gradient2(low = "darkred", 
                       mid = "white", 
                       high = "blue",
                       midpoint = 0,
                       limits = c(-1, 1),
                       name = "Residual\nCorrelation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, size = 9),
        axis.text.y = element_text(size = 9),
        axis.title = element_blank(),
        legend.position = "right",
        panel.grid = element_blank()) +
  coord_fixed() +
  labs(title = "C) Residual Co-occurrence Patterns (MVP Covariance matrix)")


#save a combined plot
ggsave("results/plots/correlation_decomposition.tiff", 
       p1 + p2 + p3, 
       width = 18, 
       height = 6, 
       dpi = 300,
       compression = "lzw")

################################################################################
#look at correlations between species in the predicted data
predictions_dataset_binomial <- read.csv("results/community_analysis/predicted_community_binomial.csv")[,2:26]
colnames(predictions_dataset_binomial) <- species_names

predicted_binomial_correlations <- cor(predictions_dataset_binomial)
# Convert matrix to long format
cor_long <- melt(predicted_mvp_correlations)
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








mvp_ann$net(as.matrix(train_set_predictors[1,]))
torch_sigmoid(mvp_ann$net$forward(as.matrix(train_set_predictors[1,]))*1.7012)

predict(mvp_ann, train_set_predictors[1,], type= "response")

m <- distr_multivariate_normal(torch_zeros(25), torch_tensor(biotic_association))
m$sample() # normally distributed with mean=`[0,0]` and covariance_matrix=`I`


