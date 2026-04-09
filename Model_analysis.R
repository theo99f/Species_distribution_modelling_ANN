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
binomial_test_eval <- as.matrix(read.csv("results/final_trained_models/binomial_testset_evaluation.csv", row.names = 1))
rowMeans(binomial_test_eval)

mvp_test_eval <- as.matrix(read.csv("results/final_trained_models/mvp_testset_evaluation.csv", row.names = 1))
rowMeans(mvp_test_eval)

gillnet_test_eval <- as.matrix(read.csv("results/final_trained_models/gillnet_mvp_testset_evaluation.csv", row.names = 1))
rowMeans(gillnet_test_eval)



################################################################################
#                        Correlation Decompositon                              #

# Compare Correlations of Raw observed data, Correlations in environmental predicted 
# occurences and residual co-occurence patterns
################################################################################
#look at correlations between species in the predicted data
whole_response_dataset <- read.csv("data/raw_data/final_raw_dataset/fish_observations_wide.csv")
whole_response_dataset <- whole_response_dataset[,species_names]
dim(whole_response_dataset)
range(whole_response_dataset)
colnames(whole_response_dataset) <- gsub("_", " ",species_names)

# compute correlations
observed_correlation <- cor(whole_response_dataset)
p1 <- create_heatmap(observed_correlation, title = "A) Observed (Binary Data)")

#################################################################################
#look at correlations between species in the predicted data
predictions_dataset_mvp <- read.csv("results/community_analysis/predicted_community_mvp.csv", row.names = 1)
colnames(predictions_dataset_mvp) <- gsub("_", " ",species_names)


# Extracting test set predictions only (rows 30966:38707) to compute unbiased
# environmental correlations. In MVP_ANN_final_train_pred_eval.R (line 66-69),
# predictions were saved as rbind(train_set_predictors, test_set_predictors),
# where train has 30966 rows and test has 7741 rows (80/20 grouped split from
# Data_preprocessing_and_split.R). This is congruent with the gillnet model,
# which saves test set predictions separately (predicted_community_gillnet_testset.csv).
predicted_correlation <- cor(predictions_dataset_mvp[30967:38707,])
p2 <- create_heatmap(predicted_correlation, title = "B) Environmental (Predicted Probabilities)", scale = "B")

################################################################################
#check the covariance matrix of the joint species distribution model
# cov2cor() scales a covariance matrix by its diagonal to become the correlation matrix.

M <- mvp_ann$parameter$parameter
residual_correlation = cov2cor(M%*%t(M) + diag(1, length(species_names)))
colnames(residual_correlation) <- gsub("_", " ",species_names)
rownames(residual_correlation) <- gsub("_", " ",species_names)

#create a heatmap for residual correlations
p3 <- create_heatmap(residual_correlation, title = "C) Residual Co-occurrence Patterns (Covariance matrix)")

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
colnames(gillnet_response_data) <- gsub("_", " ",species_names)

#correlation in gillnet data
observed_correlation_gillnet <- cor(gillnet_response_data)

p1g <- create_heatmap(observed_correlation_gillnet ,title = "A) Observed (Binary Data)")

#correlation in probability predictions of the gillnet model
predictions_test_set <- read.csv("results/community_analysis/predicted_community_gillnet_testset.csv", row.names = 1)
colnames(predictions_test_set) <- gsub("_", " ",species_names)
predicted_correlation_gillnet <- cor(predictions_test_set)

p2g <- create_heatmap(predicted_correlation_gillnet, title = "Environmental (Predicted Probabilities)", scale = "B")

#check the covariance matrix of the joint species distribution model
# Mg is the latent factor matrix used to construct the covariance matrix
# cov2cor() scales a covariance matrix by its diagonal to become the correlation matrix.
Mg <- gillnet_mvp_ann$parameter$parameter
residual_correlation_gillnet = cov2cor(Mg%*%t(Mg) + diag(1, length(species_names)))
colnames(residual_correlation_gillnet) <- gsub("_", " ",species_names)
rownames(residual_correlation_gillnet) <- gsub("_", " ",species_names)

p3g <- create_heatmap(residual_correlation_gillnet, 
                      title = "Residual Co-occurence Pattern (Covariance matrix)")


#save a combined plot
ggsave("results/plots/correlation_decomposition_gillnet.tiff", 
       p1g / p2g / p3g , 
       width = 18, 
       height = 8, 
       dpi = 300,
       compression = "lzw")

################################################################################
# Reinforcing vs. counteracting relationships between environment and residuals
# i.e. that change from positive to negative 
relationship_cor <- function(res, env, prob = 0, 
                         title = "Reinforcing and counteracting species associations\n- environmentals vs residual components", 
                         colors = c("darkblue", "darkred", "lightgrey", "darkorange", "#4682B4")){
  # filter weak relationships 
  res <- ifelse(abs(res) < quantile(abs(res[upper.tri(res)]), probs = prob), 0, res)
  env <- ifelse(abs(env) < quantile(abs(env[upper.tri(env)]), probs = prob), 0, env)
  
  # -1 for counteracting and +1 for reinforcing relationships
  relationship <- (res * env) / abs(res * env)
  # 0 for weak relationships
  relationship <- ifelse(is.na(relationship), 0, relationship)
  # positive residual correlation gets separated to -2 or 2
  relationship <- ifelse(res>0, relationship*2, relationship)
  
  cor_long <- melt(relationship)
  colnames(cor_long) <- c("Species1", "Species2", "Correlation")
  cor_long$Correlation <- factor(cor_long$Correlation, levels = c(2, 1, 0, -1, -2), 
                                 labels = c("Reinforced co-occurrence (+/+)",
                                            "Reinforced segregation (-/-)",
                                            paste("Weak correlations (<",prob*100,"th percentile)"), 
                                            "Counteracting residual segregation (+/-)",
                                            "Counteracting residual co-occurrence (-/+)"))
  
  ggplot(cor_long, aes(x = Species1, y = Species2, fill = Correlation)) +
    geom_tile(color = "white", linewidth = 0.5) +
    geom_tile(data = subset(cor_long, Species1 == Species2),
              fill = "black", color = "white", linewidth = 0.5) +
    scale_fill_manual(values = colors, name = NULL) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1, size = 9),
          axis.text.y = element_text(size = 9),
          axis.title = element_blank(),
          panel.grid = element_blank()) +
    coord_fixed() +
    labs(title = title)
}

association_relationships <- relationship_cor(residual_correlation_gillnet, predicted_correlation_gillnet, 
                            prob = 0.25)
# Save plot
ggsave("results/plots/reinforcing_counteracting_associations.tiff",
       association_relationships,
       width = 8, height = 6, dpi = 300, compression = "lzw")

################################################################################
#Inspection of correlation values for species combinations 

#saving all correlation matrices
corr_matrix_list <- list(
  Observed_Correlation = observed_correlation,
  Predicted_Correlation = predicted_correlation,
  Residual_Correlation = residual_correlation,
  Observed_Correlation_Gillnet = observed_correlation_gillnet,
  Predicted_Correlation_Gillnet = predicted_correlation_gillnet,
  Residual_Correlation_Gillnet = residual_correlation_gillnet
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
  #only values in the lower triangle of the matrix are kept
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
    top_corr$species2_negative[i] <- colnames(cor)[neg_match[1,"col"]]
    top_corr$negative_cor[i] <- min_n[i]
  }
  return(top_corr)
}


#inspect the highest correlations
get_top_correlations(n= 15, cor_matrix=residual_correlation_gillnet)
get_top_correlations(n= 30, cor_matrix=predicted_correlation_gillnet)

strong_residual_corr <- get_top_correlations(n= 15, cor_matrix=residual_correlation_gillnet)
write.csv(strong_residual_corr, "results/community_analysis/strongest_residual_correlations.csv")

#species with strong interaction or high overall residual correlation
overall_residual_correlation <- data.frame(
  mean_absolute_residual_correlation = apply(residual_correlation_gillnet, MARGIN=1, FUN= function(s){mean(abs(s[s!=1]))}),
  mean_positive_residual_correlation = apply(residual_correlation_gillnet, MARGIN=1, FUN= function(s){mean(s[s>0&s!=1])}),
  mean_negative_residual_correlation = apply(residual_correlation_gillnet, MARGIN=1, FUN= function(s){mean(s[s<0])}),
  mean_residual_correlation = apply(residual_correlation_gillnet, MARGIN=1, FUN= function(s){mean(s)})
)%>%
  dplyr::arrange(desc(mean_absolute_residual_correlation))
colMeans(overall_residual_correlation)
write.csv(overall_residual_correlation, "results/community_analysis/average_residual_correlations.csv")


#Interesting residual relationship
#counteracting co-occurrence -/+
minus_plus_rel <- ifelse(predicted_correlation_gillnet <0 & residual_correlation_gillnet>0,residual_correlation_gillnet,0)
get_top_correlations(n=10, cor_matrix = minus_plus_rel)

get_species_pair_correlations("Coregonus_albula", "Clupea_harengus",corr_matrix_list)
get_species_pair_correlations("Coregonus_maraena", "Sprattus_sprattus",corr_matrix_list)
get_species_pair_correlations("Coregonus_albula", "Sprattus_sprattus",corr_matrix_list)

get_species_pair_correlations("Neogobius_melanostomus", "Gasterosteus_aculeatus",corr_matrix_list)

#Counteracting residual segregation +/-
get_species_pair_correlations("Sander_lucioperca", "Neogobius_melanostomus", corr_matrix_list)
get_species_pair_correlations("Sander_lucioperca", "Alburnus_alburnus", corr_matrix_list)
get_species_pair_correlations("Sander_lucioperca", "Osmerus_eperlanus", corr_matrix_list)
get_species_pair_correlations("Gadus_morhua", "Gobius_niger", corr_matrix_list)
get_species_pair_correlations("Gadus_morhua", "Neogobius_melanostomus", corr_matrix_list)
get_species_pair_correlations("Gymnocephalus_cernua", "Alburnus_alburnus", corr_matrix_list)


get_species_pair_correlations("Scardinius_erythrophthalmus","Gadus_morhua", corr_matrix_list)
get_species_pair_correlations("Coregonus_maraena","Platichthys_flesus", corr_matrix_list)
get_species_pair_correlations("Coregonus_maraena","Platichthys_flesus", corr_matrix_list)


################################################################################
#load and inspect prediction maps
library(terra)
mvp_map <- rast("results/prediction_maps/prediction_map_mvp.tif")
mvp_map_binary <- rast("results/prediction_maps/prediction_map_mvp_binary.tif")
binomial_map <- rast("results/prediction_maps/prediction_map_binomial.tif")
binomial_map_binary <- rast("results/prediction_maps/prediction_map_binomial_binary.tif")


#Species maps
species <- "Sander_lucioperca"

par(mfrow = c(2, 2), oma = c(0, 0, 3, 0), mar = c(2, 2, 3, 2))
plot(binomial_map[[species]])
mtext("Binomial probabilities", side = 3, line = 1, cex = 0.9)
plot(binomial_map_binary[[species]])
mtext("Binomial presence/absence", side = 3, line = 1, cex = 0.9)
plot(mvp_map[[species]])
mtext("MVP probabilities", side = 3, line = 1, cex = 0.9)
plot(mvp_map_binary[[species]])
mtext("MVP presence/absence", side = 3, line = 1, cex = 0.9)
mtext(text = gsub("_"," ", species), outer = TRUE, line = 0.5, cex = 1.5, font = 2)

# Biodiversity map




################################################################################
# Feature importance
features_binomial <- readRDS("results/final_trained_models/summary_binomial.rds")
features_mvp <- readRDS("results/final_trained_models/summary_mvp.rds")


feature_importance <- as.data.frame(features_binomial$importance)%>%
  select(-variable)
colnames(feature_importance) <- gsub("_", " ",species_names)
rownames(feature_importance) <- gsub("_", " ",predictor_names)

#check
summary(feature_importance)

summary(rowMeans(feature_importance))
hist(rowMeans(feature_importance))

#construct a feature importance matrix sorted after predictor importance for plotting
feature_importance_long <- feature_importance %>%
  sort_by(rowSums(feature_importance))%>%
  as.matrix()%>%
  melt()

#create a plot highlighting predictor importance
predictor_importance <- ggplot(feature_importance_long, aes(x = Var2, y = Var1, fill = log(value))) +
  geom_tile(color = "white", linewidth = 0.5) +
  scale_fill_gradientn(
    colors = c("white", "darkmagenta"),
    values = scales::rescale(c(0, 
                               log(max(feature_importance$value)))),
    limits = c(0,log(max(feature_importance$value))),
    name = "log(feature importance)"
  )+
  geom_text(aes(label = ifelse(value >= 1.1, sprintf("%.1f", value), "")),
            size = 2.8, color = ifelse(feature_importance$value>1.8,"white","black"), fontface = "bold") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, size = 9),
        axis.text.y = element_text(size = 9),
        axis.title = element_blank(),
        #legend.position = "right",
        panel.grid = element_blank())+
  coord_fixed()+
  labs(title = "Predictor importance (Binomial Model)")

# Save plot
ggsave("results/plots/predictor_importance_binomial.tiff",
       predictor_importance,
       width = 8, height = 6, dpi = 300, compression = "lzw")

################################################################################








m <- distr_multivariate_normal(torch_zeros(25), torch_tensor(residual_correlation))
rowsum(m$sample(100)) # normally distributed with mean=`[0,0]` and covariance_matrix=`I`

##################################################################################
get_species_pair_correlations("Clupea_harengus", "Gasterosteus_aculeatus", corr_matrix_list)
get_species_pair_correlations("Clupea_harengus", "Sander_lucioperca", corr_matrix_list)
get_species_pair_correlations("Clupea_harengus", "Perca_fluviatilis", corr_matrix_list)
get_species_pair_correlations("Clupea_harengus", "Gadus_morhua", corr_matrix_list)
get_species_pair_correlations("Clupea_harengus", "Esox_lucius", corr_matrix_list)
get_species_pair_correlations("Clupea_harengus", "Rutilus_rutilus", corr_matrix_list)
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

get_species_pair_correlations("Gadus_morhua", "Blicca_bjoerkna", corr_matrix_list)
get_species_pair_correlations("Perca_fluviatilis", "Blicca_bjoerkna", corr_matrix_list)
