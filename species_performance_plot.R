################################################################################
#       Species-wise AUC and TSS comparison: Binomial vs MVP ANN              #
################################################################################
library(ggplot2)
source("Functions.R")

#Load final test results
bin <- read.csv("results/final_trained_models/binomial_testset_evaluation.csv", row.names = 1)
mvp <- read.csv("results/final_trained_models/mvp_testset_evaluation.csv", row.names = 1)

# Load cross-validation performance uncertainty
auc_cv_bin <- read.csv("results/final_trained_models/AUC_bin_performance_uncertainty_crossvalidation.csv", row.names = 1)
tss_cv_bin <- read.csv("results/final_trained_models/TSS_bin_performance_uncertainty_crossvalidation.csv", row.names = 1)
auc_cv_mvp <- read.csv("results/final_trained_models/AUC_mvp_performance_uncertainty_crossvalidation.csv", row.names = 1)
tss_cv_mvp <- read.csv("results/final_trained_models/TSS_mvp_performance_uncertainty_crossvalidation.csv", row.names = 1)

#mean AUC and mean TSS
mean_AUC <- mean(cbind(as.numeric(bin["AUC_species",]), as.numeric(mvp["AUC_species",])))
mean_TSS <- mean(cbind(as.numeric(bin["TSS_species",]), as.numeric(mvp["TSS_species",])))


species_names <- colnames(bin)
species_labels <- gsub("_", " ", species_names)

prevalence <- colMeans(full_response[, species_names])

################################################################################
#Plot 1: Test set performance
################################################################################
# Build long-format data
df <- data.frame(
  species = rep(species_labels, 4),
  metric  = rep(c("AUC", "AUC", "TSS", "TSS"), each = 25),
  model   = rep(c("Binomial", "MVP", "Binomial", "MVP"), each = 25),
  value   = c(as.numeric(bin["AUC_species", ]),
              as.numeric(mvp["AUC_species", ]),
              as.numeric(bin["TSS_species", ]),
              as.numeric(mvp["TSS_species", ])),
  prevalence = rep(prevalence, 4)
)

# Order species by mean AUC (across both models)
mean_auc <- tapply(df$value[df$metric == "AUC"], df$species[df$metric == "AUC"], mean)
sp_order <- names(sort(mean_auc))
df$species <- factor(df$species, levels = sp_order)

# Prevalence labels for the y-axis
prev_for_label <- prevalence[match(gsub(" ", "_", sp_order), species_names)]
y_labels <- sprintf("%s  (%.0f%%)", sp_order, prev_for_label * 100)

# Paired dot plot with connecting segments
# Reshape to wide per species+metric so we can draw segments
df_wide <- merge(
  df[df$model == "Binomial", c("species", "metric", "value")],
  df[df$model == "MVP", c("species", "metric", "value")],
  by = c("species", "metric"),
  suffixes = c("_bin", "_mvp")
)

p <- ggplot(df, aes(x = value, y = species)) +
  # Connecting segment between the two models
  geom_segment(data = df_wide,
               aes(x = value_bin, xend = value_mvp,
                   y = species, yend = species),
               color = "grey70", linewidth = 0.4) +
  # Points
  geom_point(aes(color = model, shape = model), size = 2.8) +
  # Facet by metric
  facet_wrap(~ metric, scales = "free_x", ncol = 2) +
  # Color
  scale_color_manual(values = c("Binomial" = "#2166ac", "MVP" = "#b2182b"),
                     name = "Model") +
  scale_shape_manual(values = c("Binomial" = 16, "MVP" = 17),
                     name = "Model") +
  scale_y_discrete(labels = y_labels) +
  # Reference lines
  geom_vline(data = data.frame(metric = c("AUC", "TSS"), ref = c(mean_AUC, mean_TSS)),
             aes(xintercept = ref), linetype = "dashed", color = "grey50", linewidth = 0.3) +
  # Theme
  theme_minimal(base_size = 11) +
  theme(
    axis.text.y = element_text(size = 8),
    axis.title.y = element_blank(),
    strip.text = element_text(size = 12, face = "bold"),
    legend.position = "top",
    panel.grid.major.y = element_line(color = "grey92"),
    panel.grid.minor = element_blank(),
    plot.title = element_text(size = 13, face = "bold"),
    plot.subtitle = element_text(size = 9, color = "grey40")
  ) +
  labs(
    x = "Score",
    title = "Test Set Performance by Species",
    subtitle = "Species ordered by mean AUC; prevalence in parentheses. Dashed lines: mean AUC, mean TSS"
  )

ggsave("results/plots/species_performance_comparison.png",
       p, width = 12, height = 8, dpi = 300, type = "cairo")


################################################################################
# Plot 2: Cross-validation performance uncertainty
################################################################################

# Build long-format data for CV results
df_cv <- data.frame(
  species = rep(species_labels, 4),
  metric  = rep(c("AUC", "AUC", "TSS", "TSS"), each = 25),
  model   = rep(c("Binomial", "MVP", "Binomial", "MVP"), each = 25),
  cv_mean = c(as.numeric(auc_cv_bin[, "mean"]),
              as.numeric(auc_cv_mvp[, "mean"]),
              as.numeric(tss_cv_bin[, "mean"]),
              as.numeric(tss_cv_mvp[, "mean"])),
  cv_sd   = c(as.numeric(auc_cv_bin[, "sd"]),
              as.numeric(auc_cv_mvp[, "sd"]),
              as.numeric(tss_cv_bin[, "sd"]),
              as.numeric(tss_cv_mvp[, "sd"]))
)
df_cv$species <- factor(df_cv$species, levels = sp_order)

# Mean CV AUC and TSS across both models
mean_cv_AUC <- mean(df_cv$cv_mean[df_cv$metric == "AUC"])
mean_cv_TSS <- mean(df_cv$cv_mean[df_cv$metric == "TSS"])

dodge_cv <- position_dodge(width = 0.4)

p_cv <- ggplot(df_cv, aes(x = cv_mean, y = species)) +
  # Error bars: CV mean +/- 1 SD
  geom_errorbar(aes(xmin = cv_mean - cv_sd, xmax = cv_mean + cv_sd, color = model),
                width = 0.3, linewidth = 0.35,
                orientation = "y",
                position = dodge_cv) +
  # CV mean points
  geom_point(aes(color = model, shape = model), size = 2.5,
             position = dodge_cv) +
  facet_wrap(~ metric, scales = "free_x", ncol = 2) +
  scale_color_manual(values = c("Binomial" = "#2166ac", "MVP" = "#b2182b"),
                     name = "Model") +
  scale_shape_manual(values = c("Binomial" = 16, "MVP" = 17),
                     name = "Model") +
  scale_y_discrete(labels = y_labels) +
  geom_vline(data = data.frame(metric = c("AUC", "TSS"), ref = c(mean_cv_AUC, mean_cv_TSS)),
             aes(xintercept = ref), linetype = "dashed", color = "grey50", linewidth = 0.3) +
  theme_minimal(base_size = 11) +
  theme(
    axis.text.y = element_text(size = 8),
    axis.title.y = element_blank(),
    strip.text = element_text(size = 12, face = "bold"),
    legend.position = "top",
    panel.grid.major.y = element_line(color = "grey92"),
    panel.grid.minor = element_blank(),
    plot.title = element_text(size = 13, face = "bold"),
    plot.subtitle = element_text(size = 9, color = "grey40")
  ) +
  labs(
    x = "Score",
    title = "Cross-Validation Performance Uncertainty by Species",
    subtitle = "Mean +/- 1 SD from 5-fold grouped CV on training data. Species ordered by mean test AUC"
  )

ggsave("results/plots/cv_performance_uncertainty.png",
       p_cv, width = 12, height = 8, dpi = 300, type = "cairo")

