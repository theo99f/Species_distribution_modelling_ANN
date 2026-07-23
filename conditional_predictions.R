################################################################################
# Joint community predictions
################################################################################
source("Functions.R")

mvp_ann <- readRDS("results/final_trained_models/mvp_ann")

#Get a random location where they used gillnets
#Gillnet indices
indices <- which(full_response$method == "Gillnet")



#59.3878° N latitude and 18.7358° E longitude (Stockholm Archipelago)
indices <- which(between(full_response$latitude, 57.5,60) & between(full_response$longitude, 18,19.5))
# sample one random index
selected_index <- sample(indices, 1)
# Retrieve the full row using the selected index
sampled_row <- full_response[selected_index, ]   
sampled_row
environmental_conditions <- train_set_predictors[selected_index, ]  
#59.17564  18.76873 idx 25607

#individual probabilities
predict_mvp_manual(train_set_predictors[25607, ], 
                   mvp_ann, 
                   species_names = species_names)

#joint conditional probabilities
predict_jointly(train_set_predictors[25607, ], 
                model = mvp_ann, 
                composition=list("Perca_fluviatilis"=1
                                 ,
                                 "Gasterosteus_aculeatus"=0
                ),
                species_names = species_names)
#
get_logits <- function(X, model, species_names) {
  X <- as.matrix(X)

  # Get network weights from saved model
  W1 <- model$weights[[2]]$`0.weight`
  b1 <- model$weights[[2]]$`0.bias`
  alpha1 <- model$weights[[2]]$`1.weight`
  
  W2 <- model$weights[[2]]$`2.weight`
  b2 <- model$weights[[2]]$`2.bias`
  alpha2 <- model$weights[[2]]$`3.weight`
  
  W3 <- model$weights[[2]]$`4.weight`
  b3 <- model$weights[[2]]$`4.bias`
  
  # Forward pass
  z1 <- X %*% t(W1) + matrix(b1, nrow(X), length(b1), byrow = TRUE)
  h1 <- ifelse(z1 > 0, z1, alpha1 * z1)
  
  z2 <- h1 %*% t(W2) + matrix(b2, nrow(h1), length(b2), byrow = TRUE)
  h2 <- ifelse(z2 > 0, z2, alpha2 * z2)
  
  #network output (environmental suitability)
  logits <- h2 %*% t(W3) + matrix(b3, nrow(h2), length(b3), byrow = TRUE)
  return(as.numeric(logits))
}
get_realization <- function(logits, covariance){
  draw <- distr_multivariate_normal(torch_tensor(logits), torch_tensor(covariance))
  return(as.numeric(draw$sample(1)))
}

#create the covariance matrix
M <- mvp_ann$parameter$parameter
covariance <- M%*%t(M) + diag(1, length(species_names))


plot_logits <- function(logits, species) {
  stopifnot(length(logits) == length(species))
  
  # order for nicer display (optional: by value)
  ord <- order(logits)
  logits <- logits[ord]
  species <- species[ord]
  
  y <- seq_along(species)
  
  plot(
    x = c(-10, 10),
    y = c(1, length(species)),
    type = "n",
    yaxt = "n",
    xlim = c(-10, 10),
    xlab = "Environmental suitability",
    ylab = ""
  )
  
  axis(2, at = y, labels = species, las = 1)
  abline(v = 0, lty = 2, col = "gray50")  # reference at zero log-odds
  
  segments(
    x0 = 0,
    y0 = y,
    x1 = logits,
    y1 = y,
    lwd = 2,
    col = "steelblue"
  )
  
  points(logits, y, pch = 19, col = "steelblue")
  return(ord)
}


plot_illustration <- function(logits, covariance, old_realizations,
                              species_names) {
  
  ord <- plot_logits(logits, species_names)
  
  # new realization
  x <- get_realization(logits, covariance)
  
  # plot previous realizations
  if (nrow(old_realizations) > 0) {
    for (row in 1:nrow(old_realizations)) {
      points(old_realizations[row, ord],
             seq_along(ord),
             pch = 19,
             col = "grey80")
    }
  }
  points(logits[ord],
         seq_along(ord),
         pch = 19,
         col = "steelblue")
  # plot current realization
  points(x[ord],
         seq_along(ord),
         pch = 19,
         col = "darkred")
  
  # store UNORDERED realization
  old_realizations <- rbind(old_realizations, x)
  
  return(old_realizations)
}

old_realizations <- matrix(numeric(0),
                           nrow = 0,
                           ncol = length(species_names))
plot.new()

logits <- get_logits(train_set_predictors[25607, ], 
                     model = mvp_ann, 
                     species_names = species_names)

get_realization(logits,covariance)

par(mar = c(5, 12, 4, 2) + 0.1)
plot_logits(logits, species_names)
old_realizations <- plot_illustration(
  logits,
  covariance,
  old_realizations,
  species_names
)



#ILLUSTRATIVE EXAMPLE: 63.53683  19.85183 idx 12230

#individual probabilities
predict_mvp_manual(train_set_predictors[12230, ], 
                   mvp_ann, 
                   species_names = species_names)

#joint conditional probabilities
predict_jointly(train_set_predictors[12230, ], 
                model = mvp_ann, 
                composition=list("Perca_fluviatilis"=1
                                 #"Gasterosteus_aculeatus"=1
                                 ),
                species_names = species_names)

#joint conditional probabilities
predict_jointly(train_set_predictors[12230, ], 
                model = mvp_ann, 
                composition=list(#"Perca_fluviatilis"=0
                                 "Gasterosteus_aculeatus"=1
                ),
                species_names = species_names)


predict_jointly(train_set_predictors[12230, ], 
                mvp_ann, 
                composition=list("Perca_fluviatilis"=1,
                                 "Rutilus_rutilus"=0,
                                 "Gasterosteus_aculeatus"=1),
                species_names = species_names)







#11403
predict_jointly(environmental_conditions, 
                mvp_ann, 
                composition=list("Sander_lucioperca"=1,
                                 "Rutilus_rutilus"=0,
                                 "Perca_fluviatilis"=1),
                
                species_names = species_names)


