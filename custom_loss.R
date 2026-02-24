
modelsdkjf <- dnn(X=train_set_predictors, 
                  Y=cbind(train_set_response_binary), 
                  hidden = c(30L,30L),
                  activation = "relu",
                  loss = custom_loss_latent,
                  custom_parameters = list(Lambda = nn_parameter(torch_randn(5,2)*0.1)),
                  validation = 0.1,
                  early_stopping = 20,
                  epochs = 25,
                  lr = 0.05,
                  lr_scheduler = config_lr_scheduler("reduce_on_plateau", patience = 5, factor = 0.5))
evaluate_model(modelsdkjf)


models <- dnn(X=train_set_predictors, 
              Y=cbind(train_set_response_binary), 
              hidden = c(30L,30L),
              activation = "relu",
              loss = custom_loss_latent,
              custom_parameters = list(SigmaDiag =  rep(0, length(species_names)), # Our parameters with starting values
                                       SigmaPar = matrix(rnorm(length(species_names)* (length(species_names)-1), sd = 0.001), length(species_names), length(species_names)-1)),
              validation = 0.1,
              early_stopping = 20,
              epochs = 500,
              lr = 0.005,
              lr_scheduler = config_lr_scheduler("reduce_on_plateau", patience = 5, factor = 0.5))

model_custom <- dnn(X=train_set_predictors, 
                    Y=cbind(train_set_response_binary), 
                    hidden = c(30L,30L),
                    activation = "relu",
                    loss = custom_binary_safe,
                    validation = 0.1,
                    early_stopping = 20,
                    epochs = 500,
                    lr = 0.06,
                    lr_scheduler = config_lr_scheduler("reduce_on_plateau", patience = 5, factor = 0.5))
custom_binary_safe <- function(true, pred) {
  
  pred <- torch_clamp(pred, -10, 10)
  
  bce <- torch::nnf_binary_cross_entropy_with_logits(
    input = pred,
    target = true,
    reduction = "mean"
  )
  
  penalty <- 1e-4 * torch_mean(pred^2)
  
  return(bce + penalty)
}

custom_binary <- function(true,pred){
  pred <- torch_clamp(pred, min = -10, max = 10)
  
  p <- torch_sigmoid(pred)
  loss <- -(true * torch_log(p) + (1 - true) * torch_log(1 - p))
  return(torch_mean(loss))
}
custom_binary <- function(true, pred) {
  pred <- torch_clamp(pred, min = -10, max = 10)
  
  loss <- torch::nnf_binary_cross_entropy_with_logits(
    input = pred,
    target = true,
    reduction = "mean"
  )
  return(loss)
}

custom_loss_MVP_lowrank <- function(true, pred) {
  
  N <- pred$size(1)  # number of observations in batch
  J <- pred$size(2)  # number of species
  M <- 50            # MC draws per observation
  K <- ncol(SigmaPar) # latent factor dimension (L is J x K)
  
  loglik_mc <- vector("list", M)
  
  for (m in 1:M) {
    
    # --- Step 1: sample latent factors z_i ~ N(0, I_K) ---
    # shape: N x K
    z <- torch_randn(c(N, K))
    
    # --- Step 2: project latent factors to species space ---
    # L is J x K, so for batch we do z %*% t(L)
    # shape: N x J
    xi_latent <- torch_matmul(z, SigmaPar$t())
    
    # --- Step 3: add independent residuals epsilon_i ~ N(0, D) ---
    # shape: N x J
    epsilon <- torch_randn(c(N, J)) * SigmaDiag$exp().sqrt()
    
    # --- Step 4: linear predictor with latent effect ---
    eta <- pred + xi_latent + epsilon
    
    # --- Step 5: sigmoid to get Bernoulli probabilities ---
    p <- torch_sigmoid(eta)
    
    # --- Step 6: log-likelihood for Bernoulli ---
    # shape: N x J -> sum over species
    loglik_mc[[m]] <- (true * torch_log(p) + (1 - true) * torch_log(1 - p))$sum(dim = 2)
  }
  
  # --- Step 7: Monte Carlo marginalization ---
  # shape: N
  loglik <- torch_logsumexp(torch_stack(loglik_mc), dim = 1) - log(M)
  
  # Return negative mean log-likelihood
  return(-loglik$mean())
}




t<- torch_tensor(c(0,0,0,0,0))
pr <-  torch_tensor(c(0,0,0,0,0))
K=2
J=5
#Lambda is a loading matrix with J species x K latent factors
Lambda <- nn_parameter(torch_randn(J,K)*0.1)

custom_loss_latent = function(true, pred) {
  
  #pred BxJ matrix
  #pred[1,]
  # Get batch size
  B <- true$size(1)
  #zi <- torch_tensor(matrix(0.1, nrow = B, ncol= K))
  zi <- torch_randn(B,K)*0.001
  #multiply 
  latent <- torch_mm(zi, Lambda$t()) #output is BxJ
  latent <- torch_clamp(latent, -3, 3)
  p <- torch_clamp(pred + latent,-8,8)
  p <- torch_sigmoid(p)
  p <- torch_clamp(p, min = 1e-6, max = 1 - 1e-6)  
  bce <- -(true * torch_log(p) + (1-true)* torch_log(1-p))
  lambda_reg <- 0.01 * torch_sum(Lambda^2)
  zi_reg <- 0.05 * torch_sum(zi^2)
  return(torch_mean(bce)+ 0.1 * torch_sum(Lambda^2))
}


# Change your learning rate from 0.06 to 0.005 (12x lower!)
# Add these three critical fixes:

Lambda <- nn_parameter(torch_randn(J, K) * 0.01)  # Smaller init

custom_loss_latent <- function(true, pred) {
  B <- true$size(1)
  
  zi <- ?torch_randn(B, K) / sqrt(K)  # FIX 1: Scale down zi BxK
  latent <- ?torch_mm(zi, Lambda$t()) # 
  # FIX 2: Clip latent
  
  logits <- torch_clamp(pred + latent, -8, 8)  # FIX 3: Clip logits
  p <- torch_sigmoid(logits)
  p <- torch_clamp(p, 1e-7, 1 - 1e-7)
  
  bce <- -(true * torch_log(p) + (1 - true) * torch_log(1 - p))
  
  # FIX 4: Add strong regularization
  return(torch_mean(bce) + 0.1 * torch_sum(Lambda^2))
}

# FIX 5: Lower learning rate!
model <- dnn(..., lr = 0.005)  # Not 0.06! 



custom_parameters =
  list(SigmaDiag =  rep(0, 3), # Our parameters with starting values
       SigmaPar = matrix(rnorm(6, sd = 0.001), 3, 2)) # Our parameters with starting values
)

SigmaPar <-torch_tensor(SigmaPar)
SigmaDiag <-torch_tensor(SigmaDiag)
SigmaPar$t()




custom_parameters =
  list(SigmaDiag =  rep(0, length(species_names)), # Our parameters with starting values
       SigmaPar = matrix(rnorm(length(species_names)* (length(species_names)-1), sd = 0.001), length(species_names), length(species_names)-1)) # Our parameters with starting values


torch::torch_matmul(SigmaPar, SigmaPar$t()) 

torch::torch_diag(SigmaDiag$exp()+0.001)


#custom loss function
create_cov = function(L, Diag) {
  return(torch::torch_matmul(L, L$t()) + torch::torch_diag(Diag$exp()+0.001))
}
#multiply



print(paste("true",true))
M <- 20
Sigma = create_cov(SigmaPar, SigmaDiag)
s <- torch_tensor(numeric(nrow(SigmaPar)))

loss <- c()
for (m in 1:M) {
  e <- torch::distr_multivariate_normal(s, covariance_matrix = Sigma)
  pred_r <-  pred+ e$sample() 
  p <- torch_sigmoid(pred_r)
  loss <- c(loss,(true * log(p) + (1-true)* log(1-p)))}

loglik = ?torch_logsumexp(torch_stack(loss), dim = 1) - log(M)


return(loglik$mean())
}



custom_loss_MVP = function(true, pred) {
  
  N <- pred$size(1)
  J <- pred$size(2)
  M <- 200
  
  Sigma <- create_cov(SigmaPar, SigmaDiag)
  
  loglik_mc <- vector("list", M)
  
  for (m in 1:M) {
    
    # sample N latent vectors of dimension J
    xi <- torch::distr_multivariate_normal(
      loc = torch_zeros(J),
      covariance_matrix = Sigma
    )$sample(c(N))   # shape: (N × J)
    
    eta <- pred + xi
    p <- torch_sigmoid(eta)
    
    # sum over species
    loglik_mc[[m]] <-
      (true * torch_log(p) +
         (1 - true) * torch_log(1 - p))$sum(dim = 2)
  }
  
  # MC marginalization
  loglik <- torch_logsumexp(torch_stack(loglik_mc), dim = 1) - log(M)
  
  return(-loglik$mean())
}



custom_loss_MVP(torch_tensor(c(0,1,1,0,1)),torch_tensor(c(0.1,0.3,0.4,0.9,0.7)))

model_custom <- dnn(X=train_set_predictors, 
                    Y=cbind(train_set_response_binary), 
                    hidden = c(30L,30L),
                    activation = "relu",
                    loss = custom_loss_MVP,
                    validation = 0.1,
                    early_stopping = 20,
                    epochs = 500,
                    lr = 0.01,
                    custom_parameters = custom_parameters,
                    lr_scheduler = config_lr_scheduler("reduce_on_plateau", patience = 5, factor = 0.5))



