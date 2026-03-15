################################################################################
#                                Model Analysis                                #
#                                                                              #


binomial_ann <- readRDS("results/final_trained_models/binomial_ann")

analyze_training(binomial_ann)


#looking at the covariance matrix 
#check the covariance matrix of the joint species distribution model

## cov2cor() scales a covariance matrix by its diagonal
## to become the correlation matrix.

M <- mvp_ann$parameter$parameter
biotic_association = cov2cor(M%*%t(M) + diag(1, length(species_names)))
colnames(biotic_association) <- species_names
rownames(biotic_association) <- species_names

heatmap(biotic_association, Rowv = NA, Colv =NA)


melt(data)

ggp <- ggplot(data_melt, aes(X1, X2))+
  geom_tile(aes(fill=value))
ggp

binomial_ann$loss

M <- mvp_ann$parameter$parameter
Sigma <- M%*%t(M) + diag(1,25)

cor(predictions_dataset)

mvp_ann$net(as.matrix(train_set_predictors[1,]))
torch_sigmoid(mvp_ann$net$forward(as.matrix(train_set_predictors[1,]))*1.7012)

predict(mvp_ann, train_set_predictors[1,], type= "response")

m <- distr_multivariate_normal(torch_zeros(25), torch_tensor(biotic_association))
m$sample() # normally distributed with mean=`[0,0]` and covariance_matrix=`I`


