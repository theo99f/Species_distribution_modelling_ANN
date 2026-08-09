################################################################################
# Dataset exploration
################################################################################

fish_data <- read.csv("data/raw_data/final_raw_dataset/fish_observations_wide.csv")


#3895 unique sampling dates
length(unique(fish_data$date))

#extracting the sampling (fishing) years
years <- as.numeric(sub("^([0-9]{4}).*","\\1", fish_data$date))

hist(years)
summary(years)

folder <- "data/preprocessed_data/final_data_split/"
test_set_response <- read.csv(paste0(folder,"test_set/test_set_response.csv"))
train_set_response <- read.csv(paste0(folder,"development_set/train_set_response.csv"))

years_test_set <- as.numeric(sub("^([0-9]{4}).*","\\1", test_set_response$date))
hist(years_test_set)

#catching method
unique(fish_data$method)

sum(fish_data$method=="Benthic trawl")/nrow(fish_data)
sum(fish_data$method=="Gillnet")/nrow(fish_data)
sum(fish_data$method=="Pelagic trawl")/nrow(fish_data)
sum(fish_data$method=="Fyke net")/nrow(fish_data)

################################################################################
surface_salinity <- rast("data/raw_data/final_raw_dataset/predictors_fish/salinity_surface_mean.tif")
secchi_depth <- rast("data/raw_data/final_raw_dataset/predictors_fish/secchi_depth_mean.tif")
summary(surface_salinity)
sd(surface_salinity)

plot(surface_salinity)
plot(secchi_depth)
################################################################################
#checking predictor correlations in the data set
cor.test(train_set_predictors$depth, train_set_predictors$salinity_bottom_mean)
cor.test(train_set_predictors$depth, train_set_predictors$salinity_surface_mean)

cor.test(train_set_predictors$salinity_surface_mean, train_set_predictors$temperature_bottom_mean)
cor.test(train_set_predictors$salinity_bottom_mean, train_set_predictors$temperature_bottom_mean)

cor.test(train_set_predictors$salinity_surface_mean, train_set_predictors$salinity_bottom_mean)

cor.test(train_set_predictors$sea_ice_fraction_mean, train_set_predictors$salinity_bottom_mean)
cor.test(train_set_predictors$sea_ice_fraction_mean, train_set_predictors$salinity_surface_mean)
cor.test(train_set_predictors$sea_ice_fraction_mean, train_set_predictors$depth)

cor.test(train_set_predictors$sea_ice_fraction_mean, train_set_predictors$secchi_depth_mean)

cor.test(train_set_predictors$salinity_surface_mean,train_set_predictors$secchi_depth_mean)
cor.test(train_set_predictors$salinity_bottom_mean,train_set_predictors$secchi_depth_mean)

cor.test(train_set_predictors$sea_ice_fraction_mean, train_set_predictors$temperature_bottom_mean)
cor.test(train_set_predictors$depth, train_set_predictors$temperature_bottom_mean)
cor.test(train_set_predictors$salinity_surface_mean,train_set_predictors$oxygen_bottom_mean)
cor.test(train_set_predictors$salinity_bottom_mean,train_set_predictors$oxygen_bottom_mean)
cor.test(train_set_predictors$sea_ice_fraction_mean,train_set_predictors$oxygen_bottom_mean)
cor.test(train_set_predictors$depth,train_set_predictors$oxygen_bottom_mean)
cor.test(train_set_predictors$secchi_depth_mean,train_set_predictors$oxygen_bottom_mean)

cor.test(train_set_predictors$depth,train_set_predictors$wave_significant_height_mean)

cor.test(train_set_predictors$oxy,train_set_predictors$wave_significant_height_mean)

