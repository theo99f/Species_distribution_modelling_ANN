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

