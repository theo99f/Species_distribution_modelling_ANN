train_set_response_binary <- read.csv()

library(sjSDM)
install_sjSDM()
model_joint <- sjSDM(Y=train_set_response_binary,
                     env = DNN(data = train_set_predictors,
                               formula = ~.,
                               hidden = c(30L,30L),
                               activation = "relu"))
