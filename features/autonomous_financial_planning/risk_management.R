# File name: risk_management.R
library(caret)
library(randomForest)

data <- read.csv("risk_data.csv")

model <- randomForest(risk ~ ., data = data)
predict(model, newdata = data)
