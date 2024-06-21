library(caret)
library(randomForest)

# Load the data
data <- read.csv("risk_data.csv")

# Create a random forest model
model <- randomForest(risk_level ~., data = data)

# Train the model
trainControl <- trainControl(method = "cv", number = 10)
model <- train(risk_level ~., data = data, method = "rf", trControl = trainControl)

# Use the trained model to predict risk
new_data <- data.frame(amount = 1000, country = "USA", card_type = "VISA")
prediction <- predict(model, new_data)
print(prediction)
