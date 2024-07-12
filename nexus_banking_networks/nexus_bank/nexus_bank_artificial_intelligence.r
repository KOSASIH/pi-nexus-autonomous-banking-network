library(neuralnet)

# Create a neural network with 2 inputs, 2 hidden layers, and 1 output
net <- neuralnet(Out ~ In1 + In2, data = data, hidden = c(2, 2))

# Train the network using backpropagation
net <- train(net, data)

# Make a prediction using the trained network
prediction <- predict(net, newdata = data.frame(In1 = 1, In2 = 2))
print(prediction)
