#include <cmath>
#include <iostream>
#include <vector>

class NeuralNetwork {
public:
  NeuralNetwork(int inputs, int hidden, int outputs) {
    weights1 = new double *[inputs];
    for (int i = 0; i < inputs; i++) {
      weights1[i] = new double[hidden];
    }
    weights2 = new double *[hidden];
    for (int i = 0; i < hidden; i++) {
      weights2[i] = new double[outputs];
    }
  }

  ~NeuralNetwork() {
    for (int i = 0; i < inputs; i++) {
      delete[] weights1[i];
    }
    delete[] weights1;
    for (int i = 0; i < hidden; i++) {
      delete[] weights2[i];
    }
    delete[] weights2;
  }

  void train(std::vector<std::vector<double>> inputs,
             std::vector<std::vector<double>> outputs) {
    // Train the network using backpropagation
  }

  std::vector<double> predict(std::vector<double> input) {
    // Make a prediction using the trained network
  }

private:
  int inputs;
  int hidden;
  int outputs;
  double **weights1;
  double **weights2;
};

int main() {
  NeuralNetwork nn(784, 256, 10);
  std::vector<std::vector<double>> inputs = {{1, 2, 3, 4}, {5, 6, 7, 8}};
  std::vector<std::vector<double>> outputs = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
                                              {0, 0, 0, 0, 0, 0, 0, 0, 1, 0}};
  nn.train(inputs, outputs);
  std::vector<double> prediction = nn.predict({1, 2, 3, 4});
  std::cout << "Prediction: ";
  for (double d : prediction) {
    std::cout << d << " ";
  }
  std::cout << std::endl;
  return 0;
}
