// neuromorphic_computing.cpp
#include <iostream>
#include <vector>
#include <cmath>

class Neuron {
public:
    Neuron(int num_inputs) {
        weights.resize(num_inputs);
        for (int i = 0; i < num_inputs; i++) {
            weights[i] = (double)rand() / RAND_MAX;
        }
    }

    double computeOutput(std::vector<double> inputs) {
        double sum = 0;
        for (int i = 0; i < inputs.size(); i++) {
            sum += inputs[i] * weights[i];
        }
        return sigmoid(sum);
    }

private:
    std::vector<double> weights;

    double sigmoid(double x) {
        return 1 / (1 + exp(-x));
    }
};

int main() {
    Neuron neuron(3);
    std::vector<double> inputs = {1, 2, 3};
    double output = neuron.computeOutput(inputs);
    std::cout << "Neuron output: " << output << std::endl;
    return 0;
}
