#include <iostream>
#include <vector>
#include <neuron/neuron.h>

class NeuromorphicNetwork {
public:
    NeuromorphicNetwork(int num_neurons, int num_synapses) {
        neurons_.resize(num_neurons);
        synapses_.resize(num_synapses);
    }

    void add_neuron(Neuron* neuron) {
        neurons_.push_back(neuron);
    }

    void add_synapse(Synapse* synapse) {
        synapses_.push_back(synapse);
    }

    void simulate() {
        // Simulate the neuromorphic network
        for (auto& neuron : neurons_) {
            neuron->update();
        }
        for (auto& synapse : synapses_) {
            synapse->update();
        }
    }

private:
    std::vector<Neuron*> neurons_;
    std::vector<Synapse*> synapses_;
};

int main() {
    NeuromorphicNetwork network(10, 20);
    // Create neurons and synapses
    Neuron* neuron1 = new Neuron();
    Neuron* neuron2 = new Neuron();
    Synapse* synapse = new Synapse(neuron1, neuron2);
    network.add_neuron(neuron1);
    network.add_neuron(neuron2);
    network.add_synapse(synapse);
    network.simulate();
    return 0;
}
