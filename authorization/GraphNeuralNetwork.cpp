// GraphNeuralNetwork.cpp
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <boost/algorithm/string.hpp>
#include <boost/asio.hpp>
#include <dgl/graph.h>
#include <dgl/model.h>

class GraphNeuralNetwork {
public:
    GraphNeuralNetwork(std::string modelPath) : modelPath_(modelPath) {}

    void start() {
        // Initialize graph neural network model
        dgl::Graph graph = dgl::Graph::Create();
        dgl::Model model = dgl::Model::Create(modelPath_);
        model.set_graph(graph);

        // Process incoming streaming data
        while (true) {
            std::string message = receiveMessage();
            std::vector<std::string> fields;
            boost::algorithm::split(fields, message, boost::is_any_of(","));
            // Create graph data structure from streaming data
            dgl::Node nodes[] = {{fields[0], fields[1]}, {fields[2], fields[3]}};
            dgl::Edge edges[] = {{0, 1}};
            graph.add_nodes(nodes, 2);
            graph.add_edges(edges, 1);

            // Run graph neural network model
            dgl::Tensor output = model.forward(graph);
            // Detect fraud using graph neural network output
            if (output[0] > 0.5) {
                std::cout << "Fraud detected" << std::endl;
            }
        }
    }

private:
    std::string receiveMessage() {
        // Receive streaming data from Kafka topic or file
        // Return the received message
    }

    std::string modelPath_;
};
