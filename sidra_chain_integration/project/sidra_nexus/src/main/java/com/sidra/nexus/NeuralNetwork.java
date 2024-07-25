package com.sidra.nexus;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;

public class NeuralNetwork {
    private MultilayerPerceptron neuralNetwork;

    public NeuralNetwork(Instances trainingData) throws Exception {
        neuralNetwork = new MultilayerPerceptron();
        neuralNetwork.buildClassifier(trainingData);
    }

    public double classifyInstance(Instances testData) throws Exception {
        return neuralNetwork.classifyInstance(testData.instance(0));
    }

    public Evaluation evaluate(Instances testData) throws Exception {
        Evaluation evaluation = new Evaluation(neuralNetwork);
        evaluation.evaluateModel(neuralNetwork, testData);
        return evaluation;
    }
}
