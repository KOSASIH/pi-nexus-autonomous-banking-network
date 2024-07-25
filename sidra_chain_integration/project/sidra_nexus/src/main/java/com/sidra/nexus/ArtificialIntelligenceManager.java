package com.sidra.nexus;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;

public class ArtificialIntelligenceManager {
    private MultilayerPerceptron neuralNetwork;

    public ArtificialIntelligenceManager(Instances trainingData) throws Exception {
        neuralNetwork = new MultilayerPerceptron();
        neuralNetwork.buildClassifier(trainingData);
    }

    public double makePrediction(Instances testData) throws Exception {
        return neuralNetwork.classifyInstance(testData.instance(0));
    }

    public Evaluation evaluateModel(Instances testData) throws Exception {
        Evaluation evaluation = new Evaluation(neuralNetwork);
        evaluation.evaluateModel(neuralNetwork, testData);
        return evaluation;
    }

    public void trainModel(Instances trainingData) throws Exception {
        neuralNetwork.buildClassifier(trainingData);
    }

    public void saveModel(String filePath) throws Exception {
        // Save model to file
        // ...
    }

    public void loadModel(String filePath) throws Exception {
        // Load model from file
        // ...
    }
}
