package com.sidra.nexus;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;

public class MachineLearningModel {
    private MultilayerPerceptron neuralNetwork;

    public MachineLearningModel(Instances trainingData) throws Exception {
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
}
