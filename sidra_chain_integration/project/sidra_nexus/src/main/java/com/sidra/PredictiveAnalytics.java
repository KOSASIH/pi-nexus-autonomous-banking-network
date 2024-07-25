package com.sidra;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class PredictiveAnalytics {
    public static void main(String[] args) {
        // Set up a neural network
        NeuralNetConfiguration config = new NeuralNetConfiguration.Builder()
               .seed(42)
               .weightInit(WeightInit.XAVIER)
               .updater(new Nesterovs(0.01))
               .list()
               .layer(new DenseLayer.Builder()
                       .nIn(784)
                       .nOut(256)
                       .activation(Activation.RELU)
                       .build())
               .layer(new DenseLayer.Builder()
                       .nIn(256)
                       .nOut(10)
                       .activation(Activation.SOFTMAX)
                       .build())
               .pretrain(false)
               .backprop(true)
               .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

        // Train the model
        DataSetIterator iterator = new DataSetIterator();
        model.fit(iterator);
    }
                      }
