package com.sidra;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class NLP {
    public static void main(String[] args) {
        // Set up a neural network for NLP
        NeuralNetConfiguration config = new NeuralNetConfiguration.Builder()
              .seed(42)
              .weightInit(WeightInit.XAVIER)
              .updater(new Nesterovs(0.01))
              .list()
              .layer(new LSTM.Builder()
                      .nIn(100)
                      .nOut(100)
                      .activation(Activation.TANH)
                      .build())
              .layer(new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                      .activation(Activation.SOFTMAX)
                      .nIn(100)
                      .nOut(10)
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
