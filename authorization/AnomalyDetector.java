// AnomalyDetector.java
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class AnomalyDetector {
    private MultiLayerNetwork network;

    public AnomalyDetector() {
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
               .seed(42)
               .weightInit(WeightInit.XAVIER)
               .updater(new Nesterovs(0.01))
               .list()
               .layer(new DenseLayer.Builder()
                       .nIn(10)
                       .nOut(20)
                       .activation(Activation.RELU)
                       .build())
               .layer(new DenseLayer.Builder()
                       .nIn(20)
                       .nOut(10)
                       .activation(Activation.RELU)
                       .build())
               .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                       .nIn(10)
                       .nOut(2)
                       .activation(Activation.SOFTMAX)
                       .build())
               .pretrain(false)
               .backprop(true)
               .build();

        network = new MultiLayerNetwork(conf);
        network.init();
    }

    public boolean detectAnomaly(double[] features) {
        // Use the trained network to detect anomalies
        INDArray input = Nd4j.create(features);
        INDArray output = network.output(input);
        double anomalyScore = output.getDouble(0);
        if (anomalyScore > 0.5) {
            return true;
        }
        return false;
    }
}
