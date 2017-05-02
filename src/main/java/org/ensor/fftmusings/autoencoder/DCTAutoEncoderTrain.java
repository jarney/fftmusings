package org.ensor.fftmusings.autoencoder;

import java.io.PrintStream;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Arrays;
import java.util.Random;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;

/**
 * @author Adam Gibson
 */
public class DCTAutoEncoderTrain {

    public static void main(String[] args) throws Exception {
        final int numRows = 28;
        final int numColumns = 28;
        int seed = 123;
        int miniBatchSize = 3;
        int numSamples = 100;
        int batchSize = 10;
        int iterations = 1;
        int listenerFreq = iterations/5;

        Random rng = new Random(1234);

        PrintStream log = System.out;
        
        log.println("Load data....");
        DataSetIterator iter = new DCTDataIterator(rng,
            miniBatchSize,
            batchSize,
            numSamples,
            log);
                    
        log.println("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .learningRate(0.0001)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .list()
                .layer(0, new RBM.Builder().nIn(512).nOut(1000).lossFunction(LossFunctions.LossFunction.MSE).build())
                .layer(1, new RBM.Builder().nIn(1000).nOut(500).lossFunction(LossFunctions.LossFunction.MSE).build())
                .layer(2, new RBM.Builder().nIn(500).nOut(250).lossFunction(LossFunctions.LossFunction.MSE).build())
                .layer(3, new RBM.Builder().nIn(250).nOut(100).lossFunction(LossFunctions.LossFunction.MSE).build())
                .layer(4, new RBM.Builder().nIn(100).nOut(30).lossFunction(LossFunctions.LossFunction.MSE).build()) //encoding stops
                .layer(5, new RBM.Builder().nIn(30).nOut(100).lossFunction(LossFunctions.LossFunction.MSE).build()) //decoding starts
                .layer(6, new RBM.Builder().nIn(100).nOut(250).lossFunction(LossFunctions.LossFunction.MSE).build())
                .layer(7, new RBM.Builder().nIn(250).nOut(500).lossFunction(LossFunctions.LossFunction.MSE).build())
                .layer(8, new RBM.Builder().nIn(500).nOut(1000).lossFunction(LossFunctions.LossFunction.MSE).build())
                .layer(9, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nIn(1000).nOut(512).activation(Activation.IDENTITY).build())
                .pretrain(true).backprop(true)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(log)));

        log.println("Train model....");
        int epoch = 0;
        for (int i = 0; i < 1000; i++) {
            while(iter.hasNext()) {
                DataSet next = iter.next();
                model.fit(new DataSet(next.getFeatureMatrix(),next.getFeatureMatrix()));
            }
            iter.reset();
            ModelSerializer.writeModel(model, "data/aa/model-" + epoch + ".aa", true);
            epoch++;
        }
    }
}