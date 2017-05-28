/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.rnn;

import java.io.File;
import java.io.IOException;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.ensor.fftmusings.atrain.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 *
 * @author jona
 */
public class RNNFactory {
    private static int lstmLayerSize = 200;					//Number of units in each GravesLSTM layer
    // Above is Used to 'prime' the LSTM with a character sequence to continue/complete.
    // Initialization characters must all be in CharacterIterator.getMinimalCharacterSet() by default
    
    
    public static MultiLayerNetwork create(File modelFilename, CharacterIterator iter) throws IOException {

        if (modelFilename.exists()) {
            MultiLayerNetwork net = ModelSerializer.restoreMultiLayerNetwork(modelFilename);
            net.clear();
            net.setListeners(new ScoreIterationListener(System.out));
            return net;
        }
        
        
        int nOut = iter.totalOutcomes();

        //Set up network configuration:
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .learningRate(0.1)
                .rmsDecay(0.95)
                .seed(12345)
                .regularization(true)
                .l2(0.001)
                .list()
                .layer(0, new GravesLSTM.Builder().nIn(iter.inputColumns()).nOut(lstmLayerSize)
                                .updater(Updater.RMSPROP)
                                .activation(Activation.TANH)
                                .weightInit(WeightInit.DISTRIBUTION)
                                .dist(new UniformDistribution(-0.08, 0.08)).build())
                .layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
                                .updater(Updater.RMSPROP)
                                .activation(Activation.TANH)
                                .weightInit(WeightInit.DISTRIBUTION)
                                .dist(new UniformDistribution(-0.08, 0.08)).build())
                .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                .activation(Activation.SOFTMAX) //MCXENT + softmax for classification
                                .updater(Updater.RMSPROP)
                                .nIn(lstmLayerSize).nOut(nOut).weightInit(WeightInit.DISTRIBUTION)
                                .dist(new UniformDistribution(-0.08, 0.08)).build())
                .pretrain(false)
                .backprop(true)
                .backpropType(BackpropType.TruncatedBPTT)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(System.out));

        ModelSerializer.writeModel(net, modelFilename, true);

        return net;
    }
    
    public static void persist(File modelFilename, MultiLayerNetwork net) throws IOException {

        ModelSerializer.writeModel(net, modelFilename, true);
    }
    
}
