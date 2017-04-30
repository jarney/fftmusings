/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.rnn.qft;

import java.io.File;
import java.io.IOException;
import java.util.Random;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 *
 * @author jona
 */
public class SampleLSTM {
//    private static final int lstmLayerSize = FFTProcess.FFT_WINDOW_SIZE/2*
//            (FFTProcess.MAGNITUDE_QUANTA + FFTProcess.PHASE_QUANTA);					//Number of units in each GravesLSTM layer
    
    //Number of units in each GravesLSTM layer
    private static final int lstmLayerSize = 512;
    
    public static void main(String[] args) throws Exception {
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
        
        Random rng = new Random(12345);
        SampleIterator iter = new SampleIterator(rng);
        File file = new File("data/smiths.15.rnn");
        
        MultiLayerNetwork net = SampleLSTM.create(file, iter);
//        
//        //Initialize the user interface backend
//        UIServer uiServer = UIServer.getInstance();
//        StatsStorage statsStorage = new InMemoryStatsStorage();             //Alternative: new FileStatsStorage(File) - see UIStorageExample
//        int listenerFrequency = 1;
//        net.setListeners(new StatsListener(statsStorage, listenerFrequency));
//        uiServer.attach(statsStorage);
//        
        int iterations = 300000;
        for (int i = 0; i < iterations; i++) {
            net.fit(iter);
            
            iter.reset();
            SampleLSTM.persist(file, net);
            
            System.out.println("Fit one iteration " + i);
            
        }
        
    }

    public static MultiLayerNetwork load(File modelFilename) throws IOException {
        MultiLayerNetwork net = ModelSerializer.restoreMultiLayerNetwork(modelFilename);
        net.clear();
        net.setListeners(new ScoreIterationListener());
        return net;
    }
    

    public static MultiLayerNetwork create(File modelFilename, DataSetIterator iter) throws IOException {

        if (modelFilename.exists()) {
            return load(modelFilename);
        }
        
        
        int nOut = iter.totalOutcomes();

        //Set up network configuration:
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .learningRate(0.01)
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
                .pretrain(false).backprop(true)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener());

        ModelSerializer.writeModel(net, modelFilename, true);

        return net;
    }
    
    public static void persist(File modelFilename, MultiLayerNetwork net) throws IOException {

        ModelSerializer.writeModel(net, modelFilename, true);
    }
}
