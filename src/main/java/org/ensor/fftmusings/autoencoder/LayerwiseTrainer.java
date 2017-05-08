/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.autoencoder;

import java.io.PrintStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 *
 * @author jona
 */
public class LayerwiseTrainer {
    // This is a layer-wise trainer for a deep network.
    // This uses induction on the number of layers to train.
    
    // Algorithm:
    // Begin with a 3 layer network, train up to a specified cost function
    // threshold.
    // Input -> RBN Input -> RBN Hidden -> RBN Output.
    // Once trained to the specified threshold, the input
    // from the first activation becomes the input to train
    // for the next 3 layers
    // Input0 -> RBN Input0 -> RBN1 Input -> RBN1 Hidden -> RBN1 Output -> RBN0 Output.
    // The network is then trained on "RBN1" attempting to reproduce the
    // input which leads to the needed output for the next layer.
    // Continue this process until all desired layers are trained.
    // The network will then consist of the assembled layers for use.
    
    // Net1 = loadPreviousNetwork(ioSize, hiddenSize);
    // Net2 = constructNextNetwork(hiddenSize, nextSize);
    
    // Net2Iterator = 
    //      net1.previousLayerInput();
    //      input = net1.activate(0);
    //      output = net1.activate(0,1);
    // Train net2's 3 layers to reproduce the behavior of net1's hidden layer.
    public static void main(String[] args) throws Exception {
        final int numRows = 28;
        final int numColumns = 28;
        int seed = 123;
        int miniBatchSize = 20;
        int numSamples = 100;
        int batchSize = 10;
        int iterations = 1;
        int listenerFreq = iterations/5;

        MultiLayerNetwork lastModel = ModelSerializer.restoreMultiLayerNetwork("data/daa/model-140.faa");
        
        int nLayers = lastModel.getnLayers();
        
        if (nLayers != 3) {
            System.out.println("Last model didn't have 3 layers.");
            return;
        }
        Random rng = new Random(1234);
        
        PrintStream log = System.out;
        
        Map<Integer, Double> lrSchedule = new HashMap<>();
        lrSchedule.put(1, 0.1);
        //lrSchedule.put(400, 0.02);
        //lrSchedule.put(800, 0.001);
        
        // Take the middle layer from the last model.
        // we will build an autoencoder around this layer
        // and reproduce its output.
        Layer middleLayer = lastModel.getLayer(1);
        int middleLayerSize = 800;
        int hiddenLayerSize = middleLayerSize/2;
        
        log.println("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .learningRate(0.1)
                .learningRateDecayPolicy(LearningRatePolicy.Schedule)
                .learningRateSchedule(lrSchedule)
                .iterations(iterations)
                //.regularization(true)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                //.l2(0.001)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.RMSPROP)
                .list()
                .layer(0, new RBM.Builder()
                        .nIn(middleLayerSize)
                        .nOut(hiddenLayerSize)
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .activation(Activation.TANH)
                        .build())
                .layer(1, new RBM.Builder()
                        .nIn(hiddenLayerSize)
                        .nOut(hiddenLayerSize)
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .activation(Activation.TANH)
                        .build())
                .layer(2, new OutputLayer.Builder()
                        .nIn(800)
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .nOut(middleLayerSize)
                        .activation(Activation.IDENTITY)
                        .build())
                .pretrain(false).backprop(true)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(log)));
        
        
    }

}
