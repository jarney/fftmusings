/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.mdn.example;

import java.io.PrintStream;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.ensor.fftmusings.atrain.ScoreIterationListener;
import org.ensor.fftmusings.mdn.MixtureDensityCost;
import org.ensor.fftmusings.mdn.MixtureDensityCost.MixtureDensityComponents;
import org.ensor.fftmusings.mdn.MixtureDensityOutputLayer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author jona
 */
public class MixtureDensityNetwork {
    public static void main(String[] args) {
        
        final int inputSize = 1;
        final int outputLabels = 2;
        
        final int mixtureComponentsPerOutput = 2;
        final int variablesPerMixture = 3;
        
        final int hiddenLayerSize = 50 * mixtureComponentsPerOutput;
        
        final int outputSize = outputLabels * variablesPerMixture * mixtureComponentsPerOutput;
        
        NumberFormat formatter = new DecimalFormat("#0.0000");

        Map<Integer, Double> lrSchedule = new HashMap<>();
        lrSchedule.put(0, .1);
        lrSchedule.put(5000, 0.01);
        lrSchedule.put(10000, 0.001);
        
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT).iterations(1)
//                .learningRate(0.1)
                .learningRateDecayPolicy(LearningRatePolicy.Schedule)
                .learningRateSchedule(lrSchedule)
                .rmsDecay(0.95)
                .seed(12345)
                .regularization(true)
                .l2(0.001)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.RMSPROP)
                .list()
                .layer(0, new RBM.Builder()
                        .nIn(inputSize)
                        .nOut(hiddenLayerSize)
                        .activation(Activation.TANH)
//                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .build())
                .layer(1, new RBM.Builder()
                        .nIn(hiddenLayerSize)
                        .nOut(hiddenLayerSize)
                        .activation(Activation.TANH)
//                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .build())
//                .layer(2, new OutputLayer.Builder()
//                        .nIn(hiddenLayerSize)
//                        .nOut(mixtureComponentsPerOutput*3*inputSize)
//                        .lossFunction(new MixtureDensityCost(mixtureComponentsPerOutput, inputSize))
//                        .activation(Activation.IDENTITY)
//                        .build())
                .layer(2, new MixtureDensityOutputLayer.Builder()
                        .nIn(hiddenLayerSize)
                        .mixturesPerLabel(mixtureComponentsPerOutput)
                        .labelValues(outputLabels)
                        .build())
                .pretrain(false).backprop(true)
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(System.out));
        
        int trainingEpochs = 9000;
        
        DataSetIterator iter = new InverseProblemIterator(mixtureComponentsPerOutput);
        PrintStream logOutput = System.out;
        
        Random rng = new Random();
        
        //Do training, and then generate and print samples from network
        for (int i = 0; i < trainingEpochs; i++) {
            logOutput.println("Epoch number " + i);
            int miniBatchNumber = 0;
            while (iter.hasNext()) {
                miniBatchNumber++;
                DataSet ds = iter.next();
                net.fit(ds);
            }

            iter.reset();	//Reset iterator for another epoch
            
            INDArray in = Nd4j.zeros(1);
            
            for (int j = 0; j < 11; j++) {
                double input = (.1 * j - 0.5)*10;
                in.putScalar(0, input);
                INDArray output = net.activateSelectedLayers(0, net.getnLayers()-1, in);

                MixtureDensityComponents mixtures = MixtureDensityCost.extractComponents(output, 1, outputLabels, mixtureComponentsPerOutput);
                
                System.out.print("" + formatter.format(input) + "\t");
                for (int aa = 0;  aa < outputLabels; aa++) {
                    for (int bb = 0; bb < mixtureComponentsPerOutput; bb++) {
                        System.out.print( 
                                formatter.format(mixtures.alpha.getDouble(0, aa, bb)) + "\t" +
                                formatter.format(mixtures.mu.getDouble(0, aa, bb)) + "\t" +
                                formatter.format(mixtures.sigma.getDouble(0, aa, bb)));
                        System.out.print("|\t|");
                    }
                }
                System.out.println();
            }
        }
    }
}
