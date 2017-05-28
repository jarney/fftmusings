/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.autoencoder;

import java.io.IOException;
import java.util.Arrays;
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
import org.deeplearning4j.nn.params.PretrainParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.ensor.fftmusings.audio.AudioSample;
import org.ensor.fftmusings.audio.FFTOverlap;
import org.ensor.fftmusings.audio.WAVFileWriter;
import static org.ensor.fftmusings.autoencoder.GenericAutoencoder.evaluateModel;
import org.ensor.fftmusings.pipeline.ChannelDuplicator;
import org.ensor.fftmusings.pipeline.Pipeline;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 *
 * @author jona
 */
public class StackTrainer {
    public static void main(String [] args) throws IOException, Exception {
        
        MultiLayerNetwork pretrainedLayers[] = new MultiLayerNetwork[6];
        
        pretrainedLayers[0] = ModelSerializer.restoreMultiLayerNetwork("data/daa/model-1024-1200sparse0.01.nn");
        pretrainedLayers[1] = ModelSerializer.restoreMultiLayerNetwork("data/daa/model-1200-800sparse0.01.nn");
        pretrainedLayers[2] = ModelSerializer.restoreMultiLayerNetwork("data/daa/model-800-400sparse0.01.nn");
        pretrainedLayers[3] = ModelSerializer.restoreMultiLayerNetwork("data/daa/model-400-200sparse0.01.nn");
        pretrainedLayers[4] = ModelSerializer.restoreMultiLayerNetwork("data/daa/model-200-100sparse0.01.nn");
        pretrainedLayers[5] = ModelSerializer.restoreMultiLayerNetwork("data/daa/model-100-50sparse0.01.nn");
        
        NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(System.currentTimeMillis())
                .iterations(1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.NESTEROVS)
                .regularization(false)
                .l1(0.000)
                .learningRate(Double.parseDouble(args[0]));
        int layerNo = 0;
        
        NeuralNetConfiguration.ListBuilder listBuilder = builder.list()
                .layer(layerNo++, new RBM.Builder()
                        .nIn(1024)
                        .nOut(1200)
                        .activation(Activation.SIGMOID)
                        .build())
                .layer(layerNo++, new RBM.Builder()
                        .nIn(1200)
                        .nOut(800)
                        .activation(Activation.SIGMOID)
                        .build())
                .layer(layerNo++, new RBM.Builder()
                        .nIn(800)
                        .nOut(400)
                        .activation(Activation.SIGMOID)
                        .build())
                .layer(layerNo++, new RBM.Builder()
                        .nIn(400)
                        .nOut(200)
                        .activation(Activation.SIGMOID)
                        .build())
                .layer(layerNo++, new RBM.Builder()
                        .nIn(200)
                        .nOut(100)
                        .activation(Activation.SIGMOID)
                        .build())
//                .layer(layerNo++, new RBM.Builder()
//                        .nIn(100)
//                        .nOut(50)
//                        .activation(Activation.SIGMOID)
//                        .build())
//                .layer(layerNo++, new RBM.Builder()
//                        .nIn(50)
//                        .nOut(100)
//                        .activation(Activation.SIGMOID)
//                        .build())
                .layer(layerNo++, new RBM.Builder()
                        .nIn(100)
                        .nOut(200)
                        .activation(Activation.SIGMOID)
                        .build())
                .layer(layerNo++, new RBM.Builder()
                        .nIn(200)
                        .nOut(400)
                        .activation(Activation.SIGMOID)
                        .build())
                .layer(layerNo++, new RBM.Builder()
                        .nIn(400)
                        .nOut(800)
                        .activation(Activation.SIGMOID)
                        .build())
                .layer(layerNo++, new RBM.Builder()
                        .nIn(800)
                        .nOut(1200)
                        .activation(Activation.SIGMOID)
                        .build())
                .layer(layerNo++, new OutputLayer.Builder()
                        .nIn(1200)
                        .nOut(1024)
                        .activation(Activation.IDENTITY)
                        .lossFunction(LossFunctions.LossFunction.L2)
                        .build())
                .pretrain(false).backprop(true);


        MultiLayerConfiguration conf = listBuilder.build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(System.out)));

        for (layerNo = 0; layerNo < 5; layerNo++) {
            model.getLayer(layerNo).setParam(PretrainParamInitializer.BIAS_KEY, pretrainedLayers[layerNo].getLayer(0).getParam(PretrainParamInitializer.BIAS_KEY));
            model.getLayer(layerNo).setParam(PretrainParamInitializer.WEIGHT_KEY, pretrainedLayers[layerNo].getLayer(0).getParam(PretrainParamInitializer.WEIGHT_KEY));

            model.getLayer(model.getnLayers() - layerNo - 1).setParam(PretrainParamInitializer.BIAS_KEY, pretrainedLayers[layerNo].getLayer(1).getParam(PretrainParamInitializer.BIAS_KEY));
            model.getLayer(model.getnLayers() - layerNo - 1).setParam(PretrainParamInitializer.WEIGHT_KEY, pretrainedLayers[layerNo].getLayer(1).getParam(PretrainParamInitializer.WEIGHT_KEY));
        }
        
        DataSetIterator iter = new FFTDataIterator(new Random(),
            100,
            1250,
            System.out);
        
        int epoch = 0;
        for (int i = 0; i < 300; i++) {
            model.fit(iter);
            iter.reset();
            evaluateModel(model, epoch);
            ModelSerializer.writeModel(model, "stack.rnn", true);
            epoch++;
        }
    }
    
    private static void evaluateModel(MultiLayerNetwork model, int epoch) {
        
        String inputFilename = "data/fft/20.fftd";
        try (FFTOverlap.Reader wavFileIterator = FFTOverlap.createReader(inputFilename)) {
            new Pipeline(new Layer.FFTDToINDArray())
                .add(new GenericProcessor.Process(model))
                .add(new Layer.ToFFTPNG("data/daa/dct-magnitude-" + epoch + ".png", true))
                .add(new Layer.INDArrayToFFTD())
                .add(new FFTOverlap.NormalizeToHearing(false, 11025))
                .add(new FFTOverlap.ReversePhaseDelta(1024))
                .add(new ChannelDuplicator(AudioSample.class, 2))
                .add(WAVFileWriter.create("data/daa/sample-" + epoch + ".wav"))
                .execute(wavFileIterator);
        }
        catch (Exception ex) {
            throw new RuntimeException("Could not process file " + inputFilename, ex);
        }

    }
}
