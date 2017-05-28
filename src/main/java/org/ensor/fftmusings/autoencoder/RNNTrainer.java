/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.autoencoder;

import java.io.PrintStream;
import java.util.Iterator;
import java.util.Random;
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
import org.ensor.fftmusings.audio.AudioSample;
import org.ensor.fftmusings.audio.FFTOverlap;
import org.ensor.fftmusings.audio.MagnitudeSpectrum;
import org.ensor.fftmusings.audio.WAVFileWriter;
import org.ensor.fftmusings.mdn.MixtureDensityCost;
import org.ensor.fftmusings.mdn.MixtureDensityRNNOutputLayer;
import org.ensor.fftmusings.pipeline.ChannelDuplicator;
import org.ensor.fftmusings.pipeline.Pipeline;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author jona
 */
public class RNNTrainer {
    public static void main(String[] args) throws Exception {
        
        MultiLayerNetwork stackedAutoencoder = ModelSerializer.restoreMultiLayerNetwork("stack.rnn");
        
        Random rng = new Random();
        
        RNNIterator iter = new RNNIterator(
            stackedAutoencoder,
            rng,
            100,
            1250,
            System.out
        );
        
        int mixturesPerLabel = 3;
        int parametersPerMixture = 3;
        int labels = iter.inputColumns();
        int lstmLayerSize = 200;
        int bttLength = 50;
        
        MixtureDensityCost costFunction = new MixtureDensityCost(mixturesPerLabel, labels);
        
        
        //Set up network configuration:
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .learningRate(0.1)
                .rmsDecay(0.95)
                .seed(12345)
                .iterations(1)
                .regularization(true)
                .l2(0.001)
                .list()
                .layer(0, new GravesLSTM.Builder()
                        .nIn(labels)
                        .nOut(lstmLayerSize)
                        .updater(Updater.RMSPROP)
                        .activation(Activation.TANH)
                        .weightInit(WeightInit.DISTRIBUTION)
                        .dist(new UniformDistribution(-0.08, 0.08)).build())
                .layer(1, new GravesLSTM.Builder()
                        .nIn(lstmLayerSize)
                        .nOut(lstmLayerSize)
                        .updater(Updater.RMSPROP)
                        .activation(Activation.TANH)
                        .weightInit(WeightInit.DISTRIBUTION)
                        .dist(new UniformDistribution(-0.08, 0.08)).build())
                .layer(2, new MixtureDensityRNNOutputLayer.Builder()
                        .mixturesPerLabel(mixturesPerLabel)
                        .nIn(lstmLayerSize)
                        .nOut(labels)
                        .updater(Updater.RMSPROP)
                        .weightInit(WeightInit.DISTRIBUTION)
                        .dist(new UniformDistribution(-0.08, 0.08)).build())
                .pretrain(false)
                .backprop(true)
                .backpropType(BackpropType.TruncatedBPTT)
                    .tBPTTForwardLength(bttLength)
                    .tBPTTBackwardLength(bttLength)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(System.out));
        
        for (int epoch = 0; epoch < 300; epoch++) {
            model.fit(iter);
            iter.reset();
            evaluateModel(model, costFunction, stackedAutoencoder, rng, epoch);
            ModelSerializer.writeModel(model, "stack-timeseries.rnn", true);
        }
    }
    
    public static class RNNSampleIterator implements Iterator<INDArray> {

        private int iterations = 0;
        private final MultiLayerNetwork mTimeSeriesModel;
        private final MultiLayerNetwork mStaticModel;
        private final MixtureDensityCost mCost;
        private INDArray nextInput;
        private final Random mRNG;
        
        public RNNSampleIterator(
                MultiLayerNetwork timeSeriesModel,
                MixtureDensityCost cost,
                MultiLayerNetwork staticModel,
                INDArray initialInput,
                Random rng) {
            iterations = 3200;
            mStaticModel = staticModel;
            mTimeSeriesModel = timeSeriesModel;
            mCost = cost;
            nextInput = initialInput;
            mRNG = rng;
        }
        
        @Override
        public boolean hasNext() {
            return iterations > 0;
        }

        @Override
        public INDArray next() {
            INDArray output = mTimeSeriesModel.rnnTimeStep(nextInput);

            INDArray sample = mCost.sampleFromNetwork(mRNG, output);
            mTimeSeriesModel.rnnTimeStep(sample);

            int startLayer = mStaticModel.getnLayers()/2;
            int endLayer = mStaticModel.getnLayers()-1;
            INDArray indFFTDelta = mStaticModel.activateSelectedLayers(startLayer, endLayer, sample);

            nextInput = sample;
            
            iterations--;
            
            return indFFTDelta;
        }
        
    }
    
    
    public static void evaluateModel(
            MultiLayerNetwork model,
            MixtureDensityCost cost,
            MultiLayerNetwork stackedAutoencoder,
            Random rng,
            int epoch) {
        
        
        //Create input for initialization
        INDArray initialInput = Nd4j.zeros(100);
        for (int i = 0; i < 100; i++) {
            initialInput.putScalar(i, rng.nextDouble());
        }
        
        model.rnnClearPreviousState();

        Iterator<INDArray> rnnSampleIterator = new RNNSampleIterator(model, cost, stackedAutoencoder, initialInput, rng);
        
        new Pipeline(new Layer.ToFFTPNG("data/daa/dct-magnitude-" + epoch + ".png", true))
                .add(new Layer.INDArrayToFFTD())
                .add(new FFTOverlap.NormalizeToHearing(false, 11025))
                .add(new FFTOverlap.ReversePhaseDelta(1024))
                .add(new ChannelDuplicator(AudioSample.class, 2))
                .add(WAVFileWriter.create("data/daa/sample-" + epoch + ".wav"))
                .execute(rnnSampleIterator);
        
    }
    
}
