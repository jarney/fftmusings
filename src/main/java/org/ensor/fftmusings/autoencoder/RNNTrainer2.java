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
import org.deeplearning4j.nn.api.layers.RecurrentLayer;
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
import org.ensor.fftmusings.statistics.GaussianDistribution;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

/**
 *
 * @author jona
 */
public class RNNTrainer2 {
    public static void main(String[] args) throws Exception {
        
        MultiLayerNetwork stackedAutoencoder = ModelSerializer.restoreMultiLayerNetwork("stack.rnn");
        
        Random rng = new Random();
        
        RNNIterator iter = new RNNIterator(
            stackedAutoencoder,
            rng,
            100,
            100,
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
                .layer(2, new RnnOutputLayer.Builder()
                        .nIn(lstmLayerSize)
                        .nOut(labels)
                        .lossFunction(LossFunction.MSE)
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

        private int mIterations = 0;
        private final MultiLayerNetwork mTimeSeriesModel;
        private final MultiLayerNetwork mStaticModel;
        private final MixtureDensityCost mCost;
        private INDArray nextInput;
        private final Random mRNG;
        private final GaussianDistribution mGD;

        public RNNSampleIterator(
                MultiLayerNetwork timeSeriesModel,
                MixtureDensityCost cost,
                MultiLayerNetwork staticModel,
                int iterations,
                INDArray initialInput,
                Random rng,
                GaussianDistribution gd) {
            mIterations = iterations;
            mStaticModel = staticModel;
            mTimeSeriesModel = timeSeriesModel;
            mCost = cost;
            nextInput = initialInput;
            mRNG = rng;
            mGD = gd;
        }
        
        @Override
        public boolean hasNext() {
            return mIterations > 0;
        }

        public INDArray rnnTimeStep(MultiLayerNetwork model, INDArray input) {
            
            model.setInputMiniBatchSize(input.size(0));	//Necessary for preprocessors/reshaping
            model.setInput(input);
            boolean inputIs2d = input.rank()==2;
            org.deeplearning4j.nn.api.Layer [] layers = model.getLayers();
            for( int i = 0; i < model.getnLayers(); i++) {
                if(model.getLayerWiseConfigurations().getInputPreProcess(i) != null)
                    input = model.getLayerWiseConfigurations().getInputPreProcess(i).preProcess(input,model.getInputMiniBatchSize());
                if(layers[i] instanceof RecurrentLayer){
                    if (i == 1) {
                        int n = input.shape()[1];
                        for (int k = 0; k < n; k++) {
                            input.putScalar(0, k, 0, input.getDouble(0, k, 0) + mGD.sample(mRNG));
                        }
                    }
                    input = ((RecurrentLayer)layers[i]).rnnTimeStep(input);
                } else if(layers[i] instanceof MultiLayerNetwork){
                    input = ((MultiLayerNetwork)layers[i]).rnnTimeStep(input);
                } else {
                    input = layers[i].activate(input, false);
                }
            }
            if(inputIs2d && input.rank()==3 && layers[layers.length-1].type() == org.deeplearning4j.nn.api.Layer.Type.RECURRENT){
                //Return 2d output with shape [miniBatchSize,nOut]
                // instead of 3d output with shape [miniBatchSize,nOut,1]
                return input.tensorAlongDimension(0,1,0);
            }

            model.setInput(null);
            return input;
        }

        
        @Override
        public INDArray next() {
            //INDArray sample = mTimeSeriesModel.rnnTimeStep(nextInput);
            
            // This version perturbs the hidden-state's
            // values to produce variation in the output
            // by gently giving the hidden layer some
            // small 'nudge' in random directions.
            INDArray sample = rnnTimeStep(mTimeSeriesModel, nextInput);

            int startLayer = mStaticModel.getnLayers()/2;
            int endLayer = mStaticModel.getnLayers()-1;
            INDArray indFFTDelta = mStaticModel.activateSelectedLayers(startLayer, endLayer, sample);

            nextInput = sample;
            
            mIterations--;
            
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

        int samples = 4;
        for (int i = 0; i < samples; i++) {
            
            GaussianDistribution gd = new GaussianDistribution(0, 0.0001*i);

            Iterator<INDArray> rnnSampleIterator = new RNNSampleIterator(
                    model, cost, stackedAutoencoder,
                    3200, initialInput, rng,
                    gd
            );

            new Pipeline(new Layer.ToFFTPNG("data/daa/dct-magnitude-" + epoch + "-" + i + ".png", true))
                    .add(new Layer.INDArrayToFFTD())
                    .add(new FFTOverlap.NormalizeToHearing(false, 11025))
                    .add(new FFTOverlap.ReversePhaseDelta(1024))
                    .add(new ChannelDuplicator(AudioSample.class, 2))
                    .add(WAVFileWriter.create("data/daa/sample-" + epoch + "-" + i + ".wav"))
                    .execute(rnnSampleIterator);
        
        }
    }
    
}
