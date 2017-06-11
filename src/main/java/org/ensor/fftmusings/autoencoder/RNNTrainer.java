/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.autoencoder;

import java.util.Iterator;
import java.util.Random;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
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
import org.ensor.fftmusings.audio.WAVFileWriter;
import org.ensor.fftmusings.pipeline.ChannelDuplicator;
import org.ensor.fftmusings.pipeline.Pipeline;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.NDArrayFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.ensor.fftmusings.mdn.LossMixtureDensity;
import org.ensor.fftmusings.mdn.MixtureDensityRNNOutputLayer;

/**
 *
 * @author jona
 */
public class RNNTrainer {
    static {
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
        NDArrayFactory factory = Nd4j.factory();
        factory.setDType(DataBuffer.Type.DOUBLE);

    }
    
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
        
        double learningRate = 0.0001;
        if (args.length != 0) {
            learningRate = Double.parseDouble(args[0]);
        }
        
        int nGaussians = 8;
        int labelWidth = iter.totalOutcomes();
        int inputWidth = iter.inputColumns();
        int lstmLayerSize = 400;
        int bttLength = 50;
        
        LossMixtureDensity costFunction = LossMixtureDensity.builder()
                .gaussians(nGaussians)
                .labelWidth(inputWidth)
                .build();
        
        
        //Set up network configuration:
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .learningRate(learningRate)
                .rmsDecay(0.95)
                .seed(12345)
                .iterations(1)
                .regularization(true)
                .l2(0.001)
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new GravesLSTM.Builder()
                        .nIn(inputWidth)
                        .nOut(lstmLayerSize)
                        .updater(Updater.RMSPROP)
                        .activation(Activation.TANH)
                        .build())
                .layer(1, new GravesLSTM.Builder()
                        .nIn(lstmLayerSize)
                        .nOut(lstmLayerSize)
                        .updater(Updater.RMSPROP)
                        .activation(Activation.TANH)
                        .build())
//                .layer(2, new RnnOutputLayer.Builder()
//                        .nIn(lstmLayerSize)
//                        .nOut((labelWidth + 2) * nGaussians)
//                        .activation(Activation.IDENTITY)
//                        //.lossFunction(LossFunctions.LossFunction.MSE)
//                        .lossFunction(LossMixtureDensity.builder()
//                            .gaussians(nGaussians)
//                            .labelWidth(inputWidth)
//                            .build())
//                        .updater(Updater.RMSPROP)
//                        .weightInit(WeightInit.DISTRIBUTION)
//                        .dist(new UniformDistribution(-0.08, 0.08)).build())
                .layer(2, new MixtureDensityRNNOutputLayer.Builder()
                        .gaussians(nGaussians)
                        .nIn(lstmLayerSize)
                        .nOut(labelWidth)
                        .updater(Updater.RMSPROP)
                        .build())
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
        private final LossMixtureDensity mCost;
        private INDArray nextInput;
        private final Random mRNG;
        
        public RNNSampleIterator(
                MultiLayerNetwork timeSeriesModel,
                LossMixtureDensity cost,
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

            LossMixtureDensity.MixtureDensityComponents mdc = mCost.extractComponents(output);
            INDArray sample = sampleFromNetwork(mRNG, mdc);
            //INDArray sample = output.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 100));

            int startLayer = mStaticModel.getnLayers()/2;
            int endLayer = mStaticModel.getnLayers()-1;
            INDArray indFFTDelta = mStaticModel.activateSelectedLayers(startLayer, endLayer, sample);

            nextInput = sample;
            
            iterations--;
            
            return indFFTDelta;
        }
        
        public INDArray sampleFromNetwork(Random rng, LossMixtureDensity.MixtureDensityComponents mdc) {
            INDArray alpha = mdc.getAlpha();
            int mixtures = alpha.size(1);
            double dice = rng.nextDouble();
            double cum = 0;
            int mixtureId = 0;
            for (int i = 0; i < mixtures; i++) {
                cum += alpha.getDouble(0, i);
                if (dice < cum) {
                    mixtureId = i;
                    break;
                }
            }
            
            double sigma = mdc.getSigma().getDouble(0, mixtureId);
            INDArray mu = mdc.getMu().get(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.point(mixtureId)});
            
            double[] dmean = new double[mu.size(0)];
            double[][] dsigma = new double[mu.size(0)][mu.size(0)];
            for (int i = 0; i < mu.size(0); i++) {
                dmean[i] = mu.getDouble(i, 0);
                dsigma[i][i] = sigma*sigma;
            }
            
            
            MultivariateNormalDistribution mnd = new MultivariateNormalDistribution(
                    dmean,
                    dsigma
            );

            double[] samples = mnd.sample();
            return Nd4j.create(samples).reshape(1, mu.size(0));
//            return mu;
        }
        
        
    }
    
    
    public static void evaluateModel(
            MultiLayerNetwork model,
            LossMixtureDensity cost,
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
