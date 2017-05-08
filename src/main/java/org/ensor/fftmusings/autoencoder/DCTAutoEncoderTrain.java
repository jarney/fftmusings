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
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.ensor.fftmusings.audio.AudioSample;
import org.ensor.fftmusings.audio.FFTOverlap;
import org.ensor.fftmusings.audio.FFTToPNG;
import org.ensor.fftmusings.audio.WAVFileWriter;
import org.ensor.fftmusings.pipeline.ChannelDuplicator;
import org.ensor.fftmusings.pipeline.Pipeline;
import org.nd4j.linalg.activations.Activation;

/**
 * @author Adam Gibson
 */
public class DCTAutoEncoderTrain {

    public static void main(String[] args) throws Exception {
        int seed = 123;
        int miniBatchSize = 20;
        int batchSize = 50;
        int iterations = 100;

        Random rng = new Random(1234);

        PrintStream log = System.out;
        
        log.println("Load data....");
        DCTDataIterator iter = new DCTDataIterator(rng,
            miniBatchSize,
            batchSize,
            log);

        
        Map<Integer, Double> lrSchedule = new HashMap<>();
        lrSchedule.put(1, 0.01);
        lrSchedule.put(2000, 0.01);
        
        log.println("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .learningRate(0.001)
                //.learningRateDecayPolicy(LearningRatePolicy.Schedule)
                //.learningRateSchedule(lrSchedule)
                .iterations(iterations)
                //.regularization(true)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                //.l2(0.001)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.RMSPROP)
                .list()
                .layer(0, new RBM.Builder()
                        .nIn(iter.inputColumns())
                        .nOut(800)
                        .activation(Activation.SIGMOID)
                        .build())
                .layer(1, new RBM.Builder()
                        .nIn(800)
                        .nOut(800)
                        .activation(Activation.SIGMOID)
                        .build())
                .layer(2, new OutputLayer.Builder()
                        .nIn(800)
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .nOut(iter.totalOutcomes())
                        .activation(Activation.SIGMOID)
                        .build())
                .pretrain(false).backprop(true)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(log)));

        log.println("Train model....");
        int epoch = 0;
        for (int i = 0; i < 1000; i++) {
            model.fit(iter);
//            if (epoch > 900) {
//                iter.setMiniBatchSize(1);
//            }
//            else if (epoch > 300) {
//                iter.setMiniBatchSize(50);
//            }
//            else if (epoch > 200) {
//                iter.setMiniBatchSize(100);
//            }
//            else if (epoch > 100) {
//                iter.setMiniBatchSize(250);
//            }
//            else {
//                iter.setMiniBatchSize(500);
//            }
            iter.reset();
            evaluateModel(model, epoch);
            ModelSerializer.writeModel(model, "data/daa/model-" + epoch + ".faa", true);
            epoch++;
        }
    }
    
    public static void evaluateModel(MultiLayerNetwork model, int epoch) {
        // Read DCT data from file and write a corresponding
        // .wav file based on that after passing it through the
        // auto-encoder to see what the network has learned.
        String inputFilename = "data/fft/20.fft";
        try (FFTOverlap.Reader wavFileIterator = FFTOverlap.createReader(inputFilename)) {
            new Pipeline(new AutoEncoderProcessor(model))
                .add(new FFTToPNG("data/daa/fft-magnitude-" + epoch + ".png", false))
                .add(new FFTOverlap.Reverse(1024))
                .add(new ChannelDuplicator(AudioSample.class, 2))
                .add(WAVFileWriter.create("data/daa/sample-" + epoch))
                .execute(wavFileIterator);
        }
        catch (Exception ex) {
            throw new RuntimeException("Could not process file " + inputFilename, ex);
        }
        
    }
    
}