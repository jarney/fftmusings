package org.ensor.fftmusings.autoencoder;

import java.io.PrintStream;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.ensor.fftmusings.audio.AudioSample;
import org.ensor.fftmusings.audio.DCT;
import org.ensor.fftmusings.audio.WAVFileWriter;
import org.ensor.fftmusings.pipeline.ChannelDuplicator;
import org.ensor.fftmusings.pipeline.Pipeline;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

/**
 * @author Adam Gibson
 */
public class DCTAutoEncoderTrain {

    public static void main(String[] args) throws Exception {
        int seed = 123;
        int miniBatchSize = 100;
        int batchSize = 1250;
        int iterations = 1;

        Random rng = new Random(1234);

        PrintStream log = System.out;
        
        log.println("Load data....");
        DCTDataIterator iter = new DCTDataIterator(rng,
            miniBatchSize,
            batchSize,
            log);

        
        Map<Integer, Double> lrSchedule = new HashMap<>();
        //lrSchedule.put(1, 1e-2);
        //lrSchedule.put(5000, 1e-2);
        
        log.println("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .learningRate(1)
                //.learningRateDecayPolicy(LearningRatePolicy.Schedule)
                //.learningRateSchedule(lrSchedule)
                .iterations(iterations)
                //.regularization(true)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                //.l2(0.001)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.NESTEROVS)
                .list()
                .layer(0, new RBM.Builder()
                        .nIn(iter.inputColumns())
                        .nOut(400)
                        .activation(Activation.TANH)
                        .build())
                .layer(1, new RBM.Builder()
                        .nIn(400)
                        .nOut(400)
                        .activation(Activation.TANH)
                        .build())
                .layer(2, new OutputLayer.Builder()
                        .nIn(400)
                        .nOut(iter.totalOutcomes())
                        .lossFunction(LossFunction.MSE)
                        .activation(Activation.IDENTITY)
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
        String inputFilename = "data/dct/20.dct";
        try (DCT.Reader wavFileIterator = DCT.createReader(inputFilename)) {
            new Pipeline(new DCTAutoEncoderProcessor(model))
                .add(new DCT.ToPNG("data/daa/dct-magnitude-" + epoch + ".png"))
                .add(new DCT.Reverse(true))
                .add(new ChannelDuplicator(AudioSample.class, 2))
                .add(WAVFileWriter.create("data/daa/sample-" + epoch))
                .execute(wavFileIterator);
        }
        catch (Exception ex) {
            throw new RuntimeException("Could not process file " + inputFilename, ex);
        }
        
    }
    
}