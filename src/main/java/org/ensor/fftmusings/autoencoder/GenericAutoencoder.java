/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.autoencoder;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
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
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.ensor.fftmusings.audio.AudioSample;
import org.ensor.fftmusings.audio.DCT;
import org.ensor.fftmusings.audio.FFTOverlap;
import org.ensor.fftmusings.audio.WAVFileWriter;
import org.ensor.fftmusings.pipeline.ChannelDuplicator;
import org.ensor.fftmusings.pipeline.Pipeline;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.shade.jackson.databind.ObjectMapper;

/**
 *
 * @author jona
 */
public class GenericAutoencoder {
    
    public static void main(String[] args) throws Exception {
        String encoderFile = args[0];
        //encoderFile = "data/daa/autoencoder.json";
        Autoencoder ae = readAutoencoder(encoderFile);
        PrintStream log = System.out;
        
        DataSetIterator iter = createIterator(ae);
        
        log.println("Train model....");
        int epoch = 0;
        for (int i = 0; i < ae.getMetadata().getEpochs(); i++) {
            ae.getModel().fit(iter);
            iter.reset();
            evaluateModel(ae, epoch);
            ModelSerializer.writeModel(ae.getModel(), ae.getMetadata().getFilename(), true);
            epoch++;
        }
    }

    public static void evaluateModel(Autoencoder model, int epoch) {
        // Read DCT data from file and write a corresponding
        // .wav file based on that after passing it through the
        // auto-encoder to see what the network has learned.
        String inputFilename = "data/fft/20.fftd";
        try (FFTOverlap.Reader wavFileIterator = FFTOverlap.createReader(inputFilename)) {
            new Pipeline(new Layer.FFTDToINDArray())
                .add(new GenericProcessor.Encode(model))
                .add(new Layer.ToPNG("data/daa/hidden-" + epoch + ".png"))
                .add(new GenericProcessor.Decode(model))
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
    
    public static DataSetIterator createIterator(Autoencoder ae) throws Exception {
        if (ae.getSourceAutoencoder() != null) {
            Autoencoder sourceAutoencoder = ae.getSourceAutoencoder();
            return new LayerwiseIterator(
                    createIterator(sourceAutoencoder),
                    sourceAutoencoder.getModel(),
                    ae.getMetadata().getHidden(),
                    ae.getMetadata().getMinibatchSize()
            );
        }
//        DataSetIterator iter = new FFTDataIterator(new Random(),
//            ae.getMetadata().getMinibatchSize(),
//            ae.getMetadata().getMinibatchesPerIteration(),
//            System.out);
        DataSetIterator iter = new FFTDataIterator(new Random(),
            ae.getMetadata().getMinibatchSize(),
            ae.getMetadata().getMinibatchesPerIteration(),
            System.out);
        return iter;
    }
    

    public static Autoencoder readAutoencoder(String filename) throws IOException {
        ObjectMapper m = new ObjectMapper();
        AutoencoderMetadata autoEncoderData = m.readValue(new File(filename), AutoencoderMetadata.class);
        Autoencoder ae = new Autoencoder(autoEncoderData);
        
        if ("autoencoded".equals(autoEncoderData.getInputSource())) {
            Autoencoder sourceAutoencoder = readAutoencoder(autoEncoderData.getSourceAutoencoder());
            ae.setSourceAutoencoder(sourceAutoencoder);
        }
        else if ("iterator".equals(autoEncoderData.getInputSource())) {
        }

        String modelFilename = autoEncoderData.getFilename();
        if (modelFilename == null) {
            modelFilename = "newmodel.faa";
            autoEncoderData.setFilename(modelFilename);
        }
        
        File modelFile = new File(modelFilename);
        if (!modelFile.exists()) {
            NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                    .seed(System.currentTimeMillis())
                    .iterations(autoEncoderData.getIterations())
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .weightInit(WeightInit.XAVIER)
                    .updater(Updater.NESTEROVS)
                    .regularization(autoEncoderData.getRegularization())
                    .l1(autoEncoderData.getL2Regularization());
            
            if (autoEncoderData.getLRArray().isEmpty()) {
                builder = builder.learningRate(autoEncoderData.getLearningRate());
            }
            else {
                Map<Integer, Double> lrSchedule = new HashMap<>();
                builder = builder.learningRate(autoEncoderData.getLRArray().get(0).getLR());
                for (LRSchedule schedule : autoEncoderData.getLRArray()) {
                    lrSchedule.put(schedule.getIteration(), schedule.getLR());
                }
                builder = builder
                        .learningRateDecayPolicy(LearningRatePolicy.Schedule)
                        .learningRateSchedule(lrSchedule);
            }
            
            NeuralNetConfiguration.ListBuilder listBuilder = builder.list()
                    .layer(0, new RBM.Builder()
                            .nIn(autoEncoderData.getInput())
                            .nOut(autoEncoderData.getHidden())
                            .activation(Activation.fromString(autoEncoderData.getIntermediateActivation()))
                            .sparsity(autoEncoderData.getSparsity())
                            .build())
                    .layer(1, new OutputLayer.Builder()
                            .nIn(autoEncoderData.getHidden())
                            .nOut(autoEncoderData.getInput())
                            .lossFunction(autoEncoderData.getLossFunction())
                            .activation(Activation.fromString(autoEncoderData.getFinalActivation()))
                            .build())
                    .pretrain(false).backprop(true);

            
            MultiLayerConfiguration conf = listBuilder.build();

            MultiLayerNetwork model = new MultiLayerNetwork(conf);

            
            
            model.init();
            model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(System.out)));
            ae.setModel(model);
        }
        else {
            MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelFile);
            ae.setModel(model);
        }

        
        System.out.println("Autoencoder: " + autoEncoderData);
        return ae;
    }
    
}
