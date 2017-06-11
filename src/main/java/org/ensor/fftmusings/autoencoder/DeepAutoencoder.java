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
import java.util.Random;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
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
import org.ensor.fftmusings.audio.FFTOverlap;
import org.ensor.fftmusings.audio.WAVFileWriter;
import org.ensor.fftmusings.pipeline.ChannelDuplicator;
import org.ensor.fftmusings.pipeline.IProcessor;
import org.ensor.fftmusings.pipeline.Pipeline;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.NDArrayFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 *
 * @author jona
 */
public class DeepAutoencoder {
    static {
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
        NDArrayFactory factory = Nd4j.factory();
        factory.setDType(DataBuffer.Type.DOUBLE);

    }
    
    public static void main(String[] args) throws Exception {
        
        double learningRate = 0.0001;
        if (args.length > 0) {
            learningRate = Double.parseDouble(args[0]);
        }
        
        MultiLayerNetwork model = readAutoencoder("data/daa/deepmodel.daa", learningRate);
        PrintStream log = System.out;

        DataSetIterator iter = createIterator();
        
        log.println("Train model....");
        int epoch = 0;
        for (int i = 0; i < 100; i++) {
            model.fit(iter);
            iter.reset();
            evaluateModel(model, epoch);
            ModelSerializer.writeModel(model, "data/daa/deepmodel.daa", true);
            epoch++;
        }
    }

    public static void evaluateModel(MultiLayerNetwork model, int epoch) {
        // Read DCT data from file and write a corresponding
        // .wav file based on that after passing it through the
        // auto-encoder to see what the network has learned.
        String inputFilename = "data/fft/20.fftd";
        try (FFTOverlap.Reader wavFileIterator = FFTOverlap.createReader(inputFilename)) {
            new Pipeline(new Layer.FFTDToINDArray())
                .add(new Layer.ToPNG("data/daa/input.png"))
                .add(new Encode(model, true))
                .add(new Layer.ToPNG("data/daa/hidden-" + epoch + ".png"))
                .add(new Encode(model, false))
                .add(new Layer.ToPNG("data/daa/output-" + epoch + ".png"))
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
    
    public static DataSetIterator createIterator() throws Exception {
        DataSetIterator iter = new FFTDataIterator(new Random(),
            100,
            3000,
            System.out);
        return iter;
    }
    

    public static MultiLayerNetwork readAutoencoder(String filename, double learningRate) throws IOException {
        MultiLayerNetwork model;
        
        File modelFile = new File(filename);
        if (!modelFile.exists()) {
            NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                    .seed(System.currentTimeMillis())
                    .iterations(1)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .weightInit(WeightInit.XAVIER)
                    .updater(Updater.NESTEROVS)
                    .regularization(true)
                    .l1(0.001);
            
            builder = builder.learningRate(learningRate);
            
            NeuralNetConfiguration.ListBuilder listBuilder = builder.list()
                    .layer(0, new RBM.Builder()
                            .nIn(1024)
                            .nOut(1200)
                            .activation(Activation.RELU)
                            .sparsity(0.1)
                            .build())
                    .layer(1, new RBM.Builder()
                            .nIn(1200)
                            .nOut(800)
                            .activation(Activation.RELU)
                            .sparsity(0.1)
                            .build())
                    .layer(2, new RBM.Builder()
                            .nIn(800)
                            .nOut(400)
                            .activation(Activation.RELU)
                            .sparsity(0.1)
                            .build())
                    .layer(3, new RBM.Builder()
                            .nIn(400)
                            .nOut(200)
                            .activation(Activation.RELU)
                            .sparsity(0.1)
                            .build())
                    .layer(4, new RBM.Builder()
                            .nIn(200)
                            .nOut(100)
                            .activation(Activation.RELU)
                            .sparsity(0.1)
                            .build())
                    .layer(5, new RBM.Builder()
                            .nIn(100)
                            .nOut(200)
                            .activation(Activation.RELU)
                            .sparsity(0.1)
                            .build())
                    .layer(6, new RBM.Builder()
                            .nIn(200)
                            .nOut(400)
                            .activation(Activation.RELU)
                            .sparsity(0.1)
                            .build())
                    .layer(7, new RBM.Builder()
                            .nIn(400)
                            .nOut(800)
                            .activation(Activation.RELU)
                            .sparsity(0.1)
                            .build())
                    .layer(8, new RBM.Builder()
                            .nIn(800)
                            .nOut(1200)
                            .activation(Activation.RELU)
                            .sparsity(0.1)
                            .build())
                    .layer(9, new OutputLayer.Builder()
                            .nIn(1200)
                            .nOut(1024)
                            .activation(Activation.IDENTITY)
                            .lossFunction(LossFunctions.LossFunction.L2)
                            .build())
                    .pretrain(false).backprop(true);

            
            MultiLayerConfiguration conf = listBuilder.build();

            model = new MultiLayerNetwork(conf);

            
            
            model.init();
            model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(System.out)));
        }
        else {
            model = ModelSerializer.restoreMultiLayerNetwork(modelFile);
        }

        
        return model;
    }
    public static class Encode implements IProcessor<INDArray, INDArray> {
        private final MultiLayerNetwork mModel;
        private final int start;
        private final int end;
        public Encode(MultiLayerNetwork model, boolean encode) {
            mModel = model;
            if (encode) {
                start = 0;
                end = mModel.getnLayers()/2-1;
            }
            else {
                start = mModel.getnLayers()/2;
                end = mModel.getnLayers()-1;
            }
        }

        @Override
        public void begin() {}

        @Override
        public INDArray process(INDArray input) {
            INDArray output = Layer.ModelProcessor.activateSelectedLayers(mModel, start, end, input);
            return output;
        }

        @Override
        public void end() {}
        
    }
}
