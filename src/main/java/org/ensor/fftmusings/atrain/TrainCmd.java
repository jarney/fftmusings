/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.atrain;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Random;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.ensor.fftmusings.pca.PCAFactory;
import org.ensor.fftmusings.pca.PCATransformer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 *
 * @author jona
 */
public class TrainCmd {
    public static final int QUANTA = 16;
    
    static class TrainingOptions {
        double learningRate;
        int trainingEpochs;
        String networkFilename;
        String pcaFilename;
        int hiddenLayerSize;
        Updater updater;
        int miniBatchSize;
        int iterationsPerBatch;
        int samplesPerIteration;
        String logFilename;
        
        @Override
        public String toString() {
            
            String str = "";
            
            str += "LR " + learningRate + "\n";
            str += "epochs " + trainingEpochs + "\n";
            if (networkFilename != null) {
                str += "Network Filename: " + networkFilename + "\n";
            }
            str += "pca: " + pcaFilename + "\n";
            str += "hidden layer size: " + hiddenLayerSize + "\n";
            str += "Updater: " + updater.toString() + "\n";
            
            str += "mini batch size: " + miniBatchSize + "\n";
            str += "iterations per batch " + iterationsPerBatch + "\n";
            str += "samples per iteration " + samplesPerIteration + "\n";
            str += "log filename " + logFilename + "\n";
            
            return str;
        }
        
    }

    private static TrainingOptions parseTrainingOptions(String[] args) throws ParseException {
        TrainingOptions trainingOptions = new TrainingOptions();
        CommandLineParser parser = new BasicParser();
        Options options = new Options();

        Option learningRateOption = new Option("r", "learning-rate", true, "Learning Rate");
        learningRateOption.setRequired(false);
        options.addOption(learningRateOption);

        Option networkFilenameOption = new Option("f", "network-filename", true, "Network filename (optional)");
        networkFilenameOption.setRequired(false);
        options.addOption(networkFilenameOption);
        
        Option trainingEpochsOption = new Option("e", "epochs", true, "Training Epochs");
        trainingEpochsOption.setRequired(false);
        options.addOption(trainingEpochsOption);
        
        Option hiddenLayerSizeOption = new Option("l", "hidden-layer-size", true, "Hidden Layer Size");
        hiddenLayerSizeOption.setRequired(false);
        options.addOption(hiddenLayerSizeOption);
        
        Option pcaFilenameOption = new Option("p", "pca-filename", true, "PCA Filename");
        pcaFilenameOption.setRequired(false);
        options.addOption(pcaFilenameOption);
        
        Option updaterOption = new Option("u", "updater", true, "Updater");
        updaterOption.setRequired(false);
        options.addOption(updaterOption);
        
        Option miniBatchSizeOption = new Option("b", "mini-batch-size", true, "Mini batch size");
        miniBatchSizeOption.setRequired(false);
        options.addOption(miniBatchSizeOption);
        
        Option samplesPerIterationOption = new Option("s", "samples-per-iteration", true, "Samples Per Iteration");
        samplesPerIterationOption.setRequired(false);
        options.addOption(samplesPerIterationOption);
        
        Option iterationsPerBatchOption = new Option("i", "iterations-per-batch", true, "Iterations per batch");
        iterationsPerBatchOption.setRequired(false);
        options.addOption(iterationsPerBatchOption);
        
        Option logFilename = new Option("w", "log-filename", true, "Log Filename");
        logFilename.setRequired(false);
        options.addOption(logFilename);
        
        CommandLine cmd = parser.parse(options, args);
        
        trainingOptions.learningRate = Double.parseDouble(cmd.getOptionValue("learning-rate", "0.1"));
        trainingOptions.networkFilename = cmd.getOptionValue("network-filename", null);
        trainingOptions.trainingEpochs = Integer.parseInt(cmd.getOptionValue("epochs", "10"));
        trainingOptions.pcaFilename = cmd.getOptionValue("pca-filename", "data/pca/smiths-30.pca");
        trainingOptions.updater = Updater.valueOf(cmd.getOptionValue("updater", "RMSPROP"));
        trainingOptions.hiddenLayerSize = Integer.parseInt(cmd.getOptionValue("hidden-layer-size", "3000"));
        trainingOptions.miniBatchSize = Integer.parseInt(cmd.getOptionValue("mini-batch-size", "10"));
        trainingOptions.samplesPerIteration = Integer.parseInt(cmd.getOptionValue("samples-per-iteration", "512"));
        trainingOptions.iterationsPerBatch = Integer.parseInt(cmd.getOptionValue("iterations-per-batch", "25"));
        
        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd.HH.mm.ss");
        String dateStamp = dateFormat.format(new Date());
        
        trainingOptions.logFilename = cmd.getOptionValue("log-filename", "training-") + dateStamp + ".log";

        return trainingOptions;
    }
    
    
    public static void main(String[] args) throws Exception {
        TrainingOptions trainingOptions = parseTrainingOptions(args);

        try (OutputStream os = new FileOutputStream(trainingOptions.logFilename)) {
            PrintStream logOutput = new PrintStream(os);
            System.out.println(trainingOptions);
            logOutput.println(trainingOptions);

            Random rng = new Random(12345);
            PCATransformer pca = PCAFactory.read(trainingOptions.pcaFilename);

            QFTIterator iter = new QFTIterator(
                    rng, pca, QUANTA,
                    trainingOptions.miniBatchSize,
                    trainingOptions.iterationsPerBatch,
                    trainingOptions.samplesPerIteration,
                    logOutput
            );

            MultiLayerNetwork net = create(trainingOptions, logOutput, iter);

            //Do training, and then generate and print samples from network
            for (int i = 0; i < trainingOptions.trainingEpochs; i++) {
                logOutput.println("Epoch number" + i);
                int miniBatchNumber = 0;
                while (iter.hasNext()) {
                    logOutput.println("Batch number " + miniBatchNumber);
                    miniBatchNumber++;
                    DataSet ds = iter.next();
                    net.fit(ds);
                    if (trainingOptions.networkFilename != null) {
                        ModelSerializer.writeModel(net, new File(trainingOptions.networkFilename), true);
                    }
                }

                iter.reset();	//Reset iterator for another epoch
            }
        }
    }
    
    public static MultiLayerNetwork load(File modelFilename, PrintStream logWriter) throws IOException {
        MultiLayerNetwork net = ModelSerializer.restoreMultiLayerNetwork(modelFilename);
        net.clear();
        net.setListeners(new ScoreIterationListener(logWriter));
        return net;
    }
    
    
    public static MultiLayerNetwork create(TrainingOptions trainingOptions, PrintStream logWriter, DataSetIterator iter) throws IOException {

        if (trainingOptions.networkFilename != null) {
            File modelFile = new File(trainingOptions.networkFilename);
            if (modelFile.exists()) {
                return load(modelFile, logWriter);
            }
        }
        
        int inputSize = iter.inputColumns();
        int outputSize = iter.totalOutcomes();
        int tbpttLength = 50;                       //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT).iterations(1)
                .learningRate(trainingOptions.learningRate)
                .rmsDecay(0.95)
                .seed(12345)
                .regularization(true)
                .l2(0.001)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.RMSPROP)
                .list()
                .layer(0, new GravesLSTM.Builder()
                        .nIn(inputSize)
                        .nOut(trainingOptions.hiddenLayerSize)
                        .activation(Activation.TANH).build())
                .layer(1, new GravesLSTM.Builder()
                        .nIn(trainingOptions.hiddenLayerSize)
                        .nOut(trainingOptions.hiddenLayerSize)
                        .activation(Activation.TANH).build())
                .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(trainingOptions.hiddenLayerSize)
                        .nOut(outputSize).build())
                .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
                .pretrain(false).backprop(true)
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(logWriter));

        if (trainingOptions.networkFilename != null) {
            ModelSerializer.writeModel(net, trainingOptions.networkFilename, true);
        }

        return net;
    }
    
}
