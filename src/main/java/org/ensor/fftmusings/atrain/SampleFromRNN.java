/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.atrain;

import org.ensor.fftmusings.rnn.qft.*;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.ensor.fftmusings.audio.QuantizedVector;
import org.ensor.fftmusings.audio.RNNInput;
import org.ensor.fftmusings.audio.RNNInput.Reverse.SampleStyle;
import org.ensor.fftmusings.pca.PCAFactory;
import org.ensor.fftmusings.pca.PCATransformer;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author jona
 */
public class SampleFromRNN {
    public static void main(String[] args) throws IOException, Exception {
        int nQuanta = 16;
        String networkFilename = "data/smiths-dct-30-16-0.01-500-10.rnn";
        String outputFilename = networkFilename + ".wav";
        String outputImageFilename = outputFilename + ".png";

        Random rng = new Random(System.currentTimeMillis());

        List<QuantizedVector> outputList = new ArrayList<>();
        
        MultiLayerNetwork net = SampleLSTM.load(new File(networkFilename));
        net.rnnClearPreviousState();
        
        String pcaFilename = "data/pca/smiths-30.pca";
        
        PCATransformer pca = PCAFactory.read(pcaFilename);
        QuantizedVector qsInit = new QuantizedVector(pca.getDimensions(), nQuanta);
        
        for (int j = 0; j < qsInit.size(); j++) {
            int magnitude = rng.nextInt(qsInit.getQuanta());
            qsInit.setSample(j, magnitude);
        }
        RNNInput.Forward quantaToRNN = new RNNInput.Forward();
        RNNInput.Reverse rnnToQuanta = new RNNInput.Reverse(qsInit.getQuanta(), SampleStyle.LARGEST, rng);
        
        INDArray initializationInput = quantaToRNN.process(qsInit);
        
        int samplesToMake = 25;
        
        INDArray nextInput = initializationInput;
        
        for (int i = 0; i < samplesToMake; i++) {
            System.out.println("next input");
            System.out.println(nextInput);
            INDArray output = net.rnnTimeStep(nextInput);
            
            System.out.println(output.mul(1));
        
            QuantizedVector nextSpectrum = rnnToQuanta.process(output);
            nextInput = quantaToRNN.process(nextSpectrum);
            outputList.add(nextSpectrum);
        }

//        new Pipeline(new Quantize.Reverse(nQuanta, -1, 1, true))
//                .add(new PCATransformProcessorDCTReverse(pca))
//                .add(new DCT.ToPNG(outputImageFilename))
//                .add(new DCT.Reverse(false))
//                .add(new ChannelDuplicator(AudioSample.class, 2))
//                .add(WAVFileWriter.create(outputFilename))
//                .execute(outputList.iterator());
    }    
}
