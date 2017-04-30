/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.rnn.qft;

import org.ensor.fftmusings.io.ConcatStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import javax.sound.sampled.AudioFileFormat;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.ensor.fftmusings.data.QuantizedSpectrum;
import org.ensor.fftmusings.data.Sample;
import org.ensor.fftmusings.preprocess.FFTProcess;
import org.ensor.fftmusings.preprocess.WavIO;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author jona
 */
public class SampleFromLSTM {
    public static void main(String[] args) throws IOException, Exception {
        MultiLayerNetwork net = SampleLSTM.load(new File("data/smiths.15.rnn"));

        Random rng = new Random(12345);

        List<QuantizedSpectrum> outputList = new ArrayList<>();
        
        net.rnnClearPreviousState();
        
        QuantizedSpectrum qsInit = new QuantizedSpectrum(FFTProcess.FFT_WINDOW_SIZE);
        
        for (int j = 0; j < qsInit.size(); j++) {
            int magnitude = (int)(Math.random() * QuantizedSpectrum.MAGNITUDE_QUANTA-1);
            int phase = (int)(Math.random() * QuantizedSpectrum.PHASE_QUANTA-1);
            qsInit.setSample(j, magnitude, phase);
        }
        RNNInterface rnnInterface = new RNNInterface(rng);
        INDArray initializationInput = rnnInterface.toRNN(qsInit);
        
        INDArray output = net.rnnTimeStep(initializationInput);
        
        int samplesToMake = 256;
        
        for (int i = 0; i < samplesToMake; i++) {
            QuantizedSpectrum nextSpectrum = rnnInterface.toQS(output);
            INDArray nextInput = rnnInterface.toRNN(nextSpectrum);
            outputList.add(nextSpectrum);
            output = net.rnnTimeStep(nextInput);
//            output.mul(8);
//            rnnInterface.dumpOutput(output);
        }

        AudioFormat targetFormat = new AudioFormat(AudioFormat.Encoding.PCM_SIGNED,
                11025.0f, // Sample rate
                16, // Sample size (bits)
                2, // Channels
                4, // Frame Size
                11025.0f, // Frame Rate
                false); // Big-endian

        ConcatStream cs = new ConcatStream();
        
        FFTProcess fftProcessor = new FFTProcess(FFTProcess.FFT_WINDOW_SIZE);
        
        for (QuantizedSpectrum qs : outputList) {
            Sample s0 = fftProcessor.quantizedInverseFFT(qs);
            Sample[] s = new Sample[2];
            s[0] = s0;
            s[1] = s0;
            
            
            InputStream bais = WavIO.writeSamples(targetFormat, s);
            cs.addStream(bais, s[0].size());
        }
        
        
        AudioInputStream outputAIS = new AudioInputStream(cs, targetFormat,
                        cs.size());
        
        AudioSystem.write(outputAIS,
                AudioFileFormat.Type.WAVE,
                new File(String.format("sample.wav")));
        
    }

    public static void printRNNRow(INDArray rnnInput) {
        
    }
    
}
