/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.audio;

import org.ensor.fftmusings.io.ICloseableIterator;
import org.ensor.fftmusings.pca.PCAFactory;
import org.ensor.fftmusings.pca.PCATransformer;
import org.ensor.fftmusings.pipeline.ChannelDuplicator;
import org.ensor.fftmusings.pipeline.ChannelSelector;
import org.ensor.fftmusings.pipeline.Pipeline;

/**
 *
 * @author jona
 */
public class PCATransformFFT {
    public static void main(String[] args) throws Exception {
        args = new String[2];
        args[0] = "data/wav/20.wav";
        args[1] = "data/pca/sample.fft.pca.wav";
        
        if (args.length != 2) {
            System.out.println("Usage: PCADCTData foo.wav foo.wav");
            System.exit(1);
        }
        int sampleSize = 512;
        String inputFilename = args[0];
        String pcaFilename = "data/pca/sample.fft.translator.pca";
        String outputFilename = args[1];
        
        PCATransformer pca = PCAFactory.read(pcaFilename);
        
        try (ICloseableIterator<AudioSample[]> wavFileIterator = WAVFileIterator.create(inputFilename, sampleSize)) {
            new Pipeline(new ChannelSelector(AudioSample.class, 0))
                    .add(new FFTOverlap.Forward(sampleSize*2))
                    .add(new PCATransformProcessorFFTForward(pca))
                    .add(new PCATransformProcessorFFTReverse(pca))
                    .add(new FFTOverlap.Reverse(sampleSize*2))
                    .add(new ChannelDuplicator(AudioSample.class, 2))
                    .add(WAVFileWriter.create(outputFilename))
                    .execute(wavFileIterator);
        }
    }
}
