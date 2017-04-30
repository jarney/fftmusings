/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.audio;

import org.ensor.fftmusings.io.ICloseableIterator;
import org.ensor.fftmusings.pca.PCAFactory;
import org.ensor.fftmusings.pipeline.ChannelSelector;
import org.ensor.fftmusings.pipeline.Pipeline;

/**
 *
 * @author jona
 */
public class PCAFactor {
    public static void main(String[] args) throws Exception {
        args = new String[2];
        args[0] = "data/wav/";
        args[1] = "data/pca/smiths-fft-30.pca";
        
        if (args.length != 2) {
            System.out.println("Usage: PCADCTData foo.wav foo.pca");
            System.exit(1);
        }
        int sampleSize = 512;
        String inputFilename = args[0];
        String pcaFilename = args[1];
        double maxVariance = 0.3;
        
        try (ICloseableIterator<AudioSample[]> wavFileIterator = WAVFileIterator.create(inputFilename, sampleSize)) {
            new Pipeline(new ChannelSelector(AudioSample.class, 0))
                    .add(new FFTOverlap.Forward(sampleSize*2))
                    .add(new PCAFactorProcessorFFTMagnitude(new PCAFactory(sampleSize, maxVariance), pcaFilename))
                    .execute(wavFileIterator);
        }
    }
}
