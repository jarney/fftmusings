/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.audio;

import org.ensor.fftmusings.audio.RNNInput.Reverse.SampleStyle;
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
public class PCAFactorExperiment {
    public static void main(String[] args) throws Exception {
        args = new String[2];
        args[0] = "data/wav/20.wav";
        args[1] = "data/pca/experiment.pca";
        
        if (args.length != 2) {
            System.out.println("Usage: PCADCTData foo.wav foo.pca");
            System.exit(1);
        }
        int sampleSize = 512;
        String inputFilename = args[0];
        String pcaFilename = args[1];
        double maxVariance = 0.9;
        int maxVariancePCT = (int)(100 * maxVariance);
        double dctThreshold = 0.0;
        int nQuanta = 16;
        
        try (ICloseableIterator<AudioSample[]> wavFileIterator = WAVFileIterator.create(inputFilename, sampleSize)) {
            new Pipeline(new ChannelSelector(AudioSample.class, 0))
                    .add(new FFTOverlap.Forward(sampleSize*2))
                    .add(new PCAFactorProcessorFFTMagnitude(new PCAFactory(sampleSize, maxVariance), pcaFilename))
                    .execute(wavFileIterator);
        }
        
        PCATransformer pca = PCAFactory.read(pcaFilename);
        
        String outputFilename = "sample-" + sampleSize + "-" + pca.getDimensions() + "-" + maxVariancePCT + "-" + dctThreshold + ".wav";

        try (ICloseableIterator<AudioSample[]> wavFileIterator = WAVFileIterator.create(inputFilename, sampleSize)) {
            new Pipeline(new ChannelSelector(AudioSample.class, 0))
                    .add(new FFTOverlap.Forward(sampleSize*2))
                    .add(new FFTToPNG("fft.png", false))
                    .add(new PCATransformProcessorFFTForward(pca))
                    .add(new Quantize.Forward(nQuanta, 0, 1, true))
                    .add(new RNNInput.Forward())
                    .add(new RNNInput.Reverse(nQuanta, SampleStyle.LARGEST, null))
                    .add(new Quantize.Reverse(nQuanta, 0, 1, true))
                    .add(new PCATransformProcessorFFTReverse(pca))
                    .add(new FFTOverlap.Reverse(sampleSize*2))
                    .add(new ChannelDuplicator(AudioSample.class, 2))
                    .add(WAVFileWriter.create(outputFilename))
                    .execute(wavFileIterator);
        }

    }
}
