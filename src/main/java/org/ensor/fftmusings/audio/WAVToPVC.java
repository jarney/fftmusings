/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.audio;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.ensor.fftmusings.autoencoder.Layer;
import org.ensor.fftmusings.io.ICloseableIterator;
import org.ensor.fftmusings.pipeline.ChannelDuplicator;
import org.ensor.fftmusings.pipeline.ChannelSelector;
import org.ensor.fftmusings.pipeline.Pipeline;

/**
 *
 * @author jona
 */
public class WAVToPVC {
    public static void main(String[] args) throws Exception {
        
        int sampleSize = 512;
        String inputDirectory = "data/wav";
        String outputDirectory = "data/fft";
        String reconsDirectory = "data/wav2";
        
        // Creating a pool of 16 threads to consume 8 cores.
        // About half the time spent processing is IO bound, so 16 threads
        // should keep 8 cores busy.
        ExecutorService executor = Executors.newFixedThreadPool(1);
        
        File dir = new File(inputDirectory);
//        for (File inputFile : dir.listFiles()) {
            File inputFile = new File("data/wav/20.wav");
            System.out.println("Input " + inputFile.getAbsolutePath());
        
            executor.execute(new FFTProcess(inputFile, outputDirectory, reconsDirectory, sampleSize));
//        }
        
        executor.shutdown();
        while (!executor.isTerminated()) {
            Thread.sleep(10);
        }
        System.out.println("Finished processing");
    }
    
    static class FFTProcess implements Runnable {
        
        private final int mSampleSize;
        private final String mInputFilename;
        private final String mOutputFilename;
        private final String mReconsFilename;
        private final String mPNGFilename;
        
        
        public FFTProcess(File inputFile, String outputDirectory, String reconsDirectory, int sampleSize) {
            mSampleSize = sampleSize;
            mInputFilename = inputFile.getAbsolutePath();
            mOutputFilename = outputDirectory + File.separator + inputFile.getName().replace(".wav", ".fftd");
            mReconsFilename = reconsDirectory + File.separator + inputFile.getName();
            mPNGFilename = outputDirectory + File.separator + inputFile.getName().replace(".wav", ".fftd.png");
        }
        
        public void run() {
            int fftWindowSize = mSampleSize*2;
            System.out.println("Starting " + mInputFilename);
            try (ICloseableIterator<AudioSample[]> wavFileIterator = WAVFileIterator.create(mInputFilename, mSampleSize)) {
                try (OutputStream os = new FileOutputStream(mOutputFilename)) {
                    try (DataOutputStream dos = new DataOutputStream(os)) {
                        new Pipeline(new ChannelSelector(AudioSample.class, 0))
                                .add(new FFTOverlap.ForwardPhaseDelta(fftWindowSize))
                                .add(new FFTOverlap.NormalizeToHearing(true, 11025.0))
                                .add(new FFTOverlap.Write(dos))
                                .add(new Layer.FFTDToINDArray())
                                .add(new Layer.ToFFTPNG(mPNGFilename, true))
                                .add(new Layer.INDArrayToFFTD())
                                .add(new FFTOverlap.NormalizeToHearing(false, 11025.0))
                                .add(new FFTOverlap.ReversePhaseDelta(fftWindowSize))
                                .add(new ChannelDuplicator(AudioSample.class, 2))
                                .add(WAVFileWriter.create(mReconsFilename))
                                .execute(wavFileIterator);
                    }
                }
            }
            catch (Exception ex) {
                throw new RuntimeException("Could not process file " + mInputFilename, ex);
            }
            System.out.println("Finished " + mInputFilename);
        }
    }
}
