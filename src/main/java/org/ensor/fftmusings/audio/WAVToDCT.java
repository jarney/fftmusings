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
import org.ensor.fftmusings.io.ICloseableIterator;
import org.ensor.fftmusings.pipeline.ChannelSelector;
import org.ensor.fftmusings.pipeline.Pipeline;

/**
 *
 * @author jona
 */
public class WAVToDCT {
    public static void main(String[] args) throws Exception {
        
        int sampleSize = 512;
        String inputDirectory = "data/wav";
        String outputDirectory = "data/dct";
        
        // Creating a pool of 16 threads to consume 8 cores.
        // About half the time spent processing is IO bound, so 16 threads
        // should keep 8 cores busy.
        ExecutorService executor = Executors.newFixedThreadPool(16);
        
        File dir = new File(inputDirectory);
        for (File inputFile : dir.listFiles()) {
            File outputFile = new File(outputDirectory + File.separator + inputFile.getName());
            System.out.println("Input " + inputFile.getAbsolutePath());
            System.out.println("Output " + outputFile.getAbsolutePath());
        
            executor.execute(new DCTProcess(inputFile, outputFile, sampleSize));
        }
        
        executor.shutdown();
        while (!executor.isTerminated()) {
            Thread.sleep(10);
        }
        System.out.println("Finished processing");
    }
    
    static class DCTProcess implements Runnable {
        
        private final int mSampleSize;
        private final String mInputFilename;
        private final String mOutputFilename;
        
        
        public DCTProcess(File inputFile, File outputFile, int sampleSize) {
            mSampleSize = sampleSize;
            mInputFilename = inputFile.getAbsolutePath();
            mOutputFilename = outputFile.getAbsolutePath().replace(".wav", ".dct");
        }
        
        public void run() {
            System.out.println("Starting " + mInputFilename);
            try (ICloseableIterator<AudioSample[]> wavFileIterator = WAVFileIterator.create(mInputFilename, mSampleSize)) {
                try (OutputStream os = new FileOutputStream(mOutputFilename)) {
                    try (DataOutputStream dos = new DataOutputStream(os)) {
                        new Pipeline(new ChannelSelector(AudioSample.class, 0))
                                .add(new DCT.Forward(false))
                                .add(new DCT.Write(dos))
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
