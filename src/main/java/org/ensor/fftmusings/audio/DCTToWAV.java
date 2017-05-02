/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.audio;

import java.io.File;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.ensor.fftmusings.pipeline.ChannelDuplicator;
import org.ensor.fftmusings.pipeline.ChannelSelector;
import org.ensor.fftmusings.pipeline.Pipeline;

/**
 *
 * @author jona
 */
public class DCTToWAV {
    public static void main(String[] args) throws Exception {
        
        String inputDirectory = "data/dct";
        String outputDirectory = "data/wav2";
        
        // Creating a pool of 16 threads to consume 8 cores.
        // About half the time spent processing is IO bound, so 16 threads
        // should keep 8 cores busy.
        ExecutorService executor = Executors.newFixedThreadPool(16);
        
        File dir = new File(inputDirectory);
        for (File inputFile : dir.listFiles()) {
            File outputFile = new File(outputDirectory + File.separator + inputFile.getName());
            System.out.println("Input " + inputFile.getAbsolutePath());
            System.out.println("Output " + outputFile.getAbsolutePath());
        
            executor.execute(new DCTProcess(inputFile, outputFile));
        }
        
        executor.shutdown();
        while (!executor.isTerminated()) {
            Thread.sleep(10);
        }
        System.out.println("Finished processing");
    }
    
    static class DCTProcess implements Runnable {
        
        private final String mInputFilename;
        private final String mOutputFilename;
        
        
        public DCTProcess(File inputFile, File outputFile) {
            mInputFilename = inputFile.getAbsolutePath();
            mOutputFilename = outputFile.getAbsolutePath().replace(".dct", ".wav");
        }
        
        public void run() {
            System.out.println("Starting " + mInputFilename);
            
            try (DCT.Reader wavFileIterator = DCT.createReader(mInputFilename)) {
                new Pipeline(new DCT.Reverse(false))
                    .add(new ChannelDuplicator(AudioSample.class, 2))
                    .add(WAVFileWriter.create(mOutputFilename))
                    .execute(wavFileIterator);
            }
            catch (Exception ex) {
                throw new RuntimeException("Could not process file " + mInputFilename, ex);
            }
            System.out.println("Finished " + mInputFilename);
        }
    }
}
