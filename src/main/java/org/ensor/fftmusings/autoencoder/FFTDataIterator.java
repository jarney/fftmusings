/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.autoencoder;

import java.io.File;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Random;
import org.ensor.fftmusings.audio.FFTOverlap;
import org.ensor.fftmusings.audio.MagnitudeSpectrum;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author jona
 */
public class FFTDataIterator implements DataSetIterator {
    private static final long serialVersionUID = -7287833919126626356L;
    
    private int miniBatchSize;
    private final int numExamplesToFetch;
    private int examplesSoFar = 0;
    private final Random mRNG;
    private final PrintStream mOutput;
    private final List<MagnitudeSpectrum> data = new ArrayList<>();
    private Layer.FFTDToINDArray mConv;
    
    public FFTDataIterator(Random rng,
            int aMiniBatchSize,
            int iterationsPerBatch,
            PrintStream output
    ) throws Exception {
        mRNG = rng;
        miniBatchSize = aMiniBatchSize;
        numExamplesToFetch = miniBatchSize * iterationsPerBatch;
        mOutput = output;
        mConv = new Layer.FFTDToINDArray();
        int fileno = Math.abs(mRNG.nextInt());
        fileno = fileno % 95;
        fileno = 20;
        File f = new File("data/fft/" + fileno + ".fftd");

        try (FFTOverlap.Reader dctReader = FFTOverlap.createReader(f.getAbsolutePath())) {
            while (dctReader.hasNext()) {
                data.add(dctReader.next());
            }
        }
        
    }
    
    @Override
    public boolean hasNext() {
        return examplesSoFar < numExamplesToFetch;
    }

    @Override
    public DataSet next() {
        return next(miniBatchSize);
    }

    @Override
    public DataSet next(int num) {
        try {
            DataSet nextData = nextThrows(num);
            examplesSoFar += num;
            return nextData;
        }
        catch (Exception ex) {
            throw new RuntimeException(ex);
        }
    }
    
    private DataSet nextThrows(int num) throws Exception {
        //Allocate space:
        
        int columns = inputColumns();
        INDArray input = Nd4j.zeros(new int[]{num, columns});
        INDArray labels = Nd4j.zeros(new int[]{num, columns});

        long batchStartTime = System.currentTimeMillis();
        for (int curIdx = 0; curIdx < num; curIdx++) {
            
            int offset = mRNG.nextInt(data.size());

            MagnitudeSpectrum sample = data.get(offset);
            input.putRow(curIdx, mConv.process(sample));
        }
            
        labels = input.dup();
        long batchEndTime = System.currentTimeMillis();
        
        mOutput.println("Samples calculated : total time " + (batchEndTime - batchStartTime) + " ms");
        return new DataSet(input, labels);
    }

    @Override
    public int totalExamples() {
        return numExamplesToFetch;
    }

    @Override
    public int inputColumns() {
        return 512*2;
    }

    @Override
    public int totalOutcomes() {
        return 512*2;
    }

    public void setMiniBatchSize(int aMiniBatchSize) {
        miniBatchSize = aMiniBatchSize;
    }
    
    @Override
    public void reset() {
        examplesSoFar = 0;
    }

    @Override
    public int batch() {
        return 1;
    }

    @Override
    public int cursor() {
        return examplesSoFar;
    }

    @Override
    public int numExamples() {
        return numExamplesToFetch;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
    }

    @Override
    public List<String> getLabels() {
        return null;
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return false;
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return null;
    }
}
