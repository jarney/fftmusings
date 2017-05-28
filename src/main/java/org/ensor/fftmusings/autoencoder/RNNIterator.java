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
import java.util.Random;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.ensor.fftmusings.audio.FFTOverlap;
import org.ensor.fftmusings.audio.MagnitudeSpectrum;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 *
 * @author jona
 */
public class RNNIterator implements DataSetIterator {
    private int miniBatchSize;
    private final int numExamplesToFetch;
    private int examplesSoFar = 0;
    private final Random mRNG;
    private final PrintStream mOutput;
    private final List<INDArray> data = new ArrayList<>();
    private Layer.FFTDToINDArray mConv;
    private MultiLayerNetwork mModel;

    public RNNIterator(
            MultiLayerNetwork aModel,
            Random rng,
            int aMiniBatchSize,
            int iterationsPerBatch,
            PrintStream output
    ) throws Exception {
        mModel = aModel;
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
                // Take the FFT data (1024 wide)
                MagnitudeSpectrum currentSample = dctReader.next();
                // Process it into an INDArray
                INDArray currentSampleIND = mConv.process(currentSample);
                // Use the top-half of the autoencoder to encode
                // the FFT into a learned feature-set.
                INDArray currentEncoded = mModel.activateSelectedLayers(0, mModel.getnLayers()/2-1, currentSampleIND);
                
                // This will be used as the input feature-set
                // for the RNN to learn a sequence on.
                data.add(currentEncoded);
            }
        }
        
    }
    
    @Override
    public DataSet next(int num) {
        //Allocate space:
        int exampleLength = 200;
        INDArray input = Nd4j.zeros(new int[]{num, inputColumns(), exampleLength});
        INDArray labels = Nd4j.zeros(new int[]{num, inputColumns(), exampleLength});

        int maxStartIdx = data.size() - exampleLength - 1;

        //Randomly select a subset of the file. No attempt is made to avoid overlapping subsets
        // of the file in the same minibatch
        
        
        for (int i = 0; i < num; i++) {
            int startIdx = (int) (mRNG.nextDouble() * maxStartIdx);
            for (int j = 0; j < exampleLength; j++) {
                INDArray currentSample = data.get(j + startIdx);
                INDArray nextSample = data.get(j + startIdx + 1);
                input.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(j)}, currentSample);
                labels.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(j)}, nextSample);
            }
        }
        
        examplesSoFar += num;
        return new DataSet(input, labels);
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
    public int totalExamples() {
        return numExamplesToFetch;
    }

    @Override
    public int inputColumns() {
        return 100;
    }

    @Override
    public int totalOutcomes() {
        return 100;
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
