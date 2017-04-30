/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.atrain;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Random;
import org.ensor.fftmusings.audio.QuantizedVector;
import org.ensor.fftmusings.audio.RNNInput;
import org.ensor.fftmusings.pca.PCATransformer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author jona
 */
public class QFTIterator implements DataSetIterator
{
    
    private static final long serialVersionUID = -7287833919126626356L;
    
    private final int miniBatchSize;
    private final int nsamples;
    private final int numExamplesToFetch;
    private int examplesSoFar = 0;
    private final Random mRNG;
    private final PCATransformer mPCA;
    private final int mQuanta;
    private final PrintStream mOutput;
    
    public QFTIterator(Random rng,
            PCATransformer pca,
            int nQuanta,
            int aMiniBatchSize,
            int iterationsPerBatch,
            int samplesPerBatch,
            PrintStream output
    ) {
        mRNG = rng;
        mPCA = pca;
        mQuanta = nQuanta;
        miniBatchSize = aMiniBatchSize;
        numExamplesToFetch = miniBatchSize * iterationsPerBatch;
        nsamples = samplesPerBatch;
        mOutput = output;
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
        if (examplesSoFar + num > numExamplesToFetch) {
            throw new NoSuchElementException();
        }
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
        final INDArray input = Nd4j.zeros(new int[]{num, columns, nsamples-1});
        final INDArray labels = Nd4j.zeros(new int[]{num, columns, nsamples-1});

        long batchStartTime = System.currentTimeMillis();
        
        for (int curIdx = 0; curIdx < num; curIdx++) {
            int fileno = Math.abs(mRNG.nextInt());
            fileno = fileno % 95;
            fileno = 20;
            File f = new File("data/qft/" + fileno + ".qft");
            long startTime = System.currentTimeMillis();
            List<QuantizedVector> fileData = new ArrayList<>();
            try (FileInputStream fis = new FileInputStream(f)) {
                try (DataInputStream dis = new DataInputStream(fis)) {
                    int songLength = dis.readInt();
                    for (int i = 0; i < songLength; i++) {
                        fileData.add(QuantizedVector.read(dis));
                    }
                }
            }
            
            RNNInput.Forward rnnInputProcessor = new RNNInput.Forward();
            
            int offset = mRNG.nextInt(fileData.size()-nsamples);
            for (int sampleIndex = 0; sampleIndex < nsamples-1; sampleIndex++) {
                
                QuantizedVector inputQuant = fileData.get(sampleIndex + offset);
                QuantizedVector labelQuant = fileData.get(sampleIndex + offset + 1);
                inputQuant.setSample(0, (sampleIndex + offset) % 16);
                labelQuant.setSample(0, (sampleIndex + offset+1) % 16);
                
                INDArray inputRow = rnnInputProcessor.process(inputQuant);
                INDArray labelsRow = rnnInputProcessor.process(labelQuant);
                for (int dctProbabilityArray = 0; dctProbabilityArray < columns; dctProbabilityArray++) {
                    double inputPoint = inputRow.getDouble(dctProbabilityArray);
                    double outputPoint = labelsRow.getDouble(dctProbabilityArray);
                    input.putScalar(new int[]{curIdx, dctProbabilityArray, sampleIndex}, inputPoint);
                    labels.putScalar(new int[]{curIdx, dctProbabilityArray, sampleIndex}, outputPoint);
                }
            }
            long endTime = System.currentTimeMillis();
            mOutput.println("Sampling from file " + f.getName() + "(" + offset + " - " + (offset+nsamples) + ") took " + (endTime - startTime) + " ms");
        }
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
        return mQuanta;
    }

    @Override
    public int totalOutcomes() {
        return inputColumns();
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
