/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.rnn.qft;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.ensor.fftmusings.data.QuantizedSpectrum;
import org.ensor.fftmusings.preprocess.FFTProcess;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

/**
 * See also http://dsp.stackexchange.com/questions/9877/reconstruction-of-audio-signal-from-spectrogram
 * http://dsp.stackexchange.com/questions/3406/reconstruction-of-audio-signal-from-its-absolute-spectrogram/3410#3410
 * http://web.itu.edu.tr/~ibayram/AnDwt.pdf
 * 
 * @author jona
 */
public class SampleIterator implements DataSetIterator
{
    
    private static final long serialVersionUID = -7287833919126626356L;
    
    public static final int DATA_WIDTH = FFTProcess.FFT_WINDOW_SIZE/8 *
                (QuantizedSpectrum.MAGNITUDE_QUANTA);

    private final int miniBatchSize = 32;
    private final int nsamples = 512;
    private final int numExamplesToFetch = miniBatchSize * 50;
    private int examplesSoFar = 0;
    private final Random mRNG;
    
    public SampleIterator(Random rng) {
        mRNG = rng;
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
            return nextData;
        }
        catch (Exception ex) {
            throw new RuntimeException(ex);
        }
    }
    
    private DataSet nextThrows(int num) throws Exception {
        //Allocate space:
        
        
        final INDArray input = Nd4j.zeros(new int[]{num, DATA_WIDTH, nsamples-1});
        final INDArray labels = Nd4j.zeros(new int[]{num, DATA_WIDTH, nsamples-1});

        List<Thread> threads = new ArrayList<>();
        
        for (int curIdx = 0; curIdx < num; curIdx++) {
            final int k = curIdx;
            Thread t = new Thread(new Runnable() {
                @Override
                public void run() {
                    List<QuantizedSpectrum[]> qsList = new ArrayList<>();
                    int fileno = Math.abs(mRNG.nextInt());
                    fileno = fileno % 95;
                    File f = new File("data/qft/" + fileno + ".qft");
                    try (FileInputStream fis = new FileInputStream(f)) {
                        while (true) {
                            QuantizedSpectrum[] qs = new QuantizedSpectrum[2];
                            qs[0] = new QuantizedSpectrum(FFTProcess.FFT_WINDOW_SIZE);
                            qs[1] = new QuantizedSpectrum(FFTProcess.FFT_WINDOW_SIZE);
                            int rc;
                            rc = qs[0].read(fis);
                            if (rc == -1) {
                                break;
                            }
                            rc = qs[1].read(fis);
                            if (rc == -1) {
                                break;
                            }
                            qsList.add(qs);
                        }
                    } catch (IOException ex) {
                        Logger.getLogger(SampleIterator.class.getName()).log(Level.SEVERE, null, ex);
                    }


                    int offset = mRNG.nextInt(qsList.size()-nsamples);

                    RNNInterface rnni= new RNNInterface(mRNG);

                    System.out.println("File " + fileno + " Samples " + nsamples + " starting at " + offset);

                    for (int i = 0; i < nsamples-1; i++) {
                        int sampleIndex = i + offset;
                        QuantizedSpectrum[] qs = qsList.get(sampleIndex);
                        QuantizedSpectrum[] qsNext = qsList.get(sampleIndex+1);

                        INDArray inputRow = rnni.toRNN(qs[0]);
                        INDArray labelsRow = rnni.toRNN(qsNext[0]);
                        for (int j = 0; j < DATA_WIDTH; j++) {
                            double id = inputRow.getDouble(j);
                            double ld = labelsRow.getDouble(j);
                            input.putScalar(new int[]{k, j, i}, id);
                            labels.putScalar(new int[]{k, j, i}, ld);
                        }
                    }
                }
            });
            threads.add(t);
            t.start();
        }
        
        while (!threads.isEmpty()) {
            Thread curThread = threads.get(0);
            curThread.join();
            threads.remove(0);
        }
        
        
        System.out.println("Samples calculated");
        examplesSoFar += num;
        return new DataSet(input, labels);
    }

    @Override
    public int totalExamples() {
        return numExamplesToFetch;
    }

    @Override
    public int inputColumns() {
        return DATA_WIDTH;
    }

    @Override
    public int totalOutcomes() {
        return DATA_WIDTH;
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
