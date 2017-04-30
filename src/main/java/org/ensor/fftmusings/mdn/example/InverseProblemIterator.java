/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.mdn.example;

import java.util.List;
import java.util.NoSuchElementException;
import java.util.Random;
import org.ensor.fftmusings.statistics.GaussianDistribution;
import org.ensor.fftmusings.statistics.GaussianMixture;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author jona
 */
public class InverseProblemIterator implements DataSetIterator {

    private final int iterationsPerBatch = 32;
    private final int miniBatchSize = 1000;
    private final int numExamplesToFetch = iterationsPerBatch * miniBatchSize;
    private int examplesSoFar = 0;
    private final Random mRNG;
    private final int mMixturesPerLabel;
    
    public InverseProblemIterator(int nMixturesPerLabel) {
        mRNG = new Random();
        mMixturesPerLabel = nMixturesPerLabel;
    }
    
    @Override
    public DataSet next() {
        return next(miniBatchSize);
    }

    @Override
    public boolean hasNext() {
        return (examplesSoFar < numExamplesToFetch);
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
    
    public DataSet nextThrows(int num) throws Exception {
        
        INDArray input = Nd4j.zeros(num, inputColumns());
        INDArray output = Nd4j.zeros(num, totalOutcomes());

        GaussianDistribution gd = new GaussianDistribution(0, .01);
        GaussianDistribution gd2 = new GaussianDistribution(0, 0.01);
        GaussianMixture m = new GaussianMixture();
        m.addDistribution(0.5, gd);
        m.addDistribution(0.5, gd2);
        
        for (int i = 0; i < num; i++) {
            double x = mRNG.nextDouble() - 0.5;
            //double r = mRNG.nextDouble();
            //double y = Math.sin(0.75*x)*7.0+x*0.5+r*1.0;
            gd.setMean(x*10.0);
            gd.setStandardDeviation(.2);

            gd2.setMean(-x*5.0);
            gd2.setStandardDeviation(.2);

            double y = m.sample(mRNG);
        
            // Y is a set of samples from a gaussian distribution
            // where 'x' is the mean of the distribution.  This should
            // allow us to build a simple neatly fittable dataset.

            gd.setMean(x*6.0);
            gd.setStandardDeviation(.7);

            gd2.setMean(-x*14.0);
            gd2.setStandardDeviation(.3);

            double z = m.sample(mRNG);

            input.putScalar(new int[]{i, 0}, x*10);
            output.putScalar(new int[]{i, 0}, y);
            output.putScalar(new int[]{i, 1}, z);
//            output.putScalar(new int[]{i, 1}, x*10);
//            output.putScalar(new int[]{i, 2}, gd.getMean());
//            output.putScalar(new int[]{i, 3}, gd.getStandardDeviation());
//            output.putScalar(new int[]{i, 4}, gd2.getMean());
//            output.putScalar(new int[]{i, 5}, gd2.getStandardDeviation());

        }

        
        return new DataSet(input, output);
    }

    @Override
    public int totalExamples() {
        return numExamplesToFetch;
    }

    @Override
    public int inputColumns() {
        return 1;
    }

    @Override
    public int totalOutcomes() {
        return 2;
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
    public void reset() {
        examplesSoFar = 0;
    }

    @Override
    public int batch() {
        return 1;
    }

    @Override
    public int cursor() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public int numExamples() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public List<String> getLabels() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    public static void main(String[] args) {
        InverseProblemIterator it = new InverseProblemIterator(1);
        
        int j = 0;
        while (it.hasNext()) {
            if (j == 8) break;
            DataSet next = it.next();
            INDArray features = next.getFeatures();
            INDArray labels = next.getLabels();
            
            for (int i = 0; i < features.rows(); i++) {
                System.out.println("" + features.getDouble(i) + "\t" + labels.getDouble(i, 0));
            }
            j++;
        }
    }
    
    
}
