/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.audio;

import java.io.IOException;
import org.ensor.fftmusings.pca.PCAFactory;
import org.ensor.fftmusings.pca.PCATransformer;
import org.ensor.fftmusings.pipeline.IProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author jona
 * @param <T>
 */
public abstract class PCAFactorProcessorBase<T> implements IProcessor<T, T>  {
    private final PCAFactory mPCAFactory;
    private final String mPCAFilename;
    
    public PCAFactorProcessorBase(PCAFactory aPCAFactory, String aPCAFilename) {
        mPCAFactory = aPCAFactory;
        mPCAFilename = aPCAFilename;
    }
    
    @Override
    public void begin() {
    }

    @Override
    public final T process(T input) {
        processPCA(input);
        return input;
    }
    
    public abstract void processPCA(T input);
    

    protected final void addPoint(double[] data) {
        mPCAFactory.addPoint(data);
    }
    protected final void addPoint(INDArray data) {
        mPCAFactory.addPoint(data);
    }
    
    @Override
    public void end() {
        PCATransformer transformer = null;
        transformer = mPCAFactory.createTransform();
        System.out.println("Transformer dimensions : " + transformer.getDimensions());
        try {
            PCAFactory.write(transformer, mPCAFilename);
        } catch (IOException ex) {
            throw new RuntimeException("Exception writing PCA matrix to file", ex);
        }
        
    }
}

