/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.autoencoder;

import java.util.List;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 *
 * @author jona
 */
public class LayerwiseIterator implements DataSetIterator {

    private final DataSetIterator mLastIterator;
    private final MultiLayerNetwork mLastNetwork;
    private final int mLastLayerSize;
    private final int mMiniBatchSize;
    
    LayerwiseIterator(DataSetIterator aLastIterator, MultiLayerNetwork aLastNetwork, int aLastLayerSize, int aMiniBatchSize) {
        mLastIterator = aLastIterator;
        mLastNetwork = aLastNetwork;
        mLastLayerSize = aLastLayerSize;
        mMiniBatchSize = aMiniBatchSize;
    }
    
    
    @Override
    public DataSet next(int num) {
        DataSet lastLayerData = mLastIterator.next(num);
        
        INDArray nextInput = Layer.ModelProcessor.activateSelectedLayers(mLastNetwork, 0, 0, lastLayerData.getFeatures());
        return new DataSet(nextInput, nextInput.dup());
    }

    @Override
    public int totalExamples() {
        return mLastIterator.totalExamples();
    }

    @Override
    public int inputColumns() {
        return mLastLayerSize;
    }

    @Override
    public int totalOutcomes() {
        return mLastLayerSize;
    }

    @Override
    public boolean resetSupported() {
        return mLastIterator.resetSupported();
    }

    @Override
    public boolean asyncSupported() {
        return mLastIterator.asyncSupported();
    }

    @Override
    public void reset() {
        mLastIterator.reset();
    }

    @Override
    public int batch() {
        return mLastIterator.batch();
    }

    @Override
    public int cursor() {
        return mLastIterator.cursor();
    }

    @Override
    public int numExamples() {
        return mLastIterator.numExamples();
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        mLastIterator.setPreProcessor(preProcessor);
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return mLastIterator.getPreProcessor();
    }

    @Override
    public List<String> getLabels() {
        return mLastIterator.getLabels();
    }

    @Override
    public boolean hasNext() {
        return mLastIterator.hasNext();
    }

    @Override
    public DataSet next() {
        return next(mMiniBatchSize);
    }

}
