/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.audio;

import org.ensor.fftmusings.pca.PCATransformer;
import org.ensor.fftmusings.pipeline.IProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author jona
 */
public class PCATransformProcessorDCTForward implements IProcessor<AudioDCTData, INDArray> {

    private final PCATransformer mPCA;
    
    public PCATransformProcessorDCTForward(PCATransformer pca) {
        mPCA = pca;
    }

    @Override
    public void begin() {
    }

    @Override
    public INDArray process(AudioDCTData input) {
        return mPCA.forward(Nd4j.create(input.mSamples));
    }

    @Override
    public void end() {
    }
    
}
