/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.audio;

import org.ensor.fftmusings.pca.PCATransformer;
import org.ensor.fftmusings.pipeline.IProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author jona
 */
public class PCATransformProcessorDCTReverse implements IProcessor<INDArray, AudioDCTData> {

    private final PCATransformer mPCA;
    
    public PCATransformProcessorDCTReverse(PCATransformer pca) {
        mPCA = pca;
    }

    @Override
    public void begin() {
    }

    @Override
    public AudioDCTData process(INDArray input) {
        INDArray reverse = mPCA.reverse(input);
        AudioDCTData dctData = new AudioDCTData();
        dctData.mSamples = new double[reverse.columns()];
        for (int i = 0; i < dctData.mSamples.length; i++) {
            dctData.mSamples[i] = reverse.getDouble(i);
        }
        return dctData;
    }

    @Override
    public void end() {
    }
    
}
