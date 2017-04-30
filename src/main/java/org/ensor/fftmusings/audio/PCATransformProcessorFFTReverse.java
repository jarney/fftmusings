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
class PCATransformProcessorFFTReverse implements IProcessor<INDArray, MagnitudeSpectrum> {

    private final PCATransformer mPCA;
    
    public PCATransformProcessorFFTReverse(PCATransformer pca) {
        mPCA = pca;
    }

    @Override
    public void begin() {
    }

    @Override
    public MagnitudeSpectrum process(INDArray input) {
        INDArray reverse = mPCA.reverse(input);
        MagnitudeSpectrum dctData = new MagnitudeSpectrum();
        dctData.mMagnitude = new double[reverse.columns()];
        for (int i = 0; i < dctData.mMagnitude.length; i++) {
            dctData.mMagnitude[i] = reverse.getDouble(i);
        }
        return dctData;
    }

    @Override
    public void end() {
    }
    
}
