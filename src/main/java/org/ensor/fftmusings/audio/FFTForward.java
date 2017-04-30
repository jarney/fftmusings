/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.audio;

import org.ensor.fftmusings.pipeline.IProcessor;
import org.jtransforms.fft.DoubleFFT_1D;

/**
 *
 * @author jona
 */
public class FFTForward implements IProcessor<AudioSample, AudioFFTData> {

    private DoubleFFT_1D mFFT;
    private int mSampleSize;
    
    public FFTForward() {
        mFFT = null;
        mSampleSize = 0;
    }
    
    @Override
    public void begin() {
    }

    @Override
    public AudioFFTData process(AudioSample input) {
        if (input.mSamples.length != mSampleSize) {
            mSampleSize = input.mSamples.length;
            mFFT = new DoubleFFT_1D(mSampleSize);
        }

        AudioFFTData spectrum = new AudioFFTData(mSampleSize);
        for (int i = 0; i < input.size(); i++) {
            spectrum.mSamples[i*2] = input.mSamples[i];
            spectrum.mSamples[i*2+1] = 0;
        }
        mFFT.complexForward(spectrum.mSamples);
        return spectrum;
    }

    @Override
    public void end() {
    }
    
}
