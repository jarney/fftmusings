/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.audio;

import java.util.Arrays;
import org.ensor.fftmusings.pipeline.IProcessor;
import org.jtransforms.fft.DoubleFFT_1D;

/**
 *
 * @author jona
 */
public class FFTReverse implements IProcessor<AudioFFTData, AudioSample> {
    private DoubleFFT_1D mFFT;
    private int mSampleSize;
    
    public FFTReverse() {
        mFFT = null;
        mSampleSize = 0;
    }
    

    @Override
    public void begin() {
    }

    @Override
    public AudioSample process(AudioFFTData spectrum) {
        if (spectrum.mSamples.length != mSampleSize*2) {
            mSampleSize = spectrum.mSamples.length/2;
            mFFT = new DoubleFFT_1D(mSampleSize);
        }
        
        AudioSample samples = new AudioSample(mSampleSize);
        double[] copy = Arrays.copyOf(spectrum.mSamples, spectrum.mSamples.length);
        mFFT.complexInverse(copy, true);
        for (int i = 0; i < mSampleSize; i++) {
            samples.mSamples[i] = copy[i*2];
        }
        return samples;
    }

    @Override
    public void end() {
    }
    
}
