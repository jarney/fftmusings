/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.audio;

/**
 *
 * @author jona
 */
public class AudioSample {
    protected double[] mSamples;
    
    protected AudioSample() {
        mSamples = null;
    }
    
    public AudioSample(int aSampleSize) {
        mSamples = new double[aSampleSize];
        for (int i = 0; i < aSampleSize; i++) {
            mSamples[i] = 0;
        }
    }
    public int size() {
        return mSamples.length;
    }
    
}
