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
public class AudioDCTData {
    public double[] mSamples;

    public AudioDCTData(int aSampleSize) {
        mSamples = new double[aSampleSize];
    }
    
    public AudioDCTData() {
        mSamples = null;
    }
}