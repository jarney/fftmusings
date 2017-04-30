/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package org.ensor.fftmusings.data;

/**
 *
 * @author jona
 */
public class Sample {
    private double[] mSamples;
    public Sample(int size) {
        mSamples = new double[size];
    }
    public int size() {
        return mSamples.length;
    }
    public double[] getData() {
        return mSamples;
    }
}
