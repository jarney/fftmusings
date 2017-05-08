/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.audio;

/**
 * The magnitude spectrum contains the bottom half of the FFT
 * and only represents the "magnitude" portion of the signal.
 * The upper half of an FFT is equal to the complex conjugate of
 * the bottom half, so given an FFT, we only need to preserve the bottom half
 * of it.  In addition, when we use overlapping windows, we can reconstruct
 * some of the phase data through a convolution trick.
 * 
 * Hence, this data-structure represents the bottom-half of
 * an FFT's magnitude component and this is pretty much the minimal
 * representation of an FFT we can use to reconstruct a reasonable signal.
 * 
 * See also http://dsp.stackexchange.com/questions/9877/reconstruction-of-audio-signal-from-spectrogram
 * http://dsp.stackexchange.com/questions/3406/reconstruction-of-audio-signal-from-its-absolute-spectrogram/3410#3410
 * http://web.itu.edu.tr/~ibayram/AnDwt.pdf
 * 
 * @author jona
 */
public class MagnitudeSpectrum {
    public double[] mMagnitude;
    public double[] mPhase;

    public MagnitudeSpectrum(int aSampleSize) {
        mMagnitude = new double[aSampleSize];
        mPhase = new double[aSampleSize];
    }
    
    public MagnitudeSpectrum() {
        mMagnitude = null;
    }
}
