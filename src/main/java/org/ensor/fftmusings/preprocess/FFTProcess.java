/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package org.ensor.fftmusings.preprocess;

import org.ensor.fftmusings.data.QuantizedSpectrum;
import org.ensor.fftmusings.data.Sample;
import org.jtransforms.fft.DoubleFFT_1D;

/**
 *
 * @author jona
 */
public class FFTProcess {

    public static final int FFT_WINDOW_SIZE = 1024;

    private final int mWindowSize;
    
    public FFTProcess(int windowSize) {
        mWindowSize = windowSize;
    }
    
    public QuantizedSpectrum quantizeFFT(Sample inputSample) throws Exception {
        int quantizedSize = inputSample.size()/2;
        if (quantizedSize != mWindowSize) {
            throw new Exception("Window size " + mWindowSize + " did not match input sample size");
        }
        
        double inputDouble[] = inputSample.getData();
        
        DoubleFFT_1D fft = new DoubleFFT_1D(inputSample.size());
        
        double[] fftData = new double[inputSample.size()*2];
        for (int i = 0; i < inputSample.size(); i++) {
            fftData[i*2] = inputDouble[i];
            fftData[i*2 + 1] = 0.0;
        }
        fft.complexForward(fftData);
        
        QuantizedSpectrum qs = new QuantizedSpectrum(quantizedSize);
        
        for (int i = 0; i < quantizedSize; i++) {
            double re = fftData[i*2] / inputSample.size();
            double im = fftData[i*2+1] / inputSample.size();
            double r = Math.min(Math.sqrt(re*re + im*im), 1.0);
            double phi = Math.atan2(im, re);
            if (i >= quantizedSize/8) {
                r = 0;
                phi = 0;
            }
            qs.setSample(i, r, phi);
        }
        
        return qs;
    }
    
    public Sample quantizedInverseFFT(QuantizedSpectrum qs) throws Exception {
        if (qs.size() != mWindowSize) {
            throw new Exception("Window size " + mWindowSize + " did not match given spectrum " + qs.size());
        }

        int fftSize = qs.size()*2;
        DoubleFFT_1D fft = new DoubleFFT_1D(fftSize);
        
        double[] fftData = new double[fftSize*2];
        
        for (int i = 0; i < qs.size(); i++) {
            double r = qs.getMagnitude(i);
            double phi = qs.getPhase(i);
            double re = r * Math.cos(phi);
            double im = r * Math.sin(phi);
            
            fftData[i*2] = re * fftSize;
            fftData[i*2+1] = im * fftSize;
        }
        // Ensure that the first half is the complex
        // conjugate of the second half.
        for (int i = 1; i < qs.size(); i++) {
            int idx0 = (i*2);
            int idx1 = (i*2+1);
            
            int idx = qs.size()*2;
            
            int idx2 = (idx - i)*2;
            int idx3 = (idx - i)*2+1;
            
            fftData[idx2] = fftData[idx0];
            fftData[idx3] = -fftData[idx1];
            
        }
        
        fft.complexInverse(fftData, true);
        
        Sample outputSample = new Sample(fftSize);
        double outputDouble[] = outputSample.getData();
        
        for (int i = 0; i < fftSize; i++) {
            double re = fftData[i*2];
            outputDouble[i] = re;
        }
        
        return outputSample;
    
    }
}
