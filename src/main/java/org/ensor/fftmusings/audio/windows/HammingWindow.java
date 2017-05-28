/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.audio.windows;

/**
 *
 * @author jona
 */
public class HammingWindow {
    public static double[] compute(int windowSize) {
        double[] window = new double[windowSize];
        
        for (int n = 0; n < windowSize; n++) {
            double howFar = (double)n / ((double)windowSize-1);
            window[n] =0.53836 - 0.46164 * Math.cos(Math.PI * 2 * howFar);
        }
        
        return window;
    }
}
