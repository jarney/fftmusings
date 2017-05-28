/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.audio.windows;

/**
 * This window doesn't work...
 * @author jona
 */
public class BlackmanHarrisWindow {
    public static double[] compute(int windowSize) {
        double[] window = new double[windowSize];
        
        double a0 = 0.35875;
        double a1 = 0.48829;
        double a2 = 0.14128;
        double a3 = 0.01168;
        
        for (int n = 0; n < windowSize; n++) {
            double howFar = (double)n / ((double)windowSize-1);
            window[n] =a0 -
                    a1 * Math.cos(Math.PI * 2 * howFar) +
                    a2 * Math.cos(Math.PI * 4 * howFar) -
                    a3 * Math.cos(Math.PI * 6 * howFar);
        }
        
        return window;
    }
}
