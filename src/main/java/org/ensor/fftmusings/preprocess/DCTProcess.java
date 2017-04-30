/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.preprocess;

import org.ensor.fftmusings.data.Sample;
import org.jtransforms.dct.DoubleDCT_1D;

/**
 *
 * @author jona
 */
public class DCTProcess {
    
    public static final int WINDOW_SIZE = 1024;
    
    private DoubleDCT_1D mDCT;
    
    
    DCTProcess(int windowSize) {
        mDCT = new DoubleDCT_1D(WINDOW_SIZE);
    }
    
    public void forward(Sample s) {
        mDCT.forward(s.getData(), true);
    }
    public void reverse(Sample s) {
        mDCT.inverse(s.getData(), true);
    }
}
