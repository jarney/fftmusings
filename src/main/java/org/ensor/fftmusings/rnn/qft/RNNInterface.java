/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.rnn.qft;

import java.util.Random;
import org.ensor.fftmusings.data.QuantizedSpectrum;
import org.ensor.fftmusings.preprocess.FFTProcess;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author jona
 */
public class RNNInterface {
    
    private Random mRNG;
    
    public RNNInterface(Random rng) {
        mRNG = rng;
    }
    public INDArray toRNN(QuantizedSpectrum qs) {
        INDArray input = Nd4j.zeros(FFTProcess.FFT_WINDOW_SIZE/8*
                (QuantizedSpectrum.MAGNITUDE_QUANTA));

        for (int i = 0; i < FFTProcess.FFT_WINDOW_SIZE/8; i++) {
            int index = i * QuantizedSpectrum.MAGNITUDE_QUANTA;
            int m = qs.getMagnitudeQuantized(i);
            input.putScalar(index + m, 1.0);
        }
        input = input.div(FFTProcess.FFT_WINDOW_SIZE/8);
        return input;
    }
    
    public void dumpOutput(INDArray output) throws Exception {
        int[] shape = output.shape();
        if (shape.length != 2) {
            throw new Exception("Invalid shape, should be vector");
        }
        if ((shape[1] % (QuantizedSpectrum.MAGNITUDE_QUANTA)) != 0) {
            throw new Exception("Length should be a multiple of MAGNITUDE_QUANTA");
        }
        int fftSize = shape[1] / (QuantizedSpectrum.MAGNITUDE_QUANTA) * 8;
        QuantizedSpectrum qs = new QuantizedSpectrum(fftSize);
        
        for (int i = 0; i < qs.size(); i++) {
            if (i < qs.size()/8) {
                System.out.println("Bucket " + i);
                double bucketTotal = 0;
                int index = i * (QuantizedSpectrum.MAGNITUDE_QUANTA);
                for (int j = 0; j < QuantizedSpectrum.MAGNITUDE_QUANTA; j++) {
                    System.out.println(output.getDouble(index + j) * qs.size()/8);
                    bucketTotal += output.getDouble(index + j) * qs.size()/8;
                }
                System.out.println("Total: " + bucketTotal);
            }
        }
    }
    
    public QuantizedSpectrum toQS(INDArray output) throws Exception {
        int[] shape = output.shape();
        if (shape.length != 2) {
            throw new Exception("Invalid shape, should be vector");
        }
        if ((shape[1] % (QuantizedSpectrum.MAGNITUDE_QUANTA)) != 0) {
            throw new Exception("Length should be a multiple of MAGNITUDE_QUANTA");
        }
        int fftSize = shape[1] / (QuantizedSpectrum.MAGNITUDE_QUANTA) * 8;
        QuantizedSpectrum qs = new QuantizedSpectrum(fftSize);

        int divisor = qs.size()/8;
        if (FFTProcess.FFT_WINDOW_SIZE/8 != divisor) {
            throw new Exception("Length of window does not match quanta");
        }
        
        output = output.mul(FFTProcess.FFT_WINDOW_SIZE/8);
        
        for (int i = 0; i < qs.size(); i++) {
            if (i < qs.size()/8) {
                int index = i * (QuantizedSpectrum.MAGNITUDE_QUANTA);

                double cum = 0;
                double dice = mRNG.nextDouble();
                int m = 0;
                for (int j = 0; j < QuantizedSpectrum.MAGNITUDE_QUANTA; j++) {
                    cum += output.getDouble(index + j);
                    if (cum >= dice) {
                        m = j;
                        break;
                    }
                }

                qs.setSample(i, m, 0);
            }
            else {
                qs.setSample(i, 0, 0);
            }
        }
        
        return qs;
    }
            
}
