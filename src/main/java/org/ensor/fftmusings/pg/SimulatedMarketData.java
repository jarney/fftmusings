/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.pg;

import java.util.Random;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author jona
 */
public class SimulatedMarketData {

    private double phase0;
    private double freq0 = 40;
    private double t = 0;
    private Random mRNG;
    
    SimulatedMarketData(Random rng) {
        mRNG = rng;
        phase0 = (mRNG.nextDouble()-0.5) * Math.PI*2;
    }
    
    public void update(Holdings holdings) {
        
        double newPrice = 20 + 20 * Math.sin(t / freq0*2*Math.PI + phase0);
        t += 1;
        
        holdings.setCurrencyPrice("BTC", newPrice, 2);
    }
    
}
