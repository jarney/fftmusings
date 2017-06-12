/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.pg;

import java.util.Random;
import org.ensor.fftmusings.statistics.GaussianDistribution;

/**
 *
 * @author jona
 */
public class StocasticPriceModel {
    private double btcPhase;
    private double btcFrequency = 40;
    private double btcScale = 10;
    private GaussianDistribution btcDistribution;
    private Random mRNG;
    
    public StocasticPriceModel(
            Random rng,
            double frequency, double phase, double stddev) {
        mRNG = rng;
        btcFrequency = frequency;
        btcPhase = phase;
        btcDistribution = new GaussianDistribution(0, stddev);
    }
    
    public double getPrice(double t) {
        
        // Assume price moves as a sinusoid
        // plus/minus some amount of random noise.
        
        double newPrice = btcScale + btcScale * Math.sin(t / btcFrequency*2*Math.PI + btcPhase) + 
                btcDistribution.sample(mRNG) + btcDistribution.getStandardDeviation()*2;
        if (newPrice < 0) {
            newPrice = 0.1;
        }
        return newPrice;
    }
}
