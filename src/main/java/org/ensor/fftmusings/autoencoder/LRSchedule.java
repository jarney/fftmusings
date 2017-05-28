/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.autoencoder;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 *
 * @author jona
 */
public class LRSchedule {
    private int mIteration;
    private double mLearningRate;
    
    @JsonProperty("iteration")
    public void setIteration(int iteration) {
        mIteration = iteration;
    }
    
    @JsonProperty("iteration")
    public int getIteration() {
        return mIteration;
    }
    
    @JsonProperty("lr")
    public void setLR(double aLR) {
        mLearningRate = aLR;
    }
    
    @JsonProperty("lr")
    public double getLR() {
        return mLearningRate;
    }
}
