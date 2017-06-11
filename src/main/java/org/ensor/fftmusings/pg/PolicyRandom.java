/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.pg;

import java.util.Random;

/**
 * This is a completely random policy.  It is only used to show how the
 * market works.
 * @author jona
 */
public class PolicyRandom implements IPolicy {
    
    private final Random mRNG;
    
    PolicyRandom(Random rng) {
        mRNG = rng;
    }
    
    public void updateState(Holdings holdings) {
        
    }
    
    public IAction getAction() {
        boolean nextBoolean = mRNG.nextBoolean();
        if (nextBoolean) {
            return new ActionHold();
        }
        else {
            return new ActionBuy("BTC", (mRNG.nextDouble() - 0.5) * 20);
        }
    }
    
    
}
