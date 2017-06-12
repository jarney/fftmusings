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
    
    public IAction getAction(Holdings holdings) {
        int nextBoolean = mRNG.nextInt(4);
        switch (nextBoolean) {
            case 0:
                return new ActionHold();
            case 1:
                return new ActionBuy("BTC", (mRNG.nextDouble() - 0.5) * 200);
            case 2:
                return new ActionBuy("LTC", (mRNG.nextDouble() - 0.5) * 200);
            case 3:
                return new ActionBuy("OTH", (mRNG.nextDouble() - 0.5) * 200);
        }
        return new ActionHold();
    }

    @Override
    public void nextRoundStart() {
    }

    @Override
    public void trainRound(double score) {
    }
    
    
}
