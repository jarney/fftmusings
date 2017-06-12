/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.pg;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import org.ensor.fftmusings.statistics.GaussianDistribution;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author jona
 */
public class SimulatedMarketData {

    private Map<String, StocasticPriceModel> priceModels;
    private double t = 0;
    private Random mRNG;
    
    SimulatedMarketData(Random rng) {
        mRNG = rng;
        priceModels = new HashMap<>();
        for (int i = 0; i < 50; i++) {
            priceModels.put("CUR" + i,
                new StocasticPriceModel(mRNG, 
                    40,
                    (mRNG.nextDouble()-0.5) * Math.PI*2,
                    10 // How much randomness to +/- to price.
                )
            );
        }
    }
    
    public void update(Holdings holdings) {
        t += 1;
        
        for (Map.Entry<String, StocasticPriceModel> pm : priceModels.entrySet()) {
            holdings.setCurrencyPrice(pm.getKey(), pm.getValue().getPrice(t));
        }
    }
    
}
