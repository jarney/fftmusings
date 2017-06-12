/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.pg;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Random;

/**
 *
 * @author jona
 */
public class SimulateMarket {
    
    
    public static void main(String[] args) {
        
        NumberFormat formatter = new DecimalFormat("#0.0000");
        
        Random rng = new Random();
        
        //PolicyRandom policy = new PolicyRandom(rng);
        PolicyNN policy = new PolicyNN(rng);
        

        SimulatedMarketData simulatedMarketData = new SimulatedMarketData(rng);
        
        double lastTotalGain = 0;
        int iterations = 50000;
        for (int j = 0; j < iterations; j++) {
            double initialValue = 1000;
            Holdings holdings = new Holdings();
            holdings.setAmountOf("USD", initialValue);
            holdings.setCurrencyPrice("USD", 1);
            
            
            simulatedMarketData.update(holdings);
            
            double totalValue = holdings.totalValue();
            System.out.println("Total value of holdings " + totalValue);

            policy.nextRoundStart();
            
            for (int i = 0; i < 10 + rng.nextInt(20); i++) {

                // Update the value of our holdings based on current
                // prevailing market prices.
                policy.updateState(holdings);

                // Perform any trades based on our policy network.
                IAction action = policy.getAction(holdings);
//                System.out.println("Performing action " + action.getName());
                action.perform(holdings);

                // The market needs to evolve now.  Let the market tell us
                // what is going on next.
                simulatedMarketData.update(holdings);


                totalValue = holdings.totalValue();

//                System.out.println("Total: " +
//                    formatter.format(totalValue) + " " + 
//                    " usd " + 
//                    formatter.format(holdings.getAmountOf("USD")) + 
//                    " btc " + 
//                    formatter.format(holdings.getAmountOf("BTC")) + 
//                    " @ " + 
//                    formatter.format(holdings.getMarketPriceOf("BTC")) +
//                    " ltc " + 
//                    formatter.format(holdings.getAmountOf("LTC")) + 
//                    " @ " + 
//                    formatter.format(holdings.getMarketPriceOf("LTC")) +
//                    " oth " + 
//                    formatter.format(holdings.getAmountOf("OTH")) + 
//                    " @ " + 
//                    formatter.format(holdings.getMarketPriceOf("OTH"))
//                    );
            }
            double totalGain = holdings.totalValue() - initialValue;
            double winLossScore = totalGain - lastTotalGain;
            if (totalGain <= 0) {
                winLossScore = -1;
            }
            policy.trainRound(totalGain);
            System.out.println("Total gain is " + totalGain);
            lastTotalGain = totalGain;
        }
    }
    
}
