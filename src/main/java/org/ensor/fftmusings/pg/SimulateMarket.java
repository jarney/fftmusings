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
        
        Holdings holdings = new Holdings();
        holdings.setAmountOf("USD", 1000);
        holdings.setCurrencyPrice("USD", 1, 2);
        holdings.setCurrencyPrice("BTC", 2, 2);

        Random rng = new Random();
        
        PolicyRandom policy = new PolicyRandom(rng);
        
        double totalValue = holdings.totalValue();
        System.out.println("Total value of holdings " + totalValue);

        SimulatedMarketData simulatedMarketData = new SimulatedMarketData(rng);
        
        for (int i = 0; i < 100; i++) {

            // Update the value of our holdings based on current
            // prevailing market prices.
            policy.updateState(holdings);
            
            // Perform any trades based on our policy network.
            IAction action = policy.getAction();
            System.out.println("Performing action " + action.getName());
            action.perform(holdings);

            // The market needs to evolve now.  Let the market tell us
            // what is going on next.
            simulatedMarketData.update(holdings);
            

            totalValue = holdings.totalValue();
            
            System.out.println("Total: " +
                formatter.format(totalValue) + " " + 
                " usd " + 
                formatter.format(holdings.getAmountOf("USD")) + 
                " btc " + 
                formatter.format(holdings.getAmountOf("BTC")) + 
                " @ " + 
                formatter.format(holdings.getMarketPriceOf("BTC"))
                );
        }
    }
    
}
