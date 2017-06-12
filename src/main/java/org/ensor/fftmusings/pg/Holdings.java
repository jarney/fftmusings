/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.pg;

import java.util.HashMap;
import java.util.Map;

/**
 * The holdings are the amount of money (total assets) we have.
 * We can ask what is the total value of those holdings and we can
 * also perform transactions which manipiulate the holdings.
 *
 * @author jona
 */
public class Holdings {
    private final Map<String, Asset> mAssets;
    private final double commissionRate;
    private final double commissionFee;
    
    public Holdings() {
        mAssets = new HashMap<>();
        commissionRate = 0.025;
        commissionFee = 10;
    }
    
    public double totalValue() {
        double total = 0;
        
        for (Asset a : mAssets.values()) {
            total += getValueOf(a.getName());
        }
        
        return total;
    }
    
    /**
     * Changes our holdings by reducing the amount of USD
     * currency we have by amount*toUSDCost - commission.
     * Increases our holding in the target currency
     * by amount*toUSDCost.
     * @param currency
     * @param amount 
     */
    public void buyCurrency(String currency, double amount) {
        Asset usdAsset = getAsset("USD");
        Asset otherAsset = getAsset(currency);
        
        // Reduce USD by the 'commission'
        // plus the value of the amount of currency.
        double newUSDAmount = usdAsset.getAmount() - amount * otherAsset.getToUSDFactor()
                // We pay a transaction fee here
                // equal to a percentage of the transaction
                // amount plus a constant transaction fee.
                - commissionRate * Math.abs(amount) * otherAsset.getToUSDFactor()
                - commissionFee;
        double newOtherAmount = otherAsset.getAmount() + amount;
        
        // If we're out of money, we can't make a trade, so the holding
        // amount doesn't change.
        if (newUSDAmount < 0 || newOtherAmount < 0) {
            return;
        }
        
        usdAsset.setAmount(newUSDAmount);
        otherAsset.setAmount(newOtherAmount);
    }
    
    private Asset getAsset(String currency) {
        if (mAssets.containsKey(currency)) {
            return mAssets.get(currency);
        }
        Asset a = new Asset();
        a.setName(currency);
        a.setAmount(0);
        a.setToUSDFactor(1.0);
        mAssets.put(currency, a);
        return a;
    }
    
    /**
     * Returns the number of units of the given currency we have in our
     * holdings.
     * @param currency Currency amount to return.
     * @return Amount of currency held.
     */
    public double getAmountOf(String currency) {
        Asset asset = getAsset(currency);
        return asset.getAmount();
    }
    public double getMarketPriceOf(String currency) {
        Asset asset = getAsset(currency);
        return asset.getToUSDFactor();
    }

    public void setAmountOf(String currency, double amount) {
        Asset asset = getAsset(currency);
        asset.setAmount(amount);
    }
    
    /**
     * Returns the value (in USD) of the given currency.
     * @param currency Currency to return.
     * @return Equivalent USD value.
     */
    public double getValueOf(String currency) {
        Asset asset = getAsset(currency);
        return asset.getAmount() * asset.getToUSDFactor()
                - commissionRate * Math.abs(asset.getAmount()) * asset.getToUSDFactor()
                - commissionFee;
    }
    
    
    // It's symmetric!
    public void sellCurrency(String currency, double amount) {
        buyCurrency(currency, -amount);
    }
    /**
     * Set the market prices of the currency.
     * @param currency Which currency price to establish.
     * @param toUSDFactor How much is one unit of currency worth in USD?
     */
    public void setCurrencyPrice(String currency, double toUSDFactor) {
        Asset asset = getAsset(currency);
        asset.setToUSDFactor(toUSDFactor);
    }
}
