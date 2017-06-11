/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.pg;

/**
 *
 * @author jona
 */
public class ActionBuy implements IAction {

    private final String currency;
    private final double amount;
    
    public ActionBuy(String currency, double amount) {
        this.currency = currency;
        this.amount = amount;
    }
    
    public String getName() {
        return "buy " + currency + ":" + amount;
    }
    
    @Override
    public void perform(Holdings holdings) {
        holdings.buyCurrency(currency, amount);
    }
    
}
