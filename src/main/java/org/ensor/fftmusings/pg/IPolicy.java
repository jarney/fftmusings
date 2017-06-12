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
public interface IPolicy {
    // Clear this round and make room for the
    // next round of training.
    void nextRoundStart();
    
    // Add current data to this round.
    void updateState(Holdings holdings);
    
    // Select an action for this round.
    IAction getAction(Holdings holdings);
    
    // Round is complete, train and update
    // policy weights for next round.
    void trainRound(double score);
}
