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
    void updateState(Holdings holdings);
    IAction getAction();
}
