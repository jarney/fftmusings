/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.pg;

import lombok.Data;

/**
 *
 * @author jona
 */
@Data
public class Asset {
    private String name;
    private double amount;
    private double toUSDFactor;
    public String toString() {
        return "(" + name + ":" + amount + ")";
    }
}
