/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.statistics;

/**
 *
 * @author jona
 */
public abstract class DistributionBase implements IDistribution {
    public double getProbability(double x1, double x2) {
        double startX = Math.min(x1, x2);
        double endX = Math.min(x1, x2);
        
        
        double y2 = getProbabilityDensity(endX);
        double y1 = getProbabilityDensity(startX);
        
        double startY = Math.min(y1, y2);
        double endY = Math.max(y1, y2);
        
        // Estimate as follows:
        //  /|  Top part is the little triangle = (endX-startX)*(endY-startY)/2
        // / |
        // | |  Bottom part is y1 * (endX-startX);
        // |_|
        
        // = startY * (endX - startX) + (endY - startY) * (endX - startX);
        // Factor out the (endX - startX) to get:
        // = (startY + (endY - startY)/2) * (endX - startX)
        return (startY + (endY - startY)/2) * (endX - startX);
    }

    /**
     * Returns the negative log likelihood of
     * the value.
     * @param x
     * @return 
     */
    @Override
    public double getNegativeLogLikelihood(double x) {
        return -Math.log(getProbabilityDensity(x));
    }
}
