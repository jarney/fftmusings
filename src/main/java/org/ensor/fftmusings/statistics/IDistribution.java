package org.ensor.fftmusings.statistics;

import java.util.Random;

/**
 * This method provides a model of a probability
 * distribution and predicts the probability density
 * for that distribution type and allows one to sample
 * numbers from that distribution assuming a given
 * uniform random distribution.
 *
 * @author jarney
 */
public interface IDistribution {
    /**
     * Returns a random sample from a probability distribution
     * assuming a uniformly distributed random number source.
     * @param rng
     * @return 
     */
    double sample(Random rng);
    
    /**
     * Returns the probability of a variable
     * taking a value between start and end.
     * @param startX
     * @param endX
     * @return 
     */
    double getProbability(double startX, double endX);
    
    /**
     * Returns the probability density
     */
    double getProbabilityDensity(double x);
    
    double getNegativeLogLikelihood(double x);
}
