package org.ensor.fftmusings.statistics;

import java.util.Random;

/**
 *
 * @author jarney
 */
public class GaussianDistribution extends DistributionBase {
    private double mMu;
    private double mSigma;
    private double mVariance;
    private double mExponentFactor;
    private double mNormalFactor;
    private static final double SQRT_TWO_PI = Math.sqrt(2 * Math.PI);
    
    public GaussianDistribution(double aMu, double aSigma) {
        mMu = aMu;
        mSigma = aSigma;
        derivedComponents();
    }

    public GaussianDistribution(int i, double d, int i0) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    private final void derivedComponents() {
        mVariance = mSigma * mSigma;
        mExponentFactor = -1 / (2 * mVariance);
        mNormalFactor = 1 / (SQRT_TWO_PI * mSigma);
    }
    
    public void setMean(double aMean) {
        mMu = aMean;
    }
    public void setStandardDeviation(double aStandardDeviation) {
        mSigma = aStandardDeviation;
        derivedComponents();
    }
    public void setVariance(double aVariance) {
        mVariance = Math.sqrt(mVariance);
        derivedComponents();
    }
    
    public double getMean() {
        return mMu;
    }
    public double getStandardDeviation() {
        return mSigma;
    }
    public double getVariance() {
        return mVariance;
    }
    
    public double sample(Random rng) {
        return mSigma * gaussianSample(rng) + mMu;
    }
    
    public double gaussianSample(Random rng) {
        double r, x, y;

        // find a uniform random point (x, y) inside unit circle
        do {
           x = 2.0 * rng.nextDouble() - 1.0;
           y = 2.0 * rng.nextDouble() - 1.0;
           r = x*x + y*y;
        } while (r > 1 || r == 0);    // loop executed 4 / pi = 1.273.. times on average
                                      // http://en.wikipedia.org/wiki/Box-Muller_transform

        // apply the Box-Muller formula to get standard Gaussian z    
        double z = x * Math.sqrt(-2.0 * Math.log(r) / r);
        return z;
    }
    
    public double getProbabilityDensity(double x) {
        double d = (x - mMu);
        d = d*d;
        double probability = 
                Math.exp(d * mExponentFactor) * mNormalFactor;
        return probability;
    }
    
    public double getExponentPart(double x) {
        double d = (x - mMu);
        d = d*d;
        return Math.exp(d * mExponentFactor);
    }
    
    public double getNormalPart(double x) {
        return mNormalFactor;
    }
    
    
}
