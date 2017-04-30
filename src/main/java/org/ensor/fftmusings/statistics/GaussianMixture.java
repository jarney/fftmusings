package org.ensor.fftmusings.statistics;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import javafx.util.Pair;

/**
 *
 * @author jarney
 */
public class GaussianMixture extends DistributionBase {
    private List<Pair<Double, IDistribution>> mGaussians;
    private double mWeightNormalizer;
    
    public GaussianMixture() {
        mGaussians = new ArrayList<>();
        mWeightNormalizer = 0;
    }
    public void addDistribution(double factor, GaussianDistribution g) {
        mGaussians.add(new Pair(factor, g));
        mWeightNormalizer += factor;
    }
    
    public double sample(Random rng) {
        
        double dice = rng.nextDouble() * mWeightNormalizer;
        double total = 0;
        for (Pair<Double, IDistribution> m : mGaussians) {
            total += m.getKey();
            if (dice <= total) {
                return m.getValue().sample(rng);
            }
        }
        
        return 0;
    }
    
    @Override
    public double getProbabilityDensity(double x) {
        double p = 0;
        for (Pair<Double, IDistribution> d : mGaussians) {
            double f = d.getKey();
            IDistribution m = d.getValue();
            p += m.getProbabilityDensity(x) * f;
        }
        p /= mWeightNormalizer;
        
        return p;
    }
    

    
}
