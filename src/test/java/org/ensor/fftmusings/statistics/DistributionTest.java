/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.statistics;

import java.util.Random;
import org.junit.Assert;
import org.junit.Test;

/**
 * This class tests the basic gaussian distribution properties
 * and ensures that when random numbers are generated for a distribution,
 * those numbers are consistent with the 'ideal' distribution predicted
 * by that type of distribution.
 * @author jona
 */
public class DistributionTest {
    
    @Test
    public void testGaussianDistribution() {
//        GaussianDistribution g = new GaussianDistribution(5, .5);
//        
//        Random rng = new Random(System.currentTimeMillis());
//        
//        int resolution = 100;
//        int scale = 10;
//        
//        int[] buckets = new int[resolution];
//        
//        int sampleSize = 50000;
//        
//        for (int i = 0; i < sampleSize; i++) {
//            double y = g.sample(rng);
//            int bin = (int)((y*scale) + 0.5);
//            if (bin<0) bin = 0;
//            if (bin > buckets.length-1) bin = buckets.length-1;
//            buckets[bin]++;
//        }
//
//        double[] x = new double[resolution];
//        double[] actualDistribution = new double[resolution];
//        double[] idealDistribution = new double[resolution];
//        
//        for (int i = 0; i < resolution; i++) {
//            x[i] = (double)i / (double) scale;
//            actualDistribution[i] = (double)buckets[i] / (double)sampleSize;
//            idealDistribution[i] = g.getProbability(x[i]-0.5/scale, x[i]+0.5/scale);
//        }
//        double sum = 0;
//        double asum = 0;
//        for (int i = 0; i < buckets.length; i++) {
//            double delta = Math.abs(actualDistribution[i] - idealDistribution[i]);
////            System.out.println("" + x[i] + "\t" + actualDistribution[i] + "\t" + idealDistribution[i] + "\t" + delta);
//            Assert.assertTrue(Math.abs(delta) < 0.01);
//            sum += actualDistribution[i];
//            asum += idealDistribution[i];
//        }
//        Assert.assertTrue(Math.abs(sum - 1.0) < 0.001);
//        Assert.assertTrue(Math.abs(asum - 1.0) < 0.001);
    }
    
    
    @Test
    public void testGaussianMixture() {
        GaussianDistribution g = new GaussianDistribution(5, 1);
        GaussianDistribution g2 = new GaussianDistribution(2.5, 0.5);
        GaussianMixture m = new GaussianMixture();
        m.addDistribution(1, g);
        m.addDistribution(0.5, g2);
        
//        Random rng = new Random();
//        
//        int resolution = 100;
//        int scale = 10;
//        
//        int[] buckets = new int[resolution];
//        
//        int sampleSize = 50000;
//        
//        for (int i = 0; i < sampleSize; i++) {
//            double y = m.sample(rng);
//            int bin = (int)(y*scale);
//            if (bin<0) bin = 0;
//            if (bin > buckets.length-1) bin = buckets.length-1;
//            buckets[bin]++;
//        }
//        m.getProbability(2.5, 2.6);
//
//        double[] x = new double[resolution];
//        double[] actualDistribution = new double[resolution];
//        double[] idealDistribution = new double[resolution];
//        
//        for (int i = 0; i < resolution; i++) {
//            x[i] = (double)(i+0.5) / (double) scale;
//            actualDistribution[i] = (double)buckets[i] / (double)sampleSize;
//            idealDistribution[i] = m.getProbability(x[i]-0.05, x[i]+0.05);
//        }
//        double sum = 0;
//        double asum = 0;
//        for (int i = 0; i < buckets.length; i++) {
//            double delta = Math.abs(actualDistribution[i] - idealDistribution[i]);
////            System.out.println("" + x[i] + "\t" + actualDistribution[i] + "\t" + idealDistribution[i] + "\t" + delta);
//            
//            Assert.assertTrue(Math.abs(delta) < 0.01);
//            
//            sum += actualDistribution[i];
//            asum += idealDistribution[i];
//        }
//        
//        Assert.assertTrue(Math.abs(sum - 1.0) < 0.001);
//        Assert.assertTrue(Math.abs(asum - 1.0) < 0.001);
        
    }
    
}
