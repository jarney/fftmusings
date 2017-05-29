/*
 * Copyright Jonathan S. Arney (2017), all rights reserved.
 * This may not be reproduced without permission for any purpose.
 */
package org.ensor.fftmusings.mdn;

import java.util.Random;
import org.apache.commons.math3.util.Pair;
import org.ensor.fftmusings.statistics.GaussianDistribution;
import org.ensor.fftmusings.statistics.GaussianMixture;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.io.Assert;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * This is a cost function associated with a mixture-density network.
 * For background, this is inspired by Bishop's work pioneering the mixture
 * density network.  The essence of the idea is that the cost function attempts
 * to model the output as if it were a mixture of gaussian probability
 * densities.  The network attempts to converge on a cost function which
 * involves the negative log likelihood of the output fitting a set of data
 * and estimates the "alpha" contribution to of each of the distributions
 * the 'mu' (mean) and 'sigma' (standard deviation) of each of the
 * distributions.
 * 
 * For a full description of the technique, refer to Bishop's work.
 * 
 * http://publications.aston.ac.uk/373/1/NCRG_94_004.pdf
 * 
 * The constructor for this object is the number of gaussian distributions to
 * estimate for each of the outputs.  Note that this means that the output
 * layer must provide n*3*mixtures output values in order to describe the
 * distributions of n numbers.  Please ensure that the size of the output layer
 * matches the number of mixtures provided in the constructor for the
 * cost function.
 * 
 * @author jarney
 */
public class MixtureDensityCost implements ILossFunction {

    private final int mMixturesPerLabel;
    private final int mLabelsPerSample;
    private static final double SQRT_TWO_PI = Math.sqrt(2*Math.PI);
    
    /**
     * This method constructs a mixture density cost function
     * which causes the network to learn a mixture of gaussian distributions
     * for each network output.  The network will learn the 'alpha' (weight
     * for each distribution), the 'mu' or 'mean' of each distribution,
     * and the 'sigma' (standard-deviation) of the mixture.  Together,
     * this distribution can be sampled according to the probability density
     * learned by the network.
     */
    public MixtureDensityCost(int aMixturesPerLabel, int aLabelsPerSample) {
        mMixturesPerLabel = aMixturesPerLabel;
        mLabelsPerSample = aLabelsPerSample;
    }
    
    /**
     * This class is a data holder for the mixture density
     * components for convenient manipulation.
     * These are organized as rank-3 matrices with shape
     * [nSamples, nLabelsPerSample, nMixturesPerLabel]
     * and refer to the 'alpha' (weight of that gaussian), 'mu' (mean for that
     * gaussian), and 'sigma' (standard-deviation for that gaussian).
     */
    public static class MixtureDensityComponents {
        public INDArray alpha;
        public INDArray mu;
        public INDArray sigma;
    }

    // This method extracts the "alpha", "mu", and "sigma" from the
    // output of the neural network.
    // This is done manually, but it should ultimately be done
    // through Nd4j operations in order to increase performance.
    public static MixtureDensityComponents extractComponents(INDArray output, int nSamples, int nLabelsPerSample, int nMixturesPerLabel) {
        MixtureDensityComponents mdc = new MixtureDensityComponents();
        mdc.alpha = Nd4j.zeros(nSamples, nLabelsPerSample, nMixturesPerLabel);
        mdc.mu = Nd4j.zeros(nSamples, nLabelsPerSample, nMixturesPerLabel);
        mdc.sigma = Nd4j.zeros(nSamples, nLabelsPerSample, nMixturesPerLabel);
        
        // Output is 2 dimensional (samples, labels)
        //
        // Reorganize these.
        // alpha = samples, 0-output/3
        // mu = samples, output/3-output*2/3
        // sigma = samples, output*2/3-output
        // Alpha is then sub-divided through reshape by mixtures per label and samples.

        INDArray pa2 = output.get(NDArrayIndex.all(), NDArrayIndex.interval(0, nMixturesPerLabel*nLabelsPerSample));
        INDArray pm2 = output.get(NDArrayIndex.all(), NDArrayIndex.interval(nMixturesPerLabel*nLabelsPerSample, nMixturesPerLabel*nLabelsPerSample*2));
        INDArray ps2 = output.get(NDArrayIndex.all(), NDArrayIndex.interval(nMixturesPerLabel*nLabelsPerSample*2, nMixturesPerLabel*nLabelsPerSample*3));
        mdc.alpha.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.all()}, pa2);
        mdc.mu.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.all()}, pm2);
        mdc.sigma.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.all()}, ps2);
        mdc.alpha = mdc.alpha.reshape('f', nSamples, nLabelsPerSample, nMixturesPerLabel);
        mdc.mu = mdc.mu.reshape('f', nSamples, nLabelsPerSample, nMixturesPerLabel);
        mdc.sigma = mdc.sigma.reshape('f', nSamples, nLabelsPerSample, nMixturesPerLabel);

        // Alpha is a softmax because
        // the alpha should all sum to 1 for a given gaussian mixture.
        mdc.alpha = softmax(nMixturesPerLabel, mdc.alpha);
        
        // Mu comes directly from the network as an unmolested value.
        // Note that this effectively means that the output layer of
        // the network should have an activation function at least as large as
        // the expected values.  It is best for the output
        // layer to be an IDENTITY activation function.
        //mdc.mu = mdc.mu;
        
        // Sigma comes from the network as an exponential in order to
        // ensure that it is positive-definite.
        mdc.sigma = Transforms.exp(mdc.sigma);

        return mdc;
    }
    
    /**
     * This is a 'softmax' where each output yi = e^x divided by
     * the sum of all 'sum(yi)'.
     * @param nMixturesPerLabel Number of gaussian mixtures for each label.
     * @param input Input alpha to softmax on.
     * @return The softmax of the input (as a rank-3 tensor where the 
     *         shape is [nSamples, nLabelsPerSample, nMixturesPerLabel].
     */
    private static INDArray softmax(int nMixturesPerLabel, INDArray input) {
        INDArray output = Transforms.exp(input);
        INDArray expsum = output.sum(2);
        for (int k = 0; k < nMixturesPerLabel; k++) {
            output.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(k)}).divi(expsum);
        }
        return output;
    }
    
    /**
     * Sample from the network output.
     * @param networkOutput
     * @return 
     */
    public INDArray sampleFromNetwork(Random rng, INDArray networkOutput) {
        MixtureDensityComponents mdc = extractComponents(networkOutput, 1, mLabelsPerSample, mMixturesPerLabel);
        INDArray o = Nd4j.zeros(mLabelsPerSample);
        
        for (int i = 0; i < mLabelsPerSample; i++) {
            GaussianMixture mixture = new GaussianMixture();
            for (int j = 0; j < mMixturesPerLabel; j++) {
                double alpha = mdc.alpha.getDouble(0, i, j);
                double mu = mdc.mu.getDouble(0, i, j);
                double sigma = mdc.sigma.getDouble(0, i, j);
                GaussianDistribution d = new GaussianDistribution(mu, sigma);
                mixture.addDistribution(alpha, d);
            }
            o.putScalar(i, mixture.sample(rng));
        }
        
        
        return o;
    }
    
    /**
     * Computes the aggregate score as a sum of all of the individual scores of
     * each of the labels against each of the outputs of the network.  For
     * the mixture density network, this is the negative log likelihood that
     * the given labels fall within the probability distribution described by
     * the mixture of gaussians of the network output.
     * @param labels Labels to score against the network.
     * @param preOutput Output of the network (before activation function has been called).
     * @param activationFn Activation function for the network.
     * @param mask Mask to be applied to labels (not used for MDN).
     * @param average Whether or not to return an average instead of a total score (not used).
     * @return Returns a single double which corresponds to the total score of all label values.
     */
    @Override
    public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
        // The score overall consists of the
        // sum of the negative log likelihoods for each
        // of the individual labels.
        double newValue = computeScoreArray(labels, preOutput, activationFn, mask).sumNumber().doubleValue();
        return newValue;
    }

    /**
     * This method returns the score for each of the given outputs against the
     * given set of labels.  For a mixture density network, this is done by
     * extracting the "alpha", "mu", and "sigma" components of each gaussian
     * and computing the negative log likelihood that the labels fall within
     * a linear combination of these gaussian distributions.  The smaller
     * the negative log likelihood, the higher the probability that the given
     * labels actually would fall within the distribution.  Therefore by
     * minimizing the negative log likelihood, we get to a position of highest
     * probability that the gaussian mixture explains the phenomenon.
     *
     * @param labels Labels give the sample output that the network should
     *               be trying to converge on.
     * @param preOutput The output of the last layer (before applying the activation function).
     * @param activationFn The activation function of the current layer.
     * @param mask Mask to apply to score evaluation (not supported for this cost function).
     * @return 
     */
    @Override
    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        int nSamples = labels.rows();
        INDArray output = activationFn.getActivation(preOutput.dup(), false);
        MixtureDensityComponents mdc = extractComponents(output, nSamples, mLabelsPerSample, mMixturesPerLabel);
        INDArray newCostArray = negativeLogLikelihood(labels, mdc.alpha, mdc.mu, mdc.sigma);
        return newCostArray;
        
    }

    /**
     * This method returns the gradient of the cost function with respect to the
     * output from the previous layer.  For this cost function, the gradient
     * is derived from Bishop's paper "Mixture Density Networks" (1994) which
     * gives an elegant closed-form expression for the derivatives with respect
     * to each of the output components.
     * @param labels
     * @param preOutput
     * @param activationFn
     * @param mask
     * @return 
     */
    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        int[] shape = labels.shape();
        
        int nSamples = shape[0];
        int nLabelsPerSample = shape[1];
        
        INDArray output = activationFn.getActivation(preOutput.dup(), false);
        
        MixtureDensityComponents mdc = extractComponents(output, nSamples, nLabelsPerSample, mMixturesPerLabel);
        
        INDArray gradient = Nd4j.zeros(nSamples, preOutput.columns());

        INDArray labelsMinusMu = labelsMinusMu(labels, mdc.mu);
        
        INDArray phi = phi(labelsMinusMu, mdc.sigma);

        // This computes pi_i, see Bishop equation (30).
        // TODO: Figure out the Nd4j index nonsense
        // for better efficiency here.
        INDArray pi = phi.mul(mdc.alpha);
        INDArray piDivisor = pi.sum(2);
        for (int k = 0; k < mMixturesPerLabel; k++) {
            pi.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(k)}).divi(piDivisor);
        }
        
        INDArray variance = mdc.sigma.mul(mdc.sigma);
        INDArray diffsquared = labelsMinusMu.mul(labelsMinusMu);

        // See Bishop equation (35)
        //INDArray dLdZAlpha = Nd4j.zeros(nSamples, nLabelsPerSample, mMixturesPerLabel); //mdc.alpha.sub(pi);
        INDArray dLdZAlpha = mdc.alpha.sub(pi);
        // See Bishop equation (39)
        INDArray dLdZMu = labelsMinusMu.div(variance).mul(-1).mul(pi);
        // See Bishop equation (38)
        INDArray dLdZSigma = (diffsquared.div(variance).sub(1)).mul(-1).mul(pi);
        
        // Place components of gradient into gradient holder.
        gradient.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(0, mMixturesPerLabel*nLabelsPerSample)}, dLdZAlpha);
        gradient.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(mMixturesPerLabel*nLabelsPerSample, mMixturesPerLabel*nLabelsPerSample*2)}, dLdZMu);
        gradient.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(mMixturesPerLabel*nLabelsPerSample*2, mMixturesPerLabel*nLabelsPerSample*3)}, dLdZSigma);
        
        INDArray gradients = activationFn.backprop(preOutput, gradient).getFirst();
        
        return gradients;
    }

    @Override
    public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
        
        double score = computeScore(labels, preOutput, activationFn, mask, average);
        
        INDArray gradient = computeGradient(labels, preOutput, activationFn, mask);
        
        Pair<Double, INDArray> returnCode = new Pair<>(score, gradient);
        return returnCode;
    }
    
    /**
     * This method returns an array consisting of each of the training samples,
     * for each label in each sample, the negative log likelihood of that
     * value falling within the given gaussian mixtures.
     * @param alpha
     * @param mu
     * @param sigma
     * @param labels
     * @return 
     */
    private INDArray negativeLogLikelihood(INDArray labels, INDArray alpha, INDArray mu, INDArray sigma) {
        INDArray labelsMinusMu = labelsMinusMu(labels, mu);
        INDArray phitimesalpha = phi(labelsMinusMu, sigma).mul(alpha);
        // phitimesalpha = See Bishop equation(22)
        INDArray phitimesalphasum = phitimesalpha.sum(2);
        
        // result = See Bishop(28,29)
        INDArray result = 
            Transforms.log(
                    phitimesalphasum
            ).negi();
        return result;
    }

    private INDArray labelsMinusMu(INDArray labels, INDArray mu) {
        // Now that we have the mixtures, let's compute the negative
        // log likelihodd of the label against the 
        int nSamples = labels.shape()[0];
        int labelsPerSample = labels.shape()[1];
        INDArray labelMinusMu = Nd4j.zeros(nSamples, labelsPerSample, mMixturesPerLabel);
        
        for (int k = 0; k < mMixturesPerLabel; k++) {
            labelMinusMu.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(k)}, labels);
        }
        labelMinusMu.subi(mu);
        
        return labelMinusMu;
    }
    
    /**
     * This method calculates 'phi' which is the probability
     * density function (see Bishop 23)
     * @param labelMinusMu This is the 'x-mu' term of the Gaussian distribution (distance between 'x' and the mean value of the distribution).
     * @param sigma This is the standard deviation of the Gaussian distribution.
     * @return This returns an array of shape [nsamples, nlabels, ndistributions] which contains the probability density (phi) for each of the
     *         samples * labels * distributions for the given x, sigma, mu.
     */
    INDArray phi(INDArray labelMinusMu, INDArray sigma) {
        int[] shape = sigma.shape();

        int nMixturesPerLabel = shape[2];
        
        Assert.isTrue(nMixturesPerLabel == mMixturesPerLabel, "Mixtures per label must be equal");
        
        // 1/sqrt(2PIs^2) * e^((in-u)^2/2*s^2)
        INDArray variance = sigma.mul(sigma);
        
        INDArray normalPart = sigma.mul(SQRT_TWO_PI);

        INDArray exponentPart = Transforms.exp(
            labelMinusMu.mul(labelMinusMu).div(
                    variance.mul(-2)
            )
        );

        // This is phi_i(x,mu,sigma)
        INDArray likelihoods = 
                exponentPart.div(normalPart);
        return likelihoods;
        
        
    }
    
}
