/*
 * Copyright Jonathan S. Arney (2017), all rights reserved.
 * This may not be reproduced without permission for any purpose.
 */
package org.ensor.fftmusings.mdn;

import org.apache.commons.math3.util.Pair;
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
 * @author jona
 */
public class MixtureDensityCost implements ILossFunction {

    private final int mMixturesPerLabel;
    private final int mLabelsPerSample;
    private static final int mOutputPerGaussian = 3;
    
    /**
     * This class is a data holder for the mixture density
     * components for convenient manipulation.
     */
    public static class MixtureDensityComponents {
        public INDArray alpha;
        public INDArray mu;
        public INDArray sigma;
    }

    // Index variable helpers.
    private static int getAlphaIndex(int labelNumber, int mixtureNumber, int mixturesPerLabel) {
        return (labelNumber*mixturesPerLabel + mixtureNumber)*mOutputPerGaussian;
    }
    private static int getMuIndex(int labelNumber, int mixtureNumber, int mixturesPerLabel) {
        return (labelNumber*mixturesPerLabel + mixtureNumber)*mOutputPerGaussian+1;
    }
    private static int getSigmaIndex(int labelNumber, int mixtureNumber, int mixturesPerLabel) {
        return (labelNumber*mixturesPerLabel + mixtureNumber)*mOutputPerGaussian+2;
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

        for (int j = 0; j < nLabelsPerSample; j++) {
            for (int k = 0; k < nMixturesPerLabel; k++) {
                INDArray pa = output.get(NDArrayIndex.all(), NDArrayIndex.point(getAlphaIndex(j, k, nMixturesPerLabel)));
                INDArray pm = output.get(NDArrayIndex.all(), NDArrayIndex.point(getMuIndex(j, k, nMixturesPerLabel)));
                INDArray ps = output.get(NDArrayIndex.all(), NDArrayIndex.point(getSigmaIndex(j, k, nMixturesPerLabel)));
                mdc.alpha.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(j), NDArrayIndex.point(k)}, pa);
                mdc.mu.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(j), NDArrayIndex.point(k)}, pm);
                mdc.sigma.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(j), NDArrayIndex.point(k)}, ps);
            }
        }
        
        // The alpha are extracted from the Zi of the network
        // using a softmax transformation to ensure that
        // they sum to 1 and are positive-definite.
        mdc.alpha = softmax(nSamples, nLabelsPerSample, nMixturesPerLabel, mdc.alpha);
        
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
     * TODO: Find a much more elegant way of performing this operation.
     * @param nSamples
     * @param nLabelsPerSample
     * @param nMixturesPerLabel
     * @param input
     * @return 
     */
    private static INDArray softmax(int nSamples, int nLabelsPerSample, int nMixturesPerLabel, INDArray input) {
        INDArray output = Nd4j.zeros(nSamples, nLabelsPerSample, nMixturesPerLabel);
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nLabelsPerSample; j++) {
                double smsum = 0;
                for (int k = 0; k < nMixturesPerLabel; k++) {
                    double a = Math.exp(input.getDouble(new int[]{i,j,k}));
                    smsum += a;
                }
                for (int k = 0; k < nMixturesPerLabel; k++) {
                    double a = input.getDouble(new int[]{i,j,k});
                    a = Math.exp(a);
                    output.putScalar(new int[]{i,j,k}, a/smsum);
                }
            }
        }
        return output;
    }
    
    
    public MixtureDensityCost(int aMixturesPerLabel, int aLabelsPerSample) {
        mMixturesPerLabel = aMixturesPerLabel;
        mLabelsPerSample = aLabelsPerSample;
    }
    
    /**
     * Computes the aggregate score as a sum of all of the individual scores of
     * each of the labels against each of the outputs of the network.
     * @param labels
     * @param preOutput
     * @param activationFn
     * @param mask
     * @param average
     * @return 
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

        INDArray labelsMinusMu = labelsMinusMu(labels, mdc.mu); // CHECK
        
        INDArray phi = phi(labelsMinusMu, mdc.sigma);

        // This computes pi_i, see Bishop equation (30).
        // TODO: Figure out the Nd4j index nonsense
        // for better efficiency here.
        INDArray pi = phi.mul(mdc.alpha);
        INDArray piDivisor = pi.sum(2); // CHECK
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nLabelsPerSample; j++) {
                for (int k = 0; k < mMixturesPerLabel; k++) {
                    pi.putScalar(i, j, k, pi.getDouble(i, j, k)/piDivisor.getDouble(i, j));
                }
            }
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
        
        // Place components of gradient into gradient holder.]
        // TODO: Figure out the ND4j nonsense here.
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nLabelsPerSample; j++) {
                for (int k = 0; k < mMixturesPerLabel; k++) {
                    int alphaIndex = getAlphaIndex(j, k, mMixturesPerLabel);
                    int muIndex = getMuIndex(j, k, mMixturesPerLabel);
                    int sigmaIndex = getSigmaIndex(j, k, mMixturesPerLabel);
                    gradient.putScalar(i, alphaIndex, dLdZAlpha.getDouble(i, j, k));
                    gradient.putScalar(i, muIndex, dLdZMu.getDouble(i, j, k));
                    gradient.putScalar(i, sigmaIndex, dLdZSigma.getDouble(i, j, k));
                }
            }
        }
        
        // Manual calculation of
        // gradient to act as a double-check
        // of the other gradient calculation.
        // This provides a good test of the function
        // to make sure the calculations are correct.
//        double dx = 0.001;
//        INDArray l = computeScoreArray(labelsInput, preOutput, activationFn, mask);
//        INDArray gradient2 = Nd4j.zeros(nSamples, preOutput.columns());
//        for (int j = 0; j < mMixturesPerLabel*3; j++) {
//            INDArray preOutputdx = preOutput.dup();
//            for (int i = 0; i < nSamples; i++) {
//                preOutputdx.putScalar(i, j, preOutput.getDouble(i, j)+dx);
//            }
//            INDArray ldL = computeScoreArray(labelsInput, preOutputdx, activationFn, mask);
//            INDArray g1 = ldL.sub(l).div(dx);
//            
//            
//            for (int i = 0; i < nSamples; i++) {
//                gradient2.putScalar(i, j, g1.getDouble(i));
//            }
//            
//        }
        
        
        
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
        INDArray phitimesalphasum = phitimesalpha.sum(2);
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
        
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < labelsPerSample; j++) {
                for (int k = 0; k < mMixturesPerLabel; k++) {
                    labelMinusMu.putScalar(i, j, k, labels.getDouble(i, j));
                }
            }
        }
        labelMinusMu = labelMinusMu.sub(mu);
        
        return labelMinusMu;
    }
    
    INDArray phi(INDArray labelMinusMu, INDArray sigma) {
        int[] shape = sigma.shape();

        int nSamples = shape[0];
        int labelsPerSample = shape[1];
        int nMixturesPerLabel = shape[2];
        
        Assert.isTrue(nMixturesPerLabel == mMixturesPerLabel, "Mixtures per label must be equal");
        
        INDArray SQRT_TWO_PI = Nd4j.zeros(nSamples, labelsPerSample, nMixturesPerLabel).add(Math.sqrt(2*Math.PI));

        
        // The following computes an array of scores, one for each
        // of the training samples.
        // Each score is equal to the negative log likelihood of a given label
        // falling within the distribution.
        // TODO: Extend from single gaussian to gaussian mixture.
        
        // 1/sqrt(2PIs^2) * e^((in-u)^2/2*s^2)
        INDArray variance = sigma.mul(sigma);
        
        INDArray normalPart = sigma.mul(SQRT_TWO_PI);

        INDArray exponentPart = Transforms.exp(
            labelMinusMu.mul(labelMinusMu).div(
                    variance.mul(-2)
            )
        );

        // This is Sum(a_i * phi_i(x,mu,sigma))
        // where the sum runs over all of the mixtures.
        INDArray likelihoods = 
                exponentPart.div(normalPart);
        return likelihoods;
        
        
    }
    
}
