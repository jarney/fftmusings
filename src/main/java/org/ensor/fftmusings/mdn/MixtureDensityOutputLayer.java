/*
 * Copyright Jonathan S. Arney (2017), all rights reserved.
 * This may not be reproduced without permission for any purpose.
 */
package org.ensor.fftmusings.mdn;

import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.util.LayerValidation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.util.Collection;
import java.util.Map;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.ToString;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.conf.layers.BaseOutputLayer;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.factory.Nd4j;

/**
 * This object implements a mixture density network by providing a custom
 * output layer and corresponding cost function derived from the paper
 * "Mixture Density Networks" by Bishop (1994).
 */
@Data @NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class MixtureDensityOutputLayer extends BaseOutputLayer {

    protected MixtureDensityOutputLayer(Builder builder) {
    	super(builder);
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<IterationListener> iterationListeners, int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        LayerValidation.assertNInNOutSet("OutputLayer", getLayerName(), layerIndex, getNIn(), getNOut());

        OutputLayer ret = new OutputLayer(conf);
        ret.setListeners(iterationListeners);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    @Override
    public ParamInitializer initializer() {
        return DefaultParamInitializer.getInstance();
    }

    public static class Builder extends BaseOutputLayer.Builder<Builder> {

        private int mOutputs = 1;
        private int mMixturesPerOutput = 1;
        
        public Builder() {
        }

        /**
         * This method configures the number of gaussian mixtures to use
         * for each output component.  Each output will be modeled as a
         * linear combination of this many gaussian functions.
         * @param aMixturesPerOutput Number of mixtures to model for each output.
         * @return This builder again for continued use.
         */
        public Builder mixturesPerLabel(int aMixturesPerOutput) {
            this.mMixturesPerOutput = aMixturesPerOutput;
            return this;
        }
        
        /**
         * This method configures the number of output variables to model
         * with gaussian mixtures.  The number of gaussians to model
         * for each of them is configured separately.
         * @param aOutputs Number of output variables to model.
         * @return This builder again for continued use.
         */
        public Builder labelValues(int aOutputs) {
            mOutputs = aOutputs;
            return this;
        }

        /**
         * Setting the activation function for a mixture density output
         * layer is not supported.  The mixture density network has a very
         * specific output layer and cost function that must work together
         * or convergence cannot be achieved.  In particular, because the
         * mu (mean) of a distribution is an unbounded value, using any
         * activation function other than the identity activation
         * function does not allow the mean to converge on the correct
         * value in all cases.  Therefore, configuration of
         * the activation function is not permitted.
         * @param aActivation Activation function
         * @return This builder again for continued use.
         */
        @Override
        public Builder activation(Activation aActivation) {
            throw new IllegalArgumentException(
                    "A Mixture density network uses " + 
                    "a built-in identity activation function and will not " + 
                    "function properly with any other" + 
                    "activation function");
        }
        /**
         * Setting the activation function for a mixture density output
         * layer is not supported.  The mixture density network has a very
         * specific output layer and cost function that must work together
         * or convergence cannot be achieved.  In particular, because the
         * mu (mean) of a distribution is an unbounded value, using any
         * activation function other than the identity activation
         * function does not allow the mean to converge on the correct
         * value in all cases.  Therefore, configuration of
         * the activation function is not permitted.
         * @param aActivation Activation function
         * @return This builder again for continued use.
         */
        @Override
        public Builder activation(IActivation aActivation) {
            throw new IllegalArgumentException(
                    "A Mixture density network uses " + 
                    "a built-in identity activation function and will not " + 
                    "function properly with any other" + 
                    "activation function");
        }
        
        /**
         * Setting a loss function for a mixture density output layer
         * is not supported.  The mixture density output layer comes with
         * its own specialized loss function which is intended to work
         * hand-in-hand with the output layer because it makes assumptions
         * about the size and structure of the output layer.  Using custom
         * loss functions with the mixture density output layer is not
         * supported.
         * @param aLoss Loss function.
         * @return Builder for continued use.
         */
        @Override
        public Builder lossFunction(ILossFunction aLoss) {
            throw new IllegalArgumentException(
                    "A Mixture density network uses " + 
                    "a built-in loss function and will not " + 
                    "function properly with any other" + 
                    "loss function");
        }
        
        /**
         * Setting a loss function for a mixture density output layer
         * is not supported.  The mixture density output layer comes with
         * its own specialized loss function which is intended to work
         * hand-in-hand with the output layer because it makes assumptions
         * about the size and structure of the output layer.  Using custom
         * loss functions with the mixture density output layer is not
         * supported.
         * @param aLoss Loss function.
         * @return Builder for continued use.
         */
        @Override
        public Builder lossFunction(LossFunction aLoss) {
            throw new IllegalArgumentException(
                    "A Mixture density network uses " + 
                    "a built-in loss function and will not " + 
                    "function properly with any other" + 
                    "loss function");
        }
        
        
        @Override
        @SuppressWarnings("unchecked")
        public MixtureDensityOutputLayer build() {
            super.activation(Activation.IDENTITY);
            // The number of parameters out of the network
            // will be equal to 3 parameters per gaussian (alpha, mu, sigma)
            // multiplied by the number of gaussians per output
            // multiplied by the number of outputs.  We also need to inform
            // the loss function about how many mixtures and outputs to
            // expect so that it can calculate the mixtures appropriately.
            super.nOut(3 * mMixturesPerOutput * mOutputs);
            super.lossFunction(new MixtureDensityCost(mMixturesPerOutput, mOutputs));
            return new MixtureDensityOutputLayer(this);
        }
    }
    
    /**
     * This is the actual output layer to be used.  The biggest difference
     * between this output layer and the standard output layers is that
     * this output layer allows the width of the output vector to be
     * different than the width of the labels.  This is important because
     * the cost function requires this to be the case for mixture density
     * networks.
     */
    public static class OutputLayer extends org.deeplearning4j.nn.layers.BaseOutputLayer<MixtureDensityOutputLayer> {

        public OutputLayer(NeuralNetConfiguration conf) {
            super(conf);
        }

        public OutputLayer(NeuralNetConfiguration conf, INDArray input) {
            super(conf, input);
        }

        @Override
        public Pair<Gradient,INDArray> backpropGradient(INDArray epsilon) {
            Pair<Gradient,INDArray> pair = getGradientsAndDelta(preOutput2d(true));	//Returns Gradient and delta^(this), not Gradient and epsilon^(this-1)
            INDArray delta = pair.getSecond();

            INDArray epsilonNext = params.get(DefaultParamInitializer.WEIGHT_KEY).mmul(delta.transpose()).transpose();
            return new Pair<>(pair.getFirst(),epsilonNext);
        }

        /** Returns tuple: {Gradient,Delta,Output} given preOut */
        private Pair<Gradient,INDArray> getGradientsAndDelta(INDArray preOut) {
            ILossFunction lossFunction = layerConf().getLossFn();
            INDArray labels2d = getLabels2d();
            
            INDArray delta = lossFunction.computeGradient(labels2d, preOut, layerConf().getActivationFn(), maskArray);

            Gradient gradient = new DefaultGradient();

            INDArray weightGradView = gradientViews.get(DefaultParamInitializer.WEIGHT_KEY);
            INDArray biasGradView = gradientViews.get(DefaultParamInitializer.BIAS_KEY);

            Nd4j.gemm(input,delta,weightGradView,true,false,1.0,0.0);    //Equivalent to:  weightGradView.assign(input.transpose().mmul(delta));
            biasGradView.assign(delta.sum(0));

            gradient.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY,weightGradView);
            gradient.gradientForVariable().put(DefaultParamInitializer.BIAS_KEY,biasGradView);

            return new Pair<>(gradient, delta);
        }
    }
}

