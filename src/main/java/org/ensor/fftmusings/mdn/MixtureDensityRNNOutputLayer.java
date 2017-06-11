/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.mdn;

import java.util.Collection;
import java.util.Map;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BaseOutputLayer;
import org.deeplearning4j.nn.conf.layers.InputTypeUtil;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.util.Dropout;
import org.deeplearning4j.util.LayerValidation;
import org.deeplearning4j.util.TimeSeriesUtils;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMax;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

/**Recurrent Neural Network Output Layer.<br>
 * Handles calculation of gradients etc for various objective functions.<br>
 * Functionally the same as OutputLayer, but handles output and label reshaping
 * automatically.<br>
 * Input and output activations are same as other RNN layers: 3 dimensions with shape
 * [miniBatchSize,nIn,timeSeriesLength] and [miniBatchSize,nOut,timeSeriesLength] respectively.
 * @author Alex Black
 * @see BaseOutputLayer, OutputLayer
 */
public class MixtureDensityRNNOutputLayer extends BaseOutputLayer  {
    
    protected MixtureDensityRNNOutputLayer(Builder b) {
        super(b);
    }
    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<IterationListener> iterationListeners, int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        LayerValidation.assertNInNOutSet("RnnOutputLayer", getLayerName(), layerIndex, getNIn(), getNOut());

        OutputLayer ret
                = new OutputLayer(conf);
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

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.RNN) {
            throw new IllegalStateException("Invalid input type for RnnOutputLayer (layer index = " + layerIndex +
                    ", layer name=\"" + getLayerName() + "\"): Expected RNN input, got " + inputType);
        }
        return InputType.recurrent(nOut);
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        if (inputType == null || inputType.getType() != InputType.Type.RNN) {
            throw new IllegalStateException("Invalid input type for RnnOutputLayer (layer name=\"" + getLayerName() + "\"): Expected RNN input, got " + inputType);
        }

        if (nIn <= 0 || override) {
            InputType.InputTypeRecurrent r = (InputType.InputTypeRecurrent) inputType;
            this.nIn = r.getSize();
        }
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        return InputTypeUtil.getPreprocessorForInputTypeRnnLayers(inputType, getLayerName());
    }


    public static class Builder extends org.deeplearning4j.nn.conf.layers.BaseOutputLayer.Builder<Builder> {
        private int mGaussians = 1;
        
        public Builder() {
        }

        /**
         * This method configures the number of gaussian mixtures to use
         * for each output component.  Each output will be modeled as a
         * linear combination of this many gaussian functions.
         * @param aGaussians Number of gaussians distributions to model.
         * @return This builder again for continued use.
         */
        public Builder gaussians(int aGaussians) {
            this.mGaussians = aGaussians;
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
//        @Override
//        public Builder lossFunction(ILossFunction aLoss) {
//            throw new IllegalArgumentException(
//                    "A Mixture density network uses " + 
//                    "a built-in loss function and will not " + 
//                    "function properly with any other" + 
//                    "loss function");
//        }
//        
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
        public Builder lossFunction(LossFunctions.LossFunction aLoss) {
            throw new IllegalArgumentException(
                    "A Mixture density network uses " + 
                    "a built-in loss function and will not " + 
                    "function properly with any other" + 
                    "loss function");
        }
        
        @Override
        @SuppressWarnings("unchecked")
        public MixtureDensityRNNOutputLayer build() {
            super.activation(Activation.IDENTITY.getActivationFunction());
            // The number of parameters out of the network
            // will be equal to 3 parameters per gaussian (alpha, mu, sigma)
            // multiplied by the number of gaussians per output
            // multiplied by the number of outputs.  We also need to inform
            // the loss function about how many mixtures and outputs to
            // expect so that it can calculate the mixtures appropriately.
            int labelWidth = this.nOut;
//            super.nOut(labelWidth);
//            super.lossFunction(LossFunctions.LossFunction.MSE);
            super.nOut((labelWidth + 2) * mGaussians);
            super.lossFunction(LossMixtureDensity.builder()
                    .gaussians(mGaussians)
                    .labelWidth(labelWidth)
                    .build());
            return new MixtureDensityRNNOutputLayer(this);
        }
    }
    
    public static class OutputLayer extends org.deeplearning4j.nn.layers.BaseOutputLayer<MixtureDensityRNNOutputLayer>  {

        public OutputLayer(NeuralNetConfiguration conf) {
            super(conf);
        }

        public OutputLayer(NeuralNetConfiguration conf, INDArray input) {
            super(conf, input);
        }
        
	private INDArray reshape3dTo2d(INDArray in){
		if( in.rank() != 3 ) throw new IllegalArgumentException("Invalid input: expect NDArray with rank 3");
		int[] shape = in.shape();
		if(shape[0]==1) return in.tensorAlongDimension(0,1,2).permutei(1,0);	//Edge case: miniBatchSize==1
		if(shape[2]==1) return in.tensorAlongDimension(0,1,0);	//Edge case: timeSeriesLength=1
		INDArray permuted = in.permute(0, 2, 1);	//Permute, so we get correct order after reshaping
        return permuted.reshape('f',shape[0] * shape[2], shape[1]);
	}
	
	private INDArray reshape2dTo3d(INDArray in, int miniBatchSize){
		if( in.rank() != 2 ) throw new IllegalArgumentException("Invalid input: expect NDArray with rank 2");
		//Based on: RnnToFeedForwardPreProcessor
		int[] shape = in.shape();
        if(in.ordering() != 'f') in = Shape.toOffsetZeroCopy(in, 'f');
		INDArray reshaped = in.reshape('f',miniBatchSize, shape[0] / miniBatchSize, shape[1]);
		return reshaped.permute(0, 2, 1);
	}

    @Override
    public Pair<Gradient,INDArray> backpropGradient(INDArray epsilon) {
        if(input.rank() != 3) throw new UnsupportedOperationException("Input is not rank 3");
        INDArray inputTemp = input;
        this.input = reshape3dTo2d(input);
    	Pair<Gradient,INDArray> gradAndEpsilonNext = _backpropGradient(epsilon);
        this.input = inputTemp;
    	INDArray epsilon2d = gradAndEpsilonNext.getSecond();
    	INDArray epsilon3d = reshape2dTo3d(epsilon2d, input.size(0));
		return new Pair<>(gradAndEpsilonNext.getFirst(),epsilon3d);
    }

    /**{@inheritDoc}
     */
    @Override
    public double f1Score(INDArray examples, INDArray labels) {
        if(examples.rank() == 3) examples = reshape3dTo2d(examples);
        if(labels.rank() == 3) labels = reshape3dTo2d(labels);
        return super.f1Score(examples, labels);
    }
    
    public INDArray getInput() {
        return input;
    }

    @Override
    public Layer.Type type() {
        return Layer.Type.RECURRENT;
    }
    
    @Override
    public INDArray preOutput(INDArray x, boolean training){
        setInput(x);
        return reshape2dTo3d(preOutput2d(training),input.size(0));
    }

    @Override
    protected INDArray preOutput2d(boolean training){
        if(input.rank() == 3 ) {
            //Case when called from RnnOutputLayer
            INDArray inputTemp = input;
            input = reshape3dTo2d(input);
            INDArray out = super.preOutput(input, training);
            this.input = inputTemp;
            return out;
        } else {
            //Case when called from BaseOutputLayer
            INDArray out = super.preOutput(input, training);
            return out;
        }
    }
    
    @Override
    protected INDArray getLabels2d(){
    	if(labels.rank()==3) return reshape3dTo2d(labels);
    	return labels;
    }

    @Override
    public INDArray output(INDArray input) {
        if(input.rank() != 3) throw new IllegalArgumentException("Input must be rank 3 (is: " + input.rank());
        //Returns 3d activations from 3d input
        setInput(input);
        return output(false);
    }

    @Override
    public INDArray output(boolean training){
        //Assume that input is 3d
        if(input.rank() != 3 ) throw new IllegalArgumentException("input must be rank 3");
        INDArray preOutput2d = preOutput2d(training);

        //if(conf.getLayer().getActivationFunction().equals("softmax")) {
        if(conf.getLayer().getActivationFn() instanceof ActivationSoftmax) {
            INDArray out2d = Nd4j.getExecutioner().execAndReturn(new SoftMax(preOutput2d));
            if(maskArray != null){
                out2d.muliColumnVector(maskArray);
            }
            return reshape2dTo3d(out2d,input.size(0));
        }

        if(training)
            applyDropOutIfNecessary(training);
        INDArray origInput = input;
        this.input = reshape3dTo2d(input);
        INDArray out = super.activate(true);
        this.input = origInput;
        if(maskArray != null){
            out.muliColumnVector(maskArray);
        }
        return reshape2dTo3d(out,input.size(0));
    }

    @Override
    public INDArray activate(boolean training) {
        if(input.rank() != 3) throw new UnsupportedOperationException("Input must be rank 3");
        INDArray b = getParam(DefaultParamInitializer.BIAS_KEY);
        INDArray W = getParam(DefaultParamInitializer.WEIGHT_KEY);
        if(conf.isUseDropConnect() && training) {
            W = Dropout.applyDropConnect(this, DefaultParamInitializer.WEIGHT_KEY);
        }

        INDArray input2d = reshape3dTo2d(input);

        //INDArray act2d = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getLayer().getActivationFunction(),
        //        input2d.mmul(W).addiRowVector(b)));
        INDArray act2d = conf.getLayer().getActivationFn().getActivation(input2d.mmul(W).addiRowVector(b),training);
        if(maskArray != null){
            act2d.muliColumnVector(maskArray);
        }
        return reshape2dTo3d(act2d, input.size(0));
    }

    @Override
    public void setMaskArray(INDArray maskArray) {
        if(maskArray != null && maskArray.size(1) != 1){
            maskArray = TimeSeriesUtils.reshapeTimeSeriesMaskToVector(maskArray);
        }
        this.maskArray = maskArray;
    }
    
        public Pair<Gradient,INDArray> _backpropGradient(INDArray epsilon) {
            Pair<Gradient,INDArray> pair = getGradientsAndDelta(preOutput2d(true));	//Returns Gradient and delta^(this), not Gradient and epsilon^(this-1)
            INDArray delta = pair.getSecond();

            INDArray epsilonNext = params.get(DefaultParamInitializer.WEIGHT_KEY).mmul(delta.transpose()).transpose();
            return new Pair<>(pair.getFirst(),epsilonNext);
        }
        
        /** Returns tuple: {Gradient,Delta,Output} given preOut */
        private Pair<Gradient,INDArray> getGradientsAndDelta(INDArray preOut) {
            ILossFunction lossFunction = layerConf().getLossFn();
            INDArray labels2d = getLabels2d();
//            if(labels2d.size(1) != preOut.size(1)){
//                throw new DL4JInvalidInputException("Labels array numColumns (size(1) = " + labels2d.size(1) + ") does not match output layer"
//                        + " number of outputs (nOut = " + preOut.size(1) + ")");
//            }
            //INDArray delta = lossFunction.computeGradient(labels2d, preOut, layerConf().getActivationFunction(), maskArray);
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
