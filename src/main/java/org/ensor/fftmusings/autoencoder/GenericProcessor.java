/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.autoencoder;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.ensor.fftmusings.pipeline.IProcessor;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 *
 * @author jona
 */
public class GenericProcessor {

    
    public static class Process implements IProcessor<INDArray, INDArray> {

        private MultiLayerNetwork mModel;
        
        public Process(MultiLayerNetwork model) {
            mModel = model;
        }
        
        @Override
        public void begin() {
        }

        @Override
        public INDArray process(INDArray input) {
            return mModel.activateSelectedLayers(0, mModel.getnLayers()-1, input);
        }

        @Override
        public void end() {
        }
        
    }
    
    public static class Encode implements IProcessor<INDArray, INDArray> {
        private int nSamples = 0;
        private double mTotalScore;
        private final Autoencoder mModel;
        private final ILossFunction mLoss;
        private final IActivation mActivation;
        public Encode(Autoencoder model) {
            mModel = model;
            mLoss = LossFunctions.LossFunction.L2.getILossFunction();
            mActivation = Activation.IDENTITY.getActivationFunction();
        }

        @Override
        public void begin() {
        
        }

        @Override
        public INDArray process(INDArray input) {

            INDArray output = Layer.ModelProcessor.activateSelectedLayers(mModel.getModel(), 0, 0, input);
            return output;

/*
            INDArray encodedArray = mModel.deepEncode(input);
            INDArray decodedArray = mModel.deepDecode(encodedArray);
            double score = mLoss.computeScore(input, decodedArray, mActivation, null, false);
            mTotalScore += score;
            nSamples++;
            return encodedArray;
*/
        }

        @Override
        public void end() {
            mTotalScore /= nSamples;
            System.out.println("Total score of result is " + mTotalScore);
        }
        
    }
    
    public static class Decode implements IProcessor<INDArray, INDArray> {
        private final Autoencoder mModel;
        public Decode(Autoencoder model) {
            mModel = model;
        }

        @Override
        public void begin() {
        
        }

        @Override
        public INDArray process(INDArray input) {
            INDArray output = Layer.ModelProcessor.activateSelectedLayers(mModel.getModel(), 1, 1, input);
            return output;
//            return mModel.deepDecode(input);
        }

        @Override
        public void end() {
        
        }
        
    }
    
}
