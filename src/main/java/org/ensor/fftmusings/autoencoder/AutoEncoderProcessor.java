/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.autoencoder;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.ensor.fftmusings.audio.MagnitudeSpectrum;
import org.ensor.fftmusings.pipeline.IProcessor;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 *
 * @author jona
 */
public class AutoEncoderProcessor implements IProcessor<MagnitudeSpectrum, MagnitudeSpectrum> {

    private final MultiLayerNetwork mModel;
    private final ILossFunction mLoss;
    private final IActivation mActivation;
    private double mTotalScore;
    
    public AutoEncoderProcessor(MultiLayerNetwork model) {
        mModel = model;
        mLoss = LossFunctions.LossFunction.MSE.getILossFunction();
        mActivation = Activation.IDENTITY.getActivationFunction();
    }
    
    
    @Override
    public void begin() {
        mTotalScore = 0;
    }

    @Override
    public MagnitudeSpectrum process(MagnitudeSpectrum sample) {
        INDArray inputArray = Nd4j.create(sample.mMagnitude.length);
        for (int j = 0; j < sample.mMagnitude.length; j++) {
            double x = sample.mMagnitude[j] / 170;
            //x = (x >= 0) ? Math.sqrt(x) : -Math.sqrt(-x);
            inputArray.putScalar(j, x);
        }
        
        
        // Activate the autoencoder to get the result.
        INDArray outputArray = mModel.activateSelectedLayers(0, mModel.getnLayers()-1, inputArray);
        
        double score = mLoss.computeScore(inputArray, outputArray, mActivation, null, false);
        mTotalScore += score;
        
        MagnitudeSpectrum output = new MagnitudeSpectrum();
        output.mMagnitude = new double[sample.mMagnitude.length];
        output.mPhase = new double[sample.mMagnitude.length];
        for (int j = 0; j < sample.mMagnitude.length; j++) {
            double x = outputArray.getDouble(j) * 170;
            //x = (x >= 0) ? (x*x) : -(x*x);
            output.mMagnitude[j] = x;
            output.mPhase[j] = sample.mPhase[j];
        }
        
        return output;
        
    }

    @Override
    public void end() {
        System.out.println("Total score of result is " + mTotalScore);
    }
    
}
