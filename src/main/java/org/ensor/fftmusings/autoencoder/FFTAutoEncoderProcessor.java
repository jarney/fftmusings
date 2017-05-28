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
public class FFTAutoEncoderProcessor implements IProcessor<MagnitudeSpectrum, MagnitudeSpectrum> {

    private final MultiLayerNetwork mModel;
    private final ILossFunction mLoss;
    private final IActivation mActivation;
    private double mTotalScore;
    
    public FFTAutoEncoderProcessor(MultiLayerNetwork model) {
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
        INDArray inputArray = Nd4j.create(sample.mMagnitude.length*2);
        for (int j = 0; j < sample.mMagnitude.length; j++) {
            double r = sample.mMagnitude[j];
            double p = sample.mPhase[j];
            //r = r / 170.0;
            //r = Math.sqrt(r);
            double x = r * Math.cos(p);
            double y = r * Math.sin(p);
            
            inputArray.putScalar(j*2, x);
            inputArray.putScalar(j*2+1, y);
        }
        
        
        // Activate the autoencoder to get the result.
        INDArray outputArray = mModel.activateSelectedLayers(0, mModel.getnLayers()-1, inputArray);
        
        double score = mLoss.computeScore(inputArray, outputArray, mActivation, null, false);
        mTotalScore += score;
        
        MagnitudeSpectrum output = new MagnitudeSpectrum();
        output.mMagnitude = new double[sample.mMagnitude.length];
        output.mPhase = new double[sample.mMagnitude.length];
        for (int j = 0; j < sample.mMagnitude.length; j++) {
            double x = outputArray.getDouble(j*2);
            double y = outputArray.getDouble(j*2+1);

            output.mPhase[j] = sample.mPhase[j];
            double r = Math.sqrt(x*x + y*y);
            //r = r*r;
            //r *= 170.0;
            output.mMagnitude[j] = r;
        }
        
        return output;
        
    }

    @Override
    public void end() {
        System.out.println("Total score of result is " + mTotalScore);
    }
    
}
