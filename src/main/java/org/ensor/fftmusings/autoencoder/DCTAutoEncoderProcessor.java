/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.autoencoder;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.ensor.fftmusings.audio.AudioDCTData;
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
public class DCTAutoEncoderProcessor implements IProcessor<AudioDCTData, AudioDCTData> {

    private final MultiLayerNetwork mModel;
    private final ILossFunction mLoss;
    private final IActivation mActivation;
    private double mTotalScore;
    
    public DCTAutoEncoderProcessor(MultiLayerNetwork model) {
        mModel = model;
        mLoss = LossFunctions.LossFunction.MSE.getILossFunction();
        mActivation = Activation.IDENTITY.getActivationFunction();
    }
    
    
    @Override
    public void begin() {
        mTotalScore = 0;
    }

    @Override
    public AudioDCTData process(AudioDCTData sample) {
        INDArray inputArray = Nd4j.create(sample.mSamples.length);
        for (int j = 0; j < sample.mSamples.length; j++) {
            double r = sample.mSamples[j];
            inputArray.putScalar(j, r);
        }
        
        
        // Activate the autoencoder to get the result.
        INDArray outputArray = mModel.activateSelectedLayers(0, mModel.getnLayers()-1, inputArray);
        
        double score = mLoss.computeScore(inputArray, outputArray, mActivation, null, false);
        mTotalScore += score;
        
        AudioDCTData output = new AudioDCTData();
        output.mSamples = new double[sample.mSamples.length];
        for (int j = 0; j < sample.mSamples.length; j++) {
            double r = outputArray.getDouble(j);
            output.mSamples[j] = r;
        }
        
        return output;
        
    }

    @Override
    public void end() {
        System.out.println("Total score of result is " + mTotalScore);
    }
    
}
