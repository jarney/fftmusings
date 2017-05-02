/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.autoencoder;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.ensor.fftmusings.audio.AudioDCTData;
import org.ensor.fftmusings.pipeline.IProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author jona
 */
public class AutoEncoderProcessor implements IProcessor<AudioDCTData, AudioDCTData> {

    private MultiLayerNetwork mModel;
    
    public AutoEncoderProcessor(MultiLayerNetwork model) {
        mModel = model;
    }
    
    
    @Override
    public void begin() {
    }

    @Override
    public AudioDCTData process(AudioDCTData input) {
        INDArray inputArray = Nd4j.create(input.mSamples);

        // Activate the autoencoder to get the result.
        INDArray outputArray = mModel.activateSelectedLayers(0, mModel.getnLayers()-1, inputArray);
        
        AudioDCTData output = new AudioDCTData();
        output.mSamples = new double[input.mSamples.length];
        for (int i = 0; i < input.mSamples.length; i++) {
            output.mSamples[i] = outputArray.getDouble(i);
        }
        
        return output;
        
    }

    @Override
    public void end() {
    }
    
}
