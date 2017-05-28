/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.autoencoder;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author jona
 */
public class Autoencoder {
    
    private Autoencoder mSourceAutoencoder;
    private MultiLayerNetwork mModel;
    private final AutoencoderMetadata mThisMetadata;

    public Autoencoder(AutoencoderMetadata aThisMetadata) {
        mThisMetadata = aThisMetadata;
    }

    public AutoencoderMetadata getMetadata() {
        return mThisMetadata;
    }
    
    public Autoencoder getSourceAutoencoder() {
        return mSourceAutoencoder;
    }
    
    public void setSourceAutoencoder(Autoencoder aSourceAutoencoderMetadata) {
        mSourceAutoencoder = aSourceAutoencoderMetadata;
    }
    
    public MultiLayerNetwork getModel() {
        return mModel;
    }
    public void setModel(MultiLayerNetwork aModel) {
        mModel = aModel;
    }
    public INDArray deepEncode(INDArray inputArray) {
        
        // Call deep encode on the 'parent' network to encode
        // the higher-level stuff.
        INDArray nextLayer;
        if (mSourceAutoencoder != null) {
            nextLayer = mSourceAutoencoder.deepEncode(inputArray);
        }
        else {
            nextLayer = inputArray;
            // Activate the encoder layer of the current network.
        }
        
        INDArray outputArray = Layer.ModelProcessor.activateSelectedLayers(mModel, 0, 0, nextLayer);
        
        
        return outputArray;
    }
    
    public INDArray deepDecode(INDArray inputArray) {
        
        INDArray nextLayer = Layer.ModelProcessor.activateSelectedLayers(mModel, 1, 1, inputArray);
        INDArray outputLayer;
        if (mSourceAutoencoder != null) {
            outputLayer = mSourceAutoencoder.deepDecode(nextLayer);
        }
        else {
            outputLayer = nextLayer;
        }
        return outputLayer;
    }
    

}
