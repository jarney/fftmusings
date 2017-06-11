/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.autoencoder;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.ensor.fftmusings.pipeline.IProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;

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
        private final Autoencoder mModel;
        public Encode(Autoencoder model) {
            mModel = model;
        }

        @Override
        public void begin() {
        
        }

        @Override
        public INDArray process(INDArray input) {
            return mModel.deepEncode(input);
        }

        @Override
        public void end() {
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
            return mModel.deepDecode(input);
        }

        @Override
        public void end() {
        
        }
        
    }
    
}
