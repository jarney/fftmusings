/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.autoencoder;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import javax.imageio.ImageIO;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.ensor.fftmusings.audio.AudioDCTData;
import org.ensor.fftmusings.audio.MagnitudeSpectrum;
import org.ensor.fftmusings.pipeline.IProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author jona
 */
public class Layer {
    public static class DCTToINDArray implements IProcessor<AudioDCTData, INDArray> {

        @Override
        public void begin() {

        }

        @Override
        public INDArray process(AudioDCTData input) {
            INDArray output = Nd4j.create(input.mSamples);
            return output;
        }

        @Override
        public void end() {

        }
    }
    
    public static class FFTToINDArray implements IProcessor<MagnitudeSpectrum, INDArray> {

        @Override
        public void begin() {

        }

        @Override
        public INDArray process(MagnitudeSpectrum input) {
            INDArray output = Nd4j.zeros(input.mMagnitude.length);
            for (int i = 0; i < input.mMagnitude.length; i++) {
                //double x = input.mMagnitude[i] * Math.cos(input.mPhase[i]);
                //double y = input.mMagnitude[i] * Math.sin(input.mPhase[i]);
                //double m = Math.sqrt(x*x+y*y);
                output.putScalar(i, input.mMagnitude[i]);
            }
            return output;
        }

        @Override
        public void end() {

        }
    }
    
    public static class INDArrayToFFT implements IProcessor<INDArray, MagnitudeSpectrum> {

        @Override
        public void begin() {
        
        }

        @Override
        public MagnitudeSpectrum process(INDArray input) {
            MagnitudeSpectrum output = new MagnitudeSpectrum();
            output.mMagnitude = new double[512];
            output.mPhase = new double[512];
            for (int i = 0; i < output.mMagnitude.length; i++) {
                double m = input.getDouble(i);
                output.mMagnitude[i] = m;
                output.mPhase[i] = 0;
            }
            return output;
        }

        @Override
        public void end() {
        
        }
    }
    
    public static class INDArrayToDCT implements IProcessor<INDArray, AudioDCTData> {

        @Override
        public void begin() {
        
        }

        @Override
        public AudioDCTData process(INDArray input) {
            AudioDCTData output = new AudioDCTData();
            output.mSamples = new double[512];
            for (int i = 0; i < output.mSamples.length; i++) {
                output.mSamples[i] = input.getDouble(i);
            }
            return output;
        }

        @Override
        public void end() {
        
        }
    }
    
    public static class ModelProcessor implements IProcessor<INDArray, INDArray> {

        private MultiLayerNetwork mModel;
        private int mStart;
        private int mEnd;
        
        public ModelProcessor(MultiLayerNetwork model, int start, int end) {
            mModel = model;
            mStart = start;
            mEnd = end;
        }
        
        @Override
        public void begin() {
        
        }

        @Override
        public INDArray process(INDArray input) {
            INDArray output = activateSelectedLayers(mModel, mStart, mEnd, input);
            return output;
        }
        
        public static INDArray activateSelectedLayers(MultiLayerNetwork model, int from, int to, INDArray input) {
            if (input == null) throw new IllegalStateException("Unable to perform activation; no input found");
            if (from < 0 || from >= model.getLayers().length || from > to) throw new IllegalStateException("Unable to perform activation; FROM is out of layer space");
            if (to < from || to >= model.getLayers().length) throw new IllegalStateException("Unable to perform activation; TO is out of layer space");

            INDArray res = input;
            for (int l = from; l <= to; l++) {
                res = model.activationFromPrevLayer(l, res, false);
            }
            return res;
        }

        @Override
        public void end() {
        
        }
    }
    
    public static class ToFFTPNG implements IProcessor<INDArray, INDArray> {
        private final File mFile;
        private final List<INDArray> mAllData;
        private final boolean mMagnitude;

        public ToFFTPNG(String aFilename, boolean magnitude) {
            mFile = new File(aFilename);
            mAllData = new ArrayList<>();
            mMagnitude = magnitude;
        }

        @Override
        public void begin() {

        }

        @Override
        public INDArray process(INDArray input) {
            mAllData.add(input);
            return input;
        }

        @Override
        public void end() {

            INDArray data = mAllData.get(0);
            
            int height = data.columns();
            if (!mMagnitude) {
                height /= 2;
            }
            int width = mAllData.size();

            BufferedImage bufferedImage = new BufferedImage( 
                    width, height, BufferedImage.TYPE_INT_RGB );

            double max = 0;
            if (mMagnitude) {
                for (INDArray d : mAllData) {
                    for (int j = 0; j < height; j++) {
                        double m = d.getDouble(j);
                        max = Math.max(max, m);
                    }
                }
            }
            else {
                for (INDArray d : mAllData) {
                    for (int y = 0; y < height; y++) {
                        max = Math.max(max, Math.abs(d.getDouble(y*2)));
                        max = Math.max(max, Math.abs(d.getDouble(y*2+1)));
                    }
                }
            }

            int i = 0;
            for (INDArray d : mAllData) {

                for (int j = 0; j < height; j++) {
                    if (mMagnitude) {
                        double m = d.getDouble(j)/max;
                        if (m < 0) {
                            m = 0;
                        }
                        int valueb = (int)(m * 240);
                        int value = (valueb) | (valueb << 8) | (valueb << 16);

                        bufferedImage.setRGB(i, j, value);
                    }
                    else {
                        double x = Math.abs(d.getDouble(j*2)) / max;
                        double y = Math.abs(d.getDouble(j*2+1)) / max;

                        int valueg = valueg = (int)(x * 255);

                        valueg = valueg & 0xff;

                        int valueb = (int)(y * 255);

                        valueb = valueb & 0xff;

                        int value = 0 | (valueg << 8) | (valueb << 16);

                        bufferedImage.setRGB(i, j, value);
                    }
                }


                i++;
            }

            try {
                ImageIO.write(bufferedImage, "png", mFile);
            } catch (IOException ex) {
                throw new RuntimeException("Could not write file ", ex);
            }
        }

    }
    
    public static class ToPNG implements IProcessor<INDArray, INDArray> {
        private final File mFile;
        private final List<INDArray> mAllData;

        public ToPNG(String aFilename) {
            mFile = new File(aFilename);
            mAllData = new ArrayList<>();
        }

        @Override
        public void begin() {

        }

        @Override
        public INDArray process(INDArray input) {
            mAllData.add(input);
            return input;
        }

        @Override
        public void end() {

            INDArray data = mAllData.get(0);
            
            int height = data.columns();
            int width = mAllData.size();

            BufferedImage bufferedImage = new BufferedImage( 
                    width, height, BufferedImage.TYPE_INT_RGB );

            double max = -Double.MAX_VALUE;
            double min = Double.MAX_VALUE;
            for (INDArray d : mAllData) {
                for (int y = 0; y < height; y++) {
                    double v = d.getDouble(y);
                    max = Math.max(max, v);
                    min = Math.min(min, v);
                }
            }

            int x = 0;
            for (INDArray d : mAllData) {

                for (int y = 0; y < height; y++) {

                    double dd = d.getDouble(y);

                    // Scale so that dd falls between 0=min, 1=max
                    
                    dd -= min;
                    dd /= (max-min);
                    
                    int value = (int)(dd * 255);

                    value = value & 0xff;

                    value = value | (value << 8) | (value << 16);

                    bufferedImage.setRGB(x, y, value);
                }


                x++;
            }

            try {
                ImageIO.write(bufferedImage, "png", mFile);
            } catch (IOException ex) {
                throw new RuntimeException("Could not write file ", ex);
            }
        }

    }

    public static class FFTDToINDArray implements IProcessor<MagnitudeSpectrum, INDArray> {

        @Override
        public void begin() {

        }

        @Override
        public INDArray process(MagnitudeSpectrum input) {
            INDArray output = Nd4j.zeros(input.mMagnitude.length*2);
            for (int i = 0; i < input.mMagnitude.length; i++) {
                double d = (input.mPhase[i] + (Math.PI)) / (2*Math.PI);
                if (d < 0 || d > 1) {
                    System.out.println("Invalid d");
                }
                double m0 = input.mMagnitude[i] * d;
                double m1 = input.mMagnitude[i] * (1-d);
                output.putScalar(i*2, m0);
                output.putScalar(i*2+1, m1);
            }
            return output;
        }

        @Override
        public void end() {

        }
    }
    
    public static class INDArrayToFFTD implements IProcessor<INDArray, MagnitudeSpectrum> {

        @Override
        public void begin() {
        
        }

        @Override
        public MagnitudeSpectrum process(INDArray input) {
            MagnitudeSpectrum output = new MagnitudeSpectrum();
            output.mMagnitude = new double[512];
            output.mPhase = new double[512];
            for (int i = 0; i < output.mMagnitude.length; i++) {
                double m0 = input.getDouble(i*2);
                double m1 = input.getDouble(i*2+1);
                
                output.mMagnitude[i] = m0 + m1;
                if (output.mMagnitude[i] <  0) {
                    output.mMagnitude[i] = 0;
                }
                
                if (output.mMagnitude[i] > 0.01) {
                    double d = m0 / (m0+m1);
                    output.mPhase[i] = d * (2*Math.PI) - Math.PI;
                }
                else {
                    output.mPhase[i] = 0;
                }
            }
            return output;
        }

        @Override
        public void end() {
        
        }
    }
    
    
    
}
