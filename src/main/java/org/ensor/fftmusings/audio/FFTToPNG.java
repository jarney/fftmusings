/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.audio;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import javax.imageio.ImageIO;
import org.ensor.fftmusings.pipeline.IProcessor;

/**
 *
 * @author jona
 */
public class FFTToPNG implements IProcessor<MagnitudeSpectrum, MagnitudeSpectrum> {
    private final File mFile;
    private final List<MagnitudeSpectrum> mAllData;
    private final boolean mPlotPhase;
    private boolean mPlotPolar;
    
    
    public FFTToPNG(String aFilename, boolean plotPhase) {
        mFile = new File(aFilename);
        mPlotPhase = plotPhase;
        mPlotPolar = false;
        mAllData = new ArrayList<>();
    }
    
    public FFTToPNG(String aFilename, boolean plotPhase, boolean plotPolar) {
        this(aFilename, plotPhase);
        mPlotPolar = plotPolar;
    }
    

    @Override
    public void begin() {
    
    }

    @Override
    public MagnitudeSpectrum process(MagnitudeSpectrum input) {
        mAllData.add(input);
        return input;
    }

    @Override
    public void end() {
        
        MagnitudeSpectrum data = mAllData.get(0);
        
        int height = data.mMagnitude.length;
        int width = mAllData.size();
        
        BufferedImage bufferedImage = new BufferedImage( 
                width, height, BufferedImage.TYPE_INT_RGB );
        
        
        double max = 0;
        if (!mPlotPhase || mPlotPolar) {
            for (MagnitudeSpectrum d : mAllData) {
                for (int y = 0; y < d.mMagnitude.length; y++) {
                    max = Math.max(max, Math.abs(d.mMagnitude[y]));
                }
            }
        }
        
        System.out.println("Normalization factor: " + max);
        
        int x = 0;
        for (MagnitudeSpectrum d : mAllData) {

            if (mPlotPolar) {
                for (int y = 0; y < d.mMagnitude.length; y++) {

                    d.mMagnitude[y] /= max;

                    if (d.mMagnitude[y] > 0) {
                        d.mMagnitude[y] = Math.sqrt(d.mMagnitude[y]);
                    }
                    else {
                        d.mMagnitude[y] = -Math.sqrt(-d.mMagnitude[y]);
                    }

                    if (d.mMagnitude[y] > 1.0 || d.mMagnitude[y] < -1.0) {
                        System.out.println("Out of bounds, need normalization " + d.mMagnitude[y]);
                    }

                    int value = 0;
                    double r = (d.mMagnitude[y] * 127) * Math.sin(d.mPhase[y]) + 127;
                    double g = (d.mMagnitude[y] * 127) * Math.cos(d.mPhase[y]) + 127;
                    int rv = ((int)r) & 0xff;
                    int gv = ((int)g) & 0xff;
                    int bv = 0;

                    value = rv | (gv << 8) | (bv << 16);

                    bufferedImage.setRGB(x, y, value);
                }
            }
            else if (mPlotPhase) {
                for (int y = 0; y < d.mPhase.length; y++) {
                    double v = d.mPhase[y];
                    if (v < 0) {
                        v += (Math.PI*2);
                    }
                    v /= Math.PI*2;
                    int value = 0;

                    value = (int)(v * 255);

                    value = value & 0xff;

                    value = value | (value << 8) | (value << 16);

                    bufferedImage.setRGB(x, y, value);
                }
            }
            else {
                for (int y = 0; y < d.mMagnitude.length; y++) {

                    d.mMagnitude[y] /= max;

//                    if (d.mMagnitude[y] > 0) {
//                        d.mMagnitude[y] = Math.sqrt(d.mMagnitude[y]);
//                    }
//                    else {
//                        d.mMagnitude[y] = -Math.sqrt(-d.mMagnitude[y]);
//                    }

                    if (d.mMagnitude[y] > 1.0 || d.mMagnitude[y] < 0.0) {
                        System.out.println("Out of bounds, need normalization " + d.mMagnitude[y]);
                    }

                    int value = 0;

                    value = (int)(d.mMagnitude[y] * 254);

                    value = value & 0xff;

                    value = value | (value << 8) | (value << 16);

                    bufferedImage.setRGB(x, y, value);
                }
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
