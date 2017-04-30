/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.audio;

import org.ensor.fftmusings.pipeline.IProcessor;

/**
 *
 * @author jona
 */
public class DCTLogFilter implements IProcessor<AudioDCTData, AudioDCTData> {

    @Override
    public void begin() {
    }

    @Override
    public AudioDCTData process(AudioDCTData input) {
        AudioDCTData d = new AudioDCTData(input.mSamples.length);
        
        for (int i = 0; i < 64; i++) {
            d.mSamples[i] = input.mSamples[i];
        }
        for (int i = 64; i < input.mSamples.length; i+=2) {
            d.mSamples[i] = (input.mSamples[i] + input.mSamples[i+1])/2;
            d.mSamples[i+1] = (input.mSamples[i] + input.mSamples[i+1])/2;
        }
        
        return d;
    }

    @Override
    public void end() {
    }
    
}
