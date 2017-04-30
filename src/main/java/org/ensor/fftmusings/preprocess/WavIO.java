/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package org.ensor.fftmusings.preprocess;

import org.ensor.fftmusings.data.Sample;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import javax.sound.sampled.AudioFormat;

/**
 *
 * @author jona
 */
public class WavIO {
    
    public static Sample[] readSamples(AudioFormat af, InputStream is, int sampleLength) {

        if (af.getSampleSizeInBits() != 16) {
            throw new RuntimeException("this conversion only supports 16 bit audio");
        }
        if (af.isBigEndian()) {
            throw new RuntimeException("Big-endian audio is not supported");
        }
        if (is == null) {
            throw new RuntimeException("Streams not provided");
        }
        if (af.getChannels() == 0) {
            throw new RuntimeException("No streams provided");
        }
        
        Sample[] channels = new Sample[af.getChannels()];
        for (int i = 0; i < channels.length; i++) {
            channels[i] = new Sample(sampleLength);
        }
        byte[] data = new byte[4];

        int bytesRead;
        try {
            bytesRead = is.read(data);
        } catch (IOException ex) {
            throw new RuntimeException("IO Exception ", ex);
        }
        if (bytesRead == -1) {
            return null;
        }
        for (int i = 1; i < sampleLength; i++) {
            for (int j = 0; j < channels.length; j++) {
                int i1 = (int)data[j*2 + 0] + ((int)data[j*2 + 1] << 8);
                double f1 = (double)i1;
                f1 /= (1 << 15)-1;
                double[] d = channels[j].getData();
                d[i] = f1;
            }
            
            try {
                bytesRead = is.read(data);
            } catch (IOException ex) {
                throw new RuntimeException("IO Exception ", ex);
            }
            if (bytesRead == -1) {
                data[0] = 0;
                data[1] = 0;
                data[2] = 0;
                data[3] = 0;
            }
        }
        
        return channels;
    }
    
    public static InputStream writeSamples(AudioFormat af, Sample[] channels) {
        
        if (channels.length != af.getChannels()) {
            throw new RuntimeException("Audio format is "+ af.getChannels() +
                    " channels, but " + channels.length + " supplied");
        }
        
        if (af.getSampleSizeInBits() != 16) {
            throw new RuntimeException("this conversion only supports 16 bit audio");
        }
        if (af.isBigEndian()) {
            throw new RuntimeException("Big-endian audio is not supported");
        }
        if (channels == null) {
            throw new RuntimeException("Streams not provided");
        }
        if (channels.length == 0) {
            throw new RuntimeException("No streams provided");
        }
        int size = channels[0].size();
        for (int j = 0; j < channels.length; j++) {
            if (channels[j].size() != size) {
                throw new RuntimeException("Streams are different sizes: " + channels[j].size() + "!=" + size);
            }
        }
        
        byte[] rawData = new byte[channels[0].size()*af.getFrameSize()];
        
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < channels.length; j++) {
                double[] data = channels[j].getData();
            
                double f1 = data[i];
                f1 *= (1 << 15)-1;
            
                int if1 = (int)f1;
            
                rawData[i*2*channels.length + j*2 + 0] = (byte)(if1 & 0xff);
                rawData[i*2*channels.length + j*2 + 1] = (byte)((if1 >> 8) & 0xff);
            
            }
        }
        
        ByteArrayInputStream bais = new ByteArrayInputStream(rawData);
        return bais;
    }
}
