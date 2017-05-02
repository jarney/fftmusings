/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.audio;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import javax.sound.sampled.AudioFileFormat;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import org.ensor.fftmusings.io.ConcatStream;
import org.ensor.fftmusings.pipeline.IProcessor;

/**
 *
 * @author jona
 */
public class WAVFileWriter implements IProcessor<AudioSample[], Boolean> {

    public static WAVFileWriter create(String aOutputFilename) {
        int channels = 2;
        int sampleSize = 16;
        int frameSize = sampleSize/8 * channels;
        
        AudioFormat targetFormat = new AudioFormat(AudioFormat.Encoding.PCM_SIGNED,
                11025.0f, // Sample rate
                sampleSize, // Sample size (bits)
                channels, // Channels
                frameSize, // Frame Size
                11025.0f, // Frame Rate
                false); // Big-endian
        return new WAVFileWriter(aOutputFilename, targetFormat);
    }

    private final ConcatStream mStream;
    private final AudioFormat mTargetFormat;
    private final String mOutputFilename;
    
    private WAVFileWriter(String aOutputFilename, AudioFormat aTargetFormat) {
        mOutputFilename = aOutputFilename;
        mStream = new ConcatStream();
        mTargetFormat = aTargetFormat;
    }
    
    @Override
    public void begin() {
    }

    @Override
    public Boolean process(AudioSample[] input) {
        InputStream bais = writeSamples(mTargetFormat, input);
        mStream.addStream(bais, input[0].size());
        return true;
    }

    @Override
    public void end() {
        try {
            AudioInputStream outputAIS = new AudioInputStream(mStream, mTargetFormat,
                    mStream.size());
            
            AudioSystem.write(outputAIS,
                    AudioFileFormat.Type.WAVE,
                    new File(String.format(mOutputFilename)));
        } catch (IOException ex) {
            throw new RuntimeException("Could not write file " + mOutputFilename, ex);
        }
    }
    
    private static InputStream writeSamples(AudioFormat aAudioFormat, AudioSample[] aSamples) {
        if (aSamples.length != aAudioFormat.getChannels()) {
            throw new RuntimeException("Audio format is "+ aAudioFormat.getChannels() +
                    " channels, but " + aSamples.length + " supplied");
        }
        
        if (aAudioFormat.getSampleSizeInBits() != 16) {
            throw new RuntimeException("this conversion only supports 16 bit audio");
        }
        if (aSamples == null) {
            throw new RuntimeException("Streams not provided");
        }
        if (aSamples.length == 0) {
            throw new RuntimeException("No streams provided");
        }
        int size = aSamples[0].size();
        for (int channel = 0; channel < aSamples.length; channel++) {
            if (aSamples[channel].size() != size) {
                throw new RuntimeException("Streams are different sizes: " + aSamples[channel].size() + "!=" + size);
            }
        }
        
        byte[] rawData = new byte[aSamples[0].size()*aAudioFormat.getFrameSize()];
        
        for (int i = 0; i < size; i++) {
            for (int channel = 0; channel < aSamples.length; channel++) {
                
                double f1 = aSamples[channel].mSamples[i];
                f1 *= (1 << 15)-1;
                int if1 = (int)f1;

                int firstByte;
                int secondByte;
                if (aAudioFormat.isBigEndian()) {
                    firstByte = 1;
                    secondByte = 0;
                }
                else {
                    firstByte = 0;
                    secondByte = 1;
                }
                rawData[(i*aSamples.length + channel)*2 + firstByte] = (byte)(if1 & 0xff);
                rawData[(i*aSamples.length + channel)*2 + secondByte] = (byte)((if1 >> 8) & 0xff);
            
            }
        }
        
        ByteArrayInputStream bais = new ByteArrayInputStream(rawData);
        return bais;
    }
    
}
