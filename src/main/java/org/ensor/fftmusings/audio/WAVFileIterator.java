/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.audio;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.UnsupportedAudioFileException;
import org.ensor.fftmusings.io.CompositeIterator;
import org.ensor.fftmusings.io.ICloseableIterator;
  
/**
 *
 * @author jona
 */
public class WAVFileIterator implements ICloseableIterator<AudioSample[]> {


    static ICloseableIterator<AudioSample[]> create(String aFilename, int aBufferSize) {
        return create(new File(aFilename), aBufferSize);
    }
    
    static ICloseableIterator<AudioSample[]> create(File aInputFile, int aBufferSize) {
        
        if (aInputFile.isFile()) {
            return new WAVFileIterator(aInputFile, aBufferSize);
        }
        else if (aInputFile.isDirectory()) {
            // TODO: Make this into
            // a recursive composite iterator
            // which sucks up all of the child files.
            File[] children = aInputFile.listFiles();
            List<ICloseableIterator<AudioSample[]>> iterators = new ArrayList<>();
            for (File f : children) {
                iterators.add(create(f, aBufferSize));
            }
            return new CompositeIterator(iterators);
        }
        return null;
    }
    
    
    
    private final File mFilename;
    private AudioFormat mAudioFormat;
    private AudioInputStream mAudioInputStream;
    private final int mBufferSize;
    private AudioSample[] mCurrentSample;

    private WAVFileIterator(File aInputFilename, int aBufferSize) {
        mFilename = aInputFilename;
        mAudioFormat = null;
        mAudioInputStream = null;
        mBufferSize = aBufferSize;
        mCurrentSample = null;
    }

    @Override
    public boolean hasNext() {
        if (mCurrentSample == null) {
            mCurrentSample = readSamples();
        }
        return mCurrentSample != null;
    }

    @Override
    public AudioSample[] next() {
        if (mCurrentSample == null) {
            mCurrentSample = readSamples();
        }
        AudioSample[] retSample = mCurrentSample;
        mCurrentSample = null;
        return retSample;
    }

    @Override
    public void close() throws IOException {
        if (mAudioInputStream != null) {
            mAudioInputStream.close();
            mAudioInputStream = null;
            mAudioFormat = null;
        }
    }
    
    private void open() throws IOException, UnsupportedAudioFileException {
        if (mAudioInputStream != null) {
            return;
        }
        mAudioInputStream = AudioSystem.getAudioInputStream(mFilename);
        mAudioFormat =  mAudioInputStream.getFormat();
        System.out.println("Processing " + mFilename.getAbsolutePath());
        System.out.println("Channels : " + mAudioFormat.getChannels());
        System.out.println("Encoding : " + mAudioFormat.getEncoding());
        System.out.println("Sample Rate : " + mAudioFormat.getSampleRate());
        System.out.println("Sample Size (bits) : " + mAudioFormat.getSampleSizeInBits());
        System.out.println("Frame Rate : " + mAudioFormat.getFrameRate());
        System.out.println("Frame Size : " + mAudioFormat.getFrameSize());
        System.out.println("Big-Endian? : " + mAudioFormat.isBigEndian());

    }
    
    private AudioSample[] readSamples() {
        try {
            open();
            if (mAudioFormat.getSampleSizeInBits() != 16) {
                throw new RuntimeException("this conversion only supports 16 bit audio");
            }
            if (mAudioFormat == null) {
                throw new RuntimeException("Streams not provided");
            }
            if (mAudioFormat.getChannels() == 0) {
                throw new RuntimeException("No streams provided");
            }

            AudioSample[] channels = new AudioSample[mAudioFormat.getChannels()];
            for (int i = 0; i < channels.length; i++) {
                channels[i] = new AudioSample(mBufferSize);
            }
            
            int bytesPerChannel = mAudioFormat.getSampleSizeInBits()/8;
            int bytes = bytesPerChannel * channels.length;
            
            byte[] data = new byte[bytes];

            int bytesRead;
            
            bytesRead = mAudioInputStream.read(data);
            if (bytesRead == -1) {
                return null;
            }
            for (int i = 0; i < mBufferSize; i++) {
                for (int j = 0; j < channels.length; j++) {
                    int firstByte;
                    int secondByte;
                    if (mAudioFormat.isBigEndian()) {
                        firstByte = 1;
                        secondByte = 0;
                    }
                    else {
                        firstByte = 0;
                        secondByte = 1;
                    }
                    int i1 = (int)data[j*2 + firstByte] + ((int)data[j*2 + secondByte] << 8);
                    double f1 = (double)i1;
                    f1 /= (1 << 15)-1;
                    channels[j].mSamples[i] = f1;
                }

                bytesRead = mAudioInputStream.read(data);
                if (bytesRead == -1) {
                    for (int j = 0; j < bytes; j++) {
                        data[j] = 0;
                    }
                }
            }

            return channels;
        } catch (Exception ex) {
            throw new RuntimeException("IO Exception ", ex);
        }
         
    }

}
