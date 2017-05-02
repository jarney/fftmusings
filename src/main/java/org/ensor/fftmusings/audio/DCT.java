/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.audio;

import java.awt.image.BufferedImage;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;
import org.ensor.fftmusings.io.ICloseableIterator;
import org.ensor.fftmusings.pipeline.IProcessor;
import org.jtransforms.dct.DoubleDCT_1D;

/**
 *
 * @author jona
 */
public class DCT {
    public static class Forward implements IProcessor<AudioSample, AudioDCTData> {

        private DoubleDCT_1D mFFT;
        private int mSampleSize;
        private final boolean mSQRTScale;

        public Forward(boolean aSQRTScale) {
            mFFT = null;
            mSampleSize = 0;
            mSQRTScale = aSQRTScale;
        }

        @Override
        public void begin() {
        }

        @Override
        public AudioDCTData process(AudioSample input) {
            if (input.mSamples.length != mSampleSize) {
                mSampleSize = input.mSamples.length;
                mFFT = new DoubleDCT_1D(mSampleSize);
            }

            AudioDCTData spectrum = new AudioDCTData(mSampleSize);
            System.arraycopy(input.mSamples, 0, spectrum.mSamples, 0, input.size());
            mFFT.forward(spectrum.mSamples, true);
            if (mSQRTScale) {
                for (int i = 0; i < spectrum.mSamples.length; i++) {
                    double v = spectrum.mSamples[i];
                    v = (v >= 0) ? Math.sqrt(v) : -Math.sqrt(-v);
                    spectrum.mSamples[i] = v;
                }
            }
            return spectrum;
        }

        @Override
        public void end() {
        }

    }
    public static class Reverse implements IProcessor<AudioDCTData, AudioSample> {

        private DoubleDCT_1D mFFT;
        private int mSampleSize;
        private final boolean mSQRTScale;

        public Reverse(boolean aSQRTScale) {
            mFFT = null;
            mSampleSize = 0;
            mSQRTScale = aSQRTScale;
        }

        @Override
        public void begin() {
        }

        @Override
        public AudioSample process(AudioDCTData input) {
            if (input.mSamples.length != mSampleSize) {
                mSampleSize = input.mSamples.length;
                mFFT = new DoubleDCT_1D(mSampleSize);
            }

            AudioSample sample = new AudioSample(mSampleSize);
            System.arraycopy(input.mSamples, 0, sample.mSamples, 0, input.mSamples.length);
            mFFT.inverse(sample.mSamples, true);
            if (mSQRTScale) {
                for (int i = 0; i < input.mSamples.length; i++) {
                    double v = input.mSamples[i];
                    sample.mSamples[i] = (v > 0) ? (v*v) : -(v*v);
                }
            }
            return sample;
        }

        @Override
        public void end() {
        }

    }
    public static class Threshold implements IProcessor<AudioDCTData, AudioDCTData> {

        private final double mThreshold;

        public Threshold(double lowThreshold) {
            mThreshold = lowThreshold;
        }

        @Override
        public void begin() {
        }

        @Override
        public AudioDCTData process(AudioDCTData input) {
            AudioDCTData packet = new AudioDCTData();
            packet.mSamples = new double[input.mSamples.length];
            for (int i = 0; i < packet.mSamples.length; i++) {
                double v = input.mSamples[i];
                if (v >= 0) {
                    packet.mSamples[i] = (v > mThreshold) ? v : 0;
                }
                else {
                    packet.mSamples[i] = (-v > mThreshold) ? v : 0;
                }
            }
            return packet;
        }

        @Override
        public void end() {
        }
    }

    public static Reader createReader(String inputFile) throws FileNotFoundException {
        FileInputStream fis = new FileInputStream(inputFile);
        DataInputStream dis = new DataInputStream(fis);
        return new Reader(dis);
    }
    
    public static class Reader implements ICloseableIterator<AudioDCTData> {
        
        private AudioDCTData mCurrentPacket;
        private DataInputStream mStream;
        
        public Reader(DataInputStream dis) {
            mCurrentPacket = null;
            mStream = dis;
        }

        private void readOne() {
            try {
                long nPackets = mStream.readLong();
                if (nPackets <= 0) {
                    mCurrentPacket = null;
                    return;
                }
                mCurrentPacket = new AudioDCTData();
                mCurrentPacket.mSamples = new double[(int)nPackets]; 
                for (int i = 0; i < nPackets; i++) {
                    mCurrentPacket.mSamples[i] = mStream.readDouble();
                }
            } catch (IOException ex) {
                throw new RuntimeException("EOF while reading packets", ex);
            }
        }
        
        @Override
        public boolean hasNext() {
            // If we already have a packet,
            // then we return true, next() will
            // return that packet.
            if (mCurrentPacket != null) {
                return true;
            }
            // Read a packet (if we can).
            readOne();
            return mCurrentPacket != null;
        }

        @Override
        public AudioDCTData next() {
            // If we don't have a packet ready
            // we attempt to read one.
            if (mCurrentPacket == null) {
                readOne();
            }
            AudioDCTData next = mCurrentPacket;
            // Once we read one packet,
            // the current packet is null and needs to be read.
            mCurrentPacket = null;
            return next;
        }

        @Override
        public void close() throws IOException {
            mStream.close();
        }
        
    }
    
    public static class Write implements IProcessor<AudioDCTData, AudioDCTData> {
        private DataOutputStream mDOS;
        
        public Write(DataOutputStream dos) {
            mDOS = dos;
        }

        @Override
        public void begin() {
        }

        @Override
        public AudioDCTData process(AudioDCTData input) {
            try {
                mDOS.writeLong(input.mSamples.length);
                for (int i = 0; i < input.mSamples.length; i++) {
                    mDOS.writeDouble(input.mSamples[i]);
                }
            } catch (IOException ex) {
                throw new RuntimeException("Exception writing data");
            }
            return input;
        }

        @Override
        public void end() {
            try {
                // This is effectively the EOF marker.
                mDOS.writeLong(0);
            } catch (IOException ex) {
                throw new RuntimeException("Exception writing data");
            }
        }
    }
    
    public static class ToPNG implements IProcessor<AudioDCTData, AudioDCTData> {
        private final File mFile;
        private final List<AudioDCTData> mAllData;

        public ToPNG(String aFilename) {
            mFile = new File(aFilename);
            mAllData = new ArrayList<>();
        }

        @Override
        public void begin() {

        }

        @Override
        public AudioDCTData process(AudioDCTData input) {
            mAllData.add(input);
            return input;
        }

        @Override
        public void end() {

            AudioDCTData data = mAllData.get(0);

            int height = data.mSamples.length;
            int width = mAllData.size();

            BufferedImage bufferedImage = new BufferedImage( 
                    width, height, BufferedImage.TYPE_INT_RGB );

            double max = 0;
            for (AudioDCTData d : mAllData) {
                for (int y = 0; y < d.mSamples.length; y++) {
                    max = Math.max(max, Math.abs(d.mSamples[y]));
                }
            }

            int x = 0;
            for (AudioDCTData d : mAllData) {

                for (int y = 0; y < d.mSamples.length; y++) {

                    d.mSamples[y] /= max;

                    if (d.mSamples[y] > 0) {
                        d.mSamples[y] = Math.sqrt(d.mSamples[y]);
                    }
                    else {
                        d.mSamples[y] = -Math.sqrt(-d.mSamples[y]);
                    }

                    if (d.mSamples[y] > 1.0 || d.mSamples[y] < -1.0) {
                        System.out.println("Out of bounds, need normalization " + d.mSamples[y]);
                    }

                    int value = 0;

                    value = (int)(d.mSamples[y] * 128 + 128);

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

}
