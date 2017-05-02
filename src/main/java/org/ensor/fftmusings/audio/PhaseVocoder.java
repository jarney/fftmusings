/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.audio;

import org.ensor.fftmusings.pipeline.IProcessor;

/**
 * Reference this implementation which gives the baseline on which this is
 * based.
 * https://github.com/johnglover/sound-rnn/blob/master/phase_vocoder.lua 
 * @author jona
 */
public class PhaseVocoder {
    public static final double TWO_PI = Math.PI*2;
    
    public static class VocoderData {
        private double[] mMagnitudes;
        private double[] mFrequencies;
    }
    
    public static class Encoder implements IProcessor<AudioSample, VocoderData> {

        private final double mSampleRate;
        private final int mWindowSize;
        private final int mHopSize;
        private final FFTOverlap.Forward mFFT;
        private Decoder mDecoder;
        
        public Encoder(double sampleRate, FFTOverlap.Forward fft) {
            mSampleRate = sampleRate;
            mHopSize = fft.getHopSize();
            mWindowSize = fft.getWindowSize();
            mFFT = fft;
        }
        
        double[] mLastPhase;
        
        
        @Override
        public void begin() {
        }

        @Override
        public VocoderData process(AudioSample input) {
            
            MagnitudeSpectrum magnitudeSpectrum = mFFT.process(input);
            
            VocoderData output = new VocoderData();
            output.mFrequencies = new double[mWindowSize - mHopSize];
            output.mMagnitudes = new double[mWindowSize - mHopSize];
            
            if (mLastPhase == null) {
                mLastPhase = new double[magnitudeSpectrum.mMagnitude.length];
                for (int k = 0; k < magnitudeSpectrum.mMagnitude.length; k++) {
                    mLastPhase[k] = 0;
                }
            }

//            for (int k = 0; k < magnitudeSpectrum.mMagnitude.length; k++) {
//                output.mMagnitudes[k] = magnitudeSpectrum.mMagnitude[k];
//                output.mFrequencies[k] = magnitudeSpectrum.mPhase[k];
//            }

            
            double bin_centre_freq_scalar = (TWO_PI * mHopSize) / mWindowSize;
            double radians_to_hz_scalar = mSampleRate / (TWO_PI * mHopSize);
            MagnitudeSpectrum ms2 = new MagnitudeSpectrum();
            ms2.mMagnitude = new double[magnitudeSpectrum.mMagnitude.length];
            ms2.mPhase = new double[magnitudeSpectrum.mMagnitude.length];
            
            
            for (int k = 0; k < magnitudeSpectrum.mMagnitude.length; k++) {
                double delta = magnitudeSpectrum.mPhase[k] - mLastPhase[k];
                while (delta > Math.PI) {delta = delta - TWO_PI;}
                while (delta < -Math.PI) {delta = delta + TWO_PI;}
                output.mMagnitudes[k] = magnitudeSpectrum.mMagnitude[k];
                output.mFrequencies[k] = (delta + (k) * bin_centre_freq_scalar) * radians_to_hz_scalar;
                
                mLastPhase[k] = magnitudeSpectrum.mPhase[k];
            }
            
            return output;
        }

        @Override
        public void end() {
        }
        
    }
    
    public static class Log implements IProcessor<VocoderData, VocoderData> {

        @Override
        public void begin() {
        }

        @Override
        public VocoderData process(VocoderData input) {
            System.out.println();
            for (int i = 0; i < input.mFrequencies.length; i++) {
                System.out.println(input.mFrequencies[i] + "\t" + input.mMagnitudes[i]);
            }
            return input;
        }

        @Override
        public void end() {
        }
        
    }
    
    public static class Decoder implements IProcessor<VocoderData, AudioSample> {

        private final FFTOverlap.Reverse mIFFT;
        private final double mSampleRate;
        private final int mWindowSize;
        private final int mHopSize;
        private VocoderData vlast;
        private MagnitudeSpectrum mslast;
        
        public Decoder(double sampleRate, FFTOverlap.Reverse reverseFFT) {
            mIFFT = reverseFFT;
            mSampleRate = sampleRate;
            mHopSize = reverseFFT.getHopSize();
            mWindowSize = reverseFFT.getWindowSize();
            mslast = new MagnitudeSpectrum();
            mslast.mPhase = new double[mWindowSize/2];
            mslast.mMagnitude = new double[mWindowSize/2];
            vlast = new VocoderData();
            vlast.mFrequencies = new double[mWindowSize/2];
            vlast.mMagnitudes = new double[mWindowSize/2];
        }
        
        @Override
        public void begin() {
        }

        @Override
        public AudioSample process(VocoderData input) {
            
            double bin_centre_freq_scalar = (TWO_PI * mHopSize) / mWindowSize;
            double radians_to_hz_scalar = mSampleRate / (TWO_PI * mHopSize);
            
            // Current frequency data
            // goes into last phase holder.
            for (int k = 0; k < input.mFrequencies.length; k++) {
                double delta2 = input.mFrequencies[k] / radians_to_hz_scalar - ((double)k)*bin_centre_freq_scalar;
                mslast.mPhase[k] = mslast.mPhase[k] + delta2;
            }
            
            // Process last sample since it consists
            // of the magnitude from last sample
            // plus phase of this sample.
            AudioSample as = mIFFT.process(mslast);
            
            // Current magnitude data goes
            // into last magnitude holder.
            // goes into last phase holder.
            for (int k = 0; k < input.mFrequencies.length; k++) {
                mslast.mMagnitude[k] = input.mMagnitudes[k];
            }
            
            return as;
        }

        @Override
        public void end() {
        }
        
    }
    
}
