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
        
        public Encoder(double sampleRate, FFTOverlap.Forward fft) {
            mSampleRate = sampleRate;
            mHopSize = fft.getHopSize();
            mWindowSize = fft.getWindowSize();
            mFFT = fft;
        }
        
        double[] mPhaseData;
        
        
        @Override
        public void begin() {
        }

        @Override
        public VocoderData process(AudioSample input) {
            
            MagnitudeSpectrum magnitudeSpectrum = mFFT.process(input);
            
            VocoderData output = new VocoderData();
            output.mFrequencies = new double[mWindowSize - mHopSize];
            output.mMagnitudes = new double[mWindowSize - mHopSize];
            
            if (mPhaseData == null) {
                mPhaseData = new double[magnitudeSpectrum.mMagnitude.length];
            }
            
            double bin_centre_freq_scalar = (TWO_PI * mHopSize) / mWindowSize;
            double radians_to_hz_scalar = mSampleRate / (TWO_PI * mHopSize);
            
            for (int k = 1; k < magnitudeSpectrum.mMagnitude.length; k++) {
                double delta = magnitudeSpectrum.mPhase[k] - mPhaseData[k];
                while (delta > Math.PI) {delta = delta - TWO_PI;}
                while (delta < -Math.PI) {delta = delta + TWO_PI;}
                output.mMagnitudes[k] = magnitudeSpectrum.mMagnitude[k];
                output.mFrequencies[k] = (delta + (k - 1) * bin_centre_freq_scalar) * radians_to_hz_scalar;
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
        private final double[] mPhases;
        
        public Decoder(double sampleRate, FFTOverlap.Reverse reverseFFT) {
            mIFFT = reverseFFT;
            mSampleRate = sampleRate;
            mHopSize = reverseFFT.getHopSize();
            mWindowSize = reverseFFT.getWindowSize();
            mPhases = new double[mWindowSize/2];
        }
        
        @Override
        public void begin() {
        }

        @Override
        public AudioSample process(VocoderData input) {
            
            MagnitudeSpectrum ms = new MagnitudeSpectrum();
            ms.mMagnitude = new double[mWindowSize - mHopSize];
            ms.mPhase = new double[mWindowSize - mHopSize];
            
            double hz_to_radians_scalar = (TWO_PI * mHopSize) / mSampleRate;
            double sr_over_fft = mSampleRate / mWindowSize;
            
            for (int k = 2; k <  input.mFrequencies.length; k++) {
                double delta = (input.mFrequencies[k] - ((k - 1) * sr_over_fft)) * hz_to_radians_scalar;
                double phi = mPhases[k] + delta;
                double amp = input.mMagnitudes[k];
                ms.mMagnitude[k] = amp * Math.cos(phi);
                ms.mPhase[k] = amp * Math.sin(phi);
                mPhases[k] = phi;
            }
            
            return mIFFT.process(ms);
        }

        @Override
        public void end() {
        }
        
    }
    
}
