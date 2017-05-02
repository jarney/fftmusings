/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.audio;

import org.ensor.fftmusings.pipeline.IProcessor;
import org.jtransforms.fft.DoubleFFT_1D;


/**
 *
 * @author jona
 */
public class FFTOverlap {
    public static class Forward implements IProcessor<AudioSample, MagnitudeSpectrum> {

        private DoubleFFT_1D mFFT;
        private double[] mWindow;
        private final int mFFTWindowSize;
        private AudioSample mLastSample;
        //private double[] mLastPhases;

        public Forward(int fftWindowSize) {
            mFFT = null;
            mFFTWindowSize = fftWindowSize;
            mLastSample = null;
            mWindow = HammingWindow.compute(mFFTWindowSize);
            mFFT = new DoubleFFT_1D(mFFTWindowSize);
        }

        @Override
        public void begin() {
        }

        @Override
        public MagnitudeSpectrum process(AudioSample input) {
            // e.g.
            // input sample : 512 real numbers
            // FFT window : 1024 imaginary numbers 
            if (mFFTWindowSize != input.mSamples.length*2) {
                throw new RuntimeException("Sample size does not match FFT window size");
            }

            if (mLastSample == null) {
                mLastSample = new AudioSample(input.size());
                //mLastPhases = new double[input.size()];
            }

            // Magnitude spectrum: 512 real numbers
            // because magnitude is sqrt(re*re+im*im) and
            // because we only preserve the bottom half
            // of the FFT.
            MagnitudeSpectrum spectrum = new MagnitudeSpectrum(mFFTWindowSize/2);

            // FFT buffer is FFT window size with 2 doubles (re,im).
            double[] fftData = new double[mFFTWindowSize*2];

            // Input signal is the previous sample concatenated
            // with the current signal.
            for (int i = 0; i < mLastSample.size(); i++) {
                double v = mLastSample.mSamples[i];// * mWindow[i];
                fftData[i*2] = v;
                fftData[i*2+1] = 0;
            }
            int offset = mLastSample.size();
            for (int i = 0; i < input.size(); i++) {
                double v = input.mSamples[i];// * mWindow[i+offset];
                fftData[(i + offset)*2] = v;
                fftData[(i + offset)*2+1] = 0;
            }
            mLastSample = input;

            // Compute the forward FFT.
            mFFT.complexForward(fftData);

            //dumpFFT("forward.log", fftData);
            
            // Compute the magnitude.
            for (int i = 0; i < spectrum.mMagnitude.length; i++) {
                double re = fftData[i*2];
                double im = fftData[i*2+1];
                double magnitude = Math.sqrt(re*re + im*im);
                double phase = Math.atan2(im, re);
                spectrum.mMagnitude[i] = magnitude;
                spectrum.mPhase[i] = phase;
            }

            return spectrum;
        }

        @Override
        public void end() {
        }

        public int getHopSize() {
            return mFFTWindowSize/2;
        }
        public int getWindowSize() {
            return mFFTWindowSize;
        }
    
    }
    
    
    public static void dumpFFT(String logFilename, double[]data) {
        System.out.println(logFilename);
        for (int i = 0; i < data.length/2; i++) {
            System.out.println("" + i + " " + data[i*2] + " " + data[i*2+1]);
        }
    }
    
    
    /**
     * See also http://dsp.stackexchange.com/questions/9877/reconstruction-of-audio-signal-from-spectrogram
     * http://dsp.stackexchange.com/questions/3406/reconstruction-of-audio-signal-from-its-absolute-spectrogram/3410#3410
     * http://web.itu.edu.tr/~ibayram/AnDwt.pdf
     * @author jona
     */
    public static class Reverse implements IProcessor<MagnitudeSpectrum, AudioSample> {
        private final DoubleFFT_1D mFFT;
        private final double[] mWindow;
        private final int mFFTWindowSize;
        private double[] xn;

        public Reverse(int fftWindowSize) {
            mFFTWindowSize = fftWindowSize;
            mFFT = new DoubleFFT_1D(mFFTWindowSize);
            mWindow = HammingWindow.compute(mFFTWindowSize);
            xn = null;
        }

        public int getHopSize() {
            return mFFTWindowSize/2;
        }
        public int getWindowSize() {
            return mFFTWindowSize;
        }

        @Override
        public void begin() {
        }

        @Override
        public AudioSample process(MagnitudeSpectrum spectrum) {
            // Input spectrum will be half of an FFT window
            // in length.  To get the FFT window, multiply by 2.
            if (mFFTWindowSize != spectrum.mMagnitude.length*2) {
                throw new RuntimeException("Sample size does not match FFT window size");
            }

            double[] fftBuffer = new double[mFFTWindowSize*2];
            if (xn == null) {
                xn = new double[mFFTWindowSize];
                for (int i = 0; i < mFFTWindowSize; i++) {
                    xn[i] = 0;
                }
            }

            // We fill in the bottom half of the FFT
            // based on the magnitude spectrum.
            for (int i = 0; i < spectrum.mMagnitude.length; i++) {
                double angle = spectrum.mPhase[i];
                double m = spectrum.mMagnitude[i];
                fftBuffer[i*2] = m/2 * Math.cos(angle);
                fftBuffer[i*2+1] = m/2 * Math.sin(angle);
            }
            // Next, we reconstruct the top-half of the FFT
            // from the fact that it is the complex
            // conjugate of the bottom half (tricky).
            // Ensure that the first half is the complex
            // conjugate of the second half.
            for (int i = 1; i < spectrum.mMagnitude.length; i++) {
                int idx0 = (i*2);
                int idx1 = (i*2+1);

                int idx = mFFTWindowSize;

                int idx2 = (idx - i)*2;
                int idx3 = (idx - i)*2+1;

                fftBuffer[idx2] = fftBuffer[idx0];
                fftBuffer[idx3] = -fftBuffer[idx1];
            }

            //dumpFFT("reverse.log", fftBuffer);
            
            mFFT.complexInverse(fftBuffer, true);

            for (int i = 0; i < xn.length; i++) {
                xn[i] += (fftBuffer[i*2]);// * mWindow[i]);
            }


            // Finally, we take the bottom half of the
            // FFT inverse and play that becomes our
            // reconstructed signal.
            AudioSample samples = new AudioSample(mFFTWindowSize/2);
            for (int i = 0; i < samples.size(); i++) {
                samples.mSamples[i] = xn[i];
            }

            // And finally, we shift our signal upwards
            // and this will become our next one.
            for (int i = 0; i < xn.length/2; i++) {
                xn[i] = xn[i + xn.length/2];
                xn[i + xn.length/2] = 0;
            }


            return samples;
        }

        @Override
        public void end() {
        }


    }

}
