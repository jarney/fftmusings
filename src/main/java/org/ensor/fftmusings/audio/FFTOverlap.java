/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.audio;

import org.ensor.fftmusings.audio.windows.HammingWindow;
import org.ensor.fftmusings.audio.windows.BlackmanHarrisWindow;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import org.ensor.fftmusings.io.ICloseableIterator;
import org.ensor.fftmusings.pipeline.IProcessor;
import org.jtransforms.fft.DoubleFFT_1D;


/**
 *
 * @author jona
 */
public class FFTOverlap {
    public static final double TWO_PI = Math.PI*2;
    
    public static class ForwardPhaseDelta extends Forward {
        private final double[] mLastPhase;
        public ForwardPhaseDelta(int fftWindowSize) {
            super(fftWindowSize);
            mLastPhase = new double[fftWindowSize/2];
        }
        public MagnitudeSpectrum process(AudioSample input) {
            MagnitudeSpectrum magnitudeSpectrum = super.process(input);
            for (int i = 0; i < magnitudeSpectrum.mPhase.length; i++) {
                double thisPhase = magnitudeSpectrum.mPhase[i];
                double delta = thisPhase - mLastPhase[i];
                while (delta > Math.PI) {delta = delta - TWO_PI;}
                while (delta < -Math.PI) {delta = delta + TWO_PI;}
                magnitudeSpectrum.mPhase[i] = delta;
                mLastPhase[i] = thisPhase;
            }
            return magnitudeSpectrum;
        }

        public void process(MagnitudeSpectrum currentSample) {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }
    }
    public static class ReversePhaseDelta extends Reverse {
        private final double[] mLastPhase;
        public ReversePhaseDelta(int fftWindowSize) {
            super(fftWindowSize);
            mLastPhase = new double[fftWindowSize/2];
        }
        public AudioSample process(MagnitudeSpectrum input) {
            for (int i = 0; i < input.mPhase.length; i++) {
                double thisPhaseDelta = input.mPhase[i];
                double phase = mLastPhase[i] + thisPhaseDelta;
                while (phase > TWO_PI) {phase = phase - TWO_PI;}
                while (phase < 0) {phase = phase + TWO_PI;}
                input.mPhase[i] = phase;
                mLastPhase[i] = phase;
            }
            AudioSample output = super.process(input);
            return output;
        }
    }
    
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
                double v = mLastSample.mSamples[i] * mWindow[i];
                fftData[i*2] = v;
                fftData[i*2+1] = 0;
            }
            int offset = mLastSample.size();
            for (int i = 0; i < input.size(); i++) {
                double v = input.mSamples[i] * mWindow[i+offset];
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
        private final int mFFTWindowSize;
        private double[] xn;

        public Reverse(int fftWindowSize) {
            mFFTWindowSize = fftWindowSize;
            mFFT = new DoubleFFT_1D(mFFTWindowSize);
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
                fftBuffer[i*2] = m * Math.cos(angle);
                fftBuffer[i*2+1] = m * Math.sin(angle);
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
    public static class Write implements IProcessor<MagnitudeSpectrum, MagnitudeSpectrum> {
        private DataOutputStream mDOS;
        
        public Write(DataOutputStream dos) {
            mDOS = dos;
        }

        @Override
        public void begin() {
        }

        @Override
        public MagnitudeSpectrum process(MagnitudeSpectrum input) {
            try {
                mDOS.writeLong(input.mMagnitude.length);
                for (int i = 0; i < input.mMagnitude.length; i++) {
                    mDOS.writeDouble(input.mMagnitude[i]);
                    mDOS.writeDouble(input.mPhase[i]);
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
    public static class PhaseUnwrapper implements IProcessor<MagnitudeSpectrum, MagnitudeSpectrum> {

        @Override
        public void begin() {
        }

        @Override
        public MagnitudeSpectrum process(MagnitudeSpectrum input) {
            
            int phaseNumber = 0;

            MagnitudeSpectrum unwrapped = new MagnitudeSpectrum();
            unwrapped.mPhase = new double[input.mPhase.length];
            unwrapped.mMagnitude = input.mMagnitude;
            
            for (int k = 0; k < input.mPhase.length; k++) {
                if (k > 0 && input.mPhase[k] > input.mPhase[k-1]) {
                    phaseNumber--;
                }
                unwrapped.mPhase[k] = input.mPhase[k] + phaseNumber * Math.PI*2;
            }
            return unwrapped;
        }

        @Override
        public void end() {
        }
    }
    
    public static class PhsePredictorBasicModel implements IProcessor<MagnitudeSpectrum, MagnitudeSpectrum> {

        @Override
        public void begin() {
        }

        @Override
        public MagnitudeSpectrum process(MagnitudeSpectrum input) {
            return input;
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
    
    public static class Reader implements ICloseableIterator<MagnitudeSpectrum> {
        
        private MagnitudeSpectrum mCurrentPacket;
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
                mCurrentPacket = new MagnitudeSpectrum();
                mCurrentPacket.mMagnitude = new double[(int)nPackets]; 
                mCurrentPacket.mPhase = new double[(int)nPackets]; 
                for (int i = 0; i < nPackets; i++) {
                    mCurrentPacket.mMagnitude[i] = mStream.readDouble();
                    mCurrentPacket.mPhase[i] = mStream.readDouble();
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
        public MagnitudeSpectrum next() {
            // If we don't have a packet ready
            // we attempt to read one.
            if (mCurrentPacket == null) {
                readOne();
            }
            MagnitudeSpectrum next = mCurrentPacket;
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
    /**
     * This doesn't actually normalize to human hearing range, but it does
     * do some normalization in that direction.  The intent here is that
     * the low-range signal is attenuated and the high-range signals are
     * enhanced because that's the way human hearing works.  This does NOT
     * attempt to do a realistic conversion to an actual decibel scale
     * with any precision at all, this is just a slight heuristic to get
     * better performance.
     */
    public static class NormalizeToHearing implements IProcessor<MagnitudeSpectrum, MagnitudeSpectrum> {

        private double[] mFilter;
        private double mSampleRate;
        private boolean mForward;
        private DCT.FrequencyAttenuation[] bins;
        
        public NormalizeToHearing(boolean aForward, double aSampleRate) {
            
            mForward = aForward;
            mFilter = new double[512];
            
            // Frequency Attenuation (dB)
            bins = new DCT.FrequencyAttenuation[5];
            bins[0] = new DCT.FrequencyAttenuation(0, 1);
            bins[1] = new DCT.FrequencyAttenuation(100, .25);
            bins[1] = new DCT.FrequencyAttenuation(500, .5);
            bins[2] = new DCT.FrequencyAttenuation(2000, .8);
            bins[3] = new DCT.FrequencyAttenuation(8000, 1);
            bins[4] = new DCT.FrequencyAttenuation(13025, .1);
            
            mSampleRate = aSampleRate;
            for (int i = 0; i < mFilter.length; i++) {
                mFilter[i] = interpolateSensitivity(i);
                if (!aForward) {
                    mFilter[i] = 1/mFilter[i];
                }
            }
        }
        
        int findLowFrequencyBin(double frequency) {
            for (int i = bins.length-1; i >= 0; i--) {
                if (frequency > bins[i].f) {
                    return i;
                }
            }
            return 0;
        }
        
        private double interpolateSensitivity(int bucketId) {
            double frequency = bucketId * mSampleRate / mFilter.length;
            
            // Find the nearest frequency buckets.
            int lowBin = findLowFrequencyBin(frequency);
            int highBin = lowBin+1;
            
//            System.out.println("Frequency " + frequency + ": " + lowBin + "-" + highBin);
            
            DCT.FrequencyAttenuation low = bins[lowBin];
            DCT.FrequencyAttenuation high = bins[highBin];
            
            double attenuation = low.a + (high.a - low.a) / (high.f - low.f) * (frequency - low.f);
            
            
            return attenuation;
        }
        
        
        @Override
        public void begin() {
        }

        @Override
        public MagnitudeSpectrum process(MagnitudeSpectrum input) {
            // Normalize forward or reverse.
            MagnitudeSpectrum output = new MagnitudeSpectrum();
            output.mMagnitude = new double[input.mMagnitude.length];
            output.mPhase = input.mPhase;
            
            for (int i = 0; i < input.mMagnitude.length; i++) {
                output.mMagnitude[i] = input.mMagnitude[i];
                if (mForward) {
                    output.mMagnitude[i] = Math.sqrt(output.mMagnitude[i]);
                    output.mMagnitude[i] *= mFilter[i];
                }
                else {
                    output.mMagnitude[i] *= mFilter[i];
                    output.mMagnitude[i] = output.mMagnitude[i] * output.mMagnitude[i];
                }
            }
            
            return output;
        }

        @Override
        public void end() {
        }
        
    }

}
