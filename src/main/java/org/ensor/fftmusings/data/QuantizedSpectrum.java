/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package org.ensor.fftmusings.data;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/**
 *
 * @author jona
 */
public class QuantizedSpectrum {
    private final byte[] mMagnitudes;
    private final byte[] mPhases;
    public static final int MAGNITUDE_QUANTA = 16;
    public static final int PHASE_QUANTA = 4;
    
    public QuantizedSpectrum(int samples) {
        mMagnitudes = new byte[samples];
        mPhases = new byte[samples];
    }
    public void setSample(int index, double magnitude, double phi) {
        magnitude = Math.sqrt(magnitude);
        magnitude *= MAGNITUDE_QUANTA-1;
        while (phi > (Math.PI*2.0)) {
            phi -= (Math.PI*2.0);
        }
        while (phi < 0) {
            phi += (Math.PI*2.0);
        }
        phi /= (Math.PI*2.0);
        phi *= PHASE_QUANTA;
        setSample(index, (int)magnitude, (int)phi);
    }
    public void setSample(int index, int magnitude, int phi) {
        mMagnitudes[index] = (byte)magnitude;
        mPhases[index] = (byte)phi;
    }
    
    public int getMagnitudeQuantized(int index) {
        return mMagnitudes[index];
    }
    public int getPhaseQuantized(int index) {
        return mPhases[index];
    }
    public double getMagnitude(int index) {
        double magnitude = mMagnitudes[index];
        magnitude /= (MAGNITUDE_QUANTA-1);
        magnitude = magnitude * magnitude;
        return magnitude;
    }
    public double getPhase(int index) {
        double phase = mPhases[index];
        phase /= (PHASE_QUANTA-1);
        phase *= (Math.PI*2.0);
        return phase;
    }
    public void write(OutputStream os) throws IOException {
        os.write(mMagnitudes);
        os.write(mPhases);
    }
    public int read(InputStream is) throws IOException {
        int rc;
        rc = is.read(mMagnitudes);
        if (rc == -1) {
            return -1;
        }
        rc = is.read(mPhases);
        if (rc == -1) {
            return -1;
        }
        return 0;
    }
    public int size() {
        return mMagnitudes.length;
    }
    
    public static QuantizedSpectrum add(QuantizedSpectrum a, QuantizedSpectrum b) {
        QuantizedSpectrum dif = new QuantizedSpectrum(a.size());
        for (int i = 0; i < dif.size(); i++) {
            int dm = a.getMagnitudeQuantized(i) + (b.getMagnitudeQuantized(i) - QuantizedSpectrum.MAGNITUDE_QUANTA);
            dm = Math.min(dm, QuantizedSpectrum.MAGNITUDE_QUANTA-1);
            dm = Math.max(dm, 0);
            
            int dp = a.getPhaseQuantized(i) + (b.getPhaseQuantized(i) - QuantizedSpectrum.PHASE_QUANTA);
            dp = Math.min(dp, QuantizedSpectrum.PHASE_QUANTA-1);
            dp = Math.max(dp, 0);
            
            dif.setSample(i, dm, dp);
        }
        return dif;
    }
    
    /**
     * Differences run from -MAGNITUDE_QUANTA to MAGNITUDE_QUANTA
     * so the total number of quanta is doubled.
     * @param a
     * @param b
     * @return 
     */
    public static QuantizedSpectrum diff(QuantizedSpectrum a, QuantizedSpectrum b) {
        QuantizedSpectrum dif = new QuantizedSpectrum(a.size());
        
        for (int i = 0; i < dif.size(); i++) {
            int dm = b.getMagnitudeQuantized(i) - a.getMagnitudeQuantized(i);
            dm += QuantizedSpectrum.MAGNITUDE_QUANTA;
            
            int dp = b.getPhaseQuantized(i) - a.getPhaseQuantized(i);
            dp += QuantizedSpectrum.PHASE_QUANTA;
            
            dif.setSample(i, dm, dp);
        }
        
        return dif;
    }
    
}
