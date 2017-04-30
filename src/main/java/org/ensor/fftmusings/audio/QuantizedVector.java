/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.audio;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/**
 *
 * @author jona
 */
public class QuantizedVector {
    protected int[] mSamples;
    protected int mQuanta;
    
    public QuantizedVector(int length, int nquanta) {
        mSamples = new int[length];
        mQuanta = nquanta;
    }
    public void setSample(int sampleId, int value) {
        mSamples[sampleId] = value;
    }
    public int getQuanta() {
        return mQuanta;
    }
    public int size() {
        return mSamples.length;
    }
    public static QuantizedVector read(DataInputStream dis) {
        try {
            int length = dis.readInt();
            int quanta = dis.readInt();
            QuantizedVector ret = new QuantizedVector(length, quanta);
            for (int i = 0; i < length; i++) {
                ret.mSamples[i] = dis.readInt();
            }
            return ret;
        } catch (IOException ex) {
            throw new RuntimeException("Failed to read vector", ex);
        }
    }
    public static void write(QuantizedVector v, DataOutputStream dos) {
        try {
            dos.writeInt(v.mSamples.length);
            dos.writeInt(v.mQuanta);
            for (int i = 0; i < v.mSamples.length; i++) {
                dos.writeInt(v.mSamples[i]);
            }
        } catch (IOException ex) {
            throw new RuntimeException("Failed to write vector", ex);
        }
    }
}
