/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.audio;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import org.ensor.fftmusings.pipeline.IProcessor;

/**
 *
 * @author jona
 */
public class QuantizedVectorWrite implements IProcessor<QuantizedVector, QuantizedVector> {

    private final DataOutputStream mDOS;
    private final List<QuantizedVector> mSampleList;
    
    public QuantizedVectorWrite(DataOutputStream dos) {
        mDOS = dos;
        mSampleList = new ArrayList<>();
    }
    
    @Override
    public void begin() {
    }

    @Override
    public QuantizedVector process(QuantizedVector input) {
        mSampleList.add(input);
        return input;
    }

    @Override
    public void end() {
        try {
            mDOS.writeInt(mSampleList.size());
            for (QuantizedVector input : mSampleList) {
                QuantizedVector.write(input, mDOS);
            }
        }
        catch (Exception ex) {
            throw new RuntimeException("Could not write quanta to file", ex);
        }
    }
    
}
