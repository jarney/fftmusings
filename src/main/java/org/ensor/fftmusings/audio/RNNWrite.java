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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author jona
 */
public class RNNWrite implements IProcessor<INDArray, INDArray> {

    private final List<INDArray> mData;
    private final String mFilename;
    
    public RNNWrite(String aFilename) {
        mData = new ArrayList<>();
        mFilename = aFilename;
    }
    
    @Override
    public void begin() {
    }

    @Override
    public INDArray process(INDArray input) {
        mData.add(input);
        return input;
    }

    @Override
    public void end() {
        try (OutputStream os = new FileOutputStream(new File(mFilename))) {
            DataOutputStream dos = new DataOutputStream(os);
            
            dos.writeInt(mData.size());
            for (INDArray dataPoint : mData) {
                Nd4j.write(dataPoint, dos);
            }
            
        } catch (IOException ex) {
            throw new RuntimeException("Could not write file ", ex);
        }
    }
    
}
