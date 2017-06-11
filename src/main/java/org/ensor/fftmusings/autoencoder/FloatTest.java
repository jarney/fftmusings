/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.autoencoder;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.NDArrayFactory;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author jona
 */
public class FloatTest {
    public static void main(String[] args) {
        
        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
        NDArrayFactory factory = Nd4j.factory();
        factory.setDType(DataBuffer.Type.DOUBLE);

        INDArray array = Nd4j.create(2, 4);
        System.out.println("Array is " + array);
        DataBuffer d = array.data();
        
        System.out.println("Databuffer " + d);
    }
}
