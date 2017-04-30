/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.atrain;

import java.util.ArrayList;
import java.util.List;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author jona
 */
public class TestINDArray {
    @Test
    public void testINDArraySemantics() {

        int x = 30;
        int y = 2;
        int z = 3;

        List<List<INDArray>> list = new ArrayList<>();
        
        // Get our data...
        for (int i = 0; i < x; i++) {
            List<INDArray> inputVectors = new ArrayList<>();
            for (int j = 0; j < y; j++) {
                INDArray dataPoint = Nd4j.zeros(z);
                for (int k = 0; k < z; k++) {
                    dataPoint.putScalar(k, Math.random());
                }
                inputVectors.add(dataPoint);
            }
            list.add(inputVectors);
        }


        // The old (slow) way:
        INDArray oldData = Nd4j.zeros(x, y, z);
        for (int i = 0; i < x; i++) {
            List<INDArray> points = list.get(i);
            for (int j = 0; j < y; j++) {
                INDArray dataPoint = points.get(j);
                for (int k = 0; k < z; k++) {
                    oldData.putScalar(i, j, k, dataPoint.getDouble(k));
                }
            }
        }
        // The new (elegant) way:
        // This doesn't seem to work!!!
        // Try to find a more elegant way to get this done.
        INDArray newData = Nd4j.zeros(x, y, z);
        
//        for (int i = 0; i < x; i++) {
//            List<INDArray> points = list.get(i);
//            for (int j = 0; j < y; j++) {
//                INDArray dataPoint = points.get(j);
//                newData.put(new int[]{i, j}, dataPoint);
//            }
//        }
//        
//        
//        // Compare results to make sure
//        // it works the way we expect.
//        double distance2 = oldData.distance2(newData);
//        System.out.println("Distance is " + distance2);
        
    }
}
