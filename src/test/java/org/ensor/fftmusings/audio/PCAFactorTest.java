/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.audio;

import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.FileInputStream;
import java.io.DataInputStream;
import java.util.List;
import java.util.ArrayList;
import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.ensor.fftmusings.pca.PCAFactory;
import org.ensor.fftmusings.pca.PCATransformer;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author jona
 */
public class PCAFactorTest {

    /**
     * Perform a principal component analysis on a generated
     * set of data constructed so that the last dimension can
     * easily be dropped without losing data.  Verify that the
     * system identifies the correct dimension and that the reconstructed
     * vectors differ from the originals by only a small amount.
     */
    @Test
    public void testPCAFactory() throws Exception {
        int dataPoints = 10;
        int dimensions = 3;
        double [] dx = {0.58, 0.40, 0.83, 0.03, 0.61};
        double [] dy = {0.77, 0.58, 0.60, 0.40, 0.54};
        INDArray createdMean = Nd4j.zeros(3);
        createdMean.putScalar(0, 0);
        createdMean.putScalar(1, 0);
        createdMean.putScalar(2, 0);
        PCAFactory pcaFactory = new PCAFactory(dimensions, 0.8);

        List<INDArray> points = new ArrayList<>();

        for (int i = 0; i < dataPoints/2; i++) {
            double x = dx[i];
            double y = dy[i];
            double xx = x;
            double yy = x + x;
            double zz = y;

            INDArray p1 = createdMean.add(Nd4j.create(new double[]{xx, yy, zz}));
            INDArray p2 = createdMean.sub(Nd4j.create(new double[]{xx, yy, zz}));

            points.add(p1);
            points.add(p2);
            pcaFactory.addPoint(p1);
            pcaFactory.addPoint(p2);

        }

        PCATransformer createTransform = pcaFactory.createTransform();

        DataOutputStream outputStream = new DataOutputStream(new FileOutputStream("test.pca"));
        PCAFactory.write(createTransform, outputStream);

        DataInputStream dataInputStream = new DataInputStream(new FileInputStream("test.pca"));

        PCATransformer transformer = PCAFactory.read(dataInputStream);

        Assert.assertEquals(2, transformer.getDimensions());

        INDArray origin2 = Nd4j.zeros(2);
        INDArray origin3 = Nd4j.zeros(3);
        
        for (INDArray original : points) {
            INDArray pcaData = transformer.forward(original);
            INDArray transformed = transformer.reverse(pcaData);
            Assert.assertTrue(original.distance2(transformed) < 0.0001);

        }

    }
}
