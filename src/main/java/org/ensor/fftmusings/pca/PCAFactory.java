/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.pca;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * This is a hack around a broken
 * PCA method which returns the columns
 * instead of the rows.
 * @author jona
 */
public class PCAFactory {

    private final List<INDArray> mDataPoints;
    private final int mInputDimensions;
    private final int mOutputDimensions;
    private final double mMaxVariance;
    
    public PCAFactory(int aInputDimensions, int aOutputDimensions) {
        mInputDimensions = aInputDimensions;
        mOutputDimensions = aOutputDimensions;
        mMaxVariance = 0;
        mDataPoints = new ArrayList<>();
    }
    public PCAFactory(int aInputDimensions, double aMaxVariance) {
        mInputDimensions = aInputDimensions;
        mOutputDimensions = 0;
        mMaxVariance = aMaxVariance;
        mDataPoints = new ArrayList<>();
    }
    
    public void addPoint(INDArray dataPoint) {
        if (dataPoint.columns() != mInputDimensions) {
            throw new IllegalArgumentException("Incorrect number of dimensions in input data");
        }
        mDataPoints.add(dataPoint);
    }
    public void addPoint(double[] dataPoint) {
        if (mInputDimensions != dataPoint.length) {
            throw new IllegalArgumentException("Incorrect number of dimensions in input data");
        }
        INDArray array = Nd4j.create(dataPoint);
        mDataPoints.add(array);
    }
    
    public PCATransformer createTransform() {
        if (mDataPoints.isEmpty()) {
            throw new IllegalArgumentException("Principal component analysis requires data to analyze");
        }
        INDArray data = Nd4j.zeros(new int[]{mDataPoints.size(), mInputDimensions});
        int i = 0;
        for (INDArray point : mDataPoints) {
            data.putRow(i, point);
            i++;
        }
        INDArray mean = data.sum(0);
        mean = mean.div(data.rows());
        INDArray dataMean = data.subRowVector(mean);
        INDArray reverseTransform;
        if (mOutputDimensions > 0) {
            reverseTransform = PCAFactory.pca_factor(dataMean.dup(), mOutputDimensions, false);
        }
        else {
            reverseTransform = PCAFactory.pca_factor(dataMean.dup(), mMaxVariance, false);
        }
        
        // Find the range of the dimensions.
        INDArray mmul = dataMean.mmul(reverseTransform.transpose());
        
        INDArray scale = Nd4j.zeros(mmul.columns());
        for (int j = 0; j < mmul.rows(); j++) {
            for (int k = 0; k < mmul.columns(); k++) {
                scale.putScalar(k, Math.max(scale.getDouble(k), Math.abs(mmul.getDouble(j, k))));
            }
        }
        
        return new PCATransformer(mean, reverseTransform, scale);
    }
    
    public static PCATransformer read(String aFilename) throws IOException {
        try (InputStream is = new FileInputStream(aFilename)) {
            try (DataInputStream dis = new DataInputStream(is)) {
                return read(dis);
            }
        }
    }
    public static void write(PCATransformer transformer, String aFilename) throws IOException {
        try (OutputStream os = new FileOutputStream(aFilename)) {
            try (DataOutputStream dos = new DataOutputStream(os)) {
                write(transformer, dos);
            }
        }
    }
    public static PCATransformer read(DataInputStream aInputStream) throws IOException {
        INDArray mean = Nd4j.read(aInputStream);
        INDArray matrixInverse = Nd4j.read(aInputStream);
        INDArray scale = Nd4j.read(aInputStream);
        PCATransformer transformer = new PCATransformer(mean, matrixInverse, scale);
        return transformer;
    }

    public static void write(PCATransformer transformer, DataOutputStream aOutputStream) throws IOException {
        Nd4j.write(transformer.mMean, aOutputStream);
        Nd4j.write(transformer.mPCAMatrixInverse, aOutputStream);
        Nd4j.write(transformer.mScale, aOutputStream);
    }
        
    /**
     * Calculates pca factors of a matrix, for a fixed number of reduced features
     * returns the factors to scale observations 
     *
     * The return is a factor matrix to reduce (normalized) feature sets
     *
     * @see pca(INDArray, int, boolean)
     *
     * @param A the array of features, rows are results, columns are features - will be changed
     * @param nDims the number of components on which to project the features 
     * @param normalize whether to normalize (adjust each feature to have zero mean)
     * @return the reduced feature set
     */
    private static INDArray pca_factor(INDArray A, int nDims, boolean normalize) {
        int[] shape = A.shape();
        
        if (shape.length != 2) {
            throw new IllegalArgumentException("PCA Factors should take a 2-d array with dimensions (data-points, dimensions)");
        }

        if (normalize) {
            // Normalize to mean 0 for each feature ( each column has 0 mean )
            INDArray mean = A.mean(0);
            A.subiRowVector(mean);
        }

        int m = A.rows();
        int n = A.columns();

        // The prepare SVD results, we'll decomp A to UxSxV'
        INDArray s = Nd4j.create(m < n ? m : n);
        INDArray VT = Nd4j.create(n, n, 'f');

        // Note - we don't care about U 
        Nd4j.getBlasWrapper().lapack().sgesvd(A, s, null, VT);

        // for comparison k & nDims are the equivalent values in both methods implementing PCA

        // So now let's rip out the appropriate number of left singular vectors from
        // the V output (note we pulls rows since VT is a transpose of V)
        INDArray V = VT.transpose();
        INDArray factor = Nd4j.create(nDims, n, 'f');
        for (int i = 0; i < nDims; i++) {
            factor.putRow(i, V.getRow(i));
        }

        return factor;
    }
    
    /**
     * Calculates pca vectors of a matrix, for a given variance. A larger variance (99%)
     * will result in a higher order feature set.
     *
     * To use the returned factor: multiply feature(s) by the factor to get a reduced dimension
     *
     * INDArray Areduced = A.mmul( factor ) ;
     * 
     * The array Areduced is a projection of A onto principal components
     *
     * @see pca(INDArray, double, boolean)
     *
     * @param A the array of features, rows are results, columns are features - will be changed
     * @param variance the amount of variance to preserve as a float 0 - 1
     * @param normalize whether to normalize (set features to have zero mean)
     * @return the matrix to mulitiply a feature by to get a reduced feature set
     */
    private static INDArray pca_factor(INDArray A, double variance, boolean normalize) {
        if (normalize) {
            // Normalize to mean 0 for each feature ( each column has 0 mean )
            INDArray mean = A.mean(0);
            A.subiRowVector(mean);
        }

        int m = A.rows();
        int n = A.columns();

        // The prepare SVD results, we'll decomp A to UxSxV'
        INDArray s = Nd4j.create(m < n ? m : n);
        INDArray VT = Nd4j.create(n, n, 'f');
        // Note - we don't care about U 
        Nd4j.getBlasWrapper().lapack().sgesvd(A, s, null, VT);

//        // As of here, A*VT = UNIT
//        
//        // From the original, orig*VT = X
//        
//        // U1 = A1*VT
//        // U2 = A2*VT
//        // U3 = A3*VT
//        // X1 = O2*VT
//        // X2 = O2*VT
//        // X3 = O2*VT
//        // X1 = (y1U1+y2U2+y3U3)
//        
//        INDArray mm = A.mmul(VT);
        
        // Now convert the eigs of X into the eigs of the covariance matrix
        for (int i = 0; i < s.length(); i++) {
            s.putScalar(i, Math.sqrt(s.getDouble(i)) / (m - 1));
        }

        // Now find how many features we need to preserve the required variance
        // Which is the same percentage as a cumulative sum of the eigenvalues' percentages
        double totalEigSum = s.sumNumber().doubleValue() * variance;
        int k = -1; // we will reduce to k dimensions
        double runningTotal = 0;
        for (int i = 0; i < s.length(); i++) {
            runningTotal += s.getDouble(i);
            if (runningTotal >= totalEigSum) { // OK I know it's a float, but what else can we do ?
                k = i + 1; // we will keep this many features to preserve the reqd. variance
                break;
            }
        }
        if (k == -1) { // if we need everything
            throw new IllegalArgumentException("No reduction possible for reqd. variance - use smaller variance");
        }
        // So now let's rip out the appropriate number of left singular vectors from
        // the V output (note we pulls rows since VT is a transpose of V)
        INDArray V = VT.transpose();
        
        INDArray factor = Nd4j.create(k, n, 'f');
        for (int i = 0; i < k; i++) {
            factor.putRow(i, V.getRow(i));
        }

        return factor;
    }
    
}
