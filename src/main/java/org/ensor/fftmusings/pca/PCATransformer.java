/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.pca;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author jona
 */
public final class PCATransformer {
    protected INDArray mMean;
    protected INDArray mPCAMatrix;
    protected INDArray mPCAMatrixInverse;
    protected INDArray mScale;

    protected PCATransformer(INDArray aMean, INDArray aPCAMatrixInverse, INDArray aScale) {
        mMean = aMean.dup();
        mPCAMatrixInverse = aPCAMatrixInverse.dup();
        mPCAMatrix = mPCAMatrixInverse.transpose();
        mScale = aScale.dup();
    }

    public int getDimensions() {
        return mPCAMatrixInverse.rows();
    }
    
    public INDArray forward(INDArray aPoint) {
        return aPoint.sub(mMean).mmul(mPCAMatrix).divRowVector(mScale);
    }

    public INDArray reverse(INDArray aPoint) {
        return aPoint.mulRowVector(mScale).mmul(mPCAMatrixInverse).add(mMean);
    }
}
    
