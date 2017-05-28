/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.autoencoder;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.ArrayList;
import java.util.List;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

/**
 *
 * @author jona
 */
public class AutoencoderMetadata {
    private int mInput = 100;
    private int mHidden = 10;
    private String mIntermediateActivation = "TANH";
    private String mFinalActivation = "IDENTITY";
    private int mIterations = 1;
    private double mLearningRate = 0.1;
    private String mInputSource = "iterator";
    private String mSourceAutoencoder;
    private String mUpdater = "NESTEROVS";
    private String mWeightInit = "XAVIER";
    private int mEpochs = 100;
    private int mMinibatchSize = 100;
    private int mMinibatchesPerIteration = 1250;
    private String mFilename;
    private boolean mRegularization = false;
    private double mL2Regularization = 0.01;
    private List<LRSchedule> mLRSchedule = new ArrayList<>();
    private String mLoss = "L2";
    private double mSparsity = 0.0;
    
    @JsonProperty("sparsity")
    public void setSparsity(double aSparsity) {
        mSparsity = aSparsity;
    }
    @JsonProperty("sparsity")
    public double getSparsity() {
        return mSparsity;
    }
    
    @JsonProperty("loss")
    public void setLoss(String aLoss) {
        mLoss = aLoss;
    }
    @JsonProperty("loss")
    public String getLoss() {
        return mLoss;
    }
    
    public LossFunction getLossFunction() {
        for (LossFunction fn : LossFunction.values()) {
            if (fn.name().equals(mLoss)) {
                return fn;
            }
        }
        return null;
    }
    
    @JsonProperty("lrarray")
    public void setLRArray(List<LRSchedule> aArray) {
        mLRSchedule = aArray;
    }
    @JsonProperty("lrarray")
    public List<LRSchedule> getLRArray() {
        return mLRSchedule;
    }
    
    @JsonProperty("regularization")
    public void setRegularization(boolean aRegularization) {
        mRegularization = aRegularization;
    }
    @JsonProperty("regularization")
    public boolean getRegularization() {
        return mRegularization;
    }
    
    @JsonProperty("l2Regularization")
    public void setL2Regularization(double a) {
        mL2Regularization = a;
    }
    @JsonProperty("l2Regularization")
    public double getL2Regularization() {
        return mL2Regularization;
    }

    @JsonProperty("updater")
    public void setUpdater(String aUpdater) {
        mUpdater = aUpdater;
    }
    @JsonProperty("updater")
    public String getUpdater() {
        return mUpdater;
    }
    
    @JsonProperty("weightInit")
    public void setWeightInit(String aWeightInit) {
        mWeightInit = aWeightInit;
    }
    @JsonProperty("weightInit")
    public String getWeightInit() {
        return mWeightInit;
    }
    
    @JsonProperty("minibatchesPerIteration")
    public void setMinibatchesPerIteration(int aMinibatchesPerIteration) {
        mMinibatchesPerIteration = aMinibatchesPerIteration;
    }
    
    @JsonProperty("minibatchesPerIteration")
    public int getMinibatchesPerIteration() {
        return mMinibatchesPerIteration;
    }
    
    @JsonProperty("filename")
    public void setFilename(String aFilename) {
        mFilename = aFilename;
    }
    @JsonProperty("filename")
    public String getFilename() {
        return mFilename;
    }
    
    @JsonProperty("minibatch")
    public void setMinibatchSize(int aMinibatchSize) {
        mMinibatchSize = aMinibatchSize;
    }
    @JsonProperty("minibatch")
    public int getMinibatchSize() {
        return mMinibatchSize;
    }
    
    @JsonProperty("iterations")
    public void setIterations(int aIterations) {
        mIterations = aIterations;
    }
    @JsonProperty("iterations")
    public int getIterations() {
        return mIterations;
    }
    @JsonProperty("learningRate")
    public void setLearningRate(double aLearningRate) {
        mLearningRate = aLearningRate;
    }
    @JsonProperty("learningRate")
    public double getLearningRate() {
        return mLearningRate;
    }
    
    
    @JsonProperty("input")
    public void setInput(int aInput) {
        mInput = aInput;
    }
    
    @JsonProperty("input")
    public int getInput() {
        return mInput;
    }
    @JsonProperty("hidden")
    public void setHidden(int aHidden) {
        mHidden = aHidden;
    }
    @JsonProperty("hidden")
    public int getHidden() {
        return mHidden;
    }
    @JsonProperty("intermediateActivation")
    public void setIntermediateActivation(String aIntermediateActivation) {
        mIntermediateActivation = aIntermediateActivation;
    }
    @JsonProperty("intermediateActivation")
    public String getIntermediateActivation() {
        return mIntermediateActivation;
    }
    @JsonProperty("finalActivation")
    public void setFinalActivation(String aIntermediateActivation) {
        mFinalActivation = aIntermediateActivation;
    }
    @JsonProperty("finalActivation")
    public String getFinalActivation() {
        return mFinalActivation;
    }
    @JsonProperty("inputSource")
    public void setInputSource(String aInputSource) {
        mInputSource = aInputSource;
    }
    @JsonProperty("inputSource")
    public String getInputSource() {
        return mInputSource;
    }
    @JsonProperty("epochs")
    public void setEpochs(int aEpochs) {
        mEpochs = aEpochs;
    }
    @JsonProperty("epochs")
    public int getEpochs() {
        return mEpochs;
    }
    
    @JsonProperty("sourceAutoencoder")
    public void setSourceAutoencoder(String aSourceAutoencoder) {
        mSourceAutoencoder = aSourceAutoencoder;
    }
    @JsonProperty("sourceAutoencoder")
    public String getSourceAutoencoder() {
        return mSourceAutoencoder;
    }
    

}
