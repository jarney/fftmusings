/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.audio;

import org.ensor.fftmusings.pipeline.IProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author jona
 */
public class Quantize {
    public static class Forward implements IProcessor<INDArray, QuantizedVector> {

        private final int mQuanta;
        private final double mLow;
        private final double mHigh;
        private final boolean mSQRTQuanta;

        Forward(int aQuanta, double low, double high, boolean sqrtQuanta) {
            mQuanta = aQuanta;
            mLow = low;
            mHigh = high;
            mSQRTQuanta = sqrtQuanta;
        }

        @Override
        public void begin() {
        }

        @Override
        public QuantizedVector process(INDArray input) {
            QuantizedVector quantized = new QuantizedVector(input.columns(), mQuanta);
            for (int i = 0; i < input.columns(); i++) {
                double v = input.getDouble(i);
                if (mSQRTQuanta) {
                    v = (v >= 0) ? Math.sqrt(v) : -Math.sqrt(-v);
                }
                v += -mLow;
                v /= (mHigh - mLow);
                v *= (mQuanta-1);
                int q = (int)v;
                quantized.mSamples[i] = q;
            }
            return quantized;
        }

        @Override
        public void end() {
        }

    }
    public static class Reverse implements IProcessor<QuantizedVector, INDArray> {

        private final int mQuanta;
        private final double mLow;
        private final double mHigh;
        private final boolean mSQRTQuanta;

        public Reverse(int aQuanta, double low, double high, boolean sqrtQuanta) {
            mQuanta = aQuanta;
            mLow = low;
            mHigh = high;
            mSQRTQuanta = sqrtQuanta;
        }

        @Override
        public void begin() {
        }

        @Override
        public INDArray process(QuantizedVector input) {
            INDArray normal = Nd4j.zeros(input.mSamples.length);
            for (int i = 0; i < input.mSamples.length; i++) {
                double v = input.mSamples[i];
                v /= (mQuanta-1);
                v *= (mHigh - mLow);
                v -= -mLow;
                if (mSQRTQuanta) {
                    v = (v >= 0) ? (v*v) : -(v*v);
                }
                normal.putScalar(i, v);
            }
            return normal;
        }

        @Override
        public void end() {
        }

    }

}
