/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.audio;

import java.util.Random;
import org.ensor.fftmusings.pipeline.IProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author jona
 */
public class RNNInput {
    public static class Forward implements IProcessor<QuantizedVector, INDArray> {

        @Override
        public void begin() {
        }

        @Override
        public INDArray process(QuantizedVector input) {
            INDArray array = Nd4j.zeros(input.mQuanta);

            for (int i = 0; i < 1; i++) {
                array.putScalar(i*input.mQuanta+input.mSamples[i], 1.0);
            }
            array = array.div(1);
            return array;
        }

        @Override
        public void end() {
        }
    }
    
    public static class Reverse implements IProcessor<INDArray, QuantizedVector> {

        private final int mQuanta;
        private final SampleStyle mSampleStyle;
        private final Random mRandom;

        public enum SampleStyle {
            RANDOM, LARGEST, RANDOM2
        };

        public Reverse(int aQuanta, SampleStyle aSampleStyle, Random aRandom) {
            mQuanta = aQuanta;
            mSampleStyle = aSampleStyle;
            mRandom = aRandom;
        }

        @Override
        public void begin() {
        }

        @Override
        public QuantizedVector process(INDArray input) {
            int inputLength = input.columns()/mQuanta;
            QuantizedVector qv = new QuantizedVector(inputLength, mQuanta);
            input = input.mul(inputLength);

            switch (mSampleStyle) {
                case RANDOM:
                    fillRandom(input, qv, inputLength);
                    break;
                case LARGEST:
                    fillLargest(input, qv, inputLength);
                    break;
                case RANDOM2:
                    fillRandom2(input, qv, inputLength);
                    break;
            }

            return qv;
        }

        private void fillRandom(INDArray input, QuantizedVector qv, int inputLength) {
            for (int i = 0; i < inputLength; i++) {
                double r = mRandom.nextDouble();
                double sum = 0;
                for (int j = 0; j < qv.mQuanta; j++) {
                    sum += input.getDouble(i*mQuanta+j);
                    if (sum > r) {
                        qv.mSamples[i] = j;
                        break;
                    }
                }
            }
        }
        private void fillRandom2(INDArray input, QuantizedVector qv, int inputLength) {
            for (int i = 0; i < inputLength; i++) {
                double r = mRandom.nextDouble();
                double sum = 0;
                for (int j = 0; j < qv.mQuanta; j++) {
                    double v = input.getDouble(i*mQuanta+j);
                    if (v < 0.2) {
                        v = 0;
                    }
                    sum += v;
                    if (sum > r) {
                        qv.mSamples[i] = j;
                        break;
                    }
                }
            }
        }
        private void fillLargest(INDArray input, QuantizedVector qv, int inputLength) {
            for (int i = 0; i < inputLength; i++) {
                int largestIndex = 0;
                double largestValue = 0;
                for (int j = 0; j < qv.mQuanta; j++) {
                    double v = input.getDouble(i*mQuanta+j);
                    if (v > largestValue) {
                        largestValue = v;
                        largestIndex = j;
                    }
                }
                qv.mSamples[i] = largestIndex;
            }
        }

        @Override
        public void end() {
        }

    }

}
