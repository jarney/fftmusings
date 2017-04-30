/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.pipeline;

import java.lang.reflect.Array;

/**
 *
 * @author jona
 */
public class ChannelDuplicator<A> implements IProcessor<A, A[]> {

    private final int mChannelNumber;

    public ChannelDuplicator(Class<A> classz, int aChannelNumber) {
        mChannelNumber = aChannelNumber;
    }

    @Override
    public void begin() {
    }

    @Override
    public A[] process(A input) {
        A[] output = (A[])Array.newInstance(input.getClass(), mChannelNumber);
        for (int i = 0; i < output.length; i++) {
            output[i] = input;
        }
        return output;
    }

    @Override
    public void end() {
    }
}

