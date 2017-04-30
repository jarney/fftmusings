/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.pipeline;

/**
 *
 * @author jona
 */
public class ChannelSelector<A> implements IProcessor<A[], A> {

    private final int mChannelNumber;

    public static final <R> IBuilder<IProcessor<R[], R>> builder(final Class<R> classz, final int aChannelNumber) {
        return new IBuilder<IProcessor<R[], R>>() {
            @Override
            public IProcessor<R[], R> build() {
                return new ChannelSelector(classz, aChannelNumber);
            }
        };
    };

    public ChannelSelector(Class<A> classz, int aChannelNumber) {
        mChannelNumber = aChannelNumber;
    }

    @Override
    public void begin() {
    }

    @Override
    public A process(A[] input) {
        return input[mChannelNumber];
    }

    @Override
    public void end() {
    }
}

