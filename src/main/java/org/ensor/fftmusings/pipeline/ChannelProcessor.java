/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.pipeline;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author jona
 */
public class ChannelProcessor<A, B> implements IProcessor<A[], B[]> {

    List<IProcessor<A, B>> mProcessors;

    public static final <R,S> IBuilder<ChannelProcessor<R, S>> builder(final IBuilder<IProcessor<R,S>> builder, final int n) {
        return new IBuilder<ChannelProcessor<R, S>>() {
            @Override
            public ChannelProcessor<R, S> build() {
                return new ChannelProcessor(builder, n);
            }
        };
    };


    public ChannelProcessor(IBuilder<IProcessor<A, B>> processorFactory, int channels) {
        mProcessors = new ArrayList<>();
        for (int i = 0; i < channels; i++) {
            mProcessors.add(processorFactory.build());
        }
    }

    @Override
    public void begin() {
        for (IProcessor<A, B> processor : mProcessors) {
            processor.begin();
        }
    }

    @Override
    public B[] process(A[] input) {
        if (input.length != mProcessors.size()) {
            throw new IllegalArgumentException("Wrong number of channels");
        }
        // How to create these?
        // This is kind-of a hack.
        B[] ret = (B[])new Object[input.length];

        int i = 0;
        for (A a : input) {
            ret[i] = mProcessors.get(i).process(a);
            i++;
        }

        return ret;
    }

    @Override
    public void end() {
        for (IProcessor<A, B> processor : mProcessors) {
            processor.end();
        }
    }

}