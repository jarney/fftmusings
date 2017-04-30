package org.ensor.fftmusings.pipeline;

import java.util.Iterator;

/**
 *
 * @author jarney
 */
public class Pipeline<A, B> {
    
    IProcessor<A, B> mInitial;
    
    public Pipeline(IProcessor<A, B> initial) {
        mInitial = initial;
    }
    
    public <C> Pipeline<A, C> add(final IProcessor<B, C> nextStage) {
        
        IProcessor<A, C> composed = new IProcessor<A, C>() {

            @Override
            public void begin() {
                mInitial.begin();
                nextStage.begin();
            }

            @Override
            public C process(A input) {
                B intermediate = mInitial.process(input);
                C output = nextStage.process(intermediate);
                return output;
            }

            @Override
            public void end() {
                nextStage.end();
                mInitial.end();
            }
            
        };
        return new Pipeline(composed);
    }
    
    public void execute(Iterable<A> sampleIterable) {
        execute(sampleIterable.iterator());
    }

    public void execute(Iterator<A> sampleList) {
        mInitial.begin();
        while (sampleList.hasNext()) {
            A input = sampleList.next();
            mInitial.process(input);
        }
        mInitial.end();
    }
    
}
