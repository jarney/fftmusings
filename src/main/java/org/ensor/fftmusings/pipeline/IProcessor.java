package org.ensor.fftmusings.pipeline;

/**
 *
 * @author jarney
 */
public interface IProcessor<In, Out> {
    void begin();
    Out process(In input);
    void end();
}
