/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.pipeline;

import java.util.ArrayList;
import java.util.List;
import org.junit.Test;

/**
 *
 * @author jona
 */
public class TestPipeline {
    static class Sample {
    };
    static class FFT {
    };
    static class LogFFT {
    };
    static class ND4JVector {
    };
    
    static class SampleFFT implements IProcessor<Sample, FFT> {

        public static final IBuilder<IProcessor<Sample, FFT>> BUILDER = new IBuilder() {
            @Override
            public Object build() {
                return new SampleFFT();
            }
        };
        
        @Override
        public void begin() {
            System.out.println("begin " + this.getClass());
        }

        @Override
        public FFT process(Sample input) {
            System.out.println("Input " + input);
            FFT fft = new FFT();
            return fft;
        }

        @Override
        public void end() {
            System.out.println("End " + this.getClass());
        }
        
    }

    static class FFTLogFactory implements IBuilder<FFTLog> {
        public static final FFTLogFactory INSTANCE = new FFTLogFactory();
        public FFTLog build() {
            return new FFTLog();
        }
    }
    
    
    static class FFTLog implements IProcessor<FFT, LogFFT> {

        public static final IBuilder<IProcessor<FFT, LogFFT>> BUILDER = new IBuilder() {
            @Override
            public Object build() {
                return new FFTLog();
            }
        };
        
        @Override
        public void begin() {
            System.out.println("begin " + this.getClass());
        }

        @Override
        public LogFFT process(FFT input) {
            System.out.println("Input " + input);
            LogFFT log = new LogFFT();
            return log;
        }

        @Override
        public void end() {
            System.out.println("End " + this.getClass());
        }
    };
    
    static class FFTLogToND4j implements IProcessor<LogFFT, ND4JVector> {
        public static final IBuilder<IProcessor<LogFFT, ND4JVector>> BUILDER = new IBuilder() {
            @Override
            public Object build() {
                return new FFTLogToND4j();
            }
        };

        @Override
        public void begin() {
            System.out.println("begin " + this.getClass());
        }

        @Override
        public ND4JVector process(LogFFT input) {
            System.out.println("Input " + input);
            return new ND4JVector();
        }

        @Override
        public void end() {
            System.out.println("End " + this.getClass());
        }
        
    }
    
    
    @Test
    public void testPipeline() {
        List<Sample[]> sampleList = new ArrayList<>();
        
        Sample[] s = new Sample[2];
        s[0] = new Sample();
        s[1] = new Sample();
        sampleList.add(s);
        
        new Pipeline(ChannelSelector.builder(Sample.class, 0).build())
            .add(SampleFFT.BUILDER.build())
            .add(FFTLog.BUILDER.build())
            .add(FFTLogToND4j.BUILDER.build())
            .execute(sampleList);
        
        
    }

}
