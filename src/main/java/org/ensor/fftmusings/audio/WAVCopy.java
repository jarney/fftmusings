/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.audio;

import org.ensor.fftmusings.io.ICloseableIterator;
import org.ensor.fftmusings.pipeline.ChannelDuplicator;
import org.ensor.fftmusings.pipeline.ChannelSelector;
import org.ensor.fftmusings.pipeline.Pipeline;

/**
 *
 * @author jona
 */
public class WAVCopy {

    public static void main(String[] args) throws Exception {
        args = new String[2];
        args[0] = "data/wav/20.wav";
        args[1] = "sample.wav";
        
        if (args.length != 2) {
            System.out.println("Usage: WAVToSample foo.wav foo2.wav");
            System.exit(1);
        }
        
        int audioSampleSize = 512;
        int fftWindowSize = audioSampleSize*2;
        
        try (ICloseableIterator<AudioSample[]> wavFileIterator = WAVFileIterator.create(args[0], audioSampleSize)) {
            new Pipeline(new ChannelSelector(AudioSample.class, 0))
                    .add(new PhaseVocoder.Encoder(11025, new FFTOverlap.Forward(fftWindowSize)))
                    .add(new PhaseVocoder.Decoder(11025, new FFTOverlap.Reverse(fftWindowSize)))
                    .add(new ChannelDuplicator(AudioSample.class, 2))
                    .add(WAVFileWriter.create("vocode-test.wav"))
                    .execute(wavFileIterator);

        }
    }
    
    
    
}
