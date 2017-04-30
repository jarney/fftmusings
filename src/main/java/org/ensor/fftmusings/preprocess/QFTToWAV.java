/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package org.ensor.fftmusings.preprocess;

import org.ensor.fftmusings.io.ConcatStream;
import org.ensor.fftmusings.data.QuantizedSpectrum;
import org.ensor.fftmusings.data.Sample;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import javax.sound.sampled.AudioFileFormat;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;

/**
 *
 * @author jona
 */
public class QFTToWAV {
    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.out.println("Usage: WAVFromSample foo.qft foo.wav");
            System.exit(1);
        }
        process(args[0], args[1]);
//        process("20.qft", "20.1.wav");
    }
    public static void process(String inputFilename, String outputFilename) 
            throws Exception {
        FileInputStream fis = new FileInputStream(inputFilename);
        
        AudioFormat targetFormat = new AudioFormat(AudioFormat.Encoding.PCM_SIGNED,
                11025.0f, // Sample rate
                16, // Sample size (bits)
                2, // Channels
                4, // Frame Size
                11025.0f, // Frame Rate
                false); // Big-endian
        
        ConcatStream cs = new ConcatStream();


        FFTProcess[] fftProcessor = new FFTProcess[2];
        fftProcessor[0] = new FFTProcess(FFTProcess.FFT_WINDOW_SIZE);
        fftProcessor[1] = new FFTProcess(FFTProcess.FFT_WINDOW_SIZE);
        
        while (true) {
            QuantizedSpectrum[] qs = new QuantizedSpectrum[2];
            qs[0] = new QuantizedSpectrum(FFTProcess.FFT_WINDOW_SIZE);
            qs[1] = new QuantizedSpectrum(FFTProcess.FFT_WINDOW_SIZE);
            int rc;
            rc = qs[0].read(fis);
            if (rc == -1) {
                break;
            }
            rc = qs[1].read(fis);
            if (rc == -1) {
                break;
            }
            
            Sample[] s = new Sample[2];
            s[0] = fftProcessor[0].quantizedInverseFFT(qs[0]);
            s[1] = fftProcessor[1].quantizedInverseFFT(qs[1]);
            
            InputStream bais = WavIO.writeSamples(targetFormat, s);
            cs.addStream(bais, s[0].size());
        }
        
        
        AudioInputStream outputAIS = new AudioInputStream(cs, targetFormat,
                        cs.size());

        AudioSystem.write(outputAIS,
                AudioFileFormat.Type.WAVE,
                new File(String.format(outputFilename)));
    }
}
