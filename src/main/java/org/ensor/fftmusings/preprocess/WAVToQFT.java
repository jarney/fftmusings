/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package org.ensor.fftmusings.preprocess;

import org.ensor.fftmusings.data.QuantizedSpectrum;
import org.ensor.fftmusings.data.Sample;
import java.io.File;
import java.io.FileOutputStream;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;

/**
 *
 * @author jona
 */
public class WAVToQFT {
    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.out.println("Usage: WAVToSample foo.wav foo.qft");
            System.exit(1);
        }
        process(args[0], args[1]);
//        process("data/wav/20.wav", "20.qft");
    }
    
    
    public static void process(String inputFilename, String outputFilename) 
            throws Exception {
        
        AudioInputStream audioInputStream = AudioSystem.getAudioInputStream(new File(inputFilename));
        AudioFormat sourceFormat =  audioInputStream.getFormat();


        AudioFormat targetFormat = new AudioFormat(AudioFormat.Encoding.PCM_SIGNED,
                sourceFormat.getSampleRate(),
                sourceFormat.getSampleSizeInBits(),
                sourceFormat.getChannels(),
                sourceFormat.getFrameSize(),
                sourceFormat.getFrameRate(),
                sourceFormat.isBigEndian());

        AudioInputStream inputStream = AudioSystem.getAudioInputStream(targetFormat, audioInputStream);
        
        targetFormat = inputStream.getFormat();
        
        System.out.println("Channels : " + targetFormat.getChannels());
        System.out.println("Encoding : " + targetFormat.getEncoding());
        System.out.println("Sample Rate : " + targetFormat.getSampleRate());
        System.out.println("Sample Size (bits) : " + targetFormat.getSampleSizeInBits());
        System.out.println("Frame Rate : " + targetFormat.getFrameRate());
        System.out.println("Frame Size : " + targetFormat.getFrameSize());
        System.out.println("Big-Endian? : " + targetFormat.isBigEndian());
        
        FileOutputStream fos = new FileOutputStream(outputFilename);
        
        FFTProcess[] fftProcessor = new FFTProcess[2];
        fftProcessor[0] = new FFTProcess(FFTProcess.FFT_WINDOW_SIZE);
        fftProcessor[1] = new FFTProcess(FFTProcess.FFT_WINDOW_SIZE);
        
        while (true) {
            Sample[] s = WavIO.readSamples(sourceFormat, inputStream, FFTProcess.FFT_WINDOW_SIZE*2);
            if (s == null) {
                break;
            }
            
            for (int i = 0; i < s.length; i++) {
                QuantizedSpectrum q = fftProcessor[i].quantizeFFT(s[i]);
                q.write(fos);
            }
        }
        fos.close();
    }
}
