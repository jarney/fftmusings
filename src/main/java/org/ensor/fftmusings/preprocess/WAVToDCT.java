/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.preprocess;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.PrintStream;
import javax.sound.sampled.AudioFileFormat;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import org.ensor.fftmusings.data.QuantizedSpectrum;
import org.ensor.fftmusings.data.Sample;
import org.ensor.fftmusings.io.ConcatStream;

/**
 *
 * @author jona
 */
public class WAVToDCT {
    public static void main(String[] args) throws Exception {
//        if (args.length != 2) {
//            System.out.println("Usage: WAVToSample foo.wav foo.qft");
//            System.exit(1);
//        }
//        process(args[0], args[1]);
        process("data/wav/20.wav", "20.wav");
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
        System.out.println("Big-Endial? : " + targetFormat.isBigEndian());
        
        ConcatStream cs = new ConcatStream();

        DCTProcess dct = new DCTProcess(DCTProcess.WINDOW_SIZE);
        
        FileOutputStream dctData = new FileOutputStream("20.csv");
        PrintStream ps = new PrintStream(dctData);
        
        int n = 0;
        while (true) {
            Sample[] s = WavIO.readSamples(sourceFormat, inputStream, DCTProcess.WINDOW_SIZE);
            if (s == null) {
                break;
            }
            
            for (int i = 0; i < s.length; i++) {
/*
                double[] rawData = s[i].getData();
                for (int j = 0; j < s[i].size(); j++) {
                    rawData[j] = 0.8*Math.sin((n*DCTProcess.WINDOW_SIZE + j)*Math.PI*2 / 32.0);
                }
*/
                dct.forward(s[i]);
                
                for (int j = 0; j < s[i].size(); j++) {
                    double tmp = Math.abs(s[i].getData()[j]);
                    if (tmp < 0.01) {
                        s[i].getData()[j] = 0;
                    }
//                    double tmp = (s[i].getData()[j*2] + s[i].getData()[j*2+1])/2.0;
//                    s[i].getData()[j*2] = tmp;
//                    s[i].getData()[j*2+1] = tmp;
                }
                
                // Do filtering here...
                if (n == 8 && i == 0) {
                    for (int j = 0; j < s[i].size(); j++) {
                        if ((j % 2) == 1) {
                            ps.println("" + j + "\t" + s[i].getData()[j]);
                        }
                    }
                }
                
                

                dct.reverse(s[i]);

            }
            InputStream data = WavIO.writeSamples(sourceFormat, s);
            cs.addStream(data, DCTProcess.WINDOW_SIZE);
            n++;
        }
        ps.close();
        
        AudioInputStream outputAIS = new AudioInputStream(cs, targetFormat,
                        cs.size());
        
        AudioSystem.write(outputAIS,
                AudioFileFormat.Type.WAVE,
                new File(String.format(outputFilename)));
    }
    
}
