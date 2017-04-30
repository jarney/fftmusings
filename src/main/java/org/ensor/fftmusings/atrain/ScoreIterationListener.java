/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.atrain;


import java.io.PrintStream;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;

/**
 * Score iteration listener
 *
 * @author Adam Gibson
 */
public class ScoreIterationListener implements IterationListener {
    private long iterCount = 0;
    private long lastSampleTime;
    private final PrintStream logWriter;
    /**
     * @param writer Output writer.
     */
    public ScoreIterationListener(PrintStream writer) {
        lastSampleTime = System.currentTimeMillis();
        logWriter = writer;
    }

    @Override
    public boolean invoked(){ return iterCount > 0;}

    @Override
    public void invoke() {
        iterCount++;
    }

    @Override
    public void iterationDone(Model model, int iteration) {
        invoke();
        if (iterCount % 32 != 1) return;
        double result = model.score();
        long currentTime = System.currentTimeMillis();
        logWriter.println("Score at iteration " + iterCount + " is " + result + " took " + (currentTime - lastSampleTime) + " ms");
        lastSampleTime = currentTime;
    }
}
