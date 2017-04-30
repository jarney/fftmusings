/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.rnn.qft;


import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Score iteration listener
 *
 * @author Adam Gibson
 */
public class ScoreIterationListener implements IterationListener {
    private boolean invoked = false;
    private long iterCount = 0;

    /**
     * @param printIterations    frequency with which to print scores (i.e., every printIterations parameter updates)
     */
    public ScoreIterationListener() {
    }

    @Override
    public boolean invoked(){ return invoked; }

    @Override
    public void invoke() { this.invoked = true; }

    @Override
    public void iterationDone(Model model, int iteration) {
        invoke();
        double result = model.score();
        System.out.println("Score at iteration " + iterCount + " is " + result);
        iterCount++;
    }
}
