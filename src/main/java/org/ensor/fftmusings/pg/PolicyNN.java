/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.pg;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.ensor.fftmusings.autoencoder.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 *
 * @author jona
 */
public class PolicyNN implements IPolicy {
    private final Random mRNG;
    
    private List<INDArray> roundInputs;
    private List<INDArray> roundOutputs;
    private MultiLayerNetwork mModel;
    
    private boolean initializedLastPrices = false;
    private double lastBTCPrice;
    private double lastLTCPrice;
    private double lastOTHPrice;
    private boolean initializedLastPrices2 = false;
    private double lastBTCPrice2;
    private double lastLTCPrice2;
    private double lastOTHPrice2;
    
    PolicyNN(Random rng) {
        mRNG = rng;
        
        roundInputs = new ArrayList<>();
        roundOutputs = new ArrayList<>();
        
        // This is the neural-network
        // for the policy.
        
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(System.currentTimeMillis())
                .iterations(1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.NESTEROVS)
                .regularization(true)
                .l1(0.001)
                .learningRate(0.001)
                .list()
                    .layer(0, new RBM.Builder()
                            .nIn(10)
                            .nOut(30)
                            .activation(Activation.TANH)
                            .sparsity(0.1)
                            .build())
                    .layer(1, new RBM.Builder()
                            .nIn(30)
                            .nOut(30)
                            .activation(Activation.TANH)
                            .sparsity(0.1)
                            .build())
                    .layer(2, new OutputLayer.Builder()
                            .nIn(30)
                            .nOut(12)
                            .lossFunction(LossFunctions.LossFunction.MSE)
                            .activation(Activation.SOFTMAX)
                            .build())
                .backprop(true).build();


        mModel = new MultiLayerNetwork(conf);
        
        mModel.init();
        mModel.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(System.out)));
    }
    
    public void nextRoundStart() {
        roundInputs.clear();
        roundOutputs.clear();
        initializedLastPrices = false;
        initializedLastPrices2 = false;
    }
    
    public void updateState(Holdings holdings) {
        
    }
    
    public IAction getAction(Holdings holdings) {
        // Map current market data into neural-network.
        
        // Input is mapped as follows into an input vector:
        // The same thing is replecated for 3 time-slices.
        // 0 : USD Holdings (as a % of total)
        // 1 : BTC holdings (as a % of total)
        // 2 : LTC holdings (as a % of total)
        // 3 : OTH holdings (as a % of total)
        // 4 : BTC change in price (as a %)
        // 5 : LTC change in price (as a %)
        // 6 : OTH change in price (as a %)
        INDArray currentInput = Nd4j.zeros(10);
        
        if (!initializedLastPrices) {
            lastBTCPrice = holdings.getMarketPriceOf("BTC");
            lastLTCPrice = holdings.getMarketPriceOf("LTC");
            lastOTHPrice = holdings.getMarketPriceOf("OTH");
            initializedLastPrices = true;
            return new ActionHold();
        }
        if (!initializedLastPrices2) {
            lastBTCPrice2 = lastBTCPrice;
            lastLTCPrice2 = lastLTCPrice;
            lastOTHPrice2 = lastOTHPrice;
            lastBTCPrice = holdings.getMarketPriceOf("BTC");
            lastLTCPrice = holdings.getMarketPriceOf("LTC");
            lastOTHPrice = holdings.getMarketPriceOf("OTH");
            initializedLastPrices2 = true;
            return new ActionHold();
        }
        
        currentInput.putScalar(0, holdings.getValueOf("USD") / holdings.totalValue());
        currentInput.putScalar(1, holdings.getValueOf("BTC") / holdings.totalValue());
        currentInput.putScalar(2, holdings.getValueOf("LTC") / holdings.totalValue());
        currentInput.putScalar(3, holdings.getValueOf("OTH") / holdings.totalValue());
        // Slight hack here to avoid divide by zero.
        currentInput.putScalar(4, (holdings.getValueOf("BTC") - lastBTCPrice) / (lastBTCPrice + 1e-4));
        currentInput.putScalar(5, (holdings.getValueOf("LTC") - lastLTCPrice) / (lastLTCPrice + 1e-4));
        currentInput.putScalar(6, (holdings.getValueOf("OTH") - lastOTHPrice) / (lastOTHPrice + 1e-4));
        currentInput.putScalar(7, (lastBTCPrice - lastBTCPrice2) / (lastBTCPrice2 + 1e-4));
        currentInput.putScalar(8, (lastLTCPrice - lastLTCPrice2) / (lastLTCPrice2 + 1e-4));
        currentInput.putScalar(9, (lastOTHPrice - lastOTHPrice2) / (lastOTHPrice2 + 1e-4));
        
        lastBTCPrice2 = lastBTCPrice;
        lastLTCPrice2 = lastLTCPrice;
        lastOTHPrice2 = lastOTHPrice;
        lastBTCPrice = holdings.getMarketPriceOf("BTC");
        lastLTCPrice = holdings.getMarketPriceOf("LTC");
        lastOTHPrice = holdings.getMarketPriceOf("OTH");

        
        roundInputs.add(currentInput);
        
        INDArray currentOutput = mModel.activateSelectedLayers(0, 2, currentInput);
        
        double dice = mRNG.nextDouble();
        double runningTotal = 0;
        int actionIndex = 0;
        for (actionIndex = 0; actionIndex < currentOutput.size(1); actionIndex++) {
            runningTotal += currentOutput.getDouble(actionIndex);
            if (dice < runningTotal) {
                break;
            }
        }
        
        // i is the index we chose for this iteration.
        // The it effectively the 'actual' action we selected
        // so we put this into this round's outputs.
        INDArray currentActualOutput = Nd4j.zeros(12);
        currentActualOutput.putScalar(actionIndex, 1);
        roundOutputs.add(currentActualOutput);
        
        // The policy we use is based on a neural network
        // output vector with a SOFTMAX classifier.
        // this means that the output sum to '1' and we can use it
        // as a probability that the action denoted by index 'i'
        // should be taken.
        // We map each index onto a different action and then sample
        // the probability of each index to find the correct action to take.
        // Action map:
        // 0: hold.
        // 1: buy BTC 25% of our USD holdings
        // 2: buy LTC 25% of our USD holdings
        // 3: buy OTH 25% of our USD holdings
        // 4: sell 25% of BTC holding
        // 5: sell 50% of BTC holding
        // 6: sell 100% of BTC holding
        // 7: sell 25% of LTC holding
        // 8: sell 50% of LTC holding
        // 9: sell 100% of LTC holding
        // 10: sell 25% of OTH holding
        // 11: sell 50% of OTH holding
        // 12: sell 100% of OTH holding
        
        switch (actionIndex) {
            case 0:
                return new ActionHold();
            case 1:
                return new ActionBuy("BTC", holdings.getValueOf("USD") / holdings.getMarketPriceOf("BTC") * 0.25);
            case 2:
                return new ActionBuy("LTC", holdings.getValueOf("USD") / holdings.getMarketPriceOf("LTC") * 0.25);
            case 3:
                return new ActionBuy("OTH", holdings.getValueOf("USD") / holdings.getMarketPriceOf("OTH") * 0.25);
            case 4:
                return new ActionBuy("BTC", -holdings.getAmountOf("BTC")*0.25);
            case 5:
                return new ActionBuy("BTC", -holdings.getAmountOf("BTC")*0.5);
            case 6:
                return new ActionBuy("BTC", -holdings.getAmountOf("BTC"));
            case 7:
                return new ActionBuy("LTC", -holdings.getAmountOf("LTC")*0.25);
            case 8:
                return new ActionBuy("LTC", -holdings.getAmountOf("LTC")*0.5);
            case 9:
                return new ActionBuy("LTC", -holdings.getAmountOf("LTC"));
            case 10:
                return new ActionBuy("OTH", -holdings.getAmountOf("OTH")*0.25);
            case 11:
                return new ActionBuy("OTH", -holdings.getAmountOf("OTH")*0.5);
            case 12:
                return new ActionBuy("OTH", -holdings.getAmountOf("OTH"));
        }
        return new ActionHold();
    }
    
    // Train model based on how well or badly
    // we did during our simulation.  This is a percentage
    // of how much better we did than the last round
    // of simulations.
    public void trainRound(double winLossAmount) {
        
        INDArray labels = Nd4j.zeros(roundOutputs.size(), 12);
        INDArray inputs = Nd4j.zeros(roundInputs.size(), 10);
        
        for (int i = 0; i < roundInputs.size(); i++) {
            
            // Always input the correct input we saw.
            inputs.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all()}, roundInputs.get(i));
            
            // If we succeeded, then we want to reinforce
            // these behaviors, so we put in the actual actions we took.
            if (winLossAmount > 0) {
                labels.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all()}, roundOutputs.get(i));
            }
            // If we didn't succeed, then we un-reinforce the
            // behaviors by assembling a vector that reinforces
            // 'everything except' what we did.
            else {
                // This is a vector of all '1'.
                INDArray whatWeDid = roundOutputs.get(i);
                INDArray everythingElse = Nd4j.zeros(12);
                everythingElse.addi(1).subi(whatWeDid).divi(everythingElse.size(1)-1);
                
                // We should now have 11/12 outputs asserted.
                // because we have subtracted the one that we selected.
                // We need to normalize this to sum to 1.  This means dividing
                // this by 11.
                labels.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all()}, everythingElse);
            }
        }
        
        DataSet dataSet = new DataSet(inputs, labels);
        
        // Perform one pass of data fitting.
        mModel.fit(dataSet);
        
    }
    
}
