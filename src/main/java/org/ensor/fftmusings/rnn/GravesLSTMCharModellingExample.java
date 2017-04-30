/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.rnn;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * GravesLSTM Character modelling example
 *
 * @author Alex Black
 *
 * Example: Train a LSTM RNN to generates text, one character at a time. This
 * example is somewhat inspired by Andrej Karpathy's blog post, "The
 * Unreasonable Effectiveness of Recurrent Neural Networks"
 * http://karpathy.github.io/2015/05/21/rnn-effectiveness/
 *
 * Note that this example has not been well tuned - better performance is likely
 * possible with better hyperparameters
 *
 * Some differences between this example and Karpathy's work: - The LSTM
 * architectures appear to differ somewhat. GravesLSTM has peephole connections
 * that Karpathy's char-rnn implementation appears to lack. See GravesLSTM
 * javadoc for details. There are pros and cons to both architectures (addition
 * of peephole connections is a more powerful model but has more parameters per
 * unit), though they are not radically different in practice. - Karpathy uses
 * truncated backpropagation through time (BPTT) on full character sequences,
 * whereas this example uses standard (non-truncated) BPTT on partial/subset
 * sequences. Truncated BPTT is probably the preferred method of training for
 * this sort of problem, and is configurable using the
 * .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength().tBPTTBackwardLength()
 * options
 *
 * This example is set up to train on the Complete Works of William Shakespeare,
 * downloaded from Project Gutenberg. Training on other text sources should be
 * relatively easy to implement.
 */
public class GravesLSTMCharModellingExample {

    public static void main(String[] args) throws Exception {
        int numEpochs = 30;							//Total number of training + sample generation epochs
        String generationInitialization = null;		//Optional character initialization; a random character is used if null
        int nSamplesToGenerate = 4;					//Number of samples to generate after each training epoch
        int nCharactersToSample = 300;				//Length of each sample to generate
        Random rng = new Random(12345);
        int miniBatchSize = 32;						//Size of mini batch to use when  training
        int examplesPerEpoch = 50 * miniBatchSize;	//i.e., how many examples to learn on between generating samples
        int exampleLength = 100;					//Length of each training example

        //Get a DataSetIterator that handles vectorization of text into something we can use to train
        // our GravesLSTM network.
        CharacterIterator iter = getShakespeareIterator(miniBatchSize, exampleLength, examplesPerEpoch);

        File modelFilename = new File("data/shakespere/shakespere.3.rnn");

        MultiLayerNetwork net = RNNFactory.create(modelFilename, iter);

        //Print the  number of parameters in the network (and for each layer)
        Layer[] layers = net.getLayers();
        int totalNumParams = 0;
        for (int i = 0; i < layers.length; i++) {
            int nParams = layers[i].numParams();
            System.out.println("Number of parameters in layer " + i + ": " + nParams);
            totalNumParams += nParams;
        }
        System.out.println("Total number of network parameters: " + totalNumParams);

        //Do training, and then generate and print samples from network
        for (int i = 0; i < numEpochs; i++) {
            net.fit(iter);

            System.out.println("--------------------");
            System.out.println("Completed epoch " + i);
            System.out.println("Sampling characters from network given initialization \"" + (generationInitialization == null ? "" : generationInitialization) + "\"");
            for (int j = 0; j < nSamplesToGenerate; j++) {
                String samples = sampleCharactersFromNetwork2(generationInitialization, net, iter, rng, nCharactersToSample);
                System.out.println("----- Sample " + j + " -----");
                System.out.println(samples);
                System.out.println();
            }

            RNNFactory.persist(modelFilename, net);

            iter.reset();	//Reset iterator for another epoch
        }

        System.out.println("\n\nExample complete");
    }

    /**
     * Downloads Shakespeare training data and stores it locally (temp
     * directory). Then set up and return a simple DataSetIterator that does
     * vectorization based on the text.
     *
     * @param miniBatchSize Number of text segments in each training mini-batch
     * @param exampleLength Number of characters in each text segment.
     * @param examplesPerEpoch Number of examples we want in an 'epoch'.
     */
    private static CharacterIterator getShakespeareIterator(int miniBatchSize, int exampleLength, int examplesPerEpoch) throws Exception {
        //The Complete Works of William Shakespeare
        //5.3MB file in UTF-8 Encoding, ~5.4 million characters
        //https://www.gutenberg.org/ebooks/100
        String url = "https://s3.amazonaws.com/dl4j-distribution/pg100.txt";
        String fileLocation = "data/shakespere/Shakespeare.txt";	//Storage location from downloaded file
        File f = new File(fileLocation);
        if (!f.exists()) {
            FileUtils.copyURLToFile(new URL(url), f);
            System.out.println("File downloaded to " + f.getAbsolutePath());
        } else {
            System.out.println("Using existing text file at " + f.getAbsolutePath());
        }

        if (!f.exists()) {
            throw new IOException("File does not exist: " + fileLocation);	//Download problem?
        }
        char[] validCharacters = CharacterIterator.getMinimalCharacterSet();	//Which characters are allowed? Others will be removed
        return new CharacterIterator(fileLocation, Charset.forName("UTF-8"),
                miniBatchSize, exampleLength, examplesPerEpoch, validCharacters, new Random(12345), true);
    }


    private static String sampleCharactersFromNetwork2(
            String initialization,
            MultiLayerNetwork net,
            CharacterIterator iter,
            Random rng,
            int charactersToSample) {
        
        StringBuilder sb = new StringBuilder();
        
        //Set up initialization. If no initialization: use a random character
        if (initialization == null) {
            initialization = String.valueOf(iter.getRandomCharacter());
        }
        //Create input for initialization
        INDArray initializationInput = Nd4j.zeros(iter.inputColumns());
        char[] init = initialization.toCharArray();
        for (int i = 0; i < init.length; i++) {
            int idx = iter.convertCharacterToIndex(init[i]);
            initializationInput.putScalar(new int[]{idx}, 1.0f);
        }
        
        net.rnnClearPreviousState();

        INDArray output = net.rnnTimeStep(initializationInput);
        
        for (int i = 0; i < charactersToSample; i++) {
            //Set up next input (single time step) by sampling from previous output
            INDArray nextInput = Nd4j.zeros(iter.inputColumns());
            //Output is a probability distribution. Sample from this for each example we want to generate, and add it to the new input
            double[] outputProbDistribution = new double[iter.totalOutcomes()];
            for (int j = 0; j < outputProbDistribution.length; j++) {
                outputProbDistribution[j] = output.getDouble(j);
            }
            int sampledCharacterIdx = sampleFromDistribution(outputProbDistribution, rng);

            nextInput.putScalar(new int[]{sampledCharacterIdx}, 1.0f);		//Prepare next time step input
            sb.append(iter.convertIndexToCharacter(sampledCharacterIdx));	//Add sampled character to StringBuilder (human readable output)

            output = net.rnnTimeStep(nextInput);	//Do one time step of forward pass
        }
        
        return sb.toString();
    }
    static class Dist {
        double prob;
        int charIndex;
    };
    
    static class DistComparator implements Comparator<Dist> {
        @Override
        public int compare(Dist o1, Dist o2) {
            if (o1.prob == o2.prob) {
                return 0;
            }
            return o1.prob < o2.prob ? 1 : -1;
        }
    };
    static final DistComparator comparator = new DistComparator();
    
    private static int sampleFromDistribution2(double[] distribution, Random rng) {

        List<Dist> list = new ArrayList<>();
        for (int i = 0; i < distribution.length; i++) {
            Dist d = new Dist();
            d.prob = distribution[i];
            d.charIndex = i;
            list.add(d);
        }
        list.sort(comparator);
        int idx = Math.abs(rng.nextInt()) % 3;
        return list.get(idx).charIndex;
    }

    /**
     * Given a probability distribution over discrete classes, sample from the
     * distribution and return the generated class index.
     *
     * @param distribution Probability distribution over classes. Must sum to
     * 1.0
     */
    private static int sampleFromDistribution(double[] distribution, Random rng) {
        double d = rng.nextDouble();

        double sum = 0.0;
        for (int i = 0; i < distribution.length; i++) {
            sum += distribution[i];
            if (d <= sum) {
                return i;
            }
        }
        //Should never happen if distribution is a valid probability distribution
        throw new IllegalArgumentException("Distribution is invalid? d=" + d + ", sum=" + sum);
    }
}
