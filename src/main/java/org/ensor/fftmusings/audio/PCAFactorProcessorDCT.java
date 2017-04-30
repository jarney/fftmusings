/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.audio;

import org.ensor.fftmusings.pca.PCAFactory;

/**
 *
 * @author jona
 */
public class PCAFactorProcessorDCT extends PCAFactorProcessorBase<AudioDCTData> {
    public PCAFactorProcessorDCT(PCAFactory aPCAFactory, String aPCAFilename) {
        super(aPCAFactory, aPCAFilename);
    }

    @Override
    public void processPCA(AudioDCTData input) {
        addPoint(input.mSamples);
    }
}
