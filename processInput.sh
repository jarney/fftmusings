#!/bin/bash

inputList=$(find ~/Music/JonsMusic/ogg/The\ Smiths/ -name "*.ogg"|sed 's/ /+/g')

let fileNumber=0
for file in $inputList ; do
        infile=$(echo $file|sed 's/+/ /g')
        outfile=$fileNumber
	echo $infile $fileNumber

        sox "$infile" -r 11025 data/wav/$fileNumber.wav

        java -cp target/FFTMusings-1.0-SNAPSHOT-jar-with-dependencies.jar:target/FFTMusings-1.0-SNAPSHOT.jar org.ensor.fftmusings.preprocess.WAVToQFT data/wav/$fileNumber.wav data/qft/$fileNumber.qft

        let fileNumber=$fileNumber+1
done
