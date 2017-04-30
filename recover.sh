#!/bin/bash

files=$(ls data/qft)

for file in $files ; do
    echo $file
    qft=data/qft/$file
    wav=data/wav2/$file.wav

    java -cp target/FFTMusings-1.0-SNAPSHOT-jar-with-dependencies.jar:target/FFTMusings-1.0-SNAPSHOT.jar org.ensor.fftmusings.preprocess.QFTToWAV $qft $wav
done

