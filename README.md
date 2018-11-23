# PytorchMusicLSTM
A project I made to better understand LSTM networks. 

This is currently a work in progress, I plan it make the program more easily tweakable, and eventually incorperate music with multiple instruments.

Music files are midi0 format, meaning there is 128 different frequences on a single instrument(Piano).

I used a network that had 129 inputs, so inputs from 0-127 are frequencies for notes, and 128 is a time step.


All music used to train the network was downloaded from: http://www.piano-midi.de/midicoll.htm
