# Chord Generation from Symbolic Melody Using BLSTM Networks

This is the reproduction of 'Chord Generation from Symbolic Melody Using BLSTM Networks'. \
\
The input melodies and harmonized samples are in the `inputs` and `outputs` folders respectively.\
\
This reproduced model is the same as the original setup, except that we replaced the dataset with the Nottingham Lead Sheet Dataset (the original one trained/validated on the Wikifonia Dataset).\
\
For more information, see their paper: arXiv paper.

## Install Dependencies
Python: 3.7.9\
Keras: 2.3.0\
tensorflow-gpu: 2.2.0\
music21: 6.7.1\
\
PS: Third party libraries can be installed using the `pip install` command.

## Melody Harmonization
1.　Put the melodies (MIDI or MusicXML) in the `inputs` folder;\
2.　Simply run `harmonizer.py`;\
3.　Wait a while and the harmonized melodies will be saved in the `outputs` folder.

## Use Your Own Dataset
1.　Store all the lead sheets (MusicXML) in the `dataset` folder;\
2.　Run `loader.py`, which will generate `orpus.bin`; \
3.　Run `train_model.py`, which will generate `weights.hdf5`.\
\
After that, you can use `harmonizer.py` to harmonize music that with chord progressions that fit the musical style of the new dataset. \
\
If you need to finetune the parameters, you can do so in `config.py`. It is not recommended to change the parameters in other files.
