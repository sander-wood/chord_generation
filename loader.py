import os
import pickle
import numpy as np
from copy import deepcopy
from tqdm import trange
from music21 import *
from config import *

def ks2gap(ks):
    
    if isinstance(ks, key.KeySignature):
        ks = ks.asKey()
        
    try:
        # Identify the tonic
        if ks.mode == 'major':
            tonic = ks.tonic

        else:
            tonic = ks.parallel.tonic
    
    except:
        return interval.Interval(0)

    # Transpose score
    gap = interval.Interval(tonic, pitch.Pitch('C'))

    return gap


def get_filenames(input_dir):
    
    filenames = []

    # Traverse the path
    for dirpath, dirlist, filelist in os.walk(input_dir):
        # Traverse the list of files
        for this_file in filelist:
            # Ensure that suffixes in the training set are valid
            if input_dir==DATASET_PATH and os.path.splitext(this_file)[-1] not in EXTENSION:
                continue
            filename = os.path.join(dirpath, this_file)
            filenames.append(filename)
    
    return filenames


def harmony2idx(element):

    # Extracts the MIDI pitch of each note in a harmony
    pitch_list = [sub_ele.pitch.midi for sub_ele in element.notes]
    pitch_list = sorted(pitch_list)

    bass_note = pitch_list[0]%12
    quality = pitch_list[min(1,len(pitch_list)-1)]-pitch_list[0]

    if quality<=3:
        quality = 0

    else:
        quality = 1
        
    return bass_note*2+quality


def melody_reader(score):

    melody_vecs = []
    chord_list = []
    gap_list = []
    last_chord = 0
    last_ks = key.KeySignature(0)

    for m in score.recurse():
        if isinstance(m, stream.Measure):
            vec = [0]*12
            if m.keySignature!=None:
                gap = ks2gap(m.keySignature)
                last_ks = m.keySignature
            
            else:
                gap = ks2gap(last_ks)

            gap_list.append(gap)
            this_chord = None

            for n in m:
                if isinstance(n, note.Note):
                    # midi pitch as note onset
                    token = n.transpose(gap).pitch.midi
                    
                elif isinstance(n, chord.Chord) and not isinstance(n, harmony.ChordSymbol):
                    notes = [n.transpose(gap).pitch.midi for n in n.notes]
                    notes.sort()
                    token = notes[-1]
                    
                elif isinstance(n, harmony.ChordSymbol) and np.sum(vec)==0:
                    this_chord = harmony2idx(n.transpose(gap))+1
                    last_chord = this_chord
                    continue

                else:
                     continue

                vec[token%12] += float(n.quarterLength)
            
            if np.sum(vec)!=0:
                vec = np.array(vec)/np.sum(vec)
            melody_vecs.append(vec)

            if this_chord==None:
                this_chord = last_chord
            
            chord_list.append(this_chord)
            
    return melody_vecs, chord_list, gap_list


def convert_files(filenames, fromDataset=True):

    print('\nConverting %d files...' %(len(filenames)))
    failed_list = []
    data_corpus = []

    for filename_idx in trange(len(filenames)):

        # Read this music file
        filename = filenames[filename_idx]
        
        try:
            
            score = converter.parse(filename)
            score = score.parts[0]
            if not fromDataset:
                original_score = deepcopy(score)
            song_data = []

            melody_vecs, chord_txt, gap_list = melody_reader(score)

            if fromDataset:
                song_data.append((melody_vecs, chord_txt))

            else:
                data_corpus.append((melody_vecs, gap_list, original_score, filename))
            
            if len(song_data)>0:
                data_corpus.append(song_data)

        except Exception as e:
            failed_list.append((filename, e))

    print('Successfully converted %d files.' %(len(filenames)-len(failed_list)))
    if len(failed_list)>0:
        print('Failed numbers: '+str(len(failed_list)))
        print('Failed to process: \n')
        for failed_file in failed_list:
            print(failed_file)

    if fromDataset:
        with open(CORPUS_PATH, "wb") as filepath:
            pickle.dump(data_corpus, filepath)
    
    else:
        return data_corpus


if __name__ == '__main__':

    filenames = get_filenames(input_dir=DATASET_PATH)
    convert_files(filenames)