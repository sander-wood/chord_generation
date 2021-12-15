import os
import pickle
from tqdm import trange
from config import *
from music21 import *

def transpose(score):

    # Set default interval, key signature and tempo
    gap = interval.Interval(0)
    ks = key.KeySignature(0)

    for element in score.recurse():
        
        # Found key signature
        if isinstance(element, key.KeySignature) or isinstance(element, key.Key):

            if isinstance(element, key.KeySignature):

                ks = element.asKey()
            
            else:

                ks = element

            # Identify the tonic
            if ks.mode == 'major':
                
                tonic = ks.tonic

            else:

                tonic = ks.parallel.tonic

            # Transpose score
            gap = interval.Interval(tonic, pitch.Pitch('C'))
            score = score.transpose(gap)

            break

        # No key signature found
        elif isinstance(element, note.Note) or \
             isinstance(element, note.Rest) or \
             isinstance(element, chord.Chord):
            
            break
        
        else:

            continue
    
    return score


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


def music2txt(score, fromDataset):

    # Initialization
    melody_txt = []
    chord_txt = []
    harmony_txt = [] 

    score = transpose(score)
    melody_part = score.parts[0].flat

    if fromDataset:

        # Read score
        for element in score.parts[0].flat:

            # If is ChordSymbol
            if isinstance(element, harmony.ChordSymbol):
                
                harmony_txt.append(element)

    bar_txt = [0]*12
    
    if fromDataset:

        har_cnt = 0
        cur_harmony = harmony_txt[har_cnt]

    # Read score
    for element in score.parts[0].flat:

        # If is time signature
        if isinstance(element, meter.TimeSignature):

            ts = element

        if fromDataset:
            
            if element.offset<cur_harmony.offset:

                continue
                
            if har_cnt+1<len(harmony_txt) and harmony_txt[har_cnt+1].offset<=element.offset and not isinstance(element, harmony.ChordSymbol):

                har_cnt += 1
                    
                if harmony_txt[har_cnt].offset%(ts.numerator*4/ts.denominator)==0:
                   
                    cur_harmony = harmony_txt[har_cnt]

        # If is note or chord
        if isinstance(element, note.Note) or isinstance(element, note.Rest) or isinstance(element, chord.Chord) and not isinstance(element, harmony.ChordSymbol):
            
            if isinstance(element, note.Note):

                idx = element.pitch.midi%12
                duration = element.quarterLength
            
            if isinstance(element, note.Rest):

                duration = element.quarterLength
                continue

            if isinstance(element, chord.Chord):

                idx = [sub_ele.pitch.midi for sub_ele in element.notes]
                idx = sorted(idx)[-1]%12
                duration = element.quarterLength
            
            if (element.offset+element.quarterLength-ts.offset)%(ts.numerator*4/ts.denominator)==0:

                bar_txt[idx]+=duration/(ts.numerator*4/ts.denominator)
                melody_txt.append(bar_txt)
                bar_txt = [0]*12

                if fromDataset:
    
                    chord_txt.append(harmony2idx(cur_harmony))
            
            else:

                bar_txt[idx]+=duration/(ts.numerator*4/ts.denominator)
                
    if fromDataset:

        return melody_txt, chord_txt
    
    else:

        return melody_txt, melody_part


def music_loader(path=DATASET_PATH, fromDataset=True):

    # Initialization
    melody_data = []
    chord_data = []
    melody_parts = []
    filenames = []

    # Traverse the path
    for dirpath, dirlist, filelist in os.walk(path):
        
        # Traverse the list of files
        for file_idx in trange(len(filelist)):

            this_file = filelist[file_idx]

            # Ensure that suffixes in the training set are valid
            if os.path.splitext(this_file)[-1] not in EXTENSION:

                continue
        
            filename = os.path.join(dirpath, this_file)

            # Read the this music file
            score = converter.parse(filename)

            if fromDataset:

                # Converte music to text data
                melody_txt, chord_txt = music2txt(score, fromDataset=True)
                
                if len(melody_txt)!=0:
                    
                    melody_data.append(melody_txt)
                    chord_data.append(chord_txt)
                
            else:

                # Converte music to text data
                melody_txt, melody_part = music2txt(score, fromDataset=False)

                if len(melody_txt)!=0:

                    melody_data.append(melody_txt)
                    melody_parts.append(melody_part)
                    filenames.append(this_file)
                    
    print("Successfully encoded %d pieces" %(len(melody_data)))
    
    if fromDataset:

        return (melody_data, chord_data)
    
    else:

        return (melody_data, melody_parts, filenames)


if __name__ == "__main__":

    # Read encoded music information and file names
    corpus = music_loader()
    
    # Save as corpus
    with open(CORPUS_PATH, "wb") as filepath:
        pickle.dump(corpus, filepath)
