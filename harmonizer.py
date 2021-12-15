import numpy as np
from music21 import *
from loader import music_loader
from train_model import build_model

chord_dictionary = ['Cm', 'C',
                    'C#m', 'C#',
                    'Dm', 'D',
                    'D#m', 'D#',
                    'Em', 'E',
                    'Fm', 'F',
                    'F#m', 'F#',
                    'Gm', 'G',
                    'G#m', 'G#',
                    'Am', 'A',
                    'A#m', 'A#',
                    'Bm', 'B']

def predict(song, model):
    
    chord_list = []

    # Traverse the melody sequence
    for idx in range(int(len(song)/4)):

        # Create input data
        melody = [song[idx*4], song[idx*4+1], song[idx*4+2], song[idx*4+3]]
        melody = np.array([np.array(seg) for seg in melody])[np.newaxis, ...]
        
        # Predict the next four chords
        net_output = model.predict(melody)[0]

        for chord_idx in net_output.argmax(axis=1):

            chord_list.append(chord_dictionary[chord_idx])
    
    # Create input data
    melody = [song[-4], song[-3], song[-2], song[-1]]
    melody = np.array([np.array(seg) for seg in melody])[np.newaxis, ...]

    
    # Predict the last four chords
    net_output = model.predict(melody).argmax(axis=1)[0]

    for i in range(-1*(len(song)%4), 0):

        chord_list.append(chord_dictionary[net_output[idx]])
    
    return chord_list


def export_music(melody_part, chord_list, filename):

    score = []
    next_offset = 0
    chord_cnt = 0
    ts = meter.TimeSignature('c')

    # Traverse melody part
    for element in melody_part.flat:

        score.append(element)

        # If is time signature
        if isinstance(element, meter.TimeSignature):

            ts = element

        # If is note and chord offset not greater than note offset
        if isinstance(element, note.Note) and len(chord_list)>chord_cnt and element.offset>=next_offset:
            
            # Converted to ChordSymbol
            chord_symbol = harmony.ChordSymbol(chord_list[chord_cnt])
            chord_symbol.offset = next_offset

            score.append(chord_symbol)
            chord_cnt += 1
            next_offset += ts.numerator*4/ts.denominator

    # Save as mxl
    score = stream.Stream(score)
    score.write('mxl', fp='outputs/'+filename.split('.')[-2]+'.mxl')


if __name__ == '__main__':

    # Build model
    model = build_model(weights_path='weights.hdf5')
    melody_data, melody_parts, filenames = music_loader(path='inputs', fromDataset=False)

    # Process each melody sequence
    for idx, melody in enumerate(melody_data):

        chord_list = predict(melody, model)
        export_music(melody_parts[idx], chord_list, filenames[idx])