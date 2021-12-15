import pickle
import os
import numpy as np
import keras_metrics as km
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras import Model
from keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils.np_utils import to_categorical
from config import *

def create_training_data(corpus_path=CORPUS_PATH, val_ratio=VAL_RATIO):

    # Load corpus
    with open(corpus_path, "rb") as filepath:
        corpus = pickle.load(filepath)

    # Inputs and targets for the training set
    input_melody = []
    output_chord = []

    # Inputs and targets for the validation set
    val_input_melody = []
    val_output_chord = []

    melody_data = corpus[0]
    chord_data = corpus[1]
    cnt = 0
    np.random.seed(0)

    # Process each melody sequence in the corpus
    for idx, melody_txt in enumerate(melody_data):
        
        # Load chord data
        chord_txt = chord_data[idx]

        # Randomly assigned to the training or validation set with the probability
        if np.random.rand()>val_ratio:

            train_or_val = 'train'
        
        else:

            train_or_val = 'val'

        # Create pairs
        for idx in range(len(melody_txt)-3):
            
            melody = [melody_txt[idx], melody_txt[idx+1], melody_txt[idx+2], melody_txt[idx+3]]
            melody = [np.array(seg) for seg in melody]
            chord = [chord_txt[idx], chord_txt[idx+1], chord_txt[idx+2], chord_txt[idx+3]]
            chord= to_categorical(chord, num_classes=24)

            if train_or_val=='train':
                
                input_melody.append(melody)
                output_chord.append(chord)
            
            else:

                val_input_melody.append(melody)
                val_output_chord.append(chord)

            cnt += 1

    print("Successfully read %d samples" %(cnt))
    input_melody = np.array(input_melody).reshape(len(input_melody), 4, 12)
    output_chord = np.array(output_chord).reshape(len(output_chord), 4, 24)
    val_input_melody = np.array(val_input_melody).reshape(len(val_input_melody), 4, 12)
    val_output_chord = np.array(val_output_chord).reshape(len(val_output_chord), 4, 24)
    
    return (input_melody, output_chord), (val_input_melody, val_output_chord)


def build_model(rnn_size=RNN_SIZE, num_layers=NUM_LAYERS, weights_path=None):

    # Create input layer
    input_melody = Input(shape=(4, 12), 
                        name='input_melody')
    melody = TimeDistributed(Dense(12))(input_melody)


    # Creating the hidden layer of the LSTM
    for idx in range(num_layers):
        
        melody = Bidirectional(LSTM(units=rnn_size, 
                                    return_sequences=True,
                                    name='melody_'+str(idx+1)))(melody)
        melody = TimeDistributed(Dense(units=rnn_size, activation='tanh'))(melody)
        melody = Dropout(0.2)(melody)


    # Create Dense hidden layers
    output_layer = TimeDistributed(Dense(24, activation='softmax'))(melody)

    model = Model(
                  inputs=input_melody,
                  outputs=output_layer
                 )

    model.compile(optimizer='adam',
                  loss=['categorical_crossentropy'],
                  metrics=['accuracy', km.f1_score()])
    
    if weights_path==None:

        model.summary()

    else:

        model.load_weights(weights_path)

    return model


def train_model(data,
                data_val, 
                batch_size=BATCH_SIZE, 
                epochs=EPOCHS, 
                verbose=1,
                weights_path=WEIGHTS_PATH):

    model = build_model()

    # Load or remove existing weights
    if os.path.exists(weights_path):
        
        try:

            model.load_weights(weights_path)
            print("checkpoint loaded")
        
        except:

            os.remove(weights_path)
            print("checkpoint deleted")

    # Set monitoring indicator
    if len(data_val[0])!=0:

        monitor = 'val_loss'

    else:

        monitor = 'loss'

    # Save weights
    checkpoint = ModelCheckpoint(filepath=weights_path,
                                 monitor=monitor,
                                 verbose=0,
                                 save_best_only=True,
                                 mode='min')

    if len(data_val[0])!=0:

        # With validation set
        history = model.fit(x={'input_melody': np.array(data[0])},
                            y=np.array(data[1]),
                            validation_data=({'input_melody': np.array(data_val[0])}, 
                                                data_val[1]),
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=verbose,
                            callbacks=[checkpoint])
    else:

        # Without validation set
        history = model.fit(x={'input_melody': np.array(data[0])},
                            y=np.array(data[1]),
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=verbose,
                            callbacks=[checkpoint])
    
    return history


if __name__ == "__main__":

    # Load the training and validation sets
    data, data_val = create_training_data()
    
    # Train model
    history = train_model(data, data_val)