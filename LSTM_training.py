'''
Based on keras char lstm example

https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py
12 real-valued inputs — and the targets would be a categorical distribution of possible chords.'''



#from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, Bidirectional, TimeDistributed
from keras.optimizers import RMSprop
from keras.optimizers import Nadam
from keras.optimizers import SGD
from keras.optimizers import Adam


import numpy as np
import random
import sys
import io

import matplotlib.pyplot as plt
import pydot
from keras.utils import plot_model




#           LOAD TRAINING DATA


f = open('dataset_60minor.txt', 'r')
chord_list = f.read().split()
print("Length of chord_list: ", len(chord_list))

note_zip = np.load('60min_nvs.npy.npz')
note_vecs = note_zip['arr_0']
print("Length of note_vecs: ", len(note_vecs))

#           VALIDATION DATA
f = open('dataset_60_VALmin.txt', 'r')
chord_list_VAL = f.read().split()
print("Length of chord_list_VAL: ", len(chord_list_VAL))

note_zip = np.load('nvs_VAL.npy.npz')
note_vecs_VAL = note_zip['arr_0']
print("Length of note_vecs: ", len(note_vecs_VAL))





#          LOAD TEST DATA

f_test = open('chords_test_60.txt', 'r')
chord_list_test = f_test.read().split()
print("Length of test chords: ", len(chord_list_test))

note_test_zip = np.load('nvs_test.npy.npz')
note_vecs_test = note_test_zip['arr_0']

# 60 chord tokens
chords = 'C Cm Caug Cdim Csus C# C#m C#aug C#dim C#sus D Dm Daug Ddim Dsus D# D#m D#aug D#dim D#sus E Em Eaug Edim Esus F Fm Faug Fdim Fsus F# F#m F#aug F#dim F#sus G Gm Gaug Gdim Gsus G# G#m G#aug G#dim G#sus A Am Aaug Adim Asus A# A#m A#aug A#dim A#sus B Bm Baug Bdim Bsus ENDTOKEN'.split()

# 24 chord tokens
#chords = 'C C# D D# E F F# G G# A A# B Cm C#m Dm D#m Em Fm F#m Gm G#m Am A#m Bm ENDTOKEN'.split()

chord_indices = dict((c, i) for i, c in enumerate(chords))
indices_chord = dict((i, c) for i, c in enumerate(chords))

examples = []
nv_ex = []
examples_val = []
nv_ex_val = []


seq_len = 16  #length of example
batch =  128 # how many examples to look at at once. Adjusting this effects the optimisers choice by giving more or less information at one time
step = 1 # steps to move through bars when creating trraining data
layers = 256 # units in LSTM
dropout = 0.2


#           PROCESS TRAINING/VALIDATION DATA
# cut the training data and validation data in semi-redundant sequences of seq_len lenght

for i in range(0, len(chord_list) - seq_len, step):
    examples.append(chord_list[i: i + seq_len])
    for x in range(i, i+seq_len):
        nv_ex.append(note_vecs[x])

for i in range(0, len(chord_list_VAL) - seq_len, step):
    examples_val.append(chord_list_VAL[i: i + seq_len])
    #print("First chord in example (VAL): ", chord_list_test[i])
    #print("Note vector for first chord (VAL): ", note_vecs_test[i])
    for x in range(i, i+seq_len):
        nv_ex_val.append(note_vecs_VAL[x])



# create x, y:

x = np.zeros((len(examples), seq_len, 12))
y = np.zeros((len(examples),seq_len, len(chords)))

nv_index = 0


for i , ex in enumerate(examples):
    for t, chord in enumerate(ex):
        #assert i+t < len(nv_ex)
        x[i, t] = nv_ex[nv_index]
        y[i, t, int(chord)] = 1
        nv_index += 1

print(y.shape)
print(x.shape)
#print(len(examples))
#print(len(nv_ex))

#                       VALIDATION DATA

x_val = np.zeros((len(examples_val), seq_len, 12))
y_val = np.zeros((len(examples_val), seq_len, len(chords)))

nv_index = 0

for i , ex in enumerate(examples_val):
    #print(" ------- ")
    for t, chord in enumerate(ex):
        #assert i+t < len(nv_ex_val)
        x_val[i, t] = nv_ex_val[nv_index]
        y_val[i, t, int(chord)] = 1
    #    print("Note vector for chord number ",t," ",chord," (VAL): ", nv_ex_val[nv_index])
        nv_index += 1


# build the model: a single LSTM
print('Build model...')

model = Sequential()


model.add(LSTM(layers, input_shape=(seq_len, 12), return_sequences=True)) #LSTM default activation = tanh
model.add(Dropout(dropout))
model.add(LSTM(layers, go_backwards=True,return_sequences=True))
model.add(Dropout(dropout))
model.add(LSTM(layers, return_sequences=True))
model.add(Dropout(dropout))
model.add(LSTM(layers, go_backwards=True, return_sequences=True))
model.add(Dropout(dropout))
model.add(TimeDistributed(Dense(len(chords))))
model.add(Activation('softmax'))

optimizer = Adam()
model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['acc'])


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    num_preds = 8

    start_index = random.randint(0, len(chord_list_test) - seq_len - 1 - num_preds)
    print("Start index: ", start_index)

    correct_next = chord_list_test[(start_index+seq_len):(start_index+seq_len+num_preds)]
    
    print("Correct next 8: ", correct_next)

    for diversity in [0.8, 0.1, 0.35, 0.5]:
        print('----- diversity:', diversity)

        generated = ''
        
        sentence = chord_list_test[start_index: start_index + seq_len]
        
        generated += str(sentence)
        print('----- Generating with seed: "' + str(sentence) + '"')
        sys.stdout.write(generated)
        nv_ex_idx = start_index
        sentence_start_idx = start_index
        counter = start_index


        for i in range(num_preds): # NUM OF PREDICTIONS TO ADD TO GENERATED

            nv_ex_idx = sentence_start_idx 

            x_pred = np.zeros((1, seq_len, 12))
            #print("len sentence: ", len(sentence))
            for i in range(seq_len):
                x_pred[0, i] = note_vecs_test[nv_ex_idx]
                #print("index of vec: ", nv_ex_idx, " chord name: ",chord_list[nv_ex_idx])
                nv_ex_idx +=1


            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds[-1], diversity)
            
            # GET CHORD NAME FROM PREDICTION
            next_chord = indices_chord[next_index]


            generated += " " +  next_chord + " "
            
            # sentence should contain latest generated chord for next itteration, 
            sentence = sentence[1:] + next_chord.split()
            sentence_start_idx += 1 


            sys.stdout.write(next_chord + " ")
            sys.stdout.flush()
        print()

    correct_vals = 0
    gen_list = generated.split()
    gen_list = gen_list[-num_preds:]
    for x in range(num_preds):
        #print("correct val ", x ," = ", correct_next[x])
        #print("generated ",x ," = ", gen_list[x])
        if gen_list[x] == correct_next[x]:
            correct_vals += 1
    
    print("Correct predictions: ", correct_vals)
    print("precentage of correct predictions: ", correct_vals*100/num_preds)


print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
earlyStopping = EarlyStopping(patience=1, monitor='val_acc')
tensor_board = TensorBoard(histogram_freq=1,embeddings_freq=0, write_grads=True, write_images=True)
checkpointer = ModelCheckpoint(monitor='val_acc',filepath='logs/weights.hdf5', verbose=1, save_best_only=True)
history = model.fit(x, y,
          batch_size=batch,
          epochs=5,
          #callbacks=[print_callback],
          callbacks=[checkpointer, print_callback],
          validation_data=(x_val,y_val)
          )


score = model.evaluate(x_val, y_val)
print("SCORE: ", score)



x_pred = np.zeros((1, seq_len, 12))

for i in range(seq_len):
    x_pred[0, i] = note_vecs_test[i]
    print(note_vecs_test[i])
    
preds = model.predict(x_pred)[0]
print("Preds: ", preds)

for x in range(seq_len):
    next_index = sample(preds[x], 0.8)
    print(indices_chord[next_index])



#       VISUALISE TRAINING HISTORY

''' 
history_dict = history.history
history_dict.keys()
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.figure()
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

plt.clf() 
acc_values = history_dict['acc'] 
val_acc_values = history_dict['val_acc']
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
'''
