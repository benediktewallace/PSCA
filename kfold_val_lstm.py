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
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedKFold, KFold
from keras.utils import plot_model
import pandas as pd
import seaborn as sns; sns.set()


########################################
#       Benedikte Wallace 2018
#   
#   
#   build_and_compile_model(): Build and return keras model
#   
#   run_kfold(x,y,x_val,y_val,temp,mode,n_splits,n_repeats,batch):
#   run kfold xval, return confusion matrix and perdiction accurarcy.
#   (mode arg is used when looking at differernt subsets 
#    of the data, like minor songs only in order to save 
#   results with the appropriate file name)
#   
########################################





#chords = sorted(list(set(chord_list)))
chords = 'C Cm Caug Cdim Csus C# C#m C#aug C#dim C#sus D Dm Daug Ddim Dsus D# D#m D#aug D#dim D#sus E Em Eaug Edim Esus F Fm Faug Fdim Fsus F# F#m F#aug F#dim F#sus G Gm Gaug Gdim Gsus G# G#m G#aug G#dim G#sus A Am Aaug Adim Asus A# A#m A#aug A#dim A#sus B Bm Baug Bdim Bsus ENDTOKEN'.split()

#chords = 'C C# D D# E F F# G G# A A# B Cm C#m Dm D#m Em Fm F#m Gm G#m Am A#m Bm ENDTOKEN'.split()

chord_indices = dict((c, i) for i, c in enumerate(chords))
indices_chord = dict((i, c) for i, c in enumerate(chords))


seq_len = 8 # length of example
batch =  512 # how many examples to look at at once. Adjusting this effects the optimisers choice by giving more or less information at one time
step = 1 
layers = 128 # units in LSTM
dropout = 0.2



def build_and_compile_model():
    print('Build model...')

    model = Sequential()

    model.add(TimeDistributed(Dense(12),input_shape=(seq_len, 12)))
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
    return model


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)




def run_kfold(x,y,x_val,y_val,temp=1.0,mode='mix',n_splits=10,n_repeats=1, batch=512):
    rkf = KFold(n_splits=n_splits)
    fold_pred_acc = []
    foldidx = 1
    confusion = np.zeros((len(chords), len(chords)))
    
    #dict containing chord and how many times chord was the target in our sample
    chord_counts = dict((chrd,0) for chrd in chords)
    #array containing how many times chord was predicted  
    pred_counts = np.zeros(len(chords))

    save_path = 'logs/mixed/weights.hdf5'

    if mode == 'min':
        save_path = 'logs/minor/weights.hdf5'
    elif mode == 'maj':
        save_path = 'logs/major/weights.hdf5'

    checkpointer = ModelCheckpoint(monitor='val_acc',filepath=save_path, verbose=1, save_best_only=True)



    for train, test in rkf.split(x, y=y):
        print("             Fold ",foldidx," of ",n_splits)
        print(len(train), " ", len(test))
        print(train, " ", test)
        
        foldidx += 1
       
        print(chord_counts)
       
        # Recompile model at each fold!
        model = build_and_compile_model()

        history = model.fit(x[train], y[train],
                  batch_size=batch,
                  epochs=100,
                  callbacks=[checkpointer, earlyStopping],
                  validation_data=(x_val, y_val)
                  )


        score = model.evaluate(x[test], y[test])
        print("SCORE: ", score)
        
        fold_pred_acc.append(score[1]*100)

        targets = y[test]
        targets_shape = targets.shape
        print("Shape of targets: ", targets_shape)

        preds = model.predict(x[test])
        preds_shape = preds.shape
        print("Shape of preds: ", preds_shape)

        for i in range(targets_shape[0]):
            for j in range(seq_len):
                true_val = np.argmax(targets[i, j])
                true_val_name = indices_chord[true_val]
                # add one to the chord count for true chord
                chord_counts[true_val_name] +=1
                #print("True val name: ",true_val_name)
                pred_idx = sample(preds[i, j], temperature=temp)#np.argmax(preds[i, j])
                pred_name = indices_chord[pred_idx]
                pred_counts[pred_idx] +=1
                #print("Pred name: ",pred_name)
                confusion[true_val][pred_idx] += 1


    for i in range(len(chords)):
        cur_chord = indices_chord[i]
        if chord_counts[cur_chord] != 0:
            for j in range(len(chords)):         
                confusion[i][j] = confusion[i][j] / chord_counts[cur_chord]
        else:
            print("No examples of chord: ", cur_chord)



    return fold_pred_acc, confusion, pred_counts


# CALLBACKS

#print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
earlyStopping = EarlyStopping(patience=10, monitor='val_acc')
#tensor_board = TensorBoard(histogram_freq=1,embeddings_freq=1, write_grads=True, write_images=True)
checkpointer = ModelCheckpoint(monitor='val_acc',filepath='logs/weights.hdf5', verbose=1, save_best_only=True)



#       MIXED DATASET KFOLD TEST (MAJOR & MINOR)

f = open('dataset_60mix.txt', 'r')
chord_list = f.read().split()
print("Length of chord_list: ", len(chord_list))
note_zip = np.load('60mix_nvs.npy.npz')
note_vecs = note_zip['arr_0']
print("Length of note_vecs: ", len(note_vecs))


# cut the training data in semi-redundant sequences of seq_len lenght
examples = []
nv_ex = []

for i in range(0, len(chord_list) - seq_len, step):
    examples.append(chord_list[i: i + seq_len])
    for x in range(i, i+seq_len):
        nv_ex.append(note_vecs[x])

# create x, y:

x = np.zeros((len(examples), seq_len, 12))
y = np.zeros((len(examples), seq_len, len(chords)))

nv_index = 0


for i , ex in enumerate(examples):
    for t, chord in enumerate(ex):
        x[i, t] = nv_ex[nv_index]
        y[i, t, int(chord)] = 1
        nv_index += 1

print(y.shape)
print(x.shape)


x_subarray_list = np.array_split(x, 2)
y_subarray_list = np.array_split(y, 2)
        
x_val = x_subarray_list[0]
print("X val length: ",len(x_val))
x_train = x_subarray_list[1]
print("X for kfold length: ",len(x_train))
y_val = y_subarray_list[0]
y_train = y_subarray_list[1]
        


predacc, confusion_mix, pc  = run_kfold(x_train,y_train, x_val,y_val,mode='mix')
pred_acc_mix = np.array(predacc, dtype='float64')

np.savez('60mix_pred_count.npy', pc)

mix_avgr_acc = np.sum(pred_acc_mix) / len(pred_acc_mix)
print("Mix average accuracy: ", mix_avgr_acc)



