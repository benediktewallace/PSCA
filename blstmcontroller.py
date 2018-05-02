import numpy as np
import sys
import time
import sounddevice as sd
import serial
import librosa

from keras.models import load_model



#                   ----------- GOBAL VARIABLES ---------------


SAMPLE_RATE = 48000
BPM = 80 # replaced later with arduino input, but default is 80
BEATS = 4 # beats pr measure/loop
CHANNELS = 2
LLEN = int((60/BPM)*BEATS*SAMPLE_RATE) # Use % blocksize ??

in_index = 0
out_index = 0

cycle = np.zeros((int(LLEN), 2),dtype='float32') # remove int - unnescasary?
new_layer = np.zeros((int(LLEN), 2),dtype='float32')
chord_layer = np.zeros((int(LLEN), 2),dtype='float32')


sd.default.samplerate = SAMPLE_RATE
sd.default.channels = (1,CHANNELS)
sd.default.latency = 'low'

note_names =  'C C# D D# E F F# G G# A A# B'.split()
noteBank = dict()

song_key = 0

#     ------- VARIABLES USED FOR BLSTM ----------

chords = 'C Cm Caug Cdim Csus C# C#m C#aug C#dim C#sus D Dm Daug Ddim Dsus D# D#m D#aug D#dim D#sus E Em Eaug Edim Esus F Fm Faug Fdim Fsus F# F#m F#aug F#dim F#sus G Gm Gaug Gdim Gsus G# G#m G#aug G#dim G#sus A Am Aaug Adim Asus A# A#m A#aug A#dim A#sus B Bm Baug Bdim Bsus ENDTOKEN'.split()
indices_chord = dict((i, c) for i, c in enumerate(chords))

model = load_model('looperModel2.hdf5')



#     ------- STREAM CALLBACK FUNCTIONS -------




def callback_in(indata, frames, time, status):
    global in_index

    if in_index + frames < LLEN: # Write to contiguous memory
        new_layer[in_index:in_index+frames] = indata
    else: # Wrap around
            #tilend = (in_index+frames)-LLEN
        tilend = LLEN - in_index
        fromstart = frames - tilend
        new_layer[in_index:LLEN] = indata[:tilend]
        new_layer[:fromstart] = indata[tilend:frames]

    in_index += frames
    in_index = in_index%LLEN


def callback_out(outdata, frames, time, status):
    global out_index
        
    if out_index + frames <= LLEN:
        outdata[:] = np.add(cycle[out_index:out_index+frames], chord_layer[out_index:out_index+frames])
    else: # The buffer wraps around
        fte = LLEN - out_index
        outdata[:fte] = np.add(cycle[out_index:LLEN], chord_layer[out_index:LLEN]) # Write til end of buffer
        assert frames-fte > 0
        outdata[fte:] = np.add(cycle[:frames-fte], chord_layer[:frames-fte]) # Write from beginning of buffer
    out_index += frames
    out_index = out_index%LLEN
       




#     ------- FUNCTIONS FOR CHORD CONSTRUCTION & PITCH ANALYSIS ------

def transpose(source,dest):
    return dest - source
    
def get_intervals(chord_name):
    global song_key
    # Transpose from predictied chord in C to song key by changing root of the chord, keyid
    keyId = 0
    seq = []


    if len(chord_name) == 1: # major chord
        keyId = note_names.index(chord_name) + transpose(0,song_key)
        print("transposed to ", note_names[keyId%12])
        seq = [keyId%12, (keyId+4)%12, (keyId+7)%12]
        return seq
    elif len(chord_name) == 2 and chord_name[1] == '#': # major chord
        keyId = note_names.index(chord_name) + transpose(0,song_key)
        print("transposed to ", note_names[keyId%12])
        seq = [keyId%12, (keyId+4)%12, (keyId+7)%12]
        return seq
    else:
        if chord_name[1] == '#':
            keyId = note_names.index(chord_name[:2]) + transpose(0,song_key)
        else:
            keyId = note_names.index(chord_name[0]) + transpose(0,song_key)

        # find type: min, sus, aug, dim
        if 'sus' in chord_name: # SUS4
            print("transposed to root", note_names[keyId%12])
            seq = [keyId%12, (keyId+5)%12, (keyId+6)%12]
            return seq
        elif 'aug' in chord_name:
            print("transposed to root", note_names[keyId%12])
            seq = [keyId%12, (keyId+4)%12, (keyId+8)%12]
            return seq
        elif 'dim' in chord_name:
            print("transposed to root", note_names[keyId%12])
            seq = [keyId%12, (keyId+3)%12, (keyId+6)%12]
            return seq
        else: # minor
            print("transposed to root", note_names[keyId%12])
            seq = [keyId%12, (keyId+3)%12, (keyId+7)%12]
            return seq


def to_note_vector(indata):
    global song_key
    # TO DO: 
    # overtones? are added as well, set some higher threshhold for mag or limit freq range to one reasonable for voice?
    # TRANSPOSE TO C

    pitches, magnetude = librosa.piptrack(y=indata[:,1],fmin=250.0,fmax=1050.0, sr=SAMPLE_RATE)
    nv = np.zeros(12)
    step = 1
    _, ts_max = magnetude.shape
    #print("MAX TIME SLICES ", ts_max)

    for ts in range(0, ts_max, step):
        index = magnetude[:, ts].argmax()
        pitch = pitches[index, ts]
        if pitch != 0.0:
            note = librosa.hz_to_note(pitch, octave=False)
            note_idx = note_names.index(note[0])
            nv[note_idx] += 1.

    for x in range(len(nv)):
        if nv[x] != 0:
            nv[x] = nv[x] / ts_max
    #transpose to C 
    nv = np.roll(nv, transpose(song_key, note_names.index('C')))
    print("Note vector: ", nv)
    return nv



'''def hmm_chord(indata):
    # save notevectors so that we accum observations at each call 
    #nv = []
    nv = to_note_vector(indata)
    nv_list.append(nv) 
    prediction, _, _ = viterbi(nv_list, start_probability, transition_probability, emission_probability)
    print("HMM predicted ", prediction)
    print("generating chord ",prediction[-1])
    seq = get_intervals(prediction[-1])
    construct_chord(seq) #adds the predicted chord to the playback loop'''

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def dl_chord(indata, nv_list):
    nv  = []
    nv = to_note_vector(indata)

    x = np.zeros((1,8,12))

    if nv_list == []:
        nv_list.append(nv)
        nv_list.append(nv)
        nv_list.append(nv)
        nv_list.append(nv)
        nv_list.append(nv)
        nv_list.append(nv)
        nv_list.append(nv)
        nv_list.append(nv)
    else:
        nv_list = nv_list[1:]
        nv_list.append(nv)

    # format nv into (1 example, 8 bars ,12 notes)
    # move one "step" ahead in the nv_list at evey call.
    for i in range(8):
        x[0, i] = nv_list[i]
    
    preds = model.predict(x,verbose=0)[0]
    prediction = sample(preds[-1], temperature=0.35)

    if indices_chord[prediction] != 'ENDTOKEN':
        seq = get_intervals(indices_chord[prediction])
        print("BLSTM predicted ", indices_chord[prediction])
        construct_chord(seq) #adds the predicted chord to the playback loop

    return nv_list



def find_songKey(indata):
    # find key of recent layer
    global song_key
    
    pitches, magnetude = librosa.piptrack(y=indata[:,1], sr=SAMPLE_RATE, fmin=250.0,fmax=1050.0)
    time_slice = 1
    index = magnetude[:, time_slice].argmax()
    pitch = pitches[index, time_slice]
    while pitch == 0.0:
        time_slice = (time_slice + 10)%len(pitches)
        index = magnetude[:, time_slice].argmax()
        pitch = pitches[index, time_slice]

    key = librosa.hz_to_note(pitch, octave=False)
    print("findKey: ",key)
    keyId = note_names.index(key[0])
    song_key = keyId
    

def construct_chord(seq):
    global chord_layer
    layer = np.zeros_like(cycle)
    l,_ = layer.shape
    q_size = int(l/8)
    e_size = int(l/8)

    for keyId in seq: # MINOR THIRD CHORD (major = 0, 4, 3)
        #keyId += tone
        count = 0
        if keyId in noteBank: # ELSE PITCH A NOTE TO CREATE MISSING NOTE?
            print("Adding note ", note_names[keyId])
            q = noteBank.get(keyId)
            e = np.zeros((e_size,2))
            e[:] = q #q[e_size:e_size*2]
            
            ms = int((SAMPLE_RATE/10))
            ramp = np.linspace(0.0,1.0,num=ms)
            e[:ms,1] = e[:ms,1]*ramp
            e[e_size-ms:,1] = e[e_size-ms:,1]*ramp
            e[:ms,0] = e[:ms,0]*ramp
            e[e_size-ms:,0] = e[e_size-ms:,0]*ramp
            
            for e_note in range(0,8):
                layer[count:e_size+count] = np.add(e,layer[count:e_size+count])
                count += e_size
            
        else: # key id is not in notebank
            if bool(noteBank):
                
                from_index = next(iter(noteBank))
               
                n_steps = keyId - from_index

                if n_steps < -5:
                    n_steps = n_steps+12
                elif n_steps > 5:
                    n_steps = n_steps-12
                
                n_bank = noteBank.get(from_index)
                
                print("FROM INDEX: ", from_index)

                print("Pitching from ", note_names[from_index], " to ", note_names[keyId], " n_steps=",n_steps)

                r_pitched = librosa.effects.pitch_shift(n_bank[:,1], SAMPLE_RATE, n_steps=n_steps)
                l_pitched = librosa.effects.pitch_shift(n_bank[:,0], SAMPLE_RATE, n_steps=n_steps)
                e = np.zeros((e_size,2))

                e[:,1] = r_pitched[:]#r_pitched[e_size:e_size*2]
                e[:,0] = l_pitched[:]#l_pitched[e_size:e_size*2]

                # try to make a sort of cross fade. samplerate/10 = 10ms
                ms = int((SAMPLE_RATE/10))
                ramp = np.linspace(0.0,1.0,num=ms)
                        
                e[:ms,1] = e[:ms,1]*ramp
                e[e_size-ms:,1] = e[e_size-ms:,1]*ramp

                e[:ms,0] = e[:ms,0]*ramp
                e[e_size-ms:,0] = e[e_size-ms:,0]*ramp

                for e_note in range(0,8):
                    layer[count:e_size+count] = np.add(e,layer[count:e_size+count])
                    count += e_size  
            '''if bool(noteBank):
                for n in note_names:
                    n_index = note_names.index(n)
                    if n_index in noteBank:
                        #pitch it
                        print(" adding pithched layer")
                        
                        n_steps = 0
                        if n_index < keyId:
                            n_steps = keyId - n_index
                        else:
                            n_steps = n_index + keyId
                        n_bank = noteBank.get(n_index)
                        r_pitched = librosa.effects.pitch_shift(n_bank[:,1], SAMPLE_RATE, n_steps=n_steps)
                        l_pitched = librosa.effects.pitch_shift(n_bank[:,0], SAMPLE_RATE, n_steps=n_steps)
                       
                        e = np.zeros((e_size,2))

                        e[:,1] = r_pitched[e_size:e_size*2]
                        e[:,0] = l_pitched[e_size:e_size*2]

                        # try to make a sort of cross fade. samplerate/10 = 10ms
                        ms = int((SAMPLE_RATE/10))
                        ramp = np.linspace(0.0,1.0,num=ms)
                        
                        e[:ms,1] = e[:ms,1]*ramp
                        e[e_size-ms:,1] = e[e_size-ms:,1]*ramp

                        e[:ms,0] = e[:ms,0]*ramp
                        e[e_size-ms:,0] = e[e_size-ms:,0]*ramp

                        for e_note in range(0,8):
                            layer[count:e_size+count] = np.add(e,layer[count:e_size+count])
                            count += e_size            
                        break;'''
    chord_layer = np.zeros_like(cycle)
    chord_layer[:] = layer.dot(0.6)


    
def bank(indata):
    # segment cycle into quarter-notes.
    #   llen / 4 sized segments.
    # try to find the pitch of each quarter note.
    # save the audio to bank - marked with its pitch 
    
    # TODO: 
    # Use split from librosa to get index of non silent areas in the indata?
    # find time  of highest magnitude within each q note slice of indata ? maybe rethink this and use e-note sized slices!

    l,_ = indata.shape
    #q_size = int(l/4)
    q_size = int(l/8)
    q = np.zeros((q_size, 2), dtype='float32') 
    count = 0

    for x in range(1,8):   
        q[:] = indata[count:count+q_size]
        pitches, magnetude = librosa.piptrack(y=q[:,1], sr=SAMPLE_RATE, fmin=250.0,fmax=1050.0)
        _,ts = magnetude.shape
        print(ts)
        print(pitches.shape)
        index = max(magnetude[:, i].argmax() for i in range(ts))
        prev = magnetude[:, 0].argmax()
        best_ts = 0
        for i in range(ts):
            if magnetude[:, i].argmax() > prev:
                best_ts = i

            prev = magnetude[:, i].argmax()
       


        pitch = pitches[index, best_ts]
        if pitch != 0.0: 
            note_info = librosa.hz_to_note(pitch, cents=True)
            
            # if your pretty close - >< 40cent its added to the notebank

            print(note_info)
            cents = int(note_info[0].replace('+','-').split('-',1)[-1])
            print("Cents: ", cents)
            if cents < 25:
                note = librosa.hz_to_note(pitch, octave=False)
                print("Bank: ",note)
                keyId = note_names.index(note[0])
                #if keyId in noteBank:
                #noteBank[keyId].append(q)
                #else:
                #noteBank[keyId] = list(q)
                noteBank[keyId] = q

        count += q_size
    


#
#                ----------- MAIN LOOP ---------------
#

arduino = serial.Serial('/dev/cu.usbmodem1411', 9600, timeout=.1) 

in_stream = sd.InputStream(callback=callback_in)
out_stream = sd.OutputStream(callback=callback_out)

playmode = False
recmode = False
print("Entering main loop...")
first_loop = True
nv_list = []

while True:
    ino = arduino.read()
    if ino:
        #print(ino)
        if ino == b'l':
            LLEN = (60/ino)*bars*sr
            cycle = np.zeros((int(LLEN),2))
            new_layer = np.zeros_like(cycle)
        if ino == b'p':
            if playmode:
                print("stopped play")
                playmode = False
                out_stream.stop()
            else:
                print("started play")
                playmode = True
                out_stream.start()
        elif ino == b'r':
            if recmode:
                recmode = False
                cycle = np.add(cycle,new_layer)
                in_stream.stop()
                print("stopped rec")
                bank(new_layer.copy())
                #hmm_chord(new_layer.copy())
                if first_loop:
                    first_loop = False
                    find_songKey(new_layer.copy())
                nv_list = dl_chord(new_layer.copy(), nv_list)

                #findKey_arbitrary(new_layer.copy())
                #findKey_from_notes(new_layer.copy())
                new_layer = np.zeros_like(cycle)
            else:
                recmode = True
                in_stream.start()
                print("started rec")

        elif ino == b'c': # CLEAR LOOP
            cycle = np.zeros((int(LLEN),2))
            new_layer = np.zeros_like(cycle)
            #chord_layer = np.zeros_like(cycle) # rename ml layer chord layer
            nv_list = []




'''

To do:
In construct_chord():
    find shortest pitch path, pithch down if that is closer to the correct note.


'''