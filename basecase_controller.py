import numpy as np
import sys
import time
import sounddevice as sd
import serial
import librosa




#                   ----------- GOBAL VARIABLES ---------------


SAMPLE_RATE = 48000
BPM = 80 # replaced later with arduino input, but default is 80
BEATS = 4 # beats pr measure/loop
CHANNELS = 2
LLEN = int((60/BPM)*BEATS*SAMPLE_RATE)

in_index = 0
out_index = 0

cycle = np.zeros((int(LLEN), 2),dtype='float32')
new_layer = np.zeros((int(LLEN), 2),dtype='float32')
chord_layer = np.zeros((int(LLEN), 2),dtype='float32')


sd.default.samplerate = SAMPLE_RATE
sd.default.channels = (1,CHANNELS)
sd.default.latency = 'low'

note_names =  'C C# D D# E F F# G G# A A# B'.split()
noteBank = dict()



#     ------- STREAM CALLBACK FUNCTIONS -------




def callback_in(indata, frames, time, status):
    global in_index

    if in_index + frames < LLEN: # Write to contiguous memory
        new_layer[in_index:in_index+frames] = indata
    else: # Wrap around
            
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


def findKey_arbitrary(indata):
    # find key of recent layer
    # find the notes for the appropriate chord in the note bank.

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
    seq = [keyId%12, (keyId+4)%12, (keyId+7)%12] # Major chord always
    construct_chord(seq)



def construct_chord(seq):
    global chord_layer
    layer = np.zeros_like(cycle)
    l,_ = layer.shape
    q_size = int(l/8)
    e_size = int(l/8)

    for keyId in seq: # MINOR THIRD CHORD (major = 0, 4, 3)
        #keyId += tone
        count = 0
        if keyId in noteBank: # ELSE PITCH A NOTE TO CREATE MISSING NOTE
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

                e[:,1] = r_pitched[:]
                e[:,0] = l_pitched[:]
                
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
            
    chord_layer = np.zeros_like(cycle)
    chord_layer[:] = layer.dot(0.6)


    
def bank(indata):
    # segment cycle into quarter-notes.
    #   llen / 4 sized segments.
    # try to find the pitch of each quarter note.
    # save the audio to bank - marked with its pitch 
    
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
                
                findKey_arbitrary(new_layer.copy())

                #findKey_from_notes(new_layer.copy())
                new_layer = np.zeros_like(cycle)
            else:
                recmode = True
                in_stream.start()
                print("started rec")

        elif ino == b'c': # CLEAR LOOP
            cycle = np.zeros((int(LLEN),2))
            new_layer = np.zeros_like(cycle)
            nv_list = []

