from music21 import *
import os
import numpy as np
from sklearn.preprocessing import normalize as npnorm
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

#########################################################
#	Benedikte Wallace 2018								
# 														
#	Process music xml files:							
#																		
#   Split each score (file) 							
#   into measurers and create a list of notevector 		
#   and chord name pairs which are saved and used as
#   training data.										
#														
#########################################################



FOLDER = 'foldername'
cwd = os.path.dirname(os.path.realpath(__file__))
filename_list = os.listdir(cwd+FOLDER)

measures_chords = []
measures_notes = []

mode = 'major'

FLOOR = 1e-7

note_names =  'C C# D D# E F F# G G# A A# B'.split()

#chord_names = 'C Cm Caug Cdim Csus C# C#m C#aug C#dim C#sus D Dm Daug Ddim Dsus D# D#m D#aug D#dim D#sus E Em Eaug Edim Esus F Fm Faug Fdim Fsus F# F#m F#aug F#dim F#sus G Gm Gaug Gdim Gsus G# G#m G#aug G#dim G#sus A Am Aaug Adim Asus A# A#m A#aug A#dim A#sus B Bm Baug Bdim Bsus ENDTOKEN'.split()
chord_names = 'C C# D D# E F F# G G# A A# B Cm C#m Dm D#m Em Fm F#m Gm G#m Am A#m Bm ENDTOKEN'.split()


endtoken_idx = len(chord_names)-1

def chordname_to_index(name):
	return chord_names.index(name)

def index_to_chordname(idx):
	return chord_names[idx]

def chord_type(m21_data):
	symbol = m21_data.root().step
	common_name = m21_data.pitchedCommonName
	name = symbol#note_names.index(symbol)
	note = note_names.index(symbol)
	if common_name[1] == '-' and common_name[2] == '-':
		#FLAT NOTATION
		print("FLAT name and root: ", m21_data, " ", note_names[note-2])
		name = note_names[note - 2]
	elif common_name[1] == '-':
		#FLAT NOTATION
		print("FLAT name and root: ", m21_data, " ", note_names[note-1])
		name = note_names[note-1]
	elif common_name[1] == '#':
		#FLAT NOTATION
		print("FLAT name and root: ", m21_data, " ", note_names[(note+1)%len(note_names)])
		name = note_names[(note + 1)%len(note_names)]
	if 'minor' in common_name or m21_data.quality == "minor":
		name = name + 'm'

	print(m21_data)
	print(name)
	return chord_names.index(name)

def chord_type_60(m21_data):
	symbol = m21_data.root().step
	common_name = m21_data.pitchedCommonName


	note = note_names.index(symbol)
	c_type = -1 # negative value represents chord not matching one of the five types
	
	if 'major' in common_name or m21_data.quality == "major":
		c_type = 0
	elif 'minor'in common_name or m21_data.quality == "minor":
		c_type = 1
	elif 'augmented'in common_name or m21_data.quality == "augmented":
		c_type = 2
	elif 'diminished' in common_name or m21_data.quality == "diminished":
		c_type = 3
	else:
		_, quality = harmony.chordSymbolFigureFromChord(m21_data, True)
		if "suspended" in quality or "power" in quality or 'quartal trichord' in common_name:
			c_type = 4


	if common_name[1] == '-' and common_name[2] == '-':
		#FLAT NOTATION
		print("FLAT name and root: ", m21_data, " ", note_names[note-2])
		note = note - 2
	elif common_name[1] == '-':
		#FLAT NOTATION
		print("FLAT name and root: ", m21_data, " ", note_names[note-1])
		note = note - 1
	elif common_name[1] == '#':
		#FLAT NOTATION
		print("FLAT name and root: ", m21_data, " ", note_names[(note+1)%len(note_names)])
		note = (note + 1)%len(note_names)

	
	
	if c_type != -1:
		#print("returned ",index_to_chordname((note * 5)+c_type))
		return (note * 5) + c_type
	else:
		print("********** DEBUG ME!!! Chord ", m21_data, " was not one of the 5 types. root is ", index_to_chordname(note*5)," cm: ",common_name, "\n##########DEBUG ME!!!########\n")
		return note*5





def create_alt_nv():
	nv_list = []
	for x in range(len(measures_notes)):
	# find chord for this measure
		chord = measures_chords[x]
		notes = measures_notes[x]
		notes_in_measure = np.zeros(12)


		if chord != endtoken_idx:
			for n in notes:
				n_name = n.name
				cur_name = n.name
				cur_dur = float(n.duration.quarterLength)
				n_idx = index_from_note(n_name)
				#print("Note: ", n_name, " duration: ",cur_dur, " Added as ",note_names[n_idx])
				notes_in_measure[n_idx] += cur_dur

		nv_list.append(notes_in_measure)


	return nv_list




def index_from_note(m21_data):
	root = m21_data[0]
	if len(m21_data) > 2:
		if m21_data[1] == '-' and m21_data[2] == '-':
			#print(note_names.index(root) - 2)
			return note_names.index(root) - 2
		if m21_data[1] == '#' and m21_data[2] == '#':
			#print(note_names.index(root) +2)
			return note_names.index(root) + 2 
	elif len(m21_data) > 1:
		if m21_data[1] == '-':
			#print(note_names.index(root) - 1)
			return note_names.index(root) - 1
	else:
		return note_names.index(m21_data) 




for fn in filename_list:
	
	# Parse input and transpose to key C:

	print("Filename:", fn)
	if fn[0] != '.':
		
		fn = cwd+FOLDER+'/'+fn

		# transpose to C

		s = converter.parse(fn, format='musicxml')
		k = s.analyze('key')
		print("KEY:",k)

		test_flat = s.flat
		keySigs = test_flat.getElementsByClass('KeySignature')
		assert len(keySigs) <2
		#for ks in keySigs:
		#	print("Number of keysignatures in this song",ks.measureNumber)

		i = 0 # interval
		if "minor" in str(k):
			i = interval.Interval(k.tonic, pitch.Pitch('a'))
			mode = "minor"
		else:
			i = interval.Interval(k.tonic, pitch.Pitch('C'))
			mode = "major"

		sNew = s.transpose(i)

		k = sNew.analyze('key')
		print("KEY:",k)

		# SPLIT EACH SCORE IN TO MEASURES

		sOnePart = sNew.flattenParts()
		sMeasures = sOnePart.getElementsByClass('Measure')
		print(len(sMeasures)) # -> number of measures in a score

		if len(sMeasures) > 1: # This test may not be nescescary !!!!

			if mode == "minor":
				measures_notes.append("minor")
			else:
				measures_notes.append("X")
			measures_chords.append(endtoken_idx)
			

			first_measure = sMeasures[0].getElementsByClass('Chord')

			assert len(first_measure) > 0

			for m in sMeasures:
				m_chords = m.getElementsByClass('Chord')
				m_notes = m.getElementsByClass('Note')

				
				#print("lenght of notes in measure: ", len(m_notes))
				# Measures with no notes are not added to the dataset

				if len(m_notes) > 0:

					if len(m_chords) == 1:
						
						#assert harmony.chordSymbolFigureFromChord(m_chords[0]) != 'Chord Symbol Cannot Be Identified'
						if harmony.chordSymbolFigureFromChord(m_chords[0]) != 'Chord Symbol Cannot Be Identified':

							measures_notes.append(m_notes)
							print("Chord: ", index_to_chordname(chord_type(m_chords[0])), "Data: ", m_chords[0])
							
							measures_chords.append(chord_type(m_chords[0]))


					elif len(m_chords) > 1:
						assert len(m_chords) == 2
						print("More than one chord in this measure - Only keeping FIRST chord ", index_to_chordname(chord_type(m_chords[0])))
						# only keep last chord
						measures_notes.append(m_notes)
						measures_chords.append(chord_type(m_chords[0]))
						

					else: 
						print("Zero chords in this measure - USING CHORD FROM PREV BAR")
						#assert len(measures_chords) > 0
						measures_chords.append(measures_chords[-1])
						measures_notes.append(m_notes)
						
						#print("Zero chords in this measure - not added")


				else:
					# No notes in this measure
					print("Number of notes in measure is ",len(m_notes), ", measure removed from dataset.") 




print("Number of chords in the dataset: ",len(measures_notes), ", number of note vectors (should be equal): ", len(measures_chords))

#print(measures_chords)

# print histogram example:

#notes = measures_notes[0]
#p = graph.plot.HistogramPitchClass(notes)
#p.run()


# SAVE  LIST OF CHORDS (songs seperated by ENDTOKEN) AND ARRAY OF CORRESPONDING NOTE VECTORS
nv_list = create_alt_nv()

np.savez('nvs.npy', nv_list)

num_chord_ex = 0

with open(FOLDER+".txt", "w") as text_file:    
    for index, item in enumerate(measures_chords):
    	if item == 60 and measures_chords[(index+1)%len(measures_chords)] == 60:
    		continue
    	else:
    		text_file.write("%s " % index_to_chordname(item))
    		num_chord_ex += 1


print("Number of chords in the dataset: ",num_chord_ex, ", number of note vectors (should be equal): ", len(nv_list))



# SAVE MAJOR AND MINOR SONGS IN SEPERATE FOLDERS.
'''
nv_list = create_alt_nv()

maj_nvs = []
min_nvs = []

is_minor = False

with open(FOLDER[1:] + "_24major.txt", "w") as major_file:
	with open(FOLDER[1:] + "_24minor.txt", "w") as minor_file:
		for idx, item in enumerate(measures_chords):
			if item == endtoken_idx:
				if measures_notes[idx] == 'minor':
					is_minor = True
				else:
					is_minor = False

			if is_minor:
				minor_file.write("%s " % index_to_chordname(item))
				min_nvs.append(nv_list[idx])

			else:
				major_file.write("%s " % index_to_chordname(item))
				maj_nvs.append(nv_list[idx])

np.savez('24maj_nvs.npy', maj_nvs)
np.savez('24min_nvs.npy', min_nvs)
'''


