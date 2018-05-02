import numpy as np 



# Possible states:
states = 'C Cm Caug Cdim Csus C# C#m C#aug C#dim C#sus D Dm Daug Ddim Dsus D# D#m D#aug D#dim D#sus E Em Eaug Edim Esus F Fm Faug Fdim Fsus F# F#m F#aug F#dim F#sus G Gm Gaug Gdim Gsus G# G#m G#aug G#dim G#sus A Am Aaug Adim Asus A# A#m A#aug A#dim A#sus B Bm Baug Bdim Bsus ENDTOKEN'.split()

def dptable(V):
# Print a table of steps from dictionary
    yield " ".join(("%12d" % i) for i in range(len(V)))
    for state in V[0]:
    	yield "%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)

def emission_dist_prob(emission_distribution, observation):
    # Probability measure of observation dist in emission_distritbution
    #observation = observation*0.05
    return np.dot(emission_distribution, observation)

def viterbi(obs, start_p, trans_p, emit_p):
    V = [{}] # Probability Table Prepared
    
    # Generate Probs for 0th hidden state.
    for st in states:

    	st_idx = states.index(st)

    	V[0][st] = {"prob": start_p[st_idx] * emission_dist_prob(emit_p[st_idx],obs[0]), "prev": None}
    # Generate Probs for 1st-nth hidden states 
    for t in range(1, len(obs)):
        V.append({})
        #print("ENTERED MAIN VITERBI")
        for st in states:

        	st_idx = states.index(st)

        	max_tr_prob = max(V[t-1][prev_st]["prob"] * trans_p[states.index(prev_st)][st_idx] for prev_st in states)

        	for prev_st in states:
        		prev_st_idx = states.index(prev_st)

        		if V[t-1][prev_st]["prob"] * trans_p[prev_st_idx][st_idx] == max_tr_prob:

        			max_prob = max_tr_prob * emission_dist_prob(emit_p[st_idx],obs[t]) # mult by emission prob.
        			V[t][st] = {"prob": max_prob, "prev": prev_st}
        			break
    
    # Prepare to identify best hidden state sequence from V
    output = []
    # The highest probability
    max_prob = max(value["prob"] for value in V[-1].values())
    previous = None
    
    # Get most probable final state from the end of V
    for st, data in V[-1].items():
        if data["prob"] == max_prob:
            output.append(st)
            previous = st
            break
            
    # Get the most probable state in each previous step of the list.
    for t in range(len(V) - 2, -1, -1):
        output.insert(0, V[t + 1][previous]["prev"]) # put the 'prev' state at index 0 in output
        previous = V[t + 1][previous]["prev"] # store previous 'prev' state.
    
    # Return the list of states
    return(output, max_prob, V)


