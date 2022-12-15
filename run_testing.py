"""
This file along with the seqGen folder (https://github.com/perrineruth/diffsw) should be put in the core directory of SMURF (https://github.com/spetti/SMURF).


---


In case you get the following error

module 'jax.experimental' has no attribute 'optimizers'"

, in laxy.py replace 'from jax.experimental.optimizers import adam' with 'from jax.experimental.minmax import adam', or with 'from jax.example_libraries.optimizers import adam' (one of these should work).
"""


import numpy as np
import pickle




# GLOBAL VARIABLES
data_alphabet = 'ARNDCQEGHILKMFPSTWYV' # String containing all symbols in the alphabet for the original data.
sim_alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' # String containing all symbols in the alphabet for the simulated data.




# Converts a list of numbers that encode a string into its original alphabet. The alphabet letters are coded as 0, 1, 2, ..., while all other letters are coded as negative values or larger integers.
def convert_to_alphabet (list_of_nums, alphabet):
    
    # Create the lookup table.
    lookup_table = {i : alphabet[i] for i in range(len(alphabet))}
    for n in list_of_nums:
        if not n in lookup_table.keys():
            lookup_table[n] = '.'
    
    # Return the converted string.
    return ''.join([lookup_table[n] for n in list_of_nums])




# SMURF function from the eamples/LAM_AF_examples/make_pairwise_aln_figures.ipynb notebook. Returns the aligned seq_num's sequence. Malfunctions on some shorter sequences.
def get_aln_seq(aln, seqs, seq_num):
    aln = np.where(aln[seq_num,...]>.5, 1, 0)
    pos = np.argmax(aln, axis = 0) + (np.sum(aln, axis = 0)-1)
    seq = []
    for i in pos:
        if i>0: seq.append(seqs[seq_num][i])
        else: seq.append(-1)
    return seq




# Takes in the pickle dictionary saved after model training (for the specified protein or simulated data) and returns the dictionary of sequence alignments. Only alignments for sequences for which get_aln_seq() does not produce an error are returned.
# ---
# pickle_dict: dictionary saved after model training.
# alphabet: if None, then returns the sequences encoded with numbers. Otherwise, returns the strings with the original alphabet (without spaces).
def get_aln_seqs(pickle_dict, alphabet=None):
    
    # Initialize the dictionary.
    return_dict = {}
    
    # Retrieve the alignment and MSA objects.
    aln = pickle_dict['alignments']
    seqs = pickle_dict['input_MSA']
    
    # Fill in the dictionary.
    for i in range(len(aln)):
        try:
            model_aln = get_aln_seq(aln, seqs, i)
            return_dict[i] = model_aln
        except:
            pass
        
    # Convert the entries into alphabet letters if required.
    if not alphabet is None:
        for i in return_dict.keys():
            return_dict[i] = convert_to_alphabet(return_dict[i], alphabet)
        
    # Return the result.    
    return return_dict



    
if __name__ == "__main__":
    
    # IMPLEMENTATION PENDING

    
    pass