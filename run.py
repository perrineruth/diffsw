"""
This file should be put in the core directory of SMURF. In the same directory this file must be downloaded: https://files.ipd.uw.edu/krypton/data_unalign.npz


---


In case you get the following error

module 'jax.experimental' has no attribute 'optimizers'"

, in laxy.py replace 'from jax.experimental.optimizers import adam' with 'from jax.experimental.minmax import adam', or with 'from jax.example_libraries.optimizers import adam' (one of these should work).
"""

import numpy as np
import time
import random
import pickle
import network_functions as nf # Module containing SMURF routines.




# GLOBAL VARIABLES
datafile = 'data_unalign.npz' # File containing the MSA data.
test_protein = '3A0YA' # Name of the protein to use in the tests.
N_samples = 2000 # Number of sequences to select for the procedure.
N_steps = 500 # Number of training steps.
N_proteins = 50 # Number of proteins to train the model for.
testfile = 'seqGen/out/words_generated_test.a3m'
trainfile = 'seqGen/out/words_generated_train.a3m'
pickle_file_data = 'train_data.pickle' # Pickle file for the outputs of training on the initial data.
pikle_file_sim = 'train_sim.pickle' # Pickle file for the outputs of training on the simulated data.




# Function from the 'examples/LAM_AF_examples/af_msa_backprop.ipynb' that cannot be directly imported from the notebook.
# Returns the alignments for the specified a3m file.
def get_feat(filename, alphabet="ARNDCQEGHILKMFPSTWYV"):
  '''
  Given A3M file (from hhblits)
  return MSA (aligned), MS (unaligned) and ALN (alignment)
  '''
  def parse_fasta(filename):
    '''function to parse fasta file'''    
    header, sequence = [],[]
    lines = open(filename, "r")
    for line in lines:
      line = line.rstrip()
      if len(line) == 0: pass
      else:
        if line[0] == ">":
          header.append(line[1:])
          sequence.append([])
        else:
          sequence[-1].append(line)
    lines.close()
    sequence = [''.join(seq) for seq in sequence]
    return header, sequence

  names, seqs = parse_fasta(filename)  
  a2n = {a:n for n,a in enumerate(alphabet)}
  def get_seqref(x):
    n,seq,ref,aligned_seq = 0,[],[],[]
    for aa in list(x):
      if aa != "-":
        seq.append(a2n.get(aa.upper(),-1))
        if aa.islower(): ref.append(-1); n -= 1
        else: ref.append(n); aligned_seq.append(seq[-1])
      else: aligned_seq.append(-1)
      n += 1
    return seq, ref, aligned_seq
  
  # get the multiple sequence alignment
  max_len = 0
  ms, aln, msa = [],[],[]
  for seq in seqs:
    seq_,ref_,aligned_seq_ = get_seqref(seq)
    if len(seq_) > max_len: max_len = len(seq_)
    ms.append(seq_)
    msa.append(aligned_seq_)
    aln.append(ref_)
  
  return msa, ms, aln




# Returns the number of proteins contained in the specified data file.
def get_N_proteins(npz_file = datafile):
    
    data = np.load (npz_file, allow_pickle=True)
    return len(list(data))




# Reads the specified *.npz file and returns its MSAs as a list of 1D-arrays.
# ---
# protein: name or the ID of the protein. In the first case, the protein with the given name will be used. In the second case, the n-th protein in the file will be used.
# npz_file: path to the *.npz file containing the data.
# return_protein_name: if True, returns a tuple of the array and the protein name. Otherwise, returns only the array.
# ---
# EXAMPLES:
# get_protein_MSA('3A0YA', 'data_unalign.npz')
# get_protein_MSA(5, 'data_unalign.npz')
def get_protein_MSA(protein=test_protein, npz_file = datafile, return_protein_name = False):
    
    data = np.load (npz_file, allow_pickle=True)
    
    if type(protein) == str:
        return (data[protein].item()['ms'], protein) if return_protein_name else data[protein].item()['ms']
    
    elif int(protein) == protein and protein >= 0 and protein < get_N_proteins(npz_file):
        protein_list = list(data)
        return (data[protein_list[protein]].item()['ms'], protein_list[protein]) if return_protein_name else data[protein_list[protein]]['ms']
    
    elif protein >= get_N_proteins(npz_file):
        raise ValueError ("In get_protein_MSA(), invalid protein index of " + str(protein) + " was submitted. File " + npz_file + " contains only " + str(get_N_proteins(npz_file)) + " proteins.")
        
    else:
        raise ValueError ("In get_protein_MSA(), the 'protein' argument must either be a string or a non-negative integer. Value " + str(protein) + " of type " + str(type(protein)) + " was submitted instead.")




# Trains the model on the specified dataset for the specified set of sequences. Returns the alignment and contact scores. The output of get_protein_MSA with return_protein_name=False can be used as the function argument.
# ---
# seq_array: a list of 1D arrays of sequences, with the alphabet elements encoded as non-negative integers.
# verbose: if True, then prints the diagnostic output during training.
def train_model(seq_array, verbose=False):
    
    # Take a subset of data.
    x = nf.sub_sample(seq_array, samples=min(N_samples,len(seq_array)))
            
    # Create the BasicAlign model.
    ms = nf.one_hot(nf.pad_max(x)) # pad_max converts a list of 1D-arrays into a 2D array, adding -1 to empty entries. For example, [[1,2,3],[4]] -> [[1,2,3],[4,-1,-1]]. One-hot then converts each 1D-array into an identity matrix with padding determined by the entries. Overall, this command just converts the data into the format the machine learning model expects.
    lens = np.array([len(y) for y in x]) # Array of lengths of sequences.
    gap = -3 # Alighnment gap penalty.
    model_basicAlign = nf.BasicAlign(X=ms, lengths=lens, sw_gap = gap) # This model will be trained.
    
    # Train the model and obtain the MSA parameters.
    model_basicAlign.fit(N_steps, verbose=verbose)
    msa_params = model_basicAlign.opt.get_params()
    
    # Create the TrainMRF model.
    x = nf.sub_sample(seq_array, samples=N_samples)
    lens = np.array([len(y) for y in x])
    ms = nf.one_hot(nf.pad_max(x))
    model_trainMRF = nf.MRF(X=ms, lengths=lens, sw_gap = gap)
    
    # Update it with the BasicAlign parameters.
    mrf_params = model_trainMRF.opt.get_params()
    for p in ["emb","gap","open"]:
        mrf_params[p] = msa_params[p]
    model_trainMRF.opt.set_params(mrf_params)
    
    # Train the model and obtain the MSA parameters.
    model_trainMRF.fit(N_steps, verbose=verbose)
    mrf_params = model_trainMRF.opt.get_params()
    
    # Obtain the alignments and the protein contacts.
    aln = np.copy(model_trainMRF.get_aln(np.array(range(len(x))))[0])
    contacts = model_trainMRF.get_contacts()
    
    return aln, contacts





# Reads the original *.npz file, randomly selects N_proteins proteins from it and trains the model on those. Saves the resulting alignments and contact scores into a pickle file.
def run_data():
    
    # Custom Print function to keep track of the runtime.
    time_begin = time.time()
    def Print(s):
      print (str(int(time.time()-time_begin))+' sec:        ' + str(s))   
    
    # Retrieve the list of proteins and build a dictionary of the MSAs.
    Print ("Retrieving data for " + str(N_proteins) + " randomly selected proteins. Working on protein...")
    protein_indices = random.sample(list(np.arange(0, get_N_proteins(), 1)), N_proteins)
    dict_MSA = {}
    for j, i in enumerate(protein_indices):
        Print (str(j) + ' / ' + str(len(protein_indices)))
        MSA, name = get_protein_MSA(i, datafile, True)
        dict_MSA[name] = {}
        dict_MSA[name]['input_MSA'] = MSA
    Print ("Dictionary building complete.")
        
    # Fill the dictionary with the training data.
    for i, (protein, seq_array_dict) in enumerate(dict_MSA.items()):
        seq_array = seq_array_dict['input_MSA']
        Print ("Training the model on protein " + str(i+1) + " out of " + str(len(protein_indices)) + ": " + protein + "...")
        dict_MSA[protein]['alignments'], dict_MSA[protein]['contacts'] = train_model(seq_array)
    Print ("Training complete.")
    
    # Save the results in a pickle file.
    Print ("Saving the results to " + pickle_file_data + "...")
    with open(pickle_file_data, 'wb') as file:
        pickle.dump(dict_MSA, file)
    Print ("Complete.")
        




# IMPLEMENTATION PENDING
def run_simulation():
    pass
    
    
    
    
# Generates a simulated dataset and trains the model on it.
def train_model_simulated_data():
    
    # Custom Print function to keep track of the runtime.
    time_begin = time.time()
    def Print(s):
      print (str(int(time.time()-time_begin))+' sec:        ' + str(s))   
      
    # Load the training and the testing data.
    alphabet = ''.join(['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'])
    ms_testing, ms_training = get_feat(testfile, alphabet)[1], get_feat(trainfile, alphabet)[1]
    ms_te, ms_tr = nf.one_hot(nf.pad_max(ms_testing)), nf.one_hot(nf.pad_max(ms_training))
    lens_te, lens_tr = np.array([len(y) for y in ms_te]), np.array([len(y) for y in ms_tr])
    gap = -3
    
    # Create the BasicAlign model.
    model_basicAlign = nf.BasicAlign(X=ms_tr, lengths=lens_tr, sw_gap=gap)
    
    # Train the model and obtain the MSA parameters.
    Print ("")
    Print ("Training the BasicAlign model...")
    model_basicAlign.fit(N_steps, verbose=True)
    msa_params = model_basicAlign.opt.get_params()
    
    # Create the TrainMRF model.
    model_trainMRF = nf.MRF(X=ms_tr, lengths=lens_tr, sw_gap=gap)
    
    # Update it with the BasicAlign parameters.
    mrf_params = model_trainMRF.opt.get_params()
    for p in ["emb","gap","open"]:
        mrf_params[p] = msa_params[p]
    model_trainMRF.opt.set_params(mrf_params)
    
    # Train the model and obtain the MSA parameters.
    Print ("Training the MRF model...")
    model_trainMRF.fit(N_steps, verbose=True)
    mrf_params = model_trainMRF.opt.get_params()
    
    
    
    
if __name__ == "__main__":
    run_data()