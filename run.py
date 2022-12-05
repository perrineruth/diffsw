"""
This file should be put in the core directory of SMURF. In the same directory this file must be downloaded: https://files.ipd.uw.edu/krypton/data_unalign.npz


---


In case you get the following error

module 'jax.experimental' has no attribute 'optimizers'"

, in laxy.py replace 'from jax.experimental.optimizers import adam' with 'from jax.experimental.minmax import adam', or with 'from jax.example_libraries.optimizers import adam' (one of these should work).
"""

import numpy as np
import time
import network_functions as nf # Module containing SMURF routines.




# GLOBAL VARIABLES
datafile = 'data_unalign.npz' # File containing the MSA data.
test_protein = '3A0YA' # Name of the protein to use in the tests.
N_samples = 1000 # Number of sequences to select for the procedure.
N_families = 100 # Number of proteins to train the model on.
N_steps = 100 # Number of training steps.
testfile = 'seqGen/out/words_generated_test.a3m'
trainfile = 'seqGen/out/words_generated_train.a3m'




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




# Trains the model on the specified dataset.
def train_model_preset_data():
    
    # Custom Print function to keep track of the runtime.
    time_begin = time.time()
    def Print(s):
      print (str(int(time.time()-time_begin))+' sec:        ' + str(s))   
    
    # Load the data and pick one of the proteins in it. Then load the set of multiple sequences.
    data = np.load (datafile, allow_pickle=True)
    X = data[test_protein].item()['ms']
    
    # Explore the data.
    Print ("Data for the " + test_protein + " protein has been loaded into variable x.")
    Print ("x is a " + str(np.shape(X)) + " array of entries like")
    Print (str(X[0]))
    Print ('Each number from 0 to 20 corresponds to one of 20 aminoacids.')
    
    # The data is too large for training within a reasonable timeframe. Take a smaller subset of data.
    x = nf.sub_sample(X, samples=N_samples)
            
    # Create the BasicAlign model.
    ms = nf.one_hot(nf.pad_max(x)) # pad_max converts a list of 1D-arrays into a 2D array, adding -1 to empty entries. For example, [[1,2,3],[4]] -> [[1,2,3],[4,-1,-1]]. One-hot then converts each 1D-array into an identity matrix with padding determined by the entries. Overall, this command just converts the data into the format the machine learning model expects.
    lens = np.array([len(y) for y in x]) # Array of lengths of sequences.
    gap = -3 # Alighnment gap penalty.
    model_basicAlign = nf.BasicAlign(X=ms, lengths=lens, sw_gap = gap) # This model will be trained.
    
    # Train the model, obtain and print the MSA parameters.
    Print ("")
    Print ("Training the BasicAlign model...")
    model_basicAlign.fit(N_steps, verbose=True)
    msa_params = model_basicAlign.opt.get_params()
    Print ("Parameters of the trained model: " + str(list(msa_params.keys())))
    
    # Create the TrainMRF model.
    x = nf.sub_sample(X, samples=N_samples)
    lens = np.array([len(y) for y in x])
    ms = nf.one_hot(nf.pad_max(x))
    model_trainMRF = nf.MRF(X=ms, lengths=lens, sw_gap = gap)
    
    # Update it with the BasicAlign parameters.
    mrf_params = model_trainMRF.opt.get_params()
    for p in ["emb","gap","open"]:
        mrf_params[p] = msa_params[p]
    model_trainMRF.opt.set_params(mrf_params)
    
    # Train the model, obtain and print the MSA parameters.
    Print ("Training the MRF model...")
    model_trainMRF.fit(N_steps, verbose=True)
    mrf_params = model_trainMRF.opt.get_params()
    Print ("Parameters of the trained model: " + str(list(mrf_params.keys())))
    
    # Test the model's performance.
    #model_trainMRF.predict(x)
    
    
    
    
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