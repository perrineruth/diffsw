"""
This file should be put in the core directory of SMURF. In the same directory this file must be downloaded: https://files.ipd.uw.edu/krypton/data_unalign.npz


---


In case you get the following error

module 'jax.experimental' has no attribute 'optimizers'"

, in laxy.py replace 'from jax.experimental.optimizers import adam' with 'from jax.experimental.minmax import adam', or with 'from jax.example_libraries.optimizers import adam' (one of these should work).
"""

import numpy as np
import network_functions as nf # Module containing SMURF routines.




# GLOBAL VARIABLES
datafile = 'data_unalign.npz' # File containing the MSA data.
protein = '3A0YA' # Name of the protein to use in the tests.
N_samples = 1000 # Number of sequences to select for the procedure.
N_steps = 1000 # Number of training steps.




if __name__ == "__main__":
    
    # Load the data and pick one of the proteins in it. Then load the set of multiple sequences.
    data = np.load (datafile, allow_pickle=True)
    x = data[protein].item()['ms']
    
    # Explore the data.
    print ("Data for the " + protein + " protein has been loaded into variable x.")
    print ("x is a " + str(np.shape(x)) + " array of entries like")
    print (str(x[0]))
    print ('Each number from 0 to 20 corresponds to one of 20 aminoacids.')
    
    # The data is too large for training within a reasonable timeframe. Take a smaller subset of data.
    x = nf.sub_sample(x, samples=N_samples)
            
    # Create a smooth SW model.
    ms = nf.one_hot(nf.pad_max(x)) # pad_max converts a list of 1D-arrays into a 2D array, adding -1 to empty entries. For example, [[1,2,3],[4]] -> [[1,2,3],[4,-1,-1]]. One-hot then converts each 1D-array into an identity matrix with padding determined by the entries. Overall, this command just converts the data into the format the machine learning model expects.
    lens = np.array([len(y) for y in x]) # Array of lengths of sequences.
    gap = -3 # Alighnment gap penalty.
    model = nf.BasicAlign(X=ms, lengths=lens, sw_gap = gap) # This model will be trained.
    
    # Train the model, obtain and print the MSA parameters.
    print ("")
    print ("Training model...")
    model.fit(N_steps)