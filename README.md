# diffsw

Code for the course Algorithmic Evolutionary Biology. The following code is for running and testing the SMURF algorithm. This code is meant to be placed in the main SMURF repository [https://github.com/spetti/SMURF](https://github.com/spetti/SMURF).

### List of files:
 - run_testing.py: create an msa for a single protein through SMURF
 - run_training.py: create an msa for a simulated sentences through SMURF
 - run2.py: adaptation of run_testing.py that is specific about how Jax uses GPU memory
 - MAFFT and BLOSUM data.ipynb: comparison of protein outputs from run2.py to MAFFT
 - seqGen: files used to construct MSAs based off of sentences
