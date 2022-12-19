"""
This file along with the seqGen folder (https://github.com/perrineruth/diffsw) should be put in the core directory of SMURF (https://github.com/spetti/SMURF).


---


In case you get the following error

module 'jax.experimental' has no attribute 'optimizers'"

, in laxy.py replace 'from jax.experimental.optimizers import adam' with 'from jax.experimental.minmax import adam', or with 'from jax.example_libraries.optimizers import adam' (one of these should work).
"""


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pickle




# GLOBAL VARIABLES
data_alphabet = 'ARNDCQEGHILKMFPSTWYV' # String containing all symbols in the alphabet for the original data.
sim_alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' # String containing all symbols in the alphabet for the simulated data.
data_files_list = ['train_data.pickle'] # List of the pickle files with the data training outputs.
sim_files_list = ['train_sim.pickle'] # List of the pickle files with the simulation training outputs.
sim_a3m_files_list = [['seqGen/out/sim_1.a3m', 'seqGen/out/sim_2.a3m', 'seqGen/out/sim_3.a3m']] # List of lists of the pickle files with the simulation training outputs as *.a3m files.
color_list = ['red', 'blue', 'seagreen', 'black', 'purple', 'orange', 'teal', 'darkred', 'darkblue', 'darkgreen']




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




# Returns the ground truth alignment (string) for the seq_num'th sequence in file file_a3m. Replaces all dashes with dots.
def get_ground_truth_alignment(file_a3m, seq_num):
    i = -1
    for line in open(file_a3m,'r').readlines():
        if not line[0]=='>':
            i += 1
        if i==seq_num:
            return line[:-1].replace('-','.')




# Calculates the empirical quality of the recovered alignment seq2 relative to seq1. In essence, for every two consecutive elements in seq1, the number of times they are encountered in seq1 is compared to seq2, and these numbers are treated as elements of two vectors -- and then the normalized dot product of the vectors is computed. The score is 1 for equal sequences, and 0 for perfectly unaligned sequences. Sequences must be first converted to strings using the convert_to_alphabet() function.
# ---
# seq1: reference sequence as a string.
# seq2: target sequence as a string.
# alphabet: if not None, then only two consecutive elements containing, at least, one element from the alphabet is considered. STRONGLY RECOMMENDED to not be set to None, otherwise for "sparse" sequence the scores will be abnormally high.
# ---
# NOTE: may return 1 even if the alignments are not exactly the same! Happens whenever the first alignment is a substring/subarray of the second alignment.
# WARNING: each string must have, at least, 2 elements!
def get_alignment_quality(seq1 : str, seq2 : str, alphabet : None):
    seq1, seq2 = seq1.upper(), seq2.upper()
    vec1, vec2 = {}, {}
    if type(seq1)==str and type(seq2)==str:
        for s1, s2 in zip(seq1[:-1], seq1[1:]):
            if (alphabet is None) or (s1 in alphabet or s2 in alphabet):
                if not s1+s2 in vec1.keys():
                    vec1[s1+s2] = seq1.count(s1+s2)
                    vec2[s1+s2] = seq2.count(s1+s2)
    else:
        raise ValueError ("In get_alignment_quality(), seq1 and seq2 must both be strings! Values " + str(seq1) + " and " + str(seq2) + " of types " + str(type(seq1)) + " and " + str(type(seq2)) + " were submitted instead.")
    vec1, vec2 = np.array([v for v in vec1.values()]), np.array([v for v in vec2.values()])
    return np.dot(vec1,vec2)/np.linalg.norm(vec1)/np.linalg.norm(vec2) if not np.linalg.norm(vec2)==0 else 0




# Calculates the distribution of the empirical alignment scores for the recovered alignments in the pickle dictionary. Value -1 is assigned to those alignments that have not been recovered. If file_a3m is specified, then adjusts the quality based on the ground truth.
def get_alignment_quality_distrib(pickle_dict, alphabet, file_a3m=None):
    num_of_seqs = len(pickle_dict['input_MSA'])
    seq_dict = get_aln_seqs(pickle_dict, alphabet)
    raw_alignment_scores = [get_alignment_quality(seq_dict[0], seq_dict[i], alphabet) if i in seq_dict.keys() else -1 for i in range(num_of_seqs)]
    if file_a3m is None:
        return raw_alignment_scores
    else:
        true_alignment_scores = [get_alignment_quality(get_ground_truth_alignment(file_a3m, 0), get_ground_truth_alignment(file_a3m, i), alphabet) if i in seq_dict.keys() else -1 for i in range(num_of_seqs)]
        return [a/t if not (t==0 or a==0) else 0 for a, t in zip(raw_alignment_scores, true_alignment_scores)]
    
    
    
    
# Plots multiple alignment quality histograms; the list of alignment quality distributions (list of lists of values) must be specified. The negative values are ignored.    
def plot_alignment_quality_hist(list_of_distribs, color_list=['red', 'blue', 'black'], label_list=['hist_1', 'hist_2', 'hist_3'], plot_name='alignment_quality_hist', alpha=0.7, log_scale=False):
    color_list = color_list[:len(list_of_distribs)]
    label_list = label_list[:len(list_of_distribs)]
    plt.clf()
    plt.title('Empirical alignment quality distribution', size=24)
    plt.xlabel('Empirical alignment quality', size=24)
    plt.ylabel('Number of occurrences', size=24)
    if log_scale:
        plt.gca().set_yscale('log')
    for i, distrib in enumerate(list_of_distribs):
        distrib = np.array(distrib)
        distrib = distrib[distrib>=0]
        plt.hist(distrib, bins=50, color=color_list[i], fill=True,linewidth=1,histtype='step',alpha=alpha)
    legend_parts = []
    for i in range (0,len(list_of_distribs)):
        legend_parts.append(mlines.Line2D([], [], color=color_list[i], linestyle='None', marker='s', markersize=30, label=label_list[i]))
    plt.legend(handles=legend_parts, loc='upper left',fontsize=24)
    plt.gcf().set_size_inches(25.6, 14.4)
    plt.tight_layout()
    plt.gcf().savefig("plots/"+plot_name+".eps", bbox_inches='tight',pad_inches=0.01, dpi=100, format='eps')
    plt.gcf().savefig("plots/"+plot_name+".png", bbox_inches='tight',pad_inches=0.01, dpi=100, format='png')
            


    
if __name__ == "__main__":
    
    # Distributions for the protein data.
    data_distrib_list, data_protein_list = [], []
    for d in data_files_list:
        file = open(d, 'rb')
        data = pickle.load(file)
        for p in data.keys():
            data_distrib_list.append(get_alignment_quality_distrib(data[p], data_alphabet))
            data_protein_list.append(p)
        file.close()
    
    # Distributions for the simulated data.
    sim_raw_distrib_list, sim_dataset_list = [], []
    for s in sim_files_list:
        file = open(s, 'rb')
        data = pickle.load(file)
        for p in data.keys():
            sim_raw_distrib_list.append(get_alignment_quality_distrib(data[p], sim_alphabet))
            sim_dataset_list.append(p)
            
    # Distributions for the simulated data with ground truth established.
    sim_adj_distrib_list = []
    for s, a_list in zip(sim_files_list, sim_a3m_files_list):
        file = open(s, 'rb')
        data = pickle.load(file)
        for p, a in zip(data.keys(), a_list):
            sim_adj_distrib_list.append(get_alignment_quality_distrib(data[p], sim_alphabet, a))
    
    # Create sample alignment quality histograms.
    data_distrib, data_protein, sim_raw_distrib, sim_adj_distrib, sim_dataset = data_distrib_list[0], data_protein_list[0], sim_raw_distrib_list[0], sim_adj_distrib_list[0], sim_dataset_list[0]
    plot_alignment_quality_hist([data_distrib, sim_raw_distrib], label_list = ['Protein ' + data_protein, 'Dataset ' + sim_dataset], plot_name=data_protein + '_' + sim_dataset + '_EAQ_hist')
    plot_alignment_quality_hist([sim_raw_distrib, sim_adj_distrib], label_list = ['Dataset ' + sim_dataset + " : raw", 'Dataset ' + sim_dataset + " : adjusted"], plot_name=sim_dataset + '_EAQ_hist')
    plot_alignment_quality_hist(sim_adj_distrib_list, label_list = ['Dataset ' + sim_dataset for sim_dataset in sim_dataset_list], plot_name='EAQ_hist_full', alpha=0.5, log_scale=True)
    plot_alignment_quality_hist(data_distrib_list, label_list = ['Protein '+ p for p in data_protein_list], color_list=color_list, plot_name='EAQ_hist_data_full', alpha=0.5)