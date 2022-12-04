# Set of functions for generating sequences of random words with specified restrictions and their rearrangement.


import numpy as np
import os
import random




# Pre-made alphabets.
alphabet_amino = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
alphabet_RNA = ['T', 'C', 'G', 'A']
alphabet_full = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Directory the generated files are saved to.
save_dir = 'out'




# Returns the list of words available for use, according to specified restrictions. All the returned words are capitalized.
# ---
# file_word_list: relative path to the file containing the raw list of words.
# alphabet: 'amino' for aminoacids, 'RNA' for RNA, 'full' to use all English letters.
# include_capitalized: whether to include capitalized words.
# include_abbreviations: whether to include words with more than one symbol capitalized.
# ---
# EXAMPLES:
# >>> get_list_of_words()
def get_list_of_words(file_word_list='word_list.txt', alphabet='amino', include_capitalized=True, include_abbreviations=False):
    
    # Obtain the raw list of words and convert it to the list of strings.
    full_word_list = [str(s) for s in list(np.genfromtxt(file_word_list, dtype=str, skip_header=0, usecols=0, comments=None))]
    
    # Set the alphabet.
    if alphabet == 'amino':
        alphabet = alphabet_amino
    elif alphabet == 'RNA':
        alphabet = alphabet_RNA
    elif alphabet == 'full':
        alphabet = alphabet_full
    else:
        raise ValueError ("Argument 'alphabet' in get_list_of_words() must be 'amino', 'RNA' or 'full'. Invalid value '" + str(alphabet) + "' was submitted instead.")
        
    # Remove the words based on the capitalization and abbreviation requirements.
    if not include_capitalized:
        full_word_list = [s for s in full_word_list if not s[0].isupper()]
    if not include_abbreviations:
        full_word_list = [s for s in full_word_list if len([c for c in s if c.isupper()])<2]
        
    # Capitalize all the remaining words.    
    full_word_list = [s.upper() for s in full_word_list]
        
    # Remove the words that cannot be constructed with the alphabet.
    full_word_list = [s for s in full_word_list if len([c for c in s if c in alphabet])==len(s)]
    
    # Return the result.
    return full_word_list




# Generates a sentence of the given length.
# ---
# length: length of the sentence.
# length_type: 'characters' for the length to be in the number of characters (including spaces), 'words' for it to be in the number of words.
# spaced: if true, then individual words are separated by spaces. Otherwise all words are put together into a single sequence of symbols.
# ---
# EXAMPLES:
# gen_sentence(30, get_list_of_words())
# >>> 'ETHERNET FLYCATCHER FIFTH TAM'
def gen_sentence (length : int, word_list : [str], length_type='characters', spaced=True):
    
    if length<0 or (not int(length)==length):
        raise ValueError ("Argument 'length' in gen_sentence() must be a non-negative integer. Invalid value '" + str(length) + " of type " + str(type(length)) + " was submitted instead.")
        
    if length==0:
        return ''
    
    if length_type=='characters':
        return_string = ''
        while len(return_string)<length+int(spaced):
            new_word = random.choice(word_list)
            if len (return_string + new_word) <= length and not len (return_string + new_word) == length - 1:
                return_string += new_word + (' ' if spaced else '')
        return return_string[:-1] if spaced else return_string
    
    elif length_type=='words':
        return_string = ''
        for i in range(length):
            return_string += random.choice(word_list) + (' ' if spaced else '')
        return return_string[:-1] if spaced else return_string
    
    else:
        raise ValueError ("Argument 'length_type' in gen_sentence() must be 'characters' or 'words'. Invalid value '" + str(length_type) + "' was submitted instead.")
        
        
        
        
# Alters the sentence according to the alteration parameters selected. First the required number of characters/words is removed, then inserted.
# Returns the resulting sentence, as well as two lists: the list of indices sequentially removed words/sentences, and the list of tuples of indices and characters/words sequentially inserted.
# ---
# sentence: initial capitalized sentence.
# alphabet: alphabet used when generating this sentence. 'amino', 'RNA' or 'full' (see the description of the get_list_of_words() function).
# word_list: list of words to use. Only used if alteration_type='words' and insert_frac>0.
# remove_frac: fraction of the words of characters to remove from the sentence (rounded up).
# insert_frac: fraction of the words of characters to randomly insert in sentence (rounded up).
# alteration_type: 'characters' to insert or remove characters, and 'words' for words. Words will be added between other words, while characters will be inserted at random locations throughout the sentence. If 'words', then spaced MUST BE True.
# spaced: whether the words in the sentence are separated by spaces. If False, then alteration_type MUST BE 'characters'.
# ---
# EXAMPLES:
# alter_sentence ('SAY HELLO TO MY LITTLE FRIEND, AS WELL AS HIS ENORMOUS DOG', 'full', get_list_of_words())
# >>> ['SAY HELLO TO LITTLE AS WELL AS HIS YARDSTICK ENORMOUS NEVA DOG', [3,4], (8, 'YARDSTICK'), (10, 'NEVA')]]
# alter_sentence ('SAY HELLO TO MY LITTLE FRIEND, AS WELL AS HIS ENORMOUS DOG', 'full', get_list_of_words(), alteration_type='characters', spaced=True)
# >>> ['SAY HELO TO MY LITTLE DFRIND,L AS WWEL HAS HS ENORQMOU DO',[37, 7, 25, 40, 49], [(22, 'D'), (29, 'L'), (38, 'H'), (34, 'W'), (50, 'Q')]]
def alter_sentence (sentence : str, alphabet : str, word_list : [str], remove_frac=0.1, insert_frac=0.1, alteration_type='words', spaced=True):
    
    # Check if remove_frac and insert_frac are set correctly.
    if not (remove_frac>=0 and remove_frac<=1 and insert_frac>=0 and insert_frac<=1):
        raise ValueError ("In alter_sentence(), arguments remove_frac and insert_frac must be between 0 and 1. Values " + str(remove_frac) + " and " + str(insert_frac) + " were submitted instead.")
    
    # Calculate the number of characters/words to remove.
    if alteration_type == 'words':
        if spaced:
            N_remove = np.math.ceil(len(sentence.split())*remove_frac)
            N_insert = np.math.ceil(len(sentence.split())*insert_frac)
        else:
            raise ValueError ("In alter_sentence(), argument value 'words' of alteration_type is incompatible with the value True of spaced.")
    elif alteration_type == 'characters':
        N_characters = len([c for c in sentence if not c==' '])
        N_remove = np.math.ceil(N_characters*remove_frac)
        N_insert = np.math.ceil(N_characters*insert_frac)
    else:
        raise ValueError ("In alter_sentence(), alteration_type must be 'words' or 'characters'. Value '" + str(alteration_type) + "' was submitted instead.")
    
    # Set the alphabet.
    if alphabet == 'amino':
        alphabet = alphabet_amino
    elif alphabet == 'RNA':
        alphabet = alphabet_RNA
    elif alphabet == 'full':
        alphabet = alphabet_full
    else:
        raise ValueError ("Argument 'alphabet' in alter_sentence() must be 'amino', 'RNA' or 'full'. Invalid value '" + str(alphabet) + "' was submitted instead.")
        
    # Perform the alterations.
    list_deletions, list_insertions = [], []
    
    if alteration_type == 'characters':
        for i in range(N_remove):
            while True:
                rm_index = np.random.randint(0,len(sentence))
                if not sentence[rm_index]==' ':
                    sentence = sentence[:rm_index] + sentence[rm_index+1:]
                    list_deletions.append(rm_index)
                    break
        for i in range(N_insert):
            insert_char = random.choice(alphabet)
            insert_index = np.random.randint(0, len(sentence)+1)
            sentence = sentence[:insert_index] + insert_char + sentence[insert_index:]
            list_insertions.append((insert_index, insert_char))
        return [sentence[:-1], list_deletions, list_insertions]
            
    elif alteration_type == 'words':
        words_list = sentence.split()
        for i in range(N_remove):
            j = random.choice(list(np.arange(0, len(words_list))))
            words_list = words_list[:j] + words_list[j+1:]
            list_deletions.append(j)
        for i in range(N_insert):
            j = random.choice(list(np.arange(0, len(words_list))))
            new_word = random.choice(word_list)
            words_list = words_list[:j] + [new_word] + words_list[j:]
            list_insertions.append((j, new_word))
        sentence = ''
        for w in words_list:
            sentence += w + ' '
        return [sentence[:-1], list_deletions, list_insertions]
    
    else:
        raise ValueError ("Argument 'alteration_type' in alter_sentence() must be 'characters' or 'words'. Invalid value '" + str(alteration_type) + "' was submitted instead.")




# Provides the true alignment of multiple sequences. Returns the list of alignments in the FASTA format, where the first element corresponds to the core sequence.
# ---
# core_sentence: the sentence from which the altered sentences are derived.
# altered_sentence_list: list of outputs of the alter_sentence() function.
# alphabet: alphabet used when generating this sentence. 'amino', 'RNA' or 'full' (see the description of the get_list_of_words() function).
# alteration_type: 'characters' or 'words', depending on which alteration type was used when generating the altered sequences. Setting this wrong will result in nonsensical alignments!
# preserve_spaces: if True, the spaces are not removed from the outputs. Useful for better readibility of the alignments; should not be used when saving the alignments in the a3m format.
# ---
# EXAMPLES:
def align_sentences (core_sentence : str, altered_sentence_list : [str], alphabet : str, alteration_type='words', preserve_spaces=False):
    
    # Set the alphabet.
    if alphabet == 'amino':
        alphabet = alphabet_amino
    elif alphabet == 'RNA':
        alphabet = alphabet_RNA
    elif alphabet == 'full':
        alphabet = alphabet_full
    else:
        raise ValueError ("Argument 'alphabet' in get_list_of_words() must be 'amino', 'RNA' or 'full'. Invalid value '" + str(alphabet) + "' was submitted instead.")    
    
    # Transform sequences according to the deletion operations.
    
    if alteration_type == 'characters':
        for i in range(len(altered_sentence_list)):
            list_deletions = altered_sentence_list[i][1]
            sentence = core_sentence
            for index_del in list_deletions:
                count = -1
                for j in range(len(sentence)):
                    if sentence[j] in alphabet + [' ']:
                        count +=1
                    if count == index_del:
                        sentence = sentence[:j] + '-' + sentence[j+1:]
                        break
            altered_sentence_list[i][0] = sentence
            
    elif alteration_type == 'words':
        for i in range(len(altered_sentence_list)):
            list_deletions = altered_sentence_list[i][1]
            sentence_list = core_sentence.split()
            for index_del in list_deletions:
                count = -1
                for j in range(len(sentence_list)):
                    if not sentence_list[j][0]=='-':
                        count += 1
                    if count == index_del:
                        sentence_list = sentence_list[:j] + ['-'*len(sentence_list[j])] + sentence_list[j+1:]
                        break
            altered_sentence_list[i][0] = ' '.join(sentence_list)
        
    else:
        raise ValueError ("Argument 'alteration_type' in align_sentences() must be 'characters' or 'words'. Invalid value '" + str(alteration_type) + "' was submitted instead.")
        
    # Transform sequences according to the insertion operations.
    
    if alteration_type == 'characters':
        for i in range(len(altered_sentence_list)):
            if len(altered_sentence_list[i][2]) > 0:
                list_insertion_indices, list_insertion_values = [altered_sentence_list[i][2][k][0] for k in range(len(altered_sentence_list[i][2]))], [altered_sentence_list[i][2][k][1] for k in range(len(altered_sentence_list[i][2]))]
                sentence = altered_sentence_list[i][0]
                for index_ins, value_ins in zip(list_insertion_indices, list_insertion_values):
                    count = -1
                    for j in range(len(core_sentence)):
                        if sentence[j] in alphabet + [' '] or sentence[j].upper() in alphabet:
                            count += 1
                        if count == index_ins:
                            break
                    if core_sentence[j] == '.':
                        sentence = sentence[:j] + value_ins.lower() + sentence[j+1:]
                    else:
                        core_sentence = core_sentence[:j] + '.' + core_sentence[j:]
                        for k in range(len(altered_sentence_list)):
                            if k == i:
                                altered_sentence_list[k][0] = sentence[:j] + value_ins.lower() + sentence[j:]
                            else:
                                altered_sentence_list[k][0] = altered_sentence_list[k][0][:j] + '.' + altered_sentence_list[k][0][j:]
                    
    elif alteration_type == 'words':
        for i in range(len(altered_sentence_list)):
            if len(altered_sentence_list[i][2]) > 0:
                list_insertion_indices, list_insertion_values = [altered_sentence_list[i][2][k][0] for k in range(len(altered_sentence_list[i][2]))], [altered_sentence_list[i][2][k][1] for k in range(len(altered_sentence_list[i][2]))]
                sentence_list = altered_sentence_list[i][0].split()
                core_sentence_list = core_sentence.split()
                for index_ins, value_ins in zip(list_insertion_indices, list_insertion_values):
                    count = -1
                    for j in range(len(sentence_list)):
                        if sentence_list[j][0] in alphabet or sentence_list[j][0].upper() in alphabet:
                            count += 1
                        if count == index_ins:
                            break 
                    if core_sentence_list[j][0] == '.':
                        sentence_list = sentence_list[:j] + [value_ins.lower()] + sentence_list[j+1:]
                    else:
                        core_sentence_list = core_sentence_list[:j] + ['.'*len(value_ins)] + core_sentence_list[j:]
                        core_sentence = ' '.join(core_sentence_list)
                        for k in range(len(altered_sentence_list)):
                            if k == i:
                                altered_sentence_list[k][0] = ' '.join(sentence_list[:j] + [value_ins.lower()] + sentence_list[j:])
                            else:
                                split_altered_sentence = altered_sentence_list[k][0].split()
                                altered_sentence_list[k][0] = ' '.join(split_altered_sentence[:j] + ['.'*len(value_ins)] + split_altered_sentence[j:])
                                
    # Remove the spaces if needed.
    if not preserve_spaces:
        core_sentence = core_sentence.replace(' ', '')
        for i in range(len(altered_sentence_list)):
            altered_sentence_list[i][0] = altered_sentence_list[i][0].replace(' ', '')
                                
    return [core_sentence] + [altered_sentence[0] for altered_sentence in altered_sentence_list]




# Generates an *.a3m file with the specified parameters. May also produce a *.a3m2 file if requested.
# ---
# filename: name of the file(s) to produce (minus the extension).
# seq_length: length of each generated sequence, either in words (if seq_type='words') or characters (if seq_type='characters').
# seq_file_length: number of generated sequences, including the initial sequence. Must be a positive integer.
# seq_type: 'words' or 'characters'. Whether to generate a sequence of words or random characters from the alphabet.
# seq_spaced: whether to add spaces between words. Must be False if seq_type='characters'.
# core_sentence: if None, then the initial sentence is generated randomly. If specified, then the altered sentences are generated off it.
# alt_remove_frac: a tuple of two values between 0 and 1. For each sequence other than the initial one, a value between these values will be drawn uniformly, and the resulting value is the fraction of characters/words that will be randomly removed (rounded up).
# alt_insert_frac: same as alt_remove_frac, only for insertions.
# alt_type: 'words' or 'characters'. Whether to alter the initial sentence with new words and removal of old words, or charaters randomly appended to or removed from existing words.
# w_alphabet: 'amino' for the alphabet of 20 aminoacid letters, 'RNA' for 4 RNA letters, and 'full' for all 26 English letters.
# w_include_capitalized: whether to include capitalized words in the dictionary of words.
# w_include_abbreviations: whether to include abbreviations in the dictionary of words.
# generate_a3m2: whether to generate an additional file containing the same information as *.a3m, only with spaces between words preserved. Useful for visual examination of the aligned sequences.
# ---
# EXAMPLES:
# generate_a3m ('test', 20, 10)
def generate_a3m (filename: str, seq_length : int, seq_file_length : int, seq_type='words', seq_spaced=True, core_sentence=None, alt_remove_frac=(0.1,0.1), alt_insert_frac=(0.1,0.1), alt_type='words', w_alphabet='amino', w_include_capitalized=True, w_include_abbreviations=False, generate_a3m2=True):
    
    # Obtain the full list of words.
    full_world_list = get_list_of_words(alphabet=w_alphabet, include_capitalized=w_include_capitalized, include_abbreviations=w_include_abbreviations)
    
    # Do the checks for the validity of inputs.
    if not (int(seq_length)==seq_length and seq_length>0):
        raise ValueError ("In generate_a3m(), the 'seq_length' argument must be a positive integer. Value " + str(seq_length) + " was submitted instead.")
    if not (int(seq_file_length)==seq_file_length and seq_file_length>0):
        raise ValueError ("In generate_a3m(), the 'seq_file_length' argument must be a positive integer. Value " + str(seq_file_length) + " was submitted instead.")
    if not (seq_type=='words' or seq_type=='characters'):
        raise ValueError ("In generate_a3m(), the 'seq_type' argument must be 'words' or 'characters'. Value " + str(seq_type) + " was submitted instead.")
    if not (seq_spaced==True or seq_spaced==False):
        raise ValueError ("In generate_a3m(), the 'seq_spaced' argument must be of the boolean type. Value " + str(seq_spaced) + " was submitted instead.")
    if not (core_sentence is None or type(core_sentence)==str):
        raise ValueError ("In generate_a3m(), the 'core_sentence' argument must be either None or a string. Value " + str(core_sentence) + " or type " + str(type(core_sentence)) + " was submitted instead.")
    if not (max(alt_remove_frac)<=1 and min(alt_remove_frac)>=0 and type(alt_remove_frac)==tuple and len(alt_remove_frac)==2):
        raise ValueError ("In generate_a3m(), the 'alt_remove_frac' argument must be a tuple of two float values between 0 and 1. Value " + str(alt_remove_frac) + " was submitted instead.")
    if not (alt_remove_frac[0] <= alt_remove_frac[1]):
        raise ValueError ("In generate_a3m(), the first element of the 'alt_remove_frac' tuple must not exceed the second element. The tuple " + str(alt_remove_frac) + " was submitted instead.")
    if not (max(alt_insert_frac)<=1 and min(alt_insert_frac)>=0 and type(alt_insert_frac)==tuple and len(alt_insert_frac)==2):
        raise ValueError ("In generate_a3m(), the 'alt_insert_frac' argument must be a tuple of two float values between 0 and 1. Value " + str(alt_insert_frac) + " was submitted instead.")
    if not (alt_insert_frac[0] <= alt_insert_frac[1]):
        raise ValueError ("In generate_a3m(), the first element of the 'alt_insert_frac' tuple must not exceed the second element. The tuple " + str(alt_insert_frac) + " was submitted instead.")
    if not (alt_type=='words' or alt_type=='characters'):
        raise ValueError ("In generate_a3m(), the 'alt_type' argument must be 'words' or 'characters'. Value " + str(alt_type) + " was submitted instead.")
    if (seq_type == 'characters' and seq_spaced):
        raise ValueError ("In generate_a3m(), the 'seq_type' argument must be 'words' if 'seq_spaced' is True.")
    if not (w_alphabet=='amino' or w_alphabet=='RNA' or w_alphabet=='full'):
        raise ValueError ("In generate_a3m(),, argument 'w_alphabet' must be 'amino', 'RNA' or 'full'. Invalid value '" + str(w_alphabet) + "' was submitted instead.")  
    if not (w_include_capitalized==True or w_include_capitalized==False):
        raise ValueError ("In generate_a3m(), the 'w_include_capitalized' argument must be of the boolean type. Value " + str(w_include_capitalized) + " was submitted instead.")
    if not (w_include_abbreviations==True or w_include_abbreviations==False):
        raise ValueError ("In generate_a3m(), the 'w_include_abbreviations' argument must be of the boolean type. Value " + str(w_include_abbreviations) + " was submitted instead.")
    if not (generate_a3m2==True or generate_a3m2==False):
        raise ValueError ("In generate_a3m(), the 'generate_a3m2' argument must be of the boolean type. Value " + str(generate_a3m2) + " was submitted instead.")
    if (generate_a3m2 and not seq_spaced):
        raise ValueError ("In generate_a3m(), the 'generate_a3m2' argument being True is incompatible with the 'seq_spaced' argument being False.")
        
    # Generate the initial sentence.
    if core_sentence is None:
        core_sentence = gen_sentence(seq_length, full_world_list, length_type=seq_type, spaced=seq_spaced)
        
    # Generate the full sequence list.    
    altered_sentences_list = []
    for i in range(seq_file_length-1):
        remove_frac = np.random.uniform(alt_insert_frac[0], alt_insert_frac[1])
        insert_frac = np.random.uniform(alt_insert_frac[0], alt_insert_frac[1])
        altered_sentence = alter_sentence(core_sentence, w_alphabet, full_world_list, remove_frac=remove_frac, insert_frac=insert_frac, alteration_type=alt_type, spaced=seq_spaced)
        altered_sentences_list.append(altered_sentence)
    output_sequence_list = align_sentences(core_sentence, altered_sentences_list, w_alphabet, alteration_type=alt_type, preserve_spaces=True)
            
    # Save the file(s).
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    if generate_a3m2:
        f_a3m2 = open(save_dir+'/'+filename+'.a3m2', 'w')
        for i, seq in enumerate(output_sequence_list):
            if i > 0:
                f_a3m2.write ('\n')
            f_a3m2.write ('>' + str(i) + '\n')
            f_a3m2.write(seq)
        f_a3m2.close()
    
    output_sequence_list = [s.replace(' ','') for s in output_sequence_list]
    f_a3m = open(save_dir+'/'+filename+'.a3m', 'w')
    for i, seq in enumerate(output_sequence_list):
        if i > 0:
            f_a3m.write ('\n')
        f_a3m.write ('>' + str(i) + '\n')
        f_a3m.write(seq)
    f_a3m.close()