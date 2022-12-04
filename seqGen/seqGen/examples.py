# Contains example commands for *.a3m generation.


from seqGen import generate_a3m




if __name__ == "__main__":
    
    print ("Generating an *.a3m-file of words of length 20 altered by word removal and insertion: 10 sequences...")
    generate_a3m ('words_words', 20, 10)
    
    # THE FOLLOWING TWO EXAMPLES CURRENTLY ARE NOT WORKING; DEBUGGING IN PROGRESS.
    
    #print ("Generating an *.a3m-file of words of length 10 altered by character removal and insertion: 20 sequences...")
    #generate_a3m ('words_characters', 10, 20, alt_type='characters')
    
    #print ("Generating an *.a3m-file of characters of length 50 altered by character removal and insertion: 15 sequences...")
    #generate_a3m ('characters_characters', 50, 15, seq_type='characters', alt_type='characters', seq_spaced=False, generate_a3m2=False)