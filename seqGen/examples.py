# Contains example commands for *.a3m generation.


from seqGen import generate_a3m




if __name__ == "__main__":
    
    print ("Generating an *.a3m-file of words of length 20 altered by word removal and insertion: 10 sequences, 'amino' alphabet...")
    generate_a3m ('words_generated', 20, 10)
    
    print ("Generating an *.a3m-file with the specified original sentence altered by word removal and insertion: 15 sequences, 'full' alphabet...")
    sentence = 'THE ONLY REAL PRISON IS FEAR AND THE ONLY REAL FREEDOM IS FREEDOM FROM FEAR'
    sentence_length = len(sentence.split())
    generate_a3m ('words_specified', sentence_length, 15, core_sentence=sentence, w_alphabet='full')
    
    print ("Complete.")