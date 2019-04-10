from nltk import word_tokenize
from peter_norvig_utils import known_generated
from load_and_save import load_from_pickle
from preprocessing import get_words_with_freq_from_text
from random import shuffle

class ContextApproach():
    
    def __init__(self, load_bi_grams_path=None, load_tri_grams_path=None, create_grams_path=None, encoder_decoder_path=None,
                threshold=100000000):
        assert load_bi_grams_path or load_tri_grams_path or create_grams_path
        
        try:
            if load_bi_grams_path:
                self.bi_grams = load_from_pickle(load_bi_grams_path)
            elif create_grams_path:
                self.bi_grams = get_bigrams_from_sentences_given_in_lines(create_grams_path, 2, threshold)
            else:
                self.bi_grams = None

            if load_tri_grams_path:
                self.tri_grams = load_from_pickle(load_tri_grams_path)
            elif create_grams_path:
                self.tri_grams = get_trigrams_from_sentences_given_in_lines(create_grams_path, 3, threshold)
            else:
                self.tri_grams = None
        except:
            print("wrong path")
        

    def context_approach_bigrams(self, utterance):
        w1, w2 = word_tokenize(utterance)
        c_based_on_context = self.bi_grams[w1]
        if c_based_on_context:
            vocab = {c for c in c_based_on_context}
            candidates = known_generated(w2, vocab)
            if candidates:
                p_max = 0
                approx = []
                for c in candidates:
                    p = c_based_on_context[c]
                    if p_max < p:
                        p_max = p
                        approx = [c]
                    elif p_max == p:
                        approx.append(c)
                return approx

        return ["xxxxx"]
    #     return without_context(utterance[1:], reduce_gram(bi_gram, w1))

    def context_approach_trigrams(self, utterance):
        w1, w2, w3 = word_tokenize(utterance)
        c_based_on_context = self.tri_grams[w1][w2]
        if c_based_on_context:
            vocab = {c for c in c_based_on_context}
            candidates = known_generated(w3, vocab)
            if candidates:
                p_max = 0
                approx = []
                for c in candidates:
                    p = c_based_on_context[c]
                    if p_max < p:
                        p_max = p
                        approx = [c]
                    elif p_max == p:
                        approx.append(c)
                return approx
            
        return ["xxxxx"]
    #     return context_approach_bigrams(utterance[1:], reduce_gram(tri_grams, w2))


    def get_vocab_for_bigram(self):
        vocab = []
        for w1 in self.bi_grams:
            for w2 in self.bi_grams[w1]:
                vocab.append(w2)
        return vocab


    def get_vocab_for_trigram(self):
        vocab = []
        for w1 in self.tri_grams:
            for w2 in self.tri_grams[w1]:
                for w3 in self.tri_grams[w1][w2]:
                    vocab.append(w3)
        return vocab


    def reduce_gram_test_data(self, gram_test_data):
        reduced_gram_test_data = []
        for duo in gram_test_data:
            reduce_valid = " ".join(duo[0].split(" ")[1:])
            reduced_gram_test_data.append(tuple([reduce_valid, duo[1]]))
        shuffle(reduced_gram_test_data)
        return reduced_gram_test_data
    
    
    
