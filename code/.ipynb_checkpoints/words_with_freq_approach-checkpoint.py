from peter_norvig_utils import known_generated
from nltk.metrics.distance import edit_distance
import math
from preprocessing import get_words_with_freq_from_text

class FrequencyApproach():
    
    def __init__(self, path_to_vocab):
        try:
            self.vocab = get_words_with_freq_from_text(path_to_vocab)
        except:
            print("wrong path")
        

    def scan_and_compare(self, word):
        if word in self.vocab:
            return [word]
        candidates = []
        p_max = 0
        w_len = len(word)

        for c in self.vocab:  # c stands candidate
            d = edit_distance(c, word, transpositions=True)
            if d < w_len:
                p = self.vocab[c]  * (1 - (math.log(d) / math.log(w_len))) # przemyśl to
                if p_max < p:
                    p_max = p
                    candidates = [c]
                elif p_max == p:
                    candidates.append(c)
        if candidates:
            return candidates
        else:
            return [word]


    def scan_and_compare_check_first_and_last(self, word):
        if word in self.vocab:
            return word, 0
        candidates = []
        p_max = 0
        w_len = len(word)
        for c in self.vocab:
            if word[0] == c[0] or word[-1] == c[-1]:
                d = edit_distance(c, word, transpositions=True)
                if d < w_len:
                    p = self.vocab[c]  * (1 - (math.log(d) / math.log(w_len))) # przemyśl to
                    if p_max < p:
                        p_max = p
                        candidates = [c]
                    elif p_max == p:
                        candidates.append(c)
        if candidates:
            return candidates
        else:
            return [word]


    def peter_norvig_approach(self, word):
        if word in self.vocab:
            return [word]
        p_max = 0
        approx = []
        candidates = known_generated(word, self.vocab)
        for c in candidates:
            p = self.vocab[c]
            if p > p_max:
                p_max = p
                approx = [c]
            elif p == p_max:
                approx.append(c)
        if approx:
            return approx
        else:
            return [word]