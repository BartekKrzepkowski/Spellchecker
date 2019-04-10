from peter_norvig_utils import known_generated
from nltk.metrics.distance import edit_distance
from preprocessing import get_words_with_freq_from_text

class NaiveApproach():

    def __init__(self, path_to_vocab):
        try:
            self.vocab = get_words_with_freq_from_text(path_to_vocab)
        except:
            print("wrong path")
        

    def scan_and_compare(self, word):
        if word in self.vocab:
            return [word]
        candidates = ["xxxx"]
        d_min = edit_distance(candidates[0], word, transpositions=True)
        for c in self.vocab:  # c stands candidate
            d = edit_distance(c, word, transpositions=True)
            if d < d_min:
                candidates = [c]
                d_min = d
            elif d == d_min:
                candidates.append(c)
        return candidates


    def scan_and_compare_check_first_and_last(self, word):
        if word in self.vocab:
            return [word]
        candidates = ["xxxx"]
        d_min = edit_distance(candidates[0], word, transpositions=True)
        for c in self.vocab:
            if word[0] == c[0] or word[-1] == c[-1]:
                d = edit_distance(c, word, transpositions=True)
                if d < d_min:
                    candidates = [c]
                    d_min = d
                elif d == d_min:
                    candidates.append(c)
        return candidates


    def generate_and_scan(self, word):
        candidates = list(known_generated(word, self.vocab))
        if candidates:
            return candidates
        return ["xxxx"]