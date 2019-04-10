import itertools

def generate1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def generate2(word): 
    "All edits that are two edits away from `word`."
    return set(e2 for e1 in generate1(word) for e2 in generate1(e1))

def known_generated(word, vocab):
    aproxx1 = set(w for w in generate1(word) if w in vocab)
    aproxx2 = set(w for w in generate2(word) if w in vocab)
    return aproxx1 or aproxx2
            