import os
import re
from nltk import ngrams, word_tokenize, pos_tag
#from nltk.tokenize.moses import MosesDetokenizer
from collections import defaultdict, Counter

def line_preprocessing(line):
    line = line.strip("\n").strip().lstrip()
    line = line[: -1].strip() if line[-1] == "." else line.strip()
    line = line + "."
    return line


def replace_proper_nouns(line, to_replace):
    tokenized  = word_tokenize(line)
    tagged_sent = pos_tag(list(tokenized))
    for i in range(len(tokenized)):
        if tagged_sent[i][1] == "NNP":
            tokenized[i] = tokenized[0] + to_replace
            
    return tokenized

def get_ngrams_from_sentences_given_in_lines(PATH, n, threshold):
    i = 0
    pattern = re.compile(r'/..|/:')
    n_grams_freq = Counter()
    for filename in os.listdir(PATH):
        with open(os.path.join(PATH, filename), "r") as f:
            for line in f:
                if len(line) > 20 and not pattern.search(line) :
                    line = line_preprocessing(line)
                    tokenized = replace_proper_nouns(line, "xxxx")
                    to_update = next(map(lambda x: ngrams(x, n,True, True), [tokenized]))
                    n_grams_freq.update(to_update)
                    i += 1
                    if i >= threshold:
                        break
                    
        print(i)
        i += 1
        if i >= threshold:
            break
       
    return iter(n_grams_freq.items())


#ogarnij ner-a dla 3-gramow sprawdz model jezykowy

#zrownolegl defaultdict
#https://stackoverflow.com/questions/9256687/using-defaultdict-with-multiprocessing


from multiprocessing import Pool
from multiprocessing.managers import BaseManager, DictProxy
import dill

class MyManager(BaseManager):
    pass

MyManager.register('defaultdict', defaultdict, DictProxy)

def test(line, line_preprocessing, replace_proper_nouns, model):
    line = line_preprocessing(line)
    tokenized = replace_proper_nouns(line, "xxxx")
    ngramed = list(ngrams(tokenized, n, True, True))
    for w1, w2 in ngramed:
        model[w1][w2] += 1
            

def dd():
    return defaultdict(int)

def nested_defaultdict(f, n):
    if n == 1:
        return defaultdict(f)
    else:
        return defaultdict(lambda: nested_defaultdict(f, n-1))
 

def get_ngrams_from_sentences_given_in_lines2(PATH, n, threshold):
    i = 0
    pattern = re.compile(r'/..|/:') 
    pool = Pool(processes=4)
    mgr = MyManager()
    mgr.start()
    model = mgr.defaultdict(dd)
    print(type(model))
    for filename in os.listdir(PATH):
        with open(os.path.join(PATH, filename), "r") as f:
            for line in f:
                if len(line) > 20 and not pattern.search(line) :
                    pool.apply_async(test, (line, pattern, line_preprocessing, replace_proper_nouns, model))
                    i += 1
                    if i >= threshold:
                        break
                    
    pool.close()
    pool.join()
       
    return model


def get_n_grams_from_raw_text(PATH, n, threshold):
    pass

def get_words_with_freq_from_raw_text(PATH):
    return Counter(re.findall(r'\w+', open(PATH).read()))