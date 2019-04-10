import os
import re
from nltk import ngrams, word_tokenize, pos_tag, ToktokTokenizer
#from nltk.tokenize.moses import MosesDetokenizer
from collections import defaultdict, Counter

def nested_defaultdict(f, n):
    if n == 1:
        return defaultdict(f)
    else:
        return defaultdict(lambda: nested_defaultdict(f, n-1))
    
    
    
def a():
    return defaultdict(int)

def b():
    return defaultdict(a)


def file_filter(path1, path2, low_boundry=20, high_boundry=1000, is_lower=True):
    pattern1 = re.compile(r':|-|[^\x00-\x7F]')
    pattern2 = re.compile(r'[^\w]*$')
    pattern3 = re.compile(r'^[^\w]*')
    with open(path1, "r") as f, open(path2, "w") as g:
        for line in f:
            line = re.sub(pattern2, "", line)
            line = re.sub(pattern3, "", line)
            if high_boundry > len(line) > low_boundry and (not pattern1.search(line)) :
                line = line.replace("...", " ")
                line = line.replace("..", " ")
                line = line.replace("   ", " ")
                line = re.sub(pattern2, "", line)
                line = re.sub(pattern3, "", line)
                line = line + "\n"
                if is_lower:
                    line = lone.lower()
                g.write(line)

def line_preprocessing(line):
    pattern2 = re.compile(r'[^\w]*$')
    pattern3 = re.compile(r'^[^\w]*')
    line = re.sub(pattern2, "", line)
    line = re.sub(pattern3, "", line)
    return line[: -1].strip() if line[-1] == "." else line.strip()


def replace_proper_nouns_in_line(line, to_replace):
    toktok = ToktokTokenizer()
    tokenized  = toktok.tokenize(line)
    tagged_sent = pos_tag(tokenized)
    for i in range(len(tokenized)):
        if tagged_sent[i][1] == "NNP":
            tokenized[i] = tokenized[i][:2] + to_replace
            
    return tokenized

#zrownolegl defaultdict
#https://stackoverflow.com/questions/9256687/using-defaultdict-with-multiprocessing
   
def get_bigrams_from_sentences_given_in_lines(PATH, n, threshold):
    i = 0
    pattern = re.compile(r'/..|/:')
    model = defaultdict(a)
    with open(PATH, "r") as f:
        for line in f:
            if len(line) > 20 and not pattern.search(line) :
                line = line_preprocessing(line)
                tokenized = replace_proper_nouns_in_line(line, "xxxx")
                ngramed = ngrams(tokenized, n, True, True)
                for w1, w2 in ngramed:
                    model[w1][w2] += 1
                i += 1
                if i >= threshold:
                    break
            
    #sprobuj ogarnac tak zeby zapamietac cond
    for w1 in model:
        cond = sum(model[w1].values())
        for w2 in model[w1]:
            model[w1][w2] /= cond
       
    return model

    
def get_trigrams_from_sentences_given_in_lines(PATH, n, threshold):
    i = 0
    pattern = re.compile(r'/..|/:')
    model = defaultdict(b)
    with open(PATH, "r") as f:
        for line in f:
            if len(line) > 20 and not pattern.search(line) :
                line = line_preprocessing(line)
                tokenized = replace_proper_nouns_in_line(line, "xxxx")
                ngramed = ngrams(tokenized, n, True, True)
                for w1, w2, w3 in ngramed:
                    model[w1][w2][w3] += 1
                i += 1
                if i >= threshold:
                    break
            
    #sprobuj ogarnac tak zeby zapamietac cond
    for w1 in model:
        for w2 in model[w1]:
            cond = sum(model[w1][w2].values())
            for w3 in model[w1][w2]:
                model[w1][w2][w3] /= cond
       
    return model


def get_words_with_freq_from_text(PATH):
    return Counter(re.findall(r'\w+', open(PATH).read()))


def get_words_from_text(PATH):
    return set(re.findall(r'\w+', open(PATH).read()))




























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
                    to_update = next(map(lambda x: ngrams(x, n, True, True), [tokenized]))
                    n_grams_freq.update(to_update)
                    i += 1
                    if i >= threshold:
                        break
                    
        print(i)
        i += 1
        if i >= threshold:
            break
       
    return iter(n_grams_freq.items())