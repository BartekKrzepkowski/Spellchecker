import os
import re
from nltk import ngrams, word_tokenize, pos_tag
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


def file_filter(path1, path2):
    pattern = re.compile(r'/..|/...|/:|/-|[^\x00-\x7F]')
    with open(path1, "r") as f, open(path2, "w") as g:
        for line in f:
            line = line.lstrip().lstrip("-").lstrip()
            if len(line) > 20 and (not pattern.search(line)) :
                line = line.replace("...", "")
                line = line.replace("..", "")
                g.write(line.lower())

def line_preprocessing(line):
    line = line.strip("\n").strip().lstrip()
    line = line[: -1].strip() if line[-1] == "." else line.strip()
    return line


def replace_proper_nouns_in_line(line, to_replace):
    tokenized  = word_tokenize(line)
    tagged_sent = pos_tag(list(tokenized))
    for i in range(len(tokenized)):
        if tagged_sent[i][1] == "NNP":
            tokenized[i] = tokenized[i][0] + to_replace
            
    return tokenized


def replace_proper_nouns_in_dict(vocab_freq):
    new_vocab_freq = defaultdict(int)
    for word in vocab_freq:
        new_word = word
        if pos_tag([word])[0][1] == "NNP":
            new_word = word[0] + "xxxx" 
        new_vocab_freq[new_word] += vocab_freq[word]
    return new_vocab_freq

def get_word2idx_dict(path):
    vocab_freq = get_words_with_freq_from_text(path)
    vocab_freq = replace_proper_nouns_in_dict(vocab_freq)
    word2idx_dict = defaultdict(int)
    decresed_words = sorted(vocab_freq, key=lambda x: vocab_freq[x], reverse=True)
    for i, word in enumerate(decresed_words):
        word2idx_dict[word] = i + 1
    return word2idx_dict

#zrownolegl defaultdict
#https://stackoverflow.com/questions/9256687/using-defaultdict-with-multiprocessing
   
def get_bigrams_from_sentences_given_in_lines(PATH, n, threshold):
    i = 0
    word2idx = get_word2idx_dict(PATH)
    pattern = re.compile(r'/..|/:')
    model = defaultdict(a)
    with open(PATH, "r") as f:
        for line in f:
            if len(line) > 20 and not pattern.search(line) :
                line = line_preprocessing(line)
                tokenized = replace_proper_nouns_in_line(line, "xxxx")
                tokenized = tuple([word2idx[word] for word in tokenized])
                ngramed = list(ngrams(tokenized, n, True, True))
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
   
    idx2word = defaultdict(str, [(v,k) for k,v in word2idx.items()])
       
    return model, idx2word, word2idx

    
def get_trigrams_from_sentences_given_in_lines(PATH, n, threshold):
    i = 0
    word2idx = get_word2idx(PATH)
    pattern = re.compile(r'/..|/:')
    model = defaultdict(b)
    with open(PATH, "r") as f:
        for line in f:
            if len(line) > 20 and not pattern.search(line) :
                line = line_preprocessing(line)
                tokenized = replace_proper_nouns_in_line(line, "xxxx")
                tokenized = tuple([word2idx[word] for word in tokenized])
                ngramed = list(ngrams(tokenized, n, True, True))
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
                
    idx2word = defaultdict(str, [(v,k) for k,v in word2idx.items()])
       
    return model, idx2word, word2idx


def get_words_with_freq_from_text(PATH):
    return Counter(re.findall(r'\w+', open(PATH).read()))


def get_words_from_text(PATH):
    return set(re.findall(r'\w+', open(PATH).read()))


def load_encoder_decoder(encoder_decoder_path):
    word2idx = get_word2idx_dict(encoder_decoder_path)
    idx2word = defaultdict(str, [(v,k) for k,v in word2idx.items()])
    return word2idx, idx2word
    
    




























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