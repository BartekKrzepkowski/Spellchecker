from joblib import Parallel, delayed
import math
import numpy as np
from nltk.tokenize import ToktokTokenizer
from pandas import DataFrame, read_csv


mapping = {
    "a": [["a", "q", "w", "s", "z"], 5 * [1 / 5]],
    "b": [["b", "v", "g", "h", "n"], 5 * [1 / 5]],
    "c": [["c", "x", "d", "f", "v"], 5 * [1 / 5]],
    "d": [["d", "s", "e", "r", "f", "c", "x"], 7 * [1 / 7]],
    "e": [["e", "w", "r", "d", "s", "3", "4"], 5 * [1 / 6] + 2 * [1 / 12]],
    "f": [["f", "d", "r", "t", "g", "v", "c"], 7 * [1 / 7]],
    "g": [["g", "f", "t", "y", "h", "b", "v"], 7 * [1 / 7]],
    "h": [["h", "g", "y", "u", "j", "n", "b"], 7 * [1 / 7]],
    "i": [["i", "u", "o", "k", "j", "8", "9"], 5 * [1 / 6] + 2 * [1 / 12]],
    "j": [["j", "h", "u", "i", "k", "m", "n"], 7 * [1 / 7]],
    "k": [["k", "j", "i", "o", "l", "m"], 6 * [1 / 6]],
    "l": [["l", "k", "o", "p"], 4 * [1 / 4]],
    "m": [["m", "n", "j", "k"], 4 * [1 / 4]],
    "n": [["n", "b", "h", "j", "m"], 5 * [1 / 5]],
    "o": [["o", "i", "p", "l", "k", "9", "0"], 5 * [1 / 6] + 2 * [1 / 12]],
    "p": [["p", "o", "l", "0"], 4 * [1 / 4]],
    "q": [["q", "w", "a", "1", "2"], 3 * [1 / 4] + 2 * [1 / 8]],
    "r": [["r", "e", "t", "f", "d", "4", "5"], 5 * [1 / 6] + 2 * [1 / 12]],
    "s": [["s", "a", "w", "e", "d", "x", "z"], 7 * [1 / 7]],
    "t": [["t", "r", "y", "g", "f", "5", "6"], 5 * [1 / 6] + 2 * [1 / 12]],
    "u": [list("uyijh78"), 5 * [1 / 6] + 2 * [1 / 12]],
    "v": [list("vcfgb"), 5 * [1 / 5]],
    "w": [list("wqesa23"), 5 * [1 / 6] + 2 * [1 / 12]],
    "x": [list("xzsdc"), 5 * [1 / 5]],
    "y": [list("ytuhg67"), 5 * [1 / 6] + 2 * [1 / 12]],
    "z": [list("zasx"), 4 * [1 / 4]],
    "1": [list("1qw2"), 4 * [1 / 4]],
    "2": [list("23wq1"), 5 * [1 / 5]],
    "3": [list("34ew2"), 5 * [1 / 5]],
    "4": [list("45re3"), 5 * [1 / 5]],
    "5": [list("56tr4"), 5 * [1 / 5]],
    "6": [list("67yt5"), 5 * [1 / 5]],
    "7": [list("78uy6"), 5 * [1 / 5]],
    "8": [list("89iu7"), 5 * [1 / 5]],
    "9": [list("90oi8"), 5 * [1 / 5]],
    "0": [list("0po9"), 4 * [1 / 4]]
}

alfabet = "qwertyuiopasdfghjklzxcvbnm"
row2 = list("qwertyuiop")
row1 = list("asdfghjkl")
row0 = list("zxcvbnm")


def mapping2(cr1, cr2):
    k = dict()
    chars = [cr1, cr2]
    rows = [row0, row1, row2]
    
    for cr in chars:
        if cr in alfabet:
            for i, row in enumerate(rows):
                if cr in row:
                    k[cr] = (i, row.index(cr))
        else:
            return [el for el in [cr1, cr2] if el in alfabet]

    if k[cr1][0] - k[cr2][0] == 0:
        if k[cr1][1] - k[cr2][1] == 0:
            return [cr2]
        elif k[cr1][1] - k[cr2][1] > 0:
            return rows[k[cr1][0]] [k[cr2][1]: k[cr1][1] + 1]
        else:
            return rows[k[cr1][0]] [k[cr1][1]: k[cr2][1] + 1]
    elif k[cr1][0] - k[cr2][0] > 0:
        if k[cr1][0] - k[cr2][0] == 0:
            return [cr1, cr2]
        elif k[cr1][1] - k[cr2][1] > 0:
            if k[cr1][1] - k[cr2][1] == 1:
                return rows[k[cr1][0]] [k[cr2][1] + 1: k[cr1][1] + 1] + rows[k[cr2][0]] [k[cr2][1]: k[cr1][1]]
            else:
                return rows[k[cr1][0]] [k[cr2][1] + 2: k[cr1][1] + 1] + rows[1] [k[cr2][1] + 1: k[cr1][1]] + rows[k[cr2][0]] [k[cr2][1]: k[cr1][1] - 1]
        else:
            return rows[k[cr1][0]] [k[cr1][1]: k[cr2][1] + 1] + rows[k[cr2][0]] [k[cr1][1]: k[cr2][1] + 1]
    else:
        if k[cr1][1] - k[cr2][1] == 0:
            return [cr2, cr1]
        elif k[cr1][1] - k[cr2][1] > 0:
            return rows[k[cr2][0]] [k[cr2][1]: k[cr1][1] + 1] + rows[k[cr1][0]] [k[cr2][1]: k[cr1][1] + 1]
        else:
            if k[cr2][0] - k[cr1][0] == 1:
                return rows[k[cr2][0]] [k[cr1][1] + 1: k[cr2][1] + 1] + rows[k[cr1][0]] [k[cr1][1]: k[cr2][1]]
            else:
                return rows[k[cr2][0]] [k[cr1][1] + 2: k[cr2][1] + 1] + rows[1] [k[cr1][1] + 1: k[cr2][1]] + rows[k[cr1][0]] [k[cr1][1]: k[cr2][1] - 1]
        
                

def swap_char(sentence, position, mapping=mapping):
    char = sentence.lower()[position]
    if char in mapping:
        to_swap = mapping[char]
    else:
        to_swap = [[char],[1]]
    return sentence[: position] + str(np.random.choice(to_swap[0], p=to_swap[1])) + sentence[position + 1: ]


def swap_with_neighbour(sentence, position):
    if sentence[position] in " ,./;'[]\<>?:{}!@#$% ^&*()" or sentence[position + 1] in " ,./;'[]\<> ?:{}!@#$%^&*()":
        return sentence
    return sentence[: position] + sentence[position + 1] + sentence[position] + sentence[position + 2:]


def add_char(sentence, position):
    square = mapping2(sentence.lower()[position], sentence.lower()[position + 1])
    length = len(square)
    if length == 0:
        return sentence
    else:
        return sentence[: position] + str(np.random.choice(square, p=length * [1 / length])) + sentence[position: ]

    
def loss_char(sentence, position):
    toktok = ToktokTokenizer()
    if sentence[position] in " ,./;'[]\<>?:{}!@#$% ^&*()":
        return sentence
    if sentence[position] == " ":
        return sentence
    if sentence[position] in toktok.tokenize(sentence):
        return sentence
    return sentence[: position] + sentence[position + 1: ]


functions = {
    1: swap_char,
    2: swap_with_neighbour,
    3: add_char,
    4: loss_char
}


def nb_of_errors_in_utterance(length):
    #creating distribution: Benford-like
    pre_dist = list(map(lambda x: math.log((x+2) / (x + 1), length) * (1 / (pow(x,3) + 1)), range(length)))
    norm_pre_dist = pre_dist / np.linalg.norm(pre_dist, 1)
    
    #moving most probably value
    most = length // 25
    a, b = norm_pre_dist[: most + 1], norm_pre_dist[most + 1: ]
    distribution = list(reversed(a)) + list(b)
    
    p = np.random.choice(range(length), p=distribution)
    return p



#wylosuj liczbe z przedziału [0;rang-1] - liczba bledów w wyrazeniu - rozkładem benforda
#dla każdej jedności, wylosuj miejsca, dla kazdego miejsca wylosuj zakłócenie rozkładem jednostajnym
#zakłócenia - zamiana litery, zamiana dwóch sasiednich znaków, zgubienie litery, wrzucenie dodatkowej litery

def error_generator(utterance):
    toktok = ToktokTokenizer()
    length = len(utterance)
    nb = nb_of_errors_in_utterance(length) + 1
    utterance = utterance + " "
    
    for i in range(nb):
        length = len(utterance) - 1
        position = np.random.choice(range(length), p=(length)*[1/(length)])
        l = len(toktok.tokenize(utterance))
        utterance_old = utterance
        nb = np.random.randint(1,5)
        utterance = functions[nb](utterance, position)
    
    return utterance
    

def pixie(valid_utterances):
    misspelled_utterances = []
    misspelled_utterances = Parallel(n_jobs=-1)(delayed(error_generator)(utterance) for utterance in valid_utterances)
    
    return misspelled_utterances




def get_train_data(path, file_name, sep):
    valid_utterances = []
    with open(path, "r") as f:
        for line in f:
            valid_utterances.append(line.strip("\n"))
            
    misspelled_utterances = pixie(valid_utterances)
    DataFrame([misspelled_utterances, valid_utterances]).T.to_csv(file_name, index=False, sep=sep)
