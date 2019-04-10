from joblib import Parallel, delayed
import re
import os
from collections import defaultdict
from nltk import word_tokenize



def compare(model, typo, valid):
    results = model(typo)
    if valid in results[: 1]:
        return 1  
    return 0

def get_accuracy(model, data, vocab, threads=-1):
    misspelled_data, valid_data = zip(*data)
    data_length = len(data)
    print("Ratio of OOV: ",len({el for el in valid_data if el not in vocab}) / len(valid_data))
    results = Parallel(n_jobs=threads)(delayed(compare)(model, misspelled_data[i], valid_data[i]) for i in range(data_length))
    return sum(results) / data_length


def get_test_data(path, sep):
    test_data =  []
    with open(path, "r") as f:
        for line in f:
            test_data.append(line.strip("\t").strip())
    test_data = list(set([tuple(pair.split(sep)) for pair in test_data]))
    test_data = list(map(lambda x: tuple(x), test_data))
    return list(map(lambda x: tuple([x[0].strip(), x[1]]), test_data))





#def get_test_data(path):
 #   test_data =  re.findall(r'[A-Za-z]+\s[A-Za-z]+', open(path).read())
  #  test_data = list(set([tuple(pair.split("\t")) for pair in test_data]))
   # return list(map(lambda x: tuple(x), test_data))