from error_generator_const_nb import pixie
from preprocessing import nested_defaultdict
from pandas import DataFrame


def create_test_data_for_bi_grams(bi_grams): 
    first, second = zip(*bi_grams)
    second_fuzzy = pixie(second, False)
    unvalid_bi = zip(first, second_fuzzy)
    unvalid_utt = [" ".join(utt) for utt in unvalid_bi]
    valid_utt = list(second)
    
    data =  DataFrame([unvalid_utt, valid_utt]).T
    data.columns = ["typo_with_context", "valid"]
    return data


def create_test_data_for_tri_grams(tri_grams): 
    first, second, third = zip(*tri_grams)
    third_fuzzy = pixie(third, False)
    unvalid_tri = zip(first, second, third_fuzzy)
    unvalid_utt = [" ".join(utt) for utt in unvalid_tri]
    valid_utt = list(third)
    
    data =  DataFrame([unvalid_utt, valid_utt]).T
    data.columns = ["typo_with_context", "valid"]
    return data


def bi_gram_filter(bi_grams, vocab):
    new_bi_grams = nested_defaultdict(int, 2)
    for w1 in bi_grams:
        if w1 and len(w1) > 4 and w1 in vocab:
            for w2 in bi_grams[w1]:
                if w2 and len(w2) > 4 and w2 in vocab:
                    new_bi_grams[w1][w2] = bi_grams[w1][w2]
    return new_bi_grams


def tri_gram_filter(tri_grams, vocab):
    new_tri_grams = nested_defaultdict(int, 3)
    for w1 in tri_grams:
        if w1 and len(w1) > 3 and w1 in vocab:
            for w2 in tri_grams[w1]:
                if w2 and len(w2) > 4 and w2 in vocab:
                    for w3 in tri_grams[w1][w2]:
                        if w3 and len(w3) > 4 and w3 in vocab:
                            new_tri_grams[w1][w2][w3] = tri_grams[w1][w2][w3]
    return new_tri_grams