import itertools
from nltk.metrics.distance import edit_distance
def get_top(tab, c, p):
    if len(tab) == 1:
        return [(c, p)] + tab
    low = len(tab)-1
    top = 0
    
    while top < low:
        pom = (top+low) // 2
        if tab[pom][1] > p:
            top = pom + 1
        else:
            low = pom
            
    if tab[top] == p:
        dd = low
    else:
        dd = top
        
    ret =  tab[: dd] + [(c, p)] + tab[dd: ]
    return ret[: 100]

def get_closest(tokens, token):
    best_tokens = [tokens[0]]
    min_dist = edit_distance(best_tokens[0], token)
    for t in tokens[1: ]:
        d = edit_distance(t, token, transpositions=True)
        if d < min_dist:
            min_dist = edit_distance(t, token, transpositions=True)
            best_tokens = [t]
        elif d == min_dist:
            best_tokens.append(t)
            
    print(best_tokens, min_dist, token)
    return best_tokens
