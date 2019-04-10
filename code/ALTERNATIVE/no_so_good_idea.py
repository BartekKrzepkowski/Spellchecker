def take_context1_scan_and_compare(utterance, gramed):
    old = False
    approx_word = []
    context = word_tokenize(utterance) 
    sub_grams, amount, word = utils(gramed, context)
    
    if not amount:
        old = True
    else:
        top10 = [('',0)]
        sub_grams_copy, sub_grams = itertools.tee(sub_grams)
        vocab = iter({gram[1] for gram in sub_grams_copy})
        for c in vocab:
            if c[0] == word[0] or c[-1] == word[-1]:
                p = sum([gramed[gram] for gram in sub_grams if gram[1] == c]) / amount
                if top10[-1][1] < p:
                    top10 = get_top(top10, c, p)
        print(top10)

        tokens, dist = zip(*top10)
        approx_word = get_closest(tokens, word)
    if old:
#         vocab = skads
#         aproxx = peter_norvig_approach(word, vocab)
        pass
        
        
    return approx_word

def take_context1_generate_and_scan(utterance, gramed):
    aproxx = []
    old = False
    context = word_tokenize(utterance)
    sub_grams, amount, word = utils(gramed, context)
    
    if not amount:
        old = True
    else:
        vocab = {gram[1] for gram in sub_grams}
        generated_phrases = generate(word)
        valid_candidates = {phrase for phrase in generated_phrases if phrase in vocab}
        if not valid_candidates:
            old = True
        else:
            p_max = 0
            aproxx = []
            print(valid_candidates)
            for el in valid_candidates:
                p = sum([gramed[gram] for gram in sub_grams if gram[1] == el]) / amount
                #a co jeżeli jest równe prawdopodobniestwo? popraw to
                if p_max < p:
                    print(p, el)
                    p_max = p
                    aproxx = [el]
                elif p_max == p:
                    aproxx.append(el)
                        
    if old:
        pass
#         vocab = skads
#         aproxx = peter_norvig_approach(word, vocab)
        
    return aproxx
