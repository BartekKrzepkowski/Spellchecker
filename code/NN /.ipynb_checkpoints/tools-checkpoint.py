import numpy as np

def print_vector(list_of_features, vector,end_token='\n'):
    print(''.join([list_of_features[i] for i in vector]),end=end_token)
    
    
def get_reversed_max_string_logits(list_of_features, logits):
    string_logits = logits[::-1]
    concatenated_string = ""
    for logit in string_logits:
        val_here = np.argmax(logit)
        concatenated_string += list_of_features[val_here]
    return concatenated_string


def print_sample(list_of_features, input_x, input_y, sample, chunk_size):
    sample = list(zip(*sample))
    print(list_of_features)
    sample = sample[: chunk_size]
    
    for idx, string_logits in enumerate(sample):
        print("input: ", end='')
        print_vector(list_of_features, input_x[idx])
        
        print("expected: ",end='')
        expected = input_y[idx][::-1]
        print_vector(list_of_features, expected)
        
        output = get_reversed_max_string_logits(list_of_features, string_logits)
        print("output: " + output)
        
         
        print("==============")
        
        
def split_data(self, data, splitter_cut_off):
        splitpoint = int(len(data) * splitter_cut_off)
        return data[: splitpoint], data[splitpoint:]
        
        

        
        
