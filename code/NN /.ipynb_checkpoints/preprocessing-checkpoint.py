from keras.preprocessing import sequence
from tools import print_vector



class Preprocessing:
    
    def __init__(self, character_to_pad):
        self.x_data = []
        self.y_data = []
        self.max_length_of_word = 0
        self.character_to_pad = character_to_pad
        self.dict_of_features = {character_to_pad: 0}
        self.list_of_features = [character_to_pad]
        self.max_features = 1
        
        
    def load_data(self, path):
        with open(path) as file:
            for line in file:
                input_utt, output_utt = line.strip().split("@")[:2]
                input_utt = input_utt[: -1]
                larger = max(len(input_utt), len(output_utt))
                if larger > self.max_length_of_word:
                    self.max_length_of_word = larger
                self.update_data(input_utt, output_utt)
        for i in range(10):
            print_vector(self.list_of_features, self.x_data[i], end_token='')
            print(' -> ', end='')
            print_vector(self.list_of_features, self.y_data[i])

        before_padding = self.x_data[0]
        self.x_data = sequence.pad_sequences(self.x_data, maxlen=self.max_length_of_word)
        self.y_data = sequence.pad_sequences(self.y_data, maxlen=self.max_length_of_word)
        after_padding  = self.x_data[0]

        print_vector(self.list_of_features, before_padding,end_token='')
        print(" -> after padding: ", end='')
        print_vector(self.list_of_features, after_padding)
        
        
    
        

    def update_data(self, input_utt, output_utt):
        self.x_data.append(self.utt2vec(input_utt))
        self.y_data.append(self.utt2vec(output_utt))


    def utt2vec(self, input_utt):
        vec = []
        for i in input_utt:
            if i not in self.dict_of_features:
                self.dict_of_features[i]=self.max_features
                self.list_of_features.append(i)
                self.max_features += 1
            vec.append(self.dict_of_features[i])
        return vec
        





