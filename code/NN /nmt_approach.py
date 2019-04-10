import tensorflow as tf
import numpy as np
import random
from tools import *


CHECKPOINTS_PATH = "code/NN/checkpoint/"

class ApproachNMT:

    def __init__(self, data, target, num_unit, max_length_of_word, max_features, splitter_cut_off):
        self.max_length_of_word = max_length_of_word
        self.max_features = max_features
        
        self.encoded_input = [tf.placeholder(tf.int32, shape=(None,)) for i in range(max_length_of_word)]
        self.decoded_output = [tf.placeholder(tf.int32, shape=(None,)) for t in range(max_length_of_word)]
        self.weights = [tf.ones_like(labels_t, dtype=tf.float32) for labels_t in self.encoded_input]
        self.decoded_input = ([tf.zeros_like(self.encoded_input[0], dtype=np.int32)] + self.decoded_output[:-1])
        self.empty_decoded_input = ([tf.zeros_like(self.encoded_input[0], dtype=np.int32,name="empty_dec_input") for t in range(max_length_of_word)])
        self.runtime_outputs = None
        
        self.input_keep_prob = tf.placeholder(tf.float32, name='input_keep_prob')
        self.output_keep_prob = tf.placeholder(tf.float32, name='output_keep_prob')
        cell1 = tf.nn.rnn_cell.GRUCell(num_unit)
        cell2 = tf.nn.rnn_cell.DropoutWrapper(cell1, input_keep_prob=self.input_keep_prob, output_keep_prob=self.output_keep_prob)
        self.cell = tf.nn.rnn_cell.MultiRNNCell([cell2, cell1])
        
        
        self.optimizer = tf.train.AdamOptimizer()
        self.loss = None
         
        self.x_train_data, self.x_test_data = self.split_data(data, splitter_cut_off)
        self.y_train_data, self.y_test_data = self.split_data(target, splitter_cut_off)
        
        
    def split_data(self, data, splitter_cut_off):
        splitpoint = int(len(data) * splitter_cut_off)
        return data[: splitpoint], data[splitpoint:]
        

    def prepare_model(self, embedding_dim):
        with tf.variable_scope("decoder1") as scope:
            outputs, _ = tf.nn.seq2seq.embedding_attention_seq2seq(self.encoded_input, self.decoded_input, self.cell, self.max_features,
                                                                self.max_features, embedding_dim, feed_previous=False)
        with tf.variable_scope("decoder1",reuse=True) as scope:
            self.runtime_outputs, _ = tf.nn.seq2seq.embedding_attention_seq2seq(self.encoded_input, self.empty_decoded_input, self.cell, self.max_features, 
                                                                           self.max_features, embedding_dim, feed_previous=True)            
        self.loss = tf.nn.seq2seq.sequence_loss(outputs, self.decoded_output, self.weights, self.max_features)
        
        
    def train(self, nb_of_epochs, batch_size, is_continue, input_dropout=1, output_dropout=0.7):
        train_optimizer = self.optimizer.minimize(self.loss)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer()) 
            if is_continue:
                saver.restore(sess, CHECKPOINTS_PATH + "model.ckpt-1850")
            
            for step in range(nb_of_epochs):
                input_x, input_y = self.get_random_reversed_dataset(self.x_train_data, self.y_train_data, batch_size)
                input_x = np.array(input_x).T
                input_y = np.array(input_y).T
                feed_dict = {self.encoded_input[t]: input_x[t] for t in range(self.max_length_of_word)}
                feed_dict.update({self.decoded_output[t]: input_y[t] for t in range(self.max_length_of_word)})
                feed_dict.update({self.input_keep_prob: input_dropout, self.output_keep_prob: output_dropout})
                _, l = sess.run([train_optimizer, self.loss], feed_dict)
                
                if step % 50 == 0:
                    print(step, l)
                    saver.save(sess,CHECKPOINTS_PATH + "model.ckpt", global_step=step)
                    
            saver.save(sess,CHECKPOINTS_PATH + "final-model.ckpt")
            #saver0.export_meta_graph('./myModel.ckpt.meta')
                    
                    
    def test_by_printing(self, test_size, list_of_features, input_dropout=1, output_dropout=1, test_data=None): 
        if test_data:
            x_test_data, y_test_data = zip(*test_data)
        else:
            x_test_data = self.x_test_data
            y_test_data = self.y_test_data
        
        input_x, input_y = self.get_random_reversed_dataset(x_test_data, y_test_data, test_size)
        input_x = np.array(input_x).T
        input_y = np.array(input_y).T
        feed_dict = {self.encoded_input[t]: input_x[t] for t in range(self.max_length_of_word)}
        feed_dict.update({self.input_keep_prob: input_dropout, self.output_keep_prob: output_dropout})
        saver = tf.train.Saver()
        with tf.Session() as sess:
            #saver1 = tf.train.import_meta_graph(CHECKPOINTS_PATH + 'myModel.ckpt.meta')
            saver.restore(sess, CHECKPOINTS_PATH + "model.ckpt-2700")
            samples = sess.run(self.runtime_outputs, feed_dict)
            print_sample(list_of_features, input_x.T, input_y.T, samples, test_size)  
    
    
    def get_random_reversed_dataset(self, input_x, input_y, batch_size):
        new_input_x = []
        new_input_y = []
        for _ in range(batch_size):
            index_taken = random.randint(0, len(input_x) - 1)
            new_input_x.append(input_x[index_taken])
            new_input_y.append(input_y[index_taken][:: -1])
        return new_input_x, new_input_y


    def translate_single_word(self, word):
        input_x = [get_vector_from_string(word)]
        input_x = sequence.pad_sequences(input_x, maxlen=self.max_length_of_word)
        input_x = np.array(input_x).T
        feed_dict = {self.encoded_input[t]: input_x[t] for t in range(self.max_length_of_word)}
        out = sess.run(self.runtime_outputs, feed_dict)
        return get_reversed_max_string_logits(out)

