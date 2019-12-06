import os
import json

import tensorflow as tf
import numpy as np


from model_match_gen import Encoder, QASystem, Decoder,CNN
# from config_nyt_match_gen import Config
from config import Config



from data_utils import *
from os.path import join as pjoin

'''
add cnn representation as relation embedding
'''


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)



def run_func():
    conf=json.load(open('config.json'))
    print(conf)
    config=Config(conf)
    # config = Config()
    train = dataset(config.question_train, config.context_train, config.answer_train,config.cnn_output_train,config.cnn_list_train)
    dev = dataset(config.question_dev, config.context_dev, config.answer_dev, config.cnn_output_dev,config.cnn_list_dev)
    # test = dataset(config.question_test, config.context_test, config.answer_test, config.cnn_output_test,config.cnn_list_test)
    print(len(train))
    print(len(dev))
    # print(len(test))
   


    encoder = Encoder(config.hidden_state_size)
    decoder = Decoder(config.hidden_state_size)
    cnn = CNN(config,is_training=True)
    qa = QASystem(encoder, decoder, cnn, config)
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.01)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    qa.initialize_model(sess, config.train_dir)
    # qa.train(sess, [train, dev, test], config.train_dir,config)
    qa.train(sess, [train, dev], config.train_dir,config)




if __name__ == "__main__":
    run_func()
