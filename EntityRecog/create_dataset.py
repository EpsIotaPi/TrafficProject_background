#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 00:00:15 2020

@author: adam

TO: 
"""

import os, random, pickle
from hyperparams import HyperParams as hp
from collections import Counter
from collections import OrderedDict
from sklearn.model_selection import train_test_split
import pickle as pkl


def get_sentence_target(sentence,label_list,entity_to_english):
    target = ["O" for i in range(len(sentence))]
    for line in label_list:
        try:
            _,loc,_ = line.split("\t")
            label,start,end = loc.split()
            start = int(start)
            end = int(end)
            
            if start<end-1:
                target[start] = "B-" + entity_to_english.get(label)
                target[end-1] = "E-" + entity_to_english.get(label)
                if start+1 < end-1:
                    for index in range(start+1,end-1):
                        target[index] = "I-" + entity_to_english.get(label)
        except Exception as e:
            print('error: ',e)
            print("line: ",line)
    return target
                    

outputs = [os.path.join(hp.output_dir,file) for file in os.listdir(hp.output_dir) if file.endswith("ann")]

sources = []
targets = []
for output in outputs:
    id = output.split("/")[1][:-4]
    with open(output,'r') as f:
        output_list = f.readlines()
        if output_list:
            input_ = os.path.join(hp.input_dir,id+".txt")
            with open(input_,"r") as f:
                source = f.read()
                target = get_sentence_target(source,output_list,hp.entity_to_english)
            sources.append(source)
            targets.append(target)

X_train,X_test,y_train,y_test=train_test_split(sources,targets,random_state=42,test_size=0.2)

with open(hp.train_source_dir,'w') as f:
    f.write("\n".join(X_train))

y_train = [' '.join(target) for target in y_train]
with open(hp.train_target_dir,"w") as f:
    f.write("\n".join(y_train))

with open(hp.test_source_dir,'w') as f:
    f.write("\n".join(X_test))

y_test = [' '.join(target) for target in y_test]
with open(hp.test_target_dir,"w") as f:
    f.write("\n".join(y_test))


def build_dictionary(sentences):
    """
    :param sentences:
    :param lowercase:
    :return: dict
    """
    vocabs = []
    for line in sentences:
        vocabs.extend(list(line))

    word_freqs = [
        ('_PAD_', 0), # default, padding
        ('_UNK_', 1), # out-of-vocabulary
    ]
    word_freqs = word_freqs + Counter(vocabs).most_common()
    token_to_id = OrderedDict()
    id_to_token = OrderedDict()

    for idx, ww in enumerate(word_freqs):
        token_to_id[ww[0]] = idx
        id_to_token[idx] = ww[0]
    return token_to_id, id_to_token

token_to_id, id_to_token = build_dictionary(sources)
with open(hp.vocab_dir,'wb') as f:
    pkl.dump((token_to_id, id_to_token),f)


def sentence2id(sent, vocab):
    """

    :param sent:
    :param word2id:
    :return:
    """
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in vocab:
            word = '<UNK>'
        sentence_id.append(vocab[word])
    return sentence_id


def batch_yield(data, batch_size, vocab, label_map, shuffle=False):
    """

    :param data:
    :param batch_size:
    :param vocab:
    :param label_map:
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = [label_map[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels


def read_dictionary(vocab_path):
    """

    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id