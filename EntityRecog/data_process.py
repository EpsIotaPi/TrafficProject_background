#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 16:06:45 2020

@author: adam

TO: 
"""
import pickle


class DataLoader():
    def __init__(self, hp, mode='train'):
        self.batch_size = hp.batch_size
        self.max_length = hp.max_length
        self.mode = mode
        self.token_to_id, _ = self.load_data_map(hp.vocab_dir)
        self.target_to_idx = hp.label_map
        self.label_size = hp.label_size
        self.categories_to_idx = hp.categories_map
        self.categories_size = hp.categories_size
        if mode == "train":
            source_dir = hp.train_source_dir
            target_dir = hp.train_target_dir
            categories_dir = hp.train_categories_dir
        else:
            source_dir = hp.test_source_dir
            target_dir = hp.test_target_dir
            categories_dir = hp.test_categories_dir
        self.sources = self.load_data(source_dir, self.token_to_id, 'source')
        self.targets = self.load_data(target_dir, self.target_to_idx, 'target')
        self.categories = self.load_data(categories_dir, self.categories_to_idx, 'categories')
        self.batch_data = self.get_batch_data()

    def load_data_map(self, path):
        with open(path, "rb") as f:
            token_to_id, id_to_token = pickle.load(f)
        return token_to_id, id_to_token
    
    def one_hot(self, target):
        init = [0] * self.label_size
        init[target] = 1
        return init
        
    def load_data(self, file_dir, to_idx_map, data_type):
        datas = []
        with open(file_dir, 'r', encoding='UTF-8') as f:
            for line in f:
                if data_type == "source":
                    chars = [to_idx_map.get(char, to_idx_map.get("_UNK_")) for char in line.strip()]
                    chars = chars[:self.max_length]
                    chars[-1] = len(to_idx_map)
                    datas.append(chars)
                elif data_type == "categories":
                    category = to_idx_map.get(line.strip(), to_idx_map.get("_UNK_"))
                    datas.append(category)
                else:
                    targets = [to_idx_map.get(target, to_idx_map.get("O")) for target in line.strip().split()]
                    #targets = [self.one_hot(target) for target in targets[:self.max_length]]
                    targets = targets[:self.max_length]
                    datas.append(targets)
        return datas

    def get_batch_data(self):
        '''
            prepare data for batch
        '''
        batch_data = []
        index = 0
        while True:
            if index+self.batch_size >= len(self.sources):
                sample_sources = self.padding_data(self.sources[-self.batch_size:])
                sample_targets = self.padding_data(self.targets[-self.batch_size:])
                sample_categories = self.categories[-self.batch_size:]
                batch_data.append((sample_sources, sample_targets, sample_categories))
                break
            else:
                sample_sources = self.padding_data(self.sources[index:index+self.batch_size])
                sample_targets = self.padding_data(self.targets[index:index+self.batch_size])
                sample_categories = self.categories[index:index + self.batch_size]
                batch_data.append((sample_sources, sample_targets, sample_categories))
                index += self.batch_size
        return batch_data

    def padding_data(self, data):
        return [sample + [0]*(self.max_length-len(sample)) for sample in data]

    def iteration(self):
        idx = 0
        while True:
            yield self.batch_data[idx]
            idx += 1
            if idx > len(self.batch_data)-1:
                idx = 0

    def get_batch(self):
        for data in self.batch_data:
            yield data
    
    def get_batch_data_len(self):
        return len(self.batch_data)
