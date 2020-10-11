#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 00:00:50 2020

@author: adam

TO: 
"""


class HyperParams(object):

    input_dir = "input"
    output_dir = "output"
    
    train_source_dir = "dataset/train_source.txt"
    train_target_dir = "dataset/train_target.txt"
    train_categories_dir = "dataset/train_label.txt"
    
    test_source_dir = "dataset/test_source.txt"
    test_target_dir = "dataset/test_target.txt"
    test_categories_dir = "dataset/test_label.txt"
    vocab_dir = "background/dataset/vocab.pkl"
    
    checkpoint_dir = "background/ckpt"
    checkpoint_path = "ckpt/bilstm_ner.org.ckpt"

    checkpoint_classify_dir = "ckpt_classify"
    checkpoint_path_classify = "ckpt_classify/bilstm_classify.org.ckpt"

    checkpoint_crf_dir = "ckpt_crf"
    checkpoint_path_crf = "ckpt_crf/bilstm_crf.org.ckpt"

    checkpoint_m_crf_dir = "ckpt_m_crf"
    checkpoint_path_m_crf = "ckpt_m_crf/m_bilstm_crf.org.ckpt"

    checkpoint_m_classify_dir = "ckpt_m_classify"
    checkpoint_path_m_classify = "ckpt_m_classify/m_bilstm_crf.org.ckpt"

    checkpoint_joint_dir = "ckpt_joint"
    checkpoint_path_joint = "ckpt_joint/joint_bilstm.org.ckpt"
    
    log_file = "logger/logger.txt"
    log_file_classify = "logger/logger_classify.txt"
    log_file_crf = "logger/logger_crf.txt"
    log_file_m_crf = "logger/logger_m_crf.txt"
    log_file_m_classify = "logger/logger_m_classify.txt"
    log_file_joint = "logger/logger_joint.txt"

    judge_data_file = "logger/judge_data.txt"
    voacb_size = 1483
    
    # -------------可调节参数---------------
    batch_size = 128
    epochs = 50

    max_length = 100

    learning_rate = 3e-4
    
    embedding_dim = 200
    d_ff = 800
    lstm_dim = 200
    keep_prob = 0.7
    rate = 0.3

    num_heads = 5
    blocks = 3
    # -------------可调节参数---------------

    entity_to_english = {'位置': 'position',
                         '方向': 'direction',
                         '时间': 'time',
                         '距离': 'distance',
                         '路段': 'road_section',
                         '高速名称': 'highway_name',
                         '高速编号': 'highway_number'}
    
    label_size = 22
    
    label_map = {"O": 0,
                 "B-highway_name": 1,
                 "I-highway_name": 2,
                 "E-highway_name": 3,
                 "B-highway_number": 4,
                 "I-highway_number": 5,
                 "E-highway_number": 6,
                 "B-direction": 7,
                 "I-direction": 8,
                 "E-direction": 9,
                 "B-time": 10,
                 "I-time": 11,
                 "E-time": 12,
                 "B-distance": 13,
                 "I-distance": 14,
                 "E-distance": 15,
                 "B-position": 16,
                 "I-position": 17,
                 "E-position": 18,
                 "E-road_section": 19,
                 "I-road_section": 20,
                 "B-road_section": 21}

    categories_map = {"其他情况": 0,
                      "交通事故": 1,
                      "道路施工": 2,
                      "道路拥堵": 3,
                      "恶劣天气": 4,
                      "大流量": 5,
                      }

    categories_size = 6
