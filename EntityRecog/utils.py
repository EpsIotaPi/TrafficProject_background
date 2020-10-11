#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 00:55:14 2020

@author: adam

TO: 
"""
import logging
from tensorflow.contrib.crf import viterbi_decode


def get_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def decode(scores, lengths, trans):
    paths = []
    for score, length in zip(scores, lengths):
        path, _ = viterbi_decode(score, trans)
        paths.append(path)
    return paths


def f1_score(tar_path, pre_path, tag, tag_map):
    tp = 0.
    tn = 0.
    fn = 0.
    fp = 0.
    for fetch in zip(tar_path, pre_path):
        tar, pre = fetch
        tar_tags = get_tags(tar, tag, tag_map)
        pre_tags = get_tags(pre, tag, tag_map)
        for t_tag in tar_tags:
            if t_tag in pre_tags:
                tp += 1
            else:
                fn += 1
        for p_tag in pre_tags:
            if p_tag not in tar_tags:
                fp += 1
    recall = 0. if tp+fn == 0 else (tp/(tp+fn))
    precision = 0. if tp+fp == 0 else (tp/(tp+fp))
    f1 = 0. if recall+precision == 0 else (2*precision*recall)/(precision + recall)
    return recall, precision, f1


def get_tags(path, tag, tag_map):
    begin_tag = tag_map.get("B-" + tag)
    mid_tag = tag_map.get("I-" + tag)
    end_tag = tag_map.get("E-" + tag)
    single_tag = tag_map.get("S")
    all_tag = [begin_tag, end_tag, single_tag]
    o_tag = tag_map.get("O")
    begin = -1
    end = 0
    tags = []
    last_tag = 0
    
    for index, tag in enumerate(path):
        if tag == begin_tag and index == 0:
            begin = 0
        elif tag == begin_tag:
            begin = index
        elif tag == end_tag and last_tag in [mid_tag, begin_tag] and begin > -1:
            end = index
            tags.append([begin, end])
        elif tag == o_tag or tag == single_tag:
            begin = -1
        last_tag = tag
    return tags


def get_entity(char_seq, tag_seq, label_map):
    POS = get_eve_entity(tag_seq, char_seq, label_map, "position")
    DIR = get_eve_entity(tag_seq, char_seq, label_map, "direction")
    TIME = get_eve_entity(tag_seq, char_seq, label_map, "time")
    DIS = get_eve_entity(tag_seq, char_seq, label_map, "distance")
    RSCT = get_eve_entity(tag_seq, char_seq, label_map, "road_section")
    HWN = get_eve_entity(tag_seq, char_seq, label_map, "highway_name")
    HWNB = get_eve_entity(tag_seq, char_seq, label_map, "highway_number")
    return POS, DIR, TIME, DIS, RSCT, HWN, HWNB


def get_eve_entity(tag_seq, char_seq, tag_map, tag):
    labels = []
    label = []
    begin_tag = tag_map.get("B-" + tag)
    mid_tag = tag_map.get("I-" + tag)
    end_tag = tag_map.get("E-" + tag)
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == begin_tag:
            label.append(char)
        elif tag == mid_tag:
            label.append(char)
        elif tag == end_tag:
            label.append(char)
            label = "".join(label)
            labels.append(label)
            label = []
        else:
            continue
    return labels


def get_class(classify_outputs):
    if classify_outputs == 0:
        re = "其它"
    elif classify_outputs == 1:
        re = "借道通行"
    elif classify_outputs == 2:
        re = "分流、限流"
    elif classify_outputs == 3:
        re = "占用车道"
    elif classify_outputs == 4:
        re = "封道"
    elif classify_outputs == 5:
        re = "排队缓行"
    return re