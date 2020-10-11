#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 21:24:21 2020

@author: adam

TO: 
"""
from sklearn.metrics import classification_report
import tensorflow as tf
from joint_m_models import BiLstmCRF
from data_process import DataLoader
from hyperparams import HyperParams as hp
from utils import decode, f1_score, get_logger
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tf.reset_default_graph()
logger = get_logger(hp.log_file)
judge_data = get_logger(hp.judge_data_file)


def evaluate(sess, model, test_loader, hp, epoch):
    y_true = []
    y_pred = []
    classify_true = []
    classify_pred = []
    print("restore model to evaluate......")
    ckpt = tf.train.get_checkpoint_state(hp.checkpoint_dir)
    model.saver.restore(sess, ckpt.model_checkpoint_path)
    for step, (inputs, targets, categories) in enumerate(test_loader.get_batch()):

        feed_dict = {model.inputs: inputs,
                     model.targets: targets,
                     model.categories: categories,
                     model.keep_prob: 1}
        logits, accuracy, length, trans, classify_outputs = sess.run([model.logits,
                                                                      model.accuracy,
                                                                      model.length,
                                                                      model.trans,
                                                                      model.y_pred_cls], feed_dict=feed_dict)
        predicts = decode(logits, length, trans)
        y_true += targets
        y_pred += predicts
        classify_true += categories
        classify_pred += classify_outputs.tolist()
    for tag in hp.entity_to_english.values():
        recall, precision, f1 = f1_score(y_true, y_pred, tag, hp.label_map)
        logger.info("\t{}\trecall {:.3}\tprecision {:.3}\tf1 {:.3}".format(tag, recall, precision, f1))
    labels = [0, 1, 2, 3, 4, 5]
    logger.info(classification_report(classify_true, classify_pred, labels))
    if epoch == hp.epochs - 1:
        judge_data.info("*********************分割线************************")
        judge_data.info("参数:\tclocks {}\theads {}\tmax_length {}\tkeep_prob {:.1}".format(hp.blocks, hp.num_heads, hp.max_length, hp.keep_prob))
        judge_data.info("指标")
        score = 0
        recall_all = 0
        precision_all = 0
        f1_all = 0
        for tag in hp.entity_to_english.values():
            recall, precision, f1 = f1_score(y_true, y_pred, tag, hp.label_map)
            score += f1
            recall_all += recall
            precision_all += precision
            f1_all += f1
            judge_data.info("\t{}\trecall {:.3}\tprecision {:.3}\tf1 {:.3}".format(tag, recall, precision, f1))
        score = score / 6
        recall_all = recall_all / 7
        precision_all = precision_all / 7
        f1_all = f1_all / 7
        judge_data.info("\tavg\trecall {:.3}\tprecision {:.3}\tf1 {:.3}".format(recall_all, precision_all, f1_all))
        judge_data.info("\n" + classification_report(classify_true, classify_pred, labels))
        score += classification_report(classify_true, classify_pred, labels, output_dict=True)["macro avg"]["f1-score"]
        judge_data.info("模型分数:\tscore {:.3}".format(score))


train_loader = DataLoader(hp, "train")
test_loader = DataLoader(hp, "test")

with tf.Session() as sess:

    model = BiLstmCRF(hp, is_training=True)
    ckpt = tf.train.get_checkpoint_state(hp.checkpoint_dir)

    sess.run(tf.global_variables_initializer())

    logger.info("start train......")
    for epoch in range(hp.epochs):
        for step, (inputs, targets, categories) in enumerate(train_loader.get_batch()):

            feed_dict = {model.inputs: inputs,
                         model.targets: targets,
                         model.categories: categories,
                         model.keep_prob: hp.keep_prob}
            global_steps, accuracy, loss, logits, trans, length, _ = sess.run([model.global_steps,
                                                                               model.accuracy,
                                                                               model.loss,
                                                                               model.logits,
                                                                               model.trans,
                                                                               model.length,
                                                                               model.train_op],
                                                                              feed_dict=feed_dict)
            logger.info("epoch: {}\t step: {}/{}\t loss: {:.3f}\t accuracy: {:.3f}".format(
                epoch, step, train_loader.get_batch_data_len(), loss, accuracy))
        model.saver.save(sess, hp.checkpoint_path)
        logger.info("evaluate......")
        evaluate(sess, model, test_loader, hp, epoch)












