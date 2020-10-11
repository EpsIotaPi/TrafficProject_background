#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 21:24:21 2020

@author: adam

TO:
"""
import tensorflow as tf
from crf_modules import BiLstmCRF
from crf_data_process import DataLoader
from hyperparams import HyperParams as hp
from utils import decode, f1_score, get_logger

tf.reset_default_graph()
logger = get_logger(hp.log_file_crf)


def evaluate(sess, model, test_loader, hp):
    y_true = []
    y_pred = []
    print("restore model to evaluate......")
    ckpt = tf.train.get_checkpoint_state(hp.checkpoint_crf_dir)
    model.saver.restore(sess, ckpt.model_checkpoint_path)
    for step, (inputs, targets) in enumerate(test_loader.get_batch()):
        feed_dict = {model.inputs: inputs,
                     model.targets: targets,
                     model.keep_prob: 1}
        logits, accuracy, length, trans = sess.run([model.logits,
                                                    model.accuracy,
                                                    model.length,
                                                    model.trans], feed_dict=feed_dict)
        predicts = decode(logits, length, trans)
        y_true += targets
        y_pred += predicts
    for tag in hp.entity_to_english.values():
        recall, precision, f1 = f1_score(y_true, y_pred, tag, hp.label_map)
        logger.info("\t{}\trecall {:.3}\tprecision {:.3}\tf1 {:.3}".format(tag, recall, precision, f1))


train_loader = DataLoader(hp, "train")
test_loader = DataLoader(hp, "test")

with tf.Session() as sess:
    model = BiLstmCRF(hp, is_training=True)
    ckpt = tf.train.get_checkpoint_state(hp.checkpoint_crf_dir)

    sess.run(tf.global_variables_initializer())

    logger.info("start train......")
    for epoch in range(hp.epochs):
        for step, (inputs, targets) in enumerate(train_loader.get_batch()):
            feed_dict = {model.inputs: inputs,
                         model.targets: targets,
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

        model.saver.save(sess, hp.checkpoint_path_crf)
        logger.info("evaluate......")
        evaluate(sess, model, test_loader, hp)

