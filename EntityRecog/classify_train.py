#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 21:24:21 2020

@author: adam

TO:
"""
from sklearn.metrics import classification_report
import tensorflow as tf
from classify_modules import BiLstmCRF
from classify_data_process import DataLoader
from hyperparams import HyperParams as hp
from utils import get_logger
import os
loss_n = []
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tf.reset_default_graph()
logger = get_logger(hp.log_file_classify)


def evaluate(sess, model, test_loader, hp):
    classify_true = []
    classify_pred = []
    print("restore model to evaluate......")
    ckpt = tf.train.get_checkpoint_state(hp.checkpoint_classify_dir)
    model.saver.restore(sess, ckpt.model_checkpoint_path)
    for step, (inputs, categories) in enumerate(test_loader.get_batch()):

        label = []
        for i in range(hp.batch_size):
            if categories[i] == 0:
                label.append([1, 0, 0, 0, 0, 0])
            elif categories[i] == 1:
                label.append([0, 1, 0, 0, 0, 0])
            elif categories[i] == 2:
                label.append([0, 0, 1, 0, 0, 0])
            elif categories[i] == 3:
                label.append([0, 0, 0, 1, 0, 0])
            elif categories[i] == 4:
                label.append([0, 0, 0, 0, 1, 0])
            elif categories[i] == 5:
                label.append([0, 0, 0, 0, 0, 1])

        feed_dict = {model.inputs: inputs,
                     model.categories: label,
                     model.keep_prob: 1}

        classify_outputs = sess.run([model.y_pred_cls], feed_dict=feed_dict)
        classify_true += categories
        classify_pred += classify_outputs[0].tolist()

    labels = [0, 1, 2, 3, 4, 5]
    logger.info(classification_report(classify_true, classify_pred, labels))


train_loader = DataLoader(hp, "train")
test_loader = DataLoader(hp, "test")

with tf.Session() as sess:
    model = BiLstmCRF(hp, is_training=True)
    ckpt = tf.train.get_checkpoint_state(hp.checkpoint_classify_dir)

    sess.run(tf.global_variables_initializer())

    logger.info("start train......")
    for epoch in range(hp.epochs):
        for step, (inputs, categories) in enumerate(train_loader.get_batch()):

            for i in range(hp.batch_size):
                if categories[i] == 0:
                    categories[i] = [1, 0, 0, 0, 0, 0]
                elif categories[i] == 1:
                    categories[i] = [0, 1, 0, 0, 0, 0]
                elif categories[i] == 2:
                    categories[i] = [0, 0, 1, 0, 0, 0]
                elif categories[i] == 3:
                    categories[i] = [0, 0, 0, 1, 0, 0]
                elif categories[i] == 4:
                    categories[i] = [0, 0, 0, 0, 1, 0]
                elif categories[i] == 5:
                    categories[i] = [0, 0, 0, 0, 0, 1]

            feed_dict = {model.inputs: inputs,
                         model.categories: categories,
                         model.keep_prob: hp.keep_prob}

            global_steps, classify_accuracy, loss, length, _ = sess.run([model.global_steps,
                                                                        model.classify_accuracy,
                                                                        model.loss,
                                                                        model.length,
                                                                        model.train_op],
                                                                        feed_dict=feed_dict)
            logger.info("epoch: {}\t step: {}/{}\t loss: {:.3f}\t accuracy: {:.3f}".format(
                epoch, step, train_loader.get_batch_data_len(), loss, classify_accuracy))

        model.saver.save(sess, hp.checkpoint_path_classify)
        logger.info("evaluate......")
        evaluate(sess, model, test_loader, hp)

