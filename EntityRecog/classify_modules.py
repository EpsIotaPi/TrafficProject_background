#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 19:28:44 2020

@author: adam

TO:
"""
import os
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, DropoutWrapper
from tensorflow.contrib.layers.python.layers import initializers


class BiLstmCRF:

    def __init__(self, hp, is_training):
        self.num_categories = hp.categories_size

        self.num_classes = hp.label_size
        self.lstm_dim = hp.lstm_dim
        self.embedding_dim = hp.embedding_dim
        self.num_heads = hp.num_heads
        self.learning_rate = hp.learning_rate
        self.vocab_size = hp.voacb_size
        self.is_training = is_training
        self.batch_size = hp.batch_size
        self.nums_steps = hp.max_length
        self.initializer = initializers.xavier_initializer()
        self.inputs = tf.placeholder(dtype=tf.int32, shape=[None, hp.max_length], name="inputs")
        self.categories = tf.placeholder(dtype=tf.int64, shape=[None, self.num_categories], name="categories")
        self.global_steps = tf.Variable(0, trainable=False, name="global_step")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        used = tf.sign(tf.abs(self.inputs))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.length = tf.cast(length, tf.int32)

        inputs_embedding = self.embedding_layer(self.inputs)
        states = self.bilstm_layer(inputs_embedding, self.lstm_dim, "bilstm_layer", self.length)

        states = tf.nn.dropout(states, keep_prob=self.keep_prob)
        self.logits = tf.layers.dense(states, self.num_categories)

        self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.categories)
        l2_loss = tf.losses.get_regularization_loss()
        self.loss = tf.reduce_mean(cross_entropy)
        self.loss += l2_loss
        self.train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss)

        correct_pred = tf.equal(tf.argmax(self.categories, 1), self.y_pred_cls)
        self.classify_accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    def embedding_layer(self, inputs, zero_pad=True, scale=True):
        """
        :param zero_pad:
        :param scale:
        :return: embedding inputs
        """
        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
            lookup_table = tf.get_variable('embedding_matrix',
                                           dtype=tf.float32,
                                           shape=[self.vocab_size, self.embedding_dim],
                                           initializer=tf.orthogonal_initializer(),
                                           trainable=True)
            # zero pad
            lookup_table = tf.concat((tf.zeros(shape=[1, self.embedding_dim]), lookup_table[1:, :]), 0)
            # scale
            lookup_table = lookup_table * (self.embedding_dim ** 0.5)

        inputs_embedding = tf.nn.embedding_lookup(lookup_table, inputs)
        # dropout
        inputs_embedding = tf.nn.dropout(inputs_embedding, self.keep_prob)
        return inputs_embedding

    # bidirect lstm unit
    def bilstm_layer(self, inputs, n_units, scope, seq_len=None):
        """
        :param inputs:
        :param n_units:
        :param scope:
        :param seq_len:
        :return: a 3d tensor with shape of [batch_size, seq_length, 2*n_units]
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            fw_lstm_cell = DropoutWrapper(LSTMCell(num_units=n_units), output_keep_prob=self.keep_prob)
            bw_lstm_cell = DropoutWrapper(LSTMCell(num_units=n_units), output_keep_prob=self.keep_prob)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_lstm_cell,
                                                              cell_bw=bw_lstm_cell,
                                                              inputs=inputs,
                                                              sequence_length=seq_len,
                                                              dtype=tf.float32)

            out = outputs[0] + outputs[1]

            w = tf.Variable(tf.random_normal([self.embedding_dim], stddev=0.1))
            out_h = tf.tanh(out)
            alpha = tf.matmul(tf.reshape(out_h, [-1, self.embedding_dim]), tf.reshape(w, [-1, 1]))
            alpha = tf.nn.softmax(tf.reshape(alpha, [-1, self.nums_steps]))

            att_out = tf.matmul(tf.transpose(out, [0, 2, 1]), tf.reshape(alpha, [-1, self.nums_steps, 1]))
            att_out = tf.tanh(tf.squeeze(att_out, [2]))

        return att_out


