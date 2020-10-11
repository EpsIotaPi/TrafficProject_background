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
from tensorflow.contrib.crf import crf_log_likelihood, viterbi_decode
from module import multihead_attention


# from create_dataset import batch_yield, read_dictionary


class BiLstmCRF:

    def __init__(self, hp, is_training):
        self.num_categories = hp.categories_size
        self.blocks = hp.blocks

        self.num_classes = hp.label_size
        self.lstm_dim = hp.lstm_dim
        self.embedding_dim = hp.embedding_dim
        self.num_heads = hp.num_heads
        self.rate = hp.rate
        self.learning_rate = hp.learning_rate
        self.vocab_size = hp.voacb_size
        self.is_training = is_training
        self.batch_size = hp.batch_size
        self.nums_steps = hp.max_length
        self.initializer = initializers.xavier_initializer()
        self.inputs = tf.placeholder(dtype=tf.int32, shape=[None, hp.max_length], name="inputs")
        self.targets = tf.placeholder(dtype=tf.int32, shape=[None, hp.max_length], name="targets")
        self.categories = tf.placeholder(dtype=tf.int64, shape=[None, self.num_categories], name="categories")
        self.global_steps = tf.Variable(0, trainable=False, name="global_step")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        self.filter_width = 3
        self.num_filter = self.lstm_dim
        self.repeat_times = 4
        self.cnn_output_width = self.lstm_dim
        self.layers = [
            {
                'dilation': 1
            },
            {
                'dilation': 1
            },
            {
                'dilation': 2
            },
        ]

        used = tf.sign(tf.abs(self.inputs))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.length = tf.cast(length, tf.int32)

        inputs_embedding = self.embedding_layer(self.inputs)
        model_inputs = tf.nn.dropout(inputs_embedding, self.keep_prob)
        model_inputs = self.cnn_layer(model_inputs)
        bilstm_outputs, states = self.bilstm_layer(model_inputs, self.lstm_dim, "bilstm_layer", self.length)

        # classify
        states = tf.nn.dropout(states, keep_prob=self.keep_prob)
        self.classify_logits = tf.layers.dense(states, self.num_categories)
        self.y_pred_cls = tf.argmax(tf.nn.softmax(self.classify_logits), 1)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.classify_logits, labels=self.categories)
        l2_loss = tf.losses.get_regularization_loss()
        self.classify_loss = tf.reduce_mean(cross_entropy)
        self.classify_loss += l2_loss

        # ner
        self.logits = self.logits_layer(bilstm_outputs)
        self.loss = self.loss_layer(self.logits, self.targets, self.length)
        self.loss = self.loss + self.classify_loss
        self.accuracy, self.train_op = self.optimizer_layer(self.logits, self.targets, self.loss)

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
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_lstm_cell,
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

            outputs = tf.concat(outputs, axis=2)
            outputs = tf.nn.dropout(outputs, self.keep_prob)

        return outputs, att_out

    def cnn_layer(self, model_inputs, name=None):
        """
        :param idcnn_inputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, cnn_output_width]
        """
        model_inputs = tf.expand_dims(model_inputs, 1)  # [batch_size, 1, num_steps, emb_size]
        reuse = False
        if self.keep_prob == 1.0:
            reuse = True
        with tf.variable_scope("idcnn" if not name else name):
            shape = [1, self.filter_width, self.embedding_dim,
                     self.num_filter]
            filter_w = tf.get_variable("idcnn_filter",
                                       shape=[1, self.filter_width, self.embedding_dim, self.num_filter],
                                       initializer=initializers.xavier_initializer())
            layerInput = tf.nn.conv2d(model_inputs,
                                      filter_w,
                                      strides=[1, 1, 1, 1],
                                      padding="SAME",
                                      name="init_layer", use_cudnn_on_gpu=True) # [batch_size, 1, num_steps, emb_size]
            final_out_from_layers = []
            total_w_for_last_dim = 0
            for j in range(self.repeat_times):
                for i in range(len(self.layers)):
                    dilation = self.layers[i]['dilation']
                    isLast = True if i == (len(self.layers) - 1) else False
                    with tf.variable_scope("atrous-conv-layer-%d" %i,
                                           reuse=True
                                           if(reuse or j > 0) else False):
                        w = tf.get_variable("filterW",
                                            shape=[1, self.filter_width, self.num_filter, self.num_filter],
                                            initializer=tf.contrib.layers.xavier_initializer())
                        b = tf.get_variable("filterB", shape=[self.num_filter])
                        conv = tf.nn.atrous_conv2d(layerInput, w,
                                                   rate=dilation,
                                                   padding="SAME")
                        conv = tf.nn.bias_add(conv, b)
                        conv = tf.nn.relu(conv) # [batch_size, 1, num_steps, emb_size]
                        if isLast:
                            final_out_from_layers.append(conv)
                            total_w_for_last_dim += self.num_filter
                        layerInput = conv
            final_out = tf.concat(axis=3, values=final_out_from_layers)

            keepProb = 1.0 if reuse else 0.5
            final_out = tf.nn.dropout(final_out, keepProb)

            final_out = tf.squeeze(final_out, [1])
            final_out = tf.reshape(final_out, [-1, total_w_for_last_dim])
            self.cnn_output_width = total_w_for_last_dim

            with tf.variable_scope("project" if not name else name):
                with tf.variable_scope("reshape"):
                    w = tf.get_variable("w", shape=[self.cnn_output_width, self.lstm_dim],
                                        dtype=tf.float32,
                                        initializer=initializers.xavier_initializer())
                    b = tf.get_variable("b", initializer=tf.constant(0.001, shape=[self.lstm_dim]))

                    final_out = tf.nn.xw_plus_b(final_out, w, b)
                    final_out = tf.reshape(final_out, [-1, self.nums_steps, self.lstm_dim])
        return final_out

    def logits_layer(self, lstm_outputs):
        """
        :param lstm_outputs
        :return :a 3d tensor with shape of [batch_size, seq_length, number class]
        """
        output = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim * 2])
        with tf.variable_scope("hidden", reuse=tf.AUTO_REUSE):
            w = tf.get_variable("W", shape=[self.lstm_dim * 2, self.lstm_dim],
                                dtype=tf.float32, initializer=self.initializer
                                )
            b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
                                initializer=self.initializer
                                )

            hidden = tf.tanh(tf.nn.xw_plus_b(output, w, b))
        with tf.variable_scope("logits", reuse=tf.AUTO_REUSE):
            w = tf.get_variable("W", shape=[self.lstm_dim, self.num_classes],
                                initializer=self.initializer, dtype=tf.float32
                                )
            b = tf.get_variable("b", shape=[self.num_classes], dtype=tf.float32)
            pred = tf.nn.xw_plus_b(hidden, w, b)
        logits = tf.reshape(pred, shape=[-1, self.nums_steps, self.num_classes])
        return logits

    def loss_layer(self, logits, targets, length):
        """
        :param logits:
        :param targets:
        :param length:
        :return: a loss value
        """
        with tf.variable_scope("loss_layer", reuse=tf.AUTO_REUSE):
            self.trans = tf.get_variable("transitions", shape=[self.num_classes, self.num_classes],
                                         initializer=self.initializer)
            log_likelihood, self.trans = crf_log_likelihood(inputs=logits,
                                                            tag_indices=targets,
                                                            transition_params=self.trans,
                                                            sequence_lengths=length)
            loss = tf.reduce_mean(-log_likelihood)
        return loss

    def optimizer_layer(self, logits, targets, loss):
        with tf.variable_scope("optimizer_layer", reuse=tf.AUTO_REUSE):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            correct_prediction = tf.equal(tf.argmax(logits, 2), tf.cast(targets, tf.int64))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tvars = tf.trainable_variables()
            grads = tf.gradients(loss, tvars)
            train_op = optimizer.apply_gradients(
                zip(grads, tvars), global_step=self.global_steps)
            return accuracy, train_op
