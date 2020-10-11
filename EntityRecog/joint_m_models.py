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
from .module import multihead_attention, ff
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
        self.categories = tf.placeholder(dtype=tf.int64, shape=[None], name="categories")
        self.global_steps = tf.Variable(0, trainable=False, name="global_step")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        self.src_masks = tf.math.equal(self.inputs, 0)
        self.d_ff = hp.d_ff
        
        used = tf.sign(tf.abs(self.inputs))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.length = tf.cast(length, tf.int32)
        
        inputs_embedding = self.embedding_layer(self.inputs)
        attention = self.mul_head_att_layer(queries=inputs_embedding)
        bilstm_outputs, states = self.bilstm_layer(attention, self.lstm_dim, "bilstm_layer", self.length)

        # 定义l2损失
        l2Loss = tf.constant(0.0)

        H = states[0] + states[1]
        # 得到Attention的输出
        output = self.attention(H)
        outputSize = self.lstm_dim

        # 全连接层的输出

        outputW = tf.get_variable(
            "outputW",
            shape=[outputSize, self.num_categories],
            initializer=tf.contrib.layers.xavier_initializer())

        outputB = tf.Variable(tf.constant(0.1, shape=[self.num_categories]), name="outputB")
        l2Loss += tf.nn.l2_loss(outputW)
        l2Loss += tf.nn.l2_loss(outputB)
        self.classify_logits = tf.nn.xw_plus_b(output, outputW, outputB, name="logits")
        self.y_pred_cls = tf.argmax(self.classify_logits, axis=-1, name="predictions")

        # 计算二元交叉熵损失
        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.classify_logits, labels=self.categories)
            self.classify_loss = tf.reduce_mean(losses) + 0 * l2Loss

        correct = tf.equal(self.y_pred_cls, self.categories)
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        self.logits = self.logits_layer(bilstm_outputs)
        self.loss = self.loss_layer(self.logits, self.targets, self.length)
        self.loss = self.loss + self.classify_loss

        tvars = tf.trainable_variables()
        opt = tf.train.AdamOptimizer()
        self.train_op = opt.minimize(self.loss, var_list=tvars)
        '''
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
        '''
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

            outputs = tf.concat(outputs, 2)
            att_out = tf.split(outputs, 2, -1)

        return outputs, att_out

    def mul_head_att_layer(self, queries):
        for i in range(self.blocks):
            with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                queries = multihead_attention(queries=queries,
                                              keys=queries,
                                              values=queries,
                                              key_masks=self.src_masks,
                                              num_heads=self.num_heads,
                                              dropout_rate=self.rate,
                                              training=True,
                                              causality=False)
                queries = ff(queries, num_units=[self.d_ff, self.embedding_dim])

        return queries

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

    def attention(self, H):
        """
        利用Attention机制得到句子的向量表示
        """
        # 获得最后一层LSTM的神经元数量
        hiddenSize = self.lstm_dim

        # 初始化一个权重向量，是可训练的参数
        W = tf.Variable(tf.random_normal([hiddenSize], stddev=0.1))

        # 对Bi-LSTM的输出用激活函数做非线性转换
        M = tf.tanh(H)

        # 对W和M做矩阵运算，W=[batch_size, time_step, hidden_size]，计算前做维度转换成[batch_size * time_step, hidden_size]
        # newM = [batch_size, time_step, 1]，每一个时间步的输出由向量转换成一个数字
        newM = tf.matmul(tf.reshape(M, [-1, hiddenSize]), tf.reshape(W, [-1, 1]))

        # 对newM做维度转换成[batch_size, time_step]
        restoreM = tf.reshape(newM, [-1, self.nums_steps])

        # 用softmax做归一化处理[batch_size, time_step]
        self.alpha = tf.nn.softmax(restoreM)

        # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(self.alpha, [-1, self.nums_steps, 1]))

        # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
        sequeezeR = tf.reshape(r, [-1, hiddenSize])

        sentenceRepren = tf.tanh(sequeezeR)

        # 对Attention的输出可以做dropout处理
        output = tf.nn.dropout(sentenceRepren, self.keep_prob)

        return output