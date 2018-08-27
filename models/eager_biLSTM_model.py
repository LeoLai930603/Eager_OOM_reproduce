from base.base_eager_model import BaseEagerModel
import tensorflow as tf
from tensorflow.contrib import rnn
import tensorflow.contrib.eager as tfe
from tensorflow.contrib.framework import argsort
import numpy as np


class EagerBiLSTM(BaseEagerModel):
    def __init__(self, config):
        super(EagerBiLSTM, self).__init__(config)
        self.setup_variables()
        self.init_saver()

    def __call__(self, inputs: dict, dropout_keep_prob: float, *args, **kwargs):
        with tf.name_scope("embedding"), tf.device("/cpu:0"):
            self.sequence_input = tf.nn.embedding_lookup(self.embedding_weights, inputs["sequence_input"])

        with tf.name_scope("biLSTM"):
            self.fw_cell_dropout = rnn.DropoutWrapper(self.fw_cell, input_keep_prob=dropout_keep_prob)
            self.bw_cell_dropout = rnn.DropoutWrapper(self.bw_cell, input_keep_prob=dropout_keep_prob)
            (self.fw_output, self.bw_output), _ = tf.nn.bidirectional_dynamic_rnn(self.fw_cell, self.bw_cell, self.sequence_input, inputs["sequence_length"], dtype=tf.float32)
            self.reformed_output = self.connect_forward_backward_output(self.fw_output, self.bw_output, inputs["sequence_length"])

        with tf.name_scope("output_layer"):
            self.logit = self.output_layer(self.reformed_output)

    def init_saver(self):
        self.saver = tfe.Saver(self.variables)

    def setup_variables(self):
        with tf.device("/cpu:0"), tf.variable_scope("embedding_var"):
            self.embedding_weights = tfe.Variable(tf.random_normal(shape=[self.config.vocabulary_size, self.config.embedding_size]),
                                                  name="embedding_weights")
            self.variables.append(self.embedding_weights)
        with tf.variable_scope("biLSTM_var"):
            self.fw_cell = rnn.BasicLSTMCell(num_units=self.config.num_hidden_unit)
            self.variables.append(self.fw_cell.trainable_variables)
            self.bw_cell = rnn.BasicLSTMCell(num_units=self.config.num_hidden_unit)
            self.variables.append(self.bw_cell.trainable_variables)
        with tf.variable_scope("output_var"):
            self.output_layer = tf.layers.Dense(self.config.vocabulary_size, kernel_initializer=tf.random_normal_initializer)
            self.variables.append(self.output_layer.trainable_variables)

    @staticmethod
    def connect_forward_backward_output(fw_output, bw_output, sequence_lengths):
        result = []
        for i in range(fw_output.shape[0]):
            current_sequence_length = sequence_lengths[i]
            nz_part_fw = fw_output[i][: current_sequence_length - 2, :]
            z_part_fw = fw_output[i][current_sequence_length:, :]
            nz_part_bw = bw_output[i][: current_sequence_length - 2, :]
            z_part_bw = bw_output[i][current_sequence_length:, :]
            r_nz_part_bw = nz_part_bw[::-1]
            reformed_nz_part = tf.concat([nz_part_fw, r_nz_part_bw], axis=1)
            reformed_z_part = tf.concat([z_part_fw, z_part_bw], axis=1)
            result.append(tf.concat([reformed_nz_part, reformed_z_part], axis=0))
        return tf.stack(result)