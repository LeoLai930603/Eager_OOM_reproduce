from base.base_eager_train import BaseEagerTrain
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
from timeit import default_timer as timer


class EagerBiLSTMTrainer(BaseEagerTrain):
    def __init__(self, model, config, training_data_set, validation_data_set=None):
        super(EagerBiLSTMTrainer, self).__init__(model, config, training_data_set, validation_data_set)
        self.count = 0

    def train_epoch(self):
        start_time = timer()
        training_inputs = None
        validation_inputs = None
        while True:
            try:
                step = self.model.global_step_tensor.numpy()
                training_inputs = self.train_step()
                training_loss = self.loss(training_inputs, 1.0)
                if self.validation_data_set:
                    validation_inputs = dict()
                    ((sequence_input, sequence_length), (label, _)) = self.validation_data_set.iterator.get_next()
                    validation_inputs["sequence_input"] = sequence_input
                    validation_inputs["sequence_length"] = sequence_length
                    validation_inputs["label"] = label
                    validation_loss = self.loss(validation_inputs, 1.0)
                    print("At step %d, Training Loss: %f, Validation Loss: %f" % (step, training_loss, validation_loss))
                else:
                    print("At step %d, Training Loss: %f" % (step, training_loss))
            except tf.errors.OutOfRangeError:
                if training_inputs:
                    epoch_training_loss = self.loss(training_inputs, 1.0)
                    if validation_inputs:
                        epoch_validation_loss = self.loss(validation_inputs, 1.0)
                        print("At Epoch %d, Epoch Training Loss: %f, Epoch Validation Loss: %f" % (self.cur_epoch, epoch_training_loss, epoch_validation_loss))
                    else:
                        print("At Epoch %d, Epoch Training Loss: %f" % (self.cur_epoch, epoch_training_loss))
                break
        self.training_data_set.iterator = tfe.Iterator(self.training_data_set.dataset)
        self.cur_epoch += 1
        print("It takes %f s to train this epoch..." % float(timer() - start_time))

    def train_step(self):
        inputs = dict()
        ((sequence_input, sequence_length), (label, _)) = self.training_data_set.iterator.get_next()
        inputs["sequence_input"] = sequence_input
        inputs["sequence_length"] = sequence_length
        inputs["label"] = label
        max_length = tf.reduce_max(inputs["sequence_length"])
        count = max_length * inputs["sequence_length"].shape[0]
        print(count)
        # self.count += max_length * inputs["sequence_length"].shape[0]
        # print(self.count)
        self.optimizer.minimize(lambda: self.loss(inputs, 0.5), global_step=self.model.global_step_tensor)
        return inputs

    def loss(self, inputs, dropout_keep_prob):
        self.model(inputs, dropout_keep_prob)
        return tf.losses.sparse_softmax_cross_entropy(labels=inputs["label"], logits=self.model.logit)

    def compute_gradients(self, inputs, dropout_keep_prob):
        with tf.GradientTape() as tape:
            loss_value = self.loss(inputs, dropout_keep_prob)
        return tape.gradient(loss_value, self.model.variables)
