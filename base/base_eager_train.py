import tensorflow as tf
import tensorflow.contrib.eager as tfe


class BaseEagerTrain(object):
    def __init__(self, model, config, training_data_set, validation_data_set=None):
        self.model = model
        self.config = config
        self.training_data_set = training_data_set
        self.validation_data_set = validation_data_set
        self.cur_epoch = 0
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)

    def train(self):
        for cur_epoch in range(self.config.num_epoch):
            self.train_epoch()
            self.model.save()

    def train_epoch(self):
        raise NotImplementedError

    def train_step(self):
        raise NotImplementedError