import tensorflow as tf


class BaseEagerModel(object):
    def __init__(self, config):
        self.config = config
        self.variables = []
        self.previous_check_point = ""
        self.init_global_step()

    def init_global_step(self):
        with tf.variable_scope("global_step"):
            self.global_step_tensor = tf.train.get_or_create_global_step()

    def save(self):
        print("Saving model...")
        self.previous_check_point = self.saver.save(self.config.prefix_name)
        print("Model saved...")

    def load(self):
        if self.previous_check_point:
            self.saver.restore(self.previous_check_point)

    def init_saver(self):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        raise NotImplementedError
