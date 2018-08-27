import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.contrib import lookup


class Dataset(object):
    def __init__(self, config):
        self.config = config
        self.vocabulary = lookup.index_table_from_file(self.config.vocabulary_file, num_oov_buckets=0, default_value=0)
        self.pad_id = self.vocabulary.lookup(tf.constant(self.config.pad_sign))
        self.sentence_data = self.load_dataset_from_text(self.config.sentence_data_file_path)
        self.label_data = self.load_dataset_from_text(self.config.label_data_file_path)
        self.input_fn(self.sentence_data, self.label_data)

    def load_dataset_from_text(self, file_path):
        dataset = tf.data.TextLineDataset(file_path)
        dataset = dataset.map(lambda string: tf.string_split([string]).values, num_parallel_calls=self.config.num_parallel_calls)
        dataset = dataset.map(lambda tokens: (self.vocabulary.lookup(tokens), tf.size(tokens)), num_parallel_calls=self.config.num_parallel_calls)
        return dataset

    def input_fn(self, sentences, labels):
        self.dataset = tf.data.Dataset.zip((sentences, labels))
        padded_shapes = ((tf.TensorShape([None]), tf.TensorShape([])), (tf.TensorShape([None]), tf.TensorShape([])))
        padding_values = ((self.pad_id, 0), (self.pad_id, 0))
        # self.dataset = self.dataset.shuffle(buffer_size=self.config.buffer_size)
        self.dataset = self.dataset.padded_batch(self.config.batch_size, padded_shapes=padded_shapes, padding_values=padding_values)
        self.dataset = self.dataset.prefetch(1)
        if self.config.repeat:
            self.dataset = self.dataset.repeat()
        self.iterator = tfe.Iterator(self.dataset)
