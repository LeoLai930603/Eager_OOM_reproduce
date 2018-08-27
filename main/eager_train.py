from bunch import Bunch
from models.eager_biLSTM_model import EagerBiLSTM
from trainers.eager_biLSTM_trainer import EagerBiLSTMTrainer
from data_loader.sequence_data_loader import Dataset
from datetime import datetime
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

tf.enable_eager_execution()


def get_vocabulary_size(file_path):
    with open(file_path, 'r', encoding="utf-8") as rf:
        return len(rf.readlines())


if __name__ == '__main__':
    configuration = dict()
    configuration["vocabulary_file"] = "../data/test_vocabulary.txt"
    configuration["pad_sign"] = "PAD"
    configuration["sentence_data_file_path"] = "../data/test_samples.txt"
    configuration["label_data_file_path"] = "../data/test_labels.txt"
    configuration["num_parallel_calls"] = 4
    configuration["buffer_size"] = 64
    configuration["batch_size"] = 32
    configuration["repeat"] = False
    configuration["num_epoch"] = 2
    configuration["learning_rate"] = 0.001
    configuration["vocabulary_size"] = get_vocabulary_size(configuration["vocabulary_file"])
    configuration["embedding_size"] = 150
    configuration["dropout_keep_prob"] = 0.5
    configuration["num_hidden_unit"] = 150
    configuration["prefix_name"] = "../temp/biLSTM_cache"
    config = Bunch(configuration)
    training_dataset = Dataset(config)
    model = EagerBiLSTM(config)
    trainer = EagerBiLSTMTrainer(model, config, training_dataset)
    trainer.train()
    ckpt_path = model.previous_check_point
    print("Saved files at:")
    print(ckpt_path)
    print("Date and time: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
