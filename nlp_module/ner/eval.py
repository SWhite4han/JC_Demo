# encoding=utf8
import itertools
import os
import pickle
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from nlp_module.ner import extra_config as cfg

from nlp_module.ner.data_utils import load_word2vec, input_from_line, BatchManager
from nlp_module.ner.loader import augment_with_pretrained, prepare_dataset
from nlp_module.ner.loader import char_mapping, tag_mapping
from nlp_module.ner.loader import load_sentences, update_tag_scheme
from nlp_module.ner.model import Model
from nlp_module.ner.utils import get_logger, make_path, clean, create_model, save_model
from nlp_module.ner.utils import print_config, save_config, load_config, test_ner
from Common.common_lib import batch

flags = tf.app.flags
flags.DEFINE_boolean("clean",       False,      "clean train folder")
flags.DEFINE_boolean("train",       False,      "Whether train the model")
# configurations for the model
flags.DEFINE_integer("seg_dim",     20,         "Embedding size for segmentation, 0 if not used")
flags.DEFINE_integer("char_dim",    100,        "Embedding size for characters")
flags.DEFINE_integer("lstm_dim",    100,        "Num of hidden units in LSTM, or num of filters in IDCNN")
flags.DEFINE_string("tag_schema",   "iobes",    "tagging schema iobes or iob")

# configurations for training
flags.DEFINE_float("clip",          5,          "Gradient clip")
flags.DEFINE_float("dropout",       0.5,        "Dropout rate")
flags.DEFINE_float("batch_size",    20,         "batch size")
flags.DEFINE_float("lr",            0.001,      "Initial learning rate")
flags.DEFINE_string("optimizer",    "adam",     "Optimizer for training")
flags.DEFINE_boolean("pre_emb",     True,       "Wither use pre-trained embedding")
flags.DEFINE_boolean("zeros",       False,      "Wither replace digits with zero")
flags.DEFINE_boolean("lower",       True,       "Wither lower case")

flags.DEFINE_integer("max_epoch",   100,        "maximum training epochs")
flags.DEFINE_integer("steps_check", 100,        "steps per checkpoint")
flags.DEFINE_string("ckpt_path",    "ckpt",      "Path to save model")
flags.DEFINE_string("script",       "conlleval",    "evaluation script")
flags.DEFINE_string("emb_file",     os.path.join("data", "vec.txt"),  "Path for pre_trained embedding")
flags.DEFINE_string("train_file",   os.path.join("data", "example.train"),  "Path for train data")
flags.DEFINE_string("dev_file",     os.path.join("data", "example.dev"),    "Path for dev data")
flags.DEFINE_string("test_file",    os.path.join("data", "example.test"),   "Path for test data")

flags.DEFINE_string("model_type", "idcnn", "Model type, can be idcnn or bilstm")
#flags.DEFINE_string("model_type", "bilstm", "Model type, can be idcnn or bilstm")

FLAGS = tf.app.flags.FLAGS
assert FLAGS.clip < 5.1, "gradient clip should't be too much"
assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
assert FLAGS.lr > 0, "learning rate must larger than zero"
assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]


class ner_obj(object):
    def __init__(self):
        self.config = load_config(cfg.config_file)
        self.logger = get_logger(cfg.log_file)
        # limit GPU memory
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        with open(cfg.map_file, "rb") as f:
            self.char_to_id, self.id_to_char, self.tag_to_id, self.id_to_tag = pickle.load(f)

        self.sess = tf.InteractiveSession(config=tf_config)
        self.model = create_model(self.sess, Model, cfg.ckpt_path, load_word2vec, self.config, self.id_to_char, self.logger)


    # def evaluate(sess, model, name, data, id_to_tag, logger):
    #     logger.info("evaluate:{}".format(name))
    #     ner_results = model.evaluate(sess, data, id_to_tag)
    #     eval_lines = test_ner(ner_results, cfg.result_path)
    #     for line in eval_lines:
    #         logger.info(line)
    #     f1 = float(eval_lines[1].strip().split()[-1])
    #
    #     if name == "dev":
    #         best_test_f1 = model.best_dev_f1.eval()
    #         if f1 > best_test_f1:
    #             tf.assign(model.best_dev_f1, f1).eval()
    #             logger.info("new best dev f1 score:{:>.3f}".format(f1))
    #         return f1 > best_test_f1
    #     elif name == "test":
    #         best_test_f1 = model.best_test_f1.eval()
    #         if f1 > best_test_f1:
    #             tf.assign(model.best_test_f1, f1).eval()
    #             logger.info("new best test f1 score:{:>.3f}".format(f1))
    #         return f1 > best_test_f1
    #
    #
    # def evaluate_line():
    #     config = load_config(cfg.config_file)
    #     logger = get_logger(cfg.log_file)
    #     # limit GPU memory
    #     tf_config = tf.ConfigProto()
    #     tf_config.gpu_options.allow_growth = True
    #     with open(cfg.map_file, "rb") as f:
    #         char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    #     with tf.Session(config=tf_config) as sess:
    #         model = create_model(sess, Model, cfg.ckpt_path, load_word2vec, config, id_to_char, logger)
    #         while True:
    #             # try:
    #             #     line = input("请输入测试句子:")
    #             #     result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
    #             #     print(result)
    #             # except Exception as e:
    #             #     logger.info(e)
    #
    #             line = input("Input text for test:")
    #             result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
    #             print(result)
    #
    #
    # def evaluate_lines():
    #     from termcolor import colored
    #     config = load_config(cfg.config_file)
    #     logger = get_logger(cfg.log_file)
    #     # limit GPU memory
    #     tf_config = tf.ConfigProto()
    #     tf_config.gpu_options.allow_growth = True
    #     sentences = list()
    #     with open(cfg.map_file, "rb") as f:
    #         char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    #     with open(cfg.test_file, "r") as rf:
    #         for sent in rf:
    #             if sent.strip():
    #                 sentences.append(sent)
    #     with tf.Session(config=tf_config) as sess:
    #         model = create_model(sess, Model, cfg.ckpt_path, load_word2vec, config, id_to_char, logger)
    #
    #     # sentences = ['中國商務部副部長鍾山前天證實，中美達成初步協議，近期內將不會調高人民幣匯率。',
    #     #              '乒乓球擂台赛首场半决赛战罢刘国梁王晨取得决赛权(附图片1张)本报浙江余姚1月24日电爱立信中国乒乓球擂台赛今天在浙江省余姚市举行了首场半决赛,解放军选手刘国梁和北京女选手王晨分别战胜各自对手,闯入决赛。',
    #     #              '【大陸中心】5月1日開幕的上海世博會邁入倒數，主題曲之一《2010等你來》卻讓主辦單位顏面盡失。中國網友近日爆料，該歌曲抄襲日本女歌手岡本真夜，1996年創作歌曲(不變的你就好），引起高度關注，上海世博會前晚決定暫停使用該作品。']
    #
    #         inputs = [input_from_line(line, char_to_id) for line in sentences]
    #         result = model.evaluate_line2(sess, inputs, id_to_tag)
    #         for article in result:
    #             string = article['string']
    #             pointer = 0
    #             for entity in article['entities']:
    #                 if entity['type'] == 'TIME':
    #                     color = 'blue'
    #                 elif entity['type'] == 'LOC':
    #                     color = 'magenta'
    #                 elif entity['type'] == 'ORG':
    #                     color = 'cyan'
    #                 elif entity['type'] == 'PER':
    #                     color = 'green'
    #                 else:
    #                     color = 'green'
    #                 entity_type = '(' + entity['type'] + ')'
    #                 print(string[pointer:entity['start']])
    #                 print(colored(string[entity['start']:entity['end']] + entity_type,
    #                               color))
    #                 pointer = entity['end']
    #             if pointer < len(string):
    #                 print(string[pointer:])
    #             print('-' * 30)


    def evaluate_lines_by_call(self, texts_list):
        """
           This function can process non empty single sentence or multiple sentences in a list.
        :param texts_list: a list contents single sentence or multiple sentences.
        :return: final_rlt: a list contents many lists of words with all result.
        """

        result = list()
        inputs = [input_from_line(line, self.char_to_id) for line in texts_list]

        for inps in batch(inputs, cfg.batch_size):
            result.extend(self.model.evaluate_line2(self.sess, inps, self.id_to_tag))

        # only catch words
        final_rlt = [[each['word'] for each in r['entities']] for r in result]
        return final_rlt


    # def evaluate_2(ckpt_path, test_file):
    #     config = load_config(cfg.config_file)
    #     logger = get_logger(cfg.log_file)
    #     # limit GPU memory
    #     tf_config = tf.ConfigProto()
    #     tf_config.gpu_options.allow_growth = True
    #     with open(cfg.map_file, "rb") as f:
    #         char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    #     with tf.Session(config=tf_config) as sess:
    #         model = create_model(sess, Model, ckpt_path, load_word2vec, config, id_to_char, logger)
    #
    #         test_sentences = load_sentences(test_file, FLAGS.lower, FLAGS.zeros)
    #         # Use selected tagging scheme (IOB / IOBES)
    #         update_tag_scheme(test_sentences, FLAGS.tag_schema)
    #         test_data = prepare_dataset(
    #             test_sentences, char_to_id, tag_to_id, FLAGS.lower
    #         )
    #
    #         data = BatchManager(test_data, 100)
    #
    #         logger.info("evaluate:test")
    #         ner_results = model.evaluate(sess, data, id_to_tag)
    #         eval_lines = test_ner(ner_results, cfg.result_path)
    #         for line in eval_lines:
    #             logger.info(line)
    #         f1 = float(eval_lines[1].strip().split()[-1])
    #
    #         best_test_f1 = model.best_test_f1.eval()
    #         if f1 > best_test_f1:
    #             tf.assign(model.best_test_f1, f1).eval()
    #             logger.info("new best test f1 score:{:>.3f}".format(f1))
    #         # return f1 > best_test_f1


# def main(_):
#     ner = ner_obj()
#     print(cfg.model_path.split('/')[-1])
#     if not cfg.test_file:
#         ner.evaluate_line()
#     else:
#         ner.evaluate_lines()


if __name__ == "__main__":
    # tf.app.run(main)
    # evaluate_2('ckpt_tchinese_bilstm', 'extra_data/sinica/data/sinica_ner.test')
    ner = ner_obj()
    keys = ner.evaluate_lines_by_call(['中國商務部副部長鍾山前天證實，中美達成初步協議，近期內將不會調高人民幣匯率。',
                                '乒乓球擂台赛首场半决赛战罢刘国梁王晨取得决赛权(附图片1张)本报浙江余姚1月24日电爱立信中国乒乓球擂台赛今天在浙江省余姚市举行了首场半决赛,解放军选手刘国梁和北京女选手王晨分别战胜各自对手,闯入决赛。'])
    print(keys)

