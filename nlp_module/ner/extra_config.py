import os

# --------------- Model choice --------------
model_path = os.path.join(os.path.dirname(__file__), 'models/', 'traditional_chinese_bilstm')
ckpt_path = os.path.join(model_path, 'ckpt')
# model_type = 'idcnn'
model_type = 'bilstm'

# ---------------  Train  ---------------
clean = True
# emb_file = './extra_data/sc/w2v_model/zhwiki-embeddings-100.txt'
emb_file = '/home/c11tch/workspace/PycharmProjects/rnn_cws-master/word2vec_model/zhwiki-embeddings-100.txt'
config_file = os.path.join(model_path, 'config_file')
map_file = os.path.join(model_path, 'maps.pkl')
log_dir = os.path.join(model_path, 'log')
log_file = os.path.join(log_dir, 'train.log')
vocab_file = os.path.join(model_path, 'vocab.json')
summary_path = os.path.join(model_path, 'summary')
result_path = os.path.join(model_path, 'result')

# --------------- Using mix data ----------------
use_mix_data = True
mix_data_tc_or_sc = True  # True=tw, False=sc

data_list_tw = ['./extra_data/tw/1998.tc',
                './extra_data/tw/sinica_ner.dev',
                './extra_data/tw/sinica_ner.test',
                './extra_data/tw/sinica_ner.train',
                './extra_data/tw/test.tc',
                './extra_data/tw/train.tc',
                './extra_data/tw/valid.tc']

data_list_sc = ['./extra_data/sc/1998.sc',
                './extra_data/sc/sinica_ner.dev',
                './extra_data/sc/sinica_ner.test',
                './extra_data/sc/sinica_ner.train',
                './extra_data/sc/test.sc',
                './extra_data/sc/train.sc',
                './extra_data/sc/valid.sc']

# ---------------  Eval  ---------------
# test file for multiple lines testing
test_file = 'example/test'
batch_size = 32
