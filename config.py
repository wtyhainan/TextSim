import os
from transformers import BertTokenizer, BertModel, BertConfig, AutoTokenizer, AutoModel, AutoConfig

from utils import load_stopword_from_file, mkdir_if_no_exist


BERT_DIR = './premodels/bert-base-chinese'
DATA_DIR = ''
STOPWORD_DIR = './match_data'

Stopwords = load_stopword_from_file(os.path.join(STOPWORD_DIR, 'stop_word.txt'))


class Config(object):
    def __init__(self,
                 model_name=None,
                 model_path=None,
                 seed=48,
                 learning_rate=2e-5,
                 batch_size=256,
                 mini_batch_size=16,
                 epochs=5,
                 dropout_prob=0.5,
                 save_path=None,
                 data_filepath=None,
                 Bi=False       # 是否使用双塔模型
                 ):
        self.set_random_seed(seed)
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dropout_prob = dropout_prob

        self.accumulation_steps = self.batch_size // self.mini_batch_size

        self.Bi = Bi
        self.data_filepath = data_filepath
        self.save_path = save_path
        if model_name is None:
            self.tokenizer = BertTokenizer.from_pretrained(BERT_DIR)
            self.model = BertModel.from_pretrained(BERT_DIR)
            self.model_config = BertConfig.from_pretrained(BERT_DIR)
        else:
            model_path = os.path.join(model_path, model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path)
            self.model_config = AutoConfig.from_pretrained(model_path)
        self.vocab = self.tokenizer.vocab
        if self.data_filepath is None:
            self.data_filepath = 'E:\\nlpDatasets\\TextSimDatasets\\souhu-text-match\\sohu2021_open_data_clean'

        if self.save_path is None:
            mkdir_if_no_exist('./models')
            self.save_path = './models'
        else:
            mkdir_if_no_exist(self.save_path)


    @classmethod
    def load_config_from_file(cls, filepath):
        pass

    @classmethod
    def get_default_config(cls):
        return Config()

    @property
    def max_len(self):
        return self.model_config.max_position_embeddings

    def set_random_seed(self, seed):
        pass


if __name__ == "__main__":
    config = BertConfig.from_pretrained(BERT_DIR)
    help(config)














