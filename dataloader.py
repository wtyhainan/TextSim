from typing import List, Tuple

import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader

from config import Stopwords
from utils import read_data
from config import Config
from utils import tensorized
from model import CrossEncodeClassifier


jieba.initialize()


class TextSimDataset(Dataset):
    def __init__(self,
                 datas: np.ndarray,
                 conf: Config,
                 up_sample=False):
        self.conf = conf
        self.datas = self.process(datas)        # 去除停用词
        self.tokenizer = conf.tokenizer
        self.vocab = conf.tokenizer.vocab
        self.up_sample = up_sample

    @staticmethod
    def delete_stop_word(tokens: List[str]):
        token_list = [item for item in tokens if item not in Stopwords]
        return ''.join(token_list)

    @staticmethod
    def clean(content: str) ->str:
        content = content.replace('\n', '')
        content = content.replace(' ', '')
        return content

    def process(self, datas: np.ndarray, up_sample=False):
        datas[:, 1] = [self.delete_stop_word(jieba.lcut(text)) for text in datas[:, 1]]
        datas[:, 2] = [self.delete_stop_word(jieba.lcut(text)) for text in datas[:, 2]]
        if up_sample:
            pass

        return datas

    def get_sentence_by_window(self, s: List[str], k: int, max_len: int):
        l = len(s)
        if l < max_len:
            return s
        window = max_len // k
        t_point = l // k
        res = []
        i = 0
        while i < l:
            res += s[i: i + window]
            i += t_point
        return ''.join(res)


class CrossTextSimDataset(TextSimDataset):
    def __init__(self,
                 datas: np.ndarray,
                 conf: Config,
                 up_sample=False):

        super(CrossTextSimDataset, self).__init__(datas, conf, up_sample)

    def truncate_sentence(self, s1: str, s2: str) -> str:
        s1, s2 = self.clean(s1), self.clean(s2)
        l1 = len(s1)
        l2 = len(s2)
        k = 3
        max_len = self.conf.max_len - 10
        half_len = max_len // 2
        if l1 >= half_len and l2 >= half_len:
            s1, s2 = self.get_sentence_by_window(list(s1), k=k, max_len=half_len), self.get_sentence_by_window(list(s2), k=k, max_len=half_len)
        elif l1 <= half_len and l2 >= half_len:
            s2 = s2[:max_len - l1]
        elif l1 >= half_len and l2 <= half_len:
            s1 = s1[:max_len - l2]
        else:
            pass
        exchange_prob = np.random.uniform(0, 1)
        if exchange_prob >= 0.5:
            return "[CLS]" + s2 + "[SEP]" + s1
        return "[CLS]" + s1 + "[SEP]" + s2

    def __getitem__(self, index: int) ->Tuple:
        data = self.datas[index]
        typ = int(data[0])
        label = int(data[3])
        text = self.truncate_sentence(data[1], data[2])
        tokens = self.tokenizer.tokenize(text)
        tokens_id = self.tokenizer.convert_tokens_to_ids(tokens)
        return tokens_id, label, typ

    def __len__(self):
        return len(self.datas)


class BiTextSimDataset(TextSimDataset):
    def __init__(self,
                 datas: np.ndarray,
                 conf: Config,
                 up_sample=False):
        super(BiTextSimDataset, self).__init__(datas, conf, up_sample)

    def truncate_sentence(self, s1: str, s2: str) -> Tuple[str]:
        s1, s2 = self.clean(s1), self.clean(s2)
        l1 = len(s1)
        l2 = len(s2)
        k = 3
        max_len = self.conf.max_len - 10

        if l1 > max_len:
            s1 = self.get_sentence_by_window(list(s1), k=k, max_len=max_len)
        if l2 > max_len:
            s2 = self.get_sentence_by_window(list(s2), k=k, max_len=max_len)
        return s1, s2

    def __getitem__(self, index: int) -> Tuple:
        data = self.datas[index]
        typ = int(data[0])
        label = int(data[3])
        query = data[1]
        target = data[2]
        query, target = self.truncate_sentence(query, target)
        query_tokens = self.tokenizer.tokenize(query)
        target_tokens = self.tokenizer.tokenize(target)
        query_ids = self.tokenizer.convert_tokens_to_ids(query_tokens)
        target_ids = self.tokenizer.convert_tokens_to_ids(target_tokens)
        return query_ids, target_ids, label, typ

    def __len__(self):
        return len(self.datas)


def get_dataloader(config):
    train_data, valid_data = read_data(config.data_filepath)
    if config.Bi:
        print(f'BI dataset')
        train_dataset = BiTextSimDataset(train_data.values, config)
        valid_dataset = BiTextSimDataset(valid_data.values, config)
    else:
        print(f'corss dataset')
        train_dataset = CrossTextSimDataset(train_data.values, config)
        valid_dataset = CrossTextSimDataset(valid_data.values, config)

    train_dataloader = DataLoader(train_dataset, batch_size=config.mini_batch_size, shuffle=True, collate_fn=lambda batch: np.array(batch, dtype=object))
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.mini_batch_size, collate_fn=lambda batch: np.array(batch, dtype=object))
    return train_dataloader, valid_dataloader


if __name__ == '__main__':
    from config import Config
    # from collections import Counter
    # from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    # import pandas as pd
    # from sklearn.model_selection import train_test_split

    conf = Config.get_default_config()
    conf.Bi = True
    train_dataloader, valid_dataloader = get_dataloader(conf)

    for i in train_dataloader:
        pass





