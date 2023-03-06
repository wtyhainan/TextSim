import os
import json
from typing import Tuple

import pandas as pd
import torch
import numpy as np


variants = [
    u'短短匹配A类',
    # u'短短匹配B类',
    # u'短长匹配A类',
    # u'短长匹配B类',
    # u'长长匹配A类',
    # u'长长匹配B类',
]


def read_data(file_root='E:\\nlpDatasets\\TextSimDatasets\\souhu-text-match\\sohu2021_open_data_clean'):
    train_data = []
    valid_data = []

    for i, var in enumerate(variants):
        key = 'labelA' if 'A' in var else 'labelB'
        train_filepath = os.path.join(os.path.join(file_root, var), 'train.txt')
        valid_filepath = os.path.join(os.path.join(file_root, var), 'valid.txt')

        with open(train_filepath, 'r', encoding='utf-8') as fr:
            for line in fr:
                l = json.loads(line)
                d = {'typ': i, 'source': l['source'], 'target': l['target'], 'label': l[key]}
                train_data.append(d)

        with open(valid_filepath, 'r', encoding='utf-8') as fr:
            for line in fr:
                l = json.loads(line)
                d = {'typ': i, 'source': l['source'], 'target': l['target'], 'label': l[key]}
                valid_data.append(d)
    return pd.DataFrame(train_data), pd.DataFrame(valid_data)


def load_stopword_from_file(filepath):
    stop_words = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n')
            stop_words.append(line)

    return stop_words


def tensorized(batch: list, vocab: dict) -> Tuple:
    PAD = vocab['[PAD]']
    max_len = max([len(example) for example in batch])
    batch_tensor = torch.tensor([item + [PAD for i in range(max_len - len(item))] for item in batch])
    lengths = [len(l) for l in batch]
    mask = torch.ones_like(batch_tensor)

    for index, _ in enumerate(mask):
        mask[index, lengths[index]:] = 0

    return batch_tensor, mask


def mkdir_if_no_exist(dir):
    if os.path.exists(dir):
        return
    os.mkdir(dir)


def jaccard(s1, s2):
    s1 = set(s1)
    s2 = set(s2)
    s3 = s1 | s2
    s4 = s1 & s2
    return len(s4) / len(s3)


def up_sample(train_data: np.ndarray, size=100, score_th=0.2):
    all_text = np.hstack([train_data[:, 1], train_data[:, 2]])
    ext_example = []
    examples = np.random.choice(all_text, size, replace=False)
    for example in examples:
        while True:
            sample = np.random.choice(all_text)
            score = jaccard(example, sample)
            if score < score_th:
                ext_example.append([7, example, sample, '0'])
                break
    return pd.DataFrame(ext_example, columns=['ext', 'source', 'target', 'label']).values


if __name__ == '__main__':

    train_data, valid_data = read_data()


    # from collections import Counter
    # train_data, valid_data = read_data()
    # train_data = train_data.values
    # valid_data = valid_data.values
    # ext_df = up_sample(train_data)
    # print(Counter(train_data[:, -1]))
    # for i in range(100):
    #     p = np.random.uniform(0, 1)
    #     print(p)



