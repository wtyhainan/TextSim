import argparse
from collections import Counter

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from model import CrossEncodeClassifier, BiEncoderClassifier
from dataloader import get_dataloader
from utils import tensorized
from config import Config


parse = argparse.ArgumentParser()
parse.add_argument('--model_name', default='bert-base-chinese')
parse.add_argument('--model_path', default='./premodels')
# parse.add_argument('--data_path', default='E:\\nlpDatasets\\TextSimDatasets\\souhu-text-match\\sohu2021_open_data_clean')
parse.add_argument('--data_path', default='/home/datasets/souhu-text-match/sohu2021_open_data_clean')
parse.add_argument('--learning_rate', default=2e-5)
parse.add_argument('--epochs', default=1)
parse.add_argument('--batch_size', default=32)
parse.add_argument('--dropout_prob', default=0.5)
parse.add_argument('--save_path', default='./models')
parse.add_argument('--cuda', default=True)
parse.add_argument('--mini_batch_size', default=4)


args = parse.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(model, train_dataloader, valid_dataloader, conf):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam([
        {'params': model.encoder.parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': args.learning_rate}
    ], weight_decay=2e-3)

    model.train()
    num_batch = 0
    model.to(device)

    for epoch in tqdm(range(conf.epochs)):
        model.train()
        # train
        for batch in tqdm(train_dataloader):
            if conf.Bi:
                print('BI model')
                query_ids, query_attention_mask = tensorized(batch[:, 0], conf.vocab)
                target_ids, target_attention_mask = tensorized(batch[:, 1], conf.vocab)
                label = torch.tensor(list(batch[:, 2]))

                query_ids, query_attention_mask = query_ids.to(device), query_attention_mask.to(device)
                target_ids, target_attention_mask = target_ids.to(device), target_attention_mask.to(device)
                label = label.to(device)

                output = model(query_ids, target_ids, query_attention_mask, target_attention_mask)
            else:
                print('Cross Model')
                input_ids, attention_mask = tensorized(batch[:, 0], conf.vocab)
                label = torch.tensor(list(batch[:, 1]))
                input_ids, attention_mask, label = input_ids.to(device), attention_mask.to(device), label.to(device)
                output = model(input_ids, attention_mask)

            l = loss_fn(output, label)
            l.backward()
            num_batch += 1
            if num_batch % conf.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                break

        # eval
        model.eval()
        preds, labels = [], []
        loss = 0
        for batch in tqdm(valid_dataloader):
            if conf.Bi:
                query_ids, query_attention_mask = tensorized(batch[:, 0], conf.vocab)
                target_ids, target_attention_mask = tensorized(batch[:, 1], conf.vocab)
                label = torch.tensor(list(batch[:, 2]))

                query_ids, query_attention_mask = query_ids.to(device), query_attention_mask.to(device)
                target_ids, target_attention_mask = target_ids.to(device), target_attention_mask.to(device)
                label = label.to(device)

                output = model(query_ids, target_ids, query_attention_mask, target_attention_mask)
            else:
                input_ids, attention_mask = tensorized(batch[:, 0], conf.vocab)
                label = torch.tensor(list(batch[:, 1]))
                input_ids, attention_mask, label = input_ids.to(device), attention_mask.to(device), label.to(device)
                output = model(input_ids, attention_mask)

            l = loss_fn(output, label)
            loss += l.item()
            pred = torch.argmax(output, dim=1).data.cpu().numpy()
            label = label.data.cpu().numpy()
            preds = np.concatenate((preds, pred))
            labels = np.concatenate((labels, label))
            break

        acc = accuracy_score(labels, preds)
        pre = precision_score(labels, preds)
        rec = recall_score(labels, preds)
        f1 = f1_score(labels, preds)
        print(f'batch: {num_batch}, loss: {loss}, acc: {acc}, pre: {pre}, rec: {rec}, f1: {f1}')


if __name__ == '__main__':

    conf = Config(model_name=args.model_name,
                  model_path=args.model_path,
                  seed=48,
                  learning_rate=args.learning_rate,
                  batch_size=args.batch_size,
                  mini_batch_size=args.mini_batch_size,
                  epochs=args.epochs,
                  dropout_prob=args.dropout_prob,
                  save_path=args.save_path,
                  data_filepath=args.data_path,
                  Bi=True)
    # print(conf)
    train_dataloader, valid_dataloader = get_dataloader(config=conf)

    # classifier = CrossEncodeClassifier(conf)

    if conf.Bi:
        classifier = BiEncoderClassifier(conf)
    else:
        classifier = CrossEncodeClassifier(conf)

    train(classifier, train_dataloader=train_dataloader, valid_dataloader=valid_dataloader, conf=conf)








