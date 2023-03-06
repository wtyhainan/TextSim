import torch
import torch.nn as nn


class Pooling(nn.Module):
    def __init__(self):
        super(Pooling, self).__init__()

    def forward(self, token_embedding, attention_mask):
        output_vectors = []
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embedding.size()).float()
        sum_embeddings = torch.sum(token_embedding * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=-1e-9)
        output_vectors.append(sum_embeddings / sum_mask)
        output_vector = torch.cat(output_vectors, 1)
        return output_vector


class CrossEncodeClassifier(nn.Module):
    def __init__(self, config):
        super(CrossEncodeClassifier, self).__init__()
        self.encoder = config.model
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        encode = self.encoder(input_ids, attention_mask)
        hidden_state = encode.last_hidden_state[:, 0]
        return self.classifier(hidden_state)


class BiEncoderClassifier(nn.Module):
    def __init__(self, config):
        super(BiEncoderClassifier, self).__init__()
        self.encoder = config.model
        self.dropout = nn.Dropout(config.dropout_prob)
        self.pool = Pooling()
        self.classifier = nn.Linear(768*3, 2)

    def forward(self, query_input_ids, target_input_ids, query_attention_mask, target_attention_mask):
        query_embedding = self.encoder(query_input_ids, query_attention_mask).last_hidden_state
        target_embedding = self.encoder(target_input_ids, target_attention_mask).last_hidden_state

        # 提取句子表示
        query_pool = self.pool(query_embedding, query_attention_mask)
        target_pool = self.pool(target_embedding, target_attention_mask)

        abs_hidden_state = torch.abs(query_pool - target_pool)
        hidden_state = torch.concatenate([query_pool, target_pool, abs_hidden_state], dim=-1)
        y = self.classifier(hidden_state)
        return y


if __name__ == '__main__':
    import torch

    attention_mask = torch.Tensor([[1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1]])
    embedding = torch.Tensor([[[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6], [0.5, 0.6, 0.7], [0.6, 0.7, 0.8], [0.7, 0.8, 0.9]],
                              [[1.1, 1.2, 1.3], [1.2, 1.3, 1.4], [1.3, 1.4, 1.5], [1.4, 1.5, 1.6], [1.5, 1.6, 1.7], [1.6, 1.7, 1.8], [1.7, 1.8, 1.9]]])

    print(attention_mask.size(), embedding.size())
    attention = attention_mask.unsqueeze(-1).expand(embedding.size()).float()

    embedding_sum = torch.sum(embedding * attention, dim=1)
    z = attention.sum(dim=1)
    print(attention.size())
    print(z.size(), embedding_sum.size())
    res = embedding_sum / z
    print(res.size())

    # from config import Config
    # conf = Config.get_default_config()
    # model = CrossEncodeClassifier(conf)
    #
    # input_ids = torch.ones(size=(2, 32), dtype=torch.int64)
    # attention_mask = torch.ones(size=(2, 32), dtype=torch.int64)
    # x = model(input_ids, attention_mask)









