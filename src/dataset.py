import json
import random
import numpy as np
import torch
from transformers import BertTokenizer, RobertaTokenizer
from torch.utils.data.dataset import Dataset

UNK, PAD, CLS = '[UNK]', '[PAD]', '[CLS]'  # unknown, padding, classification


class SequenceFeature:
    def __init__(self,
                 seq_token_ids,
                 seq_attention_masks,
                 seq_token_type_ids,
                 co_attention_masks,
                 senti_label,
                 topic_label
                 ):
        # BERT 输入
        self.seq_token_ids = seq_token_ids
        self.seq_attention_masks = seq_attention_masks
        self.seq_token_type_ids = seq_token_type_ids
        self.co_attention_masks = co_attention_masks
        self.senti_label = senti_label
        self.topic_label = topic_label


class TopicFeatures:
    def __init__(self,
                 top_token_ids,
                 top_attention_masks,
                 top_token_type_ids,
                 top_values
                 ):
        # BERT 输入
        self.top_token_ids = top_token_ids
        self.top_attention_masks = top_attention_masks
        self.top_token_type_ids = top_token_type_ids
        self.top_values = top_values


class InputExample:
    def __init__(self,
                 set_type,
                 text,
                 senti_label,
                 topic_label
                 ):
        self.set_type = set_type
        self.text = text
        self.senti_label = senti_label
        self.topic_label = topic_label


class DataProcessor:
    def __init__(self, max_seq_len=256):
        self.max_seq_len = max_seq_len

    @staticmethod
    def read_json(file_path):
        with open(file_path, encoding='utf-8') as f:
            raw_examples = json.load(f)
        return raw_examples

    @staticmethod
    def truncate_sentence(text, max_len):
        if len(text) <= max_len - 2:  # 考虑到需要添加特殊字符 [CLS] 和 [SEP]，需要让最大长度 -2
            return text
        else:
            return text[:max_len - 2]

    def get_examples(self, raw_examples, set_type, noise_rate=0):
        examples = []

        for i, item in enumerate(raw_examples):
            text = self.truncate_sentence(item['text'], self.max_seq_len)  # 对过长的文本进行切分
            senti_label_list = item['senti_label']
            topic_label_list = item['topic_label']
            noise_topic_label_list = []
            for topic_label in topic_label_list:
                if random.random() < noise_rate:
                    topic_label = 1 - topic_label  # 随机修改标签，0换为1，1换为0
                noise_topic_label_list.append(topic_label)

            examples.append(
                InputExample(set_type=set_type,
                             text=text,
                             senti_label=senti_label_list,
                             topic_label=noise_topic_label_list)
            )

        return examples

    def convert_seq_example(self, example: InputExample, tokenizer):  # tokenizer: BertTokenizer
        # set_type = example.set_type
        raw_text = example.text
        senti_label = example.senti_label
        topic_label = example.topic_label

        callback_info = (raw_text, senti_label, topic_label)

        context_tokens = tokenizer.tokenize(raw_text)
        # 在此仍需要判断是否超过最大长度，因为 tokenize 可能会对英文单词进行切分，导致序列长度增大
        if len(context_tokens) > self.max_seq_len - 2:
            context_tokens = context_tokens[:self.max_seq_len - 2]

        seq_token_ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + context_tokens + ["[SEP]"])
        seq_attention_masks = [1] * len(seq_token_ids)
        seq_token_type_ids = [0] * len(seq_token_ids)
        # 共注意力的掩码，除句子中的常规字符为0外，其余全部设置为1(包括[CLS]和[SEP])
        co_attention_masks = [0] * len(seq_token_ids)
        co_attention_masks[0], co_attention_masks[-1] = 1, 1

        padding_len = self.max_seq_len - len(seq_token_ids)
        seq_token_ids += ([0] * padding_len)
        seq_attention_masks += ([0] * padding_len)
        seq_token_type_ids += ([0] * padding_len)
        co_attention_masks += ([1] * padding_len)

        feature = SequenceFeature(
            # bert inputs
            seq_token_ids=seq_token_ids,
            seq_attention_masks=seq_attention_masks,
            seq_token_type_ids=seq_token_type_ids,
            co_attention_masks=co_attention_masks,
            senti_label=senti_label,
            topic_label=topic_label
        )

        return feature, callback_info

    def convert_review_sequences_to_features(self, examples, bert_path, mode):

        features = []
        if 'roberta_en' in bert_path:
            tokenizer = RobertaTokenizer.from_pretrained(bert_path)
        else:
            tokenizer = BertTokenizer.from_pretrained(bert_path)

        print(f'Convert {len(examples)} {mode} examples to features')

        for i, example in enumerate(examples):
            feature, tmp_callback = self.convert_seq_example(
                example=example,
                tokenizer=tokenizer
            )

            if feature is None:
                continue

            features.append(feature)

        print(f'Build {len(features)} {mode} features')

        return features


class DataLdaProcessor:
    def __init__(self, max_seq_len=256):
        self.max_seq_len = max_seq_len

    @staticmethod
    def read_json(file_path):
        with open(file_path, encoding='utf-8') as f:
            raw_examples = json.load(f)
        return raw_examples

    @staticmethod
    def truncate_sentence(text, max_len):
        if len(text) <= max_len - 2:  # 考虑到需要添加特殊字符 [CLS] 和 [SEP]，需要让最大长度 -2
            return text
        else:
            return text[:max_len - 2]

    @staticmethod
    def transform_label(prob_data):
        new_label_list = []
        for j in range(len(prob_data)):
            if prob_data[j] > 1:
                new_label_list.append(1)
            else:
                new_label_list.append(0)
        return new_label_list

    def get_examples(self, raw_examples, data_set, set_type, topic_num):
        examples = []

        lda_data = np.load('../lda/{}_{}_probs_tpnum{}.npy'.format(data_set[:4], set_type, topic_num))

        for i, item in enumerate(raw_examples):
            text = self.truncate_sentence(item['text'], self.max_seq_len)  # 对过长的文本进行切分
            senti_label_list = item['senti_label']
            topic_label_list = self.transform_label(lda_data[i])  # 句子在各个topic下 lda归一化概率值

            examples.append(
                InputExample(set_type=set_type,
                             text=text,
                             senti_label=senti_label_list,
                             topic_label=topic_label_list)
            )

        return examples

    def convert_seq_example(self, example: InputExample, tokenizer):  # tokenizer: BertTokenizer
        # set_type = example.set_type
        raw_text = example.text
        senti_label = example.senti_label
        topic_label = example.topic_label

        callback_info = (raw_text, senti_label, topic_label)

        context_tokens = tokenizer.tokenize(raw_text)
        if len(context_tokens) > self.max_seq_len - 2:
            context_tokens = context_tokens[:self.max_seq_len - 2]

        seq_token_ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + context_tokens + ["[SEP]"])
        seq_attention_masks = [1] * len(seq_token_ids)
        seq_token_type_ids = [0] * len(seq_token_ids)
        co_attention_masks = [0] * len(seq_token_ids)
        co_attention_masks[0], co_attention_masks[-1] = 1, 1

        padding_len = self.max_seq_len - len(seq_token_ids)
        seq_token_ids += ([0] * padding_len)
        seq_attention_masks += ([0] * padding_len)
        seq_token_type_ids += ([0] * padding_len)
        co_attention_masks += ([1] * padding_len)

        feature = SequenceFeature(
            # bert inputs
            seq_token_ids=seq_token_ids,
            seq_attention_masks=seq_attention_masks,
            seq_token_type_ids=seq_token_type_ids,
            co_attention_masks=co_attention_masks,
            senti_label=senti_label,
            topic_label=topic_label
        )

        return feature, callback_info

    def convert_review_sequences_to_features(self, examples, bert_path, mode):

        features = []
        if 'roberta_en' in bert_path:
            tokenizer = RobertaTokenizer.from_pretrained(bert_path)
        else:
            tokenizer = BertTokenizer.from_pretrained(bert_path)

        print(f'Convert {len(examples)} {mode} examples to features')

        for i, example in enumerate(examples):
            feature, tmp_callback = self.convert_seq_example(
                example=example,
                tokenizer=tokenizer
            )

            if feature is None:
                continue

            features.append(feature)

        print(f'Build {len(features)} {mode} features')

        return features

    def convert_topics_to_features(self, device, bert_path, data_set, topic_num):

        if 'roberta_en' in bert_path:
            tokenizer = RobertaTokenizer.from_pretrained(bert_path)
        else:
            tokenizer = BertTokenizer.from_pretrained(bert_path)

        top_token_ids = []
        top_attention_masks = []
        top_token_type_ids = []

        topic_path = '../lda/{}_topics_tpnum{}.json'.format(data_set[:4], topic_num)
        topic_name_list = []
        topic_value_list = []

        with open(topic_path, 'r', encoding='utf-8') as f:
            topic_json = json.load(f)
        for item in topic_json:
            topic_name_list.extend(item.keys())
            topic_value_list.extend(item.values())

        for topic in topic_name_list:
            context_tokens = tokenizer.tokenize(topic)
            if len(context_tokens) > self.max_seq_len - 2:
                context_tokens = context_tokens[:self.max_seq_len - 2]

            seq_token_ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + context_tokens + ["[SEP]"])
            seq_attention_masks = [1] * len(seq_token_ids)
            seq_token_type_ids = [0] * len(seq_token_ids)

            padding_len = self.max_seq_len - len(seq_token_ids)
            seq_token_ids += ([0] * padding_len)
            seq_attention_masks += ([0] * padding_len)
            seq_token_type_ids += ([0] * padding_len)

            top_token_ids.append(seq_token_ids)
            top_attention_masks.append(seq_attention_masks)
            top_token_type_ids.append(seq_token_type_ids)

        print('Word num of LDA:', len(top_token_ids))

        features = TopicFeatures(
            # bert inputs
            top_token_ids=torch.tensor(top_token_ids).long().to(device),
            top_attention_masks=torch.tensor(top_attention_masks).long().to(device),
            top_token_type_ids=torch.tensor(top_token_type_ids).long().to(device),
            top_values=torch.tensor(topic_value_list).float().to(device)
        )

        return features


class SentiDataset(Dataset):
    # 适用于TescaBERT的数据集
    def __init__(self, device, features):

        self.nums = len(features)

        self.seq_token_ids = [torch.tensor(example.seq_token_ids).long().to(device) for example in features]
        self.seq_attention_masks = [torch.tensor(example.seq_attention_masks).long().to(device) for example in features]
        self.seq_token_type_ids = [torch.tensor(example.seq_token_type_ids).long().to(device) for example in features]
        self.co_attention_masks = [torch.tensor(example.co_attention_masks).bool().to(device) for example in features]
        self.senti_labels = [torch.tensor(example.senti_label).long().to(device) for example in features]
        self.topic_labels = [torch.tensor(example.topic_label).float().to(device) for example in features]

    def __len__(self):
        return self.nums

    def __getitem__(self, index):

        data = {
            'seq_token_ids': self.seq_token_ids[index],
            'seq_attention_masks': self.seq_attention_masks[index],
            'seq_token_type_ids': self.seq_token_type_ids[index],
            'co_attention_masks': self.co_attention_masks[index],
            'senti_labels': self.senti_labels[index],
            'topic_labels': self.topic_labels[index]
        }

        return data
