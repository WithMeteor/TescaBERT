import json
import logging
import numpy as np
from gensim import corpora, models
import jieba.posseg as jp

FLAGS = ('n', 'nr', 'ns', 'nt', 'nz', 'eng')  # 词性


def read_text(dataset):
    train_texts = []

    train_path = '../data/{}/train.json'.format(dataset)
    with open(train_path, encoding='utf-8') as f:
        raw_examples = json.load(f)
    train_size = len(raw_examples)
    for exm in raw_examples:
        train_texts.append(exm['text'])

    dev_texts = []

    dev_path = '../data/{}/dev.json'.format(dataset)
    with open(dev_path, encoding='utf-8') as f:
        raw_examples = json.load(f)
    dev_size = len(raw_examples)
    for exm in raw_examples:
        dev_texts.append(exm['text'])

    return train_texts, dev_texts, train_size, dev_size


def get_stopwords(dataset):
    assert dataset in ['bank', 'restaurant', 'sst5', 'food']
    if dataset in ['bank', 'restaurant']:
        stopwords = [x.strip()
                     for x in open('../data/stopwords_zh.txt', encoding='utf8').readlines()]
        stopwords.append('k')
        stopwords.append('w')
        stopwords.append('e')
        stopwords.append('K')
        stopwords.append('W')
        stopwords.append('E')
    else:
        stopwords = [x.strip()
                     for x in open('../data/stopwords_en.txt', encoding='utf8').readlines()]

    return stopwords


def save_data(topic_num, dataset, train_size, dev_size, topics, probs):

    with open('../lda/{}_topics_tpnum{}.json'.format(dataset[:4], topic_num), 'w', encoding='utf-8') as data_file:
        json.dump(topics, data_file, ensure_ascii=False, indent=4)

    print('All size:', len(probs))
    print('Train size:', train_size, 'Dev size:', dev_size)
    np.save('../lda/{}_train_probs_tpnum{}.npy'.format(dataset[:4], topic_num), probs[:train_size])
    np.save('../lda/{}_dev_probs_tpnum{}.npy'.format(dataset[:4], topic_num), probs[train_size:train_size + dev_size])


def cluster_data(log_path, dataset, topic_num, word_num, flags=FLAGS):

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s  %(filename)s : %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=log_path,
                        filemode='a')

    train_texts, dev_texts, train_size, dev_size = read_text(dataset)
    stopwords = get_stopwords(dataset)

    if dataset in ['bank', 'restaurant']:

        train_words_list = []
        for text in train_texts:
            words = [w.word for w in jp.cut(text) if w.flag in flags and w.word not in stopwords]
            train_words_list.append(words)

        dev_words_list = []
        for text in dev_texts:
            words = [w.word for w in jp.cut(text) if w.flag in flags and w.word not in stopwords]
            dev_words_list.append(words)

    else:

        train_words_list = []
        for text in train_texts:
            words = [word for word in text.split() if word not in stopwords]
            train_words_list.append(words)

        dev_words_list = []
        for text in dev_texts:
            words = [word for word in text.split() if word not in stopwords]
            dev_words_list.append(words)

    # print(stopwords)

    topics = []

    # 构造词典
    dictionary = corpora.Dictionary(train_words_list)
    # 基于词典，使【词】→【稀疏向量】，并将向量放入列表，形成【稀疏向量集】
    train_corpus = [dictionary.doc2bow(words) for words in train_words_list]
    # lda模型，topic_num 设置主题的个数
    lda = models.ldamodel.LdaModel(corpus=train_corpus, id2word=dictionary, num_topics=topic_num)
    # 打印所有主题，每个主题显示 word_num 个词
    for topic in lda.print_topics(num_words=word_num):
        topic_dict = dict()
        topic_str = str(topic[1])
        topic_items = [x.strip() for x in topic_str.split('+')]
        values = []
        keys = []
        for item in topic_items:
            value = float(item.split('*')[0])
            key = item.split('*')[1].replace('"', '')
            values.append(value)
            keys.append(key)
        values = np.array(values)
        norm_values = (values / values.sum()).tolist()
        for key, value in zip(keys, norm_values):
            topic_dict[key] = round(value, 4)
        print(topic_dict)
        logging.info(topic_dict)

        topics.append(topic_dict)

    dev_corpus = [dictionary.doc2bow(words) for words in dev_words_list]

    # 主题推断
    probs = lda.inference(train_corpus + dev_corpus)[0]
    print(probs)
    logging.info(probs)
    # 保存主题结果
    save_data(topic_num, dataset, train_size, dev_size, topics, probs)


if __name__ == '__main__':
    DATASET = 'sst5'  # bank / restaurant / mr / sst2 / sst5 / amazon / food
    TOPIC_NUM = 6
    WORD_NUM = 5
    Log_Path = f'../model/model_{DATASET[:4]}/TescaBERTLda/lda.log'
    cluster_data(Log_Path, DATASET, TOPIC_NUM, WORD_NUM)
