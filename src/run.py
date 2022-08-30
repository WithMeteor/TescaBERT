import argparse
import json
import os
import random
import numpy as np
import torch
import logging
from torch.utils.data import DataLoader

from dataset import DataProcessor, DataLdaProcessor
from dataset import SentiDataset

from model_tesca import TescaBERT
from train_eval_tesca import train_tesca, test_tesca

from model_tesca_lda import TescaBERTLda
from train_eval_tesca_lda import train_tesca_lda, test_tesca_lda
from lda_clustering import cluster_data

# seed = 918
seed = random.randint(1, 1000)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# seed = 'no random seed'


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='The path of configuration file')
    parser.add_argument('--noise_rate', type=float, default=0,
                        help='The rate of noise injected into data')
    parser.add_argument('--inject_train', action='store_true',
                        help='Whether inject noise into train data')
    parser.add_argument('--loss_ratio', type=float, default=0.3,
                        help='The rate of topic loss to senti loss in stage two')
    parser.add_argument('--change_loss', action='store_true',
                        help='Whether change default loss ratio')
    parser.add_argument('--topic_num', type=float, default=5,
                        help='The num of topic in lda')
    parser.add_argument('--lda_model', action='store_true',
                        help='Whether use Tescabert-lda')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        new_args = json.load(f)
    new_args['noise_rate'] = args.noise_rate
    new_args['inject_train'] = args.inject_train
    new_args['loss_ratio'] = args.loss_ratio
    new_args['change_loss'] = args.change_loss
    new_args['model']['topic_num'] = int(args.topic_num)
    new_args['lda_model'] = args.lda_model
    return new_args


def main():
    print('Seed:', seed)

    args = parse()

    if args['inject_train']:
        args['output_path'] = os.path.join(args['output_path'],
                                           'noise_{}'.format(args['noise_rate']))
    elif args['change_loss']:
        args['output_path'] = os.path.join(args['output_path'],
                                           'loss_rate_{}'.format(args['loss_ratio']))
    elif args['lda_model']:
        args['output_path'] = os.path.join(args['output_path'],
                                           'topic_num_{}'.format(args['model']['topic_num']))
    else:
        args['output_path'] = os.path.join(args['output_path'],
                                           'try')
    if not os.path.exists(args['output_path']):
        os.mkdir(args['output_path'])
    model_path = os.path.join(args['output_path'], 'model.pkl')

    if args['use_gpu']:
        torch.cuda.set_device(args['gpu_id'])
        device = torch.device('cuda:{}'.format(args['gpu_id']) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    class_list = [x.strip()
                  for x in open('../data/{}/class.txt'.format(args['data_set']), encoding='utf8').readlines()]
    args['model']['class_num'] = len(class_list)
    args['model']['data_set'] = args['data_set']

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s  %(filename)s : %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename='{}/run.log'.format(args['output_path']),
                        filemode='w')

    print(args)
    logging.info(args)

    if args['type'] == "TescaBERT":
        net = TescaBERT(args['model'])
        mode = 'tesca'
    elif args['type'] == "TescaBERTLda":
        net = TescaBERTLda(args['model'])
        mode = 'tesca_lda'
    else:
        raise ValueError

    net.to(device)

    if args['load'] == 1:
        net.load_state_dict(torch.load(model_path))

    train_path = '../data/{}/train.txt'.format(args['data_set'])
    dev_path = '../data/{}/dev.txt'.format(args['data_set'])
    bert_path = args['model']['bert_dir']
    dev_data_loader = None
    topic_features = None

    if args['num_epochs'] > 0:
        if mode == 'tesca':
            processor = DataProcessor(max_seq_len=args['padding_len'])
            train_raw_examples = processor.read_json(train_path.replace('train.txt', 'train.json'))
            if args['inject_train']:
                train_examples = processor.get_examples(train_raw_examples, 'train', noise_rate=args['noise_rate'])
            else:
                train_examples = processor.get_examples(train_raw_examples, 'train')
            train_features = processor.convert_review_sequences_to_features(train_examples, bert_path=bert_path,
                                                                            mode='train')
            train_set = SentiDataset(device, train_features)

            dev_raw_examples = processor.read_json(dev_path.replace('dev.txt', 'dev.json'))
            dev_examples = processor.get_examples(dev_raw_examples, 'dev', noise_rate=args['noise_rate'])
            dev_features = processor.convert_review_sequences_to_features(dev_examples, bert_path=bert_path, mode='dev')

            dev_set = SentiDataset(device, dev_features)

        else:  # mode == 'tesca_lda':
            topic_num = args['model']['topic_num']
            topic_word_num = args['model']['topic_word_num']
            log_path = '{}/run.log'.format(args['output_path'])
            lda_data_path = '../lda/{}_topics_tpnum{}.json'.format(args['data_set'][:4], topic_num)

            if not os.path.exists(lda_data_path):
                # 首先对数据进行聚类
                cluster_data(log_path=log_path, dataset=args['data_set'], topic_num=topic_num, word_num=topic_word_num)

            processor = DataLdaProcessor(max_seq_len=args['padding_len'])
            train_raw_examples = processor.read_json(train_path.replace('train.txt', 'train.json'))

            train_examples = processor.get_examples(train_raw_examples, args['data_set'], 'train', topic_num)
            train_features = processor.convert_review_sequences_to_features(train_examples, bert_path=bert_path,
                                                                            mode='train')
            train_set = SentiDataset(device, train_features)

            dev_raw_examples = processor.read_json(dev_path.replace('dev.txt', 'dev.json'))
            dev_examples = processor.get_examples(dev_raw_examples, args['data_set'], 'dev', topic_num)
            dev_features = processor.convert_review_sequences_to_features(dev_examples, bert_path=bert_path, mode='dev')

            dev_set = SentiDataset(device, dev_features)
            topic_features = processor.convert_topics_to_features(device=device, bert_path=bert_path,
                                                                  data_set=args['data_set'], topic_num=topic_num)

        print('Train data size:', len(train_set))
        print('Dev data size:', len(dev_set))
        train_data_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True)
        dev_data_loader = DataLoader(dev_set, batch_size=args['batch_size'], shuffle=False)

        if mode == 'tesca':
            best_net = train_tesca(args, net, train_data_loader, dev_data_loader)
        else:  # mode == 'tesca_lda':
            best_net = train_tesca_lda(args, net, train_data_loader, dev_data_loader, topic_features)
        torch.save(best_net, model_path)
        net.load_state_dict(best_net)
    print(f"Model: {args['type']}")

    if mode == 'tesca':
        test_tesca(args, net, dev_data_loader, class_list)
    else:  # mode == 'tesca_lda':
        test_tesca_lda(args, net, dev_data_loader, topic_features, class_list)

    print('Seed:', seed)
    logging.info('Seed: {}'.format(seed))


if __name__ == '__main__':
    main()
