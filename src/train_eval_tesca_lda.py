import time
import os
from copy import deepcopy
import logging
import numpy as np
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, f1_score

from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter


class Stat:
    def __init__(self, training, writer=None):
        self.step = 0
        self.loss = []
        self.all_senti_labels = []
        self.all_senti_pred = []
        self.training = training
        self.writer = writer

    def add(self, senti_logits, senti_labels, loss):
        senti_labels = senti_labels.cpu().numpy()
        senti_logits = senti_logits.cpu().detach().numpy()
        senti_pred = np.argmax(senti_logits, axis=1)

        self.loss.append(loss)
        self.all_senti_labels.extend(senti_labels)
        self.all_senti_pred.extend(senti_pred)

    def log(self):
        self.step += 1
        senti_acc = accuracy_score(self.all_senti_labels, self.all_senti_pred)
        senti_f1 = f1_score(self.all_senti_labels, self.all_senti_pred, average='macro')
        senti_kappa = cohen_kappa_score(self.all_senti_labels, self.all_senti_pred)
        loss = sum(self.loss) / len(self.loss)
        self.loss = []
        self.all_senti_labels = []
        self.all_senti_pred = []

        if not self.writer:
            return loss, senti_acc, senti_f1, senti_kappa
        if self.training:
            self.writer.add_scalar('train_Loss', loss, self.step)
            self.writer.add_scalar('train_Accuracy', senti_acc, self.step)
            self.writer.add_scalar('train_F1Score', senti_f1, self.step)
            self.writer.add_scalar('train_Kappa', senti_kappa, self.step)
        else:
            self.writer.add_scalar('dev_Loss', loss, self.step)
            self.writer.add_scalar('dev_Accuracy', senti_acc, self.step)
            self.writer.add_scalar('dev_F1Score', senti_f1, self.step)
            self.writer.add_scalar('dev_Kappa', senti_kappa, self.step)
        return loss, senti_acc, senti_f1, senti_kappa


class StatAtt:
    def __init__(self, output_dir):
        self.seq_atten_all = []
        self.top_atten_all = []
        self.seq_attention_file = os.path.join(output_dir, "seq_attention.npy")
        self.top_attention_file = os.path.join(output_dir, "top_attention.npy")

    def add(self, seq_atten, top_atten):
        seq_atten = seq_atten.detach().cpu().numpy()
        top_atten = top_atten.detach().cpu().numpy()

        self.seq_atten_all.extend(seq_atten)
        self.top_atten_all.extend(top_atten)

    def save(self):
        np.save(self.seq_attention_file, self.seq_atten_all)
        np.save(self.top_attention_file, self.top_atten_all)

        self.seq_atten_all = []
        self.top_atten_all = []


def train_tesca_lda(args, model, train_data_loader, dev_data_loader, topic_features):

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s  %(filename)s : %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename='{}/run.log'.format(args['output_path']),
                        filemode='a')

    total_steps = len(train_data_loader) * args['num_epochs']

    if "BERT" in args['type'] or "ROBERTA" in args['type']:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': args['weight_decay'], 'lr': args['lr']},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': args['lr']}
        ]
        optimizer = AdamW(optimizer_grouped_parameters)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.05 * total_steps),
            num_training_steps=total_steps
        )
    else:
        optimizer = Adam(model.parameters(), args['lr'], weight_decay=args['weight_decay'])
        scheduler = None

    writer = SummaryWriter(args['output_path'] + '/' +
                           time.strftime('%m-%d_%H.%M', time.localtime()))
    train_stat = Stat(training=True, writer=writer)
    dev_stat = Stat(training=False, writer=writer)
    atten_stat = StatAtt(output_dir=args['output_path'])

    best_kappa, best_net = 0, None

    for epoch in range(args['num_epochs']):
        print(f"--- epoch: {epoch + 1} ---")
        logging.info("--- epoch: {} ---".format(epoch + 1))

        topic_embeds = None  # 缓存静态的主题嵌入

        for iteration, train_batch in enumerate(train_data_loader):
            model.train()
            senti_labels = train_batch['senti_labels']
            train_batch['topic_features'] = topic_features
            train_batch['topic_embeds'] = topic_embeds
            optimizer.zero_grad()
            # pred_outputs = model(inputs)

            senti_loss, senti_logits, seq_atten, topic_atten, topic_embeds = model(**train_batch)

            senti_loss.backward()
            optimizer.step()
            scheduler.step()
            train_stat.add(senti_logits, senti_labels, senti_loss.item())

            if (iteration + 1) % args['display_per_batch'] == 0:
                train_loss, train_acc, train_f1, train_kappa = train_stat.log()

                model.eval()
                with torch.no_grad():
                    for dev_batch in dev_data_loader:
                        senti_labels = dev_batch['senti_labels']
                        dev_batch['topic_features'] = topic_features
                        dev_batch['topic_embeds'] = topic_embeds

                        senti_loss, senti_logits, seq_atten, topic_atten, topic_embeds = model(**dev_batch)

                        dev_stat.add(senti_logits, senti_labels, senti_loss.item())
                        atten_stat.add(seq_atten, topic_atten)

                dev_loss, dev_acc, dev_f1, dev_kappa = dev_stat.log()
                print(f"step {(iteration + 1):5}, "
                      f"training loss: {train_loss:.4f}, acc: {train_acc:.2%}, f1: {train_f1:.2%}, kappa: {train_kappa:.2%} "
                      f"dev loss: {dev_loss:.4f}, acc: {dev_acc:.2%}, f1: {dev_f1:.2%}, kappa: {dev_kappa:.2%}.")
                logging.info("step {:5}, "
                             "training loss: {:.4f}, acc: {:.2%}, f1: {:.2%}, kappa: {:.2%} "
                             "dev loss: {:.4f}, acc: {:.2%}, f1: {:.2%}, kappa: {:.2%}."
                             .format(iteration + 1, train_loss, train_acc, train_f1, train_kappa,
                                     dev_loss, dev_acc, dev_f1, dev_kappa))

                if dev_kappa > best_kappa:
                    best_kappa = dev_kappa
                    best_net = deepcopy(model.state_dict())

                    atten_stat.save()

    print(f"best dev kappa: {best_kappa:.4f}")
    logging.info("best dev kappa: {:.2%}".format(best_kappa))
    return best_net


def test_tesca_lda(args, model, test_data_loader, topic_features, class_list):

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s  %(filename)s : %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename='{}/run.log'.format(args['output_path']),
                        filemode='a')

    test_stat = Stat(training=False)

    print("--- testing ---")
    logging.info("--- testing ---")

    model.eval()
    with torch.no_grad():
        topic_embeds = None  # 缓存静态的主题嵌入
        for batch in test_data_loader:
            senti_labels = batch['senti_labels']
            batch['topic_features'] = topic_features
            batch['topic_embeds'] = topic_embeds
            senti_loss, senti_logits, seq_atten, topic_atten, topic_embeds = model(**batch)

            test_stat.add(senti_logits, senti_labels, 0)  # dummy for loss

    report = classification_report(
        test_stat.all_senti_labels,
        test_stat.all_senti_pred,
        target_names=class_list,
        digits=4)

    print(report)
    logging.info(report)
