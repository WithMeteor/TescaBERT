import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from transformers import BertModel


class CoattentionNet(nn.Module):
    """
    使用交替-共注意力机制，获取文本序列和方面类别的隐藏状态
    """

    def __init__(self, hidden_dim=768, inner_dim=768, dropout_prob=0.5, mask_attention=True):
        super().__init__()

        self.mask_attention = mask_attention

        self.inner_dim = inner_dim
        self.dropout = nn.Dropout(dropout_prob)
        self.tanh = nn.Tanh()

        self.W_x1 = nn.Linear(hidden_dim, inner_dim)
        self.W_h1 = nn.Linear(inner_dim, 1)
        self.W_x2 = nn.Linear(hidden_dim, inner_dim)
        self.W_g2 = nn.Linear(hidden_dim, inner_dim)
        self.W_h2 = nn.Linear(inner_dim, 1)
        self.W_x3 = nn.Linear(hidden_dim, inner_dim)
        self.W_g3 = nn.Linear(hidden_dim, inner_dim)
        self.W_h3 = nn.Linear(inner_dim, 1)

    @staticmethod
    def mask_softmax(data, mask):
        mask_data = data.masked_fill(mask, -1e9)
        return fn.softmax(mask_data, dim=1)

    def forward(self, seq_input, top_input, top_mask, seq_len, top_num, output_attention=False):
        """
        交替共注意力的前向传播
        :param seq_input: 句子的序列输入 (batch_size, seq_len, hidden_dim)
        :param top_input: 方面类别的输入 (batch_size, top_num, hidden_dim)
        :param top_mask: 主题掩码，对模型预测为 0 的主题，计算掩码注意力时，注意力值为 0
        :param seq_len: 句子长度
        :param top_num: 方面类别的数目
        :param output_attention: 是否输出注意力值
        :return:
        """
        # W_x1 : (inner_dim, hidden_dim) , W_h1 : (inner_dim, 1)
        h_x1 = self.W_x1(seq_input)
        h_g1 = torch.zeros_like(h_x1)
        h1 = self.dropout(self.tanh(h_x1 + h_g1))  # H1 : (batch_size, seq_len, inner_dim)
        a1 = self.W_h1(h1).view(-1, seq_len)  # a1 : (batch_size, seq_len)
        a1 = fn.softmax(a1, dim=1).unsqueeze(-1)  # a1 : (batch_size, seq_len, 1)
        # a1 = mask_softmax(a1, seq_mask).unsqueeze(-1)  # a1 : (batch_size, seq_len, 1)
        g1 = a1 * seq_input  # g1 : (batch_size, seq_len, hidden_dim)
        g1 = torch.sum(g1, dim=1)  # g1 :(batch_size, hidden_dim)

        # W_x2 : (hidden_dim, inner_dim) , W_g2 : (hidden_dim, inner_dim) , W_h2 : (inner_dim, 1)
        h_x2 = self.W_x2(top_input)
        h_g2 = self.W_g2(g1).unsqueeze(1).expand(-1, top_num, self.inner_dim)
        h2 = self.dropout(self.tanh(h_x2 + h_g2))  # H2 : (batch_size, top_num, inner_dim)
        a2 = self.W_h2(h2).view(-1, top_num)  # a2 : (batch_size, top_num)
        if self.mask_attention:
            top_mask = (1 - top_mask).bool()
            a2 = self.mask_softmax(a2, top_mask).unsqueeze(-1)  # a2 : (batch_size, top_num, 1)
        else:
            a2 = fn.softmax(a2, dim=1).unsqueeze(-1)  # a2 : (batch_size, top_num, 1)
        g2 = a2 * top_input  # g2 : (batch_size, top_num, hidden_dim)
        g2 = torch.sum(g2, dim=1)  # g2 : (batch_size, hidden_dim)

        # W_x3 : (hidden_dim, inner_dim) , W_g3 : (hidden_dim, inner_dim) , W_h3 : (inner_dim, 1)
        h_x3 = self.W_x3(seq_input)
        h_g3 = self.W_g3(g2).unsqueeze(1).expand(-1, seq_len, self.inner_dim)
        h3 = self.dropout(self.tanh(h_x3 + h_g3))  # H3 : (batch_size, seq_len, inner_dim)
        a3 = self.W_h3(h3).view(-1, seq_len)  # a3 : (batch_size, seq_len)
        a3 = fn.softmax(a3, dim=1).unsqueeze(-1)  # a3 : (batch_size, seq_len, 1)
        # a3 = mask_softmax(a3, seq_mask).unsqueeze(-1)
        g3 = a3 * seq_input  # g3 : (batch_size, seq_len, hidden_dim)
        g3 = torch.sum(g3, dim=1)  # g3 : (batch_size, hidden_dim)

        if output_attention:
            return g3, g2, a3.squeeze(), a2.squeeze()  # seq_hidden, topic_hidden, seq_attention, topic_attention
        else:
            return g3, g2, None, None  # seq_hidden, topic_hidden


class TescaBERT(nn.Module):
    def __init__(self, args):
        super(TescaBERT, self).__init__()

        self.bert = BertModel.from_pretrained(args['bert_dir'])

        self.switch = 0
        self.output_attention = args['output_attention']

        self.topic_num = args['topic_num']  # topic_num 为主题类别个数

        assert args['gru_hidden_size'] * 2 == args['hidden_size']

        self.grus = []
        self.classifiers = []
        for i in range(self.topic_num):
            gru = nn.GRU(args['hidden_size'], args['gru_hidden_size'],
                         num_layers=args['gru_layer_num'], bidirectional=True, batch_first=True)
            topic_classifier = nn.Linear(args['gru_hidden_size'] * 2, 1)

            self.grus.append(gru)
            self.classifiers.append(topic_classifier)

        self.grus = nn.ModuleList(self.grus)
        self.classifiers = nn.ModuleList(self.classifiers)

        self.coattention = CoattentionNet(args['hidden_size'], args['coattention_inner_size'],
                                          dropout_prob=args['dropout'], mask_attention=args['mask_attention'])

        # self.senti_classifier = nn.Linear(args['hidden_size'] * 2, class_num)  # senti_num 为情感标签个数
        self.senti_classifier = nn.Linear(args['hidden_size'], args['class_num'])  # senti_num 为情感标签个数

        self.senti_loss_fct = CrossEntropyLoss()
        self.topic_loss_fct = BCEWithLogitsLoss()

        # init_blocks = [self.coattention, self.senti_classifier, self.topic0_classifier, self.topic1_classifier,
        #                self.topic2_classifier, self.topic3_classifier, self.topic4_classifier]
        # self._init_weights(init_blocks)

    @staticmethod
    def _init_weights(blocks):
        """
        初始化模型参数
        对 GRU 模型的权重进行正交初始化
        """
        for block in blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.Embedding):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)

    def forward(self,
                seq_token_ids,
                seq_attention_masks,
                seq_token_type_ids,
                co_attention_masks,
                senti_labels,
                topic_labels
                ):

        seq_bert_output = self.bert(
            input_ids=seq_token_ids,
            attention_mask=seq_attention_masks,
            token_type_ids=seq_token_type_ids
        )

        seq_embeds = seq_bert_output[0]  # (batch_size, seq_len, hidden_dim)

        batch_size = seq_embeds.shape[0]

        # 使用cls作为主题分类的输入(只用CLS)
        gru_input = seq_embeds[:, 0].reshape(batch_size, 1, -1).contiguous()

        # 使用seq作为主题分类的输入(掩盖掉SEP和PAD)
        # gru_input = seq_embeds.masked_fill(
        #     co_attention_masks.unsqueeze(2), 0)[:, 1:, :].contiguous()  # (batch_size, seq_len-1, hidden_size)
        # gru_input[:, 0] = cls_input[:, 0]

        top_hidden_list = []
        top_logits_list = []
        for i in range(self.topic_num):
            top_output, top_hidden = self.grus[i](gru_input)
            top_hidden = top_hidden.permute(1, 0, 2).reshape(batch_size, -1)  # (batch_size, num_dire * gru_hidden_size)
            top_logits = self.classifiers[i](top_hidden).squeeze()

            top_hidden_list.append(top_hidden)
            top_logits_list.append(top_logits)

        topic_logits = torch.stack(top_logits_list, dim=1)
        topic_pred = torch.round(torch.sigmoid(topic_logits))

        topic_loss = self.topic_loss_fct(topic_logits, topic_labels)

        # gru输出的隐藏状态做拼接，作为共注意力器的输入
        top_hidden = torch.stack(top_hidden_list, dim=1)  # (batch_size, top_num, hidden_size)

        # 在序列输入共注意力器之前，对除正常字符外的所有内容进行掩码(包括[CLS], [SEP], [PAD])
        seq_embeds.masked_fill_(co_attention_masks.unsqueeze(2), 0)

        seq_input = seq_embeds[:, 1:, :].contiguous()  # (batch_size, seq_len, hidden_dim)
        top_input = top_hidden.detach().contiguous()  # (batch_size, top_num, hidden_dim)

        seq_len = seq_input.shape[1]
        top_num = top_input.shape[1]

        if self.switch == 1:
            # seq_hidden, new_top_hidden, seq_atten, topic_atten = self.coattention(
            #     seq_input, top_input, topic_labels, seq_len, top_num, self.output_attention)

            seq_hidden, new_top_hidden, seq_atten, topic_atten = self.coattention(
                seq_input, top_input, topic_pred, seq_len, top_num, self.output_attention)

            # cls_hidden = seq_bert_output[0][:, 0].contiguous()  # (batch_size, hidden_size)
            # pol_hidden = seq_bert_output[1].contiguous().detach()  # (batch_size, hidden_size)

            # senti_logits = self.senti_classifier(torch.concat((pol_hidden, seq_hidden), dim=1))
            # senti_logits = self.senti_classifier(pol_hidden + seq_hidden)
            senti_logits = self.senti_classifier(seq_hidden)

        else:
            with torch.no_grad():
                # seq_hidden, new_top_hidden, seq_atten, topic_atten = self.coattention(
                #     seq_input, top_input, topic_labels, seq_len, top_num, self.output_attention)

                seq_hidden, new_top_hidden, seq_atten, topic_atten = self.coattention(
                    seq_input, top_input, topic_pred, seq_len, top_num, self.output_attention)

                # cls_hidden = seq_bert_output[0][:, 0].contiguous()  # (batch_size, hidden_size)
                # pol_hidden = seq_bert_output[1].contiguous().detach()  # (batch_size, hidden_size)

                # senti_logits = self.senti_classifier(torch.concat((pol_hidden, seq_hidden), dim=1))
                # senti_logits = self.senti_classifier(pol_hidden + seq_hidden)
                senti_logits = self.senti_classifier(seq_hidden)

        senti_loss = self.senti_loss_fct(senti_logits, senti_labels)

        return senti_loss, topic_loss, senti_logits, topic_logits, seq_atten, topic_atten
