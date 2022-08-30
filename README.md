# TescaBERT

The code and dataset for the paper _Topic Enhanced Sentiment Co-Attention BERT_, implemented in PyTorch.

## Requirements

------

- Python 3.6+
- PyTorch/PyTorch-gpu 1.7.1
- transformers 3.0.2
- numpy 1.19.3
- jieba 0.39
- scikit_learn 1.0.2
- gensim 3.8.1

## Usage

------

Provided datasets include `bank` and `asap`/`restaurant`. Both of them are **Chinese** review data.
`bank` is bank forum data of financial product, which is provided by [CCF BDCI](https://www.datafountain.cn/competitions/529/datasets).
`asap`/`restaurant` is restaurant review data from online platform, which is constructed by [Bu et al](https://arxiv.org/pdf/2103.06605v2.pdf), you can download it from [Github](https://github.com/Meituan-Dianping/asap).
The data used in this paper has been preprocessed and saved in `data`, which is **different** from the original data. Please refer to the paper for preprocessing details.

Other two English datasets are `sst-5` and `amazon-food`/`food`. 
The former is film review data, which is provided by [Stanford](https://nlp.stanford.edu/sentiment/). 
The latter is sampled Amazon fine foods review data. You can download the complete data from[ Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews). 
Both of them have 5 ratings, and we preprocess it to 3 lables (positive, negative, neutral). 
Since they have no topic label, they are not applicable to TescaBERT, and are just used to test the performance of TescaBERT-lda.

Note that , we just provide `config.json` of BERT and RoBERTa. 
Before training, you should **download pre-trained model** and put them into `./data/bert_model/bert_name`. 
For example, if you want to use RoBERTa as pre-training model, 
you should download the model from [Hugging Face](https://huggingface.co/hfl/chinese-roberta-wwm-ext/tree/main),
and put `pytorch_model.bin`, `vocab.txt` (For English model, `tokenizer.json` is also required.) into `./data/bert_model/roberta`.

Start training and inference TescaBERT as:

```
cd shell
bash ./run.sh
```

Start training and inference TescaBERT-lda as:

```
cd shell
bash ./lda.sh
```

More details

- The model and result will be saved in `model/model_DATASET`.

- The experiment of noise inject and loss function weight is conducted in `noise.sh` and `loss.sh`.

- If you want to test LDA model by yourself, please run `src/lda_clustering.py`.

- If you want to train the model on different dataset, please change the annotated code in the `.sh` file. More details of model parameters, please refer to files in  `config`.

## Citation

------

If you make advantage of the TescaBERT model in your research, please cite the following in your manuscript:

```
To be published
```
