## BERT-CRF for [RuREBus](https://github.com/dialogue-evaluation/RuREBus.git)

The main goal of this task is to train BERT-CRF model for solving Named Entity Recognition and Relation Extraction tasks
on RuREBus dataset (_Russian Relation Extraction for Business_)

### RuREBus dataset preprocessing

```shell
# Downloading RuREBus dataset
$ git clone https://github.com/dialogue-evaluation/RuREBus.git

# Unpacking train and test files
$ bash unzip_data.sh

# Tokenize text for NER and RE tasks. Run with --help flag to see documentation.
$ python scripts/tokenize_texts.py
```

### BERT Finetuning via Masked Language Modeling (MLM)

There are a lot of Russian pretrained language models, the most popular one is
[sberbank/ruBERT-base](https://huggingface.co/sberbank-ai/ruBert-base). In order to get a higher quality when solving
_NER_ and _RE_ on the RuREBus dataset, we've applied masked language modeling to sberbank/ruBERT-base model.
The dataset for finetunning was chosen from the same domain: https://disk.yandex.ru/d/9uKbo3p0ghdNpQ

1. Create masked dataset for BERT finetunning:
   ```shell
   $ python scripts/mask_texts.py 
   ```
2. Running train script with created dataset:
   ```shell
   $ python scripts/mlm.py
   ```

### BERT-CRF for Named Entity Recognition (NER):

### BERT-CRF for Relation Extraction (RE):

