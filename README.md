# BERT + CRF for [RuREBus](https://github.com/dialogue-evaluation/RuREBus.git)

The main goal of this task is to train BERT-CRF model for solving Named Entity Recognition and Relation Extraction tasks
on RuREBus dataset (_Russian Relation Extraction for Business_).

![](resources/images/general_scheme.png)

## Structure
* [`datasets`](./datasets) – implementations of torch datasets.
* [`models`](./models) – models implementation (bert+crf, classifier for RE task solving)
* [`re_utils`](./re_utils) – various useful utilities (e.g. for working with files, ner data structure, for training models).
* [`resources`](./resources) – materials for the design of the repository, it is also supposed to store data there for training and testing models.
* [`RuREBus`](https://github.com/dialogue-evaluation/RuREBus.git) – repository with original task.
* [`scripts`](./scripts) – scripts for preparing data to training and evaluation.
* [`ner_experiments.ipynb`](./ner_experiments.ipynb) – training different models to solve NER task.
* [`re_experiments.ipynb`](./re_experiments.ipynb) – training model to solve RE task.

## Requirements

Create virtual environment with `venv` or `conda` and install requirements:

```shell
pip install -r requirements.txt
```

Or build and run docker container:
```shell
./run_docker.sh
```

==========================
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

In [experiments.ipynb](experiments.ipynb) notebook we've provide code for training different version of BERT model.
Our finetunned BERT with CRF layer shows the best F1-Micro score.

|          | ruBERT | ruBERT + CRF | ruREBus-BERT | ruREBus-BERT + CRF |
|----------|--------|--------------|--------------|--------------------|
| F1-micro | 0.8023 | 0.8057       | 0.8052       | **0.8122**         |

### BERT-CRF for Relation Extraction (RE):

