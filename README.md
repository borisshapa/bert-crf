### Обработка датасета RuREBus

```shell
# Скачиваем тренировочные файлы
$ git clone https://github.com/dialogue-evaluation/RuREBus.git

# Распаковываем файлы
$ bash unzip_data.sh

# Токенезируем текст, для описания ожидаемых аргументов запустите с флагом --help
$ python scripts/tokenize_texts.py
```

### BERT Finetuning

В качестве предобученной модели была выбрана
модель [sberbank/ruBERT-base](https://huggingface.co/sberbank-ai/ruBert-base).
Однако для того, чтобы получить более высокое качество при решении задач *Named Entity Recognition* и *Relation
Extraction* на датасете RuREBus, мы дообучили BERT на неразмеченных данных того же
домена: https://disk.yandex.ru/d/9uKbo3p0ghdNpQ

1. Подготовка датасета для доубучения:
   ```shell
   $ python scripts/mask_texts.py 
   ```
2. Запуск дообучения модели:
   ```shell
   $ python scripts/mlm.py
   ```

### Обучение модели BERT-CRF