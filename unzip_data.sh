#!/bin/bash

mkdir -p resources/data/train
for train_part_id in $(seq 1 3); do
  unzip -d resources/data/train "RuREBus/train_data/train_part_$train_part_id.zip"
done