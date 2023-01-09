#!/bin/bash

mkdir -p resources/data/train
mkdir -p resources/data/test

for train_part_id in $(seq 1 3); do
  unzip -d resources/data/train "RuREBus/train_data/train_part_$train_part_id.zip"
done

unzip -d resources/data/test "RuREBus/test_data/test_full.zip"