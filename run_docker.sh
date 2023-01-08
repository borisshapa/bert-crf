#!/bin/bash

app=$PWD

docker build -t re-business . && \
docker run -it --rm \
    --net=host --ipc=host --shm-size=2048m \
    --gpus "all" \
    -v "$app":/app \
    re-business
