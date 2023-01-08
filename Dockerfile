FROM python:3.8

WORKDIR /app
COPY requirements.txt /app

RUN pip install --upgrade pip

RUN pip3 install -r requirements.txt
RUN pip3 install torch==1.8.2 torchvision==0.9.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111

ENTRYPOINT [ "bash" ]