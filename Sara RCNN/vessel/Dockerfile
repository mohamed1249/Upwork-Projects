FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

COPY . /binary_class
WORKDIR /binary_class
RUN pip3 install --upgrade pip setuptools --no-cache-dir
RUN pip3 install -r requirements.txt --no-cache-dir