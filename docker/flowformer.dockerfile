
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04




# Install basic dependencies
RUN apt update && \
    apt install -y \
    wget build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev \
    libreadline-dev libffi-dev libsqlite3-dev libbz2-dev liblzma-dev && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python 3.9.2
WORKDIR /temp

RUN wget https://www.python.org/ftp/python/3.9.2/Python-3.9.2.tgz && \
    tar xvf Python-3.9.2.tgz && \
    cd Python-3.9.2 && \
    ./configure --enable-optimizations && \
    make -j 8 && \
    make altinstall && \
    cd .. && \
    rm -rf Python-3.9.2.tgz && \
    rm -rf Python-3.9.2

WORKDIR /workspace

RUN python -m pip install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install matplotlib numpy scipy tensorboard opencv 
RUN pip3 install yacs loguru einops timm==0.4.12 imageio

