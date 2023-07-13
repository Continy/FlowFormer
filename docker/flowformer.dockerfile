
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
WORKDIR /workspace

RUN python3.9 -m pip install --upgrade pip
RUN pip3 install --upgrade setuptools
RUN pip3 --no-cache-dir install lit
RUN pip3 --no-cache-dir install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 --no-cache-dir install scipy tensorboard opencv-python opencv-python-headless
RUN pip3 --no-cache-dir install yacs loguru einops timm==0.4.12 imageio gdown



