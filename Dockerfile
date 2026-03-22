# base image with cuda 12.1
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV CC=gcc

# install only the system packages actually needed
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-dev python3-pip git build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip

# install torch + torchvision first so later packages don't pull a different version
RUN pip3 install --no-cache-dir \
    torch==2.7.0 torchvision \
    --extra-index-url https://download.pytorch.org/whl/cu121

# install all runtime dependencies (flat graph — resolves quickly)
COPY requirements.txt /requirements.txt
RUN pip3 install --no-cache-dir -r /requirements.txt

# install pruna LAST with --no-deps to avoid pip resolution-too-deep
# (its deps are already satisfied by requirements.txt above;
#  we intentionally skip llmcompressor, librosa, whisper-s2t, ctranslate2
#  because they cause resolution explosions and aren't needed for FLUX inference)
RUN pip3 install --no-cache-dir --no-deps pruna==0.2.5

# copy application files
COPY schemas.py handler.py /

CMD python3 -u /handler.py
