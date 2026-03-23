# devel image includes nvcc, CUDA toolkit & headers (needed by triton at runtime)
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV CC=gcc

# install only the system packages actually needed
# build-essential (gcc, g++, make, libc-dev) needed by triton to compile its CUDA driver module
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-dev python3-pip git build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip

# install torch + torchvision + xformers together (xformers needs torch at build time)
RUN pip3 install --no-cache-dir \
    torch==2.7.0 torchvision xformers==0.0.30 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# install all runtime dependencies
COPY requirements.txt /requirements.txt
RUN pip3 install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    -r /requirements.txt

# enable fast HF downloads at startup
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# copy application files
COPY schemas.py handler.py /

CMD ["python3", "-u", "/handler.py"]
