# base image with cuda 12.1
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# install only the system packages actually needed
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-dev python3-pip git && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip

# install torch + torchvision + xformers together (xformers needs torch at build time)
RUN pip3 install --no-cache-dir \
    torch==2.7.0 torchvision xformers==0.0.30 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# install all runtime dependencies
COPY requirements.txt /requirements.txt
RUN pip3 install --no-cache-dir -r /requirements.txt

# enable fast HF downloads at startup
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# copy application files
COPY schemas.py handler.py /

CMD ["python3", "-u", "/handler.py"]
