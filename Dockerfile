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

# install torch + torchvision first so later packages don't pull a different version
RUN pip3 install --no-cache-dir \
    torch==2.7.0 torchvision \
    --extra-index-url https://download.pytorch.org/whl/cu121

# install all runtime dependencies
# constraints.txt locks torch/numpy so pruna's dep tree can't swap versions
COPY constraints.txt /constraints.txt
COPY requirements.txt /requirements.txt
RUN pip3 install --no-cache-dir -c /constraints.txt -r /requirements.txt

# copy application files
COPY download_weights.py schemas.py handler.py test_input.json /

# download the weights from hugging face
RUN python3 /download_weights.py

CMD python3 -u /handler.py
