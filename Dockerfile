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

# install torch + torchvision + xformers + torchao from the cu126 index
# --index-url (not --extra-index-url) forces cu126 wheels matching the CUDA 12.6 base
# torchao pinned here (not in requirements.txt) to prevent pip resolving 0.16.0 from cu126
RUN pip3 install --no-cache-dir \
    torch==2.7.0+cu126 torchvision xformers==0.0.30 torchao==0.15.0 \
    --index-url https://download.pytorch.org/whl/cu126

# install all runtime dependencies (flat graph — resolves quickly)
# --extra-index-url allows cu126 wheels for torchao while still resolving PyPI packages
COPY requirements.txt /requirements.txt
RUN pip3 install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    -r /requirements.txt

# install pruna LAST with --no-deps to avoid pip resolution-too-deep
# (its deps are already satisfied by requirements.txt above;
#  we intentionally skip llmcompressor, librosa, whisper-s2t, ctranslate2
#  because they cause resolution explosions and aren't needed for FLUX inference)
RUN pip3 install --no-cache-dir --no-deps pruna==0.2.5

# build-time sanity check — will fail the build if versions are wrong
RUN python3 -c "
import torch, torchao, bitsandbytes
print(f'torch={torch.__version__}, CUDA={torch.version.cuda}')
print(f'torchao={torchao.__version__}')
print(f'bitsandbytes={bitsandbytes.__version__}')
assert 'cu126' in torch.__version__, f'wrong torch build: {torch.__version__}'
assert torchao.__version__.startswith('0.15'), f'wrong torchao: {torchao.__version__}'
"

# copy application files
COPY schemas.py handler.py /

CMD python3 -u /handler.py
