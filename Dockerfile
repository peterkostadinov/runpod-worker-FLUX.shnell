# devel image includes gcc, CUDA toolkit & headers (needed by triton at runtime)
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV CC=gcc

# install only the system packages actually needed
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-dev python3-pip git && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip

# install torch + torchvision first so later packages don't pull a different version
# --index-url (not --extra-index-url) forces cu121 wheel matching the CUDA 12.1 base
RUN pip3 install --no-cache-dir \
    torch==2.7.0+cu121 torchvision \
    --index-url https://download.pytorch.org/whl/cu121

# install all runtime dependencies (flat graph — resolves quickly)
COPY requirements.txt /requirements.txt
RUN pip3 install --no-cache-dir -r /requirements.txt

# install pruna LAST with --no-deps to avoid pip resolution-too-deep
# (its deps are already satisfied by requirements.txt above;
#  we intentionally skip llmcompressor, librosa, whisper-s2t, ctranslate2
#  because they cause resolution explosions and aren't needed for FLUX inference)
RUN pip3 install --no-cache-dir --no-deps pruna==0.2.5

# build-time sanity check — will fail the build if gcc or torch are wrong
RUN gcc --version && python3 -c "import torch; print(f'torch={torch.__version__}, CUDA={torch.version.cuda}')"

# copy application files
COPY schemas.py handler.py /

CMD python3 -u /handler.py
