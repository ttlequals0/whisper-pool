# faster-whisper server for DGX Spark (GB10, sm_121, aarch64).
# No ctranslate2[cuda] wheel exists for linux/arm64, so pip gives a CPU-only
# build. Install a CUDA build and recompile the bindings against it.
FROM nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04

ARG CT2_VERSION=4.6.0
ARG CT2_URL=https://github.com/assix/ctranslate2-aarch64-cuda13-binaries/releases/download/v4.6.0-cuda13-aarch64/ctranslate2-dgxspark-aarch64-cuda13.tar.gz

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates git build-essential python3-dev python3-venv \
        ffmpeg libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# 00- prefix keeps this ahead of pip-installed CUDA libs in the venv.
RUN curl -fsSL -o /tmp/ct2.tar.gz "${CT2_URL}" \
    && tar -xzf /tmp/ct2.tar.gz -C /opt \
    && rm /tmp/ct2.tar.gz \
    && echo "/opt/ctranslate2/lib" > /etc/ld.so.conf.d/00-ctranslate2.conf \
    && ldconfig

ENV VENV=/opt/venv
RUN python3 -m venv $VENV \
    && $VENV/bin/pip install --no-cache-dir --upgrade pip setuptools wheel pybind11

# PyPI wheel uses the old C++ ABI; swapping the .so alone breaks on std::string.
RUN git clone -q --branch "v${CT2_VERSION}" --depth 1 \
        https://github.com/OpenNMT/CTranslate2 /tmp/ct2src \
    && cd /tmp/ct2src/python \
    && CTRANSLATE2_ROOT=/opt/ctranslate2 \
       $VENV/bin/pip install --no-cache-dir --no-build-isolation . \
    && rm -rf /tmp/ct2src

RUN $VENV/bin/pip install --no-cache-dir \
        faster-whisper fastapi "uvicorn[standard]" python-multipart

ENV PATH=$VENV/bin:$PATH
WORKDIR /app
COPY hotwords.txt serve.py ./
EXPOSE 9000
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "9000", "--workers", "1"]
