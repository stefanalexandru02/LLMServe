# syntax=docker/dockerfile:1.7
#
# LLMServe: Qwen3.8-Flash-Next on an NVIDIA GPU, served over an OpenAI/Anthropic-compatible HTTP API.
# Built FROM SOURCE with CUDA 13.  Needs BuildKit (Docker Desktop and Docker 23+ have it on by default).
#
#   docker compose up -d --build                         # the usual way (docker-compose.yml)
#   docker build -t llmserve:cuda .                      # same image without compose
#
# The engine is compiled here for the GPU architectures in CUDA_ARCHS.  `docker build` has no GPU, so they are named
# instead of read from nvidia-smi.  Building for every architecture takes 25+ minutes on a fast 8-core machine and
# a lot of RAM; build only for your card:
#
#   RTX 30 series (Ampere)        --build-arg CUDA_ARCHS="86-real;86-virtual"
#   RTX 40 series (Ada)           --build-arg CUDA_ARCHS="89-real;89-virtual"
#   RTX 50 series (Blackwell)     --build-arg CUDA_ARCHS="120-real;120-virtual"
#   default (all of the above)    "86-real;89-real;120-real;120-virtual"
#
# Other build arguments:
#   PORTABLE=ON|OFF     ON (default): the CPU code needs only AVX2 and runs on any x86-64 PC.  OFF: ggml's CPU kernels
#                       are built with -march=native for the machine that BUILDS the image - only if that is the PC
#                       that will run it.  (The engine's own AVX2/AVX-512 kernels are chosen at run time either way.)
#   WITH_VISION=1|0     build strata-vision, the picture reader (LLMSERVE_VISION=on).  0 saves ~10 minutes.
#   BUILD_JOBS=N        compiler jobs; default is all CPUs.  Lower it (4) if the build runs out of memory: Docker
#                       Desktop's WSL2 VM has half of your RAM by default.
#   CUDA_VERSION        CUDA 13.0.x tag of the nvidia/cuda images (the engine needs CUDA 13: 12.8 crashes on long
#                       prompts with Blackwell; the host driver must be 580 or newer).
#
# What is in the image: /opt/llmserve/bin/{strata,libstrata_kq.so[,strata-vision]}, the Python server, the model
# preparation tools, and llama.cpp's gguf-py at the commit the engine is pinned to.  No model: the first start
# downloads it (~70 GB for Q2_0) into the /data volume.

ARG CUDA_VERSION=13.0.2
ARG UBUNTU=ubuntu24.04

# ------------------------------------------------------------------------------------------------ toolchain
# Everything that does not depend on this repository's sources, so editing serve/ or llmserve/ never rebuilds it.
FROM nvidia/cuda:${CUDA_VERSION}-devel-${UBUNTU} AS toolchain
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential git curl unzip python3-venv ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/buildvenv \
 && /opt/buildvenv/bin/pip install --no-cache-dir cmake==4.4.3 ninja==1.13.2
ENV PATH=/opt/buildvenv/bin:$PATH

# llama.cpp at the commit the engine is pinned to (ggml for the engine, mtmd for pictures, gguf-py for the tools)
ARG LLAMA_CPP_COMMIT=3cf03257f219afbe7334045ff7c6a06ac68c627d
RUN set -eux; \
    curl -fsSL --retry 5 --retry-delay 5 -o /tmp/llama.zip \
        "https://github.com/ggml-org/llama.cpp/archive/${LLAMA_CPP_COMMIT}.zip"; \
    unzip -q /tmp/llama.zip -d /tmp; \
    mv "/tmp/llama.cpp-${LLAMA_CPP_COMMIT}" /opt/llama.cpp; \
    rm /tmp/llama.zip; \
    test -f /opt/llama.cpp/ggml/CMakeLists.txt; \
    test -d /opt/llama.cpp/gguf-py/gguf

# ------------------------------------------------------------------------------------------------ engine
FROM toolchain AS engine
ARG CUDA_ARCHS="86-real;89-real;120-real;120-virtual"
ARG PORTABLE=ON
ARG BUILD_JOBS=
WORKDIR /src
# only what the engine's CMake needs: a change in serve/, llmserve/ or tools/ never rebuilds it
COPY CMakeLists.txt ./
COPY cmake/ cmake/
COPY third_party/ third_party/
COPY include/ include/
COPY src/ src/
RUN --mount=type=cache,target=/build,id=llmserve-engine-build \
    set -eux; \
    cmake -G Ninja -S . -B /build -DCMAKE_BUILD_TYPE=Release \
        -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_NATIVE_EXPERTS=ON \
        -DSTRATA_PORTABLE="${PORTABLE}" -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON \
        "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}" -DSTRATA_GGML_DIR=/opt/llama.cpp; \
    cmake --build /build --target strata strata_kq -j "${BUILD_JOBS:-$(nproc)}"; \
    mkdir -p /out; \
    cp /build/strata /build/libstrata_kq.so /out/; \
    ls -la /out

# ------------------------------------------------------------------------------------------------ picture reader
FROM toolchain AS vision
ARG CUDA_ARCHS="86-real;89-real;120-real;120-virtual"
ARG PORTABLE=ON
ARG WITH_VISION=1
ARG BUILD_JOBS=
WORKDIR /src
COPY tools/vision/ tools/vision/
RUN --mount=type=cache,target=/build-vision,id=llmserve-vision-build \
    set -eux; \
    mkdir -p /out; \
    if [ "${WITH_VISION}" = "1" ]; then \
      cmake -G Ninja -S tools/vision -B /build-vision -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=/opt/llama.cpp \
          -DSTRATA_VISION_CUDA=ON -DSTRATA_PORTABLE="${PORTABLE}" -DGGML_CUDA_NCCL=OFF \
          "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}"; \
      cmake --build /build-vision --target strata-vision -j "${BUILD_JOBS:-$(nproc)}"; \
      cp /build-vision/bin/strata-vision /out/; \
    fi; \
    ls -la /out

# ------------------------------------------------------------------------------------------------ runtime
# The "runtime" image of NVIDIA's CUDA 13 line holds libcudart and libcublas (+ libcublasLt) that the engine loads.
FROM nvidia/cuda:${CUDA_VERSION}-runtime-${UBUNTU} AS runtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv tini libgomp1 libnuma1 libatomic1 curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Python: the server (jinja2, regex, psutil, pillow), the model tools (numpy + gguf-py's own: tqdm, pyyaml, requests)
# and Hugging Face downloads that resume and use several connections (huggingface_hub + hf_xet).  Versions are the
# ones Strata's requirements.txt pins, plus the downloader.
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir \
        numpy==2.5.3 jinja2==3.1.6 markupsafe==3.0.3 regex==2026.9.10 psutil==7.2.2 pillow==12.3.0 \
        pyyaml==6.0.3 tqdm==4.70.1 requests==2.34.2 certifi==2026.10.19 truststore==0.10.5 \
        huggingface_hub==2.3.0 hf_xet==1.7.0

# gguf-py of the pinned llama.cpp: the packing tools need it (Q2_0 is ggml type 42, newer than the PyPI gguf)
COPY --from=toolchain /opt/llama.cpp/gguf-py /opt/gguf-py
COPY --from=toolchain /opt/llama.cpp/LICENSE /usr/share/doc/llmserve/LICENSE.llama.cpp
RUN PYTHONPATH=/opt/gguf-py /opt/venv/bin/python -c "import gguf, numpy, jinja2, regex, psutil, PIL, huggingface_hub; print('python deps ok, gguf', gguf.__file__)"

# the user that runs everything.  uid 1000 owns /data, so a NEW named volume mounted there starts out writable.
RUN (userdel -r ubuntu 2>/dev/null || true) \
 && groupadd -g 1000 llmserve \
 && useradd -m -u 1000 -g 1000 -s /bin/bash llmserve \
 && mkdir -p /data /opt/llmserve \
 && chown llmserve:llmserve /data

# engine (least likely to change first)
COPY --from=engine /out/ /opt/llmserve/bin/
COPY --from=vision /out/ /opt/llmserve/bin/
WORKDIR /opt/llmserve
COPY LICENSE ./
COPY tools/ tools/
COPY data/ data/
COPY strata_control/ strata_control/
COPY strata_cli/ strata_cli/
COPY serve/ serve/
COPY llmserve/ llmserve/
COPY docker/entrypoint.sh docker/healthcheck.sh docker/
RUN chmod 0755 docker/entrypoint.sh docker/healthcheck.sh bin/* \
 && /opt/venv/bin/python -m compileall -q serve strata_control strata_cli tools llmserve \
 && missing=$(ldd bin/* 2>&1 | grep 'not found' | grep -v -e 'libcuda\.so' -e 'libnvidia' || true) \
 && if [ -n "$missing" ]; then echo "the engine needs libraries this image lacks:"; echo "$missing"; exit 1; fi

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 LANG=C.UTF-8 \
    LLMSERVE_APP_DIR=/opt/llmserve LLMSERVE_ENGINE_DIR=/opt/llmserve/bin LLMSERVE_DATA_DIR=/data \
    STRATA_GGUF_PY=/opt/gguf-py STRATA_NO_PRESET=1 \
    CUDA_MODULE_LOADING=EAGER \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HOME=/data/home XDG_CONFIG_HOME=/data/home/.config XDG_STATE_HOME=/data/home/.local/state \
    XDG_CACHE_HOME=/data/home/.cache HF_HOME=/data/home/.cache/huggingface HF_HUB_DISABLE_TELEMETRY=1

USER llmserve
VOLUME ["/data"]
EXPOSE 8080
# first start: ~70 GB download + packing (an hour or two); the container is "starting" until then, not "unhealthy"
HEALTHCHECK --interval=30s --timeout=10s --start-period=2h --retries=3 CMD ["/opt/llmserve/docker/healthcheck.sh"]
STOPSIGNAL SIGTERM
LABEL org.opencontainers.image.title="LLMServe" \
      org.opencontainers.image.description="Qwen3.8-Flash-Next on NVIDIA CUDA, OpenAI/Anthropic-compatible API (a headless fork of Strata)" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/niko1221/strata"
ENTRYPOINT ["tini", "--", "/opt/llmserve/docker/entrypoint.sh"]
CMD ["serve"]
