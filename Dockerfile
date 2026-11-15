# TALOS v2, CPU. Runs the tests, the README table check and the pipeline with ONNX Runtime.
#   docker build -t talos .
#   docker run --rm talos                                   # tests + README check
#   docker run --rm -v "$PWD/work:/work" talos \
#     python -m src.TALOS.pipeline.run /work/frames --out /work/out --zone-circle CX,CY,R \
#       --device cpu --backend onnx --weights /work/talos_y26s_t1024.onnx
FROM python:3.12-slim
LABEL org.opencontainers.image.source="https://github.com/Gakshith/Tactical-Aerospace-Localization-Objective-Scoring-TALOS-"

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 YOLO_OFFLINE=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics MPLCONFIGDIR=/tmp/matplotlib

RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY env/ env/
RUN pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r env/requirements-cpu.txt \
    && bash env/patch_trackeval.sh /usr/local

COPY . .
RUN useradd -m talos && chown -R talos /app
USER talos

CMD ["bash", "-c", "python scripts/readme_tables.py --check && python -m pytest tests -q"]
