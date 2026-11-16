# syntax=docker/dockerfile:1
# TALOS inference on AWS Lambda (container image, x86_64). Build from the repo root:
#   docker build -f deploy/lambda/Dockerfile -t talos-lambda .
FROM public.ecr.aws/lambda/python:3.12

ARG MODEL_URL=https://huggingface.co/spaces/dustin-demo/talos-satellite-tracking/resolve/main/models/talos_y26s_t1024.onnx
ARG MODEL_SHA256=03a6f0d26db73279bec0ebe9ac65fe6380eea794ba06666212db1f69c73f40a0

COPY deploy/lambda/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Python, not curl: the slim Lambda base image does not promise curl. The hash pins the exact model.
RUN python - <<PY
import hashlib, os, urllib.request
os.makedirs("/opt/talos", exist_ok=True)
dst = "/opt/talos/talos_y26s_t1024.onnx"
urllib.request.urlretrieve("$MODEL_URL", dst)
got = hashlib.sha256(open(dst, "rb").read()).hexdigest()
assert got == "$MODEL_SHA256", f"model sha256 {got} != $MODEL_SHA256"
PY

COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY configs/talos_v2.json ${LAMBDA_TASK_ROOT}/configs/talos_v2.json
COPY deploy/lambda/handler.py ${LAMBDA_TASK_ROOT}/handler.py

CMD ["handler.handler"]
