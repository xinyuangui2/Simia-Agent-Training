# Dockerfile for Simia-Agent-Training
# Supports both Simia_SFT and Simia-RL

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    vim \
    sudo \
    build-essential \
    software-properties-common \
    tesseract-ocr \
    poppler-utils \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install pip for Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Set working directory
WORKDIR /workspace

# Clone the repository
ARG REPO_URL=https://github.com/xinyuangui2/Simia-Agent-Training.git
ARG BRANCH=main
RUN git clone --branch ${BRANCH} ${REPO_URL} /workspace/Simia-Agent-Training

WORKDIR /workspace/Simia-Agent-Training

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install vLLM
RUN pip install --no-cache-dir vllm==0.8.5

# Install flash-attention (requires CUDA)
RUN pip cache purge && \
    pip install --no-cache-dir flash-attn==2.7.4.post1 --no-build-isolation

# Install verl and ragen in editable mode
RUN pip install --no-cache-dir -e Simia-RL/subtrees/verl -e Simia-RL/subtrees/ragen --no-dependencies

# Install ragen requirements (without webshop external dependencies)
RUN pip install --no-cache-dir \
    IPython \
    matplotlib \
    gym \
    peft \
    accelerate \
    codetiming \
    datasets \
    dill \
    hydra-core \
    numpy \
    pandas \
    pybind11 \
    "ray>=2.10" \
    tensordict==0.6.2 \
    transformers \
    wandb \
    gymnasium \
    "gymnasium[toy-text]" \
    "pyarrow>=15.0.0" \
    pylatexenc \
    torchdata \
    debugpy \
    together \
    anthropic \
    liger-kernel

# Install remaining requirements for Simia-RL
RUN pip install --no-cache-dir \
    fire \
    python-docx \
    scikit-learn \
    openpyxl \
    tabulate \
    Pillow \
    PyMuPDF \
    PyPDF2 \
    pdf2docx \
    pytesseract \
    icalendar \
    gymnasium \
    rich \
    docker \
    mysql-connector-python \
    rpyc \
    pyyaml \
    msal \
    "ruamel.yaml==0.18.10"

# Install additional dependencies for MLflow and Azure
RUN pip install --no-cache-dir --ignore-installed azureml-mlflow mlflow

# Install specific versions of ray and opentelemetry
RUN pip install --no-cache-dir \
    ray==2.49.1 \
    opentelemetry-api==1.26.0 \
    opentelemetry-sdk==1.26.0

# Install LLaMA Factory for SFT training
RUN pip install --no-cache-dir llamafactory

# Install OpenAI SDK for API interactions
RUN pip install --no-cache-dir openai

# Unzip preprocessed data files
RUN cd Simia_SFT/Tau2 && unzip -o APIGen_5k_preprocessed_zip.zip || true
RUN cd Simia-RL && unzip -o APIGen_5k_processed_zip.zip || true

# Set default environment variables (users should override these)
ENV API_TYPE="openai"
ENV OPENAI_API_KEY=""
ENV AZURE_OPENAI_ENDPOINT=""
ENV WANDB_API_KEY=""