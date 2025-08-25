FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    curl \
    git \
    wget \
    libssl-dev \
    libopenblas-dev \
    liblapack-dev \
    libcurl4-openssl-dev \
    python3 \
    python3-pip \
    clang \
    lld \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
WORKDIR /app
COPY . .
CMD ["tail", "-f", "/dev/null"]
