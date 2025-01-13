# FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# # Install Python 3.12
# RUN apt-get update && \
#     apt-get install -y python3.10 python3.10-dev python3-pip && \
#     apt-get clean && rm -rf /var/lib/apt/lists/*

# # Set default Python version
# RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# # Upgrade pip
# RUN python -m pip install --upgrade pip

# # Copy the requirements file into the image
# COPY requirements.txt /tmp/requirements.txt

# # Install dependencies (now includes Dask)
# RUN pip install --no-cache-dir -r /tmp/requirements.txt

# # Set a working directory
# WORKDIR /

# # Default command (optional, can be overridden by kubectl or docker run)
# CMD ["python3"]

FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# Sys packages for building Python
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        wget \
        curl \
        libssl-dev \
        libffi-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev && \
        liblzma-dev \
        xz-utils && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Download & build Python 3.12
WORKDIR /tmp
RUN wget https://www.python.org/ftp/python/3.12.0/Python-3.12.0.tgz && \
    tar -xf Python-3.12.0.tgz && \
    cd Python-3.12.0 && \
    ./configure --enable-optimizations --with-ensurepip=install && \
    make -j$(nproc) && \
    make altinstall && \
    rm -rf /tmp/Python-3.12.0*

# Link python3 -> python3.12
RUN ln -s /usr/local/bin/python3.12 /usr/local/bin/python3

# Make sure pip is up to date
RUN python3 -m pip install --upgrade pip

# Copy your requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /
CMD ["python3"]

