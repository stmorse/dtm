# Use a CUDA 12.4 base image
FROM nvidia/cuda:12.4.0-base-ubuntu20.04

# Set environment variables for non-interactive apt installations
ENV DEBIAN_FRONTEND=noninteractive

# Update and install Python 3.12, pip, and necessary tools
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-distutils \
    python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set python3.12 as the default for python and pip
RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip
    
# Copy the requirements file into the image
COPY requirements.txt /tmp/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set a working directory
WORKDIR /app

# Default command (optional, can be overridden by kubectl or docker run)
CMD ["python3"]
