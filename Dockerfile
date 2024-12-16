# Use a CUDA 12.4 base image
FROM nvidia/cuda:12.4.0-base-ubuntu20.04

# Set environment variables for non-interactive apt installations
ENV DEBIAN_FRONTEND=noninteractive

# Update and install Python 3.12, pip, and necessary tools
# Install Python 3.12 and pip
RUN apt-get update && \
    apt-get install -y python3.12 python3.12-dev python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set default python and pip versions
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Copy the requirements file into the image
COPY requirements.txt /tmp/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set a working directory
WORKDIR /app

# Default command (optional, can be overridden by kubectl or docker run)
CMD ["python3"]
