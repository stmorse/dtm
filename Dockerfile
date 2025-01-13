FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# Install Python 3.12
RUN apt-get update && \
    apt-get install -y python3.10 python3.10-dev python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set default Python version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Copy the requirements file into the image
COPY requirements.txt /tmp/requirements.txt

# Install dependencies (now includes Dask)
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set a working directory
WORKDIR /

# Default command (optional, can be overridden by kubectl or docker run)
CMD ["python3"]
