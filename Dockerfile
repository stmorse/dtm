FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# Install Python 3.12
RUN apt-get update && \
    apt-get install -y python3.12 python3.12-dev

# Set default Python version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Install pip
RUN apt-get install -y python3-pip

# Upgrade pip
RUN python -m pip install --upgrade pip

# Copy the requirements file into the image
COPY requirements.txt /tmp/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set a working directory
WORKDIR /app

# Default command (optional, can be overridden by kubectl or docker run)
CMD ["python3"]
