# CUDA runtime + cuDNN (driver lives on the node)
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Prevent interactive apt prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 + system tools
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-distutils python3-pip \
    sudo rsync openssh-server \
    ffmpeg libsm6 libxext6 htop screen \
    git \
 && ln -sf /usr/bin/python3.11 /usr/bin/python \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# SSH setup 
RUN mkdir -p /var/run/sshd \
 && mkdir -p /root/.ssh \
 && chmod 700 /root/.ssh \
 && touch /root/.ssh/authorized_keys

COPY sshd_config /etc/ssh/sshd_config

# Working directory 
WORKDIR /storage/courses

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
