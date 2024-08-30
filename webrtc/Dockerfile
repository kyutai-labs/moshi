FROM ubuntu:22.04 as base
RUN apt update -q && apt install -y ca-certificates wget && \
    wget -qO /cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i /cuda-keyring.deb && apt update -q

RUN apt install -y python3 python3-pip

COPY ./msh ./msh
COPY ./webrtc ./webrtc
COPY ./requirements.txt ./requirements.txt

RUN pip install -r requirements.txt
RUN pip install -r webrtc/requirements.txt

FROM base

CMD python3 webrtc/main.py start --log-level debug
