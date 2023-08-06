FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
	golang-go git rsync python3-wheel python3-pip \
	&& pip3 install --upgrade pip \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/*
RUN pip3 install tensorflow
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN go install github.com/bazelbuild/bazelisk@latest && ln -s /root/go/bin/bazelisk /usr/bin/bazel
CMD ["/bin/bash"]
