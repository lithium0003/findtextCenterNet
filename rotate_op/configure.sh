#!/bin/bash
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"
PIP="pip3"
GCC_VER="$(gcc -dumpversion)"

function write_to_bazelrc() {
  echo "$1" >> .bazelrc
}

function write_action_env_to_bazelrc() {
  write_to_bazelrc "build --action_env $1=\"$2\""
}

function is_linux() {
  [[ "${PLATFORM}" == "linux" ]]
}

function is_macos() {
  [[ "${PLATFORM}" == "darwin" ]]
}

function is_windows() {
  # On windows, the shell script is actually running in msys
  [[ "${PLATFORM}" =~ msys_nt*|mingw*|cygwin*|uwin* ]]
}

function is_ppc64le() {
  [[ "$(uname -m)" == "ppc64le" ]]
}


# Remove .bazelrc if it already exist
[ -e .bazelrc ] && rm .bazelrc

TF_NEED_CUDA=0
TF_CUDA_VERSION=11.8

# Check if it's installed
if [[ $(${PIP} show tensorflow) == *tensorflow* ]] || [[ $(${PIP} show tf-nightly) == *tf-nightly* ]]; then
  echo 'Using installed tensorflow'
else
  # Uninstall CPU version if it is installed.
  if [[ $(${PIP} show tensorflow-cpu) == *tensorflow-cpu* ]]; then
    echo 'Already have tensorflow non-gpu installed. Uninstalling......\n'
    ${PIP} uninstall tensorflow
  elif [[ $(${PIP} show tf-nightly-cpu) == *tf-nightly-cpu* ]]; then
    echo 'Already have tensorflow non-gpu installed. Uninstalling......\n'
    ${PIP} uninstall tf-nightly
  fi
  # Install GPU version
  echo 'Installing tensorflow .....\n'
  ${PIP} install tensorflow
fi

TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS="$(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')"

write_to_bazelrc "build:cuda --define=using_cuda=true --define=using_cuda_nvcc=true"
if [[ "$PIP_MANYLINUX2010" == "0" ]]; then
  write_to_bazelrc "build:cuda --crosstool_top=@local_config_cuda//crosstool:toolchain"
fi
# Add Ubuntu toolchain flags
if is_linux; then
  if [[ "$GCC_VER" == "9" ]]; then
    write_to_bazelrc "build:manylinux2010cuda --crosstool_top=//third_party/toolchains/gcc9_manylinux2010-nvcc-cuda:toolchain"
  else
    write_to_bazelrc "build:manylinux2010cuda --crosstool_top=//third_party/toolchains/gcc11_manylinux2010-nvcc-cuda:toolchain"
  fi
fi
write_to_bazelrc "build --spawn_strategy=standalone"
write_to_bazelrc "build --strategy=Genrule=standalone"
write_to_bazelrc "build -c opt"

# MSVC (Windows): Standards-conformant preprocessor mode
# See https://docs.microsoft.com/en-us/cpp/preprocessor/preprocessor-experimental-overview
if is_windows; then
  write_to_bazelrc "build --copt=/experimental:preprocessor"
  write_to_bazelrc "build --host_copt=/experimental:preprocessor"
fi

if is_windows; then
  # Use pywrap_tensorflow instead of tensorflow_framework on Windows
  SHARED_LIBRARY_DIR=${TF_CFLAGS:2:-7}"python"
else
  SHARED_LIBRARY_DIR=${TF_LFLAGS:2}
fi
SHARED_LIBRARY_NAME=$(echo $TF_LFLAGS | rev | cut -d":" -f1 | rev)
if ! [[ $TF_LFLAGS =~ .*:.* ]]; then
  if is_macos; then
    SHARED_LIBRARY_NAME="libtensorflow_framework.dylib"
  elif is_windows; then
    # Use pywrap_tensorflow's import library on Windows. It is in the same dir as the dll/pyd.
    SHARED_LIBRARY_NAME="_pywrap_tensorflow_internal.lib"
  else
    SHARED_LIBRARY_NAME="libtensorflow_framework.so"
  fi
fi

HEADER_DIR=${TF_CFLAGS:2}
if is_windows; then
  SHARED_LIBRARY_DIR=${SHARED_LIBRARY_DIR//\\//}
  SHARED_LIBRARY_NAME=${SHARED_LIBRARY_NAME//\\//}
  HEADER_DIR=${HEADER_DIR//\\//}
fi
write_action_env_to_bazelrc "TF_HEADER_DIR" ${HEADER_DIR}
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_DIR" ${SHARED_LIBRARY_DIR}
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_NAME" ${SHARED_LIBRARY_NAME}
write_action_env_to_bazelrc "TF_NEED_CUDA" ${TF_NEED_CUDA}

# TODO(yifeif): do not hardcode path
if [[ "$TF_NEED_CUDA" == "1" ]]; then
  write_action_env_to_bazelrc "TF_CUDA_VERSION" ${TF_CUDA_VERSION}
  write_action_env_to_bazelrc "TF_CUDNN_VERSION" "8"
  if is_windows; then
    write_action_env_to_bazelrc "CUDNN_INSTALL_PATH" "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${TF_CUDA_VERSION}"
    write_action_env_to_bazelrc "CUDA_TOOLKIT_PATH" "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${TF_CUDA_VERSION}"
  else
    write_action_env_to_bazelrc "CUDNN_INSTALL_PATH" "/usr/lib/x86_64-linux-gnu"
    write_action_env_to_bazelrc "CUDA_TOOLKIT_PATH" "/usr/local/cuda"
  fi
  write_to_bazelrc "build --config=cuda"
  write_to_bazelrc "test --config=cuda"
fi

write_to_bazelrc "build --config=manylinux2010cuda"
write_to_bazelrc "test --config=manylinux2010cuda"
